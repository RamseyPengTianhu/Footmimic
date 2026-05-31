"""V3.6 Phase-Free Contact Quality and Modulated Tracking Rewards.

Implements:
1. Phase-modulated body tracking (scales tracking ERROR based on phase)
2. Phase-free D-gated contact quality rewards
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_error_magnitude

from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import MotionCommand
from soccer.tasks.tracking.mdp.rewards_v35 import _compute_d_score, _get_kick_tracker
from soccer.tasks.tracking.mdp.phase_tracking_v36 import compute_tracking_multipliers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ══════════════════════════════════════════════════════════════════════════
# Modulated Tracking Rewards
# ══════════════════════════════════════════════════════════════════════════

def _get_body_indexes(command, body_names):
    if body_names is None:
        return list(range(len(command.cfg.body_names)))
    return [list(command.cfg.body_names).index(n) for n in body_names]


def phase_modulated_body_pos(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 0.3,
    body_names: list[str] | None = None,
    kick_foot_name: str = "right_ankle_roll_link",
    support_foot_name: str = "left_ankle_roll_link",
) -> torch.Tensor:
    """Body position tracking with phase-dependent error scaling.
    
    Instead of binary masking, scales the per-body squared error using the
    multipliers from phase_tracking_v36.py.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    mults = compute_tracking_multipliers(env, command_name)
    
    all_indices = _get_body_indexes(command, body_names)
    b_names = list(command.cfg.body_names) if body_names is None else list(body_names)

    # Compute raw squared error
    body_pos_relative_w = command.body_pos_relative_w[:, all_indices]
    robot_body_selected = command.robot_body_pos_w[:, all_indices]
    diff = robot_body_selected - body_pos_relative_w
    per_body_error = torch.sum(diff * diff, dim=-1)  # (N, num_bodies)

    # Apply multipliers to error
    scaled_error = torch.zeros_like(per_body_error)
    for local_idx, b_name in enumerate(b_names):
        if b_name == kick_foot_name:
            m = mults["swing_foot"]
        elif b_name == support_foot_name:
            m = mults["support_foot"]
        else:
            m = mults["body_pos"]
        # Multiply error by the multiplier (0.0 means 0 error -> max reward)
        scaled_error[:, local_idx] = per_body_error[:, local_idx] * m

    mean_error = scaled_error.mean(dim=-1)
    return torch.exp(-mean_error / (std ** 2))


def phase_modulated_body_ori(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 0.5,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Body orientation tracking with phase-dependent error scaling."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    mults = compute_tracking_multipliers(env, command_name)
    
    all_indices = _get_body_indexes(command, body_names)
    b_names = list(command.cfg.body_names) if body_names is None else list(body_names)

    # Compute raw squared error
    body_quat_relative_w = command.body_quat_relative_w[:, all_indices]
    robot_body_selected = command.robot_body_quat_w[:, all_indices]
    per_body_error = quat_error_magnitude(body_quat_relative_w, robot_body_selected) ** 2

    # Apply multipliers to error
    scaled_error = torch.zeros_like(per_body_error)
    for local_idx, b_name in enumerate(b_names):
        # Everything gets the body_ori multiplier here
        m = mults["body_ori"]
        scaled_error[:, local_idx] = per_body_error[:, local_idx] * m

    mean_error = scaled_error.mean(dim=-1)
    return torch.exp(-mean_error / (std ** 2))


# ══════════════════════════════════════════════════════════════════════════
# Phase-Free Contact Quality Rewards
# ══════════════════════════════════════════════════════════════════════════

def _resolve_contact_event(env: ManagerBasedRLEnv, command: MotionCommand, 
                           ball_sensor_name: str, horizontal_force_threshold: float,
                           foot_cfg: SceneEntityCfg | None):
    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)
    
    correct_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if foot_cfg is not None and torch.any(event.new_contact):
        foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
        if foot_info.env_ids.numel() > 0:
            valid_expectation = foot_info.expected >= 0
            correct = (foot_info.sides == foot_info.expected) & valid_expectation
            correct_mask[foot_info.env_ids] = correct
            
    return event.new_contact, correct_mask


def v36_strike_contact(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 10.0,
    foot_cfg: SceneEntityCfg | None = None,
    discriminator_path: str = "models/strike_discriminator.pt",
    gamma: float = 1.0,
) -> torch.Tensor:
    """Reward: correct foot * D_strike^gamma (One-shot at contact)"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    contact, correct_mask = _resolve_contact_event(
        env, command, ball_sensor_name, horizontal_force_threshold, foot_cfg)
    if torch.any(contact):
        _get_kick_tracker(command).record_expected_success(contact, correct_mask)
    
    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    valid = contact & correct_mask
    if valid.any():
        d_score = _compute_d_score(env, command, discriminator_path)
        reward[valid] = d_score[valid].pow(gamma)
        
    return reward


def v36_gated_ball_speed(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 1.2,
    velocity_threshold: float = 0.5,
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 10.0,
    foot_cfg: SceneEntityCfg | None = None,
    discriminator_path: str = "models/strike_discriminator.pt",
    gamma: float = 1.0,
) -> torch.Tensor:
    """Reward: D_gated ball speed on correct foot contact."""
    from soccer.tasks.tracking.mdp.rewards import ball_speed_reward
    base_reward = ball_speed_reward(
        env, command_name, std, velocity_threshold,
        ball_sensor_name, horizontal_force_threshold, foot_cfg,
    )
    if not base_reward.any():
        return base_reward
        
    command: MotionCommand = env.command_manager.get_term(command_name)
    d_score = _compute_d_score(env, command, discriminator_path)
    return base_reward * d_score.pow(gamma)


def v36_gated_direction(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 0.8,
    velocity_threshold: float = 0.5,
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 10.0,
    foot_cfg: SceneEntityCfg | None = None,
    discriminator_path: str = "models/strike_discriminator.pt",
    gamma: float = 1.0,
) -> torch.Tensor:
    """Reward: D_gated direction alignment on correct foot contact."""
    from soccer.tasks.tracking.mdp.rewards import ball_velocity_direction_alignment
    base_reward = ball_velocity_direction_alignment(
        env, command_name, std, velocity_threshold,
        ball_sensor_name, horizontal_force_threshold, foot_cfg,
    )
    if not base_reward.any():
        return base_reward
        
    command: MotionCommand = env.command_manager.get_term(command_name)
    d_score = _compute_d_score(env, command, discriminator_path)
    return base_reward * d_score.pow(gamma)


def v36_non_strike_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 10.0,
    discriminator_path: str = "models/strike_discriminator.pt",
) -> torch.Tensor:
    """Penalty: Any contact * (1 - D_strike). Replaces early_collision."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)
    
    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if event.new_contact.any():
        d_score = _compute_d_score(env, command, discriminator_path)
        reward[event.new_contact] = - (1.0 - d_score[event.new_contact])
        
    return reward


def v36_wrong_foot_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 10.0,
    foot_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalty: Contact made with the wrong foot/body part."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    contact, correct_mask = _resolve_contact_event(
        env, command, ball_sensor_name, horizontal_force_threshold, foot_cfg)
    
    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    wrong = contact & (~correct_mask)
    if wrong.any():
        reward[wrong] = -1.0
        
    return reward
