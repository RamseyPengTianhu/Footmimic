"""v10 Event-Conditioned Kick reward functions.

Phase-aware rewards that replace hard trajectory tracking with interaction-conditioned signals:
  - r_foot_ball_relative: spatial relationship between feet and ball per event phase
  - r_contact_graph_match: actual vs desired contact events per event phase
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_inv

from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import MotionCommand
from soccer.tasks.tracking.mdp.event_phase import (
    PHASE_APPROACH, PHASE_PRESTRIKE, PHASE_STRIKE, PHASE_FOLLOWTHRU,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def r_foot_ball_relative(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    swing_foot_body: str = "right_ankle_roll_link",
    support_foot_body: str = "left_ankle_roll_link",
    std_prestrike: float = 0.3,
    std_strike: float = 0.15,
    std_support: float = 0.25,
) -> torch.Tensor:
    """Phase-aware foot-ball relative position reward.

    Rewards vary by event phase:
    - Prestrike: swing foot behind ball, support foot planted laterally
    - Strike: swing foot approaching ball, velocity aligned with target
    - Followthru: mild stability reward
    - Approach: no foot-ball constraint (tracking is sufficient)

    Returns: [N] reward values in [0, 1].
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot = command.robot
    soccer_ball = env.scene["soccer_ball"]

    phase_id = command.event_phase_id  # [N]
    ball_pos_w = soccer_ball.data.root_pos_w[:, :3]

    swing_idx = robot.body_names.index(swing_foot_body)
    support_idx = robot.body_names.index(support_foot_body)
    swing_pos_w = robot.data.body_pos_w[:, swing_idx]
    support_pos_w = robot.data.body_pos_w[:, support_idx]

    # Compute kick direction (ball → destination)
    dest_pos = command.target_destination_pos
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is not None:
        dest_w = dest_pos[:, :2] + env_origins[:, :2]
    else:
        dest_w = dest_pos[:, :2]
    kick_dir_w = dest_w - ball_pos_w[:, :2]
    kick_dir_dist = torch.norm(kick_dir_w, dim=-1, keepdim=True).clamp(min=1e-4)
    kick_dir_2d = kick_dir_w / kick_dir_dist  # [N, 2]

    # Ball-relative foot positions
    swing_to_ball = ball_pos_w[:, :2] - swing_pos_w[:, :2]  # [N, 2]
    support_to_ball = ball_pos_w[:, :2] - support_pos_w[:, :2]

    # Along-kick and lateral decomposition
    kick_perp_2d = torch.stack([-kick_dir_2d[:, 1], kick_dir_2d[:, 0]], dim=-1)

    swing_longitudinal = (swing_to_ball * kick_dir_2d).sum(dim=-1)
    support_lateral = (support_to_ball * kick_perp_2d).sum(dim=-1)

    # Swing foot distance to ball
    swing_ball_dist = torch.norm(swing_to_ball, dim=-1)

    # Swing foot velocity alignment with kick direction
    swing_vel_w = robot.data.body_lin_vel_w[:, swing_idx, :2]
    swing_vel_along_kick = (swing_vel_w * kick_dir_2d).sum(dim=-1)

    N = phase_id.shape[0]
    device = phase_id.device
    reward = torch.zeros(N, device=device)

    # Prestrike: swing behind ball (longitudinal > 0), support laterally placed
    prestrike_mask = phase_id == PHASE_PRESTRIKE
    if prestrike_mask.any():
        # Swing should be behind ball (positive longitudinal means ball is ahead)
        r_swing_behind = torch.exp(-torch.clamp(-swing_longitudinal[prestrike_mask], min=0) ** 2 / (2 * std_prestrike ** 2))
        # Support foot lateral offset should be ~0.15-0.25m
        desired_lateral = 0.20
        r_support_lat = torch.exp(-(support_lateral[prestrike_mask].abs() - desired_lateral) ** 2 / (2 * std_support ** 2))
        reward[prestrike_mask] = 0.5 * r_swing_behind + 0.5 * r_support_lat

    # Strike: swing close to ball + velocity aligned
    strike_mask = phase_id == PHASE_STRIKE
    if strike_mask.any():
        r_distance = torch.exp(-swing_ball_dist[strike_mask] ** 2 / (2 * std_strike ** 2))
        r_vel_align = torch.clamp(swing_vel_along_kick[strike_mask] / 3.0, 0, 1)  # normalize by ~3 m/s
        reward[strike_mask] = 0.6 * r_distance + 0.4 * r_vel_align

    # Followthru: mild reward for not falling (handled by stability rewards)
    followthru_mask = phase_id == PHASE_FOLLOWTHRU
    if followthru_mask.any():
        reward[followthru_mask] = 0.5  # constant baseline, stability comes from other rewards

    # Approach: no specific foot-ball constraint
    approach_mask = phase_id == PHASE_APPROACH
    if approach_mask.any():
        reward[approach_mask] = 0.5  # neutral

    return reward


def r_contact_graph_match(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "ball_contact_sensor",
    foot_sensor_name: str = "foot_contact_sensor",
    support_foot_body: str = "left_ankle_roll_link",
) -> torch.Tensor:
    """Contact graph matching reward: actual contacts vs event-phase-desired contacts.

    Desired contact graph per phase:
    - Approach: no ball contact, support foot grounded
    - Prestrike: no ball contact, support foot planted
    - Strike: ball contact expected
    - Followthru: no further ball contact

    Returns: [N] reward in [0, 1].
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    phase_id = command.event_phase_id  # [N]

    N = phase_id.shape[0]
    device = phase_id.device
    reward = torch.ones(N, device=device) * 0.5  # baseline

    # Detect ball contact from kick contact tracker
    tracker = command.kick_contact_tracker
    event = tracker.detect(
        command,
        ball_sensor_name=ball_sensor_name,
        horizontal_force_threshold=5.0,
    )
    has_ball_contact = event.new_contact  # [N] bool

    # Approach & Prestrike: penalize early ball contact
    no_contact_desired = (phase_id == PHASE_APPROACH) | (phase_id == PHASE_PRESTRIKE)
    early_contact = no_contact_desired & has_ball_contact
    reward[early_contact] = 0.0  # strong penalty for early collision

    # Strike: reward ball contact
    strike_mask = phase_id == PHASE_STRIKE
    strike_with_contact = strike_mask & has_ball_contact
    strike_no_contact = strike_mask & ~has_ball_contact
    reward[strike_with_contact] = 1.0
    reward[strike_no_contact] = 0.3  # waiting for contact

    # Followthru: mild penalty for late contact
    followthru_late = (phase_id == PHASE_FOLLOWTHRU) & has_ball_contact
    reward[followthru_late] = 0.3  # not ideal but not catastrophic

    return reward
