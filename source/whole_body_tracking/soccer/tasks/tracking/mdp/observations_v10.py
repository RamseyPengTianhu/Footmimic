"""v10 Event-Conditioned Kick observation functions.

Provides MLP-compatible observations with explicit history buffers:
  - Group 1: Current proprioception (~64D)
  - Group 2: History buffer (~290D): joint_pos/vel ×3f, action ×3f, last_action
  - Group 3: Ball history (~30D): ball_pos_local ×10f
  - Group 4: Event condition (~8D)
  - Group 5: Ball-foot relation (~24D)
  - Group 6: Event-warped weak prior (~8D)

All coordinates are in ball-relative or kick-direction-relative frames,
NEVER in absolute world coordinates.
"""
from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_apply, quat_inv

from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import MotionCommand
from soccer.tasks.tracking.mdp.event_phase import (
    compute_event_phase,
    compute_event_obs,
    event_warped_ref_index,
    query_event_warped_weak_prior,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# ============================================================================
# History Buffer Manager
# ============================================================================

class ObsHistoryBuffer:
    """Manages rolling history buffers for MLP-based policies.

    Maintains fixed-length FIFO buffers for:
      - joint_pos: [N, history_len, 29]
      - joint_vel: [N, history_len, 29]
      - action: [N, history_len, 29]
      - ball_pos_local: [N, ball_history_len, 3]

    The buffer is updated each step and flattened for MLP input.
    """

    def __init__(
        self,
        num_envs: int,
        num_joints: int,
        proprio_history_len: int = 3,
        ball_history_len: int = 10,
        device: str = "cpu",
    ):
        self.num_envs = num_envs
        self.num_joints = num_joints
        self.proprio_history_len = proprio_history_len
        self.ball_history_len = ball_history_len
        self.device = device

        # Proprio history buffers: [N, history_len, dim]
        self.joint_pos_hist = torch.zeros(num_envs, proprio_history_len, num_joints, device=device)
        self.joint_vel_hist = torch.zeros(num_envs, proprio_history_len, num_joints, device=device)
        self.action_hist = torch.zeros(num_envs, proprio_history_len, num_joints, device=device)

        # Ball position history (pelvis-local): [N, ball_history_len, 3]
        self.ball_pos_local_hist = torch.zeros(num_envs, ball_history_len, 3, device=device)

    def update(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        last_action: torch.Tensor,
        ball_pos_local: torch.Tensor,
    ):
        """Push new data into history buffers (FIFO: oldest dropped)."""
        # Shift history: drop oldest (index 0), append newest at end
        self.joint_pos_hist = torch.cat([
            self.joint_pos_hist[:, 1:], joint_pos.unsqueeze(1)
        ], dim=1)
        self.joint_vel_hist = torch.cat([
            self.joint_vel_hist[:, 1:], joint_vel.unsqueeze(1)
        ], dim=1)
        self.action_hist = torch.cat([
            self.action_hist[:, 1:], last_action.unsqueeze(1)
        ], dim=1)
        self.ball_pos_local_hist = torch.cat([
            self.ball_pos_local_hist[:, 1:], ball_pos_local.unsqueeze(1)
        ], dim=1)

    def reset(self, env_ids: torch.Tensor):
        """Reset history for specified environments (on episode reset)."""
        if env_ids.numel() == 0:
            return
        self.joint_pos_hist[env_ids] = 0.0
        self.joint_vel_hist[env_ids] = 0.0
        self.action_hist[env_ids] = 0.0
        self.ball_pos_local_hist[env_ids] = 0.0

    def get_proprio_history_flat(self) -> torch.Tensor:
        """Return flattened proprio/action history: [N, history_len * 29 * 3]."""
        return torch.cat([
            self.joint_pos_hist.flatten(1),   # [N, 3*29=87]
            self.joint_vel_hist.flatten(1),   # [N, 87]
            self.action_hist.flatten(1),      # [N, 87]
        ], dim=-1)  # [N, 261]

    def get_ball_history_flat(self) -> torch.Tensor:
        """Return flattened ball position history: [N, ball_history_len * 3]."""
        return self.ball_pos_local_hist.flatten(1)  # [N, 30]


# ============================================================================
# Observation Functions (for ManagerBasedRLEnv observation terms)
# ============================================================================

def v10_event_condition(
    env: ManagerBasedEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    """Event condition observation: [N, 8].

    Returns: phase_onehot(4) + sin(πφ)(1) + cos(πφ)(1) + t2strike(1) + t2phase_end(1)

    Requires that the command term has `event_phase_id`, `event_phase_phi`,
    and `event_segment_bounds` attributes (set up by the v10 env).
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    phase_id = command.event_phase_id        # [N]
    phase_phi = command.event_phase_phi      # [N]
    time_steps = command.time_steps          # [N]
    bounds = command.event_segment_bounds    # [N, 4]

    return compute_event_obs(phase_id, phase_phi, time_steps, bounds)


def v10_ball_foot_relation(
    env: ManagerBasedEnv,
    command_name: str = "motion",
    swing_foot_body: str = "right_ankle_roll_link",
    support_foot_body: str = "left_ankle_roll_link",
) -> torch.Tensor:
    """Ball-foot relation observation: [N, ~22D].

    Returns raw 3D vectors + task-aligned contact-ready scalars, all in local frames.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot = command.robot
    soccer_ball = env.scene["soccer_ball"]

    ball_pos_w = soccer_ball.data.root_pos_w[:, :3]     # [N, 3]
    ball_vel_w = soccer_ball.data.root_lin_vel_w[:, :3]  # [N, 3]

    pelvis_pos_w = command.robot_pelvis_pos_w            # [N, 3]
    pelvis_quat_w = command.robot_pelvis_quat_w          # [N, 4]
    pelvis_quat_inv = quat_inv(pelvis_quat_w)

    # Foot positions in world frame
    swing_idx = robot.body_names.index(swing_foot_body)
    support_idx = robot.body_names.index(support_foot_body)

    swing_pos_w = robot.data.body_pos_w[:, swing_idx]
    swing_quat_w = robot.data.body_quat_w[:, swing_idx]
    support_pos_w = robot.data.body_pos_w[:, support_idx]
    support_quat_w = robot.data.body_quat_w[:, support_idx]

    # Raw 3D: ball relative to each foot (foot-local frame)
    ball_rel_swing = quat_apply(quat_inv(swing_quat_w), ball_pos_w - swing_pos_w)      # [N, 3]
    ball_rel_support = quat_apply(quat_inv(support_quat_w), ball_pos_w - support_pos_w) # [N, 3]

    # Ball relative to pelvis (pelvis-local)
    ball_rel_pelvis = quat_apply(pelvis_quat_inv, ball_pos_w - pelvis_pos_w)  # [N, 3]

    # Ball velocity in pelvis-local frame
    ball_vel_local = quat_apply(pelvis_quat_inv, ball_vel_w)  # [N, 3]

    # Desired kick direction (ball → target destination) in pelvis-local
    dest_pos = command.target_destination_pos  # [N, 3]
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is not None:
        dest_w = dest_pos[:, :2] + env_origins[:, :2]
    else:
        dest_w = dest_pos[:, :2]
    kick_dir_w = dest_w - ball_pos_w[:, :2]
    kick_dir_dist = torch.norm(kick_dir_w, dim=-1, keepdim=True).clamp(min=1e-4)
    kick_dir_w_norm = kick_dir_w / kick_dir_dist  # [N, 2]
    kick_dir_3d = torch.cat([kick_dir_w_norm, torch.zeros_like(kick_dir_w_norm[:, :1])], dim=-1)
    kick_dir_local = quat_apply(pelvis_quat_inv, kick_dir_3d)[:, :2]  # [N, 2]

    # Contact-ready scalar features
    swing_to_ball_w = ball_pos_w - swing_pos_w  # [N, 3]
    swing_foot_ball_dist = torch.norm(swing_to_ball_w[:, :2], dim=-1, keepdim=True)  # [N, 1]

    # Along-kick and lateral distances (project onto kick direction)
    kick_dir_2d = kick_dir_w_norm  # [N, 2]
    kick_perp_2d = torch.stack([-kick_dir_2d[:, 1], kick_dir_2d[:, 0]], dim=-1)  # [N, 2]

    swing_to_ball_2d = swing_to_ball_w[:, :2]
    swing_ball_longitudinal = (swing_to_ball_2d * kick_dir_2d).sum(dim=-1, keepdim=True)  # [N, 1]
    swing_ball_lateral = (swing_to_ball_2d * kick_perp_2d).sum(dim=-1, keepdim=True)      # [N, 1]

    support_to_ball_2d = (ball_pos_w - support_pos_w)[:, :2]
    support_ball_longitudinal = (support_to_ball_2d * kick_dir_2d).sum(dim=-1, keepdim=True)
    support_ball_lateral = (support_to_ball_2d * kick_perp_2d).sum(dim=-1, keepdim=True)

    # Swing foot velocity features
    swing_lin_vel_w = robot.data.body_lin_vel_w[:, swing_idx]  # [N, 3]
    swing_vel_along_kick = (swing_lin_vel_w[:, :2] * kick_dir_2d).sum(dim=-1, keepdim=True)  # [N, 1]

    swing_to_ball_dir = swing_to_ball_2d / swing_to_ball_2d.norm(dim=-1, keepdim=True).clamp(min=1e-4)
    swing_vel_to_ball_align = (swing_lin_vel_w[:, :2] * swing_to_ball_dir).sum(dim=-1, keepdim=True)  # [N, 1]

    # Ball velocity magnitude
    ball_vel_mag = torch.norm(ball_vel_w[:, :2], dim=-1, keepdim=True)  # [N, 1]

    return torch.cat([
        ball_rel_swing,              # 3D
        ball_rel_support,            # 3D
        ball_rel_pelvis,             # 3D
        ball_vel_local,              # 3D
        kick_dir_local,              # 2D
        swing_foot_ball_dist,        # 1D
        swing_ball_longitudinal,     # 1D
        swing_ball_lateral,          # 1D
        support_ball_lateral,        # 1D
        support_ball_longitudinal,   # 1D
        swing_vel_along_kick,        # 1D
        swing_vel_to_ball_align,     # 1D
        ball_vel_mag,                # 1D
    ], dim=-1)  # [N, 22D]


def v10_motor_prior(
    env: ManagerBasedEnv,
    command_name: str = "motion",
    swing_foot_body: str = "right_ankle_roll_link",
    support_foot_body: str = "left_ankle_roll_link",
) -> torch.Tensor:
    """Event-warped motor prior: [N, 40].
    
    Includes:
    - 8D weak prior (foot-ball relative offsets)
    - 29D joint delta
    - 3D base prior (1D height delta, 2D projected gravity delta)
    """
    from soccer.tasks.tracking.mdp.event_phase import (
        query_event_warped_joint_delta,
        query_event_warped_base_prior,
        query_event_warped_weak_prior,
        event_warped_ref_index
    )
    
    command = env.command_manager.get_term(command_name)
    
    current_phase_idx = command.event_phase_id
    phase_progress = command.event_phase_phi
    original_bounds = command.event_segment_bounds
    
    # 1. 8D Weak Prior
    swing_idx = command.robot.find_bodies(swing_foot_body)[0][0]
    support_idx = command.robot.find_bodies(support_foot_body)[0][0]
    
    ref_idx = event_warped_ref_index(current_phase_idx, phase_progress, original_bounds)
    
    weak_prior_8d = query_event_warped_weak_prior(
        ref_idx,
        motion=command.motion,
        motion_idx=command.motion_idx,
        swing_foot_body_idx=swing_idx,
        support_foot_body_idx=support_idx,
        ball_pos_in_motion=command.target_point_pos,
    )
    
    # 2. 29D Joint Delta
    joint_delta_29d = query_event_warped_joint_delta(
        env, command, current_phase_idx, phase_progress, original_bounds
    )
    
    # 3. 3D Base Prior
    base_prior_3d = query_event_warped_base_prior(
        env, command, current_phase_idx, phase_progress, original_bounds
    )
    
    return torch.cat([
        weak_prior_8d,    # 8D
        joint_delta_29d,  # 29D
        base_prior_3d     # 3D
    ], dim=-1)            # Total 40D
