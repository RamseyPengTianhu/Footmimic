"""Anchor-based observation functions for the decoupled kick architecture.

These functions provide egocentric (self-centered) observations that do NOT
depend on absolute world coordinates.  They are designed to be used alongside
the existing ``observations.py`` functions without modifying them.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_apply, quat_inv

from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def anchor_ball_polar(
    env: ManagerBasedEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    """Ball position relative to the robot pelvis in polar-style coordinates.

    Returns a 3-D vector per env: ``(distance, cos_heading, sin_heading)`` where
    heading is the angle between the pelvis forward direction and the
    pelvis-to-ball vector projected onto the ground plane.

    This observation is **egocentric** — it is invariant to the absolute
    position/orientation of the robot on the field.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    ball_pos_w = soccer_ball.data.root_pos_w[:, :3]          # [N, 3]
    pelvis_pos_w = command.robot_pelvis_pos_w                 # [N, 3]
    pelvis_quat_w = command.robot_pelvis_quat_w               # [N, 4]

    # Delta in world frame, then rotate into pelvis-local frame.
    delta_w = ball_pos_w - pelvis_pos_w                       # [N, 3]
    delta_local = quat_apply(quat_inv(pelvis_quat_w), delta_w)  # [N, 3]

    # Polar decomposition on XY plane.
    dx = delta_local[:, 0]
    dy = delta_local[:, 1]
    dist = torch.norm(delta_local[:, :2], dim=-1).clamp(min=1e-4)
    cos_heading = dx / dist
    sin_heading = dy / dist

    return torch.stack([dist, cos_heading, sin_heading], dim=-1)  # [N, 3]


def anchor_ball_local(
    env: ManagerBasedEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    """Ball position relative to pelvis in local Cartesian coordinates (x, y, z).

    Same as ``constant_target_point_pos`` but explicitly decoupled from the
    original observation module for clarity.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    ball_pos_w = soccer_ball.data.root_pos_w[:, :3]
    pelvis_pos_w = command.robot_pelvis_pos_w
    pelvis_quat_w = command.robot_pelvis_quat_w

    delta_w = ball_pos_w - pelvis_pos_w
    return quat_apply(quat_inv(pelvis_quat_w), delta_w)  # [N, 3]


def ball_relative_feet(
    env: ManagerBasedEnv,
    command_name: str = "motion",
    left_foot_body: str = "left_ankle_roll_link",
    right_foot_body: str = "right_ankle_roll_link",
) -> torch.Tensor:
    """Ball position relative to each foot in foot-local frame.

    Returns a 6-D vector per env:
        ``[ball_rel_left_foot(3), ball_rel_right_foot(3)]``

    This gives the policy direct information about where the ball is
    relative to each foot, enabling adaptive foot placement.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot = command.robot
    soccer_ball = env.scene["soccer_ball"]
    ball_pos_w = soccer_ball.data.root_pos_w[:, :3]  # [N, 3]

    left_idx = robot.body_names.index(left_foot_body)
    right_idx = robot.body_names.index(right_foot_body)

    left_pos_w = robot.data.body_pos_w[:, left_idx]   # [N, 3]
    left_quat_w = robot.data.body_quat_w[:, left_idx]  # [N, 4]
    right_pos_w = robot.data.body_pos_w[:, right_idx]
    right_quat_w = robot.data.body_quat_w[:, right_idx]

    ball_rel_left = quat_apply(quat_inv(left_quat_w), ball_pos_w - left_pos_w)
    ball_rel_right = quat_apply(quat_inv(right_quat_w), ball_pos_w - right_pos_w)

    return torch.cat([ball_rel_left, ball_rel_right], dim=-1)  # [N, 6]


def kick_context(
    env: ManagerBasedEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    """Kick side indicator and phase progress.

    Returns a 2-D vector per env:
        ``[kick_side, phase_progress]``

    - kick_side: -1 = left, +1 = right, 0 = unknown
    - phase_progress: current time_step / motion_length, clipped to [0, 1]
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    # Kick side: left=-1, right=+1, unknown=0
    kick_leg = command.kick_leg  # 0=left, 1=right, -1=unknown
    kick_side = torch.where(kick_leg == 0, -1.0,
                torch.where(kick_leg == 1, 1.0, 0.0))

    # Phase progress
    t = command.time_steps.float()
    ml = command.motion_length.float().clamp(min=1.0)
    phase = (t / ml).clamp(0.0, 1.0)

    return torch.stack([kick_side, phase], dim=-1)  # [N, 2]


def target_direction_local(
    env: ManagerBasedEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    """Target kick direction in pelvis-local frame.

    Returns a 2-D vector per env: ``[cos_dir, sin_dir]``

    Direction is computed from current ball position to target destination,
    projected into the pelvis frame. Gives the policy information about
    WHERE to kick the ball.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    ball_pos_w = soccer_ball.data.root_pos_w[:, :2]   # [N, 2]
    dest_pos = command.target_destination_pos          # [N, 3]
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is not None:
        dest_w = dest_pos[:, :2] + env_origins[:, :2]
    else:
        dest_w = dest_pos[:, :2]

    # Direction from ball to destination in world frame
    dir_w = dest_w - ball_pos_w  # [N, 2]
    dist = torch.norm(dir_w, dim=-1, keepdim=True).clamp(min=1e-4)
    dir_w_norm = dir_w / dist  # [N, 2]

    # Rotate into pelvis frame (only XY heading)
    pelvis_quat_w = command.robot_pelvis_quat_w  # [N, 4]
    dir_3d = torch.cat([dir_w_norm, torch.zeros_like(dir_w_norm[:, :1])], dim=-1)
    dir_local = quat_apply(quat_inv(pelvis_quat_w), dir_3d)

    return dir_local[:, :2]  # [N, 2] = [cos_dir, sin_dir]

