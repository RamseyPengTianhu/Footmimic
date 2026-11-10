"""Pelvis drift from the reference path, in the episode-fixed frame.

The reference pelvis XY is mapped once per episode through the reset transform (reference root -> actual root, yaw
alignment) recorded by ``DribbleCGMotionCommand``; the same transform places the first-contact ball.  The target
therefore never moves with the robot.  Without this, nothing in the clean-v0 Stage-1 terminations asks the robot to
actually travel along the reference (see DRIBBLE_HANDOFF.md 18.1: marching in place beat the ball-lost termination).
"""

from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply


def reference_pelvis_xy_episode(env, command_name: str = "motion") -> torch.Tensor:
    """``[N, 2]`` reference pelvis XY at the current frame, mapped through the reset transform."""
    cmd = env.command_manager.get_term(command_name)
    delta = cmd.anchor_pos_w - cmd.s2_episode_reference_root_w
    delta[:, 2] = 0.0
    return (cmd.s2_episode_actual_root_w + quat_apply(cmd.s2_episode_yaw_alignment, delta))[:, :2]


def path_drift(env, command_name: str = "motion") -> torch.Tensor:
    """``[N]`` XY distance between the robot pelvis and the episode-fixed reference pelvis."""
    cmd = env.command_manager.get_term(command_name)
    return torch.linalg.vector_norm(cmd.robot_anchor_pos_w[:, :2] - reference_pelvis_xy_episode(env, command_name), dim=-1)


def path_drift_termination(env, command_name: str = "motion", max_distance: float = 0.5) -> torch.Tensor:
    return path_drift(env, command_name) > max_distance
