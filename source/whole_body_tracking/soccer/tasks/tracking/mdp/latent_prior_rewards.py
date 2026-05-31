"""Rewards for latent-prior stage1."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_apply, quat_inv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _cauchy_reward(error: torch.Tensor, std: float) -> torch.Tensor:
    return 1.0 / (1.0 + (error / max(std, 1.0e-6)).square())


def latent_prior_feature_tracking(
    env: "ManagerBasedRLEnv",
    command_name: str = "motion",
    joint_pos_std: float = 0.45,
    joint_vel_std: float = 8.0,
    pelvis_height_std: float = 0.12,
    pelvis_lin_vel_std: float = 2.0,
    pelvis_ang_vel_std: float = 4.0,
    joint_pos_weight: float = 0.45,
    joint_vel_weight: float = 0.20,
    pelvis_height_weight: float = 0.15,
    pelvis_vel_weight: float = 0.20,
) -> torch.Tensor:
    """Track the compact frame decoded by the content CVAE."""
    command = env.command_manager.get_term(command_name)
    robot = command.robot
    target = command.prior_target_frame
    joint_count = int(command.prior_joint_count)

    target_joint_pos = target[:, :joint_count]
    target_joint_vel = target[:, joint_count : 2 * joint_count]
    target_pelvis_height = target[:, 2 * joint_count : 2 * joint_count + 1]
    target_pelvis_lin_vel = target[:, 2 * joint_count + 1 : 2 * joint_count + 4]
    target_pelvis_ang_vel = target[:, 2 * joint_count + 4 : 2 * joint_count + 7]

    pelvis_idx = robot.body_names.index("pelvis")
    joint_pos = robot.data.joint_pos[:, :joint_count]
    joint_vel = robot.data.joint_vel[:, :joint_count]
    pelvis_height = robot.data.body_pos_w[:, pelvis_idx, 2:3]
    pelvis_lin_vel = robot.data.body_lin_vel_w[:, pelvis_idx, :]
    pelvis_ang_vel = robot.data.body_ang_vel_w[:, pelvis_idx, :]

    if command.prior_feature_frame == "local":
        pelvis_quat_inv = quat_inv(robot.data.body_quat_w[:, pelvis_idx, :])
        pelvis_lin_vel = quat_apply(pelvis_quat_inv, pelvis_lin_vel)
        pelvis_ang_vel = quat_apply(pelvis_quat_inv, pelvis_ang_vel)

    joint_pos_reward = torch.exp(-torch.mean((joint_pos - target_joint_pos) ** 2, dim=-1) / (joint_pos_std**2))
    joint_vel_reward = torch.exp(-torch.mean((joint_vel - target_joint_vel) ** 2, dim=-1) / (joint_vel_std**2))
    pelvis_height_error = torch.abs(pelvis_height - target_pelvis_height).squeeze(-1)
    pelvis_height_reward = _cauchy_reward(pelvis_height_error, pelvis_height_std)
    pelvis_lin_reward = torch.exp(
        -torch.mean((pelvis_lin_vel - target_pelvis_lin_vel) ** 2, dim=-1) / (pelvis_lin_vel_std**2)
    )
    pelvis_ang_reward = torch.exp(
        -torch.mean((pelvis_ang_vel - target_pelvis_ang_vel) ** 2, dim=-1) / (pelvis_ang_vel_std**2)
    )
    pelvis_vel_reward = 0.5 * (pelvis_lin_reward + pelvis_ang_reward)

    if hasattr(command, "metrics"):
        command.metrics["latent_prior_joint_pos_reward"] = joint_pos_reward
        command.metrics["latent_prior_joint_vel_reward"] = joint_vel_reward
        command.metrics["latent_prior_pelvis_height_reward"] = pelvis_height_reward
        command.metrics["latent_prior_pelvis_vel_reward"] = pelvis_vel_reward
        command.metrics["latent_prior_joint_pos_error"] = torch.norm(joint_pos - target_joint_pos, dim=-1)
        command.metrics["latent_prior_joint_vel_error"] = torch.norm(joint_vel - target_joint_vel, dim=-1)
        command.metrics["latent_prior_pelvis_height_error"] = pelvis_height_error
        command.metrics["latent_prior_robot_pelvis_height"] = pelvis_height.squeeze(-1)
        command.metrics["latent_prior_target_pelvis_height"] = target_pelvis_height.squeeze(-1)

    return (
        joint_pos_weight * joint_pos_reward
        + joint_vel_weight * joint_vel_reward
        + pelvis_height_weight * pelvis_height_reward
        + pelvis_vel_weight * pelvis_vel_reward
    )


def latent_prior_pelvis_height_tracking(
    env: "ManagerBasedRLEnv",
    command_name: str = "motion",
    std: float = 0.35,
) -> torch.Tensor:
    """Dense height scaffold against the decoded local-prior target."""
    command = env.command_manager.get_term(command_name)
    robot = command.robot
    joint_count = int(command.prior_joint_count)
    target_height = command.prior_target_frame[:, 2 * joint_count : 2 * joint_count + 1]
    pelvis_idx = robot.body_names.index("pelvis")
    pelvis_height = robot.data.body_pos_w[:, pelvis_idx, 2:3]
    return _cauchy_reward(torch.abs(pelvis_height - target_height).squeeze(-1), std)
