"""Tracking relaxations for V3 + content-CVAE prior experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    if body_names is None:
        return list(range(len(command.cfg.body_names)))
    return [list(command.cfg.body_names).index(name) for name in body_names]


def _cg1(command: MotionCommand, margin: int) -> torch.Tensor:
    kick_frame = command.kick_frame
    has_annotation = kick_frame >= 0
    return has_annotation & (command.time_steps >= (kick_frame - margin))


def _kick_leg(command: MotionCommand) -> torch.Tensor:
    kick_leg = getattr(command, "kick_leg", None)
    if kick_leg is None:
        return torch.ones(command.time_steps.shape[0], dtype=torch.long, device=command.time_steps.device)
    return kick_leg.to(device=command.time_steps.device, dtype=torch.long)


def _tracking_scale(
    command: MotionCommand,
    selected_names: list[str],
    cg1: torch.Tensor,
    body_cg1_scale: float,
    kick_foot_cg1_scale: float,
    support_foot_cg1_scale: float,
    left_foot_name: str,
    right_foot_name: str,
) -> torch.Tensor:
    num_envs = command.time_steps.shape[0]
    scale = torch.ones(num_envs, len(selected_names), dtype=torch.float32, device=command.time_steps.device)
    scale[cg1] = body_cg1_scale

    leg = _kick_leg(command)
    left_kick = leg == 0
    right_kick = leg == 1

    if left_foot_name in selected_names:
        idx = selected_names.index(left_foot_name)
        scale[cg1 & left_kick, idx] = kick_foot_cg1_scale
        scale[cg1 & right_kick, idx] = support_foot_cg1_scale
    if right_foot_name in selected_names:
        idx = selected_names.index(right_foot_name)
        scale[cg1 & right_kick, idx] = kick_foot_cg1_scale
        scale[cg1 & left_kick, idx] = support_foot_cg1_scale

    return scale


def cg_modulated_body_pos(
    env: "ManagerBasedRLEnv",
    command_name: str = "motion",
    std: float = 0.3,
    body_names: list[str] | None = None,
    cg_margin: int = 5,
    body_cg1_scale: float = 0.5,
    kick_foot_cg1_scale: float = 0.05,
    support_foot_cg1_scale: float = 0.8,
    left_foot_name: str = "left_ankle_roll_link",
    right_foot_name: str = "right_ankle_roll_link",
) -> torch.Tensor:
    """V3 body tracking, but softened in the kick window.

    The CG/contact terms still decide whether the kick is useful.  This term
    only prevents frame-locked tracking from over-constraining strike geometry.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    indices = _body_indexes(command, body_names)
    selected_names = list(command.cfg.body_names) if body_names is None else list(body_names)

    diff = command.robot_body_pos_w[:, indices] - command.body_pos_relative_w[:, indices]
    per_body_error = torch.sum(diff * diff, dim=-1)

    cg1 = _cg1(command, cg_margin)
    scale = _tracking_scale(
        command,
        selected_names,
        cg1,
        body_cg1_scale,
        kick_foot_cg1_scale,
        support_foot_cg1_scale,
        left_foot_name,
        right_foot_name,
    )
    scaled_error = per_body_error * scale

    if hasattr(command, "metrics"):
        command.metrics["v3cvae_tracking_scale_mean"] = scale.mean(dim=-1)
        command.metrics["v3cvae_cg1"] = cg1.to(dtype=torch.float32)

    return torch.exp(-scaled_error.mean(dim=-1) / (std**2))


def cg_modulated_foot_pos(
    env: "ManagerBasedRLEnv",
    command_name: str = "motion",
    std: float = 0.3,
    foot_body_names: list[str] | None = None,
    cg_margin: int = 5,
    kick_foot_cg1_scale: float = 0.05,
    support_foot_cg1_scale: float = 0.8,
    left_foot_name: str = "left_ankle_roll_link",
    right_foot_name: str = "right_ankle_roll_link",
) -> torch.Tensor:
    """Foot tracking that keeps support-foot guidance but frees the swing foot."""
    if foot_body_names is None:
        foot_body_names = [left_foot_name, right_foot_name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    indices = _body_indexes(command, foot_body_names)

    diff = command.robot_body_pos_w[:, indices] - command.body_pos_relative_w[:, indices]
    per_foot_error = torch.sum(diff * diff, dim=-1)

    cg1 = _cg1(command, cg_margin)
    scale = _tracking_scale(
        command,
        list(foot_body_names),
        cg1,
        body_cg1_scale=1.0,
        kick_foot_cg1_scale=kick_foot_cg1_scale,
        support_foot_cg1_scale=support_foot_cg1_scale,
        left_foot_name=left_foot_name,
        right_foot_name=right_foot_name,
    )
    return torch.exp(-(per_foot_error * scale).mean(dim=-1) / (std**2))
