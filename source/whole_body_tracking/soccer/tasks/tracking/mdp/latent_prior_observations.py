"""Observation helpers for latent-prior controllers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_pelvis_height(env: "ManagerBasedEnv", command_name: str = "motion"):
    command = env.command_manager.get_term(command_name)
    return command.robot_pelvis_pos_w[:, 2:3]
