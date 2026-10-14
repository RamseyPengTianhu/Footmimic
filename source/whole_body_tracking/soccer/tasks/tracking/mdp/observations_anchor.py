"""Anchor-based observation functions for dribbling / task-frame control.

Ball-relative observations use the **task frame** (env-local axes parallel to
world: +X forward, +Y lateral, +Z up), matching velocity rewards and spawn logic.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_apply, quat_inv, yaw_quat

from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import MotionCommand
from soccer.tasks.tracking.mdp.task_frame import task_forward_offset, task_lateral_offset

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def anchor_ball_polar(
    env: ManagerBasedEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    """Ball offset from pelvis in task-frame polar coordinates.

    Returns ``(distance, cos_heading, sin_heading)`` on the ground plane where
    heading is the angle from task +X to the pelvis→ball vector.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    ball_pos_w = soccer_ball.data.root_pos_w[:, :3]
    pelvis_pos_w = command.robot_pelvis_pos_w

    dx = task_forward_offset(ball_pos_w, pelvis_pos_w)
    dy = task_lateral_offset(ball_pos_w, pelvis_pos_w)
    dist = torch.norm(torch.stack([dx, dy], dim=-1), dim=-1).clamp(min=1e-4)
    cos_heading = dx / dist
    sin_heading = dy / dist

    return torch.stack([dist, cos_heading, sin_heading], dim=-1)


def anchor_ball_pelvis_local_polar(
    env: ManagerBasedEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    """Ball offset in the current pelvis-yaw local polar frame.

    The output shape intentionally matches ``anchor_ball_polar`` so local
    curriculum stages preserve the 163-D actor contract while eliminating the
    non-deployable task/world-frame direction.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    # Replay-only causal ablation.  Keep the physical ball at its configured
    # position, but let the actor see a stable, far, pelvis-local proxy.  This
    # preserves the 3-D observation contract while testing whether the actual
    # near-foot ball observation is what destabilizes the Stage-1 policy.
    cfg = getattr(command, "cfg", None)
    obs_mode = str(getattr(cfg, "dribble_cg_replay_ball_observation_mode", "actual")).lower().strip()
    if obs_mode == "fixed_front":
        distance = max(
            1.0e-4,
            float(getattr(cfg, "dribble_cg_replay_ball_observation_distance", 0.80)),
        )
        distance_tensor = torch.full(
            (command.num_envs,), distance, device=env.device, dtype=torch.float32
        )
        cos_heading = torch.ones_like(distance_tensor)
        sin_heading = torch.zeros_like(distance_tensor)
        return torch.stack((distance_tensor, cos_heading, sin_heading), dim=-1)
    soccer_ball = env.scene["soccer_ball"]
    offset_b = quat_apply(
        quat_inv(yaw_quat(command.robot_pelvis_quat_w)),
        soccer_ball.data.root_pos_w[:, :3] - command.robot_pelvis_pos_w,
    )
    distance = torch.norm(offset_b[:, :2], dim=-1).clamp(min=1.0e-4)
    return torch.stack((distance, offset_b[:, 0] / distance, offset_b[:, 1] / distance), dim=-1)


def s2_sparse_ball_reference_context(
    env: ManagerBasedEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    """Critic-only sparse contact-anchor context for S2.

    The 19-D result is::

        current_anchor_b[3], next_anchor_b[3],
        time_to_current, time_to_next,
        current_foot_one_hot[2], current_surface_one_hot[2],
        current_valid, next_valid,
        reference_event_eligible, cursor_contact_succeeded,
        predecessor_complete, cursor_progress, sequence_failed

    ``current`` is the reward-owned sequence cursor, so it remains on O1 when
    O1 was missed even after reference time reaches O2. ``next`` is the event
    after that cursor. Both positions are expressed in the current
    physical pelvis-yaw frame.  This exposes the sparse training target and
    event phase to the asymmetric critic without leaking it into the actor or
    constructing a frame-by-frame reference ball trajectory/velocity.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    num_envs = env.num_envs
    device = env.device
    zeros = torch.zeros(num_envs, device=device, dtype=torch.float32)
    invalid = torch.full((num_envs,), -1, device=device, dtype=torch.long)

    required = (
        "s2_contact_event_id_ref",
        "s2_event_reference",
        "s2_contact_anchor_target_w",
        "s2_next_touch_target_w",
    )
    if any(not hasattr(command, name) for name in required):
        return torch.zeros(num_envs, 19, device=device, dtype=torch.float32)

    active_event = command.s2_contact_event_id_ref.to(torch.long)
    first_event = command.s2_episode_first_event_index.to(torch.long)
    selected_count = command.s2_episode_contact_count.to(torch.long)
    current_event = command.s2_sequence_cursor.to(torch.long)
    current_frame, current_end, current_foot, current_surface, cursor_valid = (
        command.s2_event_reference(current_event)
    )

    current_anchor_w, current_valid = command.s2_contact_anchor_target_w(
        current_event
    )
    ball_pos_w = env.scene["soccer_ball"].data.root_pos_w[:, :3]
    next_anchor_w, next_frame, time_to_next, next_valid = (
        command.s2_next_touch_target_w(current_event, ball_pos_w)
    )
    locked_next_anchor_w = getattr(command, "s2_locked_next_anchor_w", None)
    locked_next_anchor_valid = getattr(
        command, "s2_locked_next_anchor_valid", torch.zeros_like(next_valid)
    ).to(torch.bool)
    if isinstance(locked_next_anchor_w, torch.Tensor):
        next_anchor_w = torch.where(
            locked_next_anchor_valid[:, None], locked_next_anchor_w, next_anchor_w
        )
        next_valid = next_valid.to(torch.bool) | locked_next_anchor_valid
    current_valid = (
        current_valid.to(torch.bool)
        & cursor_valid
        & (current_event >= 0)
        & (current_frame >= 0)
    )
    next_valid = next_valid.to(torch.bool) & current_valid

    pelvis_pos_w = command.robot_pelvis_pos_w
    pelvis_yaw_inv = quat_inv(yaw_quat(command.robot_pelvis_quat_w))
    current_anchor_b = quat_apply(
        pelvis_yaw_inv, current_anchor_w - pelvis_pos_w
    )
    next_anchor_b = quat_apply(pelvis_yaw_inv, next_anchor_w - pelvis_pos_w)
    current_anchor_b = torch.where(
        current_valid[:, None], current_anchor_b, torch.zeros_like(current_anchor_b)
    )
    next_anchor_b = torch.where(
        next_valid[:, None], next_anchor_b, torch.zeros_like(next_anchor_b)
    )

    fps = max(float(command.motion.fps.reshape(-1)[0]), 1.0e-6)
    time_to_current = (
        current_frame - command.time_steps.to(torch.long)
    ).to(torch.float32) / fps
    # Bound terminal/reset outliers while preserving the signed in-window
    # phase for the active event.
    time_to_current = torch.where(
        current_valid,
        torch.clamp(time_to_current, min=-1.0, max=5.0),
        zeros,
    )
    time_to_next = torch.where(
        next_valid,
        torch.clamp(time_to_next.to(torch.float32), min=0.0, max=5.0),
        zeros,
    )

    foot_valid = current_valid & ((current_foot == 0) | (current_foot == 1))
    surface_valid = current_valid & (
        (current_surface == 0) | (current_surface == 1)
    )
    safe_foot = torch.where(foot_valid, current_foot, invalid).clamp(min=0)
    safe_surface = torch.where(
        surface_valid, current_surface, invalid
    ).clamp(min=0)
    foot_one_hot = torch.nn.functional.one_hot(safe_foot, num_classes=2).to(
        torch.float32
    ) * foot_valid[:, None].to(torch.float32)
    surface_one_hot = torch.nn.functional.one_hot(
        safe_surface, num_classes=2
    ).to(torch.float32) * surface_valid[:, None].to(torch.float32)

    reference_event_eligible = (
        current_valid & (active_event == current_event)
    ).to(torch.float32)
    cursor_contact_succeeded = getattr(
        command, "s2_sequence_contact_succeeded", torch.zeros_like(current_valid)
    ).to(torch.float32)
    predecessor_complete = (
        current_valid & (current_event > first_event)
    ).to(torch.float32)
    cursor_progress = torch.where(
        current_valid & (selected_count > 0),
        (current_event - first_event).to(torch.float32)
        / selected_count.clamp(min=1).to(torch.float32),
        zeros,
    )
    sequence_failed = (
        current_valid
        & (command.time_steps.to(torch.long) > current_end)
        & ~getattr(
            command,
            "s2_sequence_contact_succeeded",
            torch.zeros_like(current_valid),
        ).to(torch.bool)
    ).to(torch.float32)

    return torch.cat(
        (
            current_anchor_b,
            next_anchor_b,
            time_to_current[:, None],
            time_to_next[:, None],
            foot_one_hot,
            surface_one_hot,
            current_valid[:, None].to(torch.float32),
            next_valid[:, None].to(torch.float32),
            reference_event_eligible[:, None],
            cursor_contact_succeeded[:, None],
            predecessor_complete[:, None],
            cursor_progress[:, None],
            sequence_failed[:, None],
        ),
        dim=-1,
    )


def anchor_ball_local(
    env: ManagerBasedEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    """Ball position relative to pelvis in task-frame Cartesian coordinates (x, y, z)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    ball_pos_w = soccer_ball.data.root_pos_w[:, :3]
    pelvis_pos_w = command.robot_pelvis_pos_w

    delta = ball_pos_w - pelvis_pos_w
    return delta
