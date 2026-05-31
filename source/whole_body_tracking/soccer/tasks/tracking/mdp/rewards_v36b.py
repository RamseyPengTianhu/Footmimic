"""V3.6b ball-ready gated strike rewards.

The main change from v36a is that strike/contact credit is tied to the ball
being in a hittable region and the support foot being planted.  This targets
the failure mode where the policy swings at reference time, misses, then earns
reward from a much later accidental contact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_error_magnitude

from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import MotionCommand
from soccer.tasks.tracking.mdp.rewards_v35 import _compute_d_score, _get_kick_tracker
from soccer.tasks.tracking.mdp.phase_tracking_v36b import compute_tracking_multipliers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    if body_names is None:
        return list(range(len(command.cfg.body_names)))
    return [list(command.cfg.body_names).index(name) for name in body_names]


def phase_modulated_body_pos(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 0.3,
    body_names: list[str] | None = None,
    kick_foot_name: str = "right_ankle_roll_link",
    support_foot_name: str = "left_ankle_roll_link",
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mults = compute_tracking_multipliers(env, command_name)
    indices = _get_body_indexes(command, body_names)
    selected_names = list(command.cfg.body_names) if body_names is None else list(body_names)

    diff = command.robot_body_pos_w[:, indices] - command.body_pos_relative_w[:, indices]
    per_body_error = torch.sum(diff * diff, dim=-1)
    scaled_error = torch.zeros_like(per_body_error)

    for local_idx, body_name in enumerate(selected_names):
        if body_name == kick_foot_name:
            mult = mults["swing_foot"]
        elif body_name == support_foot_name:
            mult = mults["support_foot"]
        else:
            mult = mults["body_pos"]
        scaled_error[:, local_idx] = per_body_error[:, local_idx] * mult

    return torch.exp(-scaled_error.mean(dim=-1) / (std**2))


def phase_modulated_body_ori(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 0.5,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mults = compute_tracking_multipliers(env, command_name)
    indices = _get_body_indexes(command, body_names)

    err = quat_error_magnitude(command.body_quat_relative_w[:, indices], command.robot_body_quat_w[:, indices]) ** 2
    return torch.exp(-(err * mults["body_ori"].unsqueeze(-1)).mean(dim=-1) / (std**2))


def _soft_upper(x: torch.Tensor, hi: float, std: float) -> torch.Tensor:
    excess = (x - hi).clamp(min=0.0)
    return torch.exp(-(excess * excess) / (std**2))


def _soft_lower(x: torch.Tensor, lo: float, std: float) -> torch.Tensor:
    excess = (lo - x).clamp(min=0.0)
    return torch.exp(-(excess * excess) / (std**2))


def _soft_range(x: torch.Tensor, lo: float, hi: float, std: float) -> torch.Tensor:
    return _soft_lower(x, lo, std) * _soft_upper(x, hi, std)


def _quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _get_ball_pos_w(env: ManagerBasedRLEnv, command: MotionCommand) -> torch.Tensor:
    if command.soccer_ball is not None:
        try:
            return command.soccer_ball.data.root_pos_w
        except Exception:
            pass
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is not None:
        return command.soccer_ball_pos + env_origins
    return command.soccer_ball_pos


def _get_kick_dir(env: ManagerBasedRLEnv, command: MotionCommand, ball_pos_w: torch.Tensor) -> torch.Tensor:
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is not None:
        dest_w = command.target_destination_pos + env_origins
    else:
        dest_w = command.target_destination_pos
    direction = dest_w[:, :2] - ball_pos_w[:, :2]
    return direction / torch.norm(direction, dim=-1, keepdim=True).clamp(min=1e-6)


def _get_body_pair_indices(env: ManagerBasedRLEnv, command: MotionCommand, kick_foot_name: str, support_foot_name: str):
    cache_name = f"_v36b_body_pair_{kick_foot_name}_{support_foot_name}"
    cache = getattr(env, cache_name, None)
    if cache is None:
        cache = (
            command.robot.body_names.index(kick_foot_name),
            command.robot.body_names.index(support_foot_name),
        )
        setattr(env, cache_name, cache)
    return cache


def _timing_gate(
    command: MotionCommand,
    early_grace: int = 20,
    late_grace: int = 55,
    early_decay: float = 12.0,
    late_decay: float = 25.0,
) -> torch.Tensor:
    kf = command.kick_frame
    t = command.time_steps
    has_annotation = kf >= 0
    dt = (t - kf).float()
    early_excess = (-float(early_grace) - dt).clamp(min=0.0)
    late_excess = (dt - float(late_grace)).clamp(min=0.0)
    gate = torch.exp(-(early_excess * early_excess) / (early_decay**2))
    gate = gate * torch.exp(-(late_excess * late_excess) / (late_decay**2))
    return torch.where(has_annotation, gate, torch.ones_like(gate))


def _strike_geometry(
    env: ManagerBasedRLEnv,
    command: MotionCommand,
    kick_foot_name: str,
    support_foot_name: str,
    near_ball_dist: float,
    near_ball_temp: float,
    kick_dist_std: float,
    kick_height_min: float,
    kick_height_max: float,
    kick_height_std: float,
    support_lat_min: float,
    support_lat_max: float,
    support_long_min: float,
    support_long_max: float,
    support_region_std: float,
    support_vel_std: float,
    support_height_max: float,
    support_height_std: float,
    support_yaw_std: float,
) -> dict[str, torch.Tensor]:
    robot = command.robot
    kick_idx, support_idx = _get_body_pair_indices(env, command, kick_foot_name, support_foot_name)

    ball_pos_w = _get_ball_pos_w(env, command)
    kick_dir = _get_kick_dir(env, command, ball_pos_w)
    side_dir = torch.stack((-kick_dir[:, 1], kick_dir[:, 0]), dim=-1)

    kick_pos = robot.data.body_pos_w[:, kick_idx]
    kick_vel = robot.data.body_lin_vel_w[:, kick_idx]
    support_pos = robot.data.body_pos_w[:, support_idx]
    support_vel = robot.data.body_lin_vel_w[:, support_idx]
    support_quat = robot.data.body_quat_w[:, support_idx]

    kick_rel_xy = kick_pos[:, :2] - ball_pos_w[:, :2]
    ball_to_kick = ball_pos_w[:, :2] - kick_pos[:, :2]
    support_rel_xy = support_pos[:, :2] - ball_pos_w[:, :2]

    kick_dist_xy = torch.norm(kick_rel_xy, dim=-1)
    ball_to_kick_unit = ball_to_kick / torch.norm(ball_to_kick, dim=-1, keepdim=True).clamp(min=1e-6)
    kick_speed_xy = torch.norm(kick_vel[:, :2], dim=-1)
    closing_speed = torch.sum(kick_vel[:, :2] * ball_to_kick_unit, dim=-1)
    kick_long = torch.sum(kick_rel_xy * kick_dir, dim=-1)
    kick_lat_abs = torch.abs(torch.sum(kick_rel_xy * side_dir, dim=-1))

    kick_leg = command.kick_leg
    side_sign = torch.where(
        kick_leg == 0,
        torch.full((env.num_envs,), -1.0, device=env.device),
        torch.ones(env.num_envs, device=env.device),
    )
    support_lat = torch.sum(support_rel_xy * side_dir, dim=-1) * side_sign
    support_long = torch.sum(support_rel_xy * kick_dir, dim=-1)

    pelvis_dist = torch.norm(command.robot_anchor_pos_w[:, :2] - ball_pos_w[:, :2], dim=-1)
    near_ball = torch.sigmoid((near_ball_dist - pelvis_dist) / near_ball_temp)

    kick_dist_score = torch.exp(-(kick_dist_xy * kick_dist_xy) / (kick_dist_std**2))
    kick_height_score = _soft_range(kick_pos[:, 2], kick_height_min, kick_height_max, kick_height_std)

    support_region = _soft_range(support_lat, support_lat_min, support_lat_max, support_region_std)
    support_region = support_region * _soft_range(support_long, support_long_min, support_long_max, support_region_std)

    support_speed = torch.norm(support_vel[:, :2], dim=-1)
    support_vel_score = torch.exp(-(support_speed * support_speed) / (support_vel_std**2))
    support_height_score = _soft_upper(support_pos[:, 2], support_height_max, support_height_std)

    support_yaw = _quat_to_yaw(support_quat)
    desired_yaw = torch.atan2(kick_dir[:, 1], kick_dir[:, 0])
    yaw_err = torch.atan2(torch.sin(support_yaw - desired_yaw), torch.cos(support_yaw - desired_yaw)).abs()
    support_yaw_score = torch.exp(-(yaw_err * yaw_err) / (support_yaw_std**2))

    # Yaw matters, but use it as a soft factor rather than a hard veto.  The
    # support foot should face the kick direction, while still allowing small
    # adjustments from ball randomization.
    plant_score = support_region * support_vel_score * support_height_score * (0.5 + 0.5 * support_yaw_score)
    ready_score = kick_dist_score * kick_height_score * plant_score

    if hasattr(command, "metrics"):
        command.metrics["v36b_ready_score"] = ready_score.detach()
        command.metrics["v36b_kick_ball_dist_xy"] = kick_dist_xy.detach()
        command.metrics["v36b_support_lat"] = support_lat.detach()
        command.metrics["v36b_support_long"] = support_long.detach()
        command.metrics["v36b_support_speed"] = support_speed.detach()

    return {
        "ready_score": ready_score.clamp(0.0, 1.0),
        "plant_score": plant_score.clamp(0.0, 1.0),
        "near_ball": near_ball.clamp(0.0, 1.0),
        "kick_dist_xy": kick_dist_xy,
        "kick_height": kick_pos[:, 2],
        "kick_long": kick_long,
        "kick_lat_abs": kick_lat_abs,
        "kick_speed_xy": kick_speed_xy,
        "closing_speed": closing_speed,
        "support_lat": support_lat,
        "support_long": support_long,
        "support_yaw_score": support_yaw_score,
        "support_speed": support_speed,
    }


def _default_geometry_kwargs(params: dict) -> dict:
    defaults = {
        "near_ball_dist": 1.15,
        "near_ball_temp": 0.20,
        "kick_dist_std": 0.28,
        "kick_height_min": 0.02,
        "kick_height_max": 0.45,
        "kick_height_std": 0.12,
        "support_lat_min": 0.16,
        "support_lat_max": 0.58,
        "support_long_min": -0.70,
        "support_long_max": 0.10,
        "support_region_std": 0.18,
        "support_vel_std": 0.45,
        "support_height_max": 0.16,
        "support_height_std": 0.10,
        "support_yaw_std": 0.75,
    }
    defaults.update(params)
    return defaults


def _get_empty_swing_state(env: ManagerBasedRLEnv, command: MotionCommand) -> torch.Tensor:
    prefix = getattr(command, "_state_prefix", "_motion")
    state_name = f"{prefix}_v36b_empty_swing_seen"
    state = getattr(env, state_name, None)
    if state is None or state.shape[0] != env.num_envs:
        state = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        state = state.to(device=env.device, dtype=torch.bool)
    setattr(env, state_name, state)
    return state


def _get_bool_state(env: ManagerBasedRLEnv, command: MotionCommand, suffix: str) -> torch.Tensor:
    prefix = getattr(command, "_state_prefix", "_motion")
    state_name = f"{prefix}_{suffix}"
    state = getattr(env, state_name, None)
    if state is None or state.shape[0] != env.num_envs:
        state = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        state = state.to(device=env.device, dtype=torch.bool)
    setattr(env, state_name, state)
    return state


def _get_float_state(
    env: ManagerBasedRLEnv,
    command: MotionCommand,
    suffix: str,
    default: float = 0.0,
) -> torch.Tensor:
    prefix = getattr(command, "_state_prefix", "_motion")
    state_name = f"{prefix}_{suffix}"
    state = getattr(env, state_name, None)
    if state is None or state.shape[0] != env.num_envs:
        state = torch.full((env.num_envs,), default, dtype=torch.float32, device=env.device)
    else:
        state = state.to(device=env.device, dtype=torch.float32)
    setattr(env, state_name, state)
    return state


def _update_attempt_lifecycle(
    env: ManagerBasedRLEnv,
    command: MotionCommand,
    geom: dict[str, torch.Tensor],
    contact: torch.Tensor | None = None,
    attempt_speed_threshold: float = 2.0,
    attempt_closing_speed: float = 0.5,
    attempt_max_foot_ball_dist: float = 0.9,
    attempt_min_kick_height: float = 0.02,
    attempt_max_kick_height: float = 0.65,
    attempt_near_ball_score: float = 0.5,
    attempt_early_grace: int = 25,
    attempt_window: int = 18,
) -> dict[str, torch.Tensor]:
    started = _get_bool_state(env, command, "v36b_attempt_started")
    hit = _get_bool_state(env, command, "v36b_attempt_hit")
    missed = _get_bool_state(env, command, "v36b_attempt_missed")
    late_fallback = _get_bool_state(env, command, "v36b_late_fallback")
    miss_awarded = _get_bool_state(env, command, "v36b_attempt_miss_awarded")
    frame = _get_float_state(env, command, "v36b_attempt_frame", default=-1.0)
    attempt_dist = _get_float_state(env, command, "v36b_attempt_dist_xy", default=0.0)

    if contact is None:
        contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        contact = contact.to(device=env.device, dtype=torch.bool)

    t = command.time_steps.float()
    kf = command.kick_frame
    after_attempt_time = (kf < 0) | (t >= (kf.float() - float(attempt_early_grace)))

    tracker = _get_kick_tracker(command)
    no_prior_contact = (~tracker.get_contact_awarded()) | contact
    attempt_start = (
        (~started)
        & no_prior_contact
        & after_attempt_time
        & (geom["near_ball"] > attempt_near_ball_score)
        & (geom["kick_dist_xy"] <= attempt_max_foot_ball_dist)
        & (geom["kick_height"] >= attempt_min_kick_height)
        & (geom["kick_height"] <= attempt_max_kick_height)
        & (geom["kick_speed_xy"] >= attempt_speed_threshold)
        & (geom["closing_speed"] >= attempt_closing_speed)
    )
    if torch.any(attempt_start):
        started[attempt_start] = True
        frame[attempt_start] = t[attempt_start]
        attempt_dist[attempt_start] = geom["kick_dist_xy"][attempt_start]

    elapsed = t - frame
    hit_now = contact & started & (~missed) & (elapsed <= float(attempt_window))
    if torch.any(hit_now):
        hit[hit_now] = True

    missed_now = started & (~hit) & (~missed) & (elapsed > float(attempt_window))
    if torch.any(missed_now):
        missed[missed_now] = True
        _get_empty_swing_state(env, command)[missed_now] = True

    late_now = contact & missed & (~hit)
    if torch.any(late_now):
        late_fallback[late_now] = True

    if hasattr(command, "metrics"):
        command.metrics["v36b_attempt_started"] = started.float()
        command.metrics["v36b_attempt_hit"] = hit.float()
        command.metrics["v36b_attempt_missed"] = missed.float()
        command.metrics["v36b_late_fallback"] = late_fallback.float()
        command.metrics["v36b_attempt_dist_xy"] = attempt_dist.detach()

    return {
        "started": started,
        "hit": hit,
        "missed": missed,
        "missed_now": missed_now,
        "late_fallback": late_fallback,
        "late_now": late_now,
        "miss_awarded": miss_awarded,
        "frame": frame,
        "attempt_dist": attempt_dist,
    }


def _contact_quality(
    env: ManagerBasedRLEnv,
    command: MotionCommand,
    contact: torch.Tensor,
    correct_mask: torch.Tensor,
    discriminator_path: str,
    gamma: float,
    ready_gamma: float,
    timing_gamma: float,
    kick_foot_name: str,
    support_foot_name: str,
    geom_params: dict,
    timing_params: dict,
    empty_swing_quality_scale: float = 1.0,
    require_attempt_hit: bool = False,
    contact_without_attempt_quality_scale: float = 0.0,
    attempt_speed_threshold: float = 2.0,
    attempt_closing_speed: float = 0.5,
    attempt_max_foot_ball_dist: float = 0.9,
    attempt_min_kick_height: float = 0.02,
    attempt_max_kick_height: float = 0.65,
    attempt_near_ball_score: float = 0.5,
    attempt_early_grace: int = 25,
    attempt_window: int = 18,
) -> torch.Tensor:
    quality = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if not torch.any(contact):
        return quality

    d_score = _compute_d_score(env, command, discriminator_path).pow(gamma)
    geom = _strike_geometry(env, command, kick_foot_name, support_foot_name, **geom_params)
    attempt_state = _update_attempt_lifecycle(
        env,
        command,
        geom,
        contact=contact,
        attempt_speed_threshold=attempt_speed_threshold,
        attempt_closing_speed=attempt_closing_speed,
        attempt_max_foot_ball_dist=attempt_max_foot_ball_dist,
        attempt_min_kick_height=attempt_min_kick_height,
        attempt_max_kick_height=attempt_max_kick_height,
        attempt_near_ball_score=attempt_near_ball_score,
        attempt_early_grace=attempt_early_grace,
        attempt_window=attempt_window,
    )
    timing = _timing_gate(command, **timing_params).pow(timing_gamma)
    if hasattr(command, "metrics"):
        command.metrics["v36b_timing_gate"] = timing.detach()

    quality = d_score * geom["ready_score"].pow(ready_gamma) * timing
    quality = quality * correct_mask.to(dtype=quality.dtype)

    # If the policy has already made a clear empty swing before first contact,
    # do not let a later fallback touch receive normal strike/outcome credit.
    # This is a memory term, unlike the per-frame empty swing penalty.
    empty_swing_seen = _get_empty_swing_state(env, command)
    if hasattr(command, "metrics"):
        command.metrics["v36b_empty_swing_seen"] = empty_swing_seen.float()
    quality = torch.where(empty_swing_seen, quality * float(empty_swing_quality_scale), quality)
    if require_attempt_hit:
        hit_scale = torch.where(
            attempt_state["hit"],
            torch.ones_like(quality),
            torch.full_like(quality, float(contact_without_attempt_quality_scale)),
        )
        quality = quality * hit_scale

    quality = torch.where(contact, quality, torch.zeros_like(quality))
    return quality.clamp(0.0, 1.0)


def _resolve_contact_event(
    env: ManagerBasedRLEnv,
    command: MotionCommand,
    ball_sensor_name: str,
    horizontal_force_threshold: float,
    foot_cfg: SceneEntityCfg | None,
):
    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)
    correct_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    if foot_cfg is not None and torch.any(event.new_contact):
        foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
        if foot_info.env_ids.numel() > 0:
            valid_expectation = foot_info.expected >= 0
            correct = (foot_info.sides == foot_info.expected) & valid_expectation
            correct_mask[foot_info.env_ids] = correct
    elif torch.any(event.new_contact):
        correct_mask = event.new_contact.clone()

    return event.new_contact, correct_mask


def v36b_strike_ready_prior(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    kick_foot_name: str = "right_ankle_roll_link",
    support_foot_name: str = "left_ankle_roll_link",
    near_ball_dist: float = 1.15,
    near_ball_temp: float = 0.20,
    kick_dist_std: float = 0.28,
    kick_height_min: float = 0.02,
    kick_height_max: float = 0.45,
    kick_height_std: float = 0.12,
    support_lat_min: float = 0.16,
    support_lat_max: float = 0.58,
    support_long_min: float = -0.70,
    support_long_max: float = 0.10,
    support_region_std: float = 0.18,
    support_vel_std: float = 0.45,
    support_height_max: float = 0.16,
    support_height_std: float = 0.10,
    support_yaw_std: float = 0.75,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    tracker = _get_kick_tracker(command)
    pre_contact = (~tracker.get_contact_awarded()).to(dtype=torch.float32)
    geom = _strike_geometry(
        env,
        command,
        kick_foot_name,
        support_foot_name,
        near_ball_dist=near_ball_dist,
        near_ball_temp=near_ball_temp,
        kick_dist_std=kick_dist_std,
        kick_height_min=kick_height_min,
        kick_height_max=kick_height_max,
        kick_height_std=kick_height_std,
        support_lat_min=support_lat_min,
        support_lat_max=support_lat_max,
        support_long_min=support_long_min,
        support_long_max=support_long_max,
        support_region_std=support_region_std,
        support_vel_std=support_vel_std,
        support_height_max=support_height_max,
        support_height_std=support_height_std,
        support_yaw_std=support_yaw_std,
    )
    return pre_contact * geom["near_ball"] * (0.65 * geom["plant_score"] + 0.35 * geom["ready_score"])


def v36b_empty_swing_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    kick_foot_name: str = "right_ankle_roll_link",
    support_foot_name: str = "left_ankle_roll_link",
    speed_threshold: float = 2.0,
    closing_speed_threshold: float = 0.5,
    speed_std: float = 1.2,
    empty_dist: float = 0.42,
    empty_dist_temp: float = 0.08,
    miss_penalty_scale: float = 1.0,
    near_ball_dist: float = 1.15,
    near_ball_temp: float = 0.20,
    kick_dist_std: float = 0.28,
    kick_height_min: float = 0.02,
    kick_height_max: float = 0.45,
    kick_height_std: float = 0.12,
    support_lat_min: float = 0.16,
    support_lat_max: float = 0.58,
    support_long_min: float = -0.70,
    support_long_max: float = 0.10,
    support_region_std: float = 0.18,
    support_vel_std: float = 0.45,
    support_height_max: float = 0.16,
    support_height_std: float = 0.10,
    support_yaw_std: float = 0.75,
    trigger_near_ball: float = 0.5,
    attempt_max_foot_ball_dist: float = 0.9,
    attempt_min_kick_height: float = 0.02,
    attempt_max_kick_height: float = 0.65,
    attempt_early_grace: int = 25,
    attempt_window: int = 18,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    tracker = _get_kick_tracker(command)
    pre_contact = (~tracker.get_contact_awarded()).to(dtype=torch.float32)
    geom = _strike_geometry(
        env,
        command,
        kick_foot_name,
        support_foot_name,
        near_ball_dist=near_ball_dist,
        near_ball_temp=near_ball_temp,
        kick_dist_std=kick_dist_std,
        kick_height_min=kick_height_min,
        kick_height_max=kick_height_max,
        kick_height_std=kick_height_std,
        support_lat_min=support_lat_min,
        support_lat_max=support_lat_max,
        support_long_min=support_long_min,
        support_long_max=support_long_max,
        support_region_std=support_region_std,
        support_vel_std=support_vel_std,
        support_height_max=support_height_max,
        support_height_std=support_height_std,
        support_yaw_std=support_yaw_std,
    )

    speed_excess = (geom["kick_speed_xy"] - speed_threshold).clamp(min=0.0)
    fast_swing = 1.0 - torch.exp(-(speed_excess * speed_excess) / (speed_std**2))
    closing_excess = (geom["closing_speed"] - closing_speed_threshold).clamp(min=0.0)
    closing_swing = 1.0 - torch.exp(-(closing_excess * closing_excess) / (speed_std**2))
    too_far = torch.sigmoid((geom["kick_dist_xy"] - empty_dist) / empty_dist_temp)
    penalty = pre_contact * geom["near_ball"] * fast_swing * closing_swing * too_far

    attempt_state = _update_attempt_lifecycle(
        env,
        command,
        geom,
        contact=None,
        attempt_speed_threshold=speed_threshold,
        attempt_closing_speed=closing_speed_threshold,
        attempt_max_foot_ball_dist=attempt_max_foot_ball_dist,
        attempt_min_kick_height=attempt_min_kick_height,
        attempt_max_kick_height=attempt_max_kick_height,
        attempt_near_ball_score=trigger_near_ball,
        attempt_early_grace=attempt_early_grace,
        attempt_window=attempt_window,
    )
    miss_once = attempt_state["missed"] & (~attempt_state["miss_awarded"])
    if torch.any(miss_once):
        attempt_state["miss_awarded"][miss_once] = True
        penalty = penalty + miss_once.to(dtype=penalty.dtype) * float(miss_penalty_scale)

    if hasattr(command, "metrics"):
        command.metrics["v36b_empty_swing_seen"] = _get_empty_swing_state(env, command).float()
        command.metrics["v36b_attempt_miss_once"] = miss_once.float()

    return -penalty


def v36b_strike_contact(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 10.0,
    foot_cfg: SceneEntityCfg | None = None,
    discriminator_path: str = "models/strike_discriminator.pt",
    gamma: float = 1.0,
    ready_gamma: float = 0.7,
    timing_gamma: float = 1.0,
    kick_foot_name: str = "right_ankle_roll_link",
    support_foot_name: str = "left_ankle_roll_link",
    near_ball_dist: float = 1.15,
    near_ball_temp: float = 0.20,
    kick_dist_std: float = 0.28,
    kick_height_min: float = 0.02,
    kick_height_max: float = 0.45,
    kick_height_std: float = 0.12,
    support_lat_min: float = 0.16,
    support_lat_max: float = 0.58,
    support_long_min: float = -0.70,
    support_long_max: float = 0.10,
    support_region_std: float = 0.18,
    support_vel_std: float = 0.45,
    support_height_max: float = 0.16,
    support_height_std: float = 0.10,
    support_yaw_std: float = 0.75,
    timing_early_grace: int = 20,
    timing_late_grace: int = 55,
    timing_early_decay: float = 12.0,
    timing_late_decay: float = 25.0,
    empty_swing_quality_scale: float = 1.0,
    require_attempt_hit: bool = False,
    contact_without_attempt_quality_scale: float = 0.0,
    attempt_speed_threshold: float = 2.0,
    attempt_closing_speed: float = 0.5,
    attempt_max_foot_ball_dist: float = 0.9,
    attempt_min_kick_height: float = 0.02,
    attempt_max_kick_height: float = 0.65,
    attempt_near_ball_score: float = 0.5,
    attempt_early_grace: int = 25,
    attempt_window: int = 18,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    contact, correct_mask = _resolve_contact_event(env, command, ball_sensor_name, horizontal_force_threshold, foot_cfg)
    geom_params = _default_geometry_kwargs({
        "near_ball_dist": near_ball_dist,
        "near_ball_temp": near_ball_temp,
        "kick_dist_std": kick_dist_std,
        "kick_height_min": kick_height_min,
        "kick_height_max": kick_height_max,
        "kick_height_std": kick_height_std,
        "support_lat_min": support_lat_min,
        "support_lat_max": support_lat_max,
        "support_long_min": support_long_min,
        "support_long_max": support_long_max,
        "support_region_std": support_region_std,
        "support_vel_std": support_vel_std,
        "support_height_max": support_height_max,
        "support_height_std": support_height_std,
        "support_yaw_std": support_yaw_std,
    })
    timing_params = {
        "early_grace": int(timing_early_grace),
        "late_grace": int(timing_late_grace),
        "early_decay": float(timing_early_decay),
        "late_decay": float(timing_late_decay),
    }
    quality = _contact_quality(
        env,
        command,
        contact,
        correct_mask,
        discriminator_path,
        gamma,
        ready_gamma,
        timing_gamma,
        kick_foot_name,
        support_foot_name,
        geom_params,
        timing_params,
        empty_swing_quality_scale=empty_swing_quality_scale,
        require_attempt_hit=require_attempt_hit,
        contact_without_attempt_quality_scale=contact_without_attempt_quality_scale,
        attempt_speed_threshold=attempt_speed_threshold,
        attempt_closing_speed=attempt_closing_speed,
        attempt_max_foot_ball_dist=attempt_max_foot_ball_dist,
        attempt_min_kick_height=attempt_min_kick_height,
        attempt_max_kick_height=attempt_max_kick_height,
        attempt_near_ball_score=attempt_near_ball_score,
        attempt_early_grace=attempt_early_grace,
        attempt_window=attempt_window,
    )
    if torch.any(contact):
        _get_kick_tracker(command).record_expected_success(contact, quality > 0.25)
    return quality


def v36b_invalid_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 10.0,
    foot_cfg: SceneEntityCfg | None = None,
    discriminator_path: str = "models/strike_discriminator.pt",
    gamma: float = 1.0,
    ready_gamma: float = 0.7,
    timing_gamma: float = 1.0,
    kick_foot_name: str = "right_ankle_roll_link",
    support_foot_name: str = "left_ankle_roll_link",
    near_ball_dist: float = 1.15,
    near_ball_temp: float = 0.20,
    kick_dist_std: float = 0.28,
    kick_height_min: float = 0.02,
    kick_height_max: float = 0.45,
    kick_height_std: float = 0.12,
    support_lat_min: float = 0.16,
    support_lat_max: float = 0.58,
    support_long_min: float = -0.70,
    support_long_max: float = 0.10,
    support_region_std: float = 0.18,
    support_vel_std: float = 0.45,
    support_height_max: float = 0.16,
    support_height_std: float = 0.10,
    support_yaw_std: float = 0.75,
    timing_early_grace: int = 20,
    timing_late_grace: int = 55,
    timing_early_decay: float = 12.0,
    timing_late_decay: float = 25.0,
    empty_swing_quality_scale: float = 1.0,
    require_attempt_hit: bool = False,
    contact_without_attempt_quality_scale: float = 0.0,
    attempt_speed_threshold: float = 2.0,
    attempt_closing_speed: float = 0.5,
    attempt_max_foot_ball_dist: float = 0.9,
    attempt_min_kick_height: float = 0.02,
    attempt_max_kick_height: float = 0.65,
    attempt_near_ball_score: float = 0.5,
    attempt_early_grace: int = 25,
    attempt_window: int = 18,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    contact, correct_mask = _resolve_contact_event(env, command, ball_sensor_name, horizontal_force_threshold, foot_cfg)
    geom_params = _default_geometry_kwargs({
        "near_ball_dist": near_ball_dist,
        "near_ball_temp": near_ball_temp,
        "kick_dist_std": kick_dist_std,
        "kick_height_min": kick_height_min,
        "kick_height_max": kick_height_max,
        "kick_height_std": kick_height_std,
        "support_lat_min": support_lat_min,
        "support_lat_max": support_lat_max,
        "support_long_min": support_long_min,
        "support_long_max": support_long_max,
        "support_region_std": support_region_std,
        "support_vel_std": support_vel_std,
        "support_height_max": support_height_max,
        "support_height_std": support_height_std,
        "support_yaw_std": support_yaw_std,
    })
    timing_params = {
        "early_grace": int(timing_early_grace),
        "late_grace": int(timing_late_grace),
        "early_decay": float(timing_early_decay),
        "late_decay": float(timing_late_decay),
    }
    quality = _contact_quality(
        env,
        command,
        contact,
        correct_mask,
        discriminator_path,
        gamma,
        ready_gamma,
        timing_gamma,
        kick_foot_name,
        support_foot_name,
        geom_params,
        timing_params,
        empty_swing_quality_scale=empty_swing_quality_scale,
        require_attempt_hit=require_attempt_hit,
        contact_without_attempt_quality_scale=contact_without_attempt_quality_scale,
        attempt_speed_threshold=attempt_speed_threshold,
        attempt_closing_speed=attempt_closing_speed,
        attempt_max_foot_ball_dist=attempt_max_foot_ball_dist,
        attempt_min_kick_height=attempt_min_kick_height,
        attempt_max_kick_height=attempt_max_kick_height,
        attempt_near_ball_score=attempt_near_ball_score,
        attempt_early_grace=attempt_early_grace,
        attempt_window=attempt_window,
    )
    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    reward[contact] = -(1.0 - quality[contact])
    if torch.any(contact):
        _get_kick_tracker(command).record_expected_success(contact, quality > 0.25)
    return reward


def _open_valid_contact_timer(
    env: ManagerBasedRLEnv,
    command: MotionCommand,
    timer_name: str,
    quality_name: str,
    ball_sensor_name: str,
    horizontal_force_threshold: float,
    foot_cfg: SceneEntityCfg | None,
    discriminator_path: str,
    gamma: float,
    ready_gamma: float,
    timing_gamma: float,
    min_quality: float,
    kick_foot_name: str,
    support_foot_name: str,
    geom_params: dict,
    timing_params: dict,
    window: int,
    empty_swing_quality_scale: float = 1.0,
    require_attempt_hit: bool = False,
    contact_without_attempt_quality_scale: float = 0.0,
    attempt_speed_threshold: float = 2.0,
    attempt_closing_speed: float = 0.5,
    attempt_max_foot_ball_dist: float = 0.9,
    attempt_min_kick_height: float = 0.02,
    attempt_max_kick_height: float = 0.65,
    attempt_near_ball_score: float = 0.5,
    attempt_early_grace: int = 25,
    attempt_window: int = 18,
):
    timer = getattr(env, timer_name, None)
    if timer is None or timer.shape[0] != env.num_envs:
        timer = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    else:
        timer = timer.to(device=env.device, dtype=torch.int32)

    stored_quality = getattr(env, quality_name, None)
    if stored_quality is None or stored_quality.shape[0] != env.num_envs:
        stored_quality = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    else:
        stored_quality = stored_quality.to(device=env.device, dtype=torch.float32)

    contact, correct_mask = _resolve_contact_event(env, command, ball_sensor_name, horizontal_force_threshold, foot_cfg)
    if torch.any(contact):
        quality = _contact_quality(
            env,
            command,
            contact,
            correct_mask,
            discriminator_path,
            gamma,
            ready_gamma,
            timing_gamma,
            kick_foot_name,
            support_foot_name,
            geom_params,
            timing_params,
            empty_swing_quality_scale=empty_swing_quality_scale,
            require_attempt_hit=require_attempt_hit,
            contact_without_attempt_quality_scale=contact_without_attempt_quality_scale,
            attempt_speed_threshold=attempt_speed_threshold,
            attempt_closing_speed=attempt_closing_speed,
            attempt_max_foot_ball_dist=attempt_max_foot_ball_dist,
            attempt_min_kick_height=attempt_min_kick_height,
            attempt_max_kick_height=attempt_max_kick_height,
            attempt_near_ball_score=attempt_near_ball_score,
            attempt_early_grace=attempt_early_grace,
            attempt_window=attempt_window,
        )
        trigger = contact & (quality >= min_quality)
        timer[trigger] = int(window)
        stored_quality[trigger] = quality[trigger]

    setattr(env, timer_name, timer)
    setattr(env, quality_name, stored_quality)
    return timer, stored_quality


def v36b_gated_ball_speed(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 1.2,
    velocity_threshold: float = 0.5,
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 10.0,
    foot_cfg: SceneEntityCfg | None = None,
    discriminator_path: str = "models/strike_discriminator.pt",
    gamma: float = 1.0,
    ready_gamma: float = 0.7,
    timing_gamma: float = 1.0,
    min_quality: float = 0.25,
    window: int = 5,
    kick_foot_name: str = "right_ankle_roll_link",
    support_foot_name: str = "left_ankle_roll_link",
    near_ball_dist: float = 1.15,
    near_ball_temp: float = 0.20,
    kick_dist_std: float = 0.28,
    kick_height_min: float = 0.02,
    kick_height_max: float = 0.45,
    kick_height_std: float = 0.12,
    support_lat_min: float = 0.16,
    support_lat_max: float = 0.58,
    support_long_min: float = -0.70,
    support_long_max: float = 0.10,
    support_region_std: float = 0.18,
    support_vel_std: float = 0.45,
    support_height_max: float = 0.16,
    support_height_std: float = 0.10,
    support_yaw_std: float = 0.75,
    timing_early_grace: int = 20,
    timing_late_grace: int = 55,
    timing_early_decay: float = 12.0,
    timing_late_decay: float = 25.0,
    empty_swing_quality_scale: float = 1.0,
    require_attempt_hit: bool = False,
    contact_without_attempt_quality_scale: float = 0.0,
    attempt_speed_threshold: float = 2.0,
    attempt_closing_speed: float = 0.5,
    attempt_max_foot_ball_dist: float = 0.9,
    attempt_min_kick_height: float = 0.02,
    attempt_max_kick_height: float = 0.65,
    attempt_near_ball_score: float = 0.5,
    attempt_early_grace: int = 25,
    attempt_window: int = 18,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    geom_params = _default_geometry_kwargs({
        "near_ball_dist": near_ball_dist,
        "near_ball_temp": near_ball_temp,
        "kick_dist_std": kick_dist_std,
        "kick_height_min": kick_height_min,
        "kick_height_max": kick_height_max,
        "kick_height_std": kick_height_std,
        "support_lat_min": support_lat_min,
        "support_lat_max": support_lat_max,
        "support_long_min": support_long_min,
        "support_long_max": support_long_max,
        "support_region_std": support_region_std,
        "support_vel_std": support_vel_std,
        "support_height_max": support_height_max,
        "support_height_std": support_height_std,
        "support_yaw_std": support_yaw_std,
    })
    timing_params = {
        "early_grace": int(timing_early_grace),
        "late_grace": int(timing_late_grace),
        "early_decay": float(timing_early_decay),
        "late_decay": float(timing_late_decay),
    }
    timer, quality = _open_valid_contact_timer(
        env,
        command,
        f"_{command_name}_v36b_speed_timer",
        f"_{command_name}_v36b_speed_quality",
        ball_sensor_name,
        horizontal_force_threshold,
        foot_cfg,
        discriminator_path,
        gamma,
        ready_gamma,
        timing_gamma,
        min_quality,
        kick_foot_name,
        support_foot_name,
        geom_params,
        timing_params,
        window,
        empty_swing_quality_scale=empty_swing_quality_scale,
        require_attempt_hit=require_attempt_hit,
        contact_without_attempt_quality_scale=contact_without_attempt_quality_scale,
        attempt_speed_threshold=attempt_speed_threshold,
        attempt_closing_speed=attempt_closing_speed,
        attempt_max_foot_ball_dist=attempt_max_foot_ball_dist,
        attempt_min_kick_height=attempt_min_kick_height,
        attempt_max_kick_height=attempt_max_kick_height,
        attempt_near_ball_score=attempt_near_ball_score,
        attempt_early_grace=attempt_early_grace,
        attempt_window=attempt_window,
    )

    ball_vel = env.scene["soccer_ball"].data.root_lin_vel_w
    speed_xy = torch.norm(ball_vel[:, :2], dim=-1)
    active = (timer > 0) & (speed_xy > velocity_threshold)
    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if torch.any(active):
        reward[active] = (1.0 - torch.exp(-(speed_xy[active] ** 2) / (std**2))) * quality[active]
    timer = torch.where(timer > 0, timer - 1, timer)
    setattr(env, f"_{command_name}_v36b_speed_timer", timer)
    return reward


def v36b_gated_direction(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 0.8,
    velocity_threshold: float = 0.5,
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 10.0,
    foot_cfg: SceneEntityCfg | None = None,
    discriminator_path: str = "models/strike_discriminator.pt",
    gamma: float = 1.0,
    ready_gamma: float = 0.7,
    timing_gamma: float = 1.0,
    min_quality: float = 0.25,
    window: int = 5,
    kick_foot_name: str = "right_ankle_roll_link",
    support_foot_name: str = "left_ankle_roll_link",
    near_ball_dist: float = 1.15,
    near_ball_temp: float = 0.20,
    kick_dist_std: float = 0.28,
    kick_height_min: float = 0.02,
    kick_height_max: float = 0.45,
    kick_height_std: float = 0.12,
    support_lat_min: float = 0.16,
    support_lat_max: float = 0.58,
    support_long_min: float = -0.70,
    support_long_max: float = 0.10,
    support_region_std: float = 0.18,
    support_vel_std: float = 0.45,
    support_height_max: float = 0.16,
    support_height_std: float = 0.10,
    support_yaw_std: float = 0.75,
    timing_early_grace: int = 20,
    timing_late_grace: int = 55,
    timing_early_decay: float = 12.0,
    timing_late_decay: float = 25.0,
    empty_swing_quality_scale: float = 1.0,
    require_attempt_hit: bool = False,
    contact_without_attempt_quality_scale: float = 0.0,
    attempt_speed_threshold: float = 2.0,
    attempt_closing_speed: float = 0.5,
    attempt_max_foot_ball_dist: float = 0.9,
    attempt_min_kick_height: float = 0.02,
    attempt_max_kick_height: float = 0.65,
    attempt_near_ball_score: float = 0.5,
    attempt_early_grace: int = 25,
    attempt_window: int = 18,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    geom_params = _default_geometry_kwargs({
        "near_ball_dist": near_ball_dist,
        "near_ball_temp": near_ball_temp,
        "kick_dist_std": kick_dist_std,
        "kick_height_min": kick_height_min,
        "kick_height_max": kick_height_max,
        "kick_height_std": kick_height_std,
        "support_lat_min": support_lat_min,
        "support_lat_max": support_lat_max,
        "support_long_min": support_long_min,
        "support_long_max": support_long_max,
        "support_region_std": support_region_std,
        "support_vel_std": support_vel_std,
        "support_height_max": support_height_max,
        "support_height_std": support_height_std,
        "support_yaw_std": support_yaw_std,
    })
    timing_params = {
        "early_grace": int(timing_early_grace),
        "late_grace": int(timing_late_grace),
        "early_decay": float(timing_early_decay),
        "late_decay": float(timing_late_decay),
    }
    timer, quality = _open_valid_contact_timer(
        env,
        command,
        f"_{command_name}_v36b_dir_timer",
        f"_{command_name}_v36b_dir_quality",
        ball_sensor_name,
        horizontal_force_threshold,
        foot_cfg,
        discriminator_path,
        gamma,
        ready_gamma,
        timing_gamma,
        min_quality,
        kick_foot_name,
        support_foot_name,
        geom_params,
        timing_params,
        window,
        empty_swing_quality_scale=empty_swing_quality_scale,
        require_attempt_hit=require_attempt_hit,
        contact_without_attempt_quality_scale=contact_without_attempt_quality_scale,
        attempt_speed_threshold=attempt_speed_threshold,
        attempt_closing_speed=attempt_closing_speed,
        attempt_max_foot_ball_dist=attempt_max_foot_ball_dist,
        attempt_min_kick_height=attempt_min_kick_height,
        attempt_max_kick_height=attempt_max_kick_height,
        attempt_near_ball_score=attempt_near_ball_score,
        attempt_early_grace=attempt_early_grace,
        attempt_window=attempt_window,
    )

    ball_vel = env.scene["soccer_ball"].data.root_lin_vel_w
    vel_xy = ball_vel[:, :2]
    vel_norm = torch.norm(vel_xy, dim=-1, keepdim=True)
    direction = command.target_destination_pos - command.initial_target_point_pos
    direction_xy = direction[:, :2]
    dir_norm = torch.norm(direction_xy, dim=-1, keepdim=True)
    active = (timer > 0) & (vel_norm.squeeze(-1) > velocity_threshold) & (dir_norm.squeeze(-1) > 1e-6)

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if torch.any(active):
        vel_unit = vel_xy[active] / vel_norm[active].clamp(min=1e-6)
        dir_unit = direction_xy[active] / dir_norm[active].clamp(min=1e-6)
        cos_theta = torch.sum(vel_unit * dir_unit, dim=-1).clamp(-1.0, 1.0)
        error = torch.acos(cos_theta) ** 2
        reward[active] = torch.exp(-error / (std**2)) * quality[active]

    timer = torch.where(timer > 0, timer - 1, timer)
    setattr(env, f"_{command_name}_v36b_dir_timer", timer)
    return reward
