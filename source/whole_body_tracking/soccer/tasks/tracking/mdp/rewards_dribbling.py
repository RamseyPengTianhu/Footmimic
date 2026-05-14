"""Dribbling-specific reward functions.

Encourages close ball control without strike-the-ball objectives. Contact
legality is **geometry-based**: the first ``num_ankle_links`` entries in
``all_body_cfg.body_names`` must be the ankle links (both are valid for gentle
touches); knees/wrists listed after incur ``dribbling_undesired_contact_penalty``
when closest to the ball under contact. No ``kick_leg`` motion labels required.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_error_magnitude, quat_inv

from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Shared: net contact force on the ball (world frame)
# ---------------------------------------------------------------------------


def soccer_ball_contact_net_force_w(
    env: ManagerBasedRLEnv,
    ball_sensor_name: str = "soccer_ball_contact",
) -> torch.Tensor:
    """Net contact force on the soccer ball body, shape ``(num_envs, 3)``.

    Zeros are returned when the sensor has no usable data (same convention as
    ``_identify_contact_body``).
    """
    device = env.device
    num_envs = env.num_envs
    zero = torch.zeros(num_envs, 3, device=device, dtype=torch.float32)

    contact_sensor: ContactSensor = env.scene.sensors[ball_sensor_name]
    forces_data = contact_sensor.data

    forces = None
    if hasattr(forces_data, "net_forces_w_history"):
        fh = forces_data.net_forces_w_history
        if fh is not None and fh.numel() > 0:
            forces = fh.to(device)
            if forces.ndim >= 4:
                forces = forces.max(dim=1).values
    if forces is None:
        if hasattr(forces_data, "net_forces_w"):
            f = forces_data.net_forces_w
            if f is not None and f.numel() > 0:
                forces = f.to(device)

    if forces is None or forces.ndim < 2:
        return zero

    if forces.ndim == 3:
        return forces[:, 0, :]
    if forces.shape[-1] >= 3:
        return forces[:, :3]
    return zero


def soccer_ball_contact_force_magnitude(
    env: ManagerBasedRLEnv,
    ball_sensor_name: str = "soccer_ball_contact",
) -> torch.Tensor:
    """Scalar force magnitude on the ball, shape ``(num_envs,)``."""
    f = soccer_ball_contact_net_force_w(env, ball_sensor_name)
    return torch.norm(f, dim=-1)


# ---------------------------------------------------------------------------
# 1) Velocity Tracking  — ball vel aligned with pelvis vel
# ---------------------------------------------------------------------------


def dribbling_velocity_tracking(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 1.0,
    pelvis_speed_min: float = 0.0,
    ball_speed_min: float = 0.0,
    require_contact: bool = False,
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward alignment between the soccer ball velocity and the robot pelvis velocity.

    A cosine-similarity style reward: when the ball moves in the same direction
    and at a similar speed as the robot, the reward is maximised.

    Optional **anti-cheese gates** (defaults preserve legacy behaviour):

    - ``pelvis_speed_min`` / ``ball_speed_min``: multiply the reward by
      ``clamp(|v_xy| / min, max=1)`` so near-zero speeds do not yield a full
      score from ``exp(0)==1``.
    - ``require_contact``: multiply by 1 only when ball contact force exceeds
      ``contact_force_threshold`` (same scale as dribbling touch rewards).

    Returns a value in ``[0, 1]`` per environment.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    # Ball velocity (world frame, xy only)
    ball_vel_xy = soccer_ball.data.root_lin_vel_w[:, :2]  # (N, 2)
    # Robot pelvis velocity (world frame, xy only)
    pelvis_vel_xy = command.robot_anchor_lin_vel_w[:, :2]  # (N, 2)

    # Squared difference as the error signal
    vel_diff = ball_vel_xy - pelvis_vel_xy
    error = torch.sum(vel_diff * vel_diff, dim=-1)  # (N,)

    base = torch.exp(-error / (std ** 2))

    pelvis_sp = torch.norm(pelvis_vel_xy, dim=-1)
    ball_sp = torch.norm(ball_vel_xy, dim=-1)

    gate = torch.ones_like(base)
    if pelvis_speed_min > 0.0:
        gate = gate * torch.clamp(pelvis_sp / pelvis_speed_min, max=1.0)
    if ball_speed_min > 0.0:
        gate = gate * torch.clamp(ball_sp / ball_speed_min, max=1.0)

    if require_contact:
        fmag = soccer_ball_contact_force_magnitude(env, ball_sensor_name)
        gate = gate * (fmag > contact_force_threshold).to(torch.float32)

    return base * gate


# ---------------------------------------------------------------------------
# 2) Dynamic Proximity  — ball inside the "safe zone" in front of the robot
# ---------------------------------------------------------------------------

def dribbling_dynamic_proximity(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    near_dist: float = 0.2,
    far_dist: float = 0.5,
    penalty_std: float = 0.15,
    pelvis_speed_min: float = 0.0,
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    no_contact_zone_damping: float = 1.0,
    zone_lateral_abs_max: float = 0.18,
) -> torch.Tensor:
    """Reward keeping the ball inside a longitudinal safe-zone in front of the robot.

    The ball position is projected into the robot's local frame:
    - x_local in [near_dist, far_dist] → reward = 1.0
    - Outside that range → exponential decay with ``penalty_std``

    Lateral deviation (|y_local|) is also penalised with the same decay to
    encourage straight-ahead dribbling.

    If ``pelvis_speed_min > 0``, the reward is multiplied by
    ``clamp(|v_pelvis_xy| / pelvis_speed_min, max=1)`` so standing still in the
    safe zone is not a local optimum.

    If ``no_contact_zone_damping < 1``, when the ball sits in the forward corridor
    (longitudinal band + ``|y_local| <= zone_lateral_abs_max``) but the ball
    sensor reports no contact, the proximity reward is scaled by that factor.
    This reduces the \"park in front of the ball and wiggle\" optimum.

    Returns a value in [0, 1] per environment.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    # Ball position in world frame
    ball_pos_w = soccer_ball.data.root_pos_w  # (N, 3)
    # Robot pelvis position and orientation in world frame
    pelvis_pos_w = command.robot_pelvis_pos_w  # (N, 3)
    pelvis_quat_w = command.robot_pelvis_quat_w  # (N, 4)

    # Transform ball position into the robot's local frame
    delta_w = ball_pos_w - pelvis_pos_w  # (N, 3)
    delta_local = quat_apply(quat_inv(pelvis_quat_w), delta_w)  # (N, 3)

    x_local = delta_local[:, 0]  # forward axis
    y_local = delta_local[:, 1]  # lateral axis

    # Longitudinal error: distance to the nearest edge of the safe zone
    x_error = torch.where(
        x_local < near_dist,
        near_dist - x_local,    # too close → positive error
        torch.where(
            x_local > far_dist,
            x_local - far_dist,  # too far → positive error
            torch.zeros_like(x_local),  # inside safe zone → zero error
        ),
    )

    # Lateral error: absolute lateral offset
    y_error = torch.abs(y_local)

    total_error = x_error ** 2 + y_error ** 2
    proximity_reward = torch.exp(-total_error / (penalty_std ** 2))

    if pelvis_speed_min > 0.0:
        pelvis_vel_xy = command.robot_anchor_lin_vel_w[:, :2]
        pelvis_sp = torch.norm(pelvis_vel_xy, dim=-1)
        proximity_reward = proximity_reward * torch.clamp(pelvis_sp / pelvis_speed_min, max=1.0)

    if no_contact_zone_damping < 1.0 - 1e-6:
        in_corridor = (
            (x_local >= near_dist)
            & (x_local <= far_dist)
            & (torch.abs(y_local) <= zone_lateral_abs_max)
        )
        fmag = soccer_ball_contact_force_magnitude(env, ball_sensor_name)
        no_touch = fmag <= contact_force_threshold
        damp = torch.where(
            in_corridor & no_touch,
            torch.full_like(proximity_reward, no_contact_zone_damping),
            torch.ones_like(proximity_reward),
        )
        proximity_reward = proximity_reward * damp

    return proximity_reward


# ---------------------------------------------------------------------------
# 2a) Stall in front of ball without touching — kills "back up, then freeze"
# ---------------------------------------------------------------------------


def dribbling_stall_no_touch_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    max_xy_dist: float = 0.52,
    pelvis_speed_max: float = 0.16,
) -> torch.Tensor:
    """Penalty in ``[0, 1]`` when the ball is close in XY but pelvis is nearly static and there is no contact.

    Targets the local optimum: robot brings the ball into a comfortable pose in
    front of the body, then stops or only sways without registering foot-ball
    contact on the ball sensor.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    pelvis_pos_xy = command.robot_pelvis_pos_w[:, :2]
    ball_pos_xy = soccer_ball.data.root_pos_w[:, :2]
    dist_xy = torch.norm(ball_pos_xy - pelvis_pos_xy, dim=-1)

    pelvis_vel_xy = command.robot_anchor_lin_vel_w[:, :2]
    pelvis_sp = torch.norm(pelvis_vel_xy, dim=-1)

    fmag = soccer_ball_contact_force_magnitude(env, ball_sensor_name)
    no_touch = fmag <= contact_force_threshold

    near = dist_xy <= max_xy_dist
    slow = pelvis_sp <= pelvis_speed_max
    return (near & slow & no_touch).to(torch.float32)


# ---------------------------------------------------------------------------
# Helper: identify which robot body caused the ball contact
# ---------------------------------------------------------------------------

def _identify_contact_body(
    env: ManagerBasedRLEnv,
    command: MotionCommand,
    ball_sensor_name: str,
    all_body_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Identify which robot body is closest to the ball when contact occurs.

    Returns:
        has_contact: (N,) bool — whether ball has non-zero contact force
        contact_force_mag: (N,) float — force magnitude on ball
        closest_body_idx: (N,) long — index into all_body_cfg.body_names
                                       of the body closest to ball
    """
    device = env.device
    num_envs = env.num_envs

    # Default outputs
    has_contact = torch.zeros(num_envs, dtype=torch.bool, device=device)
    closest_body_idx = torch.zeros(num_envs, dtype=torch.long, device=device)

    force_vec = soccer_ball_contact_net_force_w(env, ball_sensor_name)
    force_mag = torch.norm(force_vec, dim=-1)
    has_contact = force_mag > 1.0  # minimal threshold to filter noise
    contact_force_mag = force_mag

    if not torch.any(has_contact):
        return has_contact, contact_force_mag, closest_body_idx

    # For envs with contact, find which robot body is closest to ball
    robot = env.scene[all_body_cfg.name]
    soccer_ball = env.scene["soccer_ball"]

    # Resolve body indices (cached; invalidate if body list changes)
    cache_name = "_dribbling_body_indices_cache"
    cached = getattr(env, cache_name, None)
    names_t = tuple(all_body_cfg.body_names)
    if cached is None or cached.get("names") != names_t:
        body_indices = torch.as_tensor(
            robot.find_bodies(all_body_cfg.body_names, preserve_order=True)[0],
            dtype=torch.long, device=device,
        )
        setattr(env, cache_name, {"names": names_t, "idx": body_indices})
    body_indices = getattr(env, cache_name)["idx"]

    # Contact envs only
    contact_env_ids = torch.nonzero(has_contact, as_tuple=False).squeeze(-1)
    body_pos = robot.data.body_pos_w[contact_env_ids][:, body_indices]  # (M, B, 3)
    ball_pos = soccer_ball.data.root_pos_w[contact_env_ids]  # (M, 3)

    dist = torch.norm(body_pos - ball_pos.unsqueeze(1), dim=-1)  # (M, B)
    closest = torch.argmin(dist, dim=-1)  # (M,) — index into body_names
    closest_body_idx[contact_env_ids] = closest

    return has_contact, contact_force_mag, closest_body_idx


def _is_dribble_legal_ankle_contact(closest_body_idx: torch.Tensor, num_ankle_links: int) -> torch.Tensor:
    """True when the closest link index is one of the leading ankle entries."""
    if num_ankle_links <= 0:
        return torch.zeros_like(closest_body_idx, dtype=torch.bool)
    return closest_body_idx < num_ankle_links


# ---------------------------------------------------------------------------
# 2b) Dense approach — ankles near ball while force sensor shows no contact
# ---------------------------------------------------------------------------


def dribbling_approach_foot_ball_distance(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    foot_cfg: SceneEntityCfg | None = None,
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    std: float = 0.22,
    pelvis_speed_min: float = 0.08,
) -> torch.Tensor:
    """``[0,1]`` shaping when the ball reports no contact: minimise ankle–ball gap."""
    if foot_cfg is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    fmag = soccer_ball_contact_force_magnitude(env, ball_sensor_name)
    no_contact = fmag <= contact_force_threshold

    robot = env.scene[foot_cfg.name]
    soccer_ball = env.scene["soccer_ball"]

    cache = getattr(env, "_dribbling_foot_ball_idx_cache", None)
    if cache is None or cache.get("names") != tuple(foot_cfg.body_names):
        idx = torch.as_tensor(
            robot.find_bodies(foot_cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=env.device,
        )
        cache = {"names": tuple(foot_cfg.body_names), "idx": idx}
        setattr(env, "_dribbling_foot_ball_idx_cache", cache)
    body_idx = cache["idx"]

    feet_pos = robot.data.body_pos_w[:, body_idx, :]
    ball_pos = soccer_ball.data.root_pos_w.unsqueeze(1)
    dist = torch.norm(feet_pos - ball_pos, dim=-1)
    min_dist = dist.min(dim=-1).values

    shaping = torch.exp(-(min_dist ** 2) / (std ** 2))
    out = torch.where(no_contact, shaping, torch.zeros_like(shaping))

    if pelvis_speed_min > 0.0:
        command: MotionCommand = env.command_manager.get_term(command_name)
        pelvis_sp = torch.norm(command.robot_anchor_lin_vel_w[:, :2], dim=-1)
        out = out * torch.clamp(pelvis_sp / pelvis_speed_min, max=1.0)

    return out


# ---------------------------------------------------------------------------
# 2c) Pelvis orientation vs motion reference (reduces lean-back / arched cheat)
# ---------------------------------------------------------------------------


def dribbling_pelvis_quat_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 0.45,
) -> torch.Tensor:
    """Reward matching motion pelvis orientation (same frame as body tracking)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    names = command.cfg.body_names
    if "pelvis" not in names:
        return torch.ones(env.num_envs, device=env.device, dtype=torch.float32)
    pi = names.index("pelvis")
    ref = command.body_quat_relative_w[:, pi]
    rob = command.robot_body_quat_w[:, pi]
    err = quat_error_magnitude(ref, rob)
    return torch.exp(-(err ** 2) / (std ** 2))


# ---------------------------------------------------------------------------
# 2d) Penalise excessive horizontal ball speed (dribble vs kick)
# ---------------------------------------------------------------------------


def dribbling_ball_xy_speed_excess_penalty(
    env: ManagerBasedRLEnv,
    speed_cap: float = 3.5,
    linear_scale: float = 1.5,
) -> torch.Tensor:
    """Penalty in ``[0, 1]`` for ``|v_ball,xy|`` above ``speed_cap``."""
    soccer_ball = env.scene["soccer_ball"]
    sp = torch.norm(soccer_ball.data.root_lin_vel_w[:, :2], dim=-1)
    excess = torch.relu(sp - speed_cap)
    return torch.clamp(excess / linear_scale, max=1.0)


# ---------------------------------------------------------------------------
# 2e) Ball forward progress reward — encourage actual dribble advancement
# ---------------------------------------------------------------------------


def dribbling_ball_forward_progress_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    min_forward_speed: float = 0.2,
    speed_scale: float = 0.25,
    pelvis_speed_min: float = 0.06,
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 0.5,
    require_recent_contact: bool = True,
    recent_contact_window: int = 10,
) -> torch.Tensor:
    """Reward forward ball velocity in pelvis-local frame.

    This term rewards positive local-X ball speed (ball moving in front of the
    robot), and can optionally be gated by recent contact to avoid standing
    exploits where the policy does not engage the ball.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    ball_vel_w = soccer_ball.data.root_lin_vel_w[:, :3]
    pelvis_quat_w = command.robot_pelvis_quat_w
    ball_vel_local = quat_apply(quat_inv(pelvis_quat_w), ball_vel_w)
    forward_speed = torch.clamp(ball_vel_local[:, 0], min=0.0)

    # Smooth ramp around the target forward speed.
    base = torch.clamp((forward_speed - min_forward_speed) / max(speed_scale, 1e-6), min=0.0, max=1.0)

    pelvis_speed = torch.norm(command.robot_anchor_lin_vel_w[:, :2], dim=-1)
    gate = torch.clamp(pelvis_speed / max(pelvis_speed_min, 1e-6), max=1.0)

    if require_recent_contact:
        step_buf = getattr(env, "episode_length_buf", None)
        if step_buf is None:
            step_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

        fmag = soccer_ball_contact_force_magnitude(env, ball_sensor_name)
        has_contact = fmag > contact_force_threshold
        reset_mask = step_buf == 0

        cnt_name = "_dribbling_steps_since_contact"
        cnt = getattr(env, cnt_name, None)
        if cnt is None or cnt.shape[0] != env.num_envs:
            cnt = torch.full((env.num_envs,), fill_value=recent_contact_window + 1, device=env.device, dtype=torch.int32)

        cnt = torch.where(
            reset_mask,
            torch.full_like(cnt, recent_contact_window + 1),
            torch.where(has_contact, torch.zeros_like(cnt), cnt + 1),
        )
        setattr(env, cnt_name, cnt)
        recent = cnt <= int(recent_contact_window)
        gate = gate * recent.to(torch.float32)

    return base * gate


# ---------------------------------------------------------------------------
# 2f) Contact-graph style phase alignment (approach -> control/push)
# ---------------------------------------------------------------------------


def dribbling_phase_graph_alignment(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 0.5,
    approach_xy_dist: float = 0.55,
    approach_dist_std: float = 0.20,
    push_speed_threshold: float = 0.22,
) -> torch.Tensor:
    """Phase-style shaping without explicit per-frame labels.

    - ``approach`` phase (ball far): reward getting closer while avoiding contact.
    - ``control/push`` phase (ball near): reward contact, and reward stronger when
      ball moves forward in pelvis-local frame (push over static trapping).
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    pelvis_pos_xy = command.robot_pelvis_pos_w[:, :2]
    ball_pos_xy = soccer_ball.data.root_pos_w[:, :2]
    dist_xy = torch.norm(ball_pos_xy - pelvis_pos_xy, dim=-1)

    fmag = soccer_ball_contact_force_magnitude(env, ball_sensor_name)
    has_contact = fmag > contact_force_threshold

    ball_vel_w = soccer_ball.data.root_lin_vel_w[:, :3]
    ball_vel_local = quat_apply(quat_inv(command.robot_pelvis_quat_w), ball_vel_w)
    forward_speed = torch.clamp(ball_vel_local[:, 0], min=0.0)

    phase_approach = dist_xy > approach_xy_dist
    phase_interact = ~phase_approach

    # Approach: closer is better, but do not touch too early.
    approach_core = torch.exp(-((dist_xy - approach_xy_dist).clamp(min=0.0) ** 2) / (approach_dist_std ** 2))
    approach_reward = approach_core * (~has_contact).to(torch.float32)

    # Interact: contact is required; encourage push speed on top of stable contact.
    push_gain = torch.clamp(forward_speed / max(push_speed_threshold, 1e-6), min=0.0, max=1.0)
    interact_reward = has_contact.to(torch.float32) * (0.55 + 0.45 * push_gain)

    return torch.where(phase_approach, approach_reward, interact_reward)


# ---------------------------------------------------------------------------
# 2g) Anti-orbit penalty — discourage circling around the ball without touch
# ---------------------------------------------------------------------------


def dribbling_orbiting_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    orbit_radius_max: float = 0.9,
    tangential_deadzone: float = 0.08,
    tangential_scale: float = 0.35,
) -> torch.Tensor:
    """Penalty in ``[0,1]`` for tangential pelvis motion around the ball.

    The penalty is active only when the pelvis is near the ball (within
    ``orbit_radius_max`` in XY) and ball contact force is weak
    (``<= contact_force_threshold``). This directly suppresses the common
    local optimum of \"one foot hovering, circling around the ball\".
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    pelvis_pos_xy = command.robot_pelvis_pos_w[:, :2]
    pelvis_vel_xy = command.robot_anchor_lin_vel_w[:, :2]
    ball_pos_xy = soccer_ball.data.root_pos_w[:, :2]

    r = pelvis_pos_xy - ball_pos_xy
    r_norm = torch.norm(r, dim=-1)
    r_hat = r / (r_norm.unsqueeze(-1) + 1e-6)

    # Tangent unit vector around the ball (CCW): [-y, x]
    t_hat = torch.stack((-r_hat[:, 1], r_hat[:, 0]), dim=-1)
    v_tan = torch.abs(torch.sum(pelvis_vel_xy * t_hat, dim=-1))

    fmag = soccer_ball_contact_force_magnitude(env, ball_sensor_name)
    weak_contact = fmag <= contact_force_threshold
    near_ball = r_norm <= orbit_radius_max

    core = torch.clamp((v_tan - tangential_deadzone) / tangential_scale, min=0.0, max=1.0)
    return core * (weak_contact & near_ball).to(torch.float32)


# ---------------------------------------------------------------------------
# 3a) Legal Foot Gentle Touch — small positive reward
# ---------------------------------------------------------------------------

def dribbling_legal_foot_touch(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    force_threshold: float = 20.0,
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
) -> torch.Tensor:
    """1.0 when either ankle (first ``num_ankle_links`` in ``all_body_cfg``) gently touches the ball."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    has_contact, force_mag, closest_idx = _identify_contact_body(
        env, command, ball_sensor_name, all_body_cfg,
    )

    is_ankle = _is_dribble_legal_ankle_contact(closest_idx, num_ankle_links)
    gentle = force_mag <= force_threshold
    reward = (has_contact & is_ankle & gentle).to(torch.float32)

    return reward


# ---------------------------------------------------------------------------
# 3b) Micro-Contact Filter — moderate EMA penalty for legal foot hard kicks
# ---------------------------------------------------------------------------

def dribbling_micro_contact_filter(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    force_threshold: float = 20.0,
    max_penalty: float = 2.0,
    ema_alpha: float = 0.4,
    all_body_cfg: SceneEntityCfg | None = None,
    foot_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
) -> torch.Tensor:
    """EMA-smoothed penalty when an **ankle** hits the ball too hard."""

    command: MotionCommand = env.command_manager.get_term(command_name)
    has_contact, force_mag, closest_idx = _identify_contact_body(
        env, command, ball_sensor_name, all_body_cfg,
    )

    is_ankle = _is_dribble_legal_ankle_contact(closest_idx, num_ankle_links)

    # Only consider ankle hard contacts for this penalty
    legal_hard_force = torch.where(
        has_contact & is_ankle,
        force_mag,
        torch.zeros_like(force_mag),
    )

    # ── 5-frame EMA low-pass filter ──────────────────────────────────
    buf_name = "_dribbling_contact_ema"
    ema = getattr(env, buf_name, None)
    if ema is None or ema.shape[0] != env.num_envs:
        ema = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    ema = ema_alpha * legal_hard_force + (1.0 - ema_alpha) * ema

    # Reset EMA for environments that just reset
    step_buf = getattr(env, "episode_length_buf", None)
    if step_buf is not None:
        reset_mask = step_buf == 0
        if torch.any(reset_mask):
            ema[reset_mask] = 0.0

    setattr(env, buf_name, ema)

    # ── Clipped penalty ──────────────────────────────────────────────
    excess = torch.clamp(ema - force_threshold, min=0.0)
    penalty = (excess / force_threshold).clamp(max=max_penalty)

    return penalty


# ---------------------------------------------------------------------------
# 3c) Undesired Contact Penalty — severe instant penalty for wrong body
# ---------------------------------------------------------------------------

def dribbling_undesired_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
) -> torch.Tensor:
    """1.0 when there is contact and the closest body is **not** an ankle link."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    has_contact, force_mag, closest_idx = _identify_contact_body(
        env, command, ball_sensor_name, all_body_cfg,
    )

    is_ankle = _is_dribble_legal_ankle_contact(closest_idx, num_ankle_links)

    penalty = (has_contact & ~is_ankle).to(torch.float32)

    return penalty


# ---------------------------------------------------------------------------
# 4) Annotated contact-graph (dribbling) — demo ball + label consistency
# ---------------------------------------------------------------------------


def dribbling_cg_demo_ball_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 0.32,
) -> torch.Tensor:
    """Shaped tracking of the simulated ball toward the stitched demo trajectory."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    goal_w, mask = command.get_dribble_demo_ball_goal_world()
    if goal_w is None or mask is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    ball = env.scene["soccer_ball"].data.root_pos_w[:, :3]
    err = torch.norm(ball - goal_w, dim=-1)
    rew = torch.exp(-err / max(std, 1e-6))
    return rew * mask.to(torch.float32)


def dribbling_cg_contact_consistency(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """1.0 when sim contact presence matches the annotated CG contact bit."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    labeled = command.motion_has_dribble_cg_label
    if not torch.any(labeled):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    ref = command.dribble_cg_contact_ref
    fmag = soccer_ball_contact_force_magnitude(env, ball_sensor_name)
    sim_c = fmag > contact_force_threshold
    agree = (ref == sim_c).to(torch.float32)
    return agree * labeled.to(torch.float32)


def dribbling_cg_foot_consistency(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    all_body_cfg: SceneEntityCfg | None = None,
    left_ankle_body_name: str = "left_ankle_roll_link",
    right_ankle_body_name: str = "right_ankle_roll_link",
) -> torch.Tensor:
    """When the label specifies a foot during contact, reward matching closest ankle."""
    if all_body_cfg is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    command: MotionCommand = env.command_manager.get_term(command_name)
    labeled = command.motion_has_dribble_cg_label
    if not torch.any(labeled):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    ref_c = command.dribble_cg_contact_ref
    ref_f = command.dribble_cg_foot_ref
    active = labeled & ref_c & (ref_f >= 0)

    has_contact, _fm, closest = _identify_contact_body(env, command, ball_sensor_name, all_body_cfg)
    names = list(all_body_cfg.body_names)
    try:
        li = names.index(left_ankle_body_name)
        ri = names.index(right_ankle_body_name)
    except ValueError:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    expected = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
    expected[ref_f == 0] = li
    expected[ref_f == 1] = ri

    match = (closest == expected) & has_contact & active
    return match.to(torch.float32)


def dribbling_ball_world_forward_progress(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    axis: str = "x",
    sign: float = 1.0,
    min_forward_speed: float = 0.2,
    speed_scale: float = 0.25,
    pelvis_speed_min: float = 0.06,
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 0.5,
    require_recent_contact: bool = True,
    recent_contact_window: int = 10,
) -> torch.Tensor:
    """Forward-progress reward tied to a fixed **world axis** (default world +x).

    Mirrors :func:`dribbling_ball_forward_progress_reward` but uses the world-frame
    ball velocity along ``sign * world_axis`` instead of the pelvis-local +x
    direction. Pair with ``normalize_motion_yaw=True`` on the motion command so
    that "world +x" is also the direction the robot is reset facing — the
    policy is then asked to push the ball along a single global axis regardless
    of how the pelvis turns mid-episode.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    axis_idx = {"x": 0, "y": 1, "z": 2}.get(str(axis).lower(), 0)
    ball_vel_w = soccer_ball.data.root_lin_vel_w[:, :3]
    forward_speed = torch.clamp(float(sign) * ball_vel_w[:, axis_idx], min=0.0)

    base = torch.clamp(
        (forward_speed - min_forward_speed) / max(speed_scale, 1e-6),
        min=0.0,
        max=1.0,
    )

    pelvis_speed = torch.norm(command.robot_anchor_lin_vel_w[:, :2], dim=-1)
    gate = torch.clamp(pelvis_speed / max(pelvis_speed_min, 1e-6), max=1.0)

    if require_recent_contact:
        step_buf = getattr(env, "episode_length_buf", None)
        if step_buf is None:
            step_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

        fmag = soccer_ball_contact_force_magnitude(env, ball_sensor_name)
        has_contact = fmag > contact_force_threshold
        reset_mask = step_buf == 0

        cnt_name = "_dribbling_world_steps_since_contact"
        cnt = getattr(env, cnt_name, None)
        if cnt is None or cnt.shape[0] != env.num_envs:
            cnt = torch.full(
                (env.num_envs,),
                fill_value=recent_contact_window + 1,
                device=env.device,
                dtype=torch.int32,
            )

        cnt = torch.where(
            reset_mask,
            torch.full_like(cnt, recent_contact_window + 1),
            torch.where(has_contact, torch.zeros_like(cnt), cnt + 1),
        )
        setattr(env, cnt_name, cnt)
        recent = cnt <= int(recent_contact_window)
        gate = gate * recent.to(torch.float32)

    return base * gate


def dribbling_face_ball(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    min_distance: float = 0.05,
) -> torch.Tensor:
    """Cosine of the horizontal angle between pelvis forward and the pelvis-to-ball vector.

    Returns ``+1`` when the robot pelvis points straight at the ball, ``0`` when
    the ball is directly sideways, and ``-1`` when the ball is behind the robot.
    Defaults to ``+1`` when the ball is closer than ``min_distance`` (numerically
    unstable region right next to the foot).
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    ball_pos_w = soccer_ball.data.root_pos_w[:, :3]
    pelvis_pos_w = command.robot_pelvis_pos_w
    pelvis_quat_w = command.robot_pelvis_quat_w

    delta_w = ball_pos_w - pelvis_pos_w
    delta_local = quat_apply(quat_inv(pelvis_quat_w), delta_w)
    dx = delta_local[:, 0]
    dy = delta_local[:, 1]
    dist = torch.norm(torch.stack([dx, dy], dim=-1), dim=-1)
    safe = dist > float(min_distance)
    cos_heading = torch.where(
        safe,
        dx / dist.clamp(min=1e-4),
        torch.ones_like(dist),
    )
    return cos_heading.clamp(min=-1.0, max=1.0)
