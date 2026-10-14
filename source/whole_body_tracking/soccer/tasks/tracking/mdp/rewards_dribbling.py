"""Dribbling-specific reward functions.

Encourages close ball control without strike-the-ball objectives. Contact
legality is **side-based**: the first ``num_ankle_links`` entries in
``all_body_cfg.body_names`` are ankle links (typically both feet), and an
instep is classified only as the inside or outside side in the contacted
foot's yaw frame. Knees/wrists listed after incur
``dribbling_s2_nonfoot_ball_contact_penalty`` in S2 when closest to the ball.
S2 consumes the reference event's expected-foot and instep-side labels.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_error_magnitude, yaw_quat

from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import (
    TASK_STATE_STOP,
    MotionCommand,
    locomotion_task_state_mask,
)
from soccer.tasks.tracking.mdp.rewards import motion_relative_foot_position_error_exp
from soccer.tasks.tracking.mdp.task_frame import (
    forward_dominance_gate,
    task_forward_offset,
    task_forward_speed,
    task_lateral_offset,
    task_pelvis_heading_cos_world_x,
    task_velocity_forward_dominance,
)

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
    *,
    horizontal_only: bool = True,
) -> torch.Tensor:
    """Scalar contact-force magnitude on the ball, shape ``(num_envs,)``.

    Defaults to horizontal (XY) magnitude so ground support (mostly Z) is not
    counted as robot-ball contact — same convention as
    ``terminations.contact_phase_violation``.
    """
    f = soccer_ball_contact_net_force_w(env, ball_sensor_name)
    if horizontal_only:
        return torch.norm(f[:, :2], dim=-1)
    return torch.norm(f, dim=-1)


def dribbling_s2_excess_contact_force_penalty(
    env: ManagerBasedRLEnv,
    ball_sensor_name: str = "soccer_ball_contact",
    start_force: float = 150.0,
    full_force: float = 250.0,
    max_penalty: float = 6.0,
) -> torch.Tensor:
    """Increasing penalty for excessive horizontal robot-ball impact.

    Ball-ground support is excluded by using the ball sensor's horizontal net
    force.  Normal soft touches below ``start_force`` are free; the penalty
    is one at ``full_force`` and then continues to increase (up to
    ``max_penalty``).  This keeps a 700-N kick materially worse than a
    250-N kick instead of treating both as the same violation.
    """
    if full_force <= start_force:
        raise ValueError("full_force must exceed start_force")
    force = soccer_ball_contact_force_magnitude(env, ball_sensor_name)
    return torch.clamp(
        (force - float(start_force)) / (float(full_force) - float(start_force)),
        min=0.0,
        max=max(1.0, float(max_penalty)),
    )


def _dribbling_sim_contact(
    env: ManagerBasedRLEnv,
    ball_sensor_name: str,
    contact_force_threshold: float,
) -> torch.Tensor:
    """Bool per env: horizontal ball contact force above threshold."""
    fmag = soccer_ball_contact_force_magnitude(env, ball_sensor_name)
    return fmag > contact_force_threshold


def _dribbling_effective_cg_label_mask(command: MotionCommand, labeled: torch.Tensor) -> torch.Tensor:
    """Disable CG-only objectives in an optional synthesized style seam."""
    seam = getattr(command, "style_seam_bridge_mask", None)
    if isinstance(seam, torch.Tensor) and seam.shape == labeled.shape:
        return labeled & ~seam
    return labeled


def _dribbling_cg_gated_sim_contact(
    command: MotionCommand,
    sim_contact: torch.Tensor,
) -> torch.Tensor:
    """On CG-labeled clips, only count touches on annotated contact frames."""
    if not hasattr(command, "motion_has_dribble_cg_label"):
        return sim_contact
    labeled = _dribbling_effective_cg_label_mask(command, command.motion_has_dribble_cg_label)
    if not torch.any(labeled):
        return sim_contact
    ref = command.dribble_cg_contact_ref
    return torch.where(labeled, sim_contact & ref, sim_contact)


def _dribbling_recent_contact_gate(
    env: ManagerBasedRLEnv,
    ball_sensor_name: str,
    contact_force_threshold: float,
    recent_contact_window: int,
    buf_name: str = "_dribbling_steps_since_contact",
    command: MotionCommand | None = None,
    cg_gated: bool = False,
) -> torch.Tensor:
    """Per-env gate in ``[0, 1]``: 1 iff robot-ball contact within the last N steps."""
    sim_contact = _dribbling_sim_contact(env, ball_sensor_name, contact_force_threshold)
    if cg_gated and command is not None:
        has_contact = _dribbling_cg_gated_sim_contact(command, sim_contact)
    else:
        has_contact = sim_contact

    if recent_contact_window <= 0:
        return has_contact.to(torch.float32)

    step_buf = getattr(env, "episode_length_buf", None)
    if step_buf is None:
        step_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    reset_mask = step_buf == 0

    cnt = getattr(env, buf_name, None)
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
    setattr(env, buf_name, cnt)
    return (cnt <= int(recent_contact_window)).to(torch.float32)


def _command_direction_xy(command: MotionCommand) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the active locomotion direction and speed in world XY.

    The task frame is fixed to world ``+X`` for the legacy forward task.  In
    control, however, the direction must come from the active external command
    so that reward geometry rotates together with a requested heading.
    """
    if hasattr(command, "locomotion_lin_vel_command_w"):
        command_vel_xy = command.locomotion_lin_vel_command_w()[:, :2]
    else:
        command_vel_xy = command.anchor_lin_vel_w[:, :2]
    speed = torch.norm(command_vel_xy, dim=-1)
    # At zero speed (IDLE/STOP), keep the last commanded heading as the ball
    # corridor.  Falling back unconditionally to +X would make a stopped
    # turning run suddenly evaluate its ball geometry in the wrong direction.
    heading = getattr(command, "locomotion_cmd_heading", None)
    if isinstance(heading, torch.Tensor):
        fallback = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1)
    else:
        fallback = torch.zeros_like(command_vel_xy)
        fallback[:, 0] = 1.0
    direction = torch.where(
        (speed > 1.0e-4).unsqueeze(-1),
        command_vel_xy / speed.unsqueeze(-1).clamp(min=1.0e-4),
        fallback,
    )
    return direction, speed


def _command_frame_components(vector_xy: torch.Tensor, direction_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project XY vectors onto the commanded forward and lateral axes."""
    forward = torch.sum(vector_xy * direction_xy, dim=-1)
    lateral_axis = torch.stack((-direction_xy[:, 1], direction_xy[:, 0]), dim=-1)
    lateral = torch.sum(vector_xy * lateral_axis, dim=-1)
    return forward, lateral


def _pelvis_yaw_local_vector(command: MotionCommand, vector_w: torch.Tensor) -> torch.Tensor:
    """Express a world vector in the current pelvis yaw frame."""
    return quat_apply_inverse(yaw_quat(command.robot_pelvis_quat_w), vector_w)


def dribbling_stop_settle_state(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    settled_pelvis_speed: float = 0.05,
    settled_pelvis_angular_speed: float = 0.20,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return the STOP success predicate and its underlying physical measurements.

    The reward and successful-STOP termination call this same helper so an
    episode cannot reset under criteria different from those it was trained to
    optimize.  STOP is intentionally robot-only: the ball may continue rolling
    or leave the control corridor after the final dribble touch.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot = command.robot
    pelvis_id = robot.body_names.index("pelvis")
    soccer_ball = env.scene["soccer_ball"]
    direction_xy, _ = _command_direction_xy(command)
    offset_xy = soccer_ball.data.root_pos_w[:, :2] - command.robot_pelvis_pos_w[:, :2]
    forward_offset, lateral_offset = _command_frame_components(offset_xy, direction_xy)
    ball_speed = torch.norm(soccer_ball.data.root_lin_vel_w[:, :2], dim=-1)
    pelvis_speed = torch.norm(robot.data.body_lin_vel_w[:, pelvis_id, :2], dim=-1)
    pelvis_angular_speed = torch.norm(robot.data.body_ang_vel_w[:, pelvis_id], dim=-1)
    active = locomotion_task_state_mask(command, (TASK_STATE_STOP,))
    settled = (
        active
        & (pelvis_speed <= float(settled_pelvis_speed))
        & (pelvis_angular_speed <= float(settled_pelvis_angular_speed))
    )
    return active, pelvis_speed, pelvis_angular_speed, ball_speed, forward_offset, lateral_offset, settled


def dribbling_velocity_tracking(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 1.0,
    pelvis_speed_min: float = 0.0,
    ball_speed_min: float = 0.0,
    require_contact: bool = False,
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    recent_contact_window: int = 0,
    cg_gated_contact: bool = False,
    min_forward_dominance: float = 0.55,
) -> torch.Tensor:
    """Reward matching task +X (forward) speed between ball and pelvis after a touch.

    Only the forward (+X) velocity components are compared so lateral / crab-walking
    sync (matching ``v_y`` while ``v_x`` is small) does not score highly.

    Optional **anti-cheese gates**:

    - ``pelvis_speed_min`` / ``ball_speed_min``: multiply the reward by
      ``clamp(|v_xy| / min, max=1)`` so near-zero speeds do not yield a full
      score from ``exp(0)==1``.
    - ``require_contact``: multiply by a contact gate. If
      ``recent_contact_window > 0``, contact must have occurred within the last
      N steps (shared counter with forward-progress reward); otherwise the
      current step must show contact.
    - ``min_forward_dominance``: both ball and pelvis XY velocities must be
      sufficiently forward-dominant (task +X share of speed).

    Returns a value in ``[0, 1]`` per environment.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    ball_vel_w = soccer_ball.data.root_lin_vel_w[:, :3]
    pelvis_vel_w = command.robot_anchor_lin_vel_w[:, :3]
    ball_vel_xy = ball_vel_w[:, :2]
    pelvis_vel_xy = pelvis_vel_w[:, :2]

    # Task-frame forward speed only — lateral crab sync must not match.
    ball_fwd = task_forward_speed(ball_vel_w, clamp_forward=False)
    pelvis_fwd = task_forward_speed(pelvis_vel_w, clamp_forward=False)
    error = (ball_fwd - pelvis_fwd) ** 2
    base = torch.exp(-error / (std ** 2))

    pelvis_sp = torch.norm(pelvis_vel_xy, dim=-1)
    ball_sp = torch.norm(ball_vel_xy, dim=-1)

    gate = torch.ones_like(base)
    if pelvis_speed_min > 0.0:
        gate = gate * torch.clamp(pelvis_sp / pelvis_speed_min, max=1.0)
    if ball_speed_min > 0.0:
        gate = gate * torch.clamp(ball_sp / ball_speed_min, max=1.0)

    if require_contact:
        gate = gate * _dribbling_recent_contact_gate(
            env,
            ball_sensor_name,
            contact_force_threshold,
            recent_contact_window,
            command=command,
            cg_gated=cg_gated_contact,
        )

    if min_forward_dominance > 0.0:
        pelvis_dom = task_velocity_forward_dominance(pelvis_vel_xy)
        ball_dom = task_velocity_forward_dominance(ball_vel_xy)
        gate = gate * forward_dominance_gate(pelvis_dom, min_forward_dominance)
        gate = gate * forward_dominance_gate(ball_dom, min_forward_dominance)

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
    """Reward keeping the ball inside a longitudinal safe-zone ahead on task +X.

    Ball offset uses env-local / world-parallel axes (pelvis → ball):
    - x_task in [near_dist, far_dist] → reward = 1.0
    - Outside that range → exponential decay with ``penalty_std``

    Lateral deviation (|y_task|) is also penalised with the same decay.

    If ``pelvis_speed_min > 0``, the reward is multiplied by
    ``clamp(|v_pelvis_xy| / pelvis_speed_min, max=1)`` so standing still in the
    safe zone is not a local optimum.

    If ``no_contact_zone_damping < 1``, when the ball sits in the forward corridor
    (longitudinal band + ``|y_task| <= zone_lateral_abs_max``) but the ball
    sensor reports no contact, the proximity reward is scaled by that factor.
    This reduces the \"park in front of the ball and wiggle\" optimum.

    Returns a value in [0, 1] per environment.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    ball_pos_w = soccer_ball.data.root_pos_w  # (N, 3)
    pelvis_pos_w = command.robot_pelvis_pos_w  # (N, 3)

    x_task = task_forward_offset(ball_pos_w, pelvis_pos_w)
    y_task = task_lateral_offset(ball_pos_w, pelvis_pos_w)

    x_error = torch.where(
        x_task < near_dist,
        near_dist - x_task,
        torch.where(
            x_task > far_dist,
            x_task - far_dist,
            torch.zeros_like(x_task),
        ),
    )

    y_error = torch.abs(y_task)

    total_error = x_error ** 2 + y_error ** 2
    proximity_reward = torch.exp(-total_error / (penalty_std ** 2))

    if pelvis_speed_min > 0.0:
        pelvis_vel_xy = command.robot_anchor_lin_vel_w[:, :2]
        pelvis_sp = torch.norm(pelvis_vel_xy, dim=-1)
        proximity_reward = proximity_reward * torch.clamp(pelvis_sp / pelvis_speed_min, max=1.0)

    if no_contact_zone_damping < 1.0 - 1e-6:
        in_corridor = (
            (x_task >= near_dist)
            & (x_task <= far_dist)
            & (torch.abs(y_task) <= zone_lateral_abs_max)
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


def dribbling_pelvis_local_dynamic_proximity(
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
    """Keep the ball in front of the *current* pelvis, with no world axis.

    Unlike the task/world-frame corridor, this stays meaningful after any
    accumulated yaw drift and is therefore suitable for a local-twist policy.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    offset_b = _pelvis_yaw_local_vector(
        command, soccer_ball.data.root_pos_w[:, :3] - command.robot_pelvis_pos_w
    )
    forward_offset, lateral_offset = offset_b[:, 0], offset_b[:, 1]
    forward_error = torch.where(
        forward_offset < near_dist,
        near_dist - forward_offset,
        torch.where(forward_offset > far_dist, forward_offset - far_dist, torch.zeros_like(forward_offset)),
    )
    reward = torch.exp(-(forward_error.square() + lateral_offset.square()) / max(penalty_std, 1.0e-6) ** 2)
    if pelvis_speed_min > 0.0:
        pelvis_speed = torch.norm(command.robot_anchor_lin_vel_w[:, :2], dim=-1)
        reward = reward * torch.clamp(pelvis_speed / pelvis_speed_min, max=1.0)
    if no_contact_zone_damping < 1.0 - 1.0e-6:
        in_corridor = (
            (forward_offset >= near_dist)
            & (forward_offset <= far_dist)
            & (torch.abs(lateral_offset) <= zone_lateral_abs_max)
        )
        no_touch = soccer_ball_contact_force_magnitude(env, ball_sensor_name) <= contact_force_threshold
        reward = torch.where(in_corridor & no_touch, reward * no_contact_zone_damping, reward)
    return reward


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
    active_task_states: tuple[int, ...] | None = None,
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
    active = locomotion_task_state_mask(command, active_task_states)
    return (near & slow & no_touch & active).to(torch.float32)


# ---------------------------------------------------------------------------
# Helper: identify which robot body caused the ball contact
# ---------------------------------------------------------------------------

def _identify_contact_body(
    env: ManagerBasedRLEnv,
    command: MotionCommand,
    ball_sensor_name: str,
    all_body_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Identify the contacting foot from the ball sensor's filtered force matrix.

    Returns:
        has_contact: (N,) bool — whether ball has non-zero contact force
        contact_force_mag: (N,) float — force magnitude on ball
        closest_body_idx: (N,) long — index into ``all_body_cfg.body_names``.

    The S2 sensor filters the ball against the two ankle links in the same
    order as the leading entries of ``all_body_cfg``.  This makes a contact a
    real ball-foot collision.  The former unfiltered XY-force + nearest-body
    heuristic could classify ball-ground friction as a foot touch.
    """
    device = env.device
    num_envs = env.num_envs

    # Default outputs
    has_contact = torch.zeros(num_envs, dtype=torch.bool, device=device)
    closest_body_idx = torch.zeros(num_envs, dtype=torch.long, device=device)

    sensor = env.scene.sensors[ball_sensor_name]
    force_history = getattr(sensor.data, "force_matrix_w_history", None)
    if isinstance(force_history, torch.Tensor) and force_history.ndim == 5:
        # (env, history, ball_body=1, filtered_robot_body, xyz)
        filtered_force = force_history[:, :, 0]
        force_magnitude = torch.linalg.vector_norm(filtered_force, dim=-1)
        flat_magnitude = force_magnitude.reshape(num_envs, -1)
        contact_force_mag, flat_index = torch.max(flat_magnitude, dim=-1)
        filter_count = max(1, filtered_force.shape[2])
        closest_body_idx = flat_index % filter_count
        # Filtered PhysX contact-force entries are exactly zero when the pair
        # is not touching.  Do not insert a hand-tuned force threshold into
        # the definition of a labelled contact.
        has_contact = contact_force_mag > 0.0
        closest_body_idx = torch.where(
            has_contact, closest_body_idx, torch.zeros_like(closest_body_idx)
        )
        return has_contact, contact_force_mag, closest_body_idx

    force_matrix = getattr(sensor.data, "force_matrix_w", None)
    if isinstance(force_matrix, torch.Tensor) and force_matrix.ndim == 4:
        filtered_force = force_matrix[:, 0]
        force_magnitude = torch.linalg.vector_norm(filtered_force, dim=-1)
        contact_force_mag, closest_body_idx = torch.max(force_magnitude, dim=-1)
        has_contact = contact_force_mag > 0.0
        closest_body_idx = torch.where(
            has_contact, closest_body_idx, torch.zeros_like(closest_body_idx)
        )
        return has_contact, contact_force_mag, closest_body_idx

    # Compatibility fallback for tasks that do not configure contact filters.
    force_vec = soccer_ball_contact_net_force_w(env, ball_sensor_name)
    force_mag = torch.norm(force_vec[:, :2], dim=-1)
    has_contact = force_mag > 0.0
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


_DRIBBLE_CONTACT_SURFACES = ("any", "instep", "inside_instep", "outside_instep")


def _dribbling_contact_surface_geometry(
    env: ManagerBasedRLEnv,
    all_body_cfg: SceneEntityCfg | None,
    closest_body_idx: torch.Tensor,
    *,
    num_ankle_links: int,
    medial_y_min: float,
) -> dict[str, torch.Tensor]:
    """Compute the shared foot-yaw inside/outside contact geometry.

    The reward and playback diagnostic both consume this helper. Keeping the
    coordinates and thresholds in one place makes a recorded ``instep`` match
    exactly the match used by the learning objective.
    """
    device = env.device
    num_envs = env.num_envs
    dtype = torch.float32
    empty = {
        "ball_offset_foot_yaw": torch.full((num_envs, 3), torch.nan, device=device, dtype=dtype),
        "medial_offset": torch.full((num_envs,), torch.nan, device=device, dtype=dtype),
        "medial_sign": torch.zeros(num_envs, device=device, dtype=dtype),
        "legal_ankle": torch.zeros(num_envs, dtype=torch.bool, device=device),
        "known_foot": torch.zeros(num_envs, dtype=torch.bool, device=device),
        "instep": torch.zeros(num_envs, dtype=torch.bool, device=device),
        "inside_instep": torch.zeros(num_envs, dtype=torch.bool, device=device),
        "outside_instep": torch.zeros(num_envs, dtype=torch.bool, device=device),
    }
    if all_body_cfg is None or num_ankle_links <= 0:
        return empty
    if medial_y_min < 0.0:
        raise ValueError("medial_y_min must be non-negative.")

    names = tuple(all_body_cfg.body_names)
    if not names:
        return empty

    robot = env.scene[all_body_cfg.name]
    cache_name = "_dribbling_body_indices_cache"
    cached = getattr(env, cache_name, None)
    if cached is None or cached.get("names") != names:
        body_indices = torch.as_tensor(
            robot.find_bodies(all_body_cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=device,
        )
        setattr(env, cache_name, {"names": names, "idx": body_indices})
    body_indices = getattr(env, cache_name)["idx"]

    selected_cfg_idx = closest_body_idx.clamp(min=0, max=len(names) - 1)
    selected_body_idx = body_indices[selected_cfg_idx]
    batch_ids = torch.arange(num_envs, device=device)
    foot_pos_w = robot.data.body_pos_w[batch_ids, selected_body_idx]
    foot_yaw_w = yaw_quat(robot.data.body_quat_w[batch_ids, selected_body_idx])
    ball_pos_w = env.scene["soccer_ball"].data.root_pos_w[:, :3]
    ball_local = quat_apply_inverse(foot_yaw_w, ball_pos_w - foot_pos_w)

    # In each foot yaw frame +Y is its local left side. Thus the right foot's
    # medial direction is +Y while the left foot's medial direction is -Y.
    medial_sign = torch.zeros(num_envs, dtype=ball_local.dtype, device=device)
    for cfg_idx, body_name in enumerate(names):
        if "right_ankle" in body_name:
            medial_sign[selected_cfg_idx == cfg_idx] = 1.0
        elif "left_ankle" in body_name:
            medial_sign[selected_cfg_idx == cfg_idx] = -1.0

    medial_offset = ball_local[:, 1] * medial_sign
    known_foot = medial_sign != 0.0
    legal_ankle = _is_dribble_legal_ankle_contact(closest_body_idx, num_ankle_links)
    inside_instep = medial_offset >= medial_y_min
    outside_instep = medial_offset <= -medial_y_min
    return {
        "ball_offset_foot_yaw": ball_local,
        "medial_offset": medial_offset,
        "medial_sign": medial_sign,
        "legal_ankle": legal_ankle,
        "known_foot": known_foot,
        "instep": inside_instep | outside_instep,
        "inside_instep": inside_instep,
        "outside_instep": outside_instep,
    }


def _dribbling_cg_reference_surface_match(
    command: MotionCommand,
    geometry: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Match the current inside/outside region to the per-frame CG label.

    ``dribble_cg_surface`` stores ``0=inside_instep`` and
    ``1=outside_instep``. A missing label never silently becomes a legal
    reference-surface touch: S2 must either supply the label or use a task
    that does not request reference-surface gating.
    """
    reference_surface = getattr(command, "dribble_cg_surface_ref", None)
    if not isinstance(reference_surface, torch.Tensor):
        return torch.zeros_like(geometry["legal_ankle"])
    return (
        ((reference_surface == 0) & geometry["inside_instep"])
        | ((reference_surface == 1) & geometry["outside_instep"])
    )


def _dribbling_contact_surface_match(
    env: ManagerBasedRLEnv,
    all_body_cfg: SceneEntityCfg | None,
    closest_body_idx: torch.Tensor,
    *,
    command_name: str = "motion",
    contact_surface: str,
    num_ankle_links: int,
    medial_y_min: float = 0.018,
    cg_surface_gated: bool = False,
) -> torch.Tensor:
    """Check a requested instep side in the contacted foot's yaw frame.

    Isaac's ball contact sensor reports the net force on the ball but not the
    individual collision capsule that produced it.  We therefore identify the
    closest contacted ankle link and express the ankle-to-ball offset in that
    foot's yaw frame. Only its signed lateral coordinate is used: right-foot
    medial is ``+Y`` and left-foot medial is ``-Y``. There is deliberately no
    fore/aft or height gate; the task distinguishes only inside from outside.
    """
    surface = str(contact_surface).lower().strip()
    if surface not in _DRIBBLE_CONTACT_SURFACES:
        raise ValueError(
            f"Unsupported contact_surface={contact_surface!r}; expected one of {_DRIBBLE_CONTACT_SURFACES}."
        )
    if surface == "any" and not cg_surface_gated:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    geometry = _dribbling_contact_surface_geometry(
        env,
        all_body_cfg,
        closest_body_idx,
        num_ankle_links=num_ankle_links,
        medial_y_min=medial_y_min,
    )
    if surface == "any":
        surface_geometry = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    elif surface == "instep":
        surface_geometry = geometry["instep"]
    elif surface == "inside_instep":
        surface_geometry = geometry["inside_instep"]
    else:
        surface_geometry = geometry["outside_instep"]
    match = geometry["legal_ankle"] & geometry["known_foot"] & surface_geometry
    if cg_surface_gated:
        command: MotionCommand = env.command_manager.get_term(command_name)
        match = match & _dribbling_cg_reference_surface_match(command, geometry)
    return match


def _dribbling_s2_proximity_geometry(
    ball_from_foot: torch.Tensor,
    expected_foot: torch.Tensor,
    expected_surface: torch.Tensor,
    *,
    target_side_enabled: bool,
    side_deadzone: float,
    contact_distance_max: float,
) -> dict[str, torch.Tensor]:
    """Return physical reach and labelled inside/outside errors."""
    if float(contact_distance_max) <= 0.0:
        raise ValueError("contact_distance_max must be positive")

    physical_distance = torch.linalg.vector_norm(ball_from_foot, dim=-1)
    reach_error = torch.relu(physical_distance - float(contact_distance_max))
    lateral = ball_from_foot[:, 1]
    # Source labels are foot-independent: 0=inside, 1=outside. Convert to a
    # signed medial coordinate here, where the actual foot frame is known.
    # Right-foot medial is +Y; left-foot medial is -Y.
    medial_sign = torch.where(
        expected_foot == 1, torch.ones_like(lateral), -torch.ones_like(lateral)
    )
    medial_lateral = lateral * medial_sign
    signed_lateral = torch.where(
        expected_surface == 0, medial_lateral, -medial_lateral
    )
    if target_side_enabled:
        # The annotation tells us only which side of the foot owns the touch;
        # it does not provide a calibrated lateral clearance.  With a zero
        # dead-zone, use the sign itself as the categorical label and retain
        # only continuous wrong-side distance for shaping/telemetry.
        if float(side_deadzone) <= 0.0:
            side_error = torch.relu(-signed_lateral)
            side_correct = signed_lateral >= 0.0
            side_wrong = signed_lateral < 0.0
        else:
            side_error = torch.relu(float(side_deadzone) - signed_lateral)
            side_correct = signed_lateral >= float(side_deadzone)
            side_wrong = signed_lateral <= -float(side_deadzone)
    else:
        side_error = torch.zeros_like(lateral)
        side_correct = torch.abs(lateral) >= float(side_deadzone)
        side_wrong = torch.zeros_like(side_correct)

    target_distance = torch.sqrt(reach_error.square() + side_error.square())
    return {
        "target_distance": target_distance,
        "physical_distance": physical_distance,
        "lateral": lateral,
        "medial_lateral": medial_lateral,
        "side_correct": side_correct,
        "side_wrong": side_wrong,
        "side_error": side_error,
    }


def _dribbling_s2_target_event(command: MotionCommand) -> dict[str, torch.Tensor]:
    """Select the current-window event, otherwise the next event."""
    contact_event_id = command.s2_contact_event_id_ref
    contact_event_frame = command.s2_contact_event_frame_ref
    contact_expected_foot = command.s2_contact_event_foot_ref
    contact_expected_surface = command.s2_contact_event_surface_ref
    contact_window = (contact_event_id >= 0) & (
        (contact_expected_foot == 0) | (contact_expected_foot == 1)
    )
    (
        upcoming_event_id,
        upcoming_event_frame,
        upcoming_expected_foot,
        upcoming_expected_surface,
    ) = command.s2_upcoming_contact_event_ref()
    return {
        "contact_window": contact_window,
        "upcoming_event_id": upcoming_event_id,
        "upcoming_event_frame": upcoming_event_frame,
        "event_id": torch.where(contact_window, contact_event_id, upcoming_event_id),
        "event_frame": torch.where(
            contact_window, contact_event_frame, upcoming_event_frame
        ),
        "expected_foot": torch.where(
            contact_window, contact_expected_foot, upcoming_expected_foot
        ),
        "expected_surface": torch.where(
            contact_window, contact_expected_surface, upcoming_expected_surface
        ),
    }


def _dribbling_s2_release_components(
    robust_velocity: torch.Tensor,
    target_direction: torch.Tensor,
    nominal_speed: torch.Tensor,
    separation_distance: torch.Tensor,
    *,
    speed_lower_ratio: float,
    speed_upper_ratio: float,
    lateral_speed_std: float,
    overspeed_decay_ratio: float,
    separation_min: float,
    separation_full: float,
) -> dict[str, torch.Tensor]:
    """Pure tensor scoring for an outgoing S2 release."""
    lateral_axis = torch.stack(
        (-target_direction[:, 1], target_direction[:, 0]), dim=-1
    )
    parallel_speed = torch.sum(robust_velocity * target_direction, dim=-1)
    lateral_speed = torch.abs(torch.sum(robust_velocity * lateral_axis, dim=-1))
    speed_ratio = parallel_speed / nominal_speed.clamp(min=1.0e-6)
    velocity_speed = torch.linalg.vector_norm(robust_velocity, dim=-1)
    direction_cosine = parallel_speed / velocity_speed.clamp(min=1.0e-6)
    direction_score = torch.clamp(direction_cosine, min=0.0, max=1.0) * torch.exp(
        -torch.square(lateral_speed / max(float(lateral_speed_std), 1.0e-6))
    )
    underspeed_score = torch.clamp(
        speed_ratio / float(speed_lower_ratio), min=0.0, max=1.0
    )
    overspeed_score = torch.where(
        speed_ratio <= float(speed_upper_ratio),
        torch.ones_like(speed_ratio),
        torch.exp(
            -torch.square(
                (speed_ratio - float(speed_upper_ratio))
                / max(float(overspeed_decay_ratio), 1.0e-6)
            )
        ),
    )
    speed_band_score = underspeed_score * overspeed_score
    separation_score = torch.clamp(
        (separation_distance - float(separation_min))
        / (float(separation_full) - float(separation_min)),
        min=0.0,
        max=1.0,
    )
    return {
        "parallel_speed": parallel_speed,
        "lateral_speed": lateral_speed,
        "speed_ratio": speed_ratio,
        "direction_score": direction_score,
        "speed_band_score": speed_band_score,
        "separation_score": separation_score,
        "base_score": direction_score * speed_band_score * separation_score,
    }


def dribbling_s2_contact_event_state(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
    require_expected_foot: bool = True,
    target_side_enabled: bool = True,
    side_deadzone: float = 0.04,
    proximity_contact_distance_max: float = 0.25,
    target_region_std: float = 0.12,
    proximity_approach_seconds: float = 0.30,
    proximity_approach_min_weight: float = 0.20,
    missed_contact_grace_steps: int = 3,
    approach_distance_std: float = 0.40,
    approach_progress_max: float = 4.0,
    anchor_approach_duration: float = 0.20,
    anchor_approach_local_scales: tuple[float, float, float] = (0.126, 0.126, 0.066),
    anchor_approach_progress_max: float = 4.0,
    contact_region_approach_duration: float = 0.20,
    contact_region_progress_max: float = 4.0,
    contact_region_arrival_std: float = 0.02,
    coarse_approach_duration: float = 0.20,
    coarse_approach_progress_max: float = 4.0,
    release_window_steps: int = 8,
    release_speed_lower_ratio: float = 0.65,
    release_speed_upper_ratio: float = 1.35,
    release_lateral_speed_std: float = 0.35,
    release_overspeed_decay_ratio: float = 0.35,
    release_velocity_outlier_scale: float = 0.75,
    release_separation_min: float = 0.14,
    release_separation_full: float = 0.22,
    release_quality_threshold: float = 0.50,
    release_v2_quality_threshold: float = 0.05,
    release_min_target_distance: float = 0.05,
    release_arrival_progress_max: float = 4.0,
    acquisition_min_history_steps: int = 20,
    contact_anchor_std: float | None = None,
    contact_anchor_max_distance: float | None = None,
    contact_anchor_local_sigmas: tuple[float, float, float] | None = None,
    contact_anchor_quality_min: float | None = None,
    contact_onset_early_grace_steps: int | None = None,
    contact_release_no_force_steps: int = 3,
    pair_terminal_contact_region_max_distance: float | None = None,
    coarse_contact_longitudinal_min: float | None = 0.0,
    coarse_contact_longitudinal_max: float | None = 0.30,
    relative_node_targets: bool = False,
) -> dict[str, torch.Tensor]:
    """Evaluate S2 approach, contact, release, and diagnostics once per step."""
    if release_window_steps <= 0:
        raise ValueError("release_window_steps must be positive")
    if not 0.0 < release_speed_lower_ratio <= release_speed_upper_ratio:
        raise ValueError("Expected 0 < release_speed_lower_ratio <= release_speed_upper_ratio")
    if release_separation_full <= release_separation_min:
        raise ValueError("release_separation_full must exceed release_separation_min")
    command: MotionCommand = env.command_manager.get_term(command_name)
    required = (
        "s2_contact_event_id_ref",
        "s2_contact_event_frame_ref",
        "s2_contact_event_foot_ref",
        "s2_contact_event_surface_ref",
        "s2_upcoming_contact_event_ref",
    )
    if all_body_cfg is None or any(not hasattr(command, name) for name in required):
        zero = torch.zeros(env.num_envs, device=env.device)
        false = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        return {
            "contact_window": false,
            "approach_progress": zero,
            "coarse_contact_approach": zero,
            "anchor_approach_progress_v2": zero,
            "contact_region_progress_v3": zero,
            "contact_region_arrival": zero,
            "contact_region_distance": zero,
            "nearest_region_point_id": torch.full_like(zero, -1, dtype=torch.long),
            "nearest_region_point_timing": torch.full_like(zero, -1, dtype=torch.long),
            "contact_proximity": zero,
            "target_region_distance": zero,
            "contact_anchor_distance": zero,
            "contact_anchor_quality": zero,
            "contact_anchor_arrival": zero,
            "physical_foot_ball_distance": zero,
            "touch_occurrence": false,
            "immediate_contact": false,
            "surface_style": zero,
            "valid_contact_event": false,
            "contact_force_magnitude": zero,
            "dead_zone": false,
            "wrong_foot": false,
            "wrong_side": false,
            "premature_contact": false,
            "invalid_body_contact": false,
            "out_of_sequence_contact": false,
            "sequence_eligible": false,
            "sequence_cursor": torch.full_like(zero, -1, dtype=torch.long),
            "nonfoot_contact_penalty": zero,
            "wrong_foot_contact_penalty": zero,
            "premature_contact_penalty": zero,
            "missed_valid_contact": false,
            "release_pending": false,
            "release_arrival_progress": zero,
            "terminal_release": false,
            "terminal_transition_window": false,
            "terminal_progress_raw_rate": zero,
            "terminal_progress_clipped_rate": zero,
            "terminal_progress_positive_rate": zero,
            "terminal_progress_negative_rate": zero,
            "terminal_transition_valid": false,
            "terminal_transition_success": false,
            "terminal_transition_distance": zero,
            "terminal_transition_start_distance": zero,
            "terminal_transition_end_distance": zero,
            "terminal_transition_net_progress": zero,
            "terminal_tracking_quality": zero,
            "terminal_tracking_pulse": zero,
            "terminal_window_position_outcome": zero,
            "window_outcome_active": false,
            "window_outcome_source_event_id": torch.full_like(zero, -1, dtype=torch.long),
            "window_outcome_target_event_id": torch.full_like(zero, -1, dtype=torch.long),
            "window_outcome_candidate_frame": torch.full_like(zero, -1, dtype=torch.long),
            "window_outcome_start_distance": zero,
            "window_outcome_score_sum": zero,
            "window_outcome_sample_count": zero,
            "window_outcome_score_average": zero,
            "window_outcome_score_end": zero,
            "window_outcome_settlement_quality": zero,
            "window_outcome_settlement_pulse": zero,
            "window_outcome_pulse": zero,
            "window_outcome_emitted": false,
            "window_outcome_aborted_ball_lost": false,
            "window_outcome_aborted_other": false,
            "release_actual_touch_frame": torch.full_like(zero, -1, dtype=torch.long),
            "release_next_anchor_w": torch.zeros(env.num_envs, 3, device=env.device),
            "release_remaining_time": zero,
            "release_target_distance": zero,
            "release_target_direction": torch.zeros(env.num_envs, 2, device=env.device),
            "release_nominal_speed": zero,
            "release_parallel_speed": zero,
            "release_lateral_speed": zero,
            "release_speed_ratio": zero,
            "release_separation_score": zero,
            "release_score": zero,
            "release_v2_score": zero,
            "release_quality_pass": false,
            "release_v2_quality_pass": false,
            "quality_streak": torch.zeros_like(zero, dtype=torch.long),
            "sequence_progress": zero,
            "second_touch_conditional_success": false,
            "second_touch_conditional_success_v2": false,
            "acquisition_invalid_early_contact": false,
            "acquisition_history_steps_before_contact": zero,
            "timing_error_seconds": zero,
            "event_id": torch.full_like(zero, -1, dtype=torch.long),
            "expected_foot": torch.full_like(zero, -1, dtype=torch.long),
        }

    step = getattr(
        env, "episode_length_buf", torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    ).to(torch.long)
    frame = command.time_steps.to(torch.long)
    motion_idx = command.motion_idx.to(torch.long)
    generation = getattr(command, "s2_episode_generation", None)
    if not isinstance(generation, torch.Tensor) or generation.shape[0] != env.num_envs:
        generation = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    cached = getattr(env, "_dribbling_s2_contact_event_cache", None)
    if (
        isinstance(cached, dict)
        and isinstance(cached.get("step"), torch.Tensor)
        and torch.equal(cached["step"], step)
        and isinstance(cached.get("frame"), torch.Tensor)
        and torch.equal(cached["frame"], frame)
        and isinstance(cached.get("motion_idx"), torch.Tensor)
        and torch.equal(cached["motion_idx"], motion_idx)
        and isinstance(cached.get("generation"), torch.Tensor)
        and torch.equal(cached["generation"], generation)
    ):
        return cached["state"]

    previous_generation = getattr(env, "_dribbling_s2_episode_generation", None)
    if not isinstance(previous_generation, torch.Tensor) or previous_generation.shape[0] != env.num_envs:
        reset = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        reset = generation != previous_generation
    setattr(env, "_dribbling_s2_episode_generation", generation.detach().clone())
    zeros = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    false = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    minus_one = torch.full(
        (env.num_envs,), -1, device=env.device, dtype=torch.long
    )
    target_event = _dribbling_s2_target_event(command)
    active = target_event["contact_window"]
    # During the contact window the current label wins. Before the window,
    # only the next event's time/foot/side label is exposed for approach
    # shaping; no reference ball position participates in the reward.
    event_id = target_event["event_id"]
    event_frame = target_event["event_frame"]
    expected_foot = target_event["expected_foot"]
    expected_surface = target_event["expected_surface"]
    upcoming_event_id = target_event["upcoming_event_id"]
    upcoming_event_frame = target_event["upcoming_event_frame"]

    ball_pos_w = env.scene["soccer_ball"].data.root_pos_w[:, :3]
    anchor_lookup = getattr(command, "s2_contact_anchor_target_w", None)
    if callable(anchor_lookup):
        contact_anchor_w, contact_anchor_valid = anchor_lookup(event_id)
    else:
        contact_anchor_w = torch.zeros(env.num_envs, 3, device=env.device)
        contact_anchor_valid = false
    contact_anchor_distance = torch.linalg.vector_norm(
        ball_pos_w[:, :2] - contact_anchor_w[:, :2], dim=-1
    )
    first_event_index = getattr(command, "s2_episode_first_event_index", minus_one)

    has_contact, force_magnitude, closest_body = _identify_contact_body(
        env, command, ball_sensor_name, all_body_cfg
    )
    names = tuple(all_body_cfg.body_names)
    try:
        left_cfg_index = names.index("left_ankle_roll_link")
        right_cfg_index = names.index("right_ankle_roll_link")
    except ValueError as exc:
        raise ValueError("S2 contact body config must contain both ankle roll links") from exc
    expected_body = torch.where(
        expected_foot == 1,
        torch.full_like(expected_foot, right_cfg_index),
        torch.full_like(expected_foot, left_cfg_index),
    )
    legal_ankle = _is_dribble_legal_ankle_contact(closest_body, num_ankle_links)
    correct_foot = legal_ankle & (closest_body == expected_body)

    robot = env.scene[all_body_cfg.name]
    pose_cache_name = "_dribbling_s2_foot_pose_indices"
    pose_indices = getattr(env, pose_cache_name, None)
    if not isinstance(pose_indices, torch.Tensor):
        resolved = robot.find_bodies(
            ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
        )[0]
        pose_indices = torch.as_tensor(resolved, dtype=torch.long, device=env.device)
        setattr(env, pose_cache_name, pose_indices)
    selected_pose_index = torch.where(
        expected_foot == 1,
        torch.full_like(expected_foot, int(pose_indices[1].item())),
        torch.full_like(expected_foot, int(pose_indices[0].item())),
    )
    batch = torch.arange(env.num_envs, device=env.device)
    actual_foot_pos = robot.data.body_pos_w[batch, selected_pose_index]
    actual_foot_quat = robot.data.body_quat_w[batch, selected_pose_index]
    actual_foot_yaw = yaw_quat(robot.data.body_quat_w[batch, selected_pose_index])
    # This is ordinary live foot-ball geometry used by contact logic and
    # diagnostics.  It must remain available when the optional V3 point cloud
    # is disabled.
    ball_in_actual_foot = quat_apply_inverse(
        actual_foot_quat, ball_pos_w - actual_foot_pos
    )
    ball_from_actual_foot = quat_apply_inverse(
        actual_foot_yaw, ball_pos_w - actual_foot_pos
    )
    proximity_geometry = _dribbling_s2_proximity_geometry(
        ball_from_actual_foot,
        expected_foot,
        expected_surface,
        target_side_enabled=target_side_enabled,
        side_deadzone=side_deadzone,
        contact_distance_max=proximity_contact_distance_max,
    )
    current_lateral = proximity_geometry["lateral"]
    target_distance = proximity_geometry["target_distance"]
    physical_foot_ball_distance = proximity_geometry["physical_distance"]
    # Near the event, retain the existing side-aware region objective.
    region_scale = max(float(target_region_std), 1.0e-6)
    normalized_distance = target_distance / region_scale
    fps = float(command.motion.fps.reshape(-1)[0])
    seconds_to_event = (event_frame - frame).to(torch.float32) / max(fps, 1.0e-6)
    approach_duration = max(0.0, float(proximity_approach_seconds))
    approach_active = (
        (event_id >= 0)
        & (seconds_to_event > 0.0)
        & (seconds_to_event <= approach_duration)
    )
    proximity_active = active | approach_active
    if approach_duration > 0.0:
        approach_progress = 1.0 - torch.clamp(
            seconds_to_event / approach_duration, min=0.0, max=1.0
        )
    else:
        approach_progress = torch.ones_like(seconds_to_event)
    min_approach_weight = min(
        1.0, max(0.0, float(proximity_approach_min_weight))
    )
    approach_weight = min_approach_weight + (
        1.0 - min_approach_weight
    ) * approach_progress
    approach_weight = torch.where(active, torch.ones_like(approach_weight), approach_weight)
    proximity = (
        1.0 / (1.0 + normalized_distance.square())
    ) * proximity_active.to(torch.float32) * approach_weight
    # Sparse event-anchor supervision: guide the physical ball only during the
    # final approach/contact window.  The ball is deliberately unconstrained
    # between events, so PhysX can choose any viable rolling path.
    # Use the physical ball radius as the shaping length scale unless an
    # explicit ablation overrides it.  This keeps anchor error continuous and
    # dimensionless without inventing another centimetre tolerance.
    anchor_error_local = quat_apply_inverse(
        actual_foot_quat, ball_pos_w - contact_anchor_w
    )
    if contact_anchor_local_sigmas is not None:
        if len(contact_anchor_local_sigmas) != 3 or any(
            float(value) <= 0.0 for value in contact_anchor_local_sigmas
        ):
            raise ValueError("contact_anchor_local_sigmas must contain three positive values")
        anchor_sigmas = torch.tensor(
            contact_anchor_local_sigmas,
            dtype=anchor_error_local.dtype,
            device=env.device,
        )
        # Continuous anisotropic geometry quality in the actual foot frame.
        # The physical collision remains mandatory; this quality only rejects
        # contacts outside the calibrated manual-anchor neighbourhood.
        contact_anchor_quality = torch.exp(
            -0.5 * torch.sum(torch.square(anchor_error_local / anchor_sigmas), dim=-1)
        )
    else:
        anchor_scale = (
            float(contact_anchor_std)
            if contact_anchor_std is not None
            else float(getattr(command.cfg, "dribble_cg_ball_radius", 0.11))
        )
        contact_anchor_quality = torch.exp(
            -torch.square(contact_anchor_distance / max(anchor_scale, 1.0e-6))
        )
    contact_anchor_arrival = (
        contact_anchor_quality
        * proximity_active.to(torch.float32)
        * contact_anchor_valid.to(torch.float32)
    )
    # Keep this signal for the first event as well.  Although the physical
    # ball is spawned at the reference O1 point, ``contact_anchor_w`` is the
    # labelled foot-local offset mapped through the *actual* foot pose.  Its
    # error is therefore zero only when the policy places the real foot at the
    # labelled contact geometry; it is not a free reset-position reward.

    # Far from the event, reward only improvement in physical distance.  A
    # potential difference prevents the old stationary "stay glued to ball"
    # optimum while remaining complementary to the near side-aware term.
    approach_std = max(float(approach_distance_std), 1.0e-6)
    approach_potential = torch.exp(
        -torch.square(physical_foot_ball_distance / approach_std)
    )
    previous_approach_event = getattr(env, "_dribbling_s2_approach_event", None)
    previous_approach_potential = getattr(env, "_dribbling_s2_approach_potential", None)
    if not isinstance(previous_approach_event, torch.Tensor) or previous_approach_event.shape[0] != env.num_envs:
        previous_approach_event = minus_one.clone()
    if (
        not isinstance(previous_approach_potential, torch.Tensor)
        or previous_approach_potential.shape[0] != env.num_envs
    ):
        previous_approach_potential = zeros.clone()
    previous_approach_event = torch.where(reset, minus_one, previous_approach_event)
    previous_approach_potential = torch.where(reset, zeros, previous_approach_potential)
    far_approach_active = (
        (event_id >= 0)
        & (seconds_to_event > approach_duration)
        & ~active
    )
    same_approach_event = far_approach_active & (previous_approach_event == event_id)
    step_dt = max(float(getattr(env, "step_dt", 0.02)), 1.0e-6)
    approach_progress_reward = torch.clamp(
        (approach_potential - previous_approach_potential) / step_dt,
        min=-max(0.0, float(approach_progress_max)),
        max=max(0.0, float(approach_progress_max)),
    ) * same_approach_event.to(torch.float32)
    previous_approach_event = torch.where(
        far_approach_active, event_id, minus_one
    )
    previous_approach_potential = torch.where(
        far_approach_active, approach_potential, zeros
    )
    setattr(env, "_dribbling_s2_approach_event", previous_approach_event)
    setattr(env, "_dribbling_s2_approach_potential", previous_approach_potential)

    # Treat every uninterrupted PhysX ball-contact interval as one physical
    # episode.  A labelled event may own an episode only when that episode's
    # re-armed physical onset occurs inside the event's annotated timing
    # window.  Once owned, the episode remains eligible over its consecutive
    # contact frames so foot/side geometry may settle one step after onset.
    # An onset before the window can therefore never become a candidate merely
    # by remaining in contact until the window opens.  The same physical
    # episode also cannot complete two nearby labels: O1 and O2 require a real
    # release and re-arm between them.
    physical_episode = getattr(
        env, "_dribbling_s2_physical_contact_episode", None
    )
    physical_start_frame = getattr(
        env, "_dribbling_s2_physical_contact_start_frame", None
    )
    claimed_episode = getattr(
        env, "_dribbling_s2_claimed_contact_episode", None
    )
    claimed_event = getattr(env, "_dribbling_s2_claimed_contact_event", None)
    claimed_body = getattr(env, "_dribbling_s2_claimed_contact_body", None)
    if not isinstance(physical_episode, torch.Tensor) or physical_episode.shape[0] != env.num_envs:
        physical_episode = torch.zeros_like(frame)
    if not isinstance(physical_start_frame, torch.Tensor) or physical_start_frame.shape[0] != env.num_envs:
        physical_start_frame = minus_one.clone()
    if not isinstance(claimed_episode, torch.Tensor) or claimed_episode.shape[0] != env.num_envs:
        claimed_episode = minus_one.clone()
    if not isinstance(claimed_event, torch.Tensor) or claimed_event.shape[0] != env.num_envs:
        claimed_event = minus_one.clone()
    if not isinstance(claimed_body, torch.Tensor) or claimed_body.shape[0] != env.num_envs:
        claimed_body = minus_one.clone()
    release_off_count = getattr(env, "_dribbling_s2_contact_release_off_count", None)
    contact_rearmed = getattr(env, "_dribbling_s2_contact_rearmed", None)
    if not isinstance(release_off_count, torch.Tensor) or release_off_count.shape[0] != env.num_envs:
        release_off_count = torch.zeros_like(frame)
    if not isinstance(contact_rearmed, torch.Tensor) or contact_rearmed.shape[0] != env.num_envs:
        contact_rearmed = torch.ones_like(false)
    physical_episode = torch.where(reset, torch.zeros_like(physical_episode), physical_episode)
    physical_start_frame = torch.where(reset, minus_one, physical_start_frame)
    claimed_episode = torch.where(reset, minus_one, claimed_episode)
    claimed_event = torch.where(reset, minus_one, claimed_event)
    claimed_body = torch.where(reset, minus_one, claimed_body)
    release_off_count = torch.where(reset, torch.zeros_like(release_off_count), release_off_count)
    contact_rearmed = torch.where(reset, torch.ones_like(contact_rearmed), contact_rearmed)

    # A force threshold crossing is not sufficient evidence that the shoe
    # released the ball: solver force can briefly reach zero while the same
    # contact manifold remains geometrically closed.  The clean route uses a
    # short force-free interval as the complete re-arm condition.  Historical
    # V3 ablations additionally require leaving their loaded contact region.
    claimed_region_lookup = getattr(command, "s2_contact_region", None)
    contact_regions_available = bool(
        getattr(command, "_s2_contact_regions_available", False)
    )
    if callable(claimed_region_lookup) and contact_regions_available:
        claimed_points, claimed_point_valid, _timing, _source, claimed_region_valid = (
            claimed_region_lookup(claimed_event)
        )
        claimed_is_right = claimed_body == right_cfg_index
        claimed_pose_index = torch.where(
            claimed_is_right,
            torch.full_like(claimed_body, int(pose_indices[1].item())),
            torch.full_like(claimed_body, int(pose_indices[0].item())),
        )
        claimed_foot_pos = robot.data.body_pos_w[batch, claimed_pose_index]
        claimed_foot_quat = robot.data.body_quat_w[batch, claimed_pose_index]
        ball_in_claimed_foot = quat_apply_inverse(
            claimed_foot_quat, ball_pos_w - claimed_foot_pos
        )
        claimed_point_distance = torch.linalg.vector_norm(
            ball_in_claimed_foot.unsqueeze(1) - claimed_points, dim=-1
        )
        claimed_point_distance = torch.where(
            claimed_point_valid,
            claimed_point_distance,
            torch.full_like(claimed_point_distance, torch.inf),
        )
        claimed_cover_radius = max(
            0.0, float(getattr(command, "_s2_contact_region_cover_radius", 0.0))
        )
        claimed_region_distance = torch.clamp(
            torch.min(claimed_point_distance, dim=1).values - claimed_cover_radius,
            min=0.0,
        )
        left_claimed_region = claimed_region_valid & (claimed_region_distance > 0.0)
        # A premature/unlabelled physical onset has no claimed event region.
        # It still consumes the re-arm latch, and therefore needs only the
        # force-free condition before another onset can become a new episode.
        left_claimed_region = torch.where(
            claimed_episode == physical_episode,
            left_claimed_region,
            torch.ones_like(left_claimed_region),
        )
    else:
        claimed_region_distance = torch.full_like(zeros, torch.inf)
        left_claimed_region = torch.ones_like(false)
    release_off_count = torch.where(
        ~has_contact,
        release_off_count + 1,
        torch.zeros_like(release_off_count),
    )
    release_confirmed = (
        (physical_episode > 0)
        & (release_off_count >= max(1, int(contact_release_no_force_steps)))
        & left_claimed_region
    )
    contact_rearmed = contact_rearmed | release_confirmed

    previous_contact = getattr(env, "_dribbling_s2_previous_contact", None)
    if not isinstance(previous_contact, torch.Tensor) or previous_contact.shape[0] != env.num_envs:
        previous_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    previous_contact = previous_contact & ~reset
    new_physical_contact = has_contact & ~previous_contact & ~reset & contact_rearmed
    setattr(env, "_dribbling_s2_previous_contact", has_contact.detach().clone())
    physical_episode = physical_episode + new_physical_contact.to(torch.long)
    physical_start_frame = torch.where(
        new_physical_contact, frame, physical_start_frame
    )
    # Every recognized onset consumes the latch, including an onset before a
    # labelled window.  A one-frame force flicker must not turn that premature
    # contact into a fresh in-window candidate; the configured force-free
    # interval has to complete first.
    contact_rearmed = contact_rearmed & ~new_physical_contact
    # ``active`` already includes the configured annotation tolerance (for the
    # clean S2 tasks this is the labelled segment padded by +/-2 reference
    # frames).  Do not add a second hand-tuned grace threshold here.  Requiring
    # ``new_physical_contact`` is the essential causal gate: an uninterrupted
    # contact whose onset happened before this window cannot be claimed later.
    event_contact_claim = (
        active
        & new_physical_contact
        & (physical_episode != claimed_episode)
    )
    claimed_episode = torch.where(
        event_contact_claim, physical_episode, claimed_episode
    )
    claimed_event = torch.where(event_contact_claim, event_id, claimed_event)
    claimed_body = torch.where(event_contact_claim, closest_body, claimed_body)
    event_owns_physical_contact = (
        active
        & has_contact
        & (claimed_episode == physical_episode)
        & (claimed_event == event_id)
        & (claimed_body == closest_body)
    )
    # ``contact_onset_early_grace_steps`` is retained in the public parameter
    # list for config/checkpoint compatibility, but timing truth is now solely
    # the pre-expanded event table above.  Ownership itself proves that the
    # re-armed onset occurred in that window; accepting a continuously held
    # pre-window contact here would reintroduce the timing loophole.
    setattr(env, "_dribbling_s2_physical_contact_episode", physical_episode)
    setattr(env, "_dribbling_s2_physical_contact_start_frame", physical_start_frame)
    setattr(env, "_dribbling_s2_claimed_contact_episode", claimed_episode)
    setattr(env, "_dribbling_s2_claimed_contact_event", claimed_event)
    setattr(env, "_dribbling_s2_claimed_contact_body", claimed_body)
    setattr(env, "_dribbling_s2_contact_release_off_count", release_off_count)
    setattr(env, "_dribbling_s2_contact_rearmed", contact_rearmed)

    # The ball sensor exposes force and the closest body, but not a contact
    # point. Use the collision-frame physical ball center in the actual foot's
    # yaw frame as the surface-location proxy; no reference ball pose is used.
    side_known = (expected_surface == 0) | (expected_surface == 1)
    side_correct = proximity_geometry["side_correct"]
    side_wrong = proximity_geometry["side_wrong"]
    selected_foot_valid = correct_foot if require_expected_foot else legal_ankle
    reset_mode = getattr(command, "s2_episode_reset_mode", None)
    if not isinstance(reset_mode, torch.Tensor) or reset_mode.shape[0] != env.num_envs:
        reset_mode = torch.zeros(env.num_envs, dtype=torch.int8, device=env.device)
    acquisition = reset_mode == 1
    acquisition_ready = ~acquisition | (
        step >= max(1, int(acquisition_min_history_steps))
    )
    acquisition_invalid_early_contact = (
        acquisition
        & (step < max(1, int(acquisition_min_history_steps)))
        & has_contact
    )
    expected_foot_claim = (
        event_contact_claim
        & selected_foot_valid
        & acquisition_ready
    )
    expected_foot_touch = (
        event_owns_physical_contact
        & selected_foot_valid
        & acquisition_ready
    )

    # The first touch in each pair is a provisional preparation contact.  The
    # second touch is the pair's task endpoint and must occur on the calibrated
    # robot-aware contact patch before it can complete C2 or unlock a
    # transition.  This rejects heel/sole collisions that have the correct
    # lateral sign but the wrong longitudinal shoe location.  The threshold is
    # calibrated from positive/negative deterministic rollouts; it is not a
    # reference ball speed or trajectory target.
    pair_terminal_region_quality_pass = torch.ones_like(active)
    if pair_terminal_contact_region_max_distance is not None:
        qualification_region_lookup = getattr(command, "s2_contact_region", None)
        if callable(qualification_region_lookup) and contact_regions_available:
            (
                qualification_points,
                qualification_point_valid,
                _qualification_timing,
                _qualification_source,
                qualification_region_valid,
            ) = qualification_region_lookup(event_id)
            qualification_ball_local = quat_apply_inverse(
                actual_foot_quat, ball_pos_w - actual_foot_pos
            )
            qualification_point_distance = torch.linalg.vector_norm(
                qualification_ball_local.unsqueeze(1) - qualification_points,
                dim=-1,
            )
            qualification_point_distance = torch.where(
                qualification_point_valid,
                qualification_point_distance,
                torch.full_like(qualification_point_distance, torch.inf),
            )
            qualification_cover_radius = max(
                0.0,
                float(getattr(command, "_s2_contact_region_cover_radius", 0.0)),
            )
            qualification_region_distance = torch.clamp(
                torch.min(qualification_point_distance, dim=1).values
                - qualification_cover_radius,
                min=0.0,
            )
            pair_terminal = (event_id >= 0) & ((event_id % 2) == 1)
            pair_terminal_region_quality_pass = (
                ~pair_terminal
                | (
                    qualification_region_valid
                    & (
                        qualification_region_distance
                        <= float(pair_terminal_contact_region_max_distance)
                    )
                )
            )
        else:
            pair_terminal_region_quality_pass = (event_id < 0) | ((event_id % 2) == 0)

    # Release state is independent of the current reference window.  The
    # target direction is world-frame and is latched at the physical touch.
    release = getattr(env, "_dribbling_s2_release_state", None)
    if (
        not isinstance(release, dict)
        or release.get("num_envs") != env.num_envs
        or release.get("window_steps") != int(release_window_steps)
        or "next_event_end_frame" not in release
        or "next_event_window_start_frame" not in release
        or "arrival_previous_potential" not in release
    ):
        release = {
            "num_envs": env.num_envs,
            "window_steps": int(release_window_steps),
            "active": false.clone(),
            # Continues after the short velocity-quality window so a release
            # is credited only if the physical ball keeps making progress to
            # the *next* labelled anchor.
            "arrival_active": false.clone(),
            "event_id": minus_one.clone(),
            "touch_frame": minus_one.clone(),
            "next_event_frame": minus_one.clone(),
            "next_event_window_start_frame": minus_one.clone(),
            "next_event_end_frame": minus_one.clone(),
            "foot": minus_one.clone(),
            "next_anchor_w": torch.zeros(env.num_envs, 3, device=env.device),
            "direction_w": torch.zeros(env.num_envs, 2, device=env.device),
            "target_distance": zeros.clone(),
            "remaining_time": zeros.clone(),
            "nominal_speed": zeros.clone(),
            "touch_ball_position": torch.zeros(env.num_envs, 3, device=env.device),
            # Smallest distance achieved since the accepted physical touch.
            # Progress is measured against this monotone potential so moving
            # away and returning cannot earn the same distance twice.
            "arrival_best_distance": zeros.clone(),
            # Previous distance is used by the signed potential reward.  The
            # distance at contact remains in ``target_distance`` and is used
            # to decide whether the touch actually moved the ball toward the
            # following sparse anchor.
            "arrival_previous_distance": zeros.clone(),
            "arrival_previous_potential": zeros.clone(),
            "age": torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            "velocity_buffer": torch.zeros(
                env.num_envs, int(release_window_steps), 2, device=env.device
            ),
            "multiplier": torch.ones(env.num_envs, device=env.device),
        }
    for key in ("active", "arrival_active"):
        release[key] = release[key] & ~reset
    for key in (
        "event_id",
        "touch_frame",
        "next_event_frame",
        "next_event_window_start_frame",
        "next_event_end_frame",
        "foot",
    ):
        release[key] = torch.where(reset, minus_one, release[key])
    for key in (
        "target_distance",
        "remaining_time",
        "nominal_speed",
        "arrival_best_distance",
        "arrival_previous_distance",
        "arrival_previous_potential",
    ):
        release[key] = torch.where(reset, zeros, release[key])
    release["age"] = torch.where(reset, torch.zeros_like(release["age"]), release["age"])
    release["velocity_buffer"] = torch.where(
        reset[:, None, None], torch.zeros_like(release["velocity_buffer"]), release["velocity_buffer"]
    )
    release["multiplier"] = torch.where(reset, torch.ones_like(release["multiplier"]), release["multiplier"])

    quality_streak = getattr(env, "_dribbling_s2_quality_streak", None)
    if not isinstance(quality_streak, torch.Tensor) or quality_streak.shape[0] != env.num_envs:
        quality_streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    quality_streak = torch.where(reset, torch.zeros_like(quality_streak), quality_streak)

    tracked_event = getattr(env, "_dribbling_s2_tracked_event", None)
    valid_event_succeeded = getattr(env, "_dribbling_s2_valid_event_succeeded", None)
    occurrence_seen = getattr(env, "_dribbling_s2_occurrence_seen", None)
    grace_count = getattr(env, "_dribbling_s2_event_grace_count", None)
    if not isinstance(tracked_event, torch.Tensor) or tracked_event.shape[0] != env.num_envs:
        tracked_event = torch.full_like(event_id, -1)
    if not isinstance(valid_event_succeeded, torch.Tensor) or valid_event_succeeded.shape[0] != env.num_envs:
        valid_event_succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not isinstance(occurrence_seen, torch.Tensor) or occurrence_seen.shape[0] != env.num_envs:
        occurrence_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not isinstance(grace_count, torch.Tensor) or grace_count.shape[0] != env.num_envs:
        grace_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    tracked_event = torch.where(reset, torch.full_like(tracked_event, -1), tracked_event)
    valid_event_succeeded = torch.where(
        reset, torch.zeros_like(valid_event_succeeded), valid_event_succeeded
    )
    occurrence_seen = torch.where(reset, torch.zeros_like(occurrence_seen), occurrence_seen)
    grace_count = torch.where(reset, torch.zeros_like(grace_count), grace_count)

    # Strict causal sequence contract.  Only the event at this cursor can
    # produce any S2 ball reward.  A later labelled collision remains visible
    # in telemetry, but cannot bypass a missed predecessor.
    sequence_cursor = getattr(command, "s2_sequence_cursor", None)
    if (
        not isinstance(sequence_cursor, torch.Tensor)
        or sequence_cursor.shape[0] != env.num_envs
    ):
        sequence_cursor = first_event_index.clone()
    sequence_cursor = torch.where(
        reset, first_event_index, sequence_cursor
    ).to(torch.long)

    # Event ordering is grounded only in observable contact facts.  A strict
    # current-window/foot/side PhysX candidate completes the current contact
    # node; no hand-tuned displacement threshold participates in cursor
    # advancement.  Its outgoing transition is verified by the next ordered
    # physical candidate (O2 verifies O1->O2, I3 verifies O2->I3, ...).
    transitioned = (
        active
        & (tracked_event >= 0)
        & (event_id != tracked_event)
    )
    deadline = (
        ~active
        & (tracked_event >= 0)
        & (grace_count >= max(0, int(missed_contact_grace_steps)))
    )
    completed = transitioned | deadline
    completed_event = tracked_event.clone()
    completed_success = completed & valid_event_succeeded
    missed_valid_contact = completed & ~valid_event_succeeded
    advance_cursor = (
        completed_success
        & (completed_event == sequence_cursor)
    )
    sequence_cursor = torch.where(
        advance_cursor, sequence_cursor + 1, sequence_cursor
    )

    tracked_event = torch.where(completed, torch.full_like(tracked_event, -1), tracked_event)
    valid_event_succeeded = torch.where(
        completed, torch.zeros_like(valid_event_succeeded), valid_event_succeeded
    )
    occurrence_seen = torch.where(completed, torch.zeros_like(occurrence_seen), occurrence_seen)
    grace_count = torch.where(completed, torch.zeros_like(grace_count), grace_count)
    new_event = active & (tracked_event < 0)
    tracked_event = torch.where(new_event, event_id, tracked_event)
    valid_event_succeeded = torch.where(
        new_event, torch.zeros_like(valid_event_succeeded), valid_event_succeeded
    )
    occurrence_seen = torch.where(new_event, torch.zeros_like(occurrence_seen), occurrence_seen)
    grace_count = torch.where(new_event | active, torch.zeros_like(grace_count), grace_count)

    sequence_eligible = (
        (sequence_cursor >= 0)
        & (event_id == sequence_cursor)
    )
    command.s2_sequence_cursor.copy_(sequence_cursor)

    # Frozen robot-aware contact region for the current event. Points and the
    # live ball use the same full ankle-roll local frame as the CUDA scan.
    region_lookup = getattr(command, "s2_contact_region", None)
    if callable(region_lookup) and contact_regions_available:
        (
            region_points,
            region_point_valid,
            region_point_timing,
            region_source_id,
            region_valid,
        ) = region_lookup(event_id)
        region_point_distance = torch.linalg.vector_norm(
            ball_in_actual_foot.unsqueeze(1) - region_points, dim=-1
        )
        region_point_distance = torch.where(
            region_point_valid,
            region_point_distance,
            torch.full_like(region_point_distance, torch.inf),
        )
        nearest_region_slot = torch.argmin(region_point_distance, dim=1)
        batch_index = torch.arange(env.num_envs, device=env.device)
        nearest_region_raw_distance = region_point_distance[
            batch_index, nearest_region_slot
        ]
        cover_radius = max(
            0.0, float(getattr(command, "_s2_contact_region_cover_radius", 0.0))
        )
        contact_region_distance = torch.clamp(
            nearest_region_raw_distance - cover_radius, min=0.0
        )
        contact_region_distance = torch.where(
            region_valid, contact_region_distance, zeros
        )
        nearest_region_point_id = region_source_id[
            batch_index, nearest_region_slot
        ]
        nearest_region_point_timing = region_point_timing[
            batch_index, nearest_region_slot
        ]
        nearest_region_point_id = torch.where(
            region_valid, nearest_region_point_id, minus_one
        )
        nearest_region_point_timing = torch.where(
            region_valid, nearest_region_point_timing, minus_one
        )
    else:
        region_valid = false
        contact_region_distance = zeros
        nearest_region_point_id = minus_one
        nearest_region_point_timing = minus_one
    region_arrival_scale = max(float(contact_region_arrival_std), 1.0e-6)
    contact_region_arrival = torch.exp(
        -torch.square(contact_region_distance / region_arrival_scale)
    ) * (active & region_valid & sequence_eligible).to(torch.float32)

    # All dense S2 ball guidance obeys the same cursor.  Motion imitation is
    # deliberately outside this gate and continues over the full reference.
    contact_anchor_arrival = contact_anchor_arrival * sequence_eligible.to(
        contact_anchor_arrival.dtype
    )
    proximity = proximity * sequence_eligible.to(proximity.dtype)
    approach_progress_reward = approach_progress_reward * sequence_eligible.to(
        approach_progress_reward.dtype
    )

    # Isaac Lab's ball contact sensor does not expose a stable shoe mesh
    # contact point.  Therefore inside/outside is defined solely from the
    # physical ball centre in the *actual* ankle-yaw frame above: signed local
    # lateral position beyond ``side_deadzone``.  A correct-time/correct-foot
    # touch on the wrong side is an attempt, not an S2 success: it must not
    # unlock the next-anchor release reward or advance the curriculum.
    proximity_blocked = valid_event_succeeded & (tracked_event == event_id)
    proximity = proximity * (~proximity_blocked).to(proximity.dtype)
    side_valid_touch = expected_foot_touch & (
        ~side_known | side_correct
    )
    # A labelled anchor is approximate supervision, not contact ground truth.
    # Do not reject a real, correctly timed, correctly sided PhysX collision
    # with an uncalibrated centre-position tolerance.  Anchor accuracy remains
    # available as a continuous reward and diagnostic.
    if contact_anchor_quality_min is not None:
        anchor_valid_touch = (
            side_valid_touch
            & contact_anchor_valid
            & (contact_anchor_quality >= float(contact_anchor_quality_min))
        )
    elif contact_anchor_max_distance is None:
        anchor_valid_touch = side_valid_touch
    else:
        anchor_valid_touch = side_valid_touch & (
            ~contact_anchor_valid
            | (contact_anchor_distance <= float(contact_anchor_max_distance))
        )
    # Approximate contact position is only a broad sanity check in the actual
    # yaw-aligned foot frame.  It rejects obvious heel/behind-foot contacts;
    # it does not compare against a precise anchor point or a V3 point cloud.
    coarse_longitudinal_pass = torch.ones_like(active)
    if (
        coarse_contact_longitudinal_min is not None
        or coarse_contact_longitudinal_max is not None
    ):
        if coarse_contact_longitudinal_min is not None:
            coarse_longitudinal_pass = coarse_longitudinal_pass & (
                ball_from_actual_foot[:, 0]
                >= float(coarse_contact_longitudinal_min)
            )
        if coarse_contact_longitudinal_max is not None:
            coarse_longitudinal_pass = coarse_longitudinal_pass & (
                ball_from_actual_foot[:, 0]
                <= float(coarse_contact_longitudinal_max)
            )
        anchor_valid_touch = anchor_valid_touch & coarse_longitudinal_pass
    raw_immediate_contact = (
        anchor_valid_touch & ~valid_event_succeeded & ~occurrence_seen
    )
    raw_touch_occurrence = raw_immediate_contact & ~occurrence_seen
    out_of_sequence_contact = raw_touch_occurrence & ~sequence_eligible
    immediate_contact = raw_immediate_contact & sequence_eligible
    touch_occurrence = raw_touch_occurrence & sequence_eligible
    # ``immediate_contact`` is the current contact-node fact used by the
    # ordered cursor.  The delayed effective pulse has a different meaning:
    # the next ordered physical candidate verifies that the *previous*
    # transition was physically completed (O2 verifies O1->O2, I3 verifies
    # O2->I3, ...).  The first selected event therefore never receives this
    # bonus.  No displacement, speed, point-anchor, or region threshold is
    # involved.
    ordered_transition_completion = (
        immediate_contact
        & (first_event_index >= 0)
        & (event_id > first_event_index)
    )
    valid_contact_event = ordered_transition_completion
    valid_event_succeeded = valid_event_succeeded | immediate_contact
    surface_style = (
        immediate_contact & side_known & side_correct
    ).to(torch.float32)
    # Attempt diagnostics are one pulse at episode ownership.  The valid
    # contact itself may still become true on a later frame of this same
    # annotated/physical segment as the anchor geometry settles.
    wrong_side = (
        expected_foot_claim & side_known & side_wrong & sequence_eligible
    )
    dead_zone = expected_foot_claim & side_known & (
        (float(side_deadzone) > 0.0)
        & (torch.abs(current_lateral) < float(side_deadzone))
    ) & sequence_eligible
    occurrence_seen = occurrence_seen | raw_touch_occurrence

    # Weak acquisition shaping for the current cursor only.  The target is
    # deliberately coarse: bring the labelled foot toward the real ball while
    # correcting only the wrong-side component.  It does not use the manual
    # contact point, V3 geometry, ball velocity, or a reconstructed trajectory.
    side_error = proximity_geometry["side_error"]
    coarse_approach_distance = torch.sqrt(
        physical_foot_ball_distance.square() + side_error.square()
    )
    coarse_pre_window = (
        (seconds_to_event >= 0.0)
        & (seconds_to_event <= max(0.0, float(coarse_approach_duration)))
    )
    coarse_approach_active = (
        (active | coarse_pre_window)
        & sequence_eligible
        & ~occurrence_seen
        & ~valid_event_succeeded
    )
    previous_coarse_event = getattr(env, "_dribbling_s2_coarse_approach_event", None)
    previous_coarse_distance = getattr(env, "_dribbling_s2_coarse_approach_distance", None)
    previous_coarse_active = getattr(env, "_dribbling_s2_coarse_approach_active", None)
    if not isinstance(previous_coarse_event, torch.Tensor) or previous_coarse_event.shape[0] != env.num_envs:
        previous_coarse_event = minus_one.clone()
    if not isinstance(previous_coarse_distance, torch.Tensor) or previous_coarse_distance.shape[0] != env.num_envs:
        previous_coarse_distance = zeros.clone()
    if not isinstance(previous_coarse_active, torch.Tensor) or previous_coarse_active.shape[0] != env.num_envs:
        previous_coarse_active = false.clone()
    previous_coarse_event = torch.where(reset, minus_one, previous_coarse_event)
    previous_coarse_distance = torch.where(reset, zeros, previous_coarse_distance)
    previous_coarse_active = previous_coarse_active & ~reset
    same_coarse_phase = (
        coarse_approach_active
        & previous_coarse_active
        & (previous_coarse_event == event_id)
    )
    coarse_limit = max(0.0, float(coarse_approach_progress_max))
    coarse_contact_approach = torch.clamp(
        (previous_coarse_distance - coarse_approach_distance) / step_dt,
        min=-coarse_limit,
        max=coarse_limit,
    ) * same_coarse_phase.to(torch.float32)
    previous_coarse_event = torch.where(coarse_approach_active, event_id, minus_one)
    previous_coarse_distance = torch.where(
        coarse_approach_active, coarse_approach_distance, zeros
    )
    previous_coarse_active = coarse_approach_active
    setattr(env, "_dribbling_s2_coarse_approach_event", previous_coarse_event)
    setattr(env, "_dribbling_s2_coarse_approach_distance", previous_coarse_distance)
    setattr(env, "_dribbling_s2_coarse_approach_active", previous_coarse_active)

    # Wide, cursor-causal acquisition shaping toward the labelled foot-local
    # anchor.  This is deliberately separate from the narrow contact-quality
    # ellipsoid above: it must remain informative before physical contact.
    if (
        len(anchor_approach_local_scales) != 3
        or any(float(value) <= 0.0 for value in anchor_approach_local_scales)
    ):
        raise ValueError(
            "anchor_approach_local_scales must contain three positive values"
        )
    anchor_approach_scales = torch.tensor(
        anchor_approach_local_scales,
        dtype=anchor_error_local.dtype,
        device=env.device,
    )
    anchor_approach_distance = torch.sum(
        torch.square(anchor_error_local / anchor_approach_scales), dim=-1
    )
    anchor_approach_pre_window = (
        (seconds_to_event >= 0.0)
        & (seconds_to_event <= max(0.0, float(anchor_approach_duration)))
    )
    anchor_approach_active = (
        (active | anchor_approach_pre_window)
        & contact_anchor_valid
        & sequence_eligible
        & ~valid_event_succeeded
    )
    previous_anchor_event = getattr(
        env, "_dribbling_s2_anchor_approach_event", None
    )
    previous_anchor_distance = getattr(
        env, "_dribbling_s2_anchor_approach_distance", None
    )
    previous_anchor_active = getattr(
        env, "_dribbling_s2_anchor_approach_active", None
    )
    if (
        not isinstance(previous_anchor_event, torch.Tensor)
        or previous_anchor_event.shape[0] != env.num_envs
    ):
        previous_anchor_event = minus_one.clone()
    if (
        not isinstance(previous_anchor_distance, torch.Tensor)
        or previous_anchor_distance.shape[0] != env.num_envs
    ):
        previous_anchor_distance = zeros.clone()
    if (
        not isinstance(previous_anchor_active, torch.Tensor)
        or previous_anchor_active.shape[0] != env.num_envs
    ):
        previous_anchor_active = false.clone()
    previous_anchor_event = torch.where(reset, minus_one, previous_anchor_event)
    previous_anchor_distance = torch.where(reset, zeros, previous_anchor_distance)
    previous_anchor_active = previous_anchor_active & ~reset
    same_anchor_phase = (
        anchor_approach_active
        & previous_anchor_active
        & (previous_anchor_event == event_id)
    )
    anchor_approach_progress_v2 = torch.clamp(
        (previous_anchor_distance - anchor_approach_distance) / step_dt,
        min=-max(0.0, float(anchor_approach_progress_max)),
        max=max(0.0, float(anchor_approach_progress_max)),
    ) * same_anchor_phase.to(torch.float32)
    # Initializing from the current distance makes reset, inactive->active,
    # cursor changes, invalid->valid anchors, and successful contacts produce
    # no artificial transition pulse.
    previous_anchor_event = torch.where(
        anchor_approach_active, event_id, minus_one
    )
    previous_anchor_distance = torch.where(
        anchor_approach_active, anchor_approach_distance, zeros
    )
    previous_anchor_active = anchor_approach_active
    setattr(env, "_dribbling_s2_anchor_approach_event", previous_anchor_event)
    setattr(
        env, "_dribbling_s2_anchor_approach_distance", previous_anchor_distance
    )
    setattr(env, "_dribbling_s2_anchor_approach_active", previous_anchor_active)

    region_pre_window = (
        (seconds_to_event >= 0.0)
        & (seconds_to_event <= max(0.0, float(contact_region_approach_duration)))
    )
    contact_region_active = (
        (active | region_pre_window)
        & region_valid
        & sequence_eligible
        & ~valid_event_succeeded
    )
    previous_region_event = getattr(env, "_dribbling_s2_region_event", None)
    previous_region_distance = getattr(env, "_dribbling_s2_region_distance", None)
    previous_region_active = getattr(env, "_dribbling_s2_region_active", None)
    if not isinstance(previous_region_event, torch.Tensor) or previous_region_event.shape[0] != env.num_envs:
        previous_region_event = minus_one.clone()
    if not isinstance(previous_region_distance, torch.Tensor) or previous_region_distance.shape[0] != env.num_envs:
        previous_region_distance = zeros.clone()
    if not isinstance(previous_region_active, torch.Tensor) or previous_region_active.shape[0] != env.num_envs:
        previous_region_active = false.clone()
    previous_region_event = torch.where(reset, minus_one, previous_region_event)
    previous_region_distance = torch.where(reset, zeros, previous_region_distance)
    previous_region_active = previous_region_active & ~reset
    same_region_phase = (
        contact_region_active
        & previous_region_active
        & (previous_region_event == event_id)
    )
    contact_region_progress_v3 = torch.clamp(
        (previous_region_distance - contact_region_distance) / step_dt,
        min=-max(0.0, float(contact_region_progress_max)),
        max=max(0.0, float(contact_region_progress_max)),
    ) * same_region_phase.to(torch.float32)
    previous_region_event = torch.where(
        contact_region_active, event_id, minus_one
    )
    previous_region_distance = torch.where(
        contact_region_active, contact_region_distance, zeros
    )
    previous_region_active = contact_region_active
    setattr(env, "_dribbling_s2_region_event", previous_region_event)
    setattr(env, "_dribbling_s2_region_distance", previous_region_distance)
    setattr(env, "_dribbling_s2_region_active", previous_region_active)

    # Event-specific C2 telemetry. These are diagnostics only and never alter
    # contact truth or the frozen offline region.
    inf = torch.full_like(zeros, torch.inf)
    o1_min = getattr(env, "_dribbling_s2_o1_min_region_distance", None)
    o2_min = getattr(env, "_dribbling_s2_o2_min_region_distance", None)
    o1_occ = getattr(env, "_dribbling_s2_o1_region_occupancy", None)
    o1_samples = getattr(env, "_dribbling_s2_o1_region_samples", None)
    o1_quality = getattr(env, "_dribbling_s2_o1_region_quality", None)
    o1_contact = getattr(env, "_dribbling_s2_o1_contact", None)
    o2_given_o1 = getattr(env, "_dribbling_s2_o2_given_o1", None)
    telemetry_tensors = (
        o1_min, o2_min, o1_occ, o1_samples, o1_quality, o1_contact, o2_given_o1
    )
    if any(
        not isinstance(value, torch.Tensor) or value.shape[0] != env.num_envs
        for value in telemetry_tensors
    ):
        o1_min, o2_min = inf.clone(), inf.clone()
        o1_occ = zeros.clone()
        o1_samples = zeros.clone()
        o1_quality = zeros.clone()
        o1_contact = false.clone()
        o2_given_o1 = false.clone()
    o1_min = torch.where(reset, inf, o1_min)
    o2_min = torch.where(reset, inf, o2_min)
    o1_occ = torch.where(reset, zeros, o1_occ)
    o1_samples = torch.where(reset, zeros, o1_samples)
    o1_quality = torch.where(reset, zeros, o1_quality)
    o1_contact = torch.where(reset, false, o1_contact)
    o2_given_o1 = torch.where(reset, false, o2_given_o1)
    o1_sample = region_valid & (event_id == 0) & (active | region_pre_window)
    o2_sample = region_valid & (event_id == 1) & (active | region_pre_window)
    o1_min = torch.where(o1_sample, torch.minimum(o1_min, contact_region_distance), o1_min)
    o2_min = torch.where(o2_sample, torch.minimum(o2_min, contact_region_distance), o2_min)
    o1_occ = o1_occ + (o1_sample & (contact_region_distance <= 1.0e-6)).to(torch.float32)
    o1_samples = o1_samples + o1_sample.to(torch.float32)
    o1_quality = torch.where(
        o1_sample, torch.maximum(o1_quality, contact_region_arrival), o1_quality
    )
    o1_contact = o1_contact | (immediate_contact & (event_id == 0))
    o2_given_o1 = o2_given_o1 | (
        immediate_contact & (event_id == 1) & o1_contact
    )
    setattr(env, "_dribbling_s2_o1_min_region_distance", o1_min)
    setattr(env, "_dribbling_s2_o2_min_region_distance", o2_min)
    setattr(env, "_dribbling_s2_o1_region_occupancy", o1_occ)
    setattr(env, "_dribbling_s2_o1_region_samples", o1_samples)
    setattr(env, "_dribbling_s2_o1_region_quality", o1_quality)
    setattr(env, "_dribbling_s2_o1_contact", o1_contact)
    setattr(env, "_dribbling_s2_o2_given_o1", o2_given_o1)
    command.s2_sequence_contact_succeeded.copy_(
        valid_event_succeeded & (tracked_event == sequence_cursor)
    )

    relative_node_targets = bool(relative_node_targets)
    actual_ball_relative = None
    if relative_node_targets:
        actual_ball_relative = quat_apply_inverse(
            yaw_quat(command.robot_anchor_quat_w),
            ball_pos_w - command.robot_anchor_pos_w,
        )
    current_ball_target_frame = actual_ball_relative if relative_node_targets else ball_pos_w

    target_lookup = getattr(command, "s2_next_touch_target_w", None)
    if callable(target_lookup):
        next_anchor_w, next_event_frame, remaining_time, target_valid = target_lookup(
            event_id, ball_pos_w
        )
    else:
        next_anchor_w = torch.zeros(env.num_envs, 3, device=env.device)
        next_event_frame = minus_one
        remaining_time = zeros
        target_valid = false
    next_window_end_lookup = getattr(
        command, "s2_next_touch_window_end_frame", None
    )
    if callable(next_window_end_lookup):
        next_event_end_frame = next_window_end_lookup(event_id)
    else:
        next_event_end_frame = next_event_frame
    next_window_start_lookup = getattr(
        command, "s2_next_touch_window_start_frame", None
    )
    if callable(next_window_start_lookup):
        next_event_window_start_frame = next_window_start_lookup(event_id)
    else:
        next_event_window_start_frame = next_event_frame
    if relative_node_targets:
        relative_lookup = getattr(command, "s2_next_touch_node_relative_xy", None)
        if not callable(relative_lookup):
            raise RuntimeError("relative_node_targets requires the clean node compiler data")
        node_xy, node_z, node_valid = relative_lookup(event_id)
        next_anchor_w = torch.cat((node_xy, node_z.unsqueeze(-1)), dim=-1)
        target_valid = target_valid & node_valid
        target_delta = next_anchor_w[:, :2] - actual_ball_relative[:, :2]
    else:
        target_delta = next_anchor_w[:, :2] - ball_pos_w[:, :2]
    target_distance_out = torch.linalg.vector_norm(target_delta, dim=-1)
    target_direction = target_delta / target_distance_out.unsqueeze(-1).clamp(min=1.0e-6)
    nominal_speed = target_distance_out / remaining_time.clamp(min=1.0e-6)
    latch_release = (
        immediate_contact
        & target_valid
        & (next_event_frame > frame)
        & (target_distance_out >= max(0.0, float(release_min_target_distance)))
        & torch.isfinite(nominal_speed)
    )
    quality_multiplier = torch.clamp(
        1.0 + 0.5 * quality_streak.to(torch.float32), max=2.5
    )
    release["active"] = torch.where(latch_release, torch.ones_like(release["active"]), release["active"])
    release["arrival_active"] = torch.where(
        latch_release, torch.ones_like(release["arrival_active"]), release["arrival_active"]
    )
    release["event_id"] = torch.where(latch_release, event_id, release["event_id"])
    release["touch_frame"] = torch.where(latch_release, frame, release["touch_frame"])
    release["next_event_frame"] = torch.where(latch_release, next_event_frame, release["next_event_frame"])
    release["next_event_window_start_frame"] = torch.where(
        latch_release,
        next_event_window_start_frame,
        release["next_event_window_start_frame"],
    )
    release["next_event_end_frame"] = torch.where(
        latch_release, next_event_end_frame, release["next_event_end_frame"]
    )
    release["foot"] = torch.where(latch_release, expected_foot, release["foot"])
    release["next_anchor_w"] = torch.where(latch_release[:, None], next_anchor_w, release["next_anchor_w"])
    release["direction_w"] = torch.where(latch_release[:, None], target_direction, release["direction_w"])
    release["target_distance"] = torch.where(latch_release, target_distance_out, release["target_distance"])
    release["remaining_time"] = torch.where(latch_release, remaining_time, release["remaining_time"])
    release["nominal_speed"] = torch.where(latch_release, nominal_speed, release["nominal_speed"])
    release["touch_ball_position"] = torch.where(
        latch_release[:, None],
        current_ball_target_frame if relative_node_targets else ball_pos_w,
        release["touch_ball_position"],
    )
    release["arrival_best_distance"] = torch.where(
        latch_release, target_distance_out, release["arrival_best_distance"]
    )
    release["arrival_previous_distance"] = torch.where(
        latch_release, target_distance_out, release["arrival_previous_distance"]
    )
    release["arrival_previous_potential"] = torch.where(
        latch_release, target_distance_out, release["arrival_previous_potential"]
    )
    release["age"] = torch.where(latch_release, torch.zeros_like(release["age"]), release["age"])
    release["velocity_buffer"] = torch.where(
        latch_release[:, None, None], torch.zeros_like(release["velocity_buffer"]), release["velocity_buffer"]
    )
    release["multiplier"] = torch.where(latch_release, quality_multiplier, release["multiplier"])
    command.s2_locked_next_anchor_w.copy_(release["next_anchor_w"])
    command.s2_locked_next_anchor_valid.copy_(release["arrival_active"])

    release_pending = release["active"]
    # The velocity reward is intentionally short, but the ball must still
    # arrive at the next anchor.  This potential-difference term follows the
    # latched world-frame target over the whole inter-contact interval.  It is
    # reward-only: actor inputs remain the deployment-compatible 163-D state.
    arrival_distance = torch.linalg.vector_norm(
        current_ball_target_frame[:, :2] - release["next_anchor_w"][:, :2], dim=-1
    )
    route_displacement = (
        current_ball_target_frame[:, :2] - release["touch_ball_position"][:, :2]
    )
    route_along = torch.sum(
        route_displacement * release["direction_w"], dim=-1
    )
    route_lateral_vector = route_displacement - (
        route_along.unsqueeze(-1) * release["direction_w"]
    )
    route_lateral_distance = torch.linalg.vector_norm(
        route_lateral_vector, dim=-1
    )
    # Position-only potential.  The sparse annotations constrain only the
    # next contact node, not the unobserved path between contact nodes, so the
    # optimized potential is strictly Euclidean distance to the locked target.
    # Lateral deviation from the source-to-target line remains telemetry only.
    route_potential = arrival_distance
    # Position-only release V2. It asks whether the ball displacement points
    # toward the latched next anchor and actually reduces the remaining
    # distance. No measured or synthesized ball speed enters this score.
    release_displacement = ball_pos_w[:, :2] - release["touch_ball_position"][:, :2]
    release_displacement_norm = torch.linalg.vector_norm(release_displacement, dim=-1)
    release_direction_cosine = torch.sum(
        release_displacement * release["direction_w"], dim=-1
    ) / release_displacement_norm.clamp(min=1.0e-6)
    release_direction_score_v2 = torch.where(
        release_displacement_norm > 1.0e-6,
        torch.clamp(release_direction_cosine, min=0.0, max=1.0),
        torch.zeros_like(release_direction_cosine),
    )
    release_distance_progress_v2 = torch.clamp(
        (release["target_distance"] - arrival_distance)
        / release["target_distance"].clamp(min=1.0e-6),
        min=0.0,
        max=1.0,
    )
    release_v2_score = (
        release_direction_score_v2
        * release_distance_progress_v2
        * release_pending.to(torch.float32)
    )
    # The next annotated segment, not a synthetic symmetric tolerance, owns
    # the terminal arrival test.  For example O1=35--40 and O2=47--51 remain
    # separate even though their gap is short.
    transition_window_start = release["next_event_window_start_frame"]
    transition_window_end = release["next_event_end_frame"]
    arrival_active = release["arrival_active"] & (frame <= transition_window_end)
    # Sparse annotations say where the ball must be during the next contact
    # window, but do not define its path or require it to remain there after
    # first arrival. Signed shaping therefore stops when that window begins;
    # the absolute window-min tracking score below owns the terminal result.
    progress_active = arrival_active & (frame < transition_window_start)
    step_dt = max(float(getattr(env, "step_dt", 0.02)), 1.0e-6)
    progress_limit = max(0.0, float(release_arrival_progress_max))
    release_arrival_progress_raw = (
        release["arrival_previous_potential"] - route_potential
    ) / step_dt * progress_active.to(torch.float32)
    release_arrival_progress = torch.clamp(
        release_arrival_progress_raw,
        min=-progress_limit,
        max=progress_limit,
    )
    selected_contact_count_for_reward = getattr(
        command, "s2_episode_contact_count", torch.zeros_like(event_id)
    )
    # C1 is deliberately contact-only. It must prove the strict O1 entry
    # event without receiving credit for an unselected O1->O2 transition.
    release_arrival_progress = release_arrival_progress * (
        selected_contact_count_for_reward > 1
    ).to(torch.float32)
    release_arrival_progress_raw = release_arrival_progress_raw * (
        selected_contact_count_for_reward > 1
    ).to(torch.float32)
    selected_first_event = getattr(command, "s2_episode_first_event_index", minus_one)
    selected_contact_count = getattr(
        command, "s2_episode_contact_count", torch.zeros_like(minus_one)
    )
    selected_last_event = selected_first_event + selected_contact_count - 1
    terminal_release = (
        (selected_contact_count > 0)
        & (release["event_id"] == selected_last_event)
    )
    configured_terminal_progress_levels = tuple(
        int(value)
        for value in getattr(command.cfg, "dribble_cg_terminal_progress_levels", ())
    )
    if configured_terminal_progress_levels:
        episode_level = command.s2_episode_curriculum_level.to(torch.long)
        terminal_progress_enabled = torch.zeros_like(terminal_release)
        for configured_level in configured_terminal_progress_levels:
            terminal_progress_enabled |= episode_level == configured_level
        # Internal transitions remain active at every level.  Only the
        # outgoing transition of the last selected contact is staged: C2 can
        # first stabilize O1/O2 without simultaneously changing O2's release.
        release_arrival_progress = release_arrival_progress * (
            ~terminal_release | terminal_progress_enabled
        ).to(torch.float32)
        release_arrival_progress_raw = release_arrival_progress_raw * (
            ~terminal_release | terminal_progress_enabled
        ).to(torch.float32)
    terminal_progress_rate = release_arrival_progress * terminal_release.to(
        torch.float32
    )
    terminal_progress_raw_rate = (
        release_arrival_progress_raw * terminal_release.to(torch.float32)
    )
    terminal_progress_positive_rate = torch.clamp(terminal_progress_rate, min=0.0)
    terminal_progress_negative_rate = torch.clamp(terminal_progress_rate, max=0.0)
    release["arrival_best_distance"] = torch.where(
        arrival_active,
        torch.minimum(release["arrival_best_distance"], arrival_distance),
        release["arrival_best_distance"],
    )
    release["arrival_previous_distance"] = torch.where(
        arrival_active, arrival_distance, release["arrival_previous_distance"]
    )
    release["arrival_previous_potential"] = torch.where(
        arrival_active, route_potential, release["arrival_previous_potential"]
    )

    # Keep continuous, position-only outgoing progress for shaping and
    # telemetry.  It deliberately does not gate the contact cursor: a fixed
    # displacement deadband cannot establish touch quality.  The next ordered
    # physical contact is the transition outcome.
    # Evaluate only the outgoing transition of the last selected contact. The
    # following contact phase is present in the rollout but remains unselected,
    # so this is a ball-arrival result rather than an extra physical-touch task.
    terminal_transition_valid = getattr(
        env, "_dribbling_s2_terminal_transition_valid", None
    )
    terminal_transition_success = getattr(
        env, "_dribbling_s2_terminal_transition_success", None
    )
    terminal_transition_distance = getattr(
        env, "_dribbling_s2_terminal_transition_distance", None
    )
    terminal_transition_start_distance = getattr(
        env, "_dribbling_s2_terminal_transition_start_distance", None
    )
    terminal_transition_end_distance = getattr(
        env, "_dribbling_s2_terminal_transition_end_distance", None
    )
    terminal_transition_net_progress = getattr(
        env, "_dribbling_s2_terminal_transition_net_progress", None
    )
    if not isinstance(terminal_transition_valid, torch.Tensor) or terminal_transition_valid.shape[0] != env.num_envs:
        terminal_transition_valid = false.clone()
    if not isinstance(terminal_transition_success, torch.Tensor) or terminal_transition_success.shape[0] != env.num_envs:
        terminal_transition_success = false.clone()
    if not isinstance(terminal_transition_distance, torch.Tensor) or terminal_transition_distance.shape[0] != env.num_envs:
        terminal_transition_distance = zeros.clone()
    if not isinstance(terminal_transition_start_distance, torch.Tensor) or terminal_transition_start_distance.shape[0] != env.num_envs:
        terminal_transition_start_distance = zeros.clone()
    if not isinstance(terminal_transition_end_distance, torch.Tensor) or terminal_transition_end_distance.shape[0] != env.num_envs:
        terminal_transition_end_distance = zeros.clone()
    if not isinstance(terminal_transition_net_progress, torch.Tensor) or terminal_transition_net_progress.shape[0] != env.num_envs:
        terminal_transition_net_progress = zeros.clone()
    terminal_transition_valid = torch.where(
        reset, torch.zeros_like(terminal_transition_valid), terminal_transition_valid
    )
    terminal_transition_success = torch.where(
        reset, torch.zeros_like(terminal_transition_success), terminal_transition_success
    )
    terminal_transition_distance = torch.where(
        reset, torch.zeros_like(terminal_transition_distance), terminal_transition_distance
    )
    terminal_transition_start_distance = torch.where(
        reset,
        torch.zeros_like(terminal_transition_start_distance),
        terminal_transition_start_distance,
    )
    terminal_transition_end_distance = torch.where(
        reset,
        torch.zeros_like(terminal_transition_end_distance),
        terminal_transition_end_distance,
    )
    terminal_transition_net_progress = torch.where(
        reset,
        torch.zeros_like(terminal_transition_net_progress),
        terminal_transition_net_progress,
    )
    terminal_release_latched = latch_release & terminal_release
    live_target_w = getattr(command, "s2_last_live_next_anchor_w", release["next_anchor_w"])
    fixed_target_w = getattr(command, "s2_last_fixed_next_anchor_w", release["next_anchor_w"])
    if relative_node_targets:
        # In clean v0 these tensors carry the immutable reference-local node,
        # not a world-space target.  The suffix is retained for compatibility
        # with the historical diagnostic fields only.
        live_target_w = release["next_anchor_w"]
        fixed_target_w = release["next_anchor_w"]
    locked_live_target_w = getattr(env, "_dribbling_s2_locked_live_target_w", None)
    locked_fixed_target_w = getattr(env, "_dribbling_s2_locked_fixed_target_w", None)
    fixed_window_min_distance = getattr(env, "_dribbling_s2_fixed_window_min_distance", None)
    live_window_min_distance = getattr(env, "_dribbling_s2_live_window_min_distance", None)
    fixed_window_min_frame = getattr(env, "_dribbling_s2_fixed_window_min_frame", None)
    if not isinstance(locked_live_target_w, torch.Tensor) or locked_live_target_w.shape != release["next_anchor_w"].shape:
        locked_live_target_w = torch.zeros_like(release["next_anchor_w"])
        locked_fixed_target_w = torch.zeros_like(release["next_anchor_w"])
        fixed_window_min_distance = torch.full_like(zeros, torch.inf)
        live_window_min_distance = torch.full_like(zeros, torch.inf)
        fixed_window_min_frame = torch.full_like(frame, -1)
    locked_live_target_w = torch.where(reset[:, None], torch.zeros_like(locked_live_target_w), locked_live_target_w)
    locked_fixed_target_w = torch.where(reset[:, None], torch.zeros_like(locked_fixed_target_w), locked_fixed_target_w)
    fixed_window_min_distance = torch.where(reset, torch.full_like(zeros, torch.inf), fixed_window_min_distance)
    live_window_min_distance = torch.where(reset, torch.full_like(zeros, torch.inf), live_window_min_distance)
    fixed_window_min_frame = torch.where(reset, torch.full_like(frame, -1), fixed_window_min_frame)
    locked_live_target_w = torch.where(terminal_release_latched[:, None], live_target_w, locked_live_target_w)
    locked_fixed_target_w = torch.where(terminal_release_latched[:, None], fixed_target_w, locked_fixed_target_w)
    fixed_window_min_distance = torch.where(terminal_release_latched, torch.full_like(zeros, torch.inf), fixed_window_min_distance)
    live_window_min_distance = torch.where(terminal_release_latched, torch.full_like(zeros, torch.inf), live_window_min_distance)
    fixed_window_min_frame = torch.where(terminal_release_latched, torch.full_like(frame, -1), fixed_window_min_frame)
    terminal_transition_valid = torch.where(
        terminal_release_latched,
        torch.zeros_like(terminal_transition_valid),
        terminal_transition_valid,
    )
    terminal_transition_success = torch.where(
        terminal_release_latched,
        torch.zeros_like(terminal_transition_success),
        terminal_transition_success,
    )
    terminal_transition_start_distance = torch.where(
        terminal_release_latched,
        release["target_distance"],
        terminal_transition_start_distance,
    )
    terminal_transition_end_distance = torch.where(
        terminal_release_latched,
        torch.zeros_like(terminal_transition_end_distance),
        terminal_transition_end_distance,
    )
    terminal_transition_net_progress = torch.where(
        terminal_release_latched,
        torch.zeros_like(terminal_transition_net_progress),
        terminal_transition_net_progress,
    )
    terminal_transition_window = (
        release["arrival_active"]
        & terminal_release
        & (frame >= transition_window_start)
        & (frame <= transition_window_end)
    )
    fixed_distance_now = torch.linalg.vector_norm(
        current_ball_target_frame[:, :2] - locked_fixed_target_w[:, :2], dim=-1
    )
    live_distance_now = torch.linalg.vector_norm(
        current_ball_target_frame[:, :2] - locked_live_target_w[:, :2], dim=-1
    )
    fixed_new_min = terminal_transition_window & (fixed_distance_now < fixed_window_min_distance)
    fixed_window_min_distance = torch.where(
        terminal_transition_window,
        torch.minimum(fixed_window_min_distance, fixed_distance_now),
        fixed_window_min_distance,
    )
    live_window_min_distance = torch.where(
        terminal_transition_window,
        torch.minimum(live_window_min_distance, live_distance_now),
        live_window_min_distance,
    )
    fixed_window_min_frame = torch.where(fixed_new_min, frame, fixed_window_min_frame)
    target_shift_xy = locked_fixed_target_w[:, :2] - locked_live_target_w[:, :2]
    target_shift_distance = torch.linalg.vector_norm(target_shift_xy, dim=-1)
    setattr(env, "_dribbling_s2_locked_live_target_w", locked_live_target_w)
    setattr(env, "_dribbling_s2_locked_fixed_target_w", locked_fixed_target_w)
    setattr(env, "_dribbling_s2_fixed_window_min_distance", fixed_window_min_distance)
    setattr(env, "_dribbling_s2_live_window_min_distance", live_window_min_distance)
    setattr(env, "_dribbling_s2_fixed_window_min_frame", fixed_window_min_frame)
    # Position-only outcome at the time where the next sparse contact is
    # annotated.  Normalize by the distance latched at the outgoing touch so
    # this introduces neither a centimetre threshold nor a reference speed:
    #   0  = no net improvement from the outgoing contact,
    #   >0 = closer to the locked next anchor during its contact window,
    #   <0 = farther away than at the outgoing contact.
    # The inter-contact potential above supplies direction shaping; this term
    # supplies the missing time-window semantics.
    terminal_window_position_outcome = torch.clamp(
        1.0
        - arrival_distance
        / terminal_transition_start_distance.clamp(min=1.0e-6),
        min=-1.0,
        max=1.0,
    ) * terminal_transition_window.to(torch.float32)
    if configured_terminal_progress_levels:
        terminal_window_position_outcome = terminal_window_position_outcome * (
            terminal_progress_enabled
        ).to(torch.float32)
    terminal_transition_evaluated = terminal_transition_window & (
        frame >= transition_window_end
    )
    terminal_tracking_quality = zeros.clone()
    terminal_tracking_pulse = zeros.clone()

    # Event-level window-average outcome used by the matched Level-1 A/B
    # probe.  The accumulator is always executed (including in the weight-zero
    # control task) so telemetry and state transitions are identical between
    # the two arms.  It is initialized exactly once by the outgoing terminal
    # contact (O2 at Level 1), samples only the locked I3 contact window, and
    # emits at most one rate pulse.  RewardManager multiplies this rate by dt,
    # leaving the configured term weight to scale the event average directly.
    outcome_active = getattr(env, "_dribbling_s2_window_outcome_active", None)
    outcome_source_event_id = getattr(env, "_dribbling_s2_window_outcome_source_event_id", None)
    outcome_target_event_id = getattr(env, "_dribbling_s2_window_outcome_target_event_id", None)
    outcome_candidate_frame = getattr(env, "_dribbling_s2_window_outcome_candidate_frame", None)
    outcome_start_distance = getattr(env, "_dribbling_s2_window_outcome_start_distance", None)
    outcome_score_sum = getattr(env, "_dribbling_s2_window_outcome_score_sum", None)
    outcome_sample_count = getattr(env, "_dribbling_s2_window_outcome_sample_count", None)
    outcome_last_score = getattr(env, "_dribbling_s2_window_outcome_last_score", None)
    outcome_emitted = getattr(env, "_dribbling_s2_window_outcome_emitted", None)
    if not isinstance(outcome_active, torch.Tensor) or outcome_active.shape[0] != env.num_envs:
        outcome_active = false.clone()
        outcome_source_event_id = torch.full_like(frame, -1)
        outcome_target_event_id = torch.full_like(frame, -1)
        outcome_candidate_frame = torch.full_like(frame, -1)
        outcome_start_distance = zeros.clone()
        outcome_score_sum = zeros.clone()
        outcome_sample_count = zeros.clone()
        outcome_last_score = zeros.clone()
        outcome_emitted = false.clone()
    outcome_active = torch.where(reset, false, outcome_active)
    outcome_source_event_id = torch.where(reset, torch.full_like(frame, -1), outcome_source_event_id)
    outcome_target_event_id = torch.where(reset, torch.full_like(frame, -1), outcome_target_event_id)
    outcome_candidate_frame = torch.where(reset, torch.full_like(frame, -1), outcome_candidate_frame)
    outcome_start_distance = torch.where(reset, zeros, outcome_start_distance)
    outcome_score_sum = torch.where(reset, zeros, outcome_score_sum)
    outcome_sample_count = torch.where(reset, zeros, outcome_sample_count)
    outcome_last_score = torch.where(reset, zeros, outcome_last_score)
    outcome_emitted = torch.where(reset, false, outcome_emitted)

    legacy_window_outcome_enabled = bool(
        getattr(command.cfg, "dribble_cg_enable_legacy_window_outcome", False)
    )
    initialize_outcome = (
        terminal_release_latched
        & ~outcome_active
        & ~outcome_emitted
        & legacy_window_outcome_enabled
    )
    outcome_active = outcome_active | initialize_outcome
    outcome_source_event_id = torch.where(initialize_outcome, release["event_id"], outcome_source_event_id)
    outcome_target_event_id = torch.where(initialize_outcome, release["event_id"] + 1, outcome_target_event_id)
    outcome_candidate_frame = torch.where(initialize_outcome, frame, outcome_candidate_frame)
    outcome_start_distance = torch.where(initialize_outcome, release["target_distance"], outcome_start_distance)
    outcome_score_sum = torch.where(initialize_outcome, zeros, outcome_score_sum)
    outcome_sample_count = torch.where(initialize_outcome, zeros, outcome_sample_count)
    outcome_last_score = torch.where(initialize_outcome, zeros, outcome_last_score)

    outcome_sample = outcome_active & terminal_transition_window & ~outcome_emitted
    outcome_step_score = torch.clamp(
        (outcome_start_distance - arrival_distance)
        / outcome_start_distance.clamp(min=1.0e-6),
        min=-1.0,
        max=1.0,
    )
    outcome_score_sum = outcome_score_sum + torch.where(outcome_sample, outcome_step_score, zeros)
    outcome_sample_count = outcome_sample_count + outcome_sample.to(torch.float32)
    outcome_last_score = torch.where(outcome_sample, outcome_step_score, outcome_last_score)

    env_done = getattr(env, "reset_buf", false)
    if not isinstance(env_done, torch.Tensor) or env_done.shape[0] != env.num_envs:
        env_done = false
    else:
        env_done = env_done.to(torch.bool)
    settle_normal = outcome_active & terminal_transition_evaluated & ~outcome_emitted
    settle_partial = outcome_active & env_done & (outcome_sample_count > 0) & ~outcome_emitted
    settle = settle_normal | settle_partial
    outcome_score_average = torch.where(
        outcome_sample_count > 0,
        outcome_score_sum / outcome_sample_count.clamp(min=1.0),
        zeros,
    )
    outcome_pulse = torch.where(
        settle,
        outcome_score_average / max(float(getattr(env, "step_dt", 0.02)), 1.0e-6),
        zeros,
    )
    outcome_settlement_quality = torch.where(
        settle,
        torch.clamp(
            torch.minimum(outcome_score_average, outcome_last_score),
            min=0.0,
            max=1.0,
        ),
        zeros,
    )
    # Unlike the diagnostic/additive outcome rate above, settlement shares
    # the exact pulse convention of candidate and closure rewards.  Returning
    # q (not q / dt) makes a configured weight of 15 carry the same maximum
    # event budget as the existing closure weight of 15 after RewardManager's
    # common dt multiplication.
    outcome_settlement_pulse = torch.where(
        settle, outcome_settlement_quality, zeros
    )
    ball_lost_done = false
    try:
        ball_lost_done = env.termination_manager.get_term("ball_lost").to(torch.bool)
    except (KeyError, AttributeError):
        pass
    outcome_aborted = outcome_active & env_done & (outcome_sample_count <= 0) & ~outcome_emitted
    outcome_aborted_ball_lost = outcome_aborted & ball_lost_done
    outcome_aborted_other = outcome_aborted & ~ball_lost_done
    outcome_emitted = outcome_emitted | settle
    outcome_active = outcome_active & ~settle & ~outcome_aborted

    setattr(env, "_dribbling_s2_window_outcome_active", outcome_active)
    setattr(env, "_dribbling_s2_window_outcome_source_event_id", outcome_source_event_id)
    setattr(env, "_dribbling_s2_window_outcome_target_event_id", outcome_target_event_id)
    setattr(env, "_dribbling_s2_window_outcome_candidate_frame", outcome_candidate_frame)
    setattr(env, "_dribbling_s2_window_outcome_start_distance", outcome_start_distance)
    setattr(env, "_dribbling_s2_window_outcome_score_sum", outcome_score_sum)
    setattr(env, "_dribbling_s2_window_outcome_sample_count", outcome_sample_count)
    setattr(env, "_dribbling_s2_window_outcome_last_score", outcome_last_score)
    setattr(env, "_dribbling_s2_window_outcome_emitted", outcome_emitted)
    terminal_transition_mode = str(
        getattr(command.cfg, "dribble_cg_terminal_transition_mode", "arrival_radius")
    )
    if terminal_transition_mode == "positive_final_progress":
        # Evaluate the final state, not the minimum distance seen on the way.
        # This rejects the observed exploit in which O2 passes near I3 early
        # and the ball has already rolled away by the labelled I3 window.
        terminal_transition_valid = (
            terminal_transition_valid | terminal_transition_evaluated
        )
        final_positive = (
            arrival_distance < terminal_transition_start_distance
        )
        terminal_transition_success = torch.where(
            terminal_transition_evaluated,
            final_positive,
            terminal_transition_success,
        )
        terminal_transition_end_distance = torch.where(
            terminal_transition_evaluated,
            arrival_distance,
            terminal_transition_end_distance,
        )
        terminal_transition_net_progress = torch.where(
            terminal_transition_evaluated,
            terminal_transition_start_distance - arrival_distance,
            terminal_transition_net_progress,
        )
        terminal_transition_distance = terminal_transition_end_distance
    elif terminal_transition_mode == "absolute_window_min":
        # Clean sparse-node tracking: the annotation says that the ball center
        # should visit the fixed next node at some point in its labelled
        # contact window.  It does not say that the ball must remain there at
        # the window end, and it provides no reference velocity or path.
        # Therefore the result is the absolute minimum XY distance in the
        # complete window, never a relative improvement from the outgoing
        # touch.  The only scale is the physical ball radius.
        terminal_arrival_scale = max(
            float(getattr(command.cfg, "dribble_cg_ball_radius", 0.11)),
            1.0e-6,
        )
        finite_window_min = torch.isfinite(fixed_window_min_distance)
        evaluated_with_sample = terminal_transition_evaluated & finite_window_min
        terminal_tracking_quality_now = torch.exp(
            -0.5
            * torch.square(
                fixed_window_min_distance / terminal_arrival_scale
            )
        )
        terminal_tracking_quality = torch.where(
            evaluated_with_sample,
            terminal_tracking_quality_now,
            terminal_tracking_quality,
        )
        if configured_terminal_progress_levels:
            reward_evaluated = evaluated_with_sample & terminal_progress_enabled
        else:
            reward_evaluated = evaluated_with_sample
        # Event pulse convention matches physical candidate rewards.  Isaac
        # Lab applies dt uniformly afterwards, so the configured weight is the
        # event budget and this quality remains in [0, 1].
        terminal_tracking_pulse = torch.where(
            reward_evaluated,
            terminal_tracking_quality_now,
            terminal_tracking_pulse,
        )
        terminal_transition_valid = (
            terminal_transition_valid | evaluated_with_sample
        )
        terminal_transition_success = torch.where(
            evaluated_with_sample,
            fixed_window_min_distance <= terminal_arrival_scale,
            terminal_transition_success,
        )
        terminal_transition_distance = torch.where(
            evaluated_with_sample,
            fixed_window_min_distance,
            terminal_transition_distance,
        )
        terminal_transition_end_distance = torch.where(
            evaluated_with_sample,
            arrival_distance,
            terminal_transition_end_distance,
        )
        terminal_transition_net_progress = torch.where(
            evaluated_with_sample,
            terminal_transition_start_distance - fixed_window_min_distance,
            terminal_transition_net_progress,
        )
    elif terminal_transition_mode == "arrival_radius":
        terminal_arrival_scale = (
            float(contact_anchor_max_distance)
            if contact_anchor_max_distance is not None
            else float(getattr(command.cfg, "dribble_cg_ball_radius", 0.11))
        )
        terminal_transition_success = terminal_transition_success | (
            terminal_transition_window
            & (arrival_distance <= max(terminal_arrival_scale, 1.0e-6))
        )
        terminal_transition_valid = terminal_transition_valid | terminal_transition_evaluated
        terminal_transition_center = terminal_transition_window & (
            frame == release["next_event_frame"]
        )
        terminal_transition_distance = torch.where(
            terminal_transition_center,
            arrival_distance,
            terminal_transition_distance,
        )
    else:
        raise ValueError(
            "Unsupported dribble_cg_terminal_transition_mode: "
            f"{terminal_transition_mode!r}"
        )
    setattr(env, "_dribbling_s2_terminal_transition_valid", terminal_transition_valid)
    setattr(env, "_dribbling_s2_terminal_transition_success", terminal_transition_success)
    setattr(
        env,
        "_dribbling_s2_terminal_transition_distance",
        terminal_transition_distance,
    )
    setattr(
        env,
        "_dribbling_s2_terminal_transition_start_distance",
        terminal_transition_start_distance,
    )
    setattr(
        env,
        "_dribbling_s2_terminal_transition_end_distance",
        terminal_transition_end_distance,
    )
    setattr(
        env,
        "_dribbling_s2_terminal_transition_net_progress",
        terminal_transition_net_progress,
    )
    arrival_active_after_qualification = arrival_active
    release["arrival_active"] = arrival_active_after_qualification
    command.s2_locked_next_anchor_valid.copy_(arrival_active_after_qualification)
    command.s2_sequence_contact_succeeded.copy_(
        valid_event_succeeded & (tracked_event == sequence_cursor)
    )
    # Once a touch is accepted, holding the ball at the *old* anchor conflicts
    # directly with the release target at the next anchor.  The anchor term is
    # strictly pre-touch shaping; contact and release terms own the transition.
    contact_anchor_arrival = contact_anchor_arrival * (
        ~valid_event_succeeded & ~release_pending
    ).to(torch.float32)
    approach_progress_reward = approach_progress_reward * (
        ~release_pending
    ).to(torch.float32)
    proximity = proximity * (~release_pending).to(torch.float32)
    previous_approach_event = torch.where(
        release_pending, minus_one, previous_approach_event
    )
    previous_approach_potential = torch.where(
        release_pending, zeros, previous_approach_potential
    )
    setattr(env, "_dribbling_s2_approach_event", previous_approach_event)
    setattr(env, "_dribbling_s2_approach_potential", previous_approach_potential)
    velocity_xy = env.scene["soccer_ball"].data.root_lin_vel_w[:, :2]
    sample_index = release["age"].clamp(min=0, max=int(release_window_steps) - 1)
    pending_ids = torch.nonzero(release_pending, as_tuple=False).squeeze(-1)
    if pending_ids.numel() > 0:
        release["velocity_buffer"][pending_ids, sample_index[pending_ids]] = velocity_xy[pending_ids]
    sample_count = torch.where(
        release_pending,
        torch.clamp(release["age"] + 1, max=int(release_window_steps)),
        torch.zeros_like(release["age"]),
    )
    sample_slots = torch.arange(int(release_window_steps), device=env.device).unsqueeze(0)
    sample_mask = sample_slots < sample_count.unsqueeze(1)
    raw_mean = (
        release["velocity_buffer"] * sample_mask.unsqueeze(-1).to(torch.float32)
    ).sum(dim=1) / sample_count.to(torch.float32).clamp(min=1.0).unsqueeze(-1)
    residual = torch.linalg.vector_norm(
        release["velocity_buffer"] - raw_mean.unsqueeze(1), dim=-1
    )
    robust_weight = torch.clamp(
        max(float(release_velocity_outlier_scale), 1.0e-6)
        / residual.clamp(min=1.0e-6),
        max=1.0,
    ) * sample_mask.to(torch.float32)
    robust_velocity = (
        release["velocity_buffer"] * robust_weight.unsqueeze(-1)
    ).sum(dim=1) / robust_weight.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
    release_foot_index = torch.where(
        release["foot"] == 1,
        torch.full_like(release["foot"], int(pose_indices[1].item())),
        torch.full_like(release["foot"], int(pose_indices[0].item())),
    )
    release_foot_pos = robot.data.body_pos_w[batch, release_foot_index]
    separation_distance = torch.linalg.vector_norm(ball_pos_w - release_foot_pos, dim=-1)
    release_components = _dribbling_s2_release_components(
        robust_velocity,
        release["direction_w"],
        release["nominal_speed"],
        separation_distance,
        speed_lower_ratio=release_speed_lower_ratio,
        speed_upper_ratio=release_speed_upper_ratio,
        lateral_speed_std=release_lateral_speed_std,
        overspeed_decay_ratio=release_overspeed_decay_ratio,
        separation_min=release_separation_min,
        separation_full=release_separation_full,
    )
    parallel_speed = release_components["parallel_speed"]
    lateral_speed = release_components["lateral_speed"]
    speed_ratio = release_components["speed_ratio"]
    separation_score = release_components["separation_score"]
    release_base_score = release_components["base_score"]
    release_score = (
        release_base_score
        * release["multiplier"]
        * release_pending.to(torch.float32)
    )
    next_release_age = torch.where(
        release_pending, release["age"] + 1, release["age"]
    )
    release_finished = release_pending & (
        next_release_age >= int(release_window_steps)
    )
    release_quality_pass = release_finished & (
        release_base_score >= float(release_quality_threshold)
    )
    release_v2_quality_pass = release_finished & (
        release_v2_score >= float(release_v2_quality_threshold)
    )
    quality_streak = torch.where(
        release_finished,
        torch.where(
            release_quality_pass,
            quality_streak + 1,
            torch.zeros_like(quality_streak),
        ),
        quality_streak,
    )
    finished_release_event = release["event_id"].clone()
    release["age"] = next_release_age
    release["active"] = release["active"] & ~release_finished

    # A full-clip curriculum cannot provide grace frames beyond the source
    # boundary.  Settle its final active event on that boundary so the last
    # touch is not omitted from success/miss metrics.  Short curricula exclude
    # such terminal events and therefore retain their configured grace period.
    source_last_frame = command.motion.file_lengths[command.motion_idx] - 1
    terminal_completed = active & (tracked_event == event_id) & (frame >= source_last_frame)
    completed = completed | terminal_completed
    completed_success = completed_success | (terminal_completed & valid_event_succeeded)
    missed_valid_contact = missed_valid_contact | (
        terminal_completed & ~valid_event_succeeded
    )
    tracked_event = torch.where(
        terminal_completed, torch.full_like(tracked_event, -1), tracked_event
    )
    valid_event_succeeded = torch.where(
        terminal_completed,
        torch.zeros_like(valid_event_succeeded),
        valid_event_succeeded,
    )
    occurrence_seen = torch.where(
        terminal_completed, torch.zeros_like(occurrence_seen), occurrence_seen
    )
    grace_count = torch.where(
        terminal_completed, torch.zeros_like(grace_count), grace_count
    )
    grace_count = torch.where(
        ~active & (tracked_event >= 0) & ~release["active"],
        grace_count + 1,
        grace_count,
    )
    setattr(env, "_dribbling_s2_tracked_event", tracked_event)
    setattr(env, "_dribbling_s2_valid_event_succeeded", valid_event_succeeded)
    setattr(env, "_dribbling_s2_occurrence_seen", occurrence_seen)
    setattr(env, "_dribbling_s2_event_grace_count", grace_count)

    wrong_foot = (
        event_contact_claim
        & legal_ankle
        & ~correct_foot
        & require_expected_foot
        & sequence_eligible
    )
    premature_candidate = (
        ~active
        & (upcoming_event_id >= 0)
        & (frame < upcoming_event_frame)
        & new_physical_contact
        & legal_ankle
        & (upcoming_event_id == sequence_cursor)
    )
    premature_tracked_event = getattr(env, "_dribbling_s2_premature_tracked_event", None)
    premature_seen = getattr(env, "_dribbling_s2_premature_seen", None)
    if not isinstance(premature_tracked_event, torch.Tensor) or premature_tracked_event.shape[0] != env.num_envs:
        premature_tracked_event = torch.full_like(upcoming_event_id, -1)
    if not isinstance(premature_seen, torch.Tensor) or premature_seen.shape[0] != env.num_envs:
        premature_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    premature_tracked_event = torch.where(
        reset, torch.full_like(premature_tracked_event, -1), premature_tracked_event
    )
    premature_seen = torch.where(reset, torch.zeros_like(premature_seen), premature_seen)
    new_premature_target = (upcoming_event_id >= 0) & (
        premature_tracked_event != upcoming_event_id
    )
    premature_tracked_event = torch.where(
        new_premature_target, upcoming_event_id, premature_tracked_event
    )
    premature_seen = torch.where(
        new_premature_target, torch.zeros_like(premature_seen), premature_seen
    )
    premature_contact = premature_candidate & ~premature_seen
    premature_seen = premature_seen | premature_contact
    no_upcoming_event = upcoming_event_id < 0
    premature_tracked_event = torch.where(
        no_upcoming_event, torch.full_like(premature_tracked_event, -1), premature_tracked_event
    )
    premature_seen = torch.where(
        no_upcoming_event, torch.zeros_like(premature_seen), premature_seen
    )
    setattr(env, "_dribbling_s2_premature_tracked_event", premature_tracked_event)
    setattr(env, "_dribbling_s2_premature_seen", premature_seen)

    invalid_body_contact = new_physical_contact & ~legal_ankle
    nonfoot_contact_penalty = invalid_body_contact.to(torch.float32)
    wrong_foot_contact_penalty = wrong_foot.to(torch.float32)
    premature_contact_penalty = premature_contact.to(torch.float32)
    timing_error = (frame - event_frame).to(torch.float32) / max(fps, 1.0e-6)
    timing_error = torch.where(
        touch_occurrence, timing_error, torch.zeros_like(timing_error)
    )

    # Illegal contacts and a truly missed event break only the release-quality
    # streak; the separate contact streak below remains the curriculum's
    # literal sequence-completion measure.
    quality_streak = torch.where(
        invalid_body_contact | wrong_foot | missed_valid_contact,
        torch.zeros_like(quality_streak),
        quality_streak,
    )
    setattr(env, "_dribbling_s2_quality_streak", quality_streak)
    setattr(env, "_dribbling_s2_release_state", release)

    def _episode_counter(name: str, increment: torch.Tensor, dtype: torch.dtype = torch.float32):
        value = getattr(env, name, None)
        if not isinstance(value, torch.Tensor) or value.shape[0] != env.num_envs:
            value = torch.zeros(env.num_envs, dtype=dtype, device=env.device)
        value = torch.where(reset, torch.zeros_like(value), value)
        value = value + increment.to(value.dtype)
        setattr(env, name, value)
        return value

    completed_count = _episode_counter("_dribbling_s2_completed_count", completed)
    success_count = _episode_counter("_dribbling_s2_success_count", completed_success)
    missed_count = _episode_counter("_dribbling_s2_missed_count", missed_valid_contact)
    touch_attempt_count = _episode_counter(
        "_dribbling_s2_touch_attempt_count", event_contact_claim
    )
    correct_foot_count = _episode_counter(
        "_dribbling_s2_correct_foot_count", expected_foot_claim
    )
    occurrence_count = _episode_counter(
        "_dribbling_s2_occurrence_count", touch_occurrence
    )
    side_attempt_count = _episode_counter(
        "_dribbling_s2_side_attempt_count", expected_foot_claim & side_known
    )
    valid_contact_count = _episode_counter(
        "_dribbling_s2_valid_contact_count", surface_style > 0.0
    )
    dead_zone_count = _episode_counter("_dribbling_s2_dead_zone_count", dead_zone)
    wrong_foot_count = _episode_counter("_dribbling_s2_wrong_foot_count", wrong_foot)
    wrong_side_count = _episode_counter("_dribbling_s2_wrong_side_count", wrong_side)
    premature_contact_count = _episode_counter(
        "_dribbling_s2_premature_contact_count", premature_contact
    )
    invalid_body_count = _episode_counter("_dribbling_s2_invalid_body_count", invalid_body_contact)
    out_of_sequence_count = _episode_counter(
        "_dribbling_s2_out_of_sequence_contact_count",
        out_of_sequence_contact,
    )
    target_distance_sum = _episode_counter(
        "_dribbling_s2_target_distance_sum", target_distance * active.to(target_distance.dtype)
    )
    target_distance_count = _episode_counter("_dribbling_s2_target_distance_count", active)
    timing_signed_sum = _episode_counter("_dribbling_s2_timing_signed_sum", timing_error)
    timing_abs_sum = _episode_counter("_dribbling_s2_timing_abs_sum", torch.abs(timing_error))
    timing_count = _episode_counter("_dribbling_s2_timing_count", touch_occurrence)
    touch_force_sum = _episode_counter(
        "_dribbling_s2_touch_force_sum",
        force_magnitude * immediate_contact.to(force_magnitude.dtype),
    )
    touch_force_count = _episode_counter(
        "_dribbling_s2_touch_force_count", immediate_contact
    )
    touch_force_max = getattr(env, "_dribbling_s2_touch_force_max", None)
    if not isinstance(touch_force_max, torch.Tensor) or touch_force_max.shape[0] != env.num_envs:
        touch_force_max = torch.zeros(env.num_envs, dtype=force_magnitude.dtype, device=env.device)
    touch_force_max = torch.where(reset, torch.zeros_like(touch_force_max), touch_force_max)
    touch_force_max = torch.where(
        immediate_contact,
        torch.maximum(touch_force_max, force_magnitude),
        touch_force_max,
    )
    setattr(env, "_dribbling_s2_touch_force_max", touch_force_max)

    acquisition_history_at_contact = getattr(
        env, "_dribbling_s2_acquisition_history_at_contact", None
    )
    if (
        not isinstance(acquisition_history_at_contact, torch.Tensor)
        or acquisition_history_at_contact.shape[0] != env.num_envs
    ):
        acquisition_history_at_contact = zeros.clone()
    acquisition_history_at_contact = torch.where(
        reset, zeros, acquisition_history_at_contact
    )
    acquisition_history_at_contact = torch.where(
        acquisition & immediate_contact,
        step.to(torch.float32),
        acquisition_history_at_contact,
    )
    setattr(
        env,
        "_dribbling_s2_acquisition_history_at_contact",
        acquisition_history_at_contact,
    )

    streak = getattr(env, "_dribbling_s2_success_streak", None)
    max_streak = getattr(env, "_dribbling_s2_max_success_streak", None)
    if not isinstance(streak, torch.Tensor) or streak.shape[0] != env.num_envs:
        streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not isinstance(max_streak, torch.Tensor) or max_streak.shape[0] != env.num_envs:
        max_streak = torch.zeros_like(streak)
    streak = torch.where(reset, torch.zeros_like(streak), streak)
    max_streak = torch.where(reset, torch.zeros_like(max_streak), max_streak)
    streak = torch.where(completed, torch.where(completed_success, streak + 1, torch.zeros_like(streak)), streak)
    max_streak = torch.maximum(max_streak, streak)
    setattr(env, "_dribbling_s2_success_streak", streak)
    setattr(env, "_dribbling_s2_max_success_streak", max_streak)
    # One pulse per completed side-valid event.  Its value increases with the
    # consecutive sequence length, making later successful touches worth more
    # than repeatedly solving only the first contact.  Cap at eight so the
    # full-clip tail remains useful without overwhelming release quality.
    sequence_progress = completed_success.to(torch.float32) * torch.clamp(
        streak.to(torch.float32), max=8.0
    )

    first_release_quality = getattr(env, "_dribbling_s2_first_release_quality", None)
    first_release_quality_v2 = getattr(
        env, "_dribbling_s2_first_release_quality_v2", None
    )
    conditional_second_success = getattr(
        env, "_dribbling_s2_second_touch_conditional_success", None
    )
    conditional_second_success_v2 = getattr(
        env, "_dribbling_s2_second_touch_conditional_success_v2", None
    )
    if not isinstance(first_release_quality, torch.Tensor) or first_release_quality.shape[0] != env.num_envs:
        first_release_quality = false.clone()
    if not isinstance(first_release_quality_v2, torch.Tensor) or first_release_quality_v2.shape[0] != env.num_envs:
        first_release_quality_v2 = false.clone()
    if not isinstance(conditional_second_success, torch.Tensor) or conditional_second_success.shape[0] != env.num_envs:
        conditional_second_success = false.clone()
    if not isinstance(conditional_second_success_v2, torch.Tensor) or conditional_second_success_v2.shape[0] != env.num_envs:
        conditional_second_success_v2 = false.clone()
    first_release_quality = first_release_quality & ~reset
    first_release_quality_v2 = first_release_quality_v2 & ~reset
    conditional_second_success = conditional_second_success & ~reset
    conditional_second_success_v2 = conditional_second_success_v2 & ~reset
    real_sequence = ~acquisition
    first_release_quality = first_release_quality | (
        real_sequence
        & release_quality_pass
        & (finished_release_event == first_event_index)
    )
    first_release_quality_v2 = first_release_quality_v2 | (
        real_sequence
        & release_v2_quality_pass
        & (finished_release_event == first_event_index)
    )
    second_touch_pulse = (
        real_sequence
        & immediate_contact
        & (event_id == first_event_index + 1)
        & first_release_quality
    )
    conditional_second_success = conditional_second_success | second_touch_pulse
    second_touch_pulse_v2 = (
        real_sequence
        & immediate_contact
        & (event_id == first_event_index + 1)
        & first_release_quality_v2
    )
    conditional_second_success_v2 = (
        conditional_second_success_v2 | second_touch_pulse_v2
    )
    setattr(env, "_dribbling_s2_first_release_quality", first_release_quality)
    setattr(
        env, "_dribbling_s2_first_release_quality_v2", first_release_quality_v2
    )
    setattr(
        env,
        "_dribbling_s2_second_touch_conditional_success",
        conditional_second_success,
    )
    setattr(
        env,
        "_dribbling_s2_second_touch_conditional_success_v2",
        conditional_second_success_v2,
    )

    def _rate(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        return torch.where(denominator > 0, numerator / denominator.clamp(min=1.0), torch.zeros_like(numerator))

    command.metrics["s2_contact_success_rate"] = _rate(success_count, completed_count)
    command.metrics["s2_missed_contact_rate"] = _rate(missed_count, completed_count)
    command.metrics["s2_correct_foot_rate"] = _rate(correct_foot_count, touch_attempt_count)
    command.metrics["s2_touch_occurrence_rate"] = _rate(occurrence_count, completed_count)
    command.metrics["s2_correct_side_rate"] = _rate(valid_contact_count, side_attempt_count)
    command.metrics["s2_valid_contact_rate"] = _rate(success_count, completed_count)
    command.metrics["s2_contact_timing_error"] = _rate(timing_signed_sum, timing_count)
    command.metrics["s2_contact_timing_abs_error"] = _rate(timing_abs_sum, timing_count)
    command.metrics["s2_wrong_foot_count"] = wrong_foot_count
    command.metrics["s2_wrong_side_count"] = wrong_side_count
    command.metrics["s2_premature_contact_count"] = premature_contact_count
    command.metrics["s2_invalid_body_contact_count"] = invalid_body_count
    command.metrics["s2_out_of_sequence_contact_count"] = out_of_sequence_count
    command.metrics["s2_contact_rearmed"] = contact_rearmed.to(torch.float32)
    command.metrics["s2_contact_release_off_count"] = release_off_count.to(torch.float32)
    command.metrics["s2_contact_release_confirmed"] = release_confirmed.to(torch.float32)
    command.metrics["s2_sequence_cursor"] = sequence_cursor.to(torch.float32)
    command.metrics["s2_ordered_transition_completion"] = (
        ordered_transition_completion.to(torch.float32)
    )
    command.metrics["s2_target_region_distance"] = _rate(
        target_distance_sum, target_distance_count
    )
    command.metrics["s2_next_anchor_lateral_distance"] = torch.where(
        arrival_active, route_lateral_distance, torch.zeros_like(route_lateral_distance)
    )
    command.metrics["s2_touch_force_mean"] = _rate(touch_force_sum, touch_force_count)
    command.metrics["s2_touch_force_max"] = touch_force_max
    command.metrics["s2_dead_zone_count"] = dead_zone_count
    command.metrics["s2_complete_2"] = (max_streak >= 2).to(torch.float32)
    command.metrics["s2_complete_3"] = (max_streak >= 3).to(torch.float32)
    command.metrics["s2_complete_4"] = (max_streak >= 4).to(torch.float32)
    command.metrics["s2_complete_5"] = (max_streak >= 5).to(torch.float32)
    command.metrics["s2_complete_6"] = (max_streak >= 6).to(torch.float32)
    command.metrics["s2_complete_8"] = (max_streak >= 8).to(torch.float32)
    selected_count = getattr(command, "s2_episode_contact_count", None)
    if isinstance(selected_count, torch.Tensor):
        command.metrics["s2_complete_selected"] = (
            (selected_count > 0) & (max_streak >= selected_count)
        ).to(torch.float32)
    command.metrics["s2_approach_progress"] = approach_progress_reward
    command.metrics["s2_coarse_contact_approach"] = coarse_contact_approach
    command.metrics["s2_anchor_approach_progress_v2"] = anchor_approach_progress_v2
    if contact_regions_available:
        command.metrics["s2_contact_region_progress_v3"] = contact_region_progress_v3
        command.metrics["s2_contact_region_arrival"] = contact_region_arrival
        command.metrics["s2_o1_min_region_distance"] = torch.where(
            torch.isfinite(o1_min), o1_min, zeros
        )
        command.metrics["s2_o1_region_occupancy_rate"] = o1_occ / o1_samples.clamp(min=1.0)
        command.metrics["s2_o1_region_arrival_quality"] = o1_quality
        command.metrics["s2_o2_min_region_distance"] = torch.where(
            torch.isfinite(o2_min), o2_min, zeros
        )
        command.metrics["s2_nearest_region_point_id"] = nearest_region_point_id.to(torch.float32)
        command.metrics["s2_nearest_region_point_timing"] = nearest_region_point_timing.to(torch.float32)
    else:
        for metric_name in (
            "s2_contact_region_progress_v3",
            "s2_contact_region_arrival",
            "s2_o1_min_region_distance",
            "s2_o1_region_occupancy_rate",
            "s2_o1_region_arrival_quality",
            "s2_o2_min_region_distance",
            "s2_nearest_region_point_id",
            "s2_nearest_region_point_timing",
        ):
            command.metrics.pop(metric_name, None)
    command.metrics["s2_o1_filtered_contact_rate"] = o1_contact.to(torch.float32)
    command.metrics["s2_o2_given_o1_contact_rate"] = o2_given_o1.to(torch.float32)
    command.metrics["s2_immediate_contact"] = immediate_contact.to(torch.float32)
    command.metrics["s2_release_pending"] = release_pending.to(torch.float32)
    command.metrics["s2_release_score"] = release_score
    command.metrics["s2_release_v2_score"] = release_v2_score
    command.metrics["s2_next_anchor_progress"] = release_arrival_progress
    command.metrics["s2_terminal_progress_raw_rate"] = terminal_progress_raw_rate
    command.metrics["s2_terminal_progress_clipped_rate"] = terminal_progress_rate
    command.metrics["s2_terminal_progress_positive_rate"] = (
        terminal_progress_positive_rate
    )
    command.metrics["s2_terminal_progress_negative_rate"] = (
        terminal_progress_negative_rate
    )
    command.metrics["s2_contact_anchor_quality"] = contact_anchor_quality
    command.metrics["s2_terminal_transition_valid"] = terminal_transition_valid.to(torch.float32)
    command.metrics["s2_terminal_transition_success"] = terminal_transition_success.to(torch.float32)
    command.metrics["s2_terminal_transition_distance"] = terminal_transition_distance
    command.metrics["s2_terminal_transition_start_distance"] = (
        terminal_transition_start_distance
    )
    command.metrics["s2_terminal_transition_end_distance"] = (
        terminal_transition_end_distance
    )
    command.metrics["s2_terminal_transition_net_progress"] = (
        terminal_transition_net_progress
    )
    command.metrics["s2_terminal_tracking_quality"] = terminal_tracking_quality
    command.metrics["s2_terminal_tracking_pulse"] = terminal_tracking_pulse
    command.metrics["s2_terminal_window_position_outcome"] = (
        terminal_window_position_outcome
    )
    command.metrics["s2_window_outcome_score_average"] = outcome_score_average
    command.metrics["s2_window_outcome_score_end"] = outcome_last_score
    command.metrics["s2_window_outcome_settlement_quality"] = outcome_settlement_quality
    command.metrics["s2_window_outcome_settlement_pulse"] = outcome_settlement_pulse
    command.metrics["s2_window_outcome_pulse"] = outcome_pulse
    command.metrics["s2_window_outcome_sample_count"] = outcome_sample_count
    command.metrics["s2_window_outcome_aborted_ball_lost"] = outcome_aborted_ball_lost.to(torch.float32)
    command.metrics["s2_window_outcome_aborted_other"] = outcome_aborted_other.to(torch.float32)
    command.metrics["s2_fixed_target_shift_distance"] = target_shift_distance
    command.metrics["s2_fixed_window_min_distance"] = torch.where(
        torch.isfinite(fixed_window_min_distance), fixed_window_min_distance, zeros
    )
    command.metrics["s2_live_window_min_distance"] = torch.where(
        torch.isfinite(live_window_min_distance), live_window_min_distance, zeros
    )
    command.metrics["s2_release_quality_pass"] = release_quality_pass.to(torch.float32)
    command.metrics["s2_release_v2_quality_pass"] = release_v2_quality_pass.to(torch.float32)
    command.metrics["s2_first_release_quality"] = first_release_quality.to(torch.float32)
    command.metrics["s2_first_release_quality_v2"] = first_release_quality_v2.to(torch.float32)
    command.metrics["s2_quality_streak"] = quality_streak.to(torch.float32)
    command.metrics["s2_second_touch_conditional_success"] = conditional_second_success.to(torch.float32)
    command.metrics["s2_second_touch_conditional_success_v2"] = conditional_second_success_v2.to(torch.float32)
    command.metrics["s2_reset_mode"] = reset_mode.to(torch.float32)
    command.metrics["s2_acquisition_success_rate"] = torch.where(
        acquisition, (max_streak >= 1).to(torch.float32), zeros
    )
    command.metrics["s2_acquisition_history_steps_before_contact"] = acquisition_history_at_contact

    state = {
        # Snapshot the decision frame. ``command.time_steps`` is updated
        # in-place after manager evaluation; retaining its tensor view would
        # make playback report the following frame instead of the frame that
        # actually produced this contact decision.
        "evaluation_frame": frame.detach().clone(),
        "contact_window": active,
        "approach_progress": approach_progress_reward,
        "coarse_contact_approach": coarse_contact_approach,
        "anchor_approach_progress_v2": anchor_approach_progress_v2,
        "contact_region_progress_v3": contact_region_progress_v3,
        "contact_region_arrival": contact_region_arrival,
        "contact_region_distance": contact_region_distance,
        "contact_region_valid": region_valid,
        "nearest_region_point_id": nearest_region_point_id,
        "nearest_region_point_timing": nearest_region_point_timing,
        "ball_in_actual_foot_full": ball_in_actual_foot,
        "contact_proximity": proximity,
        "target_region_distance": target_distance,
        "contact_anchor_distance": contact_anchor_distance,
        "contact_anchor_quality": contact_anchor_quality,
        "contact_anchor_arrival": contact_anchor_arrival,
        "physical_foot_ball_distance": physical_foot_ball_distance,
        "touch_occurrence": touch_occurrence,
        "immediate_contact": immediate_contact,
        "surface_style": surface_style,
        "valid_contact_event": valid_contact_event,
        "pair_terminal_region_quality_pass": pair_terminal_region_quality_pass,
        # Contact/release-gate telemetry.  These are diagnostics only: they
        # expose why a physical onset was (or was not) allowed to become the
        # next labelled touch without changing the contact decision itself.
        "contact_has_force": has_contact,
        "contact_previous_has_force": previous_contact,
        "contact_new_physical_onset": new_physical_contact,
        "contact_event_claim": event_contact_claim,
        "contact_event_owns_physical": event_owns_physical_contact,
        "contact_closest_body": closest_body,
        "contact_correct_foot": correct_foot,
        "contact_side_correct": side_correct,
        "contact_coarse_longitudinal_pass": coarse_longitudinal_pass,
        "contact_anchor_valid_touch": anchor_valid_touch,
        "contact_rearmed": contact_rearmed,
        "contact_release_off_count": release_off_count,
        "contact_release_confirmed": release_confirmed,
        "contact_left_claimed_region": left_claimed_region,
        "contact_claimed_region_distance": claimed_region_distance,
        "contact_physical_episode": physical_episode,
        "contact_claimed_episode": claimed_episode,
        "contact_claimed_event": claimed_event,
        "contact_force_magnitude": force_magnitude,
        "dead_zone": dead_zone,
        "wrong_foot": wrong_foot,
        "wrong_side": wrong_side,
        "premature_contact": premature_contact,
        "invalid_body_contact": invalid_body_contact,
        "out_of_sequence_contact": out_of_sequence_contact,
        "sequence_eligible": sequence_eligible,
        "sequence_cursor": sequence_cursor,
        "nonfoot_contact_penalty": nonfoot_contact_penalty,
        "wrong_foot_contact_penalty": wrong_foot_contact_penalty,
        "premature_contact_penalty": premature_contact_penalty,
        "missed_valid_contact": missed_valid_contact,
        "release_pending": release_pending,
        "release_arrival_progress": release_arrival_progress,
        "terminal_release": terminal_release,
        "terminal_transition_window": terminal_transition_window,
        "terminal_progress_raw_rate": terminal_progress_raw_rate,
        "terminal_progress_clipped_rate": terminal_progress_rate,
        "terminal_progress_positive_rate": terminal_progress_positive_rate,
        "terminal_progress_negative_rate": terminal_progress_negative_rate,
        "terminal_transition_valid": terminal_transition_valid,
        "terminal_transition_success": terminal_transition_success,
        "terminal_transition_distance": terminal_transition_distance,
        "terminal_transition_start_distance": terminal_transition_start_distance,
        "terminal_transition_end_distance": terminal_transition_end_distance,
        "terminal_transition_net_progress": terminal_transition_net_progress,
        "terminal_tracking_quality": terminal_tracking_quality,
        "terminal_tracking_pulse": terminal_tracking_pulse,
        "terminal_window_position_outcome": terminal_window_position_outcome,
        "window_outcome_active": outcome_active,
        "window_outcome_source_event_id": outcome_source_event_id,
        "window_outcome_target_event_id": outcome_target_event_id,
        "window_outcome_candidate_frame": outcome_candidate_frame,
        "window_outcome_start_distance": outcome_start_distance,
        "window_outcome_score_sum": outcome_score_sum,
        "window_outcome_sample_count": outcome_sample_count,
        "window_outcome_score_average": outcome_score_average,
        "window_outcome_score_end": outcome_last_score,
        "window_outcome_settlement_quality": outcome_settlement_quality,
        "window_outcome_settlement_pulse": outcome_settlement_pulse,
        "window_outcome_pulse": outcome_pulse,
        "window_outcome_emitted": settle,
        "window_outcome_aborted_ball_lost": outcome_aborted_ball_lost,
        "window_outcome_aborted_other": outcome_aborted_other,
        "locked_live_target_w": locked_live_target_w,
        "locked_fixed_target_w": locked_fixed_target_w,
        "fixed_target_shift_xy": target_shift_xy,
        "fixed_target_shift_distance": target_shift_distance,
        "fixed_window_distance": fixed_distance_now,
        "live_window_distance": live_distance_now,
        "fixed_window_min_distance": fixed_window_min_distance,
        "live_window_min_distance": live_window_min_distance,
        "fixed_window_min_frame": fixed_window_min_frame,
        "release_actual_touch_frame": release["touch_frame"],
        "release_next_anchor_w": release["next_anchor_w"],
        "release_remaining_time": release["remaining_time"],
        "release_target_distance": release["target_distance"],
        "release_target_direction": release["direction_w"],
        "release_nominal_speed": release["nominal_speed"],
        "release_parallel_speed": parallel_speed,
        "release_lateral_speed": lateral_speed,
        "release_speed_ratio": speed_ratio,
        "release_separation_score": separation_score,
        "release_score": release_score,
        "release_v2_score": release_v2_score,
        "release_quality_pass": release_quality_pass,
        "release_v2_quality_pass": release_v2_quality_pass,
        "quality_streak": quality_streak,
        "sequence_progress": sequence_progress,
        "second_touch_conditional_success": conditional_second_success,
        "second_touch_conditional_success_v2": conditional_second_success_v2,
        "acquisition_invalid_early_contact": acquisition_invalid_early_contact,
        "acquisition_history_steps_before_contact": acquisition_history_at_contact,
        "timing_error_seconds": timing_error,
        "event_id": event_id,
        "event_frame": event_frame,
        "expected_foot": expected_foot,
        "expected_surface": expected_surface,
        # Retain the historical local-side value for diagnostics only.
        "expected_side": torch.where(
            ((expected_foot == 1) & (expected_surface == 0))
            | ((expected_foot == 0) & (expected_surface == 1)),
            torch.zeros_like(expected_surface),
            torch.ones_like(expected_surface),
        ),
        "ball_from_actual_foot": ball_from_actual_foot,
    }
    setattr(env, "_dribbling_s2_contact_event_cache", {
        "step": step.detach().clone(),
        "frame": frame.detach().clone(),
        "motion_idx": motion_idx.detach().clone(),
        "generation": generation.detach().clone(),
        "state": state,
    })
    setattr(
        env,
        "_dribbling_s2_release_telemetry",
        {
            key: state[key]
            for key in (
                "evaluation_frame",
                "approach_progress",
                "immediate_contact",
                "valid_contact_event",
                "surface_style",
                "contact_has_force",
                "contact_previous_has_force",
                "contact_new_physical_onset",
                "contact_event_claim",
                "contact_event_owns_physical",
                "contact_closest_body",
                "contact_correct_foot",
                "contact_side_correct",
                "contact_coarse_longitudinal_pass",
                "contact_anchor_valid_touch",
                "contact_rearmed",
                "contact_release_off_count",
                "contact_release_confirmed",
                "contact_left_claimed_region",
                "contact_claimed_region_distance",
                "contact_physical_episode",
                "contact_claimed_episode",
                "contact_claimed_event",
                "contact_region_distance",
                "contact_region_valid",
                "pair_terminal_region_quality_pass",
                "nearest_region_point_id",
                "nearest_region_point_timing",
                "ball_in_actual_foot_full",
                "release_pending",
                "release_actual_touch_frame",
                "release_next_anchor_w",
                "release_remaining_time",
                "release_target_distance",
                "release_target_direction",
                "release_nominal_speed",
                "release_parallel_speed",
                "release_lateral_speed",
                "release_speed_ratio",
                "release_separation_score",
                "release_score",
                "release_arrival_progress",
                "terminal_release",
                "terminal_transition_window",
                "terminal_progress_raw_rate",
                "terminal_progress_clipped_rate",
                "terminal_progress_positive_rate",
                "terminal_progress_negative_rate",
                "terminal_transition_valid",
                "terminal_transition_success",
                "terminal_transition_start_distance",
                "terminal_transition_end_distance",
                "terminal_transition_net_progress",
                "terminal_tracking_quality",
                "terminal_tracking_pulse",
                "terminal_window_position_outcome",
                "window_outcome_active",
                "window_outcome_source_event_id",
                "window_outcome_target_event_id",
                "window_outcome_candidate_frame",
                "window_outcome_start_distance",
                "window_outcome_score_sum",
                "window_outcome_sample_count",
                "window_outcome_score_average",
                "window_outcome_score_end",
                "window_outcome_settlement_quality",
                "window_outcome_settlement_pulse",
                "window_outcome_pulse",
                "window_outcome_emitted",
                "window_outcome_aborted_ball_lost",
                "window_outcome_aborted_other",
                "locked_live_target_w",
                "locked_fixed_target_w",
                "fixed_target_shift_xy",
                "fixed_target_shift_distance",
                "fixed_window_distance",
                "live_window_distance",
                "fixed_window_min_distance",
                "live_window_min_distance",
                "fixed_window_min_frame",
                "release_quality_pass",
                "quality_streak",
                "second_touch_conditional_success",
                "acquisition_history_steps_before_contact",
            )
        },
    )
    return state


def dribbling_s2_contact_proximity(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
    require_expected_foot: bool = True,
    target_side_enabled: bool = True,
    side_deadzone: float = 0.04,
    proximity_contact_distance_max: float = 0.25,
    target_region_std: float = 0.12,
    proximity_approach_seconds: float = 0.30,
    proximity_approach_min_weight: float = 0.20,
    missed_contact_grace_steps: int = 3,
) -> torch.Tensor:
    """Windowed physical reach and labelled-side proximity signal."""
    return dribbling_s2_contact_event_state(
        env, command_name, ball_sensor_name, all_body_cfg, num_ankle_links,
        require_expected_foot, target_side_enabled,
        side_deadzone, proximity_contact_distance_max, target_region_std, proximity_approach_seconds,
        proximity_approach_min_weight, missed_contact_grace_steps,
    )["contact_proximity"]


def dribbling_s2_approach_progress(
    env: ManagerBasedRLEnv,
    event_params: dict,
) -> torch.Tensor:
    """Potential-difference approach shaping outside the final 0.30 seconds."""
    return dribbling_s2_contact_event_state(env, **event_params)[
        "approach_progress"
    ]


def dribbling_s2_coarse_contact_approach(
    env: ManagerBasedRLEnv,
    event_params: dict,
) -> torch.Tensor:
    """Signed current-cursor foot progress toward the real ball and correct side."""
    return dribbling_s2_contact_event_state(env, **event_params)[
        "coarse_contact_approach"
    ]


def dribbling_s2_anchor_approach_progress_v2(
    env: ManagerBasedRLEnv,
    event_params: dict,
) -> torch.Tensor:
    """Signed progress toward the current cursor's foot-local contact anchor."""
    return dribbling_s2_contact_event_state(env, **event_params)[
        "anchor_approach_progress_v2"
    ]


def dribbling_s2_contact_region_progress_v3(
    env: ManagerBasedRLEnv,
    event_params: dict,
) -> torch.Tensor:
    """Signed distance progress toward the frozen feasible contact region."""
    return dribbling_s2_contact_event_state(env, **event_params)[
        "contact_region_progress_v3"
    ]


def dribbling_s2_contact_region_arrival(
    env: ManagerBasedRLEnv,
    event_params: dict,
) -> torch.Tensor:
    """Continuous in-window quality for the frozen feasible contact region."""
    return dribbling_s2_contact_event_state(env, **event_params)[
        "contact_region_arrival"
    ]


def _dribbling_s2_transition_region_state(
    env: ManagerBasedRLEnv,
    command_name: str,
    event_params: dict,
    progress_max: float,
    arrival_std: float,
) -> dict[str, torch.Tensor]:
    """Latch and evaluate the V3 region after the last selected C2 touch."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    contact = dribbling_s2_contact_event_state(env, **event_params)
    frame = command.time_steps.to(torch.long)
    generation = command.s2_episode_generation.to(torch.long)
    cache = getattr(env, "_dribbling_s2_t2_cache", None)
    if (
        isinstance(cache, dict)
        and torch.equal(cache.get("frame", torch.empty(0, device=env.device)), frame)
        and torch.equal(cache.get("generation", torch.empty(0, device=env.device)), generation)
    ):
        return cache["state"]

    zeros = torch.zeros(env.num_envs, device=env.device)
    false = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    previous_generation = getattr(env, "_dribbling_s2_t2_generation", None)
    if not isinstance(previous_generation, torch.Tensor):
        previous_generation = generation.clone()
    reset = generation != previous_generation
    setattr(env, "_dribbling_s2_t2_generation", generation.clone())

    selected_last = (
        command.s2_episode_first_event_index
        + command.s2_episode_contact_count
        - 1
    )
    t2_enabled = command.s2_episode_curriculum_level.to(torch.long) >= 1
    outgoing_touch = (
        contact["immediate_contact"].to(torch.bool)
        & (contact["event_id"].to(torch.long) == selected_last)
        & (command.s2_episode_contact_count == 2)
        & t2_enabled
    )
    points_w, point_valid, next_frame, next_end = (
        command.s2_next_contact_region_target_w(selected_last)
    )
    point_count = points_w.shape[1]

    locked_points = getattr(env, "_dribbling_s2_t2_points_w", None)
    locked_mask = getattr(env, "_dribbling_s2_t2_point_valid", None)
    locked_active = getattr(env, "_dribbling_s2_t2_active", None)
    locked_frame = getattr(env, "_dribbling_s2_t2_frame", None)
    locked_end = getattr(env, "_dribbling_s2_t2_end", None)
    previous_distance = getattr(env, "_dribbling_s2_t2_previous_distance", None)
    previous_active = getattr(env, "_dribbling_s2_t2_previous_active", None)
    success = getattr(env, "_dribbling_s2_t2_success", None)
    evaluated = getattr(env, "_dribbling_s2_t2_evaluated", None)
    if not isinstance(locked_points, torch.Tensor) or locked_points.shape != points_w.shape:
        locked_points = torch.zeros_like(points_w)
        locked_mask = torch.zeros_like(point_valid)
        locked_active = false.clone()
        locked_frame = torch.full_like(frame, -1)
        locked_end = torch.full_like(frame, -1)
        previous_distance = zeros.clone()
        previous_active = false.clone()
        success = false.clone()
        evaluated = false.clone()

    if bool(torch.any(reset)):
        locked_points[reset] = 0.0
        locked_mask[reset] = False
        locked_active[reset] = False
        locked_frame[reset] = -1
        locked_end[reset] = -1
        previous_distance[reset] = 0.0
        previous_active[reset] = False
        success[reset] = False
        evaluated[reset] = False

    latch = outgoing_touch & point_valid.any(dim=1) & (next_end >= next_frame)
    locked_points = torch.where(latch[:, None, None], points_w, locked_points)
    locked_mask = torch.where(latch[:, None], point_valid, locked_mask)
    locked_active = torch.where(latch, torch.ones_like(locked_active), locked_active)
    locked_frame = torch.where(latch, next_frame, locked_frame)
    locked_end = torch.where(latch, next_end, locked_end)
    success = torch.where(latch, torch.zeros_like(success), success)
    evaluated = torch.where(latch, torch.zeros_like(evaluated), evaluated)

    ball_w = env.scene["soccer_ball"].data.root_pos_w[:, :3]
    point_distance = torch.linalg.vector_norm(
        ball_w.unsqueeze(1) - locked_points, dim=-1
    )
    point_distance = torch.where(
        locked_mask, point_distance, torch.full_like(point_distance, torch.inf)
    )
    cover_radius = max(
        0.0, float(getattr(command, "_s2_contact_region_cover_radius", 0.0))
    )
    distance = torch.clamp(torch.min(point_distance, dim=1).values - cover_radius, min=0.0)
    distance = torch.where(locked_active, distance, zeros)
    transition_active = locked_active & (frame <= locked_end)
    same_phase = transition_active & previous_active & ~latch
    step_dt = max(float(getattr(env, "step_dt", 0.02)), 1.0e-6)
    progress = torch.clamp(
        (previous_distance - distance) / step_dt,
        min=-max(0.0, float(progress_max)),
        max=max(0.0, float(progress_max)),
    ) * same_phase.to(torch.float32)
    in_window = transition_active & (frame >= locked_frame) & (frame <= locked_end)
    scale = max(float(arrival_std), 1.0e-6)
    arrival = torch.exp(-torch.square(distance / scale)) * in_window.to(torch.float32)
    success = success | (in_window & (distance <= 1.0e-6))
    evaluated = evaluated | (locked_active & (frame >= locked_end))

    previous_distance = torch.where(transition_active, distance, zeros)
    previous_active = transition_active
    locked_active = locked_active & (frame <= locked_end)
    for name, value in (
        ("_dribbling_s2_t2_points_w", locked_points),
        ("_dribbling_s2_t2_point_valid", locked_mask),
        ("_dribbling_s2_t2_active", locked_active),
        ("_dribbling_s2_t2_frame", locked_frame),
        ("_dribbling_s2_t2_end", locked_end),
        ("_dribbling_s2_t2_previous_distance", previous_distance),
        ("_dribbling_s2_t2_previous_active", previous_active),
        ("_dribbling_s2_t2_success", success),
        ("_dribbling_s2_t2_evaluated", evaluated),
    ):
        setattr(env, name, value)

    # The critic retains its fixed 314-D interface: its existing next-anchor
    # slot now contains the centroid of exactly the same locked V3 region.
    centroid = (locked_points * locked_mask.unsqueeze(-1)).sum(dim=1) / locked_mask.sum(
        dim=1, keepdim=True
    ).clamp(min=1)
    command.s2_locked_next_anchor_w.copy_(centroid)
    command.s2_locked_next_anchor_valid.copy_(transition_active)
    command.metrics["s2_t2_region_progress"] = progress
    command.metrics["s2_t2_region_arrival"] = arrival
    command.metrics["s2_t2_region_distance"] = distance
    command.metrics["s2_t2_success"] = success.to(torch.float32)
    command.metrics["s2_terminal_transition_valid"] = evaluated.to(torch.float32)
    command.metrics["s2_terminal_transition_success"] = success.to(torch.float32)
    command.metrics["s2_terminal_transition_distance"] = distance
    state = {
        "progress": progress,
        "arrival": arrival,
        "distance": distance,
        "success": success,
        "evaluated": evaluated,
    }
    setattr(env, "_dribbling_s2_t2_cache", {
        "frame": frame.clone(), "generation": generation.clone(), "state": state
    })
    return state


def dribbling_s2_transition_region_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    event_params: dict,
    progress_max: float = 4.0,
    arrival_std: float = 0.02,
) -> torch.Tensor:
    return _dribbling_s2_transition_region_state(
        env, command_name, event_params, progress_max, arrival_std
    )["progress"]


def dribbling_s2_transition_region_arrival(
    env: ManagerBasedRLEnv,
    command_name: str,
    event_params: dict,
    progress_max: float = 4.0,
    arrival_std: float = 0.02,
) -> torch.Tensor:
    return _dribbling_s2_transition_region_state(
        env, command_name, event_params, progress_max, arrival_std
    )["arrival"]


def dribbling_s2_contact_anchor_arrival(
    env: ManagerBasedRLEnv,
    event_params: dict,
) -> torch.Tensor:
    """Reward physical ball proximity to the current labelled contact anchor."""
    return dribbling_s2_contact_event_state(env, **event_params)[
        "contact_anchor_arrival"
    ]


def dribbling_s2_effective_touch_bonus(
    env: ManagerBasedRLEnv,
    event_params: dict,
) -> torch.Tensor:
    """Reward completion of the preceding contact-to-contact transition.

    A pulse occurs only when the next strict ordered physical candidate is
    observed: O2 verifies O1->O2, I3 verifies O2->I3, and so on.  This reward
    does not use a displacement deadband and does not control cursor advance.
    """
    return dribbling_s2_contact_event_state(env, **event_params)[
        "valid_contact_event"
    ].to(torch.float32)


def dribbling_s2_physical_contact_candidate_bonus(
    env: ManagerBasedRLEnv,
    event_params: dict,
) -> torch.Tensor:
    """Reward one real current-window foot-ball contact discovery pulse.

    Clean-v0 deliberately keeps this pulse small.  It establishes the
    discrete contact fact; the outgoing ball result is evaluated separately
    by the fixed next-node progress/arrival terms, not by a second
    effective/closure payment on the same collision.
    """
    return dribbling_s2_contact_event_state(env, **event_params)[
        "immediate_contact"
    ].to(torch.float32)


def dribbling_s2_next_touch_release(
    env: ManagerBasedRLEnv,
    event_params: dict,
) -> torch.Tensor:
    """Score the latched eight-step outgoing release toward the next anchor."""
    return dribbling_s2_contact_event_state(env, **event_params)["release_score"]


def dribbling_s2_next_touch_release_v2(
    env: ManagerBasedRLEnv,
    event_params: dict,
) -> torch.Tensor:
    """Score position displacement toward the latched next anchor."""
    return dribbling_s2_contact_event_state(env, **event_params)["release_v2_score"]


def dribbling_s2_next_anchor_progress(
    env: ManagerBasedRLEnv,
    event_params: dict,
) -> torch.Tensor:
    """Reward signed distance progress until the next contact window starts."""
    return dribbling_s2_contact_event_state(env, **event_params)[
        "release_arrival_progress"
    ]


def dribbling_s2_next_anchor_window_tracking(
    env: ManagerBasedRLEnv,
    event_params: dict,
) -> torch.Tensor:
    """Reward absolute fixed-node tracking at the next contact window.

    The pulse is emitted once, after the complete labelled window, from the
    minimum ball-center distance to the immutable episode-fixed next anchor.
    No live root transform, relative-improvement test, ball velocity, or
    intermediate trajectory is used.
    """
    return dribbling_s2_contact_event_state(env, **event_params)[
        "terminal_tracking_pulse"
    ]


def dribbling_s2_next_window_outcome(
    env: ManagerBasedRLEnv,
    event_params: dict,
) -> torch.Tensor:
    """Emit the once-per-transition window-average position outcome rate."""
    return dribbling_s2_contact_event_state(env, **event_params)[
        "window_outcome_pulse"
    ]


def dribbling_s2_delayed_transition_settlement(
    env: ManagerBasedRLEnv,
    event_params: dict,
) -> torch.Tensor:
    """Settle a pending touch from the worse of window-average and end quality."""
    return dribbling_s2_contact_event_state(env, **event_params)[
        "window_outcome_settlement_pulse"
    ]


def dribbling_s2_surface_style(
    env: ManagerBasedRLEnv,
    event_params: dict,
) -> torch.Tensor:
    """Low-weight labelled inside/outside style pulse for a legal touch."""
    return dribbling_s2_contact_event_state(env, **event_params)["surface_style"]


def dribbling_s2_sequence_progress_bonus(
    env: ManagerBasedRLEnv,
    event_params: dict,
) -> torch.Tensor:
    """Increasing one-pulse bonus for each consecutively completed S2 event."""
    return dribbling_s2_contact_event_state(env, **event_params)["sequence_progress"]


def dribbling_s2_global_foot_ball_distance_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    foot_body_names: list[str] | None = None,
    global_foot_ball_distance_std: float = 0.40,
) -> torch.Tensor:
    """Reward actual target-foot/ball distance without a contact-sensor gate."""
    std = float(global_foot_ball_distance_std)
    if std <= 0.0:
        raise ValueError("global_foot_ball_distance_std must be positive")
    command: MotionCommand = env.command_manager.get_term(command_name)
    required = (
        "s2_contact_event_id_ref",
        "s2_contact_event_frame_ref",
        "s2_contact_event_foot_ref",
        "s2_contact_event_surface_ref",
        "s2_upcoming_contact_event_ref",
    )
    if any(not hasattr(command, name) for name in required):
        score = torch.zeros(env.num_envs, device=env.device)
        command.metrics["s2_global_foot_ball_distance"] = score
        command.metrics["s2_global_foot_ball_score"] = score
        command.metrics["s2_global_foot_ball_expected_foot"] = torch.full_like(score, -1)
        command.metrics["s2_global_foot_ball_active"] = score
        return score
    if foot_body_names is None:
        foot_body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    if len(foot_body_names) != 2:
        raise ValueError("S2 global foot-ball distance requires exactly two foot bodies")
    try:
        left_slot = foot_body_names.index("left_ankle_roll_link")
        right_slot = foot_body_names.index("right_ankle_roll_link")
        body_indices = [command.cfg.body_names.index(name) for name in foot_body_names]
    except ValueError as exc:
        raise ValueError(
            "S2 global foot-ball distance requires left/right ankle roll links in the motion command"
        ) from exc
    target_event = _dribbling_s2_target_event(command)
    expected_foot = target_event["expected_foot"]
    active = (target_event["event_id"] >= 0) & (
        (expected_foot == 0) | (expected_foot == 1)
    )
    selected_slot = torch.where(
        expected_foot == 1,
        torch.full_like(expected_foot, right_slot),
        torch.full_like(expected_foot, left_slot),
    )
    batch = torch.arange(env.num_envs, device=env.device)
    actual_foot_positions = command.robot_body_pos_w[:, body_indices]
    actual_foot_position = actual_foot_positions[batch, selected_slot]
    ball_position = env.scene["soccer_ball"].data.root_pos_w[:, :3]
    distance = torch.linalg.vector_norm(ball_position - actual_foot_position, dim=-1)
    score = torch.exp(-torch.square(distance / std)) * active.to(distance.dtype)
    command.metrics["s2_global_foot_ball_distance"] = torch.where(
        active, distance, torch.zeros_like(distance)
    )
    command.metrics["s2_global_foot_ball_score"] = score
    command.metrics["s2_global_foot_ball_expected_foot"] = torch.where(
        active, expected_foot, torch.full_like(expected_foot, -1)
    ).to(torch.float32)
    command.metrics["s2_global_foot_ball_active"] = active.to(torch.float32)
    return score


def dribbling_s2_touch_occurrence(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
    require_expected_foot: bool = True,
    target_side_enabled: bool = True,
    side_deadzone: float = 0.04,
    proximity_contact_distance_max: float = 0.25,
    target_region_std: float = 0.12,
    proximity_approach_seconds: float = 0.30,
    proximity_approach_min_weight: float = 0.20,
    missed_contact_grace_steps: int = 3,
) -> torch.Tensor:
    """Return the first in-window expected-foot contact pulse for telemetry."""
    return dribbling_s2_contact_event_state(
        env, command_name, ball_sensor_name, all_body_cfg, num_ankle_links,
        require_expected_foot, target_side_enabled,
        side_deadzone, proximity_contact_distance_max, target_region_std, proximity_approach_seconds,
        proximity_approach_min_weight, missed_contact_grace_steps,
    )["touch_occurrence"].to(torch.float32)


def dribbling_s2_valid_contact_bonus(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
    require_expected_foot: bool = True,
    target_side_enabled: bool = True,
    side_deadzone: float = 0.04,
    proximity_contact_distance_max: float = 0.25,
    target_region_std: float = 0.12,
    proximity_approach_seconds: float = 0.30,
    proximity_approach_min_weight: float = 0.20,
    missed_contact_grace_steps: int = 3,
) -> torch.Tensor:
    """Reward the first expected-foot rising edge on the labelled side."""
    return dribbling_s2_contact_event_state(
        env, command_name, ball_sensor_name, all_body_cfg, num_ankle_links,
        require_expected_foot, target_side_enabled,
        side_deadzone, proximity_contact_distance_max, target_region_std, proximity_approach_seconds,
        proximity_approach_min_weight, missed_contact_grace_steps,
    )["valid_contact_event"].to(torch.float32)


def dribbling_s2_nonfoot_ball_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
    require_expected_foot: bool = True,
    target_side_enabled: bool = True,
    side_deadzone: float = 0.04,
    proximity_contact_distance_max: float = 0.25,
    target_region_std: float = 0.12,
    proximity_approach_seconds: float = 0.30,
    proximity_approach_min_weight: float = 0.20,
    missed_contact_grace_steps: int = 3,
) -> torch.Tensor:
    """Return one pulse when a non-ankle robot link contacts the ball."""
    return dribbling_s2_contact_event_state(
        env, command_name, ball_sensor_name, all_body_cfg, num_ankle_links,
        require_expected_foot, target_side_enabled,
        side_deadzone, proximity_contact_distance_max, target_region_std, proximity_approach_seconds,
        proximity_approach_min_weight, missed_contact_grace_steps,
    )["nonfoot_contact_penalty"]


def dribbling_s2_nonfoot_contact_termination(
    env: ManagerBasedRLEnv,
    event_params: dict,
) -> torch.Tensor:
    """Terminate a clean transition on a filtered non-foot ball contact."""
    return dribbling_s2_contact_event_state(env, **event_params)["invalid_body_contact"]


def dribbling_s2_missed_contact_termination(
    env: ManagerBasedRLEnv,
    event_params: dict,
) -> torch.Tensor:
    """Terminate a clean rollout when its expected window is missed.

    The event-state function emits this pulse exactly when the labelled
    window (plus the configured grace interval) closes.  It is deliberately
    separate from the reward pulse: motion tracking may continue in legacy
    tasks, while clean-v0 treats a missed node as an ordinary episode
    failure and resets the external reference clock/LSTM.
    """
    return dribbling_s2_contact_event_state(env, **event_params)["missed_valid_contact"]


def dribbling_s2_wrong_foot_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
    require_expected_foot: bool = True,
    target_side_enabled: bool = True,
    side_deadzone: float = 0.04,
    proximity_contact_distance_max: float = 0.25,
    target_region_std: float = 0.12,
    proximity_approach_seconds: float = 0.30,
    proximity_approach_min_weight: float = 0.20,
    missed_contact_grace_steps: int = 3,
    event_params: dict | None = None,
) -> torch.Tensor:
    """Return one pulse for an in-window legal-ankle touch by the wrong foot."""
    if event_params is not None:
        return dribbling_s2_contact_event_state(env, **event_params)[
            "wrong_foot_contact_penalty"
        ]
    return dribbling_s2_contact_event_state(
        env, command_name, ball_sensor_name, all_body_cfg, num_ankle_links,
        require_expected_foot, target_side_enabled,
        side_deadzone, proximity_contact_distance_max, target_region_std, proximity_approach_seconds,
        proximity_approach_min_weight, missed_contact_grace_steps,
    )["wrong_foot_contact_penalty"]


def dribbling_s2_wrong_side_contact_penalty(
    env: ManagerBasedRLEnv,
    event_params: dict,
) -> torch.Tensor:
    """One pulse when the expected foot touches from the labelled wrong side.

    The side is the sign of the physical ball centre in the expected ankle's
    yaw-local frame; no mesh contact-point classification is assumed.
    """
    return dribbling_s2_contact_event_state(env, **event_params)["wrong_side"].to(torch.float32)


def dribbling_s2_premature_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
    require_expected_foot: bool = True,
    target_side_enabled: bool = True,
    side_deadzone: float = 0.04,
    proximity_contact_distance_max: float = 0.25,
    target_region_std: float = 0.12,
    proximity_approach_seconds: float = 0.30,
    proximity_approach_min_weight: float = 0.20,
    missed_contact_grace_steps: int = 3,
    event_params: dict | None = None,
) -> torch.Tensor:
    """Return at most one premature-contact pulse per selected target event."""
    if event_params is not None:
        return dribbling_s2_contact_event_state(env, **event_params)[
            "premature_contact_penalty"
        ]
    return dribbling_s2_contact_event_state(
        env, command_name, ball_sensor_name, all_body_cfg, num_ankle_links,
        require_expected_foot, target_side_enabled,
        side_deadzone, proximity_contact_distance_max, target_region_std, proximity_approach_seconds,
        proximity_approach_min_weight, missed_contact_grace_steps,
    )["premature_contact_penalty"]


def dribbling_s2_windowed_foot_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 0.3,
    foot_body_names: list[str] | None = None,
    contact_foot_scale: float = 0.1,
    contact_foot_release_seconds: float = 0.3,
) -> torch.Tensor:
    """Track both feet while gradually releasing the expected contact foot.

    The support foot always keeps its full reference target.  The future
    contact foot ramps from full imitation to ``contact_foot_scale`` during
    the final ``contact_foot_release_seconds`` before its event, stays released
    through the post-contact window, and then returns to full tracking.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    if foot_body_names is None:
        foot_body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    if len(foot_body_names) != 2:
        raise ValueError("S2 windowed foot tracking requires exactly two foot bodies")
    try:
        left_slot = foot_body_names.index("left_ankle_roll_link")
        right_slot = foot_body_names.index("right_ankle_roll_link")
        body_indexes = [command.cfg.body_names.index(name) for name in foot_body_names]
    except ValueError as exc:
        raise ValueError(
            "S2 windowed foot tracking requires left/right ankle roll links in the motion command"
        ) from exc
    squared_error = torch.sum(
        torch.square(
            command.body_pos_relative_w[:, body_indexes]
            - command.robot_body_pos_w[:, body_indexes]
        ),
        dim=-1,
    )
    window = getattr(command, "s2_contact_window_ref", None)
    expected_foot = getattr(command, "s2_contact_event_foot_ref", None)
    foot_weights = torch.ones_like(squared_error)
    if isinstance(window, torch.Tensor) and isinstance(expected_foot, torch.Tensor):
        contact_scale = min(1.0, max(0.0, float(contact_foot_scale)))
        target_foot = expected_foot
        target_scale = torch.ones(env.num_envs, device=env.device, dtype=foot_weights.dtype)

        upcoming = getattr(command, "s2_upcoming_contact_event_ref", None)
        release_seconds = max(0.0, float(contact_foot_release_seconds))
        if callable(upcoming) and release_seconds > 0.0:
            _, upcoming_frame, upcoming_foot, _ = upcoming()
            target_frame = torch.where(
                window, command.s2_contact_event_frame_ref, upcoming_frame
            )
            target_foot = torch.where(window, expected_foot, upcoming_foot)
            fps = max(float(command.motion.fps.reshape(-1)[0]), 1.0e-6)
            seconds_to_event = (
                target_frame - command.time_steps.to(torch.long)
            ).to(torch.float32) / fps
            approach = (
                (target_frame >= 0)
                & ((target_foot == 0) | (target_foot == 1))
                & (seconds_to_event >= 0.0)
                & (seconds_to_event <= release_seconds)
            )
            progress = 1.0 - torch.clamp(
                seconds_to_event / release_seconds, min=0.0, max=1.0
            )
            ramp_scale = 1.0 - (1.0 - contact_scale) * progress
            target_scale = torch.where(approach, ramp_scale, target_scale)

        # Once the event begins, retain the minimum scale through the complete
        # annotated contact segment.  The pre-event ramp is separate from the
        # segment and does not create an additional valid-contact window.
        post_event_window = window & (
            command.time_steps.to(torch.long) >= command.s2_contact_event_frame_ref
        )
        target_scale = torch.where(
            post_event_window,
            torch.full_like(target_scale, contact_scale),
            target_scale,
        )
        left_contact = (target_foot == 0) & (target_scale < 1.0)
        right_contact = (target_foot == 1) & (target_scale < 1.0)
        foot_weights[:, left_slot] = torch.where(
            left_contact,
            target_scale,
            foot_weights[:, left_slot],
        )
        foot_weights[:, right_slot] = torch.where(
            right_contact,
            target_scale,
            foot_weights[:, right_slot],
        )
    weighted_error = torch.sum(squared_error * foot_weights, dim=-1) / torch.sum(
        foot_weights, dim=-1
    ).clamp(min=1.0e-6)
    reward = torch.exp(-weighted_error / max(float(std), 1.0e-6) ** 2)
    return reward * locomotion_task_state_mask(command, None).to(reward.dtype)


def dribbling_contact_telemetry(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
    contact_surface: str = "any",
    medial_y_min: float = 0.018,
    cg_surface_gated: bool = False,
    contact_force_threshold: float = 14.0,
) -> dict[str, torch.Tensor]:
    """Return contact-region telemetry using the exact reward geometry.

    Values describe only a current sensor contact. ``ball_offset_foot_yaw``
    is therefore NaN and ``contact_body_idx`` is -1 when the ball has no robot
    contact. Foot ids use the CG convention: left=0, right=1, unknown=-1.
    """
    device = env.device
    num_envs = env.num_envs
    if all_body_cfg is None:
        has_contact = soccer_ball_contact_force_magnitude(env, ball_sensor_name) > 1.0
        force_mag = soccer_ball_contact_force_magnitude(env, ball_sensor_name)
        false = torch.zeros(num_envs, dtype=torch.bool, device=device)
        return {
            "has_contact": has_contact,
            "force_magnitude": force_mag,
            "contact_body_idx": torch.full((num_envs,), -1, dtype=torch.long, device=device),
            "contact_foot": torch.full((num_envs,), -1, dtype=torch.long, device=device),
            "legal_ankle": false,
            "generic_instep": false,
            "inside_instep": false,
            "outside_instep": false,
            "requested_surface_match": false,
            "gentle": force_mag <= contact_force_threshold,
            "legal_touch": false,
            "ball_offset_foot_yaw": torch.full((num_envs, 3), torch.nan, device=device),
            "medial_offset": torch.full((num_envs,), torch.nan, device=device),
        }

    command: MotionCommand = env.command_manager.get_term(command_name)
    has_contact, force_mag, closest_body_idx = _identify_contact_body(
        env, command, ball_sensor_name, all_body_cfg
    )
    geometry = _dribbling_contact_surface_geometry(
        env,
        all_body_cfg,
        closest_body_idx,
        num_ankle_links=num_ankle_links,
        medial_y_min=medial_y_min,
    )
    requested_surface_match = _dribbling_contact_surface_match(
        env,
        all_body_cfg,
        closest_body_idx,
        command_name=command_name,
        contact_surface=contact_surface,
        num_ankle_links=num_ankle_links,
        medial_y_min=medial_y_min,
        cg_surface_gated=cg_surface_gated,
    )
    contact_foot = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    contact_foot[geometry["medial_sign"] < 0.0] = 0
    contact_foot[geometry["medial_sign"] > 0.0] = 1
    contact_body_idx = torch.where(
        has_contact,
        closest_body_idx,
        torch.full_like(closest_body_idx, -1),
    )
    offset = torch.where(
        has_contact.unsqueeze(-1),
        geometry["ball_offset_foot_yaw"],
        torch.full_like(geometry["ball_offset_foot_yaw"], torch.nan),
    )
    medial_offset = torch.where(
        has_contact,
        geometry["medial_offset"],
        torch.full_like(geometry["medial_offset"], torch.nan),
    )
    gentle = force_mag <= contact_force_threshold
    return {
        "has_contact": has_contact,
        "force_magnitude": force_mag,
        "contact_body_idx": contact_body_idx,
        "contact_foot": torch.where(has_contact, contact_foot, torch.full_like(contact_foot, -1)),
        "legal_ankle": has_contact & geometry["legal_ankle"],
        "generic_instep": has_contact & geometry["legal_ankle"] & geometry["known_foot"] & geometry["instep"],
        "inside_instep": has_contact & geometry["legal_ankle"] & geometry["known_foot"] & geometry["inside_instep"],
        "outside_instep": has_contact & geometry["legal_ankle"] & geometry["known_foot"] & geometry["outside_instep"],
        "requested_surface_match": has_contact & requested_surface_match,
        "gentle": has_contact & gentle,
        "legal_touch": has_contact & geometry["legal_ankle"] & requested_surface_match & gentle,
        "ball_offset_foot_yaw": offset,
        "medial_offset": medial_offset,
    }


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


def dribbling_ball_coast_without_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    speed_threshold: float = 0.35,
    speed_scale: float = 0.45,
    max_close_xy_dist: float = 0.55,
) -> torch.Tensor:
    """Penalty when the ball slides fast **near** the robot with no contact.

    Only applies when the ball is within ``max_close_xy_dist`` of the pelvis so a
    kick-and-chase rollout (ball rolling ahead out of reach) is not penalised.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    ball_sp = torch.norm(soccer_ball.data.root_lin_vel_w[:, :2], dim=-1)
    fmag = soccer_ball_contact_force_magnitude(env, ball_sensor_name)
    no_touch = fmag <= contact_force_threshold

    pelvis_pos_xy = command.robot_pelvis_pos_w[:, :2]
    ball_pos_xy = soccer_ball.data.root_pos_w[:, :2]
    dist_xy = torch.norm(ball_pos_xy - pelvis_pos_xy, dim=-1)
    close = dist_xy <= max_close_xy_dist

    excess = torch.relu(ball_sp - speed_threshold)
    penalty = torch.clamp(excess / max(speed_scale, 1e-6), max=1.0)
    return penalty * no_touch.to(torch.float32) * close.to(torch.float32)


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
    cg_gated_contact: bool = False,
) -> torch.Tensor:
    """Reward forward ball velocity along task +X (world-parallel).

    This term rewards positive +X ball speed, and can optionally be gated by
    recent contact to avoid standing
    exploits where the policy does not engage the ball.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    ball_vel_w = soccer_ball.data.root_lin_vel_w[:, :3]
    forward_speed = task_forward_speed(ball_vel_w)
    ball_dominance = task_velocity_forward_dominance(ball_vel_w)

    base = torch.clamp((forward_speed - min_forward_speed) / max(speed_scale, 1e-6), min=0.0, max=1.0)
    base = base * forward_dominance_gate(ball_dominance, 0.4)

    pelvis_speed = torch.norm(command.robot_anchor_lin_vel_w[:, :2], dim=-1)
    gate = torch.clamp(pelvis_speed / max(pelvis_speed_min, 1e-6), max=1.0)

    if require_recent_contact:
        gate = gate * _dribbling_recent_contact_gate(
            env,
            ball_sensor_name,
            contact_force_threshold,
            recent_contact_window,
            command=command,
            cg_gated=cg_gated_contact,
        )

    return base * gate


def dribbling_command_ball_progress_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    min_forward_speed: float = 0.2,
    command_speed_ratio: float = 0.50,
    speed_scale: float = 0.25,
    lateral_ratio_max: float = 0.70,
    pelvis_speed_min: float = 0.06,
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 0.5,
    require_recent_contact: bool = True,
    recent_contact_window: int = 10,
    cg_gated_contact: bool = False,
) -> torch.Tensor:
    """Reward recent-contact ball progress along the active command direction.

    ``command_speed_ratio`` makes high-speed commands demand proportionally
    faster ball progress without asking the ball to exactly match pelvis speed.
    A lateral gate prevents a sideways kick from scoring as forward dribbling.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    direction_xy, command_speed = _command_direction_xy(command)
    forward_speed, lateral_speed = _command_frame_components(soccer_ball.data.root_lin_vel_w[:, :2], direction_xy)

    required_speed = torch.maximum(
        torch.full_like(command_speed, float(min_forward_speed)),
        command_speed * float(command_speed_ratio),
    )
    progress = torch.clamp(
        (forward_speed - required_speed) / max(speed_scale, 1.0e-6), min=0.0, max=1.0
    )
    lateral_ratio = torch.abs(lateral_speed) / forward_speed.clamp(min=1.0e-4)
    direction_gate = torch.clamp(
        1.0 - lateral_ratio / max(lateral_ratio_max, 1.0e-6), min=0.0, max=1.0
    )

    pelvis_speed = torch.norm(command.robot_anchor_lin_vel_w[:, :2], dim=-1)
    gate = torch.clamp(pelvis_speed / max(pelvis_speed_min, 1.0e-6), max=1.0)
    if require_recent_contact:
        gate = gate * _dribbling_recent_contact_gate(
            env,
            ball_sensor_name,
            contact_force_threshold,
            recent_contact_window,
            command=command,
            cg_gated=cg_gated_contact,
        )
    return progress * direction_gate * gate


# ---------------------------------------------------------------------------
# 2f) Contact-graph style phase alignment (approach -> control/push)
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
    contact_surface: str = "any",
    medial_y_min: float = 0.018,
    cg_gated: bool = False,
    cg_surface_gated: bool = False,
    min_pelvis_heading: float = 0.0,
) -> torch.Tensor:
    """Reward a new gentle touch in the selected foot/instep side.

    ``contact_surface`` is ``any`` (legacy), ``instep`` (either side),
    ``inside_instep``, or ``outside_instep``. The instep variants use the
    contacted foot's yaw frame, so selecting one is stronger than merely
    selecting the left or right foot.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    has_contact, force_mag, closest_idx = _identify_contact_body(
        env, command, ball_sensor_name, all_body_cfg,
    )

    is_ankle = _is_dribble_legal_ankle_contact(closest_idx, num_ankle_links)
    surface_match = _dribbling_contact_surface_match(
        env,
        all_body_cfg,
        closest_idx,
        command_name=command_name,
        contact_surface=contact_surface,
        num_ankle_links=num_ankle_links,
        medial_y_min=medial_y_min,
        cg_surface_gated=cg_surface_gated,
    )
    gentle = force_mag <= force_threshold
    touch = has_contact & is_ankle & surface_match & gentle
    if cg_gated:
        touch = _dribbling_cg_gated_sim_contact(command, touch)

    prev_name = "_dribbling_prev_legal_foot_touch"
    prev = getattr(env, prev_name, None)
    if prev is None or prev.shape[0] != env.num_envs:
        prev = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    new_touch = touch & ~prev
    setattr(env, prev_name, touch.detach().clone())

    reward = new_touch.to(torch.float32)
    if min_pelvis_heading > 0.0:
        heading = task_pelvis_heading_cos_world_x(command.robot_pelvis_quat_w).clamp(min=0.0, max=1.0)
        reward = reward * forward_dominance_gate(heading, min_pelvis_heading)
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
# 3c) Undesired Contact Penalty — wrong body or instep side
# ---------------------------------------------------------------------------

def dribbling_undesired_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
    contact_surface: str = "any",
    medial_y_min: float = 0.018,
    cg_surface_gated: bool = False,
    wrong_surface_penalty: float = 0.25,
) -> torch.Tensor:
    """Penalize non-ankle contact fully and wrong instep side more gently.

    The term is multiplied by its configured negative reward weight.  A
    non-ankle contact returns ``1.0``; an ankle contact on the wrong
    inside/outside instep returns ``wrong_surface_penalty``.  The latter is
    intentionally smaller: timing is already supervised independently by
    ``dribbling_cg_premature_contact_penalty``.
    """
    if not 0.0 <= wrong_surface_penalty <= 1.0:
        raise ValueError("wrong_surface_penalty must be in [0, 1].")
    command: MotionCommand = env.command_manager.get_term(command_name)
    has_contact, force_mag, closest_idx = _identify_contact_body(
        env, command, ball_sensor_name, all_body_cfg,
    )

    is_ankle = _is_dribble_legal_ankle_contact(closest_idx, num_ankle_links)
    surface_match = _dribbling_contact_surface_match(
        env,
        all_body_cfg,
        closest_idx,
        command_name=command_name,
        contact_surface=contact_surface,
        num_ankle_links=num_ankle_links,
        medial_y_min=medial_y_min,
        cg_surface_gated=cg_surface_gated,
    )

    non_ankle = has_contact & ~is_ankle
    wrong_surface = has_contact & is_ankle & ~surface_match
    return non_ankle.to(torch.float32) + wrong_surface.to(torch.float32) * wrong_surface_penalty


# ---------------------------------------------------------------------------
# 4) Annotated contact-graph (dribbling) — demo ball + label consistency
# ---------------------------------------------------------------------------


def dribbling_cg_contact_consistency(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    active_task_states: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """1.0 when sim contact presence matches the annotated CG contact bit."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    labeled = _dribbling_effective_cg_label_mask(command, command.motion_has_dribble_cg_label)
    if not torch.any(labeled):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    ref = command.dribble_cg_contact_ref
    sim_c = _dribbling_sim_contact(env, ball_sensor_name, contact_force_threshold)
    agree = (ref == sim_c).to(torch.float32)
    active = labeled & locomotion_task_state_mask(command, active_task_states)
    return agree * active.to(torch.float32)


def dribbling_cg_premature_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize sensor contact on CG non-contact (approach) frames."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    labeled = _dribbling_effective_cg_label_mask(command, command.motion_has_dribble_cg_label)
    if not torch.any(labeled):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    ref = command.dribble_cg_contact_ref
    sim_c = _dribbling_sim_contact(env, ball_sensor_name, contact_force_threshold)
    return (labeled & sim_c & ~ref).to(torch.float32)


def dribbling_ball_trapped_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    min_forward_x: float = 0.18,
    max_ball_height: float = 0.20,
) -> torch.Tensor:
    """Penalize ball under the body, behind the pelvis, or popped up (夹球 / 蹦)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    ball_pos_w = soccer_ball.data.root_pos_w
    pelvis_pos_w = command.robot_pelvis_pos_w
    x_task = task_forward_offset(ball_pos_w, pelvis_pos_w)
    too_close = x_task < min_forward_x
    behind = x_task < 0.0
    popped = ball_pos_w[:, 2] > max_ball_height
    return (too_close | behind | popped).to(torch.float32)


def dribbling_pelvis_local_ball_trapped_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    min_forward_x: float = 0.18,
    max_ball_height: float = 0.20,
) -> torch.Tensor:
    """Penalize a ball under/behind the current pelvis in pelvis-local axes."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    offset_b = _pelvis_yaw_local_vector(
        command, soccer_ball.data.root_pos_w[:, :3] - command.robot_pelvis_pos_w
    )
    too_close = offset_b[:, 0] < min_forward_x
    behind = offset_b[:, 0] < 0.0
    popped = soccer_ball.data.root_pos_w[:, 2] > max_ball_height
    return (too_close | behind | popped).to(torch.float32)


def dribbling_sustained_contact_penalty(
    env: ManagerBasedRLEnv,
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    max_contact_steps: int = 5,
) -> torch.Tensor:
    """Penalize keeping the ball squeezed between feet for too many consecutive steps."""
    in_contact = _dribbling_sim_contact(env, ball_sensor_name, contact_force_threshold)
    buf_name = "_dribbling_consecutive_contact_steps"
    cnt = getattr(env, buf_name, None)
    if cnt is None or cnt.shape[0] != env.num_envs:
        cnt = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)

    step_buf = getattr(env, "episode_length_buf", None)
    reset_mask = step_buf == 0 if step_buf is not None else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    cnt = torch.where(reset_mask, torch.zeros_like(cnt), torch.where(in_contact, cnt + 1, torch.zeros_like(cnt)))
    setattr(env, buf_name, cnt)
    return (cnt > int(max_contact_steps)).to(torch.float32)


def dribbling_ball_bounce_penalty(
    env: ManagerBasedRLEnv,
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    vz_threshold: float = 0.32,
) -> torch.Tensor:
    """Penalize large vertical ball speed while in contact (pinch-bounce)."""
    soccer_ball = env.scene["soccer_ball"]
    in_contact = _dribbling_sim_contact(env, ball_sensor_name, contact_force_threshold)
    vz = torch.abs(soccer_ball.data.root_lin_vel_w[:, 2])
    return (in_contact & (vz > vz_threshold)).to(torch.float32)


def dribbling_cg_foot_consistency(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    all_body_cfg: SceneEntityCfg | None = None,
    left_ankle_body_name: str = "left_ankle_roll_link",
    right_ankle_body_name: str = "right_ankle_roll_link",
    contact_surface: str = "any",
    medial_y_min: float = 0.018,
    cg_surface_gated: bool = False,
    active_task_states: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Reward a labeled-foot contact, optionally matching its CG instep side."""
    if all_body_cfg is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    command: MotionCommand = env.command_manager.get_term(command_name)
    labeled = _dribbling_effective_cg_label_mask(command, command.motion_has_dribble_cg_label)
    if not torch.any(labeled):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    ref_c = command.dribble_cg_contact_ref
    ref_f = command.dribble_cg_foot_ref
    active = labeled & ref_c & (ref_f >= 0) & locomotion_task_state_mask(command, active_task_states)

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

    surface_match = _dribbling_contact_surface_match(
        env,
        all_body_cfg,
        closest,
        command_name=command_name,
        contact_surface=contact_surface,
        num_ankle_links=2,
        medial_y_min=medial_y_min,
        cg_surface_gated=cg_surface_gated,
    )
    match = (closest == expected) & surface_match & has_contact & active
    return match.to(torch.float32)


def _dribbling_cg_flow_metrics(
    env: ManagerBasedRLEnv,
    *,
    command_name: str,
    ball_sensor_name: str,
    all_body_cfg: SceneEntityCfg | None,
    num_ankle_links: int,
    contact_surface: str,
    medial_y_min: float,
    contact_force_threshold: float,
    max_touch_force: float,
    release_window_steps: int,
    speed_lower_ratio: float,
    speed_upper_ratio: float,
    lateral_speed_std: float,
    overspeed_std: float,
    lateral_corridor_std: float,
    max_progress_rate: float,
    active_task_states: tuple[int, ...] | None,
) -> dict[str, torch.Tensor]:
    """Update and return the shared causal contact-flow reward state once per step."""
    if release_window_steps <= 0:
        raise ValueError("release_window_steps must be positive.")
    if not 0.0 < speed_lower_ratio <= speed_upper_ratio:
        raise ValueError("Expected 0 < speed_lower_ratio <= speed_upper_ratio.")

    step_counter = int(getattr(env, "common_step_counter", -1))
    episode_step = getattr(env, "episode_length_buf", None)
    if episode_step is None:
        episode_step = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    signature = (
        command_name,
        ball_sensor_name,
        tuple(all_body_cfg.body_names) if all_body_cfg is not None else (),
        num_ankle_links,
        contact_surface,
        medial_y_min,
        contact_force_threshold,
        max_touch_force,
        release_window_steps,
        speed_lower_ratio,
        speed_upper_ratio,
        lateral_speed_std,
        overspeed_std,
        lateral_corridor_std,
        max_progress_rate,
        active_task_states,
    )
    cache = getattr(env, "_dribbling_cg_flow_cache", None)
    if isinstance(cache, dict) and cache.get("signature") == signature:
        same_step = step_counter >= 0 and cache.get("step_counter") == step_counter
        if step_counter < 0:
            cached_episode_step = cache.get("episode_step")
            same_step = isinstance(cached_episode_step, torch.Tensor) and torch.equal(
                cached_episode_step, episode_step
            )
        if same_step:
            return cache["metrics"]

    device = env.device
    zeros = torch.zeros(env.num_envs, device=device, dtype=torch.float32)
    false = torch.zeros(env.num_envs, device=device, dtype=torch.bool)
    minus_one = torch.full((env.num_envs,), -1, device=device, dtype=torch.long)
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    has_flow = getattr(command, "motion_has_dribble_cg_flow_label", false)
    if not isinstance(has_flow, torch.Tensor):
        has_flow = false
    label_valid = getattr(command, "dribble_cg_flow_valid_ref", false)
    if not isinstance(label_valid, torch.Tensor):
        label_valid = false
    task_active = locomotion_task_state_mask(command, active_task_states)
    flow_state_enabled = _dribbling_effective_cg_label_mask(command, has_flow) & task_active
    label_valid = _dribbling_effective_cg_label_mask(command, has_flow & label_valid) & task_active
    flow_dir_local = getattr(command, "dribble_cg_flow_dir_local_ref", None)
    flow_distance = getattr(command, "dribble_cg_flow_distance_ref", None)
    flow_duration = getattr(command, "dribble_cg_flow_duration_ref", None)
    flow_anchor = getattr(command, "dribble_cg_flow_anchor_frame_ref", None)
    if not isinstance(flow_dir_local, torch.Tensor):
        flow_dir_local = torch.zeros(env.num_envs, 2, device=device)
    if not isinstance(flow_distance, torch.Tensor):
        flow_distance = torch.full_like(zeros, -1.0)
    if not isinstance(flow_duration, torch.Tensor):
        flow_duration = torch.full_like(zeros, -1.0)
    if not isinstance(flow_anchor, torch.Tensor):
        flow_anchor = minus_one

    state = getattr(env, "_dribbling_cg_flow_state", None)
    if not isinstance(state, dict) or state.get("num_envs") != env.num_envs:
        state = {
            "num_envs": env.num_envs,
            "active": false.clone(),
            "anchor_frame": minus_one.clone(),
            "target_direction_world": torch.zeros(env.num_envs, 2, device=device),
            "touch_ball_position": torch.zeros(env.num_envs, 2, device=device),
            "target_distance": zeros.clone(),
            "target_duration": zeros.clone(),
            "age_steps": torch.zeros(env.num_envs, device=device, dtype=torch.long),
            "best_progress": zeros.clone(),
            "previous_contact": false.clone(),
        }

    reset = episode_step == 0
    state["active"] = state["active"] & ~reset
    state["anchor_frame"] = torch.where(reset, minus_one, state["anchor_frame"])
    state["age_steps"] = torch.where(reset, torch.zeros_like(state["age_steps"]), state["age_steps"])
    state["best_progress"] = torch.where(reset, zeros, state["best_progress"])
    state["previous_contact"] = state["previous_contact"] & ~reset

    if all_body_cfg is None:
        has_contact = false
        force_mag = zeros
        closest = torch.zeros(env.num_envs, device=device, dtype=torch.long)
        surface_match = false
        actual_foot = minus_one
    else:
        has_contact, force_mag, closest = _identify_contact_body(
            env, command, ball_sensor_name, all_body_cfg
        )
        has_contact = has_contact & (force_mag >= contact_force_threshold)
        surface_match = _dribbling_contact_surface_match(
            env,
            all_body_cfg,
            closest,
            command_name=command_name,
            contact_surface=contact_surface,
            num_ankle_links=num_ankle_links,
            medial_y_min=medial_y_min,
            cg_surface_gated=True,
        )
        actual_foot = minus_one.clone()
        for body_idx, body_name in enumerate(all_body_cfg.body_names):
            if body_idx >= num_ankle_links:
                break
            if "left_ankle" in body_name:
                actual_foot[closest == body_idx] = 0
            elif "right_ankle" in body_name:
                actual_foot[closest == body_idx] = 1

    new_contact = has_contact & ~state["previous_contact"]
    state["previous_contact"] = has_contact.detach().clone()
    ref_contact = getattr(command, "dribble_cg_contact_ref", false)
    ref_foot = getattr(command, "dribble_cg_foot_ref", minus_one)
    direction_norm = torch.norm(flow_dir_local, dim=-1)
    label_ready = (
        label_valid
        & ref_contact
        & (flow_anchor >= 0)
        & (flow_distance > 1.0e-4)
        & (flow_duration > 1.0e-4)
        & (direction_norm > 1.0e-4)
    )
    correct_arrival_touch = (
        new_contact
        & (actual_foot == ref_foot)
        & surface_match
        & (force_mag <= max_touch_force)
        & ref_contact
    )
    correct_new_touch = (
        correct_arrival_touch
        & label_ready
        & (flow_anchor != state["anchor_frame"])
    )

    direction_local_unit = flow_dir_local / direction_norm.unsqueeze(-1).clamp(min=1.0e-6)
    direction_local_3d = torch.cat(
        (direction_local_unit, torch.zeros(env.num_envs, 1, device=device)), dim=-1
    )
    direction_world = quat_apply(yaw_quat(command.robot_pelvis_quat_w), direction_local_3d)[:, :2]
    direction_world = direction_world / torch.norm(direction_world, dim=-1, keepdim=True).clamp(min=1.0e-6)
    ball_position = soccer_ball.data.root_pos_w[:, :2]

    latch_2d = correct_new_touch.unsqueeze(-1)
    # A correct touch is the arrival condition for the previous segment. A
    # non-final anchor immediately latches the next outgoing segment; the final
    # anchor only clears the previous one because it has no flow label.
    state["active"] = torch.where(correct_arrival_touch, torch.zeros_like(state["active"]), state["active"])
    state["target_direction_world"] = torch.where(
        latch_2d, direction_world, state["target_direction_world"]
    )
    state["touch_ball_position"] = torch.where(
        latch_2d, ball_position, state["touch_ball_position"]
    )
    state["target_distance"] = torch.where(correct_new_touch, flow_distance, state["target_distance"])
    state["target_duration"] = torch.where(correct_new_touch, flow_duration, state["target_duration"])
    state["anchor_frame"] = torch.where(correct_new_touch, flow_anchor, state["anchor_frame"])
    state["age_steps"] = torch.where(correct_new_touch, torch.zeros_like(state["age_steps"]), state["age_steps"])
    state["best_progress"] = torch.where(correct_new_touch, zeros, state["best_progress"])
    state["active"] = torch.where(correct_new_touch, torch.ones_like(state["active"]), state["active"])

    segment_active = state["active"] & flow_state_enabled
    state["active"] = segment_active
    target_direction = state["target_direction_world"]
    lateral_axis = torch.stack((-target_direction[:, 1], target_direction[:, 0]), dim=-1)
    ball_velocity = soccer_ball.data.root_lin_vel_w[:, :2]
    parallel_speed = torch.sum(ball_velocity * target_direction, dim=-1)
    lateral_speed = torch.abs(torch.sum(ball_velocity * lateral_axis, dim=-1))
    nominal_speed = state["target_distance"] / state["target_duration"].clamp(min=1.0e-4)

    lower_speed = speed_lower_ratio * nominal_speed
    upper_speed = speed_upper_ratio * nominal_speed
    underspeed_score = torch.clamp(parallel_speed / lower_speed.clamp(min=1.0e-4), min=0.0, max=1.0)
    overspeed_score = torch.where(
        parallel_speed <= upper_speed,
        torch.ones_like(parallel_speed),
        torch.exp(-torch.square((parallel_speed - upper_speed) / max(overspeed_std, 1.0e-6))),
    )
    direction_score = torch.exp(-torch.square(lateral_speed / max(lateral_speed_std, 1.0e-6)))
    release_active = segment_active & (state["age_steps"] < int(release_window_steps))
    release_reward = underspeed_score * overspeed_score * direction_score * release_active.to(torch.float32)

    displacement = ball_position - state["touch_ball_position"]
    longitudinal = torch.sum(displacement * target_direction, dim=-1)
    lateral_offset = torch.abs(torch.sum(displacement * lateral_axis, dim=-1))
    raw_progress = torch.clamp(
        longitudinal / state["target_distance"].clamp(min=1.0e-4), min=0.0, max=1.0
    )
    next_best_progress = torch.where(
        segment_active, torch.maximum(state["best_progress"], raw_progress), state["best_progress"]
    )
    progress_delta = torch.clamp(next_best_progress - state["best_progress"], min=0.0)
    state["best_progress"] = next_best_progress
    step_dt = max(float(getattr(env, "step_dt", 0.02)), 1.0e-6)
    progress_rate = torch.clamp(progress_delta / step_dt, max=max_progress_rate)
    corridor_score = torch.exp(-torch.square(lateral_offset / max(lateral_corridor_std, 1.0e-6)))
    progress_reward = progress_rate * corridor_score * segment_active.to(torch.float32)
    state["age_steps"] = torch.where(
        segment_active, state["age_steps"] + 1, state["age_steps"]
    )
    setattr(env, "_dribbling_cg_flow_state", state)

    metrics = {
        "label_available": has_flow.to(torch.bool),
        "label_valid": label_valid,
        "label_anchor_frame": flow_anchor,
        "label_direction_local": flow_dir_local,
        "label_distance": flow_distance,
        "label_duration": flow_duration,
        "latched": state["active"],
        "release_active": release_active,
        "target_direction_world": target_direction,
        "nominal_speed": nominal_speed,
        "parallel_speed": parallel_speed,
        "lateral_speed": lateral_speed,
        "release_reward": release_reward,
        "progress": state["best_progress"],
        "progress_rate": progress_rate,
        "lateral_offset": lateral_offset,
        "progress_reward": progress_reward,
    }
    setattr(env, "_dribbling_cg_flow_telemetry", metrics)
    setattr(
        env,
        "_dribbling_cg_flow_cache",
        {
            "signature": signature,
            "step_counter": step_counter,
            "episode_step": episode_step.detach().clone(),
            "metrics": metrics,
        },
    )
    return metrics


def dribbling_cg_flow_release_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
    contact_surface: str = "instep",
    medial_y_min: float = 0.018,
    contact_force_threshold: float = 1.0,
    max_touch_force: float = 20.0,
    release_window_steps: int = 8,
    speed_lower_ratio: float = 0.7,
    speed_upper_ratio: float = 1.6,
    lateral_speed_std: float = 0.35,
    overspeed_std: float = 0.5,
    lateral_corridor_std: float = 0.18,
    max_progress_rate: float = 6.0,
    active_task_states: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Reward outgoing ball direction and a broad speed band after a correct touch."""
    return _dribbling_cg_flow_metrics(
        env,
        command_name=command_name,
        ball_sensor_name=ball_sensor_name,
        all_body_cfg=all_body_cfg,
        num_ankle_links=num_ankle_links,
        contact_surface=contact_surface,
        medial_y_min=medial_y_min,
        contact_force_threshold=contact_force_threshold,
        max_touch_force=max_touch_force,
        release_window_steps=release_window_steps,
        speed_lower_ratio=speed_lower_ratio,
        speed_upper_ratio=speed_upper_ratio,
        lateral_speed_std=lateral_speed_std,
        overspeed_std=overspeed_std,
        lateral_corridor_std=lateral_corridor_std,
        max_progress_rate=max_progress_rate,
        active_task_states=active_task_states,
    )["release_reward"]


def dribbling_cg_flow_progress_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
    contact_surface: str = "instep",
    medial_y_min: float = 0.018,
    contact_force_threshold: float = 1.0,
    max_touch_force: float = 20.0,
    release_window_steps: int = 8,
    speed_lower_ratio: float = 0.7,
    speed_upper_ratio: float = 1.6,
    lateral_speed_std: float = 0.35,
    overspeed_std: float = 0.5,
    lateral_corridor_std: float = 0.18,
    max_progress_rate: float = 6.0,
    active_task_states: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Reward one-way progress toward the next contact without tracking a full path."""
    return _dribbling_cg_flow_metrics(
        env,
        command_name=command_name,
        ball_sensor_name=ball_sensor_name,
        all_body_cfg=all_body_cfg,
        num_ankle_links=num_ankle_links,
        contact_surface=contact_surface,
        medial_y_min=medial_y_min,
        contact_force_threshold=contact_force_threshold,
        max_touch_force=max_touch_force,
        release_window_steps=release_window_steps,
        speed_lower_ratio=speed_lower_ratio,
        speed_upper_ratio=speed_upper_ratio,
        lateral_speed_std=lateral_speed_std,
        overspeed_std=overspeed_std,
        lateral_corridor_std=lateral_corridor_std,
        max_progress_rate=max_progress_rate,
        active_task_states=active_task_states,
    )["progress_reward"]


def dribbling_gait_foot_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 0.3,
    foot_body_names: list[str] | None = None,
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    active_task_states: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Foot imitation reward active only while the ball is **not** in contact.

    Between touches the policy should follow the reference gait (alternating
  steps) instead of freezing in a kick-ready stance with the dribble foot forward.
    """
    base = motion_relative_foot_position_error_exp(
        env, command_name, std, foot_body_names=foot_body_names
    )
    no_ball = ~_dribbling_sim_contact(env, ball_sensor_name, contact_force_threshold)
    command: MotionCommand = env.command_manager.get_term(command_name)
    active = no_ball & locomotion_task_state_mask(command, active_task_states)
    return base * active.to(torch.float32)


def dribbling_chase_ball_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    min_ball_ahead: float = 0.25,
    max_chase_xy_dist: float = 1.1,
    pelvis_forward_speed_min: float = 0.22,
    forward_speed_scale: float = 0.45,
) -> torch.Tensor:
    """Reward running forward to catch a ball that rolled ahead after a touch.

    Active when there is no ball contact, the ball lies on task +X ahead of the
    pelvis by at least ``min_ball_ahead``, and the robot is within chase range.
    This shapes kick → run → kick rather than shuffling beside the ball.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    no_ball = ~_dribbling_sim_contact(env, ball_sensor_name, contact_force_threshold)
    pelvis_pos_w = command.robot_pelvis_pos_w
    ball_pos_w = soccer_ball.data.root_pos_w

    x_ahead = task_forward_offset(ball_pos_w, pelvis_pos_w)
    dist_xy = torch.norm(ball_pos_w[:, :2] - pelvis_pos_w[:, :2], dim=-1)

    ball_ahead = x_ahead >= min_ball_ahead
    in_range = dist_xy <= max_chase_xy_dist

    pelvis_vel_w = command.robot_anchor_lin_vel_w[:, :3]
    forward = task_forward_speed(pelvis_vel_w)
    dominance = task_velocity_forward_dominance(pelvis_vel_w)
    speed_rew = torch.clamp(
        (forward - pelvis_forward_speed_min) / max(forward_speed_scale, 1e-6),
        min=0.0,
        max=1.0,
    )
    speed_rew = speed_rew * forward_dominance_gate(dominance, 0.45)

    active = no_ball & ball_ahead & in_range
    return speed_rew * active.to(torch.float32)


def dribbling_command_chase_ball_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    contact_force_threshold: float = 1.0,
    min_ball_ahead: float = 0.25,
    max_chase_xy_dist: float = 1.1,
    pelvis_forward_speed_min: float = 0.22,
    forward_speed_scale: float = 0.45,
) -> torch.Tensor:
    """Reward closing on a nearby ball ahead along the active command direction."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    direction_xy, _ = _command_direction_xy(command)
    offset_xy = soccer_ball.data.root_pos_w[:, :2] - command.robot_pelvis_pos_w[:, :2]
    ball_ahead, _ = _command_frame_components(offset_xy, direction_xy)
    dist_xy = torch.norm(offset_xy, dim=-1)
    pelvis_forward, _ = _command_frame_components(command.robot_anchor_lin_vel_w[:, :2], direction_xy)

    speed_reward = torch.clamp(
        (pelvis_forward - pelvis_forward_speed_min) / max(forward_speed_scale, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    no_ball = ~_dribbling_sim_contact(env, ball_sensor_name, contact_force_threshold)
    active = no_ball & (ball_ahead >= min_ball_ahead) & (dist_xy <= max_chase_xy_dist)
    return speed_reward * active.to(torch.float32)


def dribbling_rapid_retouch_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    force_threshold: float = 20.0,
    min_steps_between_touches: int = 22,
    all_body_cfg: SceneEntityCfg | None = None,
    num_ankle_links: int = 2,
    cg_gated: bool = False,
) -> torch.Tensor:
    """Penalty for a **new** legal gentle touch sooner than ``min_steps_between_touches``.

    Encourages kick → chase → kick cadence instead of tapping the ball every step.
    """
    if min_steps_between_touches <= 0:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    command: MotionCommand = env.command_manager.get_term(command_name)
    has_contact, force_mag, closest_idx = _identify_contact_body(
        env, command, ball_sensor_name, all_body_cfg,
    )
    is_ankle = _is_dribble_legal_ankle_contact(closest_idx, num_ankle_links)
    gentle = force_mag <= force_threshold
    touch = has_contact & is_ankle & gentle
    if cg_gated:
        touch = _dribbling_cg_gated_sim_contact(command, touch)

    prev_touch_name = "_dribbling_prev_retouch_touch"
    steps_name = "_dribbling_steps_since_legal_touch"
    prev_touch = getattr(env, prev_touch_name, None)
    if prev_touch is None or prev_touch.shape[0] != env.num_envs:
        prev_touch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    steps_since = getattr(env, steps_name, None)
    if steps_since is None or steps_since.shape[0] != env.num_envs:
        steps_since = torch.full(
            (env.num_envs,),
            fill_value=min_steps_between_touches + 1,
            device=env.device,
            dtype=torch.int32,
        )

    step_buf = getattr(env, "episode_length_buf", None)
    reset_mask = step_buf == 0 if step_buf is not None else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    steps_since = torch.where(reset_mask, torch.full_like(steps_since, min_steps_between_touches + 1), steps_since + 1)

    new_touch = touch & ~prev_touch
    too_soon = new_touch & (steps_since < int(min_steps_between_touches))
    steps_since = torch.where(new_touch, torch.zeros_like(steps_since), steps_since)

    setattr(env, prev_touch_name, touch.detach().clone())
    setattr(env, steps_name, steps_since)
    return too_soon.to(torch.float32)


def dribbling_face_ball(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    min_distance: float = 0.05,
) -> torch.Tensor:
    """Reward ball ahead on task +X **and** pelvis facing task +X.

    Returns the product of (a) cos(angle from task +X to pelvis→ball) and
    (b) cos(pelvis yaw vs task +X), each clamped to [0, 1]. Sideways crab
    dribbling (body or ball mostly on ±Y) scores near zero.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]

    ball_pos_w = soccer_ball.data.root_pos_w[:, :3]
    pelvis_pos_w = command.robot_pelvis_pos_w
    pelvis_quat_w = command.robot_pelvis_quat_w

    dx = task_forward_offset(ball_pos_w, pelvis_pos_w)
    dy = task_lateral_offset(ball_pos_w, pelvis_pos_w)
    dist = torch.norm(torch.stack([dx, dy], dim=-1), dim=-1)
    safe = dist > float(min_distance)
    ball_ahead = torch.where(
        safe,
        (dx / dist.clamp(min=1e-4)).clamp(min=0.0, max=1.0),
        torch.ones_like(dist),
    )
    pelvis_forward = task_pelvis_heading_cos_world_x(pelvis_quat_w).clamp(min=0.0, max=1.0)
    return ball_ahead * pelvis_forward


def dribbling_pelvis_local_face_ball(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    min_distance: float = 0.05,
) -> torch.Tensor:
    """Reward a ball ahead of the current pelvis, independent of world yaw."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    offset_b = _pelvis_yaw_local_vector(
        command, soccer_ball.data.root_pos_w[:, :3] - command.robot_pelvis_pos_w
    )
    distance = torch.norm(offset_b[:, :2], dim=-1)
    return torch.where(
        distance > float(min_distance),
        (offset_b[:, 0] / distance.clamp(min=1.0e-4)).clamp(min=0.0, max=1.0),
        torch.ones_like(distance),
    )


def dribbling_ball_pelvis_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 0.25,
) -> torch.Tensor:
    """Dense object-tracking reward: ball XY in the pelvis-yaw frame vs the reference.

    The reference ball is expressed in the *reference* pelvis-yaw frame at the
    current clip frame; the actual ball in the robot's own pelvis-yaw frame.
    The comparison is body-relative, so it cannot accumulate world-frame drift,
    needs no contact event, foot or surface label, and keeps the ball inside the
    pelvis-relative range the actor's ball observation was trained on.  With a
    broad ``std`` it only asks for "the ball roughly where the human had it".
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    ball = env.scene["soccer_ball"]
    motion_idx = command.motion_idx.to(torch.long)
    last = command.motion.ball_pos_w.shape[1] - 1
    frame = command.time_steps.to(torch.long).clamp(min=0, max=last)
    anchor = command.motion_anchor_body_index
    ref_ball = command.motion.ball_pos_w[motion_idx, frame]
    ref_pelvis = command.motion.body_pos_w[motion_idx, frame, anchor]
    ref_pelvis_quat = command.motion.body_quat_w[motion_idx, frame, anchor]
    ref_rel = quat_apply_inverse(yaw_quat(ref_pelvis_quat), ref_ball - ref_pelvis)
    actual_rel = quat_apply_inverse(
        yaw_quat(command.robot_anchor_quat_w), ball.data.root_pos_w - command.robot_anchor_pos_w
    )
    error_sq = torch.sum(torch.square(actual_rel[:, :2] - ref_rel[:, :2]), dim=-1)
    return torch.exp(-error_sq / float(std) ** 2)
