from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms, quat_apply, quat_inv, yaw_quat

from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import (
    TASK_STATE_DRIBBLE,
    MotionCommand,
)
from soccer.tasks.tracking.mdp.task_frame import mimic_anchor_yaw_delta_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def effective_joint_action(env: ManagerBasedEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """Return the normalized joint command that is actually sent by an action term."""
    action_term = env.action_manager.get_term(action_name)
    effective = getattr(action_term, "effective_raw_actions", None)
    if effective is not None:
        return effective
    return action_term.raw_actions


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)

def motion_anchor_ang_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.anchor_ang_vel_w.view(env.num_envs, -1)


def ball_pos_pelvis_heading(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    """Return the measured ball centre in the current pelvis-yaw frame.

    This is the deployable Stage-1/Stage-2 ball channel.  It deliberately uses
    the *actual* scene ball and the actual pelvis pose; it is not a command
    target, an offline contact anchor, or a simulator velocity signal.  Roll
    and pitch are removed so the convention matches a gravity-aligned planar
    control frame.  A body-mounted measurement needs an additional leveling
    transform before it can use this representation.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = getattr(command, "soccer_ball", None)
    if soccer_ball is None:
        soccer_ball = env.scene["soccer_ball"]
    ball_pos_w = soccer_ball.data.root_pos_w[:, :3]
    pelvis_pos_w = command.robot_anchor_pos_w[:, :3]
    pelvis_yaw_q = yaw_quat(command.robot_anchor_quat_w)
    return quat_apply(quat_inv(pelvis_yaw_q), ball_pos_w - pelvis_pos_w)


def ball_pos_pelvis_full(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    """Return the measured ball centre in the current full pelvis frame.

    Unlike :func:`ball_pos_pelvis_heading`, this observation retains pelvis
    roll and pitch.  That matches the historical Flat body-frame contract and
    the direct deployment transform from a body-mounted ball measurement.  It
    still reads the actual scene ball: no command target, annotated contact
    node, ball velocity, or simulator-only state is exposed to the actor.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = getattr(command, "soccer_ball", None)
    if soccer_ball is None:
        soccer_ball = env.scene["soccer_ball"]
    ball_pos_w = soccer_ball.data.root_pos_w[:, :3]
    # Use the explicit pelvis state rather than relying on the configurable
    # imitation anchor continuing to be named ``pelvis`` in every future task.
    pelvis_pos_w = command.robot_pelvis_pos_w[:, :3]
    pelvis_quat_w = command.robot_pelvis_quat_w
    return quat_apply(quat_inv(pelvis_quat_w), ball_pos_w - pelvis_pos_w)


def _motion_anchor_yaw_delta_quat_obs(command: MotionCommand) -> torch.Tensor:
    """Yaw delta for locomotion commands (matches mimic vel reward terms)."""
    return mimic_anchor_yaw_delta_quat(
        command.anchor_quat_w,
        command.robot_anchor_quat_w,
        align_task_frame=bool(getattr(command.cfg, "mimic_align_task_frame", False)),
    )


def motion_anchor_lin_vel_command(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Locomotion linear-velocity command exposed to the policy (reference or manual).

    Task frame: +X forward, +Y lateral, +Z up (world-parallel on flat ground).
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    if hasattr(command, "locomotion_lin_vel_command_w"):
        return command.locomotion_lin_vel_command_w().view(env.num_envs, -1)
    delta_ori_w = _motion_anchor_yaw_delta_quat_obs(command)
    ref_vel = quat_apply(delta_ori_w, command.anchor_lin_vel_w)
    return ref_vel.view(env.num_envs, -1)


def motion_anchor_ang_vel_command(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Locomotion angular-velocity command (reference or manual), rad/s in task frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    if hasattr(command, "locomotion_ang_vel_command_w"):
        return command.locomotion_ang_vel_command_w().view(env.num_envs, -1)
    delta_ori_w = _motion_anchor_yaw_delta_quat_obs(command)
    ref_ang = quat_apply(delta_ori_w, command.anchor_ang_vel_w)
    return ref_ang.view(env.num_envs, -1)


def motion_locomotion_polar_command(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Active locomotion command as ``[speed, cos(heading), sin(heading)]`` (task +X heading).

    Speed in m/s; heading radians from +X (0=forward, +pi/2=+Y). Independent of demo root vel
    when ``locomotion_command_mode`` is ``resampled`` or ``manual``.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    if getattr(command, "locomotion_command_mode", "reference") in {"manual", "resampled"}:
        speed = command.locomotion_cmd_speed
        heading = command.locomotion_cmd_heading
    else:
        lin = (
            command.locomotion_lin_vel_command_w()
            if hasattr(command, "locomotion_lin_vel_command_w")
            else command.anchor_lin_vel_w
        )
        speed = torch.norm(lin[:, :2], dim=-1)
        heading = torch.atan2(lin[:, 1], lin[:, 0])
    return torch.stack([speed, torch.cos(heading), torch.sin(heading)], dim=-1)


def motion_locomotion_twist_command_local(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Current-pelvis local locomotion command ``[vx, vy, wz]``.

    This is the deployment command contract: forward/lateral linear velocity
    and yaw rate, all relative to the robot's *current* pelvis yaw.  It does
    not encode an absolute world heading or accumulated turn angle.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    if hasattr(command, "locomotion_twist_command_b"):
        return command.locomotion_twist_command_b().view(env.num_envs, -1)
    raise RuntimeError(f"motion command '{command_name}' does not expose a local twist command")


def motion_locomotion_ang_vel_command_local(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Angular part of the local twist, padded as an angular-velocity vector.

    Kept at three dimensions so the unified 163-D observation layout remains
    stable while the legacy raw world-frame reference angular velocity is not
    leaked into the local task policy.
    """
    twist = motion_locomotion_twist_command_local(env, command_name)
    zeros = torch.zeros_like(twist[:, 2])
    return torch.stack((zeros, zeros, twist[:, 2]), dim=-1)


def robot_anchor_lin_vel_local(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Simulator-only anchor linear velocity in the current pelvis yaw frame.

    This term is intended for the critic.  The actor must infer motion from
    deployable proprioception and its command, rather than receiving this
    privileged ground-truth linear velocity.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    return quat_apply(quat_inv(yaw_quat(command.robot_anchor_quat_w)), command.robot_anchor_lin_vel_w)


def motion_locomotion_task_state(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """One-hot IDLE/DRIBBLE/STOP task input, ordered ``[idle, dribble, stop]``.

    It intentionally remains separate from the polar velocity input: zero
    speed alone is ambiguous between waiting for a start request and settling
    the ball after a completed run.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    state = getattr(command, "locomotion_task_state", None)
    if not isinstance(state, torch.Tensor):
        state = torch.full((env.num_envs,), TASK_STATE_DRIBBLE, device=env.device, dtype=torch.long)
    return torch.nn.functional.one_hot(state.to(torch.long), num_classes=3).to(torch.float32)


def _get_motion_command(env: ManagerBasedEnv, command_name: str) -> MotionCommand:
    command: MotionCommand | None = env.command_manager.get_term(command_name)
    if command is None:
        raise RuntimeError(f"motion command '{command_name}' not found in env.command_manager")
    if not hasattr(command, "target_point_pos"):
        raise RuntimeError(f"motion command '{command_name}' lacks target_point_pos attribute")
    return command


def get_target_point_world(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    command = _get_motion_command(env, command_name)
    target_local = command.target_point_pos
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is not None:
        return target_local + env_origins
    return target_local


def get_target_point_base(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    command = _get_motion_command(env, command_name)
    target_world = get_target_point_world(env, command_name)
    # delta = target_world - command.robot_anchor_pos_w
    delta = target_world - command.robot_pelvis_pos_w
    return quat_apply(quat_inv(command.robot_pelvis_quat_w), delta)


def _positional_encoding(vec: torch.Tensor, num_freqs: int = 6) -> torch.Tensor:
    """Apply sinusoidal positional encoding to a target tensor of shape (E, 3).

    The encoding follows Transformer-style frequencies: for each coordinate x,
    compute sin(2^k*pi*x) and cos(2^k*pi*x) for k=0..num_freqs-1, then
    concatenate with the original coordinates.
    """
    if num_freqs <= 0:
        return vec.view(vec.shape[0], -1)

    device = vec.device
    dtype = vec.dtype
    # freqs: [num_freqs]
    freqs = (2.0 ** torch.arange(num_freqs, device=device, dtype=dtype)) * math.pi
    # vec: [E, 3] -> vec_exp: [E, 3, num_freqs]
    vec_exp = vec.unsqueeze(-1) * freqs
    sin = torch.sin(vec_exp)
    cos = torch.cos(vec_exp)
    # sin_cos: [E, 3, 2*num_freqs] -> flatten per-sample
    sin_cos = torch.cat([sin, cos], dim=-1).view(vec.shape[0], -1)
    # Concatenate original coordinates in front.
    return torch.cat([vec.view(vec.shape[0], -1), sin_cos], dim=-1)


def target_point_pos_first_frame(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    cache_name = f"_{command_name}_target_point_cache"
    target_local = get_target_point_base(env, command_name)

    cache = getattr(env, cache_name, None)
    if cache is None or cache.shape[0] != env.num_envs:
        cache = target_local.clone()
        setattr(env, cache_name, cache)

    step_buf = getattr(env, "episode_length_buf", None)
    if step_buf is None:
        raise AttributeError("ManagerBasedEnv missing episode_length_buf required for target point caching")

    first_step_mask = (step_buf == 0)
    if torch.any(first_step_mask):
        cache = getattr(env, cache_name)
        # Only refresh the cache when an environment just reset so the policy keeps the first-frame cue.
        cache[first_step_mask] = target_local[first_step_mask]
        setattr(env, cache_name, cache)
    # Return cached target vector.
    return getattr(env, cache_name)
    return _positional_encoding(getattr(env, cache_name), num_freqs=6)


def constant_target_point_pos(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    # Constant observation path keeps the same representation as policy inputs.
    base = get_target_point_base(env, command_name)
    return base
    return _positional_encoding(base, num_freqs=6)


def blind_zone_target_point_pos(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    """Return target point in robot base frame with blind-zone simulation.
    
    If robot-ball (x, y) distance is outside [blind_distance_min, blind_distance_max],
    return the last visible position to emulate limited visibility.
    Thresholds are resampled from MotionCommandCfg ranges at each resample.
    """
    command = _get_motion_command(env, command_name)
    
    # Current target in robot base frame.
    target_base = get_target_point_base(env, command_name)
    
    # Compute robot-target (x, y) distance in world coordinates.
    target_world = get_target_point_world(env, command_name)
    robot_pos = command.robot_pelvis_pos_w
    # Horizontal distance only.
    distance_xy = torch.norm(target_world[:, :2] - robot_pos[:, :2], dim=-1)
    
    # Visible only when distance is within [min, max].
    in_visible_range = (distance_xy >= command.blind_distance_min) & (distance_xy <= command.blind_distance_max)
    
    # Update last visible target for visible environments.
    if torch.any(in_visible_range):
        command.last_visible_target_point_base[in_visible_range] = target_base[in_visible_range]
        command.is_in_blind_zone[in_visible_range] = False
    
    # Mark blind-zone environments.
    command.is_in_blind_zone[~in_visible_range] = True
    
    # Return last visible position in blind zone, otherwise current target.
    result = torch.where(
        command.is_in_blind_zone.unsqueeze(-1),
        command.last_visible_target_point_base,
        target_base
    )
    # print("blind zone target point:", command.blind_distance_min, command.blind_distance_max, result)
    return result


def target_destination_pos_local(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    if not hasattr(command, "target_destination_pos"):
        raise RuntimeError(f"motion command '{command_name}' lacks target_destination_pos attribute")
    # target_destination_pos is local to env origin; convert to world before subtracting robot pose.
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is not None:
        target_world = command.target_destination_pos + env_origins
    else:
        target_world = command.target_destination_pos

    delta = target_world - command.robot_pelvis_pos_w
    # print("position:", quat_apply(quat_inv(command.robot_pelvis_quat_w), delta))
    return quat_apply(quat_inv(command.robot_pelvis_quat_w), delta)


def target_destination_pos_local_first_frame(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    cache_name = f"_{command_name}_target_destination_local_cache"
    target_local = target_destination_pos_local(env, command_name)

    cache = getattr(env, cache_name, None)
    if cache is None or cache.shape[0] != env.num_envs:
        cache = target_local.clone()
        setattr(env, cache_name, cache)

    step_buf = getattr(env, "episode_length_buf", None)
    if step_buf is None:
        raise AttributeError("ManagerBasedEnv missing episode_length_buf required for target destination caching")

    first_step_mask = (step_buf == 0)
    if torch.any(first_step_mask):
        cache = getattr(env, cache_name)
        # Only refresh the cache when an environment just reset so the policy keeps the first-frame cue.
        cache[first_step_mask] = target_local[first_step_mask]
        setattr(env, cache_name, cache)
    # print("cache:", getattr(env, cache_name))
    return getattr(env, cache_name)
    # Positional encoding path is intentionally disabled here.
    return _positional_encoding(getattr(env, cache_name), num_freqs=6)
    


def foot_target_point_distance(env: ManagerBasedEnv, robot_cfg: SceneEntityCfg, command_name: str = "motion",) -> torch.Tensor:
    command = _get_motion_command(env, command_name)
    robot = env.scene[robot_cfg.name]
    foot_pos = robot.data.body_pos_w[:, robot_cfg.body_ids]
    target_world = get_target_point_world(env, command_name)
    diff = foot_pos - target_world.unsqueeze(1)
    dist = torch.linalg.norm(diff, dim=-1)
    return dist.view(env.num_envs, -1)
