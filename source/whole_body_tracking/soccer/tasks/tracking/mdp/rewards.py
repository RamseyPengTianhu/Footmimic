from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude, quat_apply, quat_inv, quat_apply_inverse, yaw_quat

from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import MotionCommand
from soccer.tasks.tracking.mdp.observations import get_target_point_world
from soccer.tasks.tracking.mdp.kick_detection import KickContactTracker


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def _map_names_to_indices(source_names: list[str], target_names: list[str]) -> list[int]:
    target_list = list(target_names)
    name_to_index = {name: idx for idx, name in enumerate(target_list)}
    indices: list[int] = []
    # Iterate all source names to map.
    for name in source_names:
        # Prefer exact matching for deterministic mapping.
        if name in name_to_index:
            indices.append(name_to_index[name])
            continue
        # If exact matching fails, attempt unique suffix matching.
        suffix_matches = [idx for idx, candidate in enumerate(target_list) if candidate.endswith(name)]
        # Accept only unique suffix matches to avoid ambiguity.
        if len(suffix_matches) == 1:
            indices.append(suffix_matches[0])
    return indices


def action_rate_l2_clip(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    reward = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    return reward.clamp(max=100.0)


def waist_action_rate_l2_clip(env: ManagerBasedRLEnv, waist_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    if waist_cfg is None:
        raise ValueError("waist_cfg cannot be None")
    robot = env.scene[waist_cfg.name]
    idx = torch.as_tensor(robot.find_joints(waist_cfg.joint_names, preserve_order=True)[0], device=env.device)
    return torch.sum(torch.square(env.action_manager.action[:, idx] - env.action_manager.prev_action[:, idx]), dim=1).clamp(max=100.0)


def _get_kick_tracker(command: MotionCommand) -> KickContactTracker:
    tracker = getattr(command, "kick_contact_tracker", None)
    if tracker is None:
        raise RuntimeError("MotionCommand is missing kick_contact_tracker; ensure command setup is up to date.")
    return tracker


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)

def motion_relative_foot_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, foot_body_names: list[str] | None = None
) -> torch.Tensor:
    if foot_body_names is None:
        foot_body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, foot_body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward

def foot_distance(env: ManagerBasedRLEnv, threshold: float, std: float, foot_cfg: SceneEntityCfg | None = None,) -> torch.Tensor:
    """Encourage a minimum separation between both feet to avoid crossing/overlap."""
    if foot_cfg is None:
        raise ValueError("foot_distance requires foot_cfg to identify feet.")
    robot = env.scene[foot_cfg.name]
    left_foot_idx = foot_cfg.body_ids[0]
    right_foot_idx = foot_cfg.body_ids[1]
    left_foot_pos = robot.data.body_pos_w[:, left_foot_idx]  # [num_envs, 3]
    right_foot_pos = robot.data.body_pos_w[:, right_foot_idx]  # [num_envs, 3]
    distance = torch.norm(left_foot_pos - right_foot_pos, dim=1)  # [num_envs]
    reward = torch.where(
        distance >= threshold,
        torch.tensor(1., device=distance.device),
        1.0 * torch.exp(-((distance / threshold - 1)**2) / (std ** 2))
    )
    return reward


def feet_slip_penalty(env: ManagerBasedRLEnv, foot_cfg: SceneEntityCfg, slip_force_threshold: float,) -> torch.Tensor:
    """Penalize foot linear velocity when the foot is in contact.

    A contact is detected when the contact force sensor reports an upward (positive Z)
    force larger than ``slip_force_threshold`` on the foot bodies provided by
    ``foot_cfg``. The penalty mirrors the Isaac Gym style reward, summing the squared
    linear velocity of feet that are currently in contact.
    """

    if foot_cfg is None:
        raise ValueError("foot_cfg cannot be None for _reward_feet_slip_penalty")
    contact_sensor = None
    sensors = getattr(env.scene, "sensors", None)
    if sensors is not None:
        try:
            contact_sensor = sensors["contact_forces"] if isinstance(sensors, dict) else getattr(sensors, "contact_forces", None)
        except (KeyError, AttributeError, TypeError):
            contact_sensor = None
    if contact_sensor is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    device = env.device
    num_envs = env.num_envs
    forces = None
    forces_data = contact_sensor.data
    if hasattr(forces_data, "net_forces_w_history"):
        forces_hist = forces_data.net_forces_w_history
        if forces_hist.numel() > 0:
            forces = forces_hist.to(device)
            if forces.ndim >= 4:
                forces = forces.max(dim=1).values
    if forces is None:
        if hasattr(forces_data, "net_forces_w"):
            forces = forces_data.net_forces_w
            if forces is not None and forces.numel() > 0:
                forces = forces.to(device)
            else:
                return torch.zeros(num_envs, device=device, dtype=torch.float32)
        else:
            return torch.zeros(num_envs, device=device, dtype=torch.float32)
    if forces.ndim < 3:
        return torch.zeros(num_envs, device=device, dtype=torch.float32)

    robot = env.scene[foot_cfg.name]

    foot_indices_key = tuple(foot_cfg.body_names)
    if not hasattr(contact_sensor, '_foot_indices_cache'):
        contact_sensor._foot_indices_cache = {}
    if foot_indices_key not in contact_sensor._foot_indices_cache:
        foot_sensor_indices = contact_sensor.find_bodies(foot_cfg.body_names, preserve_order=True)[0]
        contact_sensor._foot_indices_cache[foot_indices_key] = torch.as_tensor(
            foot_sensor_indices, device=device, dtype=torch.long
        )
    foot_indices = contact_sensor._foot_indices_cache[foot_indices_key]

    max_foot_idx = int(foot_indices.max()) if len(foot_indices) > 0 else -1
    if forces.shape[1] <= max_foot_idx:
        return torch.zeros(num_envs, device=device, dtype=torch.float32)
    vertical_forces = forces[:, foot_indices, 2]
    contact_mask = vertical_forces > slip_force_threshold
    foot_vel_w = robot.data.body_lin_vel_w[:, foot_indices]
    penalize = torch.where(
        contact_mask.unsqueeze(-1), 
        torch.square(foot_vel_w), 
        torch.zeros_like(foot_vel_w)
    )
    if penalize.numel() > 10000:  # Heuristic threshold; tune if needed.
        return penalize.reshape(num_envs, -1).sum(dim=1)
    else:
        return torch.sum(penalize, dim=(1, 2))
    

def target_point_proximity(env: ManagerBasedRLEnv, std: float, command_name: str = "motion",) -> torch.Tensor:
    """Reward proximity to the target point (ball) and freeze at first kick contact."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    tracker = _get_kick_tracker(command)
    
    # Compute current proximity reward.
    base_xy = command.robot_anchor_pos_w[..., :2]
    target = get_target_point_world(env, command_name).to(device=base_xy.device, dtype=base_xy.dtype)
    diff_xy = base_xy - target[..., :2]
    error = torch.sum(diff_xy * diff_xy, dim=-1)
    proximity_reward = torch.exp(-error / std**2)

    # Query kick-contact status.
    contact_awarded = tracker.get_contact_awarded()
    frozen_reward = tracker.get_frozen_proximity_reward()
    
    # Freeze reward for environments that just kicked this step.
    new_kick_mask = contact_awarded & (frozen_reward == 0.0)
    if torch.any(new_kick_mask):
        new_kick_ids = torch.nonzero(new_kick_mask, as_tuple=False).squeeze(-1)
        tracker.freeze_proximity_reward(new_kick_ids, proximity_reward[new_kick_ids])
        frozen_reward = tracker.get_frozen_proximity_reward()
        
    return torch.where(contact_awarded, frozen_reward, proximity_reward)

def target_point_relative_proximity(env: ManagerBasedRLEnv, std: float, command_name: str = "motion",) -> torch.Tensor:
    """Reward proximity to the expected reference-relative ball position.
    
    Instead of pulling the robot's pelvis to the absolute ball coordinates (which contradicts the reference motion),
    this reward pulls the robot to a position where the ball is in the correct relative position (and yaw angle) 
    as defined by the reference motion and the initial ball placement offset.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    tracker = _get_kick_tracker(command)
    
    # 1. 真实球的位置 (World)
    ball_pos_w = get_target_point_world(env, command_name).to(device=command.robot_anchor_pos_w.device, dtype=command.robot_anchor_pos_w.dtype)
    
    # 2. 参考动作中的球位置 (即带着 curve_offset 的目标点)
    ball_ref_pos_w = command.initial_target_point_pos
    
    # 3. 提取 Yaw-only Quaternion，防止骨盆俯仰侧倾污染水平面计算
    ref_yaw_quat = yaw_quat(command.anchor_quat_w)
    robot_yaw_quat = yaw_quat(command.robot_anchor_quat_w)
    
    # 4. 计算在参考动作当前帧下，球相对于骨盆的局部世界偏移
    diff_ref_world = ball_ref_pos_w - command.anchor_pos_w
    diff_ref_world[..., 2] = 0.0 # 强制只关心水平面
    
    # 将世界偏移转为参考骨盆的局部偏移 (Local Offset)
    rel_ball_ref_local = quat_apply_inverse(ref_yaw_quat, diff_ref_world)
    
    # 5. 将局部偏移转换到机器人的当前朝向下，得到机器人应该在的预期球位置
    rel_ball_cur_world = quat_apply(robot_yaw_quat, rel_ball_ref_local)
    expected_ball_world = command.robot_anchor_pos_w + rel_ball_cur_world
    
    # 6. 计算当前真实的球与预期球位置的误差
    diff_xy = expected_ball_world[..., :2] - ball_pos_w[..., :2]
    error = torch.sum(diff_xy * diff_xy, dim=-1)
    proximity_reward = torch.exp(-error / std**2)
    
    # Query kick-contact status.
    contact_awarded = tracker.get_contact_awarded()
    frozen_reward = tracker.get_frozen_proximity_reward()
    
    # Freeze reward for environments that just kicked this step.
    new_kick_mask = contact_awarded & (frozen_reward == 0.0)
    if torch.any(new_kick_mask):
        new_kick_ids = torch.nonzero(new_kick_mask, as_tuple=False).squeeze(-1)
        tracker.freeze_proximity_reward(new_kick_ids, proximity_reward[new_kick_ids])
        frozen_reward = tracker.get_frozen_proximity_reward()
        
    return torch.where(contact_awarded, frozen_reward, proximity_reward)



def target_point_contact(env: ManagerBasedRLEnv, 
        horizontal_force_threshold: float = 0.0,
        command_name: str = "motion",
        ball_sensor_name: str = "soccer_ball_contact",
        foot_cfg: SceneEntityCfg | None = None,
    ) -> torch.Tensor:
    """One-shot reward for contacting the ball at first valid touch."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if not torch.any(event.new_contact):
        return reward
    # print(event.new_contact.to(reward.dtype))
    reward_scale = torch.zeros_like(reward)
    correct_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    if foot_cfg is not None:
        foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
        if foot_info.env_ids.numel() > 0:
            valid_expectation = foot_info.expected >= 0
            correct = (foot_info.sides == foot_info.expected) & valid_expectation
            reward_scale[foot_info.env_ids] = correct.to(reward_scale.dtype)
            correct_mask[foot_info.env_ids] = correct

    tracker.record_expected_success(event.new_contact, correct_mask)
    # print("contact", event.new_contact.to(reward.dtype) * reward_scale)
    return event.new_contact.to(reward.dtype) * reward_scale

def sideways_kick(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 0.0,
    foot_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Single-shot reward encouraging foot swing along the expected lateral axis.
    Left kick expects foot velocity along local -Y; right kick expects local +Y.
    """
    if foot_cfg is None:
        raise ValueError("sideways_kick_reward requires foot_cfg to identify kicking feet.")

    command: MotionCommand = env.command_manager.get_term(command_name)
    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if not torch.any(event.new_contact):
        return reward

    foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
    if foot_info.env_ids.numel() == 0:
        return reward

    robot = command.robot
    foot_vel_w = robot.data.body_lin_vel_w[foot_info.env_ids, foot_info.body_indices]
    foot_quat_w = robot.data.body_quat_w[foot_info.env_ids, foot_info.body_indices]

    vel_local = quat_apply(quat_inv(foot_quat_w), foot_vel_w)
    vel_norm = torch.norm(vel_local, dim=-1)

    expected_leg = foot_info.expected.to(device=env.device, dtype=torch.int8)
    desired_sign = torch.zeros(expected_leg.shape, device=env.device, dtype=torch.float32)
    desired_sign = torch.where(expected_leg == 0, torch.full_like(desired_sign, -1.0), desired_sign)
    desired_sign = torch.where(expected_leg == 1, torch.full_like(desired_sign, 1.0), desired_sign)

    directional_component = vel_local[:, 1] * desired_sign
    axis_component = torch.clamp(directional_component, min=0.0)

    alignment = torch.where(vel_norm > 1e-6, axis_component / vel_norm, torch.zeros_like(vel_norm))
    reward[foot_info.env_ids] = alignment.to(reward.dtype)

    # Reward only when expected leg is valid and contact leg matches expectation.
    valid_expectation = expected_leg >= 0
    correct_foot = (foot_info.sides == foot_info.expected) & valid_expectation
    wrong_mask = ~correct_foot
    if torch.any(wrong_mask):
        reward[foot_info.env_ids[wrong_mask]] = 0.0
    # print("sideways_kick reward:", reward)
    return reward



def ball_velocity_direction_alignment(
    env: ManagerBasedRLEnv, command_name: str, std: float, velocity_threshold: float = 0.1,
    horizontal_force_threshold: float = 0.0,
    ball_sensor_name: str = "soccer_ball_contact",
    foot_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward alignment between ball velocity direction and pre-kick target-to-destination direction.

    Active only for a short window after contact with the expected foot.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    vel = soccer_ball.data.root_lin_vel_w  # [num_envs, 3]
    vel_xy = vel[:, :2]  # x-y plane projection
    vel_xy_norm = torch.norm(vel_xy, dim=-1, keepdim=True)
    vel_norm = torch.norm(vel, dim=-1, keepdim=True)
    
    # Direction vector from pre-kick target point (ball) to destination.
    direction = command.target_destination_pos - command.initial_target_point_pos  # [num_envs, 3]
    direction_xy = direction[:, :2]
    dir_norm = torch.norm(direction_xy, dim=-1, keepdim=True)

    valid_mask = (vel_norm.squeeze(-1) > velocity_threshold) & (vel_xy_norm.squeeze(-1) > 1e-6) & (
        dir_norm.squeeze(-1) > 1e-6
    )

    # Track average angle based on initial direction vectors.
    avg_angle = torch.tensor(0.0, device=env.device, dtype=torch.float32)
    if torch.any(valid_mask):
        dir_unit_valid = direction_xy[valid_mask] / dir_norm[valid_mask]
        vel_unit_valid = vel_xy[valid_mask] / vel_xy_norm[valid_mask]
        cos_theta_valid = torch.sum(vel_unit_valid * dir_unit_valid, dim=-1).clamp(-1.0, 1.0)
        theta_valid = torch.acos(cos_theta_valid)
        avg_angle = theta_valid.mean()
    if hasattr(command, "metrics"):
        command.metrics["ball_velocity_dir_alignment_angle"] = torch.full(
            (env.num_envs,), avg_angle.item(), device=env.device, dtype=torch.float32
        )
    
    # Reward window.
    timer_name = f"_{command_name}_dir_align_timer"

    timer = getattr(env, timer_name, None)
    if timer is None or timer.shape[0] != env.num_envs:
        timer = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    else:
        timer = timer.to(device=env.device, dtype=torch.int32)

    # Trigger reward window on expected-foot contact.
    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)
    
    if torch.any(event.new_contact) and foot_cfg is not None:
        foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
        if foot_info.env_ids.numel() > 0:
            valid_expectation = foot_info.expected >= 0
            correct_foot = (foot_info.sides == foot_info.expected) & valid_expectation
            # Open the window only for correct-foot contacts.
            correct_env_ids = foot_info.env_ids[correct_foot]
            if correct_env_ids.numel() > 0:
                timer[correct_env_ids] = 5

    # Validate speeds in active_mask to avoid division by zero.
    speed_valid = (vel_xy_norm.squeeze(-1) > 1e-6) & (dir_norm.squeeze(-1) > 1e-6)
    active_mask = (timer > 0) & speed_valid

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if torch.any(active_mask):
        dir_unit = direction_xy[active_mask] / dir_norm[active_mask]
        vel_unit = vel_xy[active_mask] / vel_xy_norm[active_mask]
        cos_theta = torch.sum(vel_unit * dir_unit, dim=-1).clamp(-1.0, 1.0)
        error = torch.acos(cos_theta) ** 2
        reward[active_mask] = torch.exp(-error / (std ** 2))

    # Decrement active timers.
    timer = torch.where(timer > 0, timer - 1, timer)
    setattr(env, timer_name, timer)
    # print("ball_velocity_direction_alignment reward:", timer,reward)
    return reward


def ball_speed_reward(env: ManagerBasedRLEnv, command_name: str, std: float, velocity_threshold: float = 0.1,
    horizontal_force_threshold: float = 0.0,
    ball_sensor_name: str = "soccer_ball_contact",
    foot_cfg: SceneEntityCfg | None = None,
    ) -> torch.Tensor:
    """Reward ball speed within a short window after expected-foot contact."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    soccer_ball = env.scene["soccer_ball"]
    vel = soccer_ball.data.root_lin_vel_w  # [num_envs, 3]
    speed_xy = torch.norm(vel[:, :2], dim=-1)  # x-y plane speed

    timer_name = f"_{command_name}_speed_timer"

    timer = getattr(env, timer_name, None)
    if timer is None or timer.shape[0] != env.num_envs:
        timer = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    else:
        timer = timer.to(device=env.device, dtype=torch.int32)

    # Trigger reward window on expected-foot contact.
    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)
    
    if torch.any(event.new_contact) and foot_cfg is not None:
        foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
        if foot_info.env_ids.numel() > 0:
            valid_expectation = foot_info.expected >= 0
            correct_foot = (foot_info.sides == foot_info.expected) & valid_expectation
            # Open the window only for correct-foot contacts.
            correct_env_ids = foot_info.env_ids[correct_foot]
            if correct_env_ids.numel() > 0:
                timer[correct_env_ids] = 5

    # Validate speed in active_mask to avoid division by zero.
    speed_valid = speed_xy > 1e-6
    active_mask = (timer > 0) & speed_valid

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if torch.any(active_mask):
        reward_active = 1.0 - torch.exp(-(speed_xy[active_mask] ** 2) / (std ** 2))
        reward[active_mask] = reward_active

    # Decrement active timers.
    timer = torch.where(timer > 0, timer - 1, timer)
    setattr(env, timer_name, timer)
    # print("ball_speed_reward:", reward)
    return reward

def ball_z_speed_penalty_reward(env: ManagerBasedRLEnv, command_name: str, std: float, velocity_threshold: float = 0.1,
    ) -> torch.Tensor:
    """Penalize excessive vertical ball speed in a short post-activation window."""
    soccer_ball = env.scene["soccer_ball"]
    vel = soccer_ball.data.root_lin_vel_w  # [num_envs, 3]
    z_speed = vel[:, 2]  # vertical speed
    speed = torch.norm(vel, dim=-1)

    valid_mask = speed > velocity_threshold

    timer_name = f"_{command_name}_z_speed_timer"
    prev_name = f"_{command_name}_z_speed_prev"

    timer = getattr(env, timer_name, None)
    if timer is None or timer.shape[0] != env.num_envs:
        timer = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    else:
        timer = timer.to(device=env.device, dtype=torch.int32)

    prev_valid = getattr(env, prev_name, None)
    if prev_valid is None or prev_valid.shape[0] != env.num_envs:
        prev_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        prev_valid = prev_valid.to(device=env.device, dtype=torch.bool)

    rising_mask = valid_mask & (~prev_valid)
    timer[rising_mask] = 5
    active_mask = timer > 0

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if torch.any(active_mask):
        scale = std if std > 0 else 1.0
        reward[active_mask] = torch.tanh(torch.abs(z_speed[active_mask]) / (scale + 1e-8))

    # Decrement active timers.
    timer = torch.where(timer > 0, timer - 1, timer)
    setattr(env, timer_name, timer)
    setattr(env, prev_name, valid_mask.to(dtype=torch.bool))
    # print("ball_z_speed_penalty_reward:", reward)
    return reward


def pelvis_orientation(env: ManagerBasedRLEnv, command_name: str = "motion") -> torch.Tensor:
    """Penalize pelvis pitch/roll tilt to keep the robot upright."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot = command.robot
    gravity_vec_w = robot.data.GRAVITY_VEC_W
    
    # Project gravity vector to pelvis local frame.
    pelvis_proj_gravity = quat_apply_inverse(command.robot_pelvis_quat_w, gravity_vec_w)
    # print("pelvis_proj_gravity:", gravity_vec_w, pelvis_proj_gravity)
    return torch.sum(torch.square(pelvis_proj_gravity[:, :2]), dim=1)


# ===========================================================================
# Sprint 4: Soft Contact Graph (CG) Rewards
# ===========================================================================
# These rewards implement time-gated logic based on each motion's kick_frame.
#   CG=0: time_steps < kick_frame  (approach / running phase)
#   CG=1: time_steps >= kick_frame (kick window)
# ===========================================================================

def _get_cg_phase(command: MotionCommand, margin: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-env boolean masks for CG=0 (approach) and CG=1 (kick window).

    Args:
        command: The MotionCommand instance.
        margin: Number of frames before kick_frame to start CG=1 transition.
    Returns:
        (is_cg0, is_cg1) boolean tensors of shape (num_envs,).
    """
    kf = command.kick_frame  # (num_envs,) — per-env kick start frame
    t = command.time_steps   # (num_envs,) — current frame
    has_annotation = kf >= 0  # motion has kick_frame label

    # CG=1 starts `margin` frames before kick_frame to allow preparation.
    is_cg1 = has_annotation & (t >= (kf - margin))
    is_cg0 = has_annotation & ~is_cg1

    # If no annotation, default to CG=1 (don't penalise).
    return is_cg0, is_cg1


def early_collision_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 5.0,
    cg_margin: int = 5,
) -> torch.Tensor:
    """Penalise ball contact during CG=0 (approach phase).

    During CG=0 (before kick_frame - margin), any contact with the ball
    yields a per-frame -1.0 penalty.  This teaches the robot to avoid
    accidentally bumping the ball while running.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    is_cg0, _ = _get_cg_phase(command, margin=cg_margin)

    # Detect contact from ball sensor.
    ball_contact: ContactSensor = env.scene[ball_sensor_name]
    # net_forces_w_history shape can be (N, num_bodies, H, 3) or (N, H, 3).
    net_forces = ball_contact.data.net_forces_w_history
    if net_forces.dim() == 4:
        # Sum over bodies, take latest history frame.
        force_vec = net_forces[:, :, 0, :2].sum(dim=1)  # (N, 2)
    else:
        force_vec = net_forces[:, 0, :2]  # (N, 2)
    force_mag = torch.norm(force_vec, dim=-1)  # (N,)
    has_contact = force_mag > horizontal_force_threshold

    # Penalty only during CG=0.
    penalty = torch.zeros(env.num_envs, device=env.device)
    penalty[is_cg0 & has_contact] = 1.0  # weight is negative in config
    return penalty


def time_gated_contact(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 10.0,
    foot_cfg: SceneEntityCfg | None = None,
    cg_margin: int = 5,
) -> torch.Tensor:
    """Same as target_point_contact but ONLY rewards during CG=1 window.

    Contact during CG=0 is completely ignored (no reward).
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    _, is_cg1 = _get_cg_phase(command, margin=cg_margin)
    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if not torch.any(event.new_contact):
        return reward

    reward_scale = torch.zeros_like(reward)
    correct_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    if foot_cfg is not None:
        foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
        if foot_info.env_ids.numel() > 0:
            valid_expectation = foot_info.expected >= 0
            correct = (foot_info.sides == foot_info.expected) & valid_expectation
            reward_scale[foot_info.env_ids] = correct.to(reward_scale.dtype)
            correct_mask[foot_info.env_ids] = correct

    tracker.record_expected_success(event.new_contact, correct_mask)

    # Gate: zero out reward for envs still in CG=0.
    raw_reward = event.new_contact.to(reward.dtype) * reward_scale
    raw_reward[~is_cg1] = 0.0
    return raw_reward


def dynamic_ankle_masking_body_pos(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 0.3,
    body_names: list[str] | None = None,
    kick_foot_name: str = "right_ankle_roll_link",
    kick_foot_cg1_scale: float = 0.3,
    cg_margin: int = 5,
) -> torch.Tensor:
    """Body position tracking with dynamic ankle masking based on CG phase.

    During CG=0: ALL bodies tracked (including kick foot) — stable gait.
    During CG=1: kick foot error scaled by `kick_foot_cg1_scale` — soft guidance
                 for proper kick form while allowing deviation to reach the ball.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    _, is_cg1 = _get_cg_phase(command, margin=cg_margin)

    # Get all body indices for tracking.
    all_indices = _get_body_indexes(command, body_names)

    # Find kick foot index within the body_names list.
    kick_foot_local_idx = None
    if body_names is not None and kick_foot_name in body_names:
        kick_foot_local_idx = body_names.index(kick_foot_name)
    elif body_names is None:
        # body_names is all bodies, find kick foot in cfg.body_names
        if kick_foot_name in command.cfg.body_names:
            kick_foot_local_idx = list(command.cfg.body_names).index(kick_foot_name)

    # Compute full tracking error for all bodies.
    body_pos_relative_w = command.body_pos_relative_w[:, all_indices]
    robot_body_pos_w = command.robot.data.body_pos_w[:, :, :]
    body_cfg_indices = all_indices
    robot_body_selected = robot_body_pos_w[:, body_cfg_indices]

    diff = robot_body_selected - body_pos_relative_w
    per_body_error = torch.sum(diff * diff, dim=-1)  # (num_envs, num_bodies)

    # During CG=1, scale down (not zero out) kick foot error — soft guidance.
    if kick_foot_local_idx is not None:
        if kick_foot_local_idx < len(all_indices):
            cg1_expanded = is_cg1.unsqueeze(-1)  # (num_envs, 1)
            mask = torch.zeros_like(per_body_error, dtype=torch.bool)
            mask[:, kick_foot_local_idx] = True
            # Scale kick foot error by kick_foot_cg1_scale during CG=1.
            scaled = per_body_error * kick_foot_cg1_scale
            per_body_error = torch.where(mask & cg1_expanded, scaled, per_body_error)

    mean_error = per_body_error.mean(dim=-1)
    return torch.exp(-mean_error / (std ** 2))


def ankle_lock_on_contact(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 5.0,
    ankle_cfg: SceneEntityCfg | None = None,
    cg_margin: int = 5,
    lock_margin: int = 2,
) -> torch.Tensor:
    """Penalise ankle joint velocity of the *kicking foot only* near kick_frame.

    Active in a tight window [kick_frame - lock_margin, kick_frame + lock_margin].
    Default lock_margin=2 gives a 5-frame window centered on the kick moment.

    Uses the motion's kick_leg annotation (left=0, right=1) to select
    which ankle joints to lock.

    ankle_cfg.joint_names must be ordered:
        [left_pitch, left_roll, right_pitch, right_roll]
    """
    if ankle_cfg is None:
        raise ValueError("ankle_cfg must be provided with ankle joint names")

    command: MotionCommand = env.command_manager.get_term(command_name)
    kf = command.kick_frame       # (num_envs,)
    kef = command.kick_end_frame   # (num_envs,)
    t = command.time_steps         # (num_envs,)
    has_annotation = kf >= 0

    # Gate: exactly the annotated contact window [kick_frame, kick_end_frame].
    active = has_annotation & (t >= kf)
    has_kef = kef >= 0
    active = active & (~has_kef | (t <= kef))

    # Get all ankle joint velocities: [left_pitch, left_roll, right_pitch, right_roll]
    robot = env.scene[ankle_cfg.name]
    ankle_joint_ids = torch.as_tensor(
        robot.find_joints(ankle_cfg.joint_names, preserve_order=True)[0],
        device=env.device,
    )
    ankle_vel = robot.data.joint_vel[:, ankle_joint_ids]  # (num_envs, 4)

    # Determine kicking leg per env: 0=left, 1=right
    # ankle_cfg joints are ordered [left_pitch, left_roll, right_pitch, right_roll]
    kick_leg = command.motion_kick_leg[command.motion_idx]  # (num_envs,) 0=left, 1=right

    # Build per-env mask: only penalize the kicking side's 2 joints
    # left leg → indices 0,1; right leg → indices 2,3
    num_ankle_joints = ankle_vel.shape[1]  # should be 4
    joint_mask = torch.zeros_like(ankle_vel, dtype=torch.bool)
    is_left = kick_leg == 0
    is_right = kick_leg == 1
    if num_ankle_joints >= 4:
        joint_mask[is_left, 0] = True
        joint_mask[is_left, 1] = True
        joint_mask[is_right, 2] = True
        joint_mask[is_right, 3] = True
    else:
        # Fallback: penalize all if joint count doesn't match expected layout
        joint_mask[:] = True

    # Compute penalty: sum of squared velocities for selected joints only
    masked_vel = torch.where(joint_mask, ankle_vel, torch.zeros_like(ankle_vel))
    penalty = torch.sum(torch.square(masked_vel), dim=1)

    # Zero out for envs not in the active window.
    penalty[~active] = 0.0
    return penalty


# ---------------------------------------------------------------------------
# Post-Strike Stabilization
# ---------------------------------------------------------------------------

def post_strike_stability(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 5.0,
    post_delay: int = 3,
    post_duration: int = 25,
    tilt_std: float = 0.3,
    angvel_std: float = 1.0,
    tilt_weight: float = 0.6,
    angvel_weight: float = 0.4,
) -> torch.Tensor:
    """Reward base stability AFTER successful ball contact.

    This reward is completely safe for kick acquisition because it only
    activates after ``contact_awarded`` is True AND ``post_delay`` frames
    have elapsed. It cannot interfere with the approach, pre-strike, or
    strike phases.

    The reward window is capped at ``post_duration`` frames after the delay
    to prevent unbounded reward accumulation in long episodes.

    Components:
      1. **Tilt**: penalizes base roll/pitch deviation from upright
         (using projected gravity Z component in body frame)
      2. **Angular velocity**: penalizes high roll/pitch angular velocity
         in body frame (yaw rotation is allowed)

    Args:
        post_delay: number of frames to wait after contact before activating.
        post_duration: maximum number of frames the reward is active.
        tilt_std: sigma for tilt Gaussian (smaller = stricter).
        angvel_std: sigma for angular velocity Gaussian (smaller = stricter).
        tilt_weight: relative weight of tilt component [0, 1].
        angvel_weight: relative weight of angular velocity component [0, 1].
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    tracker = _get_kick_tracker(command)
    device = env.device

    # Ensure detection is current
    tracker.detect(command, ball_sensor_name, horizontal_force_threshold)

    contact_awarded = tracker.get_contact_awarded()
    contact_frame = tracker.get_contact_frame()
    t = command.time_steps.float()

    # Gate: only activate in [contact_frame + post_delay, contact_frame + post_delay + post_duration]
    t_since_contact = t - contact_frame
    post_strike = (
        contact_awarded
        & (contact_frame >= 0)
        & (t_since_contact > post_delay)
        & (t_since_contact <= post_delay + post_duration)
    )

    reward = torch.zeros(env.num_envs, device=device)
    if not torch.any(post_strike):
        return reward

    # --- Tilt component: projected gravity Z should be close to -1 (upright) ---
    robot = command.robot
    base_quat = robot.data.root_quat_w  # (num_envs, 4)
    # Project gravity direction into body frame
    # Convention: upright → projected_gravity ≈ [0, 0, -1]
    gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=device).expand(env.num_envs, 3)
    projected_gravity = quat_apply_inverse(base_quat, gravity_vec)
    # tilt_err: 0 = perfectly upright, ~2 = inverted
    tilt_err = 1.0 + projected_gravity[:, 2]  # 0 when upright (-1 + 1 = 0)
    r_tilt = torch.exp(-tilt_err.square() / (tilt_std ** 2))

    # --- Angular velocity component: roll/pitch in body frame ---
    # Rotate world-frame angular velocity to body frame so XY = roll/pitch
    ang_vel_w = robot.data.root_ang_vel_w  # (num_envs, 3)
    ang_vel_b = quat_apply_inverse(base_quat, ang_vel_w)
    # Penalize roll (X) and pitch (Y) angular velocity, allow yaw (Z)
    ang_vel_rp_sq = ang_vel_b[:, 0].square() + ang_vel_b[:, 1].square()
    r_angvel = torch.exp(-ang_vel_rp_sq / (angvel_std ** 2))

    # Combine
    reward[post_strike] = (tilt_weight * r_tilt[post_strike] +
                           angvel_weight * r_angvel[post_strike])
    return reward


# ---------------------------------------------------------------------------
# Support-Foot Placement Prior
# ---------------------------------------------------------------------------

def support_foot_placement(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    support_foot_name: str = "left_ankle_roll_link",
    side_offset: float = 0.20,
    pos_std: float = 0.20,
    yaw_std: float = 0.4,
    yaw_weight: float = 0.5,
    plant_before: int = 20,
    plant_after: int = 5,
    near_ball_dist: float = 1.2,
) -> torch.Tensor:
    """Reward support foot placement parallel to ball and oriented toward target.

    During the plant window, this reward encourages:
      1. **Position**: The support foot should land parallel to (beside) the ball,
         offset laterally by ``side_offset`` meters along the perpendicular of
         the kick direction (ball → target destination).
      2. **Yaw**: The support foot forward axis should point toward the target
         destination, matching the intended kick direction.

    The kick direction is computed as ``normalize(target_destination - ball_pos)``.
    The side direction is the perpendicular to this, pointing toward the
    support foot side (left for right-footed kicks, right for left-footed kicks).

    Gate: Active in plant window [kick_frame - plant_before, kick_frame + plant_after]
    AND only when the pelvis is within ``near_ball_dist`` of the ball.
    At 50 Hz, plant_before=20 ≈ 0.4s before contact — enough time to guide the
    last stride before the kick.

    Args:
        support_foot_name: Body link name of the support foot.
        side_offset: Lateral distance from ball center to desired support foot (meters).
        pos_std: Gaussian std for positional reward (meters).
        yaw_std: Gaussian std for yaw alignment reward (radians).
        yaw_weight: Blending weight for yaw component (position weight is 1.0).
        plant_before: Frames before kick_frame to start plant window.
        plant_after: Frames after kick_frame to end plant window.
        near_ball_dist: Max pelvis-to-ball distance to activate (meters).
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    device = env.device

    # --- Plant window gating ---
    kf = command.kick_frame   # (N,) per-env kick frame, -1 if unannotated
    t = command.time_steps    # (N,) current frame
    has_annotation = kf >= 0
    in_plant_window = has_annotation & (t >= (kf - plant_before)) & (t <= (kf + plant_after))

    # --- Distance gating: only activate when pelvis is near ball ---
    ball_pos_w = get_target_point_world(env, command_name).to(device=device)
    ball_xy = ball_pos_w[:, :2]  # (N, 2)
    pelvis_xy = command.robot_anchor_pos_w[:, :2]  # (N, 2)
    pelvis_ball_dist = torch.norm(pelvis_xy - ball_xy, dim=-1)  # (N,)
    near_ball = pelvis_ball_dist < near_ball_dist

    # Combined gate
    active = in_plant_window & near_ball

    # --- Target destination (world) ---
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is not None:
        dest_w = command.target_destination_pos + env_origins
    else:
        dest_w = command.target_destination_pos
    dest_xy = dest_w[:, :2]  # (N, 2)

    # --- Kick direction: ball → target destination ---
    kick_dir = dest_xy - ball_xy  # (N, 2)
    kick_dir_norm = torch.norm(kick_dir, dim=-1, keepdim=True).clamp(min=1e-6)
    kick_dir = kick_dir / kick_dir_norm  # (N, 2) normalized

    # --- Side direction (perpendicular to kick_dir) ---
    # Rotate kick_dir 90° CCW → side points to the left of kick direction
    side_dir = torch.stack([-kick_dir[:, 1], kick_dir[:, 0]], dim=-1)  # (N, 2)

    # Determine side sign based on kick leg annotation.
    # If kick leg is right (1), support foot is left → side_sign = +1 (left of kick_dir)
    # If kick leg is left (0), support foot is right → side_sign = -1 (right of kick_dir)
    kick_leg = command.kick_leg  # (N,) 0=left, 1=right, -1=unknown
    side_sign = torch.where(kick_leg == 0, -1.0, 1.0).to(device=device)  # default +1 for right kick / unknown

    # --- Desired support foot position: parallel to ball, laterally offset ---
    desired_xy = ball_xy + side_sign.unsqueeze(-1) * side_offset * side_dir  # (N, 2)

    # --- Actual support foot position (world) ---
    robot = command.robot
    support_body_idx = robot.body_names.index(support_foot_name)
    support_pos_w = robot.data.body_pos_w[:, support_body_idx]  # (N, 3)
    support_xy = support_pos_w[:, :2]  # (N, 2)

    # --- Position reward ---
    pos_error = torch.sum((support_xy - desired_xy) ** 2, dim=-1)  # (N,)
    r_pos = torch.exp(-pos_error / (pos_std ** 2))

    # --- Yaw reward: support foot yaw should face kick direction ---
    # Extract yaw from support foot quaternion
    support_quat = robot.data.body_quat_w[:, support_body_idx]  # (N, 4) wxyz
    # Yaw = atan2(2*(wz + xy), 1 - 2*(yy + zz)) — standard quat-to-yaw for wxyz convention
    w, x, y, z = support_quat[:, 0], support_quat[:, 1], support_quat[:, 2], support_quat[:, 3]
    support_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))  # (N,)

    # Desired yaw = direction of kick_dir
    desired_yaw = torch.atan2(kick_dir[:, 1], kick_dir[:, 0])  # (N,)

    # Wrap yaw error to [-pi, pi]
    yaw_err = support_yaw - desired_yaw
    yaw_err = torch.atan2(torch.sin(yaw_err), torch.cos(yaw_err))  # wrap to [-pi, pi]
    r_yaw = torch.exp(-(yaw_err ** 2) / (yaw_std ** 2))

    # --- Combined reward ---
    reward = (1.0 * r_pos + yaw_weight * r_yaw) / (1.0 + yaw_weight)

    # --- Gate by plant window + distance ---
    reward = torch.where(active, reward, torch.zeros_like(reward))

    return reward


# ---------------------------------------------------------------------------
# State-Based Support Foot Stability Prior
# ---------------------------------------------------------------------------

def _soft_range_reward(x: torch.Tensor, lo: float, hi: float, std: float) -> torch.Tensor:
    """Soft range reward: 1.0 inside [lo, hi], gaussian decay outside."""
    below = torch.exp(-((x - lo) ** 2) / (std ** 2)) * (x < lo).float()
    above = torch.exp(-((x - hi) ** 2) / (std ** 2)) * (x > hi).float()
    inside = ((x >= lo) & (x <= hi)).float()
    return inside + below + above


def support_foot_stability_prior(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    support_foot_name: str = "left_ankle_roll_link",
    # --- Soft gate params ---
    near_ball_dist: float = 0.8,
    near_temp: float = 0.2,
    contact_threshold: float = 20.0,
    use_hard_contact_gate: bool = True,
    # --- Ball contact gate ---
    ball_sensor_name: str = "soccer_ball_contact",
    ball_horizontal_force_threshold: float = 5.0,
    # --- Reward params ---
    vel_std: float = 0.2,
    yaw_std: float = 0.6,
    stable_weight: float = 0.3,
    yaw_weight: float = 0.7,
    # --- Region reward (Phase A2, optional) ---
    use_region_reward: bool = False,
    lateral_min: float = 0.18,
    lateral_max: float = 0.35,
    longitudinal_min: float = -0.15,
    longitudinal_max: float = 0.08,
    region_std: float = 0.1,
    # --- Contact sensor ---
    contact_sensor_name: str = "foot_contact",
    # --- CG phase gate (v7.3) ---
    cg_margin: int = 5,
) -> torch.Tensor:
    """State-based support foot brake prior (v7.3).

    CG0-only dense reward: encourages the support foot to stabilise and
    orient toward the kick direction during the approach phase.
    Automatically disabled during CG1 (strike window) so it never
    inhibits the kicking action.

    Gates (all must be active):
      0. CG phase is CG=0 (approach, before kick_frame - margin)  [v7.3]
      1. The ball has NOT been kicked yet (pre_ball)
      2. No robot-ball contact is happening THIS STEP (no_ball_contact_now)
      3. The robot is close to the ball (soft sigmoid distance gate)
      4. The support foot is in ground contact (hard or soft force gate)

    Fail-safe: If the foot contact sensor is missing, returns zero reward.

    Changes from v7.1:
      - v7.3: Added CG0 gate — reward is zero during CG1 to prevent
        the delayed-strike behaviour observed in v7.1 (ArgΔ = +152).
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    device = env.device
    tracker = _get_kick_tracker(command)
    robot = command.robot

    # ===== Fail-safe: require foot contact sensor =====
    sensors = getattr(env.scene, "sensors", None)
    contact_sensor = None
    if sensors is not None:
        if isinstance(sensors, dict):
            contact_sensor = sensors.get(contact_sensor_name)
        else:
            contact_sensor = getattr(sensors, contact_sensor_name, None)

    if contact_sensor is None:
        # No foot contact sensor → cannot reliably gate → return zero
        return torch.zeros(env.num_envs, device=device)

    # ===== Cache body indices (robot vs sensor, may differ) =====
    cache_key = f"_support_prior_cache_{support_foot_name}"
    if not hasattr(env, cache_key):
        robot_body_ids = [robot.body_names.index(support_foot_name)]
        sensor_body_ids, _ = contact_sensor.find_bodies(
            [support_foot_name], preserve_order=True
        )
        cache = {
            "robot_body_idx": robot_body_ids[0],
            "sensor_body_idx": sensor_body_ids[0] if len(sensor_body_ids) > 0 else None,
        }
        setattr(env, cache_key, cache)

    cache = getattr(env, cache_key)
    support_body_idx = cache["robot_body_idx"]
    support_sensor_idx = cache["sensor_body_idx"]

    if support_sensor_idx is None:
        return torch.zeros(env.num_envs, device=device)

    # ===== Gate 1: ball not yet kicked =====
    pre_ball = (~tracker.get_contact_awarded()).float()  # (N,)

    # ===== Gate 2: no robot-ball contact THIS STEP =====
    # Use horizontal (XY) force only — filters out ball-ground normal force (Z).
    # Same logic as early_collision_penalty.
    try:
        ball_contact_sensor = env.scene[ball_sensor_name]
        net_forces = ball_contact_sensor.data.net_forces_w_history
        if net_forces.dim() == 4:
            force_vec = net_forces[:, :, 0, :2].sum(dim=1)  # (N, 2)
        else:
            force_vec = net_forces[:, 0, :2]  # (N, 2)
        force_mag = torch.norm(force_vec, dim=-1)  # (N,)
        no_ball_contact_now = (force_mag < ball_horizontal_force_threshold).float()
    except Exception:
        no_ball_contact_now = torch.ones(env.num_envs, device=device)

    # ===== Gate 3: robot proximity to ball (sigmoid) =====
    ball_pos_w = get_target_point_world(env, command_name).to(device=device)
    pelvis_xy = command.robot_anchor_pos_w[:, :2]
    ball_xy = ball_pos_w[:, :2]
    dist = torch.norm(pelvis_xy - ball_xy, dim=-1)  # (N,)
    near_ball_w = torch.sigmoid((near_ball_dist - dist) / near_temp)  # (N,)

    # ===== Gate 4: support foot ground contact =====
    forces = contact_sensor.data.net_forces_w  # (N, B, 3)
    if forces is not None and forces.numel() > 0:
        forces = forces.to(device=device)
        support_force_z = forces[:, support_sensor_idx, 2].clamp(min=0.0)
    else:
        support_force_z = torch.zeros(env.num_envs, device=device)

    if use_hard_contact_gate:
        support_contact_w = (support_force_z > contact_threshold).float()
    else:
        support_contact_w = torch.sigmoid(
            (support_force_z - contact_threshold) / 10.0
        )

    # ===== Gate 5 (v7.3): CG0 phase only =====
    is_cg0, _ = _get_cg_phase(command, margin=cg_margin)

    # ===== Combined activation weight =====
    active_w = is_cg0.float() * pre_ball * no_ball_contact_now * near_ball_w * support_contact_w  # (N,)

    # ===== R1: Support foot XY velocity stability =====
    support_vel_w = robot.data.body_lin_vel_w[:, support_body_idx]  # (N, 3)
    vel_sq = torch.sum(support_vel_w[:, :2] ** 2, dim=-1)  # (N,)
    r_stable = torch.exp(-vel_sq / (vel_std ** 2))  # (N,)

    # ===== R2: Support foot yaw alignment toward kick direction =====
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is not None:
        dest_w = command.target_destination_pos + env_origins
    else:
        dest_w = command.target_destination_pos
    dest_xy = dest_w[:, :2]

    kick_dir = dest_xy - ball_xy  # (N, 2)
    kick_dir_norm = torch.norm(kick_dir, dim=-1, keepdim=True).clamp(min=1e-6)
    kick_dir = kick_dir / kick_dir_norm  # (N, 2) normalized

    support_quat = robot.data.body_quat_w[:, support_body_idx]  # (N, 4) wxyz
    w, x, y, z = support_quat[:, 0], support_quat[:, 1], support_quat[:, 2], support_quat[:, 3]
    support_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    desired_yaw = torch.atan2(kick_dir[:, 1], kick_dir[:, 0])
    yaw_err = support_yaw - desired_yaw
    yaw_err = torch.atan2(torch.sin(yaw_err), torch.cos(yaw_err))
    r_yaw = torch.exp(-(yaw_err ** 2) / (yaw_std ** 2))  # (N,)

    # ===== Phase A2: Optional broad region reward =====
    if use_region_reward:
        support_pos_w = robot.data.body_pos_w[:, support_body_idx]
        support_xy = support_pos_w[:, :2]

        side_dir = torch.stack([-kick_dir[:, 1], kick_dir[:, 0]], dim=-1)
        kick_leg = command.kick_leg
        side_sign = torch.where(kick_leg == 0, -1.0, 1.0).to(device=device)

        rel = support_xy - ball_xy
        longitudinal = torch.sum(rel * kick_dir, dim=-1)
        lateral = side_sign * torch.sum(rel * side_dir, dim=-1)

        r_lat = _soft_range_reward(lateral, lateral_min, lateral_max, region_std)
        r_lon = _soft_range_reward(longitudinal, longitudinal_min, longitudinal_max, region_std)
        r_region = r_lat * r_lon

        total_w = stable_weight + yaw_weight + 0.2
        reward = active_w * (
            stable_weight / total_w * r_stable
            + yaw_weight / total_w * r_yaw
            + 0.2 / total_w * r_region
        )
    else:
        total_w = stable_weight + yaw_weight
        reward = active_w * (
            stable_weight / total_w * r_stable
            + yaw_weight / total_w * r_yaw
        )

    return reward


# ---------------------------------------------------------------------------
# Support Contact Quality Bonus (v7.3 Phase 3)
# ---------------------------------------------------------------------------

def support_contact_quality_bonus(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    support_foot_name: str = "left_ankle_roll_link",
    # --- Ball contact detection ---
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 10.0,
    foot_cfg: SceneEntityCfg | None = None,
    cg_margin: int = 5,
    # --- Support foot quality ---
    contact_threshold: float = 20.0,
    yaw_std: float = 0.6,
    contact_sensor_name: str = "foot_contact",
) -> torch.Tensor:
    """Sparse bonus for support foot quality at the moment of legal contact (v7.3).

    Only fires when ALL of the following are true in a single step:
      1. CG phase is CG=1 (legal kick window)
      2. Ball contact is detected this step (new_contact)
      3. Contact is with the correct foot

    The bonus value is:
      q_support = support_contact_w * r_yaw

    where:
      - support_contact_w: 1.0 if support foot has ground contact > threshold
      - r_yaw: gaussian reward for support foot yaw alignment to kick direction

    This reward cannot create a "don't kick but still collect support reward"
    local optimum because it is gated on actual ball contact.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    device = env.device

    # ===== Detect legal ball contact (same logic as time_gated_contact) =====
    _, is_cg1 = _get_cg_phase(command, margin=cg_margin)
    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)

    reward = torch.zeros(env.num_envs, device=device, dtype=torch.float32)
    if not torch.any(event.new_contact):
        return reward

    # ===== Check correct foot =====
    legal_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    if foot_cfg is not None:
        foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
        if foot_info.env_ids.numel() > 0:
            valid_expectation = foot_info.expected >= 0
            correct = (foot_info.sides == foot_info.expected) & valid_expectation
            legal_contact[foot_info.env_ids] = correct
    else:
        # No foot_cfg → treat all new contacts as correct
        legal_contact = event.new_contact

    # Gate by CG1
    legal_contact = legal_contact & is_cg1

    if not torch.any(legal_contact):
        return reward

    # ===== Compute support foot quality =====

    # --- Support foot ground contact ---
    robot = command.robot
    sensors = getattr(env.scene, "sensors", None)
    contact_sensor = None
    if sensors is not None:
        if isinstance(sensors, dict):
            contact_sensor = sensors.get(contact_sensor_name)
        else:
            contact_sensor = getattr(sensors, contact_sensor_name, None)

    if contact_sensor is None:
        return reward

    # Cache body indices
    cache_key = f"_support_bonus_cache_{support_foot_name}"
    if not hasattr(env, cache_key):
        robot_body_ids = [robot.body_names.index(support_foot_name)]
        sensor_body_ids, _ = contact_sensor.find_bodies(
            [support_foot_name], preserve_order=True
        )
        cache = {
            "robot_body_idx": robot_body_ids[0],
            "sensor_body_idx": sensor_body_ids[0] if len(sensor_body_ids) > 0 else None,
        }
        setattr(env, cache_key, cache)

    cache = getattr(env, cache_key)
    support_body_idx = cache["robot_body_idx"]
    support_sensor_idx = cache["sensor_body_idx"]

    if support_sensor_idx is None:
        return reward

    # Ground contact check
    forces = contact_sensor.data.net_forces_w
    if forces is not None and forces.numel() > 0:
        forces = forces.to(device=device)
        support_force_z = forces[:, support_sensor_idx, 2].clamp(min=0.0)
    else:
        support_force_z = torch.zeros(env.num_envs, device=device)

    support_contact_w = (support_force_z > contact_threshold).float()

    # --- Support foot yaw alignment ---
    ball_pos_w = get_target_point_world(env, command_name).to(device=device)
    ball_xy = ball_pos_w[:, :2]

    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is not None:
        dest_w = command.target_destination_pos + env_origins
    else:
        dest_w = command.target_destination_pos
    dest_xy = dest_w[:, :2]

    kick_dir = dest_xy - ball_xy
    kick_dir_norm = torch.norm(kick_dir, dim=-1, keepdim=True).clamp(min=1e-6)
    kick_dir = kick_dir / kick_dir_norm

    support_quat = robot.data.body_quat_w[:, support_body_idx]
    w, x, y, z = support_quat[:, 0], support_quat[:, 1], support_quat[:, 2], support_quat[:, 3]
    support_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    desired_yaw = torch.atan2(kick_dir[:, 1], kick_dir[:, 0])
    yaw_err = support_yaw - desired_yaw
    yaw_err = torch.atan2(torch.sin(yaw_err), torch.cos(yaw_err))
    r_yaw = torch.exp(-(yaw_err ** 2) / (yaw_std ** 2))

    # ===== q_support = ground_contact * yaw_quality =====
    q_support = support_contact_w * r_yaw

    # ===== Sparse bonus: only on legal contact frames =====
    reward = legal_contact.float() * q_support

    return reward