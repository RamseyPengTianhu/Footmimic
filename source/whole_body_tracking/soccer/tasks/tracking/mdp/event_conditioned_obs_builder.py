"""Standalone v10 observation builder for use across collect / BC / PPO / eval.

Computes the full ~420D v10 observation from raw env state, independent of
the ManagerBasedRLEnv observation manager. This ensures BC, PPO, and eval
all use exactly the same obs computation.

Usage:
    builder = V10ObsBuilder(num_envs, num_joints, device)
    builder.reset(env_ids)

    # Each step (BEFORE env.step):
    obs_v10 = builder.compute(env, command)

    # After env.step:
    builder.update_history(env, command, action, dones)
"""
from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_apply, quat_inv

from soccer.tasks.tracking.mdp.event_phase import (
    compute_segment_bounds,
    compute_event_phase,
    compute_event_obs,
    event_warped_ref_index,
    query_event_warped_weak_prior,
    query_event_warped_joint_delta,
    query_event_warped_base_prior,
    NUM_PHASES,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import MotionCommand


class V10ObsBuilder:
    """Builds ~420D v10 observation from raw env state.

    obs_v10 = concat(
        current_proprio,          # 64D: ang_vel(3) + gravity(3) + joint_pos(29) + joint_vel(29)
        proprio_history_flat,     # 261D: joint_pos×3(87) + joint_vel×3(87) + action×3(87)
        last_action,              # 29D
        ball_history_flat,        # 30D: ball_pos_local×10(30)
        event_condition,          # 8D
        ball_foot_relation,       # 22D
        motor_prior,              # 40D
    )  ≈ 454D
    """

    SWING_FOOT = "right_ankle_roll_link"
    SUPPORT_FOOT = "left_ankle_roll_link"
    PROPRIO_HIST_LEN = 3
    BALL_HIST_LEN = 10
    PRESTRIKE_DURATION = 20
    MIN_STRIKE_DURATION = 5

    def __init__(self, num_envs: int, num_joints: int = 29, device: str = "cuda"):
        self.num_envs = num_envs
        self.num_joints = num_joints
        self.device = device

        # History buffers: [N, hist_len, dim]
        self.joint_pos_hist = torch.zeros(num_envs, self.PROPRIO_HIST_LEN, num_joints, device=device)
        self.joint_vel_hist = torch.zeros(num_envs, self.PROPRIO_HIST_LEN, num_joints, device=device)
        self.action_hist = torch.zeros(num_envs, self.PROPRIO_HIST_LEN, num_joints, device=device)
        self.ball_pos_local_hist = torch.zeros(num_envs, self.BALL_HIST_LEN, 3, device=device)

        # Last action (most recent)
        self.last_action = torch.zeros(num_envs, num_joints, device=device)

        # Event segment bounds: [N, 4] = [approach_end, prestrike_end, strike_end, motion_length]
        self.segment_bounds = torch.zeros(num_envs, 4, dtype=torch.long, device=device)

        # Cache for event info (saved alongside obs for analysis)
        self.phase_id = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.phase_phi = torch.zeros(num_envs, device=device)

        # Body indices (resolved lazily)
        self._swing_idx = None
        self._support_idx = None
        self._pelvis_idx = None

    def _resolve_body_indices(self, robot):
        """Resolve body name → index mapping once."""
        if self._swing_idx is None:
            self._swing_idx = robot.body_names.index(self.SWING_FOOT)
            self._support_idx = robot.body_names.index(self.SUPPORT_FOOT)
            self._pelvis_idx = robot.body_names.index("pelvis")

    def reset(self, env_ids: torch.Tensor):
        """Reset history for specified environments."""
        if env_ids.numel() == 0:
            return
        self.joint_pos_hist[env_ids] = 0.0
        self.joint_vel_hist[env_ids] = 0.0
        self.action_hist[env_ids] = 0.0
        self.ball_pos_local_hist[env_ids] = 0.0
        self.last_action[env_ids] = 0.0

    def init_segment_bounds(self, command: MotionCommand):
        """Compute event segment bounds from motion kick annotations."""
        for i in range(self.num_envs):
            mid = command.motion_idx[i].item()
            kf = command.motion.kick_frames[mid].item()
            kef = command.motion.kick_end_frames[mid].item()
            ml = command.motion_length[i].item()

            if kf < 0:
                # No kick annotation — treat entire motion as approach
                kf = ml
                kef = ml

            bounds = compute_segment_bounds(
                kf, kef, ml,
                prestrike_duration=self.PRESTRIKE_DURATION,
                min_strike_duration=self.MIN_STRIKE_DURATION,
            )
            self.segment_bounds[i, 0] = bounds.approach_end
            self.segment_bounds[i, 1] = bounds.prestrike_end
            self.segment_bounds[i, 2] = bounds.strike_end
            self.segment_bounds[i, 3] = bounds.motion_length

    def compute(self, env, command: MotionCommand) -> torch.Tensor:
        """Compute full ~422D v10 observation from current env state.

        MUST be called BEFORE env.step() — the obs corresponds to the state
        that will produce the action.

        Returns: [N, ~422] tensor
        """
        robot = command.robot
        self._resolve_body_indices(robot)
        soccer_ball = env.scene["soccer_ball"]

        # ===== Group 1: Current proprio (64D) =====
        base_ang_vel = robot.data.root_ang_vel_b  # [N, 3]
        gravity = robot.data.projected_gravity_b   # [N, 3]
        joint_pos = robot.data.joint_pos           # [N, 29]
        joint_vel = robot.data.joint_vel           # [N, 29]
        current_proprio = torch.cat([base_ang_vel, gravity, joint_pos, joint_vel], dim=-1)  # [N, 64]

        # ===== Group 2: Proprio/action history (261D) + last_action (29D) =====
        proprio_hist = torch.cat([
            self.joint_pos_hist.flatten(1),  # [N, 87]
            self.joint_vel_hist.flatten(1),  # [N, 87]
            self.action_hist.flatten(1),     # [N, 87]
        ], dim=-1)  # [N, 261]

        # ===== Group 3: Ball history (30D) =====
        ball_hist = self.ball_pos_local_hist.flatten(1)  # [N, 30]

        # ===== Group 4: Event condition (8D) =====
        self.phase_id, self.phase_phi = compute_event_phase(
            command.time_steps, self.segment_bounds
        )
        event_obs = compute_event_obs(
            self.phase_id, self.phase_phi,
            command.time_steps, self.segment_bounds
        )  # [N, 8]

        # ===== Group 5: Ball-foot relation (22D) =====
        ball_foot_rel = self._compute_ball_foot_relation(
            robot, soccer_ball, command, env
        )  # [N, 22]

        # ===== Group 6: Event-Warped Motor Prior (40D) =====
        motor_prior = self._compute_motor_prior(command, soccer_ball, env)  # [N, 40]

        # Concatenate all groups
        obs_v10 = torch.cat([
            current_proprio,    # 64D
            proprio_hist,       # 261D
            self.last_action,   # 29D
            ball_hist,          # 30D
            event_obs,          # 8D
            ball_foot_rel,      # 22D
            motor_prior,        # 40D
        ], dim=-1)  # [N, ~454]

        return obs_v10

    def update_history(
        self,
        env,
        command: MotionCommand,
        action: torch.Tensor,
        dones: torch.Tensor,
    ):
        """Update history buffers AFTER env.step().

        Args:
            action: the action that was executed this step (teacher action for BC)
            dones: episode termination flags
        """
        robot = command.robot
        self._resolve_body_indices(robot)
        soccer_ball = env.scene["soccer_ball"]

        # Current state (post-step)
        joint_pos = robot.data.joint_pos
        joint_vel = robot.data.joint_vel

        # Ball position in pelvis-local frame
        ball_pos_w = soccer_ball.data.root_pos_w[:, :3]
        pelvis_pos_w = robot.data.body_pos_w[:, self._pelvis_idx]
        pelvis_quat_w = robot.data.body_quat_w[:, self._pelvis_idx]
        ball_pos_local = quat_apply(quat_inv(pelvis_quat_w), ball_pos_w - pelvis_pos_w)

        # Push into FIFO
        self.joint_pos_hist = torch.cat([self.joint_pos_hist[:, 1:], joint_pos.unsqueeze(1)], dim=1)
        self.joint_vel_hist = torch.cat([self.joint_vel_hist[:, 1:], joint_vel.unsqueeze(1)], dim=1)
        self.action_hist = torch.cat([self.action_hist[:, 1:], action.unsqueeze(1)], dim=1)
        self.ball_pos_local_hist = torch.cat([self.ball_pos_local_hist[:, 1:], ball_pos_local.unsqueeze(1)], dim=1)
        self.last_action = action.clone()

        # Reset on episode boundary
        if dones.any():
            done_ids = dones.nonzero(as_tuple=True)[0]
            self.reset(done_ids)
            # Recompute segment bounds for reset envs
            for idx in done_ids:
                i = idx.item()
                mid = command.motion_idx[i].item()
                kf = command.motion.kick_frames[mid].item()
                kef = command.motion.kick_end_frames[mid].item()
                ml = command.motion_length[i].item()
                if kf < 0:
                    kf = ml
                    kef = ml
                bounds = compute_segment_bounds(
                    kf, kef, ml,
                    prestrike_duration=self.PRESTRIKE_DURATION,
                    min_strike_duration=self.MIN_STRIKE_DURATION,
                )
                self.segment_bounds[i] = torch.tensor(
                    [bounds.approach_end, bounds.prestrike_end, bounds.strike_end, bounds.motion_length],
                    device=self.device,
                )

    def _compute_ball_foot_relation(self, robot, soccer_ball, command, env) -> torch.Tensor:
        """Compute 22D ball-foot relation (same logic as observations_v10.v10_ball_foot_relation)."""
        ball_pos_w = soccer_ball.data.root_pos_w[:, :3]
        ball_vel_w = soccer_ball.data.root_lin_vel_w[:, :3]

        pelvis_pos_w = robot.data.body_pos_w[:, self._pelvis_idx]
        pelvis_quat_w = robot.data.body_quat_w[:, self._pelvis_idx]
        pelvis_quat_inv = quat_inv(pelvis_quat_w)

        swing_pos_w = robot.data.body_pos_w[:, self._swing_idx]
        swing_quat_w = robot.data.body_quat_w[:, self._swing_idx]
        support_pos_w = robot.data.body_pos_w[:, self._support_idx]
        support_quat_w = robot.data.body_quat_w[:, self._support_idx]

        ball_rel_swing = quat_apply(quat_inv(swing_quat_w), ball_pos_w - swing_pos_w)
        ball_rel_support = quat_apply(quat_inv(support_quat_w), ball_pos_w - support_pos_w)
        ball_rel_pelvis = quat_apply(pelvis_quat_inv, ball_pos_w - pelvis_pos_w)
        ball_vel_local = quat_apply(pelvis_quat_inv, ball_vel_w)

        # Kick direction
        dest_pos = command.target_destination_pos
        env_origins = getattr(env.scene, "env_origins", None)
        dest_w = dest_pos[:, :2] + env_origins[:, :2] if env_origins is not None else dest_pos[:, :2]
        kick_dir_w = dest_w - ball_pos_w[:, :2]
        kick_dir_dist = torch.norm(kick_dir_w, dim=-1, keepdim=True).clamp(min=1e-4)
        kick_dir_2d = kick_dir_w / kick_dir_dist
        kick_dir_3d = torch.cat([kick_dir_2d, torch.zeros_like(kick_dir_2d[:, :1])], dim=-1)
        kick_dir_local = quat_apply(pelvis_quat_inv, kick_dir_3d)[:, :2]

        swing_to_ball_w = ball_pos_w - swing_pos_w
        swing_foot_ball_dist = torch.norm(swing_to_ball_w[:, :2], dim=-1, keepdim=True)
        kick_perp_2d = torch.stack([-kick_dir_2d[:, 1], kick_dir_2d[:, 0]], dim=-1)
        swing_to_ball_2d = swing_to_ball_w[:, :2]
        swing_ball_longitudinal = (swing_to_ball_2d * kick_dir_2d).sum(dim=-1, keepdim=True)
        swing_ball_lateral = (swing_to_ball_2d * kick_perp_2d).sum(dim=-1, keepdim=True)
        support_to_ball_2d = (ball_pos_w - support_pos_w)[:, :2]
        support_ball_longitudinal = (support_to_ball_2d * kick_dir_2d).sum(dim=-1, keepdim=True)
        support_ball_lateral = (support_to_ball_2d * kick_perp_2d).sum(dim=-1, keepdim=True)
        swing_lin_vel_w = robot.data.body_lin_vel_w[:, self._swing_idx]
        swing_vel_along_kick = (swing_lin_vel_w[:, :2] * kick_dir_2d).sum(dim=-1, keepdim=True)
        swing_to_ball_dir = swing_to_ball_2d / swing_to_ball_2d.norm(dim=-1, keepdim=True).clamp(min=1e-4)
        swing_vel_to_ball_align = (swing_lin_vel_w[:, :2] * swing_to_ball_dir).sum(dim=-1, keepdim=True)
        ball_vel_mag = torch.norm(ball_vel_w[:, :2], dim=-1, keepdim=True)

        return torch.cat([
            ball_rel_swing, ball_rel_support, ball_rel_pelvis,  # 9D
            ball_vel_local, kick_dir_local,                      # 5D
            swing_foot_ball_dist, swing_ball_longitudinal, swing_ball_lateral,  # 3D
            support_ball_lateral, support_ball_longitudinal,     # 2D
            swing_vel_along_kick, swing_vel_to_ball_align,       # 2D
            ball_vel_mag,                                        # 1D
        ], dim=-1)  # [N, 22]

    def _compute_motor_prior(self, command, soccer_ball, env) -> torch.Tensor:
        """Compute 40D event-warped motor prior.
        Includes 8D weak prior (offsets/facing), 29D joint delta, 1D height delta, 2D gravity delta.
        """
        # Get reference frame in original motion
        ref_idx = event_warped_ref_index(
            self.phase_id, self.phase_phi, self.segment_bounds
        )

        # Ball position in motion coordinate frame (env-origin-relative)
        ball_pos_w = soccer_ball.data.root_pos_w[:, :3]
        env_origins = getattr(env.scene, "env_origins", None)
        if env_origins is not None:
            ball_pos_local = ball_pos_w - env_origins
        else:
            ball_pos_local = ball_pos_w

        # Resolve foot body indices in the MOTION data (not robot)
        # Motion body indices correspond to command.body_indexes mapping
        motion_body_names = command.cfg.body_names
        swing_motion_idx = motion_body_names.index(self.SWING_FOOT) if self.SWING_FOOT in motion_body_names else 0
        support_motion_idx = motion_body_names.index(self.SUPPORT_FOOT) if self.SUPPORT_FOOT in motion_body_names else 0

        weak_prior_8d = query_event_warped_weak_prior(
            ref_idx=ref_idx,
            motion=command.motion,
            motion_idx=command.motion_idx,
            swing_foot_body_idx=swing_motion_idx,
            support_foot_body_idx=support_motion_idx,
            ball_pos_in_motion=ball_pos_local,
        )  # [N, 8]
        
        joint_delta_29d = query_event_warped_joint_delta(
            env=env,
            command=command,
            current_phase_idx=self.phase_id,
            phase_progress=self.phase_phi,
            original_bounds=self.segment_bounds,
        )  # [N, 29]
        
        base_prior_3d = query_event_warped_base_prior(
            env=env,
            command=command,
            current_phase_idx=self.phase_id,
            phase_progress=self.phase_phi,
            original_bounds=self.segment_bounds,
        )  # [N, 3]

        return torch.cat([weak_prior_8d, joint_delta_29d, base_prior_3d], dim=-1)  # [N, 40]

    def get_event_info(self) -> dict[str, torch.Tensor]:
        """Return current event phase info for logging/analysis."""
        phase_onehot = torch.zeros(self.num_envs, NUM_PHASES, device=self.device)
        phase_onehot.scatter_(1, self.phase_id.unsqueeze(1), 1.0)
        return {
            "phase_id": self.phase_id.clone(),
            "phase_phi": self.phase_phi.clone(),
            "phase_onehot": phase_onehot,
            "segment_bounds": self.segment_bounds.clone(),
        }
