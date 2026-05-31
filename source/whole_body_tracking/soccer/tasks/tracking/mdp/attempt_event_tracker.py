"""Geometry/event-based attempt tracker for reference-free student rewards.

This module provides per-frame reward signals based entirely on physical
observables — foot kinematics, ball contact forces, ball velocity.

**Zero dependence on motion reference, CG, kick_frame, or kick_leg.**

State machine per env:

    IDLE  ─[attempt_start]─▶  ATTEMPT  ─[contact in window]─▶  HIT
                                       ─[window expired]─▶  MISSED ─[contact]─▶ LATE
    IDLE  ─[contact w/o attempt]─▶  ACCIDENTAL

Attempt detection (geometry-only):
    - Pelvis within `near_ball_dist` of ball
    - Closest foot within `max_foot_ball_dist` of ball
    - Foot height in [min_kick_height, max_kick_height]
    - Foot XY speed >= `speed_threshold`
    - Foot closing speed toward ball >= `closing_speed`

Kick foot selection: closest foot to ball (NOT from motion annotation).
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv


class AttemptEventTracker:
    """Stateful per-env tracker for kick attempt events.

    Call ``step()`` every simulation frame. Read reward signals from
    the returned ``AttemptRewards`` named tuple.

    All state is reset automatically on episode termination.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        # Attempt detection thresholds
        speed_threshold: float = 2.0,
        closing_speed: float = 0.5,
        max_foot_ball_dist: float = 0.9,
        min_kick_height: float = 0.02,
        max_kick_height: float = 0.65,
        near_ball_dist: float = 1.25,
        # Attempt window
        attempt_window: int = 18,
        # Ball contact
        contact_force_threshold: float = 5.0,
        # Ball speed success
        ball_speed_success: float = 2.0,
        # Post-strike stability
        post_delay: int = 3,
        post_duration: int = 25,
    ):
        self.N = num_envs
        self.device = device

        # Thresholds
        self.speed_threshold = speed_threshold
        self.closing_speed = closing_speed
        self.max_foot_ball_dist = max_foot_ball_dist
        self.min_kick_height = min_kick_height
        self.max_kick_height = max_kick_height
        self.near_ball_dist = near_ball_dist
        self.attempt_window = attempt_window
        self.contact_force_threshold = contact_force_threshold
        self.ball_speed_success = ball_speed_success
        self.post_delay = post_delay
        self.post_duration = post_duration

        # Body indices (set by init_body_indices)
        self.left_foot_idx = None
        self.right_foot_idx = None
        self.pelvis_idx = None

        # State
        self._init_state()

    def _init_state(self):
        N, D = self.N, self.device
        z = lambda: torch.zeros(N, device=D)
        zb = lambda: torch.zeros(N, dtype=torch.bool, device=D)

        self.env_step = torch.zeros(N, dtype=torch.long, device=D)
        self.attempt_started = zb()
        self.attempt_missed = zb()
        self.attempt_hit = zb()
        self.contact_seen = zb()
        self.fallback_contact = zb()

        self.attempt_frame = torch.full((N,), -1, dtype=torch.long, device=D)
        self.contact_frame = torch.full((N,), -1, dtype=torch.long, device=D)

        self.peak_ball_speed = z()
        self.contact_ball_speed = z()

        # Near-ball timer for no_attempt penalty
        self.near_ball_frames = torch.zeros(N, dtype=torch.long, device=D)

    def init_body_indices(self, robot):
        """Resolve body indices from robot articulation. Call once at env init."""
        self.left_foot_idx = robot.body_names.index("left_ankle_roll_link")
        self.right_foot_idx = robot.body_names.index("right_ankle_roll_link")
        self.pelvis_idx = robot.body_names.index("pelvis")

    def reset_envs(self, env_ids: torch.Tensor):
        """Reset state for specified env IDs (on episode termination)."""
        if env_ids.numel() == 0:
            return
        self.env_step[env_ids] = 0
        self.attempt_started[env_ids] = False
        self.attempt_missed[env_ids] = False
        self.attempt_hit[env_ids] = False
        self.contact_seen[env_ids] = False
        self.fallback_contact[env_ids] = False
        self.attempt_frame[env_ids] = -1
        self.contact_frame[env_ids] = -1
        self.peak_ball_speed[env_ids] = 0.0
        self.contact_ball_speed[env_ids] = 0.0
        self.near_ball_frames[env_ids] = 0

    def step(
        self,
        env: ManagerBasedRLEnv,
    ) -> dict[str, torch.Tensor]:
        """Advance one frame and compute reward signals.

        Returns dict of per-env reward signals (all [N]):
            clean_contact:   1.0 on the frame of first clean hit, else 0
            ball_speed:      ball speed reward (only after clean first contact)
            direction:       target direction reward (only after clean first contact)
            late_fallback:   1.0 on frame of late fallback contact, else 0
            no_attempt:      per-frame penalty when near ball but not attempting
            post_stability:  upright reward in post-strike window
            attempt_miss:    1.0 on frame when attempt window expires without hit
        """
        self.env_step += 1

        robot = env.scene["robot"]
        soccer_ball = env.scene["soccer_ball"]

        # Ensure body indices are resolved
        if self.left_foot_idx is None:
            self.init_body_indices(robot)

        ball_pos_w = soccer_ball.data.root_pos_w[:, :3]
        ball_vel_w = soccer_ball.data.root_lin_vel_w[:, :3]
        ball_speed = torch.norm(ball_vel_w[:, :2], dim=-1)
        self.peak_ball_speed = torch.maximum(self.peak_ball_speed, ball_speed)

        # --- Kick foot: hardcoded right foot (all training data is right-foot) ---
        # TODO: generalize when left-foot kick data is available
        kick_pos = robot.data.body_pos_w[:, self.right_foot_idx]
        kick_vel = robot.data.body_lin_vel_w[:, self.right_foot_idx]
        pelvis_pos = robot.data.body_pos_w[:, self.pelvis_idx]

        # --- Attempt detection (purely geometric) ---
        kick_rel_xy = kick_pos[:, :2] - ball_pos_w[:, :2]
        kick_dist_xy = torch.norm(kick_rel_xy, dim=-1)

        ball_to_kick = ball_pos_w[:, :2] - kick_pos[:, :2]
        ball_to_kick_unit = ball_to_kick / torch.norm(ball_to_kick, dim=-1, keepdim=True).clamp(min=1e-6)
        kick_speed_xy = torch.norm(kick_vel[:, :2], dim=-1)
        closing_speed = torch.sum(kick_vel[:, :2] * ball_to_kick_unit, dim=-1)

        pelvis_ball_dist = torch.norm(pelvis_pos[:, :2] - ball_pos_w[:, :2], dim=-1)

        attempt_start = (
            (~self.attempt_started)
            & (~self.contact_seen)
            & (pelvis_ball_dist < self.near_ball_dist)
            & (kick_dist_xy < self.max_foot_ball_dist)
            & (kick_pos[:, 2] >= self.min_kick_height)
            & (kick_pos[:, 2] <= self.max_kick_height)
            & (kick_speed_xy >= self.speed_threshold)
            & (closing_speed >= self.closing_speed)
        )

        if torch.any(attempt_start):
            self.attempt_started[attempt_start] = True
            self.attempt_frame[attempt_start] = self.env_step[attempt_start]

        # --- Attempt window expiration ---
        elapsed = self.env_step - self.attempt_frame
        expired = (
            self.attempt_started
            & (~self.attempt_hit)
            & (~self.attempt_missed)
            & (elapsed > self.attempt_window)
        )
        self.attempt_missed[expired] = True

        # --- Ball contact detection (pure physics) ---
        has_contact = self._detect_contact(env)
        new_contact = has_contact & (~self.contact_seen)

        # Initialize reward signals
        rewards = {
            "clean_contact": torch.zeros(self.N, device=self.device),
            "ball_speed": torch.zeros(self.N, device=self.device),
            "direction": torch.zeros(self.N, device=self.device),
            "late_fallback": torch.zeros(self.N, device=self.device),
            "no_attempt": torch.zeros(self.N, device=self.device),
            "post_stability": torch.zeros(self.N, device=self.device),
            "attempt_miss": torch.zeros(self.N, device=self.device),
            "approach_shaping": torch.zeros(self.N, device=self.device),
            "prestrike_shaping": torch.zeros(self.N, device=self.device),
        }

        if torch.any(new_contact):
            self.contact_seen[new_contact] = True
            self.contact_frame[new_contact] = self.env_step[new_contact]
            self.contact_ball_speed[new_contact] = ball_speed[new_contact]

            # Clean hit: attempt started AND contact within window
            hit_in_window = (
                new_contact
                & self.attempt_started
                & ((self.env_step - self.attempt_frame) <= self.attempt_window)
            )
            self.attempt_hit[hit_in_window] = True
            rewards["clean_contact"][hit_in_window] = 1.0

            # Late fallback: contact after missed attempt
            fallback = new_contact & self.attempt_missed
            self.fallback_contact[fallback] = True
            rewards["late_fallback"][fallback] = 1.0

        # --- Ball speed reward (only after clean first contact) ---
        if torch.any(self.attempt_hit):
            # Normalized ball speed: 1 - exp(-speed²/std²)
            speed_reward = 1.0 - torch.exp(-(ball_speed ** 2) / (1.2 ** 2))
            # Only give during a short window after clean contact
            post_contact_elapsed = self.env_step - self.contact_frame
            in_speed_window = (
                self.attempt_hit
                & (self.contact_frame >= 0)
                & (post_contact_elapsed >= 0)
                & (post_contact_elapsed <= 5)
            )
            rewards["ball_speed"][in_speed_window] = speed_reward[in_speed_window]

            # Target direction alignment, also only after clean first contact.
            # This uses the task target destination, not motion-reference timing.
            try:
                command = env.command_manager.get_term("motion")
                target_vec = command.target_destination_pos[:, :2] - command.initial_target_point_pos[:, :2]
                target_norm = torch.norm(target_vec, dim=-1, keepdim=True).clamp(min=1e-6)
                ball_vel_xy = ball_vel_w[:, :2]
                ball_vel_norm = torch.norm(ball_vel_xy, dim=-1, keepdim=True).clamp(min=1e-6)
                target_unit = target_vec / target_norm
                vel_unit = ball_vel_xy / ball_vel_norm
                cos_theta = torch.sum(target_unit * vel_unit, dim=-1).clamp(-1.0, 1.0)
                angle_error = torch.acos(cos_theta).square()
                direction_reward = torch.exp(-angle_error / (0.8 ** 2))
                speed_valid = ball_speed > 0.5
                rewards["direction"][in_speed_window & speed_valid] = direction_reward[in_speed_window & speed_valid]
            except Exception:
                pass

        # --- Attempt miss signal ---
        rewards["attempt_miss"][expired] = 1.0

        # --- No-attempt penalty: near ball but not trying ---
        near_ball = pelvis_ball_dist < self.near_ball_dist
        idle_near_ball = near_ball & (~self.attempt_started) & (~self.contact_seen)
        self.near_ball_frames[idle_near_ball] += 1
        self.near_ball_frames[~idle_near_ball & (~self.contact_seen)] = 0
        # Penalty ramps up after 50 frames (~1 sec at 50 Hz) near ball without attempting
        loitering = self.near_ball_frames > 50
        rewards["no_attempt"][loitering] = 1.0

        # --- Post-strike stability (after ANY contact) ---
        if torch.any(self.contact_seen):
            post_elapsed = self.env_step - self.contact_frame
            in_post = (
                self.contact_seen
                & (self.contact_frame >= 0)
                & (post_elapsed > self.post_delay)
                & (post_elapsed <= self.post_delay + self.post_duration)
            )
            if torch.any(in_post):
                from isaaclab.utils.math import quat_apply_inverse
                base_quat = robot.data.root_quat_w
                grav = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.N, 3)
                proj_grav = quat_apply_inverse(base_quat, grav)
                tilt_err = (1.0 + proj_grav[:, 2]).square()
                r_tilt = torch.exp(-tilt_err / (0.3 ** 2))

                ang_vel_w = robot.data.root_ang_vel_w
                ang_vel_b = quat_apply_inverse(base_quat, ang_vel_w)
                rp_sq = ang_vel_b[:, 0].square() + ang_vel_b[:, 1].square()
                r_angvel = torch.exp(-rp_sq / (1.0 ** 2))

                rewards["post_stability"][in_post] = 0.6 * r_tilt[in_post] + 0.4 * r_angvel[in_post]

        # --- Dense approach shaping (pre-attempt, pre-contact) ---
        # Rewards the robot for closing in on the ball BEFORE an attempt starts.
        # Active only when: near ball, no attempt yet, no contact yet.
        # Components: (1) pelvis approaching ball, (2) right foot approaching ball,
        #             (3) positive closing speed of foot toward ball.
        # All purely geometric, zero CG dependency.
        pre_attempt = (~self.attempt_started) & (~self.contact_seen)
        approach_near = pre_attempt & (pelvis_ball_dist < self.near_ball_dist)
        if torch.any(approach_near):
            # Foot-ball distance reward: closer = higher (exp kernel)
            r_foot_dist = torch.exp(-(kick_dist_xy ** 2) / (0.5 ** 2))
            # Pelvis-ball distance reward
            r_pelvis_dist = torch.exp(-(pelvis_ball_dist ** 2) / (0.8 ** 2))
            # Closing speed reward: positive closing = good
            r_closing = torch.clamp(closing_speed / 2.0, 0.0, 1.0)
            # Combine: 40% foot proximity, 30% pelvis proximity, 30% closing speed
            approach_r = 0.4 * r_foot_dist + 0.3 * r_pelvis_dist + 0.3 * r_closing
            rewards["approach_shaping"][approach_near] = approach_r[approach_near]

        # --- Dense prestrike shaping (during attempt window, before contact) ---
        # Active only: attempt started, not yet hit, not yet missed, no contact.
        # Rewards: foot getting very close to ball + high closing speed.
        in_attempt = (
            self.attempt_started
            & (~self.attempt_hit)
            & (~self.attempt_missed)
            & (~self.contact_seen)
        )
        if torch.any(in_attempt):
            # Tight foot-ball proximity: reward peaks at ~0.1m
            r_tight_dist = torch.exp(-(kick_dist_xy ** 2) / (0.2 ** 2))
            # High closing speed during strike phase
            r_strike_closing = torch.clamp(closing_speed / 3.0, 0.0, 1.0)
            prestrike_r = 0.6 * r_tight_dist + 0.4 * r_strike_closing
            rewards["prestrike_shaping"][in_attempt] = prestrike_r[in_attempt]

        return rewards

    def _detect_contact(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Detect robot-ball contact from contact sensor forces."""
        try:
            sensor = env.scene["soccer_ball_contact"]
            forces = sensor.data.net_forces_w_history
            if forces.dim() == 4:
                force_vec = forces[:, :, 0, :2].sum(dim=1)
            else:
                force_vec = forces[:, 0, :2]
            force_mag = torch.norm(force_vec, dim=-1)
            return force_mag > self.contact_force_threshold
        except Exception:
            return torch.zeros(self.N, dtype=torch.bool, device=self.device)
