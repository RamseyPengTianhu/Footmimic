"""Anchor-based Motion Command with XGen-style ball-anchored strike.

Merges approach + strike motions into ONE combined MultiMotionLoader.
During STRIKE phase, the motion reference frame is shifted so the
kicking foot trajectory passes through the ball's actual position.

File index layout in self.motion:
    [0 .. num_approach-1]   = approach clips
    [num_approach .. total] = strike clips

State machine:
    APPROACH  ──  d ≤ threshold  ──►  STRIKE  ──  motion_finished  ──►  (resample)

XGen ball-anchoring:
    On APPROACH→STRIKE transition, compute a correction offset:
        correction = ball_pos_xy - predicted_foot_pos_xy
    This offset is applied to body_pos_relative_w during strike,
    so the entire motion shifts to align the foot with the ball.
"""
from __future__ import annotations

import os
import torch
import numpy as np
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab.utils.math import yaw_quat, quat_mul, quat_inv, quat_apply, quat_rotate_inverse

from .commands_multi_motion_soccer import (
    MotionCommand,
    MotionCommandCfg,
    MultiMotionLoader,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Per-env state constants.
STATE_APPROACH = 0
STATE_STRIKE = 1


@configclass
class AnchorMotionCommandCfg(MotionCommandCfg):
    """Config for AnchorMotionCommand."""

    class_type: type = None  # filled below after class def

    # Strike-bank motion files (populated at runtime).
    strike_motion_files: list[str] = MISSING

    # Distance threshold (metres) for APPROACH → STRIKE transition.
    strike_trigger_distance: float = 0.8

    # Name of the kicking foot body for ball-anchoring.
    kick_foot_body_name: str = "right_ankle_roll_link"


class AnchorMotionCommand(MotionCommand):
    """MotionCommand with XGen-style ball-anchored APPROACH/STRIKE state machine."""

    cfg: AnchorMotionCommandCfg

    def __init__(self, cfg: AnchorMotionCommandCfg, env: ManagerBasedRLEnv):
        # Merge approach + strike files into a single list.
        self._num_approach = len(cfg.motion_files)
        self._num_strike = len(cfg.strike_motion_files)
        assert self._num_approach > 0, "motion_files (approach) must not be empty"
        assert self._num_strike > 0, "strike_motion_files must not be empty"

        # Concatenate: approach first, then strike.
        combined_files = list(cfg.motion_files) + list(cfg.strike_motion_files)
        original_files = cfg.motion_files
        cfg.motion_files = combined_files

        super().__init__(cfg, env)

        cfg.motion_files = original_files  # restore for serialization

        # Per-env state: 0 = APPROACH, 1 = STRIKE.
        self._state = torch.full(
            (self.num_envs,), STATE_APPROACH, dtype=torch.long, device=self.device,
        )

        # Index ranges.
        self._approach_indices = torch.arange(
            0, self._num_approach, device=self.device, dtype=torch.long,
        )
        self._strike_indices = torch.arange(
            self._num_approach, self._num_approach + self._num_strike,
            device=self.device, dtype=torch.long,
        )
        self._approach_file_lengths = self.motion.file_lengths[:self._num_approach]
        self._strike_file_lengths = self.motion.file_lengths[self._num_approach:]

        # ----- XGen ball-anchoring: pre-compute foot offset at kick_frame -----
        # For each strike clip, compute: foot_pos - pelvis_pos at kick_frame
        # This tells us where the foot "wants to go" relative to the pelvis.
        self._precompute_foot_offsets(cfg)

        # Per-env strike correction (xy offset to align foot with ball).
        self._strike_correction = torch.zeros(
            self.num_envs, 3, device=self.device,
        )

    def _precompute_foot_offsets(self, cfg):
        """Pre-compute the kicking foot's offset relative to pelvis at kick_frame
        for each strike clip. Used for ball-anchoring correction."""
        # Find the foot body index in the FULL body list (not the selected subset).
        robot_body_names = self.robot.body_names
        kick_foot_name = cfg.kick_foot_body_name

        if kick_foot_name in robot_body_names:
            foot_body_idx = robot_body_names.index(kick_foot_name)
        else:
            print(f"[WARN] kick_foot_body_name '{kick_foot_name}' not found. "
                  f"Disabling ball-anchoring.")
            self._foot_offsets = None
            return

        pelvis_body_idx = robot_body_names.index("pelvis")

        # For each strike clip, get foot offset at kick_frame.
        # The raw body_pos_w in MultiMotionLoader is stored as _body_pos_w
        # with shape (num_files, max_T, num_bodies, 3).
        num_strike = self._num_strike
        foot_offsets = torch.zeros(num_strike, 3, device=self.device)

        for i in range(num_strike):
            combined_idx = self._num_approach + i
            # Get kick_frame for this strike clip.
            kick_frame = self.motion._kick_frames[combined_idx].item()
            if kick_frame < 0:
                # No kick_frame annotated, use frame 0.
                kick_frame = 0

            # Clamp to valid range.
            clip_len = self.motion.file_lengths[combined_idx].item()
            kick_frame = min(kick_frame, clip_len - 1)

            # foot pos - pelvis pos at kick_frame (in motion local frame).
            foot_pos = self.motion._body_pos_w[combined_idx, kick_frame, foot_body_idx]
            pelvis_pos = self.motion._body_pos_w[combined_idx, kick_frame, pelvis_body_idx]
            foot_offsets[i] = foot_pos - pelvis_pos

        self._foot_offsets = foot_offsets  # (num_strike, 3)
        print(f"[INFO] Ball-anchoring foot offsets pre-computed for {num_strike} strike clips:")
        for i in range(num_strike):
            print(f"  strike[{i}]: foot offset = {foot_offsets[i].cpu().numpy()}")

    # ------------------------------------------------------------------
    # State machine core
    # ------------------------------------------------------------------

    def _check_state_transition(self):
        """Check if any env should transition from APPROACH → STRIKE."""
        if self.soccer_ball is None:
            return

        approach_mask = (self._state == STATE_APPROACH)
        if not torch.any(approach_mask):
            return

        ball_pos_xy = self.soccer_ball.data.root_pos_w[:, :2]
        pelvis_pos_xy = self.robot_pelvis_pos_w[:, :2]
        dist = torch.norm(ball_pos_xy - pelvis_pos_xy, dim=-1)

        trigger = approach_mask & (dist <= self.cfg.strike_trigger_distance)
        if not torch.any(trigger):
            return

        trigger_ids = torch.where(trigger)[0]
        self._transition_to_strike(trigger_ids)

    def _transition_to_strike(self, env_ids: torch.Tensor):
        """Switch envs to STRIKE with ball-anchored correction."""
        self._state[env_ids] = STATE_STRIKE

        # Assign random strike clips.
        if self._num_strike > 1:
            local_idx = torch.randint(
                0, self._num_strike, (env_ids.numel(),), device=self.device,
            )
        else:
            local_idx = torch.zeros(env_ids.numel(), dtype=torch.long, device=self.device)

        # Map to combined index range.
        self.motion_idx[env_ids] = self._strike_indices[local_idx]
        self.time_steps[env_ids] = 0
        self.motion_length[env_ids] = self._strike_file_lengths[local_idx]

        # ----- XGen ball-anchoring correction -----
        if self._foot_offsets is not None and self.soccer_ball is not None:
            # foot_offset: where the foot goes relative to pelvis in the motion data
            foot_offset = self._foot_offsets[local_idx]  # (N, 3)

            # Get robot's pelvis world pos and yaw
            pelvis_pos = self.robot_pelvis_pos_w[env_ids]  # (N, 3)
            pelvis_quat = self.robot_pelvis_quat_w[env_ids]  # (N, 4)
            pelvis_yaw = yaw_quat(pelvis_quat)  # (N, 4)

            # Where the foot would go in world frame (without correction)
            foot_predicted_w = pelvis_pos + quat_apply(pelvis_yaw, foot_offset)

            # Ball's actual position
            ball_pos = self.soccer_ball.data.root_pos_w[env_ids]  # (N, 3)

            # Correction: shift motion so foot aligns with ball (xy only)
            correction = torch.zeros_like(ball_pos)
            correction[:, :2] = ball_pos[:, :2] - foot_predicted_w[:, :2]

            self._strike_correction[env_ids] = correction

    # ------------------------------------------------------------------
    # Override: resample (reset to APPROACH)
    # ------------------------------------------------------------------

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        ids = self._to_env_id_tensor(env_ids)
        self._state[ids] = STATE_APPROACH
        self._strike_correction[ids] = 0.0

        super()._resample_command(env_ids)

        # Override: ensure motion_idx is in approach range.
        ids = self._to_env_id_tensor(env_ids)
        if self._num_approach > 1:
            self.motion_idx[ids] = torch.randint(
                0, self._num_approach, (ids.numel(),), device=self.device,
            )
        else:
            self.motion_idx[ids] = 0
        self.motion_length[ids] = self._approach_file_lengths[self.motion_idx[ids]]

    # ------------------------------------------------------------------
    # Override: update (state machine + ball-anchoring)
    # ------------------------------------------------------------------

    def _update_command(self):
        self.kick_contact_tracker.begin_step(self)

        # Advance time for ALL envs.
        self.time_steps += 1

        # Check approach → strike transition.
        self._check_state_transition()

        # Handle end-of-motion for APPROACH: hold last frame.
        approach_mask = (self._state == STATE_APPROACH)
        approach_ended = approach_mask & (self.time_steps >= self.motion_length)
        self.time_steps[approach_ended] = (
            self.motion_length[approach_ended] - 1
        ).clamp(min=0)

        # Handle end-of-motion for STRIKE: resample.
        strike_mask = (self._state == STATE_STRIKE)
        strike_ended = strike_mask & (self.time_steps >= self.motion_length)
        resample_ids = torch.where(strike_ended)[0]
        if resample_ids.numel() > 0:
            self._resample_command(resample_ids)

        # === Rest is identical to parent, plus ball-anchoring correction ===

        self._update_target_points_from_sim()

        if hasattr(self, "kick_contact_tracker"):
            contact_awarded = self.kick_contact_tracker.get_contact_awarded()
            no_contact_mask = ~contact_awarded
            if torch.any(no_contact_mask):
                self.initial_target_point_pos[no_contact_mask] = self.target_point_pos[no_contact_mask]

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        # ----- XGen: apply ball-anchoring correction during STRIKE -----
        strike_mask = (self._state == STATE_STRIKE)
        if torch.any(strike_mask):
            # Shift the entire body tracking target so foot aligns with ball.
            self.body_pos_relative_w[strike_mask] += self._strike_correction[strike_mask, None, :]

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

        self._update_metrics()

    # ------------------------------------------------------------------
    # Expose state info
    # ------------------------------------------------------------------

    @property
    def env_state(self) -> torch.Tensor:
        return self._state

    @property
    def is_in_strike(self) -> torch.Tensor:
        return self._state == STATE_STRIKE


# Backfill the class_type.
AnchorMotionCommandCfg.class_type = AnchorMotionCommand
