"""Dribbling motion command with XGen-style demo ball stitching.

Uses per-frame ``ball_pos_w`` from motion for labels and, for unified local
stages, reset placement at the first labelled contact.  The reset position is
represented relative to the frame-0 pelvis yaw, never by a fixed simulation
``+X`` direction. Optional demo kinematic snap is controlled by
``dribble_cg_snap_mode`` for legacy spawn modes.  A
``reference_first_contact`` spawn always remains physical and cannot be
kinematically snapped:

- ``full`` (default): every step writes the demo ball pose into simulation.
- ``non_contact_only``: only overwrite the ball when the CG label says
  non-contact, leaving physics during annotated contact segments.
- ``never``: never overwrite the ball; use the labels/rewards only.

Contact / foot / surface masks come from ``dribble_cg_contact``,
``dribble_cg_foot``, and ``dribble_cg_surface`` in ``.npz``.  The legacy
``kick_frame`` / ``kick_end_frame`` / ``kick_leg`` metadata remains a fallback
for contact timing and foot side, but cannot describe an instep surface (see
:class:`MultiMotionLoader`).  Stage 2 derives its outgoing target directly
from consecutive reference contact anchors; the legacy ``dribble_cg_flow_*``
labels are not part of the 6.0.3 contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, yaw_quat

from soccer.tasks.tracking.mdp.task_frame import mimic_anchor_yaw_delta_quat, spawn_ball_ahead_env_local

from .commands_multi_motion_soccer import MotionCommand, MotionCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_REFERENCE_FIRST_CONTACT_SPAWN_MODES = frozenset(
    {"reference_first_contact", "first_contact", "contact"}
)


class DribbleCGMotionCommand(MotionCommand):
    """Soccer motion command + demo ball sync for dribbling CG."""

    def __init__(self, cfg: DribbleCGMotionCommandCfg, env: ManagerBasedRLEnv):
        # ``MotionCommand.__init__`` invokes ``_compute_soccer_ball_positions``.
        # Allocate reset provenance first because dynamic dispatch reaches the
        # overridden first-contact implementation during that base setup.
        self.ball_spawn_reference_contact_frame = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
        # 1 = reference first-contact placement; 0 = explicit local-front
        # fallback; -1 = legacy/unknown source.  Keep this state separate from
        # the live ball pose so diagnostics retain reset provenance after the
        # ball has moved under physics.
        self.ball_spawn_source = torch.full(
            (env.num_envs,), -1, dtype=torch.int8, device=env.device
        )
        self.ball_spawn_reference_local = torch.zeros(env.num_envs, 3, device=env.device)
        self.s2_episode_first_contact_frame = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
        self.s2_episode_last_contact_frame = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
        self.s2_episode_contact_count = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        self.s2_episode_first_event_index = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
        # Shared causal task state for the asymmetric S2 critic and reward.
        # It is reset after curriculum sampling and advanced only by an
        # accepted physical contact; reference time is never allowed to move
        # this cursor on its own.
        self.s2_sequence_cursor = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
        self.s2_sequence_contact_succeeded = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        # Reward latches this world-space target on an accepted touch.  The
        # critic consumes the same tensor on the following observation step;
        # it must not reconstruct a moving target from the live alignment.
        self.s2_locked_next_anchor_w = torch.zeros(
            env.num_envs, 3, dtype=torch.float32, device=env.device
        )
        self.s2_locked_next_anchor_valid = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        # Reset-time task-frame transform.  When enabled, every sparse event
        # anchor in an episode is mapped through this immutable XY+yaw frame;
        # policy motion before O2 can therefore not move the I3 target.
        self.s2_episode_reference_root_w = torch.zeros(env.num_envs, 3, device=env.device)
        self.s2_episode_actual_root_w = torch.zeros(env.num_envs, 3, device=env.device)
        self.s2_episode_yaw_alignment = torch.zeros(env.num_envs, 4, device=env.device)
        self.s2_episode_yaw_alignment[:, 0] = 1.0
        self.s2_last_live_next_anchor_w = torch.zeros(env.num_envs, 3, device=env.device)
        self.s2_last_fixed_next_anchor_w = torch.zeros(env.num_envs, 3, device=env.device)
        self.s2_last_fixed_target_valid = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self.s2_last_fixed_inverse_error = torch.zeros(env.num_envs, device=env.device)
        # Reset provenance: 0 = real frame-0 sequence, 1 = teacher-provided
        # pre-second-touch acquisition state.  Acquisition episodes never
        # enter promotion statistics.
        self.s2_episode_reset_mode = torch.zeros(
            env.num_envs, dtype=torch.int8, device=env.device
        )
        self.s2_episode_acquisition_history_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        self._s2_acquisition_probability = torch.tensor(
            float(getattr(cfg, "dribble_cg_acquisition_initial_probability", 0.25)),
            dtype=torch.float32,
            device=env.device,
        )
        # The Curriculum Manager owns this scalar. Each sampled episode keeps
        # its originating level so asynchronous resets after a promotion do
        # not contaminate the next level's evaluation window.
        start_level = max(0, int(getattr(cfg, "dribble_cg_curriculum_start_level", 0)))
        self.s2_curriculum_level = torch.tensor(
            start_level, dtype=torch.long, device=env.device
        )
        configured_levels = tuple(
            getattr(cfg, "dribble_cg_curriculum_levels", ())
        )
        history_size = max(1, len(configured_levels))
        # Model filenames call RSL-RL learning iterations "epochs". Keep the
        # exact iteration at which each global level first became active. A
        # fixed tensor is easy to checkpoint and leaves unknown legacy history
        # as -1 instead of inventing a transition time.
        self._s2_curriculum_level_entry_iteration = torch.full(
            (history_size,), -1, dtype=torch.long, device=env.device
        )
        if start_level == 0:
            self._s2_curriculum_level_entry_iteration[0] = 0
        self._s2_training_iteration = torch.tensor(
            -1, dtype=torch.long, device=env.device
        )
        self.s2_episode_curriculum_level = torch.full(
            (env.num_envs,), start_level, dtype=torch.long, device=env.device
        )
        # True episodes are drawn uniformly and are the only samples allowed
        # to update promotion statistics once hard-event replay is enabled.
        # Before replay starts every episode is uniform, so every episode is
        # also an audit episode.
        self.s2_episode_curriculum_audit = torch.ones(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        # Reward-side event state uses this generation instead of relying on
        # manager-specific episode_length_buf reset ordering.
        self.s2_episode_generation = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        super().__init__(cfg, env)
        self._build_s2_contact_event_table()
        event_slots = self._s2_contact_anchor_ref_padded.shape[1]
        # The clean task materializes every sparse reference node once per
        # episode, immediately after reset alignment.  Reward code only gathers
        # from these immutable tensors; policy root/foot motion cannot move the
        # current or next target later in the rollout.
        self.s2_episode_fixed_contact_anchors_w = torch.zeros(
            env.num_envs, event_slots, 3, device=env.device
        )
        self.s2_episode_fixed_next_anchors_w = torch.zeros_like(
            self.s2_episode_fixed_contact_anchors_w
        )
        self.s2_episode_fixed_contact_anchor_valid = torch.zeros(
            env.num_envs, event_slots, dtype=torch.bool, device=env.device
        )
        self.s2_episode_fixed_next_anchor_valid = torch.zeros_like(
            self.s2_episode_fixed_contact_anchor_valid
        )
        self._load_s2_contact_regions()
        for metric_name in (
            "s2_contact_success_rate",
            "s2_missed_contact_rate",
            "s2_correct_foot_rate",
            "s2_touch_occurrence_rate",
            "s2_correct_side_rate",
            "s2_valid_contact_rate",
            "s2_contact_timing_error",
            "s2_contact_timing_abs_error",
            "s2_target_region_distance",
            "s2_contact_anchor_quality",
            "s2_touch_force_mean",
            "s2_touch_force_max",
            "s2_dead_zone_count",
            "s2_wrong_foot_count",
            "s2_wrong_side_count",
            "s2_premature_contact_count",
            "s2_invalid_body_contact_count",
            "s2_out_of_sequence_contact_count",
            "s2_contact_rearmed",
            "s2_contact_release_off_count",
            "s2_contact_release_confirmed",
            "s2_sequence_cursor",
            "s2_complete_2",
            "s2_complete_3",
            "s2_complete_4",
            "s2_complete_5",
            "s2_complete_6",
            "s2_complete_8",
            "s2_complete_selected",
            "s2_approach_progress",
            "s2_coarse_contact_approach",
            "s2_anchor_approach_progress_v2",
            "s2_contact_region_progress_v3",
            "s2_contact_region_arrival",
            "s2_o1_min_region_distance",
            "s2_o1_region_occupancy_rate",
            "s2_o1_region_arrival_quality",
            "s2_o1_filtered_contact_rate",
            "s2_o2_min_region_distance",
            "s2_o2_given_o1_contact_rate",
            "s2_nearest_region_point_id",
            "s2_nearest_region_point_timing",
            "s2_immediate_contact",
            "s2_release_pending",
            "s2_release_score",
            "s2_release_v2_score",
            "s2_next_anchor_progress",
            "s2_terminal_progress_raw_rate",
            "s2_terminal_progress_clipped_rate",
            "s2_terminal_progress_positive_rate",
            "s2_terminal_progress_negative_rate",
            "s2_terminal_transition_valid",
            "s2_terminal_transition_success",
            "s2_terminal_transition_distance",
            "s2_terminal_transition_start_distance",
            "s2_terminal_transition_end_distance",
            "s2_terminal_transition_net_progress",
            "s2_terminal_tracking_quality",
            "s2_terminal_tracking_pulse",
            "s2_terminal_window_position_outcome",
            "s2_window_outcome_score_average",
            "s2_window_outcome_score_end",
            "s2_window_outcome_settlement_quality",
            "s2_window_outcome_settlement_pulse",
            "s2_window_outcome_pulse",
            "s2_window_outcome_sample_count",
            "s2_window_outcome_aborted_ball_lost",
            "s2_window_outcome_aborted_other",
            "s2_fixed_target_shift_distance",
            "s2_fixed_window_min_distance",
            "s2_live_window_min_distance",
            "s2_t2_region_progress",
            "s2_t2_region_arrival",
            "s2_t2_region_distance",
            "s2_t2_success",
            "s2_release_quality_pass",
            "s2_release_v2_quality_pass",
            "s2_first_release_quality",
            "s2_first_release_quality_v2",
            "s2_quality_streak",
            "s2_second_touch_conditional_success",
            "s2_second_touch_conditional_success_v2",
            "s2_reset_mode",
            "s2_acquisition_success_rate",
            "s2_acquisition_history_steps_before_contact",
        ):
            self.metrics[metric_name] = torch.zeros(self.num_envs, device=self.device)
        self._validate_fixed_touch_spec()

    def _load_s2_contact_regions(self) -> None:
        """Load a frozen offline contact-region sidecar for the active motion."""
        region_path = str(getattr(self.cfg, "dribble_cg_contact_region_path", "")).strip()
        self._s2_contact_regions_available = False
        self._s2_contact_region_points = torch.zeros(
            (1, 1, 3), dtype=torch.float32, device=self.device
        )
        self._s2_contact_region_valid = torch.zeros(
            (1, 1), dtype=torch.bool, device=self.device
        )
        self._s2_contact_region_timing = torch.zeros(
            (1, 1), dtype=torch.long, device=self.device
        )
        self._s2_contact_region_source_id = torch.full(
            (1, 1), -1, dtype=torch.long, device=self.device
        )
        self._s2_contact_region_cover_radius = 0.0
        if not region_path:
            return
        if int(self.motion.num_files) != 1:
            raise ValueError(
                "A single contact-region sidecar currently requires exactly one motion"
            )
        path = Path(region_path)
        if not path.is_file():
            raise FileNotFoundError(f"S2 contact-region sidecar not found: {path}")
        with np.load(path) as source:
            required = {
                "points", "valid", "timing_offset", "source_sample_id",
                "anchor_frame", "expected_foot", "surface", "cover_radius",
            }
            missing = sorted(required.difference(source.files))
            if missing:
                raise ValueError(f"{path}: missing {missing}")
            points = np.asarray(source["points"], dtype=np.float32)
            valid = np.asarray(source["valid"], dtype=bool)
            timing = np.asarray(source["timing_offset"], dtype=np.int64)
            source_id = np.asarray(source["source_sample_id"], dtype=np.int64)
            frames = np.asarray(source["anchor_frame"], dtype=np.int64)
            feet = np.asarray(source["expected_foot"], dtype=np.int64)
            surfaces = np.asarray(source["surface"], dtype=np.int64)
            cover_radius = float(np.asarray(source["cover_radius"]).item())
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f"{path}: points must have shape [events, points, 3]")
        if valid.shape != points.shape[:2] or timing.shape != valid.shape or source_id.shape != valid.shape:
            raise ValueError(f"{path}: region arrays have incompatible shapes")
        expected_frames = self._s2_event_frames_by_motion[0].detach().cpu().numpy()
        if not np.array_equal(frames, expected_frames):
            raise ValueError(
                f"{path}: anchor frames {frames.tolist()} do not match motion {expected_frames.tolist()}"
            )
        motion_idx = torch.zeros(len(frames), dtype=torch.long, device=self.device)
        event_frames = torch.as_tensor(frames, dtype=torch.long, device=self.device)
        expected_feet = self.motion.dribble_cg_foot[motion_idx, event_frames].detach().cpu().numpy()
        expected_surfaces = self.motion.dribble_cg_surface[motion_idx, event_frames].detach().cpu().numpy()
        if not np.array_equal(feet, expected_feet) or not np.array_equal(surfaces, expected_surfaces):
            raise ValueError(f"{path}: foot/surface labels do not match the motion")
        if cover_radius <= 0.0:
            raise ValueError(f"{path}: cover_radius must be positive")
        self._s2_contact_region_points = torch.as_tensor(
            points, dtype=torch.float32, device=self.device
        )
        self._s2_contact_region_valid = torch.as_tensor(
            valid, dtype=torch.bool, device=self.device
        )
        self._s2_contact_region_timing = torch.as_tensor(
            timing, dtype=torch.long, device=self.device
        )
        self._s2_contact_region_source_id = torch.as_tensor(
            source_id, dtype=torch.long, device=self.device
        )
        self._s2_contact_region_cover_radius = cover_radius
        self._s2_contact_regions_available = True
        print(
            f"[INFO] S2 contact regions: {path} "
            f"events={points.shape[0]} max_points={points.shape[1]} "
            f"cover_radius={cover_radius:.4f} m"
        )

    def s2_contact_region(
        self, event_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return points, mask, timing, source ids, and validity for event ids."""
        event_id = event_id.to(device=self.device, dtype=torch.long)
        event_count = self._s2_contact_region_points.shape[0]
        safe_event = event_id.clamp(min=0, max=max(0, event_count - 1))
        points = self._s2_contact_region_points[safe_event]
        point_valid = self._s2_contact_region_valid[safe_event]
        timing = self._s2_contact_region_timing[safe_event]
        source_id = self._s2_contact_region_source_id[safe_event]
        event_valid = (
            torch.full_like(event_id, self._s2_contact_regions_available, dtype=torch.bool)
            & (event_id >= 0)
            & (event_id < event_count)
            & point_valid.any(dim=1)
        )
        point_valid = point_valid & event_valid.unsqueeze(1)
        return points, point_valid, timing, source_id, event_valid

    def s2_next_contact_region_target_w(
        self, event_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map the event after ``event_id`` into the current episode world frame.

        Each frozen V3 point is attached to the reference ankle pose at the
        timing offset saved by the CUDA calibration scan.  Callers latch the
        returned world-space set when the outgoing physical touch is accepted;
        later robot/foot motion therefore cannot drag the transition target.
        """
        event_id = event_id.to(device=self.device, dtype=torch.long)
        next_event = event_id + 1
        points_local, point_valid, timing, _source_id, region_valid = (
            self.s2_contact_region(next_event)
        )
        num_points = points_local.shape[1]
        motion_idx = self.motion_idx.to(torch.long)
        max_slots = self._s2_event_frames_padded.shape[1]
        safe_next = next_event.clamp(min=0, max=max_slots - 1)
        anchor_frame = self._s2_event_frames_padded[motion_idx, safe_next]
        event_end = self._s2_event_end_frames_padded[motion_idx, safe_next]
        point_frame = anchor_frame.unsqueeze(1) + timing
        max_frame = (self.motion.file_lengths[motion_idx] - 1).unsqueeze(1)
        point_frame = torch.minimum(point_frame.clamp(min=0), max_frame)

        safe_anchor = anchor_frame.clamp(min=0)
        event_foot = self.motion.dribble_cg_foot[
            motion_idx, safe_anchor
        ].to(torch.long)
        left_index = self.cfg.body_names.index("left_ankle_roll_link")
        right_index = self.cfg.body_names.index("right_ankle_roll_link")
        foot_index = torch.where(
            event_foot == 1,
            torch.full_like(event_foot, right_index),
            torch.full_like(event_foot, left_index),
        )
        expanded_motion = motion_idx.unsqueeze(1).expand(-1, num_points)
        expanded_foot = foot_index.unsqueeze(1).expand(-1, num_points)
        raw_foot_pos = self.motion.body_pos_w[
            expanded_motion, point_frame, expanded_foot
        ] + self._env.scene.env_origins.unsqueeze(1)
        raw_foot_quat = self.motion.body_quat_w[
            expanded_motion, point_frame, expanded_foot
        ]

        alignment = self._mimic_target_yaw_delta_quat(
            self.anchor_quat_w, self.robot_anchor_quat_w
        )
        alignment_points = alignment.unsqueeze(1).expand(-1, num_points, -1)
        mapped_origin = self.robot_anchor_pos_w.clone()
        mapped_origin[:, 2] = self.anchor_pos_w[:, 2]
        mapped_foot_pos = mapped_origin.unsqueeze(1) + quat_apply(
            alignment_points.reshape(-1, 4),
            (raw_foot_pos - self.anchor_pos_w.unsqueeze(1)).reshape(-1, 3),
        ).reshape(self.num_envs, num_points, 3)
        mapped_foot_quat = quat_mul(
            alignment_points.reshape(-1, 4), raw_foot_quat.reshape(-1, 4)
        ).reshape(self.num_envs, num_points, 4)
        points_w = mapped_foot_pos + quat_apply(
            mapped_foot_quat.reshape(-1, 4), points_local.reshape(-1, 3)
        ).reshape(self.num_envs, num_points, 3)

        event_valid = (
            region_valid
            & (next_event >= 0)
            & (next_event < max_slots)
            & (anchor_frame >= 0)
            & (event_end >= anchor_frame)
            & ((event_foot == 0) | (event_foot == 1))
        )
        point_valid = point_valid & event_valid.unsqueeze(1)
        invalid = torch.full_like(anchor_frame, -1)
        return (
            points_w,
            point_valid,
            torch.where(event_valid, anchor_frame, invalid),
            torch.where(event_valid, event_end, invalid),
        )

    def _build_s2_contact_event_table(self) -> None:
        """Build windowed contact-event labels from the v5.10 CG segments.

        Every contiguous non-zero ``dribble_cg_contact`` segment is one
        logical touch event.  Its rising edge is the anchor frame, while the
        complete annotated segment, optionally padded by a configured number
        of frames at both ends, is the valid timing window.  The rising edge
        remains the anchor/reference frame.  Each event
        carries exactly one expected foot and one instep surface
        (0=inside, 1=outside).  A derived
        foot-yaw-local side is retained only for visualization and backwards-
        compatible diagnostics; learning consumes the source surface label
        directly so left/right feet cannot silently invert its meaning.
        """
        num_motions, max_frames = self.motion.dribble_cg_contact.shape
        self._s2_event_id = torch.full(
            (num_motions, max_frames), -1, dtype=torch.long, device=self.device
        )
        self._s2_event_frame = torch.full_like(self._s2_event_id, -1)
        self._s2_event_foot = torch.full(
            (num_motions, max_frames), -1, dtype=torch.int8, device=self.device
        )
        self._s2_event_surface = torch.full_like(self._s2_event_foot, -1)
        self._s2_event_side = torch.full_like(self._s2_event_foot, -1)
        self._s2_event_frames_by_motion: list[torch.Tensor] = []
        self._s2_event_window_starts_by_motion: list[torch.Tensor] = []
        self._s2_event_ends_by_motion: list[torch.Tensor] = []

        post_grace = max(
            0, int(getattr(self.cfg, "dribble_cg_missed_contact_grace_steps", 3))
        )
        for motion_idx in range(num_motions):
            length = int(self.motion.file_lengths[motion_idx].item())
            contact = self.motion.dribble_cg_contact[motion_idx, :length] > 0
            previous = torch.cat(
                (torch.zeros(1, dtype=torch.bool, device=self.device), contact[:-1])
            )
            following = torch.cat(
                (contact[1:], torch.zeros(1, dtype=torch.bool, device=self.device))
            )
            starts = torch.nonzero(contact & ~previous, as_tuple=False).squeeze(-1)
            annotated_ends = torch.nonzero(
                contact & ~following, as_tuple=False
            ).squeeze(-1)
            if starts.numel() != annotated_ends.numel():
                raise RuntimeError(
                    f"Motion {motion_idx} has malformed S2 contact segments: "
                    f"{starts.numel()} starts vs {annotated_ends.numel()} ends"
                )
            tolerance_frames = max(
                0,
                int(
                    getattr(
                        self.cfg,
                        "dribble_cg_contact_window_tolerance_frames",
                        0,
                    )
                ),
            )
            window_starts = (starts - tolerance_frames).clamp(min=0)
            window_ends = (annotated_ends + tolerance_frames).clamp(
                max=max(0, length - 1)
            )
            if starts.numel() > 1:
                overlap = window_ends[:-1] >= window_starts[1:]
                if bool(torch.any(overlap)):
                    bad = int(torch.nonzero(overlap, as_tuple=False)[0].item())
                    raise RuntimeError(
                        f"Motion {motion_idx} S2 contact tolerance of "
                        f"{tolerance_frames} frames overlaps events {bad} and "
                        f"{bad + 1}; reduce the tolerance or fix the labels."
                    )
            self._s2_event_frames_by_motion.append(starts)
            self._s2_event_window_starts_by_motion.append(window_starts)
            # Downstream timing/episode logic consumes the padded window end;
            # anchor lookup continues to consume the unmodified rising edge.
            self._s2_event_ends_by_motion.append(window_ends)
            for event_id, (
                event_frame_tensor,
                window_start_tensor,
                window_end_tensor,
            ) in enumerate(
                zip(starts, window_starts, window_ends, strict=True)
            ):
                event_frame = int(event_frame_tensor.item())
                window_start = int(window_start_tensor.item())
                window_end = int(window_end_tensor.item())
                foot = int(self.motion.dribble_cg_foot[motion_idx, event_frame].item())
                surface = int(self.motion.dribble_cg_surface[motion_idx, event_frame].item())
                # In a foot yaw frame +Y is local left.  Inside is +Y for the
                # right foot and -Y for the left foot.
                side = -1
                if foot in (0, 1) and surface in (0, 1):
                    local_left = (foot == 1 and surface == 0) or (foot == 0 and surface == 1)
                    side = 0 if local_left else 1

                # Pad the complete segment, not a single anchor frame.  The
                # explicit overlap check above keeps close same-side touches
                # distinct even when timing tolerance is enabled.
                target_frames = torch.arange(
                    window_start, window_end + 1, device=self.device
                )
                self._s2_event_id[motion_idx, target_frames] = event_id
                self._s2_event_frame[motion_idx, target_frames] = event_frame
                self._s2_event_foot[motion_idx, target_frames] = foot
                self._s2_event_surface[motion_idx, target_frames] = surface
                self._s2_event_side[motion_idx, target_frames] = side

        max_events = max((int(frames.numel()) for frames in self._s2_event_frames_by_motion), default=0)
        self._s2_event_count_by_motion = torch.as_tensor(
            [int(frames.numel()) for frames in self._s2_event_frames_by_motion],
            dtype=torch.long,
            device=self.device,
        )
        self._s2_event_frames_padded = torch.full(
            (num_motions, max(1, max_events)), -1, dtype=torch.long, device=self.device
        )
        self._s2_event_end_frames_padded = torch.full_like(
            self._s2_event_frames_padded, -1
        )
        self._s2_event_window_start_frames_padded = torch.full_like(
            self._s2_event_frames_padded, -1
        )
        for motion_idx, (frames, window_starts, ends) in enumerate(
            zip(
                self._s2_event_frames_by_motion,
                self._s2_event_window_starts_by_motion,
                self._s2_event_ends_by_motion,
                strict=True,
            )
        ):
            self._s2_event_frames_padded[motion_idx, : frames.numel()] = frames
            self._s2_event_window_start_frames_padded[
                motion_idx, : window_starts.numel()
            ] = window_starts
            self._s2_event_end_frames_padded[motion_idx, : ends.numel()] = ends

        # Each non-final touch points directly at the next reference ball
        # anchor.  The reward maps this raw reference point into the current
        # episode only when the physical touch occurs, so early/late touches
        # do not inherit a stale static velocity target.
        event_table_shape = self._s2_event_frames_padded.shape
        self._s2_next_event_frame_padded = torch.full(
            event_table_shape, -1, dtype=torch.long, device=self.device
        )
        self._s2_next_event_window_start_frame_padded = torch.full_like(
            self._s2_next_event_frame_padded, -1
        )
        self._s2_next_event_end_frame_padded = torch.full_like(
            self._s2_next_event_frame_padded, -1
        )
        self._s2_next_anchor_ref_padded = torch.zeros(
            (*event_table_shape, 3), dtype=torch.float32, device=self.device
        )
        self._s2_contact_anchor_ref_padded = torch.zeros(
            (*event_table_shape, 3), dtype=torch.float32, device=self.device
        )
        # Immutable sparse nodes in the reference pelvis-yaw frame.  These
        # are the clean v0 reward targets; unlike the historical world target
        # they cannot be moved by the policy's current root pose.
        self._s2_contact_node_rel_xy_padded = torch.zeros(
            (*event_table_shape, 2), dtype=torch.float32, device=self.device
        )
        self._s2_next_node_rel_xy_padded = torch.zeros(
            (*event_table_shape, 2), dtype=torch.float32, device=self.device
        )
        self._s2_contact_node_rel_z_padded = torch.zeros(
            event_table_shape, dtype=torch.float32, device=self.device
        )
        self._s2_next_node_rel_z_padded = torch.zeros(
            event_table_shape, dtype=torch.float32, device=self.device
        )
        self._s2_contact_node_rel_valid_padded = torch.zeros(
            event_table_shape, dtype=torch.bool, device=self.device
        )
        self._s2_next_node_rel_valid_padded = torch.zeros(
            event_table_shape, dtype=torch.bool, device=self.device
        )
        self._s2_next_anchor_valid_padded = torch.zeros(
            event_table_shape, dtype=torch.bool, device=self.device
        )
        self._s2_contact_anchor_valid_padded = torch.zeros(
            event_table_shape, dtype=torch.bool, device=self.device
        )
        for motion_idx, (frames, window_starts, ends) in enumerate(
            zip(
                self._s2_event_frames_by_motion,
                self._s2_event_window_starts_by_motion,
                self._s2_event_ends_by_motion,
                strict=True,
            )
        ):
            if not bool(self.motion.motion_has_ball_demo[motion_idx]):
                continue
            all_slots = torch.arange(frames.numel(), device=self.device)
            self._s2_contact_anchor_ref_padded[motion_idx, all_slots] = self.motion.ball_pos_w[
                motion_idx, frames
            ]
            pelvis_pos = self.motion.body_pos_w[motion_idx, frames, self.motion_anchor_body_index]
            pelvis_quat = self.motion.body_quat_w[motion_idx, frames, self.motion_anchor_body_index]
            ball_ref = self.motion.ball_pos_w[motion_idx, frames]
            pelvis_yaw = yaw_quat(pelvis_quat)
            rel = quat_apply(quat_inv(pelvis_yaw), ball_ref - pelvis_pos)
            self._s2_contact_node_rel_xy_padded[motion_idx, all_slots] = rel[:, :2]
            self._s2_contact_node_rel_z_padded[motion_idx, all_slots] = rel[:, 2]
            self._s2_contact_node_rel_valid_padded[motion_idx, all_slots] = True
            self._s2_contact_anchor_valid_padded[motion_idx, all_slots] = True
            if frames.numel() < 2:
                continue
            next_frames = frames[1:]
            next_window_starts = window_starts[1:]
            next_ends = ends[1:]
            slots = torch.arange(frames.numel() - 1, device=self.device)
            self._s2_next_event_frame_padded[motion_idx, slots] = next_frames
            self._s2_next_event_window_start_frame_padded[
                motion_idx, slots
            ] = next_window_starts
            self._s2_next_event_end_frame_padded[motion_idx, slots] = next_ends
            self._s2_next_anchor_ref_padded[motion_idx, slots] = self.motion.ball_pos_w[
                motion_idx, next_frames
            ]
            self._s2_next_node_rel_xy_padded[motion_idx, slots] = rel[1:, :2]
            self._s2_next_node_rel_z_padded[motion_idx, slots] = rel[1:, 2]
            self._s2_next_node_rel_valid_padded[motion_idx, slots] = True
            self._s2_next_anchor_valid_padded[motion_idx, slots] = True

        # Safe teacher-reset ranges for the second touch.  Never start inside
        # the first contact segment: doing so would teleport into an existing
        # collision and provide no useful recurrent ball-motion history.
        min_history = max(
            1, int(getattr(self.cfg, "dribble_cg_acquisition_min_history_steps", 20))
        )
        max_history = max(
            min_history,
            int(getattr(self.cfg, "dribble_cg_acquisition_max_history_steps", 30)),
        )
        acquisition_candidates: list[tuple[int, int, int, int]] = []
        for motion_idx, frames in enumerate(self._s2_event_frames_by_motion):
            if frames.numel() < 2 or not bool(self.motion.motion_has_ball_demo[motion_idx]):
                continue
            first_frame = int(frames[0].item())
            second_frame = int(frames[1].item())
            contact = self.motion.dribble_cg_contact[motion_idx]
            first_end = first_frame
            motion_length = int(self.motion.file_lengths[motion_idx].item())
            while first_end + 1 < motion_length and int(contact[first_end + 1].item()) > 0:
                first_end += 1
            start_min = max(first_end + 1, second_frame - max_history)
            start_max = second_frame - min_history
            if start_min <= start_max:
                acquisition_candidates.append((motion_idx, 1, start_min, start_max))
        if acquisition_candidates:
            self._s2_acquisition_candidates = torch.as_tensor(
                acquisition_candidates, dtype=torch.long, device=self.device
            )
        else:
            self._s2_acquisition_candidates = torch.empty(
                (0, 4), dtype=torch.long, device=self.device
            )
        # Persistent single-touch statistics drive level-local hard-event
        # replay. They are reset on promotion so clean and disturbed levels do
        # not contaminate one another's reachability estimates.
        event_stat_shape = self._s2_event_frames_padded.shape
        self._s2_event_attempt_count = torch.zeros(
            event_stat_shape, dtype=torch.float32, device=self.device
        )
        self._s2_event_success_count = torch.zeros_like(self._s2_event_attempt_count)
        self._s2_audit_event_attempt_count = torch.zeros_like(
            self._s2_event_attempt_count
        )
        self._s2_audit_event_success_count = torch.zeros_like(
            self._s2_event_attempt_count
        )
        self._s2_hard_replay_enabled = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        configured_levels = tuple(
            getattr(self.cfg, "dribble_cg_curriculum_levels", ())
        )
        requested_counts = {
            int(item[0])
            for level_mix in configured_levels
            for item in tuple(level_mix)
            if int(item[0]) > 0
        }
        # Prefix S2 never starts at an interior event.  Keep one candidate per
        # eligible motion and fix its event index to zero so robot state, ball
        # physics, and recurrent context all share the same frame-0 history.
        requested_counts.update((2, 4, 8))
        self._s2_sequence_candidates: dict[int, torch.Tensor] = {}
        for count in requested_counts:
            candidates: list[tuple[int, int]] = []
            for motion_idx, frames in enumerate(self._s2_event_frames_by_motion):
                final_eligible_frame = (
                    int(self.motion.file_lengths[motion_idx].item()) - 1 - post_grace - 1
                )
                if (
                    int(frames.numel()) >= count
                    and int(self._s2_event_ends_by_motion[motion_idx][count - 1].item())
                    <= final_eligible_frame
                ):
                    candidates.append((motion_idx, 0))
            if candidates:
                self._s2_sequence_candidates[count] = torch.as_tensor(
                    candidates, dtype=torch.long, device=self.device
                )

    def _s2_current_event_label(self, labels: torch.Tensor) -> torch.Tensor:
        frame = torch.minimum(self.time_steps, self.motion.file_lengths[self.motion_idx] - 1)
        value = labels[self.motion_idx, frame]
        raw_event_frame = self._s2_event_frame[self.motion_idx, frame]
        selected_first = self.s2_episode_first_contact_frame
        selected_last = self.s2_episode_last_contact_frame
        selected_curriculum = (selected_first >= 0) & (selected_last >= 0)
        selected_event = (
            (raw_event_frame >= selected_first) & (raw_event_frame <= selected_last)
        )
        return torch.where(
            selected_curriculum & ~selected_event,
            torch.full_like(value, -1),
            value,
        )

    @property
    def s2_contact_event_id_ref(self) -> torch.Tensor:
        return self._s2_current_event_label(self._s2_event_id).to(torch.long)

    @property
    def s2_contact_event_frame_ref(self) -> torch.Tensor:
        return self._s2_current_event_label(self._s2_event_frame).to(torch.long)

    @property
    def s2_contact_event_foot_ref(self) -> torch.Tensor:
        return self._s2_current_event_label(self._s2_event_foot).to(torch.long)

    @property
    def s2_contact_event_side_ref(self) -> torch.Tensor:
        """Expected ball side in the reference foot yaw frame: left=0/right=1."""
        return self._s2_current_event_label(self._s2_event_side).to(torch.long)

    @property
    def s2_contact_event_surface_ref(self) -> torch.Tensor:
        """Expected instep surface from the source label: inside=0/outside=1."""
        return self._s2_current_event_label(self._s2_event_surface).to(torch.long)

    @property
    def s2_contact_window_ref(self) -> torch.Tensor:
        return self.s2_contact_event_id_ref >= 0

    def s2_upcoming_contact_event_ref(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the next selected event's id, frame, foot, and instep surface.

        Unlike the contact-window properties above, this lookup remains valid
        throughout the pre-contact part of an episode.  It exposes only the
        event label; no reference ball or reference foot position is used.
        Invalid entries are returned as ``-1``.
        """
        event_slots = torch.arange(
            self._s2_event_frames_padded.shape[1], device=self.device
        ).unsqueeze(0)
        selected_first = self.s2_episode_first_event_index.unsqueeze(1)
        selected_count = self.s2_episode_contact_count.unsqueeze(1)
        selected = (
            (selected_first >= 0)
            & (event_slots >= selected_first)
            & (event_slots < selected_first + selected_count)
        )
        event_frames = self._s2_event_frames_padded[self.motion_idx]
        upcoming = selected & (event_frames >= self.time_steps.unsqueeze(1))
        frame_delta = torch.where(
            upcoming,
            event_frames - self.time_steps.unsqueeze(1),
            torch.full_like(event_frames, torch.iinfo(event_frames.dtype).max),
        )
        event_slot = torch.argmin(frame_delta, dim=1)
        batch = torch.arange(self.num_envs, device=self.device)
        valid = torch.any(upcoming, dim=1)
        event_frame = event_frames[batch, event_slot]
        event_foot = self.motion.dribble_cg_foot[
            self.motion_idx, event_frame.clamp(min=0)
        ].to(torch.long)
        event_surface = self._s2_event_surface[
            self.motion_idx, event_frame.clamp(min=0)
        ].to(torch.long)
        invalid = torch.full_like(event_slot, -1)
        return (
            torch.where(valid, event_slot, invalid),
            torch.where(valid, event_frame, invalid),
            torch.where(valid, event_foot, invalid),
            torch.where(valid, event_surface, invalid),
        )

    def s2_event_reference(
        self, event_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return frame/end/foot/surface/valid for explicit event slots.

        Unlike the time-driven active/upcoming helpers, this lookup can stay
        on a missed event.  It is therefore the correct source for the critic
        representation of the reward sequence cursor.
        """
        event_id = event_id.to(device=self.device, dtype=torch.long)
        max_slots = self._s2_event_frames_padded.shape[1]
        safe_event = event_id.clamp(min=0, max=max_slots - 1)
        motion_idx = self.motion_idx.to(torch.long)
        event_frame = self._s2_event_frames_padded[motion_idx, safe_event]
        event_end = self._s2_event_end_frames_padded[motion_idx, safe_event]
        safe_frame = event_frame.clamp(min=0)
        event_foot = self.motion.dribble_cg_foot[motion_idx, safe_frame].to(torch.long)
        event_surface = self._s2_event_surface[motion_idx, safe_frame].to(torch.long)
        selected_first = self.s2_episode_first_event_index
        selected_count = self.s2_episode_contact_count
        valid = (
            (event_id >= selected_first)
            & (event_id < selected_first + selected_count)
            & (event_id >= 0)
            & (event_id < max_slots)
            & (event_frame >= 0)
        )
        invalid = torch.full_like(event_id, -1)
        return (
            torch.where(valid, event_frame, invalid),
            torch.where(valid, event_end, invalid),
            torch.where(valid, event_foot, invalid),
            torch.where(valid, event_surface, invalid),
            valid,
        )

    def s2_contact_reference_foot_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the selected event's frozen reference foot position/yaw.

        The event pose is mapped through the same live yaw alignment used by
        motion imitation.  It therefore remains reference-defined without
        introducing a fixed world-forward frame.
        """
        event_frame = self.s2_contact_event_frame_ref.clamp(min=0)
        event_foot = self.s2_contact_event_foot_ref
        left_index = self.cfg.body_names.index("left_ankle_roll_link")
        right_index = self.cfg.body_names.index("right_ankle_roll_link")
        foot_index = torch.where(
            event_foot == 1,
            torch.full_like(event_foot, right_index),
            torch.full_like(event_foot, left_index),
        )
        raw_pos = self.motion.body_pos_w[self.motion_idx, event_frame, foot_index]
        raw_quat = self.motion.body_quat_w[self.motion_idx, event_frame, foot_index]
        raw_pos = raw_pos + self._env.scene.env_origins

        alignment = self._mimic_target_yaw_delta_quat(
            self.anchor_quat_w, self.robot_anchor_quat_w
        )
        mapped_origin = self.robot_anchor_pos_w.clone()
        mapped_origin[:, 2] = self.anchor_pos_w[:, 2]
        mapped_pos = mapped_origin + quat_apply(alignment, raw_pos - self.anchor_pos_w)
        mapped_yaw = yaw_quat(quat_mul(alignment, raw_quat))
        return mapped_pos, mapped_yaw

    def s2_next_touch_target_w(
        self, event_id: torch.Tensor, actual_ball_pos_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the next sparse reference node in world coordinates.

        Returns ``(next_anchor_w, next_event_frame, remaining_time, valid)``.
        Clean tasks gather an anchor precomputed at episode reset.  The live
        policy-root mapping exists only for explicitly enabled historical
        diagnostics/ablations and never participates in the formal reward.
        """
        event_id = event_id.to(device=self.device, dtype=torch.long)
        max_slots = self._s2_next_event_frame_padded.shape[1]
        safe_event = event_id.clamp(min=0, max=max_slots - 1)
        motion_idx = self.motion_idx.to(torch.long)
        current_frame = self._s2_event_frames_padded[motion_idx, safe_event]
        next_frame = self._s2_next_event_frame_padded[motion_idx, safe_event]
        safe_current_frame = current_frame.clamp(min=0)
        safe_next_frame = next_frame.clamp(min=0)
        raw_anchor = self._s2_next_anchor_ref_padded[motion_idx, safe_event]
        current_foot = self.motion.dribble_cg_foot[
            motion_idx, safe_current_frame
        ].to(torch.long)
        next_foot = self.motion.dribble_cg_foot[
            motion_idx, safe_next_frame
        ].to(torch.long)
        valid = (
            (event_id >= 0)
            & (event_id < max_slots)
            & self._s2_next_anchor_valid_padded[motion_idx, safe_event]
            & (current_frame >= 0)
            & (next_frame >= 0)
            & ((current_foot == 0) | (current_foot == 1))
            & ((next_foot == 0) | (next_foot == 1))
        )

        reference_anchor_w = raw_anchor + self._env.scene.env_origins
        batch = torch.arange(self.num_envs, device=self.device)
        fixed_next_anchor_w = self.s2_episode_fixed_next_anchors_w[
            batch, safe_event
        ]
        valid &= self.s2_episode_fixed_next_anchor_valid[batch, safe_event]
        use_fixed = bool(
            getattr(self.cfg, "dribble_cg_episode_fixed_contact_targets", False)
        )
        compare_live = bool(
            getattr(self.cfg, "dribble_cg_debug_live_target_comparison", False)
        )
        if not use_fixed or compare_live:
            alignment = self._mimic_target_yaw_delta_quat(
                self.anchor_quat_w, self.robot_anchor_quat_w
            )
            mapped_origin = self.robot_anchor_pos_w.clone()
            mapped_origin[:, 2] = self.anchor_pos_w[:, 2]
            live_next_anchor_w = mapped_origin + quat_apply(
                alignment, reference_anchor_w - self.anchor_pos_w
            )
            live_next_anchor_w[:, 2] = (
                raw_anchor[:, 2] + self._env.scene.env_origins[:, 2]
            )
        else:
            # Avoid even constructing the endogenous target in formal runs.
            live_next_anchor_w = fixed_next_anchor_w

        # Numerical closure: map the fixed world target back through the
        # immutable episode transform.  This must recover the reference I3
        # anchor (XY) and detects a fixed-but-wrong task-frame transform.
        recovered_reference_xy = self.s2_episode_reference_root_w[:, :2] + quat_apply(
            quat_inv(self.s2_episode_yaw_alignment),
            torch.cat(
                (
                    fixed_next_anchor_w[:, :2] - self.s2_episode_actual_root_w[:, :2],
                    torch.zeros_like(fixed_next_anchor_w[:, :1]),
                ),
                dim=-1,
            ),
        )[:, :2]
        fixed_inverse_error = torch.linalg.vector_norm(
            recovered_reference_xy - reference_anchor_w[:, :2], dim=-1
        )
        self.s2_last_live_next_anchor_w.copy_(live_next_anchor_w)
        self.s2_last_fixed_next_anchor_w.copy_(fixed_next_anchor_w)
        self.s2_last_fixed_target_valid.copy_(valid)
        self.s2_last_fixed_inverse_error.copy_(
            torch.where(valid, fixed_inverse_error, torch.zeros_like(fixed_inverse_error))
        )
        next_anchor_w = fixed_next_anchor_w if use_fixed else live_next_anchor_w
        fps = max(float(self.motion.fps.reshape(-1)[0]), 1.0e-6)
        remaining_time = (
            next_frame - self.time_steps.to(torch.long)
        ).to(torch.float32) / fps
        distance = torch.linalg.vector_norm(
            next_anchor_w[:, :2] - actual_ball_pos_w[:, :2], dim=-1
        )
        valid &= (remaining_time > 0.0) & torch.isfinite(distance)
        invalid_frame = torch.full_like(next_frame, -1)
        return (
            next_anchor_w,
            torch.where(valid, next_frame, invalid_frame),
            torch.where(valid, remaining_time, torch.zeros_like(remaining_time)),
            valid,
        )

    def s2_next_touch_node_relative_xy(
        self, event_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return immutable next-node coordinates in the reference pelvis-yaw frame."""
        event_id = event_id.to(device=self.device, dtype=torch.long)
        max_slots = self._s2_next_node_rel_xy_padded.shape[1]
        safe_event = event_id.clamp(min=0, max=max_slots - 1)
        motion_idx = self.motion_idx.to(torch.long)
        node_xy = self._s2_next_node_rel_xy_padded[motion_idx, safe_event]
        node_z = self._s2_next_node_rel_z_padded[motion_idx, safe_event]
        valid = (
            (event_id >= 0)
            & (event_id < max_slots)
            & self._s2_next_node_rel_valid_padded[motion_idx, safe_event]
        )
        return node_xy, node_z, valid

    def s2_next_touch_window_end_frame(self, event_id: torch.Tensor) -> torch.Tensor:
        """Return the labelled end frame of the event after ``event_id``."""
        event_id = event_id.to(device=self.device, dtype=torch.long)
        max_slots = self._s2_next_event_end_frame_padded.shape[1]
        safe_event = event_id.clamp(min=0, max=max_slots - 1)
        motion_idx = self.motion_idx.to(torch.long)
        next_end = self._s2_next_event_end_frame_padded[motion_idx, safe_event]
        valid = (
            (event_id >= 0)
            & (event_id < max_slots)
            & (next_end >= 0)
        )
        return torch.where(valid, next_end, torch.full_like(next_end, -1))

    def s2_next_touch_window_start_frame(self, event_id: torch.Tensor) -> torch.Tensor:
        """Return the tolerance-expanded start of the next contact window."""
        event_id = event_id.to(device=self.device, dtype=torch.long)
        max_slots = self._s2_next_event_window_start_frame_padded.shape[1]
        safe_event = event_id.clamp(min=0, max=max_slots - 1)
        motion_idx = self.motion_idx.to(torch.long)
        next_start = self._s2_next_event_window_start_frame_padded[
            motion_idx, safe_event
        ]
        valid = (
            (event_id >= 0)
            & (event_id < max_slots)
            & (next_start >= 0)
        )
        return torch.where(valid, next_start, torch.full_like(next_start, -1))

    def s2_contact_anchor_target_w(
        self, event_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the current event's sparse ball anchor in world coordinates.

        Clean S2 tasks map the annotated ball centre through the immutable
        reset-time reference transform.  The target must not move when the
        policy translates or rotates its pelvis/foot.  Legacy tasks may still
        request the historical actual-foot-local mapping for old ablations.

        This remains an event-only target; no intermediate ball trajectory or
        reference ball velocity is constructed.
        """
        event_id = event_id.to(device=self.device, dtype=torch.long)
        max_slots = self._s2_contact_anchor_ref_padded.shape[1]
        safe_event = event_id.clamp(min=0, max=max_slots - 1)
        motion_idx = self.motion_idx.to(torch.long)
        event_frame = self._s2_event_frames_padded[motion_idx, safe_event]
        safe_frame = event_frame.clamp(min=0)
        event_foot = self.motion.dribble_cg_foot[
            motion_idx, safe_frame
        ].to(torch.long)
        valid = (
            (event_id >= 0)
            & (event_id < max_slots)
            & self._s2_contact_anchor_valid_padded[motion_idx, safe_event]
            & (event_frame >= 0)
            & ((event_foot == 0) | (event_foot == 1))
        )

        left_index = self.cfg.body_names.index("left_ankle_roll_link")
        right_index = self.cfg.body_names.index("right_ankle_roll_link")
        foot_index = torch.where(
            event_foot == 1,
            torch.full_like(event_foot, right_index),
            torch.full_like(event_foot, left_index),
        )

        reference_foot_pos = self.motion.body_pos_w[
            motion_idx, safe_frame, foot_index
        ]
        reference_foot_quat = self.motion.body_quat_w[
            motion_idx, safe_frame, foot_index
        ]
        reference_anchor = self._s2_contact_anchor_ref_padded[
            motion_idx, safe_event
        ]
        if bool(getattr(self.cfg, "dribble_cg_episode_fixed_contact_targets", False)):
            batch = torch.arange(self.num_envs, device=self.device)
            anchor_w = self.s2_episode_fixed_contact_anchors_w[batch, safe_event]
            valid &= self.s2_episode_fixed_contact_anchor_valid[batch, safe_event]
            return anchor_w, valid

        # Historical ablations represented the annotated contact location as
        # a local offset attached to the policy-controlled actual foot.
        anchor_local = quat_apply(
            quat_inv(reference_foot_quat), reference_anchor - reference_foot_pos
        )

        batch = torch.arange(self.num_envs, device=self.device)
        actual_foot_pos = self.robot_body_pos_w[batch, foot_index]
        actual_foot_quat = self.robot_body_quat_w[batch, foot_index]
        anchor_w = actual_foot_pos + quat_apply(
            actual_foot_quat, anchor_local
        )
        return anchor_w, valid

    def s2_contact_reference_anchor_target_w(
        self, event_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map the raw labelled contact anchor with the live motion alignment.

        This read-only diagnostic target is intentionally distinct from
        :meth:`s2_contact_anchor_target_w`, which places the invariant local
        anchor on the *actual* simulated foot for reward shaping.  Comparing
        both targets exposes root/yaw or tracking mismatches without changing
        any reward, contact gate, cursor, reset, or curriculum behavior.
        """
        event_id = event_id.to(device=self.device, dtype=torch.long)
        max_slots = self._s2_contact_anchor_ref_padded.shape[1]
        safe_event = event_id.clamp(min=0, max=max_slots - 1)
        motion_idx = self.motion_idx.to(torch.long)
        event_frame = self._s2_event_frames_padded[motion_idx, safe_event]
        valid = (
            (event_id >= 0)
            & (event_id < max_slots)
            & self._s2_contact_anchor_valid_padded[motion_idx, safe_event]
            & (event_frame >= 0)
        )
        raw_anchor = self._s2_contact_anchor_ref_padded[motion_idx, safe_event]
        reference_anchor_w = raw_anchor + self._env.scene.env_origins
        alignment = self._mimic_target_yaw_delta_quat(
            self.anchor_quat_w, self.robot_anchor_quat_w
        )
        mapped_origin = self.robot_anchor_pos_w.clone()
        mapped_origin[:, 2] = self.anchor_pos_w[:, 2]
        mapped_anchor = mapped_origin + quat_apply(
            alignment, reference_anchor_w - self.anchor_pos_w
        )
        mapped_anchor[:, 2] = raw_anchor[:, 2] + self._env.scene.env_origins[:, 2]
        return mapped_anchor, valid

    def _sample_s2_contact_curriculum(self, env_ids: torch.Tensor) -> bool:
        """Sample a contact-count episode; return false when the feature is off."""
        levels = tuple(getattr(self.cfg, "dribble_cg_curriculum_levels", ()))
        if levels:
            level = min(max(0, int(self.s2_curriculum_level.item())), len(levels) - 1)
            mix = tuple(levels[level])
        else:
            return False
        counts = torch.as_tensor([int(item[0]) for item in mix], device=self.device)
        probabilities = torch.as_tensor(
            [float(item[1]) for item in mix], dtype=torch.float32, device=self.device
        )
        if bool(torch.any(probabilities < 0.0)) or float(probabilities.sum().item()) <= 0.0:
            raise ValueError(
                "Active dribble_cg_curriculum_levels probabilities must be non-negative with positive sum"
            )
        probabilities = probabilities / probabilities.sum()
        sampled_counts = counts[torch.multinomial(probabilities, env_ids.numel(), replacement=True)]
        post_grace = max(0, int(getattr(self.cfg, "dribble_cg_missed_contact_grace_steps", 3)))
        release_tail = max(
            0, int(getattr(self.cfg, "dribble_cg_release_window_steps", 8))
        )
        lookahead_contact_phases = max(
            0,
            int(
                getattr(
                    self.cfg,
                    "dribble_cg_curriculum_lookahead_contact_phases",
                    0,
                )
            ),
        )
        lookahead_start_level = max(
            0,
            int(
                getattr(
                    self.cfg,
                    "dribble_cg_curriculum_lookahead_start_level",
                    0,
                )
            ),
        )
        lookahead_levels = tuple(
            int(value)
            for value in getattr(
                self.cfg,
                "dribble_cg_curriculum_lookahead_levels",
                (),
            )
        )
        if lookahead_levels:
            if level not in lookahead_levels:
                lookahead_contact_phases = 0
        elif level < lookahead_start_level:
            lookahead_contact_phases = 0
        self.s2_episode_reset_mode[env_ids] = 0
        self.s2_episode_acquisition_history_steps[env_ids] = 0

        for requested_count_tensor in torch.unique(sampled_counts):
            requested_count = int(requested_count_tensor.item())
            group_mask = sampled_counts == requested_count
            group_env_ids = env_ids[group_mask]
            group_size = group_env_ids.numel()
            if group_size == 0:
                continue

            # Level 0 mixes real frame-0 two-touch episodes with independent
            # teacher resets before the second event.  Acquisition samples
            # are diagnostic/training-only and are excluded from promotion.
            acquisition_probability = min(
                1.0, max(0.0, float(self._s2_acquisition_probability.item()))
            )
            acquisition_mask = torch.zeros(
                group_size, dtype=torch.bool, device=self.device
            )
            if (
                level == 0
                and requested_count >= 2
                and acquisition_probability > 0.0
                and self._s2_acquisition_candidates.numel() > 0
            ):
                acquisition_mask = (
                    torch.rand(group_size, device=self.device) < acquisition_probability
                )
            acquisition_ids = group_env_ids[acquisition_mask]
            if acquisition_ids.numel() > 0:
                candidate_rows = self._s2_acquisition_candidates[
                    torch.randint(
                        self._s2_acquisition_candidates.shape[0],
                        (acquisition_ids.numel(),),
                        device=self.device,
                    )
                ]
                selected_motion = candidate_rows[:, 0]
                target_event_index = candidate_rows[:, 1]
                start_min = candidate_rows[:, 2]
                start_max = candidate_rows[:, 3]
                span = (start_max - start_min + 1).clamp(min=1)
                start_frame = start_min + torch.floor(
                    torch.rand(acquisition_ids.numel(), device=self.device)
                    * span.to(torch.float32)
                ).to(torch.long)
                target_frame = self._s2_event_frames_padded[
                    selected_motion, target_event_index
                ]
                target_end_frame = self._s2_event_end_frames_padded[
                    selected_motion, target_event_index
                ]
                true_length = self.motion.file_lengths[selected_motion]
                end_frame = torch.minimum(
                    true_length - 1,
                    target_end_frame + post_grace + release_tail + 1,
                )
                self.motion_idx[acquisition_ids] = selected_motion
                self.motion_length[acquisition_ids] = end_frame + 1
                self.time_steps[acquisition_ids] = start_frame
                self.s2_episode_first_contact_frame[acquisition_ids] = target_frame
                self.s2_episode_last_contact_frame[acquisition_ids] = target_frame
                self.s2_episode_contact_count[acquisition_ids] = 1
                self.s2_episode_first_event_index[acquisition_ids] = target_event_index
                self.s2_episode_curriculum_level[acquisition_ids] = level
                self.s2_episode_curriculum_audit[acquisition_ids] = False
                self.s2_episode_reset_mode[acquisition_ids] = 1
                self.s2_episode_acquisition_history_steps[acquisition_ids] = (
                    target_frame - start_frame
                )

            group_env_ids = group_env_ids[~acquisition_mask]
            group_size = group_env_ids.numel()
            if group_size == 0:
                continue
            uniform_audit = torch.ones(
                group_size, dtype=torch.bool, device=self.device
            )

            selected_motion = torch.empty(group_size, dtype=torch.long, device=self.device)
            sequence_start = torch.zeros(group_size, dtype=torch.long, device=self.device)
            if requested_count == 0:
                eligible_motion = torch.nonzero(
                    self._s2_event_count_by_motion > 0, as_tuple=False
                ).squeeze(-1)
                if eligible_motion.numel() == 0:
                    raise ValueError("No motion contains an S2 contact event")
                selected_motion[:] = eligible_motion[
                    torch.randint(eligible_motion.numel(), (group_size,), device=self.device)
                ]
                actual_count = self._s2_event_count_by_motion[selected_motion]
            else:
                candidates = self._s2_sequence_candidates.get(requested_count)
                if candidates is None or candidates.numel() == 0:
                    raise ValueError(
                        f"No motion contains a {requested_count}-contact curriculum sequence"
                    )
                sampled_indices = torch.randint(
                    candidates.shape[0], (group_size,), device=self.device
                )
                # Once level-local performance has plateaued, split single-
                # contact episodes into a uniform promotion audit and a hard-
                # event training pool.  Only the audit pool is allowed to
                # update promotion statistics, so replay cannot lower its own
                # measured pass rate by oversampling weak events.
                hard_replay = bool(
                    requested_count == 1
                    and int(getattr(self, "_s2_hard_replay_enabled", torch.zeros(())).item()) > 0
                )
                if hard_replay:
                    attempts = self._s2_event_attempt_count[
                        candidates[:, 0], candidates[:, 1]
                    ]
                    successes = self._s2_event_success_count[
                        candidates[:, 0], candidates[:, 1]
                    ]
                    min_attempts = max(
                        1, int(getattr(self.cfg, "dribble_cg_hard_replay_min_attempts", 30))
                    )
                    eligible = torch.nonzero(
                        attempts >= float(min_attempts), as_tuple=False
                    ).squeeze(-1)
                    if eligible.numel() > 0:
                        hard_fraction = min(
                            1.0,
                            max(
                                0.0,
                                float(getattr(self.cfg, "dribble_cg_hard_replay_fraction", 0.30)),
                            ),
                        )
                        hard_count = max(
                            1,
                            min(
                                int(eligible.numel()),
                                int(float(eligible.numel()) * hard_fraction + 0.999999),
                            ),
                        )
                        # A Beta(1, 1) posterior mean avoids extreme rankings
                        # from a small number of binary outcomes.
                        success_rate = (successes[eligible] + 1.0) / (
                            attempts[eligible] + 2.0
                        )
                        hard_local = torch.topk(
                            success_rate, k=hard_count, largest=False
                        ).indices
                        hard_indices = eligible[hard_local]
                        audit_probability = min(
                            1.0,
                            max(
                                0.0,
                                float(
                                    getattr(
                                        self.cfg, "dribble_cg_curriculum_audit_probability", 0.25
                                    )
                                ),
                            ),
                        )
                        uniform_audit = (
                            torch.rand(group_size, device=self.device) < audit_probability
                        )
                        hard_draw = ~uniform_audit
                        hard_draw_count = int(hard_draw.sum().item())
                        if hard_draw_count > 0:
                            sampled_indices[hard_draw] = hard_indices[
                                torch.randint(
                                    hard_indices.numel(),
                                    (hard_draw_count,),
                                    device=self.device,
                                )
                            ]
                sampled = candidates[sampled_indices]
                selected_motion = sampled[:, 0]
                sequence_start = sampled[:, 1]
                actual_count = torch.full_like(selected_motion, requested_count)

            first_event = self._s2_event_frames_padded[selected_motion, sequence_start]
            last_event = self._s2_event_frames_padded[
                selected_motion, sequence_start + actual_count - 1
            ]
            last_event_end = self._s2_event_end_frames_padded[
                selected_motion, sequence_start + actual_count - 1
            ]
            true_length = self.motion.file_lengths[selected_motion]
            if requested_count == 0:
                start_frame = torch.zeros_like(first_event)
                end_frame = true_length - 1
            else:
                start_frame = torch.zeros_like(first_event)
                default_end_frame = torch.minimum(
                    true_length - 1,
                    last_event_end + post_grace + release_tail + 1,
                )
                # Do not add the lookahead event to ``actual_count`` or the
                # selected event range.  It supplies only enough physical time
                # to observe the outgoing result of the last required touch.
                last_selected_slot = sequence_start + actual_count - 1
                lookahead_slot = last_selected_slot + lookahead_contact_phases
                event_count = self._s2_event_count_by_motion[selected_motion]
                has_lookahead = (
                    (lookahead_contact_phases > 0)
                    & (lookahead_slot >= 0)
                    & (lookahead_slot < event_count)
                )
                safe_lookahead_slot = torch.minimum(
                    lookahead_slot.clamp(min=0),
                    (event_count - 1).clamp(min=0),
                )
                lookahead_event_end = self._s2_event_end_frames_padded[
                    selected_motion, safe_lookahead_slot
                ]
                lookahead_end_frame = torch.minimum(
                    true_length - 1,
                    lookahead_event_end + post_grace + 1,
                )
                end_frame = torch.where(
                    has_lookahead, lookahead_end_frame, default_end_frame
                )

            self.motion_idx[group_env_ids] = selected_motion
            self.motion_length[group_env_ids] = end_frame + 1
            self.time_steps[group_env_ids] = start_frame
            self.s2_episode_first_contact_frame[group_env_ids] = first_event
            self.s2_episode_last_contact_frame[group_env_ids] = last_event
            self.s2_episode_contact_count[group_env_ids] = actual_count
            self.s2_episode_first_event_index[group_env_ids] = sequence_start
            self.s2_episode_curriculum_level[group_env_ids] = level
            self.s2_episode_curriculum_audit[group_env_ids] = uniform_audit
            self.s2_episode_reset_mode[group_env_ids] = 0
            self.s2_episode_acquisition_history_steps[group_env_ids] = 0
        return True

    def _uniform_sampling(self, env_ids: Sequence[int]):
        ids = self._to_env_id_tensor(env_ids)
        if not hasattr(self, "_s2_event_frames_by_motion"):
            super()._uniform_sampling(ids)
            return
        if self._sample_s2_contact_curriculum(ids):
            return
        super()._uniform_sampling(ids)
        self.s2_episode_first_contact_frame[ids] = -1
        self.s2_episode_last_contact_frame[ids] = -1
        self.s2_episode_contact_count[ids] = 0
        self.s2_episode_first_event_index[ids] = -1
        self.s2_episode_curriculum_level[ids] = int(self.s2_curriculum_level.item())
        self.s2_episode_curriculum_audit[ids] = True
        self.s2_episode_reset_mode[ids] = 0
        self.s2_episode_acquisition_history_steps[ids] = 0

    def _refresh_s2_episode_fixed_targets(self, env_ids: torch.Tensor) -> None:
        """Precompute every sparse world target from the reset-time transform."""
        if env_ids.numel() == 0:
            return
        motion_idx = self.motion_idx[env_ids].to(torch.long)
        env_origins = self._env.scene.env_origins[env_ids]
        reference_root = self.s2_episode_reference_root_w[env_ids]
        actual_root = self.s2_episode_actual_root_w[env_ids]
        alignment = self.s2_episode_yaw_alignment[env_ids]
        slot_count = self._s2_contact_anchor_ref_padded.shape[1]

        def _map_table(raw_table: torch.Tensor) -> torch.Tensor:
            raw = raw_table[motion_idx]
            reference_w = raw + env_origins.unsqueeze(1)
            delta = reference_w - reference_root.unsqueeze(1)
            delta[:, :, 2] = 0.0
            alignment_slots = alignment.unsqueeze(1).expand(-1, slot_count, -1)
            mapped = actual_root.unsqueeze(1) + quat_apply(
                alignment_slots.reshape(-1, 4), delta.reshape(-1, 3)
            ).reshape(env_ids.numel(), slot_count, 3)
            mapped[:, :, 2] = raw[:, :, 2] + env_origins[:, None, 2]
            return mapped

        self.s2_episode_fixed_contact_anchors_w[env_ids] = _map_table(
            self._s2_contact_anchor_ref_padded
        )
        self.s2_episode_fixed_next_anchors_w[env_ids] = _map_table(
            self._s2_next_anchor_ref_padded
        )
        self.s2_episode_fixed_contact_anchor_valid[env_ids] = (
            self._s2_contact_anchor_valid_padded[motion_idx]
        )
        self.s2_episode_fixed_next_anchor_valid[env_ids] = (
            self._s2_next_anchor_valid_padded[motion_idx]
        )

    def _resample_command(self, env_ids: Sequence[int]):
        ids = self._to_env_id_tensor(env_ids)
        super()._resample_command(ids)
        if ids.numel() > 0:
            self._realign_reference_first_contact_ball_after_robot_reset(ids)
            # Use the exact configured imitation anchor (pelvis for G1), not
            # an assumed body slot or articulation root.  This makes the fixed
            # transform algebraically identical to the live mapping at reset.
            reference_root_pos_w = self.anchor_pos_w[ids]
            reference_root_quat_w = self.anchor_quat_w[ids]
            actual_root_pos_w = self.robot_anchor_pos_w[ids]
            actual_root_quat_w = self.robot_anchor_quat_w[ids]
            self.s2_episode_reference_root_w[ids] = reference_root_pos_w
            self.s2_episode_actual_root_w[ids] = actual_root_pos_w
            self.s2_episode_yaw_alignment[ids] = self._mimic_target_yaw_delta_quat(
                reference_root_quat_w, actual_root_quat_w
            )
            if hasattr(self, "s2_episode_fixed_contact_anchors_w"):
                self._refresh_s2_episode_fixed_targets(ids)
            self.s2_episode_generation[ids] += 1
            self.s2_sequence_cursor[ids] = self.s2_episode_first_event_index[ids]
            self.s2_sequence_contact_succeeded[ids] = False
            self.s2_locked_next_anchor_w[ids] = 0.0
            self.s2_locked_next_anchor_valid[ids] = False

    def _realign_reference_first_contact_ball_after_robot_reset(
        self, env_ids: torch.Tensor
    ) -> None:
        """Apply the Flat reset's live rigid transform to the contact ball.

        ``ball_pos_w`` and the motion bodies share the source-reference world
        frame.  Flat reset may rotate the robot to task-forward (and perturb
        its root pose), after which motion imitation maps every reference body
        through that live transform.  The persistent first-contact ball must
        receive the same transform; otherwise the robot follows a rotated
        reference while the ball remains at the raw NPZ world coordinate.

        This runs after ``super()._resample_command`` has written the new
        robot root state.  It changes only the initial physical ball pose and
        its derived target/destination observations; no reward or contact
        definition is involved.
        """
        if env_ids.numel() == 0 or not self._uses_reference_first_contact_spawn():
            return
        env_origins = self._env.scene.env_origins[env_ids]
        reference_positions, valid, _contact_frames = (
            self._reference_first_contact_ball_positions(env_ids)
        )
        fallback_positions = self._anchor_local_front_ball_positions(env_ids)
        source_positions = torch.where(
            valid.unsqueeze(-1), reference_positions, fallback_positions
        )
        source_positions_w = source_positions + env_origins

        # Body slot zero is the same reference root used by the parent reset
        # when it constructs and writes ``root_pos``/``root_ori``.
        reference_root_pos_w = self.body_pos_w[env_ids, 0]
        reference_root_quat_w = self.body_quat_w[env_ids, 0]
        actual_root_pos_w = self.robot.data.root_pos_w[env_ids]
        actual_root_quat_w = self.robot.data.root_quat_w[env_ids]
        alignment = self._mimic_target_yaw_delta_quat(
            reference_root_quat_w, actual_root_quat_w
        )
        mapped_positions_w = actual_root_pos_w + quat_apply(
            alignment, source_positions_w - reference_root_pos_w
        )
        # Ground owns the ball height; yaw/planar reset alignment must not
        # rotate the source height into another axis.
        mapped_positions_w[:, 2] = source_positions[:, 2] + env_origins[:, 2]
        self.soccer_ball_pos[env_ids] = mapped_positions_w - env_origins
        self._update_soccer_ball(env_ids)
        self._update_target_points(env_ids)
        if bool(getattr(self.cfg, "soccer_ball_generate_destination", True)):
            self._update_destination_points(env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        regions_only = bool(
            getattr(self.cfg, "dribble_cg_s2_debug_regions_only", False)
        )
        if not regions_only:
            super()._set_debug_vis_impl(debug_vis)
        if debug_vis:
            if not hasattr(self, "s2_contact_event_foot_visualizer"):
                self.s2_contact_event_foot_visualizer = VisualizationMarkers(
                    self.cfg.body_visualizer_cfg.replace(
                        prim_path="/Visuals/S2Contact/reference_foot"
                    )
                )
                self.s2_contact_region_visualizer = VisualizationMarkers(
                    self.cfg.body_visualizer_cfg.replace(
                        prim_path="/Visuals/S2Contact/region_bounds"
                    )
                )
                self.s2_fixed_target_visualizer = VisualizationMarkers(
                    self.cfg.s2_fixed_target_marker_cfg
                )
                self.s2_live_target_visualizer = VisualizationMarkers(
                    self.cfg.s2_live_target_marker_cfg
                )
            self.s2_contact_event_foot_visualizer.set_visibility(True)
            self.s2_contact_region_visualizer.set_visibility(True)
            self.s2_fixed_target_visualizer.set_visibility(True)
            self.s2_live_target_visualizer.set_visibility(True)
        elif hasattr(self, "s2_contact_event_foot_visualizer"):
            self.s2_contact_event_foot_visualizer.set_visibility(False)
            self.s2_contact_region_visualizer.set_visibility(False)
            self.s2_fixed_target_visualizer.set_visibility(False)
            self.s2_live_target_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        if not bool(getattr(self.cfg, "dribble_cg_s2_debug_regions_only", False)):
            super()._debug_vis_callback(event)
        if not hasattr(self, "s2_contact_event_foot_visualizer"):
            return
        foot_pos, foot_yaw = self.s2_contact_reference_foot_pose_w()
        active = self.s2_contact_window_ref
        side = self.s2_contact_event_side_ref
        hidden_foot_pos = foot_pos.clone()
        hidden_foot_pos[~active, 2] = -100.0
        self.s2_contact_event_foot_visualizer.visualize(hidden_foot_pos, foot_yaw)

        # Four frames show the coarse front ramp (-5 to +5 cm), the 4 cm
        # lateral boundary, and the unbounded expected-side direction. +Y is
        # the foot's local left axis; these are guides, not a target rectangle.
        side_sign = torch.where(side == 0, torch.ones_like(side), -torch.ones_like(side)).to(foot_pos.dtype)
        local_corners = foot_pos.new_zeros((self.num_envs, 4, 3))
        local_corners[:, 0, 0] = -0.05
        local_corners[:, 1, 0] = 0.05
        local_corners[:, 2, 1] = side_sign * 0.04
        local_corners[:, 3, 1] = side_sign * 0.20
        corner_yaw = foot_yaw[:, None, :].expand(-1, 4, -1)
        corners = foot_pos[:, None, :] + quat_apply(
            corner_yaw.reshape(-1, 4), local_corners.reshape(-1, 3)
        ).reshape(self.num_envs, 4, 3)
        corners[~active, :, 2] = -100.0
        self.s2_contact_region_visualizer.visualize(
            corners.reshape(-1, 3), corner_yaw.reshape(-1, 4)
        )

        hidden_fixed = self.s2_last_fixed_next_anchor_w.clone()
        hidden_live = self.s2_last_live_next_anchor_w.clone()
        hidden_fixed[~self.s2_last_fixed_target_valid, 2] = -100.0
        hidden_live[~self.s2_last_fixed_target_valid, 2] = -100.0
        self.s2_fixed_target_visualizer.visualize(hidden_fixed)
        self.s2_live_target_visualizer.visualize(hidden_live)

    def _validate_fixed_touch_spec(self) -> None:
        """Reject CG clips that conflict with a fixed foot and/or instep region.

        A fixed-touch task cannot mix labels that describe another foot or a
        different contact surface.  Failing during environment creation makes
        the data-contract error explicit; callers must relabel or omit the
        conflicting motion instead of silently changing the learned contact
        graph.
        """
        configured_foot = getattr(self.cfg, "dribble_cg_fixed_touch_foot", None)
        configured_surface = getattr(self.cfg, "dribble_cg_fixed_touch_surface", None)
        require_surface_labels = bool(getattr(self.cfg, "dribble_cg_require_surface_labels", False))
        require_flow_labels = bool(getattr(self.cfg, "dribble_cg_require_flow_labels", False))

        foot_to_id = {"left": 0, "right": 1}
        surface_to_id = {"inside_instep": 0, "outside_instep": 1}

        foot_name: str | None = None
        if configured_foot is not None:
            foot_name = str(configured_foot).lower().strip()
        if foot_name is not None and foot_name not in foot_to_id:
            raise ValueError(
                "dribble_cg_fixed_touch_foot must be 'left', 'right', or None; "
                f"got {configured_foot!r}."
            )

        surface_name: str | None = None
        if configured_surface is not None:
            surface_name = str(configured_surface).lower().strip()
        if surface_name is not None and surface_name not in surface_to_id:
            raise ValueError(
                "dribble_cg_fixed_touch_surface must be 'inside_instep', "
                "'outside_instep', or None; "
                f"got {configured_surface!r}."
            )

        if foot_name is None and surface_name is None and not require_surface_labels and not require_flow_labels:
            return

        expected_foot_id = foot_to_id[foot_name] if foot_name is not None else None
        expected_surface_id = surface_to_id[surface_name] if surface_name is not None else None
        violations: list[str] = []
        for motion_idx, motion_name in enumerate(self.motion.motion_name):
            contact = self.motion.dribble_cg_contact[motion_idx] > 0
            contact_foot = self.motion.dribble_cg_foot[motion_idx]
            contact_surface = self.motion.dribble_cg_surface[motion_idx]
            distance_foot = self.motion.dribble_cg_dist_foot[motion_idx]
            kick_leg_id = int(self.motion_kick_leg[motion_idx].item())
            if require_surface_labels:
                if not bool(self.motion.motion_has_dribble_cg[motion_idx]):
                    violations.append(f"{motion_name}: missing CG contact labels")
                invalid_contact_foot = contact & ~((contact_foot == 0) | (contact_foot == 1))
                invalid_contact_surface = contact & ~(
                    (contact_surface == 0) | (contact_surface == 1)
                )
                if bool(torch.any(invalid_contact_foot)):
                    violations.append(f"{motion_name}: contact frames without a valid foot label")
                if bool(torch.any(invalid_contact_surface)):
                    violations.append(
                        f"{motion_name}: contact frames without a valid instep-surface label"
                    )
            if require_flow_labels:
                if not bool(self.motion.motion_has_dribble_cg[motion_idx]):
                    violations.append(f"{motion_name}: missing CG contact labels")
                elif not bool(self.motion.motion_has_dribble_cg_flow[motion_idx]):
                    violations.append(f"{motion_name}: missing contact-to-contact flow labels")
            if expected_foot_id is not None:
                unknown_contact_foot = contact & (contact_foot < 0)
                wrong_contact_foot = contact & (contact_foot >= 0) & (contact_foot != expected_foot_id)
                wrong_distance_foot = (distance_foot >= 0) & (distance_foot != expected_foot_id)
                wrong_kick_leg = kick_leg_id >= 0 and kick_leg_id != expected_foot_id
                if bool(torch.any(unknown_contact_foot)):
                    violations.append(f"{motion_name}: contact frames without a foot label")
                if bool(torch.any(wrong_contact_foot)):
                    violations.append(f"{motion_name}: contact labels include the other foot")
                if bool(torch.any(wrong_distance_foot)):
                    violations.append(f"{motion_name}: distance labels include the other foot")
                if wrong_kick_leg:
                    violations.append(f"{motion_name}: kick_leg disagrees with fixed {foot_name} foot")

            if expected_surface_id is not None:
                unknown_contact_surface = contact & (contact_surface < 0)
                wrong_contact_surface = contact & (contact_surface >= 0) & (
                    contact_surface != expected_surface_id
                )
                if bool(torch.any(unknown_contact_surface)):
                    violations.append(f"{motion_name}: contact frames without an instep-surface label")
                if bool(torch.any(wrong_contact_surface)):
                    violations.append(f"{motion_name}: contact labels include the other instep surface")
        if violations:
            configured = " ".join(part for part in (foot_name, surface_name) if part is not None)
            task_description = (
                f"Fixed {configured} CG task" if configured else "Labelled CG task"
            )
            raise ValueError(
                f"{task_description} received incompatible motions: " + "; ".join(violations)
            )

    def _demo_ball_world(self, env_ids: torch.Tensor) -> torch.Tensor:
        """World-frame demo ball positions for env_ids (aligned to anchor tracking)."""
        mi = self.motion_idx[env_ids]
        ts = self.time_steps[env_ids]
        env_origins = self._env.scene.env_origins[env_ids]

        mb = self.motion.ball_pos_w[mi, ts] + env_origins
        ma = self.anchor_pos_w[env_ids]
        ra = self.robot_anchor_pos_w[env_ids]
        delta = ra.clone()
        delta[:, 2] = ma[:, 2]

        anchor_quat = self.anchor_quat_w[env_ids]
        robot_quat = self.robot_anchor_quat_w[env_ids]
        dq = mimic_anchor_yaw_delta_quat(
            anchor_quat,
            robot_quat,
            align_task_frame=bool(getattr(self.cfg, "mimic_align_task_frame", False)),
        )
        rel = mb - ma
        return delta + quat_apply(dq, rel)

    def _demo_ball_lin_vel_w(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Finite-difference demo ball linear velocity in world frame."""
        mi = self.motion_idx[env_ids]
        ts = self.time_steps[env_ids]
        ts_prev = torch.clamp(ts - 1, min=0)
        env_origins = self._env.scene.env_origins[env_ids]

        mb0 = self.motion.ball_pos_w[mi, ts_prev] + env_origins
        mb1 = self.motion.ball_pos_w[mi, ts] + env_origins
        fps = float(self.motion.fps.reshape(-1)[0])
        return (mb1 - mb0) * fps

    def _should_snap_demo_ball(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Per-env bool: write sim ball from demo this step."""
        # A first-contact spawn is a physical initial condition.  Snapping it
        # to the demo trajectory would immediately replace that world-space
        # target with ``ball_pos_w[current_frame]`` (usually frame 0).  Make
        # the two modes mutually exclusive here instead of relying on every
        # stage config to repeat ``dribble_cg_snap_mode = "never"``.
        if self._uses_reference_first_contact_spawn():
            return torch.zeros(env_ids.numel(), device=self.device, dtype=torch.bool)

        mi = self.motion_idx[env_ids]
        has_demo = self.motion.motion_has_ball_demo[mi]
        mode = str(getattr(self.cfg, "dribble_cg_snap_mode", "full")).lower().strip()
        if mode == "never":
            return torch.zeros(env_ids.numel(), device=self.device, dtype=torch.bool)
        if mode == "non_contact_only":
            in_ref_contact = self.motion.dribble_cg_contact[mi, self.time_steps[env_ids]] > 0
            return has_demo & ~in_ref_contact
        return has_demo

    def _uses_reference_first_contact_spawn(self) -> bool:
        """Whether reset placement owns a persistent physical ball."""
        spawn_mode = str(
            getattr(self.cfg, "dribble_cg_ball_spawn_mode", "legacy")
        ).lower().strip()
        return spawn_mode in _REFERENCE_FIRST_CONTACT_SPAWN_MODES

    def _anchor_local_front_ball_positions(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Fallback ball ahead of the frame-0 reference pelvis in pelvis-local axes."""
        mi = self.motion_idx[env_ids]
        first_anchor = self.motion._body_pos_w[mi, 0, self.motion_anchor_body_index]
        first_anchor_yaw = yaw_quat(self.motion._body_quat_w[mi, 0, self.motion_anchor_body_index])

        distance = float(getattr(self.cfg, "dribble_cg_front_ball_distance", 0.45))
        lateral_offset = float(getattr(self.cfg, "dribble_cg_front_ball_lateral_offset", 0.0))
        height = float(getattr(self.cfg, "dribble_cg_front_ball_height", self._target_height))

        local_offset = first_anchor.new_zeros((env_ids.numel(), 3))
        local_offset[:, 0] = distance
        local_offset[:, 1] = lateral_offset
        ball_pos = first_anchor + quat_apply(first_anchor_yaw, local_offset)
        ball_pos[:, 2] = height
        return ball_pos

    def _legacy_front_ball_positions(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Historical fallback: fixed env/task +X ahead of the start anchor."""
        mi = self.motion_idx[env_ids]
        first_anchor = self.motion._body_pos_w[mi, 0, self.motion_anchor_body_index]
        distance = float(getattr(self.cfg, "dribble_cg_front_ball_distance", 0.45))
        lateral_offset = float(getattr(self.cfg, "dribble_cg_front_ball_lateral_offset", 0.0))
        height = float(getattr(self.cfg, "dribble_cg_front_ball_height", self._target_height))
        return spawn_ball_ahead_env_local(first_anchor, distance, lateral_offset, height)

    def _reference_first_contact_ball_positions(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return first-contact ball positions, validity, and source frame.

        ``ball_pos_w`` is stored in the same reference coordinate system as
        the pelvis.  Its selected point is therefore equivalent to mapping the
        frame-0 pelvis-local offset back through that pelvis yaw; retaining the
        direct value avoids introducing any simulation/world-forward axis.
        """
        mi = self.motion_idx[env_ids]
        # "Reference first contact" means the first rising edge in the whole
        # source clip.  It must not depend on a curriculum/replay selection:
        # the robot always resets at source frame 0, so an interior event would
        # put the ball at a future touch while replaying the prefix from frame 0.
        contact_frames = self.motion.first_dribble_contact_frame[mi]
        acquisition = self.s2_episode_reset_mode[env_ids] == 1
        source_frames = torch.where(acquisition, self.time_steps[env_ids], contact_frames)
        valid = (contact_frames >= 0) & self.motion.motion_has_ball_demo[mi]
        valid = torch.where(acquisition, self.motion.motion_has_ball_demo[mi], valid)
        safe_frames = source_frames.clamp(min=0)
        positions = self.motion.ball_pos_w[mi, safe_frames].clone()
        jitter_levels = tuple(
            getattr(self.cfg, "dribble_cg_curriculum_ball_spawn_jitter", ())
        )
        if jitter_levels:
            # Each pair is (clean probability, maximum perturbation radius).
            # Perturb only at sequence reset; later contacts inherit the
            # physical position/velocity errors produced by previous touches.
            episode_levels = self.s2_episode_curriculum_level[env_ids]
            clean_probability = torch.ones(env_ids.numel(), device=self.device)
            jitter_max = torch.zeros_like(clean_probability)
            for level_index, (level_clean_probability, level_jitter_max) in enumerate(
                jitter_levels
            ):
                level_mask = episode_levels == level_index
                clean_probability[level_mask] = min(
                    1.0, max(0.0, float(level_clean_probability))
                )
                jitter_max[level_mask] = max(0.0, float(level_jitter_max))
            perturb = (
                torch.rand(env_ids.numel(), device=self.device) >= clean_probability
            ) & (jitter_max > 0.0) & ~acquisition

            if bool(torch.any(perturb)):
                contact_foot = self.motion.dribble_cg_foot[mi, safe_frames].to(torch.long)
                expected_surface = self._s2_event_surface[mi, safe_frames].to(torch.long)
                left_index = self.cfg.body_names.index("left_ankle_roll_link")
                right_index = self.cfg.body_names.index("right_ankle_roll_link")
                foot_index = torch.where(
                    contact_foot == 1,
                    torch.full_like(contact_foot, right_index),
                    torch.full_like(contact_foot, left_index),
                )
                foot_pos = self.motion.body_pos_w[mi, safe_frames, foot_index]
                foot_yaw = yaw_quat(
                    self.motion.body_quat_w[mi, safe_frames, foot_index]
                )
                base_local = quat_apply(quat_inv(foot_yaw), positions - foot_pos)
                candidate_local = base_local.clone()
                accepted = torch.zeros_like(perturb)
                forward_min, forward_max = getattr(
                    self.cfg, "dribble_cg_ball_spawn_safe_forward_range", (-0.04, 0.12)
                )
                side_min, side_max = getattr(
                    self.cfg, "dribble_cg_ball_spawn_safe_side_range", (0.06, 0.14)
                )
                labels_valid = ((contact_foot == 0) | (contact_foot == 1)) & (
                    (expected_surface == 0) | (expected_surface == 1)
                )
                # Convert the source inside/outside label into the signed
                # lateral axis of the selected foot only at the geometry
                # boundary: right-foot inside is +Y; left-foot inside is -Y.
                medial_sign = torch.where(
                    contact_foot == 1,
                    torch.ones_like(base_local[:, 1]),
                    -torch.ones_like(base_local[:, 1]),
                )
                expected_lateral_sign = torch.where(
                    expected_surface == 0, medial_sign, -medial_sign
                )
                # Resampling instead of clamping guarantees the configured
                # radius remains an actual upper bound on the perturbation.
                for _ in range(8):
                    pending = perturb & labels_valid & ~accepted
                    if not bool(torch.any(pending)):
                        break
                    radius = jitter_max * torch.sqrt(
                        torch.rand(env_ids.numel(), device=self.device)
                    )
                    angle = 2.0 * torch.pi * torch.rand(
                        env_ids.numel(), device=self.device
                    )
                    proposed = base_local.clone()
                    proposed[:, 0] += radius * torch.cos(angle)
                    proposed[:, 1] += radius * torch.sin(angle)
                    signed_side = proposed[:, 1] * expected_lateral_sign
                    side_valid = (signed_side >= float(side_min)) & (
                        signed_side <= float(side_max)
                    )
                    safe = (
                        pending
                        & (proposed[:, 0] >= float(forward_min))
                        & (proposed[:, 0] <= float(forward_max))
                        & side_valid
                    )
                    candidate_local = torch.where(
                        safe.unsqueeze(-1), proposed, candidate_local
                    )
                    accepted |= safe
                perturbed_positions = foot_pos + quat_apply(foot_yaw, candidate_local)
                positions = torch.where(
                    accepted.unsqueeze(-1), perturbed_positions, positions
                )
        else:
            # Preserve the old world-XY annulus for non-curriculum users of
            # this command. S2 configures the side-preserving schedule above.
            jitter_min = max(
                0.0, float(getattr(self.cfg, "dribble_cg_ball_spawn_jitter_min", 0.0))
            )
            jitter_max = max(
                jitter_min,
                float(getattr(self.cfg, "dribble_cg_ball_spawn_jitter_max", 0.0)),
            )
            if jitter_max > 0.0:
                radius = jitter_min + torch.rand(
                    env_ids.numel(), device=self.device
                ) * (jitter_max - jitter_min)
                angle = 2.0 * torch.pi * torch.rand(env_ids.numel(), device=self.device)
                positions[:, 0] += radius * torch.cos(angle)
                positions[:, 1] += radius * torch.sin(angle)
        return positions, valid, source_frames

    def _acquisition_ball_velocity_w(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Finite-difference reference velocity and a rolling-spin fallback."""
        motion_idx = self.motion_idx[env_ids]
        frame = self.time_steps[env_ids]
        last = self.motion.file_lengths[motion_idx] - 1
        previous = torch.clamp(frame - 1, min=0)
        following = torch.minimum(frame + 1, last)
        previous_pos = self.motion.ball_pos_w[motion_idx, previous]
        following_pos = self.motion.ball_pos_w[motion_idx, following]
        fps = max(float(self.motion.fps.reshape(-1)[0]), 1.0e-6)
        elapsed = (following - previous).to(torch.float32).clamp(min=1.0) / fps
        linear_velocity = (following_pos - previous_pos) / elapsed.unsqueeze(-1)
        radius = max(float(getattr(self.cfg, "dribble_cg_ball_radius", 0.11)), 1.0e-4)
        angular_velocity = torch.zeros_like(linear_velocity)
        angular_velocity[:, 0] = -linear_velocity[:, 1] / radius
        angular_velocity[:, 1] = linear_velocity[:, 0] / radius
        return linear_velocity, angular_velocity

    def _update_soccer_ball(self, env_ids: Sequence[int] | torch.Tensor):
        """Restore reference ball velocity for acquisition teacher resets."""
        ids = self._to_env_id_tensor(env_ids)
        super()._update_soccer_ball(ids)
        acquisition_ids = ids[self.s2_episode_reset_mode[ids] == 1]
        if acquisition_ids.numel() == 0 or self.soccer_ball is None:
            return
        env_origins = getattr(self._env.scene, "env_origins", None)
        if env_origins is None:
            return
        ball_pos = self.soccer_ball_pos[acquisition_ids] + env_origins[acquisition_ids]
        ball_quat = ball_pos.new_zeros((acquisition_ids.numel(), 4))
        ball_quat[:, 0] = 1.0
        ball_lin_vel, ball_ang_vel = self._acquisition_ball_velocity_w(acquisition_ids)
        ball_state = torch.cat(
            (ball_pos, ball_quat, ball_lin_vel, ball_ang_vel), dim=-1
        )
        self.soccer_ball.write_root_state_to_sim(ball_state, env_ids=acquisition_ids)

    def _record_ball_spawn(
        self,
        env_ids: torch.Tensor,
        ball_positions: torch.Tensor,
        contact_frames: torch.Tensor,
        used_reference_contact: torch.Tensor,
    ) -> None:
        """Keep spawn provenance and frame-0 pelvis-local geometry for diagnostics."""
        mi = self.motion_idx[env_ids]
        first_anchor = self.motion._body_pos_w[mi, 0, self.motion_anchor_body_index]
        first_anchor_yaw_inv = quat_inv(
            yaw_quat(self.motion._body_quat_w[mi, 0, self.motion_anchor_body_index])
        )
        self.ball_spawn_reference_contact_frame[env_ids] = torch.where(
            used_reference_contact,
            contact_frames,
            torch.full_like(contact_frames, -1),
        )
        self.ball_spawn_source[env_ids] = used_reference_contact.to(torch.int8)
        self.ball_spawn_reference_local[env_ids] = quat_apply(
            first_anchor_yaw_inv, ball_positions - first_anchor
        )

    def ball_spawn_reference_info(self) -> dict[str, torch.Tensor]:
        """Reset-ball provenance for playback diagnostics.

        ``reference_local`` is the ball position relative to the reference
        frame-0 pelvis, expressed in that pelvis yaw frame.  A source value of
        one means first labelled contact; zero means the explicit local-front
        fallback for a clip missing the needed labels/data.
        """
        return {
            "source": self.ball_spawn_source,
            "reference_contact_frame": self.ball_spawn_reference_contact_frame,
            "reference_local": self.ball_spawn_reference_local,
        }

    def _compute_soccer_ball_positions(self, env_ids: Sequence[int] | torch.Tensor):
        ids = self._to_env_id_tensor(env_ids)
        if ids.numel() == 0:
            return

        if self._uses_reference_first_contact_spawn():
            contact_positions, valid, contact_frames = self._reference_first_contact_ball_positions(ids)
            fallback_positions = self._anchor_local_front_ball_positions(ids)
            ball_positions = torch.where(valid.unsqueeze(-1), contact_positions, fallback_positions)
            self.soccer_ball_pos[ids] = ball_positions
            self._record_ball_spawn(ids, ball_positions, contact_frames, valid)
            return

        has_demo = self._should_snap_demo_ball(ids)
        demo_ids = ids[has_demo]
        fallback_ids = ids[~has_demo]

        if fallback_ids.numel() > 0:
            fallback_mode = str(getattr(self.cfg, "dribble_cg_fallback_ball_mode", "arc_endpoint")).lower().strip()
            if fallback_mode == "front":
                self.soccer_ball_pos[fallback_ids] = self._legacy_front_ball_positions(fallback_ids)
            else:
                super()._compute_soccer_ball_positions(fallback_ids)
        if demo_ids.numel() > 0:
            # Historical +X placement is retained for legacy demo-snap tasks.
            self.soccer_ball_pos[demo_ids] = self._legacy_front_ball_positions(demo_ids)

    def _sync_demo_ball_after_step(self):
        """Kinematic sync of sim ball to demo trajectory (subset of envs)."""
        if getattr(self.cfg, "dribble_cg_use_task_frame", True):
            return
        if self.soccer_ball is None:
            return
        if hasattr(self.soccer_ball, "is_initialized") and not self.soccer_ball.is_initialized:
            return
        env_origins = getattr(self._env.scene, "env_origins", None)
        if env_origins is None:
            return

        all_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        snap = self._should_snap_demo_ball(all_ids)
        if not torch.any(snap):
            return

        ids = all_ids[snap]
        ball_pos_w = self._demo_ball_world(ids)
        ball_quat = ball_pos_w.new_zeros((ids.numel(), 4))
        ball_quat[:, 0] = 1.0
        ball_lin_vel = self._demo_ball_lin_vel_w(ids)
        ball_ang_vel = ball_pos_w.new_zeros((ids.numel(), 3))
        ball_state = torch.cat([ball_pos_w, ball_quat, ball_lin_vel, ball_ang_vel], dim=-1)
        self.soccer_ball.write_root_state_to_sim(ball_state, env_ids=ids)

        self.soccer_ball_pos[ids] = ball_pos_w - env_origins[ids]
        self.target_point_pos[ids] = self.soccer_ball_pos[ids]
        if self.target_point_marker is not None:
            self.target_point_marker.visualize(ball_pos_w)

    def _update_command(self):
        super()._update_command()
        self._sync_demo_ball_after_step()


@configclass
class DribbleCGMotionCommandCfg(MotionCommandCfg):
    """Config for :class:`DribbleCGMotionCommand`.

    Extra fields (read via ``getattr`` on older cfg objects default in command):

    dribble_cg_snap_mode:
        ``full`` — always snap to demo when ``ball_pos_w`` exists.
        ``non_contact_only`` — snap only outside annotated contact frames.
    """

    class_type: type = DribbleCGMotionCommand

    dribble_cg_snap_mode: str = "full"
    # This does not select a reset spawn frame.  Unified local stages still
    # set it to ``never`` explicitly, while ``reference_first_contact`` also
    # enforces that invariant inside the command implementation.
    dribble_cg_use_task_frame: bool = True
    # ``reference_first_contact`` places a physical reset ball at the first
    # labelled ``dribble_cg_contact`` point in ``ball_pos_w``.  ``legacy``
    # preserves the historical demo/fallback branching for older task IDs.
    dribble_cg_ball_spawn_mode: str = "legacy"
    dribble_cg_fallback_ball_mode: str = "arc_endpoint"
    dribble_cg_front_ball_distance: float = 0.45
    dribble_cg_front_ball_lateral_offset: float = 0.0
    dribble_cg_front_ball_height: float = 0.11
    # Replay-only actor observation ablation.  ``actual`` is the training
    # default; ``fixed_front`` keeps the physical ball unchanged but feeds a
    # neutral far-ball proxy through the existing 3-D actor input.
    dribble_cg_replay_ball_observation_mode: str = "actual"
    dribble_cg_replay_ball_observation_distance: float = 0.80
    # Performance-driven S2 levels. Each nested pair is
    # ``(number_of_prefix_contacts, probability)``; zero means full clip.
    # The Curriculum Manager updates ``s2_curriculum_level`` at runtime.
    dribble_cg_curriculum_levels: tuple[tuple[tuple[int, float], ...], ...] = ()
    dribble_cg_curriculum_start_level: int = 0
    dribble_cg_curriculum_lookahead_contact_phases: int = 0
    dribble_cg_curriculum_lookahead_start_level: int = 0
    # When non-empty, exact levels supersede the start-level rule. This keeps
    # unselected arrival windows confined to T2/T4 rather than leaking into
    # physical-contact-only C3/C4/Repeat levels.
    dribble_cg_curriculum_lookahead_levels: tuple[int, ...] = ()
    dribble_cg_contact_window_seconds: float = 0.10
    # Symmetric frame tolerance applied to each complete annotated contact
    # segment.  Zero preserves exact labels for legacy tasks.
    dribble_cg_contact_window_tolerance_frames: int = 0
    dribble_cg_missed_contact_grace_steps: int = 3
    dribble_cg_ball_spawn_jitter_min: float = 0.0
    dribble_cg_ball_spawn_jitter_max: float = 0.0
    # Per curriculum level: (probability of an exact reference spawn,
    # maximum radius for a foot-yaw-local, side-preserving perturbation).
    dribble_cg_curriculum_ball_spawn_jitter: tuple[tuple[float, float], ...] = ()
    dribble_cg_ball_spawn_safe_forward_range: tuple[float, float] = (-0.04, 0.12)
    dribble_cg_ball_spawn_safe_side_range: tuple[float, float] = (0.06, 0.14)
    dribble_cg_curriculum_required_contacts: tuple[int, ...] = ()
    # Empty preserves the legacy behavior in which the last selected
    # contact's outgoing transition is optimized at every curriculum level.
    # Clean staged tasks can instead name only their transition-learning
    # levels (for example T2 but not C2/C3).
    dribble_cg_terminal_progress_levels: tuple[int, ...] = ()
    # ``arrival_radius`` and ``positive_final_progress`` are legacy modes.  The
    # clean sparse curriculum uses ``absolute_window_min``: it evaluates the
    # minimum ball-centre distance to an episode-fixed reference node over the
    # next annotated contact window, scaled by the physical ball radius.
    dribble_cg_terminal_transition_mode: str = "arrival_radius"
    # Map every sparse contact anchor through the reset-time XY+yaw task frame
    # instead of the policy-dependent live root frame at outgoing contact.
    dribble_cg_episode_fixed_contact_targets: bool = False
    # Historical audit only. Formal training does not construct a live-root
    # comparison target at all.
    dribble_cg_debug_live_target_comparison: bool = False
    # Compatibility switch for the retired window-average/end and delayed
    # settlement experiments.  Formal sparse tracking leaves it disabled.
    dribble_cg_enable_legacy_window_outcome: bool = False
    # Playback markers: immutable reference target is green; the old
    # policy-dependent live-root target is red.
    s2_fixed_target_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/S2Contact/fixed_i3_target",
        markers={
            "target": sim_utils.SphereCfg(
                radius=0.06,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            )
        },
    )
    s2_live_target_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/S2Contact/live_i3_target",
        markers={
            "target": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )
        },
    )
    dribble_cg_hard_replay_min_attempts: int = 30
    dribble_cg_hard_replay_fraction: float = 0.30
    dribble_cg_curriculum_audit_probability: float = 0.25
    dribble_cg_acquisition_initial_probability: float = 0.25
    dribble_cg_acquisition_min_history_steps: int = 20
    dribble_cg_acquisition_max_history_steps: int = 30
    dribble_cg_release_window_steps: int = 8
    dribble_cg_ball_radius: float = 0.11
    # Frozen offline robot-foot collision regions. Empty disables V3.
    dribble_cg_contact_region_path: str = ""
    dribble_cg_s2_debug_regions_only: bool = False
    # ``None`` preserves the historical mixed-foot CG behavior. Unified
    # training sets this to one side and validates every motion at startup.
    dribble_cg_fixed_touch_foot: str | None = None
    # ``None`` preserves legacy foot-only labels.  A fixed inside/outside
    # instep task requires ``dribble_cg_surface`` on every annotated contact
    # frame and validates it at environment creation.
    dribble_cg_fixed_touch_surface: str | None = None
    # S2 uses per-frame inside/outside supervision. Fail early instead of
    # silently treating a legacy foot-only motion as a surface-labelled one.
    dribble_cg_require_surface_labels: bool = False
    # Legacy tasks may still opt into precomputed causal flow labels.  S2
    # 6.0.3 keeps this false and derives targets from contact anchors instead.
    dribble_cg_require_flow_labels: bool = False
