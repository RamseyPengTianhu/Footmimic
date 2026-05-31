"""Event phase system for v10 event-conditioned kick decoder.

Provides:
  - Motion segmentation into 4 semantic phases: approach, prestrike, strike, followthru
  - Event-normalized phase computation (φ ∈ [0,1] per segment)
  - Event-warped motion query for relative-offset weak prior
  - Event retiming augmentation

All coordinates are expressed in ball-relative or kick-direction-relative frames,
NEVER in absolute world coordinates.
"""
from __future__ import annotations

import math
import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .commands_multi_motion_soccer import MultiMotionLoader


# Phase IDs (used for one-hot encoding)
PHASE_APPROACH = 0
PHASE_PRESTRIKE = 1
PHASE_STRIKE = 2
PHASE_FOLLOWTHRU = 3
NUM_PHASES = 4


@dataclass
class EventSegmentBounds:
    """Segment boundaries for a single motion (frame indices).

    Segments:
      approach:   [0, approach_end)
      prestrike:  [approach_end, prestrike_end)
      strike:     [prestrike_end, strike_end)
      followthru: [strike_end, motion_length)
    """
    approach_end: int      # = kick_frame - prestrike_duration
    prestrike_end: int     # = kick_frame
    strike_end: int        # = kick_end_frame (or kick_frame + strike_duration)
    motion_length: int     # total frames

    @property
    def bounds(self) -> list[tuple[int, int]]:
        """Return [(start, end), ...] for each of the 4 segments."""
        return [
            (0, self.approach_end),
            (self.approach_end, self.prestrike_end),
            (self.prestrike_end, self.strike_end),
            (self.strike_end, self.motion_length),
        ]


def compute_segment_bounds(
    kick_frame: int,
    kick_end_frame: int,
    motion_length: int,
    prestrike_duration: int = 20,
    min_strike_duration: int = 5,
) -> EventSegmentBounds:
    """Compute event segment boundaries from kick annotations.

    Args:
        kick_frame: Frame index where kick contact begins.
        kick_end_frame: Frame index where kick contact ends. If -1, defaults
            to kick_frame + min_strike_duration.
        motion_length: Total number of frames in the motion.
        prestrike_duration: Number of frames before kick_frame for prestrike phase.
        min_strike_duration: Minimum strike phase duration if kick_end_frame not set.

    Returns:
        EventSegmentBounds with the 4 segment boundaries.
    """
    if kick_end_frame < 0:
        kick_end_frame = min(kick_frame + min_strike_duration, motion_length)

    approach_end = max(0, kick_frame - prestrike_duration)
    prestrike_end = kick_frame
    strike_end = kick_end_frame

    # Sanity: ensure ordering
    approach_end = min(approach_end, prestrike_end)
    strike_end = max(strike_end, prestrike_end + 1)
    strike_end = min(strike_end, motion_length)

    return EventSegmentBounds(
        approach_end=approach_end,
        prestrike_end=prestrike_end,
        strike_end=strike_end,
        motion_length=motion_length,
    )


def compute_event_phase(
    time_steps: torch.Tensor,
    segment_bounds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute event phase ID and normalized progress for batched envs.

    Args:
        time_steps: [N] current time step per env.
        segment_bounds: [N, 4] = [approach_end, prestrike_end, strike_end, motion_length]

    Returns:
        phase_id: [N] int tensor in {0,1,2,3}
        phase_phi: [N] float tensor in [0, 1], normalized progress within current phase.
    """
    N = time_steps.shape[0]
    device = time_steps.device
    t = time_steps.float()

    approach_end = segment_bounds[:, 0].float()
    prestrike_end = segment_bounds[:, 1].float()
    strike_end = segment_bounds[:, 2].float()
    motion_length = segment_bounds[:, 3].float()

    # Default: followthru
    phase_id = torch.full((N,), PHASE_FOLLOWTHRU, dtype=torch.long, device=device)
    phase_phi = torch.zeros(N, dtype=torch.float32, device=device)

    # Strike: prestrike_end <= t < strike_end
    strike_mask = (t >= prestrike_end) & (t < strike_end)
    strike_dur = (strike_end - prestrike_end).clamp(min=1.0)
    phase_id[strike_mask] = PHASE_STRIKE
    phase_phi[strike_mask] = ((t[strike_mask] - prestrike_end[strike_mask]) / strike_dur[strike_mask]).clamp(0, 1)

    # Prestrike: approach_end <= t < prestrike_end
    prestrike_mask = (t >= approach_end) & (t < prestrike_end)
    prestrike_dur = (prestrike_end - approach_end).clamp(min=1.0)
    phase_id[prestrike_mask] = PHASE_PRESTRIKE
    phase_phi[prestrike_mask] = ((t[prestrike_mask] - approach_end[prestrike_mask]) / prestrike_dur[prestrike_mask]).clamp(0, 1)

    # Approach: t < approach_end
    approach_mask = t < approach_end
    approach_dur = approach_end.clamp(min=1.0)
    phase_id[approach_mask] = PHASE_APPROACH
    phase_phi[approach_mask] = (t[approach_mask] / approach_dur[approach_mask]).clamp(0, 1)

    # Followthru: t >= strike_end (already default)
    followthru_mask = t >= strike_end
    followthru_dur = (motion_length - strike_end).clamp(min=1.0)
    phase_phi[followthru_mask] = ((t[followthru_mask] - strike_end[followthru_mask]) / followthru_dur[followthru_mask]).clamp(0, 1)

    return phase_id, phase_phi


def compute_event_obs(
    phase_id: torch.Tensor,
    phase_phi: torch.Tensor,
    time_steps: torch.Tensor,
    segment_bounds: torch.Tensor,
) -> torch.Tensor:
    """Compute 8D event condition observation.

    Returns: [N, 8] = [phase_onehot(4), sin(πφ), cos(πφ), t2strike_norm, t2phase_end_norm]
    """
    N = phase_id.shape[0]
    device = phase_id.device
    t = time_steps.float()

    # Phase one-hot [N, 4]
    phase_onehot = torch.zeros(N, NUM_PHASES, dtype=torch.float32, device=device)
    phase_onehot.scatter_(1, phase_id.unsqueeze(1), 1.0)

    # Smooth phase progress [N, 2]
    sin_phi = torch.sin(math.pi * phase_phi)
    cos_phi = torch.cos(math.pi * phase_phi)

    # Time to strike (normalized by motion length)
    prestrike_end = segment_bounds[:, 1].float()
    motion_length = segment_bounds[:, 3].float().clamp(min=1.0)
    t2strike = ((prestrike_end - t) / motion_length).clamp(-1.0, 1.0)

    # Time to current phase end (normalized by current phase duration)
    approach_end = segment_bounds[:, 0].float()
    strike_end = segment_bounds[:, 2].float()

    phase_end = torch.where(phase_id == PHASE_APPROACH, approach_end,
                torch.where(phase_id == PHASE_PRESTRIKE, prestrike_end,
                torch.where(phase_id == PHASE_STRIKE, strike_end, motion_length)))
    t2phase_end = ((phase_end - t) / motion_length).clamp(-1.0, 1.0)

    return torch.cat([
        phase_onehot,                           # 4D
        sin_phi.unsqueeze(1),                   # 1D
        cos_phi.unsqueeze(1),                   # 1D
        t2strike.unsqueeze(1),                  # 1D
        t2phase_end.unsqueeze(1),               # 1D
    ], dim=-1)  # [N, 8]


def event_warped_ref_index(
    phase_id: torch.Tensor,
    phase_phi: torch.Tensor,
    original_bounds: torch.Tensor,
) -> torch.Tensor:
    """Compute the reference frame index in the original motion for event-warped query.

    Given current (retimed) event phase and phi, find the corresponding frame in
    the original motion's same semantic segment.

    Args:
        phase_id: [N] current phase
        phase_phi: [N] normalized progress in current phase
        original_bounds: [N, 4] = [approach_end, prestrike_end, strike_end, motion_length]
            of the ORIGINAL (un-retimed) motion

    Returns:
        ref_idx: [N] float tensor — fractional frame index for interpolation
    """
    # Get original segment start/end for each env's current phase
    # Segment starts: [0, approach_end, prestrike_end, strike_end]
    # Segment ends:   [approach_end, prestrike_end, strike_end, motion_length]
    seg_starts = torch.zeros_like(original_bounds[:, 0].float())
    seg_ends = original_bounds[:, 0].float()

    for pid in range(NUM_PHASES):
        mask = phase_id == pid
        if pid == 0:
            seg_starts[mask] = 0.0
            seg_ends[mask] = original_bounds[mask, 0].float()
        elif pid == 1:
            seg_starts[mask] = original_bounds[mask, 0].float()
            seg_ends[mask] = original_bounds[mask, 1].float()
        elif pid == 2:
            seg_starts[mask] = original_bounds[mask, 1].float()
            seg_ends[mask] = original_bounds[mask, 2].float()
        else:  # followthru
            seg_starts[mask] = original_bounds[mask, 2].float()
            seg_ends[mask] = original_bounds[mask, 3].float()

    ref_idx = seg_starts + phase_phi * (seg_ends - seg_starts)
    return ref_idx.clamp(min=0)


def query_event_warped_weak_prior(
    ref_idx: torch.Tensor,
    motion: MultiMotionLoader,
    motion_idx: torch.Tensor,
    swing_foot_body_idx: int,
    support_foot_body_idx: int,
    ball_pos_in_motion: torch.Tensor,
) -> torch.Tensor:
    """Query event-warped weak prior as RELATIVE OFFSETS (foot-ball).

    Returns desired geometric relationships at this event phase, NOT absolute positions.

    Args:
        ref_idx: [N] fractional frame index in original motion
        motion: MultiMotionLoader with body_pos_w data
        motion_idx: [N] which motion each env is playing
        swing_foot_body_idx: body index of the swing (kicking) foot
        support_foot_body_idx: body index of the support (planted) foot
        ball_pos_in_motion: [N, 3] ball position in the original motion coordinate frame
            (typically the ball position at kick_frame, in env-origin-relative coords)

    Returns:
        weak_prior: [N, 8] = [desired_swing_offset(3), desired_support_offset(3),
                              desired_pelvis_facing_rel(2)]
    """
    N = ref_idx.shape[0]
    device = ref_idx.device

    # Interpolate body positions at fractional ref_idx
    idx_floor = ref_idx.long().clamp(0, motion._body_pos_w.shape[1] - 2)
    idx_ceil = (idx_floor + 1).clamp(max=motion._body_pos_w.shape[1] - 1)
    alpha = (ref_idx - idx_floor.float()).clamp(0, 1).unsqueeze(-1)  # [N, 1]

    # Swing foot position in original motion (before body_indexes remapping)
    swing_pos_floor = motion._body_pos_w[motion_idx, idx_floor, swing_foot_body_idx]  # [N, 3]
    swing_pos_ceil = motion._body_pos_w[motion_idx, idx_ceil, swing_foot_body_idx]
    ref_swing_pos = swing_pos_floor + alpha * (swing_pos_ceil - swing_pos_floor)  # [N, 3]

    # Support foot position
    support_pos_floor = motion._body_pos_w[motion_idx, idx_floor, support_foot_body_idx]
    support_pos_ceil = motion._body_pos_w[motion_idx, idx_ceil, support_foot_body_idx]
    ref_support_pos = support_pos_floor + alpha * (support_pos_ceil - support_pos_floor)

    # Pelvis position and orientation for facing direction
    pelvis_idx = 0  # Pelvis is typically body index 0 in the body list
    pelvis_pos_floor = motion._body_pos_w[motion_idx, idx_floor, pelvis_idx]
    pelvis_pos_ceil = motion._body_pos_w[motion_idx, idx_ceil, pelvis_idx]
    ref_pelvis_pos = pelvis_pos_floor + alpha * (pelvis_pos_ceil - pelvis_pos_floor)

    # Desired offsets: foot relative to ball (NOT absolute positions)
    desired_swing_offset = ref_swing_pos - ball_pos_in_motion   # [N, 3]
    desired_support_offset = ref_support_pos - ball_pos_in_motion  # [N, 3]

    # Pelvis facing relative to kick direction (ball → target)
    # For now, use pelvis-to-ball direction as a proxy for facing
    # This will be refined when kick_dir is available
    pelvis_to_ball = ball_pos_in_motion - ref_pelvis_pos
    pelvis_to_ball_xy = pelvis_to_ball[:, :2]
    dist = torch.norm(pelvis_to_ball_xy, dim=-1, keepdim=True).clamp(min=1e-4)
    desired_facing_rel = pelvis_to_ball_xy / dist  # [N, 2] = [cos, sin] of facing

    return torch.cat([
        desired_swing_offset,     # 3D
        desired_support_offset,   # 3D
        desired_facing_rel,       # 2D
    ], dim=-1)  # [N, 8]


def apply_event_retiming(
    original_bounds: torch.Tensor,
    approach_scale: tuple[float, float] = (0.8, 1.3),
    prestrike_scale: tuple[float, float] = (0.7, 1.5),
    strike_scale: tuple[float, float] = (0.9, 1.1),
    followthru_scale: tuple[float, float] = (0.8, 1.5),
) -> torch.Tensor:
    """Apply random retiming to event segment boundaries.

    Retiming changes the event boundary positions but NOT the motion playback speed.
    The event-warped prior automatically queries the correct semantic frame.

    Args:
        original_bounds: [N, 4] original segment boundaries
        *_scale: (min, max) random scale for each segment duration

    Returns:
        retimed_bounds: [N, 4] retimed segment boundaries
    """
    N = original_bounds.shape[0]
    device = original_bounds.device

    approach_end = original_bounds[:, 0].float()
    prestrike_end = original_bounds[:, 1].float()
    strike_end = original_bounds[:, 2].float()
    motion_length = original_bounds[:, 3].float()

    # Original segment durations
    approach_dur = approach_end
    prestrike_dur = prestrike_end - approach_end
    strike_dur = strike_end - prestrike_end
    followthru_dur = motion_length - strike_end

    # Random scales
    def rand_scale(low, high, n):
        return torch.empty(n, device=device).uniform_(low, high)

    new_approach_dur = approach_dur * rand_scale(*approach_scale, N)
    new_prestrike_dur = prestrike_dur * rand_scale(*prestrike_scale, N)
    new_strike_dur = strike_dur * rand_scale(*strike_scale, N)
    # followthru keeps the rest
    new_followthru_dur = followthru_dur * rand_scale(*followthru_scale, N)

    # Reconstruct boundaries
    new_approach_end = new_approach_dur
    new_prestrike_end = new_approach_end + new_prestrike_dur
    new_strike_end = new_prestrike_end + new_strike_dur
    new_motion_length = new_strike_end + new_followthru_dur

    # Clamp to reasonable range
    new_approach_end = new_approach_end.clamp(min=1)
    new_prestrike_end = new_prestrike_end.clamp(min=new_approach_end + 1)
    new_strike_end = new_strike_end.clamp(min=new_prestrike_end + 1)
    new_motion_length = new_motion_length.clamp(min=new_strike_end + 1)

    return torch.stack([
        new_approach_end.long().float(),
        new_prestrike_end.long().float(),
        new_strike_end.long().float(),
        new_motion_length.long().float(),
    ], dim=-1)  # [N, 4]


def query_event_warped_joint_delta(
    env,
    command,
    current_phase_idx: torch.Tensor,
    phase_progress: torch.Tensor,
    original_bounds: torch.Tensor,
) -> torch.Tensor:
    """Query the event-warped joint delta prior.
    
    delta = (ref_joint_pos[event, phi] - current_joint_pos) / 1.0
    
    Args:
        env: The RL environment instance
        command: The motion command instance
        current_phase_idx: [N] 0..3 indicating the active phase
        phase_progress: [N] 0..1 progress within the active phase
        original_bounds: [N, 4] original segment boundaries
        
    Returns:
        [N, 29] normalized joint delta
    """
    # 1. Map progress to original motion frames
    original_frames = event_warped_ref_index(current_phase_idx, phase_progress, original_bounds)
    
    # 2. Get reference joint positions at those frames
    ref_joint_pos = command.motion.joint_pos[command.motion_idx, original_frames.long()]  # [N, num_joints]
    
    # 3. Get current joint positions
    current_joint_pos = command.robot.data.joint_pos  # [N, num_joints]
    
    # 4. Compute delta (normalized by 1.0 for now)
    joint_delta = (ref_joint_pos - current_joint_pos) / 1.0
    
    return joint_delta


def query_event_warped_base_prior(
    env,
    command,
    current_phase_idx: torch.Tensor,
    phase_progress: torch.Tensor,
    original_bounds: torch.Tensor,
) -> torch.Tensor:
    """Query the event-warped base prior (height delta, projected gravity delta).
    
    height_delta = (ref_pelvis_z - current_pelvis_z) / 0.3
    gravity_delta = (ref_proj_grav[:2] - current_proj_grav[:2]) / 1.0
    
    Returns:
        [N, 3] prior (1 height, 2 gravity)
    """
    import math
    from isaaclab.utils.math import quat_apply, quat_inv
    
    # 1. Map progress to original motion frames
    original_frames = event_warped_ref_index(current_phase_idx, phase_progress, original_bounds)
    
    # 2. Get reference state directly from motion data
    # Root is typically at index 0 for motion body pos/quat
    motion_root_idx = 0 
    ref_root_pos_w = command.motion.body_pos_w[command.motion_idx, original_frames.long(), motion_root_idx]  # [N, 3]
    ref_root_quat_w = command.motion.body_quat_w[command.motion_idx, original_frames.long(), motion_root_idx]  # [N, 4]
    
    # 3. Get current state
    current_root_pos_w = command.robot.data.root_pos_w  # [N, 3]
    current_root_quat_w = command.robot.data.root_quat_w  # [N, 4]
    
    # 4. Height delta
    ref_height = ref_root_pos_w[:, 2]
    current_height = current_root_pos_w[:, 2]
    height_delta = (ref_height - current_height) / 0.3  # [N]
    
    # 5. Projected gravity delta
    gravity_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3)
    ref_proj_grav = quat_apply(quat_inv(ref_root_quat_w), gravity_w)  # [N, 3]
    current_proj_grav = quat_apply(quat_inv(current_root_quat_w), gravity_w)  # [N, 3]
    
    gravity_delta = (ref_proj_grav[:, :2] - current_proj_grav[:, :2]) / 1.0  # [N, 2]
    
    return torch.cat([height_delta.unsqueeze(-1), gravity_delta], dim=-1)  # [N, 3]
