"""Reference-free student reward functions for VQ PPO (Stage C).

These rewards use the AttemptEventTracker to provide event/geometry-gated
signals.  They have **zero dependency on CG, kick_frame, or motion reference**.

All functions conform to the IsaacLab RewardTermCfg function signature:
    func(env: ManagerBasedRLEnv, **params) -> torch.Tensor  [num_envs]

The AttemptEventTracker is lazily initialized and stored on the env object
as ``_attempt_tracker``.  It auto-resets per-env state on episode termination.
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from .attempt_event_tracker import AttemptEventTracker


def _get_tracker(env: ManagerBasedRLEnv) -> AttemptEventTracker:
    """Get or create the shared AttemptEventTracker on the env."""
    tracker = getattr(env, "_attempt_tracker", None)
    if tracker is None:
        tracker = AttemptEventTracker(
            num_envs=env.num_envs,
            device=env.device,
        )
        env._attempt_tracker = tracker
    return tracker


def _ensure_stepped(env: ManagerBasedRLEnv) -> dict[str, torch.Tensor]:
    """Ensure the tracker has been stepped exactly once this sim frame.

    The reward manager calls all reward terms sequentially in one env.step().
    We use a global step counter on the tracker to detect when a new env.step()
    has started (because some *external* code must call advance_frame() once
    per env.step, before reward computation).

    Since we can't hook into env.step(), we use a simpler trick: the FIRST
    reward term to call us in a new env.step() will find that tracker has
    NOT been stepped yet (no cached rewards, or cached rewards are stale).
    We mark the cache as fresh after stepping and clear it at the start of
    each new env.step() using the env's own step counter.
    """
    tracker = _get_tracker(env)

    # The trick: we store a "last_stepped_at" monotonic counter.
    # Each call, we increment a per-env counter.  If first call this step,
    # we step the tracker.  Subsequent calls return cached results.
    #
    # To distinguish env.step() boundaries, we use a simple approach:
    # We set a flag "_attempt_needs_step = True" at the end of step(),
    # and clear it after the first reward term steps the tracker.
    needs_step = getattr(env, "_attempt_needs_step", True)

    if needs_step:
        # Reset envs that just started a new episode
        episode_buf = getattr(env, "episode_length_buf", None)
        if episode_buf is not None:
            reset_ids = (episode_buf <= 1).nonzero(as_tuple=True)[0]
            if reset_ids.numel() > 0:
                tracker.reset_envs(reset_ids)

        # Step tracker
        rewards = tracker.step(env)
        env._attempt_tracker_rewards = rewards
        env._attempt_needs_step = False  # Mark as done for this env.step()

    return env._attempt_tracker_rewards


# ─── Individual reward functions ──────────────────────────────────────────


def attempt_clean_contact(env: ManagerBasedRLEnv) -> torch.Tensor:
    """One-shot reward for first clean contact (attempt + hit within window).

    Returns 1.0 on the single frame where clean first contact occurs.
    This is the primary positive signal — PPO should maximize this.
    """
    rewards = _ensure_stepped(env)
    return rewards["clean_contact"]


def attempt_ball_speed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Ball speed reward, gated to the window after clean first contact ONLY.

    Returns 0 if no clean contact has occurred (prevents rewarding late fallback
    or accidental contact).
    """
    rewards = _ensure_stepped(env)
    return rewards["ball_speed"]


def attempt_direction_alignment(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Ball direction reward, gated to the clean first-contact window ONLY.

    Returns 0 if no clean contact has occurred, so late fallback or accidental
    contact cannot receive target-direction reward.
    """
    rewards = _ensure_stepped(env)
    return rewards["direction"]


def attempt_late_fallback_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for contacting the ball after a missed attempt (late fallback).

    Returns 1.0 on the frame of late fallback contact.  Should be used with
    a negative weight.
    """
    rewards = _ensure_stepped(env)
    return rewards["late_fallback"]


def attempt_miss_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty when an attempt's hit window expires without contact (empty swing).

    Returns 1.0 on the frame when the attempt is marked as missed.
    Should be used with a negative weight.
    """
    rewards = _ensure_stepped(env)
    return rewards["attempt_miss"]


def attempt_no_attempt_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-frame penalty when lingering near the ball without attempting to kick.

    Returns 1.0 every frame after 50 consecutive frames (~1 sec) of being
    within near_ball_dist of the ball without starting an attempt.
    Should be used with a negative weight.
    """
    rewards = _ensure_stepped(env)
    return rewards["no_attempt"]


def attempt_post_stability(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Post-strike stability reward (upright + low angular velocity).

    Active in [contact_frame + 3, contact_frame + 28] — 25 frames post contact.
    Independent of CG; gated purely by physical contact event.
    """
    rewards = _ensure_stepped(env)
    return rewards["post_stability"]


def attempt_approach_shaping(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Dense approach shaping: rewards closing in on the ball BEFORE attempt starts.

    Active per-frame when pelvis is within near_ball_dist and no attempt/contact yet.
    Components: foot-ball proximity (40%), pelvis-ball proximity (30%),
    positive closing speed (30%).  All purely geometric, zero CG.
    """
    rewards = _ensure_stepped(env)
    return rewards["approach_shaping"]


def attempt_prestrike_shaping(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Dense prestrike shaping: rewards foot-ball proximity during attempt window.

    Active per-frame after attempt_start, before contact or window expiry.
    Components: tight foot-ball distance (60%), closing speed (40%).
    Bridges the sparse clean_contact signal back to positioning.
    """
    rewards = _ensure_stepped(env)
    return rewards["prestrike_shaping"]
