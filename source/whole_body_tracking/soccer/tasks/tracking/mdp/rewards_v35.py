"""V3.5 Strike-Gated Reward Functions.

Replaces frame-based Contact Graph rewards with discriminator-gated versions.
All functions follow the same interface as existing reward terms in rewards.py.

Key design:
  - Soft multiplier: r = D(state)^gamma * ball_outcome
  - No frame numbers involved
  - D(state) determines if the motion is strike-like
  - Bad contacts (low D) get penalty
  - Good contacts (high D) get ball outcome rewards

These functions are NEW — they do NOT modify any existing V3 reward code.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude, quat_apply, quat_inv, quat_apply_inverse

from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import MotionCommand
from soccer.tasks.tracking.mdp.kick_detection import KickContactTracker
from soccer.tasks.tracking.mdp.strike_discriminator import (
    StrikeDiscriminator, StrikeFeatureExtractor, INPUT_DIM,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ── Module-level discriminator cache ─────────────────────────────────────
# Shared across all reward terms in the same env.  Loaded once on first call.
_discriminator: StrikeDiscriminator | None = None
_extractor: StrikeFeatureExtractor | None = None
_d_score_cache: torch.Tensor | None = None
_d_cache_step: int = -1


def _get_discriminator(device: torch.device, discriminator_path: str) -> StrikeDiscriminator:
    """Load discriminator model (cached singleton)."""
    global _discriminator
    if _discriminator is None:
        import os
        ckpt = torch.load(discriminator_path, map_location=device, weights_only=False)
        _discriminator = StrikeDiscriminator(
            input_dim=ckpt.get("input_dim", INPUT_DIM),
            hidden=ckpt.get("hidden", 64),
        )
        _discriminator.load_state_dict(ckpt["model_state_dict"])
        _discriminator.to(device)
        _discriminator.eval()
        for p in _discriminator.parameters():
            p.requires_grad = False
        print(f"[v3.5] Loaded strike discriminator from {discriminator_path}")
    return _discriminator


def _get_extractor() -> StrikeFeatureExtractor:
    """Get feature extractor (cached singleton)."""
    global _extractor
    if _extractor is None:
        _extractor = StrikeFeatureExtractor()
    return _extractor


def _compute_d_score(env: ManagerBasedRLEnv, command: MotionCommand,
                      discriminator_path: str) -> torch.Tensor:
    """Compute D(state) for all envs, cached per step."""
    global _d_score_cache, _d_cache_step

    step = env.common_step_counter if hasattr(env, "common_step_counter") else -2
    if _d_cache_step == step and _d_score_cache is not None:
        return _d_score_cache

    disc = _get_discriminator(env.device, discriminator_path)
    ext = _get_extractor()
    features = ext.compute(env, command)
    with torch.no_grad():
        _d_score_cache = disc(features)
    _d_cache_step = step
    return _d_score_cache


def _get_kick_tracker(command: MotionCommand) -> KickContactTracker:
    tracker = getattr(command, "kick_contact_tracker", None)
    if tracker is None:
        raise RuntimeError("MotionCommand missing kick_contact_tracker")
    return tracker


# ══════════════════════════════════════════════════════════════════════════
# Strike-Gated Contact Reward
# ══════════════════════════════════════════════════════════════════════════

def strike_gated_contact(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 10.0,
    foot_cfg: SceneEntityCfg | None = None,
    discriminator_path: str = "models/strike_discriminator.pt",
    gamma: float = 1.0,
    bad_contact_penalty: float = 5.0,
) -> torch.Tensor:
    """Strike-gated contact reward (replaces target_point_contact).

    On first ball contact:
      - If D(state) high + correct foot → r = D^gamma (one-shot reward)
      - If D(state) low → r = -(1 - D) * penalty (running collision)

    No frame numbers involved.  D(state) determines if current motion is
    strike-like, not the wall-clock time.

    Args:
        gamma: Power exponent for soft gating. Higher = stricter.
        bad_contact_penalty: Penalty scale for non-strike contacts.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    tracker = _get_kick_tracker(command)
    event = tracker.detect(command, ball_sensor_name, horizontal_force_threshold)

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    if not torch.any(event.new_contact):
        return reward

    # Compute D(state)
    d_score = _compute_d_score(env, command, discriminator_path)

    # Check correct foot
    correct_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if foot_cfg is not None:
        foot_info = tracker.resolve_contact_foot(command, foot_cfg, event.new_contact)
        if foot_info.env_ids.numel() > 0:
            valid_expectation = foot_info.expected >= 0
            correct = (foot_info.sides == foot_info.expected) & valid_expectation
            correct_mask[foot_info.env_ids] = correct

    # Good contact: strike-like + correct foot
    good = event.new_contact & correct_mask
    if good.any():
        reward[good] = d_score[good].pow(gamma)

    # Bad contact: any contact but not good
    bad = event.new_contact & (~correct_mask)
    if bad.any():
        reward[bad] = -(1.0 - d_score[bad]) * bad_contact_penalty

    # Record for diagnostics
    tracker.record_expected_success(event.new_contact, correct_mask)

    return reward


# ══════════════════════════════════════════════════════════════════════════
# Strike-Gated Ball Outcome Rewards
# ══════════════════════════════════════════════════════════════════════════

def strike_gated_sideways_kick(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 10.0,
    foot_cfg: SceneEntityCfg | None = None,
    discriminator_path: str = "models/strike_discriminator.pt",
    gamma: float = 1.0,
) -> torch.Tensor:
    """Sideways kick reward gated by D(state)^gamma.

    Same as rewards.sideways_kick but multiplied by D(state)^gamma.
    """
    from soccer.tasks.tracking.mdp.rewards import sideways_kick
    base_reward = sideways_kick(
        env, command_name, ball_sensor_name,
        horizontal_force_threshold, foot_cfg,
    )
    if not base_reward.any():
        return base_reward

    command = env.command_manager.get_term(command_name)
    d_score = _compute_d_score(env, command, discriminator_path)
    return base_reward * d_score.pow(gamma)


def strike_gated_ball_speed(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 1.2,
    velocity_threshold: float = 0.5,
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 10.0,
    foot_cfg: SceneEntityCfg | None = None,
    discriminator_path: str = "models/strike_discriminator.pt",
    gamma: float = 1.0,
) -> torch.Tensor:
    """Ball speed reward gated by D(state)^gamma."""
    from soccer.tasks.tracking.mdp.rewards import ball_speed_reward
    base_reward = ball_speed_reward(
        env, command_name, std, velocity_threshold,
        ball_sensor_name, horizontal_force_threshold, foot_cfg,
    )
    if not base_reward.any():
        return base_reward

    command = env.command_manager.get_term(command_name)
    d_score = _compute_d_score(env, command, discriminator_path)
    return base_reward * d_score.pow(gamma)


def strike_gated_direction_alignment(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    std: float = 0.8,
    velocity_threshold: float = 0.5,
    ball_sensor_name: str = "soccer_ball_contact",
    horizontal_force_threshold: float = 10.0,
    foot_cfg: SceneEntityCfg | None = None,
    discriminator_path: str = "models/strike_discriminator.pt",
    gamma: float = 1.0,
) -> torch.Tensor:
    """Ball direction alignment reward gated by D(state)^gamma."""
    from soccer.tasks.tracking.mdp.rewards import ball_velocity_direction_alignment
    base_reward = ball_velocity_direction_alignment(
        env, command_name, std, velocity_threshold,
        ball_sensor_name, horizontal_force_threshold, foot_cfg,
    )
    if not base_reward.any():
        return base_reward

    command = env.command_manager.get_term(command_name)
    d_score = _compute_d_score(env, command, discriminator_path)
    return base_reward * d_score.pow(gamma)


# ══════════════════════════════════════════════════════════════════════════
# Diagnostic: D(state) logging reward (weight=0, used only for logging)
# ══════════════════════════════════════════════════════════════════════════

def strike_d_score(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    discriminator_path: str = "models/strike_discriminator.pt",
) -> torch.Tensor:
    """Returns D(state) for logging purposes. Set weight=0 in config."""
    command = env.command_manager.get_term(command_name)
    return _compute_d_score(env, command, discriminator_path)
