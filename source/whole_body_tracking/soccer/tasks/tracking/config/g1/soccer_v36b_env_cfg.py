"""V3.6b Ball-Ready Semantic Prior kick environment."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from soccer.tasks.tracking.mdp import content_cvae_prior
from soccer.tasks.tracking.mdp import rewards_v36b
from soccer.tasks.tracking.config.g1.soccer_v36_env_cfg import (
    DEFAULT_DISCRIMINATOR_PATH,
    G1SemanticPriorKickEnvCfg,
)

DEFAULT_CONTENT_CVAE_PRIOR_PATH = "models/content_cvae_prior_video1_clean_v3_final_local.pt"


@configclass
class G1BallReadySemanticPriorKickEnvCfg(G1SemanticPriorKickEnvCfg):
    """V3.6b: semantic prior gated by ball-ready geometry.

    Same observation and policy shape as v3/v36a.  Intended to resume from the
    v3 checkpoint, not from a newly shaped policy.
    """

    def __post_init__(self):
        super().__post_init__()

        tracked_bodies = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]

        common_geom = {
            "kick_foot_name": "right_ankle_roll_link",
            "support_foot_name": "left_ankle_roll_link",
            "near_ball_dist": 1.15,
            "near_ball_temp": 0.20,
            "kick_dist_std": 0.28,
            "kick_height_min": 0.02,
            "kick_height_max": 0.45,
            "kick_height_std": 0.12,
            "support_lat_min": 0.16,
            "support_lat_max": 0.58,
            "support_long_min": -0.70,
            "support_long_max": 0.10,
            "support_region_std": 0.18,
            "support_vel_std": 0.45,
            "support_height_max": 0.16,
            "support_height_std": 0.10,
            "support_yaw_std": 0.75,
        }
        common_timing = {
            "timing_early_grace": 20,
            "timing_late_grace": 55,
            "timing_early_decay": 12.0,
            "timing_late_decay": 25.0,
        }

        # Slightly less loose than v36a during strike: keep strike morphology,
        # but do not hard-track exact contact geometry.
        self.rewards.motion_body_pos = RewTerm(
            func=rewards_v36b.phase_modulated_body_pos,
            weight=1.0,
            params={
                "command_name": "motion",
                "std": 0.3,
                "body_names": tracked_bodies,
                "kick_foot_name": "right_ankle_roll_link",
                "support_foot_name": "left_ankle_roll_link",
            },
        )
        self.motion_body_ori = RewTerm(
            func=rewards_v36b.phase_modulated_body_ori,
            weight=0.6,
            params={
                "command_name": "motion",
                "std": 0.5,
                "body_names": tracked_bodies,
            },
        )
        self.rewards.motion_body_ori = self.motion_body_ori

        # Disable v36a contact terms; v36b replaces them with ball-ready gated
        # versions so late fallback contacts do not receive full outcome credit.
        for name in [
            "v36_strike_contact",
            "v36_gated_ball_speed",
            "v36_gated_direction",
            "v36_non_strike_penalty",
            "v36_wrong_foot_penalty",
        ]:
            if hasattr(self.rewards, name):
                getattr(self.rewards, name).weight = 0.0

        self.rewards.v36b_strike_ready_prior = RewTerm(
            func=rewards_v36b.v36b_strike_ready_prior,
            weight=3.0,
            params={
                "command_name": "motion",
                **common_geom,
            },
        )

        self.rewards.v36b_empty_swing_penalty = RewTerm(
            func=rewards_v36b.v36b_empty_swing_penalty,
            weight=4.0,
            params={
                "command_name": "motion",
                "speed_threshold": 2.0,
                "speed_std": 1.2,
                "empty_dist": 0.42,
                "empty_dist_temp": 0.08,
                **common_geom,
            },
        )

        self.rewards.v36b_strike_contact = RewTerm(
            func=rewards_v36b.v36b_strike_contact,
            weight=55.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 10.0,
                "foot_cfg": self.foot_cfg,
                "discriminator_path": DEFAULT_DISCRIMINATOR_PATH,
                "gamma": 1.0,
                "ready_gamma": 0.7,
                "timing_gamma": 1.0,
                **common_geom,
                **common_timing,
            },
        )

        self.rewards.v36b_gated_ball_speed = RewTerm(
            func=rewards_v36b.v36b_gated_ball_speed,
            weight=10.0,
            params={
                "command_name": "motion",
                "std": 1.2,
                "velocity_threshold": 0.5,
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 10.0,
                "foot_cfg": self.foot_cfg,
                "discriminator_path": DEFAULT_DISCRIMINATOR_PATH,
                "gamma": 1.0,
                "ready_gamma": 0.7,
                "timing_gamma": 1.0,
                "min_quality": 0.25,
                "window": 5,
                **common_geom,
                **common_timing,
            },
        )

        self.rewards.v36b_gated_direction = RewTerm(
            func=rewards_v36b.v36b_gated_direction,
            weight=15.0,
            params={
                "command_name": "motion",
                "std": 0.8,
                "velocity_threshold": 0.5,
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 10.0,
                "foot_cfg": self.foot_cfg,
                "discriminator_path": DEFAULT_DISCRIMINATOR_PATH,
                "gamma": 1.0,
                "ready_gamma": 0.7,
                "timing_gamma": 1.0,
                "min_quality": 0.25,
                "window": 5,
                **common_geom,
                **common_timing,
            },
        )

        self.rewards.v36b_invalid_contact_penalty = RewTerm(
            func=rewards_v36b.v36b_invalid_contact_penalty,
            weight=12.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 10.0,
                "foot_cfg": self.foot_cfg,
                "discriminator_path": DEFAULT_DISCRIMINATOR_PATH,
                "gamma": 1.0,
                "ready_gamma": 0.7,
                "timing_gamma": 1.0,
                **common_geom,
                **common_timing,
            },
        )


@configclass
class G1BallReadyPowerBalanceKickEnvCfg(G1BallReadySemanticPriorKickEnvCfg):
    """V3.6b2: keep ball-ready timing, shift credit toward stronger kicks."""

    def __post_init__(self):
        super().__post_init__()

        # v36b fixed the major timing/early-contact failure mode, but became
        # too conservative.  This balance keeps the same event gates while
        # reducing pure contact credit and increasing post-contact speed credit.
        self.rewards.v36b_strike_contact.weight = 40.0
        self.rewards.v36b_gated_ball_speed.weight = 18.0
        self.rewards.v36b_gated_direction.weight = 12.0
        self.rewards.v36b_empty_swing_penalty.weight = 3.0
        self.rewards.v36b_invalid_contact_penalty.weight = 10.0


@configclass
class G1BallReadyEmptySwingGuardKickEnvCfg(G1BallReadyPowerBalanceKickEnvCfg):
    """V3.6b3: v36b2 plus memory against empty-swing then late-contact reward."""

    def __post_init__(self):
        super().__post_init__()

        # If a clear empty swing happens before first contact, subsequent
        # fallback contact should not receive normal contact/outcome credit.
        # This directly targets the visual failure mode where the robot kicks
        # air, then later bumps the ball and still scores as success.
        for name in [
            "v36b_strike_contact",
            "v36b_gated_ball_speed",
            "v36b_gated_direction",
            "v36b_invalid_contact_penalty",
        ]:
            getattr(self.rewards, name).params["empty_swing_quality_scale"] = 0.05


@configclass
class G1BallReadyAttemptWindowKickEnvCfg(G1BallReadyPowerBalanceKickEnvCfg):
    """V3.6b4: reward only contacts that land inside the first-attempt window."""

    def __post_init__(self):
        super().__post_init__()

        attempt_params = {
            "attempt_speed_threshold": 2.0,
            "attempt_closing_speed": 0.5,
            "attempt_max_foot_ball_dist": 0.90,
            "attempt_min_kick_height": 0.02,
            "attempt_max_kick_height": 0.65,
            "attempt_near_ball_score": 0.5,
            "attempt_early_grace": 25,
            "attempt_window": 18,
        }
        empty_swing_attempt_params = {
            "attempt_max_foot_ball_dist": attempt_params["attempt_max_foot_ball_dist"],
            "attempt_min_kick_height": attempt_params["attempt_min_kick_height"],
            "attempt_max_kick_height": attempt_params["attempt_max_kick_height"],
            "attempt_early_grace": attempt_params["attempt_early_grace"],
            "attempt_window": attempt_params["attempt_window"],
        }

        # The dense term still discourages obvious air swings, but the main
        # signal is now one-shot: once the first attempt misses its hit window,
        # late fallback contact loses outcome credit.
        self.rewards.v36b_empty_swing_penalty.weight = 5.0
        self.rewards.v36b_empty_swing_penalty.params.update(
            {
                "closing_speed_threshold": 0.5,
                "empty_dist": 0.60,
                "empty_dist_temp": 0.10,
                "miss_penalty_scale": 1.0,
                "trigger_near_ball": attempt_params["attempt_near_ball_score"],
                **empty_swing_attempt_params,
            }
        )

        for name in [
            "v36b_strike_contact",
            "v36b_gated_ball_speed",
            "v36b_gated_direction",
            "v36b_invalid_contact_penalty",
        ]:
            getattr(self.rewards, name).params.update(
                {
                    "empty_swing_quality_scale": 0.0,
                    "require_attempt_hit": True,
                    "contact_without_attempt_quality_scale": 0.0,
                    **attempt_params,
                }
            )

        self.rewards.v36b_strike_contact.weight = 40.0
        self.rewards.v36b_gated_ball_speed.weight = 18.0
        self.rewards.v36b_gated_direction.weight = 12.0
        self.rewards.v36b_invalid_contact_penalty.weight = 16.0


@configclass
class G1ContentCVAEPriorKickEnvCfg(G1BallReadyAttemptWindowKickEnvCfg):
    """V3.6c: v36b4 plus a weak learned content-conditioned CVAE motion prior."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.content_cvae_prior = RewTerm(
            func=content_cvae_prior.content_cvae_prior_reward,
            weight=0.05,
            params={
                "command_name": "motion",
                "model_path": DEFAULT_CONTENT_CVAE_PRIOR_PATH,
                "reward_std": 0.5,
                "error_clip": 10.0,
                "require_full_window": True,
                "kick_foot_name": "right_ankle_roll_link",
                "support_foot_name": "left_ankle_roll_link",
                "basis_mode": "reference",
            },
        )
