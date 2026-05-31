"""V3.5 Environment Configuration — Strike Discriminator Gated Rewards.

Inherits from G1AnchorCGKickEnvCfg (V3) to keep all V3 features:
  - Dynamic ankle masking
  - Ankle lock on contact
  - Egocentric ball observations
  - Interaction termination
  - All motion tracking rewards

Then replaces ONLY the contact-gating mechanism:
  - target_point_contact (frame-gated via CG) → strike_gated_contact (D-gated)
  - early_collision_penalty (frame-gated) → REMOVED (D-gate handles this)
  - sideways_kick → strike_gated_sideways_kick
  - ball_velocity_direction_alignment → strike_gated_direction_alignment
  - ball_speed_reward → strike_gated_ball_speed

No existing V3 code is modified.  This is a NEW file.
"""
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from soccer.tasks.tracking.mdp import rewards_v35
from soccer.tasks.tracking.config.g1.soccer_anchor_env_cfg import G1AnchorCGKickEnvCfg


DEFAULT_DISCRIMINATOR_PATH = "models/strike_discriminator.pt"


@configclass
class G1StrikeGatedKickEnvCfg(G1AnchorCGKickEnvCfg):
    """V3.5: Same as V3 CG config but with strike-discriminator-gated rewards.

    Policy architecture: same LSTM, same obs. Only reward gating changes.
    Inherits from V3's G1AnchorCGKickEnvCfg to keep dynamic ankle masking,
    ankle lock, interaction termination, and all other V3 features.
    """

    def __post_init__(self):
        super().__post_init__()

        # ===== DISABLE frame-based contact rewards (set by V3) =====
        self.rewards.target_point_contact.weight = 0.0  # was time_gated_contact, 50.0
        self.rewards.early_collision_penalty.weight = 0.0  # was -15.0

        # ===== DISABLE original ball outcome rewards (will be D-gated) =====
        self.rewards.sideways_kick.weight = 0.0  # was 50.0
        self.rewards.ball_velocity_direction_alignment.weight = 0.0  # was 30.0
        self.rewards.ball_speed_reward.weight = 0.0  # was 10.0

        # ===== ADD discriminator-gated replacements =====

        # Strike-gated contact (replaces time_gated_contact)
        self.rewards.strike_gated_contact = RewTerm(
            func=rewards_v35.strike_gated_contact,
            weight=50.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 10.0,
                "foot_cfg": self.foot_cfg,
                "discriminator_path": DEFAULT_DISCRIMINATOR_PATH,
                "gamma": 1.0,
                "bad_contact_penalty": 5.0,
            },
        )

        # Strike-gated sideways kick
        self.rewards.strike_gated_sideways_kick = RewTerm(
            func=rewards_v35.strike_gated_sideways_kick,
            weight=50.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 10.0,
                "foot_cfg": self.foot_cfg,
                "discriminator_path": DEFAULT_DISCRIMINATOR_PATH,
                "gamma": 1.0,
            },
        )

        # Strike-gated ball speed
        self.rewards.strike_gated_ball_speed = RewTerm(
            func=rewards_v35.strike_gated_ball_speed,
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
            },
        )

        # Strike-gated direction alignment
        self.rewards.strike_gated_direction_alignment = RewTerm(
            func=rewards_v35.strike_gated_direction_alignment,
            weight=30.0,
            params={
                "command_name": "motion",
                "std": 0.8,
                "velocity_threshold": 0.5,
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 10.0,
                "foot_cfg": self.foot_cfg,
                "discriminator_path": DEFAULT_DISCRIMINATOR_PATH,
                "gamma": 1.0,
            },
        )

        # Diagnostic: D(state) score for TensorBoard (weight=0, log only)
        self.rewards.strike_d_score = RewTerm(
            func=rewards_v35.strike_d_score,
            weight=0.0,
            params={
                "command_name": "motion",
                "discriminator_path": DEFAULT_DISCRIMINATOR_PATH,
            },
        )
