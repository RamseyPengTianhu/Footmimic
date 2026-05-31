"""V3.6 Semantic Motion Prior Config.

Inherits from V3 (AnchorCGKick), but replaces frame-locked tracking and 
contact graph rewards with phase-modulated tracking and D-gated phase-free rewards.
"""

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm, TerminationTermCfg as TermTerm

import soccer.tasks.tracking.mdp as mdp
from .soccer_anchor_env_cfg import G1AnchorCGKickEnvCfg

DEFAULT_DISCRIMINATOR_PATH = "models/strike_discriminator.pt"

@configclass
class G1SoccerV36EnvCfg(G1AnchorCGKickEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ==============================================================================
        # 1. Terminations: Replace kick_end_frame logic with phase-free timeout fail
        # ==============================================================================
        # Remove old V3 interaction fail
        if hasattr(self.terminations, "interaction_fail"):
            self.terminations.interaction_fail = None

        self.terminations.phase_free_interaction_fail = TermTerm(
            func=mdp.phase_free_interaction_fail,
            params={
                "command_name": "motion",
                "ball_speed_threshold": 1.0,
                "margin": 5,
            }
        )

        # ==============================================================================
        # 2. Tracking Rewards: Replace static/binary masking with smooth phase decay
        # ==============================================================================
        # Remove V3's dynamic ankle masking body pos
        if hasattr(self.rewards, "motion_body_pos"):
            self.rewards.motion_body_pos = None

        self.rewards.phase_modulated_body_pos = RewTerm(
            func=mdp.phase_modulated_body_pos,
            weight=1.0,
            params={
                "command_name": "motion",
                "std": 0.3,
                "body_names": [
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
                ],
                "kick_foot_name": "right_ankle_roll_link",    # Assumes right foot baseline for config structure; dynamically resolved in command
                "support_foot_name": "left_ankle_roll_link",
            },
        )
        
        self.rewards.phase_modulated_body_ori = RewTerm(
            func=mdp.phase_modulated_body_ori,
            weight=1.0,
            params={
                "command_name": "motion",
                "std": 0.5,
                "body_names": ["pelvis", "torso_link"],
            },
        )

        # Disable old rigid body orientation if it exists
        if hasattr(self.rewards, "motion_body_ori"):
            self.rewards.motion_body_ori = None

        # ==============================================================================
        # 3. Contact Quality Rewards: Phase-Free D-Gated 
        # ==============================================================================
        # Disable all old V3 binary-gated contact rewards
        if hasattr(self.rewards, "target_point_contact"):
            self.rewards.target_point_contact = None
        if hasattr(self.rewards, "early_collision_penalty"):
            self.rewards.early_collision_penalty = None
        if hasattr(self.rewards, "sideways_kick"):
            self.rewards.sideways_kick = None
        if hasattr(self.rewards, "ball_speed_reward"):
            self.rewards.ball_speed_reward = None
        if hasattr(self.rewards, "ball_velocity_direction_alignment"):
            self.rewards.ball_velocity_direction_alignment = None
            
        # Add 5 separate V3.6 terms
        self.rewards.v36_strike_contact = RewTerm(
            func=mdp.v36_strike_contact,
            weight=50.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 10.0,
                "foot_cfg": self.foot_cfg,
                "discriminator_path": "models/strike_discriminator.pt",
                "gamma": 1.0,
            }
        )
        
        self.rewards.v36_gated_ball_speed = RewTerm(
            func=mdp.v36_gated_ball_speed,
            weight=10.0,
            params={
                "command_name": "motion",
                "std": 1.2,
                "velocity_threshold": 0.5,
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 10.0,
                "foot_cfg": self.foot_cfg,
                "discriminator_path": "models/strike_discriminator.pt",
                "gamma": 1.0,
            }
        )
        
        self.rewards.v36_gated_direction = RewTerm(
            func=mdp.v36_gated_direction,
            weight=30.0,
            params={
                "command_name": "motion",
                "std": 0.8,
                "velocity_threshold": 0.5,
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 10.0,
                "foot_cfg": self.foot_cfg,
                "discriminator_path": "models/strike_discriminator.pt",
                "gamma": 1.0,
            }
        )
        
        self.rewards.v36_non_strike_penalty = RewTerm(
            func=mdp.v36_non_strike_penalty,
            weight=-10.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 5.0,
                "discriminator_path": "models/strike_discriminator.pt",
            }
        )
        
        self.rewards.v36_wrong_foot_penalty = RewTerm(
            func=mdp.v36_wrong_foot_penalty,
            weight=-5.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 10.0,
                "foot_cfg": self.foot_cfg,
            }
        )
        
        # Diagnostic term
        self.rewards.strike_d_score = RewTerm(
            func=mdp.strike_d_score,
            weight=0.0,
            params={"discriminator_path": DEFAULT_DISCRIMINATOR_PATH}
        )

# Alias for compatibility with v36b configs
G1SemanticPriorKickEnvCfg = G1SoccerV36EnvCfg

@configclass
class G1SoccerV36EnvCfg_PLAY(G1SoccerV36EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
