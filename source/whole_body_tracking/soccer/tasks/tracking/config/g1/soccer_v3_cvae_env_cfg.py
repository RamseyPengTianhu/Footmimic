"""V3 + content-conditioned CVAE motion prior.

This config intentionally inherits directly from the proven V3/CG kick setup.
V3 contact graph and interaction rewards stay unchanged, while hard tracking is
relaxed in the kick window and the CVAE prior takes over part of the motion
style/semantic supervision.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

from soccer.tasks.tracking import mdp
from soccer.tasks.tracking.mdp import content_cvae_prior
from soccer.tasks.tracking.mdp import v3_cvae_rewards
from soccer.tasks.tracking.config.g1.soccer_anchor_env_cfg import G1AnchorCGKickEnvCfg

DEFAULT_CONTENT_CVAE_PRIOR_PATH = "models/content_cvae_prior_video1_clean_v3_final_local.pt"


@configclass
class G1AnchorCGLatentKickEnvCfg(G1AnchorCGKickEnvCfg):
    """Pure V3/CG rewards with physical terminations for latent-residual PPO.

    This is the clean latent-control counterpart to the proven V3 teacher:
    it keeps CG-v3 rewards unchanged and avoids the noisy content-CVAE motion
    prior, while relaxing reference-specific terminations that are too brittle
    for a decoded latent action policy.
    """

    def __post_init__(self):
        super().__post_init__()

        self.terminations.anchor_pos_z = None
        self.terminations.anchor_ori = None
        self.terminations.ee_body_pos = None
        self.terminations.root_height = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "minimum_height": 0.25,
            },
        )
        self.terminations.base_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "limit_angle": 1.6,
            },
        )


@configclass
class G1AnchorCGContentCVAEPriorKickEnvCfg(G1AnchorCGKickEnvCfg):
    """Anchor-CG v3 baseline with strike-window tracking relaxed by CVAE prior."""

    def __post_init__(self):
        super().__post_init__()

        # Keep V3's CG/contact/reach rewards intact, but stop per-frame tracking
        # from dominating exactly when the ball geometry should matter most.
        self.rewards.motion_body_pos = RewTerm(
            func=v3_cvae_rewards.cg_modulated_body_pos,
            weight=0.8,
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
                "cg_margin": 5,
                "body_cg1_scale": 0.5,
                "kick_foot_cg1_scale": 0.05,
                "support_foot_cg1_scale": 0.8,
            },
        )

        self.rewards.motion_foot_pos = RewTerm(
            func=v3_cvae_rewards.cg_modulated_foot_pos,
            weight=0.35,
            params={
                "command_name": "motion",
                "std": 0.3,
                "foot_body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                ],
                "cg_margin": 5,
                "kick_foot_cg1_scale": 0.05,
                "support_foot_cg1_scale": 0.8,
            },
        )

        if hasattr(self.rewards, "motion_body_lin_vel"):
            self.rewards.motion_body_lin_vel.weight *= 0.5
        if hasattr(self.rewards, "motion_body_ang_vel"):
            self.rewards.motion_body_ang_vel.weight *= 0.5

        self.rewards.content_cvae_prior = RewTerm(
            func=content_cvae_prior.content_cvae_prior_reward,
            weight=0.5,
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


@configclass
class G1AnchorCGContentCVAELatentKickEnvCfg(G1AnchorCGContentCVAEPriorKickEnvCfg):
    """V3CVAE rewards with physical, not hard-reference, terminations.

    Intended for action-CVAE latent-residual PPO.  The low-level decoder already
    constrains motion to the distilled expert manifold, so strict end-effector
    reference termination is too brittle and kills useful latent exploration.
    """

    def __post_init__(self):
        super().__post_init__()

        self.terminations.anchor_pos_z = None
        self.terminations.anchor_ori = None
        self.terminations.ee_body_pos = None
        self.terminations.root_height = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "minimum_height": 0.25,
            },
        )
        self.terminations.base_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "limit_angle": 1.6,
            },
        )
