"""Latent-prior stage1 environment for G1 soccer skills."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from soccer.tasks.tracking import mdp
from soccer.tasks.tracking.config.g1.soccer_flat_env_cfg import G1TerrainMotionEnvCfg
from soccer.tasks.tracking.mdp import latent_prior_command, latent_prior_observations, latent_prior_rewards

DEFAULT_CONTENT_CVAE_PRIOR_PATH = "models/content_cvae_prior_video1_clean_v3_final_local.pt"


@configclass
class G1LatentPriorStage1EnvCfg(G1TerrainMotionEnvCfg):
    """Stage1 controller conditioned on CVAE latent/semantic commands."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.motion.class_type = latent_prior_command.LatentPriorMotionCommand
        self.commands.motion.prior_model_path = DEFAULT_CONTENT_CVAE_PRIOR_PATH
        self.commands.motion.sampling_strategy = "uniform"

        # Actor command is now CVAE condition + latent z.  Do not add raw
        # reference angular velocity; that would leak the old frame-tracking API.
        self.observations.policy.command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        self.observations.policy.motion_ref_ang_vel = None
        self.observations.policy.base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.3, n_max=0.3))
        self.observations.policy.robot_pelvis_height = ObsTerm(
            func=latent_prior_observations.robot_pelvis_height,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.03, n_max=0.03),
        )
        if hasattr(self.observations.policy, "target_point_pos"):
            self.observations.policy.target_point_pos = None
        if hasattr(self.observations.policy, "target_destination_pos_local"):
            self.observations.policy.target_destination_pos_local = None

        self.observations.critic.command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        self.observations.critic.motion_anchor_pos_b = None
        self.observations.critic.motion_anchor_ori_b = None
        self.observations.critic.robot_pelvis_height = ObsTerm(
            func=latent_prior_observations.robot_pelvis_height,
            params={"command_name": "motion"},
        )
        if hasattr(self.observations.critic, "target_point_pos"):
            self.observations.critic.target_point_pos = None
        if hasattr(self.observations.critic, "target_destination_pos_local"):
            self.observations.critic.target_destination_pos_local = None

        # Replace hard full-body/world tracking with compact CVAE decoded
        # local-feature tracking.  Global/root yaw should be free for later
        # ball-conditioned retargeting, so keep only robot-centric balance
        # terms rather than raw reference orientation rewards.
        self.rewards.motion_global_anchor_pos.weight = 0.0
        self.rewards.motion_global_anchor_ori.weight = 0.0
        self.rewards.motion_body_pos.weight = 0.0
        self.rewards.motion_body_ori.weight = 0.0
        self.rewards.motion_body_lin_vel.weight = 0.0
        self.rewards.motion_body_ang_vel.weight = 0.0
        self.rewards.latent_prior_feature = RewTerm(
            func=latent_prior_rewards.latent_prior_feature_tracking,
            weight=6.0,
            params={
                "command_name": "motion",
                "joint_pos_std": 0.45,
                "joint_vel_std": 8.0,
                "pelvis_height_std": 0.35,
                "pelvis_lin_vel_std": 2.0,
                "pelvis_ang_vel_std": 4.0,
                "joint_pos_weight": 0.35,
                "joint_vel_weight": 0.15,
                "pelvis_height_weight": 0.35,
                "pelvis_vel_weight": 0.15,
            },
        )
        self.rewards.latent_prior_height = RewTerm(
            func=latent_prior_rewards.latent_prior_pelvis_height_tracking,
            weight=3.0,
            params={
                "command_name": "motion",
                "std": 0.35,
            },
        )
        self.rewards.alive = RewTerm(
            func=mdp.is_alive,
            weight=1.0,
        )
        self.rewards.flat_orientation = RewTerm(
            func=mdp.flat_orientation_l2,
            weight=-1.5,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.pelvis_orientation = RewTerm(
            func=mdp.pelvis_orientation,
            weight=-0.25,
            params={"command_name": "motion"},
        )

        # The prior target is compact and does not include full end-effector
        # world positions, so old full-reference failure checks are too strict.
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
