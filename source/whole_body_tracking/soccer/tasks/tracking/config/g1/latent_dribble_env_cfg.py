"""Dribble line on the shared latent interface v1 (docs/milestones/SKILL_INTERFACE_V1_PLAN_20261016_CN.md).

D-a (Stage 1' smoke): latent motion tracking on the clean-v0 Stage-1 environment (tracking rewards, ghost ball,
no ball interaction), with the observation groups re-laid-out to interface v1:

  policy (134)  student observation seen by the prior and the tracker, in the shared order
                proprio 93 | ball 10 | body command 3 | current event 13 | soft phase 4 | next event 11
  ref    (177)  encoder input (reference window), training only
  critic        unchanged privileged observation of the clean-v0 Stage-1 task
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from soccer.latent import skill_event as se
from soccer.tasks.tracking import mdp
from soccer.tasks.tracking.mdp import latent_obs

from .soccer_dribbling_env_cfg import G1CleanV0Stage1EnvCfg


@configclass
class LatentStudentObsCfg(ObsGroup):
    """Student observation, interface v1 order.  Noise of the proprio terms is copied from the parent task."""

    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    actions = ObsTerm(func=mdp.last_action)
    ball = ObsTerm(func=latent_obs.latent_ball, params={"command_name": "motion", "possession": 1.0})
    body_cmd = ObsTerm(func=mdp.motion_locomotion_twist_command_local, params={"command_name": "motion"})
    event = ObsTerm(func=latent_obs.latent_current_event, params={"command_name": "motion"})
    phase = ObsTerm(func=latent_obs.latent_soft_phase, params={"command_name": "motion"})
    next_event = ObsTerm(func=latent_obs.latent_next_event, params={"command_name": "motion"})

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class LatentRefObsCfg(ObsGroup):
    """Encoder input (reference window).  Never given to the prior or the tracker."""

    ref_window = ObsTerm(func=latent_obs.latent_ref_window, params={"command_name": "motion"})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class G1LatentDribbleDaEnvCfg(G1CleanV0Stage1EnvCfg):
    """D-a: Stage 1' smoke on interface v1 (ghost ball, possession fixed to 1, events from ``dribble_cg_*``)."""

    def __post_init__(self):
        super().__post_init__()
        parent = self.observations.policy
        student = LatentStudentObsCfg()
        for name in ("projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "actions"):
            src = getattr(parent, name, None)
            if src is not None:
                getattr(student, name).noise = src.noise
        self.observations.policy = student
        self.observations.ref = LatentRefObsCfg()


# Dimensions the groups must have (checked by scripts/dribble/check_latent_obs.py).
EXPECTED_POLICY_DIM = se.OBS_DIM  # 134
EXPECTED_REF_DIM = se.REF_DIM  # 177
