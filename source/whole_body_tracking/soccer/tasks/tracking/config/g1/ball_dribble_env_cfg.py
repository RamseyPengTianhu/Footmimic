"""Stage 2, minimal: the clean-v0 Stage-1 tracker with a real ball, on one dribble reference.

Goal: dribble through one reference clip (Reference 1, master_001675_001884: 16 right-foot touches, 6.6 s).

Everything the Actor sees is the Stage-1 contract (160-D, reference joints + ball xyz in the pelvis frame), so the
finished teacher checkpoint loads unchanged.  Differences from Stage 1:

  ball          physical (gravity, collisions), asset damping 0, constant deceleration 0.4905 m/s^2 (mu = 0.05),
                spawned at the reference first-contact position, mapped through the reset transform
  episode       always starts at frame 0 of the clip (sequential sampling over a single file)
  reward        Stage-1 tracking + ball in the front zone (pelvis-yaw box), weight 1.0
  terminations  Stage-1 ones + ball lost (> 1.1 m from the pelvis) + any non-foot link touching the ball

The front-zone weight is kept below the tracking loss of standing still (~2.5), so "stand next to the ball" earns
less than "ignore the ball and track"; scripts/dribble/eval_ball_dribble.py checks this with a zero-action policy.
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from soccer.tasks.tracking import mdp
from soccer.tasks.tracking.mdp import ball_contact, ball_physics
from soccer.tasks.tracking.mdp.commands_dribble_cg import DribbleCGMotionCommand

from .soccer_dribbling_env_cfg import G1CleanV0Stage1EnvCfg


@configclass
class G1BallDribbleEnvCfg(G1CleanV0Stage1EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Physical ball (undo the Stage-1 ghost).
        self.scene.soccer_ball.spawn = self.scene.soccer_ball.spawn.replace(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False, disable_gravity=False, linear_damping=0.0, angular_damping=0.0
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        )
        step_dt = self.sim.dt * self.decimation
        self.events.ball_deceleration = EventTerm(
            func=ball_physics.ball_constant_deceleration_event,
            mode="interval",
            interval_range_s=(step_dt, step_dt),
            is_global_time=True,
        )
        # Per-link ball contact forces (feet first; see mdp/ball_contact.py).
        self.scene.soccer_ball_contact.filter_prim_paths_expr = ball_contact.ball_filter_prim_paths()
        self.scene.soccer_ball_contact.history_length = self.decimation  # every physics substep of a control step

        # Ball at the reference first contact; episode from frame 0.
        motion = self.commands.motion
        motion.class_type = DribbleCGMotionCommand
        motion.sampling_strategy = "sequential"
        motion.soccer_ball_follow_actual_pelvis_yaw = False
        setattr(motion, "dribble_cg_use_task_frame", False)
        setattr(motion, "dribble_cg_snap_mode", "never")
        setattr(motion, "dribble_cg_ball_spawn_mode", "reference_first_contact")

        self.rewards.ball_front_zone = RewTerm(
            func=mdp.dribbling_ball_front_zone, weight=1.0, params={"command_name": "motion"}
        )
        self.terminations.ball_lost = DoneTerm(
            func=mdp.ball_lost_dribbling,
            params={"command_name": "motion", "max_distance": 1.1, "max_vel_divergence": 0.0, "grace_steps": 0},
        )
        self.terminations.nonfoot_ball_contact = DoneTerm(func=ball_contact.nonfoot_ball_contact)
