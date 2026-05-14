"""Dribbling environment configurations for the G1 robot.

Inherits proximity-level tracking and adds dribbling-specific rewards:
  - velocity / proximity gates (anti static exploit); velocity match requires contact
  - forward-progress reward: ball must move forward in pelvis-local frame
  - dense foot–ball approach when not in contact (stronger weight, lower speed gate)
  - pelvis orientation vs motion reference (anti lean-back / arched torso)
  - ball horizontal speed excess penalty
  - ankle-based gentle touch / hard-hit EMA / non-ankle contact penalty (no ``kick_leg``)
  - anti-orbit penalty; proximity damped in forward corridor without contact; stall penalty
  - slightly relaxed imitation weights to favour reaching the ball
  - ``ball_lost`` and tighter ``dribbling_no_contact`` termination
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from soccer.tasks.tracking import mdp
from soccer.tasks.tracking.mdp import observations_anchor as obs_anchor
from soccer.tasks.tracking.mdp.commands_dribble_cg import DribbleCGMotionCommand
from .soccer_flat_env_cfg import G1FlatMotionEnvCfg, G1FlatProximityEnvCfg


@configclass
class G1FlatDribblingEnvCfg(G1FlatProximityEnvCfg):
    """Flat-ground dribbling environment."""

    # Legal ankle contact mode for dribbling:
    # - "right": only right ankle is legal
    # - "left": only left ankle is legal
    # - "both": both ankles are legal
    dribble_contact_mode: str = "right"

    def __post_init__(self):
        super().__post_init__()

        # Slightly relax imitation so the policy can deviate toward the ball while
        # keeping torso/gait reference (touch-related rewards provide the main ball signal).
        if hasattr(self.rewards, "motion_body_pos"):
            self.rewards.motion_body_pos.weight = 0.72
        if hasattr(self.rewards, "motion_foot_pos"):
            self.rewards.motion_foot_pos.weight = 0.65

        # Stronger upright / anti-arch than generic proximity alone
        if hasattr(self.rewards, "pelvis_orientation"):
            self.rewards.pelvis_orientation.weight = -2.5

        # Drop world-frame yaw tracking from the motion: zig-zag motion ("slalom around
        # cones") would otherwise force the policy to copy the S-shape heading pattern.
        # The face-ball reward below replaces it with a task-aligned heading signal.
        if hasattr(self.rewards, "motion_global_anchor_ori"):
            self.rewards.motion_global_anchor_ori.weight = 0.0

        self.rewards.dribbling_face_ball = RewTerm(
            func=mdp.dribbling_face_ball,
            weight=2.0,
            params={
                "command_name": "motion",
                "min_distance": 0.05,
            },
        )

        mode = str(self.dribble_contact_mode).lower().strip()
        if mode not in {"right", "left", "both"}:
            raise ValueError(
                f"Unsupported dribble_contact_mode={self.dribble_contact_mode}. "
                "Expected one of: right, left, both."
            )

        if mode == "right":
            legal_ankles = ["right_ankle_roll_link"]
            other_ankles = ["left_ankle_roll_link"]
        elif mode == "left":
            legal_ankles = ["left_ankle_roll_link"]
            other_ankles = ["right_ankle_roll_link"]
        else:
            legal_ankles = ["right_ankle_roll_link", "left_ankle_roll_link"]
            other_ankles = []

        _foot_cfg = SceneEntityCfg("robot", body_names=legal_ankles)

        # Legal ankles must appear first for num_ankle_links gating.
        _contact_body_cfg = SceneEntityCfg(
            "robot",
            body_names=[
                *legal_ankles,
                *other_ankles,
                "right_knee_link",
                "left_knee_link",
                "right_wrist_yaw_link",
                "left_wrist_yaw_link",
            ],
        )
        _num_ankle_links = len(legal_ankles)

        self.rewards.dribbling_velocity_tracking = RewTerm(
            func=mdp.dribbling_velocity_tracking,
            weight=5.0,
            params={
                "command_name": "motion",
                "std": 1.0,
                "pelvis_speed_min": 0.14,
                "ball_speed_min": 0.0,
                # Ball–pelvis velocity match only counts after real contact (forces touches
                # instead of orbiting with a static ball).
                "require_contact": True,
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
            },
        )

        self.rewards.dribbling_dynamic_proximity = RewTerm(
            func=mdp.dribbling_dynamic_proximity,
            weight=5.0,
            params={
                "command_name": "motion",
                "near_dist": 0.2,
                "far_dist": 0.5,
                "penalty_std": 0.15,
                "pelvis_speed_min": 0.14,
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
                # In the "ball in front" corridor, do not pay full proximity without a touch.
                "no_contact_zone_damping": 0.28,
                "zone_lateral_abs_max": 0.18,
            },
        )

        self.rewards.dribbling_stall_no_touch_penalty = RewTerm(
            func=mdp.dribbling_stall_no_touch_penalty,
            weight=-5.5,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
                "max_xy_dist": 0.52,
                "pelvis_speed_max": 0.16,
            },
        )

        self.rewards.dribbling_approach_foot_ball = RewTerm(
            func=mdp.dribbling_approach_foot_ball_distance,
            weight=7.0,
            params={
                "command_name": "motion",
                "foot_cfg": _foot_cfg,
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
                "std": 0.22,
                "pelvis_speed_min": 0.05,
            },
        )

        self.rewards.dribbling_pelvis_quat_tracking = RewTerm(
            func=mdp.dribbling_pelvis_quat_tracking_exp,
            weight=2.0,
            params={
                "command_name": "motion",
                "std": 0.45,
            },
        )

        self.rewards.dribbling_ball_speed_excess = RewTerm(
            func=mdp.dribbling_ball_xy_speed_excess_penalty,
            weight=-1.5,
            params={
                "speed_cap": 4.2,
                "linear_scale": 1.5,
            },
        )

        self.rewards.dribbling_ball_forward_progress = RewTerm(
            func=mdp.dribbling_ball_world_forward_progress,
            weight=5.5,
            params={
                "command_name": "motion",
                # World-axis forward: pair with normalize_motion_yaw=True so robot resets
                # facing world +x and the ball is asked to move in that absolute direction.
                "axis": "x",
                "sign": 1.0,
                "min_forward_speed": 0.24,
                "speed_scale": 0.25,
                "pelvis_speed_min": 0.06,
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 0.5,
                "require_recent_contact": True,
                "recent_contact_window": 10,
            },
        )

        self.rewards.dribbling_orbiting_penalty = RewTerm(
            func=mdp.dribbling_orbiting_penalty,
            weight=-6.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
                "orbit_radius_max": 0.9,
                "tangential_deadzone": 0.08,
                "tangential_scale": 0.35,
            },
        )

        self.rewards.dribbling_legal_foot_touch = RewTerm(
            func=mdp.dribbling_legal_foot_touch,
            weight=9.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "force_threshold": 22.0,
                "all_body_cfg": _contact_body_cfg,
                "num_ankle_links": _num_ankle_links,
            },
        )

        self.rewards.dribbling_micro_contact_filter = RewTerm(
            func=mdp.dribbling_micro_contact_filter,
            weight=-4.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "force_threshold": 22.0,
                "max_penalty": 2.0,
                "ema_alpha": 0.35,
                "all_body_cfg": _contact_body_cfg,
                "num_ankle_links": _num_ankle_links,
            },
        )

        self.rewards.dribbling_undesired_contact_penalty = RewTerm(
            func=mdp.dribbling_undesired_contact_penalty,
            weight=-12.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "all_body_cfg": _contact_body_cfg,
                "num_ankle_links": _num_ankle_links,
            },
        )

        self.terminations.ball_lost = DoneTerm(
            func=mdp.ball_lost_dribbling,
            params={
                "command_name": "motion",
                "max_distance": 1.0,
                "max_vel_divergence": 2.0,
                "grace_steps": 50,
            },
        )

        self.terminations.dribbling_no_contact = DoneTerm(
            func=mdp.dribbling_no_ball_contact_timeout,
            params={
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
                "grace_steps": 50,
                "max_steps_without_contact": 40,
            },
        )


@configclass
class G1TerrainDribblingAnkleDisturbEnvCfg(G1FlatDribblingEnvCfg):
    """Flat dribbling with ankle disturbances (stage-1 style pretrain for dribble)."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.motion_foot_pos.weight = 0.0

        self.events.ankle_torque_disturbance = EventTerm(
            func=mdp.apply_random_ankle_torque,
            mode="interval",
            interval_range_s=(0.1, 0.3),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_ankle_pitch_joint",
                        "left_ankle_roll_joint",
                        "right_ankle_pitch_joint",
                        "right_ankle_roll_joint",
                    ],
                ),
                "torque_range": (-15.0, 15.0),
            },
        )


@configclass
class G1FlatDribblingCGHeuristicEnvCfg(G1FlatDribblingEnvCfg):
    """Legacy CG shaping: heuristic approach/interact phases (no motion labels)."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.dribbling_phase_graph_alignment = RewTerm(
            func=mdp.dribbling_phase_graph_alignment,
            weight=6.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 0.5,
                "approach_xy_dist": 0.55,
                "approach_dist_std": 0.20,
                "push_speed_threshold": 0.22,
            },
        )


@configclass
class G1FlatCGDribblingEnvCfg(G1FlatDribblingEnvCfg):
    """Dribbling with annotated contact graph + stitched demo ``ball_pos_w``.

    Uses :class:`~soccer.tasks.tracking.mdp.commands_dribble_cg.DribbleCGMotionCommand`
    to optionally drive the sim ball along the demo trajectory (anchor-relative),
    and rewards that align sensor contact / foot side with ``dribble_cg_*`` labels.

    For the previous distance-only heuristic CG, use
    :class:`G1FlatDribblingCGHeuristicEnvCfg` / gym id
    ``Tracking-CG-Heuristic-G1-Dribbling-RNN-v0``.
    """

    def __post_init__(self):
        super().__post_init__()

        self.commands.motion.class_type = DribbleCGMotionCommand
        # This CG variant uses contact labels to teach the touch timing/side, while
        # keeping the ball task simple: spawn the ball in front and let physics move it.
        setattr(self.commands.motion, "dribble_cg_use_demo_ball", False)
        setattr(self.commands.motion, "dribble_cg_fallback_ball_mode", "front")
        setattr(self.commands.motion, "dribble_cg_front_ball_distance", 0.45)
        setattr(self.commands.motion, "dribble_cg_front_ball_lateral_offset", 0.0)
        setattr(self.commands.motion, "dribble_cg_front_ball_height", 0.11)
        # Reset robot facing world +x and place ball along world +x: lets the
        # forward-progress reward be measured against an absolute direction.
        setattr(self.commands.motion, "normalize_motion_yaw", True)

        self.observations.policy.anchor_ball_polar = ObsTerm(
            func=obs_anchor.anchor_ball_polar,
            params={"command_name": "motion"},
        )
        self.observations.critic.anchor_ball_polar = ObsTerm(
            func=obs_anchor.anchor_ball_polar,
            params={"command_name": "motion"},
        )

        mode = str(self.dribble_contact_mode).lower().strip()
        if mode == "right":
            legal_ankles = ["right_ankle_roll_link"]
            other_ankles = ["left_ankle_roll_link"]
        elif mode == "left":
            legal_ankles = ["left_ankle_roll_link"]
            other_ankles = ["right_ankle_roll_link"]
        else:
            legal_ankles = ["right_ankle_roll_link", "left_ankle_roll_link"]
            other_ankles = []
        cg_body_cfg = SceneEntityCfg(
            "robot",
            body_names=[
                *legal_ankles,
                *other_ankles,
                "right_knee_link",
                "left_knee_link",
                "right_wrist_yaw_link",
                "left_wrist_yaw_link",
            ],
        )

        self.rewards.dribbling_cg_demo_ball_tracking = RewTerm(
            func=mdp.dribbling_cg_demo_ball_tracking_exp,
            weight=4.0,
            params={"command_name": "motion", "std": 0.32},
        )
        self.rewards.dribbling_cg_contact_consistency = RewTerm(
            func=mdp.dribbling_cg_contact_consistency,
            weight=5.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
            },
        )
        self.rewards.dribbling_cg_foot_consistency = RewTerm(
            func=mdp.dribbling_cg_foot_consistency,
            weight=4.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "all_body_cfg": cg_body_cfg,
            },
        )


@configclass
class G1FlatMotionCGPretrainEnvCfg(G1FlatMotionEnvCfg):
    """Stage-1 basic-motion pretraining env, obs-compatible with the CG dribble env.

    Mirrors :class:`~soccer.tasks.tracking.config.g1.soccer_flat_env_cfg.G1FlatMotionEnvCfg`
    in every respect (rewards / terminations / commands / scene), only adding the
    ``anchor_ball_polar`` observation that
    :class:`G1FlatCGDribblingEnvCfg` also adds. Keeping the observation layout
    identical lets a checkpoint trained here be resumed by
    ``Tracking-CG-G1-Dribbling-RNN-v0`` via the standard rsl_rl ``--resume`` flow.

    Use this for the Stage-1 pretrain when running CG progressive training.
    The original :class:`G1FlatMotionEnvCfg` (``Tracking-Flat-G1-Motion-RNN-v0``)
    is left untouched so existing non-CG checkpoints stay loadable.
    """

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.anchor_ball_polar = ObsTerm(
            func=obs_anchor.anchor_ball_polar,
            params={"command_name": "motion"},
        )
        self.observations.critic.anchor_ball_polar = ObsTerm(
            func=obs_anchor.anchor_ball_polar,
            params={"command_name": "motion"},
        )

        # Keep the motion frame representation consistent with the Stage-2 CG
        # dribble env: motion's first-frame anchor is rebased to world (0, 0, z0)
        # facing +x, so the Stage-1 checkpoint is obs/state-compatible.
        setattr(self.commands.motion, "normalize_motion_yaw", True)

        # Stage-1 objective: forward locomotion with the motion's gait style.
        # The demo is slalom-around-cones, so we drop the world-frame trajectory trackers
        # and replace them with a simple "walk forward at ~target speed" task. Motion
        # style is still imitated via joint reference + anchor-relative body rewards.
        if hasattr(self.rewards, "motion_global_anchor_pos"):
            self.rewards.motion_global_anchor_pos.weight = 0.0
        if hasattr(self.rewards, "motion_global_anchor_ori"):
            self.rewards.motion_global_anchor_ori.weight = 0.0
        # Slalom side-to-side motion bleeds into per-body world velocity / angular
        # velocity references too: keep them at reduced weight so they only inform gait
        # timing rather than enforcing the lateral pattern.
        if hasattr(self.rewards, "motion_body_lin_vel"):
            self.rewards.motion_body_lin_vel.weight = 0.3
        if hasattr(self.rewards, "motion_body_ang_vel"):
            self.rewards.motion_body_ang_vel.weight = 0.3

        self.rewards.forward_velocity = RewTerm(
            func=mdp.forward_velocity_reward,
            weight=3.0,
            params={
                "command_name": "motion",
                "target_speed": 0.8,
                "std": 0.4,
            },
        )
        self.rewards.lateral_velocity_penalty = RewTerm(
            func=mdp.lateral_velocity_penalty,
            weight=-0.5,
            params={"command_name": "motion"},
        )
