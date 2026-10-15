"""Dribbling environment configurations for the G1 robot.

Inherits proximity-level tracking and adds dribbling-specific rewards:
  - velocity / proximity gates (anti static exploit); velocity match requires contact
  - unified task frame (+X fwd / +Y lat): ball spawn, velocity, proximity, obs, progress
  - optional dense foot–ball approach when not in contact (disabled for CG variant)
  - pelvis orientation vs motion reference (anti lean-back / arched torso)
  - ball horizontal speed excess penalty; anti-trap / anti-sustained-contact penalties
  - both ankles legal for gentle touches; anti-trap / sustained-contact block 夹球
  - kick–chase–kick: chase reward, rapid-retouch penalty, coast penalty only when ball is close
  - gait foot tracking between touches; reduced close-proximity / foot-hover shaping
  - pelvis anchor + task-frame mimic (strip demo yaw → +X for torso/arms/legs); relaxed leg imitation weights
  - anti-crab: task-frame forward velocity match, heading-gated touches, stronger lateral penalty
  - ``ball_lost`` and tighter ``dribbling_no_contact`` termination
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from soccer.tasks.tracking import mdp
from soccer.tasks.tracking.mdp import observations_anchor as obs_anchor
from soccer.tasks.tracking.mdp.commands_dribble_cg import DribbleCGMotionCommand
from .soccer_flat_env_cfg import (
    G1FlatMotionPretrainEnvCfg,
    G1FlatMotionStrictPretrainEnvCfg,
    G1FlatProximityEnvCfg,
    SOCCER_BALL_RADIUS,
    _STAGE1_LEG_VEL_BODY_NAMES,
    _STAGE1_LOCOMOTION_BODY_NAMES,
    _STAGE1_UPPER_BODY_NAMES,
    _apply_soccer_obs,
    _apply_stage1_ghost_ball,
)


# Control-only coordinated upper-body action subsystem.
_CONTROL_UPPER_BODY_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


@configclass
class G1FlatDribblingEnvCfg(G1FlatProximityEnvCfg):
    """Flat-ground dribbling environment."""

    # Optional override: "both" (default) allows either ankle; single-foot modes remain useful for diagnostics.
    dribble_contact_mode: str = "both"
    # ``any`` preserves the old whole-foot rule. ``instep`` accepts either
    # labelled side. ``inside_instep`` and ``outside_instep`` classify only
    # the signed lateral offset in the contacted foot's yaw frame; no
    # fore/aft or height box is part of the surface definition.
    dribble_contact_surface: str = "any"

    def __post_init__(self):
        super().__post_init__()

        # Ball-on-ground: USD had mu~0.1 and restitution 1.0 → long coast after one kick.
        # Pair higher friction with moderate bounce; add linear damping on the ball body.
        self.scene.terrain.physics_material = self.scene.terrain.physics_material.replace(
            restitution=0.22,
            static_friction=1.0,
            dynamic_friction=0.95,
        )
        self.scene.soccer_ball.spawn = self.scene.soccer_ball.spawn.replace(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # Slightly lower damping so the ball rolls between discrete touches.
                linear_damping=0.18,
                angular_damping=0.18,
            ),
        )

        # Pelvis anchor + task-frame mimic: strip demo yaw so torso/arms/legs refs face +X
        # together (legacy mode only yaw-aligns to robot pelvis, leaving arms on demo heading).
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.mimic_align_task_frame = True
        _upper_body_track_names = [
            "pelvis",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]

        # Slightly relax imitation so the policy can deviate toward the ball while
        # keeping torso/gait reference (touch-related rewards provide the main ball signal).
        if hasattr(self.rewards, "motion_body_pos"):
            self.rewards.motion_body_pos.weight = 0.72
        if hasattr(self.rewards, "foot_distance"):
            self.rewards.foot_distance.weight = 0.35
        if hasattr(self.rewards, "motion_foot_pos"):
            # Baseline foot tracking; ``dribbling_gait_foot_tracking`` adds stronger signal
            # between ball touches so the policy does not freeze in a kick-ready stance.
            self.rewards.motion_foot_pos.weight = 0.55
        if hasattr(self.rewards, "motion_body_ori"):
            self.rewards.motion_body_ori.params["body_names"] = _upper_body_track_names
        # Yaw-aligned velocity match for upper body (world-frame demo vel breaks under +X task).
        if hasattr(self.rewards, "motion_body_lin_vel"):
            self.rewards.motion_body_lin_vel = RewTerm(
                func=mdp.motion_relative_body_linear_velocity_error_exp,
                weight=0.3,
                params={
                    "command_name": "motion",
                    "std": 1.0,
                    "body_names": _upper_body_track_names,
                },
            )
        if hasattr(self.rewards, "motion_body_ang_vel"):
            self.rewards.motion_body_ang_vel = RewTerm(
                func=mdp.motion_relative_body_angular_velocity_error_exp,
                weight=0.3,
                params={
                    "command_name": "motion",
                    "std": 3.14,
                    "body_names": _upper_body_track_names,
                },
            )
        # Kick-style frozen proximity is not used for dribbling.
        if hasattr(self.rewards, "target_point_proximity"):
            self.rewards.target_point_proximity = None

        # Stronger upright / anti-arch than generic proximity alone
        if hasattr(self.rewards, "pelvis_orientation"):
            self.rewards.pelvis_orientation.weight = -2.5

        # Drop world-frame yaw tracking from the motion: zig-zag motion ("slalom around
        # cones") would otherwise force the policy to copy the S-shape heading pattern.
        # The face-ball reward below replaces it with a task-aligned heading signal.
        if hasattr(self.rewards, "motion_global_anchor_ori"):
            self.rewards.motion_global_anchor_ori = None

        self.rewards.dribbling_face_ball = RewTerm(
            func=mdp.dribbling_face_ball,
            weight=2.5,
            params={
                "command_name": "motion",
                "min_distance": 0.05,
            },
        )

        # Pelvis yaw vs task +X (``dribbling_face_ball`` already includes heading × ball-ahead).
        self.rewards.task_heading_alignment = RewTerm(
            func=mdp.task_heading_alignment_reward,
            weight=1.8,
            params={"command_name": "motion"},
        )

        self.rewards.forward_velocity = RewTerm(
            func=mdp.forward_velocity_reward,
            weight=3.0,
            params={
                "command_name": "motion",
                "target_speed": 0.55,
                "std": 0.35,
                "velocity_frame": "world",
                "min_forward_dominance": 0.55,
            },
        )
        self.rewards.lateral_velocity_penalty = RewTerm(
            func=mdp.lateral_velocity_penalty,
            weight=-1.6,
            params={
                "command_name": "motion",
                "velocity_frame": "world",
                "lateral_deadzone": 0.05,
                "lateral_scale": 0.24,
            },
        )

        mode = str(self.dribble_contact_mode).lower().strip()
        if mode not in {"right", "left", "both"}:
            raise ValueError(
                f"Unsupported dribble_contact_mode={self.dribble_contact_mode}. "
                "Expected one of: right, left, both."
            )

        surface = str(self.dribble_contact_surface).lower().strip()
        if surface not in {"any", "instep", "inside_instep", "outside_instep"}:
            raise ValueError(
                f"Unsupported dribble_contact_surface={self.dribble_contact_surface}. "
                "Expected one of: any, instep, inside_instep, outside_instep."
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
        _both_feet = ["left_ankle_roll_link", "right_ankle_roll_link"]

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
                "pelvis_speed_min": 0.35,
                "ball_speed_min": 0.12,
                # Velocity match only within a short window after a real touch (not coasting).
                "require_contact": True,
                "recent_contact_window": 8,
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
                "min_forward_dominance": 0.55,
            },
        )

        self.rewards.dribbling_dynamic_proximity = RewTerm(
            func=mdp.dribbling_dynamic_proximity,
            weight=3.0,
            params={
                "command_name": "motion",
                "near_dist": 0.28,
                "far_dist": 0.72,
                "penalty_std": 0.15,
                "pelvis_speed_min": 0.35,
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
                # In the "ball in front" corridor, strongly cut proximity without a touch.
                "no_contact_zone_damping": 0.12,
                "zone_lateral_abs_max": 0.14,
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

        # Non-CG dribbling only: CG variant sets weight=0 (foot–ball distance labels cover contact).
        self.rewards.dribbling_approach_foot_ball = RewTerm(
            func=mdp.dribbling_approach_foot_ball_distance,
            weight=1.2,
            params={
                "command_name": "motion",
                "foot_cfg": _foot_cfg,
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
                "std": 0.25,
                "pelvis_speed_min": 0.12,
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
            weight=-2.5,
            params={
                # Was 2.8 — no penalty at ~2.5 m/s where the robot loses the ball.
                "speed_cap": 1.35,
                "linear_scale": 1.2,
            },
        )

        self.rewards.dribbling_ball_coast_penalty = RewTerm(
            func=mdp.dribbling_ball_coast_without_contact_penalty,
            weight=-2.2,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
                "speed_threshold": 0.55,
                "speed_scale": 0.40,
                "max_close_xy_dist": 0.50,
            },
        )

        self.rewards.dribbling_ball_trapped_penalty = RewTerm(
            func=mdp.dribbling_ball_trapped_penalty,
            weight=-8.0,
            params={
                "command_name": "motion",
                "min_forward_x": 0.32,
                "max_ball_height": 0.20,
            },
        )

        self.rewards.dribbling_sustained_contact_penalty = RewTerm(
            func=mdp.dribbling_sustained_contact_penalty,
            weight=-6.0,
            params={
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
                "max_contact_steps": 3,
            },
        )

        self.rewards.dribbling_ball_bounce_penalty = RewTerm(
            func=mdp.dribbling_ball_bounce_penalty,
            weight=-3.0,
            params={
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
                "vz_threshold": 0.32,
            },
        )

        self.rewards.dribbling_ball_forward_progress = RewTerm(
            func=mdp.dribbling_ball_forward_progress_reward,
            weight=7.5,
            params={
                "command_name": "motion",
                "min_forward_speed": 0.42,
                "speed_scale": 0.22,
                "pelvis_speed_min": 0.25,
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 0.5,
                "require_recent_contact": True,
                "recent_contact_window": 8,
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

        self.rewards.dribbling_gait_foot_tracking = RewTerm(
            func=mdp.dribbling_gait_foot_tracking_exp,
            weight=1.4,
            params={
                "command_name": "motion",
                "std": 0.28,
                "foot_body_names": _both_feet,
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
            },
        )

        self.rewards.dribbling_chase_ball = RewTerm(
            func=mdp.dribbling_chase_ball_reward,
            weight=2.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
                "min_ball_ahead": 0.28,
                "max_chase_xy_dist": 1.05,
                "pelvis_forward_speed_min": 0.25,
                "forward_speed_scale": 0.50,
            },
        )

        self.rewards.dribbling_rapid_retouch_penalty = RewTerm(
            func=mdp.dribbling_rapid_retouch_penalty,
            weight=-6.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "force_threshold": 22.0,
                "min_steps_between_touches": 32,
                "all_body_cfg": _contact_body_cfg,
                "num_ankle_links": _num_ankle_links,
            },
        )

        self.rewards.dribbling_legal_foot_touch = RewTerm(
            func=mdp.dribbling_legal_foot_touch,
            weight=5.5,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                # Was 22 — hard kicks still counted as "gentle" legal touches.
                "force_threshold": 14.0,
                "all_body_cfg": _contact_body_cfg,
                "num_ankle_links": _num_ankle_links,
                "contact_surface": surface,
                "min_pelvis_heading": 0.55,
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
                "contact_surface": surface,
                # Preserve the strong -12 signal for knees/wrists, but an
                # ankle touch on the wrong instep side is only -3.  Contact
                # timing is separately penalized by CG premature-contact.
                "wrong_surface_penalty": 0.25,
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
                "max_steps_without_contact": 50,
            },
        )


@configclass
class G1FlatCGDribblingEnvCfg(G1FlatDribblingEnvCfg):
    """Dribbling with annotated contacts and causal contact-to-contact ball flow.

    Uses :class:`~soccer.tasks.tracking.mdp.commands_dribble_cg.DribbleCGMotionCommand`
    to optionally drive the sim ball along the demo trajectory (anchor-relative),
    and rewards that align sensor contact / foot side with ``dribble_cg_*`` labels.

    """

    def __post_init__(self):
        super().__post_init__()

        self.commands.motion.class_type = DribbleCGMotionCommand
        # This CG variant uses contact labels to teach the touch timing/side,
        # while keeping the ball task simple: spawn the ball in front and let
        # physics move it.
        setattr(self.commands.motion, "dribble_cg_use_task_frame", True)
        setattr(self.commands.motion, "dribble_cg_snap_mode", "never")
        setattr(self.commands.motion, "dribble_cg_fallback_ball_mode", "front")
        setattr(self.commands.motion, "dribble_cg_front_ball_distance", 0.45)
        setattr(self.commands.motion, "dribble_cg_front_ball_lateral_offset", 0.0)
        setattr(self.commands.motion, "dribble_cg_front_ball_height", 0.11)

        self.observations.policy.anchor_ball_polar = ObsTerm(
            func=obs_anchor.anchor_ball_polar,
            params={"command_name": "motion"},
        )
        self.observations.critic.anchor_ball_polar = ObsTerm(
            func=obs_anchor.anchor_ball_polar,
            params={"command_name": "motion"},
        )

        mode = str(self.dribble_contact_mode).lower().strip()
        surface = str(self.dribble_contact_surface).lower().strip()
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

        # Event-based CG flow: a correct touch latches its outgoing direction
        # once, then earns a short velocity-band reward and capped progress.
        # Run scripts/dribble/synthesize_dribble_ball_traj.py on labeled motions first.
        cg_flow_params = {
            "command_name": "motion",
            "ball_sensor_name": "soccer_ball_contact",
            "all_body_cfg": cg_body_cfg,
            "num_ankle_links": 2,
            "contact_surface": surface,
            "medial_y_min": 0.018,
            "contact_force_threshold": 1.0,
            "max_touch_force": 20.0,
            "release_window_steps": 8,
            "speed_lower_ratio": 0.7,
            "speed_upper_ratio": 1.6,
            "lateral_speed_std": 0.35,
            "overspeed_std": 0.5,
            "lateral_corridor_std": 0.18,
            "max_progress_rate": 6.0,
        }
        self.rewards.dribbling_cg_flow_release = RewTerm(
            func=mdp.dribbling_cg_flow_release_reward,
            weight=2.0,
            params=dict(cg_flow_params),
        )
        self.rewards.dribbling_cg_flow_progress = RewTerm(
            func=mdp.dribbling_cg_flow_progress_reward,
            weight=0.5,
            params=dict(cg_flow_params),
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
        self.rewards.dribbling_cg_premature_contact = RewTerm(
            func=mdp.dribbling_cg_premature_contact_penalty,
            weight=-6.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "contact_force_threshold": 1.0,
            },
        )
        self.rewards.dribbling_cg_foot_consistency = RewTerm(
            func=mdp.dribbling_cg_foot_consistency,
            weight=2.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "all_body_cfg": cg_body_cfg,
                "contact_surface": surface,
            },
        )

        # CG: no dense "hover foot on ball" between touches — proximity + contact labels suffice.
        self.rewards.dribbling_approach_foot_ball = None

        # Velocity / forward progress use recent sim contact, not only labeled CG frames.
        self.rewards.dribbling_velocity_tracking.params["cg_gated_contact"] = False
        self.rewards.dribbling_ball_forward_progress.params["cg_gated_contact"] = False
        self.rewards.dribbling_legal_foot_touch.params["cg_gated"] = True
        self.rewards.dribbling_rapid_retouch_penalty.params["cg_gated"] = True
        self.rewards.dribbling_rapid_retouch_penalty.params["min_steps_between_touches"] = 26


def _apply_unified_control_action_contract(cfg) -> None:
    """Install the frozen 29-D action contract used by the next training run.

    The policy keeps one action for each robot joint, so Stage-1 and Stage-2
    share the action head.  Only the selected upper-body targets are projected
    onto the PCA manifold.  A zero residual limit makes this a strict
    projection; requested, projected, limited, and executed targets remain
    available on the action term for diagnostics.
    """
    old_action_cfg = cfg.actions.joint_pos
    cfg.actions.joint_pos = mdp.UpperBodyManifoldJointPositionActionCfg(
        asset_name=old_action_cfg.asset_name,
        joint_names=old_action_cfg.joint_names,
        use_default_offset=old_action_cfg.use_default_offset,
        upper_body_joint_names=_CONTROL_UPPER_BODY_JOINT_NAMES,
        command_name="motion",
        manifold_rank=6,
        latent_std_limit=3.0,
        min_latent_limit=0.03,
        orthogonal_residual_limit=0.0,
        cutoff_frequency_hz=1.8,
        reference_target_margin=0.25,
        direct_upper_body_latent_action=False,
    )
    cfg.actions.joint_pos.scale = old_action_cfg.scale
    cfg.actions.joint_pos.offset = old_action_cfg.offset
    cfg.actions.joint_pos.preserve_order = getattr(old_action_cfg, "preserve_order", False)
    if hasattr(old_action_cfg, "clip"):
        cfg.actions.joint_pos.clip = old_action_cfg.clip

    # Recurrent action feedback and rate regularization must use the command
    # after PCA projection, joint limits, and filtering rather than the raw
    # policy value that may never reach the actuator.
    cfg.observations.policy.actions.func = mdp.effective_joint_action
    cfg.observations.policy.actions.params = {"action_name": "joint_pos"}
    cfg.observations.critic.actions.func = mdp.effective_joint_action
    cfg.observations.critic.actions.params = {"action_name": "joint_pos"}
    cfg.rewards.action_rate_l2.func = mdp.effective_action_rate_l2_clip
    cfg.rewards.action_rate_l2.params = {"action_name": "joint_pos"}
    cfg.rewards.upper_body_reference_overflow = RewTerm(
        func=mdp.upper_body_reference_overflow_penalty,
        weight=-0.05,
        params={"action_name": "joint_pos"},
    )
    cfg.rewards.upper_body_manifold_nullspace = RewTerm(
        func=mdp.upper_body_manifold_nullspace_penalty,
        weight=-0.02,
        params={"action_name": "joint_pos", "scale": 0.10},
    )


def _remove_unified_kick_only_terms(cfg) -> None:
    """Remove ball Cartesian and kick-destination signals from unified dribbling.

    ``target_point_pos`` is the simulated ball position in pelvis-frame
    Cartesian coordinates, which duplicates ``anchor_ball_polar``.  The
    destination signal and the inherited frozen-proximity reward belong to the
    kick task, not continuous dribbling.  Keep the command's internal target
    buffers untouched: generic ball reset/visualization code still owns them,
    but they are no longer policy inputs or learning objectives here.
    """
    for observations in (cfg.observations.policy, cfg.observations.critic):
        observations.target_point_pos = None
        observations.target_destination_pos_local = None

    # Keep the term absent so a later parent-config change cannot reintroduce
    # a kick-style objective into the frozen unified contract.
    if hasattr(cfg.rewards, "target_point_proximity"):
        cfg.rewards.target_point_proximity = None

    # These markers communicate kick-only targets and become misleading when
    # the associated observations/objective are absent.
    cfg.commands.motion.target_point_marker_cfg = None
    cfg.commands.motion.target_destination_marker_cfg = None


def _apply_cg_pretrain_obs(cfg) -> None:
    """``anchor_ball_polar`` on policy/critic — required for CG Stage-2 resume."""
    cfg.observations.policy.anchor_ball_polar = ObsTerm(
        func=obs_anchor.anchor_ball_polar,
        params={"command_name": "motion"},
    )
    cfg.observations.critic.anchor_ball_polar = ObsTerm(
        func=obs_anchor.anchor_ball_polar,
        params={"command_name": "motion"},
    )


def _disable_reference_ball_contact_terms(cfg) -> None:
    """Remove reference-timed ball and contact-position objectives for S3."""
    for reward_name in (
        "dribbling_cg_flow_release",
        "dribbling_cg_flow_progress",
        "dribbling_cg_contact_consistency",
        "dribbling_cg_premature_contact",
        "dribbling_cg_foot_consistency",
    ):
        if hasattr(cfg.rewards, reward_name):
            setattr(cfg.rewards, reward_name, None)

    # Keep reference gait only between touches. At contact, the task owns the
    # foot, surface, time, and ball position.
    if hasattr(cfg.rewards, "motion_foot_pos"):
        cfg.rewards.motion_foot_pos = None
    if hasattr(cfg.rewards, "dribbling_legal_foot_touch"):
        cfg.rewards.dribbling_legal_foot_touch.params["cg_gated"] = False
    if hasattr(cfg.rewards, "dribbling_rapid_retouch_penalty"):
        cfg.rewards.dribbling_rapid_retouch_penalty.params["cg_gated"] = False


def _apply_unified_local_twist_observations(cfg) -> None:
    """Install the local-twist part of the 163-D unified input contract.

    Layout is deliberately preserved as ``154-D proprioception + 3-D ball +
    3-D compatibility state + 3-D twist``.  The final three policy values are
    therefore always ``[vx_local, vy_local, wz]``.  The state field remains a
    fixed DRIBBLE one-hot for S1/S2/S3; it avoids a checkpoint shape change but
    does not encode a second, competing task command.
    """
    cfg.observations.policy.motion_ref_ang_vel = ObsTerm(
        func=mdp.motion_locomotion_ang_vel_command_local,
        params={"command_name": "motion"},
    )
    # Critic-only, simulator-derived pelvis-local linear velocity.  Do not add
    # this observation to policy: it is privileged state unavailable on hardware.
    cfg.observations.critic.base_lin_vel = ObsTerm(
        func=mdp.robot_anchor_lin_vel_local,
        params={"command_name": "motion"},
    )
    cfg.observations.policy.motion_locomotion_polar_cmd = None
    cfg.observations.critic.motion_locomotion_polar_cmd = None
    cfg.observations.policy.motion_locomotion_task_state = ObsTerm(
        func=mdp.motion_locomotion_task_state,
        params={"command_name": "motion"},
    )
    cfg.observations.critic.motion_locomotion_task_state = ObsTerm(
        func=mdp.motion_locomotion_task_state,
        params={"command_name": "motion"},
    )
    cfg.observations.policy.motion_locomotion_twist_cmd = ObsTerm(
        func=mdp.motion_locomotion_twist_command_local,
        params={"command_name": "motion"},
    )
    cfg.observations.critic.motion_locomotion_twist_cmd = ObsTerm(
        func=mdp.motion_locomotion_twist_command_local,
        params={"command_name": "motion"},
    )


def _apply_flat_reference_twist_observation(cfg) -> None:
    """Append only the reference-derived local twist to the Flat S1 layout.

    This deliberately preserves ``Tracking-Flat-G1-Motion-RNN-v0``'s
    observations, rewards, reset policy, ``MotionCommand`` and randomized
    start-ahead ball.  The only interface addition is a 3-D command
    ``[vx_local, vy_local, wz]`` computed from the motion reference.
    """
    cfg.commands.motion.locomotion_command_mode = "reference"
    cfg.commands.motion.locomotion_command_frame = "pelvis_local"
    cfg.commands.motion.locomotion_task_state_enabled = False
    cfg.observations.policy.motion_locomotion_twist_cmd = ObsTerm(
        func=mdp.motion_locomotion_twist_command_local,
        params={"command_name": "motion"},
    )
    cfg.observations.critic.motion_locomotion_twist_cmd = ObsTerm(
        func=mdp.motion_locomotion_twist_command_local,
        params={"command_name": "motion"},
    )


def _apply_unified_local_163_contract(cfg) -> None:
    """Shared 163-D/29-D deployment-oriented interface for local S1/S2/S3."""
    _apply_cg_pretrain_obs(cfg)
    cfg.observations.policy.anchor_ball_polar = ObsTerm(
        func=obs_anchor.anchor_ball_pelvis_local_polar,
        params={"command_name": "motion"},
    )
    cfg.observations.critic.anchor_ball_polar = ObsTerm(
        func=obs_anchor.anchor_ball_pelvis_local_polar,
        params={"command_name": "motion"},
    )
    _remove_unified_kick_only_terms(cfg)
    _apply_unified_control_action_contract(cfg)
    _apply_unified_local_twist_observations(cfg)


def _apply_local_twist_command_mode(cfg, mode: str) -> None:
    """Make reference and task commands use one current-pelvis convention."""
    cfg.commands.motion.locomotion_command_frame = "pelvis_local"
    cfg.commands.motion.locomotion_command_mode = mode
    cfg.commands.motion.locomotion_task_state_enabled = False
    cfg.commands.motion.locomotion_task_state_sequence = ("dribble",)
    cfg.commands.motion.mimic_align_task_frame = False
    cfg.commands.motion.mimic_align_locomotion_heading = False


def _apply_reference_local_reset(cfg) -> None:
    """Reset a local-reference stage at the unmodified demo start state.

    A local twist does not need a simulation-wide forward axis.  Keep the
    reference yaw and velocities, and make any physical ball placement depend
    only on the reference pelvis-local geometry.  Simulation domain
    randomization events remain enabled; this only removes reset-state noise
    that would otherwise desynchronise a strict reference episode.
    """
    cfg.commands.motion.reset_face_task_forward = False
    cfg.commands.motion.reset_zero_velocity = False
    cfg.commands.motion.pose_range = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    cfg.commands.motion.velocity_range = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    cfg.commands.motion.joint_position_range = (0.0, 0.0)


def _apply_local_twist_velocity_rewards(
    cfg, *, lin_weight: float, lin_std: float, yaw_weight: float, yaw_std: float
) -> None:
    """Replace world/task-frame root tracking with the active pelvis-local twist."""
    for reward_name in ("forward_velocity", "lateral_velocity_penalty", "task_heading_alignment"):
        if hasattr(cfg.rewards, reward_name):
            setattr(cfg.rewards, reward_name, None)
    if hasattr(cfg.rewards, "motion_global_anchor_pos"):
        cfg.rewards.motion_global_anchor_pos = None
    if hasattr(cfg.rewards, "motion_global_anchor_ori"):
        cfg.rewards.motion_global_anchor_ori = None
    cfg.rewards.motion_anchor_lin_vel = RewTerm(
        func=mdp.motion_anchor_local_lin_vel_tracking_exp,
        weight=lin_weight,
        params={"command_name": "motion", "std": lin_std},
    )
    cfg.rewards.motion_anchor_ang_vel = RewTerm(
        func=mdp.motion_anchor_local_ang_vel_tracking_exp,
        weight=yaw_weight,
        params={"command_name": "motion", "std": yaw_std},
    )
    if hasattr(cfg.rewards, "dribbling_velocity_tracking"):
        cfg.rewards.dribbling_velocity_tracking = None


def _apply_local_ball_objectives(cfg, *, disable_pelvis_quat_tracking: bool) -> None:
    """Remove fixed-world-forward geometry from a local-twist dribbling stage."""
    cfg.rewards.dribbling_dynamic_proximity.func = mdp.dribbling_pelvis_local_dynamic_proximity
    cfg.rewards.dribbling_ball_forward_progress.func = mdp.dribbling_command_ball_progress_reward
    cfg.rewards.dribbling_ball_trapped_penalty.func = mdp.dribbling_pelvis_local_ball_trapped_penalty
    cfg.rewards.dribbling_chase_ball.func = mdp.dribbling_command_chase_ball_reward
    cfg.rewards.dribbling_face_ball.func = mdp.dribbling_pelvis_local_face_ball
    cfg.rewards.dribbling_legal_foot_touch.params["min_pelvis_heading"] = 0.0
    if disable_pelvis_quat_tracking and hasattr(cfg.rewards, "dribbling_pelvis_quat_tracking"):
        cfg.rewards.dribbling_pelvis_quat_tracking = None


def _apply_local_task_ball_objectives(cfg) -> None:
    """S3 task version: local geometry with no reference-pelvis orientation prior."""
    _apply_local_ball_objectives(cfg, disable_pelvis_quat_tracking=True)


_S2_REWARD_NAMES = frozenset(
    {
        "motion_body_pos",
        "motion_body_ori",
        "motion_body_lin_vel",
        "motion_body_ang_vel",
        "motion_anchor_lin_vel",
        "motion_anchor_ang_vel",
        "action_rate_l2",
        "waist_action_rate_l2",
        "joint_limit",
        "undesired_contacts",
        "pelvis_orientation",
        "upper_body_reference_overflow",
        "upper_body_manifold_nullspace",
        "s2_windowed_foot_tracking",
        "s2_approach_progress",
        "s2_coarse_contact_approach",
        "s2_anchor_approach_progress_v2",
        "s2_contact_region_progress_v3",
        "s2_contact_region_arrival",
        "s2_contact_proximity",
        "s2_contact_anchor_arrival",
        "s2_physical_contact_candidate_bonus",
        "s2_effective_touch_bonus",
        "s2_next_touch_release",
        "s2_next_touch_release_v2",
        "s2_next_anchor_progress",
        "s2_sequence_progress",
        "s2_surface_style",
        "s2_excess_contact_force",
        "s2_nonfoot_ball_contact",
        "s2_wrong_foot_contact",
        "s2_wrong_side_contact",
        "s2_premature_contact",
    }
)


def _prune_s2_reward_manager(cfg) -> None:
    """Remove every inherited reward outside the final S2 contract."""
    for reward_name in dir(cfg.rewards):
        if reward_name.startswith("_") or reward_name in _S2_REWARD_NAMES:
            continue
        reward_cfg = getattr(cfg.rewards, reward_name)
        if isinstance(reward_cfg, RewTerm):
            # ``None`` removes the term from RewardManager. A zero weight would
            # keep dead configuration and runtime bookkeeping alive.
            setattr(cfg.rewards, reward_name, None)
    missing = [
        reward_name
        for reward_name in sorted(_S2_REWARD_NAMES)
        if not isinstance(getattr(cfg.rewards, reward_name, None), RewTerm)
    ]
    if missing:
        raise RuntimeError(f"S2 reward contract is incomplete: {missing}")


@configclass
class G1FlatMotionCGPretrainUnifiedS1LocalStrictEnvCfg(G1FlatMotionStrictPretrainEnvCfg):
    """S1 local strict: exact demo local twist plus strict relative motion tracking.

    The reset still starts from the exact sampled demonstration state and the
    full pose/gait imitation prior remains active.  Root supervision is local
    ``[vx, vy, wz]`` rather than a non-deployable absolute world path/yaw.
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_local_twist_command_mode(self, "reference")
        # The 163-D contract includes a ball-relative input.  Reuse the S2/S3
        # first-contact spawn without introducing any ball objective in S1.
        self.commands.motion.class_type = DribbleCGMotionCommand
        setattr(self.commands.motion, "dribble_cg_use_task_frame", False)
        setattr(self.commands.motion, "dribble_cg_snap_mode", "never")
        setattr(self.commands.motion, "dribble_cg_ball_spawn_mode", "reference_first_contact")
        # S1 is raw strict imitation: preserve every demo frame and end the
        # episode at the source boundary.  ``time_out=True`` resets LSTM
        # memory while retaining normal value bootstrapping at a time limit.
        self.commands.motion.motion_clip_end_resample = True
        self.commands.motion.motion_clip_end_terminate = True
        self.commands.motion.motion_cyclic_blend_frames = 0
        self.terminations.motion_clip_end = DoneTerm(
            func=mdp.motion_clip_finished,
            params={"command_name": "motion"},
            time_out=True,
        )
        _apply_local_twist_velocity_rewards(self, lin_weight=5.0, lin_std=0.45, yaw_weight=2.0, yaw_std=0.80)
        _apply_unified_local_163_contract(self)


@configclass
class G1FlatMotionPretrainReferenceTwistEnvCfg(G1FlatMotionPretrainEnvCfg):
    """Clean Flat Stage-1 plus a reference-derived 3-D velocity command.

    Unlike the unified S1 task, this keeps the historical Flat semantics:
    target/destination observations, ``MotionCommand``, start-ahead ball with
    lateral spawn jitter, task-frame mimic rewards, and Flat reset behavior.
    The sole addition is the reference local twist observation, yielding a
    163-D actor input and 295-D critic input without changing the Flat reward.
    """

    def __post_init__(self):
        super().__post_init__()
        _apply_flat_reference_twist_observation(self)


@configclass
class G1CleanV0Stage1EnvCfg(G1FlatMotionPretrainReferenceTwistEnvCfg):
    """Deployable clean Stage-1 prior for the v0 dribbling pipeline.

    The actor sees the same current-ball channel that Stage 2 will use, but
    the ball is a bounded, independent ghost object and no reward/termination
    consumes it.  Destination is deliberately absent, so this task's actor
    contract is 160-D (Flat 154 + ball xyz 3 + reference twist 3).
    """

    def __post_init__(self):
        super().__post_init__()

        # The Actor receives the reference command as pelvis-local
        # ``[vx, vy, wz]``.  Supervise that exact deployable quantity instead
        # of comparing linear velocity in the task/world frame, and give the
        # third command component an explicit (conservative) yaw-rate term.
        # Absolute root position/orientation remain disabled: Stage 1 learns
        # the reference motion, not one fixed placement in the world.
        self.rewards.motion_global_anchor_pos = None
        self.rewards.motion_global_anchor_ori = None
        # Their selected, pelvis-relative replacements below already cover
        # pose and leg dynamics; do not keep dead global-body telemetry terms
        # in the clean RewardManager.
        self.rewards.motion_body_lin_vel = None
        self.rewards.motion_body_ang_vel = None
        self.rewards.motion_anchor_lin_vel = RewTerm(
            func=mdp.motion_anchor_local_lin_vel_tracking_exp,
            weight=2.2,
            params={"command_name": "motion", "std": 1.0},
        )
        self.rewards.motion_anchor_ang_vel = RewTerm(
            func=mdp.motion_anchor_local_ang_vel_tracking_exp,
            weight=0.4,
            params={"command_name": "motion", "std": 1.0},
        )

        # Keep the historical Flat reset noise and interval pushes. They are
        # part of the motion-prior robustness distribution, not a ball-task
        # signal, and retaining them keeps Stage 1 matched to the verified
        # Tracking-Flat-G1-Motion-RNN-v0 recipe.
        self.commands.motion.motion_clip_end_resample = True
        self.commands.motion.motion_clip_end_terminate = True
        self.commands.motion.motion_cyclic_blend_frames = 0
        self.terminations.motion_clip_end = DoneTerm(
            func=mdp.motion_clip_finished,
            params={"command_name": "motion"},
            time_out=True,
        )

        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.soccer_ball_spawn_mode = "workspace"
        self.commands.motion.soccer_ball_workspace_x_range = (0.25, 0.80)
        self.commands.motion.soccer_ball_workspace_y_range = (-0.35, 0.35)
        self.commands.motion.soccer_ball_follow_actual_pelvis_yaw = True
        self.commands.motion.soccer_ball_generate_destination = False
        self.commands.motion.target_destination_marker_cfg = None

        # Use the real scene ball in the full pelvis frame.  This preserves
        # the historical Flat body-frame semantics and maps directly from a
        # deployable body-mounted ball measurement.
        self.observations.policy.target_point_pos = ObsTerm(
            func=mdp.ball_pos_pelvis_full,
            params={"command_name": "motion"},
        )
        self.observations.critic.target_point_pos = ObsTerm(
            func=mdp.ball_pos_pelvis_full,
            params={"command_name": "motion"},
        )
        self.observations.policy.target_destination_pos_local = None
        self.observations.critic.target_destination_pos_local = None

        # Keep ball-ground visualization/state available for playback, but
        # prevent robot-ball collision from changing the motion prior.
        _apply_stage1_ghost_ball(self)


@configclass
class G1FlatCGDribblingUnifiedS2LocalReferenceEnvCfg(G1FlatCGDribblingEnvCfg):
    """S2: learn clean frame-0 contact transitions through the full clip."""

    dribble_contact_mode: str = "both"
    dribble_contact_surface: str = "instep"
    # First validate a clean frame-0 O1/O2 physical-contact unit. Keep the
    # curriculum frozen here until the separate O2->I3 transition level is
    # implemented and validated; do not jump directly from C2 to C3.
    s2_curriculum_levels: tuple[tuple[tuple[int, float], ...], ...] = (
        ((2, 1.0),),
    )

    def __post_init__(self):
        super().__post_init__()
        # Keep the first S2 training line deterministic at the episode level.
        # The base tracking task applies random interval pushes; those change
        # the ball/body state independently of the sparse contact labels and
        # make transition failures ambiguous.  Startup domain randomization is
        # retained because it is shared with the S1 checkpoint.
        self.events.push_robot = None
        _apply_local_twist_command_mode(self, "reference")
        _apply_reference_local_reset(self)
        # S2 remains an exact demo/contact episode.  Ball, robot, and LSTM
        # all reset together at the final reference frame; no synthetic
        # bridge or unlabelled contact interval is introduced here.
        self.commands.motion.motion_clip_end_resample = True
        self.commands.motion.motion_clip_end_terminate = True
        self.commands.motion.motion_cyclic_blend_frames = 0
        self.terminations.motion_clip_end = DoneTerm(
            func=mdp.motion_clip_finished,
            params={"command_name": "motion"},
            time_out=True,
        )
        # Preserve the S1 local-reference motion prior during S2 fine-tuning.
        # Only foot *position* is released by the event-window term below;
        # changing the remaining imitation objective at the S1 -> S2 boundary
        # makes a ball reward compete with a simultaneously moving prior.
        ankle_names = ["right_ankle_roll_link", "left_ankle_roll_link"]
        non_foot_body_names = [
            body_name
            for body_name in self.commands.motion.body_names
            if body_name not in ankle_names
        ]
        self.rewards.motion_body_pos = RewTerm(
            func=mdp.motion_relative_body_position_error_exp,
            weight=1.0,
            params={
                "command_name": "motion",
                "std": 0.30,
                "body_names": non_foot_body_names,
            },
        )
        # Keep the complete S1 orientation target, including both ankles.  The
        # policy may move the labelled foot toward the physical ball, but it
        # cannot rotate the ankle to manufacture an easier anchor/side test.
        self.rewards.motion_body_ori = RewTerm(
            func=mdp.motion_relative_body_orientation_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 0.40},
        )
        self.rewards.motion_body_lin_vel = RewTerm(
            func=mdp.motion_global_body_linear_velocity_error_exp,
            weight=1.0,
            params={
                "command_name": "motion",
                "std": 1.0,
                "body_names": non_foot_body_names,
            },
        )
        self.rewards.motion_body_ang_vel = RewTerm(
            func=mdp.motion_global_body_angular_velocity_error_exp,
            weight=1.0,
            params={
                "command_name": "motion",
                "std": 3.14,
                "body_names": non_foot_body_names,
            },
        )
        _apply_local_twist_velocity_rewards(
            self, lin_weight=5.0, lin_std=0.45, yaw_weight=2.0, yaw_std=0.80
        )
        _apply_unified_local_163_contract(self)

        # Asymmetric training only: the critic sees the sparse labelled ball
        # targets and contact phase, while the actor keeps the unchanged
        # deployment-oriented 163-D position-only interface.  Appending this
        # term grows the critic from 295 to 314 inputs; checkpoint loading pads
        # only the new critic columns and leaves the S1 actor untouched.
        self.observations.critic.s2_ball_reference_context = ObsTerm(
            func=obs_anchor.s2_sparse_ball_reference_context,
            params={"command_name": "motion"},
        )

        # A valid S2 touch must come from an actual ball-foot collision.  The
        # filtered matrix excludes ground contact before any event logic sees
        # the force; filter order matches ``ankle_names`` below.
        self.scene.soccer_ball_contact.filter_prim_paths_expr = [
            "{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
            "{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        ]
        self.commands.motion.dribble_cg_contact_region_path = (
            "logs/s2_calibration/all16_contact_regions_v3_closedloop.npz"
        )

        # Isolated contact-foot ablation: retain the complete S1 motion prior
        # except for the labelled foot's position target, which ramps down
        # shortly before each reference event.  Orientation and all non-foot
        # body targets remain unchanged.
        self.rewards.s2_windowed_foot_tracking = RewTerm(
            func=mdp.dribbling_s2_windowed_foot_tracking_exp,
            weight=1.0,
            params={
                "command_name": "motion",
                "std": 0.30,
                "foot_body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
                "contact_foot_scale": 0.3,
                "contact_foot_release_seconds": 0.20,
            },
        )

        # Contact-dependent terms share one event/release state.  The target
        # body list starts with both legal ankle links; later entries are
        # explicit non-foot penalties.
        num_ankle_links = len(ankle_names)
        s2_contact_body_cfg = SceneEntityCfg(
            "robot",
            body_names=[
                *ankle_names,
                *(
                    body_name
                    for body_name in self.commands.motion.body_names
                    if body_name not in ankle_names
                ),
            ],
        )
        event_params = {
            "command_name": "motion",
            "ball_sensor_name": "soccer_ball_contact",
            "all_body_cfg": s2_contact_body_cfg,
            "num_ankle_links": num_ankle_links,
            "require_expected_foot": True,
            "target_side_enabled": True,
            # The labels identify inside/outside, not a measured clearance.
            # Use the sign in the actual ankle-local frame; do not invent a
            # centimetre margin as contact truth.
            "side_deadzone": 0.0,
            "proximity_contact_distance_max": 0.25,
            "target_region_std": 0.12,
            # Sparse supervision is active only in the labelled contiguous
            # contact segment; no synthetic pre-contact time horizon is used.
            "proximity_approach_seconds": 0.0,
            "proximity_approach_min_weight": 0.0,
            "missed_contact_grace_steps": 0,
            # Anchor position is approximate supervision.  Scale its smooth
            # reward by the configured physical ball radius and never use an
            # uncalibrated distance to reject a real PhysX contact.
            "contact_anchor_std": None,
            "contact_anchor_max_distance": None,
            # Calibrated from all 264 semantically correct exact-contact hits
            # in o1_o2_i3_static_contact_scan_v2.npz.  The rounded scales are
            # just outside the observed maxima (4.113, 4.029, 2.142 cm).
            "contact_anchor_local_sigmas": (0.042, 0.042, 0.022),
            # exp(-1/2 * 3): the boundary of the 1-sigma ellipsoid.  Every
            # positive static-scan hit passes (observed minimum quality .256).
            # C2 contact truth is physical collision + window + foot + side.
            # Keep calibrated quality as shaping/telemetry until its negative
            # class has also been calibrated; it must not reject a real touch.
            "contact_anchor_quality_min": None,
            # The event table already contains the configured +/-2-frame
            # annotation tolerance.  A physical onset must occur inside that
            # expanded window; an earlier sustained contact cannot be claimed
            # after the window opens.  No second onset grace is permitted.
            "contact_onset_early_grace_steps": 0,
            # The ankle origin is not a calibrated shoe-surface origin.  Exact
            # right-foot/outside contacts in the verified Flat Stage-1 replay
            # span slightly negative local x, so an x-sign/range gate rejects
            # real contacts.  Keep full local xyz as telemetry, but define a
            # candidate only from window + PhysX foot + expected foot + side.
            "pair_terminal_contact_region_max_distance": None,
            "coarse_contact_longitudinal_min": None,
            "coarse_contact_longitudinal_max": None,
        }
        # Termination has an explicit signature and does not consume the
        # reward-only anchor-shaping parameters.
        termination_event_params = {
            key: value
            for key, value in event_params.items()
            if key
            not in {
                "contact_anchor_std",
                "contact_anchor_max_distance",
                "contact_anchor_local_sigmas",
                "contact_anchor_quality_min",
                "contact_onset_early_grace_steps",
                "pair_terminal_contact_region_max_distance",
                "coarse_contact_longitudinal_min",
                "coarse_contact_longitudinal_max",
            }
        }
        # Isaac Lab validates term call signatures.  The small wrapper rewards
        # below therefore receive this explicit dictionary instead of a
        # variadic ``**kwargs`` signature.
        packed_event_params = {"event_params": dict(event_params)}
        self.rewards.s2_approach_progress = RewTerm(
            func=mdp.dribbling_s2_approach_progress,
            # Keep the signed foot-ball approach potential as telemetry only.
            # The Stage1 motion reference owns the pre-contact foot timing;
            # optimizing this term can make the swing foot chase the ball early.
            weight=0.0,
            params={
                "event_params": {
                    **dict(event_params),
                    "approach_distance_std": 0.40,
                    "approach_progress_max": 4.0,
                },
            },
        )
        self.rewards.s2_coarse_contact_approach = RewTerm(
            func=mdp.dribbling_s2_coarse_contact_approach,
            # Disabled: minimizing ankle-origin-to-ball-centre distance
            # conflicts with the calibrated forefoot reference geometry.
            weight=0.0,
            params={
                "event_params": {
                    **dict(event_params),
                    "coarse_approach_duration": 0.20,
                    "coarse_approach_progress_max": 4.0,
                },
            },
        )
        self.rewards.s2_anchor_approach_progress_v2 = RewTerm(
            func=mdp.dribbling_s2_anchor_approach_progress_v2,
            # Point-anchor V2 remains available only for telemetry/ablation.
            weight=0.0,
            params={
                "event_params": {
                    **dict(event_params),
                    # Three times the calibrated final-contact ellipsoid keeps
                    # acquisition informative at roughly ten-centimetre error
                    # without redefining physical-contact success.
                    "anchor_approach_duration": 0.20,
                    "anchor_approach_local_scales": (0.126, 0.126, 0.066),
                    "anchor_approach_progress_max": 4.0,
                },
            },
        )
        self.rewards.s2_contact_region_progress_v3 = RewTerm(
            func=mdp.dribbling_s2_contact_region_progress_v3,
            weight=0.0,
            params={
                "event_params": {
                    **dict(event_params),
                    "contact_region_approach_duration": 0.20,
                    "contact_region_progress_max": 4.0,
                    "contact_region_arrival_std": 0.02,
                },
            },
        )
        self.rewards.s2_contact_region_arrival = RewTerm(
            func=mdp.dribbling_s2_contact_region_arrival,
            weight=0.0,
            params={
                "event_params": {
                    **dict(event_params),
                    "contact_region_approach_duration": 0.20,
                    "contact_region_progress_max": 4.0,
                    "contact_region_arrival_std": 0.02,
                },
            },
        )
        self.rewards.s2_contact_proximity = RewTerm(
            func=mdp.dribbling_s2_contact_proximity,
            # Dense side-aware guidance bridges the released reference foot to
            # the sparse physical event. It is disabled immediately after a
            # side-valid contact succeeds.
            # First reward-cleanup ablation: keep the term/telemetry active,
            # but rely on the precise foot-local anchor-arrival objective.
            weight=0.0,
            params=dict(termination_event_params),
        )
        self.rewards.s2_contact_anchor_arrival = RewTerm(
            func=mdp.dribbling_s2_contact_anchor_arrival,
            # Low-weight local-geometry bootstrap for every event, including
            # O1.  The reference foot-local anchor is mapped through the
            # actual foot, so this guides foot placement rather than paying
            # merely for the ball's reset world position.
            weight=0.0,
            params=dict(packed_event_params),
        )
        self.rewards.s2_physical_contact_candidate_bonus = RewTerm(
            func=mdp.dribbling_s2_physical_contact_candidate_bonus,
            # Observable contact fact: the strict current-window/foot/side
            # collision advances contact ordering.  Its outgoing transition
            # is verified only by the next ordered physical contact.
            # Match the successful legacy O1 bootstrap magnitude using only
            # this reliable physical-contact fact.  Transition quality remains
            # a separate delayed +15 reward at the next ordered contact.
            weight=20.0,
            params=dict(packed_event_params),
        )
        # Delayed result reward: the following strict ordered physical contact
        # verifies the preceding transition (O2 verifies O1->O2, I3 verifies
        # O2->I3, ...).  It has no centimetre, velocity, point-anchor, or region
        # gate and does not participate in cursor advancement.
        self.rewards.s2_effective_touch_bonus = RewTerm(
            func=mdp.dribbling_s2_effective_touch_bonus,
            weight=15.0,
            params=dict(packed_event_params),
        )
        self.rewards.s2_next_touch_release = RewTerm(
            func=mdp.dribbling_s2_next_touch_release,
            # Preserve the original speed-based signal for telemetry and
            # comparison, but do not optimize it in the V2 experiment.
            weight=0.0,
            params={
                "event_params": {
                    **dict(event_params),
                    "release_window_steps": 8,
                    "release_speed_lower_ratio": 0.65,
                    "release_speed_upper_ratio": 1.35,
                    "release_lateral_speed_std": 0.35,
                    "release_overspeed_decay_ratio": 0.35,
                    "release_velocity_outlier_scale": 0.75,
                    "release_separation_min": 0.14,
                    "release_separation_full": 0.22,
                    "release_quality_threshold": 0.50,
                    "release_v2_quality_threshold": 0.05,
                    "release_min_target_distance": 0.0,
                    "release_arrival_progress_max": 4.0,
                },
            },
        )
        self.rewards.s2_next_touch_release_v2 = RewTerm(
            func=mdp.dribbling_s2_next_touch_release_v2,
            # The short release score overlaps the full inter-contact
            # potential and passed on motions that never completed O2.
            # Retain telemetry only; optimize one transition signal.
            weight=0.0,
            params={
                "event_params": {
                    **dict(event_params),
                    "release_window_steps": 8,
                    "release_speed_lower_ratio": 0.65,
                    "release_speed_upper_ratio": 1.35,
                    "release_lateral_speed_std": 0.35,
                    "release_overspeed_decay_ratio": 0.35,
                    "release_velocity_outlier_scale": 0.75,
                    "release_separation_min": 0.14,
                    "release_separation_full": 0.22,
                    "release_quality_threshold": 0.50,
                    "release_v2_quality_threshold": 0.05,
                    "release_min_target_distance": 0.0,
                    "release_arrival_progress_max": 4.0,
                },
            },
        )
        self.rewards.s2_next_anchor_progress = RewTerm(
            func=mdp.dribbling_s2_next_anchor_progress,
            # Signed position-potential toward the locked next anchor.  This
            # is the only outgoing-ball shaping signal: no reference speed,
            # reconstructed trajectory, or V3 scan region is optimized.
            weight=1.0,
            params={
                "event_params": {
                    **dict(event_params),
                    "release_window_steps": 8,
                    "release_min_target_distance": 0.0,
                    "release_arrival_progress_max": 4.0,
                },
            },
        )
        self.rewards.s2_sequence_progress = RewTerm(
            func=mdp.dribbling_s2_sequence_progress_bonus,
            # Physical contact already receives an event bonus and curriculum
            # completion.  Keep the counter as telemetry without double pay.
            weight=0.0,
            params=dict(packed_event_params),
        )
        self.rewards.s2_surface_style = RewTerm(
            func=mdp.dribbling_s2_surface_style,
            # Valid contact already requires the labelled inside/outside side;
            # retain this term for telemetry without paying a duplicate pulse.
            weight=0.0,
            params=dict(packed_event_params),
        )
        self.rewards.s2_excess_contact_force = RewTerm(
            func=mdp.dribbling_s2_excess_contact_force_penalty,
            # First clean run: do not trade away contact acquisition for an
            # uncalibrated ball-impact magnitude. Retain telemetry/ablation.
            weight=0.0,
            params={
                "ball_sensor_name": "soccer_ball_contact",
                "start_force": 150.0,
                "full_force": 250.0,
                "max_penalty": 6.0,
            },
        )
        self.rewards.s2_nonfoot_ball_contact = RewTerm(
            func=mdp.dribbling_s2_nonfoot_ball_contact_penalty,
            # The clean sensor intentionally reports only the two feet.
            # Non-foot collision handling remains in robot safety terms.
            weight=0.0,
            params=dict(termination_event_params),
        )
        self.rewards.s2_wrong_foot_contact = RewTerm(
            func=mdp.dribbling_s2_wrong_foot_contact_penalty,
            weight=-1.0,
            params=dict(termination_event_params),
        )
        self.rewards.s2_wrong_side_contact = RewTerm(
            func=mdp.dribbling_s2_wrong_side_contact_penalty,
            # Wrong-side contact also cannot unlock release/curriculum. This
            # pulse makes an outer/inner switch failure explicit.
            weight=-1.5,
            params=dict(packed_event_params),
        )
        self.rewards.s2_premature_contact = RewTerm(
            func=mdp.dribbling_s2_premature_contact_penalty,
            weight=-1.0,
            params=dict(termination_event_params),
        )

        # The inherited dribbling/CG classes define many objectives for other
        # tasks. S2 exposes only its exact 28-term contract to RewardManager
        # (20 optimized terms plus eight zero-weight telemetry/ablation slots).
        _prune_s2_reward_manager(self)

        # Ball-lost is distance-only and debounced for 0.2 s.  Reference
        # ankle/wrist height and fixed no-touch terminations are removed.
        self.terminations.ball_lost.params.update(
            {
                "max_distance": 1.10,
                "max_vel_divergence": 0.0,
                "grace_steps": 10,
                "max_consecutive_steps": 10,
            }
        )
        self.terminations.dribbling_no_contact = None
        if hasattr(self.terminations, "ee_body_pos"):
            self.terminations.ee_body_pos = None
        # A missed event is an episode outcome, not an early termination.  A
        # fixed prefix endpoint preserves comparable data and lets PPO observe
        # the failed transition through its whole labelled window.
        self.terminations.missed_valid_contact = None
        self.terminations.invalid_acquisition_start = None

        # The ball always follows physics. Only the side-aware event region,
        # event approach/contact/release objectives and split penalties supervise it. Its
        # reset location is the selected first event's reference target,
        # represented in the reference pelvis-local frame rather than task +X.
        setattr(self.commands.motion, "dribble_cg_use_task_frame", False)
        setattr(self.commands.motion, "dribble_cg_snap_mode", "never")
        setattr(self.commands.motion, "dribble_cg_ball_spawn_mode", "reference_first_contact")
        setattr(self.commands.motion, "dribble_cg_curriculum_levels", self.s2_curriculum_levels)
        setattr(
            self.commands.motion,
            "dribble_cg_curriculum_required_contacts",
            # Zero means that every selected event in the full clip must be
            # completed consecutively, rather than accepting an eight-touch
            # prefix as completion of a longer reference.
            tuple(2 for _ in self.s2_curriculum_levels),
        )
        setattr(self.commands.motion, "dribble_cg_curriculum_start_level", 0)
        # Compatibility-only: S2 no longer constructs a seconds-based window
        # around one rising-edge frame; it uses the full labelled segment plus
        # the explicit frame tolerance below.
        setattr(self.commands.motion, "dribble_cg_contact_window_seconds", 0.10)
        # Manual video labels and 50-Hz PhysX contact onset can differ by a
        # small number of frames.  Accept collisions within two frames of
        # either end of the full annotated segment; event-table construction
        # rejects any expansion that would overlap adjacent O1/O2 events.
        setattr(
            self.commands.motion,
            "dribble_cg_contact_window_tolerance_frames",
            2,
        )
        setattr(self.commands.motion, "dribble_cg_missed_contact_grace_steps", 0)
        setattr(self.commands.motion, "dribble_cg_release_window_steps", 8)
        # C2 requires only the ordered physical O1/O2 contacts. Keep the
        # rollout alive through the complete unselected I3 window (plus one
        # settling step) so O2 can settle and its signed outgoing position
        # progress remains observable. I3 receives no contact reward here.
        setattr(self.commands.motion, "dribble_cg_curriculum_lookahead_contact_phases", 1)
        setattr(self.commands.motion, "dribble_cg_acquisition_initial_probability", 0.0)
        setattr(self.commands.motion, "dribble_cg_acquisition_min_history_steps", 20)
        setattr(self.commands.motion, "dribble_cg_acquisition_max_history_steps", 30)
        # Per level: (exact-reference probability, maximum initial radius).
        # Every clean-curriculum level preserves the exact frame-0 state; all
        # later ball error must arise from physical policy contacts.
        setattr(
            self.commands.motion,
            "dribble_cg_curriculum_ball_spawn_jitter",
            tuple((1.00, 0.00) for _ in self.s2_curriculum_levels),
        )
        if not (
            len(self.s2_curriculum_levels)
            == len(self.commands.motion.dribble_cg_curriculum_required_contacts)
            == len(self.commands.motion.dribble_cg_curriculum_ball_spawn_jitter)
        ):
            raise ValueError(
                "S2 curriculum levels, required-contact counts, and spawn "
                "jitter schedules must have identical lengths."
            )
        setattr(
            self.commands.motion,
            "dribble_cg_ball_spawn_safe_forward_range",
            (-0.04, 0.12),
        )
        setattr(
            self.commands.motion,
            "dribble_cg_ball_spawn_safe_side_range",
            (0.06, 0.14),
        )
        setattr(self.commands.motion, "dribble_cg_ball_spawn_jitter_min", 0.0)
        setattr(self.commands.motion, "dribble_cg_ball_spawn_jitter_max", 0.0)
        setattr(self.commands.motion, "dribble_cg_require_surface_labels", True)
        setattr(self.commands.motion, "dribble_cg_require_flow_labels", False)

        # Isaac Lab runs this term before CommandManager.reset(), allowing it
        # to evaluate the just-finished episode and promote the sampler before
        # the next episode is drawn.
        self.curriculum.s2_contact_levels = CurrTerm(
            func=mdp.s2_contact_level_curriculum,
            params={
                "command_name": "motion",
                "min_evaluation_episodes": 1024,
                "required_consecutive_passes": 2,
                "sequence_completion_threshold": 0.60,
                "contact_success_threshold": 0.60,
                # A partial level is defined only by real accepted contacts.
                # The next level's newly required PhysX contact directly
                # tests whether the previous touch produced the transition;
                # no invented position radius is used for promotion.
                "require_terminal_transition": False,
                "max_fall_rate": 0.05,
                "fall_termination_terms": ("anchor_pos_z", "anchor_ori"),
            },
        )


@configclass
class G1FlatCGDribblingUnifiedS2T2LocalReferenceEnvCfg(
    G1FlatCGDribblingUnifiedS2LocalReferenceEnvCfg
):
    """Legacy T2-shaped variant of the ordered physical-contact curriculum."""

    # One level: require O1/O2 in order and retain I3 as an unselected
    # lookahead phase for outgoing-position telemetry.
    s2_curriculum_levels: tuple[tuple[tuple[int, float], ...], ...] = (
        ((2, 1.0),),
    )

    def __post_init__(self):
        super().__post_init__()
        event_params = dict(self.rewards.s2_contact_region_progress_v3.params["event_params"])
        transition_params = {
            "command_name": "motion",
            "event_params": event_params,
            "progress_max": 4.0,
            "arrival_std": 0.02,
        }
        self.rewards.s2_transition_region_progress = RewTerm(
            func=mdp.dribbling_s2_transition_region_progress,
            weight=0.0,
            params=dict(transition_params),
        )
        self.rewards.s2_transition_region_arrival = RewTerm(
            func=mdp.dribbling_s2_transition_region_arrival,
            weight=0.0,
            params=dict(transition_params),
        )
        # Keep the physical task at exactly O1/O2, but continue through the
        # complete unselected I3 window for transition telemetry.
        setattr(
            self.commands.motion,
            "dribble_cg_curriculum_lookahead_contact_phases",
            1,
        )
        setattr(
            self.commands.motion,
            "dribble_cg_curriculum_lookahead_start_level",
            0,
        )
        self.curriculum.s2_contact_levels.params["require_terminal_transition"] = False
        self.curriculum.s2_contact_levels.params["terminal_transition_start_level"] = 1


@configclass
class G1FlatCGDribblingReferenceTwistS2EnvCfg(
    G1FlatCGDribblingUnifiedS2LocalReferenceEnvCfg
):
    """C2 -> T2 -> C3 on the verified Flat+reference-twist Stage-1 interface.

    The inherited class supplies the ordered PhysX contact cursor and sparse
    outgoing-anchor shaping.  The three levels intentionally have distinct
    semantics: Level 0 learns the ordered O1/O2 contacts, Level 1 requires the
    ball to reach the immutable offline I3 target during the annotated I3
    window, and Level 2 adds the real I3 physical contact.
    This subclass changes only the observation layout back to the one used by
    ``Tracking-Flat-G1-Motion-RNN-reference-twist-v1``:

    ``Flat proprioception + target point + destination + reference twist``.

    The actor therefore receives the same semantics as the checkpoint it is
    warm-started from.  The critic still receives the inherited privileged S2
    sparse-contact context (the 19 additional values).
    """

    # C2 and T2 deliberately select the same two-contact prefix; their
    # promotion criteria differ below. C3 then selects the third contact.
    s2_curriculum_levels: tuple[tuple[tuple[int, float], ...], ...] = (
        ((2, 1.0),),
        ((2, 1.0),),
        ((3, 1.0),),
    )

    def __post_init__(self):
        super().__post_init__()

        # Every sparse ball node is an offline annotation in the clean
        # reference frame. Map it once through the episode reset XY+yaw
        # transform; policy root/foot motion must never move the target.
        self.commands.motion.dribble_cg_episode_fixed_contact_targets = True

        # Restore the verified C2 -> C3 contact-acquisition pulse. Transition
        # quality is trained separately by the signed Euclidean next-anchor
        # potential and the absolute I3-window tracking objective.
        self.rewards.s2_physical_contact_candidate_bonus.weight = 20.0

        # The generic clean parent is also used by frozen C2 ablations and
        # therefore defaults every level to two required contacts.  This task's
        # three levels must instead be evaluated against their actual ordered
        # prefix lengths.
        self.commands.motion.dribble_cg_curriculum_required_contacts = (2, 2, 3)
        # L0 is contact-only C2. L1 is the distinct T2 ball-tracking stage:
        # signed progress is active only for O2 -> I3 and promotion uses the
        # absolute minimum fixed-target distance inside the I3 window. L2
        # verifies the same transition with a real third physical contact.
        self.commands.motion.dribble_cg_terminal_progress_levels = (1,)
        self.commands.motion.dribble_cg_terminal_transition_mode = "absolute_window_min"
        # The success scale comes from the simulated ball geometry rather
        # than an independent centimetre threshold.
        self.commands.motion.dribble_cg_ball_radius = SOCCER_BALL_RADIUS
        self.curriculum.s2_contact_levels.params.update(
            {
                # L0: ordered O1/O2 only. L1: ordered O1/O2 plus absolute I3
                # window tracking. L2 is the terminal three-contact task and
                # is not promoted further.
                "sequence_completion_thresholds": (0.80, 0.80, 0.60),
                "require_terminal_transition": True,
                "terminal_transition_start_level": 1,
            }
        )

        # The formal route registers only mechanisms that affect optimization.
        # Legacy point/region/release ablations remain available in their old
        # tasks, but do not occupy RewardManager slots here. The delayed
        # closure term is also removed: the same O2 collision must not receive
        # both the candidate pulse and a second contact pulse.
        self.commands.motion.dribble_cg_contact_region_path = ""
        for reward_name in (
            "s2_approach_progress",
            "s2_coarse_contact_approach",
            "s2_anchor_approach_progress_v2",
            "s2_contact_region_progress_v3",
            "s2_contact_region_arrival",
            "s2_contact_proximity",
            "s2_contact_anchor_arrival",
            "s2_effective_touch_bonus",
            "s2_next_touch_release",
            "s2_next_touch_release_v2",
            "s2_sequence_progress",
            "s2_surface_style",
            "s2_excess_contact_force",
            "s2_nonfoot_ball_contact",
        ):
            setattr(self.rewards, reward_name, None)

        # All active ball terms consume one canonical event-state contract.
        # This avoids the historical "first RewardManager caller wins" cache
        # behavior, where next-anchor parameters could be silently ignored if
        # the contact candidate term ran first in the same step.
        clean_event_params = dict(
            self.rewards.s2_physical_contact_candidate_bonus.params[
                "event_params"
            ]
        )
        clean_event_params.update(
            {
                "release_min_target_distance": 0.0,
                "release_arrival_progress_max": 4.0,
            }
        )
        self.rewards.s2_physical_contact_candidate_bonus.params = {
            "event_params": dict(clean_event_params)
        }
        self.rewards.s2_next_anchor_progress.params = {
            "event_params": dict(clean_event_params)
        }
        self.rewards.s2_wrong_foot_contact.params = {
            "event_params": dict(clean_event_params)
        }
        self.rewards.s2_wrong_side_contact.params = {
            "event_params": dict(clean_event_params)
        }
        self.rewards.s2_premature_contact.params = {
            "event_params": dict(clean_event_params)
        }
        self.rewards.s2_next_anchor_window_tracking = RewTerm(
            func=mdp.dribbling_s2_next_anchor_window_tracking,
            # Same maximum event budget as one +20 physical-contact pulse.
            # The continuous [0,1] quality comes only from absolute D_min / Rball.
            weight=20.0,
            params={"event_params": dict(clean_event_params)},
        )

        # Restore the exact Flat Stage-1 target/destination terms.  The
        # unified parent replaces these six values with a 3-D polar ball term
        # plus a 3-D task-state term; equal dimensionality is not semantic
        # compatibility for an RNN warm start.
        _apply_soccer_obs(self)
        self.observations.policy.anchor_ball_polar = None
        self.observations.critic.anchor_ball_polar = None
        self.observations.policy.motion_locomotion_task_state = None
        self.observations.critic.motion_locomotion_task_state = None
        self.observations.policy.motion_locomotion_polar_cmd = None
        self.observations.critic.motion_locomotion_polar_cmd = None

        # The Flat checkpoint's existing reference angular-velocity term must
        # remain in the same slot; the appended twist is the only new actor
        # input introduced by the verified Stage-1 task.
        self.observations.policy.motion_ref_ang_vel = ObsTerm(
            func=mdp.motion_anchor_ang_vel,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        self.observations.critic.motion_ref_ang_vel = ObsTerm(
            func=mdp.motion_anchor_ang_vel,
            params={"command_name": "motion"},
        )

        # Restore the actual Flat Stage-1 action path.  The unified parent
        # installs the upper-body manifold action; that has the same 29-D
        # shape but different execution semantics, so it must not be used for
        # a warm start from the verified Flat checkpoint.
        old_action_cfg = self.actions.joint_pos
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name=old_action_cfg.asset_name,
            joint_names=[".*"],
            use_default_offset=old_action_cfg.use_default_offset,
        )
        self.actions.joint_pos.scale = old_action_cfg.scale
        self.actions.joint_pos.offset = old_action_cfg.offset
        self.observations.policy.actions = ObsTerm(func=mdp.last_action)
        self.observations.critic.actions = ObsTerm(func=mdp.last_action)
        self.rewards.action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clip, weight=-1.0e-1)
        # Manifold-action telemetry has no meaning with the restored plain
        # joint-position action and is removed rather than kept at weight zero.
        self.rewards.upper_body_reference_overflow = None
        self.rewards.upper_body_manifold_nullspace = None

        # Restore Flat Stage-1 reset semantics.  The ball remains at the
        # annotated first-contact placement for C2, while robot/reference
        # initialization and domain-randomization conventions are unchanged.
        self.commands.motion.reset_face_task_forward = True
        self.commands.motion.reset_zero_velocity = True
        self.commands.motion.pose_range = {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        }
        self.commands.motion.velocity_range = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.2, 0.2),
            "roll": (-0.52, 0.52),
            "pitch": (-0.52, 0.52),
            "yaw": (-0.78, 0.78),
        }
        self.commands.motion.joint_position_range = (-0.1, 0.1)
        self.commands.motion.curve_offset_range = {
            "radius": (0.0, 0.0),
            "lateral_spawn_jitter": 0.05,
            "height": SOCCER_BALL_RADIUS,
        }

        # Keep the Flat Stage-1 motion prior exactly, with the single intended
        # C2 exception: the event-window foot-position term controls the feet.
        self.rewards.motion_body_pos.params["body_names"] = list(_STAGE1_LOCOMOTION_BODY_NAMES)
        self.rewards.motion_body_pos.weight = 1.0
        self.rewards.motion_body_ori.params["body_names"] = list(_STAGE1_LOCOMOTION_BODY_NAMES)
        self.rewards.motion_body_ori.weight = 1.0
        self.rewards.motion_body_lin_vel = None
        self.rewards.motion_body_ang_vel = None
        self.rewards.motion_upper_body_pos = RewTerm(
            func=mdp.motion_relative_body_position_error_exp,
            weight=0.85,
            params={"command_name": "motion", "std": 0.3, "body_names": list(_STAGE1_UPPER_BODY_NAMES)},
        )
        self.rewards.motion_upper_body_ori = RewTerm(
            func=mdp.motion_relative_body_orientation_error_exp,
            weight=0.35,
            params={"command_name": "motion", "std": 0.4, "body_names": list(_STAGE1_UPPER_BODY_NAMES)},
        )
        self.rewards.motion_leg_lin_vel = RewTerm(
            func=mdp.motion_relative_body_linear_velocity_error_exp,
            weight=0.4,
            params={"command_name": "motion", "std": 1.0, "body_names": list(_STAGE1_LEG_VEL_BODY_NAMES)},
        )
        self.rewards.motion_leg_ang_vel = RewTerm(
            func=mdp.motion_relative_body_angular_velocity_error_exp,
            weight=0.4,
            params={"command_name": "motion", "std": 3.14, "body_names": list(_STAGE1_LEG_VEL_BODY_NAMES)},
        )
        # Match the clean Stage-1 root-motion contract exactly.  Stage 2 may
        # release the labelled foot in contact windows, but it must not change
        # the meaning or reward coordinates of the reference twist.
        self.rewards.motion_global_anchor_pos = None
        self.rewards.motion_global_anchor_ori = None
        self.rewards.motion_anchor_lin_vel = RewTerm(
            func=mdp.motion_anchor_local_lin_vel_tracking_exp,
            weight=2.2,
            params={"command_name": "motion", "std": 1.0},
        )
        self.rewards.motion_anchor_ang_vel = RewTerm(
            func=mdp.motion_anchor_local_ang_vel_tracking_exp,
            weight=0.4,
            params={"command_name": "motion", "std": 1.0},
        )
        self.rewards.motion_foot_pos = None


@configclass
class G1CleanV0Stage2C2EnvCfg(G1FlatCGDribblingReferenceTwistS2EnvCfg):
    """Clean v0 C2 on the same 160-D deployable Actor as Stage 1.

    The verified CUDA contact core is retained, while outgoing targets use
    immutable reference pelvis-yaw XY nodes.  I3 arrival is introduced only
    by the later T2 configuration.
    """

    s2_curriculum_levels: tuple[tuple[tuple[int, float], ...], ...] = (
        ((2, 1.0),),
    )

    def __post_init__(self):
        super().__post_init__()

        # Keep the Flat reset pose/velocity/joint perturbations restored by
        # the parent.  They were already part of the successful motion prior
        # and are compatible with the episode-fixed node transform.  Interval
        # pushes remain disabled in the focused interaction stages and can be
        # reintroduced only after the ordered sequence is stable.

        self.commands.motion.soccer_ball_generate_destination = False
        self.commands.motion.dribble_cg_episode_fixed_contact_targets = True
        # The clean contact core needs one filtered matrix containing both
        # legal ankles and the remaining robot links.  Foot identity is still
        # decided from the body index; the wider filter only makes a filtered
        # non-foot collision an explicit failure instead of silently treating
        # it as a ball-force event.  Ground is not part of this robot-only
        # filter, so it cannot contaminate the foot channels.
        self.scene.soccer_ball_contact.filter_prim_paths_expr = [
            "{ENV_REGEX_NS}/Robot/.*",
        ]
        self.commands.motion.dribble_cg_contact_region_path = ""
        # C2 is evaluated through the O2 window only. T2 explicitly turns
        # the one-event lookahead back on so its I3 window can be observed.
        self.commands.motion.dribble_cg_curriculum_lookahead_contact_phases = 0
        self.commands.motion.dribble_cg_curriculum_start_level = 0
        # The historical reference-twist parent owns a three-level probe and
        # therefore writes (2, 2, 3).  Each diagnostic fixture must publish a
        # tuple matching its own number of curriculum levels.
        self.commands.motion.dribble_cg_curriculum_required_contacts = (2,)

        self.observations.policy.target_point_pos = ObsTerm(
            func=mdp.ball_pos_pelvis_full,
            params={"command_name": "motion"},
        )
        self.observations.critic.target_point_pos = ObsTerm(
            func=mdp.ball_pos_pelvis_full,
            params={"command_name": "motion"},
        )
        self.observations.policy.target_destination_pos_local = None
        self.observations.critic.target_destination_pos_local = None

        # C2 is contact-only.  T2 owns the next-window arrival pulse.
        self.rewards.s2_next_anchor_window_tracking = None
        # Clean-v0 has one physical-contact discovery pulse.  The historical
        # effective/closure term paid again on the same O2 onset and is not a
        # separate fact, so remove it rather than leaving a zero-weight
        # wrapper in the active task.
        self.rewards.s2_physical_contact_candidate_bonus.weight = 5.0
        self.rewards.s2_effective_touch_bonus = None
        for reward_name in (
            "s2_approach_progress",
            "s2_coarse_contact_approach",
            "s2_anchor_approach_progress_v2",
            "s2_contact_region_progress_v3",
            "s2_contact_region_arrival",
            "s2_contact_proximity",
            "s2_contact_anchor_arrival",
            "s2_next_touch_release",
            "s2_next_touch_release_v2",
            "s2_sequence_progress",
            "s2_surface_style",
            "s2_excess_contact_force",
            "s2_nonfoot_ball_contact",
        ):
            setattr(self.rewards, reward_name, None)
        event_params = dict(
            self.rewards.s2_physical_contact_candidate_bonus.params["event_params"]
        )
        event_params["relative_node_targets"] = True
        # C2 is deliberately acquisition-only.  The outgoing ball tracking
        # potential is introduced by T2, so C2 cannot earn a transition return
        # before the ordered O1/O2 contact fact is stable.
        self.rewards.s2_next_anchor_progress = None
        self.commands.motion.dribble_cg_terminal_progress_levels = ()
        for reward_name in (
            "s2_physical_contact_candidate_bonus",
            "s2_next_anchor_progress",
            "s2_wrong_foot_contact",
            "s2_wrong_side_contact",
            "s2_premature_contact",
        ):
            term = getattr(self.rewards, reward_name, None)
            if isinstance(term, RewTerm):
                term.params = {"event_params": dict(event_params)}
        self.terminations.s2_nonfoot_contact = DoneTerm(
            func=mdp.dribbling_s2_nonfoot_contact_termination,
            params={"event_params": dict(event_params)},
        )
        self.terminations.s2_missed_contact = DoneTerm(
            func=mdp.dribbling_s2_missed_contact_termination,
            params={"event_params": dict(event_params)},
        )


@configclass
class G1CleanV0Stage2T2EnvCfg(G1CleanV0Stage2C2EnvCfg):
    """C2 plus absolute XY arrival at the next sparse node (T2)."""

    s2_curriculum_levels: tuple[tuple[tuple[int, float], ...], ...] = (
        ((2, 1.0),),
        ((2, 1.0),),
    )

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.dribble_cg_curriculum_lookahead_contact_phases = 1
        self.commands.motion.dribble_cg_curriculum_lookahead_levels = (1,)
        self.commands.motion.dribble_cg_curriculum_start_level = 0
        self.commands.motion.dribble_cg_curriculum_required_contacts = (2, 2)
        self.commands.motion.dribble_cg_terminal_progress_levels = (1,)
        self.commands.motion.dribble_cg_terminal_transition_mode = "absolute_window_min"
        self.commands.motion.dribble_cg_ball_radius = SOCCER_BALL_RADIUS
        self.curriculum.s2_contact_levels.params.update(
            {
                "require_terminal_transition": True,
                "terminal_transition_start_level": 1,
                "terminal_transition_levels": (1,),
                "sequence_completion_thresholds": (0.80, 0.80),
            }
        )
        event_params = dict(
            self.rewards.s2_physical_contact_candidate_bonus.params["event_params"]
        )
        event_params["relative_node_targets"] = True
        self.rewards.s2_next_anchor_progress = RewTerm(
            func=mdp.dribbling_s2_next_anchor_progress,
            weight=1.0,
            params={"event_params": dict(event_params)},
        )
        self.rewards.s2_next_anchor_window_tracking = RewTerm(
            func=mdp.dribbling_s2_next_anchor_window_tracking,
            weight=20.0,
            params={"event_params": event_params},
        )


@configclass
class G1CleanV0Stage2C3EnvCfg(G1CleanV0Stage2T2EnvCfg):
    """T2 followed by a real ordered I3 physical contact (C3)."""

    s2_curriculum_levels: tuple[tuple[tuple[int, float], ...], ...] = (
        ((2, 1.0),),
        ((2, 1.0),),
        ((3, 1.0),),
    )

    def __post_init__(self):
        super().__post_init__()
        # C3 ends at the I3 contact window.  The following I4 event is not
        # needed until the C4/T4 stages.
        self.commands.motion.dribble_cg_curriculum_lookahead_contact_phases = 0
        self.commands.motion.dribble_cg_curriculum_start_level = 0
        self.commands.motion.dribble_cg_curriculum_required_contacts = (2, 2, 3)
        # Only the T2 level (L1) evaluates the fixed I3 arrival.  At L2 the
        # same target remains telemetry while the strict ordered third
        # contact is the curriculum truth.
        self.commands.motion.dribble_cg_terminal_progress_levels = (1,)
        self.curriculum.s2_contact_levels.params.update(
            {
                "sequence_completion_thresholds": (0.80, 0.80, 0.60),
                "require_terminal_transition": True,
                "terminal_transition_start_level": 1,
                "terminal_transition_levels": (1,),
            }
        )


@configclass
class G1CleanV0Stage2C4EnvCfg(G1CleanV0Stage2C3EnvCfg):
    """Complete one ordered O/O/I/I contact unit."""

    s2_curriculum_levels: tuple[tuple[tuple[int, float], ...], ...] = (
        ((2, 1.0),),
        ((2, 1.0),),
        ((3, 1.0),),
        ((4, 1.0),),
    )

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.dribble_cg_curriculum_lookahead_contact_phases = 0
        self.commands.motion.dribble_cg_curriculum_start_level = 0
        self.commands.motion.dribble_cg_curriculum_required_contacts = (2, 2, 3, 4)
        # L1 remains T2.  L2/L3 are physical C3/C4 and therefore do not get
        # an additional terminal arrival requirement in this class.
        self.commands.motion.dribble_cg_terminal_progress_levels = (1,)
        self.curriculum.s2_contact_levels.params.update(
            {
                "sequence_completion_thresholds": (0.80, 0.80, 0.60, 0.60),
                "require_terminal_transition": True,
                "terminal_transition_start_level": 1,
                "terminal_transition_levels": (1,),
            }
        )


@configclass
class G1CleanV0Stage2T4EnvCfg(G1CleanV0Stage2C4EnvCfg):
    """C4 plus absolute arrival of the next-cycle O1 node (T4)."""

    s2_curriculum_levels: tuple[tuple[tuple[int, float], ...], ...] = (
        ((2, 1.0),),
        ((2, 1.0),),
        ((3, 1.0),),
        ((4, 1.0),),
        ((4, 1.0),),
    )

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.dribble_cg_curriculum_lookahead_contact_phases = 1
        self.commands.motion.dribble_cg_curriculum_lookahead_levels = (1, 4)
        self.commands.motion.dribble_cg_curriculum_start_level = 0
        self.commands.motion.dribble_cg_curriculum_required_contacts = (2, 2, 3, 4, 4)
        # L1 evaluates O2->I3; L4 evaluates I4->next-cycle O1.  The
        # intermediate C3/C4 levels remain contact-only after their own
        # physical node is reached.
        self.commands.motion.dribble_cg_terminal_progress_levels = (1, 4)
        self.curriculum.s2_contact_levels.params.update(
            {
                "sequence_completion_thresholds": (0.80, 0.80, 0.60, 0.60, 0.60),
                "require_terminal_transition": True,
                "terminal_transition_start_level": 1,
                "terminal_transition_levels": (1, 4),
            }
        )


@configclass
class G1CleanV0Stage2EnvCfg(G1CleanV0Stage2T4EnvCfg):
    """Formal clean Stage 2 with one internal dribbling curriculum.

    The policy and optimizer remain continuous while the environment advances
    through contact acquisition, sparse-node ball tracking, physical
    verification, and a repeated O/O/I/I motif.  These are curriculum levels,
    not separate checkpoint interfaces.
    """

    s2_curriculum_levels: tuple[tuple[tuple[int, float], ...], ...] = (
        ((2, 1.0),),  # L0 C2: O1/O2 physical contacts.
        ((2, 1.0),),  # L1 T2: C2 + I3 fixed-node arrival.
        ((3, 1.0),),  # L2 C3: add physical I3 contact.
        ((4, 1.0),),  # L3 C4: complete one O/O/I/I unit.
        ((4, 1.0),),  # L4 T4: arrive at next-cycle O1.
        ((8, 1.0),),  # L5 Repeat: two ordered O/O/I/I units.
    )

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.dribble_cg_curriculum_start_level = 0
        self.commands.motion.dribble_cg_curriculum_required_contacts = (
            2,
            2,
            3,
            4,
            4,
            8,
        )
        # Only transition levels need an unselected next contact window and
        # an absolute fixed-node arrival result. Physical-contact levels are
        # judged directly by their newly required PhysX onset.
        self.commands.motion.dribble_cg_curriculum_lookahead_contact_phases = 1
        self.commands.motion.dribble_cg_curriculum_lookahead_levels = (1, 4)
        self.commands.motion.dribble_cg_terminal_progress_levels = (1, 4)
        self.curriculum.s2_contact_levels.params.update(
            {
                "sequence_completion_thresholds": (
                    0.80,
                    0.80,
                    0.60,
                    0.60,
                    0.60,
                    0.50,
                ),
                "require_terminal_transition": True,
                "terminal_transition_start_level": 1,
                "terminal_transition_levels": (1, 4),
            }
        )


@configclass
class G1FlatCGDribblingReferenceTwistS2Level1WindowOutcomeProbeBaseCfg(
    G1FlatCGDribblingReferenceTwistS2EnvCfg
):
    """Matched Level-1 probe with the outcome state machine always enabled."""

    def __post_init__(self):
        super().__post_init__()

        # A missing/legacy checkpoint must still enter the intended probe
        # level.  A restored model_16000 checkpoint already contains Level 1.
        self.commands.motion.dribble_cg_curriculum_start_level = 1
        self.commands.motion.dribble_cg_enable_legacy_window_outcome = True
        # Freeze the probe at Level 1; promotion would change the contact task
        # and invalidate the control/outcome comparison.
        self.curriculum.s2_contact_levels.params["min_evaluation_episodes"] = 1_000_000_000

        # Reuse the exact packed event parameters of next-anchor progress.
        # Both arms register and execute this term.  Only its scalar weight is
        # changed by the outcome arm below.
        self.rewards.s2_next_window_outcome = RewTerm(
            func=mdp.dribbling_s2_next_window_outcome,
            weight=0.0,
            params={
                "event_params": dict(
                    self.rewards.s2_next_anchor_progress.params["event_params"]
                )
            },
        )


@configclass
class G1FlatCGDribblingReferenceTwistS2Level1WindowOutcomeControlCfg(
    G1FlatCGDribblingReferenceTwistS2Level1WindowOutcomeProbeBaseCfg
):
    """Control arm: full outcome telemetry, zero optimization weight."""


@configclass
class G1FlatCGDribblingReferenceTwistS2Level1WindowOutcome3Cfg(
    G1FlatCGDribblingReferenceTwistS2Level1WindowOutcomeProbeBaseCfg
):
    """Treatment arm: the sole config difference is outcome weight 0 -> 3."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.s2_next_window_outcome.weight = 3.0


@configclass
class G1FlatCGDribblingReferenceTwistS2Level1DelayedSettlementCfg(
    G1FlatCGDribblingReferenceTwistS2Level1WindowOutcomeProbeBaseCfg
):
    """Budget-conserving probe: small discovery pulse, delayed result settlement."""

    def __post_init__(self):
        super().__post_init__()
        # Do not add reward budget: move 15 of the current event's immediate
        # contact value to its outgoing transition result.
        self.rewards.s2_physical_contact_candidate_bonus.weight = 5.0
        self.rewards.s2_delayed_transition_settlement = RewTerm(
            func=mdp.dribbling_s2_delayed_transition_settlement,
            weight=15.0,
            params={
                "event_params": dict(
                    self.rewards.s2_next_anchor_progress.params["event_params"]
                )
            },
        )


@configclass
class G1FlatCGDribblingReferenceTwistS2Level1FixedTargetAuditCfg(
    G1FlatCGDribblingReferenceTwistS2Level1WindowOutcomeProbeBaseCfg
):
    """No-training audit using reset-fixed XY+yaw sparse contact targets."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.dribble_cg_episode_fixed_contact_targets = True
        self.commands.motion.dribble_cg_debug_live_target_comparison = True
        self.commands.motion.debug_vis = True


@configclass
class G1FlatCGDribblingReferenceTwistS2Level1FixedTargetProgressProbeCfg(
    G1FlatCGDribblingReferenceTwistS2Level1FixedTargetAuditCfg
):
    """Level-1 causal probe: only replace live targets with reset-fixed targets.

    Reward weights, optimizer state, curriculum level, and the signed progress
    implementation are inherited unchanged from the matched Level-1 control.
    """


@configclass
class G1FlatCGDribblingUnifiedS3LocalTaskEnvCfg(G1FlatCGDribblingEnvCfg):
    """S3 local task: sampled local twist and flexible physical instep control."""

    dribble_contact_mode: str = "both"
    dribble_contact_surface: str = "instep"

    def __post_init__(self):
        super().__post_init__()
        _apply_local_twist_command_mode(self, "resampled")
        _apply_reference_local_reset(self)
        self.commands.motion.motion_clip_end_resample = False
        self.commands.motion.motion_clip_end_terminate = False
        self.commands.motion.motion_cyclic_blend_frames = 25
        self.commands.motion.locomotion_cmd_lin_vel_range = {
            "x": (0.0, 1.50),
            "y": (-0.50, 0.50),
            "z": (0.0, 0.0),
        }
        self.commands.motion.locomotion_cmd_ang_vel_range = {
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-0.80, 0.80),
        }
        self.commands.motion.locomotion_cmd_duration_range = (1.5, 3.0)
        # Exact rest commands are required for a playback plan to start/end
        # cleanly without reintroducing the old IDLE/STOP state command.
        self.commands.motion.locomotion_cmd_stationary_probability = 0.10
        # 24 env steps per rollout * 1000 PPO updates.  All local S3 reward
        # terms see the same blended command, preventing a hidden objective
        # change during the S2 -> S3 distribution transition.
        self.commands.motion.locomotion_twist_reference_blend_env_steps = 24_000
        _apply_local_twist_velocity_rewards(self, lin_weight=5.0, lin_std=0.65, yaw_weight=1.0, yaw_std=1.50)
        _apply_unified_local_163_contract(self)
        _disable_reference_ball_contact_terms(self)
        _apply_local_task_ball_objectives(self)

        # Playback command plans may request a full reset after their final
        # segment.  The underlying flag is always false during resampled
        # training, so this term is inert for PPO.
        self.terminations.locomotion_manual_sequence_end = DoneTerm(
            func=mdp.locomotion_manual_sequence_finished,
            params={"command_name": "motion"},
        )

        setattr(self.commands.motion, "dribble_cg_use_task_frame", False)
        setattr(self.commands.motion, "dribble_cg_snap_mode", "never")
        setattr(self.commands.motion, "dribble_cg_ball_spawn_mode", "reference_first_contact")
        setattr(self.commands.motion, "dribble_cg_fixed_touch_foot", None)
        setattr(self.commands.motion, "dribble_cg_fixed_touch_surface", None)


@configclass
class G1CleanV1BallTrackEnvCfg(G1CleanV0Stage2C2EnvCfg):
    """Clean-v1: Stage-1 motion prior + one dense pelvis-frame ball-tracking term.

    No contact-event machinery is used for learning.  Each episode runs the whole
    clip and ends only on a fall, a lost ball, or the clip end.  Touch counts,
    window hits, correct foot/side and node arrival are evaluation metrics (see
    ``scripts/dribble/probe_policy_foot_ball.py``), never rewards or terminations.
    """

    ball_track_weight: float = 2.0
    ball_track_std: float = 0.5
    ball_track_shape: str = "exponential"

    def __post_init__(self):
        super().__post_init__()
        cmd = self.commands.motion
        # ``0`` = every labelled event, i.e. a full-clip episode from frame 0.
        cmd.dribble_cg_curriculum_levels = (((0, 1.0),),)
        cmd.dribble_cg_curriculum_required_contacts = (0,)
        cmd.dribble_cg_curriculum_start_level = 0
        cmd.dribble_cg_curriculum_lookahead_contact_phases = 0
        cmd.dribble_cg_terminal_progress_levels = ()
        self.curriculum.s2_contact_levels = None

        for name in (
            "s2_physical_contact_candidate_bonus",
            "s2_wrong_foot_contact",
            "s2_wrong_side_contact",
            "s2_premature_contact",
            "s2_next_anchor_progress",
            "s2_next_anchor_window_tracking",
        ):
            setattr(self.rewards, name, None)
        self.terminations.s2_missed_contact = None
        self.terminations.s2_nonfoot_contact = None

        self.rewards.ball_pelvis_tracking = RewTerm(
            func=mdp.dribbling_ball_pelvis_tracking_exp,
            weight=self.ball_track_weight,
            params={
                "command_name": "motion",
                "std": self.ball_track_std,
                "shape": self.ball_track_shape,
            },
        )
