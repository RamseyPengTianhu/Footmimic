"""Anchor-based kick environment configurations.

These configs inherit from the existing soccer pipeline but override
observations and reward weights to implement the decoupled anchor
architecture (Sprint 2).  They are fully isolated — the original
MoCap-based environments are **not modified**.

Hierarchy
---------
G1TerrainMotionEnvCfg   (Stage 1 base — existing)
 └─ G1AnchorTrackingEnvCfg   (Stage 1 anchor — NEW)

G1FlatKickEnvCfg        (Stage 2 base — existing)
 └─ G1AnchorKickEnvCfg       (Stage 2 anchor — NEW)
"""

import math

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm

from soccer.tasks.tracking import mdp
from soccer.tasks.tracking.mdp import observations_anchor as obs_anchor
from .soccer_flat_env_cfg import (
    G1TerrainMotionEnvCfg,
    G1FlatKickEnvCfg,
    SOCCER_BALL_RADIUS,
)


# ---------------------------------------------------------------------------
# Stage 1 — Anchor Tracking (egocentric obs, no soccer reward)
# ---------------------------------------------------------------------------

@configclass
class G1AnchorTrackingEnvCfg(G1TerrainMotionEnvCfg):
    """Stage 1 with egocentric ball observation.

    Changes vs baseline ``G1TerrainMotionEnvCfg``:
      - Actor: ``target_point_pos``  →  ``anchor_ball_polar (d, cos_θ, sin_θ)``
      - Critic: keeps privileged world-coordinate observations
      - Velocity tracking rewards down-weighted (0.3× original)
    """

    def __post_init__(self):
        super().__post_init__()

        # --- Actor observation: replace world-coord ball pos with polar ---
        self.observations.policy.target_point_pos = ObsTerm(
            func=obs_anchor.anchor_ball_polar,
            params={"command_name": "motion"},
        )

        # Critic keeps the original privileged observations (no change).

        # --- Down-weight velocity tracking (reduce HMR noise sensitivity) ---
        if hasattr(self.rewards, "motion_body_lin_vel"):
            self.rewards.motion_body_lin_vel.weight = 0.3
        if hasattr(self.rewards, "motion_body_ang_vel"):
            self.rewards.motion_body_ang_vel.weight = 0.3


# ---------------------------------------------------------------------------
# Stage 2 — Anchor Kick (ankle masking + egocentric obs)
# ---------------------------------------------------------------------------

@configclass
class G1AnchorKickEnvCfg(G1FlatKickEnvCfg):
    """Stage 2 with egocentric observations and ankle masking.

    Changes vs baseline ``G1FlatKickEnvCfg``:
      - Actor: ``target_point_pos``  →  ``anchor_ball_polar``
      - ``motion_body_pos``: kick-foot ankle **excluded** from tracking
      - Velocity tracking rewards down-weighted
      - Critic: keeps privileged world-coordinate observations
    """

    def __post_init__(self):
        super().__post_init__()

        # --- Actor observation: egocentric ball ---
        self.observations.policy.target_point_pos = ObsTerm(
            func=obs_anchor.anchor_ball_polar,
            params={"command_name": "motion"},
        )

        # --- Ankle masking: remove kick foot from body tracking ---
        # Override motion_body_pos to exclude right_ankle_roll_link.
        # The kick foot is freed from tracking so it can reach the ball.
        self.rewards.motion_body_pos = RewTerm(
            func=mdp.motion_relative_body_position_error_exp,
            weight=1.0,
            params={
                "command_name": "motion",
                "std": 0.3,
                "body_names": [
                    "pelvis",
                    "left_hip_roll_link",
                    "left_knee_link",
                    "left_ankle_roll_link",   # support foot: KEEP tracking
                    "right_hip_roll_link",
                    "right_knee_link",
                    # "right_ankle_roll_link",  # kick foot: FREED
                    "torso_link",
                    "left_shoulder_roll_link",
                    "left_elbow_link",
                    "left_wrist_yaw_link",
                    "right_shoulder_roll_link",
                    "right_elbow_link",
                    "right_wrist_yaw_link",
                ],
            },
        )

        # --- Ankle masking for orientation too ---
        if hasattr(self, "motion_body_ori"):
            self.motion_body_ori = RewTerm(
                func=mdp.motion_relative_body_orientation_error_exp,
                weight=1.0,
                params={
                    "command_name": "motion",
                    "std": 0.4,
                    "body_names": [
                        "pelvis",
                        "left_hip_roll_link",
                        "left_knee_link",
                        "left_ankle_roll_link",
                        "right_hip_roll_link",
                        "right_knee_link",
                        # "right_ankle_roll_link",  # kick foot: FREED
                        "torso_link",
                        "left_shoulder_roll_link",
                        "left_elbow_link",
                        "left_wrist_yaw_link",
                        "right_shoulder_roll_link",
                        "right_elbow_link",
                        "right_wrist_yaw_link",
                    ],
                },
            )

        # --- Down-weight velocity tracking ---
        if hasattr(self.rewards, "motion_body_lin_vel"):
            self.rewards.motion_body_lin_vel.weight = 0.3
        if hasattr(self.rewards, "motion_body_ang_vel"):
            self.rewards.motion_body_ang_vel.weight = 0.3


# ---------------------------------------------------------------------------
# Stage 2 CG — Soft Contact Graph Kick (Sprint 4)
# ---------------------------------------------------------------------------

@configclass
class G1AnchorCGKickEnvCfg(G1FlatKickEnvCfg):
    """Stage 2 with Soft Contact Graph: time-gated rewards and dynamic masking.

    Key differences from G1AnchorKickEnvCfg (Sprint 2):
      - Ankle masking is DYNAMIC: right ankle tracked during CG=0, freed during CG=1
      - target_point_contact is TIME-GATED: only rewards during CG=1
      - early_collision_penalty: penalises ball contact during CG=0
      - interaction_termination: kills episode if ball not kicked after kick window
      - Keeps Sprint 2's egocentric polar observations and velocity downweight
    """

    def __post_init__(self):
        super().__post_init__()

        # --- Actor observation: egocentric ball (same as Sprint 2) ---
        self.observations.policy.target_point_pos = ObsTerm(
            func=obs_anchor.anchor_ball_polar,
            params={"command_name": "motion"},
        )

        # --- Dynamic ankle masking body pos (replaces static masking) ---
        # CG=0: ALL bodies tracked (right ankle included for stable gait)
        # CG=1: right ankle EXCLUDED (free to swing at ball)
        self.rewards.motion_body_pos = RewTerm(
            func=mdp.dynamic_ankle_masking_body_pos,
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
                    "right_ankle_roll_link",  # included — dynamically masked in CG=1
                    "torso_link",
                    "left_shoulder_roll_link",
                    "left_elbow_link",
                    "left_wrist_yaw_link",
                    "right_shoulder_roll_link",
                    "right_elbow_link",
                    "right_wrist_yaw_link",
                ],
                "kick_foot_name": "right_ankle_roll_link",
                "cg_margin": 5,
            },
        )


        # --- Time-gated kick reward (replace original target_point_contact) ---
        self.rewards.target_point_contact = RewTerm(
            func=mdp.time_gated_contact,
            weight=50.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 10,
                "foot_cfg": self.foot_cfg,
                "cg_margin": 5,
            },
        )

        # --- Early collision penalty (CG=0: don't bump the ball while running) ---
        self.rewards.early_collision_penalty = RewTerm(
            func=mdp.early_collision_penalty,
            weight=-15.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 5.0,
                "cg_margin": 5,
            },
        )

        # --- Ankle lock during kick contact (rigid striking surface) ---
        self.ankle_cfg = SceneEntityCfg(
            "robot",
            joint_names=[
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
            ],
        )
        self.rewards.ankle_lock_on_contact = RewTerm(
            func=mdp.ankle_lock_on_contact,
            weight=-0.0,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 5.0,
                "ankle_cfg": self.ankle_cfg,
                "cg_margin": 5,
            },
        )

        # --- Foot contact sensor for support foot stability prior ---
        # DISABLED: investigating whether this sensor affects physics simulation
        # from isaaclab.sensors import ContactSensorCfg
        # self.scene.foot_contact = ContactSensorCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/.*",
        #     history_length=1,
        #     force_threshold=0.0,
        #     debug_vis=False,
        # )

        # --- Support foot brake prior (DISABLED) ---
        # from v71-v74 experiments: all support foot rewards degrade v3 performance
        # self.rewards.support_foot_placement = RewTerm(
        #     func=mdp.support_foot_stability_prior,
        #     weight=0.0,
        #     params={
        #         "command_name": "motion",
        #         "support_foot_name": "left_ankle_roll_link",
        #         "near_ball_dist": 0.8,
        #         "near_temp": 0.2,
        #         "contact_threshold": 20.0,
        #         "use_hard_contact_gate": True,
        #         "ball_sensor_name": "soccer_ball_contact",
        #         "ball_horizontal_force_threshold": 5.0,
        #         "vel_std": 0.2,
        #         "yaw_std": 0.6,
        #         "stable_weight": 0.3,
        #         "yaw_weight": 0.7,
        #         "use_region_reward": False,
        #         "contact_sensor_name": "foot_contact",
        #         "cg_margin": 5,
        #     },
        # )

        # --- Support contact quality bonus (DISABLED) ---
        # self.rewards.support_contact_quality_bonus = RewTerm(
        #     func=mdp.support_contact_quality_bonus,
        #     weight=0.1,
        #     params={
        #         "command_name": "motion",
        #         "support_foot_name": "left_ankle_roll_link",
        #         "ball_sensor_name": "soccer_ball_contact",
        #         "horizontal_force_threshold": 10.0,
        #         "foot_cfg": self.foot_cfg,
        #         "cg_margin": 5,
        #         "contact_threshold": 20.0,
        #         "yaw_std": 0.6,
        #         "contact_sensor_name": "foot_contact",
        #     },
        # )
        # --- Post-strike stabilization (v8a) ---
        # Dense reward ONLY after ball contact + 3 frames delay.
        # Capped at 25 frames to prevent reward inflation.
        # Cannot interfere with kick acquisition.
        # Goal: reduce Fall% (esp. motion_1: 28% Fall)
        # max total reward = 0.05 × 25 = 1.25 (vs contact=50)
        self.rewards.post_strike_stability = RewTerm(
            func=mdp.post_strike_stability,
            weight=0.05,
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "horizontal_force_threshold": 5.0,
                "post_delay": 3,
                "post_duration": 25,
                "tilt_std": 0.3,
                "angvel_std": 1.0,
                "tilt_weight": 0.6,
                "angvel_weight": 0.4,
            },
        )

        # --- Push ball placement farther forward to reduce CG=0 collisions ---
        self.commands.motion.curve_offset_range = {
            "radius": (0.0, 0.4),   # was (-0.25, 0.25): now 0~0.4m forward only
            "arc_angle": math.pi / 9,
            "height": 0.11,
        }

        # --- Interaction termination (must kick or die) ---
        self.terminations.interaction_fail = DoneTerm(
            func=mdp.interaction_termination,
            params={
                "command_name": "motion",
                "ball_speed_threshold": 0.3,
                "grace_frames": 10,
            },
        )

        # --- Down-weight velocity tracking (same as Sprint 2) ---
        if hasattr(self.rewards, "motion_body_lin_vel"):
            self.rewards.motion_body_lin_vel.weight = 0.3
        if hasattr(self.rewards, "motion_body_ang_vel"):
            self.rewards.motion_body_ang_vel.weight = 0.3


# ---------------------------------------------------------------------------
# Stage 2 v9a — Adaptive Kick (rolling ball + position randomization)
# ---------------------------------------------------------------------------

@configclass
class G1AnchorAdaptiveKickEnvCfg(G1AnchorCGKickEnvCfg):
    """v9a: Rolling ball with same observation space (160D).

    Stage 1 checkpoint can be loaded directly (no obs change).
    Changes:
      - Ball rolls towards robot at 0.3~1.0 m/s with lateral jitter
      - Ball XY perturbed ±5cm each episode
      - Post-strike stability reward disabled
    """

    def __post_init__(self):
        super().__post_init__()

        # --- Ball position randomization ---
        self.commands.motion.ball_xy_perturbation = 0.05  # ±5cm

        # --- Rolling ball: towards robot ---
        self.commands.motion.enable_soccer_ball_init_vel = True
        self.commands.motion.soccer_ball_init_lin_vel_range = {
            "approach_speed": (0.3, 1.0),   # m/s towards robot
            "lateral": (-0.2, 0.2),          # m/s lateral jitter
        }

        # --- Remove post_strike_stability ---
        if hasattr(self.rewards, "post_strike_stability"):
            self.rewards.post_strike_stability.weight = 0.0


# ---------------------------------------------------------------------------
# Stage 2 SM — State Machine Kick (APPROACH/STRIKE distance trigger)
# ---------------------------------------------------------------------------

@configclass
class G1AnchorStateMachineKickEnvCfg(G1AnchorKickEnvCfg):
    """Stage 2 with distance-triggered state machine.

    Inherits sprint 2 changes (polar obs + ankle masking) and adds:
      - ``AnchorMotionCommand`` with dual APPROACH / STRIKE bank
      - Approach motions: ``*_approach.npz`` in motion_path
      - Strike motions: ``*_strike.npz`` in motion_path
      - Transition trigger: ball distance ≤ 0.8m
    """

    def __post_init__(self):
        super().__post_init__()

        # Swap command class to anchor state-machine variant.
        from soccer.tasks.tracking.mdp.commands_anchor import AnchorMotionCommand
        self.commands.motion.class_type = AnchorMotionCommand

        # NOTE: strike_motion_files will be populated at runtime by the
        # training script via the same --motion_path mechanism.
        # The AnchorMotionCommand expects cfg.strike_motion_files to be set.
        # Default to empty; the training script scanner will fill it.
        if not hasattr(self.commands.motion, "strike_motion_files"):
            self.commands.motion.strike_motion_files = []
        if not hasattr(self.commands.motion, "strike_trigger_distance"):
            self.commands.motion.strike_trigger_distance = 0.8
        if not hasattr(self.commands.motion, "kick_foot_body_name"):
            self.commands.motion.kick_foot_body_name = "right_ankle_roll_link"


# ===========================================================================
# 隔离测试用 Ablation 配置
# ===========================================================================

# ---------------------------------------------------------------------------
# 测试 2: velocity 降权 + ankle masking，但不改球位观测（保留 xyz）
# ---------------------------------------------------------------------------

@configclass
class G1AblationXyzKickEnvCfg(G1FlatKickEnvCfg):
    """Ablation Test 2: keep xyz ball obs, only apply velocity downweight + ankle masking.
    
    如果此配置能收敛 → 极坐标观测是崩溃原因
    """

    def __post_init__(self):
        super().__post_init__()

        # 球位观测：不动！保留原始 constant_target_point_pos (xyz)

        # Ankle masking: 同 G1AnchorKickEnvCfg
        self.rewards.motion_body_pos = RewTerm(
            func=mdp.motion_relative_body_position_error_exp,
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
                    # "right_ankle_roll_link",  # kick foot: FREED
                    "torso_link",
                    "left_shoulder_roll_link",
                    "left_elbow_link",
                    "left_wrist_yaw_link",
                    "right_shoulder_roll_link",
                    "right_elbow_link",
                    "right_wrist_yaw_link",
                ],
            },
        )

        # Velocity downweight: 同 G1AnchorKickEnvCfg
        if hasattr(self.rewards, "motion_body_lin_vel"):
            self.rewards.motion_body_lin_vel.weight = 0.3
        if hasattr(self.rewards, "motion_body_ang_vel"):
            self.rewards.motion_body_ang_vel.weight = 0.3


# ---------------------------------------------------------------------------
# 测试 3: 只改极坐标观测，不动 velocity / ankle
# ---------------------------------------------------------------------------

@configclass
class G1AblationPolarOnlyKickEnvCfg(G1FlatKickEnvCfg):
    """Ablation Test 3: only polar obs change, keep everything else as baseline.
    
    如果此配置崩溃 → 极坐标观测是崩溃原因
    如果此配置能收敛 → 问题在 velocity/ankle 组合
    """

    def __post_init__(self):
        super().__post_init__()

        # 只改球位观测为极坐标
        self.observations.policy.target_point_pos = ObsTerm(
            func=obs_anchor.anchor_ball_polar,
            params={"command_name": "motion"},
        )

        # velocity 权重：不动！保持 1.0
        # ankle masking：不动！保持全身追踪


# ===========================================================================
# v10 — Event-Conditioned Kick Decoder (MLP + History)
# ===========================================================================

from soccer.tasks.tracking.mdp import observations_v10 as obs_v10
from soccer.tasks.tracking.mdp import rewards_v10


@configclass
class G1EventConditionedKickEnvCfg(G1AnchorCGKickEnvCfg):
    """v10: Event-conditioned kick with MLP + flattened history.

    Completely new observation architecture (~420D):
      - Current proprio: ang_vel(3) + gravity(3) + joint_pos(29) + joint_vel(29) = 64D
      - History: joint_pos×3f(87) + joint_vel×3f(87) + action×3f(87) + last_action(29) = 290D
      - Ball history: ball_pos_local×10f = 30D
      - Event condition: 8D
      - Ball-foot relation: 22D
      - Weak prior: 8D
      Total: ~422D

    Reward structure (v10.1b nominal recovery):
      - body_prior: 0.7 (inherit from base, down-weighted)
      - foot_ball_rel: 0.05 → 0.2 (ramp)
      - contact_graph: 0.05 → 0.2 (ramp)
      - ball_outcome: 50 (kept)
      - BC regularization: via external training script

    MLP policy: [512, 256, 128] → 29D action (no LSTM).
    """

    def __post_init__(self):
        super().__post_init__()

        # ===== OBSERVATION: Replace entire policy obs with v10 design =====
        # Remove all inherited motion-tracking observations
        # We build from scratch with the new v10 obs groups.

        # NOTE: The history buffer (Group 2 & 3) is managed by the training
        # script / wrapper, not as an ObsTerm. The env computes the per-step
        # signals; the wrapper stacks them into history and concatenates.
        # ObsTerms here produce the per-step components.

        # --- Group 1: Current proprio (handled by base obs) ---
        # Keep: projected_gravity, base_ang_vel, joint_pos, joint_vel, actions
        # Remove: command (motion ref), motion_ref_ang_vel

        # Remove motion-tracking obs that don't exist in v10
        if hasattr(self.observations.policy, "command"):
            self.observations.policy.command = None
        if hasattr(self.observations.policy, "motion_ref_ang_vel"):
            self.observations.policy.motion_ref_ang_vel = None

        # Remove old ball obs (replaced by v10 ball-foot relation)
        if hasattr(self.observations.policy, "target_point_pos"):
            self.observations.policy.target_point_pos = None

        # --- Group 4: Event condition (8D) ---
        self.observations.policy.event_condition = ObsTerm(
            func=obs_v10.v10_event_condition,
            params={"command_name": "motion"},
        )

        # --- Group 5: Ball-foot relation (22D) ---
        self.observations.policy.ball_foot_relation = ObsTerm(
            func=obs_v10.v10_ball_foot_relation,
            params={
                "command_name": "motion",
                "swing_foot_body": "right_ankle_roll_link",
                "support_foot_body": "left_ankle_roll_link",
            },
        )

        # --- Group 6: Event-warped motor prior (40D) ---
        self.observations.policy.motor_prior = ObsTerm(
            func=obs_v10.v10_motor_prior,
            params={
                "command_name": "motion",
                "swing_foot_body": "right_ankle_roll_link",
                "support_foot_body": "left_ankle_roll_link",
            },
        )

        # ===== REWARDS: Event-conditioned structure =====
        # Keep body tracking rewards but down-weight for v10.1b
        # body_prior ≈ 0.7 (inherited from base)
        self.rewards.motion_global_anchor_pos.weight = 0.35
        self.rewards.motion_global_anchor_ori.weight = 0.35

        # Down-weight velocity tracking aggressively (not primary signal)
        if hasattr(self.rewards, "motion_body_lin_vel"):
            self.rewards.motion_body_lin_vel.weight = 0.1
        if hasattr(self.rewards, "motion_body_ang_vel"):
            self.rewards.motion_body_ang_vel.weight = 0.1

        # --- v10 foot-ball relative reward (phase-aware) ---
        self.rewards.foot_ball_relative = RewTerm(
            func=rewards_v10.r_foot_ball_relative,
            weight=0.05,  # v10.1b: start low, ramp to 0.2
            params={
                "command_name": "motion",
                "swing_foot_body": "right_ankle_roll_link",
                "support_foot_body": "left_ankle_roll_link",
                "std_prestrike": 0.3,
                "std_strike": 0.15,
                "std_support": 0.25,
            },
        )

        # --- v10 contact graph match reward ---
        self.rewards.contact_graph_match = RewTerm(
            func=rewards_v10.r_contact_graph_match,
            weight=0.05,  # v10.1b: start low, ramp to 0.2
            params={
                "command_name": "motion",
                "ball_sensor_name": "soccer_ball_contact",
                "support_foot_body": "left_ankle_roll_link",
            },
        )

        # ===== BALL CONFIG: static for v10.1, randomized later =====
        # Disable rolling ball (v10.1 = static ball)
        self.commands.motion.enable_soccer_ball_init_vel = False
        self.commands.motion.ball_xy_perturbation = 0.0


# ---------------------------------------------------------------------------
# Stage C — VQ Student Kick (Reference-Free, Interaction-First)
# ---------------------------------------------------------------------------

@configclass
class G1VQStudentKickEnvCfg(G1FlatKickEnvCfg):
    """Reference-free student environment for VQ PPO (Stage C).

    Design philosophy:
      - **No motion reference in rewards**: All motion tracking rewards zeroed.
      - **No CG dependency**: Does NOT inherit from G1AnchorCGKickEnvCfg.
        CG labels (kick_frame, kick_end_frame) are reference-dependent.
      - **Attempt-based rewards**: Uses AttemptEventTracker (geometry/event-gated,
        no CG, no kick_leg annotation). Kick foot hardcoded to right foot
        (matches current training data).
      - **No kick_leg from motion**: Old rewards (target_point_contact,
        sideways_kick, ball_speed_reward, ball_velocity_direction_alignment)
        read cmd.kick_leg — a motion annotation. Replaced by attempt_* rewards.

    The LatentPPOEnvWrapper (train_latent_v2_ppo.py) handles:
      - Stripping motion reference from PPO obs (--policy_obs_mode task_features)
      - Decoding VQ latent → joint actions via frozen decoder
      - Code hold (code_hold=2)
    """

    def __post_init__(self):
        super().__post_init__()

        from soccer.tasks.tracking.mdp import rewards_student

        # --- Actor observation: egocentric ball (polar coordinates) ---
        self.observations.policy.target_point_pos = ObsTerm(
            func=obs_anchor.anchor_ball_polar,
            params={"command_name": "motion"},
        )

        # =====================================================================
        # ZERO ALL MOTION TRACKING REWARDS
        # =====================================================================
        for attr_name in [
            "motion_body_pos", "motion_body_ori", "motion_foot_pos",
            "motion_body_lin_vel", "motion_body_ang_vel",
            "motion_global_anchor_pos", "motion_global_anchor_ori",
            "foot_distance",
        ]:
            term = getattr(self.rewards, attr_name, None)
            if term is not None and hasattr(term, "weight"):
                term.weight = 0.0

        # =====================================================================
        # ZERO OLD CG/KICK_LEG-DEPENDENT REWARDS
        # These read cmd.kick_leg or use kick_tracker which depends on
        # motion annotations. Replaced by attempt_* rewards below.
        # =====================================================================
        # target_point_contact: uses kick_leg via foot_cfg expected side
        self.rewards.target_point_contact.weight = 0.0
        # sideways_kick: uses kick_leg for expected lateral direction
        self.rewards.sideways_kick.weight = 0.0
        # ball_speed_reward: uses kick_tracker which reads kick_leg
        self.rewards.ball_speed_reward.weight = 0.0
        # ball_velocity_direction_alignment: same dependency
        self.rewards.ball_velocity_direction_alignment.weight = 0.0

        # =====================================================================
        # ATTEMPT-BASED REWARDS (geometry/event-gated, zero CG dependency)
        # Uses AttemptEventTracker: right foot hardcoded, no motion annotation.
        # =====================================================================

        # Primary positive signal: first clean contact (attempt + hit in window)
        self.rewards.attempt_clean_contact = RewTerm(
            func=rewards_student.attempt_clean_contact,
            weight=50.0,
            params={},
        )

        # Ball speed reward ONLY after clean first contact (not late fallback)
        self.rewards.attempt_ball_speed = RewTerm(
            func=rewards_student.attempt_ball_speed,
            weight=10.0,
            params={},
        )

        # Ball direction reward ONLY after clean first contact (not late fallback)
        self.rewards.attempt_direction_alignment = RewTerm(
            func=rewards_student.attempt_direction_alignment,
            weight=30.0,
            params={},
        )

        # Penalty: contact after missed attempt (late fallback / "补射")
        self.rewards.attempt_late_fallback = RewTerm(
            func=rewards_student.attempt_late_fallback_penalty,
            weight=-15.0,
            params={},
        )

        # Penalty: attempt window expired without hitting ball (empty swing)
        self.rewards.attempt_miss = RewTerm(
            func=rewards_student.attempt_miss_penalty,
            weight=-5.0,
            params={},
        )

        # Penalty: near ball for >1 sec without attempting to kick
        self.rewards.attempt_no_attempt = RewTerm(
            func=rewards_student.attempt_no_attempt_penalty,
            weight=-0.5,
            params={},
        )

        # Post-strike stability (gated by physical contact event, no CG)
        self.rewards.attempt_post_stability = RewTerm(
            func=rewards_student.attempt_post_stability,
            weight=0.05,
            params={},
        )

        # =====================================================================
        # DENSE SHAPING REWARDS (geometric, zero CG)
        # v3 experiment showed no improvement; disabled (w=0) for clean ablation.
        # Keep code for future experiments.
        # =====================================================================

        # Approach shaping: rewards closing in on ball before attempt
        self.rewards.attempt_approach_shaping = RewTerm(
            func=rewards_student.attempt_approach_shaping,
            weight=0.0,  # disabled
            params={},
        )

        # Prestrike shaping: rewards foot-ball proximity during attempt window
        self.rewards.attempt_prestrike_shaping = RewTerm(
            func=rewards_student.attempt_prestrike_shaping,
            weight=0.0,  # disabled
            params={},
        )

        # =====================================================================
        # KEEP: CG-free inherited rewards
        # =====================================================================
        # target_point_proximity (w=1.0): pure distance, no CG ✅
        # pelvis_orientation (w=-1.0): upright regularization, no CG ✅
        # action_rate_l2 (w=-0.1): smoothness, no CG ✅
        # waist_action_rate_l2 (w=-0.25): smoothness, no CG ✅

        # Remove old post_strike_stability that uses kick_tracker (CG-adjacent)
        if hasattr(self.rewards, "post_strike_stability"):
            self.rewards.post_strike_stability.weight = 0.0

        # =====================================================================
        # TERMINATIONS: No CG-dependent terminations
        # =====================================================================
        # interaction_termination depends on kick_end_frame (CG) — not added.
        # Episode ends via time_out or physical terminations.

        # --- Ball placement randomization ---
        self.commands.motion.curve_offset_range = {
            "radius": (0.0, 0.4),
            "arc_angle": math.pi / 9,
            "height": 0.11,
        }
