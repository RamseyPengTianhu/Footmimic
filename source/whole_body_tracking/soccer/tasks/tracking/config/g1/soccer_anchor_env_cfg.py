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
