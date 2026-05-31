"""V3.5 Strike Manifold Discriminator.

Learned classifier that recognizes valid strike states from leg kinematics,
foot velocity, and ball-relative geometry. Used to gate contact rewards —
replaces frame-based Contact Graph (CG) with a learned, phase-free gate.

Input features (~39D, all in pelvis-local frame):
  q_leg[12D]              — leg joint positions (named indices)
  dq_leg[12D]             — leg joint velocities
  foot_vel_local[3D]      — swing foot velocity in pelvis frame
  foot_vel_towards_ball[1D] — scalar projection onto foot→ball
  foot_vel_along_kick_dir[1D] — scalar projection onto kick direction
  foot_rel_pelvis_local[3D]   — swing foot pos in pelvis frame
  pelvis_roll_pitch[2D]   — from projected gravity
  pelvis_yaw_to_kick_dir[1D]  — yaw angle to kick direction
  support_foot_contact[1D]    — support foot ground contact
  ball_dist_to_foot[1D]       — distance from swing foot to ball
  foot_behind_ball[1D]        — foot behind ball relative to kick dir

Output: P(strike) ∈ [0, 1] per env.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_inv
except ImportError:
    # Offline mode (no sim): quat functions not needed for model-only usage
    quat_apply = quat_apply_inverse = quat_inv = None

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import MotionCommand


# ── G1 leg joint names (order-independent, resolved at runtime) ──────────
LEG_JOINT_NAMES = [
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

SWING_FOOT_BODY = "right_ankle_roll_link"
SUPPORT_FOOT_BODY = "left_ankle_roll_link"
PELVIS_BODY = "pelvis"

INPUT_DIM = 38


class StrikeDiscriminator(nn.Module):
    """Small MLP that classifies whether current state is a valid strike."""

    def __init__(self, input_dim: int = INPUT_DIM, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, 32),
            nn.ELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns P(strike) ∈ [0, 1] per env, shape [N]."""
        return self.net(features).squeeze(-1)


class StrikeFeatureExtractor:
    """Extracts discriminator features from live simulation state.

    Call `init_indices(robot)` once after env creation to resolve joint/body
    name→index mappings. Then call `compute(env, command)` every step.
    """

    def __init__(self):
        self._leg_joint_indices: torch.Tensor | None = None
        self._swing_body_idx: int = -1
        self._support_body_idx: int = -1
        self._pelvis_body_idx: int = -1
        self._initialized = False

    def init_indices(self, robot) -> None:
        """Resolve joint/body name→index maps from the articulation."""
        # Leg joint indices
        joint_names = list(robot.data.joint_names)
        leg_indices = []
        for name in LEG_JOINT_NAMES:
            found = [i for i, jn in enumerate(joint_names) if jn == name]
            if not found:
                # Try suffix match
                found = [i for i, jn in enumerate(joint_names) if jn.endswith(name)]
            if found:
                leg_indices.append(found[0])
            else:
                raise ValueError(f"Leg joint '{name}' not found in robot joints: {joint_names}")
        self._leg_joint_indices = torch.tensor(leg_indices, dtype=torch.long)

        # Body indices
        body_names = list(robot.data.body_names)
        for bname, attr in [
            (SWING_FOOT_BODY, "_swing_body_idx"),
            (SUPPORT_FOOT_BODY, "_support_body_idx"),
            (PELVIS_BODY, "_pelvis_body_idx"),
        ]:
            found = [i for i, bn in enumerate(body_names) if bn == bname]
            if not found:
                found = [i for i, bn in enumerate(body_names) if bn.endswith(bname)]
            if found:
                setattr(self, attr, found[0])
            else:
                raise ValueError(f"Body '{bname}' not found in robot bodies: {body_names}")

        self._initialized = True

    def compute(
        self,
        env: ManagerBasedRLEnv,
        command: MotionCommand,
    ) -> torch.Tensor:
        """Extract ~39D feature vector for all envs.

        All spatial features are computed in pelvis-local frame.

        Returns:
            features: [N, 39] tensor
        """
        if not self._initialized:
            self.init_indices(command.robot)

        robot = command.robot
        device = robot.device
        N = env.num_envs
        idx_leg = self._leg_joint_indices.to(device)

        # ── 1. Leg joint pos/vel (12D each) ──
        q_leg = robot.data.joint_pos[:, idx_leg]         # [N, 12]
        dq_leg = robot.data.joint_vel[:, idx_leg]        # [N, 12]

        # ── 2. Pelvis frame for coordinate transforms ──
        pelvis_quat = robot.data.body_quat_w[:, self._pelvis_body_idx]   # [N, 4]
        pelvis_pos = robot.data.body_pos_w[:, self._pelvis_body_idx]     # [N, 3]
        pelvis_quat_inv = quat_inv(pelvis_quat)

        # ── 3. Swing foot velocity in pelvis frame (3D) ──
        foot_vel_w = robot.data.body_lin_vel_w[:, self._swing_body_idx]  # [N, 3]
        foot_vel_local = quat_apply_inverse(pelvis_quat, foot_vel_w)     # [N, 3]

        # ── 4. Swing foot position relative to pelvis, in pelvis frame (3D) ──
        foot_pos_w = robot.data.body_pos_w[:, self._swing_body_idx]      # [N, 3]
        foot_rel_w = foot_pos_w - pelvis_pos                             # [N, 3]
        foot_rel_local = quat_apply_inverse(pelvis_quat, foot_rel_w)     # [N, 3]

        # ── 5. Ball position and kick direction ──
        soccer_ball = env.scene["soccer_ball"]
        ball_pos_w = soccer_ball.data.root_pos_w[:, :3]                  # [N, 3]

        # Foot-to-ball vector
        foot_to_ball_w = ball_pos_w - foot_pos_w                         # [N, 3]
        ball_dist = torch.norm(foot_to_ball_w[:, :2], dim=-1, keepdim=True)  # [N, 1] XY dist

        # foot_vel projected onto foot→ball direction (scalar, 1D)
        foot_to_ball_dir = foot_to_ball_w / (torch.norm(foot_to_ball_w, dim=-1, keepdim=True) + 1e-6)
        foot_vel_towards_ball = (foot_vel_w * foot_to_ball_dir).sum(dim=-1, keepdim=True)  # [N, 1]

        # Desired kick direction (ball → target destination)
        target_dest = command.target_destination_pos[:, :3]              # [N, 3]
        kick_dir_w = target_dest - ball_pos_w                           # [N, 3]
        kick_dir_w[:, 2] = 0  # XY only
        kick_dir_w = kick_dir_w / (torch.norm(kick_dir_w, dim=-1, keepdim=True) + 1e-6)

        # foot_vel projected onto kick direction (scalar, 1D)
        foot_vel_along_kick = (foot_vel_w * kick_dir_w).sum(dim=-1, keepdim=True)  # [N, 1]

        # ── 6. Pelvis orientation features ──
        # Roll/pitch from projected gravity (2D)
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=device).expand(N, 3)
        proj_grav = quat_apply_inverse(pelvis_quat, gravity_vec)         # [N, 3]
        pelvis_roll_pitch = proj_grav[:, :2]                             # [N, 2]

        # Pelvis yaw relative to kick direction (1D)
        # pelvis forward = quat_apply(pelvis_quat, [1,0,0])
        forward_local = torch.tensor([1.0, 0.0, 0.0], device=device).expand(N, 3)
        pelvis_forward_w = quat_apply(pelvis_quat, forward_local)        # [N, 3]
        pelvis_forward_w[:, 2] = 0
        pelvis_forward_w = pelvis_forward_w / (torch.norm(pelvis_forward_w, dim=-1, keepdim=True) + 1e-6)
        # cos(yaw_diff) between pelvis forward and kick direction
        yaw_cos = (pelvis_forward_w * kick_dir_w).sum(dim=-1, keepdim=True)  # [N, 1]

        # ── 7. Support foot ground contact (1D) ──
        # Use support foot Z position as proxy (below threshold = grounded)
        support_pos_w = robot.data.body_pos_w[:, self._support_body_idx]  # [N, 3]
        support_grounded = (support_pos_w[:, 2] < 0.05).float().unsqueeze(-1)  # [N, 1]

        # ── 8. Foot behind ball relative to kick direction (1D) ──
        ball_to_foot_w = foot_pos_w - ball_pos_w                         # [N, 3]
        # Negative projection = foot is behind ball
        behind_val = -(ball_to_foot_w * kick_dir_w).sum(dim=-1, keepdim=True)  # [N, 1]
        # Positive = behind, negative = in front

        # ── Concatenate all features ──
        features = torch.cat([
            q_leg,                    # 12
            dq_leg,                   # 12
            foot_vel_local,           # 3
            foot_vel_towards_ball,    # 1
            foot_vel_along_kick,      # 1
            foot_rel_local,           # 3
            pelvis_roll_pitch,        # 2
            yaw_cos,                  # 1
            support_grounded,         # 1
            ball_dist,                # 1
            behind_val,               # 1
        ], dim=-1)                    # Total: 38

        assert features.shape[-1] == INPUT_DIM, \
            f"Expected {INPUT_DIM}D features, got {features.shape[-1]}D"

        return features


def extract_reference_features(
    motion_data: dict,
    kick_frame: int,
    kick_end_frame: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract discriminator-compatible features from a reference motion NPZ.

    This produces approximate features for offline discriminator training.
    Not all features (ball pos, kick dir) are available from reference alone,
    so we use placeholder values for ball-relative features.

    Returns:
        features: [T, 39] tensor
        labels: [T] tensor (soft labels)
    """
    joint_pos = torch.tensor(motion_data["joint_pos"], dtype=torch.float32, device=device)
    T, nj = joint_pos.shape

    # Resolve leg joint indices (same ordering as LEG_JOINT_NAMES)
    # In reference motions, joints are typically ordered as in the URDF
    # We use all 12 leg joints (first 12 in standard G1 ordering)
    # This should be verified against the actual joint order in the NPZ
    q_leg = joint_pos[:, :12]  # [T, 12]

    # Approximate joint velocity via finite differences
    dq_leg = torch.zeros_like(q_leg)
    dq_leg[1:] = (q_leg[1:] - q_leg[:-1]) * 30.0  # 30 FPS

    # For reference-only features, use zeros for ball-relative features
    # These will be filled properly from rollout data
    zeros_1 = torch.zeros(T, 1, device=device)
    zeros_2 = torch.zeros(T, 2, device=device)
    zeros_3 = torch.zeros(T, 3, device=device)

    features = torch.cat([
        q_leg,          # 12
        dq_leg,         # 12
        zeros_3,        # foot_vel_local (3)
        zeros_1,        # foot_vel_towards_ball (1)
        zeros_1,        # foot_vel_along_kick (1)
        zeros_3,        # foot_rel_local (3)
        zeros_2,        # pelvis_roll_pitch (2)
        zeros_1,        # yaw_cos (1)
        zeros_1,        # support_grounded (1)
        zeros_1,        # ball_dist (1)
        zeros_1,        # behind_val (1)
    ], dim=-1)  # [T, 38]

    # Soft labels
    labels = torch.zeros(T, device=device)
    if kick_frame >= 0:
        # Load / pre-swing: 0.5 - 0.7
        load_start = max(0, kick_frame - 10)
        load_end = max(0, kick_frame - 3)
        for t in range(load_start, load_end + 1):
            if t < T:
                progress = (t - load_start) / max(1, load_end - load_start)
                labels[t] = 0.5 + 0.2 * progress

        # Contact-ready: 1.0
        for t in range(max(0, kick_frame - 2), min(T, kick_frame + 3)):
            labels[t] = 1.0

        # Early follow-through: 0.7 - 0.9
        ft_start = kick_frame + 3
        ft_end = min(T, kick_end_frame + 5) if kick_end_frame > 0 else min(T, kick_frame + 8)
        for t in range(ft_start, ft_end):
            if t < T:
                progress = (t - ft_start) / max(1, ft_end - ft_start)
                labels[t] = 0.9 - 0.2 * progress

    return features, labels
