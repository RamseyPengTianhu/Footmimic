"""Standalone task-feature computation for LATENT-v2 task_features mode.

Computes 26D task features from live simulation state.
These features are shared across ALL kick motions and are available at
deployment (from camera + FK), replacing the 58D motion reference.

Feature vector (per env, 26D):

  ── V10 spatial geometry (22D, identical to V10ObsBuilder._compute_ball_foot_relation) ──
  ball_rel_swing              3D   ball pos in swing-foot local frame
  ball_rel_support            3D   ball pos in support-foot local frame
  ball_rel_pelvis             3D   ball pos in pelvis local frame
  ball_vel_local              3D   ball velocity in pelvis frame
  kick_dir_local              2D   kick direction unit vec (pelvis frame)
  swing_foot_ball_dist        1D   horizontal dist swing foot → ball
  swing_ball_longitudinal     1D   swing→ball along kick direction
  swing_ball_lateral          1D   swing→ball perpendicular to kick dir
  support_ball_lateral        1D   support→ball perpendicular to kick dir
  support_ball_longitudinal   1D   support→ball along kick direction
  swing_vel_along_kick        1D   swing foot velocity along kick dir
  swing_vel_to_ball_align     1D   swing foot velocity toward ball
  ball_vel_mag                1D   ball speed magnitude

  ── Phase embedding (4D) ──
  progress                    1D   motion progress t/T, 0→1
  phase_sin                   1D   sin(2π · progress)
  phase_cos                   1D   cos(2π · progress)
  kick_flag                   1D   1.0 if phase ≥ STRIKE (CG=1), else 0.0
  ─────────────────────────────
  Total                       26D
"""

from __future__ import annotations

import math
import torch
from isaaclab.utils.math import quat_apply, quat_inv

# ── Feature ABI tracking ───────────────────────────────────────────────────
FEATURE_VERSION = "v10_phase_26d"
TASK_FEATURES_DIM = 26
FEATURE_NAMES = [
    "ball_rel_swing(3)", "ball_rel_support(3)", "ball_rel_pelvis(3)",
    "ball_vel_local(3)", "kick_dir_local(2)",
    "swing_foot_ball_dist(1)", "swing_ball_longitudinal(1)", "swing_ball_lateral(1)",
    "support_ball_lateral(1)", "support_ball_longitudinal(1)",
    "swing_vel_along_kick(1)", "swing_vel_to_ball_align(1)", "ball_vel_mag(1)",
    "progress(1)", "phase_sin(1)", "phase_cos(1)", "kick_flag(1)",
]

# Phase IDs (matching event_phase.py constants)
_PHASE_STRIKE = 2

# Default body names (G1 robot)
_SWING_FOOT = "right_ankle_roll_link"
_SUPPORT_FOOT = "left_ankle_roll_link"
_PELVIS = "pelvis"


def compute_ball_foot_relation(
    env,
    swing_foot: str = _SWING_FOOT,
    support_foot: str = _SUPPORT_FOOT,
    pelvis: str = _PELVIS,
    command_name: str = "motion",
) -> torch.Tensor:
    """Compute 26-D task features (V10 spatial geometry + phase embedding).

    Spatial geometry (22D) is identical to V10ObsBuilder._compute_ball_foot_relation.
    Phase embedding (4D) provides temporal context from the motion command.

    Parameters
    ----------
    env : ManagerBasedEnv (unwrapped Isaac Lab env)
    swing_foot, support_foot, pelvis : str
        Body-link names on the robot.
    command_name : str
        Name of the motion-command term.

    Returns
    -------
    torch.Tensor   shape (num_envs, 26)
    """
    # Access scene objects
    robot = env.scene["robot"]
    soccer_ball = env.scene["soccer_ball"]
    command = env.command_manager.get_term(command_name)

    # Resolve body indices
    swing_idx = robot.body_names.index(swing_foot)
    support_idx = robot.body_names.index(support_foot)
    pelvis_idx = robot.body_names.index(pelvis)

    # ── Positions & orientations ────────────────────────────────────────
    ball_pos_w = soccer_ball.data.root_pos_w[:, :3]
    ball_vel_w = soccer_ball.data.root_lin_vel_w[:, :3]

    pelvis_pos_w = robot.data.body_pos_w[:, pelvis_idx]
    pelvis_quat_w = robot.data.body_quat_w[:, pelvis_idx]
    pelvis_quat_inv = quat_inv(pelvis_quat_w)

    swing_pos_w = robot.data.body_pos_w[:, swing_idx]
    swing_quat_w = robot.data.body_quat_w[:, swing_idx]
    support_pos_w = robot.data.body_pos_w[:, support_idx]
    support_quat_w = robot.data.body_quat_w[:, support_idx]

    # ── Ball relative positions (in each body's local frame) ────────────
    ball_rel_swing = quat_apply(quat_inv(swing_quat_w), ball_pos_w - swing_pos_w)
    ball_rel_support = quat_apply(quat_inv(support_quat_w), ball_pos_w - support_pos_w)
    ball_rel_pelvis = quat_apply(pelvis_quat_inv, ball_pos_w - pelvis_pos_w)
    ball_vel_local = quat_apply(pelvis_quat_inv, ball_vel_w)

    # ── Kick direction (2D in pelvis frame) ─────────────────────────────
    dest_pos = command.target_destination_pos
    env_origins = getattr(env.scene, "env_origins", None)
    dest_w = dest_pos[:, :2] + env_origins[:, :2] if env_origins is not None else dest_pos[:, :2]
    kick_dir_w = dest_w - ball_pos_w[:, :2]
    kick_dir_dist = torch.norm(kick_dir_w, dim=-1, keepdim=True).clamp(min=1e-4)
    kick_dir_2d = kick_dir_w / kick_dir_dist
    kick_dir_3d = torch.cat([kick_dir_2d, torch.zeros_like(kick_dir_2d[:, :1])], dim=-1)
    kick_dir_local = quat_apply(pelvis_quat_inv, kick_dir_3d)[:, :2]

    # ── Spatial decomposition along kick direction ──────────────────────
    swing_to_ball_w = ball_pos_w - swing_pos_w
    swing_foot_ball_dist = torch.norm(swing_to_ball_w[:, :2], dim=-1, keepdim=True)

    kick_perp_2d = torch.stack([-kick_dir_2d[:, 1], kick_dir_2d[:, 0]], dim=-1)

    swing_to_ball_2d = swing_to_ball_w[:, :2]
    swing_ball_longitudinal = (swing_to_ball_2d * kick_dir_2d).sum(dim=-1, keepdim=True)
    swing_ball_lateral = (swing_to_ball_2d * kick_perp_2d).sum(dim=-1, keepdim=True)

    support_to_ball_2d = (ball_pos_w - support_pos_w)[:, :2]
    support_ball_longitudinal = (support_to_ball_2d * kick_dir_2d).sum(dim=-1, keepdim=True)
    support_ball_lateral = (support_to_ball_2d * kick_perp_2d).sum(dim=-1, keepdim=True)

    # ── Velocity features ───────────────────────────────────────────────
    swing_lin_vel_w = robot.data.body_lin_vel_w[:, swing_idx]
    swing_vel_along_kick = (swing_lin_vel_w[:, :2] * kick_dir_2d).sum(dim=-1, keepdim=True)

    swing_to_ball_dir = swing_to_ball_2d / swing_to_ball_2d.norm(dim=-1, keepdim=True).clamp(min=1e-4)
    swing_vel_to_ball_align = (swing_lin_vel_w[:, :2] * swing_to_ball_dir).sum(dim=-1, keepdim=True)

    ball_vel_mag = torch.norm(ball_vel_w[:, :2], dim=-1, keepdim=True)

    # ── Phase embedding (4D) ───────────────────────────────────────────
    progress = (command.time_steps.float() / command.motion_length.float().clamp(min=1)).unsqueeze(-1)
    progress = progress.clamp(0.0, 1.0)
    phase_sin = torch.sin(2.0 * math.pi * progress)
    phase_cos = torch.cos(2.0 * math.pi * progress)
    kick_flag = (command.event_phase_id >= _PHASE_STRIKE).float().unsqueeze(-1)

    # ── Concatenate 26D ─────────────────────────────────────────────────
    return torch.cat([
        # V10 spatial geometry (22D)
        ball_rel_swing, ball_rel_support, ball_rel_pelvis,         # 9D
        ball_vel_local, kick_dir_local,                             # 5D
        swing_foot_ball_dist, swing_ball_longitudinal, swing_ball_lateral,  # 3D
        support_ball_lateral, support_ball_longitudinal,            # 2D
        swing_vel_along_kick, swing_vel_to_ball_align,              # 2D
        ball_vel_mag,                                               # 1D
        # Phase embedding (4D)
        progress, phase_sin, phase_cos, kick_flag,                  # 4D
    ], dim=-1)  # [N, 26]
