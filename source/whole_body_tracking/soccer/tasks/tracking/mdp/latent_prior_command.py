"""Latent-conditioned motion-prior command.

This command is a first bridge toward a latent-prior stage1.  It still samples
windows from motion files, but the policy command is no longer the raw per-frame
joint reference.  Instead, the command is the CVAE condition plus the encoded
latent for the current short window.  Rewards can then track the decoded compact
motion feature rather than hard-tracking a full reference trajectory.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from isaaclab.utils.math import quat_apply, quat_inv

from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import MotionCommand, MotionCommandCfg
from soccer.tasks.tracking.mdp.content_cvae_prior import ContentConditionedVAE


DEFAULT_CONTENT_CVAE_PRIOR_PATH = "models/content_cvae_prior_video1_clean_v3_final_local.pt"
NUM_PHASES = 4


def _load_checkpoint(path: str, device: torch.device) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Content CVAE checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _project_local_xy(
    vec_xy: torch.Tensor,
    forward: torch.Tensor,
    side: torch.Tensor,
    kick_leg: torch.Tensor,
) -> torch.Tensor:
    side_sign = torch.where(
        kick_leg == 0,
        torch.full_like(kick_leg, -1.0, dtype=torch.float32),
        torch.ones_like(kick_leg, dtype=torch.float32),
    ).unsqueeze(-1)
    long = torch.sum(vec_xy * forward, dim=-1, keepdim=True)
    lat = torch.sum(vec_xy * side, dim=-1, keepdim=True) * side_sign
    return torch.cat([long, lat], dim=-1)


class LatentPriorMotionCommand(MotionCommand):
    """Motion command whose observation is ``condition + latent``."""

    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env):
        super().__init__(cfg, env)

        self.prior_model_path = str(getattr(cfg, "prior_model_path", DEFAULT_CONTENT_CVAE_PRIOR_PATH))
        ckpt = _load_checkpoint(self.prior_model_path, torch.device(self.device))

        self.prior_input_dim = int(ckpt["input_dim"])
        self.prior_cond_dim = int(ckpt["cond_dim"])
        self.prior_window_len = int(ckpt["window_len"])
        self.prior_latent_dim = int(ckpt["latent_dim"])
        self.prior_hidden_dim = int(ckpt["hidden_dim"])
        self.prior_feature_frame = str(ckpt.get("feature_frame", "world"))
        self.prior_time_scale = float(ckpt.get("time_scale", 40.0))

        if self.prior_input_dim % self.prior_window_len != 0:
            raise ValueError(
                f"CVAE input_dim={self.prior_input_dim} is not divisible by window_len={self.prior_window_len}"
            )
        self.prior_frame_dim = self.prior_input_dim // self.prior_window_len
        self._prior_center_index = self.prior_window_len // 2
        self.prior_joint_count = (self.prior_frame_dim - 7) // 2
        if self.prior_joint_count <= 0 or self.prior_joint_count * 2 + 7 != self.prior_frame_dim:
            raise ValueError(f"Invalid CVAE frame_dim={self.prior_frame_dim}")

        self.prior_feature_mean = torch.as_tensor(
            ckpt["feature_mean"], dtype=torch.float32, device=self.device
        )
        self.prior_feature_std = torch.as_tensor(
            ckpt["feature_std"], dtype=torch.float32, device=self.device
        ).clamp_min(1e-4)
        self.prior_frame_mean = self.prior_feature_mean.reshape(self.prior_window_len, self.prior_frame_dim)[
            self._prior_center_index
        ]
        self.prior_frame_std = self.prior_feature_std.reshape(self.prior_window_len, self.prior_frame_dim)[
            self._prior_center_index
        ]
        self.prior_cond_mean = torch.as_tensor(ckpt["cond_mean"], dtype=torch.float32, device=self.device)
        self.prior_cond_std = torch.as_tensor(ckpt["cond_std"], dtype=torch.float32, device=self.device).clamp_min(1e-4)

        self.prior_model = ContentConditionedVAE(
            input_dim=self.prior_input_dim,
            cond_dim=self.prior_cond_dim,
            latent_dim=self.prior_latent_dim,
            hidden_dim=self.prior_hidden_dim,
        ).to(self.device)
        self.prior_model.load_state_dict(ckpt["model_state_dict"])
        self.prior_model.eval()

        self.prior_cond = torch.zeros(self.num_envs, self.prior_cond_dim, device=self.device)
        self.prior_cond_norm = torch.zeros_like(self.prior_cond)
        self.prior_latent = torch.zeros(self.num_envs, self.prior_latent_dim, device=self.device)
        self.prior_target_window = torch.zeros(
            self.num_envs, self.prior_window_len, self.prior_frame_dim, device=self.device
        )
        self.prior_target_frame = torch.zeros(self.num_envs, self.prior_frame_dim, device=self.device)
        self.prior_target_frame_norm = torch.zeros_like(self.prior_target_frame)

        self._update_prior_targets()

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.prior_cond_norm, self.prior_latent, self.prior_target_frame_norm], dim=-1)

    def _reference_forward(self) -> torch.Tensor:
        body_pos = self.motion.body_pos_w
        motion_idx = self.motion_idx.to(device=body_pos.device, dtype=torch.long)
        last_idx = (self.motion_length.to(device=body_pos.device, dtype=torch.long) - 1).clamp(min=0)
        first_idx = torch.zeros_like(last_idx)
        first_anchor = body_pos[motion_idx, first_idx, self.motion_anchor_body_index]
        last_anchor = body_pos[motion_idx, last_idx, self.motion_anchor_body_index]
        forward = last_anchor[:, :2] - first_anchor[:, :2]
        fallback = torch.zeros_like(forward)
        fallback[:, 0] = 1.0
        norm = torch.norm(forward, dim=-1, keepdim=True)
        return torch.where(norm > 1e-6, forward / norm.clamp_min(1e-6), fallback)

    def _foot_indices(self) -> tuple[int, int]:
        names = list(self.cfg.body_names)
        return names.index("left_ankle_roll_link"), names.index("right_ankle_roll_link")

    def _reference_window_features(self, frame_ids: torch.Tensor) -> torch.Tensor:
        motion_idx = self.motion_idx[:, None].expand(-1, self.prior_window_len)
        joint_pos = self.motion.joint_pos[motion_idx, frame_ids, : self.prior_joint_count]
        joint_vel = self.motion.joint_vel[motion_idx, frame_ids, : self.prior_joint_count]

        pelvis_idx = list(self.cfg.body_names).index("pelvis")
        body_pos = self.motion.body_pos_w[motion_idx, frame_ids]
        body_quat = self.motion.body_quat_w[motion_idx, frame_ids]
        body_lin_vel = self.motion.body_lin_vel_w[motion_idx, frame_ids]
        body_ang_vel = self.motion.body_ang_vel_w[motion_idx, frame_ids]

        pelvis_height = body_pos[:, :, pelvis_idx, 2:3]
        pelvis_lin_vel = body_lin_vel[:, :, pelvis_idx]
        pelvis_ang_vel = body_ang_vel[:, :, pelvis_idx]
        if self.prior_feature_frame == "local":
            pelvis_quat = body_quat[:, :, pelvis_idx]
            flat_inv = quat_inv(pelvis_quat.reshape(-1, 4))
            pelvis_lin_vel = quat_apply(flat_inv, pelvis_lin_vel.reshape(-1, 3)).reshape(
                self.num_envs, self.prior_window_len, 3
            )
            pelvis_ang_vel = quat_apply(flat_inv, pelvis_ang_vel.reshape(-1, 3)).reshape(
                self.num_envs, self.prior_window_len, 3
            )
        elif self.prior_feature_frame != "world":
            raise ValueError(f"Unsupported feature_frame={self.prior_feature_frame!r}")

        return torch.cat([joint_pos, joint_vel, pelvis_height, pelvis_lin_vel, pelvis_ang_vel], dim=-1)

    def _condition(self, center_frames: torch.Tensor) -> torch.Tensor:
        num_envs = self.num_envs
        phase_id = self.event_phase_id.to(dtype=torch.long).clamp(0, NUM_PHASES - 1)
        phase = F.one_hot(phase_id, num_classes=NUM_PHASES).to(dtype=torch.float32)

        kick_leg = self.kick_leg.to(dtype=torch.long)
        leg = torch.zeros(num_envs, 3, dtype=torch.float32, device=self.device)
        leg[:, 0] = (kick_leg == 0).to(dtype=torch.float32)
        leg[:, 1] = (kick_leg == 1).to(dtype=torch.float32)
        leg[:, 2] = ((kick_leg != 0) & (kick_leg != 1)).to(dtype=torch.float32)

        kick_frame = self.kick_frame.to(dtype=torch.float32)
        dt = center_frames.to(dtype=torch.float32) - kick_frame
        time_to_kick = torch.where(
            kick_frame >= 0,
            torch.clamp(dt / max(self.prior_time_scale, 1.0), -2.0, 2.0),
            torch.zeros_like(dt),
        ).unsqueeze(-1)

        body_pos = self.motion.body_pos_w[self.motion_idx, center_frames]
        body_vel = self.motion.body_lin_vel_w[self.motion_idx, center_frames]
        pelvis_idx = list(self.cfg.body_names).index("pelvis")
        left_idx, right_idx = self._foot_indices()
        left_is_kick = (kick_leg == 0).unsqueeze(-1)

        pelvis = body_pos[:, pelvis_idx]
        left_pos, right_pos = body_pos[:, left_idx], body_pos[:, right_idx]
        left_vel, right_vel = body_vel[:, left_idx], body_vel[:, right_idx]
        kick_pos = torch.where(left_is_kick, left_pos, right_pos)
        support_pos = torch.where(left_is_kick, right_pos, left_pos)
        kick_vel = torch.where(left_is_kick, left_vel, right_vel)
        support_vel = torch.where(left_is_kick, right_vel, left_vel)

        ball = self.soccer_ball_pos
        forward = self._reference_forward()
        side = torch.stack([-forward[:, 1], forward[:, 0]], dim=-1)

        cond = torch.cat(
            [
                phase,
                leg,
                time_to_kick,
                _project_local_xy(ball[:, :2] - pelvis[:, :2], forward, side, kick_leg),
                _project_local_xy(kick_pos[:, :2] - ball[:, :2], forward, side, kick_leg),
                _project_local_xy(support_pos[:, :2] - ball[:, :2], forward, side, kick_leg),
                kick_pos[:, 2:3] - ball[:, 2:3],
                support_pos[:, 2:3] - ball[:, 2:3],
                _project_local_xy(kick_vel[:, :2], forward, side, kick_leg),
                _project_local_xy(support_vel[:, :2], forward, side, kick_leg),
                torch.norm(kick_vel[:, :2], dim=-1, keepdim=True),
                torch.norm(support_vel[:, :2], dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        if cond.shape[-1] != self.prior_cond_dim:
            raise ValueError(f"CVAE cond_dim={self.prior_cond_dim}, live cond_dim={cond.shape[-1]}")
        return cond

    @torch.no_grad()
    def _update_prior_targets(self):
        if not hasattr(self, "prior_model"):
            return
        offsets = torch.arange(self.prior_window_len, device=self.device) - self._prior_center_index
        frame_ids = self.time_steps[:, None] + offsets[None, :]
        frame_ids = frame_ids.clamp(min=0)
        frame_ids = torch.minimum(frame_ids, (self.motion_length - 1).clamp(min=0)[:, None])
        center_frames = frame_ids[:, self._prior_center_index]

        feature_window = self._reference_window_features(frame_ids)
        x = feature_window.reshape(self.num_envs, -1)
        cond = self._condition(center_frames)
        x_norm = (x - self.prior_feature_mean) / self.prior_feature_std
        cond_norm = (cond - self.prior_cond_mean) / self.prior_cond_std

        mu, _ = self.prior_model.encode(x_norm, cond_norm)
        recon = self.prior_model.decode(mu, cond_norm)
        target_window = (recon * self.prior_feature_std + self.prior_feature_mean).reshape(
            self.num_envs, self.prior_window_len, self.prior_frame_dim
        )

        self.prior_cond = cond
        self.prior_cond_norm = cond_norm
        self.prior_latent = mu
        self.prior_target_window = target_window
        self.prior_target_frame = target_window[:, self._prior_center_index]
        self.prior_target_frame_norm = (self.prior_target_frame - self.prior_frame_mean) / self.prior_frame_std

        self.metrics["latent_prior_z_norm"] = torch.norm(mu, dim=-1)
        recon_err = torch.mean((recon - x_norm) ** 2, dim=-1)
        self.metrics["latent_prior_recon_error"] = recon_err

    def _update_command(self):
        super()._update_command()
        self._update_prior_targets()
