"""Content-conditioned CVAE prior for live kick-policy rewards.

This module mirrors the offline ``scripts/rsl_rl/train_content_cvae_prior.py``
feature layout, then wraps a trained checkpoint as a dense reconstruction-score
reward.  The checkpoint is trained offline; the live reward only runs inference
and keeps a short per-environment feature window.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from isaaclab.utils.math import quat_apply, quat_inv
except ModuleNotFoundError:  # Allows offline syntax checks outside IsaacLab.
    quat_apply = quat_inv = None

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import MotionCommand


PHASE_APPROACH = 0
PHASE_PRESTRIKE = 1
PHASE_STRIKE = 2
PHASE_FOLLOW = 3
NUM_PHASES = 4

COND_NAMES = [
    "phase_approach",
    "phase_prestrike",
    "phase_strike",
    "phase_follow",
    "leg_left",
    "leg_right",
    "leg_unknown",
    "time_to_kick",
    "ball_from_pelvis_long",
    "ball_from_pelvis_lat",
    "kick_from_ball_long",
    "kick_from_ball_lat",
    "support_from_ball_long",
    "support_from_ball_lat",
    "kick_height_rel_ball",
    "support_height_rel_ball",
    "kick_vel_long",
    "kick_vel_lat",
    "support_vel_long",
    "support_vel_lat",
    "kick_speed_xy",
    "support_speed_xy",
]


class ContentConditionedVAE(nn.Module):
    """Small conditional VAE for short motion feature windows."""

    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim + cond_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([x, cond], dim=-1))
        return self.mu(h), self.logvar(h).clamp(-8.0, 8.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, cond], dim=-1))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond)
        return recon, mu, logvar


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, reconstruction, and KL losses."""
    recon_loss = F.mse_loss(recon, x)
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl


@torch.no_grad()
def reconstruction_error(model: ContentConditionedVAE, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    """Per-sample normalized reconstruction error for diagnostics."""
    mu, _ = model.encode(x, cond)
    recon = model.decode(mu, cond)
    return torch.mean((recon - x) ** 2, dim=-1)


def _load_checkpoint(path: str, device: torch.device) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Content CVAE checkpoint not found: {path}. "
            "Train it first with scripts/rsl_rl/train_content_cvae_prior.py."
        )
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _names_from_robot(robot, kind: str) -> list[str]:
    data = getattr(robot, "data", None)
    names = getattr(data, f"{kind}_names", None)
    if names is None:
        names = getattr(robot, f"{kind}_names", None)
    if names is None:
        raise AttributeError(f"Could not resolve robot {kind}_names")
    return list(names)


def _find_name(names: list[str], target: str) -> int:
    for idx, name in enumerate(names):
        if name == target or name.endswith(target):
            return idx
    raise ValueError(f"Name '{target}' not found in {names}")


def _scene_env_origins(env: "ManagerBasedRLEnv") -> torch.Tensor | None:
    return getattr(env.scene, "env_origins", None)


def _ball_pos_w(env: "ManagerBasedRLEnv", command: "MotionCommand") -> torch.Tensor:
    ball = getattr(command, "soccer_ball", None)
    if ball is not None:
        try:
            return ball.data.root_pos_w[:, :3]
        except Exception:
            pass
    try:
        ball = env.scene["soccer_ball"]
        return ball.data.root_pos_w[:, :3]
    except Exception:
        origins = _scene_env_origins(env)
        if origins is not None:
            return command.soccer_ball_pos[:, :3] + origins[:, :3]
        return command.soccer_ball_pos[:, :3]


def _destination_pos_w(env: "ManagerBasedRLEnv", command: "MotionCommand") -> torch.Tensor:
    dest = command.target_destination_pos[:, :3]
    origins = _scene_env_origins(env)
    if origins is not None:
        return dest + origins[:, :3]
    return dest


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


class ContentCVAEPrior:
    """Runtime inference wrapper for a trained content-conditioned CVAE."""

    def __init__(self, model_path: str, device: str | torch.device):
        self.model_path = model_path
        self.device = torch.device(device)
        ckpt = _load_checkpoint(model_path, self.device)

        self.input_dim = int(ckpt["input_dim"])
        self.cond_dim = int(ckpt["cond_dim"])
        self.window_len = int(ckpt["window_len"])
        self.latent_dim = int(ckpt["latent_dim"])
        self.hidden_dim = int(ckpt["hidden_dim"])
        self.lower_body_only = bool(ckpt.get("lower_body_only", True))
        self.time_scale = float(ckpt.get("time_scale", 40.0))
        self.feature_frame = str(ckpt.get("feature_frame", "world"))

        if self.input_dim % self.window_len != 0:
            raise ValueError(
                f"CVAE input_dim={self.input_dim} is not divisible by window_len={self.window_len}"
            )
        self.frame_dim = self.input_dim // self.window_len
        if (self.frame_dim - 7) % 2 != 0:
            raise ValueError(
                f"Cannot infer joint count from per-frame feature dim {self.frame_dim}; "
                "expected 2 * joint_count + 7"
            )
        self.joint_count = (self.frame_dim - 7) // 2

        self.feature_mean = torch.as_tensor(ckpt["feature_mean"], dtype=torch.float32, device=self.device)
        self.feature_std = torch.as_tensor(ckpt["feature_std"], dtype=torch.float32, device=self.device).clamp_min(1e-4)
        self.cond_mean = torch.as_tensor(ckpt["cond_mean"], dtype=torch.float32, device=self.device)
        self.cond_std = torch.as_tensor(ckpt["cond_std"], dtype=torch.float32, device=self.device).clamp_min(1e-4)

        self.model = ContentConditionedVAE(
            input_dim=self.input_dim,
            cond_dim=self.cond_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self._history: torch.Tensor | None = None
        self._valid_count: torch.Tensor | None = None
        self._prev_time_steps: torch.Tensor | None = None
        self._prev_motion_idx: torch.Tensor | None = None
        self._body_indices: dict[str, int] = {}

    def _body_index(self, command: "MotionCommand", body_name: str) -> int:
        cached = self._body_indices.get(body_name)
        if cached is not None:
            return cached
        names = _names_from_robot(command.robot, "body")
        idx = _find_name(names, body_name)
        self._body_indices[body_name] = idx
        return idx

    def _extract_frame(self, command: "MotionCommand") -> torch.Tensor:
        robot = command.robot
        joint_pos = robot.data.joint_pos[:, : self.joint_count]
        joint_vel = robot.data.joint_vel[:, : self.joint_count]
        if joint_pos.shape[1] != self.joint_count or joint_vel.shape[1] != self.joint_count:
            raise ValueError(
                f"Robot exposes {joint_pos.shape[1]} joints, but CVAE checkpoint expects {self.joint_count}"
            )

        pelvis_idx = self._body_index(command, "pelvis")
        pelvis_height = robot.data.body_pos_w[:, pelvis_idx, 2:3]
        pelvis_lin_vel = robot.data.body_lin_vel_w[:, pelvis_idx, :]
        pelvis_ang_vel = robot.data.body_ang_vel_w[:, pelvis_idx, :]
        if self.feature_frame == "local":
            if quat_apply is None or quat_inv is None:
                raise RuntimeError("Local-frame CVAE checkpoint requires isaaclab.utils.math quaternion helpers.")
            pelvis_quat = robot.data.body_quat_w[:, pelvis_idx, :]
            pelvis_quat_inv = quat_inv(pelvis_quat)
            pelvis_lin_vel = quat_apply(pelvis_quat_inv, pelvis_lin_vel)
            pelvis_ang_vel = quat_apply(pelvis_quat_inv, pelvis_ang_vel)
        elif self.feature_frame != "world":
            raise ValueError(f"Unsupported feature_frame={self.feature_frame!r}; expected 'local' or 'world'.")
        return torch.cat([joint_pos, joint_vel, pelvis_height, pelvis_lin_vel, pelvis_ang_vel], dim=-1)

    def _foot_state(
        self,
        command: "MotionCommand",
        kick_foot_name: str,
        support_foot_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        robot = command.robot
        left_idx = self._body_index(command, "left_ankle_roll_link")
        right_idx = self._body_index(command, "right_ankle_roll_link")
        default_kick_idx = self._body_index(command, kick_foot_name)
        default_support_idx = self._body_index(command, support_foot_name)

        kick_leg = getattr(command, "kick_leg", None)
        if kick_leg is None:
            kick_leg = torch.full((robot.data.body_pos_w.shape[0],), 1, dtype=torch.int8, device=self.device)
        kick_leg = kick_leg.to(device=self.device)

        left_pos = robot.data.body_pos_w[:, left_idx]
        right_pos = robot.data.body_pos_w[:, right_idx]
        left_vel = robot.data.body_lin_vel_w[:, left_idx]
        right_vel = robot.data.body_lin_vel_w[:, right_idx]
        default_kick_pos = robot.data.body_pos_w[:, default_kick_idx]
        default_kick_vel = robot.data.body_lin_vel_w[:, default_kick_idx]
        default_support_pos = robot.data.body_pos_w[:, default_support_idx]
        default_support_vel = robot.data.body_lin_vel_w[:, default_support_idx]

        left_is_kick = (kick_leg == 0).unsqueeze(-1)
        known = ((kick_leg == 0) | (kick_leg == 1)).unsqueeze(-1)
        dynamic_kick_pos = torch.where(left_is_kick, left_pos, right_pos)
        dynamic_kick_vel = torch.where(left_is_kick, left_vel, right_vel)
        dynamic_support_pos = torch.where(left_is_kick, right_pos, left_pos)
        dynamic_support_vel = torch.where(left_is_kick, right_vel, left_vel)

        kick_pos = torch.where(known, dynamic_kick_pos, default_kick_pos)
        kick_vel = torch.where(known, dynamic_kick_vel, default_kick_vel)
        support_pos = torch.where(known, dynamic_support_pos, default_support_pos)
        support_vel = torch.where(known, dynamic_support_vel, default_support_vel)
        return kick_pos, kick_vel, support_pos, support_vel

    def _reference_forward(
        self,
        command: "MotionCommand",
        fallback_forward: torch.Tensor,
    ) -> torch.Tensor:
        """Return the same forward basis used by the offline CVAE dataset."""
        try:
            motion = command.motion
            body_pos = motion.body_pos_w
            motion_idx = command.motion_idx.to(device=body_pos.device, dtype=torch.long)
            last_idx = (command.motion_length.to(device=body_pos.device, dtype=torch.long) - 1).clamp(min=0)
            first_idx = torch.zeros_like(last_idx)
            anchor_idx = int(command.motion_anchor_body_index)
            first_anchor = body_pos[motion_idx, first_idx, anchor_idx]
            last_anchor = body_pos[motion_idx, last_idx, anchor_idx]
            forward = (last_anchor[:, :2] - first_anchor[:, :2]).to(
                device=fallback_forward.device,
                dtype=fallback_forward.dtype,
            )
        except Exception:
            forward = fallback_forward

        norm = torch.norm(forward, dim=-1, keepdim=True)
        return torch.where(norm > 1e-6, forward / norm.clamp_min(1e-6), fallback_forward)

    def _condition(
        self,
        env: "ManagerBasedRLEnv",
        command: "MotionCommand",
        kick_foot_name: str,
        support_foot_name: str,
        basis_mode: str = "reference",
    ) -> torch.Tensor:
        robot = command.robot
        num_envs = robot.data.joint_pos.shape[0]
        device = robot.data.joint_pos.device

        phase_id = getattr(command, "event_phase_id", None)
        if phase_id is None:
            phase_id = torch.zeros(num_envs, dtype=torch.long, device=device)
        phase_id = phase_id.to(device=device, dtype=torch.long).clamp(0, NUM_PHASES - 1)
        phase = F.one_hot(phase_id, num_classes=NUM_PHASES).to(dtype=torch.float32)

        kick_leg = getattr(command, "kick_leg", None)
        if kick_leg is None:
            kick_leg = torch.full((num_envs,), 1, dtype=torch.int8, device=device)
        kick_leg = kick_leg.to(device=device)
        leg = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        leg[:, 0] = (kick_leg == 0).to(dtype=torch.float32)
        leg[:, 1] = (kick_leg == 1).to(dtype=torch.float32)
        leg[:, 2] = ((kick_leg != 0) & (kick_leg != 1)).to(dtype=torch.float32)

        kick_frame = getattr(command, "kick_frame", None)
        if kick_frame is None:
            time_to_kick = torch.zeros(num_envs, 1, dtype=torch.float32, device=device)
        else:
            dt = command.time_steps.to(dtype=torch.float32, device=device) - kick_frame.to(dtype=torch.float32, device=device)
            scaled = torch.clamp(dt / max(self.time_scale, 1.0), -2.0, 2.0)
            time_to_kick = torch.where(kick_frame.to(device=device) >= 0, scaled, torch.zeros_like(scaled)).unsqueeze(-1)

        pelvis_idx = self._body_index(command, "pelvis")
        pelvis_pos = robot.data.body_pos_w[:, pelvis_idx]
        ball_pos = _ball_pos_w(env, command).to(device=device)
        dest_pos = _destination_pos_w(env, command).to(device=device)
        kick_pos, kick_vel, support_pos, support_vel = self._foot_state(command, kick_foot_name, support_foot_name)

        target_forward = dest_pos[:, :2] - ball_pos[:, :2]
        forward_norm = torch.norm(target_forward, dim=-1, keepdim=True)
        fallback_forward = torch.zeros_like(target_forward)
        fallback_forward[:, 0] = 1.0
        target_forward = torch.where(
            forward_norm > 1e-6,
            target_forward / forward_norm.clamp_min(1e-6),
            fallback_forward,
        )
        if basis_mode == "reference":
            forward = self._reference_forward(command, target_forward)
        elif basis_mode == "target":
            forward = target_forward
        else:
            raise ValueError(f"Unsupported CVAE basis_mode={basis_mode!r}; expected 'reference' or 'target'.")
        side = torch.stack([-forward[:, 1], forward[:, 0]], dim=-1)

        ball_from_pelvis = _project_local_xy(ball_pos[:, :2] - pelvis_pos[:, :2], forward, side, kick_leg)
        kick_from_ball = _project_local_xy(kick_pos[:, :2] - ball_pos[:, :2], forward, side, kick_leg)
        support_from_ball = _project_local_xy(support_pos[:, :2] - ball_pos[:, :2], forward, side, kick_leg)
        kick_vel_local = _project_local_xy(kick_vel[:, :2], forward, side, kick_leg)
        support_vel_local = _project_local_xy(support_vel[:, :2], forward, side, kick_leg)

        cond = torch.cat(
            [
                phase,
                leg,
                time_to_kick,
                ball_from_pelvis,
                kick_from_ball,
                support_from_ball,
                kick_pos[:, 2:3] - ball_pos[:, 2:3],
                support_pos[:, 2:3] - ball_pos[:, 2:3],
                kick_vel_local,
                support_vel_local,
                torch.norm(kick_vel[:, :2], dim=-1, keepdim=True),
                torch.norm(support_vel[:, :2], dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"CVAE checkpoint expects cond_dim={self.cond_dim}, live condition has {cond.shape[-1]}")
        return cond

    def _ensure_history(self, frame: torch.Tensor, command: "MotionCommand", env: "ManagerBasedRLEnv") -> torch.Tensor:
        num_envs = frame.shape[0]
        device = frame.device
        needs_alloc = (
            self._history is None
            or self._history.shape[0] != num_envs
            or self._history.shape[1] != self.window_len
            or self._history.shape[2] != self.frame_dim
            or self._history.device != device
        )
        if needs_alloc:
            self._history = torch.zeros(num_envs, self.window_len, self.frame_dim, dtype=torch.float32, device=device)
            self._valid_count = torch.zeros(num_envs, dtype=torch.long, device=device)
            self._prev_time_steps = torch.full((num_envs,), -1, dtype=torch.long, device=device)
            self._prev_motion_idx = torch.full((num_envs,), -1, dtype=torch.long, device=device)

        assert self._history is not None
        assert self._valid_count is not None
        assert self._prev_time_steps is not None
        assert self._prev_motion_idx is not None

        time_steps = command.time_steps.to(device=device, dtype=torch.long)
        motion_idx = command.motion_idx.to(device=device, dtype=torch.long)
        reset = self._prev_time_steps < 0
        reset = reset | (time_steps < self._prev_time_steps) | (motion_idx != self._prev_motion_idx)
        episode_length = getattr(env, "episode_length_buf", None)
        if isinstance(episode_length, torch.Tensor) and episode_length.shape[0] == num_envs:
            reset = reset | (episode_length.to(device=device) <= 1)

        if torch.any(reset):
            self._history[reset] = frame[reset].unsqueeze(1).expand(-1, self.window_len, -1)
            self._valid_count[reset] = 0

        self._history = torch.roll(self._history, shifts=-1, dims=1)
        self._history[:, -1] = frame
        self._valid_count = torch.clamp(self._valid_count + 1, max=self.window_len)
        self._prev_time_steps = time_steps.clone()
        self._prev_motion_idx = motion_idx.clone()
        return self._valid_count >= self.window_len

    @torch.no_grad()
    def score(
        self,
        env: "ManagerBasedRLEnv",
        command: "MotionCommand",
        reward_std: float = 1.0,
        error_clip: float = 25.0,
        require_full_window: bool = True,
        kick_foot_name: str = "right_ankle_roll_link",
        support_foot_name: str = "left_ankle_roll_link",
        basis_mode: str = "reference",
    ) -> torch.Tensor:
        frame = self._extract_frame(command).to(device=self.device, dtype=torch.float32)
        full_window = self._ensure_history(frame, command, env)
        assert self._history is not None

        x = self._history.reshape(frame.shape[0], -1)
        cond = self._condition(env, command, kick_foot_name, support_foot_name, basis_mode).to(
            device=self.device,
            dtype=torch.float32,
        )
        x_norm = (x - self.feature_mean) / self.feature_std
        cond_norm = (cond - self.cond_mean) / self.cond_std

        err = reconstruction_error(self.model, x_norm, cond_norm).clamp(max=error_clip)
        score = torch.exp(-err / max(reward_std * reward_std, 1.0e-6))
        if require_full_window:
            score = score * full_window.to(dtype=score.dtype, device=score.device)

        if hasattr(command, "metrics"):
            command.metrics["content_cvae_recon_error"] = err.detach()
            command.metrics["content_cvae_score"] = score.detach()
            command.metrics["content_cvae_window_full"] = full_window.to(dtype=torch.float32, device=score.device)
        return score


def get_content_cvae_prior(
    env: "ManagerBasedRLEnv",
    model_path: str,
    device: str | torch.device | None = None,
) -> ContentCVAEPrior:
    """Return a cached runtime prior for ``model_path`` on this environment."""
    if device is None:
        device = getattr(env, "device", "cpu")
    cache = getattr(env, "_content_cvae_prior_cache", None)
    if cache is None:
        cache = {}
        setattr(env, "_content_cvae_prior_cache", cache)
    key = (os.path.abspath(model_path), str(device))
    if key not in cache:
        cache[key] = ContentCVAEPrior(model_path=model_path, device=device)
    return cache[key]


def content_cvae_prior_reward(
    env: "ManagerBasedRLEnv",
    command_name: str = "motion",
    model_path: str = "models/content_cvae_prior_video1_clean_v3_final_local.pt",
    reward_std: float = 1.0,
    error_clip: float = 25.0,
    require_full_window: bool = True,
    kick_foot_name: str = "right_ankle_roll_link",
    support_foot_name: str = "left_ankle_roll_link",
    basis_mode: str = "reference",
) -> torch.Tensor:
    """Reward high likelihood under a trained content-conditioned CVAE prior."""
    command = env.command_manager.get_term(command_name)
    prior = get_content_cvae_prior(env, model_path=model_path)
    return prior.score(
        env=env,
        command=command,
        reward_std=reward_std,
        error_clip=error_clip,
        require_full_window=require_full_window,
        kick_foot_name=kick_foot_name,
        support_foot_name=support_foot_name,
        basis_mode=basis_mode,
    )
