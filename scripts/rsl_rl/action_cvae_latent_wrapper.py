"""RSL-RL wrapper that decodes latent residual actions through an action CVAE."""

from __future__ import annotations

import importlib.util
import os

import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from soccer.tasks.tracking.mdp.event_conditioned_obs_builder import V10ObsBuilder

_model_path = os.path.join(os.path.dirname(__file__), "action_cvae_distill.py")
_spec = importlib.util.spec_from_file_location("action_cvae_distill", os.path.abspath(_model_path))
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
StateActionCVAE = _mod.StateActionCVAE

_train_path = os.path.join(os.path.dirname(__file__), "train_action_cvae_distill.py")
_train_spec = importlib.util.spec_from_file_location("train_action_cvae_distill", os.path.abspath(_train_path))
_train_mod = importlib.util.module_from_spec(_train_spec)
assert _train_spec.loader is not None
_train_spec.loader.exec_module(_train_mod)
apply_obs_slices = _train_mod.apply_obs_slices


def load_action_cvae(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = StateActionCVAE(
        obs_dim=int(ckpt["obs_dim"]),
        action_dim=int(ckpt["action_dim"]),
        latent_dim=int(ckpt["latent_dim"]),
        hidden_dims=list(ckpt["hidden_dims"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, ckpt


class ActionCVAELatentRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """Expose latent residual actions while stepping the wrapped env with decoded PD actions."""

    def __init__(
        self,
        env,
        *,
        action_cvae_path: str,
        latent_scale: float = 0.75,
        latent_clip: float = 5.0,
        pd_action_clip: float = 0.0,
        pd_residual_scale: float = 0.0,
        pd_residual_joint_scope: str = "all",
        pd_residual_gate_dist: float = 0.9,
        pd_residual_gate_temp: float = 0.2,
        pd_residual_closing_threshold: float = 0.0,
        pd_residual_closing_temp: float = 0.5,
        latent_barrier_weight: float = 0.0,
        latent_barrier_limit: float = 2.5,
    ):
        super().__init__(env)
        self.latent_scale = float(latent_scale)
        self.latent_clip = float(latent_clip)
        self.latent_barrier_weight = float(latent_barrier_weight)
        self.latent_barrier_limit = float(latent_barrier_limit)
        self.pd_action_clip = float(pd_action_clip)
        self.pd_residual_scale = float(pd_residual_scale)
        self.pd_residual_joint_scope = pd_residual_joint_scope
        self.pd_residual_gate_dist = float(pd_residual_gate_dist)
        self.pd_residual_gate_temp = float(pd_residual_gate_temp)
        self.pd_residual_closing_threshold = float(pd_residual_closing_threshold)
        self.pd_residual_closing_temp = float(pd_residual_closing_temp)
        self.use_pd_residual = self.pd_residual_scale > 0.0

        self.action_cvae, self.action_cvae_ckpt = load_action_cvae(action_cvae_path, self.device)
        self.latent_dim = int(self.action_cvae_ckpt["latent_dim"])
        self.base_action_dim = int(self.action_cvae_ckpt.get("base_action_dim", 29))
        self.action_horizon = int(self.action_cvae_ckpt.get("action_horizon", 1))
        self.obs_slices = list(self.action_cvae_ckpt["obs_slices"])
        self.obs_mean = self.action_cvae_ckpt["obs_mean"].to(self.device)
        self.obs_std = self.action_cvae_ckpt["obs_std"].to(self.device)
        self.action_mean = self.action_cvae_ckpt["action_mean"].to(self.device)
        self.action_std = self.action_cvae_ckpt["action_std"].to(self.device)

        command = self.unwrapped.command_manager.get_term("motion")
        self.pd_residual_joint_ids = self._resolve_pd_residual_joint_ids(command)
        self.pd_residual_dim = int(self.pd_residual_joint_ids.numel())
        self.v10_builder = V10ObsBuilder(
            num_envs=self.unwrapped.num_envs,
            num_joints=command.robot.data.joint_pos.shape[1],
            device=self.unwrapped.device,
        )
        self.v10_builder.init_segment_bounds(command)

        self.num_actions = self.latent_dim + (self.pd_residual_dim if self.use_pd_residual else 0)
        self.num_obs = sum(end - start for start, end in self.obs_slices)
        self.num_privileged_obs = self.num_obs
        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.latent_clip, high=self.latent_clip, shape=(self.num_actions,), dtype=float
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space, self.num_envs
        )

        self._last_obs_full = None
        self._last_latent_maha = torch.zeros(self.num_envs, device=self.device)
        self._last_latent_barrier_penalty = torch.zeros(self.num_envs, device=self.device)
        self._last_pd_action_norm = torch.zeros(self.num_envs, device=self.device)
        self._last_pd_residual_gate = torch.zeros(self.num_envs, device=self.device)
        self._last_pd_residual_norm = torch.zeros(self.num_envs, device=self.device)

    def _resolve_pd_residual_joint_ids(self, command) -> torch.Tensor:
        if not self.use_pd_residual:
            return torch.empty(0, dtype=torch.long, device=self.device)
        scope = self.pd_residual_joint_scope.strip().lower()
        if scope in ("all", "full", "29d"):
            return torch.arange(self.base_action_dim, dtype=torch.long, device=self.device)
        if scope in ("swing_leg", "kick_leg", "right_leg"):
            target_names = [
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
            ]
        elif scope == "swing_leg_no_ankle":
            target_names = [
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
            ]
        else:
            raise ValueError(f"Unknown pd_residual_joint_scope={self.pd_residual_joint_scope!r}")

        joint_names = list(command.robot.data.joint_names)
        ids = []
        for target in target_names:
            matches = [i for i, name in enumerate(joint_names) if name == target or name.endswith(target)]
            if not matches:
                raise ValueError(f"PD residual joint {target!r} not found in robot joints: {joint_names}")
            ids.append(matches[0])
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def _pd_residual_gate(self, obs_full: torch.Tensor) -> torch.Tensor:
        # V10 layout: ball-foot relation starts at 392; index 14 is swing-foot
        # XY distance to ball; index 20 is swing-foot velocity along the vector
        # toward the ball. This keeps the correction geometry-gated rather than
        # tied to the reference kick_frame schedule.
        swing_foot_ball_dist = obs_full[:, 392 + 14]
        swing_closing_speed = obs_full[:, 392 + 20]
        dist_temp = max(self.pd_residual_gate_temp, 1.0e-4)
        closing_temp = max(self.pd_residual_closing_temp, 1.0e-4)
        dist_gate = torch.sigmoid((self.pd_residual_gate_dist - swing_foot_ball_dist) / dist_temp)
        closing_gate = torch.sigmoid((swing_closing_speed - self.pd_residual_closing_threshold) / closing_temp)
        return (dist_gate * closing_gate).unsqueeze(-1)

    def _compute_v10(self) -> torch.Tensor:
        command = self.unwrapped.command_manager.get_term("motion")
        return self.v10_builder.compute(self.unwrapped, command)

    def _select_policy_obs(self, obs_full: torch.Tensor) -> torch.Tensor:
        return apply_obs_slices(obs_full, self.obs_slices)

    @torch.no_grad()
    def _decode_latent_residual(self, obs_full: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        latent_residual = residual[:, : self.latent_dim].clamp(-self.latent_clip, self.latent_clip)
        pd_residual = residual[:, self.latent_dim:] if self.use_pd_residual else None
        obs = self._select_policy_obs(obs_full)
        obs_norm = (obs - self.obs_mean) / self.obs_std
        p_mu, p_logvar = self.action_cvae.prior_stats(obs_norm)
        p_std = torch.exp(0.5 * p_logvar)
        z = p_mu + self.latent_scale * p_std * torch.tanh(latent_residual)
        action_norm = self.action_cvae.decode(obs_norm, z)
        action = action_norm * self.action_std + self.action_mean
        action = action.view(action.shape[0], self.action_horizon, self.base_action_dim)[:, 0]
        if self.use_pd_residual:
            gate = self._pd_residual_gate(obs_full)
            delta_scoped = self.pd_residual_scale * gate * torch.tanh(pd_residual)
            delta = torch.zeros_like(action)
            delta[:, self.pd_residual_joint_ids] = delta_scoped
            action = action + delta
            self._last_pd_residual_gate = gate.squeeze(-1)
            self._last_pd_residual_norm = torch.norm(delta, dim=-1)
        else:
            self._last_pd_residual_gate.zero_()
            self._last_pd_residual_norm.zero_()
        if self.pd_action_clip > 0.0:
            action = action.clamp(-self.pd_action_clip, self.pd_action_clip)
        self._last_latent_maha = torch.norm((z - p_mu) / p_std.clamp(min=1.0e-6), dim=-1)
        self._last_latent_barrier_penalty = torch.relu(
            self._last_latent_maha - self.latent_barrier_limit
        ).pow(2)
        self._last_pd_action_norm = torch.norm(action, dim=-1)
        return action

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        self._last_obs_full = self._compute_v10()
        obs = self._select_policy_obs(self._last_obs_full)
        return obs, {"observations": {"policy": obs, "critic": obs}}

    def reset(self) -> tuple[torch.Tensor, dict]:
        super().reset()
        command = self.unwrapped.command_manager.get_term("motion")
        self.v10_builder.init_segment_bounds(command)
        self.v10_builder.reset(torch.arange(self.unwrapped.num_envs, device=self.unwrapped.device))
        self._last_obs_full = self._compute_v10()
        obs = self._select_policy_obs(self._last_obs_full)
        return obs, {"observations": {"policy": obs, "critic": obs}}

    def step(self, latent_actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if self._last_obs_full is None:
            self._last_obs_full = self._compute_v10()
        pd_actions = self._decode_latent_residual(self._last_obs_full, latent_actions)
        obs_dict, rew, terminated, truncated, extras = self.env.step(pd_actions)
        if self.latent_barrier_weight > 0.0:
            barrier = self.latent_barrier_weight * self._last_latent_barrier_penalty
            while barrier.dim() < rew.dim():
                barrier = barrier.unsqueeze(-1)
            rew = rew - barrier
        dones = (terminated | truncated).to(dtype=torch.long)

        command = self.unwrapped.command_manager.get_term("motion")
        self.v10_builder.update_history(self.unwrapped, command, pd_actions, dones)
        self._last_obs_full = self._compute_v10()
        obs = self._select_policy_obs(self._last_obs_full)

        extras["observations"] = {"policy": obs, "critic": obs}
        extras.setdefault("log", {})
        extras["log"]["latent/maha"] = self._last_latent_maha.mean()
        extras["log"]["latent/barrier_penalty"] = self._last_latent_barrier_penalty.mean()
        extras["log"]["latent/barrier_reward"] = -self.latent_barrier_weight * self._last_latent_barrier_penalty.mean()
        extras["log"]["latent/pd_action_norm"] = self._last_pd_action_norm.mean()
        extras["log"]["latent/pd_residual_gate"] = self._last_pd_residual_gate.mean()
        extras["log"]["latent/pd_residual_norm"] = self._last_pd_residual_norm.mean()
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        return obs, rew, dones, extras
