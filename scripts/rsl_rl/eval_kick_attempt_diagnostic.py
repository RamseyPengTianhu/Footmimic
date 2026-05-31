"""Attempt-based kick diagnostic evaluation.

This evaluator is stricter than eval_kick_diagnostic.py.  It treats a kick as
an explicit strike attempt:

1. first high-speed swing toward the ball starts an attempt
2. contact must happen within a short window after that attempt
3. contact after a missed attempt is counted as late fallback, not clean success

This catches the failure mode where the policy kicks air first, then later bumps
the ball and still looks successful under first-contact metrics.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

from isaaclab.app import AppLauncher
import cli_args


parser = argparse.ArgumentParser(description="Attempt-based kick diagnostic eval.")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--motion_file", type=str, default=None)
parser.add_argument("--motion_path", type=str, default=None)
parser.add_argument("--eval_episodes", type=int, default=20)
parser.add_argument("--attempt_speed_threshold", type=float, default=2.0)
parser.add_argument("--attempt_closing_speed", type=float, default=0.5)
parser.add_argument("--attempt_max_foot_ball_dist", type=float, default=0.9)
parser.add_argument("--attempt_min_kick_height", type=float, default=0.02)
parser.add_argument("--attempt_max_kick_height", type=float, default=0.65)
parser.add_argument("--attempt_near_ball_dist", type=float, default=1.25)
parser.add_argument("--attempt_early_grace", type=int, default=25)
parser.add_argument("--attempt_window", type=int, default=18)
parser.add_argument("--ball_speed_success", type=float, default=2.0)
parser.add_argument("--direction_success", type=float, default=0.5)
parser.add_argument("--contact_force_threshold", type=float, default=5.0)
parser.add_argument("--ball_x_offset", type=float, default=0.0)
parser.add_argument("--ball_y_offset", type=float, default=0.0)
parser.add_argument("--ball_xy_perturb", type=float, default=0.0)
parser.add_argument("--output_json", type=str, default=None)
parser.add_argument("--action_cvae", type=str, default=None, help="Frozen action-CVAE decoder for latent-residual PPO policies.")
parser.add_argument("--latent_v2_model", type=str, default=None, help="Frozen LATENT-v2 decoder/prior for latent PPO policies.")
parser.add_argument("--decoder_only", action="store_true", help="Evaluate the latent_v2_model prior directly without a PPO policy.")
parser.add_argument("--posterior_quant", action="store_true",
                    help="Evaluate posterior quantized path: teacher_action → encoder → VQ → decoder. "
                         "Requires both --latent_v2_model and a teacher checkpoint (--load_run).")
parser.add_argument("--lab_scale", type=float, default=2.0, help="LATENT-v2 LAB scale.")
parser.add_argument("--ppo_logit_scale", type=float, default=1.0, help="VQ PPO logit residual scale.")
parser.add_argument("--kl_categorical_weight", type=float, default=0.01, help="VQ PPO categorical KL weight.")
parser.add_argument("--no_prior", action="store_true", default=False,
                    help="Ablation: PPO selects codes without Prior bias (must match training).")
parser.add_argument("--residual_alpha", type=float, default=0.0,
                    help="Route beta: continuous residual scale. Must match training.")
parser.add_argument("--prior_code_only", action="store_true", default=False,
                    help="Residual-only: code from Prior, PPO only outputs residual.")
parser.add_argument("--categorical_ppo", action="store_true", default=False,
                    help="True categorical PPO: actor outputs K logits via Categorical distribution.")
parser.add_argument("--disable_ref_terminations", action="store_true", default=False,
                    help="Disable ref-based terminations (ee_body_pos, anchor_pos_z).")
parser.add_argument("--policy_obs_mode", type=str, default="full", choices=("full", "task", "task_features"),
                    help="High-level PPO observation used by LATENT-v2 checkpoints.")
parser.add_argument("--latent_scale", type=float, default=0.75)
parser.add_argument("--latent_clip", type=float, default=5.0)
parser.add_argument("--pd_action_clip", type=float, default=0.0)
parser.add_argument("--pd_residual_scale", type=float, default=0.0)
parser.add_argument("--pd_residual_joint_scope", type=str, default="all")
parser.add_argument("--pd_residual_gate_dist", type=float, default=0.9)
parser.add_argument("--pd_residual_gate_temp", type=float, default=0.2)
parser.add_argument("--pd_residual_closing_threshold", type=float, default=0.0)
parser.add_argument("--pd_residual_closing_temp", type=float, default=0.5)
parser.add_argument("--latent_barrier_weight", type=float, default=0.0)
parser.add_argument("--latent_barrier_limit", type=float, default=2.5)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if getattr(args_cli, "headless", False) and "DISPLAY" in os.environ:
    print(f"[INFO] Headless mode: clearing DISPLAY={os.environ['DISPLAY']!r} before launching Isaac Sim")
    os.environ.pop("DISPLAY", None)
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import soccer.tasks  # noqa: F401


def _load_latent_v2_model(path: str, device: str):
    from latent_v2_models import LatentActionModel

    ckpt = torch.load(path, map_location=device, weights_only=False)
    decoder_obs_mode = ckpt.get("decoder_obs_mode", "full")
    model = LatentActionModel(
        obs_dim=int(ckpt["obs_dim"]),
        action_dim=int(ckpt["action_dim"]),
        z_dim=int(ckpt["z_dim"]),
        hidden_dims=list(ckpt["hidden_dims"]),
        decoder_obs_mode=decoder_obs_mode,
        prior_type=ckpt.get("prior_type", "mlp"),
        num_codes=int(ckpt.get("num_codes", 16)),
        commitment_weight=float(ckpt.get("commitment_weight", 0.25)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    print(f"[INFO] Loaded latent model: decoder_obs_mode={decoder_obs_mode}")
    return model, ckpt


def _latent_v2_policy_obs_dim(decoder_obs_dim: int, mode: str) -> int:
    if mode == "full":
        return decoder_obs_dim
    if mode == "task":
        if decoder_obs_dim < 160:
            raise ValueError(f"task policy obs expects obs_v3 >=160D, got {decoder_obs_dim}")
        return 3 + (decoder_obs_dim - 64)
    if mode == "task_features":
        from compute_task_features import TASK_FEATURES_DIM
        return 3 + (decoder_obs_dim - 64) + TASK_FEATURES_DIM
    raise ValueError(f"Unknown policy_obs_mode={mode!r}")


def _select_latent_v2_policy_obs(
    obs_v3: torch.Tensor, mode: str, task_features: torch.Tensor | None = None
) -> torch.Tensor:
    if mode == "full":
        return obs_v3
    if mode == "task":
        if obs_v3.shape[-1] < 160:
            raise ValueError(f"task policy obs expects obs_v3 >=160D, got {obs_v3.shape[-1]}")
        return torch.cat((obs_v3[:, 58:61], obs_v3[:, 64:]), dim=-1)
    if mode == "task_features":
        if task_features is None:
            raise ValueError("task_features must be provided for policy_obs_mode='task_features'")
        proprio = torch.cat((obs_v3[:, 58:61], obs_v3[:, 64:]), dim=-1)
        return torch.cat((proprio, task_features), dim=-1)
    raise ValueError(f"Unknown policy_obs_mode={mode!r}")


class LatentV2PPOEnvWrapper(RslRlVecEnvWrapper):
    """Decode LATENT-v2 PPO actions through a frozen LAB decoder or VQ codebook."""

    def __init__(
        self,
        env,
        *,
        latent_model_path: str,
        lab_scale: float = 2.0,
        latent_clip: float = 5.0,
        lab_barrier_weight: float = 0.0,
        lab_barrier_limit: float = 2.5,
        policy_obs_mode: str = "full",
        ppo_logit_scale: float = 1.0,
        kl_categorical_weight: float = 0.01,
        no_prior: bool = False,
        residual_alpha: float = 0.0,
        prior_code_only: bool = False,
        categorical_ppo: bool = False,
    ):
        super().__init__(env)
        self.lab_scale = float(lab_scale)
        self.latent_clip = float(latent_clip)
        self.lab_barrier_weight = float(lab_barrier_weight)
        self.lab_barrier_limit = float(lab_barrier_limit)
        self.policy_obs_mode = policy_obs_mode
        self.ppo_logit_scale = ppo_logit_scale
        self.kl_categorical_weight = kl_categorical_weight
        self.no_prior = no_prior
        self.residual_alpha = residual_alpha
        self.prior_code_only = prior_code_only
        self.categorical_ppo = categorical_ppo
        self.latent_model, self.latent_ckpt = _load_latent_v2_model(latent_model_path, self.device)
        self.z_dim = int(self.latent_ckpt["z_dim"])
        self.obs_dim_latent = int(self.latent_ckpt["obs_dim"])
        self.policy_obs_dim = _latent_v2_policy_obs_dim(self.obs_dim_latent, self.policy_obs_mode)
        self.is_vq = (self.latent_ckpt.get("prior_type", "mlp") == "vq")
        self.num_codes = int(self.latent_ckpt.get("num_codes", 16))
        self.code_hold = int(self.latent_ckpt.get("code_hold", 1))
        self.use_residual = (self.residual_alpha > 0.0)

        if self.categorical_ppo:
            self.num_actions = 1
        elif self.is_vq:
            if self.use_residual and self.prior_code_only:
                self.num_actions = self.z_dim
            elif self.use_residual:
                self.num_actions = self.num_codes + self.z_dim
            else:
                self.num_actions = self.num_codes
        else:
            self.num_actions = self.z_dim
        if self.categorical_ppo:
            self.num_obs = self.policy_obs_dim + self.num_codes
        else:
            self.num_obs = self.policy_obs_dim
        if self.policy_obs_mode in ("task", "task_features"):
            self.num_privileged_obs = self.num_obs
        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.latent_clip,
            high=self.latent_clip,
            shape=(self.num_actions,),
            dtype=float,
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space,
            self.num_envs,
        )
        self._cached_obs_v3 = None
        self._cached_task_features = None
        self._use_task_features = (self.policy_obs_mode == "task_features")
        self.base_env = self.env.unwrapped
        self._last_maha = torch.zeros(self.num_envs, device=self.device)
        self._last_barrier_penalty = torch.zeros(self.num_envs, device=self.device)
        self.eval_mode = True  # Evaluation script runs in deterministic mode

        # VQ code-hold state
        if self.is_vq:
            self._held_code = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._held_zq = torch.zeros(self.num_envs, self.z_dim, device=self.device)
            self._hold_ctr = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._last_kl_cat = torch.zeros(self.num_envs, device=self.device)
            print(f"[INFO] VQ PPO wrapper: num_codes={self.num_codes}, code_hold={self.code_hold}, "
                  f"ppo_logit_scale={self.ppo_logit_scale}")

    @torch.no_grad()
    def _decode_vq(self, obs_v3: torch.Tensor, ppo_raw: torch.Tensor) -> torch.Tensor:
        """VQ categorical decode with optional continuous residual."""
        import torch.nn.functional as F

        tf = self._cached_task_features if self._use_task_features else None
        dec_obs = self.latent_model.select_decoder_obs(obs_v3, task_features=tf)

        # Split PPO output
        if self.use_residual and self.prior_code_only:
            ppo_logits = None
            residual_raw = ppo_raw
        elif self.use_residual:
            ppo_logits = ppo_raw[:, :self.num_codes]
            residual_raw = ppo_raw[:, self.num_codes:]
        else:
            ppo_logits = ppo_raw
            residual_raw = None

        prior_logits = self.latent_model.prior(dec_obs)
        if self.prior_code_only:
            combined = prior_logits
        elif self.no_prior:
            combined = self.ppo_logit_scale * ppo_logits
            prior_logits = torch.zeros_like(combined)
        else:
            combined = prior_logits + self.ppo_logit_scale * ppo_logits

        needs_update = (self._hold_ctr % self.code_hold == 0)
        if needs_update.any():
            if getattr(self, "eval_mode", False):
                new_code = combined.argmax(dim=-1)
            else:
                import torch.distributions as D
                dist = D.Categorical(logits=combined)
                new_code = dist.sample()
            new_zq = self.latent_model.codebook.lookup(new_code)
            self._held_code[needs_update] = new_code[needs_update]
            self._held_zq[needs_update] = new_zq[needs_update]

        # Apply continuous residual
        if self.use_residual and residual_raw is not None:
            z = self._held_zq + self.residual_alpha * torch.tanh(residual_raw)
        else:
            z = self._held_zq

        action = self.latent_model.decoder(dec_obs, z)

        combined_p = F.softmax(combined, dim=-1)
        prior_p = F.softmax(prior_logits, dim=-1)
        kl = (combined_p * (combined_p.log() - prior_p.log())).sum(dim=-1)
        self._last_kl_cat = kl
        self._last_barrier_penalty = torch.zeros_like(kl) if self.no_prior else self.kl_categorical_weight * kl
        return action

    @torch.no_grad()
    def _decode_latent(self, obs_v3: torch.Tensor, latent_actions: torch.Tensor) -> torch.Tensor:
        latent_actions = latent_actions.clamp(-self.latent_clip, self.latent_clip)
        tf = self._cached_task_features if self._use_task_features else None
        dec_obs = self.latent_model.select_decoder_obs(obs_v3, task_features=tf)
        p_mu, p_logvar = self.latent_model.prior(dec_obs)
        p_std = torch.exp(0.5 * p_logvar)
        z = p_mu + self.lab_scale * p_std * torch.tanh(latent_actions)
        action = self.latent_model.decoder(dec_obs, z)
        self._last_maha = torch.norm((z - p_mu) / p_std.clamp(min=1.0e-6), dim=-1)
        self._last_barrier_penalty = torch.relu(self._last_maha - self.lab_barrier_limit).pow(2)
        return action

    def _compute_task_features(self):
        if not self._use_task_features:
            return None
        from compute_task_features import compute_ball_foot_relation
        self._cached_task_features = compute_ball_foot_relation(self.base_env)
        return self._cached_task_features

    def _select_policy_obs(self, obs_v3: torch.Tensor) -> torch.Tensor:
        tf = self._cached_task_features if self._use_task_features else None
        policy_obs = _select_latent_v2_policy_obs(obs_v3, self.policy_obs_mode, task_features=tf)
        if self.categorical_ppo:
            dec_obs = self.latent_model.select_decoder_obs(obs_v3, task_features=tf)
            prior_logits = self.latent_model.prior(dec_obs)
            policy_obs = torch.cat([policy_obs, prior_logits], dim=-1)
        return policy_obs

    def _policy_extras(self, extras: dict, policy_obs: torch.Tensor) -> dict:
        if self.policy_obs_mode not in ("task", "task_features"):
            return extras
        extras = dict(extras)
        extras["observations"] = {"policy": policy_obs, "critic": policy_obs}
        return extras

    def get_observations(self):
        obs, extras = super().get_observations()
        self._cached_obs_v3 = obs.clone()
        self._compute_task_features()
        policy_obs = self._select_policy_obs(obs)
        return policy_obs, self._policy_extras(extras, policy_obs)

    def reset(self):
        obs, extras = super().reset()
        self._cached_obs_v3 = obs.clone()
        self._compute_task_features()
        policy_obs = self._select_policy_obs(obs)
        return policy_obs, self._policy_extras(extras, policy_obs)

    def step(self, latent_actions: torch.Tensor):
        obs_v3 = self._cached_obs_v3
        if obs_v3 is None:
            obs_v3, _ = super().get_observations()
            self._cached_obs_v3 = obs_v3.clone()
        if self.categorical_ppo:
            # Categorical PPO: action is a code index [B, 1]
            tf = self._cached_task_features if self._use_task_features else None
            dec_obs = self.latent_model.select_decoder_obs(obs_v3, task_features=tf)
            code = latent_actions.squeeze(-1).long()
            needs_update = (self._hold_ctr % self.code_hold == 0)
            if needs_update.any():
                new_zq = self.latent_model.codebook.lookup(code)
                self._held_code[needs_update] = code[needs_update]
                self._held_zq[needs_update] = new_zq[needs_update]
            joint_action = self.latent_model.decoder(dec_obs, self._held_zq)
        elif self.is_vq:
            joint_action = self._decode_vq(obs_v3, latent_actions)
        else:
            joint_action = self._decode_latent(obs_v3, latent_actions)
        obs, rew, dones, extras = super().step(joint_action)
        # Apply penalty
        if self.is_vq:
            penalty = self._last_barrier_penalty
            while penalty.dim() < rew.dim():
                penalty = penalty.unsqueeze(-1)
            rew = rew - penalty
            # Hold counter management
            self._hold_ctr += 1
            if isinstance(dones, dict):
                reset_mask = dones.get("terminated", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)) | \
                             dones.get("truncated", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
            else:
                reset_mask = dones.bool()
            self._hold_ctr[reset_mask] = 0
        elif self.lab_barrier_weight > 0.0:
            barrier = self.lab_barrier_weight * self._last_barrier_penalty
            while barrier.dim() < rew.dim():
                barrier = barrier.unsqueeze(-1)
            rew = rew - barrier
        self._cached_obs_v3 = obs.clone()
        self._compute_task_features()
        policy_obs = self._select_policy_obs(obs)
        extras = self._policy_extras(extras, policy_obs)
        extras.setdefault("log", {})
        if self.is_vq:
            extras["log"]["latent/kl_categorical"] = self._last_kl_cat.mean()
            extras["log"]["latent/held_code_mean"] = self._held_code.float().mean()
        else:
            extras["log"]["latent/maha_dist"] = self._last_maha.mean()
            extras["log"]["latent/barrier_penalty"] = self._last_barrier_penalty.mean()
        return policy_obs, rew, dones, extras


def get_motion_files(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    return sorted(glob.glob(os.path.join(path, "*.npz")))


def _quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _mean(values, default=0.0):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else default


class KickAttemptEvaluator:
    def __init__(
        self,
        env,
        num_motions: int,
        motion_names: list[str],
        target_eps: int,
        device: torch.device,
        *,
        ball_x_offset: float = 0.0,
        ball_y_offset: float = 0.0,
        ball_xy_perturb: float = 0.0,
    ):
        self.env = env
        self.base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.N = env.num_envs
        self.num_motions = num_motions
        self.motion_names = motion_names
        self.target_eps = target_eps
        self.device = device
        self.ball_x_offset = ball_x_offset
        self.ball_y_offset = ball_y_offset
        self.ball_xy_perturb = ball_xy_perturb

        self.env_motion_idx = torch.zeros(self.N, dtype=torch.long, device=device)
        self.env_step = torch.zeros(self.N, dtype=torch.long, device=device)
        self.results = defaultdict(list)
        self.episodes_done = torch.zeros(num_motions, dtype=torch.long)

        cmd = self.base_env.command_manager.get_term("motion")
        robot = cmd.robot
        self.left_idx = robot.body_names.index("left_ankle_roll_link")
        self.right_idx = robot.body_names.index("right_ankle_roll_link")
        self.pelvis_idx = robot.body_names.index("pelvis")

        self._disc = None
        self._disc_extractor = None
        disc_path = "models/strike_discriminator.pt"
        if os.path.exists(disc_path):
            try:
                from soccer.tasks.tracking.mdp.strike_discriminator import (
                    INPUT_DIM,
                    StrikeDiscriminator,
                    StrikeFeatureExtractor,
                )

                ckpt = torch.load(disc_path, map_location=device, weights_only=False)
                self._disc = StrikeDiscriminator(
                    input_dim=ckpt.get("input_dim", INPUT_DIM),
                    hidden=ckpt.get("hidden", 64),
                )
                self._disc.load_state_dict(ckpt["model_state_dict"])
                self._disc.to(device)
                self._disc.eval()
                self._disc_extractor = StrikeFeatureExtractor()
                self._disc_extractor.init_indices(robot)
                print(f"[INFO] Loaded strike discriminator from {disc_path}")
            except Exception as exc:
                print(f"[WARN] Could not load discriminator: {exc}")

        self._init_accumulators()

    def _init_accumulators(self):
        N, D = self.N, self.device
        z = lambda: torch.zeros(N, device=D)
        zl = lambda: torch.zeros(N, dtype=torch.long, device=D)
        zb = lambda: torch.zeros(N, dtype=torch.bool, device=D)

        self.attempt_started = zb()
        self.attempt_missed = zb()
        self.attempt_hit = zb()
        self.fallback_contact = zb()
        self.contact_seen = zb()
        self.correct_foot = zb()

        self.attempt_frame = torch.full((N,), -1, dtype=torch.long, device=D)
        self.contact_frame = torch.full((N,), -1, dtype=torch.long, device=D)
        self.fallback_frame = torch.full((N,), -1, dtype=torch.long, device=D)

        self.attempt_kick_speed = z()
        self.attempt_closing_speed = z()
        self.attempt_dist_xy = z()
        self.attempt_dist_3d = z()
        self.attempt_kick_height = z()
        self.attempt_support_lat = z()
        self.attempt_support_long = z()
        self.attempt_support_yaw = z()
        self.attempt_support_speed = z()
        self.attempt_support_height = z()
        self.attempt_d_score = z()

        self.contact_kick_speed = z()
        self.contact_dist_xy = z()
        self.contact_support_lat = z()
        self.contact_support_long = z()
        self.contact_support_yaw = z()
        self.contact_support_height = z()
        self.contact_d_score = z()

        self.peak_ball_speed = z()
        self.ball_dir_align = z()
        self.post_frames = zl()
        self.post_max_tilt = z()
        self.post_max_angvel_xy = z()

        # ── NoAtt diagnostics: running min/max for episodes without attempt ──
        self.min_pelvis_ball_dist = torch.full((N,), 999.0, device=D)
        self.min_foot_ball_dist = torch.full((N,), 999.0, device=D)
        self.max_kick_speed = z()
        self.max_closing_speed = z()
        self.ever_near_ball = zb()  # pelvis < attempt_near_ball_dist
        self.closest_ball_frame = zl()  # step when min_foot_ball_dist was achieved

    def _reset_accumulators(self, ids: torch.Tensor):
        self.env_step[ids] = 0
        self.attempt_started[ids] = False
        self.attempt_missed[ids] = False
        self.attempt_hit[ids] = False
        self.fallback_contact[ids] = False
        self.contact_seen[ids] = False
        self.correct_foot[ids] = False
        self.attempt_frame[ids] = -1
        self.contact_frame[ids] = -1
        self.fallback_frame[ids] = -1
        for tensor in [
            self.attempt_kick_speed,
            self.attempt_closing_speed,
            self.attempt_dist_xy,
            self.attempt_dist_3d,
            self.attempt_kick_height,
            self.attempt_support_lat,
            self.attempt_support_long,
            self.attempt_support_yaw,
            self.attempt_support_speed,
            self.attempt_support_height,
            self.attempt_d_score,
            self.contact_kick_speed,
            self.contact_dist_xy,
            self.contact_support_lat,
            self.contact_support_long,
            self.contact_support_yaw,
            self.contact_support_height,
            self.contact_d_score,
            self.peak_ball_speed,
            self.ball_dir_align,
            self.post_max_tilt,
            self.post_max_angvel_xy,
        ]:
            tensor[ids] = 0.0
        self.post_frames[ids] = 0
        self.min_pelvis_ball_dist[ids] = 999.0
        self.min_foot_ball_dist[ids] = 999.0
        self.max_kick_speed[ids] = 0.0
        self.max_closing_speed[ids] = 0.0
        self.ever_near_ball[ids] = False
        self.closest_ball_frame[ids] = 0

    def assign_motions_round_robin(self):
        cmd = self.base_env.command_manager.get_term("motion")
        for i in range(self.N):
            self.env_motion_idx[i] = i % self.num_motions
        cmd.motion_idx[:] = self.env_motion_idx
        cmd.motion_length[:] = cmd.motion.file_lengths[self.env_motion_idx]
        cmd.time_steps[:] = 0

    def _perturb_ball_position(self, env_ids=None):
        if self.ball_x_offset == 0.0 and self.ball_y_offset == 0.0 and self.ball_xy_perturb == 0.0:
            return
        cmd = self.base_env.command_manager.get_term("motion")
        soccer_ball = cmd.soccer_ball
        env_origins = getattr(self.base_env.scene, "env_origins", None)
        if soccer_ball is None or env_origins is None:
            return
        ids = torch.arange(self.N, device=self.device) if env_ids is None else env_ids
        if ids.numel() == 0:
            return
        ball_state = soccer_ball.data.root_state_w[ids].clone()
        ball_state[:, 0] += self.ball_x_offset
        ball_state[:, 1] += self.ball_y_offset
        if self.ball_xy_perturb > 0:
            perturb = (torch.rand(ids.numel(), 2, device=self.device) - 0.5) * 2 * self.ball_xy_perturb
            ball_state[:, 0] += perturb[:, 0]
            ball_state[:, 1] += perturb[:, 1]
        ball_state[:, 7:] = 0.0
        local_xy = ball_state[:, :2] - env_origins[ids, :2]
        with torch.inference_mode():
            soccer_ball.write_root_state_to_sim(ball_state.clone(), env_ids=ids)
            cmd.soccer_ball_pos[ids, 0] = local_xy[:, 0]
            cmd.soccer_ball_pos[ids, 1] = local_xy[:, 1]
            cmd.target_point_pos[ids] = cmd.soccer_ball_pos[ids].clone()

    def _get_ball_contact(self):
        try:
            sensor = self.base_env.scene["soccer_ball_contact"]
            forces = sensor.data.net_forces_w_history
            if forces.dim() == 4:
                force_vec = forces[:, :, 0, :2].sum(dim=1)
            else:
                force_vec = forces[:, 0, :2]
            force_mag = torch.norm(force_vec, dim=-1)
            return force_mag > args_cli.contact_force_threshold, force_mag
        except Exception:
            z = torch.zeros(self.N, device=self.device)
            return z.bool(), z

    def _get_kick_direction(self, cmd, ball_pos_w: torch.Tensor):
        env_origins = getattr(self.base_env.scene, "env_origins", None)
        dest_w = cmd.target_destination_pos[:, :2]
        if env_origins is not None:
            dest_w = dest_w + env_origins[:, :2]
        direction = dest_w - ball_pos_w[:, :2]
        return direction / torch.norm(direction, dim=-1, keepdim=True).clamp(min=1e-6)

    def _select_feet(self, cmd, robot):
        ids = torch.arange(self.N, device=self.device)
        feet_pos = torch.stack(
            [robot.data.body_pos_w[:, self.left_idx], robot.data.body_pos_w[:, self.right_idx]],
            dim=1,
        )
        feet_vel = torch.stack(
            [robot.data.body_lin_vel_w[:, self.left_idx], robot.data.body_lin_vel_w[:, self.right_idx]],
            dim=1,
        )
        feet_quat = torch.stack(
            [robot.data.body_quat_w[:, self.left_idx], robot.data.body_quat_w[:, self.right_idx]],
            dim=1,
        )
        kick_side = cmd.kick_leg.clamp(min=0, max=1).long()
        support_side = 1 - kick_side
        return (
            kick_side,
            support_side,
            feet_pos[ids, kick_side],
            feet_vel[ids, kick_side],
            feet_pos[ids, support_side],
            feet_vel[ids, support_side],
            feet_quat[ids, support_side],
        )

    def _d_score(self, cmd):
        if self._disc is None or self._disc_extractor is None:
            return torch.zeros(self.N, device=self.device)
        with torch.no_grad():
            feats = self._disc_extractor.compute(self.base_env, cmd)
            return self._disc(feats)

    def step(self, rewards, dones, infos):
        self.env_step += 1
        cmd = self.base_env.command_manager.get_term("motion")
        robot = cmd.robot
        ball_pos_w = self.base_env.scene["soccer_ball"].data.root_pos_w
        ball_vel_w = self.base_env.scene["soccer_ball"].data.root_lin_vel_w
        has_contact, _ = self._get_ball_contact()

        (
            kick_side,
            support_side,
            kick_pos,
            kick_vel,
            support_pos,
            support_vel,
            support_quat,
        ) = self._select_feet(cmd, robot)

        kick_dir = self._get_kick_direction(cmd, ball_pos_w)
        side_dir = torch.stack([-kick_dir[:, 1], kick_dir[:, 0]], dim=-1)
        side_sign = torch.where(kick_side == 0, -1.0, 1.0).to(self.device)

        kick_rel_xy = kick_pos[:, :2] - ball_pos_w[:, :2]
        ball_to_kick = ball_pos_w[:, :2] - kick_pos[:, :2]
        kick_dist_xy = torch.norm(kick_rel_xy, dim=-1)
        kick_dist_3d = torch.norm(kick_pos - ball_pos_w, dim=-1)
        ball_to_kick_unit = ball_to_kick / torch.norm(ball_to_kick, dim=-1, keepdim=True).clamp(min=1e-6)
        kick_speed_xy = torch.norm(kick_vel[:, :2], dim=-1)
        closing_speed = torch.sum(kick_vel[:, :2] * ball_to_kick_unit, dim=-1)

        support_rel_xy = support_pos[:, :2] - ball_pos_w[:, :2]
        support_lat = torch.sum(support_rel_xy * side_dir, dim=-1) * side_sign
        support_long = torch.sum(support_rel_xy * kick_dir, dim=-1)
        support_yaw = _quat_to_yaw(support_quat)
        desired_yaw = torch.atan2(kick_dir[:, 1], kick_dir[:, 0])
        support_yaw_err = torch.atan2(torch.sin(support_yaw - desired_yaw), torch.cos(support_yaw - desired_yaw)).abs()
        support_speed = torch.norm(support_vel[:, :2], dim=-1)

        ball_speed = torch.norm(ball_vel_w[:, :2], dim=-1)
        self.peak_ball_speed = torch.maximum(self.peak_ball_speed, ball_speed)
        speed_mask = ball_speed > 0.5
        if torch.any(speed_mask):
            ball_dir = ball_vel_w[:, :2] / ball_speed.unsqueeze(-1).clamp(min=1e-6)
            align = torch.sum(ball_dir * kick_dir, dim=-1).clamp(-1.0, 1.0)
            self.ball_dir_align[speed_mask] = torch.maximum(self.ball_dir_align[speed_mask], align[speed_mask])

        pelvis_pos = robot.data.body_pos_w[:, self.pelvis_idx]
        pelvis_ball_dist = torch.norm(pelvis_pos[:, :2] - ball_pos_w[:, :2], dim=-1)

        # ── NoAtt running stats (updated every step) ──
        self.min_pelvis_ball_dist = torch.minimum(self.min_pelvis_ball_dist, pelvis_ball_dist)
        self.min_foot_ball_dist = torch.minimum(self.min_foot_ball_dist, kick_dist_xy)
        self.max_kick_speed = torch.maximum(self.max_kick_speed, kick_speed_xy)
        self.max_closing_speed = torch.maximum(self.max_closing_speed, closing_speed)
        near_now = pelvis_ball_dist < args_cli.attempt_near_ball_dist
        self.ever_near_ball = self.ever_near_ball | near_now
        # Track frame of closest foot-ball distance
        closer = kick_dist_xy < self.min_foot_ball_dist
        self.closest_ball_frame[closer] = self.env_step[closer]
        self.min_foot_ball_dist = torch.minimum(self.min_foot_ball_dist, kick_dist_xy)

        kf = cmd.kick_frame
        t = cmd.time_steps
        # kick_frame gate: filters false-positive attempts from early-episode locomotion
        after_attempt_time = (kf < 0) | (t >= (kf - args_cli.attempt_early_grace))

        attempt_start = (
            (~self.attempt_started)
            & (~self.contact_seen)
            & after_attempt_time
            & (pelvis_ball_dist < args_cli.attempt_near_ball_dist)
            & (kick_dist_xy < args_cli.attempt_max_foot_ball_dist)
            & (kick_pos[:, 2] >= args_cli.attempt_min_kick_height)
            & (kick_pos[:, 2] <= args_cli.attempt_max_kick_height)
            & (kick_speed_xy >= args_cli.attempt_speed_threshold)
            & (closing_speed >= args_cli.attempt_closing_speed)
        )

        if torch.any(attempt_start):
            self.attempt_started[attempt_start] = True
            self.attempt_frame[attempt_start] = self.env_step[attempt_start]
            self.attempt_kick_speed[attempt_start] = kick_speed_xy[attempt_start]
            self.attempt_closing_speed[attempt_start] = closing_speed[attempt_start]
            self.attempt_dist_xy[attempt_start] = kick_dist_xy[attempt_start]
            self.attempt_dist_3d[attempt_start] = kick_dist_3d[attempt_start]
            self.attempt_kick_height[attempt_start] = kick_pos[attempt_start, 2]
            self.attempt_support_lat[attempt_start] = support_lat[attempt_start]
            self.attempt_support_long[attempt_start] = support_long[attempt_start]
            self.attempt_support_yaw[attempt_start] = support_yaw_err[attempt_start]
            self.attempt_support_speed[attempt_start] = support_speed[attempt_start]
            self.attempt_support_height[attempt_start] = support_pos[attempt_start, 2]
            self.attempt_d_score[attempt_start] = self._d_score(cmd)[attempt_start]

        # Mark an attempt as missed once the hit window has elapsed.
        elapsed = self.env_step - self.attempt_frame
        expired = self.attempt_started & (~self.attempt_hit) & (~self.attempt_missed) & (elapsed > args_cli.attempt_window)
        self.attempt_missed[expired] = True

        new_contact = has_contact & (~self.contact_seen)
        if torch.any(new_contact):
            self.contact_seen[new_contact] = True
            self.contact_frame[new_contact] = self.env_step[new_contact]

            left_pos = robot.data.body_pos_w[new_contact, self.left_idx, :2]
            right_pos = robot.data.body_pos_w[new_contact, self.right_idx, :2]
            bp = ball_pos_w[new_contact, :2]
            left_dist = torch.norm(left_pos - bp, dim=-1)
            right_dist = torch.norm(right_pos - bp, dim=-1)
            contact_side = torch.where(left_dist <= right_dist, 0, 1).to(self.device)
            self.correct_foot[new_contact] = contact_side == kick_side[new_contact]

            self.contact_kick_speed[new_contact] = kick_speed_xy[new_contact]
            self.contact_dist_xy[new_contact] = kick_dist_xy[new_contact]
            self.contact_support_lat[new_contact] = support_lat[new_contact]
            self.contact_support_long[new_contact] = support_long[new_contact]
            self.contact_support_yaw[new_contact] = support_yaw_err[new_contact]
            self.contact_support_height[new_contact] = support_pos[new_contact, 2]
            self.contact_d_score[new_contact] = self._d_score(cmd)[new_contact]

            hit_in_window = (
                new_contact
                & self.attempt_started
                & ((self.env_step - self.attempt_frame) <= args_cli.attempt_window)
            )
            self.attempt_hit[hit_in_window] = True

            fallback = new_contact & self.attempt_missed
            self.fallback_contact[fallback] = True
            self.fallback_frame[fallback] = self.env_step[fallback]

        # Post-contact stability.
        post_contact = self.contact_seen
        if torch.any(post_contact):
            from isaaclab.utils.math import quat_apply_inverse as qai

            base_quat = robot.data.root_quat_w
            grav = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.N, 3)
            proj_grav = qai(base_quat, grav)
            tilt = 1.0 + proj_grav[:, 2]
            self.post_max_tilt[post_contact] = torch.maximum(self.post_max_tilt[post_contact], tilt[post_contact])
            ang_vel = robot.data.root_ang_vel_w
            angvel_xy = torch.sqrt(ang_vel[:, 0].square() + ang_vel[:, 1].square())
            self.post_max_angvel_xy[post_contact] = torch.maximum(self.post_max_angvel_xy[post_contact], angvel_xy[post_contact])
            self.post_frames[post_contact] += 1

        if isinstance(dones, dict):
            done_m = dones.get("terminated", torch.zeros(self.N, dtype=torch.bool, device=self.device))
            tout_m = dones.get("truncated", torch.zeros(self.N, dtype=torch.bool, device=self.device))
            reset = done_m | tout_m
        else:
            reset = dones.bool() if not isinstance(dones, bool) else torch.full((self.N,), dones, dtype=torch.bool, device=self.device)
            tout_m = infos.get("time_outs", torch.zeros(self.N, dtype=torch.bool, device=self.device)).to(self.device, torch.bool) if isinstance(infos, dict) else torch.zeros(self.N, dtype=torch.bool, device=self.device)
            done_m = reset & ~tout_m

        if torch.any(reset):
            self._record(reset, done_m, tout_m, cmd)
            reset_ids = torch.where(reset)[0]
            self._perturb_ball_position(env_ids=reset_ids)

    def _record(self, reset: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor, cmd):
        for idx in torch.where(reset)[0]:
            i = idx.item()
            mid = int(self.env_motion_idx[i].item())
            if self.episodes_done[mid] >= self.target_eps:
                continue
            steps = int(self.env_step[i].item())
            if steps < 2:
                continue

            kf = int(cmd.kick_frame[i].item())
            af = int(self.attempt_frame[i].item())
            cf = int(self.contact_frame[i].item())
            fb = int(self.fallback_frame[i].item())

            attempt = bool(self.attempt_started[i].item())
            contact = bool(self.contact_seen[i].item())
            hit = bool(self.attempt_hit[i].item())
            fallback = bool(self.fallback_contact[i].item())
            correct = bool(self.correct_foot[i].item())
            bspd = float(self.peak_ball_speed[i].item())
            dira = float(self.ball_dir_align[i].item())
            fell = bool(terminated[i].item())

            # Determine termination reason from robot state
            if fell:
                robot = cmd.robot
                pelvis_z = robot.data.body_pos_w[i, self.pelvis_idx, 2].item()
                # anchor_pos_z: pelvis too low (typically < 0.3m)
                if pelvis_z < 0.4:
                    term_reason = "anchor_pos_z"
                else:
                    # ee_body_pos is the most common other cause
                    term_reason = "ee_body_pos"
            elif bool(truncated[i].item()):
                term_reason = "time_out"
            else:
                term_reason = "none"

            if not attempt and fell:
                outcome = "fall_no_attempt"
            elif not attempt and contact:
                outcome = "contact_without_attempt"
            elif not attempt:
                outcome = "no_attempt"
            elif fallback:
                outcome = "late_fallback"
            elif self.attempt_missed[i] and not hit:
                outcome = "empty_swing"
            elif hit and not correct:
                outcome = "wrong_foot"
            elif hit and bspd < args_cli.ball_speed_success:
                outcome = "weak_hit"
            elif hit and dira < args_cli.direction_success:
                outcome = "wrong_direction"
            elif hit:
                outcome = "clean_success"
            else:
                outcome = "unknown"

            ep = {
                "motion": self.motion_names[mid],
                "steps": steps,
                "terminated": fell,
                "attempt_started": attempt,
                "attempt_hit": hit,
                "attempt_missed": bool(self.attempt_missed[i].item()),
                "fallback_contact": fallback,
                "contact_seen": contact,
                "correct_foot": correct,
                "outcome": outcome,
                "attempt_frame_vs_kf": af - kf if attempt and kf >= 0 else None,
                "contact_frame_vs_kf": cf - kf if contact and kf >= 0 else None,
                "contact_delay_from_attempt": cf - af if contact and attempt else None,
                "fallback_delay_from_attempt": fb - af if fallback and attempt else None,
                "attempt_kick_speed": round(float(self.attempt_kick_speed[i].item()), 3) if attempt else None,
                "attempt_closing_speed": round(float(self.attempt_closing_speed[i].item()), 3) if attempt else None,
                "attempt_dist_xy": round(float(self.attempt_dist_xy[i].item()), 3) if attempt else None,
                "attempt_dist_3d": round(float(self.attempt_dist_3d[i].item()), 3) if attempt else None,
                "attempt_kick_height": round(float(self.attempt_kick_height[i].item()), 3) if attempt else None,
                "attempt_support_lat": round(float(self.attempt_support_lat[i].item()), 3) if attempt else None,
                "attempt_support_long": round(float(self.attempt_support_long[i].item()), 3) if attempt else None,
                "attempt_support_yaw": round(float(self.attempt_support_yaw[i].item()), 3) if attempt else None,
                "attempt_support_speed": round(float(self.attempt_support_speed[i].item()), 3) if attempt else None,
                "attempt_support_height": round(float(self.attempt_support_height[i].item()), 3) if attempt else None,
                "attempt_d_score": round(float(self.attempt_d_score[i].item()), 3) if attempt else None,
                "contact_kick_speed": round(float(self.contact_kick_speed[i].item()), 3) if contact else None,
                "contact_dist_xy": round(float(self.contact_dist_xy[i].item()), 3) if contact else None,
                "contact_support_lat": round(float(self.contact_support_lat[i].item()), 3) if contact else None,
                "contact_support_long": round(float(self.contact_support_long[i].item()), 3) if contact else None,
                "contact_support_yaw": round(float(self.contact_support_yaw[i].item()), 3) if contact else None,
                "contact_support_height": round(float(self.contact_support_height[i].item()), 3) if contact else None,
                "contact_d_score": round(float(self.contact_d_score[i].item()), 3) if contact else None,
                "peak_ball_speed": round(bspd, 3),
                "ball_dir_align": round(dira, 3),
                "post_max_tilt": round(float(self.post_max_tilt[i].item()), 3) if contact else None,
                "post_max_angvel_xy": round(float(self.post_max_angvel_xy[i].item()), 3) if contact else None,
                "post_frames": int(self.post_frames[i].item()) if contact else None,
                # NoAtt diagnostics (always recorded)
                "min_pelvis_ball_dist": round(float(self.min_pelvis_ball_dist[i].item()), 3),
                "min_foot_ball_dist": round(float(self.min_foot_ball_dist[i].item()), 3),
                "max_kick_speed": round(float(self.max_kick_speed[i].item()), 3),
                "max_closing_speed": round(float(self.max_closing_speed[i].item()), 3),
                "ever_near_ball": bool(self.ever_near_ball[i].item()),
                "closest_ball_frame": int(self.closest_ball_frame[i].item()),
                "termination_reason": term_reason,
            }

            self.results[mid].append(ep)
            self.episodes_done[mid] += 1
            self._reset_accumulators(torch.tensor([i], device=self.device))

            for m in range(self.num_motions):
                if self.episodes_done[m] < self.target_eps:
                    self.env_motion_idx[i] = m
                    cmd.motion_idx[i] = m
                    cmd.motion_length[i] = cmd.motion.file_lengths[m]
                    cmd.time_steps[i] = 0
                    break

    def is_done(self):
        return all(self.episodes_done[m] >= self.target_eps for m in range(self.num_motions))

    def print_report(self):
        print("\n" + "=" * 136)
        print("  KICK ATTEMPT DIAGNOSTIC REPORT")
        print("=" * 136)
        print(
            f"  Attempt trigger: speed>={args_cli.attempt_speed_threshold:.1f}, "
            f"closing>={args_cli.attempt_closing_speed:.1f}, "
            f"foot-ball<={args_cli.attempt_max_foot_ball_dist:.2f}m, "
            f"hit_window={args_cli.attempt_window} frames"
        )

        header = (
            f"{'Motion':<35} {'Att%':>5} {'Clean%':>6} {'Empty%':>6} {'Late%':>5} "
            f"{'NoAtt%':>6} {'Term%':>5} | {'AΔ':>5} {'HitΔ':>5} {'CΔ':>5} | "
            f"{'ADst':>5} {'AKV':>5} {'ACls':>5} {'BSpd':>5} {'DirA':>5}"
        )
        print("\n" + header)
        print("-" * len(header))

        all_eps = []
        for mid in range(self.num_motions):
            eps = self.results[mid]
            if not eps:
                continue
            all_eps.extend(eps)
            self._print_motion_row(self.motion_names[mid], eps)

        if all_eps:
            self._print_motion_row("AGGREGATE", all_eps)

        print("=" * 136)
        print("\nOutcome meaning:")
        print("  clean_success        = first strike attempt hits within window and ball outcome is good")
        print("  empty_swing          = first strike attempt missed; no later first contact")
        print("  late_fallback        = first strike attempt missed, then a later contact moved the ball")
        print("  contact_without_attempt = ball contact happened before any detected strike attempt")

        if all_eps:
            print("\n" + "=" * 96)
            print("  OUTCOME BREAKDOWN")
            print("=" * 96)
            outcomes = [
                "clean_success",
                "late_fallback",
                "empty_swing",
                "weak_hit",
                "wrong_direction",
                "wrong_foot",
                "contact_without_attempt",
                "no_attempt",
                "fall_no_attempt",
            ]
            out_header = f"{'Motion':<35} " + " ".join(f"{o[:8]:>8}" for o in outcomes)
            print(out_header)
            print("-" * len(out_header))
            for mid in range(self.num_motions):
                eps = self.results[mid]
                if eps:
                    self._print_outcome_row(self.motion_names[mid], eps, outcomes)
            self._print_outcome_row("AGGREGATE", all_eps, outcomes)
            print("=" * 96)

        if all_eps:
            print("\n" + "=" * 120)
            print("  ATTEMPT / CONTACT GEOMETRY BY OUTCOME")
            print("=" * 120)
            geom_header = (
                f"{'Outcome':<24} {'N':>4} | {'AΔ':>5} {'ADst':>5} {'AKV':>5} {'ACls':>5} "
                f"{'sLat':>6} {'sLong':>6} {'sYaw':>6} {'sV':>6} | "
                f"{'CΔ':>5} {'CDly':>5} {'cDst':>5} {'cKV':>5} {'BSpd':>5}"
            )
            print(geom_header)
            print("-" * len(geom_header))
            for outcome in ["clean_success", "late_fallback", "empty_swing", "weak_hit", "fall_no_attempt"]:
                grp = [e for e in all_eps if e["outcome"] == outcome]
                if not grp:
                    continue
                print(
                    f"{outcome:<24} {len(grp):>4} | "
                    f"{_mean([e['attempt_frame_vs_kf'] for e in grp]):>5.0f} "
                    f"{_mean([e['attempt_dist_xy'] for e in grp]):>5.2f} "
                    f"{_mean([e['attempt_kick_speed'] for e in grp]):>5.2f} "
                    f"{_mean([e['attempt_closing_speed'] for e in grp]):>5.2f} "
                    f"{_mean([e['attempt_support_lat'] for e in grp]):>6.3f} "
                    f"{_mean([e['attempt_support_long'] for e in grp]):>6.3f} "
                    f"{_mean([e['attempt_support_yaw'] for e in grp]):>6.3f} "
                    f"{_mean([e['attempt_support_speed'] for e in grp]):>6.3f} | "
                    f"{_mean([e['contact_frame_vs_kf'] for e in grp]):>5.0f} "
                    f"{_mean([e['contact_delay_from_attempt'] for e in grp]):>5.0f} "
                    f"{_mean([e['contact_dist_xy'] for e in grp]):>5.2f} "
                    f"{_mean([e['contact_kick_speed'] for e in grp]):>5.2f} "
                    f"{_mean([e['peak_ball_speed'] for e in grp]):>5.2f}"
                )
            print("=" * 120)

            # ── NoAtt Breakdown ──────────────────────────────────────────────
            noatt_eps = [e for e in all_eps if e["outcome"] == "fall_no_attempt" or e["outcome"] == "no_attempt"]
            if noatt_eps:
                print("\n" + "=" * 120)
                print("  NO-ATTEMPT BREAKDOWN")
                print("=" * 120)

                # Categorize
                never_near = [e for e in noatt_eps if not e.get("ever_near_ball", False)]
                near_no_swing = [e for e in noatt_eps if e.get("ever_near_ball", False)
                                 and e.get("max_kick_speed", 0) < args_cli.attempt_speed_threshold]
                near_swing_no_close = [e for e in noatt_eps if e.get("ever_near_ball", False)
                                       and e.get("max_kick_speed", 0) >= args_cli.attempt_speed_threshold
                                       and e.get("max_closing_speed", 0) < args_cli.attempt_closing_speed]
                near_swing_close_blocked = [e for e in noatt_eps if e.get("ever_near_ball", False)
                                         and e.get("max_kick_speed", 0) >= args_cli.attempt_speed_threshold
                                         and e.get("max_closing_speed", 0) >= args_cli.attempt_closing_speed]

                total = len(noatt_eps)
                print(f"\n  Total no-attempt episodes: {total}")
                print(f"  {'Category':<45} {'Count':>5} {'%':>6} | {'MinPelvis':>9} {'MinFoot':>8} {'MaxKSpd':>8} {'MaxCls':>7} {'Steps':>6}")
                print(f"  {'-'*100}")

                for label, grp in [
                    ("never_reached_ball (pelvis>1.25m)", never_near),
                    ("near_but_no_swing (kick_spd<2.0)", near_no_swing),
                    ("swing_but_not_closing (cls<0.5)", near_swing_no_close),
                    ("swing_close_but_blocked (height/timing)", near_swing_close_blocked),
                ]:
                    if not grp:
                        continue
                    print(
                        f"  {label:<45} {len(grp):>5} {len(grp)/total*100:>5.0f}% | "
                        f"{_mean([e.get('min_pelvis_ball_dist', 0) for e in grp]):>9.3f} "
                        f"{_mean([e.get('min_foot_ball_dist', 0) for e in grp]):>8.3f} "
                        f"{_mean([e.get('max_kick_speed', 0) for e in grp]):>8.2f} "
                        f"{_mean([e.get('max_closing_speed', 0) for e in grp]):>7.2f} "
                        f"{_mean([e.get('steps', 0) for e in grp]):>6.0f}"
                    )

                # Termination stats
                termed = [e for e in noatt_eps if e.get("terminated", False)]
                timed_out = [e for e in noatt_eps if not e.get("terminated", False)]
                print(f"\n  Terminated (fell/constraint): {len(termed)} ({len(termed)/total*100:.0f}%)")
                print(f"  Timed out (survived full ep):  {len(timed_out)} ({len(timed_out)/total*100:.0f}%)")
                if termed:
                    print(f"  Avg steps before termination:  {_mean([e.get('steps', 0) for e in termed]):.0f}")
                # Per-reason breakdown
                reasons = {}
                for e in noatt_eps:
                    r = e.get("termination_reason", "unknown")
                    reasons[r] = reasons.get(r, 0) + 1
                if reasons:
                    print(f"  Termination reasons:")
                    for r, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
                        print(f"    {r:<25} {cnt:>4} ({cnt/total*100:.0f}%)")
                print("=" * 120)

    def _print_motion_row(self, name: str, eps: list[dict]):
        n = len(eps)
        attempted = [e for e in eps if e["attempt_started"]]
        clean = [e for e in eps if e["outcome"] == "clean_success"]
        empty = [e for e in eps if e["outcome"] == "empty_swing"]
        late = [e for e in eps if e["outcome"] == "late_fallback"]
        no_attempt = [e for e in eps if not e["attempt_started"]]
        term = [e for e in eps if e["terminated"]]
        hit_eps = [e for e in eps if e["attempt_hit"]]
        name = name[:34]
        print(
            f"{name:<35} {len(attempted)/n*100:>4.0f}% {len(clean)/n*100:>5.0f}% "
            f"{len(empty)/n*100:>5.0f}% {len(late)/n*100:>4.0f}% "
            f"{len(no_attempt)/n*100:>5.0f}% {len(term)/n*100:>4.0f}% | "
            f"{_mean([e['attempt_frame_vs_kf'] for e in attempted]):>5.0f} "
            f"{_mean([e['contact_delay_from_attempt'] for e in hit_eps]):>5.0f} "
            f"{_mean([e['contact_frame_vs_kf'] for e in eps]):>5.0f} | "
            f"{_mean([e['attempt_dist_xy'] for e in attempted]):>5.2f} "
            f"{_mean([e['attempt_kick_speed'] for e in attempted]):>5.2f} "
            f"{_mean([e['attempt_closing_speed'] for e in attempted]):>5.2f} "
            f"{_mean([e['peak_ball_speed'] for e in eps]):>5.1f} "
            f"{_mean([e['ball_dir_align'] for e in eps]):>5.2f}"
        )

    def _print_outcome_row(self, name: str, eps: list[dict], outcomes: list[str]):
        n = len(eps)
        counts = " ".join(f"{sum(1 for e in eps if e['outcome'] == o)/n*100:>7.0f}%" for o in outcomes)
        print(f"{name[:34]:<35} {counts}")

    def save_json(self, path: str):
        raw = {self.motion_names[k]: v for k, v in self.results.items()}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(raw, f, indent=2)
        print(f"[INFO] Saved to {path}")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.motion_file:
        motion_files = [args_cli.motion_file]
    elif args_cli.motion_path:
        motion_files = get_motion_files(args_cli.motion_path)
    else:
        raise ValueError("--motion_file or --motion_path required")
    env_cfg.commands.motion.motion_files = motion_files
    if hasattr(env_cfg.commands.motion, "strike_motion_files"):
        env_cfg.commands.motion.strike_motion_files = motion_files

    if not args_cli.decoder_only or args_cli.posterior_quant:
        resume = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Loading: {resume}")

    # ── Disable reference-based terminations if requested ──
    if getattr(args_cli, 'disable_ref_terminations', False):
        disabled = []
        for term_name in ["ee_body_pos", "anchor_pos_z"]:
            if hasattr(env_cfg.terminations, term_name):
                setattr(env_cfg.terminations, term_name, None)
                disabled.append(term_name)
        if disabled:
            print(f"[INFO] Disabled ref-based terminations: {', '.join(disabled)}")

    # ── Ball XY perturbation for generalization testing ──
    if getattr(args_cli, 'ball_xy_perturb', 0.0) > 0:
        env_cfg.commands.motion.ball_xy_perturbation = args_cli.ball_xy_perturb
        print(f"[INFO] Ball XY perturbation: ±{args_cli.ball_xy_perturb}m")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    if args_cli.action_cvae and args_cli.latent_v2_model:
        raise ValueError("Use only one of --action_cvae or --latent_v2_model.")
    if args_cli.decoder_only or args_cli.posterior_quant:
        # Pure decoder/prior or posterior eval: no PPO wrapper needed, just raw obs
        env = RslRlVecEnvWrapper(env)
    elif args_cli.latent_v2_model:
        env = LatentV2PPOEnvWrapper(
            env,
            latent_model_path=args_cli.latent_v2_model,
            lab_scale=args_cli.lab_scale,
            latent_clip=args_cli.latent_clip,
            lab_barrier_weight=args_cli.latent_barrier_weight,
            lab_barrier_limit=args_cli.latent_barrier_limit,
            policy_obs_mode=args_cli.policy_obs_mode,
            ppo_logit_scale=getattr(args_cli, 'ppo_logit_scale', 1.0),
            kl_categorical_weight=getattr(args_cli, 'kl_categorical_weight', 0.01),
            no_prior=getattr(args_cli, 'no_prior', False),
            residual_alpha=getattr(args_cli, 'residual_alpha', 0.0),
            prior_code_only=getattr(args_cli, 'prior_code_only', False),
            categorical_ppo=getattr(args_cli, 'categorical_ppo', False),
        )
        print(
            f"[INFO] LATENT-v2 wrapper: policy_obs={env.num_obs}, decoder_obs={env.obs_dim_latent}, "
            f"obs_mode={args_cli.policy_obs_mode}, latent_actions={env.num_actions}, "
            f"lab_scale={args_cli.lab_scale}, latent_clip={args_cli.latent_clip}, "
            f"barrier_weight={args_cli.latent_barrier_weight}, "
            f"barrier_limit={args_cli.latent_barrier_limit}"
        )
    elif args_cli.action_cvae:
        from action_cvae_latent_wrapper import ActionCVAELatentRslRlVecEnvWrapper

        env = ActionCVAELatentRslRlVecEnvWrapper(
            env,
            action_cvae_path=args_cli.action_cvae,
            latent_scale=args_cli.latent_scale,
            latent_clip=args_cli.latent_clip,
            pd_action_clip=args_cli.pd_action_clip,
            pd_residual_scale=args_cli.pd_residual_scale,
            pd_residual_joint_scope=args_cli.pd_residual_joint_scope,
            pd_residual_gate_dist=args_cli.pd_residual_gate_dist,
            pd_residual_gate_temp=args_cli.pd_residual_gate_temp,
            pd_residual_closing_threshold=args_cli.pd_residual_closing_threshold,
            pd_residual_closing_temp=args_cli.pd_residual_closing_temp,
            latent_barrier_weight=args_cli.latent_barrier_weight,
            latent_barrier_limit=args_cli.latent_barrier_limit,
        )
        print(
            f"[INFO] Latent action-CVAE wrapper: obs={env.num_obs}, latent_actions={env.num_actions}, "
            f"latent_scale={args_cli.latent_scale}, pd_residual_scale={args_cli.pd_residual_scale}, "
            f"pd_residual_joint_scope={args_cli.pd_residual_joint_scope}, "
            f"latent_barrier_weight={args_cli.latent_barrier_weight}, "
            f"latent_barrier_limit={args_cli.latent_barrier_limit}"
        )
    else:
        env = RslRlVecEnvWrapper(env)

    if args_cli.decoder_only or args_cli.posterior_quant:
        if not args_cli.latent_v2_model:
            raise ValueError("--latent_v2_model is required when using --decoder_only or --posterior_quant")
        model, ckpt = _load_latent_v2_model(args_cli.latent_v2_model, env.unwrapped.device)
        code_hold = ckpt.get("code_hold", 1)
        held_zq = torch.zeros(env.num_envs, model.z_dim, device=env.unwrapped.device)
        held_code = torch.zeros(env.num_envs, dtype=torch.long, device=env.unwrapped.device)
        hold_ctr = torch.zeros(env.num_envs, dtype=torch.long, device=env.unwrapped.device)
        use_task_features = ("task_features" in ckpt.get("decoder_obs_mode", "full"))
        if args_cli.posterior_quant:
            # Also need teacher policy for posterior path
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(resume)
            teacher_policy = runner.get_inference_policy(device=env.unwrapped.device)
            print(f"[INFO] Evaluating posterior quantized (code_hold={code_hold}): teacher → encoder → VQ → decoder")
        else:
            print(f"[INFO] Evaluating pure decoder/prior (code_hold={code_hold})")
    else:
        if args_cli.categorical_ppo:
            from categorical_actor_critic import CategoricalActorCriticRecurrent
            policy = CategoricalActorCriticRecurrent(
                num_actor_obs=env.num_obs,
                num_critic_obs=env.num_obs,
                num_actions=1,
                num_codes=env.num_codes,
                ppo_logit_scale=args_cli.ppo_logit_scale,
                actor_hidden_dims=list(agent_cfg.policy.actor_hidden_dims),
                critic_hidden_dims=list(agent_cfg.policy.critic_hidden_dims),
                activation=agent_cfg.policy.activation,
                rnn_type=getattr(agent_cfg.policy, 'rnn_type', 'lstm'),
                rnn_hidden_dim=getattr(agent_cfg.policy, 'rnn_hidden_size', 128),
                rnn_num_layers=getattr(agent_cfg.policy, 'rnn_num_layers', 2),
                init_temperature=1.0,
            )
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.alg.policy = policy.to(agent_cfg.device)
            print(f"[INFO] Categorical PPO eval: replaced policy with CategoricalActorCriticRecurrent")
        else:
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            
        runner.load(resume)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

    base_env = env.unwrapped
    cmd = base_env.command_manager.get_term("motion")
    evaluator = KickAttemptEvaluator(
        env,
        cmd.motion.num_files,
        cmd.motion.motion_name,
        args_cli.eval_episodes,
        base_env.device,
        ball_x_offset=args_cli.ball_x_offset,
        ball_y_offset=args_cli.ball_y_offset,
        ball_xy_perturb=args_cli.ball_xy_perturb,
    )
    evaluator.assign_motions_round_robin()
    evaluator._perturb_ball_position()
    if args_cli.action_cvae and hasattr(env, "v10_builder"):
        # The evaluator manually overrides command.motion_idx after wrapper
        # construction.  Refresh the V10 builder so the latent policy sees
        # segment bounds/history that match the assigned motions.
        env.v10_builder.init_segment_bounds(cmd)
        env.v10_builder.reset(torch.arange(env.num_envs, device=base_env.device))

    obs, _ = env.get_observations()
    step = 0
    max_steps = args_cli.eval_episodes * 500 * cmd.motion.num_files
    while simulation_app.is_running() and not evaluator.is_done() and step < max_steps:
        with torch.inference_mode():
            if args_cli.decoder_only or args_cli.posterior_quant:
                obs_v3 = obs  # obs from RslRlVecEnvWrapper is raw obs_v3
                if use_task_features:
                    from compute_task_features import compute_ball_foot_relation
                    tf = compute_ball_foot_relation(base_env)
                else:
                    tf = None

                if args_cli.posterior_quant:
                    # Get teacher action, then encode → quantize → decode
                    teacher_action = teacher_policy(obs_v3)
                    dec_obs = model.select_decoder_obs(obs_v3, task_features=tf)
                    z_e = model.encoder(dec_obs, teacher_action)
                    needs_update = (hold_ctr % code_hold == 0)
                    if needs_update.any():
                        z_q, new_code, _ = model.codebook.quantize(z_e)
                        held_code[needs_update] = new_code[needs_update]
                        held_zq[needs_update] = z_q[needs_update]
                    actions = model.decoder(dec_obs, held_zq)
                elif model.prior_type == "vq":
                    dec_obs = model.select_decoder_obs(obs_v3, task_features=tf)
                    needs_update = (hold_ctr % code_hold == 0)
                    if needs_update.any():
                        logits = model.prior(dec_obs)
                        new_code = logits.argmax(dim=-1)
                        new_zq = model.codebook.lookup(new_code)
                        held_code[needs_update] = new_code[needs_update]
                        held_zq[needs_update] = new_zq[needs_update]
                    actions = model.decoder(dec_obs, held_zq)
                else:
                    actions = model.act_prior_mean(obs_v3, task_features=tf)
            else:
                actions = policy(obs)
        # Keep env.step outside inference_mode: latent wrappers update history
        # buffers in step(), and those buffers must remain normal tensors so
        # reset-time in-place writes are legal.
        actions = actions.clone()
        obs, rew, dones, infos = env.step(actions)
        if args_cli.decoder_only or args_cli.posterior_quant:
            hold_ctr += 1
            if isinstance(dones, dict):
                reset_mask = dones.get("terminated", torch.zeros(env.num_envs, dtype=torch.bool, device=env.unwrapped.device)) | dones.get("truncated", torch.zeros(env.num_envs, dtype=torch.bool, device=env.unwrapped.device))
            else:
                reset_mask = dones.bool()
            hold_ctr[reset_mask] = 0

        evaluator.step(rew, dones, infos)
        if args_cli.action_cvae and torch.as_tensor(dones, device=base_env.device).bool().any():
            # evaluator.step may assign the reset environments to a new motion
            # after env.step already computed the next observation.  Recompute
            # latent-policy observations from the updated command state.
            env.v10_builder.init_segment_bounds(cmd)
            reset_ids = torch.as_tensor(dones, device=base_env.device).bool().nonzero(as_tuple=True)[0].clone()
            env.v10_builder.reset(reset_ids)
            obs, _ = env.get_observations()
        step += 1
        if step % 500 == 0:
            done_str = ", ".join(
                f"{evaluator.motion_names[m]}={evaluator.episodes_done[m]}"
                for m in range(evaluator.num_motions)
            )
            print(f"[EVAL] Step {step} | {done_str}")

    evaluator.print_report()
    output_json = args_cli.output_json
    if output_json is None:
        if args_cli.decoder_only or args_cli.posterior_quant:
            out_dir = os.path.join(os.path.dirname(args_cli.latent_v2_model), "eval")
            fname = "kick_attempt_diagnostic_posterior_quant.json" if args_cli.posterior_quant else "kick_attempt_diagnostic.json"
        else:
            out_dir = os.path.join(os.path.dirname(resume), "eval")
            fname = "kick_attempt_diagnostic.json"
        output_json = os.path.join(out_dir, fname)
    evaluator.save_json(output_json)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
