"""Stage 3: High-level latent PPO with LAB over frozen LATENT-style decoder.

Supports two modes based on the latent model's prior_type:

1. Continuous prior (prior_type="mlp"):
   PPO outputs u_t (z_dim=16), LAB constrains z around prior mean.

2. VQ prior (prior_type="vq", Option B categorical):
   PPO outputs logit residuals (K=16), added to prior logits.
   Code selected via categorical sampling with code_hold=2.

Usage (VQ categorical):
    CUDA_VISIBLE_DEVICES=1 python scripts/rsl_rl/train_latent_v2_ppo.py \
        --task Anchor-CG-Kick-G1-Soccer-RNN-v0 \
        --motion_path motions/Video_hmr4d_seed \
        --latent_model models/latent_v2/online_distill_vq_k16_hold2_seq.pt \
        --policy_obs_mode task_features \
        --zero_tracking_rewards \
        --ppo_logit_scale 1.0 \
        --kl_categorical_weight 0.01 \
        --init_noise_std 0.1 \
        --num_envs 4096 \
        --max_iterations 6000 \
        --run_name vq_ppo_cat \
        --headless

Usage (continuous LAB):
    CUDA_VISIBLE_DEVICES=1 python scripts/rsl_rl/train_latent_v2_ppo.py \
        --task Anchor-CG-Kick-G1-Soccer-RNN-v0 \
        --motion_path motions/Video_hmr4d_seed \
        --latent_model models/latent_v2/online_distill.pt \
        --lab_scale 2.0 \
        --num_envs 4096 \
        --max_iterations 6000 \
        --run_name latent_v2_ppo \
        --headless
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Stage 3: Latent PPO with LAB.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="Anchor-CG-Kick-G1-Soccer-RNN-v0")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=6000)
parser.add_argument("--motion_path", type=str, required=True)

# Latent model
parser.add_argument("--latent_model", type=str, required=True,
                    help="Path to Stage 2B online distillation checkpoint.")
parser.add_argument("--lab_scale", type=float, default=2.0,
                    help="LAB lambda: exploration range = lambda * sigma_p around prior mean.")
parser.add_argument("--latent_clip", type=float, default=5.0,
                    help="Clip raw PPO output before tanh.")
parser.add_argument("--lab_barrier_weight", type=float, default=0.0,
                    help="Penalty weight for Mahalanobis distance exceeding lab_barrier_limit.")
parser.add_argument("--lab_barrier_limit", type=float, default=2.5,
                    help="Mahalanobis distance threshold for barrier penalty.")
parser.add_argument("--init_noise_std", type=float, default=0.25,
                    help="Initial PPO action noise std in latent space.")
parser.add_argument("--entropy_coef", type=float, default=0.001,
                    help="PPO entropy coefficient.")
parser.add_argument("--learning_rate", type=float, default=3e-4,
                    help="PPO learning rate.")
parser.add_argument("--desired_kl", type=float, default=0.01,
                    help="Adaptive scheduler target KL.")
parser.add_argument("--ppo_logit_scale", type=float, default=1.0,
                    help="Scale for PPO logit residuals added to prior logits (VQ mode).")
parser.add_argument("--kl_categorical_weight", type=float, default=0.01,
                    help="Weight for KL(combined, prior) categorical regularization (VQ mode).")
parser.add_argument("--policy_obs_mode", type=str, default="full", choices=("full", "task", "task_features"),
                    help="High-level PPO observation: full=env obs_v3, task=remove motion-reference, task_features=proprio+ball-foot-rel.")
parser.add_argument("--zero_tracking_rewards", action="store_true", default=False,
                    help="Zero out all motion-tracking reward weights (body_pos/ori, foot_pos, vel). "
                         "Use with task/task_features mode where decoder cannot see motion reference.")
parser.add_argument("--no_prior", action="store_true", default=False,
                    help="Ablation: PPO selects codes directly without Prior bias. "
                         "combined_logits = ppo_logits (no prior added, no KL penalty).")
parser.add_argument("--warmstart_actor", type=str, default=None,
                    help="Path to pre-trained actor weights (from train_posterior_warmstart.py). "
                         "Loads actor + actor RNN into PPO, leaves critic random.")
parser.add_argument("--residual_alpha", type=float, default=0.0,
                    help="Route beta: continuous residual scale. 0.0=pure VQ (default). "
                         "z = z_q + alpha*tanh(residual). Start with 0.1.")
parser.add_argument("--residual_l2_penalty", type=float, default=0.01,
                    help="L2 penalty weight on continuous residual to keep it small.")
parser.add_argument("--prior_code_only", action="store_true", default=False,
                    help="Residual-only PPO: code from Prior (frozen), PPO only outputs residual. "
                         "Requires --residual_alpha > 0. Prevents PPO from overriding code timing.")
parser.add_argument("--categorical_ppo", action="store_true", default=False,
                    help="True categorical PPO: actor outputs K logits, samples code via "
                         "Categorical distribution. Correct logprob/entropy/ratio in discrete space.")
parser.add_argument("--no_attempt_penalty", type=float, default=None,
                    help="Override attempt_no_attempt reward weight (e.g. -2.0, -5.0). "
                         "Default None = use env config value.")
parser.add_argument("--disable_ref_terminations", action="store_true", default=False,
                    help="Disable reference-based terminations (ee_body_pos, anchor_pos_z) "
                         "that conflict with ref-free Stage C student.")
parser.add_argument("--posterior_aux_weight", type=float, default=0.0,
                    help="Weight for posterior-code auxiliary CE loss. Anneals from this value to 0.01 "
                         "over training. Requires teacher policy for posterior code computation.")

import cli_args  # isort: skip
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True
if getattr(args_cli, "headless", False) and "DISPLAY" in os.environ:
    os.environ.pop("DISPLAY", None)
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import soccer.tasks  # noqa: F401
from soccer.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner
from latent_v2_models import LatentActionModel
from compute_task_features import compute_ball_foot_relation, TASK_FEATURES_DIM
from categorical_actor_critic import CategoricalActorCriticRecurrent


def get_motion_files(motion_path: str) -> list[str]:
    if os.path.isfile(motion_path):
        return [motion_path]
    if os.path.isdir(motion_path):
        files = sorted(glob.glob(os.path.join(motion_path, "*.npz")))
        if not files:
            raise ValueError(f"No .npz files found in directory: {motion_path}")
        return files
    raise ValueError(f"Invalid path: {motion_path}")


def load_latent_model(path: str, device: str) -> tuple[LatentActionModel, dict]:
    """Load frozen latent model from Stage 2B checkpoint."""
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
    prior_type = ckpt.get("prior_type", "mlp")
    code_hold = ckpt.get("code_hold", 1)
    print(f"[INFO] Loaded latent model: decoder_obs_mode={decoder_obs_mode}, "
          f"decoder_obs_dim={model.decoder_obs_dim}, prior_type={prior_type}, "
          f"code_hold={code_hold}")
    return model, ckpt


def latent_v2_policy_obs_dim(decoder_obs_dim: int, mode: str) -> int:
    if mode == "full":
        return decoder_obs_dim
    if mode == "task":
        if decoder_obs_dim < 160:
            raise ValueError(f"task policy obs expects obs_v3 >=160D, got {decoder_obs_dim}")
        return 3 + (decoder_obs_dim - 64)
    if mode == "task_features":
        # proprio (99D) + ball_foot_relation (22D)
        return 3 + (decoder_obs_dim - 64) + TASK_FEATURES_DIM  # 121
    raise ValueError(f"Unknown policy_obs_mode={mode!r}")


def select_latent_v2_policy_obs(
    obs_v3: torch.Tensor, mode: str, task_features: torch.Tensor | None = None
) -> torch.Tensor:
    """Select high-level policy obs while keeping full obs_v3 available to the decoder.

    obs_v3 layout:
      0:58   motion reference command      (removed in task/task_features mode)
      58:61  projected gravity
      61:64  motion reference angular vel  (removed in task/task_features mode)
      64:    proprioception + previous action + ball/target
    """
    if mode == "full":
        return obs_v3
    if mode == "task":
        if obs_v3.shape[-1] < 160:
            raise ValueError(f"task policy obs expects obs_v3 >=160D, got {obs_v3.shape[-1]}")
        return torch.cat((obs_v3[:, 58:61], obs_v3[:, 64:]), dim=-1)
    if mode == "task_features":
        if task_features is None:
            raise ValueError("task_features must be provided for policy_obs_mode='task_features'")
        proprio = torch.cat((obs_v3[:, 58:61], obs_v3[:, 64:]), dim=-1)  # 99D
        return torch.cat((proprio, task_features), dim=-1)  # 121D
    raise ValueError(f"Unknown policy_obs_mode={mode!r}")


class LatentPPOEnvWrapper(RslRlVecEnvWrapper):
    """RSL-RL wrapper: PPO outputs z_dim latent actions, decoded to 29D joint actions.

    LAB constraint: z = mu_p + lab_scale * sigma_p * tanh(u)
    where u is the raw PPO output.
    """

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
        residual_l2_penalty: float = 0.01,
        prior_code_only: bool = False,
        categorical_ppo: bool = False,
    ):
        super().__init__(env)
        self.lab_scale = lab_scale
        self.latent_clip = latent_clip
        self.lab_barrier_weight = lab_barrier_weight
        self.lab_barrier_limit = lab_barrier_limit
        self.policy_obs_mode = policy_obs_mode
        self.ppo_logit_scale = ppo_logit_scale
        self.kl_categorical_weight = kl_categorical_weight
        self.no_prior = no_prior
        self.residual_alpha = residual_alpha
        self.residual_l2_penalty = residual_l2_penalty
        self.prior_code_only = prior_code_only
        self.categorical_ppo = categorical_ppo

        # Load frozen latent model
        self.latent_model, self.latent_ckpt = load_latent_model(latent_model_path, self.device)
        self.z_dim = int(self.latent_ckpt["z_dim"])
        self.obs_dim_latent = int(self.latent_ckpt["obs_dim"])  # 160 (obs_v3)
        self.policy_obs_dim = latent_v2_policy_obs_dim(self.obs_dim_latent, self.policy_obs_mode)
        self.is_vq = (self.latent_ckpt.get("prior_type", "mlp") == "vq")
        self.num_codes = int(self.latent_ckpt.get("num_codes", 16))
        self.code_hold = int(self.latent_ckpt.get("code_hold", 1))
        self.use_residual = (self.residual_alpha > 0.0)

        # Override action space: PPO outputs z_dim or K actions (+z_dim if residual)
        if self.categorical_ppo:
            # Categorical PPO: actor outputs 1 code index
            self.num_actions = 1
        elif self.is_vq:
            if self.use_residual and self.prior_code_only:
                # Residual-only: PPO outputs z_dim residual, code from Prior
                self.num_actions = self.z_dim
            elif self.use_residual:
                # Route beta: PPO outputs K code logits + z_dim residual
                self.num_actions = self.num_codes + self.z_dim
            else:
                # Pure VQ: PPO outputs K logit residuals
                self.num_actions = self.num_codes
        else:
            self.num_actions = self.z_dim
        if self.categorical_ppo:
            # Categorical PPO: obs = [features | prior_logits]
            self.num_obs = self.policy_obs_dim + self.num_codes
        else:
            self.num_obs = self.policy_obs_dim
        if self.policy_obs_mode in ("task", "task_features"):
            self.num_privileged_obs = self.num_obs
        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.latent_clip, high=self.latent_clip,
            shape=(self.num_actions,), dtype=float,
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space, self.num_envs
        )

        # Cache for current obs_v3 tensor (set in get_observations / reset)
        self._cached_obs_v3: torch.Tensor | None = None
        # Cache for current task features (computed in get_observations, used in _decode_latent)
        self._cached_task_features: torch.Tensor | None = None
        self._use_task_features = (self.policy_obs_mode == "task_features")

        # Access to unwrapped env for task feature computation
        self.base_env = self.env.unwrapped

        # Tracking
        self._last_maha = torch.zeros(self.num_envs, device=self.device)
        self._last_barrier_penalty = torch.zeros(self.num_envs, device=self.device)

        # VQ code-hold state
        if self.is_vq:
            self._held_code = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._held_zq = torch.zeros(self.num_envs, self.z_dim, device=self.device)
            self._hold_ctr = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._last_kl_cat = torch.zeros(self.num_envs, device=self.device)

    @torch.no_grad()
    def _decode_vq(self, obs_v3: torch.Tensor, ppo_raw: torch.Tensor) -> torch.Tensor:
        """VQ categorical decode with optional continuous residual.

        Pure VQ:   ppo_raw = [B, K] code logits
        Route beta: ppo_raw = [B, K+z_dim] = code_logits | residual

        z = z_q + alpha * tanh(residual)   (route beta)
        z = z_q                            (pure VQ)
        action = decoder(obs, z)
        """
        import torch.nn.functional as F
        import torch.distributions as D

        tf = self._cached_task_features if self._use_task_features else None
        dec_obs = self.latent_model.select_decoder_obs(obs_v3, task_features=tf)

        # Split PPO output into code logits and optional residual
        if self.use_residual and self.prior_code_only:
            # Residual-only mode: all PPO output is residual, code from Prior
            ppo_logits = None
            residual_raw = ppo_raw                        # [B, z_dim]
        elif self.use_residual:
            ppo_logits = ppo_raw[:, :self.num_codes]      # [B, K]
            residual_raw = ppo_raw[:, self.num_codes:]    # [B, z_dim]
        else:
            ppo_logits = ppo_raw                          # [B, K]
            residual_raw = None

        # Code selection
        prior_logits = self.latent_model.prior(dec_obs)   # [B, K]
        if self.prior_code_only or self.no_prior:
            # Prior-only or no-prior: don't mix PPO logits into code selection
            combined = prior_logits if self.prior_code_only else self.ppo_logit_scale * ppo_logits
            if self.no_prior and not self.prior_code_only:
                prior_logits = torch.zeros_like(combined)
        else:
            combined = prior_logits + self.ppo_logit_scale * ppo_logits  # [B, K]

        # Code-hold: only re-select code at hold boundaries
        needs_update = (self._hold_ctr % self.code_hold == 0)
        if needs_update.any():
            dist = D.Categorical(logits=combined)
            new_code = dist.sample()  # [B]
            new_zq = self.latent_model.codebook.lookup(new_code)
            self._held_code[needs_update] = new_code[needs_update]
            self._held_zq[needs_update] = new_zq[needs_update]

        # Apply continuous residual (route beta)
        if self.use_residual and residual_raw is not None:
            z = self._held_zq + self.residual_alpha * torch.tanh(residual_raw)
            self._last_residual_norm = residual_raw.norm(dim=-1)  # for logging
        else:
            z = self._held_zq

        # Decode
        action = self.latent_model.decoder(dec_obs, z)

        # KL(combined || prior) for logging/penalty
        combined_p = F.softmax(combined, dim=-1)
        prior_p = F.softmax(prior_logits, dim=-1)
        kl = (combined_p * (combined_p.log() - prior_p.log())).sum(dim=-1)  # [B]
        self._last_kl_cat = kl

        # Barrier penalty: KL + residual L2
        penalty = torch.zeros_like(kl)
        if not self.no_prior:
            penalty = penalty + self.kl_categorical_weight * kl
        if self.use_residual:
            penalty = penalty + self.residual_l2_penalty * self._last_residual_norm
        self._last_barrier_penalty = penalty

        return action

    @torch.no_grad()
    def _decode_vq_categorical(self, obs_v3: torch.Tensor, code_action: torch.Tensor) -> torch.Tensor:
        """Categorical PPO decode: code index → z_q → joint action.

        The actor already combined prior + PPO logits and sampled a code.
        We just look up z_q and decode.

        Args:
            code_action: [B, 1] float tensor containing code indices.
        """
        tf = self._cached_task_features if self._use_task_features else None
        dec_obs = self.latent_model.select_decoder_obs(obs_v3, task_features=tf)

        code = code_action.squeeze(-1).long()  # [B]

        # Code-hold: only update at hold boundaries
        needs_update = (self._hold_ctr % self.code_hold == 0)
        if needs_update.any():
            new_zq = self.latent_model.codebook.lookup(code)
            self._held_code[needs_update] = code[needs_update]
            self._held_zq[needs_update] = new_zq[needs_update]

        # Decode
        action = self.latent_model.decoder(dec_obs, self._held_zq)

        # No KL penalty in categorical mode (handled by PPO entropy)
        self._last_kl_cat = torch.zeros(self.num_envs, device=self.device)
        self._last_barrier_penalty = torch.zeros(self.num_envs, device=self.device)

        return action


    @torch.no_grad()
    def _decode_latent(self, obs_v3: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """LAB decode: u (PPO output) -> z -> joint action.

        LATENT Eq. 4: z = mu_p + lambda * sigma_p * tanh(u)
        """
        u_clipped = u.clamp(-self.latent_clip, self.latent_clip)

        # Slice obs for decoder/prior (handles task_features mode)
        tf = self._cached_task_features if self._use_task_features else None
        dec_obs = self.latent_model.select_decoder_obs(obs_v3, task_features=tf)

        # Get prior statistics
        p_mu, p_logvar = self.latent_model.prior(dec_obs)
        p_std = torch.exp(0.5 * p_logvar)

        # LAB: constrain z to be within lab_scale * sigma of prior mean
        z = p_mu + self.lab_scale * p_std * torch.tanh(u_clipped)

        # Decode to joint action
        action = self.latent_model.decoder(dec_obs, z)

        # Compute Mahalanobis distance for logging/barrier
        self._last_maha = torch.norm((z - p_mu) / p_std.clamp(min=1e-6), dim=-1)
        self._last_barrier_penalty = torch.relu(
            self._last_maha - self.lab_barrier_limit
        ).pow(2)

        return action

    def _compute_task_features(self) -> torch.Tensor | None:
        """Compute and cache task features from env state."""
        if not self._use_task_features:
            return None
        self._cached_task_features = compute_ball_foot_relation(self.base_env)
        return self._cached_task_features

    def _select_policy_obs(self, obs_v3: torch.Tensor) -> torch.Tensor:
        tf = self._cached_task_features if self._use_task_features else None
        policy_obs = select_latent_v2_policy_obs(obs_v3, self.policy_obs_mode, task_features=tf)
        if self.categorical_ppo:
            # Append prior logits so CategoricalActorCritic can combine them
            dec_obs = self.latent_model.select_decoder_obs(obs_v3, task_features=tf)
            prior_logits = self.latent_model.prior(dec_obs)  # [B, K]
            policy_obs = torch.cat([policy_obs, prior_logits], dim=-1)
        return policy_obs

    def _policy_extras(self, extras: dict, policy_obs: torch.Tensor) -> dict:
        if self.policy_obs_mode not in ("task", "task_features"):
            return extras
        extras = dict(extras)
        extras["observations"] = {"policy": policy_obs, "critic": policy_obs}
        return extras

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        obs, extras = super().get_observations()
        self._cached_obs_v3 = obs.clone()
        self._compute_task_features()
        policy_obs = self._select_policy_obs(obs)
        return policy_obs, self._policy_extras(extras, policy_obs)

    def reset(self) -> tuple[torch.Tensor, dict]:
        obs, extras = super().reset()
        self._cached_obs_v3 = obs.clone()
        self._compute_task_features()
        policy_obs = self._select_policy_obs(obs)
        return policy_obs, self._policy_extras(extras, policy_obs)

    def step(self, latent_actions: torch.Tensor):
        """Step env: decode latent -> joint action -> env.step()."""
        # Use cached obs_v3 from previous get_observations/step
        obs_v3 = self._cached_obs_v3
        if obs_v3 is None:
            # First step fallback: get obs from parent
            obs_v3, _ = super().get_observations()
            self._cached_obs_v3 = obs_v3.clone()

        # Decode latent to joint action
        if self.categorical_ppo:
            joint_action = self._decode_vq_categorical(obs_v3, latent_actions)
        elif self.is_vq:
            joint_action = self._decode_vq(obs_v3, latent_actions)
        else:
            joint_action = self._decode_latent(obs_v3, latent_actions)

        # Signal to AttemptEventTracker that a new sim step is starting.
        # The tracker's reward functions check this flag and step exactly once.
        self.base_env._attempt_needs_step = True

        # Step the underlying env with joint action via PARENT's step
        # This handles obs_dict → obs conversion properly
        obs, rew, dones, extras = super().step(joint_action)

        # Apply barrier/KL penalty to reward
        if self.is_vq:
            penalty = self._last_barrier_penalty  # already scaled by kl_categorical_weight
            while penalty.dim() < rew.dim():
                penalty = penalty.unsqueeze(-1)
            rew = rew - penalty
        elif self.lab_barrier_weight > 0.0:
            barrier = self.lab_barrier_weight * self._last_barrier_penalty
            while barrier.dim() < rew.dim():
                barrier = barrier.unsqueeze(-1)
            rew = rew - barrier

        # VQ hold counter management
        if self.is_vq:
            self._hold_ctr += 1
            if isinstance(dones, dict):
                reset_mask = dones.get("terminated", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)) | \
                             dones.get("truncated", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
            else:
                reset_mask = dones.bool()
            self._hold_ctr[reset_mask] = 0

        # Cache obs for next step's decoding
        self._cached_obs_v3 = obs.clone()
        self._compute_task_features()
        policy_obs = self._select_policy_obs(obs)
        extras = self._policy_extras(extras, policy_obs)

        extras.setdefault("log", {})
        if self.is_vq:
            extras["log"]["latent/kl_categorical"] = self._last_kl_cat.mean()
            extras["log"]["latent/kl_penalty"] = self._last_barrier_penalty.mean()
            extras["log"]["latent/held_code_mean"] = self._held_code.float().mean()
            # Per-code histogram: fraction of envs holding each code
            for c in range(self.num_codes):
                extras["log"][f"latent/code_{c}_frac"] = (self._held_code == c).float().mean()
        else:
            extras["log"]["latent/maha_dist"] = self._last_maha.mean()
            extras["log"]["latent/barrier_penalty"] = self._last_barrier_penalty.mean()

        return policy_obs, rew, dones, extras


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name

    # PPO hyperparams tuned for latent space
    agent_cfg.empirical_normalization = True
    agent_cfg.policy.init_noise_std = args_cli.init_noise_std
    agent_cfg.algorithm.entropy_coef = args_cli.entropy_coef
    agent_cfg.algorithm.learning_rate = args_cli.learning_rate
    agent_cfg.algorithm.desired_kl = args_cli.desired_kl
    if args_cli.categorical_ppo:
        # Categorical PPO: use fixed schedule (Gaussian KL adaptive is invalid)
        agent_cfg.algorithm.schedule = "fixed"
        agent_cfg.algorithm.entropy_coef = 0.01  # higher entropy for categorical exploration
    else:
        agent_cfg.algorithm.schedule = "adaptive"

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.commands.motion.motion_files = get_motion_files(args_cli.motion_path)

    # ── Zero out tracking rewards if requested ─────────────────────────────
    if args_cli.zero_tracking_rewards:
        tracking_terms = [
            "motion_body_pos", "motion_body_ori", "motion_foot_pos",
            "motion_body_lin_vel", "motion_body_ang_vel",
            "motion_global_anchor_pos", "motion_global_anchor_ori",
        ]
        zeroed = []
        for term_name in tracking_terms:
            term = getattr(env_cfg.rewards, term_name, None)
            if term is not None and hasattr(term, "weight"):
                old_w = term.weight
                term.weight = 0.0
                zeroed.append(f"{term_name}: {old_w} → 0.0")
        print(f"[INFO] Zeroed {len(zeroed)} tracking reward terms:")
        for z in zeroed:
            print(f"  {z}")

    # ── Override no_attempt penalty if requested ───────────────────────────
    if args_cli.no_attempt_penalty is not None:
        term = getattr(env_cfg.rewards, "attempt_no_attempt", None)
        if term is not None and hasattr(term, "weight"):
            old_w = term.weight
            term.weight = args_cli.no_attempt_penalty
            print(f"[INFO] attempt_no_attempt penalty: {old_w} → {args_cli.no_attempt_penalty}")

    # ── Disable reference-based terminations if requested ─────────────────
    if args_cli.disable_ref_terminations:
        disabled = []
        for term_name in ["ee_body_pos", "anchor_pos_z"]:
            if hasattr(env_cfg.terminations, term_name):
                setattr(env_cfg.terminations, term_name, None)
                disabled.append(term_name)
        if disabled:
            print(f"[INFO] Disabled ref-based terminations: {', '.join(disabled)}")

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    env = LatentPPOEnvWrapper(
        env,
        latent_model_path=args_cli.latent_model,
        lab_scale=args_cli.lab_scale,
        latent_clip=args_cli.latent_clip,
        lab_barrier_weight=args_cli.lab_barrier_weight,
        lab_barrier_limit=args_cli.lab_barrier_limit,
        policy_obs_mode=args_cli.policy_obs_mode,
        ppo_logit_scale=args_cli.ppo_logit_scale,
        kl_categorical_weight=args_cli.kl_categorical_weight,
        no_prior=args_cli.no_prior,
        residual_alpha=args_cli.residual_alpha,
        residual_l2_penalty=args_cli.residual_l2_penalty,
        prior_code_only=args_cli.prior_code_only,
        categorical_ppo=args_cli.categorical_ppo,
    )
    if env.is_vq:
        print(
            f"[INFO] LatentPPO VQ wrapper: policy_obs={env.num_obs}, decoder_obs={env.obs_dim_latent}, "
            f"obs_mode={args_cli.policy_obs_mode}, num_codes={env.num_codes}, code_hold={env.code_hold}, "
            f"ppo_logit_scale={args_cli.ppo_logit_scale}, kl_cat_weight={args_cli.kl_categorical_weight}, "
            f"no_prior={args_cli.no_prior}, residual_alpha={args_cli.residual_alpha}, "
            f"num_actions={env.num_actions}"
        )
    else:
        print(
            f"[INFO] LatentPPO wrapper: policy_obs={env.num_obs}, decoder_obs={env.obs_dim_latent}, "
            f"obs_mode={args_cli.policy_obs_mode}, latent_actions={env.num_actions} (z_dim), "
            f"lab_scale={args_cli.lab_scale}, barrier_weight={args_cli.lab_barrier_weight}"
        )

    # ── Create runner ──────────────────────────────────────────────────────
    if args_cli.categorical_ppo:
        # Create CategoricalActorCriticRecurrent policy manually
        policy = CategoricalActorCriticRecurrent(
            num_actor_obs=env.num_obs,
            num_critic_obs=env.num_obs,
            num_actions=1,  # ignored internally
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
        # Monkey-patch: replace the runner's default policy creation
        # We do this by creating the runner first, then swapping the policy
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=None)
        # Replace the policy in the algorithm
        runner.alg.policy = policy.to(agent_cfg.device)
        runner.alg.optimizer = torch.optim.Adam(policy.parameters(), lr=args_cli.learning_rate)
        print(f"[INFO] Categorical PPO: replaced policy with CategoricalActorCriticRecurrent")
    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=None)
    runner.add_git_repo_to_log(__file__)

    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Resuming PPO from: {resume_path}")
        runner.load(resume_path)
    elif args_cli.warmstart_actor:
        # Path B: load pre-trained actor weights (actor RNN + actor MLP only)
        # Critic, optimizer, log_std all stay at PPO default initialization.
        print(f"[INFO] Warm-starting actor from: {args_cli.warmstart_actor}")
        warmstart = torch.load(args_cli.warmstart_actor, map_location=agent_cfg.device, weights_only=False)

        # Merge actor RNN + actor MLP state dicts
        actor_sd = {}
        if "actor_rnn_state_dict" in warmstart:
            actor_sd.update(warmstart["actor_rnn_state_dict"])  # memory_a.rnn.*
        if "actor_mlp_state_dict" in warmstart:
            actor_sd.update(warmstart["actor_mlp_state_dict"])  # actor.*

        if not actor_sd:
            raise ValueError(f"No actor_rnn/actor_mlp state dicts found in {args_cli.warmstart_actor}")

        # Load actor weights into the full policy model (strict=False skips critic etc)
        policy = runner.alg.policy
        full_sd = policy.state_dict()
        loaded_keys = []
        for k, v in actor_sd.items():
            if k in full_sd:
                full_sd[k] = v
                loaded_keys.append(k)
            else:
                print(f"[WARN] Warm-start key not found in policy: {k}")
        policy.load_state_dict(full_sd)
        print(f"[INFO] Warm-start loaded {len(loaded_keys)}/{len(actor_sd)} actor keys")
        print(f"[INFO] Warm-start meta: boundary_acc={warmstart.get('best_boundary_acc', '?')}, "
              f"overall_acc={warmstart.get('best_overall_acc', '?')}, "
              f"code_hold={warmstart.get('code_hold', '?')}")

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
