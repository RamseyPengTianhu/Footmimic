"""Stage 2B: Online Distillation with DAgger (LATENT-style).

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/rsl_rl/train_latent_v2_online.py \
        --task Anchor-CG-Kick-G1-Soccer-RNN-v0 \
        --motion_path motions/Video_hmr4d_seed \
        --load_run "2026-04-28_12-15-12_cg_v3_softmask" \
        --checkpoint model_12000.pt \
        --num_envs 64 \
        --num_iters 200 \
        --output_path models/latent_v2/online_distill.pt \
        --device cuda:0 \
        --headless

Core design (following LATENT paper Section 3.2.2):
  - At each step, query teacher for a_teacher
  - Encode via POSTERIOR: z = E(obs, a_teacher)  → informed by teacher, high quality
  - Decode: a_student = D(obs, z)                → bottleneck drops info, != a_teacher
  - Step env with a_student                       → student visits its OWN distribution
  - Store (obs, a_teacher) in replay buffer
  - Train encoder + decoder + prior jointly:
      L = L_recon(decoder output, teacher action) + beta * L_KL(posterior || prior)
  - Prior P(z|obs) learns to match posterior via KL → used at deployment (no teacher)

Key difference from old approach:
  - Old: offline CVAE on teacher's distribution → decoder fails on student's states
  - New: online DAgger on student's distribution → decoder learns to handle its own errors
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Stage 2B: Online distillation with DAgger.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default="Anchor-CG-Kick-G1-Soccer-RNN-v0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--motion_path", type=str, required=True)

# Training
parser.add_argument("--num_iters", type=int, default=200,
                    help="Number of distillation iterations.")
parser.add_argument("--steps_per_iter", type=int, default=500,
                    help="Env steps per iteration (before training).")
parser.add_argument("--updates_per_iter", type=int, default=50,
                    help="Gradient updates per iteration.")
parser.add_argument("--batch_size", type=int, default=2048)
parser.add_argument("--buffer_size", type=int, default=500000,
                    help="Max transitions in replay buffer.")
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--beta", type=float, default=1e-3,
                    help="KL weight for VAE loss.")
parser.add_argument("--alpha_prior", type=float, default=0.5,
                    help="Prior recon loss weight. Directly trains D(obs, P(obs)).")

# Model
parser.add_argument("--z_dim", type=int, default=16)
parser.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 256, 128])
parser.add_argument("--decoder_obs_mode", type=str, default="full", choices=("full", "task", "task_features"),
                    help="Decoder/prior obs: full=160D obs_v3, task=99D proprioception, task_features=99D+26D.")
parser.add_argument("--prior_type", type=str, default="mlp", choices=("mlp", "lstm", "vq"),
                    help="Prior architecture: mlp=single-frame MLP, lstm=LSTM with hidden state, vq=VQ-VAE categorical.")
parser.add_argument("--lstm_hidden", type=int, default=128,
                    help="LSTM prior hidden size (only used when prior_type=lstm).")
parser.add_argument("--lstm_layers", type=int, default=1,
                    help="LSTM prior num layers (only used when prior_type=lstm).")
parser.add_argument("--seq_len", type=int, default=32,
                    help="Sequence length for LSTM prior training (only used when prior_type=lstm).")
parser.add_argument("--seq_batch_size", type=int, default=64,
                    help="Number of sequences per training batch (only used when prior_type=lstm).")
parser.add_argument("--seq_buffer_steps", type=int, default=8000,
                    help="Max timesteps per env in SequenceReplayBuffer (only used when prior_type=lstm).")
parser.add_argument("--num_codes", type=int, default=64,
                    help="Number of codebook entries (only used when prior_type=vq).")
parser.add_argument("--commitment_weight", type=float, default=0.25,
                    help="VQ commitment loss weight (only used when prior_type=vq).")
parser.add_argument("--code_hold", type=int, default=1,
                    help="VQ code hold duration: re-select code every N frames. "
                         "1=every frame (default), 2=every 2 frames. Only used when prior_type=vq.")
parser.add_argument("--alpha_switch", type=float, default=0.01,
                    help="Weight for z_e temporal smoothness penalty (only used when code_hold>1).")
parser.add_argument("--residual_noise_alpha", type=float, default=0.0,
                    help="Route beta: inject Gaussian noise N(0, alpha^2) around z_q during decoder training. "
                         "0.0=standard VQ (default). Trains decoder to handle z_q+residual. "
                         "Recommend starting with 0.1.")

# DAgger
parser.add_argument("--warmup_iters", type=int, default=5,
                    help="Initial iterations using teacher rollout (beta_mix=1).")
parser.add_argument("--beta_mix_final", type=float, default=0.0,
                    help="Final teacher mixing ratio. 0=pure student (standard DAgger).")
parser.add_argument("--prior_rollout_ratio", type=float, default=0.3,
                    help="Fraction of rollout steps using prior z (vs posterior z). "
                         "Collects teacher labels on prior-visited states to close "
                         "the prior distribution shift gap. 0=disabled.")

# Checkpointing
parser.add_argument("--output_path", type=str, default="models/latent_v2/online_distill.pt")
parser.add_argument("--resume_from", type=str, default=None,
                    help="Resume from a previous latent model checkpoint.")
parser.add_argument("--eval_interval", type=int, default=10,
                    help="Evaluate every N iterations.")

import cli_args  # isort: skip
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = False
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import glob
import time
import gymnasium as gym
import torch
import numpy as np

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import soccer.tasks  # noqa: F401
from soccer.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner
from latent_v2_models import LatentActionModel, latent_distill_loss, latent_distill_loss_masked, vq_distill_loss, vq_hold_loss
from compute_task_features import compute_ball_foot_relation, TASK_FEATURES_DIM, FEATURE_NAMES, FEATURE_VERSION


# ─── Replay Buffer ────────────────────────────────────────────────────────────


class ReplayBuffer:
    """Simple circular replay buffer for (obs, action, task_features) tuples."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: str,
                 task_features_dim: int = 0):
        self.capacity = capacity
        self.device = device
        self.obs = torch.zeros(capacity, obs_dim, device=device)
        self.actions = torch.zeros(capacity, action_dim, device=device)
        self.has_task_features = task_features_dim > 0
        if self.has_task_features:
            self.task_features = torch.zeros(capacity, task_features_dim, device=device)
        self.size = 0
        self.ptr = 0

    def add(self, obs: torch.Tensor, actions: torch.Tensor,
            task_features: torch.Tensor | None = None):
        """Add a batch of (obs, action, task_features) tuples."""
        B = obs.shape[0]
        if self.ptr + B <= self.capacity:
            self.obs[self.ptr:self.ptr + B] = obs
            self.actions[self.ptr:self.ptr + B] = actions
            if self.has_task_features and task_features is not None:
                self.task_features[self.ptr:self.ptr + B] = task_features
        else:
            # Wrap around
            remaining = self.capacity - self.ptr
            self.obs[self.ptr:] = obs[:remaining]
            self.actions[self.ptr:] = actions[:remaining]
            overflow = B - remaining
            self.obs[:overflow] = obs[remaining:]
            self.actions[:overflow] = actions[remaining:]
            if self.has_task_features and task_features is not None:
                self.task_features[self.ptr:] = task_features[:remaining]
                self.task_features[:overflow] = task_features[remaining:]

        self.ptr = (self.ptr + B) % self.capacity
        self.size = min(self.size + B, self.capacity)

    def sample(self, batch_size: int) -> tuple:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        if self.has_task_features:
            return self.obs[idx], self.actions[idx], self.task_features[idx]
        return self.obs[idx], self.actions[idx], None


class SequenceReplayBuffer:
    """Replay buffer that preserves per-env temporal order for sequence sampling.

    Data is stored as [num_envs, max_steps_per_env, D] tensors. Each env has
    its own circular timeline. Sequences are sampled as contiguous [B, T, D]
    chunks with done masks to handle episode boundaries.

    This is required for LSTM prior training: the LSTM needs to unroll over
    consecutive timesteps to build temporal context (phase/timing memory).
    """

    def __init__(
        self,
        max_steps_per_env: int,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        device: str,
        task_features_dim: int = 0,
    ):
        self.max_steps = max_steps_per_env
        self.num_envs = num_envs
        self.device = device

        self.obs = torch.zeros(num_envs, max_steps_per_env, obs_dim, device=device)
        self.actions = torch.zeros(num_envs, max_steps_per_env, action_dim, device=device)
        self.dones = torch.zeros(num_envs, max_steps_per_env, dtype=torch.bool, device=device)
        self.has_task_features = task_features_dim > 0
        if self.has_task_features:
            self.task_features = torch.zeros(
                num_envs, max_steps_per_env, task_features_dim, device=device
            )

        # Per-env write pointer and fill count
        self.ptr = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.env_size = torch.zeros(num_envs, dtype=torch.long, device=device)

    @property
    def size(self) -> int:
        """Total number of transitions stored across all envs."""
        return int(self.env_size.sum().item())

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        dones: torch.Tensor,
        task_features: torch.Tensor | None = None,
    ):
        """Add one timestep for all envs. obs/actions/dones are [num_envs, D]."""
        for i in range(self.num_envs):
            p = self.ptr[i].item()
            self.obs[i, p] = obs[i]
            self.actions[i, p] = actions[i]
            self.dones[i, p] = dones[i]
            if self.has_task_features and task_features is not None:
                self.task_features[i, p] = task_features[i]
            self.ptr[i] = (p + 1) % self.max_steps
            self.env_size[i] = min(self.env_size[i] + 1, self.max_steps)

    def sample(self, batch_size: int, seq_len: int = 32) -> dict:
        """Sample batch_size sequences of length seq_len.

        Returns dict with:
            obs:       [B, T, obs_dim]
            actions:   [B, T, action_dim]
            features:  [B, T, feat_dim] or None
            mask:      [B, T] bool — True for valid timesteps
        """
        # Only sample from envs with enough data
        valid_envs = (self.env_size >= seq_len).nonzero(as_tuple=True)[0]
        if len(valid_envs) == 0:
            # Fallback: sample from envs with most data
            min_len = self.env_size.max().item()
            actual_seq_len = min(seq_len, min_len)
            valid_envs = (self.env_size >= actual_seq_len).nonzero(as_tuple=True)[0]
            if len(valid_envs) == 0:
                return None
            seq_len = actual_seq_len

        # Random env indices and start positions
        env_idx = valid_envs[torch.randint(len(valid_envs), (batch_size,), device=self.device)]
        max_start = self.env_size[env_idx] - seq_len  # [B]
        start_idx = (torch.rand(batch_size, device=self.device) * (max_start.float() + 1)).long()
        start_idx = start_idx.clamp(min=0)

        # Gather sequences
        obs_seq = torch.zeros(batch_size, seq_len, self.obs.shape[-1], device=self.device)
        act_seq = torch.zeros(batch_size, seq_len, self.actions.shape[-1], device=self.device)
        done_seq = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device)
        feat_seq = None
        if self.has_task_features:
            feat_seq = torch.zeros(
                batch_size, seq_len, self.task_features.shape[-1], device=self.device
            )

        for b in range(batch_size):
            e = env_idx[b].item()
            s = start_idx[b].item()
            # Handle circular buffer wrap-around
            for t in range(seq_len):
                idx = (s + t) % self.max_steps
                obs_seq[b, t] = self.obs[e, idx]
                act_seq[b, t] = self.actions[e, idx]
                done_seq[b, t] = self.dones[e, idx]
                if feat_seq is not None:
                    feat_seq[b, t] = self.task_features[e, idx]

        # Build valid mask: after a done, all subsequent steps are invalid
        # (because the LSTM hidden should be reset at done boundaries)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=self.device)
        for b in range(batch_size):
            for t in range(seq_len):
                if done_seq[b, t]:
                    # The done step itself is valid, but everything after is invalid
                    mask[b, t + 1:] = False
                    break

        return {
            "obs": obs_seq,
            "actions": act_seq,
            "features": feat_seq,
            "mask": mask,
            "dones": done_seq,
        }


# ─── Eval helper ───────────────────────────────────────────────────────────────


def eval_prior_rollout(
    model: LatentActionModel,
    env_wrapped: RslRlVecEnvWrapper,
    num_episodes: int = 50,
    device: str = "cuda",
    use_task_features: bool = False,
) -> dict:
    """Quick eval: run decoder with prior mean z, count kicks/falls."""
    model.eval()
    obs_v3, _ = env_wrapped.get_observations()
    model.reset_prior_hidden(obs_v3.shape[0], device)  # Reset LSTM state for eval
    episodes = 0
    kicks = 0
    falls = 0
    ep_len = torch.zeros(obs_v3.shape[0], device=device)

    unwrapped = env_wrapped.unwrapped
    ball_contacted = torch.zeros(obs_v3.shape[0], dtype=torch.bool, device=device)

    while episodes < num_episodes:
        with torch.no_grad():
            tf = compute_ball_foot_relation(unwrapped) if use_task_features else None
            action = model.act_prior_mean(obs_v3, task_features=tf)
        obs_v3, _, dones, _ = env_wrapped.step(action)
        ep_len += 1

        # Check ball contact
        ball = unwrapped.scene["soccer_ball"]
        bvel = ball.data.root_lin_vel_w[:, :2]
        bspd = torch.norm(bvel, dim=-1)
        ball_contacted |= (bspd > 0.5)

        if dones.any():
            model.reset_prior_hidden_at(dones)
            done_ids = dones.nonzero(as_tuple=True)[0]
            for idx in done_ids:
                if episodes >= num_episodes:
                    break
                i = idx.item()
                episodes += 1
                if ball_contacted[i]:
                    kicks += 1
                if ep_len[i] < 100:
                    falls += 1
                ep_len[i] = 0
                ball_contacted[i] = False

    model.train()
    return {
        "kick_pct": kicks / max(episodes, 1) * 100,
        "fall_pct": falls / max(episodes, 1) * 100,
        "episodes": episodes,
    }


def eval_posterior_rollout(
    model: LatentActionModel,
    env_wrapped: RslRlVecEnvWrapper,
    teacher_policy,
    num_episodes: int = 50,
    device: str = "cuda",
    use_task_features: bool = False,
) -> dict:
    """Eval: run decoder with posterior z (using teacher action), count kicks/falls.
    This shows how well the decoder reconstructs teacher behavior."""
    model.eval()
    obs_v3, _ = env_wrapped.get_observations()
    episodes = 0
    kicks = 0
    falls = 0
    ep_len = torch.zeros(obs_v3.shape[0], device=device)

    unwrapped = env_wrapped.unwrapped
    ball_contacted = torch.zeros(obs_v3.shape[0], dtype=torch.bool, device=device)

    while episodes < num_episodes:
        with torch.no_grad():
            a_teacher = teacher_policy(obs_v3)
            tf = compute_ball_foot_relation(unwrapped) if use_task_features else None
            dec_obs = model.select_decoder_obs(obs_v3, task_features=tf)
            if model.prior_type == "vq":
                z_e = model.encoder(dec_obs, a_teacher)
                z_q, _, _ = model.codebook.quantize(z_e)
                action = model.decoder(dec_obs, z_q)
            else:
                q_mu, _ = model.encoder(dec_obs, a_teacher)
                action = model.decoder(dec_obs, q_mu)  # posterior mean
        obs_v3, _, dones, _ = env_wrapped.step(action)
        ep_len += 1

        ball = unwrapped.scene["soccer_ball"]
        bvel = ball.data.root_lin_vel_w[:, :2]
        bspd = torch.norm(bvel, dim=-1)
        ball_contacted |= (bspd > 0.5)

        if dones.any():
            done_ids = dones.nonzero(as_tuple=True)[0]
            for idx in done_ids:
                if episodes >= num_episodes:
                    break
                i = idx.item()
                episodes += 1
                if ball_contacted[i]:
                    kicks += 1
                if ep_len[i] < 100:
                    falls += 1
                ep_len[i] = 0
                ball_contacted[i] = False

    model.train()
    return {
        "kick_pct": kicks / max(episodes, 1) * 100,
        "fall_pct": falls / max(episodes, 1) * 100,
        "episodes": episodes,
    }

# ─── VQ Diagnostic Eval ───────────────────────────────────────────────────────


def eval_vq_diagnostic(
    model: LatentActionModel,
    env_wrapped: RslRlVecEnvWrapper,
    teacher_policy,
    num_episodes: int = 50,
    device: str = "cuda",
    use_task_features: bool = False,
    code_hold: int = 1,
) -> dict:
    """Comprehensive VQ-VAE diagnostic eval.

    Runs both prior and posterior rollouts, collecting:
      - Kick%, Fall%, BSpd (ball speed at contact)
      - cb_util, perplexity (effective code diversity)
      - code switching rate (how often code changes frame-to-frame)
      - posterior vs prior code agreement (% frames where argmax matches)
      - Episode outcome classification:
          clean_success: kicked + survived full episode
          late_fallback: kicked but then fell
          empty_swing: survived but never contacted ball
          early_fall: fell before any ball contact
    """
    assert model.prior_type == "vq", "eval_vq_diagnostic requires prior_type='vq'"
    model.eval()

    results = {}

    # ── Prior rollout ─────────────────────────────────────────────────────
    obs_v3, _ = env_wrapped.get_observations()
    N = obs_v3.shape[0]
    episodes = 0
    kicks = 0
    falls = 0
    clean_success = 0
    late_fallback = 0
    empty_swing = 0
    early_fall = 0
    ball_speeds = []
    ep_len = torch.zeros(N, device=device)
    ball_contacted = torch.zeros(N, dtype=torch.bool, device=device)
    max_bspd = torch.zeros(N, device=device)

    # Code tracking
    all_prior_codes = []
    prev_code = None

    code_switches = 0
    code_total_frames = 0

    # Code hold state
    held_zq_prior = torch.zeros(N, model.z_dim, device=device)
    held_code_prior = torch.zeros(N, dtype=torch.long, device=device)
    hold_ctr_prior = torch.zeros(N, dtype=torch.long, device=device)

    while episodes < num_episodes:
        with torch.no_grad():
            tf = compute_ball_foot_relation(env_wrapped.unwrapped) if use_task_features else None
            dec_obs = model.select_decoder_obs(obs_v3, task_features=tf)

            # Re-select code every code_hold frames
            needs_update = (hold_ctr_prior % code_hold == 0)
            if needs_update.any():
                logits = model.prior(dec_obs)
                new_code = logits.argmax(dim=-1)
                new_zq = model.codebook.lookup(new_code)
                held_code_prior[needs_update] = new_code[needs_update]
                held_zq_prior[needs_update] = new_zq[needs_update]

            action = model.decoder(dec_obs, held_zq_prior)

        all_prior_codes.append(held_code_prior.cpu().clone())

        # Code switching rate
        if prev_code is not None:
            code_switches += (held_code_prior != prev_code).sum().item()
        code_total_frames += N
        prev_code = held_code_prior.clone()
        hold_ctr_prior += 1

        obs_v3, _, dones, _ = env_wrapped.step(action)
        ep_len += 1

        # Ball speed
        ball = env_wrapped.unwrapped.scene["soccer_ball"]
        bvel = ball.data.root_lin_vel_w[:, :2]
        bspd = torch.norm(bvel, dim=-1)
        ball_contacted |= (bspd > 0.5)
        max_bspd = torch.max(max_bspd, bspd)

        if dones.any():
            done_ids = dones.nonzero(as_tuple=True)[0]
            for idx in done_ids:
                if episodes >= num_episodes:
                    break
                i = idx.item()
                episodes += 1
                kicked = ball_contacted[i].item()
                fell = ep_len[i].item() < 100

                if kicked:
                    kicks += 1
                    ball_speeds.append(max_bspd[i].item())
                if fell:
                    falls += 1

                # Outcome classification
                if kicked and not fell:
                    clean_success += 1
                elif kicked and fell:
                    late_fallback += 1
                elif not kicked and not fell:
                    empty_swing += 1
                else:  # not kicked and fell
                    early_fall += 1

                ep_len[i] = 0
                ball_contacted[i] = False
                max_bspd[i] = 0
                hold_ctr_prior[i] = 0  # Reset hold counter on episode boundary
            prev_code = None  # Reset code tracking on done

    # Aggregate prior codes
    all_prior_codes_cat = torch.cat(all_prior_codes, dim=0)
    prior_perplexity = model.codebook.perplexity_from_indices(all_prior_codes_cat)
    switch_rate = code_switches / max(code_total_frames, 1)

    results["prior"] = {
        "kick_pct": kicks / max(episodes, 1) * 100,
        "fall_pct": falls / max(episodes, 1) * 100,
        "avg_bspd": sum(ball_speeds) / max(len(ball_speeds), 1),
        "max_bspd": max(ball_speeds) if ball_speeds else 0,
        "clean_success": clean_success,
        "late_fallback": late_fallback,
        "empty_swing": empty_swing,
        "early_fall": early_fall,
        "perplexity": prior_perplexity,
        "code_switch_rate": switch_rate,
        "episodes": episodes,
    }

    # ── Posterior rollout (with teacher action for encoding) ───────────────
    obs_v3, _ = env_wrapped.get_observations()
    episodes = 0
    kicks = 0
    falls = 0
    ep_len = torch.zeros(N, device=device)
    ball_contacted = torch.zeros(N, dtype=torch.bool, device=device)
    max_bspd = torch.zeros(N, device=device)
    ball_speeds = []

    all_post_codes = []
    all_prior_codes2 = []
    agreements = 0
    agreement_total = 0

    # Code hold state for posterior
    held_zq_post = torch.zeros(N, model.z_dim, device=device)
    held_code_post = torch.zeros(N, dtype=torch.long, device=device)
    hold_ctr_post = torch.zeros(N, dtype=torch.long, device=device)

    while episodes < num_episodes:
        with torch.no_grad():
            a_teacher = teacher_policy(obs_v3)
            tf = compute_ball_foot_relation(env_wrapped.unwrapped) if use_task_features else None
            dec_obs = model.select_decoder_obs(obs_v3, task_features=tf)

            # Posterior: encode with teacher action, re-quantize every code_hold frames
            z_e = model.encoder(dec_obs, a_teacher)
            needs_update = (hold_ctr_post % code_hold == 0)
            if needs_update.any():
                z_q_new, post_code_new, _ = model.codebook.quantize(z_e)
                held_zq_post[needs_update] = z_q_new[needs_update]
                held_code_post[needs_update] = post_code_new[needs_update]
            post_code = held_code_post
            action = model.decoder(dec_obs, held_zq_post)

            # Prior: what would prior have chosen?
            prior_logits = model.prior(dec_obs)
            prior_code = prior_logits.argmax(dim=-1)

        all_post_codes.append(post_code.cpu())
        all_prior_codes2.append(prior_code.cpu())
        agreements += (post_code == prior_code).sum().item()
        agreement_total += N
        hold_ctr_post += 1

        obs_v3, _, dones, _ = env_wrapped.step(action)
        ep_len += 1

        ball = env_wrapped.unwrapped.scene["soccer_ball"]
        bvel = ball.data.root_lin_vel_w[:, :2]
        bspd = torch.norm(bvel, dim=-1)
        ball_contacted |= (bspd > 0.5)
        max_bspd = torch.max(max_bspd, bspd)

        if dones.any():
            done_ids = dones.nonzero(as_tuple=True)[0]
            for idx in done_ids:
                if episodes >= num_episodes:
                    break
                i = idx.item()
                episodes += 1
                if ball_contacted[i]:
                    kicks += 1
                    ball_speeds.append(max_bspd[i].item())
                if ep_len[i] < 100:
                    falls += 1
                ep_len[i] = 0
                ball_contacted[i] = False
                max_bspd[i] = 0
                hold_ctr_post[i] = 0

    all_post_codes_cat = torch.cat(all_post_codes, dim=0)
    post_perplexity = model.codebook.perplexity_from_indices(all_post_codes_cat)

    results["posterior"] = {
        "kick_pct": kicks / max(episodes, 1) * 100,
        "fall_pct": falls / max(episodes, 1) * 100,
        "avg_bspd": sum(ball_speeds) / max(len(ball_speeds), 1),
        "perplexity": post_perplexity,
        "episodes": episodes,
    }
    results["agreement_pct"] = agreements / max(agreement_total, 1) * 100
    results["cb_util"] = model.codebook.codebook_utilization()

    # ── Posterior Continuous rollout (z_e without quantization) ───────────
    obs_v3, _ = env_wrapped.get_observations()
    episodes = 0
    kicks = 0
    falls = 0
    ep_len = torch.zeros(N, device=device)
    ball_contacted = torch.zeros(N, dtype=torch.bool, device=device)
    max_bspd = torch.zeros(N, device=device)
    ball_speeds = []

    while episodes < num_episodes:
        with torch.no_grad():
            a_teacher = teacher_policy(obs_v3)
            tf = compute_ball_foot_relation(env_wrapped.unwrapped) if use_task_features else None
            dec_obs = model.select_decoder_obs(obs_v3, task_features=tf)

            # Posterior Continuous: encode with teacher action, but do NOT quantize
            z_e = model.encoder(dec_obs, a_teacher)
            action = model.decoder(dec_obs, z_e)

        obs_v3, _, dones, _ = env_wrapped.step(action)
        ep_len += 1

        ball = env_wrapped.unwrapped.scene["soccer_ball"]
        bvel = ball.data.root_lin_vel_w[:, :2]
        bspd = torch.norm(bvel, dim=-1)
        ball_contacted |= (bspd > 0.5)
        max_bspd = torch.max(max_bspd, bspd)

        if dones.any():
            done_ids = dones.nonzero(as_tuple=True)[0]
            for idx in done_ids:
                if episodes >= num_episodes:
                    break
                i = idx.item()
                episodes += 1
                if ball_contacted[i]:
                    kicks += 1
                    ball_speeds.append(max_bspd[i].item())
                if ep_len[i] < 100:
                    falls += 1
                ep_len[i] = 0
                ball_contacted[i] = False
                max_bspd[i] = 0

    results["posterior_continuous"] = {
        "kick_pct": kicks / max(episodes, 1) * 100,
        "fall_pct": falls / max(episodes, 1) * 100,
        "avg_bspd": sum(ball_speeds) / max(len(ball_speeds), 1),
    }

    model.train()
    return results


def print_vq_diagnostic(results: dict, iteration: int):
    """Pretty-print VQ diagnostic results."""
    p = results["prior"]
    q = results["posterior"]
    print(f"\n>>> VQ EVAL iter {iteration}:")
    print(f"  Prior:     Kick={p['kick_pct']:.0f}% Fall={p['fall_pct']:.0f}% "
          f"BSpd={p['avg_bspd']:.2f} perp={p['perplexity']:.1f} "
          f"switch={p['code_switch_rate']:.2f}")
    print(f"  Post(Quant): Kick={q['kick_pct']:.0f}% Fall={q['fall_pct']:.0f}% "
          f"BSpd={q['avg_bspd']:.2f} perp={q['perplexity']:.1f}")
    
    if "posterior_continuous" in results:
        c = results["posterior_continuous"]
        print(f"  Post(Cont):  Kick={c['kick_pct']:.0f}% Fall={c['fall_pct']:.0f}% "
              f"BSpd={c['avg_bspd']:.2f} (z_e without quantization)")
    print(f"  Agreement: {results['agreement_pct']:.1f}% "
          f"cb_util={results['cb_util']:.0%}")
    print(f"  Outcomes:  clean={p['clean_success']} late_fall={p['late_fallback']} "
          f"empty={p['empty_swing']} early_fall={p['early_fall']}")


# ─── Main ─────────────────────────────────────────────────────────────────────


def get_motion_files(motion_path):
    if os.path.isfile(motion_path):
        return [motion_path]
    elif os.path.isdir(motion_path):
        files = sorted(glob.glob(os.path.join(motion_path, "*.npz")))
        if not files:
            raise ValueError(f"No .npz files in {motion_path}")
        return files
    else:
        raise ValueError(f"Invalid path: {motion_path}")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Online distillation with DAgger."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
    env_cfg.commands.motion.motion_files = get_motion_files(args_cli.motion_path)
    device = env_cfg.sim.device

    # ── Create env ─────────────────────────────────────────────────────────
    env = gym.make(args_cli.task, cfg=env_cfg)
    env_wrapped = RslRlVecEnvWrapper(env)
    unwrapped_env = env.unwrapped

    # ── Load frozen v3 teacher ─────────────────────────────────────────────
    log_root = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    resume_path = get_checkpoint_path(
        os.path.abspath(log_root), agent_cfg.load_run, agent_cfg.load_checkpoint
    )
    print(f"[INFO] Loading v3 teacher from: {resume_path}")
    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(resume_path)
    teacher_policy = runner.get_inference_policy(device=device)

    # Get obs dim from env
    obs_v3, _ = env_wrapped.get_observations()
    obs_dim = obs_v3.shape[1]
    action_dim = env_wrapped.action_space.shape[1] if hasattr(env_wrapped.action_space, 'shape') else 29
    print(f"[INFO] obs_dim={obs_dim}, action_dim={action_dim}")

    # ── Create latent model ────────────────────────────────────────────────
    model = LatentActionModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        z_dim=args_cli.z_dim,
        hidden_dims=args_cli.hidden_dims,
        decoder_obs_mode=args_cli.decoder_obs_mode,
        prior_type=args_cli.prior_type,
        lstm_hidden=args_cli.lstm_hidden,
        lstm_layers=args_cli.lstm_layers,
        num_codes=args_cli.num_codes,
        commitment_weight=args_cli.commitment_weight,
    ).to(device)
    print(
        f"[INFO] decoder_obs_mode={args_cli.decoder_obs_mode}, "
        f"decoder_obs_dim={model.decoder_obs_dim}"
    )

    if args_cli.resume_from:
        print(f"[INFO] Resuming from: {args_cli.resume_from}")
        ckpt = torch.load(args_cli.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] LatentActionModel: z_dim={args_cli.z_dim}, params={n_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args_cli.lr)

    # ── Replay buffer ──────────────────────────────────────────────────────
    use_task_features = args_cli.decoder_obs_mode == "task_features"
    buffer = ReplayBuffer(
        capacity=args_cli.buffer_size,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        task_features_dim=TASK_FEATURES_DIM if use_task_features else 0,
    )
    # Sequence buffer for LSTM prior training or VQ hold training
    seq_buffer = None
    if args_cli.prior_type == "lstm" or (args_cli.prior_type == "vq" and args_cli.code_hold > 1):
        seq_buf_steps = args_cli.seq_buffer_steps if args_cli.prior_type == "lstm" else 4000
        seq_buffer = SequenceReplayBuffer(
            max_steps_per_env=seq_buf_steps,
            num_envs=args_cli.num_envs,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
            task_features_dim=TASK_FEATURES_DIM if use_task_features else 0,
        )

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Stage 2B: Online Distillation with DAgger")
    print(f"{'='*70}")
    print(f"  Iterations:       {args_cli.num_iters}")
    print(f"  Steps/iter:       {args_cli.steps_per_iter}")
    print(f"  Updates/iter:     {args_cli.updates_per_iter}")
    print(f"  Buffer size:      {args_cli.buffer_size}")
    print(f"  Warmup iters:     {args_cli.warmup_iters} (teacher rollout)")
    print(f"  Beta mix final:   {args_cli.beta_mix_final}")
    print(f"  KL weight (beta): {args_cli.beta}")
    print(f"  Prior recon wt:   {args_cli.alpha_prior}")
    print(f"  Prior type:       {args_cli.prior_type}")
    if args_cli.prior_type == "lstm":
        print(f"  LSTM hidden:      {args_cli.lstm_hidden}")
        print(f"  LSTM layers:      {args_cli.lstm_layers}")
        print(f"  Seq length:       {args_cli.seq_len}")
        print(f"  Seq batch size:   {args_cli.seq_batch_size}")
        print(f"  Seq buf steps:    {args_cli.seq_buffer_steps}/env")
    if args_cli.prior_type == "vq":
        print(f"  Num codes (K):    {args_cli.num_codes}")
        print(f"  Commitment wt:    {args_cli.commitment_weight}")
        print(f"  Code hold:        {args_cli.code_hold} frames")
        if args_cli.code_hold > 1:
            print(f"  Switch penalty:   {args_cli.alpha_switch}")
            print(f"  Seq buffer:       {seq_buf_steps if seq_buffer else 'N/A'}/env")
    print(f"  Prior rollout:    {args_cli.prior_rollout_ratio:.0%} of steps")
    if args_cli.residual_noise_alpha > 0:
        print(f"  Residual noise:   {args_cli.residual_noise_alpha} (route beta decoder training)")
    print(f"  Feature version:  {FEATURE_VERSION}")
    print(f"{'='*70}\n")

    best_kick_pct = 0.0
    t0 = time.time()

    # VQ code hold state (persistent across iterations, reset on episode boundaries)
    N_envs = obs_v3.shape[0]
    vq_hold_counter = torch.zeros(N_envs, dtype=torch.long, device=device)
    vq_held_zq = torch.zeros(N_envs, args_cli.z_dim, device=device)

    for iteration in range(args_cli.num_iters):
        # ── Compute DAgger beta_mix ────────────────────────────────────────
        if iteration < args_cli.warmup_iters:
            beta_mix = 1.0  # Pure teacher rollout during warmup
        else:
            # Linear decay from 1.0 to beta_mix_final
            progress = (iteration - args_cli.warmup_iters) / max(
                args_cli.num_iters - args_cli.warmup_iters, 1
            )
            beta_mix = 1.0 - progress * (1.0 - args_cli.beta_mix_final)
            beta_mix = max(beta_mix, args_cli.beta_mix_final)

        # ── Phase 1: Rollout (collect data) ────────────────────────────────
        # Interleaves posterior rollout (standard DAgger) with prior rollout.
        # Prior rollout steps use D(obs, P(obs)) to visit states the prior
        # would actually encounter, then query teacher for labels.
        # This closes the distribution shift between training and deployment.
        model.eval()
        iter_episodes = 0
        use_prior_ratio = args_cli.prior_rollout_ratio if iteration >= args_cli.warmup_iters else 0.0

        # Reset LSTM prior hidden state at start of each rollout phase
        if args_cli.prior_type == "lstm":
            model.reset_prior_hidden(obs_v3.shape[0], device)

        for step in range(args_cli.steps_per_iter):
            # 1. Teacher action (always needed for label)
            with torch.no_grad():
                a_teacher = teacher_policy(obs_v3)

            # 2. Compute task features if needed
            task_feat = None
            if use_task_features:
                with torch.no_grad():
                    task_feat = compute_ball_foot_relation(unwrapped_env)

            # 3. Student action: choose between posterior and prior rollout
            use_prior_this_step = (use_prior_ratio > 0 and
                                   torch.rand(1).item() < use_prior_ratio)

            with torch.no_grad():
                dec_obs = model.select_decoder_obs(obs_v3, task_features=task_feat)
                if args_cli.prior_type == "vq":
                    # VQ mode with code hold: re-select every code_hold frames
                    needs_hold_update = (vq_hold_counter % args_cli.code_hold == 0)
                    if use_prior_this_step:
                        if needs_hold_update.any():
                            logits = model.prior(dec_obs)
                            new_code = logits.argmax(dim=-1)
                            new_zq = model.codebook.lookup(new_code)
                            vq_held_zq[needs_hold_update] = new_zq[needs_hold_update]
                        a_student = model.decoder(dec_obs, vq_held_zq)
                    else:
                        z_e = model.encoder(dec_obs, a_teacher)
                        if needs_hold_update.any():
                            z_q_new, _, _ = model.codebook.quantize(z_e)
                            vq_held_zq[needs_hold_update] = z_q_new[needs_hold_update]
                        a_student = model.decoder(dec_obs, vq_held_zq)
                    vq_hold_counter += 1
                else:
                    # Gaussian mode: prior outputs (mu, logvar)
                    p_mu, _ = model.prior(dec_obs)
                    if use_prior_this_step:
                        # Prior rollout: z = P(obs), visits prior's own distribution
                        a_student = model.decoder(dec_obs, p_mu)
                    else:
                        # Posterior rollout: z = E(obs, a_teacher), standard DAgger
                        q_mu, q_logvar = model.encoder(dec_obs, a_teacher)
                        z = model.reparameterize(q_mu, q_logvar)
                        a_student = model.decoder(dec_obs, z)

            # 4. Store in buffer: current state + teacher label + task features
            #    For prior rollout: only store if robot is in a recoverable state
            #    (prevents buffer pollution from 'already dead' states)
            obs_store = obs_v3.detach()
            teacher_store = a_teacher.detach()
            feat_store = task_feat.detach() if task_feat is not None else None
            # SequenceReplayBuffer stores ALL envs per step (needs dones later)
            # We pass dones=False here since the step hasn't happened yet;
            # the actual dones will be stored after env.step().
            if use_prior_this_step:
                root_height = unwrapped_env.scene["robot"].data.root_pos_w[:, 2]
                recoverable = root_height > 0.3  # [N] bool
                if recoverable.any():
                    good_idx = recoverable.nonzero(as_tuple=True)[0]
                    buffer.add(obs_store[good_idx], teacher_store[good_idx],
                               feat_store[good_idx] if feat_store is not None else None)
            else:
                buffer.add(obs_store, teacher_store, feat_store)

            # 5. Choose execution action (beta_mix for warmup)
            if beta_mix > 0.999:
                a_exec = a_teacher  # Pure teacher during warmup
            elif beta_mix < 0.001:
                a_exec = a_student  # Pure student (standard DAgger)
            else:
                a_exec = beta_mix * a_teacher + (1.0 - beta_mix) * a_student

            # 6. Step env
            obs_v3, _, dones, _ = env_wrapped.step(a_exec)

            if dones.any():
                if hasattr(runner.alg.policy, "reset"):
                    runner.alg.policy.reset(dones)
                model.reset_prior_hidden_at(dones)
                iter_episodes += dones.sum().item()
                # Reset VQ hold state on episode boundaries
                done_ids = dones.nonzero(as_tuple=True)[0]
                vq_hold_counter[done_ids] = 0
                vq_held_zq[done_ids] = 0

            # Store in sequence buffer AFTER env.step() so dones are correct
            if seq_buffer is not None:
                seq_buffer.add(obs_store, teacher_store, dones,
                               feat_store)

        # ── Phase 2: Training (gradient updates) ───────────────────────────
        if buffer.size < args_cli.batch_size:
            continue  # Not enough data yet

        model.train()
        iter_recon = 0.0
        iter_kl = 0.0
        iter_prior_recon = 0.0
        iter_vq_loss = 0.0
        iter_prior_ce = 0.0
        iter_switch_pen = 0.0

        for _ in range(args_cli.updates_per_iter):
            if args_cli.prior_type == "vq" and args_cli.code_hold > 1 and seq_buffer is not None and seq_buffer.size >= args_cli.batch_size * args_cli.code_hold:
                # ── VQ hold-reconstruction training (sequence) ──
                seq_data = seq_buffer.sample(args_cli.batch_size, args_cli.code_hold)
                if seq_data is None:
                    # Fallback to IID
                    obs_b, act_b, feat_b = buffer.sample(args_cli.batch_size)
                    fwd = model.forward_vq(obs_b, act_b, task_features=feat_b,
                                           residual_noise_alpha=args_cli.residual_noise_alpha)
                    losses = vq_distill_loss(
                        fwd, act_b,
                        alpha_prior=args_cli.alpha_prior,
                        alpha_prior_recon=args_cli.alpha_prior,
                    )
                else:
                    fwd = model.forward_vq_hold(
                        seq_data["obs"], seq_data["actions"], seq_data["mask"],
                        task_features_seq=seq_data["features"],
                        code_hold=args_cli.code_hold,
                        residual_noise_alpha=args_cli.residual_noise_alpha,
                    )
                    losses = vq_hold_loss(
                        fwd, seq_data["actions"],
                        alpha_prior=args_cli.alpha_prior,
                        alpha_prior_recon=args_cli.alpha_prior,
                        alpha_switch=args_cli.alpha_switch,
                    )
            elif args_cli.prior_type == "vq":
                # ── VQ-VAE IID training (code_hold=1 fallback) ──
                obs_b, act_b, feat_b = buffer.sample(args_cli.batch_size)
                fwd = model.forward_vq(obs_b, act_b, task_features=feat_b,
                                       residual_noise_alpha=args_cli.residual_noise_alpha)
                losses = vq_distill_loss(
                    fwd, act_b,
                    alpha_prior=args_cli.alpha_prior,
                    alpha_prior_recon=args_cli.alpha_prior,
                )
            elif args_cli.prior_type == "lstm" and seq_buffer is not None and seq_buffer.size >= args_cli.seq_batch_size * args_cli.seq_len:
                # ── Sequence training for LSTM prior ──
                seq_data = seq_buffer.sample(args_cli.seq_batch_size, args_cli.seq_len)
                if seq_data is None:
                    continue
                fwd = model.forward_sequence(
                    seq_data["obs"], seq_data["actions"],
                    task_features=seq_data["features"],
                    compute_prior_recon=(args_cli.alpha_prior > 0),
                )
                losses = latent_distill_loss_masked(
                    fwd, seq_data["actions"], seq_data["mask"],
                    beta=args_cli.beta, alpha_prior=args_cli.alpha_prior,
                )
            else:
                # ── Standard i.i.d. training for MLP prior ──
                obs_b, act_b, feat_b = buffer.sample(args_cli.batch_size)
                fwd = model(obs_b, act_b, sample=True, task_features=feat_b,
                            compute_prior_recon=(args_cli.alpha_prior > 0))
                losses = latent_distill_loss(fwd, act_b, beta=args_cli.beta,
                                             alpha_prior=args_cli.alpha_prior)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            iter_recon += losses["recon"].item()
            if "kl" in losses:
                iter_kl += losses["kl"].item()
            if "prior_recon" in losses:
                iter_prior_recon += losses["prior_recon"].item()
            if "vq_loss" in losses:
                iter_vq_loss += losses["vq_loss"].item()
            if "prior_ce" in losses:
                iter_prior_ce += losses["prior_ce"].item()
            if "switch_penalty" in losses:
                iter_switch_pen += losses["switch_penalty"].item()

        avg_recon = iter_recon / args_cli.updates_per_iter
        avg_kl = iter_kl / args_cli.updates_per_iter
        avg_prior_recon = iter_prior_recon / args_cli.updates_per_iter

        # ── Logging ────────────────────────────────────────────────────────
        elapsed = time.time() - t0
        log_parts = [
            f"  Iter {iteration+1:4d}/{args_cli.num_iters}",
            f"recon={avg_recon:.6f}",
        ]
        if args_cli.prior_type == "vq":
            avg_vq = iter_vq_loss / args_cli.updates_per_iter
            avg_ce = iter_prior_ce / args_cli.updates_per_iter
            cb_util = model.codebook.codebook_utilization() if model.codebook else 0
            log_parts.append(f"vq={avg_vq:.4f} ce={avg_ce:.2f}")
            log_parts.append(f"p_recon={avg_prior_recon:.6f}")
            log_parts.append(f"cb_util={cb_util:.0%}")
            if args_cli.code_hold > 1:
                avg_sw = iter_switch_pen / args_cli.updates_per_iter
                log_parts.append(f"sw={avg_sw:.4f}")
        else:
            log_parts.append(f"kl={avg_kl:.2f}")
            if args_cli.alpha_prior > 0:
                log_parts.append(f"p_recon={avg_prior_recon:.6f}")
        buf_info = f"buf={buffer.size}"
        if seq_buffer is not None:
            buf_info += f" seq={seq_buffer.size}"
        log_parts.append(f"{buf_info} ep={iter_episodes}")
        log_parts.append(f"bmix={beta_mix:.2f}")
        log_parts.append(f"{elapsed:.0f}s")
        print(" | ".join(log_parts))

        # ── Phase 3: Eval ──────────────────────────────────────────────────
        if (iteration + 1) % args_cli.eval_interval == 0:
            if args_cli.prior_type == "vq":
                # ── VQ comprehensive diagnostic ──
                vq_results = eval_vq_diagnostic(
                    model, env_wrapped, teacher_policy,
                    num_episodes=50, device=device,
                    use_task_features=use_task_features,
                    code_hold=args_cli.code_hold,
                )
                print_vq_diagnostic(vq_results, iteration + 1)
                eval_kick_pct = vq_results["prior"]["kick_pct"]
                eval_fall_pct = vq_results["prior"]["fall_pct"]
            else:
                # ── Standard eval ──
                eval_prior = eval_prior_rollout(
                    model, env_wrapped, num_episodes=50, device=device,
                    use_task_features=use_task_features,
                )
                eval_post = eval_posterior_rollout(
                    model, env_wrapped, teacher_policy,
                    num_episodes=50, device=device,
                    use_task_features=use_task_features,
                )
                print(
                    f"  >>> EVAL iter {iteration+1}: "
                    f"Prior: Kick%={eval_prior['kick_pct']:.1f}%, Fall%={eval_prior['fall_pct']:.1f}% | "
                    f"Posterior: Kick%={eval_post['kick_pct']:.1f}%, Fall%={eval_post['fall_pct']:.1f}%"
                )
                eval_kick_pct = eval_prior["kick_pct"]
                eval_fall_pct = eval_prior["fall_pct"]

            # Save if best (based on prior eval — the deployment path)
            if eval_kick_pct > best_kick_pct:
                best_kick_pct = eval_kick_pct
                os.makedirs(os.path.dirname(os.path.abspath(args_cli.output_path)), exist_ok=True)
                ckpt = {
                    "model_state_dict": model.state_dict(),
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    "z_dim": args_cli.z_dim,
                    "hidden_dims": args_cli.hidden_dims,
                    "decoder_obs_mode": args_cli.decoder_obs_mode,
                    "prior_type": args_cli.prior_type,
                    "lstm_hidden": args_cli.lstm_hidden,
                    "lstm_layers": args_cli.lstm_layers,
                    "num_codes": args_cli.num_codes,
                    "code_hold": args_cli.code_hold,
                    "best_kick_pct": best_kick_pct,
                    "iteration": iteration + 1,
                    "feature_version": FEATURE_VERSION,
                    "task_features_dim": TASK_FEATURES_DIM,
                    "metadata": {
                        "stage": "2B_online",
                        "teacher_run": agent_cfg.load_run,
                        "teacher_ckpt": agent_cfg.load_checkpoint,
                        "beta_kl": args_cli.beta,
                        "alpha_prior": args_cli.alpha_prior,
                        "buffer_size": args_cli.buffer_size,
                        "decoder_obs_mode": args_cli.decoder_obs_mode,
                        "feature_version": FEATURE_VERSION,
                        "feature_names": FEATURE_NAMES,
                    },
                }
                torch.save(ckpt, args_cli.output_path)
                print(f"  >>> Saved best model (Prior Kick%={best_kick_pct:.1f}%)")

            # Gate check
            if eval_kick_pct >= 90 and eval_fall_pct <= 10:
                print(f"\n  *** GATE PASSED: Kick%={eval_kick_pct:.1f}%, Fall%={eval_fall_pct:.1f}% ***")
                print(f"  *** Ready for Stage 3 (PPO + LAB) ***")

            # Re-fetch obs after eval episodes
            obs_v3, _ = env_wrapped.get_observations()

    # ── Final save ─────────────────────────────────────────────────────────
    final_path = args_cli.output_path.replace(".pt", "_final.pt")
    os.makedirs(os.path.dirname(os.path.abspath(final_path)), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "z_dim": args_cli.z_dim,
        "hidden_dims": args_cli.hidden_dims,
        "decoder_obs_mode": args_cli.decoder_obs_mode,
        "iteration": args_cli.num_iters,
        "feature_version": FEATURE_VERSION,
        "task_features_dim": TASK_FEATURES_DIM,
        "metadata": {
            "stage": "2B_online_final",
            "decoder_obs_mode": args_cli.decoder_obs_mode,
            "feature_version": FEATURE_VERSION,
            "feature_names": FEATURE_NAMES,
            "alpha_prior": args_cli.alpha_prior,
        },
    }, final_path)

    print(f"\n{'='*70}")
    print(f"  Stage 2B: Online Distillation Complete")
    print(f"{'='*70}")
    print(f"  Best Kick%:  {best_kick_pct:.1f}%")
    print(f"  Best model:  {args_cli.output_path}")
    print(f"  Final model: {final_path}")
    print(f"{'='*70}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
