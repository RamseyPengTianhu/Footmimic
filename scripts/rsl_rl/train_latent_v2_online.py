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
parser.add_argument("--prior_type", type=str, default="mlp", choices=("mlp", "lstm"),
                    help="Prior architecture: mlp=single-frame MLP, lstm=LSTM with hidden state.")
parser.add_argument("--lstm_hidden", type=int, default=128,
                    help="LSTM prior hidden size (only used when prior_type=lstm).")
parser.add_argument("--lstm_layers", type=int, default=1,
                    help="LSTM prior num layers (only used when prior_type=lstm).")

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
from latent_v2_models import LatentActionModel, latent_distill_loss
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
    print(f"  Prior rollout:    {args_cli.prior_rollout_ratio:.0%} of steps")
    print(f"  Feature version:  {FEATURE_VERSION}")
    print(f"{'='*70}\n")

    best_kick_pct = 0.0
    t0 = time.time()

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

        # ── Phase 2: Training (gradient updates) ───────────────────────────
        if buffer.size < args_cli.batch_size:
            continue  # Not enough data yet

        model.train()
        iter_recon = 0.0
        iter_kl = 0.0
        iter_prior_recon = 0.0

        for _ in range(args_cli.updates_per_iter):
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
            iter_kl += losses["kl"].item()
            if "prior_recon" in losses:
                iter_prior_recon += losses["prior_recon"].item()

        avg_recon = iter_recon / args_cli.updates_per_iter
        avg_kl = iter_kl / args_cli.updates_per_iter
        avg_prior_recon = iter_prior_recon / args_cli.updates_per_iter

        # ── Logging ────────────────────────────────────────────────────────
        elapsed = time.time() - t0
        log_parts = [
            f"  Iter {iteration+1:4d}/{args_cli.num_iters}",
            f"recon={avg_recon:.6f} kl={avg_kl:.2f}",
        ]
        if args_cli.alpha_prior > 0:
            log_parts.append(f"p_recon={avg_prior_recon:.6f}")
        log_parts.append(f"buf={buffer.size} ep={iter_episodes}")
        log_parts.append(f"bmix={beta_mix:.2f}")
        log_parts.append(f"{elapsed:.0f}s")
        print(" | ".join(log_parts))

        # ── Phase 3: Eval ──────────────────────────────────────────────────
        if (iteration + 1) % args_cli.eval_interval == 0:
            # Prior eval (deployment path)
            eval_prior = eval_prior_rollout(
                model, env_wrapped, num_episodes=50, device=device,
                use_task_features=use_task_features,
            )
            # Posterior eval (reconstruction quality)
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

            # Save if best (based on prior eval — the deployment path)
            if eval_prior["kick_pct"] > best_kick_pct:
                best_kick_pct = eval_prior["kick_pct"]
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
            if eval_prior["kick_pct"] >= 90 and eval_prior["fall_pct"] <= 10:
                print(f"\n  *** GATE PASSED: Kick%={eval_prior['kick_pct']:.1f}%, Fall%={eval_prior['fall_pct']:.1f}% ***")
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
