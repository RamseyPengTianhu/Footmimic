"""Train Markov VQ Prior via DAgger: p(code_t | obs_t, code_{t-1}).

Freezes encoder/decoder/codebook from an existing VQ checkpoint and trains
only a first-order Markov categorical prior using DAgger (Dataset Aggregation).
Rollout uses a beta-mixed policy (posterior + prior) to expose the prior to
its own state distribution, solving the exposure bias problem.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/train_markov_prior.py \
        --task Anchor-CG-Kick-G1-Soccer-RNN-v0 \
        --motion_path motions/Video_hmr4d_seed \
        --vq_model models/latent_v2/online_distill_vq_k16_hold2_seq.pt \
        --load_run 2026-04-28_12-15-12_cg_v3_softmask \
        --checkpoint model_12000.pt \
        --num_envs 512 --num_iters 200 \
        --output_path models/latent_v2/markov_prior_dagger.pt \
        --device cuda:0 --headless
"""
from __future__ import annotations
import argparse, os, sys, glob, time, json
import numpy as np

parser = argparse.ArgumentParser(description="Train Markov VQ Prior.")
parser.add_argument("--task", type=str, default="Anchor-CG-Kick-G1-Soccer-RNN-v0")
parser.add_argument("--motion_path", type=str, required=True)
parser.add_argument("--vq_model", type=str, required=True, help="Frozen VQ checkpoint.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=42)
# Training
parser.add_argument("--num_iters", type=int, default=200)
parser.add_argument("--steps_per_iter", type=int, default=500)
parser.add_argument("--updates_per_iter", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=256, help="Number of sequences per batch.")
parser.add_argument("--seq_len", type=int, default=32, help="Sequence length.")
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--strike_ce_weight", type=float, default=2.0,
                    help="CE multiplier for prestrike+strike frames.")
parser.add_argument("--alpha_prior_recon", type=float, default=0.5,
                    help="Weight for prior reconstruction loss.")
# Eval
parser.add_argument("--eval_interval", type=int, default=20)
parser.add_argument("--output_path", type=str, default="models/latent_v2/markov_prior_v1.pt")

from isaaclab.app import AppLauncher
import cli_args

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if getattr(args_cli, "headless", False) and "DISPLAY" in os.environ:
    os.environ.pop("DISPLAY", None)
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import torch.nn.functional as F
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import soccer.tasks  # noqa: F401
from soccer.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner
from latent_v2_models import LatentActionModel, LatentPriorCategorical
from compute_task_features import compute_ball_foot_relation, TASK_FEATURES_DIM

# Phase constants
PHASE_APPROACH = 0
PHASE_PRESTRIKE = 1
PHASE_STRIKE = 2
PHASE_FOLLOW = 3


class SequenceBuffer:
    """Buffer that stores (obs, posterior_code, exec_code, phase_id, dones)
    per env per timestep for DAgger sequence training.

    exec_codes = the code actually used for stepping (mixed posterior/prior).
    post_codes = the posterior label (always from encoder, used as CE target).
    """

    def __init__(self, num_envs, max_steps, obs_dim, device,
                 task_features_dim=0):
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.device = device

        self.obs = torch.zeros(num_envs, max_steps, obs_dim, device=device)
        self.post_codes = torch.zeros(num_envs, max_steps, dtype=torch.long, device=device)
        self.exec_codes = torch.zeros(num_envs, max_steps, dtype=torch.long, device=device)
        self.phase_ids = torch.zeros(num_envs, max_steps, dtype=torch.long, device=device)
        self.dones = torch.zeros(num_envs, max_steps, dtype=torch.bool, device=device)
        self.has_tf = task_features_dim > 0
        if self.has_tf:
            self.task_features = torch.zeros(num_envs, max_steps, task_features_dim, device=device)

        self.ptr = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.env_size = torch.zeros(num_envs, dtype=torch.long, device=device)

    @property
    def size(self):
        return int(self.env_size.sum().item())

    def add(self, obs, post_codes, exec_codes, phase_ids, dones, task_features=None):
        """Add one timestep for all envs (vectorized)."""
        p = self.ptr  # [num_envs]
        idx = torch.arange(self.num_envs, device=self.device)
        self.obs[idx, p] = obs
        self.post_codes[idx, p] = post_codes
        self.exec_codes[idx, p] = exec_codes
        self.phase_ids[idx, p] = phase_ids
        self.dones[idx, p] = dones.to(torch.bool)
        if self.has_tf and task_features is not None:
            self.task_features[idx, p] = task_features
        self.ptr = (p + 1) % self.max_steps
        self.env_size = torch.min(self.env_size + 1,
                                  torch.full_like(self.env_size, self.max_steps))

    def sample(self, batch_size, seq_len):
        """Sample [B, T] sequences — fully vectorized."""
        valid_envs = (self.env_size >= seq_len).nonzero(as_tuple=True)[0]
        if len(valid_envs) == 0:
            return None

        env_idx = valid_envs[torch.randint(len(valid_envs), (batch_size,), device=self.device)]
        max_start = self.env_size[env_idx] - seq_len
        start_idx = (torch.rand(batch_size, device=self.device) * (max_start.float() + 1)).long().clamp(min=0)

        offsets = torch.arange(seq_len, device=self.device).unsqueeze(0)
        indices = (start_idx.unsqueeze(1) + offsets) % self.max_steps

        e = env_idx.unsqueeze(1).expand(-1, seq_len)
        obs_seq = self.obs[e, indices]
        code_seq = self.post_codes[e, indices]
        exec_seq = self.exec_codes[e, indices]
        phase_seq = self.phase_ids[e, indices]
        done_seq = self.dones[e, indices]
        feat_seq = self.task_features[e, indices] if self.has_tf else None

        done_cumsum = done_seq.long().cumsum(dim=1)
        mask = (done_cumsum == 0) | (done_seq & (done_cumsum == 1))

        return {
            "obs": obs_seq, "codes": code_seq, "exec_codes": exec_seq,
            "phases": phase_seq, "mask": mask, "features": feat_seq,
            "dones": done_seq,
        }


def get_motion_files(path):
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*.npz")))
    return [path]


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if args_cli.motion_path:
        env_cfg.commands.motion.motion_files = get_motion_files(args_cli.motion_path)
        if hasattr(env_cfg.commands.motion, "strike_motion_files"):
            env_cfg.commands.motion.strike_motion_files = env_cfg.commands.motion.motion_files

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)
    base_env = env.unwrapped
    device = base_env.device

    # Load teacher
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(resume)
    teacher_policy = runner.get_inference_policy(device=device)
    print(f"[INFO] Teacher: {resume}")

    # Load frozen VQ model
    ckpt = torch.load(args_cli.vq_model, map_location=device, weights_only=False)
    assert ckpt.get("prior_type") == "vq"
    num_codes = int(ckpt.get("num_codes", 16))
    code_hold = int(ckpt.get("code_hold", 1))
    obs_dim = int(ckpt["obs_dim"])
    action_dim = int(ckpt["action_dim"])
    z_dim = int(ckpt["z_dim"])
    hidden_dims = list(ckpt["hidden_dims"])
    decoder_obs_mode = ckpt.get("decoder_obs_mode", "full")
    use_tf = decoder_obs_mode == "task_features"

    # Create model with markov_prior=True (new prior, rest from checkpoint)
    model = LatentActionModel(
        obs_dim=obs_dim, action_dim=action_dim, z_dim=z_dim,
        hidden_dims=hidden_dims, decoder_obs_mode=decoder_obs_mode,
        prior_type="vq", num_codes=num_codes,
        commitment_weight=float(ckpt.get("commitment_weight", 0.25)),
        markov_prior=True,
    ).to(device)

    # Load encoder/decoder/codebook from checkpoint (prior weights won't match)
    old_state = ckpt["model_state_dict"]
    new_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []
    for k, v in old_state.items():
        if k.startswith("prior."):
            skipped_keys.append(k)
            continue
        if k in new_state and new_state[k].shape == v.shape:
            new_state[k] = v
            loaded_keys.append(k)
        else:
            skipped_keys.append(k)
    model.load_state_dict(new_state)
    print(f"[INFO] Loaded {len(loaded_keys)} keys, skipped {len(skipped_keys)} (prior re-initialized)")

    # Freeze encoder, decoder, codebook
    for name, param in model.named_parameters():
        if not name.startswith("prior."):
            param.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Markov prior: {trainable} trainable / {total} total params")
    print(f"[INFO] K={num_codes}, code_hold={code_hold}, obs_mode={decoder_obs_mode}")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args_cli.lr
    )

    # Sequence buffer (smaller for DAgger — want on-policy data to dominate)
    cmd = base_env.command_manager.get_term("motion")
    buf = SequenceBuffer(
        args_cli.num_envs, 2000, obs_dim, device,
        task_features_dim=TASK_FEATURES_DIM if use_tf else 0,
    )

    # DAgger schedule: beta = prob of using posterior code for stepping
    dagger_betas = [0.9, 0.7, 0.5, 0.35, 0.2]
    iters_per_round = args_cli.num_iters // len(dagger_betas)
    start_token = model.prior.start_token

    # VQ hold state — track both posterior (label) and executed (prev_code)
    N = args_cli.num_envs
    hold_ctr = torch.zeros(N, dtype=torch.long, device=device)
    held_post_code = torch.zeros(N, dtype=torch.long, device=device)
    held_exec_code = torch.full((N,), start_token, dtype=torch.long, device=device)
    held_zq = torch.zeros(N, z_dim, device=device)

    obs, _ = env.get_observations()
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"  Markov VQ Prior DAgger Training")
    print(f"  Iters: {args_cli.num_iters}, steps/iter: {args_cli.steps_per_iter}")
    print(f"  seq_len: {args_cli.seq_len}, batch: {args_cli.batch_size}")
    print(f"  strike_ce_weight: {args_cli.strike_ce_weight}")
    print(f"  DAgger betas: {dagger_betas}")
    print(f"  iters/round: {iters_per_round}, buffer: 2000 steps/env")
    print(f"{'='*60}\n")

    for iteration in range(args_cli.num_iters):
        # ── DAgger beta schedule ──
        round_idx = min(iteration // iters_per_round, len(dagger_betas) - 1)
        beta = dagger_betas[round_idx]

        # ═══════════ Phase 1: DAgger Rollout ═══════════
        # Always rollout — DAgger needs fresh on-policy data
        model.eval()
        iter_episodes = 0
        for step in range(args_cli.steps_per_iter):
            with torch.no_grad():
                obs_v3 = obs
                a_teacher = teacher_policy(obs_v3)
                tf = compute_ball_foot_relation(base_env) if use_tf else None
                dec_obs = model.select_decoder_obs(obs_v3, task_features=tf)

                # Always compute posterior code (= label)
                z_e = model.encoder(dec_obs, a_teacher)
                needs_update = (hold_ctr % code_hold == 0)

                if needs_update.any():
                    z_q_post, post_code, _ = model.codebook.quantize(z_e)
                    held_post_code[needs_update] = post_code[needs_update]

                    # Prior prediction using EXECUTED prev_code (all envs)
                    prior_logits = model.prior(dec_obs, prev_code=held_exec_code)
                    prior_code = prior_logits.argmax(dim=-1)  # [N]

                    # DAgger mix: per-env coin flip
                    use_posterior = (torch.rand(N, device=device) < beta)
                    choose_post = needs_update & use_posterior
                    choose_prior = needs_update & (~use_posterior)

                    # Update executed code
                    held_exec_code[choose_post] = post_code[choose_post]
                    held_exec_code[choose_prior] = prior_code[choose_prior]

                    # Update zq for decoding
                    held_zq[needs_update] = model.codebook.lookup(
                        held_exec_code[needs_update])

                # Decode with the executed code
                actions = model.decoder(dec_obs, held_zq)

            # Get phase
            ref_phase = cmd.event_phase_id.long()

            # Step environment
            obs, _, dones, _ = env.step(actions.clone())
            hold_ctr += 1

            # Store: obs, posterior_code (label), exec_code (for prev_code)
            buf.add(obs_v3, held_post_code, held_exec_code, ref_phase, dones,
                    task_features=tf)

            # Handle resets
            if dones.any():
                if hasattr(runner.alg.policy, "reset"):
                    runner.alg.policy.reset(dones)
                done_ids = dones.nonzero(as_tuple=True)[0]
                hold_ctr[done_ids] = 0
                held_post_code[done_ids] = 0
                held_exec_code[done_ids] = start_token  # START on reset
                held_zq[done_ids] = 0
                iter_episodes += dones.sum().item()

        # ═══════════ Phase 2: Train Markov prior ═══════════
        if buf.size < args_cli.batch_size * args_cli.seq_len:
            print(f"  Iter {iteration+1}: filling buffer ({buf.size} samples)")
            continue

        model.prior.train()
        iter_ce = 0.0
        iter_recon = 0.0
        iter_switch_pen = 0.0

        for _ in range(args_cli.updates_per_iter):
            data = buf.sample(args_cli.batch_size, args_cli.seq_len)
            if data is None:
                continue

            B, T = data["codes"].shape
            mask = data["mask"]  # [B, T]
            target_codes = data["codes"]  # [B, T] posterior codes (LABEL)
            exec_codes = data["exec_codes"]  # [B, T] executed codes (for prev_code)
            phases = data["phases"]  # [B, T]

            # Build prev_code from EXECUTED codes (not posterior!)
            # This is the DAgger key: prev_code reflects what was actually done
            prev_codes = torch.full_like(exec_codes, start_token)
            prev_codes[:, 1:] = exec_codes[:, :-1]
            # Reset prev_code to START after done boundaries (vectorized)
            dones_seq = data["dones"]  # [B, T]
            done_shifted = torch.zeros_like(dones_seq)
            done_shifted[:, 1:] = dones_seq[:, :-1]
            prev_codes[done_shifted] = start_token

            # Identify code-hold boundary frames (based on target/posterior)
            is_boundary = torch.ones(B, T, dtype=torch.bool, device=device)
            is_boundary[:, 1:] = (target_codes[:, 1:] != target_codes[:, :-1])
            is_boundary[done_shifted] = True

            # ── Mixed CE: boundary_weight=1.0, hold_weight=0.2 ──
            # Batch all T frames into one forward pass: [B,T] → [B*T]
            HOLD_WEIGHT = 0.2

            # Flatten [B, T] → [B*T]
            obs_flat = data["obs"].reshape(B * T, -1)         # [B*T, obs_dim]
            pc_flat = prev_codes.reshape(B * T)                # [B*T]
            tgt_flat = target_codes.reshape(B * T)             # [B*T]
            mask_flat = mask.reshape(B * T)                    # [B*T]
            bnd_flat = (is_boundary & mask).reshape(B * T)     # [B*T]
            hold_flat = (~is_boundary & mask).reshape(B * T)   # [B*T]
            phase_flat = phases.reshape(B * T)                 # [B*T]

            # Build dec_obs for all frames at once
            if data["features"] is not None:
                feat_flat = data["features"].reshape(B * T, -1)
                dec_obs_flat = model.select_decoder_obs(obs_flat, task_features=feat_flat)
            else:
                dec_obs_flat = model.select_decoder_obs(obs_flat)

            # Single batched forward pass
            logits_flat = model.prior(dec_obs_flat, prev_code=pc_flat)  # [B*T, K]

            # CE loss
            ce_flat = F.cross_entropy(logits_flat, tgt_flat, reduction="none")  # [B*T]

            # Frame weight: boundary=1.0, hold=0.2
            frame_w = torch.where(bnd_flat, torch.ones(B * T, device=device),
                                  torch.full((B * T,), HOLD_WEIGHT, device=device))
            # Phase weight on top
            strike_flat = (phase_flat == PHASE_PRESTRIKE) | (phase_flat == PHASE_STRIKE)
            frame_w[strike_flat] *= args_cli.strike_ce_weight

            n_valid = mask_flat.float().sum().clamp(min=1)
            ce_total = (ce_flat * frame_w * mask_flat.float()).sum() / n_valid



            # Track agreement
            pred_flat = logits_flat.detach().argmax(-1)
            n_correct_bnd = (pred_flat == tgt_flat)[bnd_flat].sum().item()
            n_total_bnd = bnd_flat.sum().item()
            n_correct_hold = (pred_flat == tgt_flat)[hold_flat].sum().item()
            n_total_hold = hold_flat.sum().item()

            loss = ce_total

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.prior.parameters(), max_norm=1.0)
            optimizer.step()

            iter_ce += ce_total.item()
            iter_recon += n_correct_bnd / max(n_total_bnd, 1)
            iter_switch_pen += n_correct_hold / max(n_total_hold, 1)

        avg_ce = iter_ce / args_cli.updates_per_iter
        avg_bnd = iter_recon / args_cli.updates_per_iter * 100
        avg_hold = iter_switch_pen / args_cli.updates_per_iter * 100
        elapsed = time.time() - t0
        print(f"  Iter {iteration+1:3d}/{args_cli.num_iters} | β={beta:.2f} | "
              f"CE={avg_ce:.3f} bnd={avg_bnd:.1f}% hold={avg_hold:.1f}% | "
              f"buf={buf.size} ep={iter_episodes} | {elapsed:.0f}s")

        # ═══════════ Phase 3: Eval ═══════════
        if (iteration + 1) % args_cli.eval_interval == 0:
            model.eval()
            # Quick eval: prior rollout with Markov state
            eval_kicks = 0
            eval_eps = 0
            eval_c15_approach = 0
            eval_approach_frames = 0
            target_eps = 50

            eval_obs, _ = env.get_observations()
            eval_prev_code = torch.full((args_cli.num_envs,), model.prior.start_token,
                                        dtype=torch.long, device=device)
            eval_hold_ctr = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
            eval_held_code = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
            eval_held_zq = torch.zeros(args_cli.num_envs, z_dim, device=device)
            ball_contacted = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)

            while eval_eps < target_eps:
                with torch.no_grad():
                    tf = compute_ball_foot_relation(base_env) if use_tf else None
                    dec_obs = model.select_decoder_obs(eval_obs, task_features=tf)
                    needs = (eval_hold_ctr % code_hold == 0)
                    if needs.any():
                        logits = model.prior(dec_obs, prev_code=eval_prev_code)
                        new_code = logits[needs].argmax(dim=-1)
                        eval_held_code[needs] = new_code
                        eval_held_zq[needs] = model.codebook.lookup(new_code)
                        eval_prev_code[needs] = new_code
                    action = model.decoder(dec_obs, eval_held_zq)

                # Track c15 during approach
                ref_ph = cmd.event_phase_id.long()
                approach_mask = (ref_ph == PHASE_APPROACH)
                eval_approach_frames += approach_mask.sum().item()
                eval_c15_approach += ((eval_held_code == 15) & approach_mask).sum().item()

                eval_obs, _, dones, _ = env.step(action.clone())
                eval_hold_ctr += 1

                bvel = base_env.scene["soccer_ball"].data.root_lin_vel_w[:, :2]
                ball_contacted |= (torch.norm(bvel, dim=-1) > 0.5)

                if dones.any():
                    if hasattr(runner.alg.policy, "reset"):
                        runner.alg.policy.reset(dones)
                    for i in dones.nonzero(as_tuple=True)[0]:
                        if eval_eps >= target_eps:
                            break
                        eval_eps += 1
                        if ball_contacted[i]:
                            eval_kicks += 1
                        ball_contacted[i] = False
                        eval_hold_ctr[i] = 0
                        eval_held_code[i] = 0
                        eval_held_zq[i] = 0
                        eval_prev_code[i] = model.prior.start_token

            kick_pct = eval_kicks / max(eval_eps, 1) * 100
            c15_app_pct = eval_c15_approach / max(eval_approach_frames, 1) * 100
            print(f"  >>> EVAL: Kick={kick_pct:.0f}% ({eval_kicks}/{eval_eps}) | "
                  f"c15_in_approach={c15_app_pct:.1f}%")

        # ═══════════ Save ═══════════
        if (iteration + 1) % args_cli.eval_interval == 0 or iteration == args_cli.num_iters - 1:
            save_dict = {
                "model_state_dict": model.state_dict(),
                "obs_dim": obs_dim, "action_dim": action_dim, "z_dim": z_dim,
                "hidden_dims": hidden_dims, "decoder_obs_mode": decoder_obs_mode,
                "prior_type": "vq", "num_codes": num_codes,
                "commitment_weight": float(ckpt.get("commitment_weight", 0.25)),
                "code_hold": code_hold, "markov_prior": True,
                "strike_ce_weight": args_cli.strike_ce_weight,
            }
            os.makedirs(os.path.dirname(args_cli.output_path), exist_ok=True)
            torch.save(save_dict, args_cli.output_path)
            print(f"  [SAVE] {args_cli.output_path}")

    print(f"\n[DONE] Total time: {time.time()-t0:.0f}s")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
