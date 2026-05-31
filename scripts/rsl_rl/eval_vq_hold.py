"""Code-Hold diagnostic: test VQ posterior with different code hold durations.

For each hold value H, the encoder re-selects a code every H frames.
Between re-selections, the same z_q is reused.

This answers: is the Fall% problem caused by code switching jitter,
or by the codebook primitives themselves being bad?

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/rsl_rl/eval_vq_hold.py \
        --task Anchor-CG-Kick-G1-Soccer-RNN-v0 \
        --motion_path motions/Video_hmr4d_seed \
        --latent_model models/latent_v2/online_distill_vq_k16.pt \
        --load_run "2026-04-28_12-15-12_cg_v3_softmask" \
        --checkpoint model_12000.pt \
        --hold_values 1 2 4 8 \
        --num_envs 32 --num_episodes 100 --device cuda:0 --headless
"""

import argparse
import glob
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="VQ code-hold diagnostic.")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--task", type=str, default="Anchor-CG-Kick-G1-Soccer-RNN-v0")
parser.add_argument("--motion_path", type=str, required=True)
parser.add_argument("--latent_model", type=str, required=True,
                    help="Path to VQ checkpoint (.pt)")
parser.add_argument("--hold_values", type=int, nargs="+", default=[1, 2, 4, 8],
                    help="Code hold durations to test (1 = every frame, default behavior)")
parser.add_argument("--num_episodes", type=int, default=100)

import cli_args  # isort: skip
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = False
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import numpy as np

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner

import soccer.tasks  # noqa: F401
from latent_v2_models import LatentActionModel
from compute_task_features import compute_ball_foot_relation


def get_checkpoint_path(log_root, run_dir, checkpoint):
    """Resolve checkpoint path (same logic as train script)."""
    if os.path.isabs(run_dir):
        run_path = run_dir
    else:
        run_path = os.path.join(log_root, run_dir)
    if checkpoint == "model_latest.pt":
        ckpts = sorted(glob.glob(os.path.join(run_path, "model_*.pt")))
        if ckpts:
            return ckpts[-1]
    return os.path.join(run_path, checkpoint)


def load_vq_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = LatentActionModel(
        obs_dim=int(ckpt["obs_dim"]),
        action_dim=int(ckpt["action_dim"]),
        z_dim=int(ckpt["z_dim"]),
        hidden_dims=list(ckpt["hidden_dims"]),
        decoder_obs_mode=ckpt.get("decoder_obs_mode", "full"),
        prior_type=ckpt.get("prior_type", "vq"),
        num_codes=int(ckpt.get("num_codes", 16)),
        commitment_weight=float(ckpt.get("commitment_weight", 0.25)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


def get_motion_files(motion_path):
    if os.path.isfile(motion_path):
        return [motion_path]
    if os.path.isdir(motion_path):
        return sorted(glob.glob(os.path.join(motion_path, "*.npz")))
    raise ValueError(f"Invalid: {motion_path}")


def run_hold_eval(model, env_wrapped, teacher_policy, hold_H, num_episodes, device, use_task_features):
    """Run posterior rollout with code held for H frames."""
    N = env_wrapped.num_envs
    obs_v3, _ = env_wrapped.get_observations()

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

    # Code hold state: current z_q and frame counter per env
    held_zq = torch.zeros(N, model.z_dim, device=device)
    held_code = torch.zeros(N, dtype=torch.long, device=device)
    hold_counter = torch.zeros(N, dtype=torch.long, device=device)  # counts up

    code_switches = 0
    code_total = 0
    prev_code = None

    while episodes < num_episodes:
        with torch.no_grad():
            a_teacher = teacher_policy(obs_v3)
            tf = compute_ball_foot_relation(env_wrapped.unwrapped) if use_task_features else None
            dec_obs = model.select_decoder_obs(obs_v3, task_features=tf)

            # Encode
            z_e = model.encoder(dec_obs, a_teacher)

            # Only re-quantize every H frames (or on first frame of episode)
            needs_update = (hold_counter % hold_H == 0)
            if needs_update.any():
                z_q_new, code_new, _ = model.codebook.quantize(z_e)
                held_zq[needs_update] = z_q_new[needs_update]
                held_code[needs_update] = code_new[needs_update]

            # Track switching (only on frames where code could change)
            if prev_code is not None:
                switched = (held_code != prev_code)
                code_switches += switched.sum().item()
                code_total += N
            prev_code = held_code.clone()

            # Decode from held code
            action = model.decoder(dec_obs, held_zq)

        obs_v3, _, dones, _ = env_wrapped.step(action)
        ep_len += 1
        hold_counter += 1

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
                fell = (ep_len[i].item() < 100)

                if kicked:
                    kicks += 1
                    ball_speeds.append(max_bspd[i].item())
                if fell:
                    falls += 1

                # 4-way outcome
                if kicked and not fell:
                    clean_success += 1
                elif kicked and fell:
                    late_fallback += 1
                elif not kicked and not fell:
                    empty_swing += 1
                else:
                    early_fall += 1

                # Reset per-env state
                ep_len[i] = 0
                ball_contacted[i] = False
                max_bspd[i] = 0
                hold_counter[i] = 0  # reset hold counter on episode boundary

    switch_rate = code_switches / max(code_total, 1)

    return {
        "hold_H": hold_H,
        "kick_pct": kicks / max(episodes, 1) * 100,
        "fall_pct": falls / max(episodes, 1) * 100,
        "avg_bspd": sum(ball_speeds) / max(len(ball_speeds), 1),
        "clean_success": clean_success,
        "late_fallback": late_fallback,
        "empty_swing": empty_swing,
        "early_fall": early_fall,
        "switch_rate": switch_rate,
        "episodes": episodes,
    }


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
    env_cfg.commands.motion.motion_files = get_motion_files(args_cli.motion_path)
    device = env_cfg.sim.device

    # Load VQ model
    model, ckpt = load_vq_model(args_cli.latent_model, device)
    use_task_features = ckpt.get("decoder_obs_mode", "full") == "task_features"
    num_codes = ckpt.get("num_codes", 16)
    print(f"[INFO] VQ model: K={num_codes}, z_dim={model.z_dim}, "
          f"best_kick={ckpt.get('best_kick_pct', '?')}%, "
          f"iter={ckpt.get('iteration', '?')}")

    # Load teacher
    env = gym.make(args_cli.task, cfg=env_cfg)
    env_wrapped = RslRlVecEnvWrapper(env)

    log_root = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    resume_path = get_checkpoint_path(
        os.path.abspath(log_root), agent_cfg.load_run, agent_cfg.load_checkpoint
    )
    print(f"[INFO] Loading teacher from: {resume_path}")
    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(resume_path)
    teacher_policy = runner.get_inference_policy(device=device)

    # Run hold evaluations
    print(f"\n{'='*70}")
    print(f"  VQ Code-Hold Diagnostic")
    print(f"  Hold values: {args_cli.hold_values}")
    print(f"  Episodes per hold: {args_cli.num_episodes}")
    print(f"{'='*70}")

    all_results = []
    for H in args_cli.hold_values:
        print(f"\n  Running hold={H} ...")
        result = run_hold_eval(
            model, env_wrapped, teacher_policy,
            hold_H=H, num_episodes=args_cli.num_episodes,
            device=device, use_task_features=use_task_features,
        )
        all_results.append(result)
        print(f"    Kick={result['kick_pct']:.0f}% Fall={result['fall_pct']:.0f}% "
              f"BSpd={result['avg_bspd']:.2f} switch={result['switch_rate']:.3f}")
        print(f"    clean={result['clean_success']} late_fall={result['late_fallback']} "
              f"empty={result['empty_swing']} early_fall={result['early_fall']}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY: Code-Hold Results")
    print(f"{'='*70}")
    print(f"  {'Hold':>6} | {'Kick%':>6} | {'Fall%':>6} | {'BSpd':>6} | {'Switch':>7} | {'Clean':>5} | {'LateFall':>8} | {'Empty':>5} | {'EarlyFall':>9}")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*5}-+-{'-'*8}-+-{'-'*5}-+-{'-'*9}")
    for r in all_results:
        print(f"  {r['hold_H']:>6} | {r['kick_pct']:>5.0f}% | {r['fall_pct']:>5.0f}% | "
              f"{r['avg_bspd']:>6.2f} | {r['switch_rate']:>7.3f} | "
              f"{r['clean_success']:>5} | {r['late_fallback']:>8} | "
              f"{r['empty_swing']:>5} | {r['early_fall']:>9}")
    print(f"{'='*70}")

    # Interpretation
    h1 = all_results[0]  # hold=1 (baseline)
    best = min(all_results, key=lambda r: r["fall_pct"])
    if best["fall_pct"] < h1["fall_pct"] - 10:
        print(f"\n  >>> DIAGNOSIS: Code switching IS the problem!")
        print(f"      hold=1 Fall={h1['fall_pct']:.0f}% -> hold={best['hold_H']} Fall={best['fall_pct']:.0f}%")
        print(f"      Next step: Train sticky/option prior with hold duration.")
    else:
        print(f"\n  >>> DIAGNOSIS: Code primitives themselves are weak.")
        print(f"      Holding codes did NOT significantly reduce Fall%.")
        print(f"      Next step: Fix codebook training (Gumbel-Softmax, phase-aware, etc.)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
