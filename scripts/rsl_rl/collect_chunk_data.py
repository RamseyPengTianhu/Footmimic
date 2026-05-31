"""Collect teacher rollout data with task features for Chunk VAE training.

Rolls out the teacher policy in Isaac Gym and saves:
  - decoder_obs: proprio(99D) + ball-foot spatial features(22D) = 121D
  - actions: teacher actions (29D)
  - dones: episode boundaries
  - phase_ids: geometric phase (approach/prestrike/strike/follow)

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/collect_chunk_data.py \\
        --task Anchor-CG-Kick-G1-Soccer-RNN-v0 \\
        --motion_path motions/Video_hmr4d_seed \\
        --load_run 2026-04-28_12-15-12_cg_v3_softmask \\
        --checkpoint model_12000.pt \\
        --num_envs 256 --num_steps 4000 \\
        --output_path data/teacher_manifold/chunk_vae_task22.pt \\
        --device cuda:0 --headless
"""
from __future__ import annotations
import argparse, os, sys, glob, time

parser = argparse.ArgumentParser(description="Collect teacher data with task features.")
parser.add_argument("--task", type=str, default="Anchor-CG-Kick-G1-Soccer-RNN-v0")
parser.add_argument("--motion_path", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_steps", type=int, default=4000,
                    help="Steps per env to collect.")
parser.add_argument("--output_path", type=str,
                    default="data/teacher_manifold/chunk_vae_task22.pt")
parser.add_argument("--include_phase", action="store_true",
                    help="Include 4D phase embedding (26D total). Default: 22D spatial only.")

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
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import soccer.tasks  # noqa: F401
from soccer.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner
from compute_task_features import compute_ball_foot_relation, TASK_FEATURES_DIM


def get_motion_files(path):
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*.npz")))
    return [path]


def build_decoder_obs(obs_v3, task_features, include_phase):
    """Build deployable obs: proprio(99D) + spatial features (22D or 26D)."""
    proprio = torch.cat((obs_v3[:, 58:61], obs_v3[:, 64:]), dim=-1)  # 99D
    if include_phase:
        return torch.cat((proprio, task_features), dim=-1)  # 99+26=125D
    else:
        # Spatial only: first 22D of task_features (skip last 4D phase)
        return torch.cat((proprio, task_features[:, :22]), dim=-1)  # 99+22=121D


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

    cmd = base_env.command_manager.get_term("motion")

    # Determine obs dim
    obs, _ = env.get_observations()
    obs_v3_dim = obs.shape[-1]
    tf_sample = compute_ball_foot_relation(base_env)
    dec_obs_sample = build_decoder_obs(obs, tf_sample, args_cli.include_phase)
    dec_obs_dim = dec_obs_sample.shape[-1]
    action_dim = 29  # G1 joint actions

    N = args_cli.num_envs
    T = args_cli.num_steps
    total_frames = N * T

    print(f"[INFO] obs_v3_dim={obs_v3_dim}, dec_obs_dim={dec_obs_dim}, action_dim={action_dim}")
    print(f"[INFO] Collecting {T} steps x {N} envs = {total_frames} frames")
    print(f"[INFO] include_phase={args_cli.include_phase}")

    # Pre-allocate flat storage
    all_dec_obs = torch.zeros(total_frames, dec_obs_dim, device=device)
    all_actions = torch.zeros(total_frames, action_dim, device=device)
    all_dones = torch.zeros(total_frames, dtype=torch.bool, device=device)
    all_phases = torch.zeros(total_frames, dtype=torch.long, device=device)

    t0 = time.time()
    ptr = 0

    for step in range(T):
        with torch.no_grad():
            a_teacher = teacher_policy(obs)
            tf = compute_ball_foot_relation(base_env)
            dec_obs = build_decoder_obs(obs, tf, args_cli.include_phase)

        ref_phase = cmd.event_phase_id.long()

        # Store
        all_dec_obs[ptr:ptr + N] = dec_obs
        all_actions[ptr:ptr + N] = a_teacher
        all_phases[ptr:ptr + N] = ref_phase

        # Step
        obs, _, dones, _ = env.step(a_teacher.clone())
        all_dones[ptr:ptr + N] = dones.to(torch.bool)
        ptr += N

        # Handle resets
        if dones.any():
            if hasattr(runner.alg.policy, "reset"):
                runner.alg.policy.reset(dones)

        if (step + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  Step {step+1}/{T} | {ptr} frames | {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"\n[INFO] Collection done: {ptr} frames in {elapsed:.0f}s")

    # Save
    save_dict = {
        "decoder_obs": all_dec_obs.cpu(),
        "actions": all_actions.cpu(),
        "dones": all_dones.cpu(),
        "phase_ids": all_phases.cpu(),
        "metadata": {
            "dec_obs_dim": dec_obs_dim,
            "action_dim": action_dim,
            "num_envs": N,
            "num_steps": T,
            "include_phase": args_cli.include_phase,
            "teacher_run": str(resume),
            "motion_path": args_cli.motion_path,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.output_path)), exist_ok=True)
    torch.save(save_dict, args_cli.output_path)
    file_mb = os.path.getsize(args_cli.output_path) / 1e6
    print(f"[SAVED] {args_cli.output_path} ({file_mb:.0f} MB)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
