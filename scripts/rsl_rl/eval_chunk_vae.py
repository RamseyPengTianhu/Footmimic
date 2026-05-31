"""Evaluate Chunk VAE in Isaac Gym closed-loop rollout.

Modes:
  --mode posterior_teacher:
      Maintains a sliding window of (obs, teacher_action) of size H.
      At each step, encodes the window to get z, decodes it, and executes
      the *last* action of the reconstructed chunk. Tests if the Decoder
      can physically control the robot stably.
  --mode prior_receding:
      Maintains a history window of observations of size K+1.
      At each step, queries the History Prior for z, decodes a future chunk
      of length H, and executes the *first* action of the chunk.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/eval_chunk_vae.py \\
    --task Anchor-CG-Kick-G1-Soccer-RNN-v0 \\
    --motion_path motions/Video_hmr4d_seed \\
    --teacher_run 2026-04-28_12-15-12_cg_v3_softmask \\
    --teacher_ckpt model_12000.pt \\
    --vae_ckpt models/chunk_vae/chunk_vae_H4_z32_hist8.pt \\
    --mode prior_receding \\
    --num_envs 64
"""
import argparse
import os
import sys
import time
import torch

parser = argparse.ArgumentParser(description="Evaluate Chunk VAE.")
parser.add_argument("--task", type=str, default="Anchor-CG-Kick-G1-Soccer-RNN-v0")
parser.add_argument("--motion_path", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--teacher_run", type=str, required=True)
parser.add_argument("--teacher_ckpt", type=str, default="model_12000.pt")
parser.add_argument("--vae_ckpt", type=str, required=True)
parser.add_argument("--mode", type=str, choices=["posterior_teacher", "prior_receding"], required=True)

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
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import soccer.tasks  # noqa: F401
from soccer.utils.my_on_policy_runner import MotionOnPolicyRunner
from compute_task_features import compute_ball_foot_relation
from chunk_vae_models import ChunkVAE
import glob

def get_motion_files(path):
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*.npz")))
    return [path]

def build_decoder_obs(obs_v3, task_features, include_phase):
    proprio = torch.cat((obs_v3[:, 58:61], obs_v3[:, 64:]), dim=-1)
    if include_phase:
        return torch.cat((proprio, task_features), dim=-1)
    else:
        return torch.cat((proprio, task_features[:, :22]), dim=-1)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume = get_checkpoint_path(log_root, args_cli.teacher_run, args_cli.teacher_ckpt)

    if args_cli.motion_path:
        env_cfg.commands.motion.motion_files = get_motion_files(args_cli.motion_path)
        if hasattr(env_cfg.commands.motion, "strike_motion_files"):
            env_cfg.commands.motion.strike_motion_files = env_cfg.commands.motion.motion_files

    render_mode = None if getattr(args_cli, "headless", False) else "rgb_array"
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)
    base_env = env.unwrapped
    device = base_env.device

    # 1. Load Teacher
    runner = MotionOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(resume)
    teacher_policy = runner.get_inference_policy(device=device)
    print(f"[INFO] Loaded Teacher: {resume}")

    # 2. Load VAE
    print(f"[INFO] Loading VAE: {args_cli.vae_ckpt}")
    vae_ckpt = torch.load(args_cli.vae_ckpt, map_location=device, weights_only=False)
    meta = vae_ckpt["metadata"]
    H = vae_ckpt["chunk_len"]
    K = vae_ckpt.get("history_len", 0)
    
    vae = ChunkVAE(
        obs_dim=vae_ckpt["obs_dim"],
        action_dim=vae_ckpt["action_dim"],
        chunk_len=H,
        z_dim=vae_ckpt["z_dim"],
        hidden_dims=vae_ckpt["hidden_dims"],
        history_len=K,
    ).to(device)
    vae.load_state_dict(vae_ckpt["model_state_dict"])
    vae.eval()
    
    act_mean = vae_ckpt["act_mean"].to(device)
    act_std = vae_ckpt["act_std"].to(device)
    print(f"[INFO] VAE Mode: {args_cli.mode} | H={H}, K={K}, z={vae_ckpt['z_dim']}")

    # 3. Setup Buffers
    num_envs = args_cli.num_envs
    obs_dim = vae_ckpt["obs_dim"]
    action_dim = vae_ckpt["action_dim"]

    # History buffers
    if args_cli.mode == "posterior_teacher":
        buf_len = H
        obs_buf = torch.zeros(num_envs, buf_len, obs_dim, device=device)
        act_buf = torch.zeros(num_envs, buf_len, action_dim, device=device)
    else:
        buf_len = K + 1
        obs_buf = torch.zeros(num_envs, buf_len, obs_dim, device=device)

    obs, _ = env.get_observations()

    # Fill initial buffers by repeating the first observation
    with torch.no_grad():
        tf = compute_ball_foot_relation(base_env)
        dec_obs = build_decoder_obs(obs, tf, include_phase=meta.get("include_phase", False))
        
        obs_buf[:] = dec_obs.unsqueeze(1).repeat(1, buf_len, 1)
        if args_cli.mode == "posterior_teacher":
            a_teach = teacher_policy(obs)
            a_teach_norm = (a_teach - act_mean) / act_std
            act_buf[:] = a_teach_norm.unsqueeze(1).repeat(1, buf_len, 1)

    print("[INFO] Starting Rollout...")
    
    kick_count = 0
    total_steps = 0

    while simulation_app.is_running():
        with torch.no_grad():
            # Update current observation
            tf = compute_ball_foot_relation(base_env)
            dec_obs = build_decoder_obs(obs, tf, include_phase=meta.get("include_phase", False))
            
            if args_cli.mode == "posterior_teacher":
                # Get teacher intended action
                a_teach = teacher_policy(obs)
                a_teach_norm = (a_teach - act_mean) / act_std
                
                # Shift buffers and append
                obs_buf = torch.roll(obs_buf, shifts=-1, dims=1)
                obs_buf[:, -1, :] = dec_obs
                
                act_buf = torch.roll(act_buf, shifts=-1, dims=1)
                act_buf[:, -1, :] = a_teach_norm

                # Encode the past H frames to reconstruct them
                mu_e, _ = vae.encoder(obs_buf, act_buf)
                # The decoder reconstructs the chunk [t-H+1, ..., t]. 
                recon = vae.decoder(obs_buf[:, 0], mu_e)
                # Action at current step t is the LAST frame of the chunk
                a_exec_norm = recon[:, -1, :]
                
            elif args_cli.mode == "prior_receding":
                # Shift history buffer and append
                obs_buf = torch.roll(obs_buf, shifts=-1, dims=1)
                obs_buf[:, -1, :] = dec_obs
                
                # Query Prior for future chunk
                chunk_act_norm = vae.act_prior_chunk(obs_t=dec_obs, sample=False, obs_history=obs_buf)
                
                # Receding horizon: execute the FIRST action of the predicted future chunk
                a_exec_norm = chunk_act_norm[:, 0, :]

            # Denormalize
            a_exec = a_exec_norm * act_std + act_mean

        obs, _, dones, _ = env.step(a_exec)
        total_steps += 1
        
        # Reset teacher policy RNN states if episode done
        if dones.any():
            if hasattr(runner.alg.policy, "reset"):
                runner.alg.policy.reset(dones)
            
            # Reset buffers for done envs
            done_idx = dones.nonzero(as_tuple=True)[0]
            if len(done_idx) > 0:
                tf_done = compute_ball_foot_relation(base_env)
                dec_obs_done = build_decoder_obs(obs, tf_done, include_phase=meta.get("include_phase", False))
                
                obs_buf[done_idx] = dec_obs_done[done_idx].unsqueeze(1).repeat(1, buf_len, 1)
                if args_cli.mode == "posterior_teacher":
                    a_teach_done = teacher_policy(obs)
                    a_teach_norm_done = (a_teach_done - act_mean) / act_std
                    act_buf[done_idx] = a_teach_norm_done[done_idx].unsqueeze(1).repeat(1, buf_len, 1)

        # Basic tracking just to show it's running
        cmd = base_env.command_manager.get_term("motion")
        phase = cmd.event_phase_id
        kick_count += (phase == 2).sum().item()
        
        if total_steps % 100 == 0:
            print(f"Step {total_steps} | Accum Kicks: {kick_count}")


if __name__ == "__main__":
    main()
    simulation_app.close()
