"""Measure support foot contact force distribution during actual policy execution."""
import argparse, sys, os, glob

from isaaclab.app import AppLauncher
import cli_args

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--motion_path", type=str, default=None)
parser.add_argument("--motion_file", type=str, default=None)
parser.add_argument("--measure_steps", type=int, default=2000)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnvCfg, multi_agent_to_single_agent, DirectMARLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import soccer.tasks  # noqa

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    import gymnasium as gym
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.motion_path:
        mfiles = sorted(glob.glob(os.path.join(args_cli.motion_path, "*.npz")))
        env_cfg.commands.motion.motion_files = mfiles

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading: {resume}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base_env = env.unwrapped
    device = base_env.device

    # Get sensor
    foot_contact = base_env.scene.sensors.get("foot_contact") if isinstance(base_env.scene.sensors, dict) else getattr(base_env.scene.sensors, "foot_contact", None)
    if foot_contact is None:
        print("[FAIL] foot_contact sensor not found!")
        env.close()
        return

    ids, _ = foot_contact.find_bodies(["left_ankle_roll_link"], preserve_order=True)
    support_idx = ids[0]
    print(f"[OK] left_ankle_roll_link sensor index: {support_idx}")

    # Collect force data
    all_forces = []
    obs, _ = env.get_observations()

    for step in range(args_cli.measure_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        forces = foot_contact.data.net_forces_w
        if forces is not None and forces.numel() > 0:
            fz = forces[:, support_idx, 2].clamp(min=0.0).cpu().numpy()
            all_forces.extend(fz.tolist())

        if step % 500 == 499:
            print(f"  Step {step+1}/{args_cli.measure_steps}")

    all_forces = np.array(all_forces)

    print("\n" + "="*60)
    print("  SUPPORT FOOT CONTACT FORCE DISTRIBUTION")
    print("="*60)
    print(f"  Total samples: {len(all_forces)}")
    print(f"  Min:    {np.min(all_forces):.1f} N")
    print(f"  Max:    {np.max(all_forces):.1f} N")
    print(f"  Mean:   {np.mean(all_forces):.1f} N")
    print(f"  Median: {np.median(all_forces):.1f} N")
    print(f"  Std:    {np.std(all_forces):.1f} N")
    print()

    thresholds = [1, 2, 5, 10, 15, 20, 30, 50, 100]
    print("  Threshold → % samples above:")
    for t in thresholds:
        pct = np.mean(all_forces > t) * 100
        marker = " ← current" if t == 20 else ""
        print(f"    > {t:3d} N: {pct:5.1f}%{marker}")

    print("="*60)
    env.close()

main()
simulation_app.close()
