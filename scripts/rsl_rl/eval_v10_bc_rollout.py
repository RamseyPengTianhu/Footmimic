"""Evaluate v10 BC-pretrained MLP in simulation (rollout test).

Usage:
    python scripts/rsl_rl/eval_v10_bc_rollout.py \
        --task Anchor-CG-Kick-G1-Soccer-RNN-v0 \
        --motion_path motions/Video/ \
        --bc_checkpoint logs/rsl_rl/v10_bc/bc_pretrained.pt \
        --num_episodes 50 \
        --headless

Runs the BC-pretrained v10 MLP in the v3 env (same env as teacher rollout).
Uses V10ObsBuilder to compute 422D obs, feeds to MLP, steps env.

Reports: Kick%, BSpd, Fall%, episode length, per-phase stats.
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Eval v10 BC-pretrained policy.")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--task", type=str, default="Anchor-CG-Kick-G1-Soccer-RNN-v0")
parser.add_argument("--motion_path", type=str, required=True)
parser.add_argument("--bc_checkpoint", type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)

import cli_args  # isort: skip
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = False
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import glob
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import soccer.tasks  # noqa: F401
from soccer.tasks.tracking.mdp.event_conditioned_obs_builder import V10ObsBuilder


class V10MLPActor(nn.Module):
    """v10 MLP actor (must match train_v10_bc.py)."""
    def __init__(self, obs_dim, action_dim=29, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ELU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)


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
    """Evaluate BC policy via rollout."""
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
    env_cfg.commands.motion.motion_files = get_motion_files(args_cli.motion_path)
    device = env_cfg.sim.device

    # Load BC model
    print(f"[INFO] Loading BC checkpoint from: {args_cli.bc_checkpoint}")
    ckpt = torch.load(args_cli.bc_checkpoint, map_location=device, weights_only=False)
    model = V10MLPActor(
        obs_dim=ckpt["obs_dim"],
        action_dim=ckpt["action_dim"],
        hidden_dims=ckpt.get("hidden_dims", [512, 256, 128]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[INFO] BC model: epoch {ckpt.get('epoch', '?')}, "
          f"MSE={ckpt.get('best_loss', 0):.6f}, "
          f"obs_dim={ckpt['obs_dim']}, params={sum(p.numel() for p in model.parameters())}")

    # Create env (v3 env, same as teacher rollout)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env_wrapped = RslRlVecEnvWrapper(env)
    unwrapped_env = env.unwrapped

    # Initialize V10ObsBuilder
    command = unwrapped_env.command_manager.get_term("motion")
    num_joints = command.robot.data.joint_pos.shape[1]
    v10_builder = V10ObsBuilder(
        num_envs=args_cli.num_envs,
        num_joints=num_joints,
        device=device,
    )
    v10_builder.init_segment_bounds(command)

    # Get initial env obs (not used by MLP, just to init env)
    obs_v3, _ = env_wrapped.get_observations()

    # Metrics
    episodes_completed = 0
    total_kicks = 0
    total_falls = 0
    ball_speeds = []
    ep_lengths = []
    ep_length_counter = torch.zeros(args_cli.num_envs, device=device)

    # Track ball contact per env
    ball_contacted = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)

    print(f"\n[INFO] Evaluating {args_cli.num_episodes} episodes "
          f"({args_cli.num_envs} parallel envs)...")

    step_count = 0
    while episodes_completed < args_cli.num_episodes:
        # 1. Compute v10 obs
        obs_v10 = v10_builder.compute(unwrapped_env, command)

        # 2. MLP action (deterministic)
        with torch.no_grad():
            actions = model(obs_v10)

        # 3. Step env
        obs_v3, _, dones, infos = env_wrapped.step(actions)
        ep_length_counter += 1
        step_count += 1

        # 4. Update v10 history
        v10_builder.update_history(unwrapped_env, command, actions, dones)

        # 5. Check ball contact (simple: ball moved significantly)
        soccer_ball = unwrapped_env.scene["soccer_ball"]
        ball_vel = soccer_ball.data.root_lin_vel_w[:, :3]
        ball_speed = torch.norm(ball_vel[:, :2], dim=-1)
        ball_contacted |= (ball_speed > 0.5)

        # 6. Handle episode ends
        if dones.any():
            done_ids = dones.nonzero(as_tuple=True)[0]
            for idx in done_ids:
                if episodes_completed >= args_cli.num_episodes:
                    break
                i = idx.item()
                episodes_completed += 1
                ep_lengths.append(ep_length_counter[i].item())

                # Check kick success
                if ball_contacted[i]:
                    total_kicks += 1
                    ball_speeds.append(ball_speed[i].item())

                # Check fall (short episode = likely fell)
                if ep_length_counter[i] < 100:  # < 2s at 50Hz
                    total_falls += 1

                # Reset per-env state
                ep_length_counter[i] = 0
                ball_contacted[i] = False

        if step_count % 200 == 0:
            print(f"  Step {step_count}, episodes: {episodes_completed}/{args_cli.num_episodes}")

    # Report
    kick_rate = total_kicks / max(episodes_completed, 1) * 100
    fall_rate = total_falls / max(episodes_completed, 1) * 100
    avg_bspd = np.mean(ball_speeds) if ball_speeds else 0
    avg_ep_len = np.mean(ep_lengths) if ep_lengths else 0

    print(f"\n{'='*60}")
    print(f"v10 BC Rollout Evaluation ({episodes_completed} episodes)")
    print(f"{'='*60}")
    print(f"  Kick%:          {kick_rate:.1f}%  ({total_kicks}/{episodes_completed})")
    print(f"  Fall%:          {fall_rate:.1f}%  ({total_falls}/{episodes_completed})")
    print(f"  Avg BSpd:       {avg_bspd:.2f} m/s  (kicked episodes only)")
    print(f"  Avg Ep Length:  {avg_ep_len:.1f} steps")
    print(f"  Total steps:    {step_count}")
    print(f"{'='*60}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
