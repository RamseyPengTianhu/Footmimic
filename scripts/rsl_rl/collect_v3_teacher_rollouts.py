"""Collect v3 teacher rollouts with v10 observations for BC pretraining.

Usage:
    python scripts/rsl_rl/collect_v3_teacher_rollouts.py \
        --task Anchor-CG-Kick-G1-Soccer-RNN-v0 \
        --motion_path data/motions/kick/ \
        --load_run 2026-04-28_12-15-12_cg_v3_softmask \
        --checkpoint model_12000.pt \
        --num_episodes 200 \
        --output_path data/bc_rollouts/v3_teacher_v10obs.pt \
        --headless

Key design:
  - ENV is v3 env (Anchor-CG-Kick-G1-Soccer-RNN-v0) so v3 LSTM runs correctly
  - Teacher action is DETERMINISTIC MEAN (act_inference), not sampled
  - V10ObsBuilder computes ~422D v10 obs from raw env state each step
  - Both obs_v10 and obs_v3 are saved; BC trains on obs_v10 → a_teacher
  - History buffer uses TEACHER actions (matches v3 rollout distribution)
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect v3 teacher rollouts with v10 obs.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default="Anchor-CG-Kick-G1-Soccer-RNN-v0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--motion_path", type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=200)
parser.add_argument("--output_path", type=str, default="data/bc_rollouts/v3_teacher_v10obs.pt")

# RSL-RL args
import cli_args  # isort: skip
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = False
if getattr(args_cli, "headless", False) and "DISPLAY" in os.environ:
    print(f"[INFO] Headless mode: clearing DISPLAY={os.environ['DISPLAY']!r} before launching Isaac Sim")
    os.environ.pop("DISPLAY", None)
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import glob
import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import soccer.tasks  # noqa: F401
from soccer.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner
from soccer.tasks.tracking.mdp.event_conditioned_obs_builder import V10ObsBuilder


def get_motion_files(motion_path: str) -> list[str]:
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
    """Collect teacher rollouts with v10 obs."""
    # Override config
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
    env_cfg.commands.motion.motion_files = get_motion_files(args_cli.motion_path)

    # Create v3 env
    env = gym.make(args_cli.task, cfg=env_cfg)
    env_wrapped = RslRlVecEnvWrapper(env)
    unwrapped_env = env.unwrapped
    device = env_cfg.sim.device

    # Load frozen v3 teacher
    log_root = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    resume_path = get_checkpoint_path(
        os.path.abspath(log_root), agent_cfg.load_run, agent_cfg.load_checkpoint
    )
    print(f"[INFO] Loading v3 teacher from: {resume_path}")

    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=device)

    # Initialize V10ObsBuilder
    command = unwrapped_env.command_manager.get_term("motion")
    num_joints = command.robot.data.joint_pos.shape[1]
    v10_builder = V10ObsBuilder(
        num_envs=args_cli.num_envs,
        num_joints=num_joints,
        device=device,
    )
    v10_builder.init_segment_bounds(command)

    # Collection buffers
    all_obs_v10 = []
    all_obs_v3 = []
    all_actions = []
    all_dones = []
    all_episode_steps = []
    all_phase_ids = []
    all_phase_phis = []
    all_motion_ids = []

    # Get initial observation
    obs_v3, _ = env_wrapped.get_observations()
    episodes_completed = 0
    step_count = 0

    print(f"[INFO] Collecting {args_cli.num_episodes} episodes "
          f"({args_cli.num_envs} parallel envs)...")
    print(f"[INFO] Teacher: {resume_path}")

    while episodes_completed < args_cli.num_episodes:
        # 1. Compute v10 obs from current state (BEFORE step)
        obs_v10 = v10_builder.compute(unwrapped_env, command)
        
        if step_count == 0:
            motor_prior = obs_v10[:, -32:]  # The last 32 dims are the new motor prior
            joint_delta = motor_prior[:, :29]
            height_delta = motor_prior[:, 29:30]
            gravity_delta = motor_prior[:, 30:32]
            print("\n[DEBUG] Motor Prior Stats (First Step):")
            print(f"  Joint Delta (29D)  | mean: {joint_delta.mean().item():.3f}, min: {joint_delta.min().item():.3f}, max: {joint_delta.max().item():.3f}")
            print(f"  Height Delta (1D)  | mean: {height_delta.mean().item():.3f}, min: {height_delta.min().item():.3f}, max: {height_delta.max().item():.3f}")
            print(f"  Gravity Delta (2D) | mean: {gravity_delta.mean().item():.3f}, min: {gravity_delta.min().item():.3f}, max: {gravity_delta.max().item():.3f}\n")


        # 2. Teacher deterministic mean action
        with torch.no_grad():
            a_teacher = policy(obs_v3)

        # 3. Record
        all_obs_v10.append(obs_v10.cpu())
        all_obs_v3.append(obs_v3.cpu())
        all_actions.append(a_teacher.cpu())
        all_episode_steps.append(command.time_steps.cpu())
        all_motion_ids.append(command.motion_idx.cpu())

        event_info = v10_builder.get_event_info()
        all_phase_ids.append(event_info["phase_id"].cpu())
        all_phase_phis.append(event_info["phase_phi"].cpu())

        # 4. Step env with teacher action
        obs_v3, _, dones, infos = env_wrapped.step(a_teacher)
        all_dones.append(dones.cpu())

        # 5. Update v10 history with teacher action (post-step)
        v10_builder.update_history(unwrapped_env, command, a_teacher, dones)

        # 6. Count completed episodes
        if dones.any():
            n_done = dones.sum().item()
            episodes_completed += n_done

        step_count += 1
        if step_count % 100 == 0:
            print(f"  Step {step_count}, episodes: {episodes_completed}/{args_cli.num_episodes}")

    # Stack and save
    obs_v10_dim = all_obs_v10[0].shape[-1]

    dataset = {
        "obs_v10": torch.cat(all_obs_v10, dim=0),           # [T*N, ~422]
        "actions_teacher": torch.cat(all_actions, dim=0),     # [T*N, 29]
        "obs_v3": torch.cat(all_obs_v3, dim=0),              # [T*N, 160] (debug)
        "done": torch.cat(all_dones, dim=0),                  # [T*N]
        "episode_step": torch.cat(all_episode_steps, dim=0),  # [T*N]
        "motion_id": torch.cat(all_motion_ids, dim=0),        # [T*N]
        "phase_id": torch.cat(all_phase_ids, dim=0),          # [T*N]
        "phase_phi": torch.cat(all_phase_phis, dim=0),        # [T*N]
        "metadata": {
            "teacher_run": agent_cfg.load_run,
            "teacher_ckpt": agent_cfg.load_checkpoint,
            "obs_dim_v10": obs_v10_dim,
            "num_envs": args_cli.num_envs,
            "num_steps": step_count,
            "num_episodes": episodes_completed,
            "deterministic_teacher": True,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.output_path)), exist_ok=True)
    torch.save(dataset, args_cli.output_path)

    print(f"\n{'='*60}")
    print(f"Collection complete")
    print(f"{'='*60}")
    print(f"  Episodes:     {episodes_completed}")
    print(f"  Total steps:  {step_count * args_cli.num_envs}")
    print(f"  obs_v10 dim:  {obs_v10_dim}")
    print(f"  obs_v10:      {dataset['obs_v10'].shape}")
    print(f"  actions:      {dataset['actions_teacher'].shape}")
    print(f"  obs_v3 (dbg): {dataset['obs_v3'].shape}")
    print(f"  Output:       {args_cli.output_path}")
    print(f"{'='*60}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
