"""DAgger: collect on-policy data with student rollout + teacher labels.

Usage:
    python scripts/rsl_rl/dagger_v10.py \
        --task Anchor-CG-Kick-G1-Soccer-RNN-v0 \
        --motion_path motions/Video/ \
        --load_run 2026-04-28_12-15-12_cg_v3_softmask \
        --checkpoint model_12000.pt \
        --bc_checkpoint logs/rsl_rl/v10_bc/bc_pretrained.pt \
        --num_episodes 200 \
        --output_path data/bc_rollouts/v3_teacher_v10obs_dagger1.pt \
        --headless

Key design:
  - Student (v10 MLP) controls the robot (visits its OWN state distribution)
  - Teacher (frozen v3 LSTM) provides the LABEL for that state
  - V10ObsBuilder computes obs_v10 from the STUDENT's visited states
  - This fixes covariate shift: student sees its own mistakes during training
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="DAgger: student rollout + teacher labels.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default="Anchor-CG-Kick-G1-Soccer-RNN-v0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--motion_path", type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=200)
parser.add_argument("--output_path", type=str, default="data/bc_rollouts/v3_teacher_v10obs_dagger1.pt")
parser.add_argument("--bc_checkpoint", type=str, required=True, help="BC-pretrained v10 MLP checkpoint.")
parser.add_argument("--beta", type=float, default=0.0,
                    help="Teacher mixing ratio. 0=pure student, 1=pure teacher. "
                         "Use 0 for standard DAgger (student rollout, teacher label).")

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

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import soccer.tasks  # noqa: F401
from soccer.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner
from soccer.tasks.tracking.mdp.event_conditioned_obs_builder import V10ObsBuilder


class V10MLPActor(nn.Module):
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
    """DAgger data collection."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
    env_cfg.commands.motion.motion_files = get_motion_files(args_cli.motion_path)
    device = env_cfg.sim.device

    # Create env
    env = gym.make(args_cli.task, cfg=env_cfg)
    env_wrapped = RslRlVecEnvWrapper(env)
    unwrapped_env = env.unwrapped

    # Load frozen v3 teacher
    log_root = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    resume_path = get_checkpoint_path(
        os.path.abspath(log_root), agent_cfg.load_run, agent_cfg.load_checkpoint
    )
    print(f"[INFO] Loading v3 teacher from: {resume_path}")
    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(resume_path)
    teacher_policy = runner.get_inference_policy(device=device)

    # Load student (BC-pretrained v10 MLP)
    print(f"[INFO] Loading BC student from: {args_cli.bc_checkpoint}")
    ckpt = torch.load(args_cli.bc_checkpoint, map_location=device, weights_only=False)
    student = V10MLPActor(
        obs_dim=ckpt["obs_dim"],
        action_dim=ckpt["action_dim"],
        hidden_dims=ckpt.get("hidden_dims", [512, 256, 128]),
    ).to(device)
    student.load_state_dict(ckpt["model_state_dict"])
    student.eval()

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
    all_actions = []  # teacher labels
    all_dones = []
    all_phase_ids = []

    obs_v3, _ = env_wrapped.get_observations()
    episodes_completed = 0
    step_count = 0
    beta = args_cli.beta

    print(f"[INFO] DAgger: {args_cli.num_episodes} episodes, beta={beta}")
    print(f"[INFO] beta=0 → pure student rollout; beta=1 → pure teacher rollout")

    while episodes_completed < args_cli.num_episodes:
        # 1. Compute v10 obs
        obs_v10 = v10_builder.compute(unwrapped_env, command)

        # 2. Student action (for rollout)
        with torch.no_grad():
            a_student = student(obs_v10)

        # 3. Teacher action (for labeling)
        with torch.no_grad():
            a_teacher = teacher_policy(obs_v3)

        # 4. Record: v10 obs + teacher label
        all_obs_v10.append(obs_v10.cpu())
        all_actions.append(a_teacher.cpu())

        event_info = v10_builder.get_event_info()
        all_phase_ids.append(event_info["phase_id"].cpu())

        # 5. Choose execution action (beta-mixing)
        if beta > 0:
            a_exec = beta * a_teacher + (1 - beta) * a_student
        else:
            a_exec = a_student  # Pure student rollout (standard DAgger)

        # 6. Step env with execution action
        obs_v3, _, dones, infos = env_wrapped.step(a_exec)
        all_dones.append(dones.cpu())

        # 7. Update history with EXECUTED action (not teacher label)
        v10_builder.update_history(unwrapped_env, command, a_exec, dones)

        # 8. Count
        if dones.any():
            episodes_completed += dones.sum().item()

        step_count += 1
        if step_count % 100 == 0:
            print(f"  Step {step_count}, episodes: {episodes_completed}/{args_cli.num_episodes}")

    # Save
    dataset = {
        "obs_v10": torch.cat(all_obs_v10, dim=0),
        "actions_teacher": torch.cat(all_actions, dim=0),
        "done": torch.cat(all_dones, dim=0),
        "phase_id": torch.cat(all_phase_ids, dim=0),
        "metadata": {
            "type": "dagger",
            "beta": beta,
            "teacher_run": agent_cfg.load_run,
            "teacher_ckpt": agent_cfg.load_checkpoint,
            "student_ckpt": args_cli.bc_checkpoint,
            "obs_dim_v10": all_obs_v10[0].shape[-1],
            "num_envs": args_cli.num_envs,
            "num_steps": step_count,
            "num_episodes": episodes_completed,
            "deterministic_teacher": True,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.output_path)), exist_ok=True)
    torch.save(dataset, args_cli.output_path)

    print(f"\n{'='*60}")
    print(f"DAgger collection complete")
    print(f"{'='*60}")
    print(f"  Episodes:     {episodes_completed}")
    print(f"  Total steps:  {step_count * args_cli.num_envs}")
    print(f"  obs_v10:      {dataset['obs_v10'].shape}")
    print(f"  actions:      {dataset['actions_teacher'].shape}")
    print(f"  Output:       {args_cli.output_path}")
    print(f"{'='*60}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
