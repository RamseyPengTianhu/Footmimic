"""DAgger collection for action-space CVAE distillation.

The first offline dataset is collected under the expert policy, so the action
CVAE only learns on expert states.  This script rolls out the current decoder
student, queries the frozen expert on the visited states, and saves new
``obs_v10 -> expert action`` labels in the same format consumed by
``train_action_cvae_distill.py``.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect DAgger labels for action-CVAE decoder.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default="Anchor-V3CVAE-Kick-G1-Soccer-RNN-v0")
parser.add_argument("--motion_path", type=str, required=True)
parser.add_argument("--student_model", type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=500)
parser.add_argument("--output_path", type=str, default="data/action_latent/action_cvae_dagger.pt")
parser.add_argument("--teacher_mix", type=float, default=0.25, help="Execution blend: beta*teacher + (1-beta)*student.")
parser.add_argument("--mode", choices=["prior_mean", "prior_sample"], default="prior_mean")
parser.add_argument("--sample_scale", type=float, default=0.5)
parser.add_argument("--clip_actions", type=float, default=0.0, help="If >0, clamp student and executed actions.")
parser.add_argument("--seed", type=int, default=42)

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

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import soccer.tasks  # noqa: F401
from soccer.tasks.tracking.mdp.event_conditioned_obs_builder import V10ObsBuilder
from soccer.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

_model_path = os.path.join(os.path.dirname(__file__), "action_cvae_distill.py")
_spec = importlib.util.spec_from_file_location("action_cvae_distill", os.path.abspath(_model_path))
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
StateActionCVAE = _mod.StateActionCVAE

_train_path = os.path.join(os.path.dirname(__file__), "train_action_cvae_distill.py")
_train_spec = importlib.util.spec_from_file_location("train_action_cvae_distill", os.path.abspath(_train_path))
_train_mod = importlib.util.module_from_spec(_train_spec)
assert _train_spec.loader is not None
_train_spec.loader.exec_module(_train_mod)
apply_obs_slices = _train_mod.apply_obs_slices


def get_motion_files(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    files = sorted(glob.glob(os.path.join(path, "*.npz")))
    if not files:
        raise ValueError(f"No .npz files found in {path}")
    return files


def load_student(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = StateActionCVAE(
        obs_dim=int(ckpt["obs_dim"]),
        action_dim=int(ckpt["action_dim"]),
        latent_dim=int(ckpt["latent_dim"]),
        hidden_dims=list(ckpt["hidden_dims"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def student_action(model, ckpt: dict, obs_full: torch.Tensor) -> torch.Tensor:
    obs = apply_obs_slices(obs_full, ckpt["obs_slices"])
    obs = (obs - ckpt["obs_mean"].to(obs.device)) / ckpt["obs_std"].to(obs.device)
    if args_cli.mode == "prior_mean":
        action_norm = model.act_prior_mean(obs)
    else:
        p_mu, p_logvar = model.prior_stats(obs)
        p_std = torch.exp(0.5 * p_logvar)
        z = p_mu + args_cli.sample_scale * p_std * torch.randn_like(p_std)
        action_norm = model.decode(obs, z)
    action = action_norm * ckpt["action_std"].to(obs.device) + ckpt["action_mean"].to(obs.device)
    base_action_dim = int(ckpt.get("base_action_dim", 29))
    action = action[:, :base_action_dim]
    if args_cli.clip_actions > 0.0:
        action = action.clamp(-args_cli.clip_actions, args_cli.clip_actions)
    return action


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
    env_cfg.commands.motion.motion_files = get_motion_files(args_cli.motion_path)
    if hasattr(env_cfg.commands.motion, "strike_motion_files"):
        env_cfg.commands.motion.strike_motion_files = env_cfg.commands.motion.motion_files
    device = env_cfg.sim.device

    student, student_ckpt = load_student(args_cli.student_model, device)
    print(f"[INFO] Loaded student action CVAE: {args_cli.student_model}")
    print(
        f"[INFO] student obs_slices={student_ckpt['obs_slices']}, mode={args_cli.mode}, "
        f"action_horizon={student_ckpt.get('action_horizon', 1)}"
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    env_wrapped = RslRlVecEnvWrapper(env)
    base_env = env.unwrapped

    log_root = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    resume_path = get_checkpoint_path(os.path.abspath(log_root), agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading frozen teacher: {resume_path}")
    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    teacher_policy = runner.get_inference_policy(device=device)

    command = base_env.command_manager.get_term("motion")
    builder = V10ObsBuilder(
        num_envs=args_cli.num_envs,
        num_joints=command.robot.data.joint_pos.shape[1],
        device=device,
    )
    builder.init_segment_bounds(command)
    obs_v3, _ = env_wrapped.get_observations()

    beta = float(args_cli.teacher_mix)
    beta = max(0.0, min(1.0, beta))
    print(
        f"[INFO] Collecting {args_cli.num_episodes} episodes with beta={beta:.2f} "
        f"({args_cli.num_envs} envs). Executed = beta*teacher + (1-beta)*student"
    )

    all_obs_v10 = []
    all_obs_v3 = []
    all_actions_teacher = []
    all_actions_student = []
    all_actions_exec = []
    all_dones = []
    all_episode_steps = []
    all_motion_ids = []
    all_phase_ids = []
    all_phase_phis = []

    episodes_completed = 0
    step_count = 0
    while simulation_app.is_running() and episodes_completed < args_cli.num_episodes:
        obs_v10 = builder.compute(base_env, command)
        with torch.no_grad():
            a_teacher = teacher_policy(obs_v3)
            a_student = student_action(student, student_ckpt, obs_v10)
            a_exec = beta * a_teacher + (1.0 - beta) * a_student
            if args_cli.clip_actions > 0.0:
                a_exec = a_exec.clamp(-args_cli.clip_actions, args_cli.clip_actions)

        all_obs_v10.append(obs_v10.cpu())
        all_obs_v3.append(obs_v3.cpu())
        all_actions_teacher.append(a_teacher.cpu())
        all_actions_student.append(a_student.cpu())
        all_actions_exec.append(a_exec.cpu())
        all_episode_steps.append(command.time_steps.cpu())
        all_motion_ids.append(command.motion_idx.cpu())
        event_info = builder.get_event_info()
        all_phase_ids.append(event_info["phase_id"].cpu())
        all_phase_phis.append(event_info["phase_phi"].cpu())

        obs_v3, _, dones, _ = env_wrapped.step(a_exec)
        builder.update_history(base_env, command, a_exec, dones)
        all_dones.append(dones.cpu())

        if dones.any():
            episodes_completed += int(dones.sum().item())
        step_count += 1
        if step_count % 100 == 0:
            print(f"  Step {step_count}, episodes: {episodes_completed}/{args_cli.num_episodes}")

    dataset = {
        "obs_v10": torch.cat(all_obs_v10, dim=0),
        "actions_teacher": torch.cat(all_actions_teacher, dim=0),
        "actions_student": torch.cat(all_actions_student, dim=0),
        "actions_exec": torch.cat(all_actions_exec, dim=0),
        "obs_v3": torch.cat(all_obs_v3, dim=0),
        "done": torch.cat(all_dones, dim=0),
        "episode_step": torch.cat(all_episode_steps, dim=0),
        "motion_id": torch.cat(all_motion_ids, dim=0),
        "phase_id": torch.cat(all_phase_ids, dim=0),
        "phase_phi": torch.cat(all_phase_phis, dim=0),
        "metadata": {
            "teacher_run": agent_cfg.load_run,
            "teacher_ckpt": agent_cfg.load_checkpoint,
            "student_model": args_cli.student_model,
            "student_mode": args_cli.mode,
            "teacher_mix": beta,
            "obs_dim_v10": all_obs_v10[0].shape[-1],
            "num_envs": args_cli.num_envs,
            "num_steps": step_count,
            "num_episodes": episodes_completed,
            "dagger": True,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.output_path)), exist_ok=True)
    torch.save(dataset, args_cli.output_path)
    print("\n" + "=" * 60)
    print("DAgger collection complete")
    print("=" * 60)
    print(f"  Episodes:    {episodes_completed}")
    print(f"  Total steps: {step_count * args_cli.num_envs}")
    print(f"  obs_v10:     {dataset['obs_v10'].shape}")
    print(f"  actions:     {dataset['actions_teacher'].shape}")
    print(f"  Output:      {args_cli.output_path}")
    print("=" * 60)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
