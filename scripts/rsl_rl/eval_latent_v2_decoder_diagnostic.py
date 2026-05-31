"""Closed-loop diagnostics for a LATENT-v2 distillation checkpoint.

Compares three action sources in the same environment:

  teacher          : frozen v3 policy action
  prior_mean       : D(obs, mean[p(z|obs)])              deployment path
  posterior_mean   : D(obs, mean[q(z|obs, teacher_a)])   oracle decoder path

The gap between posterior_mean and prior_mean tells us whether the decoder can
represent the teacher action but the prior cannot predict the needed latent.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diagnose LATENT-v2 decoder/prior closed-loop quality.")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--task", type=str, default="Anchor-CG-Kick-G1-Soccer-RNN-v0")
parser.add_argument("--motion_path", type=str, required=True)
parser.add_argument("--latent_model", type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=100)
parser.add_argument("--modes", type=str, nargs="+", default=["teacher", "prior_mean", "posterior_mean"],
                    choices=("teacher", "prior_mean", "posterior_mean", "posterior_sample"))
parser.add_argument("--output_json", type=str, default=None)

import cli_args  # isort: skip

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = False
if getattr(args_cli, "headless", False) and "DISPLAY" in os.environ:
    os.environ.pop("DISPLAY", None)
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import soccer.tasks  # noqa: F401
from soccer.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner
from latent_v2_models import LatentActionModel, diag_gaussian_kl
from compute_task_features import compute_ball_foot_relation


def get_motion_files(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.npz")))
        if files:
            return files
    raise ValueError(f"Invalid motion_path: {path}")


def load_latent_model(path: str, device: str) -> tuple[LatentActionModel, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = LatentActionModel(
        obs_dim=int(ckpt["obs_dim"]),
        action_dim=int(ckpt["action_dim"]),
        z_dim=int(ckpt["z_dim"]),
        hidden_dims=list(ckpt["hidden_dims"]),
        decoder_obs_mode=ckpt.get("decoder_obs_mode", "full"),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, ckpt


@torch.no_grad()
def action_from_mode(
    mode: str,
    *,
    model: LatentActionModel,
    teacher_policy,
    obs_v3: torch.Tensor,
    task_features: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    teacher_action = teacher_policy(obs_v3).clone()
    info: dict[str, torch.Tensor] = {}

    if mode == "teacher":
        info["action_mse"] = torch.zeros(obs_v3.shape[0], device=obs_v3.device)
        info["action_norm"] = torch.norm(teacher_action, dim=-1)
        info["teacher_action_norm"] = torch.norm(teacher_action, dim=-1)
        return teacher_action, info

    dec_obs = model.select_decoder_obs(obs_v3, task_features=task_features)
    p_mu, p_logvar = model.prior(dec_obs)

    if mode == "prior_mean":
        z = p_mu
    elif mode in ("posterior_mean", "posterior_sample"):
        q_mu, q_logvar = model.encoder(dec_obs, teacher_action)
        z = q_mu if mode == "posterior_mean" else model.reparameterize(q_mu, q_logvar)
        info["kl_q_prior"] = diag_gaussian_kl(q_mu, q_logvar, p_mu, p_logvar)
        info["latent_mu_l2"] = torch.norm(q_mu - p_mu, dim=-1)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    action = model.decoder(dec_obs, z)
    info["action_mse"] = (action - teacher_action).pow(2).mean(dim=-1)
    info["action_norm"] = torch.norm(action, dim=-1)
    info["teacher_action_norm"] = torch.norm(teacher_action, dim=-1)
    return action, info


def summarize(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def reset_motion_command(env, env_ids: torch.Tensor | None = None) -> None:
    """Start evaluated episodes from the beginning of their motion clips.

    The regular training command can reset with randomized command state.  This
    diagnostic is meant to compare action sources, so keep the motion timing
    deterministic and round-robin across loaded clips.
    """
    base_env = env.unwrapped
    cmd = base_env.command_manager.get_term("motion")
    num_envs = env.num_envs
    ids = (
        torch.arange(num_envs, device=base_env.device)
        if env_ids is None
        else env_ids.to(device=base_env.device, dtype=torch.long)
    )
    if ids.numel() == 0:
        return
    num_files = int(cmd.motion.num_files)
    motion_ids = ids % num_files
    cmd.motion_idx[ids] = motion_ids
    cmd.motion_length[ids] = cmd.motion.file_lengths[motion_ids]
    cmd.time_steps[ids] = 0


def run_mode(
    mode: str,
    *,
    env: RslRlVecEnvWrapper,
    model: LatentActionModel,
    teacher_policy,
    teacher_module,
    num_episodes: int,
    device: str,
) -> dict:
    obs_v3, _ = env.reset()
    reset_motion_command(env)
    obs_v3, _ = env.get_observations()
    if hasattr(teacher_module, "reset"):
        teacher_module.reset()
    unwrapped = env.unwrapped
    use_task_features = model.decoder_obs_mode == "task_features"
    num_envs = env.num_envs

    ep_len = torch.zeros(num_envs, device=device)
    ball_contacted = torch.zeros(num_envs, dtype=torch.bool, device=device)
    max_ball_speed = torch.zeros(num_envs, device=device)

    episodes = 0
    kicks = 0
    falls = 0
    no_contact = 0
    ball_speeds: list[float] = []
    action_mse: list[float] = []
    kl_values: list[float] = []
    latent_mu_l2: list[float] = []
    action_norms: list[float] = []
    teacher_norms: list[float] = []

    step = 0
    while episodes < num_episodes:
        task_features = compute_ball_foot_relation(unwrapped) if use_task_features else None
        action, info = action_from_mode(
            mode,
            model=model,
            teacher_policy=teacher_policy,
            obs_v3=obs_v3,
            task_features=task_features,
        )
        if "action_mse" in info:
            action_mse.append(info["action_mse"].mean().item())
            action_norms.append(info["action_norm"].mean().item())
            teacher_norms.append(info["teacher_action_norm"].mean().item())
        if "kl_q_prior" in info:
            kl_values.append(info["kl_q_prior"].mean().item())
        if "latent_mu_l2" in info:
            latent_mu_l2.append(info["latent_mu_l2"].mean().item())

        obs_v3, _, dones, _ = env.step(action)
        ep_len += 1
        step += 1

        ball = unwrapped.scene["soccer_ball"]
        ball_speed = torch.norm(ball.data.root_lin_vel_w[:, :2], dim=-1)
        ball_contacted |= ball_speed > 0.5
        max_ball_speed = torch.maximum(max_ball_speed, ball_speed)

        if dones.any():
            done_tensor = torch.as_tensor(dones, device=device).bool()
            if hasattr(teacher_module, "reset"):
                teacher_module.reset(done_tensor)
            done_ids = dones.nonzero(as_tuple=True)[0]
            reset_motion_command(env, done_ids)
            obs_v3, _ = env.get_observations()
            for idx in done_ids:
                if episodes >= num_episodes:
                    break
                i = idx.item()
                episodes += 1
                contacted = bool(ball_contacted[i].item())
                if contacted:
                    kicks += 1
                    ball_speeds.append(max_ball_speed[i].item())
                else:
                    no_contact += 1
                if ep_len[i] < 100:
                    falls += 1
                ep_len[i] = 0
                ball_contacted[i] = False
                max_ball_speed[i] = 0.0

        if step % 500 == 0:
            print(f"[{mode}] step={step}, episodes={episodes}/{num_episodes}")

    return {
        "mode": mode,
        "episodes": episodes,
        "kick_pct": kicks / max(episodes, 1) * 100.0,
        "fall_pct": falls / max(episodes, 1) * 100.0,
        "no_contact_pct": no_contact / max(episodes, 1) * 100.0,
        "avg_ball_speed": summarize(ball_speeds),
        "max_ball_speed": max(ball_speeds) if ball_speeds else 0.0,
        "action_mse": summarize(action_mse),
        "kl_q_prior": summarize(kl_values),
        "latent_mu_l2": summarize(latent_mu_l2),
        "action_norm": summarize(action_norms),
        "teacher_action_norm": summarize(teacher_norms),
    }


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
    env_cfg.commands.motion.motion_files = get_motion_files(args_cli.motion_path)
    device = env_cfg.sim.device

    env_raw = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env_raw)

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading teacher: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(resume_path)
    teacher_policy = runner.get_inference_policy(device=device)

    model, ckpt = load_latent_model(args_cli.latent_model, device)
    print(
        "[INFO] Loaded latent model: "
        f"decoder_obs_mode={model.decoder_obs_mode}, obs_dim={ckpt['obs_dim']}, "
        f"decoder_obs_dim={model.decoder_obs_dim}, z_dim={ckpt['z_dim']}"
    )

    results = []
    for mode in args_cli.modes:
        print(f"\n[INFO] Running mode={mode}")
        results.append(
            run_mode(
                mode,
                env=env,
                model=model,
                teacher_policy=teacher_policy,
                teacher_module=runner.alg.policy,
                num_episodes=args_cli.num_episodes,
                device=device,
            )
        )

    print("\n" + "=" * 118)
    print("  LATENT-v2 DECODER / PRIOR DIAGNOSTIC")
    print("=" * 118)
    print(
        f"{'Mode':<17} {'Kick%':>7} {'Fall%':>7} {'NoCt%':>7} "
        f"{'BSpd':>7} {'ActMSE':>9} {'KL(q||p)':>9} {'|muq-mup|':>10} {'ActN/TchN':>13}"
    )
    print("-" * 118)
    for row in results:
        print(
            f"{row['mode']:<17} {row['kick_pct']:7.1f} {row['fall_pct']:7.1f} "
            f"{row['no_contact_pct']:7.1f} {row['avg_ball_speed']:7.2f} "
            f"{row['action_mse']:9.5f} {row['kl_q_prior']:9.3f} "
            f"{row['latent_mu_l2']:10.3f} {row['action_norm']:6.2f}/{row['teacher_action_norm']:<6.2f}"
        )
    print("=" * 118)

    if args_cli.output_json:
        output_dir = os.path.dirname(args_cli.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args_cli.output_json, "w", encoding="utf-8") as f:
            json.dump({"latent_model": args_cli.latent_model, "results": results}, f, indent=2)
        print(f"[INFO] Saved JSON: {args_cli.output_json}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
