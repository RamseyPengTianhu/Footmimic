"""Closed-loop rollout evaluation for an action-space CVAE decoder.

This is the first online check for a LATENT-style action prior:

    state -> p(z | state), decoder(state, z) -> low-level action

The model is trained offline by ``train_action_cvae_distill.py`` from expert
policy rollouts.  This script runs the decoder directly in the simulator using
the same V10 observation builder used during data collection.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
from collections import defaultdict

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate action-CVAE decoder in closed-loop simulation.")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--task", type=str, default="Anchor-V3CVAE-Kick-G1-Soccer-RNN-v0")
parser.add_argument("--motion_path", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--eval_episodes", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--mode", choices=["prior_mean", "prior_sample", "teacher_posterior"], default="prior_mean")
parser.add_argument("--sample_scale", type=float, default=0.5)
parser.add_argument(
    "--posterior_rollout_data",
    type=str,
    nargs="*",
    default=None,
    help=(
        "Rollout data used to build H-step teacher action chunks for "
        "teacher_posterior when action_horizon>1."
    ),
)
parser.add_argument(
    "--posterior_chunk_source",
    choices=["lookup", "repeat_live_teacher"],
    default="lookup",
    help=(
        "For action_horizon>1 teacher_posterior: use H-step chunks from offline lookup, "
        "or repeat the live teacher's current action across the chunk as a decoder sanity check."
    ),
)
parser.add_argument(
    "--posterior_missing",
    choices=["error", "repeat_teacher"],
    default="error",
    help="Fallback if a (motion_id, time_step) posterior chunk is missing.",
)
parser.add_argument(
    "--execute_chunk",
    action="store_true",
    help="For action_horizon>1 checkpoints, decode a chunk and execute queued actions before decoding again.",
)
parser.add_argument(
    "--chunk_steps",
    type=int,
    default=0,
    help="Number of decoded chunk steps to execute before replanning. 0 means the checkpoint action_horizon.",
)
parser.add_argument("--clip_actions", type=float, default=0.0, help="If >0, clamp decoded actions to +/- this value.")
parser.add_argument("--ball_speed_success", type=float, default=2.0)
parser.add_argument("--direction_success", type=float, default=0.5)
parser.add_argument("--output_json", type=str, default=None)

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
import numpy as np
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


class PosteriorActionLookup:
    """Mean teacher action chunk indexed by (motion_id, episode_step)."""

    def __init__(self, mean_chunks: torch.Tensor, valid: torch.Tensor):
        self.mean_chunks = mean_chunks
        self.valid = valid

    def to(self, device: str):
        self.mean_chunks = self.mean_chunks.to(device)
        self.valid = self.valid.to(device)
        return self

    def lookup(self, motion_ids: torch.Tensor, time_steps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        motion_ids = motion_ids.long().clamp(min=0)
        time_steps = time_steps.long().clamp(min=0)
        in_range = (motion_ids < self.mean_chunks.shape[0]) & (time_steps < self.mean_chunks.shape[1])
        safe_motion = motion_ids.clamp(max=self.mean_chunks.shape[0] - 1)
        safe_step = time_steps.clamp(max=self.mean_chunks.shape[1] - 1)
        chunks = self.mean_chunks[safe_motion, safe_step]
        valid = in_range & self.valid[safe_motion, safe_step]
        return chunks, valid


def get_motion_files(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    files = sorted(glob.glob(os.path.join(path, "*.npz")))
    if not files:
        raise ValueError(f"No .npz files found in {path}")
    return files


def _mean(values: list[float], default: float = 0.0) -> float:
    return float(np.mean(values)) if values else default


def _make_posterior_diag() -> dict[str, float]:
    return {
        "lookup_valid": 0.0,
        "lookup_total": 0.0,
        "first_recon_mse_sum": 0.0,
        "full_recon_mse_sum": 0.0,
        "first_std_recon_mse_sum": 0.0,
        "full_std_recon_mse_sum": 0.0,
        "recon_count": 0.0,
        "teacher_first_norm_sum": 0.0,
        "decoded_first_norm_sum": 0.0,
        "teacher_full_norm_sum": 0.0,
        "decoded_full_norm_sum": 0.0,
    }


@torch.no_grad()
def _accumulate_posterior_diag(
    diag: dict[str, float],
    decoded_chunk: torch.Tensor,
    teacher_chunk: torch.Tensor,
    valid_mask: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
) -> None:
    valid_mask = valid_mask.bool()
    diag["lookup_valid"] += float(valid_mask.sum().item())
    diag["lookup_total"] += float(valid_mask.numel())
    if not bool(valid_mask.any().item()):
        return

    teacher_chunk = teacher_chunk.view_as(decoded_chunk)
    decoded_valid = decoded_chunk[valid_mask]
    teacher_valid = teacher_chunk[valid_mask]
    first_diff = decoded_valid[:, 0] - teacher_valid[:, 0]
    full_diff = decoded_valid - teacher_valid
    mean = action_mean.to(decoded_chunk.device).view_as(teacher_chunk[:1])
    std = action_std.to(decoded_chunk.device).view_as(teacher_chunk[:1]).clamp(min=1.0e-6)
    decoded_valid_std = (decoded_valid - mean) / std
    teacher_valid_std = (teacher_valid - mean) / std
    first_std_diff = decoded_valid_std[:, 0] - teacher_valid_std[:, 0]
    full_std_diff = decoded_valid_std - teacher_valid_std
    n = float(decoded_valid.shape[0])
    diag["first_recon_mse_sum"] += float(first_diff.pow(2).mean(dim=-1).sum().item())
    diag["full_recon_mse_sum"] += float(full_diff.pow(2).mean(dim=(1, 2)).sum().item())
    diag["first_std_recon_mse_sum"] += float(first_std_diff.pow(2).mean(dim=-1).sum().item())
    diag["full_std_recon_mse_sum"] += float(full_std_diff.pow(2).mean(dim=(1, 2)).sum().item())
    diag["teacher_first_norm_sum"] += float(torch.norm(teacher_valid[:, 0], dim=-1).sum().item())
    diag["decoded_first_norm_sum"] += float(torch.norm(decoded_valid[:, 0], dim=-1).sum().item())
    diag["teacher_full_norm_sum"] += float(torch.norm(teacher_valid.reshape(decoded_valid.shape[0], -1), dim=-1).sum().item())
    diag["decoded_full_norm_sum"] += float(torch.norm(decoded_valid.reshape(decoded_valid.shape[0], -1), dim=-1).sum().item())
    diag["recon_count"] += n


def _finalize_posterior_diag(diag: dict[str, float]) -> dict[str, float]:
    total = max(diag["lookup_total"], 1.0)
    count = max(diag["recon_count"], 1.0)
    return {
        "lookup_valid_rate": diag["lookup_valid"] / total,
        "lookup_valid": diag["lookup_valid"],
        "lookup_total": diag["lookup_total"],
        "first_recon_mse": diag["first_recon_mse_sum"] / count,
        "full_recon_mse": diag["full_recon_mse_sum"] / count,
        "first_std_recon_mse": diag["first_std_recon_mse_sum"] / count,
        "full_std_recon_mse": diag["full_std_recon_mse_sum"] / count,
        "teacher_first_action_norm": diag["teacher_first_norm_sum"] / count,
        "decoded_first_action_norm": diag["decoded_first_norm_sum"] / count,
        "teacher_full_action_norm": diag["teacher_full_norm_sum"] / count,
        "decoded_full_action_norm": diag["decoded_full_norm_sum"] / count,
        "recon_count": diag["recon_count"],
    }


def _repeat_teacher_action_chunk(teacher_action: torch.Tensor, action_horizon: int) -> torch.Tensor:
    return teacher_action[:, None, :].expand(-1, action_horizon, -1).reshape(teacher_action.shape[0], -1)


def _get_kick_direction(base_env, cmd, ball_pos_w: torch.Tensor) -> torch.Tensor:
    env_origins = getattr(base_env.scene, "env_origins", None)
    dest_w = cmd.target_destination_pos[:, :2]
    if env_origins is not None:
        dest_w = dest_w + env_origins[:, :2]
    direction = dest_w - ball_pos_w[:, :2]
    return direction / torch.norm(direction, dim=-1, keepdim=True).clamp(min=1.0e-6)


@torch.no_grad()
def decode_action(model, ckpt: dict, obs_full: torch.Tensor, mode: str, sample_scale: float) -> torch.Tensor:
    chunk = decode_action_chunk(model, ckpt, obs_full, mode, sample_scale)
    return chunk[:, 0]


@torch.no_grad()
def decode_action_chunk(model, ckpt: dict, obs_full: torch.Tensor, mode: str, sample_scale: float) -> torch.Tensor:
    obs = apply_obs_slices(obs_full, ckpt["obs_slices"])
    obs = (obs - ckpt["obs_mean"].to(obs.device)) / ckpt["obs_std"].to(obs.device)
    if mode == "prior_mean":
        action_norm = model.act_prior_mean(obs)
    else:
        p_mu, p_logvar = model.prior_stats(obs)
        p_std = torch.exp(0.5 * p_logvar)
        z = p_mu + sample_scale * p_std * torch.randn_like(p_std)
        action_norm = model.decode(obs, z)
    action = action_norm * ckpt["action_std"].to(obs.device) + ckpt["action_mean"].to(obs.device)
    base_action_dim = int(ckpt.get("base_action_dim", 29))
    action_horizon = int(ckpt.get("action_horizon", 1))
    action = action.view(action.shape[0], action_horizon, base_action_dim)
    if args_cli.clip_actions > 0.0:
        action = action.clamp(-args_cli.clip_actions, args_cli.clip_actions)
    return action


@torch.no_grad()
def decode_teacher_posterior_chunk(
    model,
    ckpt: dict,
    obs_full: torch.Tensor,
    teacher_action_chunk: torch.Tensor,
) -> torch.Tensor:
    obs = apply_obs_slices(obs_full, ckpt["obs_slices"])
    obs = (obs - ckpt["obs_mean"].to(obs.device)) / ckpt["obs_std"].to(obs.device)
    base_action_dim = int(ckpt.get("base_action_dim", 29))
    action_horizon = int(ckpt.get("action_horizon", 1))
    if teacher_action_chunk.dim() == 2 and teacher_action_chunk.shape[-1] == base_action_dim:
        teacher_action_chunk = teacher_action_chunk[:, None, :].expand(-1, action_horizon, -1).reshape(
            teacher_action_chunk.shape[0],
            -1,
        )
    elif teacher_action_chunk.dim() == 3:
        teacher_action_chunk = teacher_action_chunk.reshape(teacher_action_chunk.shape[0], -1)
    if teacher_action_chunk.shape[-1] != int(ckpt["action_dim"]):
        raise ValueError(
            f"teacher_action_chunk dim={teacher_action_chunk.shape[-1]} but checkpoint action_dim={ckpt['action_dim']}"
        )
    action_norm = (teacher_action_chunk - ckpt["action_mean"].to(obs.device)) / ckpt["action_std"].to(obs.device)
    q_mu, _ = model.encode(obs, action_norm)
    decoded_norm = model.decode(obs, q_mu)
    action = decoded_norm * ckpt["action_std"].to(obs.device) + ckpt["action_mean"].to(obs.device)
    action = action.view(action.shape[0], action_horizon, base_action_dim)
    if args_cli.clip_actions > 0.0:
        action = action.clamp(-args_cli.clip_actions, args_cli.clip_actions)
    return action


@torch.no_grad()
def decode_teacher_posterior(model, ckpt: dict, obs_full: torch.Tensor, teacher_action: torch.Tensor) -> torch.Tensor:
    return decode_teacher_posterior_chunk(model, ckpt, obs_full, teacher_action)[:, 0]


def load_model(path: str, device: str):
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


def _build_valid_action_chunks(data: dict, action_horizon: int):
    actions = data["actions_teacher"].float()
    motion_id = data["motion_id"].long()
    episode_step = data["episode_step"].long()
    done = data.get("done")
    if done is None:
        done = torch.zeros(actions.shape[0], dtype=torch.bool)
    else:
        done = done.bool()
    num_envs = int(data.get("metadata", {}).get("num_envs", 0))
    if num_envs <= 0:
        raise ValueError("posterior rollout data requires metadata['num_envs']")
    if actions.shape[0] % num_envs != 0:
        raise ValueError(f"Cannot reshape {actions.shape[0]} samples into num_envs={num_envs}")
    steps = actions.shape[0] // num_envs
    valid_steps = steps - action_horizon + 1
    if valid_steps <= 0:
        raise ValueError(f"Rollout has only {steps} steps, shorter than action_horizon={action_horizon}")
    actions_t = actions.view(steps, num_envs, -1)
    motion_t = motion_id.view(steps, num_envs)
    step_t = episode_step.view(steps, num_envs)
    done_t = done.view(steps, num_envs)

    chunks = torch.stack(
        [actions_t[offset : offset + valid_steps] for offset in range(action_horizon)],
        dim=2,
    ).flatten(2)
    valid = torch.ones(valid_steps, num_envs, dtype=torch.bool)
    for offset in range(action_horizon):
        valid &= ~done_t[offset : offset + valid_steps]
    valid = valid.reshape(-1)
    return (
        motion_t[:valid_steps].reshape(-1)[valid],
        step_t[:valid_steps].reshape(-1)[valid],
        chunks.reshape(-1, chunks.shape[-1])[valid],
    )


def build_posterior_action_lookup(paths: list[str], ckpt: dict) -> PosteriorActionLookup:
    action_horizon = int(ckpt.get("action_horizon", 1))
    action_dim = int(ckpt["action_dim"])
    prepared = []
    max_motion = 0
    max_step = 0
    total = 0
    for path in paths:
        data = torch.load(path, map_location="cpu", weights_only=False)
        motion_ids, time_steps, chunks = _build_valid_action_chunks(data, action_horizon)
        if chunks.shape[-1] != action_dim:
            raise ValueError(f"{path}: chunk action_dim={chunks.shape[-1]} differs from checkpoint action_dim={action_dim}")
        prepared.append((motion_ids, time_steps, chunks))
        max_motion = max(max_motion, int(motion_ids.max().item()))
        max_step = max(max_step, int(time_steps.max().item()))
        total += int(chunks.shape[0])
        print(f"[INFO] Posterior lookup loaded {path}: chunks={chunks.shape[0]}")

    sums = torch.zeros(max_motion + 1, max_step + 1, action_dim)
    counts = torch.zeros(max_motion + 1, max_step + 1, 1)
    for motion_ids, time_steps, chunks in prepared:
        sums.index_put_((motion_ids, time_steps), chunks, accumulate=True)
        counts.index_put_(
            (motion_ids, time_steps),
            torch.ones(chunks.shape[0], 1, dtype=counts.dtype),
            accumulate=True,
        )
    valid = counts.squeeze(-1) > 0
    mean_chunks = sums / counts.clamp(min=1.0)
    print(
        f"[INFO] Posterior lookup table: motions={mean_chunks.shape[0]}, steps={mean_chunks.shape[1]}, "
        f"valid_keys={int(valid.sum().item())}, chunks={total}"
    )
    return PosteriorActionLookup(mean_chunks, valid)


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

    model, ckpt = load_model(args_cli.model, device)
    print(f"[INFO] Loaded action CVAE: {args_cli.model}")
    print(
        f"[INFO] obs_dim={ckpt['obs_dim']} base_obs_dim={ckpt.get('base_obs_dim')} "
        f"action_dim={ckpt['action_dim']} base_action_dim={ckpt.get('base_action_dim', 29)} "
        f"action_horizon={ckpt.get('action_horizon', 1)} latent_dim={ckpt['latent_dim']} "
        f"obs_slices={ckpt['obs_slices']}"
    )
    action_horizon = int(ckpt.get("action_horizon", 1))
    base_action_dim = int(ckpt.get("base_action_dim", 29))
    posterior_lookup = None
    if args_cli.mode == "teacher_posterior" and action_horizon > 1:
        if args_cli.posterior_chunk_source == "lookup" and not args_cli.posterior_rollout_data:
            raise ValueError(
                "--mode teacher_posterior with action_horizon>1 requires --posterior_rollout_data "
                "so the evaluator can build real H-step teacher action chunks."
            )
        if args_cli.posterior_chunk_source == "lookup":
            posterior_lookup = build_posterior_action_lookup(args_cli.posterior_rollout_data, ckpt).to(device)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    base_env = env.unwrapped
    cmd = base_env.command_manager.get_term("motion")

    teacher_policy = None
    needs_live_teacher = args_cli.mode == "teacher_posterior" and (
        action_horizon == 1
        or args_cli.posterior_missing == "repeat_teacher"
        or args_cli.posterior_chunk_source == "repeat_live_teacher"
    )
    if needs_live_teacher:
        if agent_cfg.load_run is None or agent_cfg.load_checkpoint is None:
            raise ValueError("--mode teacher_posterior requires --load_run and --checkpoint for the expert policy.")
        log_root = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        resume_path = get_checkpoint_path(os.path.abspath(log_root), agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Loading teacher for posterior oracle: {resume_path}")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        teacher_policy = runner.get_inference_policy(device=device)

    builder = V10ObsBuilder(
        num_envs=args_cli.num_envs,
        num_joints=cmd.robot.data.joint_pos.shape[1],
        device=device,
    )
    builder.init_segment_bounds(cmd)
    obs_v3, _ = env.get_observations()

    episode_step = torch.zeros(args_cli.num_envs, device=device)
    peak_ball_speed = torch.zeros(args_cli.num_envs, device=device)
    best_dir_align = torch.full((args_cli.num_envs,), -1.0, device=device)
    action_norm_sum = torch.zeros(args_cli.num_envs, device=device)
    action_norm_count = torch.zeros(args_cli.num_envs, device=device)
    last_motion_idx = cmd.motion_idx.clone()

    results: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    episodes = 0
    step = 0
    max_steps = args_cli.eval_episodes * 800
    print(f"[INFO] Rollout mode={args_cli.mode}, episodes={args_cli.eval_episodes}, envs={args_cli.num_envs}")
    chunk_steps = args_cli.chunk_steps if args_cli.chunk_steps > 0 else action_horizon
    chunk_steps = max(1, min(chunk_steps, action_horizon))
    use_chunk_queue = bool(args_cli.execute_chunk and action_horizon > 1)
    action_queue = torch.zeros(args_cli.num_envs, action_horizon, base_action_dim, device=device)
    queue_pos = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
    queue_remaining = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
    posterior_diag = _make_posterior_diag()
    if use_chunk_queue:
        print(f"[INFO] Chunk execution enabled: action_horizon={action_horizon}, chunk_steps={chunk_steps}")

    while simulation_app.is_running() and episodes < args_cli.eval_episodes and step < max_steps:
        obs_full = builder.compute(base_env, cmd)
        motion_idx_before = cmd.motion_idx.clone()
        time_step_before = cmd.time_steps.clone()
        with torch.no_grad():
            if args_cli.mode == "teacher_posterior" and not use_chunk_queue:
                if action_horizon == 1:
                    assert teacher_policy is not None
                    teacher_action = teacher_policy(obs_v3)
                    actions = decode_teacher_posterior(model, ckpt, obs_full, teacher_action)
                else:
                    if args_cli.posterior_chunk_source == "repeat_live_teacher":
                        assert teacher_policy is not None
                        teacher_action = teacher_policy(obs_v3)
                        teacher_chunk = _repeat_teacher_action_chunk(teacher_action, action_horizon)
                        valid_chunk = torch.ones(args_cli.num_envs, dtype=torch.bool, device=device)
                    else:
                        assert posterior_lookup is not None
                        teacher_chunk, valid_chunk = posterior_lookup.lookup(motion_idx_before, time_step_before)
                        if not bool(valid_chunk.all().item()):
                            if args_cli.posterior_missing == "error":
                                missing = int((~valid_chunk).sum().item())
                                raise RuntimeError(f"Missing {missing} posterior chunks for current rollout state.")
                            assert teacher_policy is not None
                            teacher_action = teacher_policy(obs_v3)
                            fallback = _repeat_teacher_action_chunk(teacher_action, action_horizon)
                            teacher_chunk = torch.where(valid_chunk[:, None], teacher_chunk, fallback)
                    decoded_chunk = decode_teacher_posterior_chunk(model, ckpt, obs_full, teacher_chunk)
                    _accumulate_posterior_diag(
                        posterior_diag,
                        decoded_chunk,
                        teacher_chunk,
                        valid_chunk,
                        ckpt["action_mean"],
                        ckpt["action_std"],
                    )
                    actions = decoded_chunk[:, 0]
            elif use_chunk_queue:
                refresh = queue_remaining <= 0
                if torch.any(refresh):
                    if args_cli.mode == "teacher_posterior":
                        if args_cli.posterior_chunk_source == "repeat_live_teacher":
                            assert teacher_policy is not None
                            teacher_action = teacher_policy(obs_v3)
                            new_chunk_teacher = _repeat_teacher_action_chunk(teacher_action, action_horizon)
                            valid_chunk = torch.ones(args_cli.num_envs, dtype=torch.bool, device=device)
                        else:
                            assert posterior_lookup is not None
                            new_chunk_teacher, valid_chunk = posterior_lookup.lookup(motion_idx_before, time_step_before)
                            if not bool(valid_chunk[refresh].all().item()):
                                if args_cli.posterior_missing == "error":
                                    missing = int((~valid_chunk[refresh]).sum().item())
                                    raise RuntimeError(f"Missing {missing} posterior chunks for refreshed chunk states.")
                                assert teacher_policy is not None
                                teacher_action = teacher_policy(obs_v3)
                                fallback = _repeat_teacher_action_chunk(teacher_action, action_horizon)
                                new_chunk_teacher = torch.where(valid_chunk[:, None], new_chunk_teacher, fallback)
                        new_chunk = decode_teacher_posterior_chunk(model, ckpt, obs_full, new_chunk_teacher)
                        _accumulate_posterior_diag(
                            posterior_diag,
                            new_chunk,
                            new_chunk_teacher,
                            refresh & valid_chunk,
                            ckpt["action_mean"],
                            ckpt["action_std"],
                        )
                    else:
                        new_chunk = decode_action_chunk(model, ckpt, obs_full, args_cli.mode, args_cli.sample_scale)
                    action_queue[refresh] = new_chunk[refresh]
                    queue_pos[refresh] = 0
                    queue_remaining[refresh] = chunk_steps
                env_ids = torch.arange(args_cli.num_envs, device=device)
                actions = action_queue[env_ids, queue_pos]
                queue_pos.add_(1).clamp_(max=action_horizon - 1)
                queue_remaining.sub_(1)
            else:
                actions = decode_action(model, ckpt, obs_full, args_cli.mode, args_cli.sample_scale)
            obs_v3, _, dones, _ = env.step(actions)
        builder.update_history(base_env, cmd, actions, dones)

        soccer_ball = base_env.scene["soccer_ball"]
        ball_pos_w = soccer_ball.data.root_pos_w[:, :3]
        ball_vel_w = soccer_ball.data.root_lin_vel_w[:, :3]
        ball_speed = torch.norm(ball_vel_w[:, :2], dim=-1)
        kick_dir = _get_kick_direction(base_env, cmd, ball_pos_w)
        ball_dir = ball_vel_w[:, :2] / ball_speed.unsqueeze(-1).clamp(min=1.0e-6)
        align = torch.sum(ball_dir * kick_dir, dim=-1).clamp(-1.0, 1.0)
        moving = ball_speed > 0.5

        peak_ball_speed = torch.maximum(peak_ball_speed, ball_speed)
        best_dir_align = torch.where(moving, torch.maximum(best_dir_align, align), best_dir_align)
        episode_step += 1
        action_norm_sum += torch.norm(actions, dim=-1)
        action_norm_count += 1
        step += 1

        if dones.any():
            done_ids = dones.nonzero(as_tuple=True)[0]
            for idx in done_ids:
                if episodes >= args_cli.eval_episodes:
                    break
                i = int(idx.item())
                mid = int(motion_idx_before[i].item())
                motion_name = cmd.motion.motion_name[mid] if mid < len(cmd.motion.motion_name) else str(mid)
                bspd = float(peak_ball_speed[i].item())
                dira = float(best_dir_align[i].item())
                length = int(episode_step[i].item())
                success = bspd >= args_cli.ball_speed_success and dira >= args_cli.direction_success
                fall_like = length < 100 and bspd < args_cli.ball_speed_success
                results[motion_name].append(
                    {
                        "motion_id": mid,
                        "length": length,
                        "peak_ball_speed": bspd,
                        "best_dir_align": dira,
                        "success": bool(success),
                        "fall_like": bool(fall_like),
                        "mean_action_norm": float((action_norm_sum[i] / action_norm_count[i].clamp(min=1)).item()),
                    }
                )
                episodes += 1

            peak_ball_speed[done_ids] = 0.0
            best_dir_align[done_ids] = -1.0
            episode_step[done_ids] = 0.0
            action_norm_sum[done_ids] = 0.0
            action_norm_count[done_ids] = 0.0
            last_motion_idx[done_ids] = cmd.motion_idx[done_ids]
            queue_remaining[done_ids] = 0
            queue_pos[done_ids] = 0

        if step % 500 == 0:
            print(f"[EVAL] Step {step} | episodes={episodes}/{args_cli.eval_episodes}")

    all_eps = [ep for eps in results.values() for ep in eps]
    success_rate = 100.0 * sum(ep["success"] for ep in all_eps) / max(len(all_eps), 1)
    fall_rate = 100.0 * sum(ep["fall_like"] for ep in all_eps) / max(len(all_eps), 1)
    posterior_summary = _finalize_posterior_diag(posterior_diag)

    print("\n" + "=" * 96)
    print("  ACTION-CVAE DECODER CLOSED-LOOP ROLLOUT")
    print("=" * 96)
    print(f"Model: {args_cli.model}")
    print(f"Mode:  {args_cli.mode}")
    if args_cli.mode == "teacher_posterior" and action_horizon > 1:
        print(f"Posterior chunk source: {args_cli.posterior_chunk_source}")
    if use_chunk_queue:
        print(f"Chunk: execute_chunk=True, chunk_steps={chunk_steps}")
    print(f"Episodes={len(all_eps)}  Success={success_rate:.1f}%  FallLike={fall_rate:.1f}%")
    print(f"Peak ball speed={_mean([ep['peak_ball_speed'] for ep in all_eps]):.2f} m/s  "
          f"DirAlign={_mean([ep['best_dir_align'] for ep in all_eps]):.2f}  "
          f"Length={_mean([ep['length'] for ep in all_eps]):.1f}")
    if args_cli.mode == "teacher_posterior" and action_horizon > 1:
        print(
            "Posterior diag: "
            f"lookup={100.0 * posterior_summary['lookup_valid_rate']:.1f}% "
            f"first_mse={posterior_summary['first_recon_mse']:.6f} "
            f"full_mse={posterior_summary['full_recon_mse']:.6f} "
            f"first_std_mse={posterior_summary['first_std_recon_mse']:.6f} "
            f"full_std_mse={posterior_summary['full_std_recon_mse']:.6f} "
            f"first_norm teacher/dec="
            f"{posterior_summary['teacher_first_action_norm']:.2f}/"
            f"{posterior_summary['decoded_first_action_norm']:.2f}"
        )
    print("-" * 96)
    print(f"{'Motion':45s} {'N':>4s} {'Succ%':>7s} {'Fall%':>7s} {'BSpd':>7s} {'DirA':>7s} {'Len':>7s} {'ActN':>7s}")
    for motion_name, eps in sorted(results.items()):
        n = len(eps)
        succ = 100.0 * sum(ep["success"] for ep in eps) / max(n, 1)
        fall = 100.0 * sum(ep["fall_like"] for ep in eps) / max(n, 1)
        print(
            f"{motion_name[:45]:45s} {n:4d} {succ:7.1f} {fall:7.1f} "
            f"{_mean([ep['peak_ball_speed'] for ep in eps]):7.2f} "
            f"{_mean([ep['best_dir_align'] for ep in eps]):7.2f} "
            f"{_mean([ep['length'] for ep in eps]):7.1f} "
            f"{_mean([ep['mean_action_norm'] for ep in eps]):7.2f}"
        )
    print("=" * 96)

    output_json = args_cli.output_json
    if output_json is None:
        stem = os.path.splitext(os.path.basename(args_cli.model))[0]
        suffix = args_cli.mode
        if use_chunk_queue:
            suffix = f"{suffix}_chunk{chunk_steps}"
        output_json = os.path.join("logs", "action_cvae_rollout", f"{stem}_{suffix}.json")
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args_cli.model,
                "mode": args_cli.mode,
                "posterior_chunk_source": args_cli.posterior_chunk_source
                if args_cli.mode == "teacher_posterior" and action_horizon > 1
                else None,
                "execute_chunk": use_chunk_queue,
                "chunk_steps": chunk_steps if use_chunk_queue else 1,
                "success_rate": success_rate,
                "fall_like_rate": fall_rate,
                "posterior_diagnostics": posterior_summary
                if args_cli.mode == "teacher_posterior" and action_horizon > 1
                else None,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"[INFO] Saved JSON: {output_json}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
