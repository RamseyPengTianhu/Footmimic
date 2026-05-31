"""Train a high-level latent-residual PPO policy over a frozen action-CVAE.

This is the LATENT-style step after action-space distillation:

    high-level action u_t
    z_t = mu_prior(s_t) + lambda * sigma_prior(s_t) * tanh(u_t)
    low-level action = decoder(s_t, z_t)

The wrapped RSL-RL environment exposes ``u_t`` as the policy action while the
underlying IsaacLab task still receives the decoded 29D joint-position action.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train latent-residual PPO over frozen action-CVAE decoder.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="Anchor-V3CVAE-Kick-G1-Soccer-RNN-v0")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--motion_path", type=str, required=True)
parser.add_argument("--action_cvae", type=str, required=True)
parser.add_argument("--latent_scale", type=float, default=0.75)
parser.add_argument("--latent_clip", type=float, default=5.0)
parser.add_argument("--pd_action_clip", type=float, default=0.0)
parser.add_argument("--pd_residual_scale", type=float, default=0.0)
parser.add_argument("--pd_residual_joint_scope", type=str, default="all")
parser.add_argument("--pd_residual_gate_dist", type=float, default=0.9)
parser.add_argument("--pd_residual_gate_temp", type=float, default=0.2)
parser.add_argument("--pd_residual_closing_threshold", type=float, default=0.0)
parser.add_argument("--pd_residual_closing_temp", type=float, default=0.5)
parser.add_argument("--latent_barrier_weight", type=float, default=0.0)
parser.add_argument("--latent_barrier_limit", type=float, default=2.5)

import cli_args  # isort: skip

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True
if getattr(args_cli, "headless", False) and "DISPLAY" in os.environ:
    print(f"[INFO] Headless mode: clearing DISPLAY={os.environ['DISPLAY']!r} before launching Isaac Sim")
    os.environ.pop("DISPLAY", None)
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
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


def get_motion_files(motion_path: str) -> list[str]:
    if os.path.isfile(motion_path):
        return [motion_path]
    if os.path.isdir(motion_path):
        files = sorted(glob.glob(os.path.join(motion_path, "*.npz")))
        if not files:
            raise ValueError(f"No .npz files found in directory: {motion_path}")
        print(f"Found {len(files)} motion files in {motion_path}")
        for file in files:
            print(f"  - {os.path.basename(file)}")
        return files
    raise ValueError(f"Invalid path: {motion_path}")


def load_action_cvae(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = StateActionCVAE(
        obs_dim=int(ckpt["obs_dim"]),
        action_dim=int(ckpt["action_dim"]),
        latent_dim=int(ckpt["latent_dim"]),
        hidden_dims=list(ckpt["hidden_dims"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, ckpt


class ActionCVAELatentRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """RSL-RL wrapper exposing latent residuals instead of raw PD actions."""

    def __init__(
        self,
        env,
        *,
        action_cvae_path: str,
        latent_scale: float = 0.75,
        latent_clip: float = 5.0,
        pd_action_clip: float = 0.0,
        pd_residual_scale: float = 0.0,
        pd_residual_joint_scope: str = "all",
        pd_residual_gate_dist: float = 0.9,
        pd_residual_gate_temp: float = 0.2,
        pd_residual_closing_threshold: float = 0.0,
        pd_residual_closing_temp: float = 0.5,
        latent_barrier_weight: float = 0.0,
        latent_barrier_limit: float = 2.5,
    ):
        super().__init__(env)
        self.latent_scale = float(latent_scale)
        self.latent_clip = float(latent_clip)
        self.latent_barrier_weight = float(latent_barrier_weight)
        self.latent_barrier_limit = float(latent_barrier_limit)
        self.pd_action_clip = float(pd_action_clip)
        self.pd_residual_scale = float(pd_residual_scale)
        self.pd_residual_joint_scope = pd_residual_joint_scope
        self.pd_residual_gate_dist = float(pd_residual_gate_dist)
        self.pd_residual_gate_temp = float(pd_residual_gate_temp)
        self.pd_residual_closing_threshold = float(pd_residual_closing_threshold)
        self.pd_residual_closing_temp = float(pd_residual_closing_temp)
        self.use_pd_residual = self.pd_residual_scale > 0.0

        self.action_cvae, self.action_cvae_ckpt = load_action_cvae(action_cvae_path, self.device)
        self.latent_dim = int(self.action_cvae_ckpt["latent_dim"])
        self.base_action_dim = int(self.action_cvae_ckpt.get("base_action_dim", 29))
        self.action_horizon = int(self.action_cvae_ckpt.get("action_horizon", 1))
        self.obs_slices = list(self.action_cvae_ckpt["obs_slices"])
        self.obs_mean = self.action_cvae_ckpt["obs_mean"].to(self.device)
        self.obs_std = self.action_cvae_ckpt["obs_std"].to(self.device)
        self.action_mean = self.action_cvae_ckpt["action_mean"].to(self.device)
        self.action_std = self.action_cvae_ckpt["action_std"].to(self.device)

        command = self.unwrapped.command_manager.get_term("motion")
        self.pd_residual_joint_ids = self._resolve_pd_residual_joint_ids(command)
        self.pd_residual_dim = int(self.pd_residual_joint_ids.numel())
        self.v10_builder = V10ObsBuilder(
            num_envs=self.unwrapped.num_envs,
            num_joints=command.robot.data.joint_pos.shape[1],
            device=self.unwrapped.device,
        )
        self.v10_builder.init_segment_bounds(command)

        # RSL-RL reads these dimensions when constructing the actor-critic.
        self.num_actions = self.latent_dim + (self.pd_residual_dim if self.use_pd_residual else 0)
        self.num_obs = sum(end - start for start, end in self.obs_slices)
        self.num_privileged_obs = self.num_obs
        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.latent_clip, high=self.latent_clip, shape=(self.num_actions,), dtype=float
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space, self.num_envs
        )

        self._last_obs_full = None
        self._last_latent_maha = torch.zeros(self.num_envs, device=self.device)
        self._last_latent_barrier_penalty = torch.zeros(self.num_envs, device=self.device)
        self._last_pd_action_norm = torch.zeros(self.num_envs, device=self.device)
        self._last_pd_residual_gate = torch.zeros(self.num_envs, device=self.device)
        self._last_pd_residual_norm = torch.zeros(self.num_envs, device=self.device)

    def _resolve_pd_residual_joint_ids(self, command) -> torch.Tensor:
        if not self.use_pd_residual:
            return torch.empty(0, dtype=torch.long, device=self.device)
        scope = self.pd_residual_joint_scope.strip().lower()
        if scope in ("all", "full", "29d"):
            return torch.arange(self.base_action_dim, dtype=torch.long, device=self.device)
        if scope in ("swing_leg", "kick_leg", "right_leg"):
            target_names = [
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
            ]
        elif scope == "swing_leg_no_ankle":
            target_names = [
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
            ]
        else:
            raise ValueError(f"Unknown pd_residual_joint_scope={self.pd_residual_joint_scope!r}")

        joint_names = list(command.robot.data.joint_names)
        ids = []
        for target in target_names:
            matches = [i for i, name in enumerate(joint_names) if name == target or name.endswith(target)]
            if not matches:
                raise ValueError(f"PD residual joint {target!r} not found in robot joints: {joint_names}")
            ids.append(matches[0])
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def _pd_residual_gate(self, obs_full: torch.Tensor) -> torch.Tensor:
        swing_foot_ball_dist = obs_full[:, 392 + 14]
        swing_closing_speed = obs_full[:, 392 + 20]
        dist_temp = max(self.pd_residual_gate_temp, 1.0e-4)
        closing_temp = max(self.pd_residual_closing_temp, 1.0e-4)
        dist_gate = torch.sigmoid((self.pd_residual_gate_dist - swing_foot_ball_dist) / dist_temp)
        closing_gate = torch.sigmoid((swing_closing_speed - self.pd_residual_closing_threshold) / closing_temp)
        return (dist_gate * closing_gate).unsqueeze(-1)

    def _compute_v10(self) -> torch.Tensor:
        command = self.unwrapped.command_manager.get_term("motion")
        return self.v10_builder.compute(self.unwrapped, command)

    def _select_policy_obs(self, obs_full: torch.Tensor) -> torch.Tensor:
        return apply_obs_slices(obs_full, self.obs_slices)

    @torch.no_grad()
    def _decode_latent_residual(self, obs_full: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        latent_residual = residual[:, : self.latent_dim].clamp(-self.latent_clip, self.latent_clip)
        pd_residual = residual[:, self.latent_dim:] if self.use_pd_residual else None
        obs = self._select_policy_obs(obs_full)
        obs_norm = (obs - self.obs_mean) / self.obs_std
        p_mu, p_logvar = self.action_cvae.prior_stats(obs_norm)
        p_std = torch.exp(0.5 * p_logvar)
        z = p_mu + self.latent_scale * p_std * torch.tanh(latent_residual)
        action_norm = self.action_cvae.decode(obs_norm, z)
        action = action_norm * self.action_std + self.action_mean
        action = action.view(action.shape[0], self.action_horizon, self.base_action_dim)[:, 0]
        if self.use_pd_residual:
            gate = self._pd_residual_gate(obs_full)
            delta_scoped = self.pd_residual_scale * gate * torch.tanh(pd_residual)
            delta = torch.zeros_like(action)
            delta[:, self.pd_residual_joint_ids] = delta_scoped
            action = action + delta
            self._last_pd_residual_gate = gate.squeeze(-1)
            self._last_pd_residual_norm = torch.norm(delta, dim=-1)
        else:
            self._last_pd_residual_gate.zero_()
            self._last_pd_residual_norm.zero_()
        if self.pd_action_clip > 0.0:
            action = action.clamp(-self.pd_action_clip, self.pd_action_clip)
        self._last_latent_maha = torch.norm((z - p_mu) / p_std.clamp(min=1.0e-6), dim=-1)
        self._last_latent_barrier_penalty = torch.relu(
            self._last_latent_maha - self.latent_barrier_limit
        ).pow(2)
        self._last_pd_action_norm = torch.norm(action, dim=-1)
        return action

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        self._last_obs_full = self._compute_v10()
        obs = self._select_policy_obs(self._last_obs_full)
        return obs, {"observations": {"policy": obs, "critic": obs}}

    def reset(self) -> tuple[torch.Tensor, dict]:
        super().reset()
        command = self.unwrapped.command_manager.get_term("motion")
        self.v10_builder.init_segment_bounds(command)
        self.v10_builder.reset(torch.arange(self.unwrapped.num_envs, device=self.unwrapped.device))
        self._last_obs_full = self._compute_v10()
        obs = self._select_policy_obs(self._last_obs_full)
        return obs, {"observations": {"policy": obs, "critic": obs}}

    def step(self, latent_actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if self._last_obs_full is None:
            self._last_obs_full = self._compute_v10()
        pd_actions = self._decode_latent_residual(self._last_obs_full, latent_actions)
        obs_dict, rew, terminated, truncated, extras = self.env.step(pd_actions)
        if self.latent_barrier_weight > 0.0:
            barrier = self.latent_barrier_weight * self._last_latent_barrier_penalty
            while barrier.dim() < rew.dim():
                barrier = barrier.unsqueeze(-1)
            rew = rew - barrier
        dones = (terminated | truncated).to(dtype=torch.long)

        command = self.unwrapped.command_manager.get_term("motion")
        self.v10_builder.update_history(self.unwrapped, command, pd_actions, dones)
        self._last_obs_full = self._compute_v10()
        obs = self._select_policy_obs(self._last_obs_full)

        extras["observations"] = {"policy": obs, "critic": obs}
        extras.setdefault("log", {})
        extras["log"]["latent/maha"] = self._last_latent_maha.mean()
        extras["log"]["latent/barrier_penalty"] = self._last_latent_barrier_penalty.mean()
        extras["log"]["latent/barrier_reward"] = -self.latent_barrier_weight * self._last_latent_barrier_penalty.mean()
        extras["log"]["latent/pd_action_norm"] = self._last_pd_action_norm.mean()
        extras["log"]["latent/pd_residual_gate"] = self._last_pd_residual_gate.mean()
        extras["log"]["latent/pd_residual_norm"] = self._last_pd_residual_norm.mean()
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        return obs, rew, dones, extras


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name

    # Start close to the frozen prior; PPO explores bounded residuals through tanh.
    agent_cfg.empirical_normalization = True
    agent_cfg.policy.init_noise_std = 0.25
    agent_cfg.algorithm.entropy_coef = 0.001
    agent_cfg.algorithm.learning_rate = 3.0e-4
    agent_cfg.algorithm.schedule = "adaptive"

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    motion_files = get_motion_files(args_cli.motion_path)
    env_cfg.commands.motion.motion_files = motion_files
    if hasattr(env_cfg.commands.motion, "strike_motion_files"):
        env_cfg.commands.motion.strike_motion_files = motion_files

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = ActionCVAELatentRslRlVecEnvWrapper(
        env,
        action_cvae_path=args_cli.action_cvae,
        latent_scale=args_cli.latent_scale,
        latent_clip=args_cli.latent_clip,
        pd_action_clip=args_cli.pd_action_clip,
        pd_residual_scale=args_cli.pd_residual_scale,
        pd_residual_joint_scope=args_cli.pd_residual_joint_scope,
        pd_residual_gate_dist=args_cli.pd_residual_gate_dist,
        pd_residual_gate_temp=args_cli.pd_residual_gate_temp,
        pd_residual_closing_threshold=args_cli.pd_residual_closing_threshold,
        pd_residual_closing_temp=args_cli.pd_residual_closing_temp,
        latent_barrier_weight=args_cli.latent_barrier_weight,
        latent_barrier_limit=args_cli.latent_barrier_limit,
    )
    print(
        f"[INFO] Latent wrapper: obs={env.num_obs}, latent_actions={env.num_actions}, "
        f"latent_scale={args_cli.latent_scale}, decoder_horizon={env.action_horizon}, "
        f"pd_residual_scale={args_cli.pd_residual_scale}, "
        f"pd_residual_joint_scope={args_cli.pd_residual_joint_scope}, "
        f"latent_barrier_weight={args_cli.latent_barrier_weight}, "
        f"latent_barrier_limit={args_cli.latent_barrier_limit}"
    )

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=None)
    runner.add_git_repo_to_log(__file__)

    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading latent PPO checkpoint from: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
