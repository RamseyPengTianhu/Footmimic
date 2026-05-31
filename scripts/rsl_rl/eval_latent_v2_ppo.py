"""Evaluate latent v2 PPO checkpoint: Kick%, Fall%, BallSpeed over N episodes.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/eval_latent_v2_ppo.py \
        --task Anchor-CG-Kick-G1-Soccer-RNN-v0 \
        --motion_path motions/Video_hmr4d_seed \
        --latent_model models/latent_v2/online_distill.pt \
        --load_run "2026-05-25_20-32-16_latent_v2_ppo" \
        --checkpoint model_1000.pt \
        --lab_scale 2.0 \
        --num_envs 32 \
        --num_episodes 100 \
        --device cuda:0 --headless
"""

import argparse
import glob
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate latent v2 PPO.")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--task", type=str, default="Anchor-CG-Kick-G1-Soccer-RNN-v0")
parser.add_argument("--motion_path", type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=100)

# Latent model
parser.add_argument("--latent_model", type=str, required=True)
parser.add_argument("--lab_scale", type=float, default=2.0)
parser.add_argument("--latent_clip", type=float, default=5.0)
parser.add_argument("--policy_obs_mode", type=str, default="full", choices=("full", "task", "task_features"),
                    help="High-level PPO observation used by the checkpoint.")

import cli_args  # isort: skip
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = False
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import numpy as np

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import soccer.tasks  # noqa: F401
from soccer.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner
from latent_v2_models import LatentActionModel
from compute_task_features import compute_ball_foot_relation, TASK_FEATURES_DIM


def load_latent_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    decoder_obs_mode = ckpt.get("decoder_obs_mode", "full")
    model = LatentActionModel(
        obs_dim=int(ckpt["obs_dim"]),
        action_dim=int(ckpt["action_dim"]),
        z_dim=int(ckpt["z_dim"]),
        hidden_dims=list(ckpt["hidden_dims"]),
        decoder_obs_mode=decoder_obs_mode,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[INFO] Loaded latent model: decoder_obs_mode={decoder_obs_mode}")
    return model, ckpt


def latent_v2_policy_obs_dim(decoder_obs_dim: int, mode: str) -> int:
    if mode == "full":
        return decoder_obs_dim
    if mode == "task":
        if decoder_obs_dim < 160:
            raise ValueError(f"task policy obs expects obs_v3 >=160D, got {decoder_obs_dim}")
        return 3 + (decoder_obs_dim - 64)
    if mode == "task_features":
        return 3 + (decoder_obs_dim - 64) + TASK_FEATURES_DIM
    raise ValueError(f"Unknown policy_obs_mode={mode!r}")


def select_latent_v2_policy_obs(
    obs_v3: torch.Tensor, mode: str, task_features: torch.Tensor | None = None
) -> torch.Tensor:
    if mode == "full":
        return obs_v3
    if mode == "task":
        if obs_v3.shape[-1] < 160:
            raise ValueError(f"task policy obs expects obs_v3 >=160D, got {obs_v3.shape[-1]}")
        return torch.cat((obs_v3[:, 58:61], obs_v3[:, 64:]), dim=-1)
    if mode == "task_features":
        if task_features is None:
            raise ValueError("task_features must be provided for policy_obs_mode='task_features'")
        proprio = torch.cat((obs_v3[:, 58:61], obs_v3[:, 64:]), dim=-1)
        return torch.cat((proprio, task_features), dim=-1)
    raise ValueError(f"Unknown policy_obs_mode={mode!r}")


class LatentPPOEnvWrapper(RslRlVecEnvWrapper):
    def __init__(self, env, *, latent_model_path, lab_scale=2.0, latent_clip=5.0, policy_obs_mode="full"):
        super().__init__(env)
        self.lab_scale = lab_scale
        self.latent_clip = latent_clip
        self.policy_obs_mode = policy_obs_mode
        self.latent_model, self.latent_ckpt = load_latent_model(latent_model_path, self.device)
        self.z_dim = int(self.latent_ckpt["z_dim"])
        self.obs_dim_latent = int(self.latent_ckpt["obs_dim"])
        self.policy_obs_dim = latent_v2_policy_obs_dim(self.obs_dim_latent, self.policy_obs_mode)
        self.num_actions = self.z_dim
        self.num_obs = self.policy_obs_dim
        if self.policy_obs_mode in ("task", "task_features"):
            self.num_privileged_obs = self.policy_obs_dim
        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-latent_clip, high=latent_clip, shape=(self.z_dim,), dtype=float)
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space, self.num_envs)
        self._cached_obs_v3 = None
        self._cached_task_features = None
        self._use_task_features = (self.policy_obs_mode == "task_features")
        self.base_env = self.env.unwrapped

    @torch.no_grad()
    def _decode(self, obs_v3, u):
        u = u.clamp(-self.latent_clip, self.latent_clip)
        tf = self._cached_task_features if self._use_task_features else None
        dec_obs = self.latent_model.select_decoder_obs(obs_v3, task_features=tf)
        p_mu, p_logvar = self.latent_model.prior(dec_obs)
        p_std = torch.exp(0.5 * p_logvar)
        z = p_mu + self.lab_scale * p_std * torch.tanh(u)
        return self.latent_model.decoder(dec_obs, z)

    def _compute_task_features(self):
        if not self._use_task_features:
            return None
        self._cached_task_features = compute_ball_foot_relation(self.base_env)
        return self._cached_task_features

    def _select_policy_obs(self, obs_v3):
        tf = self._cached_task_features if self._use_task_features else None
        return select_latent_v2_policy_obs(obs_v3, self.policy_obs_mode, task_features=tf)

    def _policy_extras(self, extras, policy_obs):
        if self.policy_obs_mode not in ("task", "task_features"):
            return extras
        extras = dict(extras)
        extras["observations"] = {"policy": policy_obs, "critic": policy_obs}
        return extras

    def get_observations(self):
        obs, extras = super().get_observations()
        self._cached_obs_v3 = obs.clone()
        self._compute_task_features()
        policy_obs = self._select_policy_obs(obs)
        return policy_obs, self._policy_extras(extras, policy_obs)

    def reset(self):
        obs, extras = super().reset()
        self._cached_obs_v3 = obs.clone()
        self._compute_task_features()
        policy_obs = self._select_policy_obs(obs)
        return policy_obs, self._policy_extras(extras, policy_obs)

    def step(self, latent_actions):
        obs_v3 = self._cached_obs_v3
        if obs_v3 is None:
            obs_v3, _ = super().get_observations()
        joint_action = self._decode(obs_v3, latent_actions)
        obs, rew, dones, extras = super().step(joint_action)
        self._cached_obs_v3 = obs.clone()
        self._compute_task_features()
        policy_obs = self._select_policy_obs(obs)
        return policy_obs, rew, dones, self._policy_extras(extras, policy_obs)


def get_motion_files(motion_path):
    if os.path.isfile(motion_path):
        return [motion_path]
    if os.path.isdir(motion_path):
        return sorted(glob.glob(os.path.join(motion_path, "*.npz")))
    raise ValueError(f"Invalid: {motion_path}")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
    env_cfg.commands.motion.motion_files = get_motion_files(args_cli.motion_path)
    device = env_cfg.sim.device

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading PPO from: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = LatentPPOEnvWrapper(
        env, latent_model_path=args_cli.latent_model,
        lab_scale=args_cli.lab_scale, latent_clip=args_cli.latent_clip,
        policy_obs_mode=args_cli.policy_obs_mode)
    print(
        f"[INFO] LatentPPO wrapper: policy_obs={env.num_obs}, decoder_obs={env.obs_dim_latent}, "
        f"obs_mode={args_cli.policy_obs_mode}, actions={env.num_actions}, lab_scale={args_cli.lab_scale}"
    )

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=device)

    # ── Eval loop ──────────────────────────────────────────────────────────
    num_eps = args_cli.num_episodes
    N = args_cli.num_envs
    unwrapped = env.unwrapped

    obs, _ = env.get_observations()
    ep_len = torch.zeros(N, device=device)
    ball_contacted = torch.zeros(N, dtype=torch.bool, device=device)
    max_ball_speed = torch.zeros(N, device=device)

    episodes = 0
    kicks = 0
    falls = 0
    ball_speeds = []

    print(f"\n[INFO] Evaluating {num_eps} episodes ({N} parallel envs)...")

    step = 0
    while episodes < num_eps:
        with torch.inference_mode():
            actions = policy(obs).clone()
        obs, _, dones, _ = env.step(actions)
        ep_len += 1
        step += 1

        # Track ball
        try:
            ball = unwrapped.scene["soccer_ball"]
            bvel = ball.data.root_lin_vel_w[:, :2]
            bspd = torch.norm(bvel, dim=-1)
            ball_contacted |= (bspd > 0.5)
            max_ball_speed = torch.maximum(max_ball_speed, bspd)
        except Exception:
            pass

        if dones.any():
            done_ids = dones.nonzero(as_tuple=True)[0]
            for idx in done_ids:
                if episodes >= num_eps:
                    break
                i = idx.item()
                episodes += 1
                if ball_contacted[i]:
                    kicks += 1
                    ball_speeds.append(max_ball_speed[i].item())
                if ep_len[i] < 100:
                    falls += 1
                ep_len[i] = 0
                ball_contacted[i] = False
                max_ball_speed[i] = 0.0

        if step % 200 == 0:
            print(f"  Step {step}, episodes: {episodes}/{num_eps}")

    # ── Report ─────────────────────────────────────────────────────────────
    kick_pct = kicks / max(episodes, 1) * 100
    fall_pct = falls / max(episodes, 1) * 100
    avg_bspd = np.mean(ball_speeds) if ball_speeds else 0.0

    print(f"\n{'='*60}")
    print(f"  Latent v2 PPO Evaluation ({episodes} episodes)")
    print(f"{'='*60}")
    print(f"  Kick%:     {kick_pct:.1f}%  ({kicks}/{episodes})")
    print(f"  Fall%:     {fall_pct:.1f}%  ({falls}/{episodes})")
    print(f"  Avg BSpd:  {avg_bspd:.2f} m/s  (kick episodes only)")
    if ball_speeds:
        print(f"  Max BSpd:  {max(ball_speeds):.2f} m/s")
        print(f"  Min BSpd:  {min(ball_speeds):.2f} m/s")
    print(f"{'='*60}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
