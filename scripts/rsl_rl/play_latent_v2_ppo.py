"""Play/visualize a latent v2 PPO checkpoint.

Usage:
    # Live viewer:
    CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/play_latent_v2_ppo.py \
        --task Anchor-CG-Kick-G1-Soccer-RNN-v0 \
        --motion_path motions/Video_hmr4d_seed \
        --latent_model models/latent_v2/online_distill.pt \
        --load_run "2026-05-25_20-32-16_latent_v2_ppo" \
        --checkpoint model_1000.pt \
        --lab_scale 2.0 \
        --num_envs 1

    # Record video:
    CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/play_latent_v2_ppo.py \
        --task Anchor-CG-Kick-G1-Soccer-RNN-v0 \
        --motion_path motions/Video_hmr4d_seed \
        --latent_model models/latent_v2/online_distill.pt \
        --load_run "2026-05-25_20-32-16_latent_v2_ppo" \
        --checkpoint model_1000.pt \
        --lab_scale 2.0 \
        --num_envs 1 \
        --video --video_length 600 --headless
"""

import argparse
import datetime
import glob
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play latent v2 PPO checkpoint.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=600)
parser.add_argument("--dual_view", action="store_true", default=False)
parser.add_argument("--path_tracing", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Anchor-CG-Kick-G1-Soccer-RNN-v0")
parser.add_argument("--motion_path", type=str, required=True)
parser.add_argument("--motion_file", type=str, default=None)

# Latent model
parser.add_argument("--latent_model", type=str, required=True,
                    help="Path to Stage 2B latent model checkpoint.")
parser.add_argument("--lab_scale", type=float, default=2.0)
parser.add_argument("--latent_clip", type=float, default=5.0)
parser.add_argument("--lab_barrier_weight", type=float, default=0.0)
parser.add_argument("--lab_barrier_limit", type=float, default=2.5)
parser.add_argument("--policy_obs_mode", type=str, default="full", choices=("full", "task", "task_features"),
                    help="High-level PPO observation used by the checkpoint.")

import cli_args  # isort: skip
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video or args_cli.dual_view:
    args_cli.enable_cameras = True
    if not hasattr(args_cli, 'headless'):
        args_cli.headless = True
if getattr(args_cli, "headless", False) and "DISPLAY" in os.environ:
    os.environ.pop("DISPLAY", None)
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import soccer.tasks  # noqa: F401
from soccer.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner
from latent_v2_models import LatentActionModel
from compute_task_features import compute_ball_foot_relation, TASK_FEATURES_DIM


def load_latent_model(path: str, device: str) -> tuple[LatentActionModel, dict]:
    """Load frozen latent model from Stage 2B checkpoint."""
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
    for param in model.parameters():
        param.requires_grad_(False)
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
    """RSL-RL wrapper: PPO outputs z_dim latent actions, decoded to 29D joint actions."""

    def __init__(self, env, *, latent_model_path, lab_scale=2.0, latent_clip=5.0,
                 lab_barrier_weight=0.0, lab_barrier_limit=2.5, policy_obs_mode="full"):
        super().__init__(env)
        self.lab_scale = lab_scale
        self.latent_clip = latent_clip
        self.lab_barrier_weight = lab_barrier_weight
        self.lab_barrier_limit = lab_barrier_limit
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
            low=-self.latent_clip, high=self.latent_clip,
            shape=(self.z_dim,), dtype=float,
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space, self.num_envs
        )
        self._cached_obs_v3 = None
        self._cached_task_features = None
        self._use_task_features = (self.policy_obs_mode == "task_features")
        self.base_env = self.env.unwrapped
        self._last_maha = torch.zeros(self.num_envs, device=self.device)
        self._last_barrier_penalty = torch.zeros(self.num_envs, device=self.device)

    @torch.no_grad()
    def _decode_latent(self, obs_v3, u):
        u_clipped = u.clamp(-self.latent_clip, self.latent_clip)
        tf = self._cached_task_features if self._use_task_features else None
        dec_obs = self.latent_model.select_decoder_obs(obs_v3, task_features=tf)
        p_mu, p_logvar = self.latent_model.prior(dec_obs)
        p_std = torch.exp(0.5 * p_logvar)
        z = p_mu + self.lab_scale * p_std * torch.tanh(u_clipped)
        action = self.latent_model.decoder(dec_obs, z)
        self._last_maha = torch.norm((z - p_mu) / p_std.clamp(min=1e-6), dim=-1)
        return action

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
            self._cached_obs_v3 = obs_v3.clone()
        joint_action = self._decode_latent(obs_v3, latent_actions)
        obs, rew, dones, extras = super().step(joint_action)
        self._cached_obs_v3 = obs.clone()
        self._compute_task_features()
        policy_obs = self._select_policy_obs(obs)
        return policy_obs, rew, dones, self._policy_extras(extras, policy_obs)


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
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device

    if args_cli.motion_file:
        env_cfg.commands.motion.motion_files = [args_cli.motion_file]
    else:
        env_cfg.commands.motion.motion_files = get_motion_files(args_cli.motion_path)

    # Camera setup for video
    if args_cli.video:
        env_cfg.viewer.eye = (5.0, 5.0, 3.0)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.5)
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    print(f"[INFO] Loading PPO checkpoint from: {resume_path}")

    # Create env
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Video wrapper
    if args_cli.video:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_dir = os.path.join(log_dir, "videos", f"play_{timestamp}")
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording video ({args_cli.video_length} steps) to: {video_dir}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Latent PPO wrapper (decodes 16D → 29D)
    env = LatentPPOEnvWrapper(
        env,
        latent_model_path=args_cli.latent_model,
        lab_scale=args_cli.lab_scale,
        latent_clip=args_cli.latent_clip,
        lab_barrier_weight=args_cli.lab_barrier_weight,
        lab_barrier_limit=args_cli.lab_barrier_limit,
        policy_obs_mode=args_cli.policy_obs_mode,
    )
    print(
        f"[INFO] LatentPPO wrapper: policy_obs={env.num_obs}, decoder_obs={env.obs_dim_latent}, "
        f"obs_mode={args_cli.policy_obs_mode}, actions={env.num_actions} (z_dim), "
        f"lab_scale={args_cli.lab_scale}"
    )

    # Load PPO runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Dual-view recorder setup
    dual_recorder = None
    if args_cli.dual_view:
        from dual_view_recorder import DualViewRecorder
        video_dir = os.path.join(log_dir, "videos",
                                 f"dual_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        dual_recorder = DualViewRecorder(
            env=env.unwrapped if hasattr(env, 'unwrapped') else env,
            output_dir=video_dir,
            resolution=(960, 540),
            front_offset=(4.0, 3.0, 2.5),
            back_offset=(-4.0, -3.0, 2.5),
            lookat_offset=0.5,
            fps=30,
            path_tracing=args_cli.path_tracing,
        )
        dual_recorder.setup()
        for _ in range(5):
            env.unwrapped.sim.render()
        print(f"[INFO] Dual-view recording: {args_cli.video_length} steps → {video_dir}")

    # Play loop
    obs, _ = env.get_observations()
    timestep = 0

    print(f"[INFO] Playing... (Ctrl+C to stop)")
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                actions = policy(obs).clone()
            obs, _, _, _ = env.step(actions)

            if dual_recorder is not None:
                dual_recorder.capture()

            if args_cli.video or args_cli.dual_view:
                timestep += 1
                if timestep == args_cli.video_length:
                    break
    except KeyboardInterrupt:
        print(f"[INFO] Stopped at timestep={timestep}")
    finally:
        if dual_recorder is not None:
            dual_recorder.save()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
