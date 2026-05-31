"""Quick eval: run DAgger decoder with prior mean z (no PPO), measure Kick%/Fall%/BSpd.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/rsl_rl/eval_dagger_decoder.py \
        --task Anchor-CG-Kick-G1-Soccer-RNN-v0 \
        --motion_path motions/Video_hmr4d_seed \
        --latent_model models/latent_v2/online_distill_phase26.pt \
        --num_envs 32 --num_episodes 200 --device cuda:0 --headless
"""

import argparse
import glob
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate DAgger decoder (prior mean, no PPO).")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--task", type=str, default="Anchor-CG-Kick-G1-Soccer-RNN-v0")
parser.add_argument("--motion_path", type=str, required=True)
parser.add_argument("--latent_model", type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=200)

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
from isaaclab_tasks.utils.hydra import hydra_task_config

import soccer.tasks  # noqa: F401
from latent_v2_models import LatentActionModel
from compute_task_features import compute_ball_foot_relation


def load_latent_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    decoder_obs_mode = ckpt.get("decoder_obs_mode", "full")
    model = LatentActionModel(
        obs_dim=int(ckpt["obs_dim"]),
        action_dim=int(ckpt["action_dim"]),
        z_dim=int(ckpt["z_dim"]),
        hidden_dims=list(ckpt["hidden_dims"]),
        decoder_obs_mode=decoder_obs_mode,
        prior_type=ckpt.get("prior_type", "mlp"),
        num_codes=int(ckpt.get("num_codes", 16)),
        commitment_weight=float(ckpt.get("commitment_weight", 0.25)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


def get_motion_files(motion_path):
    if os.path.isfile(motion_path):
        return [motion_path]
    if os.path.isdir(motion_path):
        return sorted(glob.glob(os.path.join(motion_path, "*.npz")))
    raise ValueError(f"Invalid: {motion_path}")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
    env_cfg.commands.motion.motion_files = get_motion_files(args_cli.motion_path)
    device = env_cfg.sim.device

    # Load decoder
    model, ckpt = load_latent_model(args_cli.latent_model, device)
    decoder_obs_mode = ckpt.get("decoder_obs_mode", "full")
    use_task_features = decoder_obs_mode == "task_features"
    print(f"[INFO] Decoder: mode={decoder_obs_mode}, obs_dim={model.decoder_obs_dim}, "
          f"z_dim={model.z_dim}, best_kick={ckpt.get('best_kick_pct', '?')}%")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env_wrapped = RslRlVecEnvWrapper(env)
    unwrapped = env.unwrapped
    N = args_cli.num_envs

    obs_v3, _ = env_wrapped.get_observations()
    ep_len = torch.zeros(N, device=device)
    ball_contacted = torch.zeros(N, dtype=torch.bool, device=device)
    max_ball_speed = torch.zeros(N, device=device)

    episodes = 0
    kicks = 0
    falls = 0
    no_attempts = 0
    ball_speeds = []

    code_hold = ckpt.get("code_hold", 1)
    print(f"\n[INFO] Evaluating decoder (prior mean, NO PPO) for {args_cli.num_episodes} episodes...")
    print(f"[INFO] code_hold={code_hold}")

    # Code hold state
    held_zq = torch.zeros(N, model.z_dim, device=device)
    held_code = torch.zeros(N, dtype=torch.long, device=device)
    hold_ctr = torch.zeros(N, dtype=torch.long, device=device)

    step = 0
    while episodes < args_cli.num_episodes:
        with torch.inference_mode():
            tf = compute_ball_foot_relation(unwrapped) if use_task_features else None
            if model.prior_type == "vq":
                dec_obs = model.select_decoder_obs(obs_v3, task_features=tf)
                needs_update = (hold_ctr % code_hold == 0)
                if needs_update.any():
                    logits = model.prior(dec_obs)
                    new_code = logits.argmax(dim=-1)
                    new_zq = model.codebook.lookup(new_code)
                    held_code[needs_update] = new_code[needs_update]
                    held_zq[needs_update] = new_zq[needs_update]
                action = model.decoder(dec_obs, held_zq)
            else:
                action = model.act_prior_mean(obs_v3, task_features=tf)

        obs_v3, _, dones, _ = env_wrapped.step(action)
        ep_len += 1
        step += 1
        hold_ctr += 1

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
                if episodes >= args_cli.num_episodes:
                    break
                i = idx.item()
                episodes += 1
                if ball_contacted[i]:
                    kicks += 1
                    ball_speeds.append(max_ball_speed[i].item())
                else:
                    no_attempts += 1
                if ep_len[i] < 100:
                    falls += 1
                ep_len[i] = 0
                ball_contacted[i] = False
                max_ball_speed[i] = 0.0
                hold_ctr[i] = 0

        if step % 200 == 0:
            print(f"  Step {step}, episodes: {episodes}/{args_cli.num_episodes}")

    # Report
    kick_pct = kicks / max(episodes, 1) * 100
    early_fall_pct = falls / max(episodes, 1) * 100
    no_attempt_pct = no_attempts / max(episodes, 1) * 100
    avg_bspd = np.mean(ball_speeds) if ball_speeds else 0.0

    print(f"\n{'='*60}")
    print(f"  DAgger Decoder Eval (prior mean, NO PPO)")
    print(f"  {episodes} episodes, model: {args_cli.latent_model}")
    print(f"{'='*60}")
    print(f"  Kick%:       {kick_pct:.1f}%  ({kicks}/{episodes})")
    print(f"  EarlyFall%:  {early_fall_pct:.1f}%  ({falls}/{episodes})  (ep_len<100)")
    print(f"  NoAttempt%:  {no_attempt_pct:.1f}%  ({no_attempts}/{episodes})")
    print(f"  Avg BSpd:    {avg_bspd:.2f} m/s  (kick episodes only)")
    if ball_speeds:
        print(f"  Max BSpd:    {max(ball_speeds):.2f} m/s")
        print(f"  Min BSpd:    {min(ball_speeds):.2f} m/s")
    print(f"{'='*60}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
