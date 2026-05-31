"""V3.5 Strike Data Collection — Collect discriminator training data from V3 rollouts.

Runs the V3 policy in evaluation mode and records per-frame features at
contact events. Also extracts reference motion strike/approach frames.

Outputs: `models/strike_data.pt` containing:
  - features: [N_total, 39]
  - labels:   [N_total]  (soft labels 0.0–1.0)
  - sources:  [N_total]  (0=ref_neg, 1=ref_pos, 2=rollout_pos, 3=rollout_hard_neg)

Usage:
  python scripts/rsl_rl/collect_strike_data.py \\
    --task Anchor-CG-Kick-G1-Soccer-RNN-v0 \\
    --motion_path motions/Video \\
    --load_run <v3_run_dir> --checkpoint model_12000.pt \\
    --num_envs 64 --num_episodes 200 --headless
"""
import argparse, sys, os, glob
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect strike discriminator training data")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default="Anchor-CG-Kick-G1-Soccer-RNN-v0")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--motion_path", type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=200)
parser.add_argument("--output", type=str, default="models/strike_data.pt")

import cli_args
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import numpy as np
from collections import defaultdict

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import soccer.tasks
from soccer.utils.my_on_policy_runner import MotionOnPolicyRunner
from soccer.tasks.tracking.mdp.strike_discriminator import (
    StrikeFeatureExtractor, extract_reference_features, INPUT_DIM,
)


def get_motion_files(p):
    if os.path.isfile(p): return [p]
    if os.path.isdir(p):
        f = sorted(glob.glob(os.path.join(p, "*.npz")))
        if not f: raise ValueError(f"No .npz in {p}")
        return f
    raise ValueError(f"Invalid: {p}")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.motion.motion_files = get_motion_files(args_cli.motion_path)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # Load V3 policy
    log_root = os.path.join("logs", "rsl_rl", "g1_flat")
    ckpt_path = os.path.join(log_root, args_cli.load_run, args_cli.checkpoint)
    if not os.path.exists(ckpt_path):
        ckpt_path = get_checkpoint_path(log_root, args_cli.load_run, args_cli.checkpoint)

    runner = MotionOnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device,
    )
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=agent_cfg.device)

    # Feature extractor
    uw = env.unwrapped
    command = uw.command_manager.get_term("motion")
    extractor = StrikeFeatureExtractor()
    extractor.init_indices(command.robot)

    # ── Collect rollout data ──
    print(f"\n[collect] Running V3 policy for ~{args_cli.num_episodes} episodes...")
    all_features = []
    all_labels = []
    all_sources = []  # 0=ref_neg, 1=ref_pos, 2=rollout_pos, 3=rollout_hard_neg

    # Per-env state tracking
    n = uw.num_envs
    dev = uw.device
    contact_awarded = torch.zeros(n, dtype=torch.bool, device=dev)
    contact_frame = torch.full((n,), -1, dtype=torch.long, device=dev)
    contact_ball_speed = torch.zeros(n, device=dev)  # ball speed AT contact time
    # Buffer: store features for frames around contact
    WINDOW = 15
    frame_buffer = torch.zeros(n, 500, INPUT_DIM, device=dev)  # max 500 frames per episode
    frame_count = torch.zeros(n, dtype=torch.long, device=dev)

    episodes_done = 0
    obs, _ = env.reset()

    while episodes_done < args_cli.num_episodes:
        with torch.no_grad():
            actions = policy(obs)
        obs, _, dones, infos = env.step(actions)

        # Extract features for all envs
        features = extractor.compute(uw, command)  # [N, 38]

        # Store in per-env buffer
        for i in range(n):
            fc = frame_count[i].item()
            if fc < 500:
                frame_buffer[i, fc] = features[i]
                frame_count[i] += 1

        # Detect ball contact (using ball speed as proxy)
        soccer_ball = uw.scene["soccer_ball"]
        ball_vel = soccer_ball.data.root_lin_vel_w[:, :3]
        ball_speed = torch.norm(ball_vel[:, :2], dim=-1)
        new_contact = (~contact_awarded) & (ball_speed > 0.5)

        if new_contact.any():
            contact_awarded[new_contact] = True
            contact_frame[new_contact] = frame_count[new_contact] - 1
            contact_ball_speed[new_contact] = ball_speed[new_contact]  # save speed AT contact

        # Process done episodes
        if dones.any():
            done_ids = dones.nonzero(as_tuple=True)[0]
            for idx in done_ids:
                i = idx.item()
                fc = frame_count[i].item()
                cf = contact_frame[i].item()

                if cf >= 0 and fc > 0:
                    # This episode had a contact
                    ep_features = frame_buffer[i, :fc].clone()

                    # Check kick quality using ball speed AT CONTACT TIME
                    kicked = contact_awarded[i].item()
                    bs = contact_ball_speed[i].item()

                    # Get kick_frame from motion for comparison
                    mid = command.motion_idx[i].item()
                    kf = command.motion.kick_frames[mid].item()

                    # Determine if this was a "good" kick
                    is_good_kick = kicked and bs > 1.0

                    # Label frames around contact
                    ep_labels = torch.zeros(fc, device=dev)
                    if is_good_kick:
                        # Positive: frames around contact (±5)
                        for t in range(max(0, cf - 5), min(fc, cf + 6)):
                            dist_to_cf = abs(t - cf)
                            if dist_to_cf <= 2:
                                ep_labels[t] = 1.0
                            elif dist_to_cf <= 5:
                                ep_labels[t] = 0.7
                        source = 2  # rollout positive
                    else:
                        # Hard negative: contact happened but kick was bad
                        for t in range(max(0, cf - 3), min(fc, cf + 4)):
                            ep_labels[t] = 0.0  # negative label
                        source = 3  # rollout hard negative

                    # Also label approach frames as negative
                    # (all frames far from contact)
                    ep_sources = torch.full((fc,), source, dtype=torch.long, device=dev)
                    for t in range(fc):
                        if ep_labels[t] == 0.0:
                            ep_sources[t] = 3 if source == 3 else 0

                    # Sample: keep contact-nearby and some random approach
                    # Contact window
                    contact_mask = torch.zeros(fc, dtype=torch.bool, device=dev)
                    for t in range(max(0, cf - WINDOW), min(fc, cf + WINDOW + 1)):
                        contact_mask[t] = True

                    # Random approach sample (up to 2x contact frames)
                    non_contact = (~contact_mask).nonzero(as_tuple=True)[0]
                    n_contact = contact_mask.sum().item()
                    n_sample = min(len(non_contact), n_contact * 2)
                    if n_sample > 0:
                        perm = torch.randperm(len(non_contact), device=dev)[:n_sample]
                        sample_idx = non_contact[perm]
                        keep_mask = contact_mask.clone()
                        keep_mask[sample_idx] = True
                    else:
                        keep_mask = contact_mask

                    if keep_mask.any():
                        all_features.append(ep_features[keep_mask].cpu())
                        all_labels.append(ep_labels[keep_mask].cpu())
                        all_sources.append(ep_sources[keep_mask].cpu())

                episodes_done += 1

                # Reset per-env state
                contact_awarded[i] = False
                contact_frame[i] = -1
                contact_ball_speed[i] = 0.0
                frame_count[i] = 0

            if episodes_done % 50 == 0:
                print(f"  [collect] {episodes_done}/{args_cli.num_episodes} episodes done")

    # ── Add reference motion data ──
    print("\n[collect] Extracting reference motion features...")
    motion_files = get_motion_files(args_cli.motion_path)
    for mf in motion_files:
        data = np.load(mf)
        kf = int(data.get("kick_frame", -1))
        kef = int(data.get("kick_end_frame", -1))
        ref_feat, ref_labels = extract_reference_features(
            {k: data[k] for k in data.files}, kf, kef
        )
        ref_sources = torch.where(
            ref_labels > 0.1,
            torch.ones_like(ref_labels, dtype=torch.long),   # 1 = ref_pos
            torch.zeros_like(ref_labels, dtype=torch.long),   # 0 = ref_neg
        )
        all_features.append(ref_feat)
        all_labels.append(ref_labels)
        all_sources.append(ref_sources)
        print(f"  {os.path.basename(mf)}: {ref_feat.shape[0]} frames, "
              f"{(ref_labels > 0.1).sum().item()} positive")

    # ── Save ──
    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    sources = torch.cat(all_sources, dim=0)

    os.makedirs(os.path.dirname(args_cli.output), exist_ok=True)
    torch.save({
        "features": features,
        "labels": labels,
        "sources": sources,
        "feature_dim": INPUT_DIM,
    }, args_cli.output)

    n_pos = (labels > 0.3).sum().item()
    n_neg = (labels <= 0.3).sum().item()
    print(f"\n[collect] Saved {features.shape[0]} samples to {args_cli.output}")
    print(f"  Positive (label > 0.3): {n_pos}")
    print(f"  Negative (label <= 0.3): {n_neg}")
    print(f"  Sources: ref_neg={int((sources==0).sum())}, ref_pos={int((sources==1).sum())}, "
          f"rollout_pos={int((sources==2).sum())}, rollout_hard_neg={int((sources==3).sum())}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
