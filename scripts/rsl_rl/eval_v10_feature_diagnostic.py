"""v10.2 Step 0: Combined Feature Diagnostics.

Diagnostic 1: Event Action Sensitivity
  - Freeze env state at various time steps
  - Recompute obs with different event shifts
  - Measure ||a_shift - a_0|| per joint group

Diagnostic 2: Minimal Feature Ablation
  - Run rollouts with zeroed obs components
  - A. zero event_obs (8D)
  - B. zero motor_prior (40D)
  - C. zero action_history (87D within proprio_hist)
  - D. zero all history (261D proprio_hist + 29D last_action)
  - Measure Kick%, ep_len, CΔ_original

Usage:
    python scripts/rsl_rl/eval_v10_feature_diagnostic.py \
        --task Event-Conditioned-Kick-G1-Soccer-v0 \
        --motion_file motions/Video/ \
        --bc_checkpoint logs/rsl_rl/v10_bc/bc_pretrained_v2.pt \
        --num_episodes 50 \
        --headless
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="v10.2 Feature Diagnostics")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--task", type=str, default="Event-Conditioned-Kick-G1-Soccer-v0")
parser.add_argument("--motion_file", type=str, required=True)
parser.add_argument("--bc_checkpoint", type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)

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
import numpy as np
from collections import defaultdict

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import soccer.tasks  # noqa: F401
from soccer.tasks.tracking.mdp.event_conditioned_obs_builder import V10ObsBuilder
from soccer.tasks.tracking.mdp.event_phase import compute_segment_bounds, compute_event_phase, compute_event_obs


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


# ============================================================================
# Obs layout: 454D total
# [0:64]    current_proprio (ang_vel:3, gravity:3, joint_pos:29, joint_vel:29)
# [64:325]  proprio_hist (joint_pos_hist:87 + joint_vel_hist:87 + action_hist:87)
# [325:354] last_action (29)
# [354:384] ball_hist (30)
# [384:392] event_obs (8)
# [392:414] ball_foot_rel (22)
# [414:454] motor_prior (40)
# ============================================================================

OBS_SLICES = {
    "current_proprio": (0, 64),
    "proprio_hist": (64, 325),
    "last_action": (325, 354),
    "ball_hist": (354, 384),
    "event_obs": (384, 392),
    "ball_foot_rel": (392, 414),
    "motor_prior": (414, 454),
    # Sub-slices within proprio_hist
    "joint_pos_hist": (64, 151),   # 87D
    "joint_vel_hist": (151, 238),  # 87D
    "action_hist": (238, 325),     # 87D
}

# Joint groups for action analysis
JOINT_GROUPS = {
    "hip": list(range(0, 6)),     # left/right hip (3 each)
    "knee": [6, 7],                # left/right knee
    "ankle": list(range(8, 12)),   # left/right ankle (2 each)
    "waist": list(range(12, 15)),  # waist joints
    "shoulder": list(range(15, 21)), # shoulders
    "elbow": list(range(21, 25)),  # elbows
    "wrist": list(range(25, 29)),  # wrists
}


def diagnostic_1_sensitivity(env, env_wrapped, unwrapped_env, model,
                              v10_builder, command, device):
    """Diagnostic 1: Event action sensitivity.
    
    Collects states at various time steps, then recomputes obs with
    different event shifts and measures action change.
    """
    print("\n" + "=" * 70)
    print("  DIAGNOSTIC 1: Event Action Sensitivity")
    print("=" * 70)

    shifts = [-20, -10, -5, -2, 0, 2, 5, 10, 20]
    num_envs = unwrapped_env.num_envs

    # Collect states at different phases by running a few episodes
    env_wrapped.reset()
    v10_builder.init_segment_bounds(command)

    # Run until we have enough data
    all_obs = []
    all_time_steps = []
    
    for step in range(500):
        obs_v10 = v10_builder.compute(unwrapped_env, command)
        
        # Save obs snapshots at various time steps
        if step % 20 == 0 and step > 0:
            all_obs.append(obs_v10.clone())
            all_time_steps.append(command.time_steps.clone())

        with torch.no_grad():
            actions = model(obs_v10)
        obs_v3, _, dones, _ = env_wrapped.step(actions)
        v10_builder.update_history(unwrapped_env, command, actions, dones)

    if not all_obs:
        print("  No obs collected. Skipping.")
        return

    # For each collected state, perturb event_obs and measure action change
    print(f"\n  Collected {len(all_obs)} state snapshots")
    print(f"\n  {'Shift':>6} | {'||Δa||':>8} | {'Hip':>8} | {'Knee':>8} | {'Ankle':>8} | {'Max joint':>10}")
    print("  " + "-" * 70)

    shift_action_diffs = defaultdict(list)

    for obs_snapshot in all_obs:
        # Get baseline action (shift=0)
        with torch.no_grad():
            a_base = model(obs_snapshot)

        for shift in shifts:
            # Modify event_obs in the snapshot
            obs_modified = obs_snapshot.clone()
            
            # Recompute event_obs with shifted segment_bounds
            shifted_bounds = v10_builder.segment_bounds.clone()
            shifted_bounds[:, 0] = (shifted_bounds[:, 0] + shift).clamp(min=1)
            shifted_bounds[:, 1] = (shifted_bounds[:, 1] + shift).clamp(min=shifted_bounds[:, 0] + 1)
            shifted_bounds[:, 2] = (shifted_bounds[:, 2] + shift).clamp(min=shifted_bounds[:, 1] + 1)
            shifted_bounds[:, 2] = shifted_bounds[:, 2].clamp(max=shifted_bounds[:, 3] - 1)
            
            phase_id_s, phase_phi_s = compute_event_phase(
                command.time_steps, shifted_bounds
            )
            event_obs_s = compute_event_obs(
                phase_id_s, phase_phi_s,
                command.time_steps, shifted_bounds
            )
            
            # Replace event_obs slice
            s, e = OBS_SLICES["event_obs"]
            obs_modified[:, s:e] = event_obs_s
            
            # Also update motor_prior (which depends on event phase)
            # For now, just change event_obs — motor_prior stays from original
            # This isolates the event_obs signal
            
            with torch.no_grad():
                a_shifted = model(obs_modified)
            
            diff = (a_shifted - a_base).abs()
            shift_action_diffs[shift].append(diff.cpu())

    # Aggregate results
    for shift in shifts:
        all_diffs = torch.cat(shift_action_diffs[shift], dim=0)  # [total_samples, 29]
        mean_diff = all_diffs.mean(dim=0)  # [29]
        total_norm = mean_diff.norm().item()
        
        hip_diff = mean_diff[JOINT_GROUPS["hip"]].norm().item()
        knee_diff = mean_diff[JOINT_GROUPS["knee"]].norm().item()
        ankle_diff = mean_diff[JOINT_GROUPS["ankle"]].norm().item()
        max_joint = mean_diff.max().item()
        max_idx = mean_diff.argmax().item()
        
        print(f"  {shift:>+6d} | {total_norm:>8.4f} | {hip_diff:>8.4f} | "
              f"{knee_diff:>8.4f} | {ankle_diff:>8.4f} | {max_joint:>8.4f} (j{max_idx})")

    # Interpretation
    baseline_norms = [torch.cat(shift_action_diffs[s], dim=0).mean(dim=0).norm().item() 
                      for s in shifts if s != 0]
    avg_sensitivity = np.mean(baseline_norms)
    
    print(f"\n  Average action sensitivity to event shift: {avg_sensitivity:.4f}")
    if avg_sensitivity < 0.01:
        print("  ❌ VERY LOW: Policy essentially ignores event_obs.")
    elif avg_sensitivity < 0.05:
        print("  ⚠️ WEAK: Policy has slight sensitivity to event_obs.")
    else:
        print("  ✅ SIGNIFICANT: Policy uses event_obs for action selection.")


def diagnostic_2_ablation(env, env_wrapped, unwrapped_env, model,
                           v10_builder, command, device, num_episodes):
    """Diagnostic 2: Minimal feature ablation rollout.
    
    Runs rollouts with zeroed obs components and measures impact.
    """
    print("\n" + "=" * 70)
    print("  DIAGNOSTIC 2: Feature Ablation Rollout")
    print("=" * 70)

    ablation_configs = {
        "baseline": {},   # no zeroing
        "zero_event_obs": {"event_obs"},
        "zero_motor_prior": {"motor_prior"},
        "zero_action_hist": {"action_hist"},
        "zero_all_history": {"proprio_hist", "last_action"},
    }

    num_envs = unwrapped_env.num_envs
    BALL_SPEED_THRESH = 0.5

    results = {}

    for abl_name, zero_features in ablation_configs.items():
        print(f"\n  --- Ablation: {abl_name} ---")
        
        # Fresh builder
        builder = V10ObsBuilder(
            num_envs=num_envs,
            num_joints=command.robot.data.joint_pos.shape[1],
            device=device,
        )
        
        env_wrapped.reset()
        builder.init_segment_bounds(command)

        episodes_completed = 0
        kick_successes = 0
        falls = 0
        ep_lengths = []
        contact_deltas = []  # CΔ_original = contact_frame - original_kick_frame

        ep_length_counter = torch.zeros(num_envs, device=device)
        ball_contacted = torch.zeros(num_envs, dtype=torch.bool, device=device)
        contact_step = torch.full((num_envs,), -1.0, device=device)
        
        env_orig_kf = torch.zeros(num_envs, device=device)
        for i in range(num_envs):
            mid = command.motion_idx[i].item()
            env_orig_kf[i] = command.motion.kick_frames[mid].item()

        step_count = 0
        max_steps = num_episodes * 600

        while episodes_completed < num_episodes and step_count < max_steps:
            obs_v10 = builder.compute(unwrapped_env, command)

            # Apply ablation: zero out specified features
            for feat in zero_features:
                s, e = OBS_SLICES[feat]
                obs_v10[:, s:e] = 0.0

            with torch.no_grad():
                actions = model(obs_v10)

            obs_v3, _, dones, _ = env_wrapped.step(actions)
            ep_length_counter += 1
            step_count += 1

            builder.update_history(unwrapped_env, command, actions, dones)

            # Detect ball contact
            soccer_ball = unwrapped_env.scene["soccer_ball"]
            ball_vel = soccer_ball.data.root_lin_vel_w[:, :3]
            ball_speed = torch.norm(ball_vel[:, :2], dim=-1)
            new_contact = (~ball_contacted) & (ball_speed > BALL_SPEED_THRESH)
            if new_contact.any():
                ball_contacted[new_contact] = True
                contact_step[new_contact] = command.time_steps[new_contact].float()

            if dones.any():
                done_ids = dones.nonzero(as_tuple=True)[0]
                for idx in done_ids:
                    if episodes_completed >= num_episodes:
                        break
                    i = idx.item()
                    episodes_completed += 1
                    ep_lengths.append(ep_length_counter[i].item())

                    if ball_contacted[i]:
                        kick_successes += 1
                        cd_orig = contact_step[i].item() - env_orig_kf[i].item()
                        contact_deltas.append(cd_orig)

                    if ep_length_counter[i] < 100:
                        falls += 1

                    ep_length_counter[i] = 0
                    ball_contacted[i] = False
                    contact_step[i] = -1.0
                    mid = command.motion_idx[i].item()
                    env_orig_kf[i] = command.motion.kick_frames[mid].item()

        kick_pct = kick_successes / max(episodes_completed, 1) * 100
        fall_pct = falls / max(episodes_completed, 1) * 100
        avg_ep_len = float(np.mean(ep_lengths)) if ep_lengths else 0
        avg_cd = float(np.mean(contact_deltas)) if contact_deltas else float('nan')
        std_cd = float(np.std(contact_deltas)) if contact_deltas else float('nan')

        results[abl_name] = {
            "kick_pct": kick_pct,
            "fall_pct": fall_pct,
            "avg_ep_len": avg_ep_len,
            "avg_cd_original": avg_cd,
            "std_cd_original": std_cd,
        }

        cd_str = f"{avg_cd:+.1f}±{std_cd:.1f}" if not np.isnan(avg_cd) else "N/A"
        print(f"    Kick%={kick_pct:.1f}%  Fall%={fall_pct:.1f}%  "
              f"EpLen={avg_ep_len:.1f}  CΔ_orig={cd_str}")

    # Summary table
    print(f"\n\n{'='*80}")
    print("Feature Ablation Summary")
    print(f"{'='*80}")
    print(f"  {'Ablation':<20} | {'Kick%':>6} | {'Fall%':>6} | {'EpLen':>7} | {'CΔ_original':>12}")
    print("  " + "-" * 70)
    for name, r in results.items():
        cd_str = f"{r['avg_cd_original']:+.1f}±{r['std_cd_original']:.1f}" if not np.isnan(r['avg_cd_original']) else "N/A"
        print(f"  {name:<20} | {r['kick_pct']:>5.1f}% | {r['fall_pct']:>5.1f}% | "
              f"{r['avg_ep_len']:>7.1f} | {cd_str:>12}")
    print(f"{'='*80}")

    # Interpretation
    bl = results.get("baseline", {})
    for name, r in results.items():
        if name == "baseline":
            continue
        kick_drop = bl.get("kick_pct", 0) - r["kick_pct"]
        eplen_drop = bl.get("avg_ep_len", 0) - r["avg_ep_len"]
        if kick_drop > 30 or eplen_drop > 100:
            severity = "CRITICAL"
        elif kick_drop > 10 or eplen_drop > 50:
            severity = "MODERATE"
        else:
            severity = "MINIMAL"
        print(f"  {name}: {severity} impact (Kick% Δ={kick_drop:+.1f}%, EpLen Δ={eplen_drop:+.1f})")

    return results


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
    env_cfg.commands.motion.motion_files = get_motion_files(args_cli.motion_file)
    device = env_cfg.sim.device

    # Load BC model
    print(f"[INFO] Loading BC checkpoint from: {args_cli.bc_checkpoint}")
    ckpt = torch.load(args_cli.bc_checkpoint, map_location=device, weights_only=False)
    model = V10MLPActor(
        obs_dim=ckpt["obs_dim"],
        action_dim=ckpt["action_dim"],
        hidden_dims=ckpt.get("hidden_dims", [512, 256, 128]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Create env
    env = gym.make(args_cli.task, cfg=env_cfg)
    env_wrapped = RslRlVecEnvWrapper(env)
    unwrapped_env = env.unwrapped
    command = unwrapped_env.command_manager.get_term("motion")

    # V10 obs builder
    v10_builder = V10ObsBuilder(
        num_envs=args_cli.num_envs,
        num_joints=command.robot.data.joint_pos.shape[1],
        device=device,
    )

    # Run diagnostics
    diagnostic_1_sensitivity(
        env, env_wrapped, unwrapped_env, model,
        v10_builder, command, device
    )

    diagnostic_2_ablation(
        env, env_wrapped, unwrapped_env, model,
        v10_builder, command, device, args_cli.num_episodes
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
