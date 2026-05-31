"""Script to evaluate a trained policy quantitatively across all motions."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during eval.")
parser.add_argument("--video_length", type=int, default=600, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to a single motion file.")
parser.add_argument("--motion_path", type=str, default=None, help="Directory containing motion files.")
parser.add_argument("--eval_episodes", type=int, default=50, help="Number of episodes per motion for evaluation.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os
import glob
import torch
import numpy as np
from collections import defaultdict

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import soccer.tasks  # noqa: F401


def get_motion_files(motion_path: str) -> list[str]:
    """Get a list of motion files from a file or directory."""
    if os.path.isfile(motion_path):
        return [motion_path]
    elif os.path.isdir(motion_path):
        motion_files = sorted(glob.glob(os.path.join(motion_path, "*.npz")))
        if not motion_files:
            raise ValueError(f"No .npz files found in directory: {motion_path}")
        return motion_files
    else:
        raise ValueError(f"Invalid path: {motion_path}")


class PolicyEvaluator:
    """Collects per-episode metrics across multiple environments and motions."""

    def __init__(self, env, num_motions: int, motion_names: list[str], target_episodes: int, device: str):
        self.env = env
        self.base_env = env.unwrapped if hasattr(env, 'unwrapped') else env
        self.num_envs = env.num_envs
        self.num_motions = num_motions
        self.motion_names = motion_names
        self.target_episodes = target_episodes
        self.device = device

        # Per-env tracking
        self.env_motion_idx = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.env_step_count = torch.zeros(self.num_envs, dtype=torch.long, device=device)

        # Per-env accumulators (reset each episode)
        self.ep_body_pos_err = torch.zeros(self.num_envs, device=device)
        self.ep_body_ori_err = torch.zeros(self.num_envs, device=device)
        self.ep_foot_pos_err = torch.zeros(self.num_envs, device=device)
        self.ep_action_rate = torch.zeros(self.num_envs, device=device)
        self.ep_joint_limit = torch.zeros(self.num_envs, device=device)
        self.ep_early_collision_count = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.ep_peak_ball_speed = torch.zeros(self.num_envs, device=device)

        # CG precision tracking
        self.ep_cg_match_frames = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.ep_cg_annotated_frames = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.ep_cg_ref1_actual1 = torch.zeros(self.num_envs, dtype=torch.long, device=device)  # true positive
        self.ep_cg_ref1_actual0 = torch.zeros(self.num_envs, dtype=torch.long, device=device)  # false negative
        self.ep_cg_ref0_actual1 = torch.zeros(self.num_envs, dtype=torch.long, device=device)  # false positive
        self.ep_actual_contact_frame_first = torch.full((self.num_envs,), -1, dtype=torch.long, device=device)
        self.ep_ref_kick_start_frame = torch.full((self.num_envs,), -1, dtype=torch.long, device=device)

        # Episode results storage: dict[motion_idx] -> list of episode dicts
        self.results: dict[int, list[dict]] = defaultdict(list)

        # Completed episodes count per motion
        self.episodes_done = torch.zeros(num_motions, dtype=torch.long)

    def assign_motions_round_robin(self):
        """Assign motions to envs in round-robin for even coverage."""
        for i in range(self.num_envs):
            self.env_motion_idx[i] = i % self.num_motions

        # Force the command manager to use our assigned motions
        cmd = self.base_env.command_manager.get_term("motion")
        cmd.motion_idx[:] = self.env_motion_idx
        cmd.motion_length[:] = cmd.motion.file_lengths[self.env_motion_idx]
        cmd.time_steps[:] = 0

    def step(self, rewards, dones, infos):
        """Called after each env.step(). Accumulate metrics and record completed episodes."""
        self.env_step_count += 1

        # --- Accumulate per-step metrics ---
        cmd = self.base_env.command_manager.get_term("motion")

        # Body position tracking error
        if hasattr(cmd, 'body_pos_error'):
            self.ep_body_pos_err += cmd.body_pos_error.mean(dim=-1) if cmd.body_pos_error.ndim > 1 else cmd.body_pos_error

        # Action rate
        action_diff = self.base_env.action_manager.action - self.base_env.action_manager.prev_action
        self.ep_action_rate += torch.sum(torch.square(action_diff), dim=1)

        # Ball speed
        try:
            soccer_ball = self.base_env.scene["soccer_ball"]
            ball_vel_xy = soccer_ball.data.root_lin_vel_w[:, :2]
            ball_speed = torch.norm(ball_vel_xy, dim=-1)
            self.ep_peak_ball_speed = torch.maximum(self.ep_peak_ball_speed, ball_speed)
        except (KeyError, RuntimeError):
            pass

        # CG comparison: Reference CG vs Actual CG (contact sensor)
        try:
            from soccer.tasks.tracking.mdp.rewards import _get_cg_phase
            is_cg0, is_cg1 = _get_cg_phase(cmd, margin=5)

            ball_contact = self.base_env.scene["soccer_ball_contact"]
            net_forces = ball_contact.data.net_forces_w_history
            if net_forces.dim() == 4:
                force_vec = net_forces[:, :, 0, :2].sum(dim=1)
            else:
                force_vec = net_forces[:, 0, :2]
            force_mag = torch.norm(force_vec, dim=-1)
            has_contact = force_mag > 5.0

            # Early collision (CG=0 contact)
            self.ep_early_collision_count += (is_cg0 & has_contact).long()

            # CG precision tracking
            has_annotation = cmd.kick_frame >= 0
            ref_cg = is_cg1  # True = CG=1 (kick window)
            actual_cg = has_contact

            # Only count annotated envs
            self.ep_cg_annotated_frames += has_annotation.long()
            self.ep_cg_match_frames += (has_annotation & (ref_cg == actual_cg)).long()
            self.ep_cg_ref1_actual1 += (has_annotation & ref_cg & actual_cg).long()
            self.ep_cg_ref1_actual0 += (has_annotation & ref_cg & ~actual_cg).long()
            self.ep_cg_ref0_actual1 += (has_annotation & ~ref_cg & actual_cg).long()

            # Track first actual contact frame
            new_first_contact = has_contact & (self.ep_actual_contact_frame_first < 0)
            self.ep_actual_contact_frame_first[new_first_contact] = cmd.time_steps[new_first_contact]

            # Track ref kick start frame
            no_ref_yet = self.ep_ref_kick_start_frame < 0
            self.ep_ref_kick_start_frame[no_ref_yet & (cmd.kick_frame >= 0)] = cmd.kick_frame[no_ref_yet & (cmd.kick_frame >= 0)]
        except Exception:
            pass

        # --- Handle completed episodes (resets) ---
        if isinstance(dones, dict):
            done_mask = dones.get("terminated", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
            timeout_mask = dones.get("truncated", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
            reset_mask = done_mask | timeout_mask
        else:
            reset_mask = dones.bool() if not isinstance(dones, bool) else torch.full((self.num_envs,), dones, dtype=torch.bool, device=self.device)
            if isinstance(infos, dict) and "time_outs" in infos:
                timeout_mask = infos["time_outs"].to(device=self.device, dtype=torch.bool)
                done_mask = reset_mask & ~timeout_mask
            else:
                timeout_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                done_mask = reset_mask

        if torch.any(reset_mask):
            self._record_episodes(reset_mask, done_mask, timeout_mask, cmd)

    def _record_episodes(self, reset_mask, terminated_mask, truncated_mask, cmd):
        """Record metrics for completed episodes."""
        reset_ids = torch.where(reset_mask)[0]

        for idx in reset_ids:
            i = idx.item()
            motion_id = int(self.env_motion_idx[i].item())

            if self.episodes_done[motion_id] >= self.target_episodes:
                continue

            steps = int(self.env_step_count[i].item())
            if steps < 2:
                continue

            # Kick success: check if ball reached meaningful speed
            kick_success = bool(self.ep_peak_ball_speed[i].item() > 1.0)

            # Check if kick contact was properly detected
            tracker = cmd.kick_contact_tracker
            contact_awarded = tracker.get_contact_awarded()
            kicked = bool(contact_awarded[i].item())

            # CG precision metrics
            ann_frames = max(int(self.ep_cg_annotated_frames[i].item()), 1)
            cg_accuracy = round(self.ep_cg_match_frames[i].item() / ann_frames, 3)
            cg_tp = int(self.ep_cg_ref1_actual1[i].item())
            cg_fn = int(self.ep_cg_ref1_actual0[i].item())
            cg_fp = int(self.ep_cg_ref0_actual1[i].item())
            first_contact = int(self.ep_actual_contact_frame_first[i].item())
            ref_kf = int(self.ep_ref_kick_start_frame[i].item())
            contact_timing_delta = (first_contact - ref_kf) if (first_contact >= 0 and ref_kf >= 0) else None

            episode_data = {
                "motion_name": self.motion_names[motion_id],
                "motion_idx": motion_id,
                "episode_length": steps,
                "terminated": bool(terminated_mask[i].item()),
                "truncated": bool(truncated_mask[i].item()),
                "kick_success": kick_success,
                "contact_awarded": kicked,
                "peak_ball_speed": round(self.ep_peak_ball_speed[i].item(), 3),
                "early_collision_frames": int(self.ep_early_collision_count[i].item()),
                "action_smoothness": round(self.ep_action_rate[i].item() / max(steps, 1), 4),
                # CG precision
                "cg_accuracy": cg_accuracy,
                "cg_true_positive": cg_tp,
                "cg_false_negative": cg_fn,
                "cg_false_positive": cg_fp,
                "contact_timing_delta": contact_timing_delta,
            }

            self.results[motion_id].append(episode_data)
            self.episodes_done[motion_id] += 1

            # Reset accumulators for this env
            self.env_step_count[i] = 0
            self.ep_body_pos_err[i] = 0
            self.ep_body_ori_err[i] = 0
            self.ep_foot_pos_err[i] = 0
            self.ep_action_rate[i] = 0
            self.ep_joint_limit[i] = 0
            self.ep_early_collision_count[i] = 0
            self.ep_peak_ball_speed[i] = 0
            self.ep_cg_match_frames[i] = 0
            self.ep_cg_annotated_frames[i] = 0
            self.ep_cg_ref1_actual1[i] = 0
            self.ep_cg_ref1_actual0[i] = 0
            self.ep_cg_ref0_actual1[i] = 0
            self.ep_actual_contact_frame_first[i] = -1
            self.ep_ref_kick_start_frame[i] = -1

            # Reassign to a motion that needs more episodes
            for m in range(self.num_motions):
                if self.episodes_done[m] < self.target_episodes:
                    self.env_motion_idx[i] = m
                    cmd.motion_idx[i] = m
                    cmd.motion_length[i] = cmd.motion.file_lengths[m]
                    break

    def is_done(self) -> bool:
        """Check if all motions have enough episodes."""
        return all(self.episodes_done[m] >= self.target_episodes for m in range(self.num_motions))

    def aggregate(self) -> dict:
        """Compute aggregate statistics per motion and overall."""
        summary = {"per_motion": {}, "aggregate": {}}

        all_episodes = []
        for motion_id in range(self.num_motions):
            episodes = self.results[motion_id]
            if not episodes:
                continue

            name = self.motion_names[motion_id]
            n = len(episodes)

            kick_successes = sum(1 for e in episodes if e["kick_success"])
            contacts = sum(1 for e in episodes if e["contact_awarded"])
            falls = sum(1 for e in episodes if e["terminated"])
            early_collisions = sum(1 for e in episodes if e["early_collision_frames"] > 0)
            peak_speeds = [e["peak_ball_speed"] for e in episodes]
            ep_lengths = [e["episode_length"] for e in episodes]
            smoothness = [e["action_smoothness"] for e in episodes]

            cg_accuracies = [e["cg_accuracy"] for e in episodes]
            timing_deltas = [e["contact_timing_delta"] for e in episodes if e["contact_timing_delta"] is not None]

            motion_stats = {
                "motion_name": name,
                "num_episodes": n,
                "kick_success_rate": round(kick_successes / n, 3),
                "contact_rate": round(contacts / n, 3),
                "fall_rate": round(falls / n, 3),
                "early_collision_rate": round(early_collisions / n, 3),
                "ball_speed_mean": round(float(np.mean(peak_speeds)), 3),
                "ball_speed_max": round(float(np.max(peak_speeds)), 3),
                "episode_length_mean": round(float(np.mean(ep_lengths)), 1),
                "action_smoothness_mean": round(float(np.mean(smoothness)), 4),
                # CG precision
                "cg_accuracy_mean": round(float(np.mean(cg_accuracies)), 3),
                "contact_timing_delta_mean": round(float(np.mean(timing_deltas)), 1) if timing_deltas else None,
                "contact_timing_delta_std": round(float(np.std(timing_deltas)), 1) if timing_deltas else None,
            }
            summary["per_motion"][name] = motion_stats
            all_episodes.extend(episodes)

        # Aggregate across all motions
        if all_episodes:
            n = len(all_episodes)
            all_timing = [e["contact_timing_delta"] for e in all_episodes if e["contact_timing_delta"] is not None]
            summary["aggregate"] = {
                "total_episodes": n,
                "kick_success_rate": round(sum(1 for e in all_episodes if e["kick_success"]) / n, 3),
                "contact_rate": round(sum(1 for e in all_episodes if e["contact_awarded"]) / n, 3),
                "fall_rate": round(sum(1 for e in all_episodes if e["terminated"]) / n, 3),
                "early_collision_rate": round(sum(1 for e in all_episodes if e["early_collision_frames"] > 0) / n, 3),
                "ball_speed_mean": round(float(np.mean([e["peak_ball_speed"] for e in all_episodes])), 3),
                "ball_speed_max": round(float(np.max([e["peak_ball_speed"] for e in all_episodes])), 3),
                "episode_length_mean": round(float(np.mean([e["episode_length"] for e in all_episodes])), 1),
                "action_smoothness_mean": round(float(np.mean([e["action_smoothness"] for e in all_episodes])), 4),
                "cg_accuracy_mean": round(float(np.mean([e["cg_accuracy"] for e in all_episodes])), 3),
                "contact_timing_delta_mean": round(float(np.mean(all_timing)), 1) if all_timing else None,
                "contact_timing_delta_std": round(float(np.std(all_timing)), 1) if all_timing else None,
            }

        return summary

    def print_report(self, summary: dict):
        """Print a formatted evaluation report."""
        print("\n" + "=" * 100)
        print("  POLICY EVALUATION REPORT")
        print("=" * 100)

        # Per-motion table
        if summary["per_motion"]:
            header = f"{'Motion':<40} {'Kick%':>6} {'Fall%':>6} {'EarlyCol%':>10} {'BallSpd':>8} {'CG Acc%':>8} {'Timing':>8} {'Smooth':>8}"
            print(f"\n{header}")
            print("-" * len(header))

            for name, stats in summary["per_motion"].items():
                timing_str = f"{stats['contact_timing_delta_mean']:>+5.0f}f" if stats.get('contact_timing_delta_mean') is not None else "  N/A"
                print(
                    f"{name:<40} "
                    f"{stats['kick_success_rate']*100:>5.1f}% "
                    f"{stats['fall_rate']*100:>5.1f}% "
                    f"{stats['early_collision_rate']*100:>9.1f}% "
                    f"{stats['ball_speed_mean']:>7.2f}m "
                    f"{stats['cg_accuracy_mean']*100:>7.1f}% "
                    f"{timing_str:>7} "
                    f"{stats['action_smoothness_mean']:>7.4f}"
                )

        # Aggregate
        if summary["aggregate"]:
            agg = summary["aggregate"]
            timing_str = f"{agg['contact_timing_delta_mean']:>+5.0f}f" if agg.get('contact_timing_delta_mean') is not None else "  N/A"
            print(f"\n{'AGGREGATE':<40} "
                  f"{agg['kick_success_rate']*100:>5.1f}% "
                  f"{agg['fall_rate']*100:>5.1f}% "
                  f"{agg['early_collision_rate']*100:>9.1f}% "
                  f"{agg['ball_speed_mean']:>7.2f}m "
                  f"{agg['cg_accuracy_mean']*100:>7.1f}% "
                  f"{timing_str:>7} "
                  f"{agg['action_smoothness_mean']:>7.4f}")
            print(f"\nTotal episodes: {agg['total_episodes']}")
            print(f"Best ball speed: {agg['ball_speed_max']:.2f} m/s")
            if agg.get('contact_timing_delta_mean') is not None:
                print(f"Contact timing: {agg['contact_timing_delta_mean']:+.1f} ± {agg['contact_timing_delta_std']:.1f} frames vs ref kick_frame")

        print("=" * 100 + "\n")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Evaluate policy across all motions."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    # Load motion files
    if args_cli.motion_file is not None:
        motion_files = [args_cli.motion_file]
    elif args_cli.motion_path is not None:
        motion_files = get_motion_files(args_cli.motion_path)
    else:
        raise ValueError("Either --motion_file or --motion_path must be specified.")

    # For state-machine environments: auto-split approach/strike files.
    approach_files = [f for f in motion_files if f.endswith("_approach.npz")]
    strike_files = [f for f in motion_files if f.endswith("_strike.npz")]

    if approach_files and strike_files:
        env_cfg.commands.motion.motion_files = approach_files
        if hasattr(env_cfg.commands.motion, "strike_motion_files"):
            env_cfg.commands.motion.strike_motion_files = strike_files
        else:
            env_cfg.commands.motion.motion_files = motion_files
    else:
        env_cfg.commands.motion.motion_files = motion_files
        if hasattr(env_cfg.commands.motion, "strike_motion_files"):
            env_cfg.commands.motion.strike_motion_files = motion_files

    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    import gymnasium as gym
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # Get motion names from the command manager
    base_env = env.unwrapped
    cmd = base_env.command_manager.get_term("motion")
    num_motions = cmd.motion.num_files
    motion_names = cmd.motion.motion_name

    print(f"\n[INFO] Evaluating {num_motions} motions, {args_cli.eval_episodes} episodes each")
    print(f"[INFO] Using {args_cli.num_envs} parallel environments")
    for i, name in enumerate(motion_names):
        print(f"  [{i}] {name}")

    # Create evaluator
    evaluator = PolicyEvaluator(
        env=env,
        num_motions=num_motions,
        motion_names=motion_names,
        target_episodes=args_cli.eval_episodes,
        device=base_env.device,
    )
    evaluator.assign_motions_round_robin()

    # Run evaluation loop
    obs, _ = env.get_observations()
    step = 0
    max_steps = args_cli.eval_episodes * 500 * num_motions  # safety limit

    while simulation_app.is_running() and not evaluator.is_done() and step < max_steps:
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, infos = env.step(actions)

        evaluator.step(rewards, dones, infos)
        step += 1

        if step % 500 == 0:
            done_str = ", ".join(f"{motion_names[m]}={evaluator.episodes_done[m]}" for m in range(num_motions))
            print(f"[EVAL] Step {step} | Episodes: {done_str}")

    # Generate report
    summary = evaluator.aggregate()
    evaluator.print_report(summary)

    # Save results
    eval_dir = os.path.join(os.path.dirname(resume_path), "eval")
    os.makedirs(eval_dir, exist_ok=True)

    json_path = os.path.join(eval_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Results saved to: {json_path}")

    # Also save raw episode data
    raw_path = os.path.join(eval_dir, "eval_episodes.json")
    raw_data = {motion_names[k]: v for k, v in evaluator.results.items()}
    with open(raw_path, "w") as f:
        json.dump(raw_data, f, indent=2)
    print(f"[INFO] Raw episodes saved to: {raw_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
