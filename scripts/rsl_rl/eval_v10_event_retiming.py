"""v10.2 Diagnostic: Event Retiming Test.

Tests whether the BC policy learned event-conditioned timing or memorized
fixed-timing trajectories from the V3 teacher.

Shifts kick_frame by Δ ∈ {-20, -10, -5, -2, +2, +5, +10, +20} frames when
computing segment_bounds, then measures contact_frame relative to both
retimed and original kick timing.

Success: CΔ_event ≈ 0  (policy follows retimed event)
Failure: CΔ_original ≈ 0 (policy ignores retiming, fixed trajectory)

Usage:
    python scripts/rsl_rl/eval_v10_event_retiming.py \
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

parser = argparse.ArgumentParser(description="v10.2 Event Retiming Diagnostic")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--task", type=str, default="Event-Conditioned-Kick-G1-Soccer-v0")
parser.add_argument("--motion_file", type=str, required=True)
parser.add_argument("--bc_checkpoint", type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--shifts", type=str, default="-20,-10,-5,-2,0,2,5,10,20",
    help="Comma-separated list of event shifts in frames"
)

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

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import soccer.tasks  # noqa: F401
from soccer.tasks.tracking.mdp.event_conditioned_obs_builder import V10ObsBuilder
from soccer.tasks.tracking.mdp.event_phase import compute_segment_bounds


class V10MLPActor(nn.Module):
    """v10 MLP actor (must match train_v10_bc.py)."""
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


class RetimingV10ObsBuilder(V10ObsBuilder):
    """V10ObsBuilder that shifts kick_frame before computing segment bounds."""

    def __init__(self, event_shift: int, **kwargs):
        super().__init__(**kwargs)
        self.event_shift = event_shift

    def init_segment_bounds(self, command):
        """Compute segment bounds with shifted kick_frame."""
        for i in range(self.num_envs):
            mid = command.motion_idx[i].item()
            kf = command.motion.kick_frames[mid].item()
            kef = command.motion.kick_end_frames[mid].item()
            ml = command.motion_length[i].item()

            if kf < 0:
                kf = ml
                kef = ml

            # Apply event shift
            shifted_kf = max(1, min(int(kf + self.event_shift), ml - 2))
            shifted_kef = max(shifted_kf + 1, min(int(kef + self.event_shift), ml - 1))

            bounds = compute_segment_bounds(
                shifted_kf, shifted_kef, ml,
                prestrike_duration=self.PRESTRIKE_DURATION,
                min_strike_duration=self.MIN_STRIKE_DURATION,
            )
            self.segment_bounds[i, 0] = bounds.approach_end
            self.segment_bounds[i, 1] = bounds.prestrike_end
            self.segment_bounds[i, 2] = bounds.strike_end
            self.segment_bounds[i, 3] = bounds.motion_length


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


def run_retiming_eval(env_wrapped, unwrapped_env, model, v10_builder, command,
                      event_shift, num_episodes, device):
    """Run rollout with a specific event shift and return metrics."""
    num_envs = unwrapped_env.num_envs
    BALL_SPEED_THRESH = 0.5  # m/s — same as eval_v10_bc_rollout.py

    # Metrics accumulators
    episodes_completed = 0
    contact_frames = []      # Frame when ball was first contacted
    original_kick_frames = []  # Original kick_frame from motion
    retimed_kick_frames = []   # Shifted kick_frame
    ep_lengths = []
    kick_successes = 0
    falls = 0

    ep_length_counter = torch.zeros(num_envs, device=device)
    # Ball contact detection via speed (simple, reliable)
    ball_contacted = torch.zeros(num_envs, dtype=torch.bool, device=device)
    contact_step = torch.full((num_envs,), -1.0, device=device)  # time_step at first contact
    # Save original kick_frame per env at episode start
    env_orig_kf = torch.zeros(num_envs, device=device)
    for i in range(num_envs):
        mid = command.motion_idx[i].item()
        env_orig_kf[i] = command.motion.kick_frames[mid].item()

    step_count = 0
    max_steps = num_episodes * 600  # safety limit

    while episodes_completed < num_episodes and step_count < max_steps:
        # 1. Compute v10 obs
        obs_v10 = v10_builder.compute(unwrapped_env, command)

        # 2. MLP action (deterministic)
        with torch.no_grad():
            actions = model(obs_v10)

        # 3. Step env
        obs_v3, _, dones, infos = env_wrapped.step(actions)
        ep_length_counter += 1
        step_count += 1

        # 4. Update v10 history
        v10_builder.update_history(unwrapped_env, command, actions, dones)

        # 5. Detect ball contact via speed
        soccer_ball = unwrapped_env.scene["soccer_ball"]
        ball_vel = soccer_ball.data.root_lin_vel_w[:, :3]
        ball_speed = torch.norm(ball_vel[:, :2], dim=-1)
        new_contact = (~ball_contacted) & (ball_speed > BALL_SPEED_THRESH)
        if new_contact.any():
            ball_contacted[new_contact] = True
            contact_step[new_contact] = command.time_steps[new_contact].float()

        # 6. Handle episode ends
        if dones.any():
            done_ids = dones.nonzero(as_tuple=True)[0]
            for idx in done_ids:
                if episodes_completed >= num_episodes:
                    break
                i = idx.item()
                episodes_completed += 1
                ep_lengths.append(ep_length_counter[i].item())

                orig_kf = env_orig_kf[i].item()
                retimed_kf = orig_kf + event_shift

                if ball_contacted[i]:  # Contact happened
                    kick_successes += 1
                    contact_frames.append(contact_step[i].item())
                    original_kick_frames.append(orig_kf)
                    retimed_kick_frames.append(retimed_kf)
                else:
                    contact_frames.append(float('nan'))
                    original_kick_frames.append(orig_kf)
                    retimed_kick_frames.append(retimed_kf)

                if ep_length_counter[i] < 100:
                    falls += 1

                # Reset per-env state and update orig_kf for next episode
                ep_length_counter[i] = 0
                ball_contacted[i] = False
                contact_step[i] = -1.0
                mid = command.motion_idx[i].item()
                env_orig_kf[i] = command.motion.kick_frames[mid].item()

    # Compute CΔ metrics
    cf_arr = np.array(contact_frames)
    okf_arr = np.array(original_kick_frames)
    rkf_arr = np.array(retimed_kick_frames)

    valid = ~np.isnan(cf_arr)
    if valid.sum() > 0:
        cd_event = cf_arr[valid] - rkf_arr[valid]     # should be ≈ 0 if event-conditioned
        cd_original = cf_arr[valid] - okf_arr[valid]   # should be ≈ shift if event-conditioned
        mean_cd_event = float(np.mean(cd_event))
        std_cd_event = float(np.std(cd_event))
        mean_cd_original = float(np.mean(cd_original))
        std_cd_original = float(np.std(cd_original))
    else:
        mean_cd_event = float('nan')
        std_cd_event = float('nan')
        mean_cd_original = float('nan')
        std_cd_original = float('nan')

    return {
        "shift": event_shift,
        "episodes": episodes_completed,
        "kick_pct": kick_successes / max(episodes_completed, 1) * 100,
        "fall_pct": falls / max(episodes_completed, 1) * 100,
        "avg_ep_len": float(np.mean(ep_lengths)) if ep_lengths else 0,
        "mean_cd_event": mean_cd_event,
        "std_cd_event": std_cd_event,
        "mean_cd_original": mean_cd_original,
        "std_cd_original": std_cd_original,
    }


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
    print(f"[INFO] BC model: obs_dim={ckpt['obs_dim']}, "
          f"action_dim={ckpt['action_dim']}, "
          f"params={sum(p.numel() for p in model.parameters())}")

    # Parse shifts
    shifts = [int(s.strip()) for s in args_cli.shifts.split(",")]
    print(f"[INFO] Testing event shifts: {shifts}")

    # Create env
    env = gym.make(args_cli.task, cfg=env_cfg)
    env_wrapped = RslRlVecEnvWrapper(env)
    unwrapped_env = env.unwrapped
    command = unwrapped_env.command_manager.get_term("motion")
    num_joints = command.robot.data.joint_pos.shape[1]

    # Run for each shift
    results = []
    for shift in shifts:
        print(f"\n{'='*60}")
        print(f"  Event shift = {shift:+d} frames")
        print(f"{'='*60}")

        # Create builder with this shift
        v10_builder = RetimingV10ObsBuilder(
            event_shift=shift,
            num_envs=args_cli.num_envs,
            num_joints=num_joints,
            device=device,
        )
        v10_builder.init_segment_bounds(command)

        # Reset env for clean start
        env_wrapped.reset()

        result = run_retiming_eval(
            env_wrapped, unwrapped_env, model, v10_builder, command,
            shift, args_cli.num_episodes, device,
        )
        results.append(result)

        print(f"  Kick%: {result['kick_pct']:.1f}%  "
              f"Fall%: {result['fall_pct']:.1f}%  "
              f"EpLen: {result['avg_ep_len']:.1f}")
        print(f"  CΔ_event:    {result['mean_cd_event']:+.1f} ± {result['std_cd_event']:.1f}")
        print(f"  CΔ_original: {result['mean_cd_original']:+.1f} ± {result['std_cd_original']:.1f}")

    # Summary table
    print(f"\n\n{'='*80}")
    print("v10.2 Event Retiming Diagnostic Results")
    print(f"{'='*80}")
    print(f"{'Shift':>6} | {'Kick%':>6} | {'Fall%':>6} | {'EpLen':>7} | "
          f"{'CΔ_event':>12} | {'CΔ_original':>12}")
    print("-" * 80)
    for r in results:
        cd_e = f"{r['mean_cd_event']:+.1f}±{r['std_cd_event']:.1f}" if not np.isnan(r['mean_cd_event']) else "N/A"
        cd_o = f"{r['mean_cd_original']:+.1f}±{r['std_cd_original']:.1f}" if not np.isnan(r['mean_cd_original']) else "N/A"
        print(f"{r['shift']:>+6d} | {r['kick_pct']:>5.1f}% | {r['fall_pct']:>5.1f}% | "
              f"{r['avg_ep_len']:>7.1f} | {cd_e:>12} | {cd_o:>12}")
    print(f"{'='*80}")

    # Interpretation
    if results:
        valid_results = [r for r in results if not np.isnan(r['mean_cd_event']) and r['shift'] != 0]
        if valid_results:
            avg_abs_cd_event = np.mean([abs(r['mean_cd_event']) for r in valid_results])
            avg_abs_cd_original = np.mean([abs(r['mean_cd_original']) for r in valid_results])

            # Check correlation: does CΔ_original track the shift?
            shifts_arr = np.array([r['shift'] for r in valid_results])
            cd_orig_arr = np.array([r['mean_cd_original'] for r in valid_results])

            if len(shifts_arr) > 1:
                correlation = np.corrcoef(shifts_arr, cd_orig_arr)[0, 1]
            else:
                correlation = 0.0

            print(f"\nDiagnostic Summary:")
            print(f"  Avg |CΔ_event|:    {avg_abs_cd_event:.1f} frames")
            print(f"  Avg |CΔ_original|: {avg_abs_cd_original:.1f} frames")
            print(f"  Correlation(shift, CΔ_original): {correlation:.3f}")

            if correlation > 0.7 and avg_abs_cd_event < 10:
                print(f"\n  ✅ EVENT-CONDITIONED: Policy follows retimed events.")
                print(f"     CΔ_event stays small, CΔ_original tracks shift.")
            elif abs(correlation) < 0.3:
                print(f"\n  ❌ FIXED TIMING: Policy ignores retiming.")
                print(f"     CΔ_original does NOT track shift — v3-style fixed trajectory.")
            else:
                print(f"\n  ⚠️ MIXED: Partial event conditioning detected.")
                print(f"     Some retiming sensitivity but not strong.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
