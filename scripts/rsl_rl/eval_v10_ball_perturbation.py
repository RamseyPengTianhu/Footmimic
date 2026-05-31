"""v10.2 Diagnostic: Ball Position Perturbation Test.

Tests whether the BC policy adapts its foot geometry to ball position
changes, or uses fixed trajectories that absorb perturbation through tolerance.

Offsets ball position by δ ∈ {-0.20, -0.15, -0.10, 0, +0.10, +0.15, +0.20} m
in X and Y, then measures swing/support foot geometry at contact time.

Success: Geometry slopes ≠ 0 (policy adapts)
Failure: Geometry slopes ≈ 0 (fixed trajectory tolerance)

Usage:
    python scripts/rsl_rl/eval_v10_ball_perturbation.py \
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

parser = argparse.ArgumentParser(description="v10.2 Ball Perturbation Diagnostic")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--task", type=str, default="Event-Conditioned-Kick-G1-Soccer-v0")
parser.add_argument("--motion_file", type=str, required=True)
parser.add_argument("--bc_checkpoint", type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--offsets", type=str, default="-0.20,-0.15,-0.10,0.0,0.10,0.15,0.20",
    help="Comma-separated list of ball offsets in meters"
)
parser.add_argument(
    "--axis", type=str, default="x", choices=["x", "y"],
    help="Axis along which to perturb ball position"
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
from isaaclab.utils.math import quat_apply, quat_inv

import soccer.tasks  # noqa: F401
from soccer.tasks.tracking.mdp.event_conditioned_obs_builder import V10ObsBuilder


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


def apply_ball_offset(env, command, offset_m, axis):
    """Apply a deterministic ball offset to all envs after reset.
    
    Modifies soccer_ball_pos and writes new ball state to sim.
    """
    soccer_ball = env.scene["soccer_ball"]
    env_origins = env.scene.env_origins
    
    axis_idx = 0 if axis == "x" else 1
    
    # Shift the command's internal ball position
    command.soccer_ball_pos[:, axis_idx] += offset_m
    command.target_point_pos[:, axis_idx] += offset_m
    command.initial_target_point_pos[:, axis_idx] += offset_m
    
    # Write updated ball state to sim
    all_ids = torch.arange(env.num_envs, device=env.device)
    ball_world_pos = command.soccer_ball_pos + env_origins
    ball_state = torch.zeros(env.num_envs, 13, device=env.device)
    ball_state[:, :3] = ball_world_pos
    ball_state[:, 3] = 1.0  # quaternion w
    soccer_ball.write_root_state_to_sim(ball_state, env_ids=all_ids)


def run_perturbation_eval(env_wrapped, unwrapped_env, model, v10_builder, command,
                          offset_m, axis, num_episodes, device):
    """Run rollout with a specific ball offset and record geometry at contact."""
    num_envs = unwrapped_env.num_envs
    tracker = command.kick_contact_tracker

    # Resolve body indices
    robot = command.robot
    body_names = [b for b in robot.body_names]
    swing_name = "right_ankle_roll_link"
    support_name = "left_ankle_roll_link"
    pelvis_name = "pelvis"
    
    swing_idx = body_names.index(swing_name) if swing_name in body_names else 0
    support_idx = body_names.index(support_name) if support_name in body_names else 1
    pelvis_idx = body_names.index(pelvis_name) if pelvis_name in body_names else 0

    # Accumulators
    episodes_completed = 0
    kick_successes = 0
    falls = 0
    ep_lengths = []
    
    # Geometry at contact time
    swing_ball_offsets = []     # [dx, dy, dz] of swing foot relative to ball
    support_ball_offsets = []   # [dx, dy, dz] of support foot relative to ball
    pelvis_yaws = []           # pelvis yaw angle at contact
    dir_alignments = []        # ball velocity direction alignment

    ep_length_counter = torch.zeros(num_envs, device=device)
    
    # Track whether we've recorded geometry for each env
    geometry_recorded = torch.zeros(num_envs, dtype=torch.bool, device=device)
    
    # Per-env geometry buffers
    env_swing_offset = torch.zeros(num_envs, 3, device=device)
    env_support_offset = torch.zeros(num_envs, 3, device=device)
    env_pelvis_yaw = torch.zeros(num_envs, device=device)
    env_dir_alignment = torch.zeros(num_envs, device=device)

    step_count = 0
    max_steps = num_episodes * 600

    while episodes_completed < num_episodes and step_count < max_steps:
        # 1. Compute v10 obs
        obs_v10 = v10_builder.compute(unwrapped_env, command)

        # 2. MLP action
        with torch.no_grad():
            actions = model(obs_v10)

        # 3. Step env
        obs_v3, _, dones, infos = env_wrapped.step(actions)
        ep_length_counter += 1
        step_count += 1

        # 4. Update v10 history
        v10_builder.update_history(unwrapped_env, command, actions, dones)

        # 5. Check for new contacts and record geometry
        contact_awarded = tracker.get_contact_awarded()
        new_records = contact_awarded & ~geometry_recorded
        
        if new_records.any():
            rec_ids = new_records.nonzero(as_tuple=True)[0]
            
            # Ball position
            soccer_ball = unwrapped_env.scene["soccer_ball"]
            ball_pos_w = soccer_ball.data.root_pos_w[:, :3]
            ball_vel_w = soccer_ball.data.root_lin_vel_w[:, :3]
            
            # Foot positions
            swing_pos_w = robot.data.body_pos_w[:, swing_idx]
            support_pos_w = robot.data.body_pos_w[:, support_idx]
            
            # Pelvis orientation
            pelvis_quat_w = robot.data.body_quat_w[:, pelvis_idx]
            
            # Compute foot-ball offsets (in pelvis-local frame)
            pelvis_quat_inv = quat_inv(pelvis_quat_w)
            swing_offset = quat_apply(pelvis_quat_inv[rec_ids], 
                                       swing_pos_w[rec_ids] - ball_pos_w[rec_ids])
            support_offset = quat_apply(pelvis_quat_inv[rec_ids],
                                         support_pos_w[rec_ids] - ball_pos_w[rec_ids])
            
            # Pelvis yaw (from quaternion)
            # Extract yaw from quaternion: atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
            q = pelvis_quat_w[rec_ids]
            yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                              1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
            
            # Direction alignment
            direction = command.target_destination_pos[rec_ids] - command.initial_target_point_pos[rec_ids]
            dir_xy = direction[:, :2]
            dir_norm = torch.norm(dir_xy, dim=-1, keepdim=True).clamp(min=1e-6)
            vel_xy = ball_vel_w[rec_ids, :2]
            vel_norm = torch.norm(vel_xy, dim=-1, keepdim=True).clamp(min=1e-6)
            cos_align = torch.sum((dir_xy / dir_norm) * (vel_xy / vel_norm), dim=-1).clamp(-1, 1)
            
            env_swing_offset[rec_ids] = swing_offset
            env_support_offset[rec_ids] = support_offset
            env_pelvis_yaw[rec_ids] = yaw
            env_dir_alignment[rec_ids] = cos_align
            geometry_recorded[rec_ids] = True

        # 6. Handle episode ends
        if dones.any():
            done_ids = dones.nonzero(as_tuple=True)[0]
            for idx in done_ids:
                if episodes_completed >= num_episodes:
                    break
                i = idx.item()
                episodes_completed += 1
                ep_lengths.append(ep_length_counter[i].item())

                if geometry_recorded[i]:
                    kick_successes += 1
                    swing_ball_offsets.append(env_swing_offset[i].cpu().numpy())
                    support_ball_offsets.append(env_support_offset[i].cpu().numpy())
                    pelvis_yaws.append(env_pelvis_yaw[i].item())
                    dir_alignments.append(env_dir_alignment[i].item())

                if ep_length_counter[i] < 100:
                    falls += 1

                # Reset per-env state
                ep_length_counter[i] = 0
                geometry_recorded[i] = False

    return {
        "offset": offset_m,
        "axis": axis,
        "episodes": episodes_completed,
        "kick_pct": kick_successes / max(episodes_completed, 1) * 100,
        "fall_pct": falls / max(episodes_completed, 1) * 100,
        "avg_ep_len": float(np.mean(ep_lengths)) if ep_lengths else 0,
        "swing_offsets": np.array(swing_ball_offsets) if swing_ball_offsets else np.zeros((0, 3)),
        "support_offsets": np.array(support_ball_offsets) if support_ball_offsets else np.zeros((0, 3)),
        "pelvis_yaws": np.array(pelvis_yaws),
        "dir_alignments": np.array(dir_alignments),
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

    # Parse offsets
    offsets = [float(o.strip()) for o in args_cli.offsets.split(",")]
    axis = args_cli.axis
    print(f"[INFO] Testing ball offsets along {axis.upper()}: {offsets}")

    # Create env
    env = gym.make(args_cli.task, cfg=env_cfg)
    env_wrapped = RslRlVecEnvWrapper(env)
    unwrapped_env = env.unwrapped
    command = unwrapped_env.command_manager.get_term("motion")
    num_joints = command.robot.data.joint_pos.shape[1]

    # Run for each offset
    results = []
    for offset in offsets:
        print(f"\n{'='*60}")
        print(f"  Ball offset = {offset:+.2f}m along {axis.upper()}")
        print(f"{'='*60}")

        # Create fresh builder
        v10_builder = V10ObsBuilder(
            num_envs=args_cli.num_envs,
            num_joints=num_joints,
            device=device,
        )
        
        # Reset env and apply offset
        env_wrapped.reset()
        v10_builder.init_segment_bounds(command)
        
        if abs(offset) > 1e-6:
            apply_ball_offset(unwrapped_env, command, offset, axis)

        result = run_perturbation_eval(
            env_wrapped, unwrapped_env, model, v10_builder, command,
            offset, axis, args_cli.num_episodes, device,
        )
        results.append(result)

        print(f"  Kick%: {result['kick_pct']:.1f}%  "
              f"Fall%: {result['fall_pct']:.1f}%  "
              f"EpLen: {result['avg_ep_len']:.1f}")
        if len(result['swing_offsets']) > 0:
            mean_swing = np.mean(result['swing_offsets'], axis=0)
            mean_support = np.mean(result['support_offsets'], axis=0)
            print(f"  Swing-ball offset:   [{mean_swing[0]:+.3f}, {mean_swing[1]:+.3f}, {mean_swing[2]:+.3f}]")
            print(f"  Support-ball offset: [{mean_support[0]:+.3f}, {mean_support[1]:+.3f}, {mean_support[2]:+.3f}]")
            print(f"  Mean dir alignment:  {np.mean(result['dir_alignments']):.3f}")

    # Summary table
    print(f"\n\n{'='*90}")
    print(f"v10.2 Ball Perturbation Diagnostic ({axis.upper()}-axis)")
    print(f"{'='*90}")
    print(f"{'Offset':>7} | {'Kick%':>6} | {'Fall%':>6} | {'EpLen':>7} | "
          f"{'Swing_X':>8} | {'Swing_Y':>8} | {'Supp_X':>8} | {'Supp_Y':>8} | {'DirAlign':>8}")
    print("-" * 90)
    
    offsets_arr = []
    swing_x_arr = []
    swing_y_arr = []
    support_x_arr = []
    support_y_arr = []
    
    for r in results:
        if len(r['swing_offsets']) > 0:
            ms = np.mean(r['swing_offsets'], axis=0)
            msp = np.mean(r['support_offsets'], axis=0)
            da = np.mean(r['dir_alignments'])
            offsets_arr.append(r['offset'])
            swing_x_arr.append(ms[0])
            swing_y_arr.append(ms[1])
            support_x_arr.append(msp[0])
            support_y_arr.append(msp[1])
        else:
            ms = [0, 0, 0]
            msp = [0, 0, 0]
            da = 0
        print(f"{r['offset']:>+7.2f} | {r['kick_pct']:>5.1f}% | {r['fall_pct']:>5.1f}% | "
              f"{r['avg_ep_len']:>7.1f} | {ms[0]:>+8.3f} | {ms[1]:>+8.3f} | "
              f"{msp[0]:>+8.3f} | {msp[1]:>+8.3f} | {da:>8.3f}")
    print(f"{'='*90}")

    # Compute geometry slopes
    if len(offsets_arr) >= 3:
        offsets_np = np.array(offsets_arr)
        
        # Linear regression: geometry_metric = a * offset + b
        from numpy.polynomial import polynomial as P
        
        axis_idx = 0 if axis == "x" else 1
        
        # Swing foot slope
        swing_data = np.array(swing_x_arr) if axis == "x" else np.array(swing_y_arr)
        swing_coefs = np.polyfit(offsets_np, swing_data, 1)
        swing_slope = swing_coefs[0]
        
        # Support foot slope
        support_data = np.array(support_x_arr) if axis == "x" else np.array(support_y_arr)
        support_coefs = np.polyfit(offsets_np, support_data, 1)
        support_slope = support_coefs[0]
        
        print(f"\nGeometry Slopes ({axis.upper()}-axis perturbation):")
        print(f"  Swing foot {axis.upper()} slope:   {swing_slope:+.3f} (m/m)")
        print(f"  Support foot {axis.upper()} slope: {support_slope:+.3f} (m/m)")
        
        # Interpretation
        # slope ≈ 0 → fixed trajectory (no adaptation)
        # slope ≈ -1 → perfect tracking (foot moves opposite to ball to maintain relative pos)
        # slope > 0 → foot moves with ball (some adaptation)
        
        if abs(swing_slope) > 0.3 or abs(support_slope) > 0.3:
            print(f"\n  ✅ GEOMETRY ADAPTATION DETECTED")
            print(f"     Foot positions change with ball offset — event-conditioned behavior.")
        elif abs(swing_slope) < 0.1 and abs(support_slope) < 0.1:
            print(f"\n  ❌ NO GEOMETRY ADAPTATION")
            print(f"     Foot positions fixed — v3-style tolerance absorption.")
        else:
            print(f"\n  ⚠️ WEAK ADAPTATION")
            print(f"     Some sensitivity to ball position, but not strong.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
