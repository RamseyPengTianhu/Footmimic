"""V0b probe: how close does a (frozen) policy get its foot to the ball in every touch window?

No learning happens.  The policy is loaded from a checkpoint and rolled out on a
dribbling task over the *whole* clip (all labelled events), with the miss/non-foot/
ball-lost terminations removed so an episode is only ended by a fall or clip end.

For each (env, event) it records, inside the tolerance-expanded labelled window:
  * min 3-D distance between the expected ankle link and the ball centre, plus the
    ball centre in the ankle-local frame at that moment,
  * whether the filtered robot-ball contact sensor ever read force (real contact).
It also records the ball speed 0.3 s after the window (did the ball get kicked?).

Example:
  python scripts/dribble/probe_policy_foot_ball.py \
      --task Tracking-Flat-G1-CleanV0-Stage2-C2-RNN --motion_path motions/master-aligned \
      --load_run <run dir> --checkpoint model_6999.pt --num_envs 32 --output out.npz --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rsl_rl"))
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--motion_path", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--noise_std", type=float, default=0.0, help="Gaussian noise added to policy actions (emulates exploration std)")
parser.add_argument("--hold_ball_ahead", action="store_true", help="Causal test: teleport the ball to 0.5 m ahead of the pelvis every step (Stage-1 like ball observation)")
parser.add_argument("--keep_ball_lost", action="store_true", help="Keep the ball_lost termination (evaluate under the task's normal terminations)")
parser.add_argument("--max_steps", type=int, default=0, help="0 = clip length + 20")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import glob  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import soccer.tasks  # noqa: E402,F401
from soccer.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner  # noqa: E402


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    files = [args_cli.motion_path] if os.path.isfile(args_cli.motion_path) else sorted(
        glob.glob(os.path.join(args_cli.motion_path, "*.npz"))
    )
    env_cfg.commands.motion.motion_files = files

    # Whole-clip episode: select every labelled event, end only on fall / clip end.
    with np.load(files[0], allow_pickle=True) as z:
        contact = z["dribble_cg_contact"] > 0
    num_events = int(np.count_nonzero(contact & ~np.r_[False, contact[:-1]]))
    cmd_cfg = env_cfg.commands.motion
    is_s2 = hasattr(cmd_cfg, "dribble_cg_curriculum_required_contacts") and hasattr(cmd_cfg, "dribble_cg_contact_region_path")
    cmd_cfg.dribble_cg_curriculum_levels = (((0, 1.0),),)  # 0 = every event, full clip
    cmd_cfg.dribble_cg_curriculum_required_contacts = (0,)
    cmd_cfg.dribble_cg_curriculum_start_level = 0
    cmd_cfg.dribble_cg_curriculum_lookahead_contact_phases = 0
    cmd_cfg.dribble_cg_terminal_progress_levels = ()
    for name in ("s2_missed_contact", "s2_nonfoot_contact") + (() if args_cli.keep_ball_lost else ("ball_lost",)):
        if hasattr(env_cfg.terminations, name):
            setattr(env_cfg.terminations, name, None)
    if hasattr(env_cfg.curriculum, "s2_contact_levels"):
        env_cfg.curriculum.s2_contact_levels = None

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device, registry_name=None)
    runner.git_status_repos = []
    path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[PROBE] loading {path}")
    runner.load(path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base = env.unwrapped
    cmd = base.command_manager.get_term("motion")
    robot = base.scene["robot"]
    ball = base.scene["soccer_ball"]
    sensor = base.scene["soccer_ball_contact"]
    right_id = robot.find_bodies("right_ankle_roll_link")[0][0]
    left_id = robot.find_bodies("left_ankle_roll_link")[0][0]
    n = base.num_envs
    dev = base.device
    has_s2 = hasattr(cmd, "_s2_event_frames_padded")
    max_events = int(cmd._s2_event_frames_padded.shape[1]) if has_s2 else 1
    clip_len = int(cmd.motion.file_lengths[0].item())
    steps = args_cli.max_steps or clip_len + 20
    after = int(round(0.3 / base.step_dt))

    INF = float("inf")
    min_dist = torch.full((n, max_events), INF, device=dev)
    min_local = torch.full((n, max_events, 3), float("nan"), device=dev)
    min_frame = torch.full((n, max_events), -1, device=dev, dtype=torch.long)
    touched = torch.zeros((n, max_events), device=dev, dtype=torch.bool)
    ball_speed_after = torch.full((n, max_events), float("nan"), device=dev)
    window_end = cmd._s2_event_ends_by_motion[0].to(dev) if has_s2 else torch.zeros(1, device=dev, dtype=torch.long)
    alive = torch.ones(n, device=dev, dtype=torch.bool)  # first episode only
    fell = torch.zeros(n, device=dev, dtype=torch.bool)
    last_frame = torch.zeros(n, device=dev, dtype=torch.long)

    ts_frame, ts_ball_speed, ts_force, ts_dist_r, ts_alive, ts_ball_xy, ts_root_xy, ts_event = [], [], [], [], [], [], [], []
    done_reason = ["" for _ in range(n)]
    done_frame = torch.full((n,), -1, device=dev, dtype=torch.long)
    obs, _ = env.get_observations()
    for _ in range(steps):
        frame = cmd.time_steps.clone()
        mi = cmd.motion_idx.to(torch.long)
        if has_s2:
            ev = cmd._s2_event_id[mi, frame.clamp(max=cmd._s2_event_id.shape[1] - 1)]
            foot = cmd._s2_event_foot[mi, frame.clamp(max=cmd._s2_event_foot.shape[1] - 1)].to(torch.long)
        else:
            ev = torch.full_like(frame, -1)
            foot = torch.ones_like(frame)
        in_window = (ev >= 0) & alive
        bid = torch.where(foot == 1, torch.full_like(foot, right_id), torch.full_like(foot, left_id))
        idx = torch.arange(n, device=dev)
        fpos = robot.data.body_pos_w[idx, bid]
        fquat = robot.data.body_quat_w[idx, bid]
        bpos = ball.data.root_pos_w
        d = torch.linalg.vector_norm(bpos - fpos, dim=-1)
        # ball centre in ankle-local frame
        from isaaclab.utils.math import quat_apply_inverse

        local = quat_apply_inverse(fquat, bpos - fpos)
        fm = sensor.data.force_matrix_w
        force = (
            torch.linalg.vector_norm(fm, dim=-1).amax(dim=(-1, -2))
            if fm is not None
            else torch.zeros(n, device=dev)
        )
        evc = ev.clamp(min=0)
        better = in_window & (d < min_dist[idx, evc])
        min_dist[idx[better], evc[better]] = d[better]
        min_local[idx[better], evc[better]] = local[better]
        min_frame[idx[better], evc[better]] = frame[better]
        touched[idx[in_window], evc[in_window]] |= force[in_window] > 1.0
        # ball speed shortly after each window ended
        for e in range(min(max_events, window_end.numel())):
            at = alive & (frame == int(window_end[e].item()) + after)
            if at.any():
                ball_speed_after[at, e] = torch.linalg.vector_norm(ball.data.root_lin_vel_w[at, :2], dim=-1)
        last_frame = torch.where(alive, frame, last_frame)
        ts_frame.append(frame.clone()); ts_alive.append(alive.clone())
        ts_ball_speed.append(torch.linalg.vector_norm(ball.data.root_lin_vel_w[:, :2], dim=-1))
        ts_force.append(force.clone()); ts_event.append(ev.clone())
        ts_dist_r.append(torch.linalg.vector_norm(ball.data.root_pos_w - robot.data.body_pos_w[:, right_id], dim=-1))
        ts_ball_xy.append(ball.data.root_pos_w[:, :2].clone()); ts_root_xy.append(robot.data.root_pos_w[:, :2].clone())

        with torch.inference_mode():
            actions = policy(obs)
            if args_cli.noise_std > 0.0:
                actions = actions + args_cli.noise_std * torch.randn_like(actions)
            obs, _, dones, _ = env.step(actions)
        if args_cli.hold_ball_ahead:
          with torch.inference_mode():
            from isaaclab.utils.math import yaw_quat, quat_apply

            q = yaw_quat(robot.data.root_quat_w)
            off = torch.tensor([0.5, -0.15, 0.0], device=dev).expand(n, 3)
            pos = robot.data.root_pos_w + quat_apply(q, off)
            pos[:, 2] = 0.11
            pose = torch.cat([pos, torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).expand(n, 4)], dim=-1)
            ball.write_root_pose_to_sim(pose)
            ball.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
        newly_done = dones.bool() & alive
        if newly_done.any():
            tm = base.termination_manager
            for i in torch.nonzero(newly_done).flatten().tolist():
                done_reason[i] = ",".join(t for t in tm.active_terms if bool(tm.get_term(t)[i]))
                done_frame[i] = frame[i]
            fell |= newly_done & (
                tm.get_term("anchor_pos_z").bool() | tm.get_term("anchor_ori").bool()
                if "anchor_pos_z" in tm.active_terms
                else torch.zeros_like(newly_done)
            )
        alive = alive & ~dones.bool()
        if not alive.any():
            break

    num_ev = int(cmd._s2_event_count_by_motion[0].item()) if has_s2 else 0
    out = dict(
        min_dist=min_dist[:, :num_ev].cpu().numpy(),
        min_local=min_local[:, :num_ev].cpu().numpy(),
        min_frame=min_frame[:, :num_ev].cpu().numpy(),
        touched=touched[:, :num_ev].cpu().numpy(),
        ball_speed_after=ball_speed_after[:, :num_ev].cpu().numpy(),
        last_frame=last_frame.cpu().numpy(),
        fell=fell.cpu().numpy(),
        event_frames=cmd._s2_event_frames_by_motion[0].cpu().numpy() if has_s2 else np.zeros(0),
        window_starts=cmd._s2_event_window_starts_by_motion[0].cpu().numpy() if has_s2 else np.zeros(0),
        window_ends=cmd._s2_event_ends_by_motion[0].cpu().numpy() if has_s2 else np.zeros(0),
        clip_len=clip_len,
        done_reason=np.array(done_reason), done_frame=done_frame.cpu().numpy(),
        ts_frame=torch.stack(ts_frame).cpu().numpy(), ts_alive=torch.stack(ts_alive).cpu().numpy(),
        ts_ball_speed=torch.stack(ts_ball_speed).cpu().numpy(), ts_force=torch.stack(ts_force).cpu().numpy(),
        ts_event=torch.stack(ts_event).cpu().numpy(), ts_dist_r=torch.stack(ts_dist_r).cpu().numpy(),
        ts_ball_xy=torch.stack(ts_ball_xy).cpu().numpy(), ts_root_xy=torch.stack(ts_root_xy).cpu().numpy(),
    )
    np.savez_compressed(args_cli.output, **out)
    print(f"[PROBE] saved {args_cli.output}")
    os._exit(0)  # Isaac teardown hangs; results are already on disk.


if __name__ == "__main__":
    main()
