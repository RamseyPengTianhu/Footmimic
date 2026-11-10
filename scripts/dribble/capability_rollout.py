"""Capability rollout on one dribble reference with a frozen tracker (no learning).  Raw per-step log -> npz.

Task Tracking-Flat-G1-BallDribble (clean-v0 Stage-1 actor contract, ball spawned at the reference first contact,
episode from frame 0).  Ball-related terminations (ball lost, non-foot contact) are removed so the episode runs until
a fall / tracking termination or the end of the clip; only each env's FIRST episode is logged.

Conditions (--condition):
  ghost_follow       ball without collisions, written every step to (0.5, -0.15) m in the pelvis-yaw frame
                     (Stage-1-like ball observation; reproduces the earlier 64/64)
  ghost_static       ball without collisions, left at the first-contact spawn
  real               physical ball, constant deceleration 0.4905 m/s^2, asset damping 0
  real_no_resist     physical ball, no deceleration event, asset damping 0
  hold_ahead         physical ball written every step to (0.5, -0.15) m in the pelvis-yaw frame, zero velocity
                     (reproduces the earlier 64/64 causal test)
  reset_after_touch  physical ball; whenever a foot touch ends, the ball is put back to the reference ball state of
                     the current frame, mapped from the reference pelvis-yaw frame into the robot pelvis-yaw frame
  offset             physical ball, spawn moved by --offset m in a random horizontal direction (angle logged)
--no_resist can be combined with any physical condition.

  python scripts/dribble/capability_rollout.py --condition real --load_run <run> --checkpoint model_X.pt \
      --motion_path <npz> --num_envs 128 --output out.npz --headless [--video out.mp4 --enable_cameras]
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rsl_rl"))
import cli_args  # isort: skip

CONDITIONS = ("ghost_follow", "ghost_static", "real", "real_no_resist", "hold_ahead", "reset_after_touch", "offset")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Tracking-Flat-G1-BallDribble")
parser.add_argument("--condition", required=True, choices=CONDITIONS)
parser.add_argument("--offset", type=float, default=0.0)
parser.add_argument("--no_resist", action="store_true")
parser.add_argument("--motion_path", required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=0, help="0 = clip length + 5 (first episode only)")
parser.add_argument("--output", required=True)
parser.add_argument("--video", default="", help="also record env 0 to this mp4 (needs --enable_cameras)")
parser.add_argument("--video_steps", type=int, default=600)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_inv, yaw_quat  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import soccer.tasks  # noqa: E402,F401
from soccer.tasks.tracking.mdp import ball_contact, path_drift  # noqa: E402
from soccer.utils.my_on_policy_runner import MotionOnPolicyRunner  # noqa: E402

GHOST = args_cli.condition.startswith("ghost")
HOLD = args_cli.condition in ("ghost_follow", "hold_ahead")
HOLD_OFFSET = (0.5, -0.15, 0.0)
BALL_Z = 0.11
CONTACT_N = 1.0


def _yaw_frame_apply(q, v):
    return quat_apply(yaw_quat(q), v)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.motion.motion_files = [args_cli.motion_path]
    env_cfg.terminations.ball_lost = None
    env_cfg.terminations.nonfoot_ball_contact = None
    if GHOST:
        import isaaclab.sim as sim_utils

        env_cfg.scene.soccer_ball.spawn = env_cfg.scene.soccer_ball.spawn.replace(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        )
    if GHOST or args_cli.no_resist or args_cli.condition == "real_no_resist":
        env_cfg.events.ball_deceleration = None
    if args_cli.video:
        env_cfg.viewer.eye = (4.0, 4.0, 3.5)  # high and wide: a ball left 1-2 m behind stays in view
        env_cfg.viewer.lookat = (0.0, 0.0, 0.3)
        env_cfg.commands.motion.debug_vis = False
        env_cfg.scene.contact_forces.debug_vis = False
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None))
    base = env.unwrapped
    dev = base.device
    N = base.num_envs

    runner = MotionOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.git_status_repos = []
    ckpt = get_checkpoint_path(os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)),
                               agent_cfg.load_run, agent_cfg.load_checkpoint)
    runner.load(ckpt)
    policy = runner.alg.policy
    policy.eval()
    act = runner.get_inference_policy(device=agent_cfg.device)

    cmd = base.command_manager.get_term("motion")
    tm = base.termination_manager
    robot, ball = base.scene["robot"], base.scene["soccer_ball"]
    foot_ids = robot.find_bodies(list(ball_contact.FOOT_BODY_NAMES), preserve_order=True)[0]
    clip_len = int(cmd.motion.file_lengths[0])
    fps = float(cmd.motion.fps if not hasattr(cmd.motion.fps, "__len__") else cmd.motion.fps[0])
    ref_ball = cmd.motion.ball_pos_w[0, :clip_len]  # raw reference coordinates
    ref_ball_vel = torch.zeros_like(ref_ball)
    ref_ball_vel[1:] = (ref_ball[1:] - ref_ball[:-1]) * fps
    steps = args_cli.steps or clip_len + 5
    if args_cli.video:
        steps = max(steps, args_cli.video_steps)
    origins = base.scene.env_origins

    def write_ball(ids, pos_w, vel_w=None):
        if ids.numel() == 0:
            return
        pose = torch.cat([pos_w, torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).expand(ids.numel(), 4)], dim=-1)
        ball.write_root_pose_to_sim(pose, env_ids=ids)
        vel = torch.zeros(ids.numel(), 6, device=dev)
        if vel_w is not None:
            vel[:, :2] = vel_w[:, :2]
        ball.write_root_velocity_to_sim(vel, env_ids=ids)

    def hold_ahead(ids):
        q = robot.data.root_quat_w[ids]
        pos = robot.data.root_pos_w[ids] + _yaw_frame_apply(q, torch.tensor(HOLD_OFFSET, device=dev).expand(ids.numel(), 3))
        pos[:, 2] = origins[ids, 2] + BALL_Z
        write_ball(ids, pos)

    all_ids = torch.arange(N, device=dev)
    offset_angle = torch.zeros(N, device=dev)
    if args_cli.condition == "offset" and args_cli.offset > 0:
        offset_angle = torch.rand(N, device=dev) * 2 * torch.pi  # relative to the frame-0 pelvis yaw
        d = torch.stack([torch.cos(offset_angle), torch.sin(offset_angle), torch.zeros_like(offset_angle)], -1) * args_cli.offset
        pos = ball.data.root_pos_w.clone() + _yaw_frame_apply(robot.data.root_quat_w, d)
        pos[:, 2] = origins[:, 2] + BALL_Z
        write_ball(all_ids, pos)
    if HOLD:
        hold_ahead(all_ids)

    keys = ("frame", "pelvis_pos", "pelvis_quat", "pelvis_vel", "ref_pelvis_pos_ep", "ref_pelvis_pos", "ref_pelvis_quat",
            "cmd_local", "ball_pos", "ball_vel", "foot_pos", "foot_quat", "ball_force_link", "ball_force_feet_vec",
            "ball_reset")
    log = {k: [] for k in keys}
    alive = torch.ones(N, dtype=torch.bool, device=dev)
    end_step = torch.full((N,), -1, dtype=torch.long, device=dev)
    end_reason = ["running"] * N
    prev_foot_touch = torch.zeros(N, dtype=torch.bool, device=dev)
    frames = []
    obs, _ = env.get_observations()
    with torch.inference_mode():
        policy.reset()
        for step in range(steps):
            if HOLD:
                hold_ahead(all_ids)
            o = origins
            log["frame"].append(cmd.time_steps.clone())
            log["pelvis_pos"].append(cmd.robot_anchor_pos_w - o)
            log["pelvis_quat"].append(cmd.robot_anchor_quat_w.clone())
            log["pelvis_vel"].append(cmd.robot_anchor_lin_vel_w.clone())
            log["ref_pelvis_pos_ep"].append(path_drift.reference_pelvis_xy_episode(base) - o[:, :2])
            log["ref_pelvis_pos"].append(cmd.anchor_pos_w - o)
            log["ref_pelvis_quat"].append(cmd.anchor_quat_w.clone())
            log["cmd_local"].append(cmd.locomotion_twist_command_b().view(N, -1)[:, :3].clone())
            log["ball_pos"].append(ball.data.root_pos_w - o)
            log["ball_vel"].append(ball.data.root_lin_vel_w.clone())
            log["foot_pos"].append(robot.data.body_pos_w[:, foot_ids] - o.unsqueeze(1))
            log["foot_quat"].append(robot.data.body_quat_w[:, foot_ids].clone())

            obs, _, dones, _ = env.step(act(obs))
            policy.reset(dones)

            fm_h = ball_contact.ball_force_matrix_steps(base)  # [N, substeps, 30, 3] force on the ball
            mag_h = torch.linalg.vector_norm(fm_h, dim=-1)
            log["ball_force_link"].append(mag_h.amax(dim=1))
            pick = mag_h[:, :, :2].argmax(dim=1)  # per foot: substep with the largest force
            feet_vec = torch.gather(fm_h[:, :, :2], 1, pick[:, None, :, None].expand(-1, 1, -1, 3))[:, 0]
            log["ball_force_feet_vec"].append(feet_vec)
            foot_touch = (mag_h[:, :, :2].amax(dim=1) > CONTACT_N).any(dim=-1)
            reset_mask = torch.zeros(N, dtype=torch.bool, device=dev)
            if args_cli.condition == "reset_after_touch":
                ended = prev_foot_touch & ~foot_touch & ~dones.bool()
                ids = ended.nonzero().flatten()
                if ids.numel():
                    fr = cmd.time_steps[ids].clamp(max=clip_len - 1)
                    rel = ref_ball[fr] + o[ids] - cmd.anchor_pos_w[ids]
                    rel[:, 2] = 0.0
                    q_ref, q_rob = cmd.anchor_quat_w[ids], cmd.robot_anchor_quat_w[ids]
                    local = quat_apply(quat_inv(yaw_quat(q_ref)), rel)
                    pos = cmd.robot_anchor_pos_w[ids] + _yaw_frame_apply(q_rob, local)
                    pos[:, 2] = o[ids, 2] + BALL_Z
                    vel = _yaw_frame_apply(q_rob, quat_apply(quat_inv(yaw_quat(q_ref)), ref_ball_vel[fr]))
                    write_ball(ids, pos, vel)
                    reset_mask[ids] = True
            log["ball_reset"].append(reset_mask)
            prev_foot_touch = foot_touch & ~dones.bool()

            d = dones.bool() & alive
            if d.any():
                for i in d.nonzero().flatten().tolist():
                    end_step[i] = step
                    end_reason[i] = next((t for t in tm.active_terms if bool(tm.get_term(t)[i])), "unknown")
                alive &= ~d
            if args_cli.video:
                frame_rgb = base.render()
                if frame_rgb is not None:
                    frames.append((np.asarray(frame_rgb), step))
            if not alive.any() and not args_cli.video:
                break

    out = {k: torch.stack(v, dim=1).cpu().numpy() for k, v in log.items()}  # [N, T, ...]
    out["end_step"] = end_step.cpu().numpy()
    out["end_reason"] = np.array(end_reason)
    out["offset_angle"] = offset_angle.cpu().numpy()
    out["ref_ball"] = ref_ball.cpu().numpy()
    out["ref_contact"] = cmd.motion.dribble_cg_contact[0, :clip_len].cpu().numpy()
    out["ref_foot"] = cmd.motion.dribble_cg_foot[0, :clip_len].cpu().numpy()
    out["ref_surface"] = cmd.motion.dribble_cg_surface[0, :clip_len].cpu().numpy()
    out["link_names"] = np.array(ball_contact.BALL_FILTER_BODY_NAMES)
    meta = {"condition": args_cli.condition, "offset": args_cli.offset, "no_resist": bool(args_cli.no_resist),
            "checkpoint": ckpt, "motion_path": args_cli.motion_path, "num_envs": N, "steps": steps,
            "clip_len": clip_len, "fps": fps, "step_dt": float(base.step_dt), "contact_threshold_N": CONTACT_N,
            "hold_offset_pelvis_yaw": HOLD_OFFSET}
    out["meta"] = np.array(json.dumps(meta))
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)), exist_ok=True)
    np.savez_compressed(args_cli.output, **out)
    print("[ROLLOUT]", json.dumps(meta), "ended:", {r: end_reason.count(r) for r in set(end_reason)})

    if args_cli.video and frames:
        _write_video(args_cli.video, frames, out, meta)
    sys.stdout.flush()
    os._exit(0)


def _write_video(path, frames, out, meta):
    """Env 0, overlay: frame, touch count (foot touches since the episode start), ball speed, contact link."""
    import cv2

    force = out["ball_force_link"][0]  # [T, 30]
    names = list(out["link_names"])
    ball_vel = out["ball_vel"][0]
    ref_frame = out["frame"][0]
    h, w = frames[0][0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), int(round(1.0 / meta["step_dt"])), (w, h))
    touches, prev = 0, False
    for img, step in frames:
        if step > 0 and ref_frame[step] < ref_frame[step - 1]:
            touches, prev = 0, False  # new episode
        foot = force[step, :2].max() > meta["contact_threshold_N"]
        if foot and not prev:
            touches += 1
        prev = foot
        links = [names[j] for j in np.nonzero(force[step] > meta["contact_threshold_N"])[0]]
        speed = float(np.linalg.norm(ball_vel[step, :2]))
        bgr = cv2.cvtColor(img[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
        lines = [f"{meta['condition']}  ckpt {os.path.basename(meta['checkpoint'])}",
                 f"ref frame {int(ref_frame[step])}  t={step * meta['step_dt']:.2f}s",
                 f"foot touches {touches}   ball speed {speed:.2f} m/s",
                 f"ball contact: {', '.join(links) if links else '-'}"]
        for k, s in enumerate(lines):
            cv2.putText(bgr, s, (12, 30 + 28 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(bgr, s, (12, 30 + 28 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(bgr)
    vw.release()
    print("[ROLLOUT] video", path, len(frames), "frames")


if __name__ == "__main__":
    main()
