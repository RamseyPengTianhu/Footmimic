"""Kinematic validation of retargeted G1 motion clips (no learning, no dynamics).

For every clip:
  * FK consistency: pose the simulated G1 at the clip's root pose + joint angles and compare all 30 body
    positions with the clip's ``body_pos_w`` (this also proves the joint order matches the robot);
  * joint limits: frames with any joint outside the robot's position limits, and the worst violation;
  * joint speed: 99th-percentile |joint_vel| compared with the dribbling references;
  * foot skating: horizontal ankle speed while that foot is planted (ankle within 1.5 cm of its lowest level).
This checks that the retargeting is geometrically valid for G1.  It does NOT show that a tracker can execute
the motion (balance / dynamics) -- that needs a trained tracker.

  python scripts/dribble/validate_retarget.py --inputs motions/amass_walk_stand_g1_v1 \
      --references motions/master-aligned motions/master-annotated-003178-grounded --output out.json --headless
"""

import argparse
import glob
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--inputs", nargs="+", required=True, help="npz files or directories (recursive)")
parser.add_argument("--references", nargs="*", default=[], help="clips used as the normal-range baseline")
parser.add_argument("--stride", type=int, default=5, help="FK check every N-th frame")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--output", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False
simulation_app = AppLauncher(args_cli).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

from soccer.robots.g1 import G1_CYLINDER_CFG  # noqa: E402
from soccer.tasks.tracking.config.g1.soccer_flat_env_cfg import G1FlatSoccerSceneCfg  # noqa: E402


def files_of(items):
    out = []
    for it in items:
        out += [it] if os.path.isfile(it) else sorted(glob.glob(os.path.join(it, "**", "*.npz"), recursive=True))
    return out


def offline_stats(z):
    fps = int(np.asarray(z["fps"]).reshape(-1)[0])
    jv = np.abs(np.asarray(z["joint_vel"]))
    bp = np.asarray(z["body_pos_w"])
    skate = []
    for fi in (18, 19):
        h = bp[:, fi, 2]
        planted = h < np.percentile(h, 10) + 0.015
        v = np.linalg.norm(np.diff(bp[:, fi, :2], axis=0), axis=1) * fps
        pv = v[planted[1:] & planted[:-1]]
        if pv.size:
            skate.append(pv)
    skate = np.concatenate(skate) if skate else np.zeros(1)
    return {"joint_vel_p99": float(np.percentile(jv, 99)), "joint_vel_max": float(jv.max()),
            "skate_median": float(np.median(skate)), "skate_p90": float(np.percentile(skate, 90))}


def main():
    dt = 0.005
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=dt))
    cfg = G1FlatSoccerSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    cfg.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene = InteractiveScene(cfg)
    sim.reset()
    robot = scene["robot"]
    ball = scene["soccer_ball"]
    dev = sim.device
    n = scene.num_envs
    origins = scene.env_origins
    limits = robot.data.joint_pos_limits[0].cpu().numpy()  # [J, 2]
    print(f"[RETARGET] robot bodies {robot.num_bodies}, joints {robot.num_joints}")
    # park the ball far away so it cannot touch the robots
    bs = ball.data.default_root_state.clone(); bs[:, :3] = origins + torch.tensor([0.0, 0.0, -10.0], device=dev)
    ball.write_root_state_to_sim(bs)

    results = {}
    for role, paths in (("reference", files_of(args_cli.references)), ("input", files_of(args_cli.inputs))):
        for p in paths:
            z = np.load(p, allow_pickle=True)
            jp = np.asarray(z["joint_pos"], np.float32); bp = np.asarray(z["body_pos_w"], np.float32)
            bq = np.asarray(z["body_quat_w"], np.float32)
            frames = np.arange(0, len(jp), args_cli.stride)
            errs = []
            for c in range(0, len(frames), n):
                fr = frames[c:c + n]; k = len(fr)
                ids = torch.arange(k, device=dev)
                root = torch.zeros(k, 13, device=dev)
                root[:, :3] = torch.as_tensor(bp[fr, 0], device=dev) + origins[:k]
                root[:, 3:7] = torch.as_tensor(bq[fr, 0], device=dev)
                robot.write_root_state_to_sim(root, env_ids=ids)
                q = torch.as_tensor(jp[fr], device=dev)
                robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
                scene.write_data_to_sim(); sim.step(render=False); scene.update(dt)
                sim_bp = (robot.data.body_pos_w[:k] - origins[:k, None, :]).cpu().numpy()
                errs.append(np.linalg.norm(sim_bp - bp[fr], axis=-1))
            e = np.concatenate(errs, axis=0)  # [frames, bodies]
            lo = jp < limits[None, :, 0] - 1e-3; hi = jp > limits[None, :, 1] + 1e-3
            viol = np.maximum(limits[None, :, 0] - jp, jp - limits[None, :, 1]).clip(min=0)
            r = {"role": role, "frames": int(len(jp)),
                 "fk_err_median_cm": float(np.median(e) * 100), "fk_err_max_cm": float(e.max() * 100),
                 "limit_frames_frac": float((lo | hi).any(axis=1).mean()), "limit_max_rad": float(viol.max()),
                 "limit_joints": sorted({int(j) for j in np.flatnonzero((lo | hi).any(axis=0))})}
            r.update(offline_stats(z))
            results[p] = r
            print(f"[RETARGET] {role:9s} {os.path.basename(p)[:60]:60s} fk {r['fk_err_median_cm']:.2f}/{r['fk_err_max_cm']:.2f} cm  "
                  f"limits {r['limit_frames_frac']:.2f} ({r['limit_max_rad']:.3f} rad)  jv_p99 {r['joint_vel_p99']:.1f}  skate {r['skate_median']:.2f}/{r['skate_p90']:.2f} m/s")
    json.dump({"joint_names": list(robot.joint_names), "joint_limits": limits.tolist(), "clips": results},
              open(args_cli.output, "w"), indent=1)
    print(f"[RETARGET] saved {args_cli.output}")
    os._exit(0)


if __name__ == "__main__":
    main()
