#!/usr/bin/env python3
"""Legacy unfiltered ball-force timing check for an S2 motion.

The robot is reset to the reference pose at every frame while a physical ball
starts stationary at the first labelled contact anchor.  This isolates the
reference geometry question from policy quality:

    Does the ball receive horizontal force near the manually labelled frame?

This tool does **not** identify a robot-ball body pair.  Its unfiltered ball
force includes ground interaction and must not be used as proof of foot-ball
contact, contact success, or contact-quality calibration.  Use
``calibrate_foot_ball_contact.py`` for filtered logical-foot validation.

Run with Isaac Lab's Python, for example::

    /path/to/IsaacLab/isaaclab.sh -p scripts/dribble/validate_first_contact_alignment.py \
      --motion motions/master-aligned/master_001675_001884_unitree_g1.npz --headless
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--motion", type=Path, required=True)
parser.add_argument("--force_threshold", type=float, default=1.0, help="Horizontal ball-force threshold in N")
parser.add_argument("--frame_tolerance", type=int, default=2)
parser.add_argument("--post_frames", type=int, default=8, help="Frames to run after the labelled event")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from soccer.robots.g1 import G1_CYLINDER_CFG
from soccer.tasks.tracking.config.g1.soccer_flat_env_cfg import G1FlatSoccerSceneCfg


def _first_contact_frame(contact: np.ndarray) -> int:
    mask = np.asarray(contact, dtype=np.int8).reshape(-1) > 0
    starts = np.flatnonzero(mask & np.concatenate(([True], ~mask[:-1])))
    if starts.size == 0:
        raise ValueError("Motion has no dribble_cg_contact rising edge")
    return int(starts[0])


def main() -> int:
    with np.load(args_cli.motion, allow_pickle=True) as motion:
        required = {"joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "ball_pos_w", "dribble_cg_contact"}
        missing = sorted(required.difference(motion.files))
        if missing:
            raise ValueError(f"{args_cli.motion}: missing {missing}")
        data = {key: np.asarray(motion[key]) for key in required}

    first_label = _first_contact_frame(data["dribble_cg_contact"])
    final_frame = min(data["joint_pos"].shape[0] - 1, first_label + max(0, args_cli.post_frames))

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=0.005))
    scene_cfg = G1FlatSoccerSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    ball = scene["soccer_ball"]
    ball_sensor = scene["soccer_ball_contact"]
    device = sim.device
    origin = scene.env_origins[0]
    fps = float(np.asarray(data.get("fps", 50.0)).reshape(-1)[0]) if "fps" in data else 50.0
    physics_steps = max(1, int(round((1.0 / fps) / sim.get_physics_dt())))

    # Keep the ball stationary until the reference robot reaches it.  Ball
    # position is the first labelled contact anchor, not ball_pos_w[0] by
    # assumption, so this also catches malformed source data.
    ball_pos = torch.as_tensor(data["ball_pos_w"][first_label], dtype=torch.float32, device=device) + origin
    ball_state = torch.zeros((1, 13), dtype=torch.float32, device=device)
    ball_state[:, :3] = ball_pos
    ball_state[:, 3] = 1.0
    ball.write_root_state_to_sim(ball_state)
    scene.write_data_to_sim()
    for _ in range(2):
        sim.step(render=False)
        scene.update(sim.get_physics_dt())

    first_physics_contact: int | None = None
    force_by_frame: list[float] = []
    for frame in range(final_frame + 1):
        root_state = torch.cat(
            [
                torch.as_tensor(data["body_pos_w"][frame, 0], dtype=torch.float32, device=device).unsqueeze(0) + origin,
                torch.as_tensor(data["body_quat_w"][frame, 0], dtype=torch.float32, device=device).unsqueeze(0),
                torch.as_tensor(data["body_lin_vel_w"][frame, 0], dtype=torch.float32, device=device).unsqueeze(0),
                torch.as_tensor(data["body_ang_vel_w"][frame, 0], dtype=torch.float32, device=device).unsqueeze(0),
            ],
            dim=-1,
        )
        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(
            torch.as_tensor(data["joint_pos"][frame], dtype=torch.float32, device=device).unsqueeze(0),
            torch.as_tensor(data["joint_vel"][frame], dtype=torch.float32, device=device).unsqueeze(0),
        )
        scene.write_data_to_sim()

        max_horizontal_force = 0.0
        for _ in range(physics_steps):
            sim.step(render=False)
            scene.update(sim.get_physics_dt())
            force = ball_sensor.data.net_forces_w[0, 0, :2]
            max_horizontal_force = max(max_horizontal_force, float(torch.linalg.vector_norm(force).item()))
        force_by_frame.append(max_horizontal_force)
        if first_physics_contact is None and max_horizontal_force > args_cli.force_threshold:
            first_physics_contact = frame

    print(f"motion: {args_cli.motion}")
    print(f"first labelled contact frame: {first_label}")
    print(f"stationary ball anchor: {data['ball_pos_w'][first_label].tolist()}")
    print(f"first legacy unfiltered ball-force frame: {first_physics_contact}")
    print("WARNING: NOT verified robot-ball contact; link and foot are unknown.")
    print("frame  horizontal_force_N")
    for frame, force in enumerate(force_by_frame):
        if abs(frame - first_label) <= 5 or force > args_cli.force_threshold:
            print(f"{frame:5d}  {force:.5f}")

    if first_physics_contact is None:
        print("RESULT: FAIL — no legacy unfiltered horizontal ball-force event was detected.")
        return 1
    delta = first_physics_contact - first_label
    if abs(delta) > args_cli.frame_tolerance:
        print(f"RESULT: FAIL — collision is {delta:+d} frames from the label.")
        return 1
    print(f"RESULT: PASS — unfiltered ball-force timing is {delta:+d} frames (contact NOT verified).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
