#!/usr/bin/env python3
"""Static filtered foot-ball contact calibration around labelled anchors.

The robot is held at a selected reference pose while a physical ball is reset
on a grid around the manually labelled ball-centre anchor.  Only the two
configured logical-foot filtered channels are accepted as contact facts;
unfiltered ball/ground force is recorded separately and never counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--motion", type=Path, required=True)
parser.add_argument("--event_ids", type=int, nargs="+", default=[0])
parser.add_argument("--timing_offsets", type=int, nargs="+", default=[-2, -1, 0, 1, 2])
parser.add_argument("--long_offsets", type=float, nargs="+", default=[-0.04, -0.02, 0.0, 0.02, 0.04])
parser.add_argument("--lat_offsets", type=float, nargs="+", default=[-0.04, -0.02, 0.0, 0.02, 0.04])
parser.add_argument("--z_offsets", type=float, nargs="+", default=[-0.02, 0.0, 0.02])
parser.add_argument("--physics_steps", type=int, default=4)
parser.add_argument("--contact_epsilon", type=float, default=1.0e-6)
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from soccer.robots.g1 import G1_CYLINDER_CFG
from soccer.tasks.tracking.config.g1.soccer_flat_env_cfg import G1FlatSoccerSceneCfg


FOOT_NAMES = ("right_ankle_roll_link", "left_ankle_roll_link")


def _event_starts(contact: np.ndarray) -> np.ndarray:
    mask = np.asarray(contact).reshape(-1) > 0
    return np.flatnonzero(mask & np.concatenate(([True], ~mask[:-1])))


def _write_robot_pose(robot, data: dict[str, np.ndarray], frame: int, origin: torch.Tensor, device: str) -> None:
    root = torch.cat(
        (
            torch.as_tensor(data["body_pos_w"][frame, 0], device=device).unsqueeze(0) + origin,
            torch.as_tensor(data["body_quat_w"][frame, 0], device=device).unsqueeze(0),
            torch.zeros((1, 6), dtype=torch.float32, device=device),
        ),
        dim=-1,
    ).to(torch.float32)
    robot.write_root_state_to_sim(root)
    robot.write_joint_state_to_sim(
        torch.as_tensor(data["joint_pos"][frame], dtype=torch.float32, device=device).unsqueeze(0),
        torch.zeros_like(torch.as_tensor(data["joint_vel"][frame], dtype=torch.float32, device=device)).unsqueeze(0),
    )


def main() -> int:
    with np.load(args_cli.motion, allow_pickle=True) as source:
        required = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "ball_pos_w", "dribble_cg_contact", "dribble_cg_foot", "dribble_cg_surface")
        missing = [name for name in required if name not in source.files]
        if missing:
            raise ValueError(f"{args_cli.motion}: missing {missing}")
        data = {name: np.asarray(source[name]) for name in required}
    starts = _event_starts(data["dribble_cg_contact"])
    if starts.size == 0:
        raise ValueError("motion has no contact events")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=0.005))
    scene_cfg = G1FlatSoccerSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # Filter order is the logical-foot order used in this output.
    scene_cfg.soccer_ball_contact.filter_prim_paths_expr = [
        f"{{ENV_REGEX_NS}}/Robot/{name}" for name in FOOT_NAMES
    ]
    scene_cfg.soccer_ball_contact.history_length = 1
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    robot = scene["robot"]
    ball = scene["soccer_ball"]
    sensor = scene["soccer_ball_contact"]
    device = sim.device
    origin = scene.env_origins[0]
    foot_body_ids = robot.find_bodies(list(FOOT_NAMES), preserve_order=True)[0]
    if len(foot_body_ids) != 2:
        raise RuntimeError(f"unable to resolve logical feet {FOOT_NAMES}: {foot_body_ids}")

    rows: list[tuple] = []
    for event_id in args_cli.event_ids:
        if event_id < 0 or event_id >= starts.size:
            raise ValueError(f"event_id {event_id} outside 0..{starts.size - 1}")
        anchor_frame = int(starts[event_id])
        expected_foot = int(data["dribble_cg_foot"][anchor_frame])
        surface = int(data["dribble_cg_surface"][anchor_frame])
        if expected_foot not in (0, 1):
            raise ValueError(f"event {event_id} has invalid foot {expected_foot}")
        # Dataset foot ids are 1=right, 0=left; sensor channels are right,left.
        expected_channel = 0 if expected_foot == 1 else 1
        for timing_offset in args_cli.timing_offsets:
            frame = int(np.clip(anchor_frame + timing_offset, 0, data["joint_pos"].shape[0] - 1))
            _write_robot_pose(robot, data, frame, origin, device)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim.get_physics_dt())
            foot_pos = robot.data.body_pos_w[0, foot_body_ids[expected_channel]].clone()
            foot_quat = robot.data.body_quat_w[0, foot_body_ids[expected_channel]].clone()
            anchor_w = torch.as_tensor(data["ball_pos_w"][anchor_frame], dtype=torch.float32, device=device) + origin
            for long_offset in args_cli.long_offsets:
                for lat_offset in args_cli.lat_offsets:
                    for z_offset in args_cli.z_offsets:
                        local_offset = torch.tensor([long_offset, lat_offset, z_offset], dtype=torch.float32, device=device)
                        ball_pos = anchor_w + quat_apply(foot_quat.unsqueeze(0), local_offset.unsqueeze(0))[0]
                        ball_state = torch.zeros((1, 13), dtype=torch.float32, device=device)
                        ball_state[:, :3] = ball_pos
                        ball_state[:, 3] = 1.0
                        ball.write_root_state_to_sim(ball_state)
                        sensor.reset(torch.tensor([0], dtype=torch.long, device=device))
                        max_force = torch.zeros(2, dtype=torch.float32, device=device)
                        max_unfiltered = 0.0
                        for _ in range(max(1, args_cli.physics_steps)):
                            _write_robot_pose(robot, data, frame, origin, device)
                            scene.write_data_to_sim()
                            sim.step(render=False)
                            scene.update(sim.get_physics_dt())
                            matrix = sensor.data.force_matrix_w
                            if isinstance(matrix, torch.Tensor) and matrix.ndim == 4:
                                max_force = torch.maximum(max_force, torch.linalg.vector_norm(matrix[0, 0], dim=-1))
                            max_unfiltered = max(
                                max_unfiltered,
                                float(torch.linalg.vector_norm(sensor.data.net_forces_w[0, 0]).item()),
                            )
                        actual_foot_pos = robot.data.body_pos_w[0, foot_body_ids[expected_channel]]
                        actual_foot_quat = robot.data.body_quat_w[0, foot_body_ids[expected_channel]]
                        right_force, left_force = (float(max_force[0].item()), float(max_force[1].item()))
                        contact_channel = int(torch.argmax(max_force).item()) if float(max_force.max().item()) > args_cli.contact_epsilon else -1
                        expected_contact = contact_channel == expected_channel
                        ball_center_foot = quat_apply_inverse(
                            actual_foot_quat.unsqueeze(0),
                            (ball_pos - actual_foot_pos).unsqueeze(0),
                        )[0]
                        anchor_center_foot = quat_apply_inverse(
                            actual_foot_quat.unsqueeze(0),
                            (anchor_w - actual_foot_pos).unsqueeze(0),
                        )[0]
                        # Dataset surface: 0=inside, 1=outside. Right-foot
                        # inside is +Y; left-foot inside is -Y.
                        inside_sign = 1.0 if expected_foot == 1 else -1.0
                        expected_side_sign = inside_sign if surface == 0 else -inside_sign
                        semantic_side_match = float(ball_center_foot[1].item()) * expected_side_sign >= 0.0
                        rows.append((
                            event_id, anchor_frame, frame, timing_offset, expected_foot, surface,
                            long_offset, lat_offset, z_offset, right_force, left_force,
                            max_unfiltered, contact_channel, expected_contact, semantic_side_match,
                            *ball_pos.detach().cpu().tolist(),
                            *actual_foot_pos.detach().cpu().tolist(),
                            *actual_foot_quat.detach().cpu().tolist(),
                            *ball_center_foot.detach().cpu().tolist(),
                            *anchor_center_foot.detach().cpu().tolist(),
                        ))

    names = (
        "event_id", "anchor_frame", "reference_frame", "timing_offset", "expected_foot", "surface",
        "offset_long", "offset_lat", "offset_z", "right_force", "left_force", "unfiltered_force",
        "contact_channel", "expected_foot_contact", "semantic_side_match",
        "ball_x", "ball_y", "ball_z", "foot_x", "foot_y", "foot_z", "foot_qw", "foot_qx", "foot_qy", "foot_qz",
        "ball_center_foot_x", "ball_center_foot_y", "ball_center_foot_z",
        "anchor_center_foot_x", "anchor_center_foot_y", "anchor_center_foot_z",
    )
    output = args_cli.output or args_cli.motion.with_name(args_cli.motion.stem + "_static_contact_scan.npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = list(zip(*rows, strict=True)) if rows else [[] for _ in names]
    np.savez_compressed(output, **{name: np.asarray(column) for name, column in zip(names, columns, strict=True)})
    expected_hits = sum(bool(row[13]) for row in rows)
    cross_talk = sum(bool(row[13]) and ((row[10] > args_cli.contact_epsilon) if row[4] == 1 else (row[9] > args_cli.contact_epsilon)) for row in rows)
    ground_only_false_positive = sum(row[12] < 0 and row[11] > args_cli.contact_epsilon for row in rows)
    print(f"motion: {args_cli.motion}")
    print(f"samples: {len(rows)}  expected-foot hits: {expected_hits}  cross-talk hits: {cross_talk}")
    print(f"unfiltered-only samples (ground/other): {ground_only_false_positive}")
    print(f"saved: {output}")
    if expected_hits == 0:
        print("RESULT: FAIL — no labelled logical-foot filtered contact in the scan grid.")
        return 1
    if cross_talk > 0:
        print("RESULT: FAIL — left/right filtered channels show cross-talk.")
        return 1
    print("RESULT: PASS — filtered logical-foot contact is reachable; inspect NPZ before defining quality.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
