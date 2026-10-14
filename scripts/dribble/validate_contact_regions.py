#!/usr/bin/env python3
"""CUDA closed-loop validation for frozen foot-local contact regions."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--motion", type=Path, required=True)
parser.add_argument("--regions", type=Path, required=True)
parser.add_argument("--scan", type=Path, required=True)
parser.add_argument("--event_ids", type=int, nargs="+", default=[0, 1])
parser.add_argument("--positive_samples", type=int, default=3)
parser.add_argument("--negative_samples", type=int, default=3)
parser.add_argument("--validate_all_positive", action="store_true")
parser.add_argument("--output_regions", type=Path, default=None)
parser.add_argument("--physics_steps", type=int, default=4)
parser.add_argument("--contact_epsilon", type=float, default=1.0e-6)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import quat_apply

from soccer.robots.g1 import G1_CYLINDER_CFG
from soccer.tasks.tracking.config.g1.soccer_flat_env_cfg import G1FlatSoccerSceneCfg


FOOT_NAMES = ("right_ankle_roll_link", "left_ankle_roll_link")


def _write_robot(robot, motion, frame: int, origin, device: str) -> None:
    root = torch.cat(
        (
            torch.as_tensor(motion["body_pos_w"][frame, 0], device=device).unsqueeze(0) + origin,
            torch.as_tensor(motion["body_quat_w"][frame, 0], device=device).unsqueeze(0),
            torch.zeros((1, 6), dtype=torch.float32, device=device),
        ),
        dim=-1,
    ).to(torch.float32)
    robot.write_root_state_to_sim(root)
    robot.write_joint_state_to_sim(
        torch.as_tensor(motion["joint_pos"][frame], dtype=torch.float32, device=device).unsqueeze(0),
        torch.zeros_like(
            torch.as_tensor(motion["joint_vel"][frame], dtype=torch.float32, device=device)
        ).unsqueeze(0),
    )


def _select_evenly(ids: np.ndarray, count: int) -> np.ndarray:
    if ids.size <= count:
        return ids
    slots = np.linspace(0, ids.size - 1, count).round().astype(int)
    return ids[slots]


def main() -> int:
    with np.load(args_cli.motion) as source:
        motion = {name: np.asarray(source[name]) for name in source.files}
    with np.load(args_cli.regions) as source:
        region = {name: np.asarray(source[name]) for name in source.files}
    with np.load(args_cli.scan) as source:
        scan = {name: np.asarray(source[name]) for name in source.files}
    cover_radius = float(region["cover_radius"].item())
    reproducible_valid = region["valid"].astype(bool).copy()

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(device=args_cli.device, dt=0.005)
    )
    scene_cfg = G1FlatSoccerSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene_cfg.soccer_ball_contact.filter_prim_paths_expr = [
        f"{{ENV_REGEX_NS}}/Robot/{name}" for name in FOOT_NAMES
    ]
    scene_cfg.soccer_ball_contact.history_length = 1
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    robot = scene["robot"]
    ball = scene["soccer_ball"]
    sensor = scene["soccer_ball_contact"]
    origin = scene.env_origins[0]
    device = sim.device
    foot_ids = robot.find_bodies(list(FOOT_NAMES), preserve_order=True)[0]

    failures: list[str] = []
    for event_id in args_cli.event_ids:
        anchor_frame = int(region["anchor_frame"][event_id])
        expected_foot = int(region["expected_foot"][event_id])
        surface = int(region["surface"][event_id])
        channel = 0 if expected_foot == 1 else 1
        valid_slots = np.flatnonzero(region["valid"][event_id])
        positive_slots = (
            valid_slots
            if args_cli.validate_all_positive
            else _select_evenly(valid_slots, args_cli.positive_samples)
        )

        scan_event = scan["event_id"] == event_id
        scan_xyz = np.column_stack(
            [
                scan["ball_center_foot_x"],
                scan["ball_center_foot_y"],
                scan["ball_center_foot_z"],
            ]
        )
        points = region["points"][event_id, valid_slots]
        raw_distance = np.sqrt(
            ((scan_xyz[scan_event, None] - points[None]) ** 2).sum(axis=-1)
        ).min(axis=1)
        # d_G > 1 cm means raw point distance exceeds cover radius + 1 cm.
        negative_local_ids = np.flatnonzero(
            (~scan["expected_foot_contact"][scan_event].astype(bool))
            & scan["semantic_side_match"][scan_event].astype(bool)
            & (raw_distance > cover_radius + 0.01)
        )
        event_scan_ids = np.flatnonzero(scan_event)
        negative_scan_ids = _select_evenly(
            event_scan_ids[negative_local_ids], args_cli.negative_samples
        )
        cases = [
            (
                "positive",
                int(slot),
                region["points"][event_id, slot],
                int(region["timing_offset"][event_id, slot]),
                True,
            )
            for slot in positive_slots
        ]
        cases += [
            (
                "negative",
                int(scan_id),
                scan_xyz[scan_id],
                int(scan["timing_offset"][scan_id]),
                False,
            )
            for scan_id in negative_scan_ids
        ]
        for kind, sample_id, local_point_np, timing_offset, expected_contact in cases:
            frame = int(np.clip(anchor_frame + timing_offset, 0, len(motion["joint_pos"]) - 1))
            _write_robot(robot, motion, frame, origin, device)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim.get_physics_dt())
            foot_pos = robot.data.body_pos_w[0, foot_ids[channel]].clone()
            foot_quat = robot.data.body_quat_w[0, foot_ids[channel]].clone()
            local_point = torch.as_tensor(local_point_np, dtype=torch.float32, device=device)
            world_point = foot_pos + quat_apply(foot_quat.unsqueeze(0), local_point.unsqueeze(0))[0]
            ball_state = torch.zeros((1, 13), dtype=torch.float32, device=device)
            ball_state[:, :3] = world_point
            ball_state[:, 3] = 1.0
            ball.write_root_state_to_sim(ball_state)
            sensor.reset(torch.tensor([0], dtype=torch.long, device=device))
            max_force = torch.zeros(2, dtype=torch.float32, device=device)
            for _ in range(max(1, args_cli.physics_steps)):
                _write_robot(robot, motion, frame, origin, device)
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(sim.get_physics_dt())
                matrix = sensor.data.force_matrix_w
                if isinstance(matrix, torch.Tensor) and matrix.ndim == 4:
                    max_force = torch.maximum(
                        max_force, torch.linalg.vector_norm(matrix[0, 0], dim=-1)
                    )
            actual_contact = float(max_force[channel].item()) > args_cli.contact_epsilon
            other_contact = float(max_force[1 - channel].item()) > args_cli.contact_epsilon
            inside_sign = 1.0 if expected_foot == 1 else -1.0
            expected_side_sign = inside_sign if surface == 0 else -inside_sign
            side_match = float(local_point_np[1]) * expected_side_sign >= 0.0
            d_g = max(
                float(np.linalg.norm(points - local_point_np, axis=1).min()) - cover_radius,
                0.0,
            )
            ok = (
                side_match
                and not other_contact
                and ((d_g <= 1.0e-6 and actual_contact) if expected_contact else (d_g > 0.01 and not actual_contact))
            )
            print(
                f"event={event_id} {kind}={sample_id} timing={timing_offset:+d} "
                f"d_G={d_g:.4f}m force={float(max_force[channel]):.3f}N "
                f"contact={int(actual_contact)} side={int(side_match)} result={'PASS' if ok else 'FAIL'}"
            )
            if not ok:
                failures.append(f"event {event_id} {kind} {sample_id}")
                if kind == "positive":
                    reproducible_valid[event_id, sample_id] = False
    if args_cli.output_regions is not None:
        for event_id in args_cli.event_ids:
            if not np.any(reproducible_valid[event_id]):
                raise RuntimeError(
                    f"event {event_id} has no reproducible points after CUDA validation"
                )
        args_cli.output_regions.parent.mkdir(parents=True, exist_ok=True)
        output = dict(region)
        output["valid"] = reproducible_valid
        output["closed_loop_validated_events"] = np.asarray(
            args_cli.event_ids, dtype=np.int64
        )
        output["closed_loop_source_regions"] = np.asarray(str(args_cli.regions))
        np.savez_compressed(args_cli.output_regions, **output)
        counts = reproducible_valid.sum(axis=1)
        print(f"saved reproducible regions: {args_cli.output_regions}")
        print(
            "points/event after closed loop: "
            + ", ".join(
                f"{event_id}:{int(counts[event_id])}" for event_id in args_cli.event_ids
            )
        )
        # Failed positive candidates are the samples this mode is designed to
        # remove. Negative false positives remain fatal.
        negative_failures = [item for item in failures if " negative " in item]
        if not negative_failures:
            print("RESULT: PASS — reproducible CUDA region mask generated.")
            return 0
    if failures:
        print("RESULT: FAIL — " + ", ".join(failures))
        return 1
    print("RESULT: PASS — region/local/world transform and CUDA filtered contact agree.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
