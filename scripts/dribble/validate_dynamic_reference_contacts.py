#!/usr/bin/env python3
"""Validate labelled ball anchors with a free-root dynamic reference-PD replay.

This is a calibration/diagnostic tool, not a training environment.  The robot
is initialized from the motion shortly before one labelled contact event and
then receives joint reference targets only.  Neither the robot root nor the
ball is teleported after initialization.  Contacts are read from an all-robot-
link PhysX manifold view and cross-checked against filtered logical-foot
channels, so ball-ground force can never count as a foot contact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--motion", type=Path, required=True)
parser.add_argument("--pre_seconds", type=float, default=0.30)
parser.add_argument("--post_seconds", type=float, default=0.10)
parser.add_argument("--physics_dt", type=float, default=0.005)
parser.add_argument("--contact_force_epsilon", type=float, default=1.0)
parser.add_argument("--output", type=Path, default=Path("logs/s2_calibration/dynamic_reference_pd_cases.npz"))
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


LOGICAL_FOOT_NAMES = ("right_ankle_roll_link", "left_ankle_roll_link")
CASES = (
    ("O1_nominal", 0, 0.00, 0.00),
    ("O1_lat_p2cm", 0, 0.02, 0.00),
    ("O1_z_m2cm", 0, 0.00, -0.02),
    ("O1_lat_p2cm_z_m2cm", 0, 0.02, -0.02),
    ("O2_nominal", 1, 0.00, 0.00),
    ("I3_nominal", 2, 0.00, 0.00),
)


def _event_segments(contact: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(contact).reshape(-1) > 0
    starts = np.flatnonzero(mask & np.concatenate(([True], ~mask[:-1])))
    ends = np.flatnonzero(mask & np.concatenate((~mask[1:], [True])))
    return starts, ends


def _quat_angle_error(q: torch.Tensor, q_ref: torch.Tensor) -> float:
    dot = torch.sum(q * q_ref).abs().clamp(0.0, 1.0)
    return float((2.0 * torch.acos(dot)).item())


def _filtered_forces(sensor, device: str) -> torch.Tensor:
    matrix = sensor.data.force_matrix_w
    if not isinstance(matrix, torch.Tensor) or matrix.ndim != 4 or matrix.shape[2] < 2:
        return torch.zeros((2, 3), dtype=torch.float32, device=device)
    return matrix[0, 0, :2]


def _read_exact_contacts(view, sensor_index: int, body_names: list[str], dt: float) -> list[dict]:
    forces, points, normals, separations, counts, starts = view.get_contact_data(dt=dt)
    rows: list[dict] = []
    for filter_index, body_name in enumerate(body_names):
        count = int(counts[sensor_index, filter_index].item())
        if count <= 0:
            continue
        start = int(starts[sensor_index, filter_index].item())
        sl = slice(start, start + count)
        normal_forces = forces[sl, 0]
        weights = normal_forces.clamp_min(0.0)
        weight_sum = weights.sum()
        if float(weight_sum.item()) > 0.0:
            point = (points[sl] * weights.unsqueeze(-1)).sum(0) / weight_sum
            normal = (normals[sl] * weights.unsqueeze(-1)).sum(0) / weight_sum
        else:
            point = points[sl].mean(0)
            normal = normals[sl].mean(0)
        rows.append({
            "body_index": filter_index,
            "body_name": body_name,
            "force": float(normal_forces.sum().item()),
            "count": count,
            "point": point.detach().cpu().numpy(),
            "normal": normal.detach().cpu().numpy(),
            "min_separation": float(separations[sl, 0].min().item()),
        })
    return rows


def main() -> int:
    required = (
        "fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
        "body_lin_vel_w", "body_ang_vel_w", "ball_pos_w", "dribble_cg_contact",
        "dribble_cg_foot", "dribble_cg_surface",
    )
    with np.load(args_cli.motion, allow_pickle=True) as source:
        missing = [key for key in required if key not in source.files]
        if missing:
            raise ValueError(f"{args_cli.motion}: missing {missing}")
        data = {key: np.asarray(source[key]) for key in required}
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    starts, ends = _event_segments(data["dribble_cg_contact"])
    if starts.size < 3:
        raise ValueError(f"need at least O1/O2/I3 events, found {starts.size}")

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(device=args_cli.device, dt=args_cli.physics_dt)
    )
    scene_cfg = G1FlatSoccerSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene_cfg.soccer_ball_contact.filter_prim_paths_expr = [
        f"{{ENV_REGEX_NS}}/Robot/{name}" for name in LOGICAL_FOOT_NAMES
    ]
    scene_cfg.soccer_ball_contact.history_length = 1
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    robot = scene["robot"]
    ball = scene["soccer_ball"]
    sensor = scene["soccer_ball_contact"]
    device = sim.device
    origin = scene.env_origins[0]

    logical_ids, logical_names = robot.find_bodies(list(LOGICAL_FOOT_NAMES), preserve_order=True)
    if list(logical_names) != list(LOGICAL_FOOT_NAMES):
        raise RuntimeError(f"logical feet did not resolve: {logical_names}")
    all_body_names = list(robot.body_names)
    physics_view = getattr(sensor, "_physics_sim_view", None)
    if physics_view is None:
        raise RuntimeError("ball ContactSensor has no initialized PhysX view")
    exact_view = physics_view.create_rigid_contact_view(
        "/World/envs/env_*/SoccerBall",
        filter_patterns=[f"/World/envs/env_*/Robot/{name}" for name in all_body_names],
        max_contact_data_count=max(64, len(all_body_names) * 8),
    )
    sensor_paths = list(exact_view.sensor_paths)
    sensor_index = next((i for i, path in enumerate(sensor_paths) if "/env_0/" in path), 0)
    if int(exact_view.filter_count) != len(all_body_names):
        raise RuntimeError(
            f"all-link view expected {len(all_body_names)} filters, got {exact_view.filter_count}"
        )
    resolved_filter_paths = np.asarray(exact_view.filter_paths)[sensor_index]
    resolved_body_names = [str(path).rsplit("/", 1)[-1] for path in resolved_filter_paths]
    if set(resolved_body_names) != set(all_body_names):
        raise RuntimeError(
            "all-link PhysX filters do not match robot bodies: "
            f"missing={sorted(set(all_body_names) - set(resolved_body_names))}, "
            f"extra={sorted(set(resolved_body_names) - set(all_body_names))}"
        )

    # The motion body order is the same robot body order used by tracking.
    if data["body_pos_w"].shape[1] != len(all_body_names):
        raise RuntimeError(
            f"motion has {data['body_pos_w'].shape[1]} bodies, robot has {len(all_body_names)}"
        )
    physics_substeps = max(1, int(round((1.0 / fps) / args_cli.physics_dt)))
    env_ids = torch.tensor([0], dtype=torch.long, device=device)
    results: list[dict] = []

    for case_name, event_id, lateral_offset, z_offset in CASES:
        anchor_frame = int(starts[event_id])
        segment_end = int(ends[event_id])
        start_frame = max(0, anchor_frame - int(round(args_cli.pre_seconds * fps)))
        final_frame = min(
            data["joint_pos"].shape[0] - 1,
            segment_end + int(round(args_cli.post_seconds * fps)),
        )
        expected_foot = int(data["dribble_cg_foot"][anchor_frame])
        surface = int(data["dribble_cg_surface"][anchor_frame])
        expected_channel = 0 if expected_foot == 1 else 1  # dataset: right=1, left=0
        expected_body_id = int(logical_ids[expected_channel])
        expected_filter_index = resolved_body_names.index(LOGICAL_FOOT_NAMES[expected_channel])

        root_state = torch.zeros((1, 13), dtype=torch.float32, device=device)
        root_state[0, :3] = torch.as_tensor(data["body_pos_w"][start_frame, 0], device=device) + origin
        root_state[0, 3:7] = torch.as_tensor(data["body_quat_w"][start_frame, 0], device=device)
        root_state[0, 7:10] = torch.as_tensor(data["body_lin_vel_w"][start_frame, 0], device=device)
        root_state[0, 10:13] = torch.as_tensor(data["body_ang_vel_w"][start_frame, 0], device=device)
        robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        joint_pos = torch.as_tensor(data["joint_pos"][start_frame], dtype=torch.float32, device=device).unsqueeze(0)
        joint_vel = torch.as_tensor(data["joint_vel"][start_frame], dtype=torch.float32, device=device).unsqueeze(0)
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        anchor_w = torch.as_tensor(data["ball_pos_w"][anchor_frame], dtype=torch.float32, device=device) + origin
        ref_foot_quat = torch.as_tensor(
            data["body_quat_w"][anchor_frame, expected_body_id], dtype=torch.float32, device=device
        )
        local_offset = torch.tensor([0.0, lateral_offset, z_offset], dtype=torch.float32, device=device)
        initial_ball_pos = anchor_w + quat_apply(ref_foot_quat.unsqueeze(0), local_offset.unsqueeze(0))[0]
        ball_state = torch.zeros((1, 13), dtype=torch.float32, device=device)
        ball_state[0, :3] = initial_ball_pos
        ball_state[0, 3] = 1.0
        ball.write_root_state_to_sim(ball_state, env_ids=env_ids)
        sensor.reset(env_ids)

        contact_active = False
        onset_frame = -1
        end_frame = -1
        duration_substeps = 0
        expected_impulse = 0.0
        max_expected_force = 0.0
        primary_link = ""
        primary_link_force = 0.0
        contact_point = np.full(3, np.nan, dtype=np.float32)
        contact_normal = np.full(3, np.nan, dtype=np.float32)
        min_separation = np.nan
        onset_ball_foot = np.full(3, np.nan, dtype=np.float32)
        onset_rel_vel_foot = np.full(3, np.nan, dtype=np.float32)
        onset_root_pos_error = np.nan
        onset_root_ori_error = np.nan
        onset_joint_rms_error = np.nan
        onset_foot_pos_error = np.nan
        precontact_ball_displacement = np.nan
        min_foot_ball_center_distance = np.inf
        min_distance_frame = -1
        min_distance_ball_foot = np.full(3, np.nan, dtype=np.float32)
        min_distance_root_pos_error = np.nan
        min_distance_joint_rms_error = np.nan
        min_distance_foot_pos_error = np.nan
        anchor_root_pos_error = np.nan
        anchor_joint_rms_error = np.nan
        anchor_foot_pos_error = np.nan
        anchor_ball_foot = np.full(3, np.nan, dtype=np.float32)
        right_peak = 0.0
        left_peak = 0.0
        any_other_robot_contact = False

        for frame in range(start_frame, final_frame + 1):
            target_pos = torch.as_tensor(data["joint_pos"][frame], dtype=torch.float32, device=device).unsqueeze(0)
            target_vel = torch.as_tensor(data["joint_vel"][frame], dtype=torch.float32, device=device).unsqueeze(0)
            robot.set_joint_position_target(target_pos, env_ids=env_ids)
            robot.set_joint_velocity_target(target_vel, env_ids=env_ids)
            for _substep in range(physics_substeps):
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(args_cli.physics_dt)
                filtered = _filtered_forces(sensor, device)
                magnitudes = torch.linalg.vector_norm(filtered, dim=-1)
                right_peak = max(right_peak, float(magnitudes[0].item()))
                left_peak = max(left_peak, float(magnitudes[1].item()))
                exact = _read_exact_contacts(exact_view, sensor_index, resolved_body_names, args_cli.physics_dt)
                expected_rows = [row for row in exact if row["body_index"] == expected_filter_index]
                any_other_robot_contact |= any(row["body_index"] != expected_filter_index for row in exact)
                foot_pos = robot.data.body_pos_w[0, expected_body_id]
                foot_quat = robot.data.body_quat_w[0, expected_body_id]
                ball_pos = ball.data.root_pos_w[0]
                ball_center_foot = quat_apply_inverse(foot_quat.unsqueeze(0), (ball_pos - foot_pos).unsqueeze(0))[0]
                center_distance = float(torch.linalg.vector_norm(ball_pos - foot_pos).item())
                ref_root_pos_now = torch.as_tensor(data["body_pos_w"][frame, 0], device=device) + origin
                ref_foot_pos_now = torch.as_tensor(data["body_pos_w"][frame, expected_body_id], device=device) + origin
                root_pos_error_now = float(
                    torch.linalg.vector_norm(robot.data.root_pos_w[0] - ref_root_pos_now).item()
                )
                joint_rms_error_now = float(
                    torch.sqrt(torch.mean((robot.data.joint_pos[0] - target_pos[0]) ** 2)).item()
                )
                foot_pos_error_now = float(torch.linalg.vector_norm(foot_pos - ref_foot_pos_now).item())
                if center_distance < min_foot_ball_center_distance:
                    min_foot_ball_center_distance = center_distance
                    min_distance_frame = frame
                    min_distance_ball_foot = ball_center_foot.detach().cpu().numpy()
                    min_distance_root_pos_error = root_pos_error_now
                    min_distance_joint_rms_error = joint_rms_error_now
                    min_distance_foot_pos_error = foot_pos_error_now
                if frame == anchor_frame:
                    anchor_root_pos_error = root_pos_error_now
                    anchor_joint_rms_error = joint_rms_error_now
                    anchor_foot_pos_error = foot_pos_error_now
                    anchor_ball_foot = ball_center_foot.detach().cpu().numpy()
                has_expected = bool(expected_rows) and float(magnitudes[expected_channel].item()) > args_cli.contact_force_epsilon
                if has_expected:
                    row = max(expected_rows, key=lambda item: item["force"])
                    expected_impulse += row["force"] * args_cli.physics_dt
                    max_expected_force = max(max_expected_force, row["force"])
                    if row["force"] > primary_link_force:
                        primary_link_force = row["force"]
                        primary_link = row["body_name"]
                        contact_point = row["point"]
                        contact_normal = row["normal"]
                        min_separation = row["min_separation"]
                    if not contact_active:
                        contact_active = True
                        onset_frame = frame
                        onset_ball_foot = ball_center_foot.detach().cpu().numpy()
                        ball_vel = ball.data.root_lin_vel_w[0]
                        foot_vel = robot.data.body_lin_vel_w[0, expected_body_id]
                        onset_rel_vel_foot = quat_apply_inverse(
                            foot_quat.unsqueeze(0), (ball_vel - foot_vel).unsqueeze(0)
                        )[0].detach().cpu().numpy()
                        precontact_ball_displacement = float(
                            torch.linalg.vector_norm(ball_pos - initial_ball_pos).item()
                        )
                        ref_root_pos = torch.as_tensor(data["body_pos_w"][frame, 0], device=device) + origin
                        ref_root_quat = torch.as_tensor(data["body_quat_w"][frame, 0], device=device)
                        onset_root_pos_error = float(
                            torch.linalg.vector_norm(robot.data.root_pos_w[0] - ref_root_pos).item()
                        )
                        onset_root_ori_error = _quat_angle_error(robot.data.root_quat_w[0], ref_root_quat)
                        onset_joint_rms_error = float(torch.sqrt(torch.mean((robot.data.joint_pos[0] - target_pos[0]) ** 2)).item())
                        ref_foot_pos = torch.as_tensor(data["body_pos_w"][frame, expected_body_id], device=device) + origin
                        onset_foot_pos_error = float(torch.linalg.vector_norm(foot_pos - ref_foot_pos).item())
                    duration_substeps += 1
                    end_frame = frame
                elif contact_active:
                    # Aggregate only the first continuous expected-foot event.
                    contact_active = False

        side_sign = float(onset_ball_foot[1]) if np.isfinite(onset_ball_foot[1]) else np.nan
        inside_sign = 1.0 if expected_foot == 1 else -1.0
        expected_sign = inside_sign if surface == 0 else -inside_sign
        side_match = bool(np.isfinite(side_sign) and side_sign * expected_sign >= 0.0)
        timing_error = float(onset_frame - anchor_frame) if onset_frame >= 0 else np.nan
        results.append({
            "case_name": case_name, "event_id": event_id, "anchor_frame": anchor_frame,
            "segment_end": segment_end, "expected_foot": expected_foot, "surface": surface,
            "offset_lateral": lateral_offset, "offset_z": z_offset,
            "contact": onset_frame >= 0, "onset_frame": onset_frame, "end_frame": end_frame,
            "timing_error": timing_error, "duration_seconds": duration_substeps * args_cli.physics_dt,
            "primary_link": primary_link, "right_peak_force": right_peak, "left_peak_force": left_peak,
            "max_expected_force": max_expected_force, "expected_impulse": expected_impulse,
            "side_match": side_match, "other_robot_contact": any_other_robot_contact,
            "contact_point_w": contact_point, "contact_normal_w": contact_normal,
            "min_separation": min_separation, "onset_ball_center_foot": onset_ball_foot,
            "onset_relative_velocity_foot": onset_rel_vel_foot,
            "min_foot_ball_center_distance": min_foot_ball_center_distance,
            "min_distance_frame": min_distance_frame,
            "min_distance_ball_center_foot": min_distance_ball_foot,
            "min_distance_root_pos_error": min_distance_root_pos_error,
            "min_distance_joint_rms_error": min_distance_joint_rms_error,
            "min_distance_foot_pos_error": min_distance_foot_pos_error,
            "anchor_root_pos_error": anchor_root_pos_error,
            "anchor_joint_rms_error": anchor_joint_rms_error,
            "anchor_foot_pos_error": anchor_foot_pos_error,
            "anchor_ball_center_foot": anchor_ball_foot,
            "final_ball_displacement": float(
                torch.linalg.vector_norm(ball.data.root_pos_w[0] - initial_ball_pos).item()
            ),
            "precontact_ball_displacement": precontact_ball_displacement,
            "root_pos_error": onset_root_pos_error, "root_ori_error": onset_root_ori_error,
            "joint_rms_error": onset_joint_rms_error, "foot_pos_error": onset_foot_pos_error,
        })

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    keys = tuple(results[0])
    np.savez_compressed(args_cli.output, **{key: np.asarray([row[key] for row in results]) for key in keys})
    print(f"motion: {args_cli.motion}")
    print(f"reference fps: {fps:g}  physics substeps/control: {physics_substeps}")
    print("case                         hit  onset  dt(fr)  link                         peak(N) impulse(Ns) side")
    for row in results:
        print(
            f"{row['case_name']:<28} {int(row['contact']):>3} {row['onset_frame']:>6} "
            f"{row['timing_error']:>7.1f}  {row['primary_link']:<28} "
            f"{row['max_expected_force']:>7.2f} {row['expected_impulse']:>11.3f} "
            f"{int(row['side_match']):>4}"
        )
    print(f"saved: {args_cli.output}")
    print("tracking audit (at labelled anchor frame):")
    for row in results:
        print(
            f"  {row['case_name']:<28} min_dist={row['min_foot_ball_center_distance']:.3f}m "
            f"@f{row['min_distance_frame']}  anchor_err="
            f"(root={row['anchor_root_pos_error']:.3f}m, joint={row['anchor_joint_rms_error']:.3f}rad, "
            f"foot={row['anchor_foot_pos_error']:.3f}m) "
            f"ball_drift={row['final_ball_displacement']:.3f}m"
        )
    print("NOTE: velocity and impulse are diagnostics only; they are not reference targets or rewards.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
