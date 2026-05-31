"""Offline diagnostic for kick reference geometry.

This script checks whether each reference motion is geometrically compatible
with the soccer task's inferred ball placement and kick_frame annotation.

It does not run Isaac Sim. It only reads NPZ motion files and reports whether
the reference swing-foot arc passes near the inferred ball around kick_frame.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


# Legacy G1 NPZ files do not store body_names. These indices match the observed
# raw body_pos_w order for the current Video motions.
BODY_INDEX = {
    "pelvis": 0,
    "left_ankle_roll_link": 14,
    "right_ankle_roll_link": 18,
    "torso_link": 13,
}


@dataclass
class MotionGeometryReport:
    motion: str
    frames: int
    kick_leg: str
    kick_frame: int
    kick_end_frame: int
    nominal_min_xy: float
    nominal_min_3d: float
    nominal_argmin_delta: int
    nominal_foot_height: float
    nominal_foot_speed: float
    nominal_closing_speed: float
    best_min_xy: float
    best_min_3d: float
    best_argmin_delta: int
    best_ball_radius_offset: float
    best_ball_angle_deg: float
    support_lat_at_kf: float
    support_long_at_kf: float
    support_height_at_kf: float
    confidence: str
    notes: str


def get_motion_files(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    files = sorted(glob.glob(os.path.join(path, "**", "*.npz"), recursive=True))
    if not files:
        raise ValueError(f"No .npz files found under {path}")
    return files


def _scalar(data: np.lib.npyio.NpzFile, key: str, default):
    if key not in data.files:
        return default
    try:
        return np.asarray(data[key]).flat[0].item()
    except Exception:
        return default


def infer_kick_leg(path: str, data: np.lib.npyio.NpzFile) -> str:
    raw = _scalar(data, "kick_leg", None)
    if raw is not None:
        label = str(raw).strip().lower()
        if label in {"left", "right"}:
            return label
    stem = Path(path).stem.lower()
    if "_left" in stem:
        return "left"
    return "right"


def infer_ball_candidates(
    body_pos: np.ndarray,
    radius_min: float,
    radius_max: float,
    arc_angle: float,
    ball_height: float,
    radius_samples: int,
    angle_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match MotionCommand._compute_soccer_ball_positions, but offline.

    Returns candidates, radius_offsets, angle_offsets.
    """
    first_anchor = body_pos[0, BODY_INDEX["torso_link"]]
    last_anchor = body_pos[-1, BODY_INDEX["torso_link"]]
    radius_vec = last_anchor[:2] - first_anchor[:2]
    radius = float(np.linalg.norm(radius_vec))
    if radius > 1.0e-8:
        base_angle = math.atan2(float(radius_vec[1]), float(radius_vec[0]))
    else:
        base_angle = 0.0

    if radius_samples <= 1:
        radius_offsets = np.array([(radius_min + radius_max) * 0.5], dtype=np.float32)
    else:
        radius_offsets = np.linspace(radius_min, radius_max, radius_samples, dtype=np.float32)

    if angle_samples <= 1 or abs(arc_angle) < 1.0e-8:
        angle_offsets = np.array([0.0], dtype=np.float32)
    else:
        angle_offsets = np.linspace(-arc_angle, arc_angle, angle_samples, dtype=np.float32)

    candidates = []
    candidate_radius_offsets = []
    candidate_angle_offsets = []
    for ro in radius_offsets:
        r = max(0.0, radius + float(ro))
        for ao in angle_offsets:
            theta = base_angle + float(ao)
            xy = first_anchor[:2] + r * np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
            candidates.append([float(xy[0]), float(xy[1]), float(ball_height)])
            candidate_radius_offsets.append(float(ro))
            candidate_angle_offsets.append(float(ao))
    return (
        np.asarray(candidates, dtype=np.float32),
        np.asarray(candidate_radius_offsets, dtype=np.float32),
        np.asarray(candidate_angle_offsets, dtype=np.float32),
    )


def foot_names_for_leg(kick_leg: str) -> tuple[str, str]:
    if kick_leg == "left":
        return "left_ankle_roll_link", "right_ankle_roll_link"
    return "right_ankle_roll_link", "left_ankle_roll_link"


def _eval_candidate(
    foot_pos: np.ndarray,
    foot_vel: np.ndarray,
    ball: np.ndarray,
    frame_start: int,
    kick_frame: int,
) -> tuple[float, float, int, float, float, float]:
    rel = foot_pos - ball[None, :]
    dist_xy = np.linalg.norm(rel[:, :2], axis=-1)
    idx = int(np.argmin(dist_xy))
    dist_3d = float(np.linalg.norm(rel[idx]))
    ball_to_foot = ball[:2] - foot_pos[idx, :2]
    norm = float(np.linalg.norm(ball_to_foot))
    if norm > 1.0e-8:
        closing = float(np.dot(foot_vel[idx, :2], ball_to_foot / norm))
    else:
        closing = 0.0
    return (
        float(dist_xy[idx]),
        dist_3d,
        int(frame_start + idx - kick_frame),
        float(foot_pos[idx, 2]),
        float(np.linalg.norm(foot_vel[idx, :2])),
        closing,
    )


def analyze_motion(path: str, args: argparse.Namespace) -> MotionGeometryReport:
    data = np.load(path, allow_pickle=True)
    body_pos = data["body_pos_w"].astype(np.float32)
    body_vel = data["body_lin_vel_w"].astype(np.float32)
    frames = int(body_pos.shape[0])
    kick_frame = int(_scalar(data, "kick_frame", -1))
    kick_end_frame = int(_scalar(data, "kick_end_frame", -1))
    kick_leg = infer_kick_leg(path, data)
    kick_foot, support_foot = foot_names_for_leg(kick_leg)
    kick_idx = BODY_INDEX[kick_foot]
    support_idx = BODY_INDEX[support_foot]

    if kick_frame < 0 or kick_frame >= frames:
        kf = min(max(kick_frame, 0), frames - 1)
    else:
        kf = kick_frame
    start = max(0, kf - args.window_before)
    end = min(frames, kf + args.window_after + 1)

    candidates, radius_offsets, angle_offsets = infer_ball_candidates(
        body_pos,
        radius_min=args.radius_offset_min,
        radius_max=args.radius_offset_max,
        arc_angle=args.arc_angle,
        ball_height=args.ball_height,
        radius_samples=args.radius_samples,
        angle_samples=args.angle_samples,
    )
    nominal_ball = infer_ball_candidates(
        body_pos,
        radius_min=(args.radius_offset_min + args.radius_offset_max) * 0.5,
        radius_max=(args.radius_offset_min + args.radius_offset_max) * 0.5,
        arc_angle=0.0,
        ball_height=args.ball_height,
        radius_samples=1,
        angle_samples=1,
    )[0][0]

    foot_window = body_pos[start:end, kick_idx]
    vel_window = body_vel[start:end, kick_idx]
    nominal = _eval_candidate(foot_window, vel_window, nominal_ball, start, kf)

    best_tuple = None
    best_i = 0
    for i, ball in enumerate(candidates):
        cur = _eval_candidate(foot_window, vel_window, ball, start, kf)
        if best_tuple is None or cur[0] < best_tuple[0]:
            best_tuple = cur
            best_i = i
    assert best_tuple is not None

    # Support foot geometry at annotated kick_frame relative to nominal ball.
    first_anchor = body_pos[0, BODY_INDEX["torso_link"]]
    last_anchor = body_pos[-1, BODY_INDEX["torso_link"]]
    kick_dir = last_anchor[:2] - first_anchor[:2]
    norm = np.linalg.norm(kick_dir)
    if norm < 1.0e-8:
        kick_dir = np.array([1.0, 0.0], dtype=np.float32)
    else:
        kick_dir = kick_dir / norm
    side_dir = np.array([-kick_dir[1], kick_dir[0]], dtype=np.float32)
    side_sign = -1.0 if kick_leg == "left" else 1.0
    support_rel = body_pos[kf, support_idx, :2] - nominal_ball[:2]
    support_lat = float(np.dot(support_rel, side_dir) * side_sign)
    support_long = float(np.dot(support_rel, kick_dir))
    support_height = float(body_pos[kf, support_idx, 2])

    notes = []
    if nominal[0] > args.good_xy:
        notes.append("nominal_ball_not_on_swing_arc")
    if abs(nominal[2]) > args.good_frame_delta:
        notes.append("kick_frame_not_near_nominal_argmin")
    if best_tuple[0] > args.good_xy:
        notes.append("no_ball_candidate_close_to_swing_arc")
    if abs(best_tuple[2]) > args.good_frame_delta:
        notes.append("kick_frame_not_near_best_argmin")
    if nominal[5] < args.min_closing_speed:
        notes.append("low_nominal_closing_speed")

    if nominal[0] <= args.good_xy and abs(nominal[2]) <= args.good_frame_delta:
        confidence = "high"
    elif best_tuple[0] <= args.good_xy and abs(best_tuple[2]) <= args.good_frame_delta:
        confidence = "needs_ball_fix"
    elif best_tuple[0] <= args.warn_xy:
        confidence = "medium"
    else:
        confidence = "low"

    return MotionGeometryReport(
        motion=Path(path).name,
        frames=frames,
        kick_leg=kick_leg,
        kick_frame=kick_frame,
        kick_end_frame=kick_end_frame,
        nominal_min_xy=round(nominal[0], 4),
        nominal_min_3d=round(nominal[1], 4),
        nominal_argmin_delta=nominal[2],
        nominal_foot_height=round(nominal[3], 4),
        nominal_foot_speed=round(nominal[4], 4),
        nominal_closing_speed=round(nominal[5], 4),
        best_min_xy=round(best_tuple[0], 4),
        best_min_3d=round(best_tuple[1], 4),
        best_argmin_delta=best_tuple[2],
        best_ball_radius_offset=round(float(radius_offsets[best_i]), 4),
        best_ball_angle_deg=round(math.degrees(float(angle_offsets[best_i])), 2),
        support_lat_at_kf=round(support_lat, 4),
        support_long_at_kf=round(support_long, 4),
        support_height_at_kf=round(support_height, 4),
        confidence=confidence,
        notes=";".join(notes),
    )


def print_table(reports: list[MotionGeometryReport]):
    header = (
        f"{'Motion':<42} {'leg':>5} {'kf':>4} | "
        f"{'NomXY':>6} {'NomD':>5} {'NΔ':>4} {'NCls':>5} | "
        f"{'BestXY':>6} {'BΔ':>4} {'rOff':>5} {'ang':>5} | "
        f"{'sLat':>6} {'sLong':>6} {'conf':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in reports:
        print(
            f"{r.motion[:41]:<42} {r.kick_leg:>5} {r.kick_frame:>4} | "
            f"{r.nominal_min_xy:>6.3f} {r.nominal_min_3d:>5.3f} {r.nominal_argmin_delta:>4} "
            f"{r.nominal_closing_speed:>5.2f} | "
            f"{r.best_min_xy:>6.3f} {r.best_argmin_delta:>4} {r.best_ball_radius_offset:>5.2f} "
            f"{r.best_ball_angle_deg:>5.1f} | "
            f"{r.support_lat_at_kf:>6.3f} {r.support_long_at_kf:>6.3f} {r.confidence:>12}"
        )


def main():
    parser = argparse.ArgumentParser(description="Analyze reference swing-foot and inferred ball geometry.")
    parser.add_argument("--motion_path", type=str, required=True)
    parser.add_argument("--output_json", type=str, default="logs/motion_reference_geometry.json")
    parser.add_argument("--output_csv", type=str, default="logs/motion_reference_geometry.csv")
    parser.add_argument("--window_before", type=int, default=25)
    parser.add_argument("--window_after", type=int, default=35)
    parser.add_argument("--radius_offset_min", type=float, default=0.0)
    parser.add_argument("--radius_offset_max", type=float, default=0.4)
    parser.add_argument("--arc_angle", type=float, default=math.pi / 9.0)
    parser.add_argument("--ball_height", type=float, default=0.11)
    parser.add_argument("--radius_samples", type=int, default=9)
    parser.add_argument("--angle_samples", type=int, default=11)
    parser.add_argument("--good_xy", type=float, default=0.12)
    parser.add_argument("--warn_xy", type=float, default=0.20)
    parser.add_argument("--good_frame_delta", type=int, default=12)
    parser.add_argument("--min_closing_speed", type=float, default=0.5)
    args = parser.parse_args()

    reports = [analyze_motion(path, args) for path in get_motion_files(args.motion_path)]
    print_table(reports)

    counts: dict[str, int] = {}
    for r in reports:
        counts[r.confidence] = counts.get(r.confidence, 0) + 1
    print("\nConfidence counts:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump([asdict(r) for r in reports], f, indent=2)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(reports[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(r) for r in reports)
    print(f"[INFO] Saved JSON: {args.output_json}")
    print(f"[INFO] Saved CSV:  {args.output_csv}")


if __name__ == "__main__":
    main()
