"""Audit whether motion-only references are compatible with virtual ball sampling.

The video motions do not contain a real ball. This script therefore does not
try to recover ground-truth ball position. Instead it asks a narrower question:

    Does the current environment's virtual ball sampling distribution place the
    ball in positions that the reference swing-foot arc can naturally reach?

This is a data audit tool only. It does not import IsaacLab and does not modify
training code.
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


# Legacy G1 NPZ files in motions/Video do not store body_names. These defaults
# match the observed low foot proxies in current Video motions. Override them
# from CLI if your converted files use a different body order.
DEFAULT_BODY_INDEX = {
    "pelvis": 0,
    "left_foot": 14,
    "right_foot": 18,
}


@dataclass
class CompatibilityReport:
    motion: str
    frames: int
    kick_leg: str
    kick_frame: int
    kick_end_frame: int
    anchor_index: int
    swing_index: int
    support_index: int
    num_ball_samples: int
    spatial_hit_fraction: float
    timed_hit_fraction: float
    good_contact_fraction: float
    median_min_xy: float
    p25_min_xy: float
    p10_min_xy: float
    best_min_xy: float
    best_min_3d: float
    best_frame: int
    best_frame_delta: int
    best_radius_offset: float
    best_angle_deg: float
    best_foot_height: float
    best_foot_speed: float
    best_closing_speed: float
    support_lat_at_best: float
    support_long_at_best: float
    support_height_at_best: float
    support_speed_at_best: float
    classification: str
    recommended_action: str
    notes: str


def get_motion_files(path: str | list[str] | tuple[str, ...]) -> list[str]:
    paths = [path] if isinstance(path, str) else list(path)
    files: list[str] = []
    for p in paths:
        if os.path.isfile(p):
            files.append(p)
        else:
            files.extend(glob.glob(os.path.join(p, "**", "*.npz"), recursive=True))
    files = sorted(set(files))
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


def body_indices_for_leg(kick_leg: str, args: argparse.Namespace) -> tuple[int, int]:
    left_idx = args.left_foot_index
    right_idx = args.right_foot_index
    if kick_leg == "left":
        return left_idx, right_idx
    return right_idx, left_idx


def infer_ball_samples(
    body_pos: np.ndarray,
    anchor_index: int,
    radius_min: float,
    radius_max: float,
    arc_angle: float,
    ball_height: float,
    radius_samples: int,
    angle_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the virtual ball distribution used by the current env config."""
    first_anchor = body_pos[0, anchor_index]
    last_anchor = body_pos[-1, anchor_index]
    radius_vec = last_anchor[:2] - first_anchor[:2]
    radius = float(np.linalg.norm(radius_vec))
    base_angle = math.atan2(float(radius_vec[1]), float(radius_vec[0])) if radius > 1.0e-8 else 0.0

    if radius_samples <= 1:
        radius_offsets = np.array([(radius_min + radius_max) * 0.5], dtype=np.float32)
    else:
        radius_offsets = np.linspace(radius_min, radius_max, radius_samples, dtype=np.float32)

    if angle_samples <= 1 or abs(arc_angle) < 1.0e-8:
        angle_offsets = np.array([0.0], dtype=np.float32)
    else:
        angle_offsets = np.linspace(-arc_angle, arc_angle, angle_samples, dtype=np.float32)

    balls: list[list[float]] = []
    ball_radius_offsets: list[float] = []
    ball_angle_offsets: list[float] = []
    for ro in radius_offsets:
        sampled_radius = max(0.0, radius + float(ro))
        for ao in angle_offsets:
            theta = base_angle + float(ao)
            xy = first_anchor[:2] + sampled_radius * np.array(
                [math.cos(theta), math.sin(theta)], dtype=np.float32
            )
            balls.append([float(xy[0]), float(xy[1]), float(ball_height)])
            ball_radius_offsets.append(float(ro))
            ball_angle_offsets.append(float(ao))

    return (
        np.asarray(balls, dtype=np.float32),
        np.asarray(ball_radius_offsets, dtype=np.float32),
        np.asarray(ball_angle_offsets, dtype=np.float32),
    )


def local_basis(body_pos: np.ndarray, anchor_index: int) -> tuple[np.ndarray, np.ndarray]:
    first_anchor = body_pos[0, anchor_index]
    last_anchor = body_pos[-1, anchor_index]
    forward = last_anchor[:2] - first_anchor[:2]
    norm = float(np.linalg.norm(forward))
    if norm < 1.0e-8:
        forward = np.array([1.0, 0.0], dtype=np.float32)
    else:
        forward = (forward / norm).astype(np.float32)
    side = np.array([-forward[1], forward[0]], dtype=np.float32)
    return forward, side


def eval_ball_against_swing(
    foot_pos: np.ndarray,
    foot_vel: np.ndarray,
    ball: np.ndarray,
    start_frame: int,
) -> dict[str, float | int]:
    rel = foot_pos - ball[None, :]
    dist_xy = np.linalg.norm(rel[:, :2], axis=-1)
    idx = int(np.argmin(dist_xy))
    ball_to_foot = ball[:2] - foot_pos[idx, :2]
    norm = float(np.linalg.norm(ball_to_foot))
    closing = float(np.dot(foot_vel[idx, :2], ball_to_foot / norm)) if norm > 1.0e-8 else 0.0
    return {
        "min_xy": float(dist_xy[idx]),
        "min_3d": float(np.linalg.norm(rel[idx])),
        "frame": int(start_frame + idx),
        "foot_height": float(foot_pos[idx, 2]),
        "foot_speed": float(np.linalg.norm(foot_vel[idx, :2])),
        "closing_speed": closing,
    }


def classify(
    spatial_fraction: float,
    timed_fraction: float,
    good_fraction: float,
    best_min_xy: float,
    best_frame_delta: int,
    args: argparse.Namespace,
) -> tuple[str, str, list[str]]:
    notes: list[str] = []
    if best_min_xy > args.good_xy:
        return "style_only_or_reject", "do_not_use_for_v3_hard_tracking", ["no_sampled_ball_on_swing_arc"]

    if abs(best_frame_delta) > args.good_frame_delta:
        notes.append("best_spatial_match_far_from_kick_frame")

    if good_fraction >= args.ok_good_fraction:
        return "hard_tracking_ok", "can_use_in_v3_candidate_set", notes

    if spatial_fraction >= args.fixable_spatial_fraction:
        if timed_fraction < args.fixable_timed_fraction:
            notes.append("spatial_overlap_but_timing_weak")
            return "kickframe_or_phase_review", "manual_review_kick_frame_or_phase_window", notes
        notes.append("sampler_overlap_is_narrow")
        return "ball_sampler_too_wide", "use_narrower_or_per_motion_ball_sampling", notes

    return "ball_position_fixable", "per_motion_ball_offset_or_style_only", notes


def analyze_motion(path: str, args: argparse.Namespace) -> CompatibilityReport:
    data = np.load(path, allow_pickle=True)
    body_pos = data["body_pos_w"].astype(np.float32)
    body_vel = data["body_lin_vel_w"].astype(np.float32)
    frames = int(body_pos.shape[0])
    kick_frame = int(_scalar(data, "kick_frame", -1))
    kick_end_frame = int(_scalar(data, "kick_end_frame", -1))
    kick_leg = infer_kick_leg(path, data)
    swing_index, support_index = body_indices_for_leg(kick_leg, args)
    anchor_index = args.anchor_index

    if not (0 <= swing_index < body_pos.shape[1]):
        raise ValueError(f"{path}: swing index {swing_index} out of range for body_pos_w shape {body_pos.shape}")
    if not (0 <= support_index < body_pos.shape[1]):
        raise ValueError(f"{path}: support index {support_index} out of range for body_pos_w shape {body_pos.shape}")
    if not (0 <= anchor_index < body_pos.shape[1]):
        raise ValueError(f"{path}: anchor index {anchor_index} out of range for body_pos_w shape {body_pos.shape}")

    kf = min(max(kick_frame, 0), frames - 1) if kick_frame >= 0 else frames // 2
    start = max(0, kf - args.window_before)
    end = min(frames, kf + args.window_after + 1)
    foot_pos = body_pos[start:end, swing_index]
    foot_vel = body_vel[start:end, swing_index]

    balls, radius_offsets, angle_offsets = infer_ball_samples(
        body_pos=body_pos,
        anchor_index=anchor_index,
        radius_min=args.radius_offset_min,
        radius_max=args.radius_offset_max,
        arc_angle=args.arc_angle,
        ball_height=args.ball_height,
        radius_samples=args.radius_samples,
        angle_samples=args.angle_samples,
    )

    evals = [eval_ball_against_swing(foot_pos, foot_vel, ball, start) for ball in balls]
    min_xy = np.asarray([float(e["min_xy"]) for e in evals], dtype=np.float32)
    frame_delta = np.asarray([int(e["frame"]) - kf for e in evals], dtype=np.int32)
    closing = np.asarray([float(e["closing_speed"]) for e in evals], dtype=np.float32)
    foot_speed = np.asarray([float(e["foot_speed"]) for e in evals], dtype=np.float32)
    foot_height = np.asarray([float(e["foot_height"]) for e in evals], dtype=np.float32)

    spatial_hit = min_xy <= args.good_xy
    timed_hit = spatial_hit & (np.abs(frame_delta) <= args.good_frame_delta)
    good_contact = (
        timed_hit
        & (closing >= args.min_closing_speed)
        & (foot_speed >= args.min_foot_speed)
        & (foot_height >= args.foot_height_min)
        & (foot_height <= args.foot_height_max)
    )

    best_idx = int(np.argmin(min_xy))
    best = evals[best_idx]
    best_frame = int(best["frame"])

    forward, side = local_basis(body_pos, anchor_index)
    side_sign = -1.0 if kick_leg == "left" else 1.0
    support_rel = body_pos[best_frame, support_index, :2] - balls[best_idx, :2]
    support_lat = float(np.dot(support_rel, side) * side_sign)
    support_long = float(np.dot(support_rel, forward))
    support_height = float(body_pos[best_frame, support_index, 2])
    support_speed = float(np.linalg.norm(body_vel[best_frame, support_index, :2]))

    spatial_fraction = float(np.mean(spatial_hit))
    timed_fraction = float(np.mean(timed_hit))
    good_fraction = float(np.mean(good_contact))
    classification, action, notes = classify(
        spatial_fraction,
        timed_fraction,
        good_fraction,
        float(min_xy[best_idx]),
        int(frame_delta[best_idx]),
        args,
    )

    if support_speed > args.support_speed_max:
        notes.append("support_moving_at_best_contact")
    if support_height > args.support_height_max:
        notes.append("support_not_grounded_at_best_contact")

    return CompatibilityReport(
        motion=Path(path).name,
        frames=frames,
        kick_leg=kick_leg,
        kick_frame=kick_frame,
        kick_end_frame=kick_end_frame,
        anchor_index=anchor_index,
        swing_index=swing_index,
        support_index=support_index,
        num_ball_samples=int(len(balls)),
        spatial_hit_fraction=round(spatial_fraction, 4),
        timed_hit_fraction=round(timed_fraction, 4),
        good_contact_fraction=round(good_fraction, 4),
        median_min_xy=round(float(np.median(min_xy)), 4),
        p25_min_xy=round(float(np.percentile(min_xy, 25)), 4),
        p10_min_xy=round(float(np.percentile(min_xy, 10)), 4),
        best_min_xy=round(float(min_xy[best_idx]), 4),
        best_min_3d=round(float(best["min_3d"]), 4),
        best_frame=best_frame,
        best_frame_delta=int(frame_delta[best_idx]),
        best_radius_offset=round(float(radius_offsets[best_idx]), 4),
        best_angle_deg=round(math.degrees(float(angle_offsets[best_idx])), 2),
        best_foot_height=round(float(best["foot_height"]), 4),
        best_foot_speed=round(float(best["foot_speed"]), 4),
        best_closing_speed=round(float(best["closing_speed"]), 4),
        support_lat_at_best=round(support_lat, 4),
        support_long_at_best=round(support_long, 4),
        support_height_at_best=round(support_height, 4),
        support_speed_at_best=round(support_speed, 4),
        classification=classification,
        recommended_action=action,
        notes=";".join(notes),
    )


def print_table(reports: list[CompatibilityReport]):
    header = (
        f"{'Motion':<42} {'leg':>5} {'kf':>4} | "
        f"{'Spat%':>6} {'Time%':>6} {'Good%':>6} | "
        f"{'MedXY':>6} {'P10':>6} {'Best':>6} {'dF':>4} | "
        f"{'rOff':>5} {'ang':>5} {'Cls':>5} {'FSpd':>5} | "
        f"{'class':>22}"
    )
    print(header)
    print("-" * len(header))
    for r in reports:
        print(
            f"{r.motion[:41]:<42} {r.kick_leg:>5} {r.kick_frame:>4} | "
            f"{100.0 * r.spatial_hit_fraction:>5.1f}% "
            f"{100.0 * r.timed_hit_fraction:>5.1f}% "
            f"{100.0 * r.good_contact_fraction:>5.1f}% | "
            f"{r.median_min_xy:>6.3f} {r.p10_min_xy:>6.3f} {r.best_min_xy:>6.3f} "
            f"{r.best_frame_delta:>4} | "
            f"{r.best_radius_offset:>5.2f} {r.best_angle_deg:>5.1f} "
            f"{r.best_closing_speed:>5.2f} {r.best_foot_speed:>5.2f} | "
            f"{r.classification:>22}"
        )


def write_manifest(reports: list[CompatibilityReport], output_path: str):
    manifest: dict[str, list[str]] = {
        "hard_tracking_ok": [],
        "ball_sampler_too_wide": [],
        "ball_position_fixable": [],
        "kickframe_or_phase_review": [],
        "style_only_or_reject": [],
    }
    for r in reports:
        manifest.setdefault(r.classification, []).append(r.motion)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Audit virtual-ball compatibility for motion-only kick references.")
    parser.add_argument("--motion_path", type=str, nargs="+", required=True)
    parser.add_argument("--output_json", type=str, default="logs/motion_data_audit/virtual_ball_compatibility.json")
    parser.add_argument("--output_csv", type=str, default="logs/motion_data_audit/virtual_ball_compatibility.csv")
    parser.add_argument("--output_manifest", type=str, default="logs/motion_data_audit/motion_manifest.json")
    parser.add_argument("--anchor_index", type=int, default=7, help="Raw body_pos_w index used by current env ball sampler.")
    parser.add_argument("--left_foot_index", type=int, default=DEFAULT_BODY_INDEX["left_foot"])
    parser.add_argument("--right_foot_index", type=int, default=DEFAULT_BODY_INDEX["right_foot"])
    parser.add_argument("--window_before", type=int, default=25)
    parser.add_argument("--window_after", type=int, default=35)
    parser.add_argument("--radius_offset_min", type=float, default=0.0)
    parser.add_argument("--radius_offset_max", type=float, default=0.4)
    parser.add_argument("--arc_angle", type=float, default=math.pi / 9.0)
    parser.add_argument("--ball_height", type=float, default=0.11)
    parser.add_argument("--radius_samples", type=int, default=17)
    parser.add_argument("--angle_samples", type=int, default=21)
    parser.add_argument("--good_xy", type=float, default=0.12)
    parser.add_argument("--good_frame_delta", type=int, default=12)
    parser.add_argument("--min_closing_speed", type=float, default=0.3)
    parser.add_argument("--min_foot_speed", type=float, default=1.0)
    parser.add_argument("--foot_height_min", type=float, default=0.02)
    parser.add_argument("--foot_height_max", type=float, default=0.35)
    parser.add_argument("--support_speed_max", type=float, default=0.35)
    parser.add_argument("--support_height_max", type=float, default=0.16)
    parser.add_argument("--ok_good_fraction", type=float, default=0.12)
    parser.add_argument("--fixable_spatial_fraction", type=float, default=0.04)
    parser.add_argument("--fixable_timed_fraction", type=float, default=0.02)
    args = parser.parse_args()

    reports = [analyze_motion(path, args) for path in get_motion_files(args.motion_path)]
    print_table(reports)

    counts: dict[str, int] = {}
    for r in reports:
        counts[r.classification] = counts.get(r.classification, 0) + 1
    print("\nClassification counts:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump([asdict(r) for r in reports], f, indent=2)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(reports[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(r) for r in reports)
    write_manifest(reports, args.output_manifest)

    print(f"[INFO] Saved JSON:     {args.output_json}")
    print(f"[INFO] Saved CSV:      {args.output_csv}")
    print(f"[INFO] Saved manifest: {args.output_manifest}")


if __name__ == "__main__":
    main()
