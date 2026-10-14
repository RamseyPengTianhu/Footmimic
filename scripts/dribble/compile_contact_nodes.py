#!/usr/bin/env python3
"""Compile sparse dribbling annotations into the clean v0 node product.

The compiler is the only place where raw ``ball_pos_w`` and contact labels are
interpreted for the clean Stage-2 task.  It emits immutable phase-conditioned
human--ball nodes in the reference pelvis-yaw frame; the training reward never
re-parses the source arrays.

It also writes a nearest-neighbour phase-ambiguity report.  The report is
diagnostic rather than a hidden hard threshold: if similar reference states
carry different event intent, the caller must add a deployable phase channel
to *both* Stage 1 and Stage 2 and retrain the interface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask).reshape(-1).astype(bool)
    if mask.size == 0:
        return []
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _yaw_xy_inverse(quat_wxyz: np.ndarray, delta_xy: np.ndarray) -> np.ndarray:
    """Apply the inverse yaw of a wxyz quaternion to XY vectors."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    w, x, y, z = np.moveaxis(q, -1, 0)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    c = np.cos(yaw)
    s = np.sin(yaw)
    d = np.asarray(delta_xy, dtype=np.float64)
    return np.stack((c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]), axis=-1)


def _surface_name(value: int) -> str:
    return {0: "inside_instep", 1: "outside_instep"}.get(int(value), "unknown")


def _event_id_per_frame(length: int, segments: list[tuple[int, int]]) -> np.ndarray:
    ids = np.full(length, -1, dtype=np.int32)
    for event_id, (start, end) in enumerate(segments):
        ids[start : end + 1] = event_id
    return ids


def _features(data: np.lib.npyio.NpzFile, pelvis_index: int) -> np.ndarray:
    q = np.asarray(data["joint_pos"], dtype=np.float64)
    dq = np.asarray(data["joint_vel"], dtype=np.float64)
    feats = [q, dq]
    # Root local twist is included when the source motion contains body
    # velocities.  It matches the deployable reference-derived command.
    if "body_lin_vel_w" in data.files and "body_quat_w" in data.files:
        lin = np.asarray(data["body_lin_vel_w"][:, pelvis_index, :3], dtype=np.float64)
        ang = np.asarray(data["body_ang_vel_w"][:, pelvis_index, :3], dtype=np.float64)
        quat = np.asarray(data["body_quat_w"][:, pelvis_index, :4], dtype=np.float64)
        twist_xy = _yaw_xy_inverse(quat, lin[:, :2])
        feats.append(np.concatenate((twist_xy, ang[:, 2:3]), axis=-1))
    out = np.concatenate(feats, axis=-1)
    scale = np.std(out, axis=0)
    scale[scale < 1e-6] = 1.0
    return (out - np.mean(out, axis=0)) / scale


def _compile_one(path: Path, pelvis_index: int, exclude_frames: int) -> tuple[dict, dict]:
    data = np.load(path, allow_pickle=True)
    required = {"body_pos_w", "body_quat_w", "ball_pos_w", "dribble_cg_contact", "dribble_cg_foot", "dribble_cg_surface"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise ValueError(f"{path}: missing required arrays {missing}")
    body_pos = np.asarray(data["body_pos_w"], dtype=np.float64)
    body_quat = np.asarray(data["body_quat_w"], dtype=np.float64)
    ball = np.asarray(data["ball_pos_w"], dtype=np.float64)
    length = min(len(ball), len(body_pos), len(body_quat))
    if not (0 <= pelvis_index < body_pos.shape[1]):
        raise ValueError(f"{path}: pelvis index {pelvis_index} outside body dimension {body_pos.shape[1]}")
    contact = np.asarray(data["dribble_cg_contact"][:length]) > 0
    foot = np.asarray(data["dribble_cg_foot"][:length])
    surface = np.asarray(data["dribble_cg_surface"][:length])
    segments = _segments(contact)
    frame_event = _event_id_per_frame(length, segments)
    pelvis_pos = body_pos[:length, pelvis_index, :3]
    pelvis_quat = body_quat[:length, pelvis_index, :4]
    relative_xy = _yaw_xy_inverse(pelvis_quat, ball[:length, :2] - pelvis_pos[:, :2])
    relative_z = ball[:length, 2] - pelvis_pos[:, 2]

    events = []
    for event_id, (start, end) in enumerate(segments):
        next_id = event_id + 1 if event_id + 1 < len(segments) else -1
        next_start = segments[next_id][0] if next_id >= 0 else -1
        event = {
            "event_id": event_id,
            "start_frame": start,
            "end_frame": end,
            "foot_id": int(foot[start]),
            "surface_id": int(surface[start]),
            "foot": "right" if int(foot[start]) == 1 else "left" if int(foot[start]) == 0 else "unknown",
            "surface": _surface_name(surface[start]),
            "node_xy": relative_xy[start].astype(float).tolist(),
            "node_z": float(relative_z[start]),
            "next_event_id": next_id,
            "next_node_xy": relative_xy[next_start].astype(float).tolist() if next_id >= 0 else None,
            "next_node_z": float(relative_z[next_start]) if next_id >= 0 else None,
            "time_to_next_contact_frames": int(next_start - start) if next_id >= 0 else None,
            "phase": float(start / max(1, length - 1)),
        }
        events.append(event)

    # Nearest-neighbour audit across non-adjacent frames.  Keep all values in
    # the report so a later decision is based on tracking-noise scale, not a
    # silently embedded distance threshold.
    feat = _features(data, pelvis_index)
    nearest = []
    for frame in range(length):
        candidates = np.arange(length)
        keep = np.abs(candidates - frame) > int(exclude_frames)
        candidates = candidates[keep]
        if candidates.size == 0:
            continue
        dist = np.linalg.norm(feat[candidates] - feat[frame], axis=1)
        idx = int(np.argmin(dist))
        nn = int(candidates[idx])
        a = int(frame_event[frame])
        b = int(frame_event[nn])
        intent_a = None if a < 0 else (events[a]["foot"], events[a]["surface"], events[a]["next_event_id"])
        intent_b = None if b < 0 else (events[b]["foot"], events[b]["surface"], events[b]["next_event_id"])
        nearest.append({
            "frame": frame,
            "nearest_frame": nn,
            "feature_rms": float(dist[idx] / np.sqrt(feat.shape[1])),
            "event_id": a,
            "nearest_event_id": b,
            "intent_mismatch": bool(intent_a != intent_b),
        })

    mismatch = [row for row in nearest if row["intent_mismatch"]]
    product = {
        "schema_version": "clean_dribble_v0",
        "source": str(path),
        "fps": int(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data.files else None,
        "num_frames": length,
        "pelvis_index": pelvis_index,
        "node_frame": "reference_pelvis_heading",
        "distance_dimensions": "xy",
        "events": events,
    }
    audit = {
        "source": str(path),
        "num_frames": length,
        "num_events": len(events),
        "exclude_adjacent_frames": exclude_frames,
        "nearest_frame_count": len(nearest),
        "nearest_intent_mismatch_count": len(mismatch),
        "nearest_intent_mismatch_rate": float(len(mismatch) / max(1, len(nearest))),
        "nearest": nearest,
    }
    return product, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="motion NPZ or directory")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pelvis-index", type=int, default=0)
    parser.add_argument("--exclude-adjacent-frames", type=int, default=5)
    args = parser.parse_args()
    paths = [args.input] if args.input.is_file() else sorted(args.input.glob("*.npz"))
    if not paths:
        raise SystemExit(f"no NPZ files found under {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for path in paths:
        product, audit = _compile_one(path, args.pelvis_index, args.exclude_adjacent_frames)
        stem = path.stem
        (args.output_dir / f"{stem}.nodes.json").write_text(json.dumps(product, indent=2) + "\n")
        (args.output_dir / f"{stem}.phase_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        summary.append({k: audit[k] for k in ("source", "num_frames", "num_events", "nearest_intent_mismatch_count", "nearest_intent_mismatch_rate")})
        print(json.dumps(summary[-1]))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
