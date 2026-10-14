#!/usr/bin/env python3
"""Build conservative per-event contact regions from a static CUDA scan."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ground_height", type=float, default=0.0)
    parser.add_argument("--min_ball_clearance", type=float, default=0.105)
    parser.add_argument("--force_percentile", type=float, default=90.0)
    parser.add_argument("--min_expected_foot_force", type=float, default=5.0)
    parser.add_argument("--min_filtered_force_fraction", type=float, default=0.10)
    parser.add_argument("--cover_radius", type=float, default=0.01)
    args = parser.parse_args()

    with np.load(args.scan) as source:
        data = {name: np.asarray(source[name]) for name in source.files}
    required = {
        "event_id", "anchor_frame", "expected_foot", "surface",
        "expected_foot_contact", "semantic_side_match", "ball_z",
        "right_force", "left_force", "timing_offset",
        "ball_center_foot_x", "ball_center_foot_y", "ball_center_foot_z",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"{args.scan}: missing {missing}")

    event_ids = np.unique(data["event_id"]).astype(np.int64)
    if not np.array_equal(event_ids, np.arange(event_ids.size)):
        raise ValueError(f"event ids must be contiguous from zero, got {event_ids}")
    rows: list[np.ndarray] = []
    timings: list[np.ndarray] = []
    sample_ids: list[np.ndarray] = []
    force_limits: list[float] = []
    frames: list[int] = []
    feet: list[int] = []
    surfaces: list[int] = []
    xyz = np.column_stack(
        [
            data["ball_center_foot_x"],
            data["ball_center_foot_y"],
            data["ball_center_foot_z"],
        ]
    ).astype(np.float32)
    force = np.where(
        data["expected_foot"] == 1, data["right_force"], data["left_force"]
    )
    for event_id in event_ids:
        event = data["event_id"] == event_id
        semantic_contact = (
            event
            & data["expected_foot_contact"].astype(bool)
            & data["semantic_side_match"].astype(bool)
        )
        if not np.any(semantic_contact):
            raise ValueError(f"event {event_id} has no semantic filtered contact")
        force_limit = float(
            np.percentile(force[semantic_contact], args.force_percentile)
        )
        clean = (
            semantic_contact
            & ((data["ball_z"] - args.ground_height) >= args.min_ball_clearance)
            & (force <= force_limit)
            & (force >= args.min_expected_foot_force)
            & (
                force / np.maximum(data["unfiltered_force"], 1.0e-6)
                >= args.min_filtered_force_fraction
            )
        )
        ids = np.flatnonzero(clean)
        if ids.size == 0:
            raise ValueError(f"event {event_id} has no points after cleaning")
        rows.append(xyz[ids])
        timings.append(data["timing_offset"][ids].astype(np.int64))
        sample_ids.append(ids.astype(np.int64))
        force_limits.append(force_limit)
        frames.append(int(data["anchor_frame"][np.flatnonzero(event)[0]]))
        feet.append(int(data["expected_foot"][np.flatnonzero(event)[0]]))
        surfaces.append(int(data["surface"][np.flatnonzero(event)[0]]))

    max_points = max(len(points) for points in rows)
    points = np.zeros((len(rows), max_points, 3), dtype=np.float32)
    valid = np.zeros((len(rows), max_points), dtype=bool)
    timing = np.zeros((len(rows), max_points), dtype=np.int64)
    source_sample_id = np.full((len(rows), max_points), -1, dtype=np.int64)
    for event_id, (event_points, event_timing, event_ids_) in enumerate(
        zip(rows, timings, sample_ids, strict=True)
    ):
        count = len(event_points)
        points[event_id, :count] = event_points
        valid[event_id, :count] = True
        timing[event_id, :count] = event_timing
        source_sample_id[event_id, :count] = event_ids_

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        points=points,
        valid=valid,
        timing_offset=timing,
        source_sample_id=source_sample_id,
        anchor_frame=np.asarray(frames, dtype=np.int64),
        expected_foot=np.asarray(feet, dtype=np.int64),
        surface=np.asarray(surfaces, dtype=np.int64),
        force_limit=np.asarray(force_limits, dtype=np.float32),
        cover_radius=np.asarray(args.cover_radius, dtype=np.float32),
        ground_height=np.asarray(args.ground_height, dtype=np.float32),
        min_ball_clearance=np.asarray(args.min_ball_clearance, dtype=np.float32),
        force_percentile=np.asarray(args.force_percentile, dtype=np.float32),
        min_expected_foot_force=np.asarray(
            args.min_expected_foot_force, dtype=np.float32
        ),
        min_filtered_force_fraction=np.asarray(
            args.min_filtered_force_fraction, dtype=np.float32
        ),
        source_scan=np.asarray(str(args.scan)),
    )
    print(f"saved: {args.output}")
    print(
        "points/event: "
        + ", ".join(f"{event_id}:{len(event_points)}" for event_id, event_points in enumerate(rows))
    )
    print(f"cover radius: {args.cover_radius:.4f} m")


if __name__ == "__main__":
    main()
