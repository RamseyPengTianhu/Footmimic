#!/usr/bin/env python3
"""Audit whether a labelled motion's stored ball anchors match current geometry.

This is a data-level preflight check for S2.  It regenerates the synthetic
contact anchors in memory using exactly the production geometry in
``synthesize_dribble_ball_traj.py`` and compares them against ``ball_pos_w``.
It deliberately does not modify the motion file.

The companion Isaac-Lab replay remains the final collision check: a stationary
ball at the first anchor must first contact the reference robot at the first
labelled contact frame, within the chosen tolerance.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np


def _load_synth_module():
    module_path = Path(__file__).with_name("synthesize_dribble_ball_traj.py")
    spec = importlib.util.spec_from_file_location("synthesize_dribble_ball_traj", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True, help="One labelled motion .npz")
    parser.add_argument("--max_anchor_error", type=float, default=1.0e-5)
    parser.add_argument("--inside_foot_contact_x", type=float, default=0.10)
    parser.add_argument("--outside_foot_contact_x", type=float, default=0.16)
    args = parser.parse_args()

    synth = _load_synth_module()
    with np.load(args.motion, allow_pickle=True) as data:
        required = {"body_pos_w", "body_quat_w", "dribble_cg_contact", "dribble_cg_foot", "dribble_cg_surface", "ball_pos_w"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{args.motion}: missing required arrays {missing}")
        body_pos = np.asarray(data["body_pos_w"], dtype=np.float32)
        body_quat = np.asarray(data["body_quat_w"], dtype=np.float32)
        contact = np.asarray(data["dribble_cg_contact"], dtype=np.int8).reshape(-1)
        foot = np.asarray(data["dribble_cg_foot"], dtype=np.int8).reshape(-1)
        surface = np.asarray(data["dribble_cg_surface"], dtype=np.int8).reshape(-1)
        stored_ball = np.asarray(data["ball_pos_w"], dtype=np.float32)

    left_foot, right_foot = synth.resolve_foot_body_indices(body_pos)
    regenerated, _, _, _, _ = synth.synthesize_ball_trajectory(
        body_pos,
        body_quat,
        contact,
        foot,
        surface,
        left_foot_index=left_foot,
        right_foot_index=right_foot,
        anchor_body_index=-1,
        ball_radius=0.11,
        foot_offset_x=0.06,
        foot_offset_y=0.0,
        foot_inner_offset=0.035,
        foot_surface_offset=None,
        foot_collision_radius=0.01,
        foot_contact_x=args.inside_foot_contact_x,
        outside_foot_contact_x=args.outside_foot_contact_x,
        foot_half_width=0.026,
        foot_contact_z=-0.025,
        terminal_forward_limit=0.12,
        init_forward_dist=0.45,
        use_foot_yaw=True,
        traj_mode="hybrid",
    )

    mask = contact > 0
    starts = np.flatnonzero(mask & np.concatenate(([True], ~mask[:-1])))
    if starts.size == 0:
        raise ValueError(f"{args.motion}: no labelled contact events")

    anchor_error = np.linalg.norm(stored_ball[starts, :2] - regenerated[starts, :2], axis=1)
    pre_first = stored_ball[: starts[0]]
    stationary_error = float(np.max(np.abs(pre_first - stored_ball[starts[0]]))) if pre_first.size else 0.0

    print(f"motion: {args.motion}")
    print(f"feet: left={left_foot}, right={right_foot}; events={starts.size}")
    print(f"first contact frame: {int(starts[0])}")
    print(f"pre-first-contact stationary error: {stationary_error:.8f} m")
    print("event  frame  foot  surface  anchor_error_m")
    for event_id, (frame, error) in enumerate(zip(starts, anchor_error, strict=True)):
        print(
            f"{event_id:5d}  {int(frame):5d}  {int(foot[frame]):4d}"
            f"  {int(surface[frame]):7d}  {float(error):.8f}"
        )
    print(f"max anchor error: {float(anchor_error.max()):.8f} m")

    if stationary_error > args.max_anchor_error or bool(np.any(anchor_error > args.max_anchor_error)):
        print("RESULT: FAIL — stored ball anchors do not match the current production geometry.")
        return 1
    print("RESULT: PASS — stored anchors match current production geometry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
