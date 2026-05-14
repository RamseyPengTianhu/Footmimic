#!/usr/bin/env python3
"""Dribble label helper for RSL-RL motion datasets.

This tool provides two subcommands:

1) template
   Generate a JSON template for manual annotation from a motion directory.

2) apply
   Apply labels from the JSON file back into each motion ``.npz``.

Written fields (compatible with current loader):
  - kick_leg: "right" | "left"
  - kick_frame: int
  - kick_end_frame: int

Contact-graph (for ``Tracking-CG-G1-Dribbling-RNN-v0``):
  - contact_segments: list of {start, end, foot} with foot in {left, right}
  - apply writes ``dribble_cg_contact`` / ``dribble_cg_foot`` per frame

3) autolabel
   Semi-automatic pre-labeling from foot–ball proximity (XY by default).
   Requires ``ball_pos_w`` either inside motion ``.npz`` or in a sidecar file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _collect_npz_files(motion_path: Path) -> list[Path]:
    if motion_path.is_file() and motion_path.suffix == ".npz":
        return [motion_path]
    if motion_path.is_dir():
        files = sorted(motion_path.glob("*.npz"))
        if files:
            return files
    raise ValueError(f"No .npz files found at: {motion_path}")


def _default_entry(path: Path) -> dict[str, Any]:
    return {
        "kick_leg": "right",
        "kick_frame": -1,
        "kick_end_frame": -1,
        # CG: [{"start": 10, "end": 40, "foot": "right"}, ...]
        "contact_segments": [],
        "notes": "",
        "file": path.name,
    }


def cmd_template(args: argparse.Namespace) -> None:
    motion_path = Path(args.motion_path)
    files = _collect_npz_files(motion_path)

    root: dict[str, Any] = {
        "version": 1,
        "description": "Manual dribble labels for motion files.",
        "labels": {},
    }
    for f in files:
        root["labels"][f.name] = _default_entry(f)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(root, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[OK] Template written: {out} ({len(files)} files)")


def _validate_entry(name: str, entry: dict[str, Any]) -> None:
    leg = str(entry.get("kick_leg", "")).lower().strip()
    if leg not in {"right", "left"}:
        raise ValueError(f"{name}: kick_leg must be 'right' or 'left', got {entry.get('kick_leg')}")

    for key in ["kick_frame", "kick_end_frame"]:
        val = int(entry.get(key, -1))
        if val < -1:
            raise ValueError(f"{name}: {key} must be >= -1, got {val}")

    csegs = entry.get("contact_segments") or []
    if not isinstance(csegs, list):
        raise ValueError(f"{name}: contact_segments must be a list")
    for i, seg in enumerate(csegs):
        if not isinstance(seg, dict):
            raise ValueError(f"{name}: contact_segments[{i}] must be an object")
        foot = str(seg.get("foot", "")).lower().strip()
        if foot not in {"left", "right"}:
            raise ValueError(f"{name}: contact_segments[{i}].foot must be left|right, got {seg.get('foot')}")
        s = int(seg.get("start", -1))
        e = int(seg.get("end", -1))
        if s < 0 or e < s:
            raise ValueError(f"{name}: invalid contact segment at [{i}] start={s}, end={e}")


def _npz_replace(npz_path: Path, updates: dict[str, Any], remove_keys: tuple[str, ...] = ()) -> None:
    with np.load(npz_path, allow_pickle=True) as old:
        payload = {k: old[k] for k in old.files}
    for k in remove_keys:
        payload.pop(k, None)
    payload.update(updates)
    np.savez(npz_path, **payload)


def cmd_apply(args: argparse.Namespace) -> None:
    motion_path = Path(args.motion_path)
    files = _collect_npz_files(motion_path)
    file_map = {p.name: p for p in files}

    label_json = Path(args.label_json)
    data = json.loads(label_json.read_text(encoding="utf-8"))
    labels = data.get("labels", {})
    if not isinstance(labels, dict):
        raise ValueError("label_json must contain object field: labels")

    updated = 0
    missing = []
    for fname, entry_any in labels.items():
        if fname not in file_map:
            missing.append(fname)
            continue
        if not isinstance(entry_any, dict):
            raise ValueError(f"{fname}: label entry must be object")
        entry = entry_any
        _validate_entry(fname, entry)

        kick_leg = str(entry["kick_leg"]).lower().strip()
        kick_frame = int(entry.get("kick_frame", -1))
        kick_end_frame = int(entry.get("kick_end_frame", -1))

        updates: dict[str, Any] = {
            "kick_leg": np.asarray(kick_leg),
            "kick_frame": np.asarray(kick_frame, dtype=np.int32),
            "kick_end_frame": np.asarray(kick_end_frame, dtype=np.int32),
        }

        with np.load(file_map[fname], allow_pickle=True) as d0:
            T = int(np.asarray(d0["joint_pos"]).shape[0])
        csegs = entry.get("contact_segments") or []
        if isinstance(csegs, list) and len(csegs) > 0:
            cc = np.zeros(T, dtype=np.int8)
            cf = np.full(T, -1, dtype=np.int8)
            for seg in csegs:
                s = max(0, min(T - 1, int(seg["start"])))
                e = max(0, min(T - 1, int(seg["end"])))
                if e < s:
                    continue
                foot = str(seg["foot"]).lower().strip()
                kid = 0 if foot == "left" else 1
                cc[s : e + 1] = 1
                cf[s : e + 1] = kid
            updates["dribble_cg_contact"] = cc
            updates["dribble_cg_foot"] = cf

        _npz_replace(
            file_map[fname],
            updates,
            remove_keys=("dribble_phase_starts", "dribble_phase_ends", "dribble_phase_types"),
        )
        updated += 1

    if missing:
        print(f"[WARN] {len(missing)} labeled files not found under motion_path.")
        for n in missing[:10]:
            print(f"  - {n}")

    print(f"[OK] Updated {updated} motion files.")


def _run_length_smooth(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Remove short True/False runs to reduce flickering labels."""
    if min_len <= 1 or mask.size == 0:
        return mask.astype(bool)
    out = mask.astype(bool).copy()
    n = out.size
    i = 0
    while i < n:
        j = i + 1
        while j < n and out[j] == out[i]:
            j += 1
        if j - i < min_len:
            fill = out[i - 1] if i > 0 else (out[j] if j < n else out[i])
            out[i:j] = fill
        i = j
    return out


def _load_ball_pos(npz_path: Path, sidecar_path: Path | None) -> np.ndarray:
    with np.load(npz_path, allow_pickle=True) as d:
        if "ball_pos_w" in d.files:
            return np.asarray(d["ball_pos_w"], dtype=np.float32)

    if sidecar_path is None or not sidecar_path.exists():
        raise ValueError(
            f"{npz_path.name}: ball_pos_w not found in motion npz and no valid sidecar provided."
        )
    with np.load(sidecar_path, allow_pickle=True) as d:
        if "ball_pos_w" not in d.files:
            raise ValueError(f"{sidecar_path}: missing ball_pos_w")
        return np.asarray(d["ball_pos_w"], dtype=np.float32)


def _contact_segments_from_mask(
    contact: np.ndarray, foot_side: np.ndarray, default_foot: str
) -> list[dict[str, Any]]:
    """Merge consecutive contact frames into ``contact_segments`` with foot side."""
    n = int(contact.size)
    if n == 0:
        return []
    df = str(default_foot).lower().strip()
    out: list[dict[str, Any]] = []
    i = 0
    c = contact.astype(bool)
    while i < n:
        if not c[i]:
            i += 1
            continue
        j = i + 1
        while j < n and c[j]:
            j += 1
        votes = foot_side[i:j]
        cnt_r = int(np.sum(votes == 1))
        cnt_l = int(np.sum(votes == -1))
        if cnt_r == 0 and cnt_l == 0:
            foot = "right" if df != "left" else "left"
        elif cnt_r > cnt_l:
            foot = "right"
        elif cnt_l > cnt_r:
            foot = "left"
        else:
            foot = "right" if df != "left" else "left"
        out.append({"start": i, "end": j - 1, "foot": foot})
        i = j
    return out


def cmd_autolabel(args: argparse.Namespace) -> None:
    motion_path = Path(args.motion_path)
    files = _collect_npz_files(motion_path)
    sidecar_dir = Path(args.ball_trace_path) if args.ball_trace_path else None

    labels_root: dict[str, Any]
    out = Path(args.output)
    if args.base_json and Path(args.base_json).exists():
        labels_root = json.loads(Path(args.base_json).read_text(encoding="utf-8"))
        if "labels" not in labels_root or not isinstance(labels_root["labels"], dict):
            labels_root["labels"] = {}
        labels_root.pop("phase_type_map", None)
    else:
        labels_root = {
            "version": 1,
            "description": "Auto prelabels for dribble contact; please review manually.",
            "labels": {},
        }

    updated = 0
    for f in files:
        with np.load(f, allow_pickle=True) as d:
            body_pos = np.asarray(d["body_pos_w"], dtype=np.float32)  # [T, B, 3]
        T, B, _ = body_pos.shape
        if args.right_foot_index >= B or args.left_foot_index >= B:
            raise ValueError(
                f"{f.name}: foot index out of range for body_pos_w shape {body_pos.shape}; "
                f"got right={args.right_foot_index}, left={args.left_foot_index}"
            )

        sidecar = None
        if sidecar_dir is not None:
            cand = sidecar_dir / f.name
            sidecar = cand if cand.exists() else None
        ball_pos = _load_ball_pos(f, sidecar)
        if ball_pos.shape[0] != T:
            n = min(T, ball_pos.shape[0])
            body_pos = body_pos[:n]
            ball_pos = ball_pos[:n]
            T = n

        r_foot = body_pos[:, args.right_foot_index, :3]
        l_foot = body_pos[:, args.left_foot_index, :3]
        ball = ball_pos[:, :3]

        # Pseudo sidecar places the ball on the ground (low z) while ankles sit ~0.7–0.9 m up.
        # 3D distance is then dominated by vertical offset (~0.75 m), forcing absurd thresholds (~0.7 m).
        if getattr(args, "contact_dist_3d", False):
            dist_r = np.linalg.norm(r_foot - ball, axis=1)
            dist_l = np.linalg.norm(l_foot - ball, axis=1)
        else:
            dist_r = np.linalg.norm(r_foot[:, :2] - ball[:, :2], axis=1)
            dist_l = np.linalg.norm(l_foot[:, :2] - ball[:, :2], axis=1)
        min_dist = np.minimum(dist_r, dist_l)
        contact = min_dist <= args.contact_dist_threshold
        contact = _run_length_smooth(contact, args.min_contact_run)

        # Foot-side votes: ONLY where ball is realistically near a foot (tight radius).
        # If ``contact_dist_threshold`` is huge (e.g. 0.9 m), naive voting biases one side forever.
        foot_vote_max = float(args.foot_vote_max_dist)
        plausible_touch = min_dist <= foot_vote_max

        default_side = 1 if str(args.default_foot).lower().strip() == "right" else -1

        # +1 right, -1 left, 0 unknown
        foot_side = np.zeros(T, dtype=np.int32)
        side_pick = np.where(dist_r <= dist_l, 1, -1)
        vote_mask = contact & plausible_touch
        foot_side[vote_mask] = side_pick[vote_mask]

        # Global kick_leg from plausible-touch votes only; ties -> default_foot.
        cnt_r = int(np.sum(foot_side == 1))
        cnt_l = int(np.sum(foot_side == -1))
        if cnt_r > cnt_l:
            kick_leg = "right"
        elif cnt_l > cnt_r:
            kick_leg = "left"
        else:
            kick_leg = "right" if default_side == 1 else "left"

        contact_idx = np.flatnonzero(contact)
        kf = int(contact_idx[0]) if contact_idx.size > 0 else -1
        kef = int(contact_idx[-1]) if contact_idx.size > 0 else -1

        ent = labels_root["labels"].get(f.name, _default_entry(f))
        ent["kick_leg"] = kick_leg
        ent["kick_frame"] = kf
        ent["kick_end_frame"] = kef
        ent.pop("phases", None)
        ent["contact_segments"] = _contact_segments_from_mask(contact, foot_side, str(args.default_foot))
        note_extra = ""
        warn_large = 0.55 if not getattr(args, "contact_dist_3d", False) else 0.45
        if float(args.contact_dist_threshold) > warn_large:
            note_extra += " WARN:large_contact_dist hurts label quality;"
        contact_mode = "3d" if getattr(args, "contact_dist_3d", False) else "xy"
        ent["notes"] = (
            "AUTO_PRELABEL: review contact_segments/foot. "
            f"contact_dist={args.contact_dist_threshold} ({contact_mode}), "
            f"foot_vote_max={foot_vote_max}, default_foot={args.default_foot}.{note_extra}"
        )
        ent["file"] = f.name
        labels_root["labels"][f.name] = ent
        updated += 1

    labels_root.pop("phase_type_map", None)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(labels_root, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[OK] Auto-labeled {updated} files -> {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dribble label template/apply tool")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_t = sub.add_parser("template", help="Generate label JSON template from motion path")
    p_t.add_argument("--motion_path", type=str, required=True, help="Directory or single .npz")
    p_t.add_argument("--output", type=str, required=True, help="Output JSON path")
    p_t.set_defaults(func=cmd_template)

    p_a = sub.add_parser("apply", help="Apply label JSON back to .npz files")
    p_a.add_argument("--motion_path", type=str, required=True, help="Directory or single .npz")
    p_a.add_argument("--label_json", type=str, required=True, help="Input label JSON")
    p_a.set_defaults(func=cmd_apply)

    p_auto = sub.add_parser("autolabel", help="Generate semi-automatic prelabels from kinematics")
    p_auto.add_argument("--motion_path", type=str, required=True, help="Directory or single .npz")
    p_auto.add_argument(
        "--ball_trace_path",
        type=str,
        default=None,
        help="Optional dir with sidecar npz containing ball_pos_w (same filenames).",
    )
    p_auto.add_argument("--output", type=str, required=True, help="Output label JSON path")
    p_auto.add_argument("--base_json", type=str, default=None, help="Optional existing label JSON to merge into")
    p_auto.add_argument("--right_foot_index", type=int, default=6, help="Right ankle index in body_pos_w")
    p_auto.add_argument("--left_foot_index", type=int, default=3, help="Left ankle index in body_pos_w")
    p_auto.add_argument(
        "--contact_dist_3d",
        action="store_true",
        help="Use 3D foot–ball distance. Default: XY only (for ground-height ball_pos_w vs elevated ankles).",
    )
    p_auto.add_argument("--contact_dist_threshold", type=float, default=0.20, help="Foot-ball distance threshold for contact")
    p_auto.add_argument(
        "--foot_vote_max_dist",
        type=float,
        default=0.32,
        help=(
            "Max foot-ball distance to count toward left/right votes. "
            "Keep modest (e.g. 0.25-0.35) even when contact_dist_threshold is large."
        ),
    )
    p_auto.add_argument(
        "--default_foot",
        type=str,
        default="right",
        choices=("right", "left"),
        help="When foot votes tie or absent, label kick_leg / segment foot with this side.",
    )
    p_auto.add_argument("--min_contact_run", type=int, default=3, help="Minimum run length for contact mask smoothing")
    p_auto.set_defaults(func=cmd_autolabel)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

