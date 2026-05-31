"""Diagnose where a content-conditioned CVAE motion prior fails.

This script maps per-window reconstruction error back to:

  - source motion file
  - semantic phase
  - window start / center / end frame
  - frame offset from kick_frame

It is offline-only and does not import IsaacLab.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

from train_content_cvae_prior import (
    NUM_PHASES,
    PHASE_NAMES,
    ContentConditionedVAE,
    build_dataset,
    get_motion_files,
    load_filter_manifest,
    motion_pattern_matches,
)


def make_build_args(ckpt: dict, args: argparse.Namespace) -> SimpleNamespace:
    ball_args = ckpt.get("ball_args", {})
    return SimpleNamespace(
        time_scale=float(ckpt.get("time_scale", 40.0)),
        ball_mode=args.ball_mode or ckpt.get("ball_mode", "best_swing"),
        feature_frame=args.feature_frame or ckpt.get("feature_frame", "world"),
        filter_manifest_data=args.filter_manifest_data,
        filter_range_match=args.filter_range_match,
        radius_offset_min=float(ball_args.get("radius_offset_min", 0.0)),
        radius_offset_max=float(ball_args.get("radius_offset_max", 0.4)),
        arc_angle=float(ball_args.get("arc_angle", 0.3490658503988659)),
        ball_height=float(ball_args.get("ball_height", 0.11)),
        radius_samples=int(ball_args.get("radius_samples", 9)),
        angle_samples=int(ball_args.get("angle_samples", 11)),
        ball_search_before=int(ball_args.get("ball_search_before", 25)),
        ball_search_after=int(ball_args.get("ball_search_after", 35)),
    )


@torch.no_grad()
def reconstruction_errors(
    model: ContentConditionedVAE,
    x: torch.Tensor,
    cond: torch.Tensor,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    loader = DataLoader(TensorDataset(x, cond), batch_size=batch_size, shuffle=False)
    chunks = []
    model.eval()
    for xb, cb in loader:
        xb = xb.to(device)
        cb = cb.to(device)
        mu, _ = model.encode(xb, cb)
        recon = model.decode(mu, cb)
        chunks.append(torch.mean((recon - xb) ** 2, dim=-1).cpu())
    return torch.cat(chunks, dim=0)


def fmt(value: float | None, width: int = 8) -> str:
    if value is None:
        return " " * (width - 2) + "--"
    return f"{value:{width}.5f}"


def relpath(path: str) -> str:
    return os.path.relpath(path)


def stem(path: str, max_len: int = 42) -> str:
    name = Path(path).name
    return name[:max_len]


def mean(vals: list[float]) -> float | None:
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def source_split(source: str, val_files: list[str]) -> str:
    for val_file in val_files:
        if motion_pattern_matches(str(val_file), source):
            return "val"
    return "train"


def main():
    parser = argparse.ArgumentParser(description="Diagnose per-motion CVAE reconstruction failures")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--motion_path", type=str, nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--ball_mode", choices=["best_swing", "nominal"], default=None)
    parser.add_argument("--filter_manifest", type=str, default=None)
    parser.add_argument("--filter_range_match", choices=["overlap", "center"], default=None)
    parser.add_argument("--split", choices=["all", "train", "val"], default="all")
    parser.add_argument(
        "--ignore_checkpoint_filter",
        action="store_true",
        help="Diagnose all requested motions instead of the checkpoint's saved filter manifest.",
    )
    parser.add_argument("--feature_frame", choices=["local", "world"], default=None)
    parser.add_argument("--top_windows", type=int, default=30)
    parser.add_argument("--top_motions", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    if args.filter_manifest:
        args.filter_manifest_data = load_filter_manifest(args.filter_manifest)
    elif args.ignore_checkpoint_filter:
        args.filter_manifest_data = {}
    else:
        args.filter_manifest_data = ckpt.get("filter_manifest", {})
    if args.motion_path is None:
        motion_files = ckpt.get("motion_files", [])
        if not motion_files:
            raise ValueError("Checkpoint has no motion_files; pass --motion_path explicitly.")
    else:
        motion_files = get_motion_files(args.motion_path)

    build_args = make_build_args(ckpt, args)
    x, cond_raw, phase, sources, metadata = build_dataset(
        motion_files,
        window_len=int(ckpt["window_len"]),
        stride=args.stride,
        prestrike_len=int(ckpt["prestrike_len"]),
        lower_body_only=bool(ckpt.get("lower_body_only", True)),
        args=build_args,
        return_metadata=True,
    )
    filter_stats = getattr(build_args, "_filter_stats", {})
    x = (x - ckpt["feature_mean"]) / ckpt["feature_std"]
    cond = (cond_raw - ckpt["cond_mean"]) / ckpt["cond_std"]

    model = ContentConditionedVAE(
        input_dim=int(ckpt["input_dim"]),
        cond_dim=int(ckpt["cond_dim"]),
        latent_dim=int(ckpt["latent_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
    ).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])

    errors = reconstruction_errors(model, x, cond, args.batch_size, args.device)
    val_files = [str(path) for path in ckpt.get("val_files", [])]
    splits = [source_split(src, val_files) for src in sources]
    selected = torch.tensor([args.split == "all" or split == args.split for split in splits], dtype=torch.bool)
    if not selected.any():
        raise ValueError(f"No windows selected for --split {args.split!r}.")

    phase_means = torch.zeros(NUM_PHASES)
    phase_stds = torch.ones(NUM_PHASES)
    for phase_id in range(NUM_PHASES):
        mask = (phase == phase_id) & selected
        if mask.any():
            vals = errors[mask]
            phase_means[phase_id] = vals.mean()
            phase_stds[phase_id] = vals.std(unbiased=False).clamp(min=1.0e-6)

    window_rows = []
    motion_stats: dict[str, dict] = {}
    for idx, (err_t, phase_t, src, meta, split) in enumerate(zip(errors, phase, sources, metadata, splits)):
        if args.split != "all" and split != args.split:
            continue
        err = float(err_t.item())
        phase_id = int(phase_t.item())
        z = float(((err_t - phase_means[phase_id]) / phase_stds[phase_id]).item())
        kick_frame = int(meta["kick_frame"])
        center = int(meta["center"])
        row = {
            "index": int(idx),
            "source": src,
            "motion": relpath(src),
            "split": split,
            "phase": PHASE_NAMES[phase_id],
            "phase_id": phase_id,
            "error": err,
            "phase_z": z,
            "start": int(meta["start"]),
            "center": center,
            "end": int(meta["end"]),
            "kick_frame": kick_frame,
            "kick_delta": None if kick_frame < 0 else int(center - kick_frame),
            "kick_end_frame": int(meta["kick_end_frame"]),
            "kick_leg": str(meta["kick_leg"]),
            "motion_len": int(meta["motion_len"]),
        }
        window_rows.append(row)

        stats = motion_stats.setdefault(
            src,
            {
                "source": src,
                "motion": relpath(src),
                "split": split,
                "count": 0,
                "errors": [],
                "z_values": [],
                "phase_errors": {phase_name: [] for phase_name in PHASE_NAMES.values()},
                "worst_window": row,
            },
        )
        stats["count"] += 1
        stats["errors"].append(err)
        stats["z_values"].append(z)
        stats["phase_errors"][PHASE_NAMES[phase_id]].append(err)
        if row["error"] > stats["worst_window"]["error"]:
            stats["worst_window"] = row

    motion_rows = []
    for src, stats in motion_stats.items():
        phase_error_means = {
            phase_name: mean(stats["phase_errors"][phase_name])
            for phase_name in PHASE_NAMES.values()
        }
        present_phase_means = [(k, v) for k, v in phase_error_means.items() if v is not None]
        worst_phase = max(present_phase_means, key=lambda item: item[1])[0] if present_phase_means else "--"
        motion_rows.append(
            {
                "source": src,
                "motion": stats["motion"],
                "split": stats["split"],
                "count": int(stats["count"]),
                "mean_error": mean(stats["errors"]) or 0.0,
                "mean_phase_z": mean(stats["z_values"]) or 0.0,
                "worst_phase": worst_phase,
                "worst_window": stats["worst_window"],
                **{f"{phase_name}_error": phase_error_means[phase_name] for phase_name in PHASE_NAMES.values()},
            }
        )

    motion_rows.sort(key=lambda row: row["mean_error"], reverse=True)
    worst_windows = sorted(window_rows, key=lambda row: row["error"], reverse=True)[: args.top_windows]
    z_worst_windows = sorted(window_rows, key=lambda row: row["phase_z"], reverse=True)[: args.top_windows]

    print(f"Model: {args.model}")
    print(
        f"Motions: {len(motion_files)} files, windows={len(window_rows)}, "
        f"split={args.split}, ball_mode={build_args.ball_mode}, feature_frame={build_args.feature_frame}"
    )
    if filter_stats.get("active"):
        print(
            "Filter: "
            f"included_motions={filter_stats.get('included_motion_files', 0)}, "
            f"excluded_motions={filter_stats.get('excluded_motion_files', 0)}, "
            f"included_windows={filter_stats.get('included_windows', 0)}, "
            f"excluded_windows={filter_stats.get('excluded_windows', 0)}"
        )
    print(
        "Phase baseline error: "
        + ", ".join(
            f"{PHASE_NAMES[i]}={phase_means[i].item():.5f}+/-{phase_stds[i].item():.5f}"
            for i in range(NUM_PHASES)
        )
    )

    print("\n" + "=" * 144)
    print("  PER-MOTION / PER-PHASE RECONSTRUCTION ERROR")
    print("=" * 144)
    print(
        f"{'Motion':42s} {'N':>5s} {'All':>8s} {'Z':>7s} "
        f"{'Split':>5s} "
        f"{'Approach':>9s} {'Pre':>9s} {'Strike':>9s} {'Follow':>9s} "
        f"{'Worst':>9s} {'Win':>13s} {'WErr':>8s} {'WZ':>7s} {'KDelta':>7s}"
    )
    print("-" * 144)
    for row in motion_rows[: args.top_motions]:
        worst = row["worst_window"]
        k_delta = "--" if worst["kick_delta"] is None else str(worst["kick_delta"])
        win = f"{worst['start']}-{worst['end']}"
        print(
            f"{stem(row['source']):42s} {row['count']:5d} {row['mean_error']:8.5f} {row['mean_phase_z']:7.2f} "
            f"{row['split']:>5s} "
            f"{fmt(row['approach_error'], 9)} {fmt(row['prestrike_error'], 9)} "
            f"{fmt(row['strike_error'], 9)} {fmt(row['follow_error'], 9)} "
            f"{row['worst_phase']:>9s} {win:>13s} {worst['error']:8.5f} {worst['phase_z']:7.2f} {k_delta:>7s}"
        )

    print("\n" + "=" * 126)
    print("  WORST WINDOWS BY RAW ERROR")
    print("=" * 126)
    print(f"{'Rank':>4s} {'Motion':42s} {'Split':>5s} {'Phase':>9s} {'Err':>8s} {'Z':>7s} {'Win':>13s} {'Center':>6s} {'KDelta':>7s} {'Leg':>6s}")
    print("-" * 126)
    for rank, row in enumerate(worst_windows, start=1):
        k_delta = "--" if row["kick_delta"] is None else str(row["kick_delta"])
        win = f"{row['start']}-{row['end']}"
        print(
            f"{rank:4d} {stem(row['source']):42s} {row['split']:>5s} {row['phase']:>9s} "
            f"{row['error']:8.5f} {row['phase_z']:7.2f} {win:>13s} "
            f"{row['center']:6d} {k_delta:>7s} {row['kick_leg']:>6s}"
        )

    print("\n" + "=" * 126)
    print("  WORST WINDOWS BY PHASE-RELATIVE Z")
    print("=" * 126)
    print(f"{'Rank':>4s} {'Motion':42s} {'Split':>5s} {'Phase':>9s} {'Err':>8s} {'Z':>7s} {'Win':>13s} {'Center':>6s} {'KDelta':>7s} {'Leg':>6s}")
    print("-" * 126)
    for rank, row in enumerate(z_worst_windows, start=1):
        k_delta = "--" if row["kick_delta"] is None else str(row["kick_delta"])
        win = f"{row['start']}-{row['end']}"
        print(
            f"{rank:4d} {stem(row['source']):42s} {row['split']:>5s} {row['phase']:>9s} "
            f"{row['error']:8.5f} {row['phase_z']:7.2f} {win:>13s} "
            f"{row['center']:6d} {k_delta:>7s} {row['kick_leg']:>6s}"
        )

    output_dir = args.output_dir
    if output_dir is None:
        output_name = Path(args.model).stem
        if filter_stats.get("active"):
            if args.filter_manifest:
                output_name += f"__{Path(args.filter_manifest).stem}"
            else:
                output_name += "__checkpoint_filter"
        if args.split != "all":
            output_name += f"__{args.split}"
        output_dir = os.path.join("logs", "content_cvae_diagnostics", output_name)
    os.makedirs(output_dir, exist_ok=True)

    motion_csv = os.path.join(output_dir, "motion_phase_errors.csv")
    with open(motion_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "motion",
            "split",
            "count",
            "mean_error",
            "mean_phase_z",
            "approach_error",
            "prestrike_error",
            "strike_error",
            "follow_error",
            "worst_phase",
            "worst_window_start",
            "worst_window_center",
            "worst_window_end",
            "worst_window_phase",
            "worst_window_error",
            "worst_window_phase_z",
            "worst_window_kick_delta",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in motion_rows:
            worst = row["worst_window"]
            writer.writerow(
                {
                    "motion": row["motion"],
                    "split": row["split"],
                    "count": row["count"],
                    "mean_error": row["mean_error"],
                    "mean_phase_z": row["mean_phase_z"],
                    "approach_error": row["approach_error"],
                    "prestrike_error": row["prestrike_error"],
                    "strike_error": row["strike_error"],
                    "follow_error": row["follow_error"],
                    "worst_phase": row["worst_phase"],
                    "worst_window_start": worst["start"],
                    "worst_window_center": worst["center"],
                    "worst_window_end": worst["end"],
                    "worst_window_phase": worst["phase"],
                    "worst_window_error": worst["error"],
                    "worst_window_phase_z": worst["phase_z"],
                    "worst_window_kick_delta": worst["kick_delta"],
                }
            )

    windows_csv = os.path.join(output_dir, "window_errors.csv")
    with open(windows_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "motion",
            "split",
            "phase",
            "error",
            "phase_z",
            "start",
            "center",
            "end",
            "kick_frame",
            "kick_delta",
            "kick_end_frame",
            "kick_leg",
            "motion_len",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(window_rows, key=lambda item: item["error"], reverse=True):
            writer.writerow({key: row[key] for key in fieldnames})

    json_path = os.path.join(output_dir, "diagnostic_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "motion_paths": args.motion_path,
                "num_motions": len(motion_files),
                "num_windows": len(window_rows),
                "split": args.split,
                "ball_mode": build_args.ball_mode,
                "feature_frame": build_args.feature_frame,
                "filter_stats": filter_stats,
                "phase_baseline": {
                    PHASE_NAMES[i]: {
                        "mean": float(phase_means[i].item()),
                        "std": float(phase_stds[i].item()),
                    }
                    for i in range(NUM_PHASES)
                },
                "motion_rows": motion_rows,
                "worst_windows": worst_windows,
                "worst_windows_by_phase_z": z_worst_windows,
            },
            f,
            indent=2,
        )

    print(f"\n[INFO] Saved motion CSV: {motion_csv}")
    print(f"[INFO] Saved window CSV: {windows_csv}")
    print(f"[INFO] Saved JSON:       {json_path}")


if __name__ == "__main__":
    main()
