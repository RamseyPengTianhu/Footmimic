"""Evaluate a content-conditioned CVAE motion prior.

The main diagnostic is cross-condition reconstruction error. A useful prior
should reconstruct a strike window best when the condition also says "strike"
and should get worse when the phase part of the condition is swapped.
"""

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

from train_content_cvae_prior import (
    COND_NAMES,
    NUM_PHASES,
    PHASE_NAMES,
    ContentConditionedVAE,
    build_dataset,
    get_motion_files,
    load_filter_manifest,
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


@torch.no_grad()
def cross_phase_errors(
    model: ContentConditionedVAE,
    x: torch.Tensor,
    cond_raw: torch.Tensor,
    cond_mean: torch.Tensor,
    cond_std: torch.Tensor,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    phase_chunks = []
    model.eval()
    for phase_id in range(NUM_PHASES):
        swapped = cond_raw.clone()
        swapped[:, :NUM_PHASES] = 0.0
        swapped[:, phase_id] = 1.0
        swapped = (swapped - cond_mean) / cond_std
        phase_chunks.append(reconstruction_errors(model, x, swapped, batch_size, device))
    return torch.stack(phase_chunks, dim=1)


def format_matrix(matrix: torch.Tensor) -> str:
    header = "true\\cond      " + "  ".join(f"{PHASE_NAMES[i]:>10}" for i in range(NUM_PHASES))
    rows = [header]
    for i in range(NUM_PHASES):
        vals = "  ".join(f"{matrix[i, j].item():10.5f}" for j in range(NUM_PHASES))
        rows.append(f"{PHASE_NAMES[i]:>10}  {vals}")
    return "\n".join(rows)


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


def main():
    parser = argparse.ArgumentParser(description="Evaluate content-conditioned CVAE prior")
    parser.add_argument("--model", type=str, default="models/content_cvae_prior_video.pt")
    parser.add_argument("--motion_path", type=str, nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--ball_mode", choices=["best_swing", "nominal"], default=None)
    parser.add_argument("--filter_manifest", type=str, default=None)
    parser.add_argument("--filter_range_match", choices=["overlap", "center"], default=None)
    parser.add_argument(
        "--ignore_checkpoint_filter",
        action="store_true",
        help="Evaluate all requested motions instead of the checkpoint's saved filter manifest.",
    )
    parser.add_argument(
        "--feature_frame",
        choices=["local", "world"],
        default=None,
        help="Override checkpoint feature frame. Old checkpoints default to world.",
    )
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
    x, cond_raw, phase, sources = build_dataset(
        motion_files,
        window_len=int(ckpt["window_len"]),
        stride=args.stride,
        prestrike_len=int(ckpt["prestrike_len"]),
        lower_body_only=bool(ckpt.get("lower_body_only", True)),
        args=build_args,
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

    correct = reconstruction_errors(model, x, cond, args.batch_size, args.device)
    errors = cross_phase_errors(
        model,
        x,
        cond_raw,
        ckpt["cond_mean"],
        ckpt["cond_std"],
        args.batch_size,
        args.device,
    )

    matrix = torch.zeros(NUM_PHASES, NUM_PHASES)
    counts = torch.bincount(phase, minlength=NUM_PHASES)
    for true_phase in range(NUM_PHASES):
        mask = phase == true_phase
        if mask.any():
            matrix[true_phase] = errors[mask].mean(dim=0)
        else:
            matrix[true_phase] = float("nan")

    pred_phase = errors.argmin(dim=1)
    acc = (pred_phase == phase).float().mean()
    wrong_sum = errors.sum(dim=1) - correct
    wrong_mean = wrong_sum / max(NUM_PHASES - 1, 1)
    margin = wrong_mean - correct

    print(f"Model: {args.model}")
    print(
        f"Motions: {len(motion_files)} files, windows={x.shape[0]}, "
        f"input_dim={x.shape[1]}, cond_dim={cond.shape[1]}, "
        f"ball_mode={build_args.ball_mode}, feature_frame={build_args.feature_frame}"
    )
    if filter_stats.get("active"):
        print(
            "Filter: "
            f"included_motions={filter_stats.get('included_motion_files', 0)}, "
            f"excluded_motions={filter_stats.get('excluded_motion_files', 0)}, "
            f"included_windows={filter_stats.get('included_windows', 0)}, "
            f"excluded_windows={filter_stats.get('excluded_windows', 0)}"
        )
    print("Condition layout:")
    print("  " + ", ".join(ckpt.get("cond_names", COND_NAMES)))
    print(
        "Phase counts: "
        + ", ".join(f"{PHASE_NAMES[i]}={counts[i].item()}" for i in range(NUM_PHASES))
    )

    print("\nCross-phase condition reconstruction error (lower is better):")
    print(format_matrix(matrix))

    print("\nPhase discrimination from argmin reconstruction error:")
    print(f"  accuracy: {acc.item():.3f}")
    print(f"  mean correct error: {correct.mean().item():.5f}")
    print(f"  mean wrong error:   {wrong_mean.mean().item():.5f}")
    print(f"  mean margin:        {margin.mean().item():.5f}")

    print("\nPer-phase margins (wrong_mean - correct):")
    for phase_id in range(NUM_PHASES):
        mask = phase == phase_id
        if not mask.any():
            continue
        phase_acc = (pred_phase[mask] == phase[mask]).float().mean().item()
        print(
            f"  {PHASE_NAMES[phase_id]:>9}: "
            f"n={mask.sum().item():4d}, acc={phase_acc:.3f}, "
            f"correct={correct[mask].mean().item():.5f}, "
            f"wrong={wrong_mean[mask].mean().item():.5f}, "
            f"margin={margin[mask].mean().item():.5f}"
        )

    source_to_errors: dict[str, list[float]] = {}
    for src, err in zip(sources, correct.tolist()):
        source_to_errors.setdefault(src, []).append(float(err))
    ranked = sorted(
        ((sum(vals) / len(vals), src, len(vals)) for src, vals in source_to_errors.items()),
        reverse=True,
    )
    print("\nHighest reconstruction-error source files:")
    for mean_err, src, n in ranked[:10]:
        print(f"  {mean_err:.5f}  n={n:4d}  {os.path.relpath(src)}")


if __name__ == "__main__":
    main()
