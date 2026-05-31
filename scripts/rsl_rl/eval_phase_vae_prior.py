"""Evaluate a phase-conditioned VAE motion prior.

The key diagnostic is cross-phase reconstruction error:

    true strike window decoded as strike should have lower error than the same
    window decoded as approach/prestrike/follow.

If all columns are similar, the model is behaving like a generic autoencoder
instead of a useful semantic phase prior.
"""

from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from train_phase_vae_prior import NUM_PHASES, PHASE_APPROACH, PHASE_PRESTRIKE, PHASE_STRIKE, PHASE_FOLLOW
from train_phase_vae_prior import build_dataset, get_motion_files
from train_phase_vae_prior import PhaseConditionedVAE


PHASE_NAMES = {
    PHASE_APPROACH: "approach",
    PHASE_PRESTRIKE: "prestrike",
    PHASE_STRIKE: "strike",
    PHASE_FOLLOW: "follow",
}


@torch.no_grad()
def cross_phase_errors(
    model: PhaseConditionedVAE,
    x: torch.Tensor,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Return per-sample reconstruction error for every decode phase."""
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
    chunks = []
    model.eval()
    for (xb,) in loader:
        xb = xb.to(device)
        phase_errors = []
        for phase_id in range(NUM_PHASES):
            phase = torch.full((xb.shape[0],), phase_id, dtype=torch.long, device=device)
            mu, _ = model.encode(xb, phase)
            recon = model.decode(mu, phase)
            phase_errors.append(torch.mean((recon - xb) ** 2, dim=-1))
        chunks.append(torch.stack(phase_errors, dim=1).cpu())
    return torch.cat(chunks, dim=0)


def format_matrix(matrix: torch.Tensor) -> str:
    header = "true\\decode     " + "  ".join(f"{PHASE_NAMES[i]:>10}" for i in range(NUM_PHASES))
    rows = [header]
    for i in range(NUM_PHASES):
        vals = "  ".join(f"{matrix[i, j].item():10.5f}" for j in range(NUM_PHASES))
        rows.append(f"{PHASE_NAMES[i]:>11}  {vals}")
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate phase VAE prior")
    parser.add_argument("--model", type=str, default="models/phase_vae_prior_video.pt")
    parser.add_argument("--motion_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    motion_path = args.motion_path
    if motion_path is None:
        motion_files = ckpt.get("motion_files", [])
        if not motion_files:
            raise ValueError("Checkpoint has no motion_files; pass --motion_path explicitly.")
    else:
        motion_files = get_motion_files(motion_path)

    x, phase, sources = build_dataset(
        motion_files,
        window_len=int(ckpt["window_len"]),
        stride=args.stride,
        prestrike_len=int(ckpt["prestrike_len"]),
        lower_body_only=bool(ckpt.get("lower_body_only", True)),
    )
    x = (x - ckpt["feature_mean"]) / ckpt["feature_std"]

    model = PhaseConditionedVAE(
        input_dim=int(ckpt["input_dim"]),
        num_phases=int(ckpt.get("num_phases", NUM_PHASES)),
        latent_dim=int(ckpt["latent_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
    ).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])

    errors = cross_phase_errors(model, x, batch_size=args.batch_size, device=args.device)

    matrix = torch.zeros(NUM_PHASES, NUM_PHASES)
    counts = torch.bincount(phase, minlength=NUM_PHASES)
    for true_phase in range(NUM_PHASES):
        mask = phase == true_phase
        if mask.any():
            matrix[true_phase] = errors[mask].mean(dim=0)
        else:
            matrix[true_phase] = float("nan")

    correct = errors.gather(1, phase[:, None]).squeeze(1)
    wrong_sum = errors.sum(dim=1) - correct
    wrong_mean = wrong_sum / max(NUM_PHASES - 1, 1)
    margin = wrong_mean - correct
    pred_phase = errors.argmin(dim=1)
    acc = (pred_phase == phase).float().mean()

    print(f"Model: {args.model}")
    print(f"Motions: {len(motion_files)} files, windows={x.shape[0]}, input_dim={x.shape[1]}")
    print(
        "Phase counts: "
        + ", ".join(f"{PHASE_NAMES[i]}={counts[i].item()}" for i in range(NUM_PHASES))
    )
    print("\nCross-phase reconstruction error (lower is better):")
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
            f"n={mask.sum().item():3d}, acc={phase_acc:.3f}, "
            f"correct={correct[mask].mean().item():.5f}, "
            f"wrong={wrong_mean[mask].mean().item():.5f}, "
            f"margin={margin[mask].mean().item():.5f}"
        )

    print("\nSource files:")
    for path in motion_files:
        print(f"  {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
