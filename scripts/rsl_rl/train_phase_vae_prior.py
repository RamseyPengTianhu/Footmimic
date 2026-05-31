"""Train an offline phase-conditioned VAE motion prior from NPZ motions.

This is a VAE-as-prior prototype. It does not control the policy. It learns a
phase-conditioned likelihood over short windows of low-dimensional motion
features and saves a checkpoint that can later be used for diagnostics or a
reward term.

Usage:
  python scripts/rsl_rl/train_phase_vae_prior.py \
    --motion_path motions/Video \
    --output models/phase_vae_prior.pt \
    --epochs 400
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

# Direct import to avoid pulling in IsaacLab for offline training.
import importlib.util

_model_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "source",
    "whole_body_tracking",
    "soccer",
    "tasks",
    "tracking",
    "mdp",
    "phase_vae_prior.py",
)
_spec = importlib.util.spec_from_file_location("phase_vae_prior", os.path.abspath(_model_path))
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
PhaseConditionedVAE = _mod.PhaseConditionedVAE
vae_loss = _mod.vae_loss


PHASE_APPROACH = 0
PHASE_PRESTRIKE = 1
PHASE_STRIKE = 2
PHASE_FOLLOW = 3
NUM_PHASES = 4


def get_motion_files(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    files = sorted(glob.glob(os.path.join(path, "**", "*.npz"), recursive=True))
    if not files:
        raise ValueError(f"No .npz files found under {path}")
    return files


def phase_at_frame(t: int, kick_frame: int, kick_end_frame: int, motion_len: int, prestrike_len: int) -> int:
    if kick_frame < 0:
        return PHASE_APPROACH
    if kick_end_frame < 0:
        kick_end_frame = min(kick_frame + 5, motion_len)
    approach_end = max(0, kick_frame - prestrike_len)
    if t < approach_end:
        return PHASE_APPROACH
    if t < kick_frame:
        return PHASE_PRESTRIKE
    if t <= kick_end_frame:
        return PHASE_STRIKE
    return PHASE_FOLLOW


def extract_frame_features(data: np.lib.npyio.NpzFile, lower_body_only: bool = True) -> np.ndarray:
    """Extract compact motion-style features from one reference motion.

    Defaults to lower-body + waist joint pos/vel, pelvis height, and pelvis
    linear/angular velocity. This keeps the prototype useful even with only a
    couple of motions.
    """
    joint_pos = data["joint_pos"].astype(np.float32)
    joint_vel = data["joint_vel"].astype(np.float32)
    body_pos = data["body_pos_w"].astype(np.float32)
    body_lin_vel = data["body_lin_vel_w"].astype(np.float32)
    body_ang_vel = data["body_ang_vel_w"].astype(np.float32)

    if lower_body_only:
        # Joint order from the conversion scripts: 12 leg joints + 3 waist.
        joint_slice = slice(0, 15)
    else:
        joint_slice = slice(0, joint_pos.shape[1])

    pelvis_height = body_pos[:, 0, 2:3]
    pelvis_lin_vel = body_lin_vel[:, 0, :]
    pelvis_ang_vel = body_ang_vel[:, 0, :]

    return np.concatenate(
        [
            joint_pos[:, joint_slice],
            joint_vel[:, joint_slice],
            pelvis_height,
            pelvis_lin_vel,
            pelvis_ang_vel,
        ],
        axis=-1,
    ).astype(np.float32)


def build_dataset(
    motion_files: list[str],
    window_len: int,
    stride: int,
    prestrike_len: int,
    lower_body_only: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    windows: list[np.ndarray] = []
    phases: list[int] = []
    sources: list[str] = []

    for motion_file in motion_files:
        data = np.load(motion_file, allow_pickle=True)
        feat = extract_frame_features(data, lower_body_only=lower_body_only)
        motion_len = feat.shape[0]
        kick_frame = int(np.asarray(data.get("kick_frame", -1)).item())
        kick_end_frame = int(np.asarray(data.get("kick_end_frame", -1)).item())

        for start in range(0, max(motion_len - window_len + 1, 0), stride):
            end = start + window_len
            center = start + window_len // 2
            phase = phase_at_frame(center, kick_frame, kick_end_frame, motion_len, prestrike_len)
            windows.append(feat[start:end].reshape(-1))
            phases.append(phase)
            sources.append(motion_file)

    if not windows:
        raise ValueError("No windows generated. Try a smaller --window_len.")

    x = torch.tensor(np.stack(windows), dtype=torch.float32)
    phase = torch.tensor(phases, dtype=torch.long)
    return x, phase, sources


def main():
    parser = argparse.ArgumentParser(description="Train phase-conditioned VAE motion prior")
    parser.add_argument("--motion_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="models/phase_vae_prior.pt")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--beta", type=float, default=1.0e-3)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--window_len", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--prestrike_len", type=int, default=20)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--full_body", action="store_true", help="Use all joints instead of lower-body + waist")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    motion_files = get_motion_files(args.motion_path)
    x, phase, sources = build_dataset(
        motion_files,
        window_len=args.window_len,
        stride=args.stride,
        prestrike_len=args.prestrike_len,
        lower_body_only=not args.full_body,
    )

    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp(min=1.0e-4)
    x = (x - mean) / std

    dataset = TensorDataset(x, phase)
    n_val = max(1, int(len(dataset) * args.val_split)) if len(dataset) > 1 else 0
    n_train = len(dataset) - n_val
    if n_val > 0:
        train_ds, val_ds = random_split(dataset, [n_train, n_val])
    else:
        train_ds, val_ds = dataset, dataset

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = PhaseConditionedVAE(
        input_dim=x.shape[1],
        num_phases=NUM_PHASES,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1.0e-5)

    print(f"Loaded {len(motion_files)} files, {len(dataset)} windows, input_dim={x.shape[1]}")
    counts = torch.bincount(phase, minlength=NUM_PHASES)
    print(
        "Phase counts: "
        f"approach={counts[0].item()}, prestrike={counts[1].item()}, "
        f"strike={counts[2].item()}, follow={counts[3].item()}"
    )

    best_val = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        train_total = train_recon = train_kl = 0.0
        n_batches = 0
        for xb, pb in train_loader:
            xb = xb.to(args.device)
            pb = pb.to(args.device)
            recon, mu, logvar = model(xb, pb)
            loss, recon_loss, kl = vae_loss(recon, xb, mu, logvar, beta=args.beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total += loss.item()
            train_recon += recon_loss.item()
            train_kl += kl.item()
            n_batches += 1

        model.eval()
        val_total = val_recon = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for xb, pb in val_loader:
                xb = xb.to(args.device)
                pb = pb.to(args.device)
                recon, mu, logvar = model(xb, pb)
                loss, recon_loss, _ = vae_loss(recon, xb, mu, logvar, beta=args.beta)
                val_total += loss.item()
                val_recon += recon_loss.item()
                n_val_batches += 1

        train_total /= max(n_batches, 1)
        train_recon /= max(n_batches, 1)
        train_kl /= max(n_batches, 1)
        val_total /= max(n_val_batches, 1)
        val_recon /= max(n_val_batches, 1)

        if val_total < best_val:
            best_val = val_total
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 0 or (epoch + 1) % 25 == 0:
            print(
                f"epoch {epoch + 1:04d}/{args.epochs}: "
                f"train={train_total:.5f} recon={train_recon:.5f} kl={train_kl:.5f} "
                f"val={val_total:.5f} val_recon={val_recon:.5f}"
            )

    assert best_state is not None
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "input_dim": x.shape[1],
            "num_phases": NUM_PHASES,
            "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim,
            "window_len": args.window_len,
            "prestrike_len": args.prestrike_len,
            "lower_body_only": not args.full_body,
            "feature_mean": mean,
            "feature_std": std,
            "motion_files": motion_files,
            "sources": sources,
            "best_val_loss": best_val,
        },
        args.output,
    )
    print(f"Saved phase VAE prior to {args.output} (best_val={best_val:.5f})")


if __name__ == "__main__":
    main()
