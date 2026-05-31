"""Stage 2A: Offline CVAE pre-training from teacher rollout data.

Usage:
    python scripts/rsl_rl/train_latent_v2_offline.py \
        --rollout_data data/teacher_manifold/v3_softmask_hmr4d_clean.pt \
        --output_path models/latent_v2/offline_cvae.pt \
        --obs_key obs_v3 \
        --z_dim 16 \
        --epochs 300 \
        --batch_size 2048 \
        --lr 1e-3 \
        --beta_max 1e-3 \
        --beta_warmup 20 \
        --device cuda

This trains the full LatentActionModel (encoder + decoder + prior) offline
on pre-collected teacher rollout data. No simulator needed.

Purpose: quickly validate that the VAE bottleneck can reconstruct teacher
actions with acceptable MSE before moving to online distillation.
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from latent_v2_models import LatentActionModel, latent_distill_loss


def main():
    parser = argparse.ArgumentParser(description="Stage 2A: Offline CVAE pre-training.")
    parser.add_argument("--rollout_data", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="models/latent_v2/offline_cvae.pt")
    parser.add_argument("--obs_key", type=str, default="obs_v3",
                        choices=["obs_v3", "obs_v10"],
                        help="Which observation to use for encoder/decoder/prior.")
    parser.add_argument("--z_dim", type=int, default=16)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta_max", type=float, default=1e-3,
                        help="Max KL weight (LATENT uses small beta).")
    parser.add_argument("--beta_warmup", type=int, default=20,
                        help="Epochs to linearly warm up beta from 0 to beta_max.")
    parser.add_argument("--val_ratio", type=float, default=0.05,
                        help="Fraction of data for validation.")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"[INFO] Loading data from: {args.rollout_data}")
    data = torch.load(args.rollout_data, map_location=args.device, weights_only=False)

    obs = data[args.obs_key]
    actions = data["actions_teacher"]
    phase_ids = data.get("phase_id", None)
    meta = data.get("metadata", {})

    obs_dim = obs.shape[1]
    action_dim = actions.shape[1]
    N = obs.shape[0]

    print(f"[INFO] obs_key={args.obs_key}, obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"[INFO] {N} transitions from teacher: {meta.get('teacher_run', '?')}")

    # ── Train/val split ────────────────────────────────────────────────────
    n_val = int(N * args.val_ratio)
    n_train = N - n_val
    perm = torch.randperm(N, device=args.device)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_loader = DataLoader(
        TensorDataset(obs[train_idx], actions[train_idx]),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_obs, val_act = obs[val_idx], actions[val_idx]

    if phase_ids is not None:
        val_phases = phase_ids[val_idx]
    else:
        val_phases = None

    print(f"[INFO] Train: {n_train}, Val: {n_val}")

    # ── Create model ───────────────────────────────────────────────────────
    model = LatentActionModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        z_dim=args.z_dim,
        hidden_dims=args.hidden_dims,
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model: z_dim={args.z_dim}, hidden={args.hidden_dims}, params={n_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_recon = float("inf")
    t0 = time.time()

    print(f"\n[INFO] Training for {args.epochs} epochs...")
    print(f"[INFO] beta warmup: 0 -> {args.beta_max} over {args.beta_warmup} epochs")

    for epoch in range(args.epochs):
        # Beta warmup
        if epoch < args.beta_warmup:
            beta = args.beta_max * (epoch / max(args.beta_warmup, 1))
        else:
            beta = args.beta_max

        # Train
        model.train()
        epoch_recon = 0.0
        epoch_kl = 0.0
        n_batches = 0

        for obs_b, act_b in train_loader:
            fwd = model(obs_b, act_b, sample=True)
            losses = latent_distill_loss(fwd, act_b, beta=beta)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_recon += losses["recon"].item()
            epoch_kl += losses["kl"].item()
            n_batches += 1

        scheduler.step()
        avg_recon = epoch_recon / max(n_batches, 1)
        avg_kl = epoch_kl / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_fwd = model(val_obs, val_act, sample=False)
            val_losses = latent_distill_loss(val_fwd, val_act, beta=beta)
            val_recon = val_losses["recon"].item()
            val_kl = val_losses["kl"].item()

            # Also check prior-only reconstruction (no teacher action at test time)
            prior_action = model.act_prior_mean(val_obs)
            prior_recon = (prior_action - val_act).pow(2).mean().item()

        # Log
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch+1:4d}/{args.epochs} | "
                f"train_recon={avg_recon:.6f} kl={avg_kl:.2f} | "
                f"val_recon={val_recon:.6f} val_kl={val_kl:.2f} | "
                f"prior_recon={prior_recon:.6f} | "
                f"beta={beta:.1e} lr={scheduler.get_last_lr()[0]:.1e} | "
                f"{elapsed:.0f}s"
            )

        # Save best
        if val_recon < best_val_recon:
            best_val_recon = val_recon
            os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
            ckpt = {
                "model_state_dict": model.state_dict(),
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "z_dim": args.z_dim,
                "hidden_dims": args.hidden_dims,
                "obs_key": args.obs_key,
                "best_val_recon": best_val_recon,
                "epoch": epoch + 1,
                "metadata": {
                    "stage": "2A_offline",
                    "teacher_run": meta.get("teacher_run", "?"),
                    "teacher_ckpt": meta.get("teacher_ckpt", "?"),
                    "beta_max": args.beta_max,
                    "n_train": n_train,
                },
            }
            torch.save(ckpt, args.output_path)

    # ── Final report ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Stage 2A: Offline CVAE Training Complete")
    print(f"{'='*70}")
    print(f"  Best val recon MSE: {best_val_recon:.6f}")
    print(f"  Saved to: {args.output_path}")

    # Per-phase MSE analysis
    if val_phases is not None:
        model.eval()
        model.load_state_dict(
            torch.load(args.output_path, map_location=args.device, weights_only=False)["model_state_dict"]
        )
        with torch.no_grad():
            # Posterior reconstruction
            post_fwd = model(val_obs, val_act, sample=False)
            post_mse = (post_fwd["recon"] - val_act).pow(2).mean(dim=-1)

            # Prior-only reconstruction
            prior_act = model.act_prior_mean(val_obs)
            prior_mse = (prior_act - val_act).pow(2).mean(dim=-1)

        phase_names = ["approach", "prestrike", "strike", "followthru"]
        print(f"\n  Per-phase MSE (validation):")
        print(f"  {'Phase':12s} | {'Posterior':>10s} | {'Prior':>10s} | {'Count':>6s}")
        print(f"  {'-'*50}")
        for pid, name in enumerate(phase_names):
            mask = val_phases == pid
            if mask.any():
                p_mse = post_mse[mask.to(args.device)].mean().item()
                pr_mse = prior_mse[mask.to(args.device)].mean().item()
                print(f"  {name:12s} | {p_mse:10.6f} | {pr_mse:10.6f} | {mask.sum().item():6d}")
            else:
                print(f"  {name:12s} | {'n/a':>10s} | {'n/a':>10s} | {0:6d}")

    # Gate check
    print(f"\n  Gate check: recon MSE {'PASS' if best_val_recon < 0.005 else 'FAIL'} "
          f"(threshold: 0.005, actual: {best_val_recon:.6f})")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
