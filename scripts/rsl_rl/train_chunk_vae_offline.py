"""Offline Continuous Chunk VAE training with action normalization.

Supports both snapshot prior (history_len=0) and history prior (history_len>0).

Usage:
    # Snapshot prior (original)
    python scripts/rsl_rl/train_chunk_vae_offline.py \\
        --data data/teacher_manifold/chunk_vae_task22.pt \\
        --chunk_len 4 --z_dim 32 --epochs 300

    # History prior (K=4 past frames)
    python scripts/rsl_rl/train_chunk_vae_offline.py \\
        --data data/teacher_manifold/chunk_vae_task22.pt \\
        --chunk_len 4 --z_dim 32 --history_len 4 --epochs 300 \\
        --output_path models/chunk_vae/chunk_vae_H4_z32_hist4.pt
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from chunk_vae_models import ChunkVAE, chunk_vae_loss


def extract_chunks(data: dict, chunk_len: int, history_len: int = 0,
                   device: str = "cpu"):
    """Extract non-overlapping H-frame chunks from flat data, respecting episode boundaries.

    If history_len > 0, also extracts a history window of (K+1) frames
    ending at the chunk start frame t: obs[t-K : t+1].

    Returns:
        obs_chunks: [N_chunks, H, obs_dim]
        action_chunks: [N_chunks, H, action_dim]
        phase_chunks: [N_chunks, H]
        history_chunks: [N_chunks, K+1, obs_dim] or None if history_len=0
    """
    dec_obs = data["decoder_obs"]  # [T_total, obs_dim]
    actions = data["actions"]      # [T_total, action_dim]
    dones = data["dones"]          # [T_total]
    phases = data["phase_ids"]     # [T_total]

    meta = data["metadata"]
    num_envs = meta["num_envs"]
    num_steps = meta["num_steps"]
    T_total = dec_obs.shape[0]

    assert T_total == num_envs * num_steps, \
        f"Data size mismatch: {T_total} vs {num_envs}*{num_steps}"

    # Reshape to [num_envs, num_steps, ...]
    obs_dim = dec_obs.shape[-1]
    action_dim = actions.shape[-1]
    dec_obs = dec_obs.reshape(num_envs, num_steps, obs_dim)
    actions = actions.reshape(num_envs, num_steps, action_dim)
    dones = dones.reshape(num_envs, num_steps)
    phases = phases.reshape(num_envs, num_steps)

    K = history_len  # number of past frames needed

    obs_chunks = []
    act_chunks = []
    phase_chunks = []
    hist_chunks = [] if K > 0 else None

    for e in range(num_envs):
        # Track episode start for history boundary checking
        ep_start = 0
        t = 0
        while t + chunk_len <= num_steps:
            # Update episode start if we crossed a done
            while ep_start < t and dones[e, ep_start:t].any():
                # Find last done before t
                done_mask = dones[e, ep_start:t]
                last_done_rel = done_mask.nonzero(as_tuple=True)[0][-1].item()
                ep_start = ep_start + last_done_rel + 1

            chunk_dones = dones[e, t:t + chunk_len]
            if chunk_dones[:-1].any():
                first_done = chunk_dones.nonzero(as_tuple=True)[0][0].item()
                ep_start = t + first_done + 1
                t = ep_start
                continue

            # Check if we have enough history within the same episode
            if K > 0:
                hist_start = t - K
                if hist_start < ep_start:
                    # Not enough history in this episode, skip forward
                    t += chunk_len
                    continue
                # Also check no dones in the history window
                hist_dones = dones[e, hist_start:t]
                if hist_dones.any():
                    t += chunk_len
                    continue
                # History window: [t-K, t-K+1, ..., t] = K+1 frames
                hist_chunks.append(dec_obs[e, hist_start:t + 1])

            obs_chunks.append(dec_obs[e, t:t + chunk_len])
            act_chunks.append(actions[e, t:t + chunk_len])
            phase_chunks.append(phases[e, t:t + chunk_len])
            t += chunk_len  # non-overlapping

    if len(obs_chunks) == 0:
        raise ValueError("No valid chunks extracted! Check data and chunk_len.")

    obs_chunks = torch.stack(obs_chunks).to(device)
    act_chunks = torch.stack(act_chunks).to(device)
    phase_chunks = torch.stack(phase_chunks).to(device)
    if hist_chunks is not None:
        hist_chunks = torch.stack(hist_chunks).to(device)

    return obs_chunks, act_chunks, phase_chunks, hist_chunks


def main():
    parser = argparse.ArgumentParser(description="Train Chunk VAE offline.")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to collected data (.pt)")
    parser.add_argument("--output_path", type=str,
                        default="models/chunk_vae/chunk_vae_H8_z16.pt")
    parser.add_argument("--chunk_len", type=int, default=8)
    parser.add_argument("--z_dim", type=int, default=16)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--history_len", type=int, default=0,
                        help="Number of past frames for history prior. 0=snapshot.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta_max", type=float, default=1e-3,
                        help="Max KL weight.")
    parser.add_argument("--beta_warmup", type=int, default=30,
                        help="Epochs to warm up beta from 0 to beta_max.")
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    K = args.history_len

    # ── Load and chunk data ─────────────────────────────────────────────
    print(f"[INFO] Loading data: {args.data}")
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    meta = data["metadata"]
    print(f"[INFO] dec_obs_dim={meta['dec_obs_dim']}, action_dim={meta['action_dim']}")
    print(f"[INFO] {meta['num_envs']} envs × {meta['num_steps']} steps")

    obs_chunks, act_chunks, phase_chunks, hist_chunks = extract_chunks(
        data, args.chunk_len, history_len=K, device=args.device
    )
    N = obs_chunks.shape[0]
    obs_dim = obs_chunks.shape[-1]
    action_dim = act_chunks.shape[-1]
    H = args.chunk_len

    print(f"[INFO] Extracted {N} chunks of length {H}")
    print(f"[INFO] obs_dim={obs_dim}, action_dim={action_dim}")
    if K > 0:
        print(f"[INFO] History prior: K={K} past frames (window={K+1})")
        print(f"[INFO] History shape: {hist_chunks.shape}")

    # ── Action normalization ────────────────────────────────────────────
    all_actions_flat = act_chunks.reshape(-1, action_dim)
    act_mean = all_actions_flat.mean(dim=0)
    act_std = all_actions_flat.std(dim=0).clamp(min=1e-6)
    print(f"[INFO] Action stats: mean range [{act_mean.min():.3f}, {act_mean.max():.3f}], "
          f"std range [{act_std.min():.3f}, {act_std.max():.3f}]")

    act_chunks_norm = (act_chunks - act_mean.view(1, 1, -1)) / act_std.view(1, 1, -1)

    # ── Train/val split ─────────────────────────────────────────────────
    n_val = max(int(N * args.val_ratio), 1)
    n_train = N - n_val
    perm = torch.randperm(N, device=args.device)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_obs = obs_chunks[train_idx]
    train_act = act_chunks_norm[train_idx]
    val_obs = obs_chunks[val_idx]
    val_act_norm = act_chunks_norm[val_idx]
    val_act_raw = act_chunks[val_idx]
    val_phases = phase_chunks[val_idx]

    # History splits
    train_hist = hist_chunks[train_idx] if hist_chunks is not None else None
    val_hist = hist_chunks[val_idx] if hist_chunks is not None else None

    # DataLoader
    if train_hist is not None:
        train_dataset = TensorDataset(train_obs, train_act, train_hist)
    else:
        train_dataset = TensorDataset(train_obs, train_act)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
    )

    print(f"[INFO] Train: {n_train}, Val: {n_val}")

    # ── Create model ────────────────────────────────────────────────────
    model = ChunkVAE(
        obs_dim=obs_dim,
        action_dim=action_dim,
        chunk_len=H,
        z_dim=args.z_dim,
        hidden_dims=args.hidden_dims,
        history_len=K,
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    prior_type = f"history(K={K})" if K > 0 else "snapshot"
    print(f"[INFO] ChunkVAE: H={H}, z_dim={args.z_dim}, prior={prior_type}, "
          f"hidden={args.hidden_dims}, params={n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ───────────────────────────────────────────────────
    best_val_recon = float("inf")
    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"  Chunk VAE Offline Training (with action normalization)")
    print(f"  H={H}, z_dim={args.z_dim}, prior={prior_type}, beta_max={args.beta_max}")
    print(f"  {args.epochs} epochs, batch={args.batch_size}")
    print(f"{'='*70}\n")

    for epoch in range(args.epochs):
        beta = args.beta_max * min(1.0, epoch / max(args.beta_warmup, 1))

        model.train()
        epoch_recon = 0.0
        epoch_kl = 0.0
        n_batches = 0

        for batch in train_loader:
            if K > 0:
                obs_b, act_b, hist_b = batch
            else:
                obs_b, act_b = batch
                hist_b = None

            fwd = model(obs_b, act_b, sample=True, obs_history=hist_b)
            losses = chunk_vae_loss(fwd, act_b, beta=beta)

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
            val_fwd = model(val_obs, val_act_norm, sample=False, obs_history=val_hist)
            val_losses = chunk_vae_loss(val_fwd, val_act_norm, beta=beta)
            val_recon_norm = val_losses["recon"].item()
            val_kl_raw = val_losses["kl_raw"].item()

            # Denormalize for raw-space MSE
            val_recon_raw_act = val_fwd["recon"] * act_std.view(1, 1, -1) + act_mean.view(1, 1, -1)
            val_recon_raw = F.mse_loss(val_recon_raw_act, val_act_raw).item()

            # Prior-only
            prior_actions_norm = model.act_prior_chunk(
                val_obs[:, 0], sample=False, obs_history=val_hist
            )
            prior_raw_act = prior_actions_norm * act_std.view(1, 1, -1) + act_mean.view(1, 1, -1)
            prior_recon_raw = F.mse_loss(prior_raw_act, val_act_raw).item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch+1:4d}/{args.epochs} | "
                f"norm_recon={avg_recon:.6f} kl={avg_kl:.4f} | "
                f"val_norm={val_recon_norm:.6f} val_raw={val_recon_raw:.6f} | "
                f"prior_raw={prior_recon_raw:.6f} kl_raw={val_kl_raw:.1f} | "
                f"β={beta:.1e} | {elapsed:.0f}s"
            )

        # Save best
        if val_recon_norm < best_val_recon:
            best_val_recon = val_recon_norm
            os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
            ckpt = {
                "model_state_dict": model.state_dict(),
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "z_dim": args.z_dim,
                "chunk_len": H,
                "hidden_dims": args.hidden_dims,
                "history_len": K,
                "act_mean": act_mean.cpu(),
                "act_std": act_std.cpu(),
                "best_val_recon": best_val_recon,
                "epoch": epoch + 1,
                "metadata": {
                    "stage": "chunk_vae_offline",
                    "data_path": args.data,
                    "beta_max": args.beta_max,
                    "n_train": n_train,
                    "include_phase": meta.get("include_phase", False),
                    "prior_type": prior_type,
                },
            }
            torch.save(ckpt, args.output_path)

    # ── Final report ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Chunk VAE Training Complete")
    print(f"{'='*70}")
    print(f"  Prior type: {prior_type}")
    print(f"  Best val normalized recon MSE: {best_val_recon:.6f}")
    print(f"  Saved to: {args.output_path}")

    # Per-phase analysis
    model.eval()
    model.load_state_dict(
        torch.load(args.output_path, map_location=args.device, weights_only=False)[
            "model_state_dict"
        ]
    )
    with torch.no_grad():
        post_fwd = model(val_obs, val_act_norm, sample=False, obs_history=val_hist)
        post_raw = post_fwd["recon"] * act_std.view(1, 1, -1) + act_mean.view(1, 1, -1)
        post_mse = (post_raw - val_act_raw).pow(2).mean(dim=(1, 2))

        prior_norm = model.act_prior_chunk(val_obs[:, 0], sample=False, obs_history=val_hist)
        prior_raw = prior_norm * act_std.view(1, 1, -1) + act_mean.view(1, 1, -1)
        prior_mse = (prior_raw - val_act_raw).pow(2).mean(dim=(1, 2))

    chunk_phase = val_phases[:, 0]
    phase_names = ["approach", "prestrike", "strike", "followthru"]
    print(f"\n  Per-phase RAW MSE (validation):")
    print(f"  {'Phase':12s} | {'Posterior':>10s} | {'Prior':>10s} | {'Count':>6s}")
    print(f"  {'-'*50}")
    for pid, name in enumerate(phase_names):
        mask = chunk_phase == pid
        if mask.any():
            p_mse = post_mse[mask].mean().item()
            pr_mse = prior_mse[mask].mean().item()
            print(f"  {name:12s} | {p_mse:10.6f} | {pr_mse:10.6f} | {mask.sum().item():6d}")
        else:
            print(f"  {name:12s} | {'n/a':>10s} | {'n/a':>10s} | {0:6d}")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
