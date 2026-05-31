"""BC pretraining for v10 MLP policy from v3 teacher rollout data.

Usage:
    python scripts/rsl_rl/train_v10_bc.py \
        --rollout_data data/bc_rollouts/v3_teacher_v10obs.pt \
        --output_path logs/rsl_rl/v10_bc/bc_pretrained.pt \
        --epochs 100 \
        --batch_size 256 \
        --lr 1e-3

This is standard single-step supervised learning — no sequence-level training
needed since v10 uses MLP (not LSTM). The history buffer is part of the obs.
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class V10MLPActor(nn.Module):
    """v10 MLP actor matching rsl_rl ActorCritic architecture.

    Architecture: [512, 256, 128] → 29D action (same as G1FlatPPORunnerCfg).
    """

    def __init__(self, obs_dim: int, action_dim: int = 29, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ELU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def main():
    parser = argparse.ArgumentParser(description="BC pretrain v10 MLP from v3 rollouts.")
    parser.add_argument("--rollout_data", type=str, required=True,
                        help="Path to v3 teacher rollout .pt file.")
    parser.add_argument("--extra_data", type=str, nargs="*", default=[],
                        help="Additional .pt files (e.g. DAgger rounds) to merge.")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from a previous BC checkpoint (warm-start weights).")
    parser.add_argument("--output_path", type=str, default="logs/rsl_rl/v10_bc/bc_pretrained.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--action_dim", type=int, default=29)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load primary dataset
    print(f"[INFO] Loading rollout data from: {args.rollout_data}")
    data = torch.load(args.rollout_data, map_location=args.device, weights_only=False)

    obs = data["obs_v10"]              # [N, ~422]
    actions = data["actions_teacher"]   # [N, 29]
    phase_ids = data.get("phase_id", None)  # [N] for per-phase MSE

    meta = data.get("metadata", {})
    print(f"[INFO] Primary: {obs.shape[0]} transitions, obs_dim={obs.shape[1]}")
    print(f"[INFO] Teacher: {meta.get('teacher_run', '?')} / {meta.get('teacher_ckpt', '?')}")

    # Merge extra datasets (DAgger rounds)
    for extra_path in args.extra_data:
        print(f"[INFO] Merging extra data: {extra_path}")
        extra = torch.load(extra_path, map_location=args.device, weights_only=False)
        extra_obs = extra["obs_v10"]
        extra_act = extra["actions_teacher"]
        obs = torch.cat([obs, extra_obs], dim=0)
        actions = torch.cat([actions, extra_act], dim=0)
        if phase_ids is not None and "phase_id" in extra:
            phase_ids = torch.cat([phase_ids, extra["phase_id"]], dim=0)
        extra_meta = extra.get("metadata", {})
        print(f"  +{extra_obs.shape[0]} transitions (type={extra_meta.get('type', '?')})")

    print(f"[INFO] Total dataset: {obs.shape[0]} transitions")

    # Create dataset and dataloader
    dataset = TensorDataset(obs, actions)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)


    # Create model
    obs_dim = obs.shape[1]
    model = V10MLPActor(obs_dim, args.action_dim).to(args.device)

    # Optionally resume from previous BC checkpoint
    if args.resume_from:
        print(f"[INFO] Resuming weights from: {args.resume_from}")
        prev_ckpt = torch.load(args.resume_from, map_location=args.device, weights_only=False)
        model.load_state_dict(prev_ckpt["model_state_dict"])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"[INFO] Model: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"[INFO] Training for {args.epochs} epochs, batch_size={args.batch_size}")

    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for obs_batch, action_batch in dataloader:
            pred_action = model(obs_batch)
            loss = ((pred_action - action_batch) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{args.epochs}: MSE={avg_loss:.6f}, lr={scheduler.get_last_lr()[0]:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save best model
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "obs_dim": obs_dim,
                "action_dim": args.action_dim,
                "hidden_dims": [512, 256, 128],
                "best_loss": best_loss,
                "epoch": epoch + 1,
            }
            torch.save(checkpoint, args.output_path)

    print(f"\n[INFO] Training complete. Best MSE: {best_loss:.6f}")
    print(f"[INFO] Saved to: {args.output_path}")

    # Per-phase MSE analysis
    if phase_ids is not None:
        model.eval()
        with torch.no_grad():
            pred_all = model(obs.to(args.device))
            mse_all = ((pred_all - actions.to(args.device)) ** 2).mean(dim=-1)  # [N]

        phase_names = ["approach", "prestrike", "strike", "followthru"]
        print(f"\n  Per-phase MSE:")
        for pid, name in enumerate(phase_names):
            mask = phase_ids == pid
            if mask.any():
                phase_mse = mse_all[mask.to(args.device)].mean().item()
                print(f"    {name:12s}: {phase_mse:.6f} ({mask.sum().item()} samples)")
            else:
                print(f"    {name:12s}: no samples")


if __name__ == "__main__":
    main()
