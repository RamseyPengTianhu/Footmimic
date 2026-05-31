"""V3.5 Strike Discriminator Training — Offline binary classifier.

Trains a small MLP to distinguish strike states from approach/running states.
Uses soft-label BCE with data from `collect_strike_data.py`.

Usage:
  python scripts/rsl_rl/train_strike_discriminator.py \\
    --data models/strike_data.pt \\
    --output models/strike_discriminator.pt \\
    --epochs 200 --lr 1e-3 --batch_size 256
"""
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

# Direct import to avoid IsaacLab dependency chain (no sim needed for offline training)
import importlib.util
_disc_path = os.path.join(os.path.dirname(__file__), "..", "..",
    "source", "whole_body_tracking", "soccer", "tasks", "tracking", "mdp",
    "strike_discriminator.py")
_spec = importlib.util.spec_from_file_location("strike_discriminator", os.path.abspath(_disc_path))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
StrikeDiscriminator = _mod.StrikeDiscriminator
INPUT_DIM = _mod.INPUT_DIM


def main():
    parser = argparse.ArgumentParser(description="Train strike discriminator")
    parser.add_argument("--data", type=str, default="models/strike_data.pt")
    parser.add_argument("--output", type=str, default="models/strike_discriminator.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load data
    data = torch.load(args.data, weights_only=False)
    features = data["features"].to(args.device)
    labels = data["labels"].to(args.device)
    sources = data["sources"]

    print(f"Loaded {features.shape[0]} samples, feature_dim={features.shape[1]}")
    print(f"  Label distribution: mean={labels.mean():.3f}, "
          f"positive(>0.3)={(labels > 0.3).sum().item()}, "
          f"negative(<=0.3)={(labels <= 0.3).sum().item()}")
    print(f"  Sources: ref_neg={int((sources==0).sum())}, ref_pos={int((sources==1).sum())}, "
          f"rollout_pos={int((sources==2).sum())}, rollout_hard_neg={int((sources==3).sum())}")

    # Dataset & split
    dataset = TensorDataset(features, labels)
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Model
    model = StrikeDiscriminator(input_dim=INPUT_DIM, hidden=args.hidden).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCELoss()

    print(f"\nTraining: {n_train} train, {n_val} val, {args.epochs} epochs")
    print(f"Model: {sum(p.numel() for p in model.parameters())} params\n")

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        n_batches = 0
        for feat_b, label_b in train_loader:
            pred = model(feat_b)
            loss = criterion(pred, label_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= n_batches
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0
        n_val_batches = 0
        all_pred = []
        all_true = []
        with torch.no_grad():
            for feat_b, label_b in val_loader:
                pred = model(feat_b)
                loss = criterion(pred, label_b)
                val_loss += loss.item()
                n_val_batches += 1
                all_pred.append(pred)
                all_true.append(label_b)
        val_loss /= max(n_val_batches, 1)

        # Metrics
        all_pred = torch.cat(all_pred)
        all_true = torch.cat(all_true)

        # Binary threshold metrics
        pred_pos = all_pred > 0.5
        true_pos = all_true > 0.3  # positive = label > 0.3
        tp = (pred_pos & true_pos).sum().item()
        fp = (pred_pos & ~true_pos).sum().item()
        fn = (~pred_pos & true_pos).sum().item()
        tn = (~pred_pos & ~true_pos).sum().item()
        tpr = tp / max(tp + fn, 1)
        tnr = tn / max(tn + fp, 1)
        precision = tp / max(tp + fp, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
                  f"train={train_loss:.4f}, val={val_loss:.4f} | "
                  f"TPR={tpr:.3f}, TNR={tnr:.3f}, Prec={precision:.3f}")

    # Save best model
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "input_dim": INPUT_DIM,
        "hidden": args.hidden,
        "best_val_loss": best_val_loss,
    }, args.output)

    print(f"\nSaved best model (val_loss={best_val_loss:.4f}) to {args.output}")

    # Final evaluation
    model.load_state_dict(best_state)
    model.to(args.device)
    model.eval()
    with torch.no_grad():
        all_pred = model(features)
    pred_pos = all_pred > 0.5
    true_pos = labels > 0.3
    tp = (pred_pos & true_pos).sum().item()
    fp = (pred_pos & ~true_pos).sum().item()
    fn = (~pred_pos & true_pos).sum().item()
    tn = (~pred_pos & ~true_pos).sum().item()
    print(f"\nFull dataset evaluation:")
    print(f"  TPR (recall):  {tp / max(tp + fn, 1):.3f}")
    print(f"  TNR:           {tn / max(tn + fp, 1):.3f}")
    print(f"  Precision:     {tp / max(tp + fp, 1):.3f}")
    print(f"  F1:            {2 * tp / max(2 * tp + fp + fn, 1):.3f}")
    print(f"  Total: TP={tp}, FP={fp}, FN={fn}, TN={tn}")


if __name__ == "__main__":
    main()
