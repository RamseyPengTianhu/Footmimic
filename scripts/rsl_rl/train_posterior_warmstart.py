"""Path B: Posterior warm-start for RNN Actor.

Trains an LSTM actor (matching PPO's architecture exactly) to predict
posterior VQ codes from task_features policy obs, using CE loss.

The pre-trained actor can then be loaded into Stage C PPO via --warmstart_actor.

Pipeline:
  1. Load teacher rollout data (decoder_obs + actions + dones)
  2. Compute posterior codes from frozen VQ-VAE encoder
  3. Downsample codes to code_hold cadence using 2-frame mode
  4. Train LSTM actor with CE loss on code prediction
  5. Save actor_rnn + actor_mlp weights separately

Usage:
    python scripts/rsl_rl/train_posterior_warmstart.py \
        --data_path data/teacher_manifold/chunk_vae_task26.pt \
        --vq_model_path models/latent_v2/online_distill_vq_k16_hold2_seq.pt \
        --output_path models/warmstart/posterior_warmstart_actor.pt \
        --seq_len 64 --batch_size 64 --epochs 50 --lr 1e-3 \
        --code_hold 2
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ─── CLI ────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Posterior warm-start: train LSTM actor on VQ codes.")
parser.add_argument("--data_path", type=str, required=True,
                    help="Path to teacher rollout data (.pt file with decoder_obs, actions, dones).")
parser.add_argument("--vq_model_path", type=str, required=True,
                    help="Path to frozen VQ-VAE checkpoint.")
parser.add_argument("--output_path", type=str, default="models/warmstart/posterior_warmstart_actor.pt",
                    help="Where to save the pre-trained actor weights.")
parser.add_argument("--seq_len", type=int, default=64,
                    help="Sequence length for LSTM training windows.")
parser.add_argument("--batch_size", type=int, default=64,
                    help="Batch size for training.")
parser.add_argument("--epochs", type=int, default=50,
                    help="Number of training epochs.")
parser.add_argument("--lr", type=float, default=1e-3,
                    help="Learning rate.")
parser.add_argument("--code_hold", type=int, default=2,
                    help="Code hold duration. Downsample posterior codes to this cadence via mode.")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--seed", type=int, default=42)

# LSTM architecture (must match PPO actor exactly)
parser.add_argument("--lstm_hidden", type=int, default=128,
                    help="LSTM hidden size (must match PPO).")
parser.add_argument("--lstm_layers", type=int, default=2,
                    help="LSTM num_layers (must match PPO).")
parser.add_argument("--mlp_dims", type=str, default="128,64,32",
                    help="Comma-separated MLP hidden dims (must match PPO).")

args = parser.parse_args()


# ─── LSTM Actor (mirrors RSL-RL ActorCriticRecurrent actor) ─────────────────

class LSTMCodeActor(nn.Module):
    """LSTM + MLP actor that predicts VQ code logits.

    Architecture matches RSL-RL's ActorCriticRecurrent actor exactly:
        obs -> LSTM(input_size, hidden_size, num_layers) -> MLP -> logits[K]

    State dict key mapping to RSL-RL format:
        self.rnn.*        -> memory_a.rnn.*
        self.mlp.0.*      -> actor.0.*
        self.mlp.2.*      -> actor.2.*
        ...
    """

    def __init__(
        self,
        obs_dim: int,
        num_codes: int,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        mlp_dims: list[int] | None = None,
    ):
        super().__init__()
        if mlp_dims is None:
            mlp_dims = [128, 64, 32]

        self.obs_dim = obs_dim
        self.num_codes = num_codes
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

        # LSTM (matches memory_a.rnn in RSL-RL)
        self.rnn = nn.LSTM(
            input_size=obs_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
        )

        # MLP head (matches actor.* in RSL-RL)
        layers: list[nn.Module] = []
        dim = lstm_hidden
        for h in mlp_dims:
            layers.append(nn.Linear(dim, h))
            layers.append(nn.ELU())
            dim = h
        layers.append(nn.Linear(dim, num_codes))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self, obs_seq: torch.Tensor, hidden: tuple | None = None
    ) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            obs_seq: [B, T, obs_dim] observation sequence
            hidden: optional (h0, c0) for LSTM

        Returns:
            logits: [B, T, num_codes]
            hidden: (h_n, c_n)
        """
        rnn_out, hidden = self.rnn(obs_seq, hidden)  # [B, T, H]
        logits = self.mlp(rnn_out)  # [B, T, K]
        return logits, hidden

    def get_rsl_rl_state_dicts(self) -> tuple[dict, dict]:
        """Get separate RSL-RL compatible state dicts for actor RNN and actor MLP.

        Returns:
            actor_rnn_sd: keys like memory_a.rnn.*
            actor_mlp_sd: keys like actor.*
        """
        rnn_sd = {}
        for k, v in self.rnn.state_dict().items():
            rnn_sd[f"memory_a.rnn.{k}"] = v

        mlp_sd = {}
        for k, v in self.mlp.state_dict().items():
            mlp_sd[f"actor.{k}"] = v

        return rnn_sd, mlp_sd


# ─── Sequence Dataset ────────────────────────────────────────────────────────

class EpisodeSequenceDataset(Dataset):
    """Chop flat rollout data into fixed-length LSTM training sequences.

    Splits at episode boundaries (dones=True), then extracts windows of seq_len.
    Posterior codes are downsampled to code_hold cadence using mode (majority vote).
    """

    def __init__(
        self,
        obs: torch.Tensor,       # [N_total, obs_dim]
        codes: torch.Tensor,     # [N_total] long
        dones: torch.Tensor,     # [N_total] bool
        phase_ids: torch.Tensor, # [N_total] long
        seq_len: int,
        num_envs: int,
        code_hold: int = 1,
        num_codes: int = 16,
    ):
        self.seq_len = seq_len
        self.code_hold = code_hold
        self.num_codes = num_codes
        self.sequences: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        # (obs_window, held_codes, boundary_mask, phase_ids)

        # Reshape to [num_envs, T, ...] for per-env episode splitting
        T = obs.shape[0] // num_envs
        obs_2d = obs.reshape(num_envs, T, -1)
        codes_2d = codes.reshape(num_envs, T)
        dones_2d = dones.reshape(num_envs, T)
        phase_2d = phase_ids.reshape(num_envs, T)

        for env_i in range(num_envs):
            self._extract_env_sequences(
                obs_2d[env_i], codes_2d[env_i], dones_2d[env_i], phase_2d[env_i]
            )

        print(f"[DATA] {len(self.sequences)} sequences of length {seq_len} "
              f"from {num_envs} envs, code_hold={code_hold}")

    def _downsample_codes_mode(self, codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Downsample codes to code_hold cadence using mode (majority vote).

        Returns:
            held_codes: [T] with held code at each frame
            boundary_mask: [T] bool, True at code_hold boundaries
        """
        T = codes.shape[0]
        held_codes = codes.clone()
        boundary_mask = torch.zeros(T, dtype=torch.bool)

        for t in range(0, T, self.code_hold):
            window_end = min(t + self.code_hold, T)
            window = codes[t:window_end]
            # Mode: most frequent code in the window
            counts = torch.bincount(window, minlength=self.num_codes)
            mode_code = counts.argmax()
            held_codes[t:window_end] = mode_code
            boundary_mask[t] = True

        return held_codes, boundary_mask

    def _extract_env_sequences(self, obs, codes, dones, phases):
        """Extract non-overlapping windows from one env, respecting episode boundaries."""
        # Find episode starts
        ep_starts = [0]
        done_idxs = torch.where(dones)[0]
        for d in done_idxs:
            if d.item() + 1 < len(dones):
                ep_starts.append(d.item() + 1)

        for start in ep_starts:
            # Find episode end
            future_dones = torch.where(dones[start:])[0]
            if len(future_dones) > 0:
                end = start + future_dones[0].item() + 1
            else:
                end = len(dones)

            ep_len = end - start
            if ep_len < self.seq_len:
                continue

            ep_obs = obs[start:end]
            ep_codes = codes[start:end]
            ep_phases = phases[start:end]

            # Downsample codes to code_hold cadence using mode
            held_codes, boundary_mask = self._downsample_codes_mode(ep_codes)

            # Extract non-overlapping windows
            for w_start in range(0, ep_len - self.seq_len + 1, self.seq_len):
                w_end = w_start + self.seq_len
                self.sequences.append((
                    ep_obs[w_start:w_end],
                    held_codes[w_start:w_end],
                    boundary_mask[w_start:w_end],
                    ep_phases[w_start:w_end],
                ))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


# ─── Load VQ-VAE and compute posterior codes ─────────────────────────────────

def compute_posterior_codes(
    data_path: str,
    vq_model_path: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Load data + frozen VQ-VAE, return (obs, posterior_codes, dones, phase_ids, num_envs, num_codes)."""

    print(f"[INFO] Loading data from {data_path}")
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    obs = data["decoder_obs"]       # [N, 125]
    actions = data["actions"]       # [N, 29]
    dones = data["dones"]           # [N]
    phase_ids = data["phase_ids"]   # [N]
    meta = data["metadata"]
    num_envs = meta["num_envs"]
    print(f"[INFO] Data: {obs.shape[0]} frames, obs_dim={obs.shape[1]}, "
          f"action_dim={actions.shape[1]}, num_envs={num_envs}")

    # Load VQ-VAE (add scripts/rsl_rl to path for imports)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from latent_v2_models import LatentActionModel

    print(f"[INFO] Loading VQ-VAE from {vq_model_path}")
    ckpt = torch.load(vq_model_path, map_location=device, weights_only=False)
    vq_model = LatentActionModel(
        obs_dim=int(ckpt["obs_dim"]),
        action_dim=int(ckpt["action_dim"]),
        z_dim=int(ckpt["z_dim"]),
        hidden_dims=list(ckpt.get("hidden_dims", [512, 256, 128])),
        decoder_obs_mode=ckpt.get("decoder_obs_mode", "full"),
        prior_type=ckpt.get("prior_type", "mlp"),
        num_codes=int(ckpt.get("num_codes", 16)),
        commitment_weight=float(ckpt.get("commitment_weight", 0.25)),
        markov_prior=bool(ckpt.get("markov_prior", False)),
    ).to(device)
    vq_model.load_state_dict(ckpt["model_state_dict"])
    vq_model.eval()

    num_codes = vq_model.num_codes
    print(f"[INFO] VQ-VAE: num_codes={num_codes}, z_dim={vq_model.z_dim}, "
          f"decoder_obs_mode={vq_model.decoder_obs_mode}, "
          f"decoder_obs_dim={vq_model.decoder_obs_dim}")

    # Compute posterior codes in batches
    print("[INFO] Computing posterior codes...")
    batch_sz = 4096
    all_codes = torch.zeros(obs.shape[0], dtype=torch.long)

    with torch.no_grad():
        for i in range(0, obs.shape[0], batch_sz):
            j = min(i + batch_sz, obs.shape[0])
            obs_b = obs[i:j].to(device)
            act_b = actions[i:j].to(device)

            # VQ encoder: obs is already decoder_obs (125D), feed directly
            z_e = vq_model.encoder(obs_b, act_b)
            _, code_idx, _ = vq_model.codebook.quantize(z_e)
            all_codes[i:j] = code_idx.cpu()

            if (i // batch_sz) % 50 == 0:
                print(f"  {i}/{obs.shape[0]} frames processed")

    # Code distribution
    counts = torch.bincount(all_codes, minlength=num_codes)
    print(f"[INFO] Posterior code distribution: {counts.tolist()}")
    print(f"[INFO] Active codes: {(counts > 0).sum().item()}/{num_codes}")

    # Code usage entropy
    probs = counts.float() / counts.sum()
    entropy = -(probs * (probs + 1e-10).log()).sum().item()
    max_entropy = math.log(num_codes)
    print(f"[INFO] Code entropy: {entropy:.3f} / {max_entropy:.3f} "
          f"(utilization={entropy/max_entropy:.1%})")

    # Phase-code cross-tab
    print("\n[INFO] Phase × Code cross-tabulation:")
    phase_names = ["approach", "prestrike", "strike", "follow"]
    for p in range(4):
        mask = (phase_ids == p)
        if mask.sum() == 0:
            continue
        p_counts = torch.bincount(all_codes[mask], minlength=num_codes)
        top3 = p_counts.topk(3)
        top3_str = ", ".join(f"c{top3.indices[i]}:{top3.values[i]}" for i in range(3))
        print(f"  {phase_names[p]:>12s}: n={mask.sum().item():>7d}  top3=[{top3_str}]")

    return obs, all_codes, dones, phase_ids, num_envs, num_codes


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(
    logits: torch.Tensor,    # [N, K]
    targets: torch.Tensor,   # [N]
    boundary: torch.Tensor,  # [N] bool
    phases: torch.Tensor,    # [N]
    num_codes: int,
) -> dict[str, float]:
    """Compute comprehensive prediction metrics."""
    preds = logits.argmax(dim=-1)
    K = num_codes

    metrics = {}

    # Overall accuracy
    metrics["overall_acc"] = (preds == targets).float().mean().item()

    # Boundary accuracy (only at code_hold boundaries)
    if boundary.any():
        metrics["boundary_acc"] = (preds[boundary] == targets[boundary]).float().mean().item()

    # Top-3 accuracy
    top3 = logits.topk(3, dim=-1).indices  # [N, 3]
    metrics["top3_acc"] = (top3 == targets.unsqueeze(-1)).any(dim=-1).float().mean().item()

    # Phase-specific accuracy
    phase_names = ["approach", "prestrike", "strike", "follow"]
    for p in range(4):
        mask = (phases == p)
        if mask.sum() > 0:
            metrics[f"acc_{phase_names[p]}"] = (preds[mask] == targets[mask]).float().mean().item()

    # Per-code precision/recall for each code
    for c in range(K):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if (tp + fp + fn) > 0:
            metrics[f"c{c}_prec"] = prec
            metrics[f"c{c}_recall"] = recall

    # Code usage entropy of predictions
    pred_counts = torch.bincount(preds, minlength=K).float()
    pred_probs = pred_counts / pred_counts.sum()
    pred_entropy = -(pred_probs * (pred_probs + 1e-10).log()).sum().item()
    metrics["pred_entropy"] = pred_entropy
    metrics["pred_entropy_ratio"] = pred_entropy / math.log(K)

    return metrics


# ─── Training loop ──────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. Compute posterior codes
    obs, codes, dones, phase_ids, num_envs, num_codes = compute_posterior_codes(
        args.data_path, args.vq_model_path, device
    )

    obs_dim = obs.shape[1]  # 125
    mlp_dims = [int(x) for x in args.mlp_dims.split(",")]

    # 2. Build dataset
    dataset = EpisodeSequenceDataset(
        obs, codes, dones, phase_ids,
        seq_len=args.seq_len,
        num_envs=num_envs,
        code_hold=args.code_hold,
        num_codes=num_codes,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )

    # 3. Build LSTM actor (same architecture as PPO)
    model = LSTMCodeActor(
        obs_dim=obs_dim,
        num_codes=num_codes,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        mlp_dims=mlp_dims,
    ).to(device)
    print(f"\n[MODEL] {model}")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] {param_count:,} parameters")

    # 4. Training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_boundary_acc = 0.0
    best_state = None
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        all_logits = []
        all_targets = []
        all_boundary = []
        all_phases = []

        for batch_obs, batch_codes, batch_boundary, batch_phases in loader:
            batch_obs = batch_obs.to(device)       # [B, T, obs_dim]
            batch_codes = batch_codes.to(device)    # [B, T]

            logits, _ = model(batch_obs)  # [B, T, K]

            # Loss on ALL frames (not just boundaries) — LSTM needs dense signal
            loss = criterion(logits.reshape(-1, num_codes), batch_codes.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * batch_obs.shape[0]

            # Collect for metrics
            all_logits.append(logits.detach().cpu().reshape(-1, num_codes))
            all_targets.append(batch_codes.cpu().reshape(-1))
            all_boundary.append(batch_boundary.reshape(-1))
            all_phases.append(batch_phases.reshape(-1))

        scheduler.step()

        # Compute full metrics
        cat_logits = torch.cat(all_logits)
        cat_targets = torch.cat(all_targets)
        cat_boundary = torch.cat(all_boundary)
        cat_phases = torch.cat(all_phases)
        metrics = compute_metrics(cat_logits, cat_targets, cat_boundary, cat_phases, num_codes)

        avg_loss = total_loss / len(loader)
        elapsed = time.time() - t0

        # Print summary
        phase_accs = " | ".join(
            f"{k.split('_')[1][:3]}={v:.2f}"
            for k, v in sorted(metrics.items())
            if k.startswith("acc_")
        )
        print(
            f"  Epoch {epoch+1:3d}/{args.epochs} | "
            f"loss={avg_loss:.4f} | "
            f"acc={metrics['overall_acc']:.3f} | "
            f"bnd={metrics.get('boundary_acc', 0):.3f} | "
            f"top3={metrics['top3_acc']:.3f} | "
            f"ent={metrics['pred_entropy_ratio']:.2f} | "
            f"{phase_accs} | "
            f"lr={scheduler.get_last_lr()[0]:.1e} | "
            f"{elapsed:.0f}s"
        )

        # Detailed per-code metrics every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            print("    Per-code P/R:")
            for c in range(num_codes):
                p = metrics.get(f"c{c}_prec", 0)
                r = metrics.get(f"c{c}_recall", 0)
                if p > 0 or r > 0:
                    print(f"      c{c:2d}: prec={p:.3f} recall={r:.3f}")

        # Track best by boundary accuracy (what matters for PPO)
        ba = metrics.get("boundary_acc", metrics["overall_acc"])
        if ba > best_boundary_acc:
            best_boundary_acc = ba
            best_acc = metrics["overall_acc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\n[DONE] Best boundary_acc={best_boundary_acc:.3f}, overall_acc={best_acc:.3f}")

    # 5. Save in clean format
    model.load_state_dict(best_state)
    rnn_sd, mlp_sd = model.get_rsl_rl_state_dicts()

    save_dict = {
        "actor_rnn_state_dict": rnn_sd,     # memory_a.rnn.* keys
        "actor_mlp_state_dict": mlp_sd,     # actor.* keys
        "model_state_dict": best_state,     # native format (backup)
        "obs_dim": obs_dim,
        "num_codes": num_codes,
        "rnn_hidden_dim": args.lstm_hidden,
        "rnn_num_layers": args.lstm_layers,
        "mlp_dims": mlp_dims,
        "code_hold": args.code_hold,
        "best_boundary_acc": best_boundary_acc,
        "best_overall_acc": best_acc,
        "epochs": args.epochs,
        "data_path": args.data_path,
        "vq_model_path": args.vq_model_path,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save(save_dict, args.output_path)
    print(f"[SAVED] {args.output_path}")
    print(f"[INFO] actor_rnn keys: {list(rnn_sd.keys())[:5]}...")
    print(f"[INFO] actor_mlp keys: {list(mlp_sd.keys())[:5]}...")


if __name__ == "__main__":
    train(args)
