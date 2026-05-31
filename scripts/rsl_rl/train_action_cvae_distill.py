"""Train a state-conditioned action CVAE from teacher rollout data.

This is the first offline step toward a LATENT-style latent action space:

  q(z | obs, teacher_action), p(z | obs), D(action | obs, z)

The rollout format matches scripts/rsl_rl/collect_v3_teacher_rollouts.py.
"""

from __future__ import annotations

import argparse
import os
import importlib.util

import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


_model_path = os.path.join(os.path.dirname(__file__), "action_cvae_distill.py")
_spec = importlib.util.spec_from_file_location("action_cvae_distill", os.path.abspath(_model_path))
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
StateActionCVAE = _mod.StateActionCVAE
action_cvae_loss = _mod.action_cvae_loss


PHASE_NAMES = ["approach", "prestrike", "strike", "follow"]


def parse_hidden_dims(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_obs_slices(text: str | None, obs_dim: int) -> list[tuple[int, int]]:
    if text is None or text.strip().lower() in {"", "all"}:
        return [(0, obs_dim)]
    slices: list[tuple[int, int]] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            idx = int(part)
            start, end = idx, idx + 1
        else:
            start_s, end_s = part.split(":", 1)
            start = 0 if start_s == "" else int(start_s)
            end = obs_dim if end_s == "" else int(end_s)
        if start < 0:
            start += obs_dim
        if end < 0:
            end += obs_dim
        if start < 0 or end > obs_dim or start >= end:
            raise ValueError(f"Invalid obs slice {part!r} for obs_dim={obs_dim}")
        slices.append((start, end))
    if not slices:
        raise ValueError("No valid obs slices were parsed.")
    return slices


def apply_obs_slices(obs: torch.Tensor, slices: list[tuple[int, int]]) -> torch.Tensor:
    return torch.cat([obs[:, start:end] for start, end in slices], dim=-1)


def _build_action_chunks(
    obs: torch.Tensor,
    actions: torch.Tensor,
    phase: torch.Tensor,
    done: torch.Tensor | None,
    *,
    num_envs: int,
    action_horizon: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if action_horizon <= 1:
        return obs, actions, phase
    if num_envs <= 0:
        raise ValueError("action_horizon > 1 requires metadata['num_envs'] in rollout data.")
    if obs.shape[0] % num_envs != 0:
        raise ValueError(f"Cannot reshape {obs.shape[0]} samples into num_envs={num_envs}")

    steps = obs.shape[0] // num_envs
    if steps < action_horizon:
        raise ValueError(f"Rollout has only {steps} steps, shorter than action_horizon={action_horizon}")

    obs_t = obs.view(steps, num_envs, -1)
    actions_t = actions.view(steps, num_envs, -1)
    phase_t = phase.view(steps, num_envs)
    if done is None:
        done_t = torch.zeros(steps, num_envs, dtype=torch.bool)
    else:
        done_t = done.view(steps, num_envs).bool()

    valid_steps = steps - action_horizon + 1
    chunks = torch.stack(
        [actions_t[offset : offset + valid_steps] for offset in range(action_horizon)],
        dim=2,
    ).flatten(2)

    # Drop windows crossing an episode boundary.  We require all actions in the
    # chunk to belong to a non-terminal continuation; this is conservative and
    # avoids training primitives that mix two reset states.
    valid = torch.ones(valid_steps, num_envs, dtype=torch.bool)
    for offset in range(action_horizon):
        valid &= ~done_t[offset : offset + valid_steps]

    return (
        obs_t[:valid_steps].reshape(-1, obs.shape[-1])[valid.reshape(-1)],
        chunks.reshape(-1, chunks.shape[-1])[valid.reshape(-1)],
        phase_t[:valid_steps].reshape(-1)[valid.reshape(-1)],
    )


def load_rollout_file(
    path: str,
    obs_key: str,
    action_key: str,
    phase_key: str | None,
    action_horizon: int = 1,
):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if obs_key not in data:
        raise KeyError(f"{path} has no obs key {obs_key!r}. Available: {list(data.keys())}")
    if action_key not in data:
        raise KeyError(f"{path} has no action key {action_key!r}. Available: {list(data.keys())}")
    obs = data[obs_key].float()
    actions = data[action_key].float()
    if obs.shape[0] != actions.shape[0]:
        raise ValueError(f"{path}: obs/action length mismatch: {obs.shape[0]} vs {actions.shape[0]}")
    if phase_key and phase_key in data:
        phase = data[phase_key].long()
    else:
        phase = torch.full((obs.shape[0],), -1, dtype=torch.long)
    meta = data.get("metadata", {})
    done = data.get("done")
    num_envs = int(meta.get("num_envs", 0))
    obs, actions, phase = _build_action_chunks(
        obs,
        actions,
        phase,
        done,
        num_envs=num_envs,
        action_horizon=action_horizon,
    )
    return obs, actions, phase, meta


def load_dataset(args: argparse.Namespace):
    obs_chunks: list[torch.Tensor] = []
    action_chunks: list[torch.Tensor] = []
    phase_chunks: list[torch.Tensor] = []
    source_chunks: list[torch.Tensor] = []
    metadata = []
    base_obs_dim = None
    action_dim = None
    base_action_dim = None
    obs_slices = None

    for source_id, path in enumerate(args.rollout_data):
        obs, actions, phase, meta = load_rollout_file(
            path,
            args.obs_key,
            args.action_key,
            args.phase_key,
            action_horizon=args.action_horizon,
        )
        if base_obs_dim is None:
            base_obs_dim = obs.shape[1]
            action_dim = actions.shape[1]
            base_action_dim = int(meta.get("action_dim", 0)) or (
                action_dim // args.action_horizon if args.action_horizon > 1 else action_dim
            )
            obs_slices = parse_obs_slices(args.obs_slices, base_obs_dim)
        elif obs.shape[1] != base_obs_dim:
            raise ValueError(f"{path}: obs_dim={obs.shape[1]} differs from first dataset obs_dim={base_obs_dim}")
        elif actions.shape[1] != action_dim:
            raise ValueError(f"{path}: action_dim={actions.shape[1]} differs from first dataset action_dim={action_dim}")

        obs = apply_obs_slices(obs, obs_slices)
        obs_chunks.append(obs)
        action_chunks.append(actions)
        phase_chunks.append(phase)
        source_chunks.append(torch.full((obs.shape[0],), source_id, dtype=torch.long))
        metadata.append({"path": path, **meta})
        print(f"[INFO] Loaded {path}: n={obs.shape[0]}, obs_dim={obs.shape[1]}, action_dim={actions.shape[1]}")

    obs = torch.cat(obs_chunks, dim=0)
    actions = torch.cat(action_chunks, dim=0)
    phases = torch.cat(phase_chunks, dim=0)
    sources = torch.cat(source_chunks, dim=0)

    if args.max_samples and args.max_samples > 0 and obs.shape[0] > args.max_samples:
        generator = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(obs.shape[0], generator=generator)[: args.max_samples]
        obs = obs[idx]
        actions = actions[idx]
        phases = phases[idx]
        sources = sources[idx]

    return obs, actions, phases, sources, metadata, base_obs_dim, obs_slices, base_action_dim


def split_indices(n: int, val_split: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if val_split <= 0.0 or n <= 1:
        mask = torch.ones(n, dtype=torch.bool)
        return mask, mask
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    n_val = max(1, int(round(n * val_split)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    return train_mask, val_mask


@torch.no_grad()
def evaluate(model, loader, device: str):
    model.eval()
    recon_total = prior_total = kl_total = 0.0
    n = 0
    for obs_b, act_b, _, _ in loader:
        obs_b = obs_b.to(device)
        act_b = act_b.to(device)
        recon, q_mu, q_logvar, p_mu, p_logvar = model(obs_b, act_b, sample=False)
        prior_recon = model.decode(obs_b, p_mu)
        recon_mse = (recon - act_b).pow(2).mean(dim=-1)
        prior_mse = (prior_recon - act_b).pow(2).mean(dim=-1)
        kl = _mod.diag_gaussian_kl(q_mu, q_logvar, p_mu, p_logvar)
        bs = obs_b.shape[0]
        recon_total += recon_mse.sum().item()
        prior_total += prior_mse.sum().item()
        kl_total += kl.sum().item()
        n += bs
    return {
        "posterior_mse": recon_total / max(n, 1),
        "prior_mse": prior_total / max(n, 1),
        "kl": kl_total / max(n, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train action CVAE from teacher rollout data.")
    parser.add_argument("--rollout_data", type=str, nargs="+", required=True)
    parser.add_argument("--output_path", type=str, default="models/action_cvae_v3_teacher.pt")
    parser.add_argument("--init_model", type=str, default=None, help="Optional checkpoint to warm-start model weights from.")
    parser.add_argument("--obs_key", type=str, default="obs_v10")
    parser.add_argument("--action_key", type=str, default="actions_teacher")
    parser.add_argument("--phase_key", type=str, default="phase_id")
    parser.add_argument(
        "--obs_slices",
        type=str,
        default="all",
        help="Comma-separated obs slices, e.g. '0:414' to drop v10 motor_prior from 454D obs.",
    )
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--beta", type=float, default=1.0e-3)
    parser.add_argument("--prior_recon_weight", type=float, default=0.0)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument(
        "--action_horizon",
        type=int,
        default=1,
        help="If >1, train decoder to reconstruct a short future action chunk from each state.",
    )
    parser.add_argument("--hidden_dims", type=str, default="512,256,128")
    parser.add_argument(
        "--phase_balance",
        action="store_true",
        help="Use inverse-frequency phase weights for training batches when phase_id is available.",
    )
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    obs, actions, phases, sources, metadata, base_obs_dim, obs_slices, base_action_dim = load_dataset(args)
    train_mask, val_mask = split_indices(obs.shape[0], args.val_split, args.seed)

    obs_mean = obs[train_mask].mean(dim=0, keepdim=True)
    obs_std = obs[train_mask].std(dim=0, keepdim=True).clamp(min=1.0e-4)
    action_mean = actions[train_mask].mean(dim=0, keepdim=True)
    action_std = actions[train_mask].std(dim=0, keepdim=True).clamp(min=1.0e-4)

    obs_norm = (obs - obs_mean) / obs_std
    actions_norm = (actions - action_mean) / action_std

    dataset = TensorDataset(obs_norm, actions_norm, phases, sources)
    train_ds = torch.utils.data.Subset(dataset, train_mask.nonzero(as_tuple=True)[0].tolist())
    val_ds = torch.utils.data.Subset(dataset, val_mask.nonzero(as_tuple=True)[0].tolist())
    sampler = None
    shuffle = True
    if args.phase_balance and (phases[train_mask] >= 0).any():
        train_indices = train_mask.nonzero(as_tuple=True)[0]
        train_phases = phases[train_indices]
        valid = train_phases >= 0
        counts = torch.bincount(train_phases[valid].clamp(min=0), minlength=len(PHASE_NAMES)).float().clamp(min=1.0)
        sample_weights = torch.ones_like(train_phases, dtype=torch.float)
        sample_weights[valid] = 1.0 / counts[train_phases[valid]]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_indices),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        shuffle = False
        print(f"[INFO] Phase-balanced sampler counts={counts.tolist()}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    hidden_dims = parse_hidden_dims(args.hidden_dims)
    model = StateActionCVAE(
        obs_dim=obs.shape[1],
        action_dim=actions.shape[1],
        latent_dim=args.latent_dim,
        hidden_dims=hidden_dims,
    ).to(args.device)
    if args.init_model:
        init_ckpt = torch.load(args.init_model, map_location="cpu", weights_only=False)
        init_state = init_ckpt["model_state_dict"]
        model_state = model.state_dict()
        compatible_state = {
            key: value
            for key, value in init_state.items()
            if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
        }
        missing = sorted(set(model_state) - set(compatible_state))
        unexpected = sorted(set(init_state) - set(compatible_state))
        model_state.update(compatible_state)
        model.load_state_dict(model_state)
        print(
            f"[INFO] Warm-started from {args.init_model}: loaded={len(compatible_state)}, "
            f"missing={len(missing)}, skipped={len(unexpected)}"
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1.0e-5)

    print(
        f"[INFO] Dataset: n={obs.shape[0]}, train={int(train_mask.sum())}, val={int(val_mask.sum())}, "
        f"base_obs_dim={base_obs_dim}, obs_dim={obs.shape[1]}, "
        f"base_action_dim={base_action_dim}, action_dim={actions.shape[1]}, action_horizon={args.action_horizon}"
    )
    print(f"[INFO] obs_slices={obs_slices}, latent_dim={args.latent_dim}, hidden_dims={hidden_dims}")

    best_val = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        total = recon_total = kl_total = prior_total = 0.0
        batches = 0
        for obs_b, act_b, _, _ in train_loader:
            obs_b = obs_b.to(args.device)
            act_b = act_b.to(args.device)
            recon, q_mu, q_logvar, p_mu, p_logvar = model(obs_b, act_b, sample=True)
            loss, recon_loss, kl = action_cvae_loss(
                recon,
                act_b,
                q_mu,
                q_logvar,
                p_mu,
                p_logvar,
                beta=args.beta,
            )
            prior_loss = torch.zeros((), device=args.device)
            if args.prior_recon_weight > 0.0:
                prior_recon = model.decode(obs_b, p_mu)
                prior_loss = (prior_recon - act_b).pow(2).mean()
                loss = loss + args.prior_recon_weight * prior_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total += loss.item()
            recon_total += recon_loss.item()
            kl_total += kl.item()
            prior_total += prior_loss.item()
            batches += 1

        train_loss = total / max(batches, 1)
        train_recon = recon_total / max(batches, 1)
        train_kl = kl_total / max(batches, 1)
        train_prior = prior_total / max(batches, 1)
        val_metrics = evaluate(model, val_loader, args.device)
        val_score = val_metrics["posterior_mse"] + args.beta * val_metrics["kl"]

        if val_score < best_val:
            best_val = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"epoch {epoch + 1:04d}/{args.epochs}: "
                f"train={train_loss:.6f} recon={train_recon:.6f} kl={train_kl:.4f} prior_aux={train_prior:.6f} | "
                f"val_post={val_metrics['posterior_mse']:.6f} val_prior={val_metrics['prior_mse']:.6f} "
                f"val_kl={val_metrics['kl']:.4f}"
            )

    assert best_state is not None
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    checkpoint = {
        "model_state_dict": best_state,
        "obs_dim": obs.shape[1],
        "base_obs_dim": base_obs_dim,
        "action_dim": actions.shape[1],
        "base_action_dim": base_action_dim,
        "action_horizon": args.action_horizon,
        "latent_dim": args.latent_dim,
        "hidden_dims": hidden_dims,
        "obs_key": args.obs_key,
        "action_key": args.action_key,
        "phase_key": args.phase_key,
        "obs_slices": obs_slices,
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "rollout_metadata": metadata,
        "best_val_score": best_val,
        "beta": args.beta,
        "prior_recon_weight": args.prior_recon_weight,
        "init_model": args.init_model,
    }
    torch.save(checkpoint, args.output_path)
    print(f"[INFO] Saved action CVAE to {args.output_path} (best_val={best_val:.6f})")


if __name__ == "__main__":
    main()
