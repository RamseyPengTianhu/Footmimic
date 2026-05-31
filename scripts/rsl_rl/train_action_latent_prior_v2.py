"""Conservative LATENT/PULSE-style prior refinement for an action CVAE.

The original ``train_action_cvae_distill.py`` trains posterior, prior and
decoder together.  That is useful for the first offline distillation pass, but
it is too destructive for online/DAgger refinement: a few bad on-policy states
can move the decoder and break the action manifold.

This script keeps the action manifold fixed by default:

  q(z | obs, teacher_action)  frozen posterior
  D(obs, z)                  frozen decoder
  p(z | obs)                 trainable prior

It is the safer v2 path for comparing teachers/data while keeping the rest of
the latent-action pipeline aligned with LATENT/PULSE.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import os

import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


_model_path = os.path.join(os.path.dirname(__file__), "action_cvae_distill.py")
_spec = importlib.util.spec_from_file_location("action_cvae_distill", os.path.abspath(_model_path))
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
StateActionCVAE = _mod.StateActionCVAE
diag_gaussian_kl = _mod.diag_gaussian_kl

_train_path = os.path.join(os.path.dirname(__file__), "train_action_cvae_distill.py")
_train_spec = importlib.util.spec_from_file_location("train_action_cvae_distill", os.path.abspath(_train_path))
_train_mod = importlib.util.module_from_spec(_train_spec)
assert _train_spec.loader is not None
_train_spec.loader.exec_module(_train_mod)
PHASE_NAMES = _train_mod.PHASE_NAMES
apply_obs_slices = _train_mod.apply_obs_slices
load_rollout_file = _train_mod.load_rollout_file
split_indices = _train_mod.split_indices


def _load_model(ckpt: dict, device: str) -> StateActionCVAE:
    model = StateActionCVAE(
        obs_dim=int(ckpt["obs_dim"]),
        action_dim=int(ckpt["action_dim"]),
        latent_dim=int(ckpt["latent_dim"]),
        hidden_dims=list(ckpt["hidden_dims"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def _load_dataset(args: argparse.Namespace, ckpt: dict):
    obs_chunks: list[torch.Tensor] = []
    action_chunks: list[torch.Tensor] = []
    phase_chunks: list[torch.Tensor] = []
    source_chunks: list[torch.Tensor] = []
    metadata = []
    action_horizon = int(ckpt.get("action_horizon", 1))
    if action_horizon != 1 and not args.allow_horizon_gt1:
        raise ValueError(
            f"Checkpoint action_horizon={action_horizon}; v2 prior refinement is intended for H=1. "
            "Pass --allow_horizon_gt1 only for debugging."
        )

    for source_id, path in enumerate(args.rollout_data):
        obs, actions, phases, meta = load_rollout_file(
            path,
            ckpt.get("obs_key", "obs_v10"),
            ckpt.get("action_key", "actions_teacher"),
            ckpt.get("phase_key", "phase_id"),
            action_horizon=action_horizon,
        )
        base_obs_dim = int(ckpt.get("base_obs_dim", obs.shape[1]))
        if obs.shape[1] != base_obs_dim:
            raise ValueError(f"{path}: obs_dim={obs.shape[1]} differs from checkpoint base_obs_dim={base_obs_dim}")
        obs = apply_obs_slices(obs.float(), ckpt["obs_slices"])
        actions = actions.float()
        if obs.shape[1] != int(ckpt["obs_dim"]):
            raise ValueError(f"{path}: selected obs_dim={obs.shape[1]} differs from checkpoint obs_dim={ckpt['obs_dim']}")
        if actions.shape[1] != int(ckpt["action_dim"]):
            raise ValueError(f"{path}: action_dim={actions.shape[1]} differs from checkpoint action_dim={ckpt['action_dim']}")

        obs_chunks.append(obs)
        action_chunks.append(actions)
        phase_chunks.append(phases.long())
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

    obs_norm = (obs - ckpt["obs_mean"]) / ckpt["obs_std"]
    actions_norm = (actions - ckpt["action_mean"]) / ckpt["action_std"]
    return obs_norm, actions_norm, phases, sources, metadata


def _make_loaders(
    obs: torch.Tensor,
    actions: torch.Tensor,
    phases: torch.Tensor,
    sources: torch.Tensor,
    args: argparse.Namespace,
):
    train_mask, val_mask = split_indices(obs.shape[0], args.val_split, args.seed)
    dataset = TensorDataset(obs, actions, phases, sources)
    train_indices = train_mask.nonzero(as_tuple=True)[0]
    val_indices = val_mask.nonzero(as_tuple=True)[0]
    train_ds = torch.utils.data.Subset(dataset, train_indices.tolist())
    val_ds = torch.utils.data.Subset(dataset, val_indices.tolist())

    sampler = None
    shuffle = True
    if args.phase_balance and (phases[train_mask] >= 0).any():
        train_phases = phases[train_indices]
        valid = train_phases >= 0
        counts = torch.bincount(train_phases[valid].clamp(min=0), minlength=len(PHASE_NAMES)).float().clamp(min=1.0)
        weights = torch.ones_like(train_phases, dtype=torch.float)
        weights[valid] = 1.0 / counts[train_phases[valid]]
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(train_indices),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        shuffle = False
        print(f"[INFO] Phase-balanced sampler counts={counts.tolist()}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader, train_mask, val_mask


def _set_trainable(model: StateActionCVAE, *, train_decoder: bool):
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.prior.parameters():
        param.requires_grad_(True)
    if train_decoder:
        for param in model.decoder.parameters():
            param.requires_grad_(True)


def _parameter_anchor(module: torch.nn.Module, initial_state: dict[str, torch.Tensor]) -> torch.Tensor:
    loss = None
    for name, param in module.named_parameters():
        term = (param - initial_state[name].to(param.device)).pow(2).mean()
        loss = term if loss is None else loss + term
    if loss is None:
        device = next(module.parameters()).device
        return torch.zeros((), device=device)
    return loss


@torch.no_grad()
def _evaluate(
    model: StateActionCVAE,
    initial_model: StateActionCVAE,
    loader: DataLoader,
    args: argparse.Namespace,
):
    model.eval()
    totals = {
        "post_mse": 0.0,
        "prior_mse": 0.0,
        "kl": 0.0,
        "maha": 0.0,
        "prior_anchor_mu": 0.0,
    }
    n = 0
    for obs_b, act_b, _, _ in loader:
        obs_b = obs_b.to(args.device)
        act_b = act_b.to(args.device)
        q_mu, q_logvar = model.encode(obs_b, act_b)
        p_mu, p_logvar = model.prior_stats(obs_b)
        p0_mu, _ = initial_model.prior_stats(obs_b)
        post_recon = model.decode(obs_b, q_mu)
        prior_recon = model.decode(obs_b, p_mu)
        bs = obs_b.shape[0]
        totals["post_mse"] += (post_recon - act_b).pow(2).mean(dim=-1).sum().item()
        totals["prior_mse"] += (prior_recon - act_b).pow(2).mean(dim=-1).sum().item()
        totals["kl"] += diag_gaussian_kl(q_mu, q_logvar, p_mu, p_logvar).sum().item()
        totals["maha"] += (((q_mu - p_mu) ** 2) / p_logvar.exp().clamp(min=1.0e-8)).sum(dim=-1).sqrt().sum().item()
        totals["prior_anchor_mu"] += (p_mu - p0_mu).pow(2).mean(dim=-1).sum().item()
        n += bs
    return {key: value / max(n, 1) for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser(description="Conservatively refine only p(z|obs) in an action CVAE.")
    parser.add_argument("--init_model", type=str, required=True)
    parser.add_argument("--rollout_data", type=str, nargs="+", required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--kl_weight", type=float, default=1.0)
    parser.add_argument("--prior_recon_weight", type=float, default=0.1)
    parser.add_argument("--mu_mse_weight", type=float, default=0.0)
    parser.add_argument("--logvar_mse_weight", type=float, default=0.0)
    parser.add_argument("--prior_anchor_weight", type=float, default=0.01)
    parser.add_argument("--decoder_anchor_weight", type=float, default=10.0)
    parser.add_argument("--train_decoder", action="store_true")
    parser.add_argument("--phase_balance", action="store_true", default=True)
    parser.add_argument("--no_phase_balance", action="store_false", dest="phase_balance")
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--allow_horizon_gt1", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    ckpt = torch.load(args.init_model, map_location="cpu", weights_only=False)
    obs, actions, phases, sources, metadata = _load_dataset(args, ckpt)
    train_loader, val_loader, train_mask, val_mask = _make_loaders(obs, actions, phases, sources, args)

    model = _load_model(ckpt, args.device)
    initial_model = _load_model(ckpt, args.device)
    initial_model.eval()
    for param in initial_model.parameters():
        param.requires_grad_(False)
    initial_prior_state = {k: v.detach().clone().cpu() for k, v in initial_model.prior.state_dict().items()}
    initial_decoder_state = {k: v.detach().clone().cpu() for k, v in initial_model.decoder.state_dict().items()}

    _set_trainable(model, train_decoder=args.train_decoder)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=1.0e-5)

    print(
        f"[INFO] Dataset: n={obs.shape[0]}, train={int(train_mask.sum())}, val={int(val_mask.sum())}, "
        f"obs_dim={ckpt['obs_dim']}, action_dim={ckpt['action_dim']}, action_horizon={ckpt.get('action_horizon', 1)}"
    )
    print(
        f"[INFO] v2 prior refinement: train_decoder={args.train_decoder}, lr={args.lr}, "
        f"prior_anchor={args.prior_anchor_weight}, decoder_anchor={args.decoder_anchor_weight}"
    )

    best_score = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        running = {
            "loss": 0.0,
            "kl": 0.0,
            "prior_recon": 0.0,
            "mu_mse": 0.0,
            "anchor": 0.0,
        }
        batches = 0
        for obs_b, act_b, _, _ in train_loader:
            obs_b = obs_b.to(args.device)
            act_b = act_b.to(args.device)

            with torch.no_grad():
                q_mu, q_logvar = initial_model.encode(obs_b, act_b)
                p0_mu, p0_logvar = initial_model.prior_stats(obs_b)

            p_mu, p_logvar = model.prior_stats(obs_b)
            prior_recon = model.decode(obs_b, p_mu)

            kl = diag_gaussian_kl(q_mu, q_logvar, p_mu, p_logvar).mean()
            prior_recon_loss = (prior_recon - act_b).pow(2).mean()
            mu_mse = (p_mu - q_mu).pow(2).mean()
            logvar_mse = (p_logvar - q_logvar).pow(2).mean()
            prior_anchor = (p_mu - p0_mu).pow(2).mean() + 0.25 * (p_logvar - p0_logvar).pow(2).mean()
            param_anchor = args.prior_anchor_weight * _parameter_anchor(model.prior, initial_prior_state)
            if args.train_decoder:
                param_anchor = param_anchor + args.decoder_anchor_weight * _parameter_anchor(
                    model.decoder,
                    initial_decoder_state,
                )

            loss = (
                args.kl_weight * kl
                + args.prior_recon_weight * prior_recon_loss
                + args.mu_mse_weight * mu_mse
                + args.logvar_mse_weight * logvar_mse
                + args.prior_anchor_weight * prior_anchor
                + param_anchor
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)
            optimizer.step()

            running["loss"] += loss.item()
            running["kl"] += kl.item()
            running["prior_recon"] += prior_recon_loss.item()
            running["mu_mse"] += mu_mse.item()
            running["anchor"] += prior_anchor.item()
            batches += 1

        val = _evaluate(model, initial_model, val_loader, args)
        val_score = val["prior_mse"] + args.kl_weight * val["kl"]
        if val_score < best_score:
            best_score = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 0 or (epoch + 1) % 10 == 0:
            denom = max(batches, 1)
            print(
                f"epoch {epoch + 1:04d}/{args.epochs}: "
                f"train={running['loss'] / denom:.6f} kl={running['kl'] / denom:.4f} "
                f"prior_recon={running['prior_recon'] / denom:.6f} "
                f"mu_mse={running['mu_mse'] / denom:.6f} anchor={running['anchor'] / denom:.6f} | "
                f"val_post={val['post_mse']:.6f} val_prior={val['prior_mse']:.6f} "
                f"val_kl={val['kl']:.4f} val_maha={val['maha']:.4f} "
                f"val_prior_anchor={val['prior_anchor_mu']:.6f}"
            )

    assert best_state is not None
    out_ckpt = copy.deepcopy(ckpt)
    out_ckpt["model_state_dict"] = best_state
    out_ckpt["rollout_metadata"] = metadata
    out_ckpt["init_model"] = args.init_model
    out_ckpt["latent_prior_v2"] = {
        "train_decoder": args.train_decoder,
        "lr": args.lr,
        "kl_weight": args.kl_weight,
        "prior_recon_weight": args.prior_recon_weight,
        "mu_mse_weight": args.mu_mse_weight,
        "logvar_mse_weight": args.logvar_mse_weight,
        "prior_anchor_weight": args.prior_anchor_weight,
        "decoder_anchor_weight": args.decoder_anchor_weight,
        "best_val_score": best_score,
    }

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.save(out_ckpt, args.output_path)
    print(f"[INFO] Saved v2 prior-refined action CVAE to {args.output_path} (best_val={best_score:.6f})")


if __name__ == "__main__":
    main()
