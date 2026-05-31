"""Evaluate an offline state-conditioned action CVAE."""

from __future__ import annotations

import argparse
import os
import importlib.util

import torch
from torch.utils.data import DataLoader, TensorDataset

from train_action_cvae_distill import (
    PHASE_NAMES,
    apply_obs_slices,
    load_rollout_file,
)


_model_path = os.path.join(os.path.dirname(__file__), "action_cvae_distill.py")
_spec = importlib.util.spec_from_file_location("action_cvae_distill", os.path.abspath(_model_path))
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
StateActionCVAE = _mod.StateActionCVAE
diag_gaussian_kl = _mod.diag_gaussian_kl


def load_dataset(paths: list[str], ckpt: dict, max_samples: int, seed: int):
    obs_chunks = []
    action_chunks = []
    phase_chunks = []
    source_chunks = []
    for source_id, path in enumerate(paths):
        obs, actions, phases, _ = load_rollout_file(
            path,
            ckpt.get("obs_key", "obs_v10"),
            ckpt.get("action_key", "actions_teacher"),
            ckpt.get("phase_key", "phase_id"),
            action_horizon=int(ckpt.get("action_horizon", 1)),
        )
        if obs.shape[1] != int(ckpt.get("base_obs_dim", obs.shape[1])):
            raise ValueError(
                f"{path}: base obs_dim={obs.shape[1]} differs from checkpoint base_obs_dim={ckpt.get('base_obs_dim')}"
            )
        obs = apply_obs_slices(obs.float(), ckpt["obs_slices"])
        if obs.shape[1] != int(ckpt["obs_dim"]):
            raise ValueError(f"{path}: selected obs_dim={obs.shape[1]} differs from checkpoint obs_dim={ckpt['obs_dim']}")
        obs_chunks.append(obs)
        action_chunks.append(actions.float())
        phase_chunks.append(phases.long())
        source_chunks.append(torch.full((obs.shape[0],), source_id, dtype=torch.long))

    obs = torch.cat(obs_chunks, dim=0)
    actions = torch.cat(action_chunks, dim=0)
    phases = torch.cat(phase_chunks, dim=0)
    sources = torch.cat(source_chunks, dim=0)

    if max_samples and max_samples > 0 and obs.shape[0] > max_samples:
        generator = torch.Generator().manual_seed(seed)
        idx = torch.randperm(obs.shape[0], generator=generator)[:max_samples]
        obs, actions, phases, sources = obs[idx], actions[idx], phases[idx], sources[idx]
    return obs, actions, phases, sources


@torch.no_grad()
def compute_metrics(model, obs, actions, phases, sources, batch_size: int, device: str):
    loader = DataLoader(TensorDataset(obs, actions, phases, sources), batch_size=batch_size, shuffle=False)
    post_errors = []
    prior_errors = []
    kls = []
    latent_maha = []
    phase_out = []
    source_out = []
    model.eval()
    for obs_b, act_b, phase_b, source_b in loader:
        obs_b = obs_b.to(device)
        act_b = act_b.to(device)
        q_mu, q_logvar = model.encode(obs_b, act_b)
        p_mu, p_logvar = model.prior_stats(obs_b)
        post_recon = model.decode(obs_b, q_mu)
        prior_recon = model.decode(obs_b, p_mu)
        post_errors.append((post_recon - act_b).pow(2).mean(dim=-1).cpu())
        prior_errors.append((prior_recon - act_b).pow(2).mean(dim=-1).cpu())
        kls.append(diag_gaussian_kl(q_mu, q_logvar, p_mu, p_logvar).cpu())
        latent_maha.append((((q_mu - p_mu) ** 2) / p_logvar.exp().clamp(min=1.0e-8)).sum(dim=-1).sqrt().cpu())
        phase_out.append(phase_b.cpu())
        source_out.append(source_b.cpu())
    return {
        "posterior_mse": torch.cat(post_errors),
        "prior_mse": torch.cat(prior_errors),
        "kl": torch.cat(kls),
        "latent_maha": torch.cat(latent_maha),
        "phase": torch.cat(phase_out),
        "source": torch.cat(source_out),
    }


def summarize(name: str, values: torch.Tensor) -> str:
    return (
        f"{name}: mean={values.mean().item():.6f}, "
        f"p50={values.quantile(0.50).item():.6f}, "
        f"p90={values.quantile(0.90).item():.6f}, "
        f"p99={values.quantile(0.99).item():.6f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate action CVAE distillation checkpoint.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--rollout_data", type=str, nargs="+", required=True)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    obs, actions, phases, sources = load_dataset(args.rollout_data, ckpt, args.max_samples, args.seed)
    obs = (obs - ckpt["obs_mean"]) / ckpt["obs_std"]
    actions = (actions - ckpt["action_mean"]) / ckpt["action_std"]

    model = StateActionCVAE(
        obs_dim=int(ckpt["obs_dim"]),
        action_dim=int(ckpt["action_dim"]),
        latent_dim=int(ckpt["latent_dim"]),
        hidden_dims=list(ckpt["hidden_dims"]),
    ).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])

    metrics = compute_metrics(model, obs, actions, phases, sources, args.batch_size, args.device)

    print(f"Model: {args.model}")
    print(f"Data: n={obs.shape[0]}, obs_dim={obs.shape[1]}, action_dim={actions.shape[1]}")
    print(
        f"obs_key={ckpt.get('obs_key')}, obs_slices={ckpt.get('obs_slices')}, "
        f"latent_dim={ckpt.get('latent_dim')}, action_horizon={ckpt.get('action_horizon', 1)}, "
        f"base_action_dim={ckpt.get('base_action_dim', ckpt.get('action_dim'))}"
    )
    print(summarize("posterior_mse", metrics["posterior_mse"]))
    print(summarize("prior_mean_mse", metrics["prior_mse"]))
    print(summarize("kl_q_prior", metrics["kl"]))
    print(summarize("latent_maha", metrics["latent_maha"]))

    print("\nPer-phase:")
    for phase_id, phase_name in enumerate(PHASE_NAMES):
        mask = metrics["phase"] == phase_id
        if not mask.any():
            continue
        print(
            f"  {phase_name:10s} n={mask.sum().item():6d} "
            f"post={metrics['posterior_mse'][mask].mean().item():.6f} "
            f"prior={metrics['prior_mse'][mask].mean().item():.6f} "
            f"kl={metrics['kl'][mask].mean().item():.4f} "
            f"maha={metrics['latent_maha'][mask].mean().item():.4f}"
        )

    print("\nPer-source:")
    for source_id, path in enumerate(args.rollout_data):
        mask = metrics["source"] == source_id
        if not mask.any():
            continue
        print(
            f"  {os.path.relpath(path):55s} n={mask.sum().item():6d} "
            f"post={metrics['posterior_mse'][mask].mean().item():.6f} "
            f"prior={metrics['prior_mse'][mask].mean().item():.6f} "
            f"kl={metrics['kl'][mask].mean().item():.4f}"
        )


if __name__ == "__main__":
    main()
