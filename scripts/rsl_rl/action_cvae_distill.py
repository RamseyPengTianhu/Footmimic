"""State-conditioned action CVAE for teacher-policy distillation.

This is closer to LATENT/PULSE-style latent action modeling than the
kinematic content CVAE:

  posterior q(z | obs, action)
  prior     p(z | obs)
  decoder   D(action | obs, z)

The model is still offline-only. It does not import IsaacLab.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def build_mlp(input_dim: int, output_dim: int, hidden_dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = input_dim
    for hidden in hidden_dims:
        layers.append(nn.Linear(dim, hidden))
        layers.append(nn.ELU())
        dim = hidden
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


class StateActionCVAE(nn.Module):
    """Conditional VAE that reconstructs teacher actions from state and latent z."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 16,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dims = list(hidden_dims)

        self.posterior = build_mlp(obs_dim + action_dim, 2 * latent_dim, hidden_dims)
        self.prior = build_mlp(obs_dim, 2 * latent_dim, hidden_dims)
        self.decoder = build_mlp(obs_dim + latent_dim, action_dim, hidden_dims)

    @staticmethod
    def _split_stats(stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar.clamp(-8.0, 8.0)

    def encode(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._split_stats(self.posterior(torch.cat([obs, action], dim=-1)))

    def prior_stats(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._split_stats(self.prior(obs))

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([obs, z], dim=-1))

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q_mu, q_logvar = self.encode(obs, action)
        p_mu, p_logvar = self.prior_stats(obs)
        z = self.reparameterize(q_mu, q_logvar) if sample else q_mu
        recon = self.decode(obs, z)
        return recon, q_mu, q_logvar, p_mu, p_logvar

    @torch.no_grad()
    def act_prior_mean(self, obs: torch.Tensor) -> torch.Tensor:
        p_mu, _ = self.prior_stats(obs)
        return self.decode(obs, p_mu)

    @torch.no_grad()
    def act_from_latent_residual(self, obs: torch.Tensor, residual: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        p_mu, p_logvar = self.prior_stats(obs)
        p_std = torch.exp(0.5 * p_logvar)
        z = p_mu + scale * p_std * torch.tanh(residual)
        return self.decode(obs, z)


def diag_gaussian_kl(
    q_mu: torch.Tensor,
    q_logvar: torch.Tensor,
    p_mu: torch.Tensor,
    p_logvar: torch.Tensor,
) -> torch.Tensor:
    """Per-sample KL(q || p) for diagonal Gaussians."""
    q_var = q_logvar.exp()
    p_var = p_logvar.exp()
    kl = p_logvar - q_logvar + (q_var + (q_mu - p_mu).pow(2)) / p_var.clamp(min=1.0e-8) - 1.0
    return 0.5 * kl.sum(dim=-1)


def action_cvae_loss(
    recon: torch.Tensor,
    action: torch.Tensor,
    q_mu: torch.Tensor,
    q_logvar: torch.Tensor,
    p_mu: torch.Tensor,
    p_logvar: torch.Tensor,
    beta: float = 1.0e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = (recon - action).pow(2).mean()
    kl = diag_gaussian_kl(q_mu, q_logvar, p_mu, p_logvar).mean()
    return recon_loss + beta * kl, recon_loss, kl
