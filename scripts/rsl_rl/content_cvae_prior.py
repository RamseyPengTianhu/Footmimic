"""Content-conditioned CVAE motion prior for offline diagnostics."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentConditionedVAE(nn.Module):
    """Small conditional VAE for short motion feature windows."""

    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim + cond_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([x, cond], dim=-1))
        return self.mu(h), self.logvar(h).clamp(-8.0, 8.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, cond], dim=-1))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond)
        return recon, mu, logvar


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, reconstruction, and KL losses."""
    recon_loss = F.mse_loss(recon, x)
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl


@torch.no_grad()
def reconstruction_error(model: ContentConditionedVAE, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    """Per-sample normalized reconstruction error for diagnostics."""
    mu, _ = model.encode(x, cond)
    recon = model.decode(mu, cond)
    return torch.mean((recon - x) ** 2, dim=-1)
