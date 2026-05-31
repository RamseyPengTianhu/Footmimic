"""Phase-conditioned VAE motion prior.

The first use of this model is offline: learn whether a short motion feature
window looks like approach, prestrike, strike, or follow-through. It is not a
controller and does not generate robot actions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseConditionedVAE(nn.Module):
    """Small conditional VAE for short motion feature windows."""

    def __init__(
        self,
        input_dim: int,
        num_phases: int = 4,
        latent_dim: int = 32,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_phases = num_phases
        self.latent_dim = latent_dim

        cond_dim = input_dim + num_phases
        self.encoder = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_phases, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def _phase_onehot(self, phase_id: torch.Tensor) -> torch.Tensor:
        return F.one_hot(phase_id.long(), num_classes=self.num_phases).to(dtype=torch.float32)

    def encode(self, x: torch.Tensor, phase_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        phase = self._phase_onehot(phase_id).to(device=x.device)
        h = self.encoder(torch.cat([x, phase], dim=-1))
        return self.mu(h), self.logvar(h).clamp(-8.0, 8.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, phase_id: torch.Tensor) -> torch.Tensor:
        phase = self._phase_onehot(phase_id).to(device=z.device)
        return self.decoder(torch.cat([z, phase], dim=-1))

    def forward(self, x: torch.Tensor, phase_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, phase_id)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, phase_id)
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
def reconstruction_error(model: PhaseConditionedVAE, x: torch.Tensor, phase_id: torch.Tensor) -> torch.Tensor:
    """Per-sample normalized reconstruction error for reward/diagnostics."""
    mu, _ = model.encode(x, phase_id)
    recon = model.decode(mu, phase_id)
    return torch.mean((recon - x) ** 2, dim=-1)
