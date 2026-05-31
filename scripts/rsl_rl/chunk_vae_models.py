"""Continuous Chunk VAE model for action chunking.

Architecture:
  Encoder (posterior):  q(z | obs_seq[t:t+H], action_seq[t:t+H])
     - Flattened (H * obs_dim + H * action_dim) -> MLP -> mu, logvar
  Prior (snapshot):     p(z | obs_t)
     - obs_t -> MLP -> mu, logvar
  Prior (history):      p(z | obs_{t-K:t})
     - Flattened (K+1) * obs_dim -> MLP -> mu, logvar
     - Allows the prior to infer velocity, intent, and phase from
       temporal changes in ball-foot geometry.
  Decoder:             p(action_chunk | obs_t, z)
     - (obs_t, z) -> MLP -> H * action_dim (flattened chunk)

obs = decoder_obs = proprio(99D) + ball-foot task_features(22D) = 121D
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: list[int],
    activation: type[nn.Module] = nn.ELU,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(dim, h))
        layers.append(activation())
        dim = h
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


class ChunkEncoder(nn.Module):
    """Posterior: q(z | obs_chunk, action_chunk).

    Takes flattened H-frame obs + action sequence, outputs mu/logvar.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        chunk_len: int,
        z_dim: int = 16,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.z_dim = z_dim

        input_dim = chunk_len * (obs_dim + action_dim)
        self.net = _build_mlp(input_dim, 2 * z_dim, hidden_dims)

    def forward(
        self, obs_chunk: torch.Tensor, action_chunk: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs_chunk: [B, H, obs_dim]
            action_chunk: [B, H, action_dim]
        Returns:
            mu, logvar: each [B, z_dim]
        """
        B = obs_chunk.shape[0]
        x = torch.cat([
            obs_chunk.reshape(B, -1),
            action_chunk.reshape(B, -1),
        ], dim=-1)
        stats = self.net(x)
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar.clamp(-8.0, 8.0)


class ChunkPrior(nn.Module):
    """Prior: p(z | obs_t).

    Takes first-frame observation only, outputs mu/logvar.
    """

    def __init__(
        self,
        obs_dim: int,
        z_dim: int = 16,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        self.z_dim = z_dim
        self.net = _build_mlp(obs_dim, 2 * z_dim, hidden_dims)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: [B, obs_dim]  (first frame of chunk)
        Returns:
            mu, logvar: each [B, z_dim]
        """
        stats = self.net(obs)
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar.clamp(-8.0, 8.0)


class ChunkPriorHistory(nn.Module):
    """History-based Prior: p(z | obs_{t-K:t}).

    Takes a window of (K+1) past observations, flattens them, and uses
    an MLP to output mu/logvar. This allows the prior to infer velocity,
    acceleration, closing speed, and movement intent from temporal changes
    in the ball-foot geometry — without requiring explicit phase labels.

    The history window includes the current frame (t) plus K past frames,
    for a total of (K+1) frames.
    """

    def __init__(
        self,
        obs_dim: int,
        history_len: int,
        z_dim: int = 16,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        self.obs_dim = obs_dim
        self.history_len = history_len  # K past frames (total window = K+1)
        self.z_dim = z_dim

        # Input: (K+1) frames * obs_dim, flattened
        input_dim = (history_len + 1) * obs_dim
        self.net = _build_mlp(input_dim, 2 * z_dim, hidden_dims)

    def forward(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs_history: [B, K+1, obs_dim]  (past K frames + current frame)
        Returns:
            mu, logvar: each [B, z_dim]
        """
        B = obs_history.shape[0]
        x = obs_history.reshape(B, -1)
        stats = self.net(x)
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar.clamp(-8.0, 8.0)


class ChunkDecoder(nn.Module):
    """Decoder: p(action_chunk | obs_t, z).

    Takes first-frame obs + latent, outputs H-frame action sequence.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        chunk_len: int,
        z_dim: int = 16,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        self.action_dim = action_dim
        self.chunk_len = chunk_len

        output_dim = chunk_len * action_dim
        self.net = _build_mlp(obs_dim + z_dim, output_dim, hidden_dims)

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: [B, obs_dim]  (first frame)
            z:   [B, z_dim]
        Returns:
            actions: [B, H, action_dim]
        """
        B = obs.shape[0]
        x = torch.cat([obs, z], dim=-1)
        out = self.net(x)
        return out.reshape(B, self.chunk_len, self.action_dim)


class ChunkVAE(nn.Module):
    """Full Chunk VAE: encoder + prior + decoder.

    obs_dim should be decoder_obs_dim (e.g., 121D = 99D proprio + 22D task features).

    Args:
        history_len: If 0, uses snapshot prior p(z|obs_t).
                     If >0, uses history prior p(z|obs_{t-K:t}) with K=history_len.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        chunk_len: int = 8,
        z_dim: int = 16,
        hidden_dims: list[int] | None = None,
        history_len: int = 0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.z_dim = z_dim
        self.history_len = history_len

        self.encoder = ChunkEncoder(obs_dim, action_dim, chunk_len, z_dim, hidden_dims)
        if history_len > 0:
            self.prior = ChunkPriorHistory(obs_dim, history_len, z_dim, hidden_dims)
        else:
            self.prior = ChunkPrior(obs_dim, z_dim, hidden_dims)
        self.decoder = ChunkDecoder(obs_dim, action_dim, chunk_len, z_dim, hidden_dims)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor, sample: bool = True
    ) -> torch.Tensor:
        if sample:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(
        self,
        obs_chunk: torch.Tensor,
        action_chunk: torch.Tensor,
        sample: bool = True,
        obs_history: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass for training.

        Args:
            obs_chunk: [B, H, obs_dim]
            action_chunk: [B, H, action_dim]
            sample: whether to sample z or use mu
            obs_history: [B, K+1, obs_dim] history window for history prior.
                         If None and history_len>0, falls back to chunk obs.

        Returns dict with:
            recon: [B, H, action_dim]  reconstructed actions
            mu_e, logvar_e: encoder posterior params
            mu_p, logvar_p: prior params
            z: [B, z_dim]  sampled/mean latent
        """
        # Posterior
        mu_e, logvar_e = self.encoder(obs_chunk, action_chunk)
        z = self.reparameterize(mu_e, logvar_e, sample)

        # Prior
        obs_t0 = obs_chunk[:, 0]
        if self.history_len > 0 and obs_history is not None:
            mu_p, logvar_p = self.prior(obs_history)
        else:
            mu_p, logvar_p = self.prior(obs_t0)

        # Decode
        recon = self.decoder(obs_t0, z)

        return {
            "recon": recon,
            "mu_e": mu_e,
            "logvar_e": logvar_e,
            "mu_p": mu_p,
            "logvar_p": logvar_p,
            "z": z,
        }

    def act_prior_chunk(
        self, obs_t: torch.Tensor, sample: bool = False,
        obs_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Prior-only inference: obs -> z -> H-frame actions.

        Args:
            obs_t: [B, obs_dim]  current observation (used for decoder)
            sample: if True, sample from prior; if False, use mean
            obs_history: [B, K+1, obs_dim] for history prior. If None,
                         uses obs_t as snapshot prior input.

        Returns:
            actions: [B, H, action_dim]
        """
        if self.history_len > 0 and obs_history is not None:
            mu_p, logvar_p = self.prior(obs_history)
        else:
            mu_p, logvar_p = self.prior(obs_t)
        z = self.reparameterize(mu_p, logvar_p, sample)
        return self.decoder(obs_t, z)


def chunk_vae_loss(
    fwd: dict[str, torch.Tensor],
    target_actions: torch.Tensor,
    beta: float = 1e-3,
) -> dict[str, torch.Tensor]:
    """Compute chunk VAE loss.

    Args:
        fwd: output from ChunkVAE.forward()
        target_actions: [B, H, action_dim] ground truth
        beta: KL weight

    Returns dict with:
        total, recon, kl, kl_raw
    """
    # Reconstruction: MSE over full chunk
    recon_loss = F.mse_loss(fwd["recon"], target_actions)

    # KL(q(z|x) || p(z|obs_0)) — analytical for two Gaussians
    mu_e, logvar_e = fwd["mu_e"], fwd["logvar_e"]
    mu_p, logvar_p = fwd["mu_p"], fwd["logvar_p"]

    kl = 0.5 * (
        logvar_p - logvar_e
        + (logvar_e.exp() + (mu_e - mu_p).pow(2)) / logvar_p.exp().clamp(min=1e-6)
        - 1.0
    ).sum(dim=-1).mean()

    total = recon_loss + beta * kl

    return {
        "total": total,
        "recon": recon_loss.detach(),
        "kl": (beta * kl).detach(),
        "kl_raw": kl.detach(),
    }
