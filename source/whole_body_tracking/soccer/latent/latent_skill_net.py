"""Shared latent skill network (interface v1, docs/milestones/SKILL_INTERFACE_V1_PLAN_20261016_CN.md).

reference window --encoder--> z_post --+
                                       +--> tracker(obs, z) --> action
obs --recurrent prior--> z_prior ------+

- The encoder only sees the future reference window and is used in training only.
- The prior and the tracker see the same student observation (proprioception, gated ball, body
  command, contact event, soft phase, next-event preview; see `skill_event.py`), never the reference.
- Parameter names match `LatentStudent` in `scripts/rsl_rl/train_latent_student_distill.py`, so A1
  checkpoints load into this class.

This file is the single source of truth for the shoot, pass, and dribble lines. Do not fork it; propose
changes on the main line. It depends only on torch, so it can be imported without Isaac.
"""

from __future__ import annotations

import torch
import torch.nn as nn

LOGVAR_MIN, LOGVAR_MAX = -8.0, 4.0


class RunningNorm(nn.Module):
    """Running mean/variance normaliser (parallel-variance update), clipped to [-10, 10]."""

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1])
        b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = b_mean - self.mean
        total = self.count + b_n
        self.mean += delta * b_n / total
        self.var = (self.var * self.count + b_var * b_n + delta**2 * self.count * b_n / total) / total
        self.count = total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / torch.sqrt(self.var + 1e-6)).clamp(-10.0, 10.0)


def mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


def kl_gauss(mu_q, lv_q, mu_p, lv_p) -> torch.Tensor:
    """KL(q || p) for diagonal Gaussians, summed over the last dimension."""
    return 0.5 * (lv_p - lv_q + (lv_q.exp() + (mu_q - mu_p) ** 2) / lv_p.exp() - 1.0).sum(-1)


def sample_z(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return mu + torch.randn_like(mu) * (0.5 * logvar).exp()


Hidden = tuple[torch.Tensor, torch.Tensor]


class LatentSkillNet(nn.Module):
    """Encoder (MLP), recurrent prior (LSTM 1x256), tracker (LSTM 2x256)."""

    def __init__(
        self,
        obs_dim: int,
        ref_dim: int,
        act_dim: int = 29,
        z_dim: int = 8,
        prior_hidden: int = 256,
        tracker_hidden: int = 256,
        tracker_layers: int = 2,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.stud_norm = RunningNorm(obs_dim)
        self.ref_norm = RunningNorm(ref_dim)
        self.encoder = mlp([ref_dim, 256, 256, 2 * z_dim])
        self.prior_rnn = nn.LSTM(obs_dim, prior_hidden, num_layers=1)
        self.prior_head = mlp([prior_hidden, 128, 2 * z_dim])
        self.dec_rnn = nn.LSTM(obs_dim + z_dim, tracker_hidden, num_layers=tracker_layers)
        self.dec_head = mlp([tracker_hidden, 256, 128, act_dim])

    # --- components -----------------------------------------------------------------------------
    def encode(self, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(self.ref_norm(ref)).chunk(2, dim=-1)
        return mu, logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)

    def prior_step(self, obs: torch.Tensor, hidden: Hidden) -> tuple[torch.Tensor, torch.Tensor, Hidden]:
        out, hidden = self.prior_rnn(self.stud_norm(obs).unsqueeze(0), hidden)
        mu, logvar = self.prior_head(out.squeeze(0)).chunk(2, dim=-1)
        return mu, logvar.clamp(LOGVAR_MIN, LOGVAR_MAX), hidden

    def track_step(self, obs: torch.Tensor, z: torch.Tensor, hidden: Hidden) -> tuple[torch.Tensor, Hidden]:
        out, hidden = self.dec_rnn(torch.cat([self.stud_norm(obs), z], -1).unsqueeze(0), hidden)
        return self.dec_head(out.squeeze(0)), hidden

    # A1 name, kept for checkpoint and script compatibility.
    decode_step = track_step

    # --- state helpers --------------------------------------------------------------------------
    def init_hidden(self, n: int, device) -> dict[str, Hidden]:
        def zeros(rnn: nn.LSTM) -> Hidden:
            shape = (rnn.num_layers, n, rnn.hidden_size)
            return torch.zeros(shape, device=device), torch.zeros(shape, device=device)

        return {"prior": zeros(self.prior_rnn), "tracker": zeros(self.dec_rnn)}

    @staticmethod
    def reset_hidden(hidden: dict[str, Hidden], done: torch.Tensor) -> dict[str, Hidden]:
        keep = (~done.bool()).float().view(1, -1, 1)
        return {k: (h * keep, c * keep) for k, (h, c) in hidden.items()}

    @torch.no_grad()
    def update_norms(self, obs: torch.Tensor, ref: torch.Tensor | None = None) -> None:
        self.stud_norm.update(obs)
        if ref is not None:
            self.ref_norm.update(ref)

    # --- one control step -----------------------------------------------------------------------
    def step(
        self,
        obs: torch.Tensor,
        hidden: dict[str, Hidden],
        ref: torch.Tensor | None = None,
        use_prior: torch.Tensor | bool = False,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Hidden]]:
        """Advance prior and tracker by one step.

        `ref` is required unless every environment uses the prior. `use_prior` may be a per-env bool
        tensor (mixed posterior/prior batches, as in A1). The prior always steps so its memory stays
        in sync; its output is returned for the KL term.
        """
        mu_p, lv_p, h_prior = self.prior_step(obs, hidden["prior"])
        if ref is None:
            mu_q, lv_q = mu_p, lv_p
        else:
            mu_q, lv_q = self.encode(ref)
        if isinstance(use_prior, torch.Tensor):
            sel = use_prior.bool().view(-1, 1)
            mu, lv = torch.where(sel, mu_p, mu_q), torch.where(sel, lv_p, lv_q)
        elif use_prior:
            mu, lv = mu_p, lv_p
        else:
            mu, lv = mu_q, lv_q
        z = mu if deterministic else sample_z(mu, lv)
        action, h_tracker = self.track_step(obs, z, hidden["tracker"])
        stats = {"mu_q": mu_q, "lv_q": lv_q, "mu_p": mu_p, "lv_p": lv_p, "z": z}
        return action, stats, {"prior": h_prior, "tracker": h_tracker}
