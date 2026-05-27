"""LATENT-style latent action models for teacher distillation.

Architecture follows the LATENT paper (Learn Athletic humanoid TEnnis skills):
  - Posterior Encoder:  E(z | obs, action)  -> N(mu_e, sigma_e)
  - Learnable Prior:    P(z | obs)          -> N(mu_p, sigma_p)
  - Decoder:            D(action | obs, z)

Key differences from the old action_cvae_distill.py:
  - Separate, clean module definitions (no legacy code)
  - Designed for both offline pre-training and online DAgger distillation
  - Prior is learnable & conditional (state-dependent), not fixed N(0,1)
  - LAB-compatible: act_with_lab() method for Stage 3 PPO
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: list[int],
    activation: type[nn.Module] = nn.ELU,
) -> nn.Sequential:
    """Build a simple MLP with activation between hidden layers."""
    layers: list[nn.Module] = []
    dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(dim, h))
        layers.append(activation())
        dim = h
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


class LatentEncoder(nn.Module):
    """Posterior encoder: q(z | obs, action) -> N(mu, sigma).

    Takes concatenated (obs, action) and outputs mean + logvar of latent z.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        z_dim: int = 16,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        self.net = _build_mlp(obs_dim + action_dim, 2 * z_dim, hidden_dims)
        self.z_dim = z_dim

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, logvar), each [B, z_dim]."""
        stats = self.net(torch.cat([obs, action], dim=-1))
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar.clamp(-8.0, 8.0)


class LatentPrior(nn.Module):
    """Learnable conditional prior (MLP): P(z | obs) -> N(mu_p, sigma_p).

    At deployment, this replaces the encoder (no teacher action available).
    Single-frame, no temporal memory.
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
        self.net = _build_mlp(obs_dim, 2 * z_dim, hidden_dims)
        self.z_dim = z_dim

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu_p, logvar_p), each [B, z_dim]."""
        stats = self.net(obs)
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar.clamp(-8.0, 8.0)


class LatentPriorLSTM(nn.Module):
    """Learnable conditional prior with LSTM: P(z | obs, h) -> N(mu_p, sigma_p).

    Uses LSTM hidden state to capture temporal context (phase, timing).
    This addresses the fundamental limitation of MLP prior: teacher policy
    is LSTM-based, so the same obs can correspond to different actions
    depending on the hidden state (approach vs plant vs swing vs recovery).

    Architecture:
        obs -> LSTM -> MLP head -> (mu_p, logvar_p)

    Hidden state management:
        - reset_hidden(batch_size): initialize h,c to zeros
        - reset_hidden_at(dones): reset specific envs on episode end
        - forward() updates hidden state in-place
    """

    def __init__(
        self,
        obs_dim: int,
        z_dim: int = 16,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
        )
        # MLP head: LSTM output -> (mu, logvar)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ELU(),
            nn.Linear(128, 2 * z_dim),
        )

        # Hidden state (h, c) — managed externally during rollout
        self._h: torch.Tensor | None = None
        self._c: torch.Tensor | None = None

    def reset_hidden(self, batch_size: int, device: torch.device | str = "cuda"):
        """Initialize hidden state to zeros for all envs."""
        self._h = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=device)
        self._c = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=device)

    def reset_hidden_at(self, dones: torch.Tensor):
        """Reset hidden state for envs that just terminated."""
        if self._h is None:
            return
        if dones.any():
            done_idx = dones.nonzero(as_tuple=True)[0]
            self._h[:, done_idx, :] = 0.0
            self._c[:, done_idx, :] = 0.0

    def get_hidden(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return current hidden state (for checkpointing)."""
        if self._h is None:
            return None
        return (self._h.clone(), self._c.clone())

    def set_hidden(self, hidden: tuple[torch.Tensor, torch.Tensor]):
        """Restore hidden state."""
        self._h, self._c = hidden[0].clone(), hidden[1].clone()

    def forward(
        self,
        obs: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Supports both online (single-step) and batch (sequence) modes.

        Online mode: obs is [B, obs_dim]
            Uses self._h, self._c as hidden state. Updates in-place.
            Returns (mu_p, logvar_p) each [B, z_dim].

        Batch mode: obs is [B, T, obs_dim], hidden is provided
            Processes full sequence. Returns (mu_p, logvar_p) each [B, T, z_dim].
        """
        if obs.dim() == 2:
            # Online: [B, obs_dim] -> [B, 1, obs_dim]
            B = obs.shape[0]
            obs_seq = obs.unsqueeze(1)
            # Use stored hidden only if batch size matches (online rollout).
            # Otherwise use fresh zeros (training with random batch from buffer).
            if self._h is not None and self._h.shape[1] == B:
                lstm_out, (self._h, self._c) = self.lstm(obs_seq, (self._h, self._c))
            else:
                h0 = torch.zeros(self.lstm_layers, B, self.lstm_hidden, device=obs.device)
                c0 = torch.zeros(self.lstm_layers, B, self.lstm_hidden, device=obs.device)
                lstm_out, _ = self.lstm(obs_seq, (h0, c0))
            stats = self.head(lstm_out.squeeze(1))  # [B, 2*z_dim]
        elif obs.dim() == 3:
            # Batch: [B, T, obs_dim]
            if hidden is not None:
                lstm_out, _ = self.lstm(obs, hidden)
            else:
                lstm_out, _ = self.lstm(obs)
            stats = self.head(lstm_out)  # [B, T, 2*z_dim]
        else:
            raise ValueError(f"Expected 2D or 3D obs, got {obs.dim()}D")

        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar.clamp(-8.0, 8.0)


class LatentDecoder(nn.Module):
    """Action decoder: D(action | obs, z).

    Reconstructs teacher action from state and sampled latent code.
    """

    def __init__(
        self,
        obs_dim: int,
        z_dim: int = 16,
        action_dim: int = 29,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        self.net = _build_mlp(obs_dim + z_dim, action_dim, hidden_dims)

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Returns reconstructed action [B, action_dim]."""
        return self.net(torch.cat([obs, z], dim=-1))


class LatentActionModel(nn.Module):
    """Complete LATENT-style model: Encoder + Prior + Decoder.

    Provides methods for:
      - forward(): full CVAE pass (training)
      - act_prior_mean(): deterministic action from prior mean (eval / Stage 2B rollout)
      - act_prior_sample(): stochastic action from prior sample
      - act_with_lab(): LAB-constrained action from high-level policy output (Stage 3)

    Supports three decoder_obs_mode:
      - 'full':  encoder/prior/decoder see full obs_v3 (160D), including motion reference
      - 'task':  encoder/prior/decoder see only proprioception (99D), no motion reference
                 This forces the latent z to encode all kick-relevant information,
                 removing dependency on motion reference at the decoder level.
      - 'task_features':  encoder/prior/decoder see proprioception (99D) + ball-foot
                 relation (22D) = 121D.  The 22D task features are computed from live
                 sim state by compute_task_features.compute_ball_foot_relation() and
                 passed in via the `task_features` parameter.  These features are
                 shared across all kick motions and available at deployment.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 29,
        z_dim: int = 16,
        hidden_dims: list[int] | None = None,
        decoder_obs_mode: str = "full",
        prior_type: str = "mlp",
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.obs_dim = obs_dim  # always the raw env obs dim (e.g. 160)
        self.action_dim = action_dim
        self.z_dim = z_dim
        self.hidden_dims = list(hidden_dims)
        self.decoder_obs_mode = decoder_obs_mode
        self.prior_type = prior_type

        # Compute the actual input dim for encoder/prior/decoder
        self.decoder_obs_dim = self._compute_decoder_obs_dim(obs_dim, decoder_obs_mode)

        self.encoder = LatentEncoder(self.decoder_obs_dim, action_dim, z_dim, hidden_dims)
        if prior_type == "lstm":
            self.prior = LatentPriorLSTM(
                self.decoder_obs_dim, z_dim,
                lstm_hidden=lstm_hidden, lstm_layers=lstm_layers,
            )
        else:
            self.prior = LatentPrior(self.decoder_obs_dim, z_dim, hidden_dims)
        self.decoder = LatentDecoder(self.decoder_obs_dim, z_dim, action_dim, hidden_dims)

    @staticmethod
    def _compute_decoder_obs_dim(obs_dim: int, mode: str) -> int:
        if mode == "full":
            return obs_dim
        if mode == "task":
            # Remove 58D motion command (0:58) and 3D motion_ref_ang_vel (61:64)
            # Keep: projected_gravity(3) + base_ang_vel(3) + joint_pos(29) + joint_vel(29)
            #       + prev_action(29) + ball_pos(3) + target_dir(3) = 99D
            return 3 + (obs_dim - 64)  # = obs_dim - 61
        if mode == "task_features":
            # proprio (99D) + task_features (26D from compute_task_features)
            from compute_task_features import TASK_FEATURES_DIM
            return 3 + (obs_dim - 64) + TASK_FEATURES_DIM  # = 99 + 26 = 125
        raise ValueError(f"Unknown decoder_obs_mode={mode!r}")

    def select_decoder_obs(
        self, obs_v3: torch.Tensor, task_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Extract the observation subset used by encoder/prior/decoder.

        obs_v3 layout (160D):
          0:58   motion reference command      (removed in 'task'/'task_features' mode)
          58:61  projected gravity
          61:64  motion reference angular vel  (removed in 'task'/'task_features' mode)
          64:    proprioception + prev_action + ball/target

        Args:
            obs_v3: raw env observation [B, obs_dim] (always full 160D)
            task_features: task features (required for 'task_features' mode)
        """
        if self.decoder_obs_mode == "full":
            return obs_v3
        if self.decoder_obs_mode == "task":
            return torch.cat((obs_v3[..., 58:61], obs_v3[..., 64:]), dim=-1)
        if self.decoder_obs_mode == "task_features":
            if task_features is None:
                raise ValueError(
                    "task_features must be provided when decoder_obs_mode='task_features'"
                )
            proprio = torch.cat((obs_v3[..., 58:61], obs_v3[..., 64:]), dim=-1)  # 99D
            return torch.cat((proprio, task_features), dim=-1)
        raise ValueError(f"Unknown decoder_obs_mode={self.decoder_obs_mode!r}")


    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps."""
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(
        self,
        obs_v3: torch.Tensor,
        action: torch.Tensor,
        sample: bool = True,
        task_features: torch.Tensor | None = None,
        compute_prior_recon: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Full CVAE forward pass for training.

        Args:
            obs_v3: raw env observation [B, obs_dim] (always full 160D).
                    Will be sliced internally based on decoder_obs_mode.
            action: teacher action [B, action_dim]
            sample: whether to sample z or use mean
            task_features: [B, 26] task features (required for 'task_features' mode)
            compute_prior_recon: if True, also compute D(obs, prior_mean) for
                prior_recon loss (directly trains the deployment path)

        Returns dict with: recon, q_mu, q_logvar, p_mu, p_logvar, z,
                           and optionally prior_recon
        """
        obs = self.select_decoder_obs(obs_v3, task_features)
        q_mu, q_logvar = self.encoder(obs, action)
        p_mu, p_logvar = self.prior(obs)

        z = self.reparameterize(q_mu, q_logvar) if sample else q_mu
        recon = self.decoder(obs, z)

        result = {
            "recon": recon,
            "q_mu": q_mu,
            "q_logvar": q_logvar,
            "p_mu": p_mu,
            "p_logvar": p_logvar,
            "z": z,
        }

        # Prior recon: decode using prior mean z (the actual deployment path)
        if compute_prior_recon:
            result["prior_recon"] = self.decoder(obs, p_mu)

        return result

    def reset_prior_hidden(self, batch_size: int, device: str = "cuda"):
        """Reset LSTM prior hidden state (no-op for MLP prior)."""
        if self.prior_type == "lstm":
            self.prior.reset_hidden(batch_size, device)

    def reset_prior_hidden_at(self, dones: torch.Tensor):
        """Reset LSTM prior hidden state for terminated envs (no-op for MLP prior)."""
        if self.prior_type == "lstm":
            self.prior.reset_hidden_at(dones)

    @torch.no_grad()
    def act_prior_mean(
        self,
        obs_v3: torch.Tensor,
        task_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Deterministic action using prior mean z. For eval / Stage 2B rollout.
        For LSTM prior, this also updates the hidden state."""
        obs = self.select_decoder_obs(obs_v3, task_features)
        p_mu, _ = self.prior(obs)
        action = self.decoder(obs, p_mu)
        return action

    @torch.no_grad()
    def act_prior_sample(
        self,
        obs_v3: torch.Tensor,
        task_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Stochastic action using sampled z from prior."""
        obs = self.select_decoder_obs(obs_v3, task_features)
        p_mu, p_logvar = self.prior(obs)
        z = self.reparameterize(p_mu, p_logvar)
        action = self.decoder(obs, z)
        return action

    @torch.no_grad()
    def act_with_lab(
        self,
        obs_v3: torch.Tensor,
        a_latent: torch.Tensor,
        lab_scale: float = 2.0,
        task_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """LAB-constrained action for Stage 3 PPO.

        LATENT Eq. (4): a_full = D(obs, mu_p + lambda * sigma_p * tanh(a_latent))
        For LSTM prior, this also updates the hidden state.

        Args:
            obs_v3: raw env observation [B, obs_dim] (always full, sliced internally)
            a_latent: raw PPO output [B, z_dim]
            lab_scale: lambda, controls exploration range around prior mean
            task_features: [B, 22] ball-foot relation (required for 'task_features' mode)
        """
        obs = self.select_decoder_obs(obs_v3, task_features)
        p_mu, p_logvar = self.prior(obs)
        p_std = torch.exp(0.5 * p_logvar)
        z = p_mu + lab_scale * p_std * torch.tanh(a_latent)
        action = self.decoder(obs, z)
        return action


# ─── Loss functions ───────────────────────────────────────────────────────────


def diag_gaussian_kl(
    q_mu: torch.Tensor,
    q_logvar: torch.Tensor,
    p_mu: torch.Tensor,
    p_logvar: torch.Tensor,
) -> torch.Tensor:
    """KL(q || p) for diagonal Gaussians. Returns per-sample KL [B]."""
    q_var = q_logvar.exp()
    p_var = p_logvar.exp()
    kl = p_logvar - q_logvar + (q_var + (q_mu - p_mu).pow(2)) / p_var.clamp(min=1e-8) - 1.0
    return 0.5 * kl.sum(dim=-1)


def latent_distill_loss(
    fwd: dict[str, torch.Tensor],
    action_target: torch.Tensor,
    beta: float = 1e-3,
    alpha_prior: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Combined distillation loss: L_recon + beta * L_KL + alpha * L_prior_recon.

    Args:
        fwd: output dict from LatentActionModel.forward()
        action_target: teacher action [B, action_dim]
        beta: KL weight (LATENT uses small beta to avoid posterior collapse)
        alpha_prior: weight for prior recon loss (directly trains deployment path).
            When > 0, fwd must contain 'prior_recon' key.

    Returns:
        dict with: total, recon, kl, and optionally prior_recon (all scalar tensors)
    """
    recon_loss = (fwd["recon"] - action_target).pow(2).mean()
    kl_loss = diag_gaussian_kl(
        fwd["q_mu"], fwd["q_logvar"], fwd["p_mu"], fwd["p_logvar"]
    ).mean()

    total = recon_loss + beta * kl_loss

    result = {"total": total, "recon": recon_loss, "kl": kl_loss}

    # Prior recon loss: ||D(obs, prior_mean) - a_teacher||²
    # Directly trains the deployment path (decoder + prior mean)
    if alpha_prior > 0.0 and "prior_recon" in fwd:
        prior_recon_loss = (fwd["prior_recon"] - action_target).pow(2).mean()
        total = total + alpha_prior * prior_recon_loss
        result["prior_recon"] = prior_recon_loss
        result["total"] = total

    return result
