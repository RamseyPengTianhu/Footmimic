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


# ─── VQ-VAE Components ────────────────────────────────────────────────────────


class VQCodebook(nn.Module):
    """Vector Quantization codebook with EMA updates.

    Maintains K embedding vectors of dimension z_dim. Given a continuous
    input z_e, finds the nearest codebook entry and returns it with a
    straight-through gradient estimator.

    Inspired by VQ-VAE (van den Oord et al.) and Neural Categorical Priors.

    Args:
        num_codes: number of discrete codes (K)
        z_dim: dimension of each code vector
        commitment_weight: weight for commitment loss ||z_e - sg(z_q)||²
        ema_decay: decay rate for exponential moving average codebook updates
    """

    def __init__(
        self,
        num_codes: int = 64,
        z_dim: int = 16,
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.z_dim = z_dim
        self.commitment_weight = commitment_weight
        self.ema_decay = ema_decay

        # Codebook embeddings
        self.embedding = nn.Embedding(num_codes, z_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_codes, 1.0 / num_codes)

        # EMA tracking
        self.register_buffer("ema_count", torch.zeros(num_codes))
        self.register_buffer("ema_weight", self.embedding.weight.clone())

    def quantize(
        self, z_e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vector quantize a continuous embedding.

        Args:
            z_e: continuous encoder output [..., z_dim]

        Returns:
            z_q: quantized embedding (same shape as z_e), with straight-through grad
            code_indices: [...] integer indices of selected codes
            vq_loss: scalar commitment + codebook loss
        """
        flat_shape = z_e.shape[:-1]
        z_e_flat = z_e.reshape(-1, self.z_dim)  # [N, z_dim]

        # Compute distances to all codebook entries
        # ||z_e - e_k||² = ||z_e||² + ||e_k||² - 2 * z_e · e_k
        dist = (
            z_e_flat.pow(2).sum(dim=-1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=-1, keepdim=False)
            - 2.0 * z_e_flat @ self.embedding.weight.t()
        )  # [N, K]

        # Nearest code
        code_indices = dist.argmin(dim=-1)  # [N]
        z_q_flat = self.embedding(code_indices)  # [N, z_dim]

        # EMA codebook update (only during training)
        if self.training:
            with torch.no_grad():
                # One-hot encode assignments
                onehot = torch.zeros(z_e_flat.shape[0], self.num_codes, device=z_e.device)
                onehot.scatter_(1, code_indices.unsqueeze(1), 1.0)

                # Update counts and weights
                self.ema_count = self.ema_decay * self.ema_count + (1 - self.ema_decay) * onehot.sum(0)
                self.ema_weight = self.ema_decay * self.ema_weight + (1 - self.ema_decay) * (onehot.t() @ z_e_flat)

                # Laplace smoothing to avoid dead codes
                n = self.ema_count.sum()
                self.ema_count = (
                    (self.ema_count + 1e-5) / (n + self.num_codes * 1e-5) * n
                )
                self.embedding.weight.data = self.ema_weight / self.ema_count.unsqueeze(1)

        # Losses
        # Commitment loss: encoder commits to codebook (gradient only to encoder)
        commitment_loss = (z_e_flat - z_q_flat.detach()).pow(2).mean()
        vq_loss = self.commitment_weight * commitment_loss

        # Straight-through estimator: gradient flows through z_q to z_e
        z_q_st = z_e_flat + (z_q_flat - z_e_flat).detach()

        # Reshape back
        z_q = z_q_st.reshape(*flat_shape, self.z_dim)
        code_indices = code_indices.reshape(*flat_shape)

        return z_q, code_indices, vq_loss

    def lookup(self, code_indices: torch.Tensor) -> torch.Tensor:
        """Look up codebook entries by index.

        Args:
            code_indices: [...] integer indices
        Returns:
            z_q: [..., z_dim] codebook vectors
        """
        return self.embedding(code_indices)

    def codebook_utilization(self) -> float:
        """Fraction of codes with non-negligible EMA count (for monitoring)."""
        active = (self.ema_count > 1.0).float().mean().item()
        return active

    def perplexity_from_indices(self, code_indices: torch.Tensor) -> float:
        """Compute perplexity from a batch of code indices.

        Perplexity = exp(H(p)) where H is entropy of the empirical code distribution.
        Measures effective number of codes used in a batch.
        - If all frames use the same code: perplexity = 1
        - If codes are uniformly distributed: perplexity = K
        """
        flat = code_indices.reshape(-1)
        counts = torch.bincount(flat, minlength=self.num_codes).float()
        probs = counts / counts.sum()
        # Entropy with log-sum stability
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum()
        return torch.exp(entropy).item()


class LatentEncoderVQ(nn.Module):
    """Deterministic encoder for VQ-VAE: E(z_e | obs, action) -> z_e.

    Unlike the Gaussian LatentEncoder, this outputs a single continuous
    vector (no mu/logvar split). The vector will be quantized by the
    VQCodebook to select a discrete primitive.
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
        self.net = _build_mlp(obs_dim + action_dim, z_dim, hidden_dims)
        self.z_dim = z_dim

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Returns continuous embedding z_e [..., z_dim]."""
        return self.net(torch.cat([obs, action], dim=-1))


class LatentPriorCategorical(nn.Module):
    """Categorical prior: P(code_t | obs_t, code_{t-1}) -> logits over K codes.

    When markov=True (first-order Markov prior), the input is augmented with
    a one-hot encoding of the previous code.  Index K serves as a START token
    for the first frame of each episode, avoiding pollution of code 0.

    At deployment, this replaces the encoder + codebook:
      1. Prior outputs logits -> select code (argmax or sample)
      2. Look up codebook embedding for that code
      3. Decode action from (obs, z_q)

    This avoids the continuous Gaussian averaging problem entirely.
    The prior must make a discrete CHOICE, not a blended average.
    """

    def __init__(
        self,
        obs_dim: int,
        num_codes: int = 64,
        hidden_dims: list[int] | None = None,
        markov: bool = False,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        self.num_codes = num_codes
        self.markov = markov
        # When markov=True, input dim += (num_codes + 1) for one-hot prev_code
        # The +1 is for the START token (index K)
        input_dim = obs_dim + (num_codes + 1 if markov else 0)
        self.net = _build_mlp(input_dim, num_codes, hidden_dims)

    @property
    def start_token(self) -> int:
        """Index used as START token (= num_codes)."""
        return self.num_codes

    def forward(
        self, obs: torch.Tensor, prev_code: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Returns logits [..., num_codes].

        Args:
            obs: [B, obs_dim] observation
            prev_code: [B] long tensor of previous code indices.
                       If None and markov=True, uses START token.
                       Ignored when markov=False.
        """
        if not self.markov:
            return self.net(obs)
        # Markov mode: concatenate one-hot(prev_code, K+1)
        if prev_code is None:
            # Use START token for all envs
            prev_code = torch.full(
                (obs.shape[0],), self.start_token,
                dtype=torch.long, device=obs.device,
            )
        import torch.nn.functional as F
        one_hot = F.one_hot(prev_code, self.num_codes + 1).float()  # [B, K+1]
        return self.net(torch.cat([obs, one_hot], dim=-1))


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
        num_codes: int = 64,
        commitment_weight: float = 0.25,
        markov_prior: bool = False,
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
        self.num_codes = num_codes
        self.markov_prior = markov_prior

        # Compute the actual input dim for encoder/prior/decoder
        self.decoder_obs_dim = self._compute_decoder_obs_dim(obs_dim, decoder_obs_mode)

        if prior_type == "vq":
            # VQ-VAE: deterministic encoder + codebook + categorical prior
            self.encoder = LatentEncoderVQ(self.decoder_obs_dim, action_dim, z_dim, hidden_dims)
            self.codebook = VQCodebook(num_codes, z_dim, commitment_weight)
            self.prior = LatentPriorCategorical(
                self.decoder_obs_dim, num_codes, hidden_dims, markov=markov_prior,
            )
        else:
            # Gaussian VAE: probabilistic encoder + gaussian prior (MLP or LSTM)
            self.encoder = LatentEncoder(self.decoder_obs_dim, action_dim, z_dim, hidden_dims)
            self.codebook = None
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
            # proprio (99D) + task_features (26D: 22D spatial + 4D phase)
            # Hardcoded to avoid importing compute_task_features which requires isaaclab runtime.
            # Must stay in sync with compute_task_features.TASK_FEATURES_DIM.
            _TASK_FEATURES_DIM = 26
            return 3 + (obs_dim - 64) + _TASK_FEATURES_DIM  # = 99 + 26 = 125
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

    def forward_sequence(
        self,
        obs_v3: torch.Tensor,
        action: torch.Tensor,
        task_features: torch.Tensor | None = None,
        compute_prior_recon: bool = False,
    ) -> dict[str, torch.Tensor]:
        """CVAE forward pass for sequence training (LSTM prior).

        Unlike forward(), this processes [B, T, D] sequences.
        - Encoder and Decoder are MLPs: naturally handle [B*T, D] (reshaped).
        - Prior is LSTM: processes [B, T, obs_dim] with temporal unrolling.

        Args:
            obs_v3: [B, T, obs_dim] sequence of observations
            action: [B, T, action_dim] teacher actions
            task_features: [B, T, 26] or None
            compute_prior_recon: if True, also compute D(obs, prior_mean)

        Returns dict with same keys as forward(), all shaped [B, T, ...]:
            recon, q_mu, q_logvar, p_mu, p_logvar, z, and optionally prior_recon
        """
        B, T = obs_v3.shape[:2]

        # Select decoder obs: works with [B, T, D] thanks to [... , start:end]
        obs = self.select_decoder_obs(obs_v3, task_features)  # [B, T, obs_dec_dim]

        # Flatten for MLP encoder/decoder
        obs_flat = obs.reshape(B * T, -1)
        act_flat = action.reshape(B * T, -1)

        # Encoder (MLP): process all frames
        q_mu_flat, q_logvar_flat = self.encoder(obs_flat, act_flat)
        q_mu = q_mu_flat.reshape(B, T, -1)
        q_logvar = q_logvar_flat.reshape(B, T, -1)

        # Prior (LSTM): process sequences with temporal context
        # Pass [B, T, obs_dim] directly — LSTM handles the sequence
        p_mu, p_logvar = self.prior(obs, hidden=None)  # [B, T, z_dim]

        # Sample z from posterior
        z = self.reparameterize(q_mu, q_logvar)  # [B, T, z_dim]

        # Decoder (MLP): process all frames
        z_flat = z.reshape(B * T, -1)
        recon_flat = self.decoder(obs_flat, z_flat)
        recon = recon_flat.reshape(B, T, -1)

        result = {
            "recon": recon,
            "q_mu": q_mu,
            "q_logvar": q_logvar,
            "p_mu": p_mu,
            "p_logvar": p_logvar,
            "z": z,
        }

        if compute_prior_recon:
            p_mu_flat = p_mu.reshape(B * T, -1)
            prior_recon_flat = self.decoder(obs_flat, p_mu_flat)
            result["prior_recon"] = prior_recon_flat.reshape(B, T, -1)

        return result

    def forward_vq(
        self,
        obs_v3: torch.Tensor,
        action: torch.Tensor,
        task_features: torch.Tensor | None = None,
        residual_noise_alpha: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """VQ-VAE forward pass for training.

        Flow:
          1. Encoder maps (obs, action) -> z_e (continuous embedding)
          2. Codebook quantizes z_e -> z_q (nearest code) + code_indices
          3. Decoder reconstructs action from (obs, z_q + noise)
          4. Prior predicts categorical distribution over codes from obs alone
          5. Loss = recon + vq_commitment + prior_crossentropy

        Args:
            obs_v3: [B, obs_dim] raw env observation
            action: [B, action_dim] teacher action
            task_features: [B, 26] or None
            residual_noise_alpha: if > 0, add N(0, alpha^2) noise to z_q
                before decoding. Trains decoder to handle z_q + residual.

        Returns dict with:
            recon: [B, action_dim] reconstructed action
            z_e: [B, z_dim] continuous encoder output
            z_q: [B, z_dim] quantized code (straight-through)
            code_indices: [B] selected code index
            vq_loss: scalar commitment loss
            prior_logits: [B, K] categorical logits from prior
            prior_recon: [B, action_dim] action decoded from prior's argmax code
        """
        obs = self.select_decoder_obs(obs_v3, task_features)

        # Encoder -> continuous embedding
        z_e = self.encoder(obs, action)  # [B, z_dim]

        # Quantize
        z_q, code_indices, vq_loss = self.codebook.quantize(z_e)  # [B, z_dim], [B], scalar

        # Optional: add noise around z_q to train residual-aware decoder
        if residual_noise_alpha > 0.0 and self.training:
            noise = residual_noise_alpha * torch.randn_like(z_q)
            z_decode = z_q + noise
        else:
            z_decode = z_q

        # Decode from (possibly noised) quantized code
        recon = self.decoder(obs, z_decode)  # [B, action_dim]

        # Prior: predict which code to use from obs alone
        prior_logits = self.prior(obs)  # [B, K]

        # Prior deployment path: decode from prior's top code (no noise)
        prior_code = prior_logits.argmax(dim=-1)  # [B]
        prior_z_q = self.codebook.lookup(prior_code)  # [B, z_dim]
        prior_recon = self.decoder(obs, prior_z_q)  # [B, action_dim]

        return {
            "recon": recon,
            "z_e": z_e,
            "z_q": z_q,
            "code_indices": code_indices,
            "vq_loss": vq_loss,
            "prior_logits": prior_logits,
            "prior_recon": prior_recon,
        }

    def forward_vq_hold(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        mask: torch.Tensor,
        task_features_seq: torch.Tensor | None = None,
        code_hold: int = 2,
        residual_noise_alpha: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """VQ-VAE forward pass for hold-reconstruction training.

        Encodes at every `code_hold` frames and reuses the same z_q for
        subsequent frames. This trains the decoder to treat a single code
        as a multi-frame primitive.

        Args:
            obs_seq: [B, T, obs_dim] observation sequence
            action_seq: [B, T, action_dim] teacher action sequence
            mask: [B, T] bool — True for valid timesteps
            task_features_seq: [B, T, feat_dim] or None
            code_hold: re-encode every N frames (default=2)

        Returns dict with:
            recon_seq: [B, T, action_dim] reconstructed actions
            z_e_seq: [B, T, z_dim] continuous encoder outputs (all frames)
            code_indices_0: [B] code selected at frame 0
            vq_loss: scalar commitment loss (from frame-0 encoding)
            prior_logits_seq: [B, T, K] prior logits at all frames
            prior_recon_seq: [B, T, action_dim] prior decoded actions
            mask: [B, T] pass-through
        """
        B, T, _ = obs_seq.shape

        # Prepare decoder obs for all frames
        dec_obs_seq = []
        for t in range(T):
            tf_t = task_features_seq[:, t] if task_features_seq is not None else None
            dec_obs_seq.append(self.select_decoder_obs(obs_seq[:, t], task_features=tf_t))
        dec_obs_all = torch.stack(dec_obs_seq, dim=1)  # [B, T, dec_obs_dim]

        # Encode ALL frames to get z_e (needed for switch penalty)
        z_e_list = []
        for t in range(T):
            z_e_t = self.encoder(dec_obs_all[:, t], action_seq[:, t])  # [B, z_dim]
            z_e_list.append(z_e_t)
        z_e_seq = torch.stack(z_e_list, dim=1)  # [B, T, z_dim]

        # Quantize at hold boundaries and hold z_q across frames
        recon_list = []
        code_indices_0 = None
        total_vq_loss = 0.0
        n_quant = 0
        held_zq = None

        for t in range(T):
            if t % code_hold == 0:
                z_q_t, code_t, vq_loss_t = self.codebook.quantize(z_e_seq[:, t])
                held_zq = z_q_t
                total_vq_loss = total_vq_loss + vq_loss_t
                n_quant += 1
                if t == 0:
                    code_indices_0 = code_t
            # Optional noise for residual-aware decoder training
            if residual_noise_alpha > 0.0 and self.training:
                z_decode = held_zq + residual_noise_alpha * torch.randn_like(held_zq)
            else:
                z_decode = held_zq
            recon_t = self.decoder(dec_obs_all[:, t], z_decode)
            recon_list.append(recon_t)

        recon_seq = torch.stack(recon_list, dim=1)  # [B, T, action_dim]
        avg_vq_loss = total_vq_loss / max(n_quant, 1)

        # Prior logits + prior recon at all frames
        prior_logits_list = []
        prior_recon_list = []
        for t in range(T):
            pl_t = self.prior(dec_obs_all[:, t])  # [B, K]
            pc_t = pl_t.argmax(dim=-1)
            pz_t = self.codebook.lookup(pc_t)
            pr_t = self.decoder(dec_obs_all[:, t], pz_t)
            prior_logits_list.append(pl_t)
            prior_recon_list.append(pr_t)
        prior_logits_seq = torch.stack(prior_logits_list, dim=1)  # [B, T, K]
        prior_recon_seq = torch.stack(prior_recon_list, dim=1)    # [B, T, action_dim]

        return {
            "recon_seq": recon_seq,
            "z_e_seq": z_e_seq,
            "code_indices_0": code_indices_0,
            "vq_loss": avg_vq_loss,
            "prior_logits_seq": prior_logits_seq,
            "prior_recon_seq": prior_recon_seq,
            "mask": mask,
        }


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
        For LSTM prior, this also updates the hidden state.
        For VQ prior, uses argmax code selection."""
        obs = self.select_decoder_obs(obs_v3, task_features)
        if self.prior_type == "vq":
            logits = self.prior(obs)  # [B, K]
            code = logits.argmax(dim=-1)  # [B]
            z_q = self.codebook.lookup(code)  # [B, z_dim]
            return self.decoder(obs, z_q)
        p_mu, _ = self.prior(obs)
        action = self.decoder(obs, p_mu)
        return action

    @torch.no_grad()
    def act_prior_sample(
        self,
        obs_v3: torch.Tensor,
        task_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Stochastic action using sampled z from prior.
        For VQ prior, uses Gumbel-softmax sampling."""
        obs = self.select_decoder_obs(obs_v3, task_features)
        if self.prior_type == "vq":
            logits = self.prior(obs)  # [B, K]
            # Gumbel-max sampling for stochastic code selection
            gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
            code = (logits + gumbel).argmax(dim=-1)  # [B]
            z_q = self.codebook.lookup(code)  # [B, z_dim]
            return self.decoder(obs, z_q)
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


def latent_distill_loss_masked(
    fwd: dict[str, torch.Tensor],
    action_target: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 1e-3,
    alpha_prior: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Sequence-aware distillation loss with done mask.

    Same as latent_distill_loss but for [B, T, D] tensors with a validity mask.
    Only timesteps where mask=True contribute to the loss.

    Args:
        fwd: output dict from LatentActionModel.forward_sequence()
            All values shaped [B, T, ...]
        action_target: teacher action [B, T, action_dim]
        mask: [B, T] bool — True for valid timesteps
        beta: KL weight
        alpha_prior: weight for prior recon loss

    Returns:
        dict with: total, recon, kl, and optionally prior_recon (all scalar tensors)
    """
    num_valid = mask.sum().clamp(min=1).float()

    # Recon loss: [B, T, action_dim] -> per-timestep MSE -> masked mean
    recon_per_t = (fwd["recon"] - action_target).pow(2).mean(dim=-1)  # [B, T]
    recon_loss = (recon_per_t * mask.float()).sum() / num_valid

    # KL loss: [B, T] per-timestep KL -> masked mean
    kl_per_t = diag_gaussian_kl(
        fwd["q_mu"], fwd["q_logvar"], fwd["p_mu"], fwd["p_logvar"]
    )  # [B, T]
    kl_loss = (kl_per_t * mask.float()).sum() / num_valid

    total = recon_loss + beta * kl_loss
    result = {"total": total, "recon": recon_loss, "kl": kl_loss}

    if alpha_prior > 0.0 and "prior_recon" in fwd:
        p_recon_per_t = (fwd["prior_recon"] - action_target).pow(2).mean(dim=-1)  # [B, T]
        prior_recon_loss = (p_recon_per_t * mask.float()).sum() / num_valid
        total = total + alpha_prior * prior_recon_loss
        result["prior_recon"] = prior_recon_loss
        result["total"] = total

    return result


def vq_distill_loss(
    fwd: dict[str, torch.Tensor],
    action_target: torch.Tensor,
    alpha_prior: float = 1.0,
    alpha_prior_recon: float = 0.5,
) -> dict[str, torch.Tensor]:
    """VQ-VAE distillation loss.

    Components:
      1. recon: ||D(obs, z_q) - a_teacher||²  (reconstruction from encoder's code)
      2. vq_loss: commitment loss from codebook (already computed in forward_vq)
      3. prior_ce: CrossEntropy(prior_logits, encoder_code_indices)
         Trains the prior to predict which code the encoder would select
      4. prior_recon: ||D(obs, prior_z_q) - a_teacher||²
         Directly optimizes the deployment path

    Args:
        fwd: output dict from LatentActionModel.forward_vq()
        action_target: teacher action [B, action_dim]
        alpha_prior: weight for prior cross-entropy loss
        alpha_prior_recon: weight for prior reconstruction loss

    Returns:
        dict with: total, recon, vq_loss, prior_ce, prior_recon (all scalar tensors)
    """
    import torch.nn.functional as F

    # Reconstruction from encoder's quantized code
    recon_loss = (fwd["recon"] - action_target).pow(2).mean()

    # VQ commitment loss (from codebook)
    vq_loss = fwd["vq_loss"]

    # Prior cross-entropy: train prior to predict encoder's code
    prior_ce = F.cross_entropy(fwd["prior_logits"], fwd["code_indices"])

    # Prior reconstruction: deployment path quality
    prior_recon_loss = (fwd["prior_recon"] - action_target).pow(2).mean()

    total = recon_loss + vq_loss + alpha_prior * prior_ce + alpha_prior_recon * prior_recon_loss

    return {
        "total": total,
        "recon": recon_loss,
        "vq_loss": vq_loss,
        "prior_ce": prior_ce,
        "prior_recon": prior_recon_loss,
    }


def vq_hold_loss(
    fwd: dict[str, torch.Tensor],
    action_seq: torch.Tensor,
    alpha_prior: float = 1.0,
    alpha_prior_recon: float = 0.5,
    alpha_switch: float = 0.01,
) -> dict[str, torch.Tensor]:
    """VQ-VAE hold-reconstruction loss for sequence training.

    The forward pass (forward_vq_hold) encodes at frame 0, holds z_q
    for `code_hold` frames, and decodes all frames with the held code.

    Loss components:
      1. recon: masked MSE across all T frames (same z_q → multiple actions)
      2. vq_loss: commitment loss from codebook
      3. prior_ce: CE(prior_logits[t], code_0) for all valid frames
         (trains prior to be temporally consistent with encoder)
      4. prior_recon: masked MSE of prior-decoded actions
      5. switch_penalty: ||z_e[t] - z_e[t-1]||² for consecutive frames
         (encourages smooth z_e to reduce quantization jitter)

    Args:
        fwd: output dict from LatentActionModel.forward_vq_hold()
        action_seq: [B, T, action_dim] teacher actions
        alpha_prior: weight for prior CE loss
        alpha_prior_recon: weight for prior reconstruction
        alpha_switch: weight for z_e temporal smoothness penalty

    Returns:
        dict with: total, recon, vq_loss, prior_ce, prior_recon, switch_penalty
    """
    import torch.nn.functional as F

    mask = fwd["mask"]  # [B, T]
    mask_f = mask.float()
    n_valid = mask_f.sum().clamp(min=1)

    # 1. Hold-reconstruction: same z_q must reconstruct all T frames
    recon_err = (fwd["recon_seq"] - action_seq).pow(2).sum(dim=-1)  # [B, T]
    recon_loss = (recon_err * mask_f).sum() / n_valid

    # 2. VQ commitment loss
    vq_loss = fwd["vq_loss"]

    # 3. Prior CE: train prior at ALL frames to predict code_0
    #    This teaches the prior temporal consistency
    B, T, K = fwd["prior_logits_seq"].shape
    code_target = fwd["code_indices_0"].unsqueeze(1).expand(B, T)  # [B, T]
    logits_flat = fwd["prior_logits_seq"].reshape(B * T, K)
    target_flat = code_target.reshape(B * T)
    mask_flat = mask.reshape(B * T)
    ce_per_elem = F.cross_entropy(logits_flat, target_flat, reduction="none")
    prior_ce = (ce_per_elem * mask_flat.float()).sum() / mask_flat.float().sum().clamp(min=1)

    # 4. Prior recon
    prior_recon_err = (fwd["prior_recon_seq"] - action_seq).pow(2).sum(dim=-1)
    prior_recon_loss = (prior_recon_err * mask_f).sum() / n_valid

    # 5. Switch penalty: z_e temporal smoothness
    #    ||z_e[t] - z_e[t-1]||² for t=1..T-1, masked
    switch_penalty = torch.tensor(0.0, device=action_seq.device)
    if T > 1:
        z_e = fwd["z_e_seq"]  # [B, T, z_dim]
        z_diff = (z_e[:, 1:] - z_e[:, :-1]).pow(2).sum(dim=-1)  # [B, T-1]
        switch_mask = mask_f[:, 1:] * mask_f[:, :-1]  # Both frames must be valid
        n_switch = switch_mask.sum().clamp(min=1)
        switch_penalty = (z_diff * switch_mask).sum() / n_switch

    total = (recon_loss + vq_loss
             + alpha_prior * prior_ce
             + alpha_prior_recon * prior_recon_loss
             + alpha_switch * switch_penalty)

    return {
        "total": total,
        "recon": recon_loss,
        "vq_loss": vq_loss,
        "prior_ce": prior_ce,
        "prior_recon": prior_recon_loss,
        "switch_penalty": switch_penalty,
    }
