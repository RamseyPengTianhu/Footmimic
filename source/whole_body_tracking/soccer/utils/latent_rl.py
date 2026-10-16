"""PPO wrapper around the shared latent skill net (dribble line, D-a).

The shared network ``soccer.latent.latent_skill_net.LatentSkillNet`` is used unchanged.  This file only adapts it to
RSL-RL 2.3.3:

* ``LatentActorCritic`` -- actor: encoder posterior z -> tracker (deterministic z = posterior mean during PPO, so the
  action log-probability is a plain Gaussian around the tracker output); critic: LSTM + MLP on the privileged
  observation.  RSL-RL hands the actor a single observation tensor, so the env's ``policy`` group is the student
  observation (134) followed by the encoder's reference window (177); the prior and the tracker only ever see the
  first 134 columns.
* ``LatentPPO`` -- PPO whose loss additionally contains a KL(posterior || prior) term.  Its ``update`` is the core path
  of ``rsl_rl.algorithms.PPO.update`` (2.3.3) plus that term; symmetry, RND and multi-GPU are not supported and raise
  instead of silently doing nothing.

  The KL term is ``kl_coef(t) * sum_d max(free_bits, mean_batch KL_d)`` ("free bits", per z dimension) with
  ``kl_coef(t)`` 0 for the first ``kl_warmup_iters`` updates and then ramped linearly to ``kl_coef`` over
  ``kl_ramp_iters``.  D-b run 1 used a plain ``0.01 * KL`` from the start and the posterior collapsed onto the prior
  (posterior and prior gave the same tracking, KL -> 0.001, joint error flat at ~2x clean-v0 Stage 1); see
  DRIBBLE_HANDOFF.md section 17.7.

Importing this module registers both class names in the RSL-RL runner namespace (the runner resolves
``class_name`` with ``eval``); RSL-RL itself is not modified.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.networks import Memory
from rsl_rl.utils import unpad_trajectories
from torch.distributions import Normal

from soccer.latent import skill_event as se
from soccer.latent.latent_skill_net import LatentSkillNet, kl_gauss


def _mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


def _kl_per_dim(mu_q, lv_q, mu_p, lv_p) -> torch.Tensor:
    """Per-dimension KL(q || p) for diagonal Gaussians -- the summand of the shared ``kl_gauss``."""
    return 0.5 * (lv_p - lv_q + (lv_q.exp() + (mu_q - mu_p) ** 2) / lv_p.exp() - 1.0)


class LatentActorCritic(nn.Module):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        student_dim: int = se.OBS_DIM,
        ref_dim: int = se.REF_DIM,
        z_dim: int = 8,
        critic_hidden_dims=(512, 256, 128),
        critic_rnn_hidden_dim: int = 256,
        init_noise_std: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        if num_actor_obs != student_dim + ref_dim:
            raise ValueError(
                f"actor observation must be student ({student_dim}) + reference window ({ref_dim}) = "
                f"{student_dim + ref_dim}, got {num_actor_obs}"
            )
        self.student_dim, self.ref_dim = student_dim, ref_dim
        self.net = LatentSkillNet(obs_dim=student_dim, ref_dim=ref_dim, act_dim=num_actions, z_dim=z_dim)
        self.memory_c = Memory(num_critic_obs, type="lstm", num_layers=1, hidden_size=critic_rnn_hidden_dim)
        self.critic = _mlp([critic_rnn_hidden_dim, *critic_hidden_dims, 1])
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution: Normal | None = None
        # Actor memory packed as one LSTM-style (h, c) pair: prior layers then tracker layers along the layer axis.
        # RSL-RL's storage indexes actor and critic hidden tuples together, so both must have two tensors.
        if self.net.prior_rnn.hidden_size != self.net.dec_rnn.hidden_size:
            raise ValueError("prior and tracker hidden sizes must match to pack the actor memory")
        self.n_prior_layers = self.net.prior_rnn.num_layers
        self.hidden_a: tuple[torch.Tensor, torch.Tensor] | None = None
        self.use_prior = False  # evaluation switch: run on the prior instead of the encoder
        self.kl_batch: torch.Tensor | None = None  # per-sample KL of the last batch-mode forward (for logging)
        self.kl_batch_dims: torch.Tensor | None = None  # per-sample, per-z-dimension KL (for the free-bits loss)
        self.last_kl = torch.zeros(())  # mean KL of the last rollout step (diagnostics)
        Normal.set_default_validate_args(False)

    # --- helpers ---------------------------------------------------------------------------------------
    def _split(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return obs[..., : self.student_dim], obs[..., self.student_dim :]

    def _as_dict(self, h: tuple | list) -> dict:
        k = self.n_prior_layers
        hh, cc = h[0], h[1]
        return {
            "prior": (hh[:k].contiguous(), cc[:k].contiguous()),
            "tracker": (hh[k:].contiguous(), cc[k:].contiguous()),
        }

    @staticmethod
    def _as_tuple(h: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.cat([h["prior"][0], h["tracker"][0]], 0), torch.cat([h["prior"][1], h["tracker"][1]], 0)

    def _rollout_step(self, obs: torch.Tensor) -> torch.Tensor:
        stud, ref = self._split(obs)
        if self.training:
            self.net.update_norms(stud, ref)
        hidden = self.net.init_hidden(obs.shape[0], obs.device) if self.hidden_a is None else self._as_dict(self.hidden_a)
        mean, st, hidden = self.net.step(
            stud, hidden, ref=None if self.use_prior else ref, use_prior=self.use_prior, deterministic=True
        )
        self.hidden_a = self._as_tuple(hidden)
        self.last_kl = kl_gauss(st["mu_q"], st["lv_q"], st["mu_p"], st["lv_p"]).mean().detach()
        return mean

    def _batch_forward(self, obs: torch.Tensor, masks: torch.Tensor, hidden_states) -> torch.Tensor:
        """obs: padded trajectories [T, B, D]; hidden_states: packed actor (h, c) at each trajectory start."""
        stud, ref = self._split(obs)
        hidden = self._as_dict(hidden_states)
        means, kls = [], []
        for t in range(obs.shape[0]):
            mean_t, st, hidden = self.net.step(stud[t], hidden, ref=ref[t], use_prior=False, deterministic=True)
            means.append(mean_t)
            kls.append(_kl_per_dim(st["mu_q"], st["lv_q"], st["mu_p"], st["lv_p"]))
        mean = unpad_trajectories(torch.stack(means), masks)
        self.kl_batch_dims = unpad_trajectories(torch.stack(kls), masks)  # [T, B, z]
        self.kl_batch = self.kl_batch_dims.sum(-1)
        return mean

    # --- RSL-RL actor-critic interface -------------------------------------------------------------------
    def reset(self, dones=None):
        if self.hidden_a is not None:
            if dones is None:
                self.hidden_a = None
            else:
                for h in self.hidden_a:
                    h[..., dones == 1, :] = 0.0
        self.memory_c.reset(dones)

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def act(self, observations, masks=None, hidden_states=None):
        mean = self._rollout_step(observations) if masks is None else self._batch_forward(observations, masks, hidden_states)
        self.distribution = Normal(mean, mean * 0.0 + self.std)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        return self._rollout_step(observations)

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        return self.critic(input_c.squeeze(0))

    def get_hidden_states(self):
        return self.hidden_a, self.memory_c.hidden_states

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True


class LatentPPO(PPO):
    """PPO + KL(posterior || prior) on the latent z.  See the module docstring."""

    def __init__(
        self,
        policy,
        kl_coef: float = 0.01,
        kl_free_bits: float = 0.0,
        kl_warmup_iters: int = 0,
        kl_ramp_iters: int = 0,
        **kwargs,
    ):
        super().__init__(policy, **kwargs)
        if self.rnd or self.symmetry or self.is_multi_gpu:
            raise NotImplementedError("LatentPPO does not support RND, symmetry or multi-GPU")
        self.kl_coef = float(kl_coef)
        self.kl_free_bits = float(kl_free_bits)  # nats per z dimension
        self.kl_warmup_iters = int(kl_warmup_iters)
        self.kl_ramp_iters = int(kl_ramp_iters)
        self.num_updates = 0  # one per learning iteration; not saved in checkpoints, so a resumed run restarts the warm-up

    def current_kl_coef(self) -> float:
        t = self.num_updates - self.kl_warmup_iters
        if t < 0:
            return 0.0
        if self.kl_ramp_iters <= 0:
            return self.kl_coef
        return self.kl_coef * min(1.0, t / self.kl_ramp_iters)

    def kl_loss(self) -> torch.Tensor:
        """Free-bits KL: sum over z dimensions of max(free_bits, batch-mean KL of that dimension)."""
        per_dim = self.policy.kl_batch_dims.reshape(-1, self.policy.kl_batch_dims.shape[-1]).mean(0)
        return torch.clamp(per_dim, min=self.kl_free_bits).sum()

    def update(self):  # noqa: C901
        mean_value_loss = mean_surrogate_loss = mean_entropy = mean_kl = 0.0
        kl_coef_now = self.current_kl_coef()
        generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch,
            old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, _rnd,
        ) in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch, sigma_batch = self.policy.action_mean, self.policy.action_std
            entropy_batch = self.policy.entropy
            kl_latent = self.policy.kl_batch.mean()  # logged (raw KL, nats)
            kl_term = self.kl_loss()

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    ).mean()
                    if kl > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif 0.0 < kl < self.desired_kl / 2.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.max((value_batch - returns_batch).pow(2), (value_clipped - returns_batch).pow(2)).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + kl_coef_now * kl_term
            )
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_kl += kl_latent.item()

        n = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        self.num_updates += 1
        return {
            "latent_kl_coef": kl_coef_now,
            "value_function": mean_value_loss / n,
            "surrogate": mean_surrogate_loss / n,
            "entropy": mean_entropy / n,
            "latent_kl": mean_kl / n,
        }


def register() -> None:
    """Expose the class names to RSL-RL's runner, which resolves ``class_name`` with ``eval``."""
    import rsl_rl.runners.on_policy_runner as runner_module

    runner_module.LatentActorCritic = LatentActorCritic
    runner_module.LatentPPO = LatentPPO


register()
