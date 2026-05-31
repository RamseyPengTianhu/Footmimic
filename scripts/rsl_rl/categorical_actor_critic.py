"""Categorical Actor-Critic for VQ code selection.

The actor outputs K *residual* logits via an RNN+MLP, which are added to
prior logits (appended to the observation by the wrapper):

    combined = prior_logits + scale * residual_logits
    code ~ Categorical(combined / temperature)

This preserves the prior's safety bias while letting PPO learn corrections.
The prior logits go through the same unpad_trajectories transform as the
RNN output, ensuring shapes match in recurrent mini-batches.

``action_mean`` and ``action_std`` return dummy [B,1] tensors for RSL-RL
PPO storage/logging compatibility.  Use ``schedule="fixed"`` to avoid
the Gaussian-KL adaptive LR logic.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from rsl_rl.networks import Memory
from rsl_rl.utils import resolve_nn_activation


class CategoricalActorCriticRecurrent(nn.Module):
    """RNN actor-critic with categorical action head for VQ code selection."""

    is_recurrent = True

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,  # ignored – we use num_codes
        *,
        num_codes: int = 16,
        ppo_logit_scale: float = 1.0,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        activation: str = "elu",
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 128,
        rnn_num_layers: int = 2,
        init_temperature: float = 1.0,
        **kwargs,  # absorb unexpected args from RSL-RL config
    ):
        if kwargs:
            print(
                "CategoricalActorCriticRecurrent.__init__ got unexpected arguments, "
                f"which will be ignored: {list(kwargs.keys())}"
            )
        super().__init__()

        if actor_hidden_dims is None:
            actor_hidden_dims = [128, 64, 32]
        if critic_hidden_dims is None:
            critic_hidden_dims = [128, 64, 32]

        self.num_codes = num_codes
        self.ppo_logit_scale = ppo_logit_scale

        # Feature dims (obs without appended prior logits)
        self.num_features = num_actor_obs - num_codes
        self.num_critic_features = num_critic_obs - num_codes

        activation_fn = resolve_nn_activation(activation)

        # ── Actor ──────────────────────────────────────────────────────────
        self.memory_a = Memory(
            self.num_features, type=rnn_type,
            num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim,
        )
        actor_layers = []
        dims = [rnn_hidden_dim] + actor_hidden_dims
        for i in range(len(dims) - 1):
            actor_layers.append(nn.Linear(dims[i], dims[i + 1]))
            actor_layers.append(activation_fn)
        actor_layers.append(nn.Linear(dims[-1], num_codes))
        self.actor = nn.Sequential(*actor_layers)

        # ── Critic ─────────────────────────────────────────────────────────
        self.memory_c = Memory(
            self.num_critic_features, type=rnn_type,
            num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim,
        )
        critic_layers = []
        dims_c = [rnn_hidden_dim] + critic_hidden_dims
        for i in range(len(dims_c) - 1):
            critic_layers.append(nn.Linear(dims_c[i], dims_c[i + 1]))
            critic_layers.append(activation_fn)
        critic_layers.append(nn.Linear(dims_c[-1], 1))
        self.critic = nn.Sequential(*critic_layers)

        # ── Temperature (exploration knob) ─────────────────────────────────
        self.log_temperature = nn.Parameter(
            torch.tensor(init_temperature).log()
        )

        # ── State ──────────────────────────────────────────────────────────
        self.distribution: Categorical | None = None

        print(f"Categorical Actor MLP: {self.actor}")
        print(f"Categorical Critic MLP: {self.critic}")
        print(f"Categorical Actor RNN: {self.memory_a}")
        print(f"Categorical Critic RNN: {self.memory_c}")
        print(f"Num codes: {num_codes}, PPO logit scale: {ppo_logit_scale}, "
              f"init temperature: {init_temperature:.2f}")

    # ── helpers ────────────────────────────────────────────────────────────

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=0.1, max=5.0)

    # ── ActorCritic API (called by RSL-RL PPO) ─────────────────────────────

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def _split_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split [features | prior_logits] observation."""
        features = obs[..., :-self.num_codes]
        prior_logits = obs[..., -self.num_codes:]
        return features, prior_logits

    def act(self, observations, masks=None, hidden_states=None):
        """Sample a code from the categorical distribution.

        Returns:
            actions: [B, 1] float tensor containing the code index.
        """
        features, prior_logits = self._split_obs(observations)
        
        # RNN processing
        input_a = self.memory_a(features, masks, hidden_states)
        residual_logits = self.actor(input_a.squeeze(0))  # [N_valid, K]

        # Ensure prior_logits matches residual_logits shape in recurrent batch mode
        if masks is not None:
            from rsl_rl.utils.utils import unpad_trajectories
            prior_logits = unpad_trajectories(prior_logits, masks)
        else:
            prior_logits = prior_logits.squeeze(0)

        # Explicit additive prior
        combined_logits = prior_logits + self.ppo_logit_scale * residual_logits
        
        # Temperature-scaled sampling
        scaled_logits = combined_logits / self.temperature
        self.distribution = Categorical(logits=scaled_logits)

        code = self.distribution.sample()  # [B]
        return code.unsqueeze(-1).float()  # [B, 1]

    def act_inference(self, observations):
        """Deterministic (argmax) code selection for evaluation."""
        features, prior_logits = self._split_obs(observations)
        
        input_a = self.memory_a(features)
        residual_logits = self.actor(input_a.squeeze(0))
        
        prior_logits = prior_logits.squeeze(0)
        combined_logits = prior_logits + self.ppo_logit_scale * residual_logits
        
        scaled_logits = combined_logits / self.temperature
        return scaled_logits.argmax(dim=-1).unsqueeze(-1).float()  # [B, 1]

    def get_actions_log_prob(self, actions):
        """Categorical log-probability of the stored code.

        Args:
            actions: [B, 1] float tensor → code index.
        Returns:
            log_prob: [B] tensor.
        """
        code = actions.squeeze(-1).long()
        return self.distribution.log_prob(code)

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        """Compute value estimate."""
        features, _ = self._split_obs(critic_observations)
        input_c = self.memory_c(features, masks, hidden_states)
        return self.critic(input_c.squeeze(0))

    # ── Properties expected by RSL-RL PPO ──────────────────────────────────
    # PPO accesses these during act() and update().
    # We return dummy values compatible with actions_shape=[1].

    @property
    def action_mean(self) -> torch.Tensor:
        """Dummy [B, 1] mean for PPO storage.  Returns temperature."""
        B = self.distribution.probs.shape[0]
        return self.temperature.expand(B, 1)

    @property
    def action_std(self) -> torch.Tensor:
        """Dummy [B, 1] std for PPO storage.  Returns temperature."""
        B = self.distribution.probs.shape[0]
        return self.temperature.expand(B, 1)

    @property
    def entropy(self) -> torch.Tensor:
        """Categorical entropy [B]."""
        return self.distribution.entropy()

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
