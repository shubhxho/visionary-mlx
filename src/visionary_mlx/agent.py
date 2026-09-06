"""Actor-critic that acts on tokenizer latents and trains inside imagined rollouts."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from visionary_mlx.config import AgentConfig


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


class ActorCritic(nn.Module):
    def __init__(self, latent_dim: int, cfg: AgentConfig | None = None):
        super().__init__()
        self.cfg = cfg or AgentConfig()
        self.actor = _mlp(latent_dim, self.cfg.hidden_dim, self.cfg.num_actions)
        self.critic = _mlp(latent_dim, self.cfg.hidden_dim, 1)

    def flatten_z(self, z: mx.array) -> mx.array:
        # (B, N, C) or (B, T, N, C)
        if z.ndim == 4:
            b, t, n, c = z.shape
            return z.reshape(b, t, n * c)
        b, n, c = z.shape
        return z.reshape(b, n * c)

    def policy(self, z: mx.array) -> tuple[mx.array, mx.array]:
        h = self.flatten_z(z)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

    def act(self, z: mx.array, greedy: bool = False) -> tuple[mx.array, mx.array, mx.array]:
        logits, value = self.policy(z)
        if greedy:
            action = mx.argmax(logits, axis=-1)
        else:
            action = mx.random.categorical(logits)
        logp = self.log_prob(logits, action)
        return action, logp, value

    @staticmethod
    def log_prob(logits: mx.array, actions: mx.array) -> mx.array:
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        # ``one_hot`` is not part of MLX Core. Gather works across supported
        # MLX versions and avoids materialising an action-sized dense tensor.
        indices = actions.astype(mx.int32)[..., None]
        return mx.take_along_axis(logp, indices, axis=-1).squeeze(-1)

    @staticmethod
    def entropy(logits: mx.array) -> mx.array:
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        p = mx.exp(logp)
        return -mx.sum(p * logp, axis=-1)


def lambda_return(rewards: mx.array, values: mx.array, continues: mx.array, gamma: float, lam: float) -> mx.array:
    """rewards/values/continues: (B, H). returns bootstrap from last value."""
    # values includes bootstrap at the end: (B, H+1)
    b, h = rewards.shape
    gae = mx.zeros((b,))
    outs = []
    next_v = values[:, -1]
    for t in range(h - 1, -1, -1):
        cont = continues[:, t]
        delta = rewards[:, t] + gamma * cont * next_v - values[:, t]
        gae = delta + gamma * lam * cont * gae
        ret = gae + values[:, t]
        outs.append(ret)
        next_v = values[:, t]
    outs.reverse()
    return mx.stack(outs, axis=1)
