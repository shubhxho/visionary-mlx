"""Action-conditioned flow-matching dynamics (Visionary / Dreamer 4 shortcut forcing)."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from visionary_mlx.config import DynamicsConfig, TokenizerConfig
from visionary_mlx.layers import SpatioTemporalTransformer, temporal_rope, windowed_causal_mask


class ActionEmbedding(nn.Module):
    def __init__(self, dim: int, num_actions: int):
        super().__init__()
        self.embed = nn.Embedding(num_actions, dim)
        self.base = mx.random.normal((dim,)) * 0.02

    def __call__(self, actions: mx.array) -> mx.array:
        # actions: (B, T) int
        safe = mx.maximum(actions, 0)
        tok = self.embed(safe)
        valid = (actions >= 0)[..., None]
        return mx.where(valid, tok, 0) + self.base


class ShortcutEmbedding(nn.Module):
    def __init__(self, dim: int, max_step_size: int):
        super().__init__()
        self.max_step_size = max_step_size
        half = dim // 2
        self.step_embed = nn.Embedding(max_step_size, half)
        self.signal_embed = nn.Embedding((1 << (max_step_size - 1)) + 1, dim - half)

    def __call__(self, step_levels: mx.array, signal_levels: mx.array) -> mx.array:
        return mx.concatenate(
            [self.step_embed(step_levels), self.signal_embed(signal_levels)],
            axis=-1,
        )


class DynamicsModel(nn.Module):
    def __init__(self, cfg: DynamicsConfig | None = None, tok: TokenizerConfig | None = None):
        super().__init__()
        self.cfg = cfg or DynamicsConfig()
        tok = tok or TokenizerConfig()
        self.num_obs = tok.num_latents
        self.token_dim = tok.channel_dim
        self.action_emb = ActionEmbedding(self.cfg.model_dim, self.cfg.num_actions)
        self.shortcut_emb = ShortcutEmbedding(self.cfg.model_dim, self.cfg.max_step_size)
        self.register_tokens = mx.random.normal((self.cfg.num_registers, self.cfg.model_dim)) * 0.02
        self.obs_proj = nn.Linear(self.token_dim, self.cfg.model_dim)
        self.blocks = SpatioTemporalTransformer(
            num_layers=self.cfg.num_layers,
            dim=self.cfg.model_dim,
            num_heads=self.cfg.num_heads,
            num_kv_heads=self.cfg.num_kv_heads,
            head_dim=self.cfg.head_dim,
            mlp_hidden=self.cfg.mlp_hidden_dim,
            temporal_period=self.cfg.temporal_period,
            temporal_offset=1,
        )
        self.pred = nn.Linear(self.cfg.model_dim, self.token_dim)
        self.reward_head = nn.Linear(self.cfg.model_dim, 1)
        self.continue_head = nn.Linear(self.cfg.model_dim, 1)
        # spatial mask: action cannot see later tokens? Visionary: action is token 0, others see all including action
        n = 1 + 1 + self.cfg.num_registers + self.num_obs
        levels = mx.concatenate([mx.zeros((1,), dtype=mx.int32), mx.ones((n - 1,), dtype=mx.int32)])
        self._spatial_mask = levels[:, None] >= levels[None, :]
        self._num_tokens = n
        self._obs_offset = 2 + self.cfg.num_registers

    def __call__(
        self,
        z: mx.array,
        actions: mx.array,
        step_levels: mx.array,
        signal_levels: mx.array,
        segment_ids: mx.array | None = None,
    ) -> mx.array:
        """z: (B, T, N, C). Returns predicted clean latents of same shape."""
        b, t, n, _ = z.shape
        act = self.action_emb(actions)[:, :, None, :]
        short = self.shortcut_emb(step_levels, signal_levels)[:, :, None, :]
        reg = mx.broadcast_to(self.register_tokens, (b, t, self.cfg.num_registers, self.cfg.model_dim))
        obs = self.obs_proj(z)
        tokens = mx.concatenate([act, short, reg, obs], axis=2)

        srope = temporal_rope(self.cfg.rope_base, self.cfg.head_dim, self._num_tokens)
        trope = temporal_rope(self.cfg.rope_base, self.cfg.head_dim, t)
        tmask = windowed_causal_mask(t, self.cfg.context_length)
        if segment_ids is not None:
            same = segment_ids[:, :, None] == segment_ids[:, None, :]
            tmask = tmask[None, :, :] & same
        hidden = self.blocks(tokens, srope, self._spatial_mask, trope, tmask)
        obs_h = hidden[:, :, self._obs_offset :, :]
        return self.pred(obs_h)

    def hidden_for_heads(
        self,
        z: mx.array,
        actions: mx.array,
        step_levels: mx.array,
        signal_levels: mx.array,
    ) -> mx.array:
        b, t, _, _ = z.shape
        act = self.action_emb(actions)[:, :, None, :]
        short = self.shortcut_emb(step_levels, signal_levels)[:, :, None, :]
        reg = mx.broadcast_to(self.register_tokens, (b, t, self.cfg.num_registers, self.cfg.model_dim))
        obs = self.obs_proj(z)
        tokens = mx.concatenate([act, short, reg, obs], axis=2)
        srope = temporal_rope(self.cfg.rope_base, self.cfg.head_dim, self._num_tokens)
        trope = temporal_rope(self.cfg.rope_base, self.cfg.head_dim, t)
        tmask = windowed_causal_mask(t, self.cfg.context_length)
        hidden = self.blocks(tokens, srope, self._spatial_mask, trope, tmask)
        return hidden[:, :, self._obs_offset :, :].mean(axis=2)

    def flow_loss(self, z: mx.array, actions: mx.array) -> tuple[mx.array, dict]:
        """Shortcut / flow matching: predict clean z from noised z at random tau."""
        b, t, n, c = z.shape
        finest = self.cfg.max_step_size - 1
        step_levels = mx.full((b, t), finest, dtype=mx.int32)
        step_counts = 1 << finest
        signal_levels = mx.random.randint(0, step_counts + 1, (b, t))
        tau = signal_levels.astype(mx.float32) / float(step_counts)
        noise = mx.random.normal(z.shape)
        z_noised = tau[..., None, None] * z + (1.0 - tau[..., None, None]) * noise
        pred = self(z_noised, actions, step_levels, signal_levels)
        err = (pred - z) ** 2
        weight = 0.9 * tau[..., None, None] + 0.1
        loss = mx.mean(weight * err)
        # reward/continue are trained separately
        return loss, {"flow_mse": mx.mean(err), "pred_abs": mx.mean(mx.abs(pred))}

    def reward_loss(self, z: mx.array, actions: mx.array, rewards: mx.array, dones: mx.array) -> mx.array:
        finest = self.cfg.max_step_size - 1
        step_levels = mx.full(actions.shape, finest, dtype=mx.int32)
        k_max = 1 << finest
        signal_levels = mx.full(actions.shape, k_max, dtype=mx.int32)
        h = self.hidden_for_heads(z, actions, step_levels, signal_levels)
        r_hat = self.reward_head(h).squeeze(-1)
        c_hat = self.continue_head(h).squeeze(-1)
        r_loss = mx.mean((r_hat - rewards) ** 2)
        # dones: 1 = terminal
        c_target = 1.0 - dones
        c_loss = mx.mean(mx.logaddexp(0, c_hat) - c_target * c_hat)
        return r_loss + 0.5 * c_loss

    def generate_next(
        self,
        z_prefix: mx.array,
        actions: mx.array,
        target_index: int,
        sample_steps: int | None = None,
        context_tau: float | None = None,
    ) -> mx.array:
        """Denoise one latent frame at `target_index` given clean prefix < index."""
        sample_steps = int(sample_steps or self.cfg.sample_steps)
        context_tau = float(context_tau if context_tau is not None else self.cfg.context_tau)
        step_level = int(round(math.log2(max(sample_steps, 1))))
        step_count = 1 << step_level
        step_size = 1.0 / step_count
        finest = self.cfg.max_step_size - 1
        k_max = 1 << finest
        b, t, n, c = z_prefix.shape

        current = mx.random.normal((b, n, c))
        for s in range(step_count):
            z_in = z_prefix
            # write current noisy target
            # mlx doesn't have easy slice assign on compiled graphs; use mask blend
            pos = mx.arange(t) == target_index
            z_in = mx.where(pos[None, :, None, None], current[:, None, :, :], z_in)
            step_lv = mx.full((b, t), finest, dtype=mx.int32)
            sig_lv = mx.full((b, t), k_max, dtype=mx.int32)
            # target row uses current integration level
            step_lv = mx.where(pos[None, :], mx.array(step_level, dtype=mx.int32), step_lv)
            sig_lv = mx.where(pos[None, :], mx.array(s * (k_max // step_count), dtype=mx.int32), sig_lv)
            pred = self(z_in, actions, step_lv, sig_lv)[:, target_index]
            tau = s / step_count
            velocity = (pred - current) / max(1.0 - tau, 1e-6)
            current = current + velocity * step_size
        return current

    def rollout(self, z_context: mx.array, actions: mx.array, horizon: int) -> mx.array:
        """Open-loop latent rollout. z_context: (B, C, N, D), actions: (B, C+horizon)."""
        b, ctx, n, d = z_context.shape
        frames = [z_context[:, i] for i in range(ctx)]
        total = ctx + horizon
        # pad prefix with zeros to full length
        zeros = mx.zeros((b, n, d))
        for h in range(horizon):
            prefix_list = frames + [zeros] * (total - len(frames))
            prefix = mx.stack(prefix_list[:total], axis=1)
            nxt = self.generate_next(prefix, actions[:, :total], target_index=ctx + h)
            frames.append(nxt)
        return mx.stack(frames, axis=1)
