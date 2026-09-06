"""Spatiotemporal transformer pieces, ported from Visionary's Dreamer 4 stack to MLX."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def swish(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(swish(self.gate(x)) * self.up(x))


def apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Split-half RoPE matching Visionary. x: (B, T, H, D), cos/sin: (T, D/2)."""
    x_left, x_right = mx.split(x, 2, axis=-1)
    cos = cos.astype(x.dtype)[None, :, None, :]
    sin = sin.astype(x.dtype)[None, :, None, :]
    rot_left = x_left * cos - x_right * sin
    rot_right = x_right * cos + x_left * sin
    return mx.concatenate([rot_left, rot_right], axis=-1)


def temporal_rope(base: float, head_dim: int, seq_len: int) -> tuple[mx.array, mx.array]:
    half = head_dim // 2
    theta = 1.0 / (base ** (mx.arange(half).astype(mx.float32) / half))
    idx = mx.arange(seq_len).astype(mx.float32)
    angles = mx.outer(idx, theta)
    return mx.cos(angles), mx.sin(angles)


def spatial_rope(base: float, head_dim: int, x_len: int, y_len: int) -> tuple[mx.array, mx.array]:
    quarter = head_dim // 4
    theta = 1.0 / (base ** (mx.arange(quarter).astype(mx.float32) / quarter))
    n = x_len * y_len
    idx = mx.arange(n)
    x_idx = (idx % x_len).astype(mx.float32)
    y_idx = (idx // x_len).astype(mx.float32)
    x_ang = mx.outer(x_idx, theta)
    y_ang = mx.outer(y_idx, theta)
    cos = mx.concatenate([mx.cos(x_ang), mx.cos(y_ang)], axis=-1)
    sin = mx.concatenate([mx.sin(x_ang), mx.sin(y_ang)], axis=-1)
    return cos, sin


def pad_spatial_rope_for_latents(
    cos: mx.array, sin: mx.array, num_latents: int
) -> tuple[mx.array, mx.array]:
    latent_cos = mx.ones((num_latents, cos.shape[-1]), dtype=cos.dtype)
    latent_sin = mx.zeros((num_latents, sin.shape[-1]), dtype=sin.dtype)
    return mx.concatenate([latent_cos, cos], axis=0), mx.concatenate([latent_sin, sin], axis=0)


def spatial_mask(num_image: int, num_latent: int, encoder: bool) -> mx.array:
    """Bool mask, True = keep. Encoder: latents see image; decoder: image sees latents."""
    n_lat, n_img = num_latent, num_image
    lat_lat = mx.ones((n_lat, n_lat), dtype=mx.bool_)
    img_img = mx.ones((n_img, n_img), dtype=mx.bool_)
    if encoder:
        lat_img = mx.ones((n_lat, n_img), dtype=mx.bool_)
        img_lat = mx.zeros((n_img, n_lat), dtype=mx.bool_)
    else:
        lat_img = mx.zeros((n_lat, n_img), dtype=mx.bool_)
        img_lat = mx.ones((n_img, n_lat), dtype=mx.bool_)
    top = mx.concatenate([lat_lat, lat_img], axis=1)
    bot = mx.concatenate([img_lat, img_img], axis=1)
    return mx.concatenate([top, bot], axis=0)


def causal_mask(t: int) -> mx.array:
    return mx.tril(mx.ones((t, t), dtype=mx.bool_))


def identity_mask(t: int) -> mx.array:
    return mx.eye(t, dtype=mx.bool_)


def windowed_causal_mask(t: int, context: int) -> mx.array:
    mask = mx.tril(mx.ones((t, t), dtype=mx.bool_))
    # keep only the last `context` keys
    idx_q = mx.arange(t)[:, None]
    idx_k = mx.arange(t)[None, :]
    return mask & ((idx_q - idx_k) < context)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.repeats = num_heads // num_kv_heads
        self.q = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.k = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.v = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.o = nn.Linear(num_heads * head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)

    def __call__(self, x: mx.array, rope: tuple[mx.array, mx.array] | None, mask: mx.array | None) -> mx.array:
        b, t, _ = x.shape
        q = self.q(x).reshape(b, t, self.num_heads, self.head_dim)
        k = self.k(x).reshape(b, t, self.num_kv_heads, self.head_dim)
        v = self.v(x).reshape(b, t, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope is not None:
            q = apply_rope(q, rope[0], rope[1])
            k = apply_rope(k, rope[0], rope[1])
        if self.repeats > 1:
            k = mx.repeat(k, self.repeats, axis=2)
            v = mx.repeat(v, self.repeats, axis=2)
        # SDPA wants (B, heads, T, D)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_mask = None
        if mask is not None:
            # mask: (T, T) or (B, T, T) bool -> broadcast to (B, 1, T, T)
            if mask.ndim == 2:
                attn_mask = mask[None, None, :, :]
            elif mask.ndim == 3:
                attn_mask = mask[:, None, :, :]
            else:
                attn_mask = mask
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=attn_mask)
        out = out.transpose(0, 2, 1, 3).reshape(b, t, -1)
        return self.o(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, head_dim: int, mlp_hidden: int):
        super().__init__()
        self.n1 = nn.RMSNorm(dim)
        self.attn = Attention(dim, num_heads, num_kv_heads, head_dim)
        self.n2 = nn.RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_hidden)

    def __call__(self, x: mx.array, rope: tuple[mx.array, mx.array], mask: mx.array | None) -> mx.array:
        x = x + self.attn(self.n1(x), rope, mask)
        x = x + self.mlp(self.n2(x))
        return x


class SpatioTemporalTransformer(nn.Module):
    """Alternating spatial / temporal attention. Temporal every `period` layers at `offset`."""

    def __init__(
        self,
        num_layers: int,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        mlp_hidden: int,
        temporal_period: int = 4,
        temporal_offset: int = 1,
    ):
        super().__init__()
        if not 0 <= temporal_offset < temporal_period:
            raise ValueError("temporal_offset out of range")
        if num_layers % temporal_period != 0:
            raise ValueError("num_layers must be divisible by temporal_period")
        self.num_layers = num_layers
        self.period = temporal_period
        self.offset = temporal_offset
        self.blocks = [
            TransformerBlock(dim, num_heads, num_kv_heads, head_dim, mlp_hidden) for _ in range(num_layers)
        ]
        self.final_norm = nn.RMSNorm(dim)

    def is_temporal(self, i: int) -> bool:
        return i % self.period == self.offset

    def __call__(
        self,
        x: mx.array,
        spatial_rope: tuple[mx.array, mx.array],
        spatial_mask: mx.array,
        temporal_rope: tuple[mx.array, mx.array],
        temporal_mask: mx.array,
    ) -> mx.array:
        # x: (B, T, N, D)
        b, t, n, d = x.shape
        for i, block in enumerate(self.blocks):
            if self.is_temporal(i):
                h = x.transpose(0, 2, 1, 3).reshape(b * n, t, d)
                # temporal_mask (B, T, T) or (T, T) — expand per token
                if temporal_mask.ndim == 2:
                    m = temporal_mask
                else:
                    m = mx.repeat(temporal_mask, n, axis=0)
                h = block(h, temporal_rope, m)
                x = h.reshape(b, n, t, d).transpose(0, 2, 1, 3)
            else:
                h = x.reshape(b * t, n, d)
                h = block(h, spatial_rope, spatial_mask)
                x = h.reshape(b, t, n, d)
        return self.final_norm(x)


def patchify(video: mx.array, patch: int) -> mx.array:
    """(B, T, H, W, C) -> (B, T, N, P)."""
    b, t, h, w, c = video.shape
    ph, pw = h // patch, w // patch
    x = video.reshape(b, t, ph, patch, pw, patch, c)
    x = x.transpose(0, 1, 2, 4, 3, 5, 6)
    return x.reshape(b, t, ph * pw, patch * patch * c)


def unpatchify(patches: mx.array, patch: int, height: int, width: int, channels: int = 3) -> mx.array:
    """(B, T, N, P) -> (B, T, H, W, C)."""
    b, t, _, _ = patches.shape
    ph, pw = height // patch, width // patch
    x = patches.reshape(b, t, ph, pw, patch, patch, channels)
    x = x.transpose(0, 1, 2, 4, 3, 5, 6)
    return x.reshape(b, t, height, width, channels)


def count_params(module: nn.Module) -> int:
    n = 0
    for _, arr in tree_flatten(module.parameters()):
        n += arr.size
    return n


def tree_flatten(tree) -> list[tuple[str, mx.array]]:
    from mlx.utils import tree_flatten as _tf

    return _tf(tree)
