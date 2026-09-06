"""MAE video tokenizer: latent tokens + spatiotemporal transformer encoder/decoder."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from visionary_mlx.config import TokenizerConfig
from visionary_mlx.layers import (
    SpatioTemporalTransformer,
    pad_spatial_rope_for_latents,
    patchify,
    spatial_mask,
    spatial_rope,
    temporal_rope,
    unpatchify,
)


class TokenizerEncoder(nn.Module):
    def __init__(self, cfg: TokenizerConfig, x_len: int, y_len: int):
        super().__init__()
        self.cfg = cfg
        self.x_len = x_len
        self.y_len = y_len
        patch_dim = cfg.patch_size * cfg.patch_size * 3
        self.in_proj = nn.Linear(patch_dim, cfg.model_dim)
        self.mask_token = mx.random.normal((cfg.model_dim,)) * 0.02
        self.latent_tokens = mx.random.normal((cfg.num_latents, cfg.model_dim)) * 0.02
        self.blocks = SpatioTemporalTransformer(
            num_layers=cfg.encoder_layers,
            dim=cfg.model_dim,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            mlp_hidden=cfg.mlp_hidden_dim,
            temporal_period=cfg.temporal_period,
            temporal_offset=cfg.temporal_offset,
        )
        self.out_proj = nn.Linear(cfg.model_dim, cfg.channel_dim)
        self._smask = spatial_mask(x_len * y_len, cfg.num_latents, encoder=True)

    def __call__(self, patches: mx.array, temporal_mask: mx.array, mask: mx.array | None = None) -> mx.array:
        # patches: (B, T, N, P) in [0, 1]
        b, t, n, _ = patches.shape
        x = self.in_proj(patches)
        if mask is not None:
            x = mx.where(mask[..., None], self.mask_token.astype(x.dtype), x)
        lat = mx.broadcast_to(self.latent_tokens, (b, t, self.cfg.num_latents, self.cfg.model_dim))
        x = mx.concatenate([lat, x], axis=2)

        scos, ssin = spatial_rope(self.cfg.rope_base, self.cfg.head_dim, self.x_len, self.y_len)
        scos, ssin = pad_spatial_rope_for_latents(scos, ssin, self.cfg.num_latents)
        tcos, tsin = temporal_rope(self.cfg.rope_base, self.cfg.head_dim, t)
        x = self.blocks(x, (scos, ssin), self._smask, (tcos, tsin), temporal_mask)
        z = x[:, :, : self.cfg.num_latents, :]
        return mx.tanh(self.out_proj(z))


class TokenizerDecoder(nn.Module):
    def __init__(self, cfg: TokenizerConfig, x_len: int, y_len: int):
        super().__init__()
        self.cfg = cfg
        self.x_len = x_len
        self.y_len = y_len
        self.num_tokens = x_len * y_len
        patch_dim = cfg.patch_size * cfg.patch_size * 3
        self.lat_proj = nn.Linear(cfg.channel_dim, cfg.model_dim)
        self.image_tokens = mx.random.normal((self.num_tokens, cfg.model_dim)) * 0.02
        self.blocks = SpatioTemporalTransformer(
            num_layers=cfg.decoder_layers,
            dim=cfg.model_dim,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            mlp_hidden=cfg.mlp_hidden_dim,
            temporal_period=cfg.temporal_period,
            temporal_offset=cfg.temporal_offset,
        )
        self.out_proj = nn.Linear(cfg.model_dim, patch_dim)
        self._smask = spatial_mask(self.num_tokens, cfg.num_latents, encoder=False)

    def __call__(self, latent: mx.array, temporal_mask: mx.array) -> mx.array:
        b, t, _, _ = latent.shape
        lat = self.lat_proj(latent)
        img = mx.broadcast_to(self.image_tokens, (b, t, self.num_tokens, self.cfg.model_dim))
        x = mx.concatenate([lat, img], axis=2)
        scos, ssin = spatial_rope(self.cfg.rope_base, self.cfg.head_dim, self.x_len, self.y_len)
        scos, ssin = pad_spatial_rope_for_latents(scos, ssin, self.cfg.num_latents)
        tcos, tsin = temporal_rope(self.cfg.rope_base, self.cfg.head_dim, t)
        x = self.blocks(x, (scos, ssin), self._smask, (tcos, tsin), temporal_mask)
        x = x[:, :, self.cfg.num_latents :, :]
        return mx.sigmoid(self.out_proj(x))


class VideoTokenizer(nn.Module):
    def __init__(self, cfg: TokenizerConfig | None = None):
        super().__init__()
        self.cfg = cfg or TokenizerConfig()
        s = self.cfg.image_size // self.cfg.patch_size
        self.x_len = s
        self.y_len = s
        self.encoder = TokenizerEncoder(self.cfg, s, s)
        self.decoder = TokenizerDecoder(self.cfg, s, s)

    def encode_patches(self, patches: mx.array, independent: mx.array | None = None) -> mx.array:
        b, t, _, _ = patches.shape
        tmask = self._temporal_mask(b, t, independent)
        return self.encoder(patches, tmask, mask=None)

    def decode_latents(self, latent: mx.array, independent: mx.array | None = None) -> mx.array:
        b, t, _, _ = latent.shape
        tmask = self._temporal_mask(b, t, independent)
        return self.decoder(latent, tmask)

    def encode_video(self, video: mx.array) -> mx.array:
        patches = patchify(video, self.cfg.patch_size)
        return self.encode_patches(patches)

    def decode_video(self, latent: mx.array) -> mx.array:
        patches = self.decode_latents(latent)
        return unpatchify(patches, self.cfg.patch_size, self.cfg.image_size, self.cfg.image_size, 3)

    def reconstruct(
        self,
        video: mx.array,
        mask_prob: float | None = None,
        independent: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Returns reconstructed video in [0,1], bool mask, latent."""
        patches = patchify(video, self.cfg.patch_size)
        b, t, n, _ = patches.shape
        if independent is None:
            independent = mx.random.bernoulli(self.cfg.independent_prob, (b,))
        tmask = self._temporal_mask(b, t, independent)
        mask = self._sample_mask(b, t, n, mask_prob)
        latent = self.encoder(patches, tmask, mask=mask)
        recon_p = self.decoder(latent, tmask)
        recon = unpatchify(recon_p, self.cfg.patch_size, self.cfg.image_size, self.cfg.image_size, 3)
        return recon, mask, latent

    def _temporal_mask(self, batch: int, t: int, independent: mx.array | None) -> mx.array:
        causal = mx.tril(mx.ones((t, t), dtype=mx.bool_))
        ident = mx.eye(t, dtype=mx.bool_)
        if independent is None:
            return causal
        # (B, T, T)
        flag = independent.astype(mx.bool_)[:, None, None]
        return mx.where(flag, ident[None, :, :], causal[None, :, :])

    def _sample_mask(self, b: int, t: int, n: int, mask_prob: float | None) -> mx.array:
        if mask_prob is None:
            p = mx.random.uniform(self.cfg.mask_prob_min, self.cfg.mask_prob_max, (b, t))
        else:
            p = mx.full((b, t), float(mask_prob))
        r = mx.random.uniform(0, 1, (b, t, n))
        return r < p[:, :, None]
