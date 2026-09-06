"""Classic RSSM baseline (VAE encoder + GRU dynamics + decoder). Useful for debugging."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, z: int = 32):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, 32, 4, stride=2, padding=1)  # 64->32
        self.c2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)  # 32->16
        self.c3 = nn.Conv2d(64, 128, 4, stride=2, padding=1)  # 16->8
        self.c4 = nn.Conv2d(128, 128, 4, stride=2, padding=1)  # 8->4
        self.fc = nn.Linear(128 * 4 * 4, z)

    def __call__(self, x: mx.array) -> mx.array:
        h = nn.silu(self.c1(x))
        h = nn.silu(self.c2(h))
        h = nn.silu(self.c3(h))
        h = nn.silu(self.c4(h))
        return self.fc(h.reshape(h.shape[0], -1))


class ConvDecoder(nn.Module):
    def __init__(self, z: int = 32, out_ch: int = 3):
        super().__init__()
        self.fc = nn.Linear(z, 128 * 4 * 4)
        self.d1 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)
        self.d2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.d3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.d4 = nn.ConvTranspose2d(32, out_ch, 4, stride=2, padding=1)

    def __call__(self, z: mx.array) -> mx.array:
        h = self.fc(z).reshape(-1, 4, 4, 128)
        h = nn.silu(self.d1(h))
        h = nn.silu(self.d2(h))
        h = nn.silu(self.d3(h))
        return mx.sigmoid(self.d4(h))


class RSSM(nn.Module):
    """Ha & Schmidhuber / Dreamer-v1 shape: encode, GRU next-latent, decode."""

    def __init__(self, z: int = 32, hdim: int = 128, num_actions: int = 5, in_ch: int = 3):
        super().__init__()
        self.z = z
        self.enc = ConvEncoder(in_ch, z)
        self.dec = ConvDecoder(z, in_ch)
        self.act_emb = nn.Embedding(num_actions, z)
        self.gru = nn.GRU(z + z, hdim)
        self.to_z = nn.Linear(hdim, z)
        self.znorm = nn.LayerNorm(z)

    def encode_seq(self, x: mx.array) -> mx.array:
        b, t = x.shape[:2]
        z = self.znorm(self.enc(x.reshape(b * t, *x.shape[2:])))
        return z.reshape(b, t, self.z)

    def decode_seq(self, z: mx.array) -> mx.array:
        b, t = z.shape[:2]
        x = self.dec(z.reshape(b * t, self.z))
        return x.reshape(b, t, *x.shape[1:])

    def imagine(self, x_ctx: mx.array, actions: mx.array, steps: int) -> mx.array:
        """actions: (B, ctx+steps) the action taken at each frame."""
        z_ctx = self.encode_seq(x_ctx)
        b, ctx, _ = z_ctx.shape
        a_ctx = self.act_emb(actions[:, :ctx])
        inp = mx.concatenate([z_ctx, a_ctx], axis=-1)
        hseq = self.gru(inp)
        h = hseq[:, -1, :]
        z = self.znorm(self.to_z(h))
        preds = [z]
        for i in range(steps - 1):
            a = self.act_emb(actions[:, ctx + i])
            step_in = mx.concatenate([z, a], axis=-1)[:, None, :]
            h = self.gru(step_in, h)[:, -1, :]
            z = self.znorm(self.to_z(h))
            preds.append(z)
        return mx.stack(preds, axis=1)

    def loss(self, x: mx.array, actions: mx.array, context: int = 4) -> tuple[mx.array, dict]:
        z = self.encode_seq(x)
        recon = self.decode_seq(z)
        l_recon = mx.mean((recon - x) ** 2)
        # teacher-forced next latent
        a = self.act_emb(actions[:, :-1])
        inp = mx.concatenate([z[:, :-1], a], axis=-1)
        h = self.gru(inp)
        z_hat = self.znorm(self.to_z(h))
        l_dyn = mx.mean((z_hat - mx.stop_gradient(z[:, 1:])) ** 2)
        # imagination on the tail
        steps = x.shape[1] - context
        z_img = self.imagine(x[:, :context], actions, steps)
        x_img = self.decode_seq(z_img)
        l_img = mx.mean((x_img - x[:, context:]) ** 2)
        loss = l_recon + l_dyn + 0.5 * l_img
        return loss, {"recon": l_recon, "dyn": l_dyn, "imagine": l_img}
