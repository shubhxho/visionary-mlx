"""Tiny forward-pass shape tests. Instantiates models at reduced depth."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from visionary_mlx.config import AgentConfig, DynamicsConfig, TokenizerConfig
from visionary_mlx.agent import ActorCritic
from visionary_mlx.dynamics import DynamicsModel
from visionary_mlx.layers import count_params, patchify, unpatchify
from visionary_mlx.tokenizer import VideoTokenizer


def _tok_cfg() -> TokenizerConfig:
    return TokenizerConfig(
        encoder_layers=4,
        decoder_layers=4,
        num_latents=4,
        num_heads=2,
        num_kv_heads=1,
        model_dim=32,
        head_dim=16,
        mlp_hidden_dim=64,
        channel_dim=4,
        patch_size=8,
        image_size=16,
        temporal_period=4,
        temporal_offset=1,
    )


def test_patchify_roundtrip():
    x = mx.random.uniform(0, 1, (2, 3, 16, 16, 3))
    p = patchify(x, 8)
    y = unpatchify(p, 8, 16, 16, 3)
    assert p.shape == (2, 3, 4, 192)
    assert y.shape == x.shape
    assert float(mx.mean(mx.abs(x - y)).item()) < 1e-5


def test_tokenizer_forward():
    mx.random.seed(0)
    cfg = _tok_cfg()
    m = VideoTokenizer(cfg)
    video = mx.random.uniform(0, 1, (2, 4, 16, 16, 3))
    recon, mask, z = m.reconstruct(video)
    assert recon.shape == video.shape
    assert z.shape == (2, 4, cfg.num_latents, cfg.channel_dim)
    assert mask.shape == (2, 4, 4)
    n = count_params(m)
    assert n > 10_000


def test_dynamics_flow():
    mx.random.seed(0)
    tcfg = _tok_cfg()
    dcfg = DynamicsConfig(
        num_layers=4,
        num_heads=2,
        num_kv_heads=1,
        num_registers=1,
        num_actions=5,
        max_step_size=3,
        model_dim=32,
        head_dim=16,
        mlp_hidden_dim=64,
        context_length=4,
        temporal_period=4,
        sample_steps=2,
    )
    dyn = DynamicsModel(dcfg, tcfg)
    z = mx.random.normal((2, 4, tcfg.num_latents, tcfg.channel_dim))
    a = mx.zeros((2, 4), dtype=mx.int32)
    loss, mets = dyn.flow_loss(z, a)
    assert loss.shape == ()
    assert "flow_mse" in mets
    nxt = dyn.generate_next(z, a, target_index=3, sample_steps=2)
    assert nxt.shape == (2, tcfg.num_latents, tcfg.channel_dim)


def test_agent_act():
    ag = ActorCritic(latent_dim=16, cfg=AgentConfig(hidden_dim=32, num_actions=5))
    z = mx.random.normal((3, 4, 4))
    action, logp, v = ag.act(z)
    assert action.shape == (3,)
    assert logp.shape == (3,)
    assert v.shape == (3,)


def test_env_step():
    from visionary_mlx.envs import make_env

    env = make_env("arena", n=4, size=32, seed=1)
    f0 = env.render()
    assert f0.shape == (4, 32, 32, 3)
    f1, r, d = env.step(np.array([0, 1, 2, 3]))
    assert f1.shape == f0.shape
    assert r.shape == (4,)
    assert d.shape == (4,)
