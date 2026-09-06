"""Model and training configs. Tiny is the M4 16GB default; atari matches Visionary's 7M-ish shape."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class TokenizerConfig:
    encoder_layers: int = 4
    decoder_layers: int = 4
    num_latents: int = 16
    num_heads: int = 4
    num_kv_heads: int = 2
    model_dim: int = 64
    head_dim: int = 16
    mlp_hidden_dim: int = 192
    channel_dim: int = 8
    patch_size: int = 8
    image_size: int = 64
    rope_base: float = 10000.0
    temporal_period: int = 4
    temporal_offset: int = 1
    independent_prob: float = 0.3
    mask_prob_min: float = 0.0
    mask_prob_max: float = 0.9


@dataclass
class DynamicsConfig:
    num_layers: int = 8
    num_heads: int = 4
    num_kv_heads: int = 2
    num_registers: int = 2
    num_actions: int = 5
    max_step_size: int = 4
    model_dim: int = 64
    head_dim: int = 16
    mlp_hidden_dim: int = 192
    context_length: int = 8
    rope_base: float = 10000.0
    temporal_period: int = 4
    sample_steps: int = 4
    context_tau: float = 0.9


@dataclass
class AgentConfig:
    hidden_dim: int = 128
    num_actions: int = 5
    imagine_horizon: int = 8
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5


@dataclass
class ModelConfig:
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    @property
    def image_size(self) -> int:
        return self.tokenizer.image_size

    @property
    def patch_size(self) -> int:
        return self.tokenizer.patch_size

    @property
    def num_patches(self) -> int:
        s = self.image_size // self.patch_size
        return s * s

    @property
    def num_latents(self) -> int:
        return self.tokenizer.num_latents

    @property
    def channel_dim(self) -> int:
        return self.tokenizer.channel_dim

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        return cls(
            tokenizer=TokenizerConfig(**d.get("tokenizer", {})),
            dynamics=DynamicsConfig(**d.get("dynamics", {})),
            agent=AgentConfig(**d.get("agent", {})),
        )


@dataclass
class TrainConfig:
    seed: int = 0
    batch_size: int = 8
    seq_len: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.01
    tokenizer_steps: int = 400
    dynamics_steps: int = 400
    agent_steps: int = 200
    log_every: int = 20
    eval_every: int = 100
    ema_decay: float = 0.99
    lpips_proxy_weight: float = 0.1  # gradient-free perceptual proxy (blur L1)
    data_dir: str = "data"
    run_dir: str = "runs/default"
    env: str = "arena"
    num_episodes: int = 256
    episode_len: int = 32


def tiny_config() -> ModelConfig:
    """Fits comfortably on 16GB unified memory."""
    return ModelConfig()


def atari_config() -> ModelConfig:
    """Closer to Visionary's 7M Dream Atari models. Needs more memory."""
    tok = TokenizerConfig(
        encoder_layers=8,
        decoder_layers=8,
        num_latents=32,
        num_heads=8,
        num_kv_heads=2,
        model_dim=128,
        head_dim=32,
        mlp_hidden_dim=384,
        channel_dim=16,
        patch_size=8,
        image_size=64,
    )
    dyn = DynamicsConfig(
        num_layers=16,
        num_heads=8,
        num_kv_heads=2,
        num_registers=2,
        num_actions=5,
        max_step_size=6,
        model_dim=128,
        head_dim=32,
        mlp_hidden_dim=384,
        context_length=16,
        sample_steps=4,
    )
    agent = AgentConfig(hidden_dim=256, num_actions=5, imagine_horizon=16)
    return ModelConfig(tokenizer=tok, dynamics=dyn, agent=agent)


def champion_config() -> ModelConfig:
    """High-capacity candidate for this repository's 64px video workload.

    This is a configuration, not a performance claim: publish held-out scores
    from a trained checkpoint before calling it a champion.
    """
    tok = TokenizerConfig(
        encoder_layers=16, decoder_layers=16, num_latents=64, num_heads=8,
        num_kv_heads=2, model_dim=256, head_dim=32, mlp_hidden_dim=1024,
        channel_dim=32, patch_size=8, image_size=64,
    )
    dyn = DynamicsConfig(
        num_layers=24, num_heads=8, num_kv_heads=2, num_registers=4,
        num_actions=5, max_step_size=6, model_dim=256, head_dim=32,
        mlp_hidden_dim=1024, context_length=32, sample_steps=8,
    )
    agent = AgentConfig(hidden_dim=512, num_actions=5, imagine_horizon=32)
    return ModelConfig(tokenizer=tok, dynamics=dyn, agent=agent)


CONFIGS = {
    "tiny": tiny_config,
    "atari": atari_config,
    "champion": champion_config,
}
