"""Visionary-MLX: Dreamer-style world models on Apple Silicon."""

from visionary_mlx.config import ModelConfig, TrainConfig, tiny_config
from visionary_mlx.tokenizer import VideoTokenizer
from visionary_mlx.dynamics import DynamicsModel
from visionary_mlx.agent import ActorCritic
from visionary_mlx.rssm import RSSM

__version__ = "0.1.0"
__all__ = [
    "ModelConfig",
    "TrainConfig",
    "tiny_config",
    "VideoTokenizer",
    "DynamicsModel",
    "ActorCritic",
    "RSSM",
]
