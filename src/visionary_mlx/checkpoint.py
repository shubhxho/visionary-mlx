"""Safetensors checkpoints + json sidecar config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from visionary_mlx.config import ModelConfig


def save_model(model: nn.Module, path: str | Path, config: ModelConfig | dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(path))
    if config is not None:
        cfg = config.to_dict() if hasattr(config, "to_dict") else config
        path.with_suffix(".json").write_text(json.dumps(cfg, indent=2))
    return path


def load_weights(model: nn.Module, path: str | Path) -> nn.Module:
    model.load_weights(str(path))
    mx.eval(model.parameters())
    return model


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())
