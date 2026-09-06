"""Reproducible checkpoint-only world-model benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import json

import mlx.core as mx
import numpy as np

from visionary_mlx.data import ClipLoader
from visionary_mlx.dynamics import DynamicsModel
from visionary_mlx.metrics import mae, psnr
from visionary_mlx.tokenizer import VideoTokenizer


@dataclass(frozen=True)
class BenchmarkResult:
    """Held-out scores for one named, immutable checkpoint."""

    name: str
    tokenizer_psnr: float
    tokenizer_mae: float
    dream_psnr: float | None = None
    dream_mae: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(name: str, tokenizer: VideoTokenizer, loader: ClipLoader, dynamics: DynamicsModel | None = None, batches: int = 8) -> BenchmarkResult:
    """Score reconstruction and optional open-loop prediction on held-out clips."""
    if batches < 1:
        raise ValueError("batches must be positive")
    tokenizer.eval()
    if dynamics is not None:
        dynamics.eval()
    recon_psnrs, recon_maes, dream_psnrs, dream_maes = [], [], [], []
    for _ in range(batches):
        batch = loader.sample()
        video, actions = mx.array(batch["video"]), mx.array(batch["actions"])
        # Disable both spatial masking and random independent-frame masking:
        # the same checkpoint/data/seed must always yield the same score.
        independent = mx.zeros((video.shape[0],), dtype=mx.bool_)
        recon, _, z = tokenizer.reconstruct(video, mask_prob=0.0, independent=independent)
        mx.eval(recon, z)
        target, reconstruction = np.array(video), np.array(recon)
        recon_psnrs.append(psnr(target, reconstruction))
        recon_maes.append(mae(target, reconstruction))
        if dynamics is not None:
            context = max(1, video.shape[1] // 4)
            latent_rollout = dynamics.rollout(z[:, :context], actions, video.shape[1] - context)
            dreamed = tokenizer.decode_video(latent_rollout)
            mx.eval(dreamed)
            dream = np.array(dreamed)
            dream_psnrs.append(psnr(target[:, context:], dream[:, context:]))
            dream_maes.append(mae(target[:, context:], dream[:, context:]))
    tokenizer.train()
    if dynamics is not None:
        dynamics.train()
    return BenchmarkResult(name, float(np.mean(recon_psnrs)), float(np.mean(recon_maes)), float(np.mean(dream_psnrs)) if dream_psnrs else None, float(np.mean(dream_maes)) if dream_maes else None)


def leaderboard_markdown(results: Iterable[BenchmarkResult]) -> str:
    ranked = sorted(results, key=lambda item: (-item.tokenizer_psnr, item.tokenizer_mae, item.name))
    lines = ["# Visionary MLX benchmark", "", "Held-out scores: higher PSNR and lower MAE are better.", "", "| Rank | Checkpoint | Tokenizer PSNR | Tokenizer MAE | Dream PSNR | Dream MAE |", "| ---: | --- | ---: | ---: | ---: | ---: |"]
    for rank, result in enumerate(ranked, 1):
        dream_psnr = "—" if result.dream_psnr is None else f"{result.dream_psnr:.3f}"
        dream_mae = "—" if result.dream_mae is None else f"{result.dream_mae:.5f}"
        lines.append(f"| {rank} | {result.name} | {result.tokenizer_psnr:.3f} | {result.tokenizer_mae:.5f} | {dream_psnr} | {dream_mae} |")
    return "\n".join(lines) + "\n"


def write_leaderboard(results: Iterable[BenchmarkResult], out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    materialized = list(results)
    json_path, markdown_path = out / "benchmark.json", out / "BENCHMARK.md"
    json_path.write_text(json.dumps([item.to_dict() for item in materialized], indent=2) + "\n")
    markdown_path.write_text(leaderboard_markdown(materialized))
    return json_path, markdown_path
