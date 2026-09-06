"""Pixel metrics and GIF writers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def psnr(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    mse = np.mean((a - b) ** 2)
    if mse < eps:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def to_uint8(frames: np.ndarray) -> np.ndarray:
    x = np.clip(frames, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)


def save_gif(frames: np.ndarray, path: str | Path, fps: int = 12, scale: int = 4) -> Path:
    """frames: (T, H, W, C) float or uint8."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if frames.dtype != np.uint8:
        frames = to_uint8(frames)
    imgs = []
    for f in frames:
        im = Image.fromarray(f)
        if scale != 1:
            im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
        imgs.append(im)
    duration = int(1000 / max(fps, 1))
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=duration, loop=0)
    return path


def montage_row(clips: list[np.ndarray], gap: int = 2) -> np.ndarray:
    """clips: list of (T, H, W, C) -> (T, H, W*k, C)."""
    t = min(c.shape[0] for c in clips)
    clips = [c[:t] for c in clips]
    h, w = clips[0].shape[1:3]
    c = clips[0].shape[-1]
    g = np.zeros((t, h, gap, c), dtype=clips[0].dtype)
    parts = []
    for i, cl in enumerate(clips):
        if i:
            parts.append(g)
        parts.append(cl)
    return np.concatenate(parts, axis=2)
