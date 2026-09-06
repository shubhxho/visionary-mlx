"""Dream: encode context, roll dynamics, decode video. Optionally overlay the actor."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from visionary_mlx.agent import ActorCritic
from visionary_mlx.dynamics import DynamicsModel
from visionary_mlx.metrics import montage_row, save_gif
from visionary_mlx.tokenizer import VideoTokenizer


def dream_clip(
    tok: VideoTokenizer,
    dyn: DynamicsModel,
    video: mx.array,
    actions: mx.array,
    context: int,
) -> dict[str, np.ndarray]:
    z = tok.encode_video(video)
    horizon = video.shape[1] - context
    z_roll = dyn.rollout(z[:, :context], actions, horizon)
    recon = tok.decode_video(z)
    dream = tok.decode_video(z_roll)
    mx.eval(recon, dream)
    return {
        "gt": np.array(video),
        "recon": np.array(recon),
        "dream": np.array(dream),
        "z": np.array(z),
        "z_dream": np.array(z_roll),
    }


def actor_dream(
    tok: VideoTokenizer,
    dyn: DynamicsModel,
    agent: ActorCritic,
    z0: mx.array,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll the world model with the actor choosing actions. Returns decoded video, actions."""
    z = z0
    frames = [z]
    acts = []
    for _ in range(horizon):
        action, _, _ = agent.act(z, greedy=True)
        acts.append(np.array(action))
        z_seq = z[:, None, :, :]
        a_seq = action[:, None]
        z = dyn.generate_next(z_seq, a_seq, target_index=0, sample_steps=dyn.cfg.sample_steps)
        frames.append(z)
    z_seq = mx.stack(frames, axis=1)
    video = tok.decode_video(z_seq)
    mx.eval(video)
    return np.array(video), np.stack(acts, axis=1)


def write_dream_gifs(out: dict[str, np.ndarray], dest: Path, scale: int = 4) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    n = out["gt"].shape[0]
    for i in range(n):
        grid = montage_row([out["gt"][i], out["recon"][i], out["dream"][i]])
        p = dest / f"dream_{i:02d}.gif"
        save_gif(grid, p, fps=8, scale=scale)
        paths.append(p)
    # montage of first 4 dreams
    tops = [out["dream"][i] for i in range(min(4, n))]
    save_gif(montage_row(tops), dest / "dream_montage.gif", fps=8, scale=scale)
    return paths
