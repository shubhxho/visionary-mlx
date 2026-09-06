"""Collect rollouts and sample training clips."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from visionary_mlx.envs import heuristic_policy, make_env


def collect_rollouts(
    env_name: str,
    num_episodes: int,
    episode_len: int,
    size: int = 64,
    seed: int = 0,
    parallel: int = 16,
    expert_frac: float = 0.7,
) -> dict[str, np.ndarray]:
    """Return uint8 videos (E, T, H, W, C), int actions, float rewards, float dones."""
    videos, actions, rewards, dones = [], [], [], []
    left = num_episodes
    ep_i = 0
    while left > 0:
        n = min(parallel, left)
        env = make_env(env_name, n=n, size=size, seed=seed + ep_i * 997)
        env.reset()
        f_seq, a_seq, r_seq, d_seq = [], [], [], []
        for t in range(episode_len):
            use_expert = env.rng.random(n) < expert_frac
            a = heuristic_policy(env, env.render())
            rnd = env.rng.integers(0, 5, n)
            a = np.where(use_expert, a, rnd).astype(np.int32)
            frame = env.render()
            nxt, r, d = env.step(a)
            f_seq.append(frame)
            a_seq.append(a)
            r_seq.append(r)
            d_seq.append(d)
            _ = nxt
        videos.append(np.stack(f_seq, axis=1))  # (n, T, H, W, C)
        actions.append(np.stack(a_seq, axis=1))
        rewards.append(np.stack(r_seq, axis=1))
        dones.append(np.stack(d_seq, axis=1))
        left -= n
        ep_i += 1
    out = {
        "video": np.clip(np.concatenate(videos, axis=0) * 255.0, 0, 255).astype(np.uint8),
        "actions": np.concatenate(actions, axis=0).astype(np.int32),
        "rewards": np.concatenate(rewards, axis=0).astype(np.float32),
        "dones": np.concatenate(dones, axis=0).astype(np.float32),
        "env": np.array(env_name),
    }
    return out


def save_dataset(ds: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **ds)
    return path


def load_dataset(path: str | Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


class ClipLoader:
    def __init__(self, ds: dict, seq_len: int, batch_size: int, seed: int = 0):
        self.video = ds["video"]  # (E, T, H, W, C) uint8
        self.actions = ds["actions"]
        self.rewards = ds.get("rewards")
        self.dones = ds.get("dones")
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        e, t = self.video.shape[:2]
        if t < seq_len:
            raise ValueError(f"episode_len {t} < seq_len {seq_len}")
        self.e = e
        self.t = t
        self.max_start = t - seq_len + 1

    def sample(self) -> dict[str, np.ndarray]:
        idx = self.rng.integers(0, self.e, self.batch_size)
        start = self.rng.integers(0, self.max_start, self.batch_size)
        sl = self.seq_len
        v = np.stack([self.video[i, s : s + sl] for i, s in zip(idx, start)], axis=0)
        a = np.stack([self.actions[i, s : s + sl] for i, s in zip(idx, start)], axis=0)
        batch = {
            "video": (v.astype(np.float32) / 255.0),
            "actions": a.astype(np.int32),
        }
        if self.rewards is not None:
            batch["rewards"] = np.stack(
                [self.rewards[i, s : s + sl] for i, s in zip(idx, start)], axis=0
            ).astype(np.float32)
        if self.dones is not None:
            batch["dones"] = np.stack(
                [self.dones[i, s : s + sl] for i, s in zip(idx, start)], axis=0
            ).astype(np.float32)
        return batch
