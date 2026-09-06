"""Vectorized visual sandboxes. No ALE — pure numpy, 64x64 RGB, Atari-like physics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ACTIONS = ["noop", "up", "down", "left", "right"]
ACTION_DELTA = {
    0: (0.0, 0.0),
    1: (0.0, -1.0),
    2: (0.0, 1.0),
    3: (-1.0, 0.0),
    4: (1.0, 0.0),
}


def _disks(frames: np.ndarray, ys: np.ndarray, xs: np.ndarray, r: float, rgb: tuple[float, float, float]):
    """Paint filled disks. ys/xs: (N,) pixel coords."""
    h, w = frames.shape[1:3]
    yy = np.arange(h)[None, :, None]
    xx = np.arange(w)[None, None, :]
    d2 = (yy - ys[:, None, None]) ** 2 + (xx - xs[:, None, None]) ** 2
    m = d2 <= r * r
    for c, v in enumerate(rgb):
        ch = frames[..., c]
        ch[m] = v
        frames[..., c] = ch


@dataclass
class EnvSpec:
    name: str
    num_actions: int = 5
    size: int = 64


class ArenaEnv:
    """Agent (cyan) chases a gold target while two balls bounce. Reward on overlap."""

    spec = EnvSpec("arena")

    def __init__(self, n: int, size: int = 64, seed: int = 0):
        self.n = n
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.agent = np.zeros((n, 2), dtype=np.float32)
        self.target = np.zeros((n, 2), dtype=np.float32)
        self.ball = np.zeros((n, 2, 2), dtype=np.float32)  # two balls, xy
        self.ball_v = np.zeros((n, 2, 2), dtype=np.float32)
        self.reset()

    def reset(self, mask: np.ndarray | None = None) -> np.ndarray:
        m = np.ones(self.n, dtype=bool) if mask is None else mask
        k = int(m.sum())
        if k == 0:
            return self.render()
        lo, hi = 6.0, self.size - 6.0
        self.agent[m] = self.rng.uniform(lo, hi, (k, 2)).astype(np.float32)
        self.target[m] = self.rng.uniform(lo, hi, (k, 2)).astype(np.float32)
        self.ball[m] = self.rng.uniform(lo, hi, (k, 2, 2)).astype(np.float32)
        ang = self.rng.uniform(0, 2 * np.pi, (k, 2))
        spd = self.rng.uniform(0.8, 1.6, (k, 2, 1))
        self.ball_v[m] = np.stack([np.cos(ang), np.sin(ang)], axis=-1).astype(np.float32) * spd
        return self.render()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        actions = np.asarray(actions, dtype=np.int32)
        delta = np.array([ACTION_DELTA[int(a)] for a in actions], dtype=np.float32)
        self.agent = np.clip(self.agent + delta * 2.2, 4.0, self.size - 5.0)
        # bounce balls
        nxt = self.ball + self.ball_v
        hit = (nxt < 3.0) | (nxt > self.size - 4.0)
        self.ball_v = np.where(hit, -self.ball_v, self.ball_v)
        self.ball = np.clip(self.ball + self.ball_v, 3.0, self.size - 4.0)
        # kick: if agent close to a ball, add impulse
        for i in range(2):
            d = self.ball[:, i] - self.agent
            dist = np.linalg.norm(d, axis=-1, keepdims=True) + 1e-6
            kick = (dist[:, 0] < 6.0)[:, None]
            self.ball_v[:, i] = np.where(kick, self.ball_v[:, i] + 0.6 * d / dist, self.ball_v[:, i])
        dist_t = np.linalg.norm(self.agent - self.target, axis=-1)
        reward = (dist_t < 6.0).astype(np.float32)
        # relocate target on hit
        hit_t = dist_t < 6.0
        if hit_t.any():
            k = int(hit_t.sum())
            self.target[hit_t] = self.rng.uniform(6.0, self.size - 6.0, (k, 2)).astype(np.float32)
        done = np.zeros(self.n, dtype=np.float32)
        return self.render(), reward, done

    def render(self) -> np.ndarray:
        s = self.size
        frames = np.zeros((self.n, s, s, 3), dtype=np.float32)
        # dim grid
        frames[:, :, :, :] = 0.05
        frames[:, ::8, :, 1] = 0.12
        frames[:, :, ::8, 1] = 0.12
        # border
        frames[:, :2, :] = 0.25
        frames[:, -2:, :] = 0.25
        frames[:, :, :2] = 0.25
        frames[:, :, -2:] = 0.25
        _disks(frames, self.ball[:, 0, 1], self.ball[:, 0, 0], 3.2, (0.85, 0.25, 0.25))
        _disks(frames, self.ball[:, 1, 1], self.ball[:, 1, 0], 3.2, (0.35, 0.45, 0.95))
        _disks(frames, self.target[:, 1], self.target[:, 0], 3.5, (0.95, 0.8, 0.15))
        _disks(frames, self.agent[:, 1], self.agent[:, 0], 3.8, (0.2, 0.9, 0.85))
        return frames


class PongEnv:
    """Single-paddle pong. Reward +1 on hit, -1 on miss (then reset ball)."""

    spec = EnvSpec("pong")

    def __init__(self, n: int, size: int = 64, seed: int = 0):
        self.n = n
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.paddle_y = np.zeros((n,), dtype=np.float32)
        self.ball = np.zeros((n, 2), dtype=np.float32)
        self.ball_v = np.zeros((n, 2), dtype=np.float32)
        self.reset()

    def reset(self, mask: np.ndarray | None = None) -> np.ndarray:
        m = np.ones(self.n, dtype=bool) if mask is None else mask
        k = int(m.sum())
        if k:
            self.paddle_y[m] = self.size / 2
            self._serve(m)
        return self.render()

    def _serve(self, m: np.ndarray) -> None:
        k = int(m.sum())
        self.ball[m, 0] = self.size * 0.5
        self.ball[m, 1] = self.rng.uniform(10, self.size - 10, k).astype(np.float32)
        vx = self.rng.choice([-1.0, 1.0], size=k).astype(np.float32) * 1.4
        vy = self.rng.uniform(-1.2, 1.2, k).astype(np.float32)
        self.ball_v[m, 0] = vx
        self.ball_v[m, 1] = vy

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dy = np.zeros(self.n, dtype=np.float32)
        dy[actions == 1] = -2.4
        dy[actions == 2] = 2.4
        self.paddle_y = np.clip(self.paddle_y + dy, 6.0, self.size - 7.0)
        self.ball = self.ball + self.ball_v
        # top/bottom
        hit_y = (self.ball[:, 1] < 3) | (self.ball[:, 1] > self.size - 4)
        self.ball_v[:, 1] = np.where(hit_y, -self.ball_v[:, 1], self.ball_v[:, 1])
        self.ball[:, 1] = np.clip(self.ball[:, 1], 3, self.size - 4)
        # right wall bounce
        hit_r = self.ball[:, 0] > self.size - 4
        self.ball_v[:, 0] = np.where(hit_r, -np.abs(self.ball_v[:, 0]), self.ball_v[:, 0])
        # paddle at x=6
        reward = np.zeros(self.n, dtype=np.float32)
        px = 6.0
        near = (self.ball[:, 0] < px + 3) & (self.ball_v[:, 0] < 0)
        aligned = np.abs(self.ball[:, 1] - self.paddle_y) < 7.0
        hit = near & aligned
        miss = (self.ball[:, 0] < 2) & (self.ball_v[:, 0] < 0) & ~aligned
        self.ball_v[:, 0] = np.where(hit, np.abs(self.ball_v[:, 0]) * 1.02, self.ball_v[:, 0])
        self.ball[hit, 0] = px + 3
        reward[hit] = 1.0
        reward[miss] = -1.0
        if miss.any():
            self._serve(miss)
        done = np.zeros(self.n, dtype=np.float32)
        return self.render(), reward, done

    def render(self) -> np.ndarray:
        s = self.size
        frames = np.zeros((self.n, s, s, 3), dtype=np.float32)
        frames[..., 1] = 0.04
        # center dashed line
        frames[:, :, s // 2 - 1 : s // 2 + 1, :] = 0.2
        # paddle
        for i in range(self.n):
            y = int(self.paddle_y[i])
            frames[i, max(0, y - 6) : min(s, y + 7), 4:7, :] = (0.9, 0.9, 0.95)
        _disks(frames, self.ball[:, 1], self.ball[:, 0], 2.4, (0.95, 0.95, 0.9))
        return frames


class BallsEnv:
    """Two bouncing disks. No agent. Action is ignored (noop world). Good tokenizer smoke test."""

    spec = EnvSpec("balls")

    def __init__(self, n: int, size: int = 64, seed: int = 0):
        self.n = n
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.p = np.zeros((n, 2, 2), dtype=np.float32)
        self.v = np.zeros((n, 2, 2), dtype=np.float32)
        self.reset()

    def reset(self, mask: np.ndarray | None = None) -> np.ndarray:
        m = np.ones(self.n, dtype=bool) if mask is None else mask
        k = int(m.sum())
        if k:
            lo, hi = 8.0, self.size - 8.0
            self.p[m] = self.rng.uniform(lo, hi, (k, 2, 2)).astype(np.float32)
            self.v[m] = self.rng.choice([-1.0, 1.0], size=(k, 2, 2)).astype(np.float32)
        return self.render()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nxt = self.p + self.v
        hit = (nxt < 4) | (nxt > self.size - 5)
        self.v = np.where(hit, -self.v, self.v)
        self.p = np.clip(self.p + self.v, 4, self.size - 5)
        # elastic swap on overlap
        d = self.p[:, 0] - self.p[:, 1]
        dist = np.linalg.norm(d, axis=-1)
        coll = dist < 7.0
        if coll.any():
            self.v[coll, 0], self.v[coll, 1] = self.v[coll, 1].copy(), self.v[coll, 0].copy()
        reward = np.zeros(self.n, dtype=np.float32)
        done = np.zeros(self.n, dtype=np.float32)
        return self.render(), reward, done

    def render(self) -> np.ndarray:
        s = self.size
        frames = np.full((self.n, s, s, 3), 0.06, dtype=np.float32)
        _disks(frames, self.p[:, 0, 1], self.p[:, 0, 0], 3.5, (0.95, 0.35, 0.25))
        _disks(frames, self.p[:, 1, 1], self.p[:, 1, 0], 3.5, (0.25, 0.75, 0.95))
        return frames


ENVS = {"arena": ArenaEnv, "pong": PongEnv, "balls": BallsEnv}


def make_env(name: str, n: int, size: int = 64, seed: int = 0):
    if name not in ENVS:
        raise KeyError(f"unknown env {name!r}, have {list(ENVS)}")
    return ENVS[name](n=n, size=size, seed=seed)


def heuristic_policy(env, frames: np.ndarray) -> np.ndarray:
    """Cheap expert-ish policy so dynamics see more than random noise."""
    n = frames.shape[0]
    a = np.zeros(n, dtype=np.int32)
    if isinstance(env, ArenaEnv):
        d = env.target - env.agent
        ax = np.abs(d[:, 0]) >= np.abs(d[:, 1])
        a[ax & (d[:, 0] > 0)] = 4
        a[ax & (d[:, 0] < 0)] = 3
        a[~ax & (d[:, 1] > 0)] = 2
        a[~ax & (d[:, 1] < 0)] = 1
        # epsilon random
        rnd = env.rng.random(n) < 0.15
        a[rnd] = env.rng.integers(0, 5, int(rnd.sum()))
    elif isinstance(env, PongEnv):
        d = env.ball[:, 1] - env.paddle_y
        a[d < -1.5] = 1
        a[d > 1.5] = 2
        rnd = env.rng.random(n) < 0.1
        a[rnd] = env.rng.integers(0, 5, int(rnd.sum()))
    else:
        a[:] = 0
    return a
