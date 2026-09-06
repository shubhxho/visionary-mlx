"""Training loops for tokenizer, dynamics, RSSM, and the imagination agent."""

from __future__ import annotations

import time
from errno import ENOSPC
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from visionary_mlx.agent import ActorCritic, lambda_return
from visionary_mlx.checkpoint import save_json, save_model
from visionary_mlx.config import ModelConfig, TrainConfig
from visionary_mlx.data import ClipLoader
from visionary_mlx.dynamics import DynamicsModel
from visionary_mlx.layers import count_params
from visionary_mlx.metrics import mae, montage_row, psnr, save_gif
from visionary_mlx.rssm import RSSM
from visionary_mlx.tokenizer import VideoTokenizer


def _to_mx(batch: dict) -> dict:
    out = {}
    for k, v in batch.items():
        if k == "actions":
            out[k] = mx.array(v)
        else:
            out[k] = mx.array(v.astype(np.float32))
    return out


def _blur(x: mx.array) -> mx.array:
    """Cheap 3x3 box blur as a perceptual proxy (no VGG). x: (..., H, W, C)."""
    k = mx.ones((3, 3, 1, 1), dtype=x.dtype) / 9.0
    # depthwise via channel loop is slow; average spatially with pad
    xp = mx.pad(x, [(0, 0), (0, 0), (1, 1), (1, 1), (0, 0)])
    acc = None
    for dy in range(3):
        for dx in range(3):
            sl = xp[:, :, dy : dy + x.shape[2], dx : dx + x.shape[3], :]
            acc = sl if acc is None else acc + sl
    return acc / 9.0


def train_tokenizer(
    model: VideoTokenizer,
    loader: ClipLoader,
    cfg: TrainConfig,
    run_dir: Path,
    steps: int | None = None,
) -> dict:
    steps = steps or cfg.tokenizer_steps
    opt = optim.AdamW(learning_rate=cfg.lr, weight_decay=cfg.weight_decay)
    mx.eval(model.parameters())

    def loss_fn(m: VideoTokenizer, video: mx.array):
        recon, mask, latent = m.reconstruct(video)
        mse = mx.mean((recon - video) ** 2)
        blur_l = mx.mean(mx.abs(_blur(recon) - _blur(video)))
        # latent energy — keep tanh from saturating at ±1
        sat = mx.mean(mx.abs(latent) ** 2)
        loss = mse + cfg.lpips_proxy_weight * blur_l + 0.01 * sat
        return loss, {"mse": mse, "blur": blur_l, "sat": sat, "mask": mx.mean(mask.astype(mx.float32))}

    vg = nn.value_and_grad(model, loss_fn)
    logs = []
    t0 = time.time()
    for step in range(1, steps + 1):
        batch = _to_mx(loader.sample())
        (loss, metrics), grads = vg(model, batch["video"])
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        if step % cfg.log_every == 0 or step == 1:
            rec = {k: float(metrics[k].item()) for k in metrics}
            rec["loss"] = float(loss.item())
            rec["step"] = step
            rec["sec"] = time.time() - t0
            logs.append(rec)
            print(
                f"[tok {step:5d}/{steps}] loss={rec['loss']:.4f} mse={rec['mse']:.4f} "
                f"blur={rec['blur']:.4f} {rec['sec']:.1f}s",
                flush=True,
            )
        if step % cfg.eval_every == 0 or step == steps:
            try:
                _eval_tokenizer(model, loader, run_dir / "tokenizer", step)
            except OSError as exc:
                if exc.errno != ENOSPC:
                    raise
                print("  skipping tokenizer preview: disk is full", flush=True)
    save_model(model, run_dir / "tokenizer.safetensors")
    save_json(logs, run_dir / "tokenizer_log.json")
    return {"logs": logs, "params": count_params(model)}


def _eval_tokenizer(model: VideoTokenizer, loader: ClipLoader, out_dir: Path, step: int) -> None:
    batch = loader.sample()
    video = mx.array(batch["video"][:4])
    recon, _, _ = model.reconstruct(video, mask_prob=0.0)
    mx.eval(recon)
    v = np.array(video)
    r = np.array(recon)
    print(f"  eval psnr={psnr(v, r):.2f} mae={mae(v, r):.4f}", flush=True)
    clips = []
    for i in range(min(4, v.shape[0])):
        clips.append(montage_row([v[i], r[i]]))
    grid = np.concatenate(clips, axis=1) if len(clips) > 1 else clips[0]
    save_gif(grid, out_dir / f"recon_step{step:05d}.gif", fps=8, scale=3)


def train_dynamics(
    dyn: DynamicsModel,
    tok: VideoTokenizer,
    loader: ClipLoader,
    cfg: TrainConfig,
    run_dir: Path,
    steps: int | None = None,
) -> dict:
    steps = steps or cfg.dynamics_steps
    tok.eval()
    opt = optim.AdamW(learning_rate=cfg.lr, weight_decay=cfg.weight_decay)
    mx.eval(dyn.parameters())

    def loss_fn(m: DynamicsModel, z, actions, rewards, dones):
        flow, mets = m.flow_loss(z, actions)
        rloss = m.reward_loss(z, actions, rewards, dones)
        return flow + 0.1 * rloss, {**mets, "rloss": rloss}

    vg = nn.value_and_grad(dyn, loss_fn)
    logs = []
    t0 = time.time()
    for step in range(1, steps + 1):
        batch = _to_mx(loader.sample())
        z = mx.stop_gradient(tok.encode_video(batch["video"]))
        (loss, metrics), grads = vg(
            dyn,
            z,
            batch["actions"],
            batch.get("rewards", mx.zeros(batch["actions"].shape)),
            batch.get("dones", mx.zeros(batch["actions"].shape)),
        )
        opt.update(dyn, grads)
        mx.eval(dyn.parameters(), opt.state, loss)
        if step % cfg.log_every == 0 or step == 1:
            rec = {k: float(metrics[k].item()) for k in metrics}
            rec["loss"] = float(loss.item())
            rec["step"] = step
            rec["sec"] = time.time() - t0
            logs.append(rec)
            print(
                f"[dyn {step:5d}/{steps}] loss={rec['loss']:.4f} flow={rec['flow_mse']:.4f} "
                f"{rec['sec']:.1f}s",
                flush=True,
            )
        if step % cfg.eval_every == 0 or step == steps:
            # Persist the learned dynamics before optional preview creation so
            # an out-of-space GIF never loses hours of model training.
            save_model(dyn, run_dir / "dynamics.safetensors")
            try:
                _eval_dynamics(dyn, tok, loader, run_dir / "dynamics", step)
            except OSError as exc:
                if exc.errno != ENOSPC:
                    raise
                print("  skipping dynamics preview: disk is full", flush=True)
    tok.train()
    save_model(dyn, run_dir / "dynamics.safetensors")
    save_json(logs, run_dir / "dynamics_log.json")
    return {"logs": logs, "params": count_params(dyn)}


def _eval_dynamics(dyn: DynamicsModel, tok: VideoTokenizer, loader: ClipLoader, out_dir: Path, step: int) -> None:
    batch = loader.sample()
    video = mx.array(batch["video"][:2])
    actions = mx.array(batch["actions"][:2])
    ctx = max(2, video.shape[1] // 4)
    z = mx.stop_gradient(tok.encode_video(video))
    horizon = video.shape[1] - ctx
    z_roll = dyn.rollout(z[:, :ctx], actions, horizon)
    recon_gt = tok.decode_video(z)
    recon_dream = tok.decode_video(z_roll)
    mx.eval(recon_gt, recon_dream)
    gt = np.array(video)
    dream = np.array(recon_dream)
    print(f"  dream mae={mae(gt, dream):.4f} psnr={psnr(gt, dream):.2f}", flush=True)
    clips = []
    for i in range(gt.shape[0]):
        clips.append(montage_row([gt[i], np.array(recon_gt)[i], dream[i]]))
    grid = np.concatenate(clips, axis=1)
    save_gif(grid, out_dir / f"dream_step{step:05d}.gif", fps=8, scale=3)


def train_agent(
    agent: ActorCritic,
    dyn: DynamicsModel,
    tok: VideoTokenizer,
    loader: ClipLoader,
    cfg: TrainConfig,
    mcfg: ModelConfig,
    run_dir: Path,
    steps: int | None = None,
) -> dict:
    steps = steps or cfg.agent_steps
    tok.eval()
    dyn.eval()
    opt = optim.AdamW(learning_rate=cfg.lr, weight_decay=0.0)
    horizon = mcfg.agent.imagine_horizon
    gamma = mcfg.agent.gamma
    lam = mcfg.agent.lam
    mx.eval(agent.parameters())

    def loss_fn(ag: ActorCritic, z0: mx.array):
        # z0: (B, N, C) starting latent. Imagine H steps with the frozen dynamics.
        b = z0.shape[0]
        z = z0
        logps = []
        values = []
        rewards = []
        continues = []
        entropies = []
        for t in range(horizon):
            logits, v = ag.policy(z)
            action = mx.random.categorical(logits)
            logp = ag.log_prob(logits, action)
            ent = ag.entropy(logits)
            # one-step dynamics: wrap as T=1 sequence
            z_seq = z[:, None, :, :]
            a_seq = action[:, None]
            z_next = mx.stop_gradient(dyn.generate_next(z_seq, a_seq, target_index=0, sample_steps=2))
            finest = dyn.cfg.max_step_size - 1
            step_lv = mx.full((b, 1), finest, dtype=mx.int32)
            sig_lv = mx.full((b, 1), 1 << finest, dtype=mx.int32)
            h = dyn.hidden_for_heads(z_seq, a_seq, step_lv, sig_lv)
            r = dyn.reward_head(h).squeeze(-1).squeeze(-1)
            cont = mx.sigmoid(dyn.continue_head(h).squeeze(-1).squeeze(-1))
            logps.append(logp)
            values.append(v)
            rewards.append(mx.stop_gradient(r))
            continues.append(mx.stop_gradient(cont))
            entropies.append(ent)
            z = mx.stop_gradient(z_next)
        _, v_last = ag.policy(z)
        logps = mx.stack(logps, axis=1)
        values = mx.stack(values + [v_last], axis=1)
        rewards = mx.stack(rewards, axis=1)
        continues = mx.stack(continues, axis=1)
        ent = mx.stack(entropies, axis=1)
        ret = lambda_return(rewards, values, continues, gamma, lam)
        v_t = values[:, :-1]
        adv = mx.stop_gradient(ret - v_t)
        actor = -mx.mean(logps * adv)
        critic = mx.mean((v_t - mx.stop_gradient(ret)) ** 2)
        entropy = -mcfg.agent.entropy_coef * mx.mean(ent)
        loss = actor + mcfg.agent.value_coef * critic + entropy
        return loss, {
            "actor": actor,
            "critic": critic,
            "ent": mx.mean(ent),
            "ret": mx.mean(ret),
            "rew": mx.mean(rewards),
        }

    vg = nn.value_and_grad(agent, loss_fn)
    logs = []
    t0 = time.time()
    for step in range(1, steps + 1):
        batch = _to_mx(loader.sample())
        z = mx.stop_gradient(tok.encode_video(batch["video"]))
        z0 = z[:, 0]
        (loss, metrics), grads = vg(agent, z0)
        opt.update(agent, grads)
        mx.eval(agent.parameters(), opt.state, loss)
        if step % cfg.log_every == 0 or step == 1:
            rec = {k: float(metrics[k].item()) for k in metrics}
            rec["loss"] = float(loss.item())
            rec["step"] = step
            rec["sec"] = time.time() - t0
            logs.append(rec)
            print(
                f"[agent {step:5d}/{steps}] loss={rec['loss']:.4f} rew={rec['rew']:.3f} "
                f"ret={rec['ret']:.3f} {rec['sec']:.1f}s",
                flush=True,
            )
        if step % cfg.eval_every == 0 or step == steps:
            save_model(agent, run_dir / "agent.safetensors")
    tok.train()
    dyn.train()
    save_model(agent, run_dir / "agent.safetensors")
    save_json(logs, run_dir / "agent_log.json")
    return {"logs": logs, "params": count_params(agent)}


def train_rssm(model: RSSM, loader: ClipLoader, cfg: TrainConfig, run_dir: Path, steps: int | None = None) -> dict:
    steps = steps or cfg.tokenizer_steps
    opt = optim.AdamW(learning_rate=cfg.lr, weight_decay=cfg.weight_decay)
    mx.eval(model.parameters())

    def loss_fn(m, video, actions):
        return m.loss(video, actions, context=max(2, video.shape[1] // 4))

    vg = nn.value_and_grad(model, loss_fn)
    logs = []
    t0 = time.time()
    for step in range(1, steps + 1):
        batch = _to_mx(loader.sample())
        (loss, metrics), grads = vg(model, batch["video"], batch["actions"])
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        if step % cfg.log_every == 0 or step == 1:
            rec = {k: float(metrics[k].item()) for k in metrics}
            rec["loss"] = float(loss.item())
            rec["step"] = step
            rec["sec"] = time.time() - t0
            logs.append(rec)
            print(
                f"[rssm {step:5d}/{steps}] loss={rec['loss']:.4f} recon={rec['recon']:.4f} "
                f"dyn={rec['dyn']:.4f} img={rec['imagine']:.4f} {rec['sec']:.1f}s",
                flush=True,
            )
    save_model(model, run_dir / "rssm.safetensors")
    save_json(logs, run_dir / "rssm_log.json")
    return {"logs": logs, "params": count_params(model)}
