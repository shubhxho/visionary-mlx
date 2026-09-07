"""CLI: collect, train-tokenizer, train-dynamics, train-agent, train-rssm, dream, pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from visionary_mlx.agent import ActorCritic
from visionary_mlx.benchmark import evaluate, write_leaderboard
from visionary_mlx.checkpoint import load_weights, save_json
from visionary_mlx.config import CONFIGS, ModelConfig, TrainConfig
from visionary_mlx.data import ClipLoader, collect_rollouts, load_dataset, save_dataset
from visionary_mlx.dynamics import DynamicsModel
from visionary_mlx.imagine import dream_clip, write_dream_gifs
from visionary_mlx.layers import count_params
from visionary_mlx.rssm import RSSM
from visionary_mlx.tokenizer import VideoTokenizer
from visionary_mlx.train import train_agent, train_dynamics, train_rssm, train_tokenizer


def _paths(args) -> tuple[Path, Path]:
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    data = Path(args.data_dir)
    return run, data


def _model_cfg(name: str) -> ModelConfig:
    if name not in CONFIGS:
        raise SystemExit(f"unknown config {name}, have {list(CONFIGS)}")
    return CONFIGS[name]()


def cmd_collect(args) -> None:
    run, data = _paths(args)
    print(f"collecting {args.num_episodes} {args.env} episodes of {args.episode_len} frames")
    ds = collect_rollouts(
        args.env,
        num_episodes=args.num_episodes,
        episode_len=args.episode_len,
        size=args.image_size,
        seed=args.seed,
        parallel=args.parallel,
    )
    path = save_dataset(ds, data / f"{args.env}.npz")
    print(f"wrote {path} video={ds['video'].shape} actions={ds['actions'].shape}")
    save_json(
        {"env": args.env, "episodes": int(ds["video"].shape[0]), "T": int(ds["video"].shape[1])},
        run / "dataset.json",
    )


def cmd_train_tokenizer(args) -> None:
    run, data = _paths(args)
    mcfg = _model_cfg(args.config)
    save_json(mcfg.to_dict(), run / "model_config.json")
    tcfg = _train_cfg(args)
    ds = load_dataset(data / f"{args.env}.npz")
    loader = ClipLoader(ds, seq_len=tcfg.seq_len, batch_size=tcfg.batch_size, seed=tcfg.seed)
    mx.random.seed(tcfg.seed)
    model = VideoTokenizer(mcfg.tokenizer)
    print(f"tokenizer params={count_params(model):,}")
    train_tokenizer(model, loader, tcfg, run, steps=args.steps)


def cmd_train_dynamics(args) -> None:
    run, data = _paths(args)
    mcfg = _model_cfg(args.config)
    save_json(mcfg.to_dict(), run / "model_config.json")
    tcfg = _train_cfg(args)
    ds = load_dataset(data / f"{args.env}.npz")
    loader = ClipLoader(ds, seq_len=tcfg.seq_len, batch_size=tcfg.batch_size, seed=tcfg.seed)
    mx.random.seed(tcfg.seed)
    tok = VideoTokenizer(mcfg.tokenizer)
    load_weights(tok, run / "tokenizer.safetensors")
    dyn = DynamicsModel(mcfg.dynamics, mcfg.tokenizer)
    print(f"dynamics params={count_params(dyn):,}")
    train_dynamics(dyn, tok, loader, tcfg, run, steps=args.steps)


def cmd_train_agent(args) -> None:
    run, data = _paths(args)
    mcfg = _model_cfg(args.config)
    save_json(mcfg.to_dict(), run / "model_config.json")
    tcfg = _train_cfg(args)
    ds = load_dataset(data / f"{args.env}.npz")
    loader = ClipLoader(ds, seq_len=tcfg.seq_len, batch_size=tcfg.batch_size, seed=tcfg.seed)
    mx.random.seed(tcfg.seed)
    tok = VideoTokenizer(mcfg.tokenizer)
    load_weights(tok, run / "tokenizer.safetensors")
    dyn = DynamicsModel(mcfg.dynamics, mcfg.tokenizer)
    load_weights(dyn, run / "dynamics.safetensors")
    latent_dim = mcfg.tokenizer.num_latents * mcfg.tokenizer.channel_dim
    agent = ActorCritic(latent_dim, mcfg.agent)
    print(f"agent params={count_params(agent):,}")
    train_agent(agent, dyn, tok, loader, tcfg, mcfg, run, steps=args.steps)


def cmd_train_rssm(args) -> None:
    run, data = _paths(args)
    tcfg = _train_cfg(args)
    ds = load_dataset(data / f"{args.env}.npz")
    loader = ClipLoader(ds, seq_len=tcfg.seq_len, batch_size=tcfg.batch_size, seed=tcfg.seed)
    mx.random.seed(tcfg.seed)
    model = RSSM(z=32, hdim=128, num_actions=5)
    print(f"rssm params={count_params(model):,}")
    train_rssm(model, loader, tcfg, run, steps=args.steps)


def cmd_dream(args) -> None:
    run, data = _paths(args)
    mcfg = _model_cfg(args.config)
    save_json(mcfg.to_dict(), run / "model_config.json")
    ds = load_dataset(data / f"{args.env}.npz")
    loader = ClipLoader(ds, seq_len=args.seq_len, batch_size=args.num, seed=args.seed)
    batch = loader.sample()
    tok = VideoTokenizer(mcfg.tokenizer)
    load_weights(tok, run / "tokenizer.safetensors")
    dyn = DynamicsModel(mcfg.dynamics, mcfg.tokenizer)
    load_weights(dyn, run / "dynamics.safetensors")
    video = mx.array(batch["video"])
    actions = mx.array(batch["actions"])
    out = dream_clip(tok, dyn, video, actions, context=args.context)
    paths = write_dream_gifs(out, Path(args.out or (run / "dreams")))
    print("wrote", ", ".join(str(p) for p in paths))


def cmd_pipeline(args) -> None:
    cmd_collect(args)
    cmd_train_tokenizer(args)
    cmd_train_dynamics(args)
    cmd_train_agent(args)
    cmd_dream(args)


def cmd_benchmark(args) -> None:
    """Score a trained checkpoint and write release-ready benchmark artifacts."""
    run, _ = _paths(args)
    mcfg = _model_cfg(args.config)
    save_json(mcfg.to_dict(), run / "model_config.json")
    eval_data = Path(args.eval_data_dir or args.data_dir)
    ds = load_dataset(eval_data / f"{args.env}.npz")
    loader = ClipLoader(ds, seq_len=args.seq_len, batch_size=args.batch_size, seed=args.seed)
    tok = VideoTokenizer(mcfg.tokenizer)
    load_weights(tok, run / "tokenizer.safetensors")
    dyn = None
    if (run / "dynamics.safetensors").exists():
        dyn = DynamicsModel(mcfg.dynamics, mcfg.tokenizer)
        load_weights(dyn, run / "dynamics.safetensors")
    result = evaluate(args.name or run.name, tok, loader, dyn, batches=args.batches)
    json_path, markdown_path = write_leaderboard([result], args.out_dir or (run / "benchmark"))
    print(f"wrote {json_path} and {markdown_path}")


def _train_cfg(args) -> TrainConfig:
    return TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        tokenizer_steps=args.steps,
        dynamics_steps=args.steps,
        agent_steps=max(args.steps // 2, 50),
        log_every=args.log_every,
        eval_every=args.eval_every,
        data_dir=args.data_dir,
        run_dir=args.run_dir,
        env=args.env,
        num_episodes=args.num_episodes,
        episode_len=args.episode_len,
    )


def _add_shared(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="tiny", choices=list(CONFIGS))
    p.add_argument("--env", default="arena", choices=["arena", "pong", "balls"])
    p.add_argument("--run-dir", default="runs/default")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--num-episodes", type=int, default=128)
    p.add_argument("--episode-len", type=int, default=32)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--parallel", type=int, default=16)
    p.add_argument("--context", type=int, default=2)
    p.add_argument("--num", type=int, default=4)
    p.add_argument("--out", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--eval-data-dir", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--batches", type=int, default=8)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="visionary-mlx", description="World models on MLX")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn in [
        ("collect", cmd_collect),
        ("train-tokenizer", cmd_train_tokenizer),
        ("train-dynamics", cmd_train_dynamics),
        ("train-agent", cmd_train_agent),
        ("train-rssm", cmd_train_rssm),
        ("dream", cmd_dream),
        ("pipeline", cmd_pipeline),
        ("benchmark", cmd_benchmark),
    ]:
        p = sub.add_parser(name)
        _add_shared(p)
        p.set_defaults(func=fn)
    args = parser.parse_args(argv)
    mx.set_default_device(mx.gpu)
    print(f"mlx device={mx.default_device()}")
    args.func(args)


if __name__ == "__main__":
    main()
