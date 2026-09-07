---
library_name: mlx
tags:
  - mlx
  - world-model
  - video
  - reinforcement-learning
---

# Visionary MLX Champion v0.2.0

This is an end-to-end MLX world-model release trained on the repository's
synthetic `arena` environment: video tokenizer, action-conditioned flow
dynamics, and imagination-trained actor-critic agent.

## Contents

- `tokenizer.safetensors` — 30.6M-parameter video tokenizer.
- `dynamics.safetensors` — 22.8M-parameter flow dynamics model.
- `agent.safetensors` — 2.6M-parameter imagined-rollout actor-critic.
- `model_config.json` — exact architecture configuration.
- `benchmark.json` and `BENCHMARK.md` — deterministic held-out results.
- `dream_montage.gif` — qualitative open-loop prediction preview.

## Reproducible held-out evaluation

The checkpoint was benchmarked on 32 batches drawn from a separate 256-episode
synthetic `arena` rollout set (seed 100), never used for training.

| Metric | Score |
| --- | ---: |
| Tokenizer PSNR | 18.611 |
| Tokenizer MAE | 0.02022 |
| Open-loop dream PSNR | 18.610 |
| Open-loop dream MAE | 0.02023 |

These are synthetic-workload results only. They are not comparable to external
vision/video benchmarks and do not establish a general state-of-the-art claim.

## Use

On Apple Silicon with MLX installed, place this directory under `releases/` in
the source checkout, then use the matching `champion` configuration:

```sh
uv run visionary-mlx dream --config champion --run-dir releases/v0.2.0 \
  --data-dir data/eval --env arena --out dreams
uv run visionary-mlx benchmark --config champion --run-dir releases/v0.2.0 \
  --eval-data-dir data/eval --env arena --name visionary-mlx-champion-v0.2.0
```

Source commit: `f1ff2c74d9fec812c7579439a6bded99ccf3b729`.
