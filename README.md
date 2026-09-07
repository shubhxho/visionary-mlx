---
library_name: mlx
tags:
  - mlx
  - world-model
  - video
  - reinforcement-learning
---

# Visionary MLX

MLX video tokenizer, action-conditioned world model, and imagination-trained agent.

`champion` is the high-capacity configuration for the included 64px synthetic-video workload. It is not a state-of-the-art claim: release a trained checkpoint and its held-out report first.

```sh
visionary-mlx pipeline --config champion --env arena --run-dir runs/champion
visionary-mlx collect --env arena --data-dir data/eval --seed 100 --num-episodes 256
visionary-mlx benchmark --config champion --env arena --run-dir runs/champion \
  --eval-data-dir data/eval --name visionary-mlx-champion --out-dir releases/v0.2.0
python -m pytest
```

The benchmark command writes JSON and Markdown evidence from a separate,
deterministic held-out rollout set. The shipped environments are synthetic
(`arena`, `pong`, and `balls`), so their scores are not comparable to external
vision or video benchmarks.

## Release status

The `champion` run is complete. Release `v0.2.0` contains tokenizer, dynamics,
and agent safetensors, the exact configuration, a dream preview, checksums,
and deterministic held-out benchmark reports. On the repository's separate
synthetic `arena` evaluation set (32 batches), it achieved tokenizer PSNR
18.611 / MAE 0.02022 and open-loop dream PSNR 18.610 / MAE 0.02023. These are
synthetic-workload results only, not an external state-of-the-art claim.

Long MLX runs checkpoint tokenizer, dynamics, and agent weights periodically;
preview generation is non-fatal when local storage is exhausted.

## Use a released checkpoint

The release directory contains `model_config.json` and three safetensors files
(`tokenizer`, `dynamics`, and `agent`). Install MLX on Apple Silicon, then use
the matching `--config` name and run directory:

```sh
# A dream requires a video/action rollout. Create one if you do not already
# have a compatible arena.npz dataset.
uv run visionary-mlx collect --config champion --env arena --data-dir data/eval \
  --num-episodes 256 --episode-len 32 --image-size 64 --seed 100
uv run visionary-mlx dream --config champion --run-dir releases/v0.2.0 \
  --data-dir data/eval --env arena --out dreams
uv run visionary-mlx benchmark --config champion --run-dir releases/v0.2.0 \
  --eval-data-dir data/eval --env arena --name visionary-mlx-champion
```
