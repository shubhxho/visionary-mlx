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

This repository is the source release and model card for the ongoing
`champion` training run. No pretrained weights or external benchmark claims
are published yet. The first weight release will include safetensors,
configuration, deterministic held-out benchmark JSON/Markdown, and the
training commit SHA.

Long MLX runs checkpoint tokenizer, dynamics, and agent weights periodically;
preview generation is non-fatal when local storage is exhausted.
