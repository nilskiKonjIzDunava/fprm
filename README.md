# fprm
This code accompanies the paper:\
\[[arXiv](https://neurips.cc/)\] \[[Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers](https://arxiv.org/abs/2606.18206)\]\

## Installation

Requirements: Linux with an NVIDIA GPU (CUDA 12.x driver) and Python 3.11–3.12. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/nilskiKonjIzDunava/fprm.git
cd fprm
uv sync
```

`uv sync` creates a project-local `.venv/` and installs everything (PyTorch is pulled from the CUDA 12.9 wheel index). Run commands either via `uv run <cmd>` or after `source .venv/bin/activate` — always from the repository root.

Training logs to Weights & Biases: run `wandb login` first, or set `WANDB_MODE=offline` to disable.

## Usage

### 1. Build the datasets

Datasets are not shipped; build them into `data/` first. Sudoku and Maze download automatically from the Hugging Face Hub:

```bash
# Sudoku-Extreme: 1000 puzzles x 1000 augmentations
uv run python dataset/build_sudoku_dataset.py \
    --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000

# Maze-Hard 30x30: non-augmented (the default; the 8x augmented build collapses to ~5%)
uv run python dataset/build_maze_dataset.py \
    --output-dir data/maze-30x30-hard-1k-noaug
```

### 2. Train

`pretrain.py` is the (Hydra) training entrypoint; the reproduced experiment configs live in `config/`.

Sudoku fits on a single GPU:

```bash
uv run python pretrain.py --config-name cfg_pretrain_sudoku
```

Any field can be overridden on the command line — e.g. the seed and W&B naming:

```bash
uv run python pretrain.py --config-name cfg_pretrain_sudoku \
    seed=1 +project_name=fprm-sudoku +run_name=my-run +checkpoint_path=checkpoints/my-run
```

`arch=fprm` is the paper's model; `trm`, `trm_singlez`, `trm_hier6`, `hrm`, and `transformers_baseline` are also available.

#### Multi-GPU (Maze)

The Maze config uses `n_backwards_L=6`, which exceeds a single 40 GB GPU at `global_batch_size=768`. Train it across multiple GPUs with `torchrun` — e.g. 4× A100-80GB on one node, or 8× A100-40GB across two:

```bash
# 1 node, 4 GPUs
torchrun --standalone --nproc-per-node=4 \
    pretrain.py --config-name cfg_pretrain_maze

# 2 nodes x 4 GPUs (one torchrun per node; HEAD_NODE = rank-0 hostname)
torchrun --nnodes=2 --nproc-per-node=4 \
    --rdzv-backend=c10d --rdzv-endpoint="$HEAD_NODE:29850" \
    pretrain.py --config-name cfg_pretrain_maze
```

`global_batch_size=768` is fixed regardless of GPU count, so results match across these configurations. Sudoku trains on a single GPU as shown above. For the multi-seed runs, set `seed=0`, `seed=1`, `seed=2`.

## Citation

```bibtex
@misc{movahedi2026fixedpointreasonersstableadaptive,
      title={Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers}, 
      author={Sajad Movahedi and Vera Milovanović and Shlomo Libo Feigin and Alexander Theus and Thomas Hofmann and Valentina Boeva and T. Konstantin Rusch and Antonio Orvieto},
      year={2026},
      eprint={2606.18206},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2606.18206}, 
}
```

## Acknowledgments

The puzzle tasks built on [TRM](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) and state-tracking on [FP-RNN](https://github.com/dr-faustus/fp-rnn)
