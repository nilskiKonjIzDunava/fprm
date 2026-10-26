"""Eval-only / inference entrypoint.

Loads a trained checkpoint (local file or Hugging Face Hub) into the model
described by a normal training config, then runs it through the *same*
`pretrain.evaluate()` used at the end of every training iteration: same data
loader, same ACT/fixed-point inference loop, same metrics, same length-gen grid,
same evaluators.

Usage (from the repo root; same Hydra syntax as pretrain.py):

    # checkpoint on the Hub:  hf://<org>/<repo>[@revision]/<path/in/repo>
    uv run python inference.py --config-name cfg_pretrain_state_tracking_fprm_s5 \
        +load_checkpoint=hf://some-org/fprm-s5/step_390625 \
        +metrics_out=results/s5.json

    # local checkpoint, test-time scaling on the fixed-point budget
    uv run python inference.py --config-name cfg_pretrain_sudoku \
        +load_checkpoint=checkpoints/my-run/step_60000 arch.max_iter_eval=2000

    # also dump the predictions (written to <checkpoint_path>/step_<N>_all_preds.0)
    ... +checkpoint_path=outputs/infer-s5 '+eval_save_outputs=[preds]'

    # evaluate only 1000 evenly spaced test puzzles (one every ~N/1000) instead of all
    ... +n_subset=1000

The test split of `data_paths` (or `data_paths_test`) must have been built with
dataset/build_*.py, exactly as for training. Private repos: set HF_TOKEN.
"""
import os
import re
import json

import numpy as np
import torch
import hydra
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from huggingface_hub import hf_hub_download

from pretrain import TrainState, create_evaluators, evaluate
from pretrain_config import PretrainConfig
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from utils.flop_profiler import FlopProfiler
from utils.functions import load_model_class
from plots.io import write_curves


class SubsetPuzzleDataset(PuzzleDataset):
    """PuzzleDataset whose (test) data is thinned to `n_subset` evenly spaced examples.

    With n_subset=1000 on the ~423k Sudoku-Extreme test puzzles this keeps one
    puzzle every ~423 (indices floor(i * N / n_subset)). Only valid for datasets
    with one example per puzzle (sudoku, maze, state tracking), which is checked.
    """

    def __init__(self, config: PuzzleDatasetConfig, split: str = "test", n_subset=None):
        super().__init__(config, split)
        self.n_subset = n_subset

    def _lazy_load_dataset(self):
        if self._data is not None:
            return
        super()._lazy_load_dataset()
        if not self.n_subset:
            return
        for set_name, d in self._data.items():  # type: ignore
            n = len(d["inputs"])
            assert np.array_equal(d["puzzle_indices"], np.arange(n + 1)), \
                "n_subset assumes one example per puzzle"
            k = min(self.n_subset, n)
            idx = (np.arange(k) * n) // k  # evenly spaced, sorted -> cheap memmap read
            self._data[set_name] = {
                "inputs": d["inputs"][idx],
                "labels": d["labels"][idx],
                "puzzle_identifiers": d["puzzle_identifiers"][idx],
                "puzzle_indices": np.arange(k + 1, dtype=d["puzzle_indices"].dtype),
                "group_indices": np.arange(k + 1, dtype=d["group_indices"].dtype),
            }


def create_test_loader(config: PretrainConfig, n_subset=None):
    """Same as pretrain.create_dataloader(config, "test", test_set_mode=True, ...), single
    process, with optional subsampling of the test set."""
    dataset = SubsetPuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if len(config.data_paths_test) > 0 else config.data_paths,
        global_batch_size=config.global_batch_size,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    ), split="test", n_subset=n_subset)
    loader = DataLoader(dataset, batch_size=None, num_workers=1, prefetch_factor=8,
                        pin_memory=True, persistent_workers=True)
    return loader, dataset.metadata


def resolve_checkpoint(path: str) -> str:
    """`hf://org/repo[@rev]/file` -> local cached file; anything else is returned as is."""
    if not path.startswith("hf://"):
        return path
    org, name, filename = path[len("hf://"):].split("/", 2)
    repo_id, _, revision = f"{org}/{name}".partition("@")
    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision or None)


def build_model(config: PretrainConfig, metadata, world_size: int = 1) -> torch.nn.Module:
    """Same construction as create_model.create_model (model + loss head), minus
    optimizers / compile / checkpoint loading."""
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
    )
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)
    with torch.device("cuda"):
        model = model_cls(model_cfg)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
    return model


def load_weights(model: torch.nn.Module, path: str) -> None:
    """Load `step_<N>` (plain state_dict) or a `step_<N>_train_state.pt` bundle.

    Checkpoints are saved from the torch.compile'd model, so keys carry an
    `_orig_mod.` prefix; it is stripped so the same file loads compiled or not.
    """
    state = torch.load(path, map_location="cuda")
    if "model" in state:  # resume bundle (utils/resume.py)
        state = state["model"]
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}

    # Puzzle-embedding table: re-init from the mean if the identifier count differs
    # (same behaviour as create_model.load_checkpoint). Absent when puzzle_emb_ndim == 0.
    puzzle_emb = getattr(model.model, "puzzle_emb", None)  # type: ignore
    name = "model.inner.puzzle_emb.weights"
    if puzzle_emb is not None and name in state and state[name].shape != puzzle_emb.weights.shape:
        print(f"Resetting puzzle embedding: found {tuple(state[name].shape)}, expected {tuple(puzzle_emb.weights.shape)}")
        state[name] = state[name].mean(dim=0, keepdim=True).expand(puzzle_emb.weights.shape).contiguous()

    # Tolerate checkpoints that predate the CastedLinear `.mask` buffer (as in utils/resume.py).
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = [k for k in missing if not k.endswith(".mask")]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"checkpoint does not match the config's architecture: "
            f"missing={bad_missing[:5]} unexpected={list(unexpected)[:5]}"
        )


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    raw = OmegaConf.to_container(hydra_config, resolve=True)
    n_subset = raw.pop("n_subset", None)  # not a PretrainConfig field: +n_subset=1000
    config = PretrainConfig(**raw)  # type: ignore
    assert config.load_checkpoint, "set +load_checkpoint=<local path> or hf://<org>/<repo>/<file>"

    torch.random.manual_seed(config.seed)  # fixed-point init is random (trunc-normal at reset)

    # Data: same dataset class/metadata as training, single process.
    eval_loader, eval_metadata = create_test_loader(config, n_subset=n_subset)
    try:
        evaluators = create_evaluators(config, eval_metadata)
    except Exception:
        print("No evaluator found")
        evaluators = []

    # Model + weights (load before compile so state-dict keys are unprefixed).
    ckpt_path = resolve_checkpoint(config.load_checkpoint)
    print(f"Loading checkpoint {ckpt_path}")
    model = build_model(config, eval_metadata)
    load_weights(model, ckpt_path)
    if "DISABLE_COMPILE" not in os.environ:
        model = torch.compile(model)  # type: ignore
    model.eval()

    m = re.search(r"step_(\d+)", os.path.basename(ckpt_path))
    step = int(m.group(1)) if m else 0
    train_state = TrainState(model=model, optimizers=[], optimizer_lrs=[], carry=None, step=step, total_steps=step)

    metrics = evaluate(
        config, train_state, eval_loader, eval_metadata, evaluators,
        rank=0, world_size=1, cpu_group=None, profiler=FlopProfiler(enabled=config.profile_flops),
    )

    print(json.dumps(metrics, indent=2, default=float))
    if config.metrics_out and metrics and "length_gen" in metrics:
        write_curves({config.run_name or "run": metrics["length_gen"]}, config.metrics_out)
        print(f"Wrote per-k metrics to {config.metrics_out}")


if __name__ == "__main__":
    launch()