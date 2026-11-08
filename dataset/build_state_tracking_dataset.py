"""Build a state-tracking (group word-problem) dataset in PuzzleDataset format.

Each input sequence is a sequence of group elements; the label at position i is
the running product of the first i+1 inputs.
"""

from typing import List
import os
import json

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel
from abstract_algebra.finite_algebras import (
    generate_cyclic_group,
    generate_symmetric_group,
)

from common import PuzzleDatasetMetadata


cli = ArgParser()


class StateTrackingConfig(BaseModel):
    group: str = "S5"               # one of S<n>, Z<n>, A<n>
    k_train: int = 20               # max training sequence length
    k_eval_max: int = 50            # max evaluation length (= test seq_len, model rotary cache)
    train_samples: int = 100_000    # total training samples (split across train lengths)
    test_samples: int = 1_000       # number of test samples (full-length)
    strict_len: bool = True         # train only on length k_train; else lengths 2..k_train
    seed: int = 0
    output_dir: str = "data/state_tracking-S5"


def _make_group(spec: str):
    """Generate an group from a string identifier."""
    kind, n = spec[0], int(spec[1:])
    if kind == "S":
        return generate_symmetric_group(n)
    if kind == "Z":
        return generate_cyclic_group(n)
    if kind == "A":
        s_n = generate_symmetric_group(n)
        a_n = s_n.commutator_subalgebra()
        a_n.name = f"A{n}"
        return a_n
    raise ValueError(f"Unknown group spec {spec!r} (expected S<n>, Z<n>, or A<n>)")


def _build_cayley_table(group) -> np.ndarray:
    """Index-space multiplication table: cayley[i, j] = idx(elements[i] * elements[j])."""
    elems = group.elements
    n = len(elems)
    table = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            table[i, j] = elems.index(group.op(elems[i], elems[j]))
    return table


def _generate(cayley: np.ndarray, k: int, num_samples: int, rng: np.random.Generator):
    """Returns (inputs, labels) of shape (num_samples, k); index 0 is the identity (acc start).

    Sequences are deduplicated.
    """
    n = cayley.shape[0]

    # Cap requested samples at the number of distinct length-k sequences (n**k).
    capacity = n ** k
    if num_samples > capacity:
        print(f"Warning: requested {num_samples} samples but only {capacity} unique "
              f"length-{k} sequences exist; truncating to {capacity}.")
        num_samples = capacity

    seen = set()
    rows = []
    # Oversample in batches to amortise rejection cost; the loop terminates as
    # soon as `num_samples` unique sequences have been collected.
    while len(rows) < num_samples:
        remaining = num_samples - len(rows)
        cand = rng.integers(0, n, size=(remaining, k), dtype=np.int64)
        for row in cand:
            key = row.tobytes()
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
            if len(rows) == num_samples:
                break

    inputs = np.stack(rows)
    labels = np.empty_like(inputs)
    acc = np.zeros(num_samples, dtype=np.int64)
    for j in range(k):
        acc = cayley[acc, inputs[:, j]]
        labels[:, j] = acc
    return inputs, labels


def _save_set(out_dir: str, set_name: str, inputs: np.ndarray, labels: np.ndarray):
    n = len(inputs)
    np.save(os.path.join(out_dir, f"{set_name}__inputs.npy"), inputs.astype(np.int32))
    np.save(os.path.join(out_dir, f"{set_name}__labels.npy"), labels.astype(np.int32))
    np.save(os.path.join(out_dir, f"{set_name}__puzzle_identifiers.npy"), np.zeros((n,), dtype=np.int32))
    np.save(os.path.join(out_dir, f"{set_name}__puzzle_indices.npy"), np.arange(n + 1, dtype=np.int32))
    np.save(os.path.join(out_dir, f"{set_name}__group_indices.npy"), np.arange(n + 1, dtype=np.int32))


def _write_metadata(
    out_dir: str,
    sets: List[str],
    total_examples: int,
    seq_len: int,
    vocab_size: int,
):
    metadata = PuzzleDatasetMetadata(
        seq_len=seq_len,
        vocab_size=vocab_size,
        pad_id=0,
        ignore_label_id=0,                 # PAD positions are ignored in loss
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=total_examples,
        mean_puzzle_examples=1.0,
        total_puzzles=total_examples,
        sets=sets,
    )
    with open(os.path.join(out_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)


@cli.command(singleton=True)
def build(config: StateTrackingConfig):
    rng = np.random.default_rng(config.seed)
    group = _make_group(config.group)
    cayley = _build_cayley_table(group)
    n_elements = cayley.shape[0]

    pad_id = 0
    elem_offset = 1
    vocab_size = n_elements + 1
    seq_len = config.k_eval_max  # model rotary cache length

    # ---- TRAIN ----
    train_lens = [config.k_train] if config.strict_len else list(range(2, config.k_train + 1))
    samples_per_len = max(1, config.train_samples // len(train_lens))

    n_train = samples_per_len * len(train_lens)
    train_in = np.full((n_train, seq_len), pad_id, dtype=np.int64)
    train_lb = np.full((n_train, seq_len), pad_id, dtype=np.int64)
    cursor = 0
    for k in train_lens:
        raw_in, raw_lb = _generate(cayley, k, samples_per_len, rng)
        train_in[cursor:cursor + samples_per_len, :k] = raw_in + elem_offset
        train_lb[cursor:cursor + samples_per_len, :k] = raw_lb + elem_offset
        cursor += samples_per_len

    perm = rng.permutation(n_train)
    train_in = train_in[perm]
    train_lb = train_lb[perm]

    train_dir = os.path.join(config.output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    _save_set(train_dir, "all", train_in, train_lb)
    _write_metadata(train_dir, ["all"], n_train, seq_len, vocab_size)

    # ---- TEST ---- one physical set at full length. The eval grid (which
    # prefix lengths to score) lives in PretrainConfig (k_eval_min/max/step),
    # not in the dataset.
    raw_in, raw_lb = _generate(cayley, config.k_eval_max, config.test_samples, rng)
    full_in = (raw_in + elem_offset).astype(np.int64)
    full_lb = (raw_lb + elem_offset).astype(np.int64)

    test_dir = os.path.join(config.output_dir, "test")
    os.makedirs(test_dir, exist_ok=True)
    _save_set(test_dir, "all", full_in, full_lb)

    _write_metadata(
        test_dir,
        sets=["all"],
        total_examples=config.test_samples,
        seq_len=seq_len,
        vocab_size=vocab_size,
    )

    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

    print(f"Wrote state-tracking dataset to {config.output_dir} "
          f"(group={config.group}, vocab_size={vocab_size}, "
          f"train_examples={n_train}, seq_len={seq_len})")


if __name__ == "__main__":
    cli()
