from typing import Optional
import os
import csv
import json
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from common import PuzzleDatasetMetadata


cli = ArgParser()


class DataProcessConfig(BaseModel):
    source_repo: str = "sapientinc/sudoku-extreme"
    output_dir: str = "data/sudoku-extreme-full"

    subsample_size: Optional[int] = None
    min_difficulty: Optional[int] = None
    num_aug: int = 0

    # Easy-to-hard empty-cell OOD split (see build_easy_to_hard_split).
    easy_to_hard_split: bool = False
    easy_to_hard_source: Optional[str] = None
    easy_to_hard_train_size: int = 1_000_000
    train_empty_min: int = 39
    train_empty_max: int = 50
    val_test_empty_threshold: int = 50
    train_min_rating: Optional[int] = None
    train_max_rating: Optional[int] = None
    val_test_min_rating: Optional[int] = None
    val_test_max_rating: Optional[int] = None
    val_test_in_dist_per_empty: int = 5000
    seed: int = 0


def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    # Create a random digit mapping: a permutation of 1..9, with zero (blank) unchanged
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))
    
    # Randomly decide whether to transpose.
    transpose_flag = np.random.rand() < 0.5

    # Generate a valid row permutation:
    # - Shuffle the 3 bands (each band = 3 rows) and for each band, shuffle its 3 rows.
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])

    # Similarly for columns (stacks).
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

    # Build an 81->81 mapping. For each new cell at (i, j)
    # (row index = i // 9, col index = i % 9),
    # its value comes from old row = row_perm[i//9] and old col = col_perm[i%9].
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        # Apply transpose flag
        if transpose_flag:
            x = x.T
        # Apply the position mapping.
        new_board = x.flatten()[mapping].reshape(9, 9).copy()
        # Apply digit mapping
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)


def convert_subset(set_name: str, config: DataProcessConfig):
    # Read CSV
    inputs = []
    labels = []
    ratings = []
    
    with open(hf_hub_download(config.source_repo, f"{set_name}.csv", repo_type="dataset"), newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for source, q, a, rating in reader:
            if (config.min_difficulty is None) or (int(rating) >= config.min_difficulty):
                assert len(q) == 81 and len(a) == 81
                
                inputs.append(np.frombuffer(q.replace('.', '0').encode(), dtype=np.uint8).reshape(9, 9) - ord('0'))
                labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(9, 9) - ord('0'))
                ratings.append(int(rating))

    # If subsample_size is specified for the training set,
    # randomly sample the desired number of examples.
    if set_name == "train" and config.subsample_size is not None:
        total_samples = len(inputs)
        if config.subsample_size < total_samples:
            indices = np.random.choice(total_samples, size=config.subsample_size, replace=False)
            inputs = [inputs[i] for i in indices]
            labels = [labels[i] for i in indices]
            ratings = [ratings[i] for i in indices]

    # Generate dataset
    num_augments = config.num_aug if set_name == "train" else 0

    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    group_difficulties = []
    puzzle_id = 0
    example_id = 0
    
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)
    
    for orig_inp, orig_out, rating in zip(tqdm(inputs), labels, ratings):
        for aug_idx in range(1 + num_augments):
            # First index is not augmented
            if aug_idx == 0:
                inp, out = orig_inp, orig_out
            else:
                inp, out = shuffle_sudoku(orig_inp, orig_out)

            # Push puzzle (only single example)
            results["inputs"].append(inp)
            results["labels"].append(out)
            example_id += 1
            puzzle_id += 1
            
            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)
            
        # Push group
        results["group_indices"].append(puzzle_id)
        group_difficulties.append(int(rating))
        
    # To Numpy
    def _seq_to_numpy(seq):
        arr = np.concatenate(seq).reshape(len(seq), -1)
        
        assert np.all((arr >= 0) & (arr <= 9))
        return arr + 1
    
    results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),
        
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=81,
        vocab_size=10 + 1,  # PAD + "0" ... "9"
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(results["group_indices"]) - 1,
        sets=["all"]
    )

    # Save metadata as JSON.
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
        
    # Save data
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
    np.save(os.path.join(save_dir, "all__group_difficulties.npy"), np.array(group_difficulties, dtype=np.int32))
        
    # Save IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


def _fill_cells_to_target(puzzle: np.ndarray, solution: np.ndarray,
                          target_empty: int, rng: np.random.Generator) -> np.ndarray:
    """Reveal random blanks from `solution` until exactly `target_empty` remain.
    Adapted from `add_random_hints` in hrm-mechanistic-analysis (random count there,
    fixed count here). Pre-shift digit space: 0 = blank.
    """
    out = puzzle.copy()
    blank_positions = np.where(puzzle == 0)
    blanks = list(zip(blank_positions[0], blank_positions[1]))
    n_fill = len(blanks) - target_empty
    if n_fill <= 0:
        return out
    sel = rng.choice(len(blanks), size=n_fill, replace=False)
    for idx in sel:
        r, c = blanks[idx]
        out[r, c] = solution[r, c]
    return out


def _build_and_save(inputs, labels, ratings, num_augments: int, save_dir: str):
    """Build index arrays, optionally augment, and write the split to disk.
    inputs/labels: lists of (9,9) uint8 in pre-shift digit space (0 = blank).
    """
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    group_difficulties = []
    puzzle_id = 0
    example_id = 0

    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for orig_inp, orig_out, rating in zip(tqdm(inputs), labels, ratings):
        for aug_idx in range(1 + num_augments):
            # First index is not augmented
            if aug_idx == 0:
                inp, out = orig_inp, orig_out
            else:
                inp, out = shuffle_sudoku(orig_inp, orig_out)

            # Push puzzle (only single example)
            results["inputs"].append(inp)
            results["labels"].append(out)
            example_id += 1
            puzzle_id += 1

            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)

        # Push group
        results["group_indices"].append(puzzle_id)
        group_difficulties.append(int(rating))

    # To Numpy
    def _seq_to_numpy(seq):
        arr = np.concatenate(seq).reshape(len(seq), -1)

        assert np.all((arr >= 0) & (arr <= 9))
        return arr + 1

    results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),

        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=81,
        vocab_size=10 + 1,  # PAD + "0" ... "9"
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(results["group_indices"]) - 1,
        sets=["all"]
    )

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
    np.save(os.path.join(save_dir, "all__group_difficulties.npy"), np.array(group_difficulties, dtype=np.int32))


def build_easy_to_hard_split(config: DataProcessConfig):
    """Build val/test = (source-dataset val/test, empty>threshold) + synthesized
    uniform in-dist (empties in [train_empty_min, val_test_empty_threshold]),
    and a train of `easy_to_hard_train_size` puzzles uniform over [train_empty_min,
    train_empty_max] drawn from HF train.csv. Train, val in-dist and test in-dist
    share a `used` mask over the HF pool — strict no-leak.
    """
    if config.easy_to_hard_source is None:
        raise ValueError("easy_to_hard_split requires --easy-to-hard-source.")
    src_dataset = config.easy_to_hard_source

    rng = np.random.default_rng(config.seed)

    low, high = config.train_empty_min, config.train_empty_max
    if low > high:
        raise ValueError(f"train_empty_min ({low}) must be <= train_empty_max ({high})")
    val_test_threshold = config.val_test_empty_threshold
    total_train = config.easy_to_hard_train_size
    in_dist_per_empty = config.val_test_in_dist_per_empty

    os.makedirs(config.output_dir, exist_ok=True)

    # Stage natural OOD val/test in lists; concatenated with in-dist synth before save.
    print(f"Loading val/test from {src_dataset}...")
    fingerprints: set[bytes] = set()
    natural: dict[str, dict[str, list]] = {
        "val":  {"in": [], "lab": [], "ratings": []},
        "test": {"in": [], "lab": [], "ratings": []},
    }
    for split in ("val", "test"):
        split_dir = os.path.join(src_dataset, split)
        if not os.path.isdir(split_dir):
            print(f"  WARNING: {split_dir} not found, skipping {split}")
            continue
        inputs = np.load(os.path.join(split_dir, "all__inputs.npy"), mmap_mode="r")
        labels = np.load(os.path.join(split_dir, "all__labels.npy"), mmap_mode="r")
        diffs_path = os.path.join(split_dir, "all__group_difficulties.npy")
        if os.path.isfile(diffs_path):
            diffs = np.load(diffs_path)
        else:
            print(f"  WARNING: no {diffs_path}; ratings recorded as 0")
            diffs = np.zeros(len(inputs), dtype=np.int32)
        empty = (inputs == 1).sum(axis=1)
        keep = empty > val_test_threshold
        if config.val_test_min_rating is not None:
            keep &= diffs >= config.val_test_min_rating
        if config.val_test_max_rating is not None:
            keep &= diffs <= config.val_test_max_rating
        print(f"  {split} natural OOD: {keep.sum()}/{len(inputs)} after filter")

        # Pre-filter fingerprints prevent ANY leakage into train.
        for row in inputs[:]:
            fingerprints.add((row.reshape(9, 9) - 1).astype(np.uint8).tobytes())

        keep_idx = np.where(keep)[0]
        natural[split]["in"]      = [(inputs[i].reshape(9, 9) - 1).astype(np.uint8) for i in keep_idx]
        natural[split]["lab"]     = [(labels[i].reshape(9, 9) - 1).astype(np.uint8) for i in keep_idx]
        natural[split]["ratings"] = diffs[keep].tolist()
    print(f"Val+test fingerprint set: {len(fingerprints)} unique puzzles")

    print(f"Loading source puzzles from {config.source_repo} train.csv...")
    csv_path = hf_hub_download(config.source_repo, "train.csv", repo_type="dataset")
    src_inputs, src_labels, src_ratings = [], [], []
    n_excluded_fp = n_excluded_rating = 0
    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for _, q, a, rating in reader:
            assert len(q) == 81 and len(a) == 81
            r = int(rating)
            inp = np.frombuffer(q.replace(".", "0").encode(), dtype=np.uint8).reshape(9, 9) - ord("0")
            if inp.tobytes() in fingerprints:
                n_excluded_fp += 1
                continue
            if config.train_min_rating is not None and r < config.train_min_rating:
                n_excluded_rating += 1
                continue
            if config.train_max_rating is not None and r > config.train_max_rating:
                n_excluded_rating += 1
                continue
            lab = np.frombuffer(a.encode(), dtype=np.uint8).reshape(9, 9) - ord("0")
            src_inputs.append(inp.astype(np.uint8))
            src_labels.append(lab.astype(np.uint8))
            src_ratings.append(r)

    if not src_inputs:
        raise RuntimeError("No source puzzles left after fingerprint exclusion + rating filters.")

    src_empty = np.array([int(np.sum(p == 0)) for p in src_inputs], dtype=np.int32)
    print(f"Source pool: {len(src_inputs)} puzzles "
          f"(excluded {n_excluded_fp} by fingerprint, {n_excluded_rating} by rating); "
          f"empty cells [{src_empty.min()}, {src_empty.max()}]")
    if src_empty.max() < high:
        raise RuntimeError(f"Source max empty ({src_empty.max()}) < train_empty_max ({high}).")

    # Shared mask across val_in_dist, test_in_dist, and train: strict no-leak.
    used = np.zeros(len(src_inputs), dtype=bool)

    # Pre-allocate val/test in-dist before train so train can't re-use the same source.
    in_dist: dict[str, dict[str, list]] = {
        "val":  {"in": [], "lab": [], "ratings": []},
        "test": {"in": [], "lab": [], "ratings": []},
    }
    if in_dist_per_empty > 0:
        in_dist_targets = list(range(low, val_test_threshold + 1))
        N = in_dist_per_empty
        print(f"Synthesizing val/test in-dist: {N} puzzles per empty target across "
              f"{low}..{val_test_threshold} ({len(in_dist_targets)} buckets).")
        for X in sorted(in_dist_targets, reverse=True):
            for split in ("val", "test"):
                eligible = np.where((src_empty >= X) & ~used)[0]
                if len(eligible) < N:
                    raise RuntimeError(
                        f"Not enough source for {split} in-dist X={X}: have {len(eligible)}, need {N}."
                    )
                chosen = rng.choice(eligible, size=N, replace=False)
                used[chosen] = True
                for i in tqdm(chosen, desc=f"  {split}_in_dist target={X}"):
                    filled = _fill_cells_to_target(src_inputs[i], src_labels[i], X, rng)
                    in_dist[split]["in"].append(filled)
                    in_dist[split]["lab"].append(src_labels[i])
                    in_dist[split]["ratings"].append(src_ratings[i])

    targets = list(range(low, high + 1))
    n_per_target = total_train // len(targets)
    print(f"Sampling {n_per_target} puzzles per target across empty-cell targets {low}..{high}")
    train_inputs, train_labels, train_ratings = [], [], []
    for X in sorted(targets, reverse=True):
        eligible = np.where((src_empty >= X) & ~used)[0]
        if len(eligible) < n_per_target:
            raise RuntimeError(f"Not enough source for train X={X}: have {len(eligible)}, need {n_per_target}.")
        chosen = rng.choice(eligible, size=n_per_target, replace=False)
        used[chosen] = True
        for i in tqdm(chosen, desc=f"  target={X}"):
            filled = _fill_cells_to_target(src_inputs[i], src_labels[i], X, rng)
            train_inputs.append(filled)
            train_labels.append(src_labels[i])
            train_ratings.append(src_ratings[i])

    perm = rng.permutation(len(train_inputs))
    train_inputs = [train_inputs[i] for i in perm]
    train_labels = [train_labels[i] for i in perm]
    train_ratings = [train_ratings[i] for i in perm]
    _build_and_save(train_inputs, train_labels, train_ratings, num_augments=0,
                    save_dir=os.path.join(config.output_dir, "train"))

    for split in ("val", "test"):
        all_in   = natural[split]["in"]      + in_dist[split]["in"]
        all_lab  = natural[split]["lab"]     + in_dist[split]["lab"]
        all_rats = natural[split]["ratings"] + in_dist[split]["ratings"]
        if not all_in:
            continue
        perm = rng.permutation(len(all_in))
        all_in   = [all_in[i]   for i in perm]
        all_lab  = [all_lab[i]  for i in perm]
        all_rats = [all_rats[i] for i in perm]
        _build_and_save(all_in, all_lab, all_rats, num_augments=0,
                        save_dir=os.path.join(config.output_dir, split))

    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    if config.easy_to_hard_split:
        build_easy_to_hard_split(config)
        return
    convert_subset("train", config)
    convert_subset("test", config)


if __name__ == "__main__":
    cli()
