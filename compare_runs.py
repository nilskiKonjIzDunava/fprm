"""Compare two trace runs of the same subset (e.g. batch 32 vs 64, both with +deterministic=true).

    python compare_runs.py results/det_b32 results/det_b64
"""
import sys

import numpy as np
import pandas as pd

from plot_trajectories import read_csv


def main(a, b):
    sa, sb = (read_csv(f"{p}_summary.csv").set_index("seq_id").sort_index() for p in (a, b))
    ta, tb = (read_csv(f"{p}_trace.csv") for p in (a, b))
    assert sa.index.equals(sb.index), "different sequence sets"
    print(f"{len(sa)} sequences")
    for col in ["first_step_below_tol", "correct_at_natural_stop", "correct_at_max_iter"]:
        same = ((sa[col] == sb[col]) | (sa[col].isna() & sb[col].isna())).mean()
        print(f"identical {col}: {same:.1%}")
    m = ta.merge(tb, on=["seq_id", "step"], suffixes=("_a", "_b"))
    d = (m.residue_a - m.residue_b).abs()
    print(f"residue |diff|: max {d.max():.3g}, mean {d.mean():.3g}  (0 = bit-identical)")
    print(f"eta     |diff|: max {(m.eta_a - m.eta_b).abs().max():.3g}")
    acc = [(s.correct_at_max_iter == 1).mean() for s in (sa, sb)]
    print(f"sequence accuracy at max_iter: {acc[0]:.3f} vs {acc[1]:.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])