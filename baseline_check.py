"""Step 0/1 checks from a trace run (no GPU needed).

    python baseline_check.py results/sudoku_1000/sudoku

Prints: NaN/inf check on the trajectories, and the SFPRM Table-5 style numbers:
halting applied afterwards with tau = the tolerance used for the trace.
  % halted         sequences with first_step_below_tol defined
  median steps     median of first_step_below_tol over halted sequences
  accuracy         halted -> correct at natural stop; not halted -> correct at max_iter
"""
import sys

import numpy as np

from plot_trajectories import read_csv


def main(prefix):
    t, s = read_csv(f"{prefix}_trace.csv"), read_csv(f"{prefix}_summary.csv")
    bad = ~np.isfinite(t[["residue", "eta"]].to_numpy(dtype=float))
    print(f"{len(s)} sequences, {len(t)} trace rows")
    print(f"NaN/inf in trace (residue, eta): {int(bad.sum())}"
          + ("   <-- PROBLEM" if bad.any() else "   (ok)"))
    halted = s.first_step_below_tol.notna()
    ok = np.where(halted, s.correct_at_natural_stop == 1, s.correct_at_max_iter == 1)
    print(f"halted:            {halted.mean():.1%}")
    print(f"median steps:      {s.first_step_below_tol[halted].median():g}")
    print(f"accuracy (halting afterwards): {ok.mean():.1%}")
    print(f"accuracy at max_iter (no halting): {(s.correct_at_max_iter == 1).mean():.1%}")
    print(f"accuracy among halted:     {ok[halted].mean():.1%}")
    print(f"accuracy among not halted: {ok[~halted].mean():.1%}" if (~halted).any() else "")
    print("target (SFPRM Table 5, FPRM Sudoku): 91.5% halted, median 50 steps, 88.0% accuracy")


if __name__ == "__main__":
    main(sys.argv[1])