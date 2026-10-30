"""Sanity test of make_anderson_step on a synthetic linear contraction (CPU, a few seconds).

    python test_anderson.py

f(y) = A y + b with a known fixed point. Prints the iterations each solver needs to reach residue < 1e-6
and the error to the true fixed point. Anderson should be faster than Picard on most seeds. Broyden with a
large memory (m=60, about unlimited for this 32-dim problem) must converge fast; Broyden m=5 drops old
corrections and, as the paper reports, may drift away (None = did not converge in 500 iterations).
"""
import types

import numpy as np
import torch

from inference import make_anderson_step, make_broyden_step


class FakeOpt:
    """Minimal stand-in for FixedPointOptimizer.step: Picard update + residue + iter_idx."""
    stepsize_decay_eval = 1.0

    def step(self, state, y):
        x = state["y"]
        state["residues"] = ((x - y).abs().amax(-1) / (y.abs().amax(-1) + 1e-5)).amax(-1)
        state["y"] = y.clone()
        state["iter_idx"] = state["iter_idx"] + 1
        return state


def run(A, b, y0, solver, m, iters=500, tol=1e-6):
    opt = FakeOpt()
    if solver == "anderson":
        make_anderson_step(opt, m=m)
    elif solver == "broyden":
        make_broyden_step(opt, m=m)
    B, S, H = y0.shape
    st = dict(y=y0.clone(), residues=torch.full((B,), float("inf")), iter_idx=torch.zeros(B, dtype=torch.int32))
    for k in range(iters):
        fy = (st["y"].reshape(B, -1) @ A.T + b).reshape(B, S, H)
        st = opt.step(st, fy)
        if bool((st["residues"] < tol).all()):
            return k + 1, st["y"]
    return None, st["y"]


def main():
    torch.manual_seed(0)
    B, S, H = 2, 4, 8
    N = S * H
    for seed in range(3):
        g = torch.Generator().manual_seed(seed)
        A = torch.randn(N, N, generator=g)
        A *= 0.9 / torch.linalg.eigvals(A).abs().max()
        b = torch.randn(N, generator=g)
        ystar = torch.linalg.solve(torch.eye(N) - A, b)
        y0 = torch.randn(B, S, H, generator=g)
        res = []
        for solver, m in (("picard", 0), ("anderson", 3), ("broyden", 5), ("broyden", 60)):
            k, y = run(A, b, y0, solver, m)
            err = (y.reshape(B, -1) - ystar).abs().max().item()
            res.append(f"{solver} m={m}: {k} it, err {err:.1e}")
        print(f"seed {seed}: " + " | ".join(res))


if __name__ == "__main__":
    main()