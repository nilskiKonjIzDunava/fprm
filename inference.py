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

    # per-sequence fixed-point trajectories -> CSV (FPRM; use with +n_subset):
    # residue and eta every 10 iterations, first step below tolerance, correctness at the
    # natural stop and at max_iter  ->  <prefix>_trace.csv, <prefix>_summary.csv
    ... +n_subset=1000 +trace_out=results/sudoku +trace_every=10 [+trace_tol=0.1] [+trace_excel_it=false]

    # results independent of batch composition / batch size (per-sequence seeded y0, every
    # sequence runs the full max_iter_eval, deterministic kernels). For bit-level equality across
    # batch sizes add arch.forward_dtype=float32. Check with compare_runs.py.
    ... +deterministic=true [arch.forward_dtype=float32]

    # solvers: default = Algorithm 1; Picard = arch.stepsize_decay_eval=1.0;
    # Anderson (memory m, mixing beta, relative regularization lam):
    ... +deterministic=true +solver=anderson [+aa_m=3 +aa_beta=1.0 +aa_lam=1e-8]
    # Broyden (memory m):
    ... +deterministic=true +solver=broyden [+bro_m=5]

The test split of `data_paths` (or `data_paths_test`) must have been built with
dataset/build_*.py, exactly as for training. Private repos: set HF_TOKEN.
"""
import os
import re
import csv
import json

import numpy as np
import torch
import hydra
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from huggingface_hub import hf_hub_download

from models.losses import IGNORE_LABEL_ID
from pretrain import TrainState, autocast_ctx, create_evaluators, evaluate
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


# ---------------------------------------------------------------------------------------
# Determinism: results independent of batch composition (+deterministic=true)
#
# What couples the sequences of a batch in FPRM eval (everything else is per-row):
#   1. the initial state y0 is drawn from the *global* RNG at reset -> depends on batch
#      shape/order. Fix: y0 of sequence k comes from a generator seeded with (seed, k).
#   2. eval halting is batch-wide (residues.max() < fp_thresh | stepsize.max() < 1e-3):
#      a sequence stops when the *whole* batch (padding rows included) has converged.
#      Fix: nobody stops before max_iter_eval, so every sequence gets the same budget.
#   3. kernel numerics (TF32, bf16 reductions, cuBLAS algorithm choice) depend on shapes.
#      Fix: disable TF32 + deterministic algorithms; for bit-level equality across batch
#      sizes also run in fp32 (arch.forward_dtype=float32).
# ---------------------------------------------------------------------------------------
class SeqIdLoader:
    """Wraps the eval loader and tells the optimizer the global index of each row."""

    def __init__(self, loader, opt):
        self.loader, self.opt = loader, opt

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        offset = 0
        for set_name, batch, gbs in self.loader:
            valid = (batch["labels"] != IGNORE_LABEL_ID).any(-1)
            ids = torch.where(valid, offset + torch.cumsum(valid.long(), 0) - 1,
                              -1 - torch.arange(valid.numel()))  # padding rows: negative, unused
            self.opt.seq_ids = ids.tolist()
            offset += int(valid.sum())
            yield set_name, batch, gbs


def make_deterministic(model, seed: int):
    """Per-sequence seeded init of y0 + TF32 off + deterministic kernels. Call before compile."""
    import math
    import models.fixed_point_reasoning.model_utils as mu

    opt = model.model.inner.L_optimizer  # type: ignore
    assert float(opt.additive_noise_std) == 0.0, "additive_noise_std > 0 uses the global RNG at every step"
    opt.seq_ids = None
    orig = mu.trunc_normal_init_

    @torch._dynamo.disable
    def seeded(tensor, std=1.0, lower=-2.0, upper=2.0):
        ids = getattr(opt, "seq_ids", None)
        if tensor.ndim != 3 or ids is None or std == 0 or len(ids) != tensor.shape[0]:
            return orig(tensor, std=std, lower=lower, upper=upper)
        # same truncated-normal as models.common.trunc_normal_init_, but row k <- seed (seed, ids[k])
        sqrt2 = math.sqrt(2)
        a, b = math.erf(lower / sqrt2), math.erf(upper / sqrt2)
        z = (b - a) / 2
        c = (2 * math.pi) ** -0.5
        pdf_u, pdf_l = c * math.exp(-0.5 * lower ** 2), c * math.exp(-0.5 * upper ** 2)
        comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)
        key = (tuple(ids), tuple(tensor.shape), tensor.dtype)
        if getattr(opt, "_y0_key", None) == key:  # reset() runs at every model call: draw y0 once per batch
            tensor.copy_(opt._y0)
            return tensor
        gen = torch.Generator(device=tensor.device)
        with torch.no_grad():
            for k, sid in enumerate(ids):
                gen.manual_seed(seed * 1_000_003 + sid + 10 ** 9)
                row = torch.empty(tensor.shape[1:], dtype=torch.float32, device=tensor.device)
                row.uniform_(a, b, generator=gen).erfinv_().mul_(sqrt2 * comp_std)
                row.clip_(lower * comp_std, upper * comp_std)
                tensor[k] = row.to(tensor.dtype)
        opt._y0_key, opt._y0 = key, tensor.clone()
        return tensor

    mu.trunc_normal_init_ = seeded
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return opt


class FixedBudget(torch.nn.Module):
    """No early stop: all_finish only when every row has used max_iter steps (fixes #2)."""

    def __init__(self, model):
        super().__init__()
        self.inner = model

    def forward(self, *args, **kwargs):
        carry, *rest, _ = self.inner(*args, **kwargs)
        if int(carry.steps.max()) < int(self.inner.model.max_iter):  # type: ignore
            carry.halted.zero_()
            return (carry, *rest, False)
        return (carry, *rest, True)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.__dict__["_modules"]["inner"], name)



# ---------------------------------------------------------------------------------------
# Alternative fixed-point solvers (+solver=anderson): replace FixedPointOptimizer.step only.
# The model call, residue definition, halting and trace stay the same.
# ---------------------------------------------------------------------------------------
def make_anderson_step(opt, m: int = 3, beta: float = 1.0, lam: float = 1e-8):
    """Anderson acceleration (type II, as in DEQ / Walker-Ni) with memory m, per sequence.

    With x_i the iterates, f_i = f(x_i), g_i = f_i - x_i (flattened over positions x channels):
        alpha = argmin || sum_i alpha_i g_i ||   s.t. sum_i alpha_i = 1
        x_next = beta * sum_i alpha_i f_i + (1 - beta) * sum_i alpha_i x_i
    The bordered system [[0, 1^T], [1, G G^T + lam * mean(diag) I]] alpha~ = [1, 0] is solved in
    float64. beta=1, lam=1e-8 are my defaults: confirm them against the paper.
    The eta decay is switched off (eta = 1 always), as for Picard. Eval only: the batch moves in
    lockstep (all rows share the same iteration index), which FixedBudget / the trace guarantee.
    """
    orig_step = opt.step
    opt.stepsize_decay_eval = 1.0  # no eta decay: eta stays 1, Anderson does the acceleration
    buf = {}

    @torch._dynamo.disable
    def step(state, y):
        k = int(state["iter_idx"].max())          # iteration index (0 = first step of this batch)
        x = state["y"].detach()                   # current iterate x_k, before the update
        B = x.shape[0]
        f = (y if y.dtype == x.dtype else y.to(x.dtype)).detach()
        state = orig_step(state, y)               # residues / best / patience / iter_idx; y <- f(x_k)
        with torch.no_grad():
            X, F = x.reshape(B, -1).float(), f.reshape(B, -1).float()
            if k == 0 or buf.get("X") is None or buf["X"].shape[0] != B or buf["X"].shape[2] != X.shape[1]:
                buf["X"] = torch.zeros(B, m, X.shape[1], device=X.device)
                buf["F"] = torch.zeros_like(buf["X"])
            slot = k % m
            buf["X"][:, slot], buf["F"][:, slot] = X, F
            n = min(k + 1, m)
            Xh, Fh = buf["X"][:, :n], buf["F"][:, :n]
            G = Fh - Xh                                            # [B, n, N]
            gram = torch.bmm(G, G.transpose(1, 2)).double()        # [B, n, n]
            eye = torch.eye(n, dtype=gram.dtype, device=gram.device)
            reg = lam * gram.diagonal(dim1=1, dim2=2).mean(-1).clamp_min(1e-30)
            H = torch.zeros(B, n + 1, n + 1, dtype=gram.dtype, device=gram.device)
            H[:, 0, 1:] = 1
            H[:, 1:, 0] = 1
            H[:, 1:, 1:] = gram + reg.view(-1, 1, 1) * eye
            rhs = torch.zeros(B, n + 1, 1, dtype=gram.dtype, device=gram.device)
            rhs[:, 0] = 1
            alpha = torch.linalg.solve(H, rhs)[:, 1:, 0]           # [B, n]
            newest = torch.zeros_like(alpha)
            newest[:, slot] = 1                                    # n-1 == slot (before wrap-around)
            alpha = torch.where(torch.isfinite(alpha).all(-1, keepdim=True), alpha, newest)  # fallback: Picard
            alpha = alpha.float()
            mix = beta * torch.einsum("bn,bnd->bd", alpha, Fh) + (1 - beta) * torch.einsum("bn,bnd->bd", alpha, Xh)
            state["y"] = mix.reshape(x.shape).to(x.dtype)
        return state

    opt.step = step
    return opt



def make_broyden_step(opt, m: int = 5, eps: float = 1e-10):
    """Limited-memory 'good' Broyden on g(z) = f(z) - z (as in the SFPRM paper, App. C.5), per sequence.

        z_{i+1} = z_i - B_i g_i,   B_0 = -I,   B_i = B_{i-1} + (s_i - B_{i-1} y_i) s_i^T B_{i-1} / (s_i^T B_{i-1} y_i)
        s_i = z_i - z_{i-1},  y_i = g_i - g_{i-1}
    B is stored as the last m rank-one corrections B = -I + sum_j u_j v_j^T (oldest dropped when full).
    Safeguards, as in the paper: a non-finite extrapolation falls back to the fixed step f(z_i), and an
    update whose denominator is (relatively) ~0 is skipped. eta = 1 (no damping), eta decay off.
    All tokens of a sequence are flattened into one vector; one history per sequence; eval only (lockstep).
    """
    orig_step = opt.step
    opt.stepsize_decay_eval = 1.0
    buf = {}

    @torch._dynamo.disable
    def step(state, y):
        k = int(state["iter_idx"].max())
        x = state["y"].detach()
        B = x.shape[0]
        f = (y if y.dtype == x.dtype else y.to(x.dtype)).detach()
        state = orig_step(state, y)
        with torch.no_grad():
            X, Fz = x.reshape(B, -1).float(), f.reshape(B, -1).float()
            N, dev = X.shape[1], X.device
            g = Fz - X
            if k == 0 or buf.get("U") is None or buf["U"].shape[0] != B or buf["U"].shape[2] != N:
                buf["U"] = torch.zeros(B, m, N, device=dev)
                buf["V"] = torch.zeros_like(buf["U"])
                buf["cnt"] = torch.zeros(B, dtype=torch.long, device=dev)
            else:
                U, V, cnt = buf["U"], buf["V"], buf["cnt"]
                valid = (torch.arange(m, device=dev)[None] < cnt.clamp(max=m)[:, None]).float()  # [B, m]
                s_, y_ = X - buf["x"], g - buf["g"]
                By = -y_ + torch.einsum("bm,bmd->bd", valid * torch.einsum("bmd,bd->bm", V, y_), U)
                den = (s_ * By).sum(-1)
                ok = torch.isfinite(den) & (den.abs() > eps * s_.norm(dim=-1) * By.norm(dim=-1))
                u = (s_ - By) / torch.where(ok, den, torch.ones_like(den)).unsqueeze(-1)
                v = -s_ + torch.einsum("bm,bmd->bd", valid * torch.einsum("bmd,bd->bm", U, s_), V)
                idx = ok.nonzero(as_tuple=True)[0]
                slot = cnt[idx] % m
                U[idx, slot], V[idx, slot] = u[idx], v[idx]
                cnt[idx] += 1
            U, V, cnt = buf["U"], buf["V"], buf["cnt"]
            valid = (torch.arange(m, device=dev)[None] < cnt.clamp(max=m)[:, None]).float()
            corr = torch.einsum("bm,bmd->bd", valid * torch.einsum("bmd,bd->bm", V, g), U)
            new = X + g - corr                                    # z - B g
            new = torch.where(torch.isfinite(new).all(-1, keepdim=True), new, Fz)  # fallback: fixed step
            buf["x"], buf["g"] = X, g
            state["y"] = new.reshape(x.shape).to(x.dtype)
        return state

    opt.step = step
    return opt



@torch.inference_mode()
def trace_trajectories(config: PretrainConfig, model, eval_loader, out_prefix: str,
                       every: int, tol: float, excel_it: bool = True) -> None:
    """Per-sequence fixed-point diagnostics, written as two CSV files (FPRM only).

    Replaces pretrain.evaluate(): it runs the same model calls, but keeps going until
    `max_iter_eval` even after the batch has converged, so every sequence is followed
    for the whole budget. One model call == one fixed-point iteration (eval: n_steps=1).

    <out_prefix>_trace.csv    seq_id, step, residue, eta   (every `every` steps, + step 1 and the last)
    <out_prefix>_summary.csv  seq_id, first_step_below_tol, correct_at_natural_stop,
                              correct_at_max_iter, final_residue

    residue_i / eta_i: residue of iterate y_i and the step size used in
    y_{i+1} = eta_i * f(y_i) + (1 - eta_i) * y_i. "Natural stop" of a sequence = first step
    whose residue is below `tol`; correctness there uses the predictions returned by that
    same model call (as the model itself would when halting).
    """
    sep, dec = (";", ",") if excel_it else (",", ".")  # Italian Excel: ';' separator, ',' decimal

    def cell(x):
        if x is None:
            return ""
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, float):
            return f"{x:.6g}".replace(".", dec)
        return x

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    f_trace = open(f"{out_prefix}_trace.csv", "w", newline="")
    f_sum = open(f"{out_prefix}_summary.csv", "w", newline="")
    w_trace, w_sum = csv.writer(f_trace, delimiter=sep), csv.writer(f_sum, delimiter=sep)
    w_trace.writerow(["seq_id", "step", "residue", "eta"])
    w_sum.writerow(["seq_id", "first_step_below_tol", "correct_at_natural_stop",
                    "correct_at_max_iter", "final_residue"])

    model.set_num_iters()  # eval budget = arch.max_iter_eval, as evaluate() does
    max_iter = int(model.model.max_iter)  # type: ignore
    eta0 = float((config.arch.__pydantic_extra__ or {}).get("stepsize", 1.0))
    seq_offset = 0

    for set_name, batch, _ in eval_loader:
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.device("cuda"):
            carry = model.initial_carry(batch)
        labels = batch["labels"]
        mask = labels != IGNORE_LABEL_ID
        valid = mask.any(-1).cpu().numpy()  # padding rows of the last batch are dropped
        B = labels.shape[0]
        first = torch.zeros(B, dtype=torch.long, device="cuda")   # 0 = never below tol
        ok_stop = torch.zeros(B, dtype=torch.bool, device="cuda")
        eta = torch.full((B,), eta0, device="cuda")               # eta_i used by the update at step i
        step = 0

        while True:
            with autocast_ctx(config):
                carry, _, _, _, preds, all_finish = model(carry=carry, batch=batch, return_keys={"preds"})
            step += 1
            assert step <= max_iter, f"step {step} exceeded max_iter {max_iter}"

            state = carry.inner_carry.z_L_state
            res = state["residues"].float()
            correct = ((preds["preds"] == labels) | ~mask).all(-1)

            newly = (first == 0) & (res < tol)
            first = torch.where(newly, torch.full_like(first, step), first)
            ok_stop = torch.where(newly, correct, ok_stop)

            if step == 1 or step % every == 0 or step == max_iter:
                r, e = res.cpu().numpy(), eta.cpu().numpy()
                for b in np.flatnonzero(valid):
                    w_trace.writerow([seq_offset + int(b), step, cell(float(r[b])), cell(float(e[b]))])
            eta = state["stepsize"].float().view(-1)  # decayed value = eta for the next update

            if all_finish:
                if step >= max_iter:
                    break
                carry.halted.zero_()  # batch converged early: keep iterating, don't let it reset

        first_c, ok_c, fin_c = first.cpu().numpy(), ok_stop.cpu().numpy(), correct.cpu().numpy()
        res_c = res.cpu().numpy()
        for b in np.flatnonzero(valid):
            conv = first_c[b] > 0
            w_sum.writerow([seq_offset + int(b),
                            int(first_c[b]) if conv else "",
                            cell(bool(ok_c[b])) if conv else "",
                            cell(bool(fin_c[b])),
                            cell(float(res_c[b]))])
        seq_offset += int(valid.sum())
        print(f"traced {seq_offset} sequences")

    f_trace.close()
    f_sum.close()
    print(f"Wrote {out_prefix}_trace.csv and {out_prefix}_summary.csv")


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    raw = OmegaConf.to_container(hydra_config, resolve=True)
    n_subset = raw.pop("n_subset", None)  # not a PretrainConfig field: +n_subset=1000
    trace_out = raw.pop("trace_out", None)  # +trace_out=results/s5  -> CSV trajectory export
    trace_every = int(raw.pop("trace_every", 10))
    trace_tol = raw.pop("trace_tol", None)  # measurement tolerance; default arch.fp_thresh
    trace_excel_it = bool(raw.pop("trace_excel_it", True))  # ';' separator + ',' decimal
    deterministic = bool(raw.pop("deterministic", False))  # +deterministic=true: batch-independent results
    solver = str(raw.pop("solver", "default"))  # default (Algorithm 1) | anderson | broyden
    aa_m = int(raw.pop("aa_m", 3))
    aa_beta = float(raw.pop("aa_beta", 1.0))
    aa_lam = float(raw.pop("aa_lam", 1e-8))
    bro_m = int(raw.pop("bro_m", 5))
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
    if solver == "anderson":
        make_anderson_step(model.model.inner.L_optimizer, aa_m, aa_beta, aa_lam)  # type: ignore
        print(f"Solver: Anderson m={aa_m} beta={aa_beta} lam={aa_lam} (eta decay off)")
    elif solver == "broyden":
        make_broyden_step(model.model.inner.L_optimizer, bro_m)  # type: ignore
        print(f"Solver: Broyden m={bro_m} (eta decay off)")
    elif solver != "default":
        raise SystemExit(f"unknown +solver={solver}")
    if deterministic:
        opt = make_deterministic(model, config.seed)
        eval_loader = SeqIdLoader(eval_loader, opt)
    if "DISABLE_COMPILE" not in os.environ:
        model = torch.compile(model)  # type: ignore
    if deterministic:
        model = FixedBudget(model)  # type: ignore
    model.eval()

    if trace_out:
        tol = float(trace_tol) if trace_tol is not None else float(config.arch.__pydantic_extra__.get("fp_thresh", 0.1))  # type: ignore
        trace_trajectories(config, model, eval_loader, trace_out, trace_every, tol, trace_excel_it)
        return

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