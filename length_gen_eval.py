"""Single-pass length-generalisation evaluation.

Used by state tracking (and any other causal model with dense per-position
labels). One forward per batch; per-k accuracy and sequence_accuracy are read off
by cumsum'ing per-position correctness along the time axis.

Triggered when PretrainConfig sets k_eval_min/max (and optionally k_eval_step).
All helpers no-op when those fields are unset.
"""
from typing import Optional, List, Dict

import torch
import wandb

from models.losses import IGNORE_LABEL_ID
from pretrain_config import PretrainConfig


def length_gen_ks(config: PretrainConfig) -> Optional[List[int]]:
    """Return the eval-grid prefix lengths, or None if length-gen is disabled."""
    if config.k_eval_min is None or config.k_eval_max is None:
        return None
    return list(range(config.k_eval_min, config.k_eval_max + 1, config.k_eval_step))


def k_pos_tensor(ks: List[int], seq_len: int, device) -> torch.Tensor:
    """Column indices into a [B, seq_len] tensor for each k (= k - 1, clamped)."""
    return torch.tensor(
        [min(k - 1, seq_len - 1) for k in ks], device=device, dtype=torch.long,
    )


def accumulate(state, preds: torch.Tensor, labels: torch.Tensor, k_pos: torch.Tensor):
    """Add one batch's per-k contributions to the running sums.

    state : dict[str, Tensor[|ks|]] (running sums) or None on first call.
    preds, labels : [B, L] int tensors.
    k_pos  : LongTensor[|ks|] from `k_pos_tensor(...)`.
    """
    valid   = labels != IGNORE_LABEL_ID
    correct = valid & (preds == labels)
    cum_cor = correct.to(torch.float32).cumsum(dim=1).index_select(1, k_pos)  # [B, |ks|]
    cum_val = valid.to(torch.float32).cumsum(dim=1).index_select(1, k_pos)    # [B, |ks|]

    valid_k = cum_val > 0
    acc_b   = torch.where(valid_k, cum_cor / cum_val.clamp_min(1.0), torch.zeros_like(cum_cor)).sum(0)
    exact_b = (valid_k & (cum_cor == cum_val)).to(torch.float32).sum(0)
    count_b = valid_k.to(torch.float32).sum(0)

    if state is None:
        return {"accuracy": acc_b, "sequence_accuracy": exact_b, "count": count_b}
    state["accuracy"]       += acc_b
    state["sequence_accuracy"] += exact_b
    state["count"]          += count_b
    return state


def finalize(state: Dict[str, torch.Tensor], ks: List[int]) -> dict:
    """Reduce running sums to per-k means and pack as a structured curve."""
    cnt = state["count"].clamp_min(1.0)
    return {
        "ks":             list(ks),
        "accuracy":       (state["accuracy"]       / cnt).cpu().tolist(),
        "sequence_accuracy": (state["sequence_accuracy"] / cnt).cpu().tolist(),
    }


def to_wandb(curve: dict) -> dict:
    """Build wandb Table + line plots from a {ks, accuracy, sequence_accuracy} curve."""
    cols = [c for c in curve if c != "ks"]
    table = wandb.Table(
        columns=["k"] + cols,
        data=[[curve["ks"][i]] + [float(curve[c][i]) for c in cols]
              for i in range(len(curve["ks"]))],
    )
    return {f"eval/{c}_vs_k": wandb.plot.line(table, "k", c, title=f"{c} vs k") for c in cols}
