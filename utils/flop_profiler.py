"""FLOP tracking: profile per-segment cost once with FlopCounterMode,
then multiply by the number of ACT segments executed per batch.

Works for TRM and FPTRM-RecDoubleLoopACT, where each `model(carry, batch)` call
is exactly one ACT segment with deterministic per-segment compute. Training
calls run one segment; eval batches run `inference_steps` segments via the
external `while True` loop. Disabled by default; flip `profile_flops=True` in
the pretrain config to enable. When disabled, all methods are no-ops and the
training/eval hot paths are unchanged.

If `torch.compile` swallows the dispatched ops, set `DISABLE_COMPILE=1` for the
profiling run so FlopCounterMode can see them.
"""
from contextlib import contextmanager
from typing import Optional, Dict


class FlopProfiler:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._train_per_seg: Optional[int] = None
        self._eval_per_seg: Optional[int] = None
        self.train_segments = 0
        self.eval_segments = 0

    @contextmanager
    def measure(self, kind: str):
        """One-shot: count FLOPs of the wrapped block and store as the
        per-segment cost for `kind` in {'train','eval'}. Subsequent calls are
        no-ops once the cost has been recorded."""
        if not self.enabled or self._get(kind) is not None:
            yield
            return
        from torch.utils.flop_counter import FlopCounterMode
        fcm = FlopCounterMode(display=False)
        with fcm:
            yield
        self._set(kind, int(fcm.get_total_flops()))

    def add_train(self, segments: int = 1) -> None:
        if self.enabled:
            self.train_segments += segments

    def add_eval(self, segments: int) -> None:
        if self.enabled:
            self.eval_segments += segments

    def metrics(self) -> Dict[str, float]:
        if not self.enabled:
            return {}
        return {
            "flops/train_per_segment": float(self._train_per_seg or 0),
            "flops/train_segments": float(self.train_segments),
            "flops/train_total": float((self._train_per_seg or 0) * self.train_segments),
            "flops/eval_per_segment": float(self._eval_per_seg or 0),
            "flops/eval_segments": float(self.eval_segments),
            "flops/eval_total": float((self._eval_per_seg or 0) * self.eval_segments),
        }

    def _get(self, kind: str) -> Optional[int]:
        return self._train_per_seg if kind == "train" else self._eval_per_seg

    def _set(self, kind: str, value: int) -> None:
        if kind == "train":
            self._train_per_seg = value
        else:
            self._eval_per_seg = value
