import pydantic

from typing import Optional, List

class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str

class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig

class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_paths: List[str]
    data_paths_test: List[str] = []
    # Evaluators
    evaluators: List[EvaluatorConfig] = []

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Optimizer for dense parameters
    optimizer: str = "adamw"

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None  # weights-only load (transfer / eval)
    resume_from: Optional[str] = None  # full resume: bundle path or its directory
    checkpoint_path: Optional[str] = None
    metrics_out: Optional[str] = None  # eval-only: dump per-k metrics as plot JSON

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    checkpoint_every_n_steps: Optional[int] = None  # if set, also save a checkpoint every N training steps
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0 # number of eval iters (each = eval_interval epochs) to skip before starting eval
    eval_save_outputs: List[str] = []

    # Length-generalisation eval (state tracking, causal models): when all three
    # are set, evaluation does ONE forward per batch and per-k accuracy is read
    # off via cumulative cumsum over the time axis at prefix lengths
    # range(k_eval_min, k_eval_max + 1, k_eval_step). Only correct under causal
    # attention. Default None disables it; sudoku/maze/arc keep the standard
    # per-set eval path.
    k_eval_min: Optional[int] = None
    k_eval_max: Optional[int] = None
    k_eval_step: int = 1

    ema: bool = False # use Exponential-Moving-Average
    ema_rate: float = 0.999 # EMA-rate
    freeze_weights: bool = False # If True, freeze weights and only learn the embeddings

    profile_flops: bool = False  # one-shot per-segment FLOPs + accumulator (utils.flop_profiler)
