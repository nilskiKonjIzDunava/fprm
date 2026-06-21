import math
from typing import Literal, Optional

from models.config import ReasoningModelConfig


class FPRMConfig(ReasoningModelConfig):
    # Halting fields are required on the base class but unused by most FP archs;
    # default them so existing FP yamls don't need to set them.
    halt_max_steps: int = 1
    halt_exploration_prob: float = 0.0

    halting_mechanism: Literal["act", "fixed_point", "fixed_iterations"] = "fixed_point"

    # Fixed-point iteration controls
    max_iter: int
    # Eval-time max_iter cap. None = fall back to max_iter
    max_iter_eval: Optional[int] = None

    # Variational dropout on the L_level output (multiplicative mask). Same
    # mask is reused for all FP iterations on a given sample (resampled when
    # the sample is halted). 0.0 = off.
    variational_dropout: float = 0.0

    # Jacobian-norm regularizer (Hutchinson estimator).
    # jacobian_reg selects the JVP method:
    #   "none"  - regularizer disabled (no extra L_level forwards, no outputs["jacobian_loss"])
    #   "fda"   - central-difference finite approximation (2 extra forwards per sample)
    #   "exact" - autograd VJP with create_graph=True (1 forward + 1 backward per sample;
    #             requires MATH SDPA backend for double-backward)
    # n_jacobian_samples is the number of Hutchinson samples; jacobian_eps is the
    # central-difference step (only used by "fda").
    jacobian_reg: Literal["none", "fda", "exact"] = "none"
    n_jacobian_samples: int = 0
    jacobian_eps: float = 1.0e-3

    # Fixed-point solver
    fp_thresh: float = 0.1
    outlier_quantile: float = 0.25
    stepsize: float = 1.0
    stepsize_decay: float = 0.9
    decay_patience: int = 5
    eps: float = 1e-8
    # Std of the truncated-normal init for the FP iterate y at carry reset.
    init_std: float = 1.0
    # Std of additive Gaussian noise injected at every FP iteration update.
    # 0.0 = off.
    additive_noise_std: float = 0.0
    # If True, use a persistent (hidden_size,) buffer (random at construction,
    # frozen thereafter) broadcast over (batch, seq, hidden) at reset — matching
    # TRM's H_init/L_init scheme. If False (default), draw fresh i.i.d. random
    # of shape (batch, seq, hidden) at every reset.
    fixed_init: bool = False
    gamma_alpha: float = 4.0
    gamma_scale: float = 50 / 3
    max_iter_dist: str = "gamma"
    expon_scale: float = 50.0 / math.log(2.0)
