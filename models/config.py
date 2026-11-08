from pydantic import BaseModel
from typing import Optional

class ReasoningModelConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int
    n_backwards_L: int = 1   # number of with-grad L-level steps in the final pass; was implicitly L_cycles before

    H_layers: int # ignored
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Alexia: added
    mlp_t: bool = False # use mlp on L instead of transformer
    puzzle_emb_len: int = 16 # if non-zero, its specified to this value
    no_ACT_continue: bool =  True # No continue ACT loss, only use the sigmoid of the halt which makes much more sense

    # Q-head input source: True -> read from z_H[:, 0] (first puzzle_emb register, original behavior).
    # False -> mean-pool z_H over the non-puzzle_emb token positions.
    q_logit_from_puzzle_emb: bool = True
    # When True, q-head loss only updates the q-head itself, not the reasoning trunk.
    q_logit_detach_model: bool = False

    # When True, self-attention uses a causal mask (left-to-right). Default False
    # for non-autoregressive tasks (ARC, sudoku, maze); set True for tasks like
    # state tracking where prefix-k accuracy from a single forward is desired.
    causal: bool = False

    # Fields introduced by FPTRM.
    # Defaults reproduce the original TRM block: post-norm, no conv, no residual scaling.
    softmax_temp: float = 1.0
    norm_type: str = "post-norm"            # 'pre-norm' | 'peri-norm' | 'post-norm'
    norm_placement: str = "none"            # 'input' | 'output' | 'none'
    conv_type: str = "none"                 # 'conv1d' | 'conv2d' | 'none'
    conv_kernel_size: int = 4
    conv_bias: bool = False
    residual_scale: Optional[str] = None    # None | 'fixed' | 'input-independent' | 'input-dependent'
    alpha_1_init: float = 0.5
    alpha_2_init: float = 0.5
    normalize_input_injection: bool = False
    use_spec_norm_linear: bool = False

    # When True, only alpha_2 (the outer input-injection mix) is learnable;
    # the per-block alpha_1/beta_1 are pinned to 1 so blocks behave as plain residuals.
    outer_only: bool = False

    # DropConnect-style weight dropout applied to the trunk CastedLinear layers
    # (Attention.qkv_proj/o_proj and SwiGLU.gate_up_proj/down_proj). 0.0 = off.
    # The mask is resampled per training batch by the train loop.
    weight_dropout: float = 0.0
