from dataclasses import dataclass
from typing import Tuple, Dict, Generator, Union
import math
import random
import torch
import torch.nn.functional as F
from torch import nn

from scipy.stats import gamma, expon

from ..layers import (
    RotaryEmbedding,
    CastedEmbedding,
    CastedLinear,
    rms_norm,
)
from ..sparse_embedding import CastedSparseEmbedding
from ..transformer import FixedPointTransformer

from .fprm_config import FPRMConfig
from .model_utils import FixedPointOptimizer

IGNORE_LABEL_ID = -100


@dataclass
class FixedPointReasoningModel_ACTV1InnerCarry:
    z_L_state: dict
    dropout_mask: torch.Tensor

@dataclass
class FixedPointReasoningModel_ACTV1Carry:
    inner_carry: FixedPointReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class FixedPointReasoningModel_Inner(nn.Module):
    def __init__(self, config: FPRMConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        # Reasoning Layers
        self.L_level = FixedPointTransformer(self.config, self.config.L_layers)

        self.L_optimizer = FixedPointOptimizer(self.config)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return FixedPointReasoningModel_ACTV1InnerCarry(
            z_L_state = None,
            dropout_mask = None,
        )

    def reset_carry(self, 
                    reset_flag: torch.Tensor, 
                    batch: torch.Tensor,
                    carry: FixedPointReasoningModel_ACTV1InnerCarry):        
        shape = (batch.shape[0], batch.shape[1] + self.puzzle_emb_len, self.config.hidden_size)
        device = batch.device
        dtype = self.forward_dtype
        
        if self.training:
            dropout_mask = torch.empty(*shape, device=device, dtype=dtype).bernoulli_(p=1 - self.config.variational_dropout).div_(1 - self.config.variational_dropout)
            if carry.dropout_mask is not None:
                dropout_mask = torch.where(reset_flag.view(-1, 1, 1), dropout_mask, carry.dropout_mask)
        else:
            dropout_mask = torch.ones(*shape, device=device, dtype=dtype)

        return FixedPointReasoningModel_ACTV1InnerCarry(
            z_L_state = self.L_optimizer.reset(reset_flag, shape, dtype, device, carry.z_L_state),
            dropout_mask=dropout_mask,
        )
    
    # Disable torch.compile for this method: the 'exact' branch backprops through
    # autograd.grad(..., create_graph=True), and torch.compile's aot_autograd
    # backend does not support double-backward. Eager-mode autograd does.
    @torch._dynamo.disable
    def _jacobian_reg(self, state: Dict[str, torch.Tensor], input_embeddings: torch.Tensor,
                      dropout_mask: torch.Tensor, seq_info: Dict[str, any]):
        # Eval path discards the regularizer loss; skip the work. Required for
        # 'exact' too, where autograd.grad would fail under eval's no_grad.
        if not self.training or self.config.jacobian_reg == 'none':
            return state["y"].new_zeros(())

        estimate = 0

        from torch.nn.attention import SDPBackend, sdpa_kernel
        _sdpa_ctx = sdpa_kernel(SDPBackend.MATH)

        for n in range(self.config.n_jacobian_samples):
            v = (torch.randn_like(state["y"]) / math.sqrt(state["y"].shape[-1])).detach()
            z_in = state["y"].detach().requires_grad_(True)

            if self.config.jacobian_reg == 'fda':
                z_p = dropout_mask * self.L_level(z_in + self.config.jacobian_eps * v, input_embeddings, **seq_info)
                z_m = dropout_mask * self.L_level(z_in - self.config.jacobian_eps * v, input_embeddings, **seq_info)

                jvp = (z_p - z_m) / (2 * self.config.jacobian_eps)

                estimate += jvp.pow(2).mean() / self.config.n_jacobian_samples

            elif self.config.jacobian_reg == 'exact':
                with torch.enable_grad(), _sdpa_ctx:
                    f_z = dropout_mask * self.L_level(z_in, input_embeddings, **seq_info)
                    s = (v * f_z).sum()
                    # create_graph=True so the loss can backprop through Jt_v;
                    # retain_graph defaults to create_graph here (we need it).
                    (jvp,) = torch.autograd.grad(s, z_in, create_graph=True)
                estimate += jvp.pow(2).mean() / self.config.n_jacobian_samples

            else:
                raise ValueError(f"Unknown jacobian_reg: {self.config.jacobian_reg!r}")

        return estimate

    def _z_step(self, state: Dict[str, torch.Tensor], input_embeddings: torch.Tensor, 
                dropout_mask: torch.Tensor, seq_info: Dict[str, any]):
        z_new = dropout_mask * self.L_level(state["y"], input_embeddings, **seq_info)
        return self.L_optimizer.step(state, z_new)

    def forward(self, 
                carry: FixedPointReasoningModel_ACTV1InnerCarry, 
                batch: Dict[str, torch.Tensor],
                force_grad: bool,
                n_steps: int) -> Tuple[
        Tuple[FixedPointReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ]:
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Slice rotary cache to the actual input length so the same model
        # can be evaluated at sequence lengths shorter than config.seq_len
        # (length-generalisation for state tracking).
        cos_sin = None
        if hasattr(self, "rotary_emb"):
            cos, sin = self.rotary_emb()
            s = input_embeddings.shape[1]
            cos_sin = (cos[:s], sin[:s])
        seq_info = dict(cos_sin=cos_sin, puzzle_emb_len=self.puzzle_emb_len)
        z_state, dropout_mask = carry.z_L_state, carry.dropout_mask
        
        with torch.set_grad_enabled(force_grad):
            for _ in range(n_steps):
                z_state = self._z_step(z_state, input_embeddings, dropout_mask, seq_info)
        
        output = self.lm_head(z_state['y'])[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_state['y'][:, 0]).to(torch.float32) # Q-head; uses the first puzzle_emb position
        jacobian_loss = self._jacobian_reg(z_state, input_embeddings, dropout_mask, seq_info)
        new_carry = FixedPointReasoningModel_ACTV1InnerCarry(z_L_state=self.L_optimizer.detach_state(z_state),
                                                                         dropout_mask=carry.dropout_mask)  # New carry no grad
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1], jacobian_loss)


class FixedPointReasoningModel_ACTV1(nn.Module):
    """Single-state FPTRM wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = FPRMConfig(**config_dict)
        self.inner = FixedPointReasoningModel_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb
    
    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return FixedPointReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
    
    def set_num_iters(self):
        if self.training:
            if self.config.max_iter_dist == 'gamma':
                sampled_max_iter = gamma.rvs(a=self.config.gamma_alpha, scale=self.config.gamma_scale)
            elif self.config.max_iter_dist == 'expon':
                sampled_max_iter = expon.rvs(scale=self.config.expon_scale)
            elif self.config.max_iter_dist == 'det':
                sampled_max_iter = self.config.max_iter

            if self.config.max_iter_dist == 'det':
                self.max_iter = max(0, int(sampled_max_iter))
            else:
                self.max_iter = max(1, int(sampled_max_iter))
        else:
            self.max_iter = self.config.max_iter_eval if self.config.max_iter_eval is not None else self.config.max_iter

    def forward(
        self,
        carry: FixedPointReasoningModel_ACTV1Carry,
        batch: Dict[str, torch.Tensor],
    ):
        # Update data, carry (removing halted sequences)
        # Handled inside the optimizer
        new_inner_carry = self.inner.reset_carry(carry.halted, batch['inputs'], carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward-backward inner model
        n_steps = self.config.n_backwards_L if self.training else 1
        new_inner_carry, logits, (q_halt_logits, q_continue_logits, jacobian_loss) = self.inner(new_inner_carry, new_current_data, 
                                                                                                force_grad=self.training, n_steps=n_steps)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        if self.training and self.config.jacobian_reg != 'none' and self.config.n_jacobian_samples > 0:
            outputs["jacobian_loss"] = jacobian_loss

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1

            if self.config.halting_mechanism == 'act':
                is_last_step = new_steps >= self.config.halt_max_steps
            else:
                is_last_step = new_steps >= self.max_iter
            
            halted = is_last_step

            # if testing, use fixed-points
            if not self.training:
                # during inference we only halt for the entire sequence
                halted = halted | (new_inner_carry.z_L_state['residues'].max() < self.config.fp_thresh) \
                                | (new_inner_carry.z_L_state['stepsize'].max() < 1e-3)

            # if training, and ACT is enabled
            cap = self.config.halt_max_steps if self.config.halting_mechanism == 'act' else self.max_iter            
            if self.training and (cap > 1):
                
                if self.config.halting_mechanism == 'act':
                    if self.config.no_ACT_continue:
                        halted = halted | (q_halt_logits > 0)
                    else:
                        halted = halted | (q_halt_logits > q_continue_logits)
                
                    # Exploration
                    min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                    halted = halted & (new_steps >= min_halt_steps)

                elif self.config.halting_mechanism == 'fixed_point':
                    # Exploration is implemented by self.max_iter if we choose to use it
                    halted = halted | (new_inner_carry.z_L_state['residues'] < self.config.fp_thresh) \
                                    | (new_inner_carry.z_L_state['stepsize'].view(-1) < 1e-3) 
                
                elif self.config.halting_mechanism == 'fixed_iterations':
                    pass

                else:
                    raise ValueError("FPRM only accepts ACT, Fixed_point, and Fixed_iterations as its halting mechanism.")                

                if not self.config.no_ACT_continue:
                    # Compute target Q
                    # NOTE: No replay buffer and target networks for computing target Q-value.
                    # As batch_size is large, there're many parallel envs.
                    # Similar concept as PQN https://arxiv.org/abs/2407.04811
                    _, _, (next_q_halt_logits, next_q_continue_logits, _) = self.inner(new_inner_carry, new_current_data, force_grad=False, n_steps=1)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return FixedPointReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
