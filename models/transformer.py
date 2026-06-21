from typing import Tuple, List, Optional
import torch
from torch import nn

import math

from .layers import rms_norm, CastedLinear, SpecNormalizedLinear, Attention, SwiGLU
from .config import ReasoningModelConfig

CosSin = Tuple[torch.Tensor, torch.Tensor]

'''
    modified some of the TRM blocks to use in the FP variant
    main modifications:
        - added conv1d over qkv and swiglu; absolutely necessary for FP stability
        - added QK normalization; helpful to stability
        - added weight norm to all linear layers; helpful to stability
'''

    
class FixedPointTransformerBlock(nn.Module):
    def __init__(self, config: ReasoningModelConfig) -> None:
        super().__init__()

        self.config = config
        weight_dropout = getattr(config, "weight_dropout", 0.0)
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len, # L
                expansion=config.expansion,
                weight_dropout=weight_dropout,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=config.causal,
                softmax_temp=config.softmax_temp,
                weight_dropout=weight_dropout,
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
            weight_dropout=weight_dropout,
        )
        self.norm_eps = config.rms_norm_eps
        self.norm_type = config.norm_type

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor, alpha_1: torch.Tensor = None, beta_1: torch.Tensor = None) -> torch.Tensor:
        # B, L, D = hidden_states.shape
        
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1, 2)
            out = self.mlp_t(rms_norm(hidden_states, variance_epsilon=self.norm_eps) if (self.norm_type == 'pre-norm' or self.norm_type == 'peri-norm') else hidden_states)
          
            if self.norm_type == 'peri-norm':
                out = rms_norm(out, variance_epsilon=self.norm_eps)

            hidden_states = alpha_1.transpose(1, 2) * hidden_states + beta_1.transpose(1, 2) * out

            if self.norm_type == 'post-norm':
                hidden_states = rms_norm(hidden_states, variance_epsilon=self.norm_eps)

            hidden_states = hidden_states.transpose(1, 2)
        else:
            # Self Attention
            out = self.self_attn(
                cos_sin=cos_sin,
                hidden_states=rms_norm(hidden_states, variance_epsilon=self.norm_eps) if (self.norm_type == 'pre-norm' or self.norm_type == 'peri-norm') else hidden_states,
            )
         
            if self.norm_type == 'peri-norm':
                out = rms_norm(out, variance_epsilon=self.norm_eps)

            hidden_states = alpha_1 * hidden_states + beta_1 * out

            if self.norm_type == 'post-norm':
                hidden_states = rms_norm(hidden_states, variance_epsilon=self.norm_eps)
        
        # Fully Connected
        out = self.mlp(rms_norm(hidden_states, variance_epsilon=self.norm_eps) if self.norm_type == 'pre-norm' or self.norm_type == 'peri-norm' else hidden_states)
 
        if self.norm_type == 'peri-norm':
            out = rms_norm(out, variance_epsilon=self.norm_eps)

        hidden_states = alpha_1 * hidden_states + beta_1 * out
 
        if self.norm_type == 'post-norm':
            hidden_states = rms_norm(hidden_states, variance_epsilon=self.norm_eps)
        
        return hidden_states


class FixedPointTransformer(nn.Module):
    def __init__(
        self,
        config: dict,
        n_layers: int
    ) -> None:
        super().__init__()

        self.config = config
        self.layers = torch.nn.ModuleList([FixedPointTransformerBlock(config) for _ in range(n_layers)])

        hidden_size = self.config.hidden_size
        # conv_params
        conv_kernel_size = self.config.conv_kernel_size
        conv_bias = self.config.conv_bias
        # residual scale params
        residual_scale = self.config.residual_scale
        self.outer_only = self.config.outer_only
        alpha_1_init = self.config.alpha_1_init
        alpha_2_init = self.config.alpha_2_init
        norm_placement = self.config.norm_placement
        normalize_input_injection = self.config.normalize_input_injection

        self.conv_type = self.config.conv_type

        if self.conv_type == 'conv1d':
            self.conv = nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size - 1,
                groups=hidden_size,
                bias=conv_bias,
            )
        elif self.conv_type == 'conv2d':
            if conv_kernel_size % 2 == 0:
                raise ValueError(f"conv_kernel_size must be odd for conv2d (got {conv_kernel_size})")
            self.conv = nn.Conv2d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2,
                groups=hidden_size,
                bias=conv_bias,
            )
        else:
            pass

        if residual_scale in ("none", "None"):
            residual_scale = None

        valid_residual_scale = {None, "fixed", "input-independent", "input-dependent"}
        if residual_scale not in valid_residual_scale:
            raise ValueError(f"Unknown residual_scale: {residual_scale}")

        valid_norm_placement = {"input", "output", "none"}
        if norm_placement not in valid_norm_placement:
            raise ValueError(f"Unknown norm_placement: {norm_placement}")

        if not (0.0 < alpha_1_init < 1.0):
            raise ValueError("alpha_1_init must be in (0, 1)")
        if not (0.0 < alpha_2_init < 1.0):
            raise ValueError("alpha_2_init must be in (0, 1)")

        self.residual_scale = residual_scale
        self.norm_placement = norm_placement
        self.normalize_input_injection = normalize_input_injection
        
        if self.residual_scale is not None:
            alpha_1_init_logit = math.log(alpha_1_init / (1 - alpha_1_init))
            alpha_2_init_logit = math.log(alpha_2_init / (1 - alpha_2_init))
            if self.residual_scale == 'fixed':
                self.alpha_2_param = nn.Parameter(torch.tensor(alpha_2_init), requires_grad=False)
                if not self.outer_only:
                    self.alpha_1_param = nn.Parameter(torch.tensor(alpha_1_init), requires_grad=False)

            elif self.residual_scale == 'input-independent':
                self.alpha_2_param = nn.Parameter(alpha_2_init_logit * torch.ones(hidden_size), requires_grad=True)
                self.alpha_2_param._no_weight_decay = True
                if not self.outer_only:
                    self.alpha_1_param = nn.Parameter(alpha_1_init_logit * torch.ones(hidden_size), requires_grad=True)
                    self.alpha_1_param._no_weight_decay = True

            elif self.residual_scale == 'input-dependent':
                if self.config.use_spec_norm_linear:
                    self.alpha_2_layer = SpecNormalizedLinear(hidden_size, hidden_size, bias=True)
                    if not self.outer_only:
                        self.alpha_1_layer = SpecNormalizedLinear(hidden_size, hidden_size, bias=True)
                
                else:
                    self.alpha_2_layer = CastedLinear(hidden_size, hidden_size, bias=True)
                    if not self.outer_only:
                        self.alpha_1_layer = CastedLinear(hidden_size, hidden_size, bias=True)

    def forward(
        self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        if self.normalize_input_injection:
            input_injection = rms_norm(input_injection, variance_epsilon=self.config.rms_norm_eps)

        puzzle_emb_len = kwargs.pop('puzzle_emb_len', 0)

        if self.conv_type != 'none':
            batch_size, total_seq_len, hidden_size = hidden_states.shape
            puzzle_part = hidden_states[:, :puzzle_emb_len]
            grid_part = hidden_states[:, puzzle_emb_len:]

            grid_len = total_seq_len - puzzle_emb_len

            if self.conv_type == 'conv1d':
                grid_part = self.conv(grid_part.transpose(1, 2)).transpose(1, 2)[:, :grid_part.shape[1], :].contiguous()
            elif self.conv_type == 'conv2d':
                hw = int(math.sqrt(grid_len))
                assert hw * hw == grid_len, f"grid_len {grid_len} is not a perfect square"

                grid_part = grid_part.reshape(batch_size, hw, hw, hidden_size).permute(0, 3, 1, 2).contiguous()
                grid_part = self.conv(grid_part).permute(0, 2, 3, 1).reshape(batch_size, grid_len, hidden_size).contiguous()
            else:
                raise ValueError("Unknown convolution type.")
            
            hidden_states = torch.cat([puzzle_part, grid_part], dim=1).contiguous()
        else:
            pass 

        if self.residual_scale is not None:
            if self.residual_scale == 'fixed':
                alpha_2 = self.alpha_2_param
                if not self.outer_only:
                    alpha_1 = self.alpha_1_param
            elif self.residual_scale == 'input-independent':
                alpha_2 = torch.sigmoid(self.alpha_2_param).to(dtype=hidden_states.dtype).reshape(1, 1, -1)
                if not self.outer_only:
                    alpha_1 = torch.sigmoid(self.alpha_1_param).to(dtype=hidden_states.dtype).reshape(1, 1, -1)
            elif self.residual_scale == 'input-dependent':
                alpha_2 = torch.sigmoid(self.alpha_2_layer(hidden_states)).to(dtype=hidden_states.dtype)
                if not self.outer_only:
                    alpha_1 = torch.sigmoid(self.alpha_1_layer(hidden_states)).to(dtype=hidden_states.dtype)
            
            # weight tying for beta_1, beta_2
            # based on asymptotic analysis
            if self.outer_only:
                one = hidden_states.new_ones(1, 1, 1)
                beta_2 = 1 - alpha_2
                alpha_1, beta_1 = one, one
            else:
                beta_2 = 1 - alpha_2 * alpha_1.pow(2 * len(self.layers))
                beta_1 = beta_2 * (1 - alpha_1) / (1 - alpha_1.pow(2 * len(self.layers)) + 1e-5)    
        else:
            one = hidden_states.new_ones(1, 1, 1)
            alpha_1, alpha_2, beta_1, beta_2 = one, one, one, one

        if self.norm_placement == 'input':
            hidden_states = rms_norm(hidden_states, variance_epsilon=self.config.rms_norm_eps)
        
        hidden_states = alpha_2 * hidden_states + beta_2 * input_injection
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                alpha_1=alpha_1,
                beta_1=beta_1,
                **kwargs,
            )

        if self.norm_placement == 'output':
            hidden_states = rms_norm(hidden_states, variance_epsilon=self.config.rms_norm_eps)
        
        return hidden_states
    
