import math
from typing import Tuple, Optional
import einops
import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.functional import scaled_dot_product_attention

from models.common import trunc_normal_init_


CosSin = Tuple[torch.Tensor, torch.Tensor]


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class SpecNormalizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.skip_power_iter = (self.in_features == 1 or self.out_features == 1)
        
        # 1. Weights - kept in float32 for spectral norm precision
        self.weight = nn.Parameter(trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5)))
        
        # 2. Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        self.scale = nn.Parameter(torch.tensor(torch.linalg.matrix_norm(self.weight, ord=2).item()), requires_grad=True)
        self.scale._no_weight_decay = True
        
        self.register_buffer("u", torch.ones(out_features))
        self.register_buffer("v", torch.ones(in_features))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.skip_power_iter:
            # exact spectral norm for rank-1 matrices
            sigma = torch.norm(self.weight, p=2.0)
        else:
            u_init = self.u.detach().clone()
            v_init = self.v.detach().clone()
            # power iterations to estimate spectral norm
            sigma, u, v = self._power_iterations(A=self.weight, u_init=u_init, v_init=v_init, num_iterations=50)

            # store the results for the next forward pass
            with torch.no_grad():
                self.u = u.detach()
                self.v = v.detach()

        weight_sn = (self.weight / sigma) * self.scale

        return F.linear(
            input,
            weight_sn.to(input.dtype),
            bias=self.bias.to(input.dtype) if self.bias is not None else None,
        )
    
    @staticmethod
    @torch.compile(fullgraph=True)
    def _power_iterations(A: torch.Tensor, u_init: torch.Tensor, v_init: torch.Tensor, num_iterations: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Power iterations to estimate spectral norm 
        """
        v = v_init
        u = u_init

        for _ in range(num_iterations):
            v = nn.functional.normalize(torch.mv(A.t(), u), dim=0)
            u = nn.functional.normalize(torch.mv(A, v), dim=0)

        sigma = torch.dot(u, torch.mv(A, v))
        return sigma, u, v


class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool,
                 dropout: float = 0.0):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))
        
        self.dropout = dropout
        self.mask = nn.Buffer(torch.ones_like(self.weight))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight * self.mask if self.training else self.weight
        return F.linear(input, weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
    
    def reset_mask(self):
        if not self.training or self.dropout == 0.0:
            self.mask.fill_(1.0)
        else:
            self.mask.bernoulli_(1 - self.dropout).div_(1 - self.dropout)

class CastedEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    def __init__(self, hidden_size: int, head_dim: int, num_heads: int, num_key_value_heads: int, causal: bool = False,
                 softmax_temp: float = 1.0, use_spec_norm_linear: bool = False, weight_dropout: float = 0.0):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal
        self.softmax_temp = softmax_temp

        proj_dim = (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim
        # added feat: spectral normalization
        if use_spec_norm_linear:
            self.qkv_proj = SpecNormalizedLinear(self.hidden_size, proj_dim, bias=False)
        else:
            self.qkv_proj = CastedLinear(self.hidden_size, proj_dim, bias=False, dropout=weight_dropout)

        # added feat: spectral normalization
        if use_spec_norm_linear:
            self.o_proj = SpecNormalizedLinear(self.output_size, self.hidden_size, bias=False)
        else:
            self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False, dropout=weight_dropout)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor, return_entropy: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # flash attn
        query, key, value = map(lambda t: einops.rearrange(t, 'B S H D -> B H S D'), (query, key, value)) # needed for scaled_dot_product_attention but not flash_attn_func
        attn_output = scaled_dot_product_attention(query=query, key=key, value=value, is_causal=self.causal, scale=1.0 / (math.sqrt(self.head_dim) * self.softmax_temp))
        attn_output = einops.rearrange(attn_output, 'B H S D -> B S H D')
        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)  # type: ignore
        return self.o_proj(attn_output)


class LinearSwish(nn.Module):
    def __init__(self, hidden_size: int, reverse: bool = False, use_spec_norm_linear: bool = False):
        super().__init__()

        if use_spec_norm_linear:
            self.linear = SpecNormalizedLinear(hidden_size, hidden_size, bias=False)
        else:
            self.linear = CastedLinear(hidden_size, hidden_size, bias=False)
        self.reverse = reverse

    def forward(self, x):
        if self.reverse:
            return F.silu(self.linear(x))
        else:
            return self.linear(F.silu(x))


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float, use_spec_norm_linear: bool = False, weight_dropout: float = 0.0):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        if use_spec_norm_linear:
            self.gate_up_proj = SpecNormalizedLinear(hidden_size, inter * 2, bias=False)
            self.down_proj    = SpecNormalizedLinear(inter, hidden_size, bias=False)
        else:
            self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False, dropout=weight_dropout)
            self.down_proj    = CastedLinear(inter, hidden_size, bias=False, dropout=weight_dropout)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)
