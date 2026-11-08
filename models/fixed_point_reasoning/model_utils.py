import math
import torch
import torch.nn as nn
from typing import Dict

from ..common import trunc_normal_init_


def shift(input:torch.Tensor, shift, dim=0, fillval=0):
    # torch.roll without the copy of the wrap-around section
    size = input.size(dim)
    fill = torch.full_like(input.narrow(dim, 0, abs(shift)), fillval)
    if shift > 0:
        output = torch.cat([fill, input.narrow(dim, 0, size-shift)], dim=dim)
    if shift < 0:
        output = torch.cat([input.narrow(dim, -shift, size+shift), fill], dim=dim)
    return output

class FixedPointOptimizer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.stepsize = config.stepsize
        self.stepsize_decay_train = config.stepsize_decay_train
        self.stepsize_decay_eval = config.stepsize_decay_eval
        self.decay_patience = config.decay_patience
        self.eps = config.eps
        self.outlier_quantile = config.outlier_quantile
        self.max_iter = config.max_iter
        self.init_std = config.init_std
        self.additive_noise_std = config.additive_noise_std
        self.fp_thresh = config.fp_thresh

        self.fixed_init = config.fixed_init
        if self.fixed_init:
            fwd_dtype = getattr(torch, config.forward_dtype)
            self.init_vec = nn.Buffer(
                trunc_normal_init_(torch.empty(config.hidden_size, dtype=fwd_dtype), std=self.init_std),
                persistent=True,
            )

    def detach_state(self, state: dict):
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.detach()
        return state
        
    def reset(self, reset_flag: torch.Tensor, shape: tuple, dtype: torch.dtype, 
              device: torch.device, state: dict, reset_metadata: bool = False):
        batch_size, seq_len, hidden_size = shape[0], shape[1], shape[2]
        reset_flag_1d = reset_flag.view(-1)
        reset_flag_3d = reset_flag_1d.view(-1, 1, 1)

        if self.fixed_init:
            y = self.init_vec.to(dtype=dtype, device=device).expand(batch_size, seq_len, hidden_size).contiguous()
        else:
            y = trunc_normal_init_(torch.empty(batch_size, seq_len, hidden_size, dtype=dtype, device=device), std=self.init_std)
        residues = torch.inf * torch.ones(batch_size).to(device)
        stepsize = (self.stepsize * torch.ones(batch_size, 1, 1, dtype=dtype, device=device))
        patience = self.decay_patience * torch.ones(batch_size).to(device)
        iter_idx = torch.zeros(batch_size, dtype=torch.int32, device=device)
        best_residues = torch.inf * torch.ones(batch_size).to(device)

        if state is None:
            state = dict(y=y.contiguous(),
                         residues=residues,
                         stepsize=stepsize,
                         patience=patience,
                         iter_idx=iter_idx,
                         best_residues=best_residues)
        else:
            state = dict(y=torch.where(reset_flag_3d, y.contiguous(), state['y']),
                         residues=residues if reset_metadata else torch.where(reset_flag_1d, residues, state['residues']),
                         stepsize=stepsize if reset_metadata else torch.where(reset_flag_3d, stepsize, state['stepsize']),
                         patience=patience if reset_metadata else torch.where(reset_flag_1d, patience, state['patience']),
                         iter_idx=iter_idx if reset_metadata else torch.where(reset_flag_1d, iter_idx, state['iter_idx']),
                         best_residues=best_residues if reset_metadata else torch.where(reset_flag_1d, best_residues, state['best_residues']))
        
        return state

    def step(self, state:Dict[str, torch.Tensor], y:torch.Tensor):
        state_dtype = state["y"].dtype
        if y.dtype != state_dtype:
            y = y.to(state_dtype)

        with torch.no_grad():
            residues = (state['y'].detach() - y.detach()).norm(p=torch.inf, dim=-1) / (y.detach().norm(p=torch.inf, dim=-1) + self.eps)
            residues = residues.max(dim=1)[0]

        stepsize = state['stepsize']
        if stepsize.dtype != state_dtype:
            stepsize = stepsize.to(state_dtype)
        state['y'] = y * stepsize + state['y'] * (1 - stepsize) + self.additive_noise_std * torch.randn_like(state['y'])

        improved = residues < state['best_residues']

        # update patience and lowest residue
        state['residues'] = residues
        state['best_residues'] = torch.where(improved, residues, state['best_residues'])
        state['patience'] = torch.where(improved, self.decay_patience, state['patience']-1)

        # update damping factor, reset patience
        adapt = (state['patience'] <= 0) & (state['residues'] >= self.fp_thresh)
        state['patience'] = torch.where(adapt, self.decay_patience, state['patience'])
        stepsize_dtype = state['stepsize'].dtype
        # Separate train/eval decay (selected by module mode set via model.train()/eval()).
        decay = self.stepsize_decay_train if self.training else self.stepsize_decay_eval
        state['stepsize'] = state['stepsize'] * torch.where(adapt, decay, 1).to(stepsize_dtype).reshape(-1, 1, 1)

        state['iter_idx'] = state['iter_idx'] + 1
        return state

    def cont(self, state: Dict[str, torch.Tensor], thresh: float):
        if int(state['iter_idx'].max().item()) == 0:
            return self.max_iter > 0

        q = 1 - self.outlier_quantile if self.training else 1.0
        return (
            (torch.quantile(state['residues'].float(), q=q) >= thresh)
            & (torch.quantile(state['iter_idx'].float(), q=q) < self.max_iter)
            & (torch.quantile(state['stepsize'].float(), q=q) > 1e-3)
        )

class VariationalDropout(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        assert 0.0 <= dropout < 1.0, f"dropout must be in [0, 1), got {dropout}"
        self.dropout = dropout
        self._mask: torch.Tensor | None = None

    def sample_mask(self, x: torch.Tensor) -> None:
        if not self.training or self.dropout == 0.0:
            self._mask = None
            return
        keep_prob = 1.0 - self.dropout
        self._mask = torch.bernoulli(torch.full_like(x, keep_prob)) / keep_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout == 0.0 or self._mask is None:
            return x
        return x * self._mask


class VariationalDropToken1d(nn.Module):
    def __init__(self, dropout: float = 0.0, token_first: bool = True):
        super().__init__()
        assert 0.0 <= dropout < 1.0, f"dropout must be in [0, 1), got {dropout}"
        self.dropout = dropout
        self.token_first = token_first
        self._mask: torch.Tensor | None = None

    def sample_mask(self, x: torch.Tensor) -> None:
        if not self.training or self.dropout == 0.0:
            self._mask = None
            return
        keep_prob = 1.0 - self.dropout
        if self.token_first:
            B, L, _ = x.shape
            self._mask = torch.bernoulli(torch.full((B, L, 1), keep_prob, device=x.device, dtype=x.dtype)) / keep_prob
        else:
            B, _, L = x.shape
            self._mask = torch.bernoulli(torch.full((B, 1, L), keep_prob, device=x.device, dtype=x.dtype)) / keep_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout == 0.0 or self._mask is None:
            return x
        return x * self._mask
