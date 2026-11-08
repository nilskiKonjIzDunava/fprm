import os

import torch
import torch.nn as nn
import torch.distributed as dist

from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from pretrain_config import PretrainConfig
from puzzle_dataset import PuzzleDatasetMetadata

from utils.functions import load_model_class


def load_checkpoint(model: nn.Module, config: PretrainConfig, strict: bool = True):
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint {config.load_checkpoint} (strict={strict})")

        # Load state dict
        state_dict = torch.load(config.load_checkpoint, map_location="cuda")

        # Resize and reset puzzle emb if needed
        puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
        expected_shape: torch.Size = model.model.puzzle_emb.weights.shape  # type: ignore
        if puzzle_emb_name in state_dict:
            puzzle_emb = state_dict[puzzle_emb_name]
            if puzzle_emb.shape != expected_shape:
                print(f"Resetting puzzle embedding as shape is different. Found {puzzle_emb.shape}, Expected {expected_shape}")
                # Re-initialize using mean
                state_dict[puzzle_emb_name] = (
                    torch.mean(puzzle_emb, dim=0, keepdim=True).expand(expected_shape).contiguous()
                )
        result = model.load_state_dict(state_dict, assign=True, strict=strict)
        if not strict:
            if result.missing_keys:
                print(f"  missing ({len(result.missing_keys)}): {result.missing_keys[:5]}{'...' if len(result.missing_keys) > 5 else ''}")
            if result.unexpected_keys:
                print(f"  unexpected ({len(result.unexpected_keys)}): {result.unexpected_keys[:5]}{'...' if len(result.unexpected_keys) > 5 else ''}")


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int, strict_load: bool = True):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)  # type: ignore

        # Load checkpoint
        if rank == 0:
            load_checkpoint(model, config, strict=strict_load)

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Optimizers and lr
    dense_optimizer = _build_dense_optimizer(model, config)

    if config.arch.puzzle_emb_ndim == 0:
        optimizers = [dense_optimizer]
        optimizer_lrs = [config.lr]
    elif config.freeze_weights:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr
        ]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            ),
            dense_optimizer,
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr,
            config.lr
        ]

    return model, optimizers, optimizer_lrs


def _split_decay_param_groups(model: nn.Module, weight_decay: float):
    """Split params into weight-decay / no-weight-decay groups.

    Any parameter tagged ``p._no_weight_decay = True`` (e.g. residual-scale
    alphas, the conv-over-fixed-point-state weights/bias, and SpecNormalizedLinear's
    scale) is placed in the no-decay group.
    """
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if getattr(p, "_no_weight_decay", False):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _build_dense_optimizer(model: nn.Module, config: PretrainConfig) -> torch.optim.Optimizer:
    name = config.optimizer.lower()
    param_groups = _split_decay_param_groups(model, config.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(
            param_groups,
            lr=0,  # Needs to be set by scheduler
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )
    raise ValueError(f"Unknown optimizer {config.optimizer!r}; expected 'adamw'")