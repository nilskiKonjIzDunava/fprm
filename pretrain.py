from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import math
import yaml
import shutil
import copy

import torch
import torch._dynamo
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

# Per-batch set_num_iters() writes a fresh int to model.max_iter (an nn.Module
# attribute). Dynamo treats nn.Module ints as static, so each new value blows
# the recompile cache (limit=8) and falls back to eager — ~2.3x slowdown.
# Marking nn.Module ints as unspecialized lets dynamo trace max_iter as
# symbolic so the compiled graph is reused across values.
torch._dynamo.config.allow_unspec_int_on_nn_module = True

import tqdm
import wandb
import coolname
import hydra
from omegaconf import DictConfig

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from utils.flop_profiler import FlopProfiler
from utils.resume import restore_train_state, save_resume_bundle
from models.ema import EMAHelper
from plots.io import write_curves
import length_gen_eval


from pretrain_config import PretrainConfig
from create_model import create_model

@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if len(config.data_paths_test)>0 and split=="test" else config.data_paths,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata


def mix_weights_direct(device, alpha, net, nets):
    sd = []
    for i in range(len(nets)):
        sd += [nets[i].state_dict()]
    sd_alpha = {}
    for k in sd[0].keys():
        comb_net = alpha[0]*sd[0][k].to(device)
        for i in range(1,len(nets)):
            comb_net += alpha[i]*sd[i][k].to(device)
        sd_alpha[k] =  comb_net
    net.load_state_dict(sd_alpha)
    return net

def autocast_ctx(config: PretrainConfig):
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_name = (config.arch.__pydantic_extra__ or {}).get("forward_dtype", "bfloat16")
    dtype = getattr(torch, dtype_name)
    return torch.amp.autocast(device_type=device_type, dtype=dtype, cache_enabled=False)


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    # Estimated total training steps
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    # Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, rank=rank, world_size=world_size)

    return TrainState(
        step=0,
        total_steps=total_steps,

        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    # FIXME: Only saved model.
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}"))


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )

def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    data_paths = config.data_paths_test if len(config.data_paths_test) > 0 else config.data_paths
    # Initialize evaluators
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "evaluators.")(
                data_path=data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__
            )  # type: ignore
            evaluators.append(cls)

    return evaluators

def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int, profiler: FlopProfiler):
    train_state.step += 1
    if train_state.step > train_state.total_steps:  # At most train_total_steps
        return

    # One-time per-rank batch sanity print: with world_size GPUs each rank
    # should receive global_batch_size / world_size examples per step.
    if train_state.step == 1:
        any_key = next(iter(batch))
        per_rank_bs = batch[any_key].shape[0]
        print(
            f"[gpu-diag] rank={rank}/{world_size} step=1 "
            f"per_rank_batch={per_rank_bs} global_batch={global_batch_size} "
            f"expected_per_rank={global_batch_size // max(world_size, 1)}",
            flush=True,
        )

    # To device
    batch = {k: v.cuda() for k, v in batch.items()}

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward + backward (FlopCounterMode wraps once, then is a no-op)
    with profiler.measure("train"):
        with autocast_ctx(config):
            train_state.carry, masked_loss, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])
            scaled_loss = (1 / global_batch_size) * masked_loss
        scaled_loss.backward()
    profiler.add_train(1)

    # Allreduce
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)

    # Apply optimizer
    lr_this_step = None    
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
            
        optim.step()
        optim.zero_grad()

    # Reduce metrics
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
        # Reduce and reconstruct
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            
            # Postprocess
            count = max(reduced_metrics["count"], 1)  # Avoid NaNs
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

            reduced_metrics["train/lr"] = lr_this_step
            reduced_metrics.update(profiler.metrics())
            return reduced_metrics

def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
    profiler: FlopProfiler,
):
    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        # Length-generalisation mode: single forward per batch, per-k metrics
        # via cumulative cumsum. Runs in parallel with the standard per-set
        # aggregation below; both are populated from the same forward.
        ks = length_gen_eval.length_gen_ks(config)
        lg_state = None
        k_pos = None
        if ks is not None:
            return_keys = set(return_keys) | {"preds"}
            k_pos = length_gen_eval.k_pos_tensor(ks, eval_metadata.seq_len, device="cuda")

        # Run evaluation
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        eval_model = train_state.model

        # Models with iterative reasoning (FPTRM variants) read self.max_iter
        # inside forward; their set_num_iters() pins it to config.max_iter when
        # called in eval mode. Call it here so eval has a deterministic budget
        # regardless of whether we got here from training or from an eval-only
        # entry point.
        if hasattr(eval_model, "set_num_iters"):
            eval_model.set_num_iters()

        save_preds = {}

        metric_keys = []
        metric_values = None

        carry = None
        processed_batches = 0

        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")

            # To device
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = eval_model.initial_carry(batch)  # type: ignore

            # Forward
            inference_steps = 0
            while True:
                with profiler.measure("eval"), autocast_ctx(config):
                    carry, _, loss, metrics, preds, all_finish = eval_model(
                        carry=carry, batch=batch, return_keys=return_keys
                    )
                inference_steps += 1

                if all_finish:
                    break
            profiler.add_eval(inference_steps)

            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            if ks is not None:
                lg_state = length_gen_eval.accumulate(
                    lg_state, preds["preds"], batch["labels"], k_pos,
                )

            del carry, loss, preds, batch, all_finish

            # Aggregate metrics
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(
                    sorted(metrics.keys())
                )  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda"
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

            del metrics

        # concatenate save preds
        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

        # Save preds
        if config.checkpoint_path is not None and len(save_preds):
            # Each rank save predictions independently
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(
                save_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}")
            )

        del save_preds

        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                # Postprocess
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        # Length-generalisation curve (if enabled): reduce running sums across
        # ranks and pack as {ks, accuracy, sequence_accuracy} under key "length_gen".
        if lg_state is not None and ks is not None:
            if world_size > 1:
                for t in lg_state.values():
                    dist.reduce(t, dst=0)
            if rank == 0:
                if reduced_metrics is None:
                    reduced_metrics = {}
                reduced_metrics["length_gen"] = length_gen_eval.finalize(lg_state, ks)

        # Run evaluators
        if rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")
            
        for i, evaluator in enumerate(evaluators):
            if rank == 0:
                print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")
                
            # Path for saving
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            # Run and log
            metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group)
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}

                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")
                
        if rank == 0:
            print("All evaluators completed!")

    return reduced_metrics

def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype.
        # Default NCCL timeout is 10 min; FPTRM eval can have multi-minute
        # rank-to-rank divergence accumulating over many eval batches, so we
        # raise it. Override via NCCL_TIMEOUT_MIN env var if needed.
        from datetime import timedelta as _timedelta
        _nccl_timeout_min = int(os.environ.get("NCCL_TIMEOUT_MIN", "60"))
        dist.init_process_group(backend="nccl", timeout=_timedelta(minutes=_nccl_timeout_min))

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
        # CPU GLOO process group
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")
        assert (
            dist.get_rank(CPU_PROCESS_GROUP) == RANK and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
        )

    # GPU diagnostics: confirm each rank sees its own device.
    _local_rank = int(os.environ.get("LOCAL_RANK", 0))
    _dev_idx = torch.cuda.current_device() if torch.cuda.is_available() else -1
    _dev_name = torch.cuda.get_device_name(_dev_idx) if _dev_idx >= 0 else "cpu"
    _dev_uuid = "n/a"
    if _dev_idx >= 0:
        _props = torch.cuda.get_device_properties(_dev_idx)
        _u = getattr(_props, "uuid", None)
        if _u is not None:
            try:
                _dev_uuid = str(_u)
            except Exception:
                _dev_uuid = "n/a"
    print(
        f"[gpu-diag] rank={RANK}/{WORLD_SIZE} local_rank={_local_rank} "
        f"pid={os.getpid()} host={os.uname().nodename} "
        f"cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
        f"device_count={torch.cuda.device_count()} "
        f"current_device={_dev_idx} name={_dev_name} uuid={_dev_uuid}",
        flush=True,
    )
    if WORLD_SIZE > 1:
        dist.barrier()

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)
    assert not (config.resume_from and config.load_checkpoint), \
        "resume_from and load_checkpoint are mutually exclusive"

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    try:
        eval_loader,  eval_metadata  = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    except:
        print("NO EVAL DATA FOUND")
        eval_loader = eval_metadata = None

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except:
        print("No evaluator found")
        evaluators = []

    # Train state
    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)
    steps_per_epoch = max(train_state.total_steps / config.epochs, 1)
    profiler = FlopProfiler(enabled=config.profile_flops)

    # EMA setup (must precede resume so EMA shadow can be restored)
    ema_helper = None
    if config.ema:
        print('Setup EMA')
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    # Resume from bundle (all ranks read the same file)
    start_iter_id = 0
    prev_wandb_id = None
    if config.resume_from is not None:
        start_iter_id, prev_wandb_id = restore_train_state(
            resume_from=config.resume_from,
            train_state=train_state,
            ema_helper=ema_helper,
            dataset=train_loader.dataset,
        )

    # Progress bar and wandb
    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps, initial=train_state.step)
        wandb.init(
            project=config.project_name, name=config.run_name,
            id=prev_wandb_id, resume="allow" if prev_wandb_id else None,
            config=config.model_dump(),
            settings=wandb.Settings(_disable_stats=True),
        )  # type: ignore
        if config.resume_from is None:
            wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config)

    # Training Loop
    for _iter_id in range(start_iter_id, total_iters):
        print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        ############ Train Iter
        if RANK == 0:
            print("TRAIN")
        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:
            train_state.model.set_num_iters()  # type: ignore[attr-defined]  # resample max_iter per batch
            for m in train_state.model.modules():
                if hasattr(m, "reset_mask"):
                    m.reset_mask()
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE, profiler=profiler)

            if RANK == 0 and metrics is not None:
                metrics["epoch"] = train_state.step / steps_per_epoch
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
            if config.ema:
                ema_helper.update(train_state.model)

            if (
                RANK == 0
                and config.checkpoint_every_n_steps is not None
                and train_state.step > 0
                and train_state.step % config.checkpoint_every_n_steps == 0
            ):
                save_train_state(config, train_state)

        if _iter_id >= config.min_eval_interval:
            ############ Evaluation
            if RANK == 0:
                print("EVALUATE")
            if config.ema:
                print("SWITCH TO EMA")
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state
            train_state_eval.model.eval()
            metrics = evaluate(config,
                train_state_eval,
                eval_loader,
                eval_metadata,
                evaluators,
                rank=RANK,
                world_size=WORLD_SIZE,
                cpu_group=CPU_PROCESS_GROUP,
                profiler=profiler)

            if RANK == 0 and metrics is not None:
                set_keys = [k for k, v in metrics.items() if isinstance(v, dict) and k != "length_gen"]
                single_set = len(set_keys) == 1
                log = {}
                for k, v in metrics.items():
                    if k == "length_gen":
                        continue
                    if isinstance(v, dict):
                        prefix = "eval/" if single_set else f"eval/{k}/"
                        for mk, mv in v.items():
                            log[f"{prefix}{mk}"] = mv
                    else:
                        log[k] = v
                if "length_gen" in metrics:
                    log.update(length_gen_eval.to_wandb(metrics["length_gen"]))
                log["epoch"] = train_state.step / steps_per_epoch
                wandb.log(log, step=train_state.step)
                if config.metrics_out and _iter_id == total_iters - 1 and "length_gen" in metrics:
                    write_curves({config.run_name or "run": metrics["length_gen"]},
                                 config.metrics_out)
                    print(f"Wrote per-k metrics to {config.metrics_out}")

            ############ Checkpointing
            if RANK == 0:
                print("SAVE CHECKPOINT")
            if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
                save_train_state(config, train_state_eval)
                save_resume_bundle(
                    checkpoint_path=config.checkpoint_path,
                    train_state=train_state,
                    ema_helper=ema_helper,
                    dataset_iters=(_iter_id + 1) * len(train_metadata.sets),
                    iter_id=_iter_id,
                    wandb_run_id=wandb.run.id if wandb.run is not None else None,
                )

            if config.ema:
                del train_state_eval

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
