from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Iterable, Optional, Sequence, Tuple, Type

import torch
import torch.distributed as dist


@dataclass
class DistEnv:
    mode: str
    rank: int
    local_rank: int
    world_size: int
    master: bool
    distributed: bool


def default_seq_len_for_profile(profile: str) -> int:
    table = {
        "d12": 65536,
        "d24": 65536,
        "d36": 32768,
        "d48": 24576,
    }
    return table.get(profile, 65536)


def fallback_seq_len_for_profile(profile: str, current: int) -> int:
    ladder = {
        "d36": [32768, 24576, 16384],
        "d48": [24576, 16384],
    }
    vals = ladder.get(profile, [])
    for v in vals:
        if v < current:
            return v
    return current


def resolve_parallel_mode(parallel_mode: str, world_size: int, fsdp_default: bool = True) -> str:
    if parallel_mode not in {"auto", "single", "ddp", "fsdp"}:
        raise ValueError(f"Invalid parallel_mode={parallel_mode}")
    if parallel_mode == "auto":
        if world_size <= 1:
            return "single"
        return "fsdp" if fsdp_default else "ddp"
    if parallel_mode == "single" and world_size > 1:
        return "ddp"
    return parallel_mode


def init_distributed(parallel_mode: str = "auto", fsdp_default: bool = True) -> DistEnv:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    resolved_mode = resolve_parallel_mode(parallel_mode, world_size, fsdp_default=fsdp_default)
    use_dist = world_size > 1 and resolved_mode in {"ddp", "fsdp"}
    if use_dist and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return DistEnv(
        mode=resolved_mode,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        master=(rank == 0),
        distributed=use_dist,
    )


def destroy_distributed(env: DistEnv) -> None:
    if env.distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def rank0_only(env: DistEnv, fn, *args, **kwargs):
    if env.master:
        return fn(*args, **kwargs)
    return None


def resolve_compile_enabled(compile_mode: str, model_profile: str, world_size: int) -> bool:
    if compile_mode not in {"auto", "on", "off"}:
        raise ValueError(f"Invalid compile_mode={compile_mode}")
    if compile_mode == "on":
        return True
    if compile_mode == "off":
        return False
    if world_size > 1:
        return False
    return model_profile not in {"d36", "d48"}


def resolve_loss_fp32_enabled(loss_fp32: str, model_profile: str) -> bool:
    if loss_fp32 not in {"auto", "on", "off"}:
        raise ValueError(f"Invalid loss_fp32={loss_fp32}")
    if loss_fp32 == "on":
        return True
    if loss_fp32 == "off":
        return False
    return model_profile in {"d12", "d24"}


def filter_global_batch_sizes(
    batch_sizes: Sequence[int], device_batch_size: int, world_size: int
) -> Tuple[list[int], list[int]]:
    valid: list[int] = []
    dropped: list[int] = []
    denom = max(1, int(device_batch_size) * max(1, int(world_size)))
    for bs in batch_sizes:
        if int(bs) % denom == 0:
            valid.append(int(bs))
        else:
            dropped.append(int(bs))
    return valid, dropped


def all_reduce_mean_inplace(tensor: torch.Tensor, env: DistEnv) -> torch.Tensor:
    if env.distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= env.world_size
    return tensor


def dist_broadcast_object(obj: Any, env: DistEnv, src: int = 0) -> Any:
    if not env.distributed:
        return obj
    payload = [obj]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def dist_barrier(env: DistEnv) -> None:
    if env.distributed:
        dist.barrier()


def promote_scalar_parameters_for_fsdp_(model: torch.nn.Module) -> list[str]:
    """FSDP in some torch versions rejects 0-dim parameters; promote them to shape [1]."""
    promoted: list[str] = []
    for module_name, module in model.named_modules():
        for param_name, param in list(module._parameters.items()):
            if param is None or param.ndim != 0:
                continue
            new_param = torch.nn.Parameter(param.detach().reshape(1), requires_grad=param.requires_grad)
            module._parameters[param_name] = new_param
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            promoted.append(full_name)
    return promoted


def wrap_model_fsdp(
    model: torch.nn.Module,
    env: DistEnv,
    block_types: Optional[Iterable[Type[torch.nn.Module]]] = None,
    activation_checkpointing: str = "none",
) -> torch.nn.Module:
    if env.mode != "fsdp":
        return model

    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )

    promoted = promote_scalar_parameters_for_fsdp_(model)
    if promoted and env.master:
        shown = ", ".join(promoted[:8])
        extra = "" if len(promoted) <= 8 else f", ... (+{len(promoted) - 8} more)"
        print(f"[INFO] Promoted scalar params for FSDP compatibility: {shown}{extra}")

    auto_wrap_policy = None
    block_type_tuple: Tuple[Type[torch.nn.Module], ...] = tuple(block_types or ())
    if block_type_tuple:
        def _policy(module, recurse, nonwrapped_numel):
            if recurse:
                return True
            return isinstance(module, block_type_tuple)
        auto_wrap_policy = _policy

    if activation_checkpointing == "block" and block_type_tuple:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )
        check_fn = lambda m: isinstance(m, block_type_tuple)
        wrapper = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=wrapper, check_fn=check_fn)

    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp,
        use_orig_params=True,
        limit_all_gathers=True,
        sync_module_states=True,
        device_id=torch.device("cuda", env.local_rank),
    )
