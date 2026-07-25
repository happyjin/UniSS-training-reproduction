"""Small torchrun helpers shared by Simul-UniSS non-Megatron stages."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TypeVar

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


ModuleT = TypeVar("ModuleT", bound=nn.Module)


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @classmethod
    def initialize(cls, requested_device: str) -> "DistributedContext":
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        enabled = world_size > 1
        wants_cuda = requested_device.startswith("cuda")
        if enabled:
            if wants_cuda:
                if not torch.cuda.is_available():
                    raise RuntimeError("torchrun requested CUDA but CUDA is unavailable")
                device = torch.device("cuda", local_rank)
                torch.cuda.set_device(device)
                backend = "nccl"
            else:
                device = torch.device(requested_device)
                backend = "gloo"
            dist.init_process_group(backend=backend, init_method="env://")
        else:
            device = torch.device(requested_device)
            if device.type == "cuda" and device.index is not None:
                torch.cuda.set_device(device)
        return cls(enabled, rank, local_rank, world_size, device)

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def wrap(self, model: ModuleT) -> nn.Module:
        if not self.enabled:
            return model
        device_ids = [self.local_rank] if self.device.type == "cuda" else None
        output_device = self.local_rank if self.device.type == "cuda" else None
        return DistributedDataParallel(
            model,
            device_ids=device_ids,
            output_device=output_device,
        )

    @staticmethod
    def unwrap(model: nn.Module) -> nn.Module:
        return model.module if isinstance(model, DistributedDataParallel) else model

    def reduce_sums(self, values: list[float]) -> list[float]:
        tensor = torch.tensor(values, dtype=torch.float64, device=self.device)
        if self.enabled:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.cpu().tolist()

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def close(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()

