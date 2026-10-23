"""Utilidades mínimas para DDP con torchrun (si WORLD_SIZE no está definido, todo es de 1 proceso)."""
from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def init_dist() -> tuple[int, int, int]:
    """Devuelve (rank, world, local_rank) e inicia NCCL si hay más de un proceso."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1 and not dist.is_initialized():
        torch.cuda.set_device(local)
        # timeout largo: mientras unos procesos escriben informes/checkpoints el resto espera
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
    return rank, world, local


def device_map(world: int, local: int):
    """Un modelo completo por GPU en DDP; reparto automático entre GPUs si hay un solo proceso."""
    return {"": local} if world > 1 else "auto"


def gather_rows(rows: list[dict], world: int) -> list[dict]:
    if world == 1:
        return rows
    parts = [None] * world
    dist.all_gather_object(parts, rows)
    return [r for p in parts for r in p]


def barrier(world: int) -> None:
    if world > 1:
        dist.barrier()


def cleanup(world: int) -> None:
    if world > 1 and dist.is_initialized():
        dist.destroy_process_group()
