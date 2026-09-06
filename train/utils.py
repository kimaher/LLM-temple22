"""Shared training helpers: LR schedule, precision selection, checkpointing."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from model.config import ModelConfig
from model.transformer import Transformer


# --------------------------------------------------------------------------- #
# Learning-rate schedule
# --------------------------------------------------------------------------- #
def cosine_lr(step: int, *, base_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup, then cosine decay from base_lr down to min_lr.

    Warmup exists because Adam's second-moment estimate is garbage for the first
    few steps; taking full-size steps then is how you get an early loss spike.
    Cosine decay spends most of the run at a high LR and anneals smoothly at the
    end, which reliably beats a constant LR at a fixed step budget.
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (base_lr - min_lr)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


# --------------------------------------------------------------------------- #
# Devices and precision
# --------------------------------------------------------------------------- #
def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def pick_dtype(device: str, requested: str = "auto") -> torch.dtype:
    """bf16 where supported, fp16 on older CUDA, fp32 on CPU.

    bf16 has the same exponent range as fp32, so it needs no loss scaling; fp16
    does, which is why the GradScaler below is enabled only for fp16.
    """
    if requested != "auto":
        return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[requested]
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


class AmpContext:
    """Bundles the autocast context and the (fp16-only) gradient scaler."""

    def __init__(self, device: str, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype
        self.enabled = device.startswith("cuda") and dtype != torch.float32
        self.scaler = torch.amp.GradScaler(device, enabled=(self.enabled and dtype == torch.float16))

    def autocast(self):
        device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        return torch.amp.autocast(device_type=device_type, dtype=self.dtype, enabled=self.enabled)


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def save_checkpoint(
    path: str | Path,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
    best_val_loss: float = float("inf"),
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a self-describing checkpoint (weights + the config that built them)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "config": model.config.to_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "saved_at": time.time(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload["extra"] = extra
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic: never leave a half-written checkpoint behind


def load_checkpoint(
    path: str | Path,
    device: str = "cpu",
    model: Optional[Transformer] = None,
) -> tuple[Transformer, Dict[str, Any]]:
    """Rebuild a model straight from a checkpoint (config included)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if model is None:
        model = Transformer(ModelConfig.from_dict(ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    return model, ckpt


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
@dataclass
class Timer:
    """Wall-clock stopwatch for tokens/second reporting."""

    t0: float = 0.0

    def start(self) -> None:
        self.t0 = time.perf_counter()

    def lap(self) -> float:
        now = time.perf_counter()
        dt = now - self.t0
        self.t0 = now
        return dt


def human(n: float) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000
    return f"{n:.2f}T"
