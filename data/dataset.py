"""Batching for pretraining, straight off memory-mapped token shards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


class TokenShardDataset:
    """Random fixed-length windows over a directory of .bin token shards.

    Deliberately *not* a torch Dataset/DataLoader: for LM pretraining there is
    no epoch and no shuffling of examples, just "give me B random windows of
    length T from the stream".  Sampling offsets directly is simpler, has no
    worker processes to babysit, and memory-maps so the corpus never has to fit
    in RAM.
    """

    def __init__(self, data_dir: str | Path, split: str = "train", seq_len: int = 256) -> None:
        self.dir = Path(data_dir)
        self.split = split
        self.seq_len = seq_len

        meta_path = self.dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} not found - run `python -m data.prepare` first")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.dtype = np.dtype(self.meta["dtype"])
        self.vocab_size: int = self.meta["vocab_size"]

        paths = sorted(self.dir.glob(f"{split}_*.bin"))
        if not paths:
            raise FileNotFoundError(f"no {split}_*.bin shards in {self.dir}")
        self.shards = [np.memmap(p, dtype=self.dtype, mode="r") for p in paths]
        self.shard_lens = [len(s) for s in self.shards]
        self.total_tokens = sum(self.shard_lens)
        # Sample a shard in proportion to its length so every token is equally
        # likely, even when the last shard is a short tail.
        usable = np.array([max(0, n - seq_len - 1) for n in self.shard_lens], dtype=np.float64)
        if usable.sum() <= 0:
            raise ValueError(
                f"every {split} shard is shorter than seq_len+1={seq_len + 1}; "
                f"use a smaller --seq-len or more data"
            )
        self.shard_probs = usable / usable.sum()

    def __len__(self) -> int:
        return self.total_tokens

    def get_batch(
        self,
        batch_size: int,
        device: str = "cpu",
        generator: np.random.Generator | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (x, y) of shape (batch_size, seq_len); y is x shifted by one."""
        rng = generator or np.random
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        shard_ids = (
            rng.choice(len(self.shards), size=batch_size, p=self.shard_probs)
            if len(self.shards) > 1
            else np.zeros(batch_size, dtype=int)
        )
        for si in shard_ids:
            shard = self.shards[int(si)]
            hi = len(shard) - self.seq_len - 1
            start = int(rng.integers(0, hi)) if hasattr(rng, "integers") else int(rng.randint(0, hi))
            window = np.asarray(shard[start : start + self.seq_len + 1], dtype=np.int64)
            xs.append(window[:-1])
            ys.append(window[1:])

        x = torch.from_numpy(np.stack(xs))
        y = torch.from_numpy(np.stack(ys))
        if device.startswith("cuda"):
            # pin_memory + non_blocking overlaps the host->device copy with
            # compute; on CPU it is a no-op we skip.
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
