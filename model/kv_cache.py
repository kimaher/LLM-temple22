"""Static key/value cache for autoregressive decoding.

During generation every new token attends to *all* previous tokens.  Without a
cache we would recompute the keys and values for the whole prefix at every
step, which makes generation quadratic.  The cache stores them once.

The buffers are preallocated to `max_seq_len` so no reallocation happens inside
the decode loop; `pos` tracks how much of them is currently valid.
"""

from __future__ import annotations

from typing import List, Tuple

import torch


class KVCache:
    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        shape = (batch_size, n_kv_heads, max_seq_len, head_dim)
        self.k: List[torch.Tensor] = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.v: List[torch.Tensor] = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.max_seq_len = max_seq_len
        # Number of tokens already written.  Shared by every layer, and only
        # advanced by the model after the last layer has consumed it.
        self.pos = 0

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append this layer's new k/v and return the full history so far.

        k, v: (batch, n_kv_heads, new_tokens, head_dim)
        """
        new = k.shape[2]
        end = self.pos + new
        if end > self.max_seq_len:
            raise ValueError(f"KV cache overflow: {end} > max_seq_len={self.max_seq_len}")
        self.k[layer_idx][:, :, self.pos : end] = k
        self.v[layer_idx][:, :, self.pos : end] = v
        return self.k[layer_idx][:, :, :end], self.v[layer_idx][:, :, :end]

    def advance(self, n: int) -> None:
        self.pos += n

    def reset(self) -> None:
        self.pos = 0
