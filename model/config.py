"""Model configuration.

A single dataclass describes the whole architecture.  Everything downstream
(training, inference, the web server) round-trips this object through JSON so a
checkpoint is always self-describing: you never have to remember which flags a
given `.pt` file was trained with.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional


@dataclass
class ModelConfig:
    # --- vocabulary -------------------------------------------------------
    vocab_size: int = 50257  # overwritten by the tokenizer's real vocab size

    # --- transformer shape ------------------------------------------------
    n_layers: int = 8
    n_heads: int = 8
    # Number of *key/value* heads.  None => n_kv_heads == n_heads, i.e. plain
    # multi-head attention.  Setting it lower turns on grouped-query attention
    # (GQA), which shrinks the KV cache at inference time.
    n_kv_heads: Optional[int] = None
    d_model: int = 512

    # SwiGLU inner dimension.  None => the LLaMA heuristic below, which keeps
    # the parameter count of a SwiGLU block roughly equal to a vanilla
    # 4*d_model GELU block (SwiGLU has three matrices instead of two).
    ffn_hidden: Optional[int] = None
    ffn_multiple_of: int = 64  # round the heuristic up to a hardware-friendly size

    # --- sequence / regularisation ---------------------------------------
    max_seq_len: int = 512
    dropout: float = 0.0

    # --- numerical details ------------------------------------------------
    rope_theta: float = 10000.0  # RoPE base frequency
    norm_eps: float = 1e-5  # RMSNorm epsilon
    tie_embeddings: bool = True  # share the token embedding with the output head

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be even for RoPE")
        if self.ffn_hidden is None:
            self.ffn_hidden = self._default_ffn_hidden()

    # ------------------------------------------------------------------ #
    # derived quantities
    # ------------------------------------------------------------------ #
    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_rep(self) -> int:
        """How many query heads share each key/value head (1 == plain MHA)."""
        return self.n_heads // self.n_kv_heads

    def _default_ffn_hidden(self) -> int:
        hidden = int(8 * self.d_model / 3)  # 2/3 * 4 * d_model
        m = self.ffn_multiple_of
        return m * ((hidden + m - 1) // m)

    # ------------------------------------------------------------------ #
    # (de)serialisation
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown ModelConfig keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# Handy presets.  `tiny` is what every smoke test uses: it trains in seconds on
# a CPU, which is the whole point of having it.
TINY = ModelConfig(vocab_size=512, n_layers=2, n_heads=4, d_model=128, max_seq_len=128)
