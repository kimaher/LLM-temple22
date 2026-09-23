"""A decoder-only transformer, written from scratch in plain PyTorch.

Architecture (the modern "LLaMA-style" recipe):
  * pre-norm residual blocks with RMSNorm
  * rotary position embeddings (RoPE) applied to queries and keys
  * causal multi-head self-attention (optionally grouped-query)
  * SwiGLU feed-forward network
  * optional weight tying between the embedding and the output head

Nothing here comes from `transformers`; the only external pieces are
`torch.nn.functional.scaled_dot_product_attention` (a fused kernel that is
mathematically identical to the four lines it replaces, see the comment at the
call site) and standard `nn.Linear` / `nn.Embedding` containers.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .kv_cache import KVCache


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    """Root-mean-square layer norm.

    Like LayerNorm but without the mean subtraction and without a bias: only a
    learned per-channel gain.  Cheaper, and empirically just as good for LMs.

        y = x / sqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise in fp32 for stability, then cast back to the input dtype so
        # this stays cheap under autocast / bf16 training.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


# --------------------------------------------------------------------------- #
# Rotary position embeddings (RoPE)
# --------------------------------------------------------------------------- #
def build_rope_cache(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | str | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute the cos/sin tables used by RoPE.

    Each pair of channels in a head is treated as a 2-D vector and rotated by an
    angle proportional to the token's absolute position.  A dot product between
    two rotated vectors depends only on the *difference* of their angles, so
    attention scores end up depending on relative position - which is what we
    want, and it extrapolates better than learned absolute embeddings.

    Returns two tensors of shape (max_seq_len, head_dim).
    """
    # inv_freq[i] = 1 / theta^(2i/head_dim) for i in [0, head_dim/2)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (max_seq_len, head_dim/2)
    # Duplicated so the table lines up with the "rotate half" split below.
    emb = torch.cat((freqs, freqs), dim=-1)  # (max_seq_len, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(x1, x2) -> (-x2, x1): a 90 degree rotation of each channel pair."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate x by the angles in cos/sin.

    x:        (batch, heads, seq, head_dim)
    cos, sin: (seq, head_dim), already sliced to the right absolute positions
    """
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return x * cos + _rotate_half(x) * sin


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand key/value heads so every query head has a partner (GQA).

    (B, n_kv_heads, T, D) -> (B, n_kv_heads * n_rep, T, D)
    """
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = config.n_rep

        # Separate projections rather than one fused qkv matrix: with GQA the q
        # and kv projections have different output sizes anyway.
        self.wq = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)
        self.dropout_p = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: Optional[KVCache] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        # Project, then split into heads: (B, heads, T, head_dim)
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Positional information enters here, on q and k only (never on v).
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # softmax(q @ k^T / sqrt(d) + mask) @ v, fused.  `is_causal=True` is only
        # correct when the queries and keys are the same tokens (the no-cache
        # training path).  With a cache we pass an explicit mask built by the
        # model, and for single-token decoding no mask is needed at all: the new
        # token is allowed to attend to everything already in the cache.
        is_causal = attn_mask is None and cache is None
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.resid_dropout(self.wo(y))


# --------------------------------------------------------------------------- #
# Feed-forward
# --------------------------------------------------------------------------- #
class SwiGLU(nn.Module):
    """Gated feed-forward block: down(silu(gate(x)) * up(x)).

    The elementwise gate lets the network suppress channels multiplicatively,
    which works better than a plain ReLU/GELU MLP at equal parameter count.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.ffn_hidden
        self.w_gate = nn.Linear(config.d_model, hidden, bias=False)
        self.w_up = nn.Linear(config.d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


# --------------------------------------------------------------------------- #
# Block
# --------------------------------------------------------------------------- #
class Block(nn.Module):
    """Pre-norm residual block: x + attn(norm(x)), then x + ffn(norm(x)).

    Pre-norm (normalising the *input* of each sublayer rather than its output)
    keeps a clean identity path from the embeddings to the logits, which is what
    lets deep transformers train stably.
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = CausalSelfAttention(config, layer_idx)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: Optional[KVCache] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, cache=cache, attn_mask=attn_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class Transformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            # One matrix, used both to look tokens up and to score them.  Saves
            # vocab_size * d_model parameters and usually helps small models.
            self.lm_head.weight = self.tok_emb.weight

        # The RoPE tables are identical for every layer, so build them once and
        # carry them with the module (non-persistent: derived, not learned).
        cos, sin = build_rope_cache(config.head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2 style scaled init: the residual stream accumulates 2*n_layers
        # sublayer outputs, so shrink the projections that write into it to keep
        # the variance of that stream roughly constant with depth.
        std = 0.02 / math.sqrt(2 * config.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------ #
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if not self.config.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def _causal_mask(self, T: int, start_pos: int, device: torch.device) -> Optional[torch.Tensor]:
        """Mask for the cached case when we feed more than one token at a time.

        Query i (absolute position start_pos + i) may attend to key j whenever
        j <= start_pos + i.  Returns None when no mask is needed.
        """
        if T == 1:
            return None  # a single new token may attend to the whole cache
        k_len = start_pos + T
        q_idx = torch.arange(start_pos, k_len, device=device)[:, None]
        k_idx = torch.arange(k_len, device=device)[None, :]
        return (k_idx <= q_idx)[None, None, :, :]  # broadcasts over batch/heads

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        idx:       (B, T) int64 token ids
        targets:   (B, T) int64 next-token labels; -100 entries are ignored
        cache:     KVCache for incremental decoding (its `pos` gives the offset)
        loss_mask: optional (B, T) mask, used by SFT to train only on response
                   tokens

        Returns (logits, loss); `loss` is None when no targets are given.
        """
        B, T = idx.shape
        start_pos = cache.pos if cache is not None else 0
        if start_pos + T > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {start_pos + T} exceeds max_seq_len={self.config.max_seq_len}"
            )

        x = self.emb_dropout(self.tok_emb(idx))

        # Slice the RoPE tables at the *absolute* positions of these tokens.
        cos = self.rope_cos[start_pos : start_pos + T]
        sin = self.rope_sin[start_pos : start_pos + T]

        attn_mask = self._causal_mask(T, start_pos, idx.device) if cache is not None else None

        for block in self.blocks:
            x = block(x, cos, sin, cache=cache, attn_mask=attn_mask)
        x = self.norm(x)

        if cache is not None:
            cache.advance(T)

        if targets is None:
            # Inference: only the last position matters, so skip the rest of the
            # (vocab_size-wide, therefore expensive) output projection.
            logits = self.lm_head(x[:, -1:, :])
            return logits, None

        logits = self.lm_head(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="none",
        )
        if loss_mask is not None:
            m = loss_mask.reshape(-1).to(loss.dtype)
        else:
            m = (targets.reshape(-1) != -100).to(loss.dtype)
        loss = (loss * m).sum() / m.sum().clamp(min=1.0)
        return logits, loss

    # ------------------------------------------------------------------ #
    def configure_optimizers(
        self,
        lr: float,
        weight_decay: float,
        betas: Tuple[float, float] = (0.9, 0.95),
        device: str = "cpu",
    ) -> torch.optim.Optimizer:
        """AdamW with weight decay on matrices only.

        Biases and 1-D gains (the RMSNorm weights) are excluded: shrinking them
        just fights the normalisation they exist to provide.
        """
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        extra = {"fused": True} if "cuda" in str(device) else {}
        return torch.optim.AdamW(groups, lr=lr, betas=betas, **extra)

    def new_cache(self, batch_size: int, max_seq_len: Optional[int] = None) -> KVCache:
        """Allocate a KV cache matching this model's device/dtype."""
        p = next(self.parameters())
        return KVCache(
            n_layers=self.config.n_layers,
            batch_size=batch_size,
            max_seq_len=max_seq_len or self.config.max_seq_len,
            n_kv_heads=self.config.n_kv_heads,
            head_dim=self.config.head_dim,
            device=p.device,
            dtype=p.dtype,
        )
