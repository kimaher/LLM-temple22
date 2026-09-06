"""Autoregressive generation: sampling strategies on top of a KV-cached decode loop.

The loop is the standard two-phase one:

  * **prefill** - run the whole prompt through the model once, filling the KV
    cache and producing the first next-token distribution;
  * **decode** - feed back one token at a time.  Each step only computes
    attention for the single new query against the cached keys/values, so a step
    costs O(context) instead of O(context^2).

Everything is exposed as a generator (`generate_stream`) because that is what
the SSE endpoint needs; `generate` is a thin wrapper that drains it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence

import torch
import torch.nn.functional as F

from model.transformer import Transformer


@dataclass
class SamplingConfig:
    max_new_tokens: int = 256
    # temperature = 0 means greedy (argmax).  Otherwise logits are divided by it
    # before the softmax: <1 sharpens the distribution, >1 flattens it.
    temperature: float = 0.8
    # top-k: keep only the k most likely tokens.  0 disables.
    top_k: int = 50
    # top-p (nucleus): keep the smallest set of tokens whose probabilities sum
    # to p.  Adapts to how peaked the distribution is, where top-k does not.
    # 1.0 disables.
    top_p: float = 0.95
    # Divides the logit of tokens already in the context; >1 discourages loops.
    # A crude but effective fix for the repetition that under-trained models fall into.
    repetition_penalty: float = 1.0
    stop_ids: Optional[Sequence[int]] = None
    seed: Optional[int] = None


def apply_repetition_penalty(logits: torch.Tensor, generated: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    # Scale positive logits down and negative logits up: both push the token
    # towards being less likely (Keskar et al., CTRL).
    score = torch.gather(logits, 1, generated)
    score = torch.where(score > 0, score / penalty, score * penalty)
    return logits.scatter(1, generated, score)


def filter_logits(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Mask out tokens excluded by top-k / top-p before sampling.

    logits: (batch, vocab). Returns the same shape with -inf in filtered slots.
    """
    if top_k and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        # Drop everything after the nucleus, but always keep the top token so we
        # can never end up with an all -inf row.
        remove = cumulative - probs > top_p
        remove[..., 0] = False
        logits = logits.masked_fill(remove.scatter(1, sorted_idx, remove), float("-inf"))
    return logits


def sample_from_logits(
    logits: torch.Tensor,
    config: SamplingConfig,
    generated: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Pick the next token id from (batch, vocab) logits. Returns (batch, 1)."""
    if generated is not None and config.repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits.float(), generated, config.repetition_penalty)

    if config.temperature <= 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits.float() / config.temperature
    logits = filter_logits(logits, top_k=config.top_k, top_p=config.top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


@torch.no_grad()
def generate_stream(
    model: Transformer,
    prompt_ids: Sequence[int],
    config: Optional[SamplingConfig] = None,
    device: Optional[str] = None,
) -> Iterator[int]:
    """Yield generated token ids one at a time (batch size 1).

    Stops at `max_new_tokens`, at any id in `stop_ids`, or when the context
    window is full.
    """
    config = config or SamplingConfig()
    was_training = model.training
    model.eval()
    device = device or str(next(model.parameters()).device)

    generator = None
    if config.seed is not None:
        generator = torch.Generator(device=device).manual_seed(config.seed)

    max_len = model.config.max_seq_len
    ids = list(prompt_ids)
    if len(ids) >= max_len:
        # Keep the most recent context; the oldest tokens are the ones we can
        # most afford to lose.  (A real system would summarise instead.)
        ids = ids[-(max_len - 1) :]

    idx = torch.tensor([ids], dtype=torch.long, device=device)
    cache = model.new_cache(batch_size=1, max_seq_len=max_len)
    stop = set(config.stop_ids or [])

    try:
        # --- prefill -----------------------------------------------------
        logits, _ = model(idx, cache=cache)
        context = idx

        # --- decode ------------------------------------------------------
        for _ in range(config.max_new_tokens):
            next_id = sample_from_logits(logits[:, -1, :], config, generated=context, generator=generator)
            token = int(next_id.item())
            if token in stop:
                break
            yield token
            context = torch.cat([context, next_id], dim=1)
            if cache.pos >= max_len:
                break  # context window exhausted
            logits, _ = model(next_id, cache=cache)
    finally:
        if was_training:
            model.train()


def generate(
    model: Transformer,
    prompt_ids: Sequence[int],
    config: Optional[SamplingConfig] = None,
    device: Optional[str] = None,
) -> List[int]:
    """Non-streaming convenience wrapper: returns the list of new token ids."""
    return list(generate_stream(model, prompt_ids, config, device))
