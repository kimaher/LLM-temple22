"""Tests for the transformer internals.

The interesting ones are `test_causality` (a bug in the mask is invisible in the
loss curve but destroys the model at inference time) and `test_kv_cache_matches_full_forward`
(the cache is an optimisation, so it must be numerically equivalent to the slow path).
"""

from __future__ import annotations

import math

import pytest
import torch

from model.config import ModelConfig
from model.transformer import RMSNorm, Transformer, apply_rope, build_rope_cache


@pytest.fixture
def config() -> ModelConfig:
    return ModelConfig(vocab_size=97, n_layers=2, n_heads=4, d_model=64, max_seq_len=32, dropout=0.0)


@pytest.fixture
def model(config: ModelConfig) -> Transformer:
    torch.manual_seed(0)
    return Transformer(config).eval()


def test_config_rejects_bad_shapes():
    with pytest.raises(ValueError):
        ModelConfig(d_model=100, n_heads=8)  # not divisible
    with pytest.raises(ValueError):
        ModelConfig(n_heads=8, n_kv_heads=3)  # not divisible


def test_forward_shapes(model: Transformer, config: ModelConfig):
    x = torch.randint(0, config.vocab_size, (3, 16))
    logits, loss = model(x, targets=x)
    assert logits.shape == (3, 16, config.vocab_size)
    assert loss.ndim == 0
    # Inference path returns only the last position.
    logits, loss = model(x)
    assert logits.shape == (3, 1, config.vocab_size)
    assert loss is None


def test_initial_loss_is_near_uniform(config: ModelConfig):
    """A freshly initialised model should be about as good as a coin flip."""
    torch.manual_seed(0)
    model = Transformer(config)
    x = torch.randint(0, config.vocab_size, (8, 32))
    y = torch.randint(0, config.vocab_size, (8, 32))
    _, loss = model(x, targets=y)
    assert abs(loss.item() - math.log(config.vocab_size)) < 0.3


def test_causality(model: Transformer, config: ModelConfig):
    """Changing a token must not affect the logits at any earlier position."""
    x = torch.randint(0, config.vocab_size, (1, 12))
    base, _ = model(x, targets=x)

    changed = x.clone()
    changed[0, 7] = (changed[0, 7] + 1) % config.vocab_size
    after, _ = model(changed, targets=changed)

    assert torch.allclose(base[:, :7], after[:, :7], atol=1e-5)
    assert not torch.allclose(base[:, 7], after[:, 7], atol=1e-5)


def test_kv_cache_matches_full_forward(model: Transformer, config: ModelConfig):
    """Token-by-token decoding with the cache == one big forward pass."""
    x = torch.randint(0, config.vocab_size, (1, 10))
    full, _ = model(x, targets=x)

    cache = model.new_cache(batch_size=1)
    stepwise = []
    for t in range(x.shape[1]):
        logits, _ = model(x[:, t : t + 1], cache=cache)
        stepwise.append(logits[:, -1, :])
    stepwise = torch.stack(stepwise, dim=1)

    assert torch.allclose(full, stepwise, atol=1e-4)


def test_prefill_then_decode_matches(model: Transformer, config: ModelConfig):
    """Chunked prefill (prompt in one pass, then one token at a time) is equivalent."""
    x = torch.randint(0, config.vocab_size, (1, 10))
    full, _ = model(x, targets=x)

    cache = model.new_cache(batch_size=1)
    model(x[:, :6], cache=cache)
    last, _ = model(x[:, 6:], cache=cache)
    assert torch.allclose(full[:, -1:], last, atol=1e-4)


def test_cache_overflow_raises(model: Transformer, config: ModelConfig):
    cache = model.new_cache(batch_size=1, max_seq_len=8)
    with pytest.raises(ValueError):
        model(torch.randint(0, config.vocab_size, (1, 9)), cache=cache)


def test_rmsnorm_normalises():
    norm = RMSNorm(16)
    x = torch.randn(4, 16) * 7 + 3
    y = norm(x)
    rms = y.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rope_preserves_norm_and_relative_position():
    """RoPE is a rotation: it preserves lengths, and inner products depend only
    on the distance between positions."""
    head_dim, seq = 16, 8
    cos, sin = build_rope_cache(head_dim, seq)
    x = torch.randn(1, 1, seq, head_dim)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)

    # The same vector at positions (2,5) and (4,7) - equal distance - should
    # produce the same dot product.
    v = torch.randn(head_dim)
    q = v.view(1, 1, 1, -1)
    dot = lambda i, j: (  # noqa: E731
        apply_rope(q, cos[i : i + 1], sin[i : i + 1]) * apply_rope(q, cos[j : j + 1], sin[j : j + 1])
    ).sum()
    assert torch.allclose(dot(2, 5), dot(4, 7), atol=1e-5)


def test_weight_tying(config: ModelConfig):
    tied = Transformer(config)
    assert tied.lm_head.weight is tied.tok_emb.weight
    untied = Transformer(ModelConfig(**{**config.to_dict(), "tie_embeddings": False}))
    assert untied.lm_head.weight is not untied.tok_emb.weight
    assert untied.num_params() > tied.num_params()


def test_gqa_runs_and_shrinks_the_cache(config: ModelConfig):
    gqa = ModelConfig(**{**config.to_dict(), "n_kv_heads": 1})
    model = Transformer(gqa).eval()
    x = torch.randint(0, gqa.vocab_size, (2, 6))
    logits, _ = model(x)
    assert logits.shape == (2, 1, gqa.vocab_size)
    assert model.new_cache(1).k[0].shape[1] == 1  # one kv head, not four


def test_loss_mask_ignores_masked_positions(model: Transformer, config: ModelConfig):
    x = torch.randint(0, config.vocab_size, (2, 8))
    mask = torch.zeros(2, 8)
    mask[:, -1] = 1.0  # score only the final position
    _, masked = model(x, targets=x, loss_mask=mask)
    _, full = model(x, targets=x)
    assert not torch.allclose(masked, full)
    assert masked.item() > 0


def test_one_optimizer_step_reduces_loss_on_a_fixed_batch(config: ModelConfig):
    """The smallest possible end-to-end training assertion."""
    torch.manual_seed(0)
    model = Transformer(config)
    opt = model.configure_optimizers(lr=1e-2, weight_decay=0.0)
    x = torch.randint(0, config.vocab_size, (4, 16))
    y = torch.randint(0, config.vocab_size, (4, 16))

    _, before = model(x, targets=y)
    for _ in range(5):
        opt.zero_grad()
        _, loss = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    _, after = model(x, targets=y)
    assert after.item() < before.item()
