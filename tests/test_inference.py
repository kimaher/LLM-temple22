"""Tests for sampling, streaming detokenisation, and SFT batching."""

from __future__ import annotations

import json

import pytest
import torch

from data.sft_dataset import SFTDataset
from inference.generate import (
    SamplingConfig,
    apply_repetition_penalty,
    filter_logits,
    generate,
    generate_stream,
)
from inference.stream import TokenStreamDecoder
from model.config import ModelConfig
from model.transformer import Transformer
from tokenizer.bpe import BPETokenizer
from tokenizer.tokenizer import SPECIAL_TOKENS, CustomBPETokenizer


@pytest.fixture(scope="module")
def tokenizer() -> CustomBPETokenizer:
    corpus = "hello world, the cat sat on the mat. " * 50
    return CustomBPETokenizer(BPETokenizer.train(corpus, vocab_size=400, special_tokens=SPECIAL_TOKENS))


@pytest.fixture(scope="module")
def model(tokenizer: CustomBPETokenizer) -> Transformer:
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size, n_layers=2, n_heads=4, d_model=64, max_seq_len=64
    )
    return Transformer(config).eval()


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def test_top_k_keeps_exactly_k_candidates():
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
    filtered = filter_logits(logits, top_k=2)
    assert torch.isfinite(filtered).sum().item() == 2


def test_top_p_keeps_the_nucleus():
    # probabilities ~ [0.64, 0.24, 0.09, 0.03] -> p=0.9 keeps the first three
    logits = torch.log(torch.tensor([[0.64, 0.24, 0.09, 0.03]]))
    filtered = filter_logits(logits, top_p=0.9)
    assert torch.isfinite(filtered).sum().item() == 3


def test_top_p_always_keeps_at_least_one_token():
    logits = torch.log(torch.tensor([[0.99, 0.005, 0.005]]))
    filtered = filter_logits(logits, top_p=0.1)
    assert torch.isfinite(filtered).sum().item() == 1


def test_greedy_is_deterministic(model: Transformer):
    cfg = SamplingConfig(max_new_tokens=10, temperature=0.0)
    assert generate(model, [1, 2, 3], cfg) == generate(model, [1, 2, 3], cfg)


def test_seeded_sampling_is_reproducible(model: Transformer):
    cfg = SamplingConfig(max_new_tokens=10, temperature=1.0, top_k=20, seed=42)
    assert generate(model, [1, 2, 3], cfg) == generate(model, [1, 2, 3], cfg)


def test_generation_respects_max_new_tokens(model: Transformer):
    out = generate(model, [1, 2, 3], SamplingConfig(max_new_tokens=7, temperature=0.9, seed=0))
    assert len(out) == 7


def test_generation_stops_at_stop_token(model: Transformer):
    """Force a stop by making every token a stop token."""
    stop = list(range(model.config.vocab_size))
    out = generate(model, [1, 2, 3], SamplingConfig(max_new_tokens=10, temperature=0.0, stop_ids=stop))
    assert out == []


def test_generation_does_not_exceed_context_window(model: Transformer):
    prompt = list(range(1, model.config.max_seq_len + 20))  # longer than the window
    out = generate(model, prompt, SamplingConfig(max_new_tokens=200, temperature=0.0))
    assert len(out) <= model.config.max_seq_len


def test_repetition_penalty_pushes_seen_tokens_down():
    """Positive logits are divided, negative ones multiplied: both mean 'less likely'."""
    logits = torch.tensor([[3.0, -2.0, 0.5]])
    seen = torch.tensor([[0, 1]])
    out = apply_repetition_penalty(logits.clone(), seen, penalty=2.0)
    assert out[0, 0].item() == pytest.approx(1.5)  # 3.0 / 2
    assert out[0, 1].item() == pytest.approx(-4.0)  # -2.0 * 2
    assert out[0, 2].item() == pytest.approx(0.5)  # untouched: not in the context


def test_repetition_penalty_breaks_a_greedy_loop(model: Transformer):
    """An untrained model repeats one token forever; a strong penalty stops that."""
    plain = generate(model, [1, 2, 3], SamplingConfig(max_new_tokens=12, temperature=0.0))
    assert len(set(plain)) == 1  # the degenerate loop we are trying to break
    penalised = generate(
        model, [1, 2, 3], SamplingConfig(max_new_tokens=12, temperature=0.0, repetition_penalty=3.0)
    )
    assert len(set(penalised)) > 1


def test_streaming_yields_incrementally(model: Transformer):
    stream = generate_stream(model, [1, 2, 3], SamplingConfig(max_new_tokens=5, temperature=0.0))
    first = next(stream)
    assert isinstance(first, int)
    stream.close()


def test_model_returns_to_training_mode(model: Transformer):
    model.train()
    generate(model, [1, 2], SamplingConfig(max_new_tokens=2, temperature=0.0))
    assert model.training
    model.eval()


# --------------------------------------------------------------------------- #
# streaming detokenisation
# --------------------------------------------------------------------------- #
def test_stream_decoder_reassembles_split_characters(tokenizer: CustomBPETokenizer):
    """An emoji spans several byte tokens; the decoder must not emit garbage."""
    text = "hi 🙂 there"
    ids = tokenizer.encode(text, allowed_special=False)
    decoder = TokenStreamDecoder(tokenizer)
    out = "".join(decoder.push(i) for i in ids) + decoder.flush()
    assert out == text
    assert "�" not in out


# --------------------------------------------------------------------------- #
# SFT data
# --------------------------------------------------------------------------- #
def test_sft_masks_everything_but_the_assistant(tokenizer: CustomBPETokenizer, tmp_path):
    path = tmp_path / "sft.jsonl"
    path.write_text(
        json.dumps(
            {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]}
        )
        + "\n",
        encoding="utf-8",
    )
    ds = SFTDataset(path, tokenizer, max_seq_len=64)
    ids, mask = ds.examples[0]

    assert len(ids) == len(mask)
    assert sum(mask) > 0
    assert mask[0] == 0  # <|bos|>
    # The supervised span must be a suffix: the reply plus its <|eot|>.
    first_supervised = mask.index(1)
    assert all(m == 1 for m in mask[first_supervised:])
    assert ids[-1] == tokenizer.eot_id and mask[-1] == 1


def test_sft_accepts_instruction_format(tokenizer: CustomBPETokenizer, tmp_path):
    path = tmp_path / "sft.jsonl"
    path.write_text(
        json.dumps({"instruction": "Say hi", "input": "", "response": "hi"}) + "\n", encoding="utf-8"
    )
    ds = SFTDataset(path, tokenizer, max_seq_len=64)
    assert len(ds) == 1


def test_sft_batch_shapes_and_padding(tokenizer: CustomBPETokenizer, tmp_path):
    path = tmp_path / "sft.jsonl"
    lines = [
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "hello " * (i + 1)},
                    {"role": "assistant", "content": "world " * (i + 1)},
                ]
            }
        )
        for i in range(4)
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    ds = SFTDataset(path, tokenizer, max_seq_len=64)
    x, y, m = ds.get_batch(4)
    assert x.shape == y.shape == m.shape
    # Padded slots are excluded from the loss.
    assert ((y == -100) & (m != 0)).sum().item() == 0


def test_sft_split(tokenizer: CustomBPETokenizer, tmp_path):
    path = tmp_path / "sft.jsonl"
    lines = [
        json.dumps({"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": "a"}]})
        for i in range(10)
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    train, val = SFTDataset(path, tokenizer, max_seq_len=64).split(0.2)
    assert len(train) == 8 and val is not None and len(val) == 2
