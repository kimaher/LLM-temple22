"""Tokenizer tests: round-tripping, special tokens, and the chat template."""

from __future__ import annotations

import pytest

from tokenizer.bpe import BPETokenizer
from tokenizer.tokenizer import ASSISTANT, EOT, SPECIAL_TOKENS, USER, CustomBPETokenizer

CORPUS = ("the cat sat on the mat. " * 40) + ("a quick brown fox jumps over the lazy dog. " * 40)


@pytest.fixture(scope="module")
def bpe() -> BPETokenizer:
    return BPETokenizer.train(CORPUS, vocab_size=400, special_tokens=SPECIAL_TOKENS)


def test_roundtrip_ascii(bpe: BPETokenizer):
    assert bpe.decode(bpe.encode(CORPUS)) == CORPUS


def test_roundtrip_unicode(bpe: BPETokenizer):
    """Byte-level BPE has no <unk>: anything encodable in UTF-8 survives."""
    text = "café — 日本語 🙂 \t newline\n"
    assert bpe.decode(bpe.encode(text)) == text


def test_merges_actually_compress(bpe: BPETokenizer):
    ids = bpe.encode(CORPUS)
    assert len(ids) < len(CORPUS.encode("utf-8")) / 2


def test_special_tokens_are_single_ids(bpe: BPETokenizer):
    ids = bpe.encode(f"{USER}hello{EOT}")
    assert ids[0] == bpe.special_tokens[USER]
    assert ids[-1] == bpe.special_tokens[EOT]


def test_special_tokens_are_not_forgeable(bpe: BPETokenizer):
    """User text must never produce a control token, or a user could fake a turn."""
    ids = bpe.encode(f"{USER}hello", allowed_special=False)
    assert bpe.special_tokens[USER] not in ids
    assert bpe.decode(ids) == f"{USER}hello"


def test_vocab_size_matches_ids(bpe: BPETokenizer):
    assert max(bpe.encode(CORPUS)) < bpe.vocab_size
    assert bpe.vocab_size == 256 + len(bpe.merges) + len(bpe.special_tokens)


def test_save_and_load(bpe: BPETokenizer, tmp_path):
    path = tmp_path / "bpe.json"
    bpe.save(path)
    other = BPETokenizer.load(path)
    assert other.encode(CORPUS) == bpe.encode(CORPUS)
    assert other.vocab_size == bpe.vocab_size


def test_train_rejects_impossible_vocab_size():
    with pytest.raises(ValueError):
        BPETokenizer.train("hello", vocab_size=10)


def test_chat_template_layout(bpe: BPETokenizer):
    tok = CustomBPETokenizer(bpe)
    ids = tok.render_chat(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        add_generation_prompt=False,
    )
    assert ids[0] == tok.bos_id
    assert ids[1] == tok.special_tokens[USER]
    assert ids[-1] == tok.eot_id
    assert tok.special_tokens[ASSISTANT] in ids


def test_generation_prompt_ends_with_assistant_marker(bpe: BPETokenizer):
    tok = CustomBPETokenizer(bpe)
    ids = tok.render_chat([{"role": "user", "content": "hi"}], add_generation_prompt=True)
    assert ids[-1] == tok.special_tokens[ASSISTANT]


def test_missing_special_tokens_rejected():
    plain = BPETokenizer.train(CORPUS, vocab_size=300)
    with pytest.raises(ValueError):
        CustomBPETokenizer(plain)
