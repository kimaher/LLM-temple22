"""Clean a raw corpus, tokenize it, and shard it into flat binary token files.

    python -m data.prepare --input data/raw/tinyshakespeare.txt --tokenizer bpe:tokenizer/artifacts/bpe.json

Output layout (data/processed/<name>/):
    train_000000.bin   uint16/uint32 token ids, no headers, no padding
    val_000000.bin
    meta.json          dtype, token counts, vocab size, tokenizer spec

Why a flat binary blob rather than, say, one example per row: language-model
pretraining doesn't have "examples".  It has one long stream of tokens that we
slice into fixed-length windows at random offsets.  A flat file memory-maps
cleanly, so the OS page cache does the data loading and we never hold the corpus
in RAM.

Documents are separated by the `<|eot|>` token so the model learns where a
document ends instead of hallucinating across boundaries.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterator, List

import numpy as np

from tokenizer.tokenizer import load_tokenizer

PROCESSED_DIR = Path("data/processed")

# Zero-width and other invisible characters that survive most copy-paste
# pipelines and would otherwise become their own tokens.
_INVISIBLE = re.compile(r"[​-‏‪-‮﻿]")
_MANY_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_SPACE = re.compile(r"[ \t]+\n")


def clean_text(text: str) -> str:
    """Light normalisation only.

    Aggressive cleaning (lowercasing, stripping punctuation) throws away signal
    the model should learn.  We only remove things that are invisible to a
    reader but expensive in tokens.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INVISIBLE.sub("", text)
    text = _TRAILING_SPACE.sub("\n", text)
    text = _MANY_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def split_documents(text: str, doc_sep: str, min_chars: int) -> Iterator[str]:
    """Split the corpus into documents and drop the near-empty ones."""
    if not doc_sep:
        yield text
        return
    for doc in text.split(doc_sep):
        doc = doc.strip()
        if len(doc) >= min_chars:
            yield doc


class ShardWriter:
    """Buffers token ids and flushes fixed-size .bin shards to disk."""

    def __init__(self, out_dir: Path, split: str, dtype: np.dtype, shard_tokens: int) -> None:
        self.out_dir = out_dir
        self.split = split
        self.dtype = dtype
        self.shard_tokens = shard_tokens
        self.buffer: List[int] = []
        self.shard_index = 0
        self.total = 0

    def add(self, ids: List[int]) -> None:
        self.buffer.extend(ids)
        while len(self.buffer) >= self.shard_tokens:
            self._flush(self.buffer[: self.shard_tokens])
            self.buffer = self.buffer[self.shard_tokens :]

    def close(self) -> None:
        if self.buffer:
            self._flush(self.buffer)
            self.buffer = []

    def _flush(self, ids: List[int]) -> None:
        path = self.out_dir / f"{self.split}_{self.shard_index:06d}.bin"
        np.array(ids, dtype=self.dtype).tofile(path)
        self.total += len(ids)
        self.shard_index += 1
        print(f"  wrote {path.name}: {len(ids):,} tokens")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="raw .txt file")
    ap.add_argument("--name", default=None, help="output dir name (default: input stem)")
    ap.add_argument("--tokenizer", default=None, help="tokenizer spec, e.g. bpe:path.json or tiktoken:gpt2")
    ap.add_argument("--doc-sep", default="\n\n", help="document separator; empty string = one document")
    ap.add_argument("--min-chars", type=int, default=1, help="drop documents shorter than this")
    ap.add_argument("--val-fraction", type=float, default=0.01)
    ap.add_argument("--shard-tokens", type=int, default=10_000_000)
    ap.add_argument("--limit-docs", type=int, default=0, help="dev mode: only process the first N documents")
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    name = args.name or Path(args.input).stem
    out_dir = PROCESSED_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.bin"):
        old.unlink()

    # uint16 covers vocabularies up to 65535 and halves the file size; anything
    # bigger (e.g. tiktoken's 50k+specials is still fine, but be safe) uses uint32.
    dtype = np.uint16 if tok.vocab_size < 2**16 else np.uint32

    text = clean_text(Path(args.input).read_text(encoding="utf-8", errors="replace"))
    docs = list(split_documents(text, args.doc_sep, args.min_chars))
    if args.limit_docs:
        docs = docs[: args.limit_docs]
    n_val = max(1, int(len(docs) * args.val_fraction)) if len(docs) > 1 else 0
    # Hold out a contiguous tail rather than a random sample: with a random
    # sample, near-duplicate neighbouring documents leak between the splits.
    train_docs, val_docs = docs[: len(docs) - n_val], docs[len(docs) - n_val :]
    print(f"{len(docs):,} documents -> {len(train_docs):,} train / {len(val_docs):,} val")

    counts = {}
    for split, split_docs in (("train", train_docs), ("val", val_docs)): # this thing runs twice, once for train and once for val
        if not split_docs:
            continue
        writer = ShardWriter(out_dir, split, dtype, args.shard_tokens)
        for i, doc in enumerate(split_docs):
            ids = tok.encode(doc, allowed_special=False)
            ids.append(tok.eot_id)  # document boundary
            writer.add(ids)
            if (i + 1) % 5000 == 0:
                print(f"  {split}: {i + 1:,}/{len(split_docs):,} docs")
        writer.close()
        counts[split] = writer.total

    meta = {
        "name": name,
        "source": str(args.input),
        "tokenizer": args.tokenizer,
        "vocab_size": tok.vocab_size,
        "dtype": np.dtype(dtype).name,
        "tokens": counts,
        "doc_sep": args.doc_sep,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    total = sum(counts.values())
    print(f"\n{total:,} tokens total -> {out_dir}")
    print(f"compression: {len(text) / max(1, total):.2f} chars/token")


if __name__ == "__main__":
    main()
