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

The input is streamed from disk and tokenized on `--workers` processes, so a
multi-GB corpus needs neither multi-GB of RAM nor an hour on one core.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import time
import unicodedata
from multiprocessing import Pool
from pathlib import Path
from typing import Iterator, List, Optional

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


_READ_CHARS = 1 << 24  # 16M chars per read; bounds memory regardless of corpus size


def iter_documents(path: Path, doc_sep: str, min_chars: int) -> Iterator[str]:
    """Stream cleaned documents from `path`, dropping the near-empty ones.

    Text mode already turns \\r\\n and \\r into \\n.  Invisible characters and
    trailing whitespace are removed *before* splitting, because a "blank" line
    holding a space or a zero-width char should still separate documents.  The
    text after the last separator in each read is carried into the next one,
    so separators (and those regexes) straddling a read boundary still match.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        if not doc_sep:
            doc = clean_text(f.read())
            if len(doc) >= min_chars:
                yield doc
            return
        carry = ""
        while True:
            chunk = f.read(_READ_CHARS)
            buf = _TRAILING_SPACE.sub("\n", _INVISIBLE.sub("", carry + chunk))
            parts = buf.split(doc_sep)
            carry = parts.pop() if chunk else ""
            for part in parts:
                doc = clean_text(part)
                if len(doc) >= min_chars:
                    yield doc
            if not chunk:
                return


# Per-process tokenizer, built once by the pool initializer rather than pickled
# into every task.
_tok = None


def _init_worker(spec: Optional[str]) -> None:
    global _tok
    _tok = load_tokenizer(spec)


def _encode(doc: str) -> np.ndarray:
    ids = _tok.encode(doc, allowed_special=False)
    ids.append(_tok.eot_id)  # document boundary
    # uint32 pickles far smaller than a list of Python ints on the way back
    # from the worker; ShardWriter narrows it to the output dtype.
    return np.asarray(ids, dtype=np.uint32)


class ShardWriter:
    """Buffers token ids and flushes fixed-size .bin shards to disk."""

    def __init__(self, out_dir: Path, split: str, dtype: np.dtype, shard_tokens: int) -> None:
        self.out_dir = out_dir
        self.split = split
        self.dtype = dtype
        self.shard_tokens = shard_tokens
        self.buffer: List[np.ndarray] = []
        self.buffered = 0
        self.shard_index = 0
        self.total = 0

    def add(self, ids: np.ndarray) -> None:
        self.buffer.append(ids)
        self.buffered += len(ids)
        while self.buffered >= self.shard_tokens:
            flat = np.concatenate(self.buffer)
            self._flush(flat[: self.shard_tokens])
            rest = flat[self.shard_tokens :]
            self.buffer, self.buffered = [rest], len(rest)

    def close(self) -> None:
        if self.buffered:
            self._flush(np.concatenate(self.buffer))
        self.buffer, self.buffered = [], 0

    def _flush(self, ids: np.ndarray) -> None:
        path =self.out_dir / f"{self.split}_{self.shard_index:06d}.bin"
        ids.astype(self.dtype).tofile(path)
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
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="tokenizer processes (1 = in-process)")
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

    # Pass 1 only counts, so the train/val boundary is known before pass 2
    # streams the same documents through the tokenizer.
    def docs() -> Iterator[str]:
        it = iter_documents(Path(args.input), args.doc_sep, args.min_chars)
        return itertools.islice(it, args.limit_docs) if args.limit_docs else it

    n_docs = n_chars = 0
    for doc in docs():
        n_docs += 1
        n_chars += len(doc)
    n_val = max(1, int(n_docs * args.val_fraction)) if n_docs > 1 else 0
    # Hold out a contiguous tail rather than a random sample: with a random
    # sample, near-duplicate neighbouring documents leak between the splits.
    n_train = n_docs - n_val
    print(f"{n_docs:,} documents -> {n_train:,} train / {n_val:,} val, {args.workers} worker(s)")

    # imap keeps document order, so the first n_train results are train and
    # the rest are val, exactly as if tokenized serially.
    writers = {s: ShardWriter(out_dir, s, dtype, args.shard_tokens) for s in ("train", "val")}
    pool = Pool(args.workers, _init_worker, (args.tokenizer,)) if args.workers > 1 else None
    if pool is None:
        _init_worker(args.tokenizer)
    encoded = pool.imap(_encode, docs(), chunksize=64) if pool else map(_encode, docs())
    t0 = time.perf_counter()
    try:
        for i, ids in enumerate(encoded):
            writers["train" if i < n_train else "val"].add(ids)
            if (i + 1) % 20_000 == 0:
                rate = (i + 1) / (time.perf_counter() - t0)
                print(f"  {i + 1:,}/{n_docs:,} docs ({rate:,.0f} docs/s)")
    finally:
        if pool:
            pool.close()
            pool.join()
    counts = {}
    for split, writer in writers.items():
        writer.close()
        if writer.total:
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
    print(f"compression: {n_chars / max(1, total):.2f} chars/token")


if __name__ == "__main__":
    main()
