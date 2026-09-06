"""Train the byte-level BPE tokenizer on a text corpus.

    python -m tokenizer.train_tokenizer --input data/raw/tinyshakespeare.txt \
        --vocab-size 4096 --output tokenizer/artifacts/bpe.json

Rules of thumb for `--vocab-size`: roughly sqrt-ish scaling with corpus size -
~4k for a 1 MB corpus, ~8-16k for tens of MB.  Too large and most embeddings
never get a gradient; too small and sequences get long, which costs attention
time quadratically.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .bpe import BPETokenizer
from .tokenizer import SPECIAL_TOKENS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="path to a UTF-8 text file")
    ap.add_argument("--output", default="tokenizer/artifacts/bpe.json")
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument(
        "--max-chars",
        type=int,
        default=20_000_000,
        help="cap the training text; merges converge long before you need the whole corpus",
    )
    args = ap.parse_args()

    text = Path(args.input).read_text(encoding="utf-8", errors="replace")
    if len(text) > args.max_chars:
        print(f"truncating corpus {len(text):,} -> {args.max_chars:,} chars for tokenizer training")
        text = text[: args.max_chars]

    print(f"training BPE: {len(text):,} chars -> vocab_size={args.vocab_size}")
    t0 = time.perf_counter()
    tok = BPETokenizer.train(text, vocab_size=args.vocab_size, special_tokens=SPECIAL_TOKENS, verbose=True)
    dt = time.perf_counter() - t0

    tok.save(args.output)
    sample = text[:2000]
    ids = tok.encode(sample)
    ratio = len(sample.encode("utf-8")) / max(1, len(ids))
    print(f"done in {dt:.1f}s -> {args.output}")
    print(f"vocab_size={tok.vocab_size} (merges={len(tok.merges)}, special={len(tok.special_tokens)})")
    print(f"compression: {ratio:.2f} bytes/token on a sample of the corpus")
    assert tok.decode(ids) == sample, "round-trip failed"
    print("round-trip OK")


if __name__ == "__main__":
    main()
