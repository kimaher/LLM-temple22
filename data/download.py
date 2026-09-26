"""Fetch a raw text corpus into data/raw/.

    python -m data.download --corpus tinyshakespeare
    python -m data.download --corpus cosmopedia-stanford --shards 4
    python -m data.download --list

Corpora are deliberately small and plain-text: the point of this project is to
train end to end on a laptop/single GPU, not to reproduce a frontier run.

Two kinds of source:
  * text     a single .txt file, saved as-is (pretraining).
  * parquet  the first --shards shards of a Hugging Face dataset.  The .parquet
             files are cached next to the output and converted to either
               - .txt   documents joined by DOC_SEP (pretraining), or
               - .jsonl one {"messages": [...]} conversation per line (SFT,
                        the format data/sft_dataset.py reads).
             Needs `pip install pyarrow`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple

import requests

RAW_DIR = Path("data/raw")

# Parquet text corpora contain paragraph breaks inside documents, so the usual
# "\n\n" separator would chop every document into paragraphs.  Pass this to
# data.prepare as --doc-sep instead.
DOC_SEP = "<|doc|>"

_HF = "https://huggingface.co/datasets"


class Corpus(NamedTuple):
    url: str  # parquet: a template with {i}, the shard index
    filename: str
    description: str
    kind: str = "text"  # "text" | "parquet-text" | "parquet-chat"
    column: str = "text"  # parquet column holding the document / messages
    shards: int = 1  # parquet shards available upstream


CORPORA: Dict[str, Corpus] = {
    # --- pretraining ------------------------------------------------------
    "tinyshakespeare": Corpus(
        url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        filename="tinyshakespeare.txt",
        description="~1.1 MB of Shakespeare. The classic 'does my training loop work' corpus.",
    ),
    "tinystories": Corpus(
        url=f"{_HF}/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt",
        filename="tinystories.txt",
        description="~22 MB of simple synthetic children's stories. Small models produce real English on it.",
    ),
    "cosmopedia-stanford": Corpus(
        url=f"{_HF}/HuggingFaceTB/cosmopedia/resolve/main/data/stanford/train-{{i:05d}}-of-00013.parquet",
        filename="cosmopedia-stanford.txt",
        description="Synthetic textbook chapters (254 MB/shard, ~78k docs). Explanatory prose.",
        kind="parquet-text",
        shards=13,
    ),
    "cosmopedia-wikihow": Corpus(
        url=f"{_HF}/HuggingFaceTB/cosmopedia/resolve/main/data/wikihow/train-{{i:05d}}-of-00002.parquet",
        filename="cosmopedia-wikihow.txt",
        description="Synthetic how-to articles (251 MB/shard). Step-by-step instructions.",
        kind="parquet-text",
        shards=2,
    ),
    "fineweb-edu": Corpus(
        url=f"{_HF}/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/{{i:03d}}_00000.parquet",
        filename="fineweb-edu.txt",
        description="Educational web pages (2.2 GB/shard). Real-world knowledge.",
        kind="parquet-text",
        shards=14,
    ),
    # --- instruction tuning (SFT) ------------------------------------------
    "smol-smoltalk": Corpus(
        url=f"{_HF}/HuggingFaceTB/smol-smoltalk/resolve/main/data/train-{{i:05d}}-of-00004.parquet",
        filename="smol-smoltalk.jsonl",
        description="SFT: assistant chats built for <1B models (230 MB/shard).",
        kind="parquet-chat",
        column="messages",
        shards=4,
    ),
    "everyday-conversations": Corpus(
        url=f"{_HF}/HuggingFaceTB/smoltalk/resolve/main/data/everyday-conversations/train-00000-of-00001.parquet",
        filename="everyday-conversations.jsonl",
        description="SFT: ~2k short greetings and small-talk chats (1 MB). Good first SFT set.",
        kind="parquet-chat",
        column="messages",
    ),
}


def _fetch(url: str, dest: Path) -> None:
    """Stream `url` to `dest` via a .part file, so an interrupted download never looks complete."""
    print(f"downloading {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written = 0
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                written += len(chunk)
                if total:
                    print(f"\r  {written / 1e6:.1f}/{total / 1e6:.1f} MB", end="", flush=True)
        print()
        tmp.replace(dest)


def _iter_column(parquets: List[Path], column: str, max_docs: int) -> Iterator:
    """Yield one column's values across shards, in batches so a 2 GB shard never sits in RAM."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit("this corpus is stored as parquet; run `pip install pyarrow` first") from None
    n = 0
    for parquet in parquets:
        for batch in pq.ParquetFile(parquet).iter_batches(columns=[column], batch_size=1024):
            for value in batch.column(0).to_pylist():
                if max_docs and n >= max_docs:
                    return
                yield value
                n += 1


# Roles our chat template has special tokens for (see tokenizer/tokenizer.py).
_ROLES = {"system", "user", "assistant"}


def _convert(corpus: Corpus, parquets: List[Path], dest: Path, max_docs: int) -> int:
    tmp = dest.with_suffix(dest.suffix + ".part")
    n = 0
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for value in _iter_column(parquets, corpus.column, max_docs):
            if corpus.kind == "parquet-text":
                if n:
                    f.write(f"\n{DOC_SEP}\n")
                f.write(value.strip())
            else:
                messages = [{"role": m["role"], "content": m["content"]} for m in value if m["role"] in _ROLES]
                if not any(m["role"] == "assistant" for m in messages):
                    continue  # nothing for the model to learn from
                f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(dest)
    return n


def download(name: str, force: bool = False, max_docs: int = 0, shards: int = 1) -> Path:
    if name not in CORPORA:
        raise SystemExit(f"unknown corpus {name!r}; choose from {sorted(CORPORA)}")
    corpus = CORPORA[name]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / corpus.filename
    if dest.exists() and not force:
        print(f"{dest} already exists ({dest.stat().st_size:,} bytes); use --force to re-download")
        return dest

    if corpus.kind == "text":
        _fetch(corpus.url, dest)
    else:
        if not 1 <= shards <= corpus.shards:
            raise SystemExit(f"{name} has {corpus.shards} shard(s); --shards must be 1..{corpus.shards}")
        # Shards are the expensive part, so they are cached: rerunning with
        # --force and different --shards/--max-docs only fetches what's new.
        parquets = [dest.with_name(f"{dest.stem}-{i:03d}.parquet") for i in range(shards)]
        for i, parquet in enumerate(parquets):
            if not parquet.exists():
                _fetch(corpus.url.format(i=i), parquet)
        n = _convert(corpus, parquets, dest, max_docs)
        unit = "conversations" if corpus.kind == "parquet-chat" else "documents"
        print(f"converted {n:,} {unit}")

    print(f"saved {dest} ({dest.stat().st_size:,} bytes)")
    if corpus.kind == "parquet-text":
        print(f'next: python -m data.prepare --input {dest.as_posix()} --doc-sep "{DOC_SEP}"')
    elif corpus.kind == "parquet-chat":
        print(f"next: python -m train.sft --init checkpoints/pretrain/best.pt --data {dest.as_posix()}")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="tinyshakespeare")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="parquet corpora only: keep the first N documents/conversations (0 = all)",
    )
    ap.add_argument("--shards", type=int, default=1, help="parquet corpora only: how many shards to fetch")
    ap.add_argument("--list", action="store_true", help="list available corpora and exit")
    args = ap.parse_args()

    if args.list:
        for name, c in CORPORA.items():
            shards = f"{c.shards} shard(s)" if c.kind != "text" else ""
            print(f"{name:24s} {shards:11s} {c.description}")
        return
    download(args.corpus, force=args.force, max_docs=args.max_docs, shards=args.shards)


if __name__ == "__main__":
    main()
