"""Fetch a raw text corpus into data/raw/.

    python -m data.download --corpus tinyshakespeare
    python -m data.download --list

Corpora are deliberately small and plain-text: the point of this project is to
train end to end on a laptop/single GPU, not to reproduce a frontier run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, NamedTuple

import requests

RAW_DIR = Path("data/raw")


class Corpus(NamedTuple):
    url: str
    filename: str
    description: str


CORPORA: Dict[str, Corpus] = {
    "tinyshakespeare": Corpus(
        url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        filename="tinyshakespeare.txt",
        description="~1.1 MB of Shakespeare. The classic 'does my training loop work' corpus.",
    ),
    "tinystories": Corpus(
        url="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt",
        filename="tinystories.txt",
        description="~22 MB of simple synthetic children's stories. Small models produce real English on it.",
    ),
}


def download(name: str, force: bool = False) -> Path:
    if name not in CORPORA:
        raise SystemExit(f"unknown corpus {name!r}; choose from {sorted(CORPORA)}")
    corpus = CORPORA[name]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / corpus.filename
    if dest.exists() and not force:
        print(f"{dest} already exists ({dest.stat().st_size:,} bytes); use --force to re-download")
        return dest

    print(f"downloading {corpus.url}")
    with requests.get(corpus.url, stream=True, timeout=60) as r:
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
    print(f"saved {dest} ({dest.stat().st_size:,} bytes)")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="tinyshakespeare")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true", help="list available corpora and exit")
    args = ap.parse_args()

    if args.list:
        for name, c in CORPORA.items():
            print(f"{name:16s} {c.description}")
        return
    download(args.corpus, force=args.force)


if __name__ == "__main__":
    main()
