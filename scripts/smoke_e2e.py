"""End-to-end smoke test: tokenizer -> data -> pretrain -> SFT -> generate -> API.

Runs every stage of the pipeline at its smallest useful size (a few hundred KB
of text, a ~1M parameter model, a handful of steps) and fails loudly if any
stage breaks.  Takes well under a minute on a GPU, a couple of minutes on CPU.

    python scripts/smoke_e2e.py

Everything it produces lands in runs/smoke/ and data/processed/smoke/, so it
never touches your real checkpoints.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "smoke"
CORPUS = ROOT / "data" / "raw" / "tinyshakespeare.txt"
SMOKE_CORPUS = RUN_DIR / "corpus.txt"
TOKENIZER = RUN_DIR / "bpe.json"
DATA_DIR = ROOT / "data" / "processed" / "smoke"
PRETRAIN_DIR = RUN_DIR / "pretrain"
SFT_DIR = RUN_DIR / "sft"


def run(step: str, *args: str) -> None:
    print(f"\n=== {step} ===", flush=True)
    print("$ python " + " ".join(args), flush=True)
    t0 = time.perf_counter()
    result = subprocess.run([sys.executable, *args], cwd=ROOT, env={**_env()})
    if result.returncode != 0:
        raise SystemExit(f"FAILED at: {step}")
    print(f"[{step} ok in {time.perf_counter() - t0:.1f}s]", flush=True)


def _env() -> dict:
    import os

    # Force UTF-8 so sample previews don't blow up on a legacy Windows codepage.
    return {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(ROOT)}


def main() -> None:
    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR)
    RUN_DIR.mkdir(parents=True)

    # 0. corpus ---------------------------------------------------------
    if not CORPUS.exists():
        run("download corpus", "-m", "data.download", "--corpus", "tinyshakespeare")
    # A slice is plenty: this test is about plumbing, not model quality.
    SMOKE_CORPUS.write_text(
        CORPUS.read_text(encoding="utf-8")[:200_000], encoding="utf-8"
    )

    # 1. tokenizer ------------------------------------------------------
    run(
        "train tokenizer",
        "-m", "tokenizer.train_tokenizer",
        "--input", str(SMOKE_CORPUS),
        "--vocab-size", "512",
        "--output", str(TOKENIZER),
    )

    # 2. data -----------------------------------------------------------
    run(
        "prepare data",
        "-m", "data.prepare",
        "--input", str(SMOKE_CORPUS),
        "--name", "smoke",
        "--tokenizer", f"bpe:{TOKENIZER}",
        "--val-fraction", "0.05",
    )

    # 3. pretrain -------------------------------------------------------
    run(
        "pretrain",
        "-m", "train.pretrain",
        "--data", str(DATA_DIR),
        "--model-config", "configs/dev.json",
        "--out-dir", str(PRETRAIN_DIR),
        "--tokenizer", f"bpe:{TOKENIZER}",
        "--max-steps", "30",
        "--warmup-steps", "5",
        "--eval-interval", "15",
        "--eval-iters", "3",
        "--log-interval", "10",
        "--batch-size", "8",
        "--lr", "1e-3",
        "--sample-prompt", "ROMEO:",
    )

    # 4. SFT ------------------------------------------------------------
    if not (ROOT / "data" / "sft_sample.jsonl").exists():
        run("make sft sample", "scripts/make_sft_sample.py")
    run(
        "sft",
        "-m", "train.sft",
        "--init", str(PRETRAIN_DIR / "best.pt"),
        "--data", "data/sft_sample.jsonl",
        "--tokenizer", f"bpe:{TOKENIZER}",
        "--out-dir", str(SFT_DIR),
        "--max-steps", "20",
        "--warmup-steps", "5",
        "--eval-interval", "10",
        "--eval-iters", "2",
        "--log-interval", "10",
        "--batch-size", "4",
    )

    # 5. inference ------------------------------------------------------
    run(
        "generate",
        "-m", "inference.chat",
        "--checkpoint", str(SFT_DIR / "best.pt"),
        "--tokenizer", f"bpe:{TOKENIZER}",
        "--prompt", "Who wrote Romeo and Juliet?",
        "--max-new-tokens", "32",
        "--seed", "0",
    )

    # 6. web API against the freshly trained checkpoint -------------------
    run("web api", "scripts/check_api.py", str(SFT_DIR / "best.pt"), f"bpe:{TOKENIZER}")

    print("\nALL STAGES PASSED")
    print(f"artifacts in {RUN_DIR}")


if __name__ == "__main__":
    main()
