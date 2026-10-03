"""Command-line launcher for the web backend.

    python -m web.backend.serve                          # checkpoints/sft/best.pt
    python -m web.backend.serve --mock                   # mock engine
    python -m web.backend.serve --hf                     # Qwen/Qwen3.5-2B
    python -m web.backend.serve --hf Qwen/Qwen3.5-0.8B   # any HF chat model

The flags are translated into the LLM_* environment variables that
`engine.build_engine` reads, so this is equivalent to setting them by hand and
running uvicorn - and it keeps working under --reload, which re-imports the app
in a child process.
"""

from __future__ import annotations

import argparse
import os

import uvicorn

DEFAULT_HF_MODEL = "Qwen/Qwen3.5-2B"


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the LLM-temple22 chat API")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--mock", action="store_true", help="serve canned responses, no model")
    source.add_argument(
        "--hf",
        nargs="?",
        const=DEFAULT_HF_MODEL,
        metavar="MODEL_ID",
        help=f"serve a pretrained Hugging Face model (default: {DEFAULT_HF_MODEL})",
    )
    source.add_argument("--checkpoint", help="checkpoint to serve (default: checkpoints/sft/best.pt)")
    parser.add_argument("--tokenizer", help="tokenizer spec for --checkpoint, must match training")
    parser.add_argument("--thinking", action="store_true", help="enable Qwen3-style thinking mode with --hf")
    parser.add_argument("--device", help="cuda | cpu (default: auto)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    args = parser.parse_args()

    if args.mock:
        os.environ["LLM_MOCK"] = "1"
    if args.hf:
        os.environ["LLM_HF_MODEL"] = args.hf
    if args.thinking:
        os.environ["LLM_HF_THINKING"] = "1"
    if args.checkpoint:
        os.environ["LLM_CHECKPOINT"] = args.checkpoint
    if args.tokenizer:
        os.environ["LLM_TOKENIZER"] = args.tokenizer
    if args.device:
        os.environ["LLM_DEVICE"] = args.device

    uvicorn.run("web.backend.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
