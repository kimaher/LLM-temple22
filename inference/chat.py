"""Terminal chat / completion client - the fastest way to eyeball a checkpoint.

    # chat with an SFT'd checkpoint
    python -m inference.chat --checkpoint checkpoints/sft/best.pt

    # raw completion from a pretrained (not yet instruction-tuned) checkpoint
    python -m inference.chat --checkpoint checkpoints/pretrain/best.pt --raw --prompt "ROMEO:"

Commands inside the REPL: /reset (clear history), /quit.
"""

from __future__ import annotations

import argparse
import sys

import torch

from inference.generate import SamplingConfig, generate_stream
from inference.stream import TokenStreamDecoder
from tokenizer.tokenizer import load_tokenizer
from train.utils import load_checkpoint, pick_device


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/sft/best.pt")
    ap.add_argument("--tokenizer", default=None, help="tokenizer spec; must match training")
    ap.add_argument("--prompt", default=None, help="answer one prompt and exit (no REPL)")
    ap.add_argument("--raw", action="store_true", help="plain completion, bypassing the chat template")
    ap.add_argument("--system", default=None, help="optional system message")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="auto")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    # Windows terminals default to a legacy codepage; without this, any
    # non-ASCII token the model emits raises UnicodeEncodeError.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    device = pick_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device=device)
    model.eval()
    tok = load_tokenizer(args.tokenizer)
    print(
        f"loaded {args.checkpoint} (step {ckpt.get('step', 0)}, "
        f"{model.num_params() / 1e6:.1f}M params) on {device}"
    )

    sampling = SamplingConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        stop_ids=None if args.raw else tok.stop_ids,
        seed=args.seed,
    )

    def respond(prompt_ids: list[int]) -> str:
        """Stream one completion to stdout and return the full text."""
        decoder = TokenStreamDecoder(tok)
        pieces = []
        for token in generate_stream(model, prompt_ids, sampling, device=device):
            chunk = decoder.push(token)
            if chunk:
                pieces.append(chunk)
                print(chunk, end="", flush=True)
        tail = decoder.flush()
        if tail:
            pieces.append(tail)
            print(tail, end="", flush=True)
        print()
        return "".join(pieces)

    # --- one-shot mode ---------------------------------------------------
    if args.prompt is not None:
        if args.raw:
            ids = tok.encode(args.prompt, allowed_special=False)
        else:
            messages = ([{"role": "system", "content": args.system}] if args.system else []) + [
                {"role": "user", "content": args.prompt}
            ]
            ids = tok.render_chat(messages)
        respond(ids)
        return

    # --- REPL ------------------------------------------------------------
    history: list[dict[str, str]] = []
    if args.system and not args.raw:
        history.append({"role": "system", "content": args.system})
    print("chat ready. /reset clears history, /quit exits.\n")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/quit", "/exit"):
            break
        if user == "/reset":
            history = [h for h in history if h["role"] == "system"]
            print("(history cleared)")
            continue

        if args.raw:
            ids = tok.encode(user, allowed_special=False)
        else:
            history.append({"role": "user", "content": user})
            ids = tok.render_chat(history)

        print("bot> ", end="", flush=True)
        with torch.no_grad():
            reply = respond(ids)
        if not args.raw:
            history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
