"""Supervised fine-tuning: teach the pretrained model the chat format.

    python -m train.sft --init checkpoints/pretrain/best.pt \
        --data data/sft_sample.jsonl --max-steps 200

Differences from pretraining, and why:

  * we start from pretrained weights, so the LR is ~10x lower - a big LR here
    would wash out everything the pretraining run learned ("catastrophic
    forgetting");
  * examples are conversations rendered with the chat template, not random
    windows of a token stream;
  * the loss is masked to the assistant's tokens only (see data/sft_dataset.py);
  * runs are short: a few hundred steps over a small, high-quality set beats a
    long run over a noisy one.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from data.sft_dataset import SFTDataset
from model.transformer import Transformer
from tokenizer.tokenizer import load_tokenizer
from train.utils import (
    AmpContext,
    Timer,
    cosine_lr,
    human,
    load_checkpoint,
    pick_device,
    pick_dtype,
    save_checkpoint,
    set_lr,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default="checkpoints/pretrain/best.pt", help="pretrained checkpoint to fine-tune")
    ap.add_argument("--data", default="data/sft_sample.jsonl")
    ap.add_argument("--tokenizer", default=None, help="must match the one used for pretraining")
    ap.add_argument("--out-dir", default="checkpoints/sft")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--min-lr", type=float, default=2e-6)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--eval-interval", type=int, default=100)
    ap.add_argument("--eval-iters", type=int, default=10)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    ap.add_argument("--seed", type=int, default=1337)
    return ap.parse_args()


@torch.no_grad()
def evaluate(model: Transformer, ds: SFTDataset, amp: AmpContext, iters: int, batch_size: int, device: str) -> float:
    model.eval()
    total = 0.0
    for _ in range(iters):
        x, y, m = ds.get_batch(batch_size, device=device)
        with amp.autocast():
            _, loss = model(x, targets=y, loss_mask=m)
        total += loss.item()
    model.train()
    return total / max(1, iters)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    dtype = pick_dtype(device, args.dtype)
    amp = AmpContext(device, dtype)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_checkpoint(args.init, device=device)
    config = model.config
    print(f"loaded {args.init} (pretrained {ckpt.get('step', 0)} steps, {human(model.num_params())} params)")

    tok = load_tokenizer(args.tokenizer)
    if tok.vocab_size > config.vocab_size:
        raise SystemExit(
            f"tokenizer vocab ({tok.vocab_size}) exceeds the model's ({config.vocab_size}); "
            f"pass the same --tokenizer used for pretraining"
        )

    dataset = SFTDataset(args.data, tok, max_seq_len=config.max_seq_len)
    train_ds, val_ds = dataset.split(args.val_fraction)
    print(f"{len(train_ds)} train examples" + (f", {len(val_ds)} val examples" if val_ds else ""))

    optimizer = model.configure_optimizers(args.lr, args.weight_decay, device=device)
    log_file = (out_dir / "log.jsonl").open("a", encoding="utf-8")
    timer = Timer()
    timer.start()
    best_val = float("inf")

    model.train()
    for step in range(args.max_steps):
        lr = cosine_lr(
            step, base_lr=args.lr, min_lr=args.min_lr, warmup_steps=args.warmup_steps, total_steps=args.max_steps
        )
        set_lr(optimizer, lr)

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(args.grad_accum):
            x, y, m = train_ds.get_batch(args.batch_size, device=device)
            with amp.autocast():
                _, loss = model(x, targets=y, loss_mask=m)
                loss = loss / args.grad_accum
            amp.scaler.scale(loss).backward()
            total_loss += loss.item()

        amp.scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        amp.scaler.step(optimizer)
        amp.scaler.update()

        if step % args.log_interval == 0:
            log_file.write(json.dumps({"step": step, "loss": total_loss, "lr": lr}) + "\n")
            log_file.flush()
            print(
                f"step {step:5d} | loss {total_loss:.4f} | ppl {math.exp(min(total_loss, 20)):7.2f} | lr {lr:.2e}"
            )

        is_last = step == args.max_steps - 1
        if (step > 0 and step % args.eval_interval == 0) or is_last:
            val = evaluate(model, val_ds, amp, args.eval_iters, args.batch_size, device) if val_ds else total_loss
            print(f"  eval @ {step}: {'val' if val_ds else 'train'} {val:.4f}")
            save_checkpoint(out_dir / "last.pt", model, optimizer, step=step, best_val_loss=best_val)
            if val < best_val:
                best_val = val
                save_checkpoint(out_dir / "best.pt", model, optimizer, step=step, best_val_loss=best_val)

    log_file.close()
    print(f"\ndone in {timer.lap():.1f}s. checkpoints in {out_dir}")
    print("try it:  python -m inference.chat --checkpoint " + str(out_dir / "best.pt"))


if __name__ == "__main__":
    main()
