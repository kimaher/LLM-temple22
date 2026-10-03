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

Guarding against forgetting (both optional):

  * --eval-pretrain-data DIR  tracks loss on the pretraining corpus (a dir from
    data.prepare) at every eval, so you can *see* how much the model forgets;
  * --replay-data DIR         mixes plain next-token batches from that corpus
    into every step ("replay"), weighted by --replay-weight.  This is the most
    reliable fix for small models.  The forgetting check uses this corpus too
    unless --eval-pretrain-data says otherwise.

    python -m train.sft --init checkpoints/pretrain/best.pt \
        --data data/raw/smol-smoltalk.jsonl --replay-data data/processed/fineweb-edu
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from data.dataset import TokenShardDataset
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
    ap.add_argument(
        "--replay-data",
        default=None,
        help="pretraining token dir (from data.prepare); if given, its batches are mixed into every step",
    )
    ap.add_argument("--replay-weight", type=float, default=0.5, help="loss weight of the replay batches")
    ap.add_argument(
        "--replay-batch-size", type=int, default=None, help="replay windows per micro-step (default: --batch-size)"
    )
    ap.add_argument(
        "--eval-pretrain-data",
        default=None,
        help="pretraining token dir to measure forgetting on (default: --replay-data, if given)",
    )
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


@torch.no_grad()
def evaluate_pretrain(
    model: Transformer, ds: TokenShardDataset, amp: AmpContext, iters: int, batch_size: int, device: str
) -> float:
    """Plain LM loss on the pretraining corpus - if this climbs, the model is forgetting."""
    model.eval()
    total = 0.0
    for _ in range(iters):
        x, y = ds.get_batch(batch_size, device=device)
        with amp.autocast():
            _, loss = model(x, targets=y)
        total += loss.item()
    model.train()
    return total / max(1, iters)


def load_pretrain_shards(data_dir: str, split: str, seq_len: int, vocab_size: int) -> TokenShardDataset:
    ds = TokenShardDataset(data_dir, split, seq_len=seq_len)
    if ds.vocab_size > vocab_size:
        raise SystemExit(
            f"{data_dir} was tokenized with vocab {ds.vocab_size}, larger than the model's ({vocab_size}); "
            f"point at the corpus this checkpoint was pretrained on"
        )
    return ds


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  # TokenShardDataset samples windows with numpy
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

    # Replay: windows from the train split, trained on with the plain LM loss.
    replay_ds = None
    if args.replay_data:
        replay_ds = load_pretrain_shards(args.replay_data, "train", config.max_seq_len, config.vocab_size)
        print(f"replaying {args.replay_data} ({human(len(replay_ds))} tokens) at weight {args.replay_weight}")
    replay_bs = args.replay_batch_size or args.batch_size

    # Forgetting check: prefer the held-out val split, fall back to train.
    pre_eval_ds = None
    pre_eval_dir = args.eval_pretrain_data or args.replay_data
    if pre_eval_dir:
        try:
            pre_eval_ds = load_pretrain_shards(pre_eval_dir, "val", config.max_seq_len, config.vocab_size)
        except (FileNotFoundError, ValueError) as exc:
            print(f"no usable val split in {pre_eval_dir} ({exc}); measuring forgetting on train")
            pre_eval_ds = load_pretrain_shards(pre_eval_dir, "train", config.max_seq_len, config.vocab_size)
    pre_base = None
    if pre_eval_ds is not None:
        pre_base = evaluate_pretrain(model, pre_eval_ds, amp, args.eval_iters, replay_bs, device)
        print(f"pretrain loss before SFT: {pre_base:.4f}")

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
        replay_loss = 0.0
        for _ in range(args.grad_accum):
            x, y, m = train_ds.get_batch(args.batch_size, device=device)
            with amp.autocast():
                _, loss = model(x, targets=y, loss_mask=m)
                loss = loss / args.grad_accum
            amp.scaler.scale(loss).backward()
            total_loss += loss.item()
            if replay_ds is not None:
                # A separate backward (rather than summing the losses) frees the
                # SFT activations before the full-length replay windows go in.
                rx, ry = replay_ds.get_batch(replay_bs, device=device)
                with amp.autocast():
                    _, rloss = model(rx, targets=ry)
                    rloss = rloss / args.grad_accum
                amp.scaler.scale(args.replay_weight * rloss).backward()
                replay_loss += rloss.item()

        amp.scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        amp.scaler.step(optimizer)
        amp.scaler.update()

        if step % args.log_interval == 0:
            record = {"step": step, "loss": total_loss, "lr": lr}
            line = f"step {step:5d} | loss {total_loss:.4f} | ppl {math.exp(min(total_loss, 20)):7.2f} | lr {lr:.2e}"
            if replay_ds is not None:
                record["replay_loss"] = replay_loss
                line += f" | replay {replay_loss:.4f}"
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
            print(line)

        is_last = step == args.max_steps - 1
        if (step > 0 and step % args.eval_interval == 0) or is_last:
            val = evaluate(model, val_ds, amp, args.eval_iters, args.batch_size, device) if val_ds else total_loss
            msg = f"  eval @ {step}: {'val' if val_ds else 'train'} {val:.4f}"
            if pre_eval_ds is not None:
                pre = evaluate_pretrain(model, pre_eval_ds, amp, args.eval_iters, replay_bs, device)
                msg += f" | pretrain {pre:.4f} ({pre - pre_base:+.4f} vs before SFT)"
                log_file.write(json.dumps({"step": step, "val": val, "pretrain_loss": pre}) + "\n")
                log_file.flush()
            print(msg)
            # update best_val before saving last.pt so it records the true best
            is_best = val < best_val
            if is_best:
                best_val = val
            save_checkpoint(out_dir / "last.pt", model, optimizer, step=step, best_val_loss=best_val)
            if is_best:
                save_checkpoint(out_dir / "best.pt", model, optimizer, step=step, best_val_loss=best_val)

    log_file.close()
    print(f"\ndone in {timer.lap():.1f}s. checkpoints in {out_dir}")
    print("try it:  python -m inference.chat --checkpoint " + str(out_dir / "best.pt"))


if __name__ == "__main__":
    main()
