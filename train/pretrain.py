"""Pretraining loop: next-token prediction on a tokenized corpus.

    # tiny end-to-end run on CPU or GPU, finishes in under a minute
    python -m train.pretrain --data data/processed/shakespeare \
        --model-config configs/tiny.json --max-steps 200 --eval-interval 50

    # scale up
    python -m train.pretrain --data data/processed/tinystories \
        --model-config configs/small.json --max-steps 20000 --batch-size 32 --grad-accum 4

Everything that makes training stable is here and nowhere else: LR warmup +
cosine decay, gradient accumulation, gradient clipping, mixed precision, and
checkpointing that can resume mid-run.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from data.dataset import TokenShardDataset
from model.config import ModelConfig
from model.transformer import Transformer
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
    # data / io
    ap.add_argument("--data", default="data/processed/shakespeare", help="dir produced by data.prepare")
    ap.add_argument("--model-config", default="configs/tiny.json")
    ap.add_argument("--out-dir", default="checkpoints/pretrain")
    ap.add_argument("--resume", action="store_true", help="continue from out-dir/last.pt")
    ap.add_argument("--tokenizer", default=None, help="only used for the sample previews during eval")
    # optimisation
    ap.add_argument("--batch-size", type=int, default=16, help="sequences per micro-step")
    ap.add_argument("--grad-accum", type=int, default=1, help="micro-steps per optimizer step")
    ap.add_argument("--max-steps", type=int, default=2000, help="optimizer steps (not micro-steps)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    # evaluation / logging
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--sample-prompt", default="\n", help="preview prompt printed at each eval")
    # system
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (slow first step, faster after)")
    return ap.parse_args()


@torch.no_grad()
def estimate_loss(model: Transformer, datasets: dict, amp: AmpContext, iters: int, batch_size: int, device: str):
    """Average loss over a few random batches from each split."""
    model.eval()
    out = {}
    for split, ds in datasets.items():
        losses = torch.zeros(iters)
        for i in range(iters):
            x, y = ds.get_batch(batch_size, device=device)
            with amp.autocast():
                _, loss = model(x, targets=y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def preview_sample(model: Transformer, tokenizer_spec: str | None, prompt: str, device: str) -> str | None:
    """Decode a short continuation so you can watch English emerge."""
    try:
        from inference.generate import SamplingConfig, generate
        from tokenizer.tokenizer import load_tokenizer

        tok = load_tokenizer(tokenizer_spec)
        ids = tok.encode(prompt, allowed_special=False) or [tok.bos_id]
        out = generate(model, ids, SamplingConfig(max_new_tokens=64, temperature=0.8, top_k=50, top_p=0.95))
        return tok.decode(out)
    except Exception as exc:  # a preview must never take the training run down
        return f"(sample unavailable: {exc})"


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    dtype = pick_dtype(device, args.dtype)
    amp = AmpContext(device, dtype)
    if device.startswith("cuda"):
        # TF32 matmuls: noticeably faster on Ampere+, and the precision loss is
        # irrelevant next to the noise of SGD.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- data
    config = ModelConfig.load(args.model_config)
    train_ds = TokenShardDataset(args.data, "train", seq_len=config.max_seq_len)
    datasets = {"train": train_ds}
    try:
        datasets["val"] = TokenShardDataset(args.data, "val", seq_len=config.max_seq_len)
    except (FileNotFoundError, ValueError) as exc:
        print(f"no usable val split ({exc}); reporting train loss only")

    # The corpus decides the vocabulary, so the config's placeholder is
    # overwritten here.  Padding to a multiple of 64 keeps the output matmul on
    # tensor-core-friendly shapes; the extra rows are simply never sampled.
    config.vocab_size = ((train_ds.vocab_size + 63) // 64) * 64
    print(f"data: {human(train_ds.total_tokens)} train tokens, vocab_size={config.vocab_size}")

    # --------------------------------------------------------------- model
    start_step = 0
    best_val = float("inf")
    if args.resume:
        model, ckpt = load_checkpoint(out_dir / "last.pt", device=device)
        start_step = ckpt.get("step", 0)
        best_val = ckpt.get("best_val_loss", float("inf"))
        config = model.config
        print(f"resumed from step {start_step}")
    else:
        model = Transformer(config).to(device)
    config.save(out_dir / "model_config.json")

    print(
        f"model: {human(model.num_params())} params "
        f"({human(model.num_params(non_embedding=True))} non-embedding), "
        f"{config.n_layers}L x {config.d_model}d x {config.n_heads}h, ctx={config.max_seq_len}"
    )
    print(f"device={device} dtype={dtype} amp={amp.enabled}")

    optimizer = model.configure_optimizers(args.lr, args.weight_decay, (args.beta1, args.beta2), device)
    if args.resume and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    if args.compile:
        print("compiling model (first step will be slow)...")
        model = torch.compile(model)

    tokens_per_step = args.batch_size * args.grad_accum * config.max_seq_len
    print(f"tokens/optimizer step: {tokens_per_step:,}")
    log_path = out_dir / "log.jsonl"
    log_file = log_path.open("a", encoding="utf-8")
    timer = Timer()
    timer.start()

    # ---------------------------------------------------------------- loop
    model.train()
    for step in range(start_step, args.max_steps):
        lr = cosine_lr(
            step,
            base_lr=args.lr,
            min_lr=args.min_lr,
            warmup_steps=args.warmup_steps,
            total_steps=args.max_steps,
        )
        set_lr(optimizer, lr)

        # Gradient accumulation: several micro-batches make up one optimizer
        # step, which is how you get a large effective batch on small hardware.
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = train_ds.get_batch(args.batch_size, device=device)
            with amp.autocast():
                _, loss = model(x, targets=y)
                loss = loss / args.grad_accum  # so the gradient matches one big batch
            amp.scaler.scale(loss).backward()
            total_loss += loss.item()

        # Unscale before clipping, otherwise we would clip the fp16 loss-scaled
        # gradients and the threshold would mean nothing.
        amp.scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        amp.scaler.step(optimizer)
        amp.scaler.update()

        if step % args.log_interval == 0:
            dt = timer.lap()
            tps = tokens_per_step * args.log_interval / dt if step else tokens_per_step / dt
            record = {
                "step": step,
                "loss": total_loss,
                "lr": lr,
                "grad_norm": float(grad_norm),
                "tokens_per_sec": tps,
            }
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
            print(
                f"step {step:6d} | loss {total_loss:.4f} | ppl {math.exp(min(total_loss, 20)):8.2f} | "
                f"lr {lr:.2e} | gnorm {float(grad_norm):.2f} | {human(tps)} tok/s"
            )

        is_last = step == args.max_steps - 1
        if (step > 0 and step % args.eval_interval == 0) or is_last:
            losses = estimate_loss(model, datasets, amp, args.eval_iters, args.batch_size, device)
            msg = " | ".join(f"{k} {v:.4f}" for k, v in losses.items())
            print(f"  eval @ {step}: {msg}")
            val = losses.get("val", losses["train"])

            raw_model = getattr(model, "_orig_mod", model)  # unwrap torch.compile
            # update best_val before saving last.pt so --resume restores the true best
            is_best = val < best_val
            if is_best:
                best_val = val
            save_checkpoint(out_dir / "last.pt", raw_model, optimizer, step=step, best_val_loss=best_val)
            if is_best:
                save_checkpoint(out_dir / "best.pt", raw_model, optimizer, step=step, best_val_loss=best_val)
                print(f"  new best val loss {best_val:.4f} -> {out_dir / 'best.pt'}")

            sample = preview_sample(raw_model, args.tokenizer, args.sample_prompt, device)
            if sample:
                print(f"  sample: {sample!r}")
            timer.start()  # don't count eval time in tokens/sec

    log_file.close()
    print(f"\ndone. best val loss {best_val:.4f}; checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
