"""Smallest possible end-to-end proof that the model + training loop work.

No tokenizer, no dataset, no checkpoints on disk to worry about: we generate a
synthetic corpus with an obvious rule (the next token is always the previous one
plus a fixed stride, modulo the vocab), train a tiny model on it for a few
hundred steps and check that the loss collapses and that greedy decoding
reproduces the rule.

If this passes, the architecture, the loss, the optimizer and the LR schedule
are all wired up correctly - everything after this is data plumbing.

    python scripts/smoke_train.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import ModelConfig, Transformer  # noqa: E402
from train.utils import AmpContext, cosine_lr, pick_device, pick_dtype, set_lr  # noqa: E402

VOCAB = 64
STRIDE = 3


def make_batch(batch_size: int, seq_len: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """x[t+1] = (x[t] + STRIDE) % VOCAB, starting from a random token.

    Learnable, but only if the model can actually see the previous token - so a
    broken causal mask or broken positions show up immediately as a stuck loss.
    """
    starts = torch.randint(0, VOCAB, (batch_size, 1), device=device)
    offsets = torch.arange(seq_len + 1, device=device)[None, :] * STRIDE
    seq = (starts + offsets) % VOCAB
    return seq[:, :-1], seq[:, 1:]  # inputs, next-token targets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    torch.manual_seed(1337)
    device = pick_device(args.device)
    dtype = pick_dtype(device)
    amp = AmpContext(device, dtype)

    config = ModelConfig(
        vocab_size=VOCAB, n_layers=2, n_heads=4, d_model=128, max_seq_len=args.seq_len, dropout=0.0
    )
    model = Transformer(config).to(device)
    print(f"device={device} dtype={dtype} params={model.num_params():,}")

    optimizer = model.configure_optimizers(lr=args.lr, weight_decay=0.1, device=device)

    initial_loss = None
    for step in range(args.steps):
        lr = cosine_lr(step, base_lr=args.lr, min_lr=args.lr * 0.1, warmup_steps=20, total_steps=args.steps)
        set_lr(optimizer, lr)

        x, y = make_batch(args.batch_size, args.seq_len, device)
        with amp.autocast():
            _, loss = model(x, targets=y)

        optimizer.zero_grad(set_to_none=True)
        amp.scaler.scale(loss).backward()
        amp.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        amp.scaler.step(optimizer)
        amp.scaler.update()

        if initial_loss is None:
            initial_loss = loss.item()
        if step % 50 == 0 or step == args.steps - 1:
            print(f"step {step:4d} | loss {loss.item():.4f} | lr {lr:.2e}")

    final_loss = loss.item()

    # Greedy decode: feed a prefix, check the model continues the pattern.
    model.eval()
    prompt = torch.tensor([[7]], device=device)
    cache = model.new_cache(batch_size=1)
    generated = [7]
    cur = prompt
    with torch.no_grad():
        for _ in range(8):
            logits, _ = model(cur, cache=cache)
            cur = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(int(cur.item()))
    expected = [(7 + i * STRIDE) % VOCAB for i in range(len(generated))]
    print(f"generated: {generated}")
    print(f"expected:  {expected}")

    ok = final_loss < 0.1 and generated == expected
    print(f"\ninitial loss {initial_loss:.4f} -> final loss {final_loss:.4f}  (chance = {math.log(VOCAB):.2f})")
    print("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
