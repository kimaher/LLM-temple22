"""Instruction-tuning data: JSONL conversations -> padded batches + loss masks.

Accepted line formats (both are common in the wild, so we take either):

    {"messages": [{"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}]}
    {"instruction": "...", "input": "...", "response": "..."}

The important part is the loss mask.  We render the full conversation with the
chat template, but only score the assistant's tokens: training the model to
predict the *user's* text teaches it to interview itself, and dilutes the
gradient signal we actually want.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from tokenizer.tokenizer import ASSISTANT, BaseTokenizer

IGNORE_INDEX = -100  # matches F.cross_entropy's default ignore_index


def _normalise(record: Dict) -> List[Dict[str, str]]:
    """Coerce one JSONL record into a list of chat messages."""
    if "messages" in record:
        return list(record["messages"])
    instruction = record.get("instruction", "")
    extra = record.get("input", "")
    prompt = f"{instruction}\n\n{extra}".strip() if extra else instruction
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": record.get("response") or record.get("output", "")},
    ]


class SFTDataset:
    """Tokenizes every conversation up front (SFT sets are small) and batches them."""

    def __init__(self, path: str | Path, tokenizer: BaseTokenizer, max_seq_len: int = 512) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples: List[Tuple[List[int], List[int]]] = []  # (ids, mask)

        skipped = 0
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            messages = _normalise(json.loads(line))
            ids, mask = self._encode_conversation(messages)
            if len(ids) < 2 or sum(mask) == 0:
                skipped += 1
                continue
            self.examples.append((ids[:max_seq_len], mask[:max_seq_len]))
        if not self.examples:
            raise ValueError(f"no usable examples in {path}")
        self.skipped = skipped

    def _encode_conversation(self, messages: Sequence[Dict[str, str]]) -> Tuple[List[int], List[int]]:
        """Render the conversation and mark which tokens are supervised.

        A token is supervised when it belongs to an assistant reply (including
        the closing <|eot|>, so the model learns to stop) but not the
        <|assistant|> marker itself, which is part of the prompt.
        """
        tok = self.tokenizer
        ids: List[int] = [tok.bos_id]
        mask: List[int] = [0]
        for msg in messages:
            role, content = msg["role"], msg["content"]
            role_id = tok.special_tokens[{"system": "<|system|>", "user": "<|user|>", "assistant": ASSISTANT}[role]]
            body = tok.encode(content, allowed_special=False)
            ids.append(role_id)
            mask.append(0)
            supervised = 1 if role == "assistant" else 0
            ids.extend(body)
            mask.extend([supervised] * len(body))
            ids.append(tok.eot_id)
            mask.append(supervised)
        return ids, mask

    @classmethod
    def from_examples(
        cls, examples: List[Tuple[List[int], List[int]]], tokenizer: BaseTokenizer, max_seq_len: int
    ) -> "SFTDataset":
        """Build a dataset from already-tokenized examples (used by `split`)."""
        obj = cls.__new__(cls)
        obj.tokenizer, obj.max_seq_len, obj.examples, obj.skipped = tokenizer, max_seq_len, list(examples), 0
        return obj

    def split(self, val_fraction: float) -> Tuple["SFTDataset", Optional["SFTDataset"]]:
        """Split off a validation slice; returns (train, val | None)."""
        n_val = int(len(self.examples) * val_fraction)
        if n_val < 1 or len(self.examples) - n_val < 1:
            return self, None
        make = lambda ex: SFTDataset.from_examples(ex, self.tokenizer, self.max_seq_len)  # noqa: E731
        return make(self.examples[:-n_val]), make(self.examples[-n_val:])

    def __len__(self) -> int:
        return len(self.examples)

    def get_batch(
        self,
        batch_size: int,
        device: str = "cpu",
        generator: torch.Generator | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (x, y, loss_mask), right-padded to the longest example in the batch.

        Padding is masked out of the loss, so the pad id itself is irrelevant.
        """
        idx = torch.randint(0, len(self.examples), (batch_size,), generator=generator)
        chosen = [self.examples[int(i)] for i in idx]
        width = max(len(ids) for ids, _ in chosen)

        x = torch.zeros(batch_size, width - 1, dtype=torch.long)
        y = torch.full((batch_size, width - 1), IGNORE_INDEX, dtype=torch.long)
        m = torch.zeros(batch_size, width - 1, dtype=torch.float)
        for row, (ids, mask) in enumerate(chosen):
            t = torch.tensor(ids, dtype=torch.long)
            n = len(ids) - 1
            x[row, :n] = t[:-1]
            y[row, :n] = t[1:]
            # mask[i+1] says "token i+1 is supervised", and predicting it is the
            # job of position i - hence the shift, mirroring the x/y shift.
            m[row, :n] = torch.tensor(mask[1:], dtype=torch.float)
        return x.to(device), y.to(device), m.to(device)
