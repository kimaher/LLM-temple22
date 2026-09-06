"""One tokenizer interface, two backends.

Backends
--------
`bpe:<path>`      our own byte-level BPE (tokenizer/bpe.py), trained on the
                  project's corpus.  **This is the default** - the model is
                  pretrained from scratch on a small corpus, so a small
                  corpus-specific vocabulary spends the embedding matrix far
                  better than a general-purpose 50k one.
`tiktoken:<name>` GPT-2's (or any tiktoken) merge table, with our chat special
                  tokens appended.  Configurable escape hatch: use it when you
                  want a known-good vocabulary and don't care that ~50k
                  embeddings dominate a small model's parameter count.

Both backends expose the same chat special tokens, so the SFT data format, the
inference stop conditions and the web server never need to know which is in use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .bpe import BPETokenizer

# Chat control tokens.  They are ordinary vocabulary entries as far as the model
# is concerned; what makes them "special" is that the tokenizer never produces
# them from raw user text, so a user cannot forge a turn boundary by typing one.
BOS = "<|bos|>"
EOT = "<|eot|>"  # end of turn / end of document
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
SYSTEM = "<|system|>"
SPECIAL_TOKENS: List[str] = [BOS, EOT, USER, ASSISTANT, SYSTEM]

DEFAULT_TOKENIZER_PATH = Path("tokenizer/artifacts/bpe.json")


class BaseTokenizer(ABC):
    """Encode/decode plus the chat template shared by SFT and inference."""

    special_tokens: Dict[str, int]

    # ------------------------------------------------------------------ #
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Number of ids the model's embedding matrix must cover."""

    @abstractmethod
    def encode(self, text: str, allowed_special: bool = True) -> List[int]: ...

    @abstractmethod
    def decode(self, ids: Iterable[int]) -> str: ...

    # ------------------------------------------------------------------ #
    @property
    def bos_id(self) -> int:
        return self.special_tokens[BOS]

    @property
    def eot_id(self) -> int:
        return self.special_tokens[EOT]

    @property
    def assistant_id(self) -> int:
        return self.special_tokens[ASSISTANT]

    @property
    def stop_ids(self) -> List[int]:
        """Ids that should end generation in a chat setting."""
        return [self.eot_id, self.special_tokens[USER], self.bos_id]

    # ------------------------------------------------------------------ #
    def render_chat(
        self,
        messages: Sequence[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> List[int]:
        """Turn [{role, content}, ...] into token ids.

        Layout (identical at SFT time and at inference time - that alignment is
        the whole point of a template):

            <|bos|> <|system|> ... <|eot|> <|user|> ... <|eot|> <|assistant|> ... <|eot|>

        With `add_generation_prompt`, the sequence ends right after the final
        `<|assistant|>`, which is exactly the state the model was trained to
        continue from.
        """
        ids: List[int] = [self.bos_id]
        role_tokens = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}
        for msg in messages:
            role = msg["role"]
            if role not in role_tokens:
                raise ValueError(f"unknown role: {role!r}")
            ids.append(self.special_tokens[role_tokens[role]])
            ids.extend(self.encode(msg["content"], allowed_special=False))
            ids.append(self.eot_id)
        if add_generation_prompt:
            ids.append(self.special_tokens[ASSISTANT])
        return ids


class CustomBPETokenizer(BaseTokenizer):
    """Our own trained BPE (the default backend)."""

    def __init__(self, bpe: BPETokenizer) -> None:
        self.bpe = bpe
        self.special_tokens = bpe.special_tokens
        missing = [t for t in SPECIAL_TOKENS if t not in self.special_tokens]
        if missing:
            raise ValueError(f"tokenizer is missing chat special tokens: {missing}")

    @classmethod
    def load(cls, path: str | Path) -> "CustomBPETokenizer":
        return cls(BPETokenizer.load(path))

    @property
    def vocab_size(self) -> int:
        return self.bpe.vocab_size

    def encode(self, text: str, allowed_special: bool = True) -> List[int]:
        return self.bpe.encode(text, allowed_special=allowed_special)

    def decode(self, ids: Iterable[int]) -> str:
        return self.bpe.decode(ids)


class TiktokenTokenizer(BaseTokenizer):
    """A tiktoken merge table (default: gpt2) plus our chat special tokens."""

    def __init__(self, encoding_name: str = "gpt2") -> None:
        import tiktoken  # imported lazily so the BPE path has no dependency

        base = tiktoken.get_encoding(encoding_name)
        # Append our tokens after the base vocabulary, keeping base ids intact.
        extra = {tok: base.n_vocab + i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.enc = tiktoken.Encoding(
            name=f"{encoding_name}-chat",
            pat_str=base._pat_str,
            mergeable_ranks=base._mergeable_ranks,
            special_tokens={**base._special_tokens, **extra},
        )
        self.special_tokens = extra
        self._allowed = set(extra) | set(base._special_tokens)

    @property
    def vocab_size(self) -> int:
        return self.enc.n_vocab

    def encode(self, text: str, allowed_special: bool = True) -> List[int]:
        if allowed_special:
            return self.enc.encode(text, allowed_special=self._allowed)
        return self.enc.encode(text, disallowed_special=())

    def decode(self, ids: Iterable[int]) -> str:
        return self.enc.decode([int(i) for i in ids])


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #
def load_tokenizer(spec: Optional[str] = None) -> BaseTokenizer:
    """Build a tokenizer from a spec string.

    Accepted forms:
        None                       -> the default trained BPE, if present
        "tiktoken" / "tiktoken:r50k_base"
        "bpe:path/to/bpe.json" or a bare path ending in .json
    """
    if spec is None:
        if DEFAULT_TOKENIZER_PATH.exists():
            return CustomBPETokenizer.load(DEFAULT_TOKENIZER_PATH)
        raise FileNotFoundError(
            f"no tokenizer at {DEFAULT_TOKENIZER_PATH}. Train one with\n"
            f"  python -m tokenizer.train_tokenizer --input data/raw/<corpus>.txt\n"
            f"or pass --tokenizer tiktoken:gpt2"
        )
    if spec.startswith("tiktoken"):
        _, _, name = spec.partition(":")
        return TiktokenTokenizer(name or "gpt2")
    path = spec[4:] if spec.startswith("bpe:") else spec
    return CustomBPETokenizer.load(path)
