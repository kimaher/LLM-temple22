"""Incremental detokenisation for streaming output.

Decoding tokens one at a time is not safe: a BPE token is a sequence of *bytes*,
and a multi-byte character (an em dash, an emoji) is often split across two
tokens.  Decoding each token alone would emit a replacement character where the
text should be.

So we keep the ids seen so far, decode the whole prefix, and emit only the newly
completed text - holding back anything that still ends in a partial codepoint
until the next token arrives.
"""

from __future__ import annotations

from typing import List

from tokenizer.tokenizer import BaseTokenizer

REPLACEMENT = "�"


class TokenStreamDecoder:
    def __init__(self, tokenizer: BaseTokenizer) -> None:
        self.tokenizer = tokenizer
        self.ids: List[int] = []
        self.emitted = 0

    def push(self, token_id: int) -> str:
        """Add a token and return the text that became complete because of it."""
        self.ids.append(int(token_id))
        text = self.tokenizer.decode(self.ids)
        if text.endswith(REPLACEMENT):
            return ""  # mid-character: wait for the rest of the bytes
        chunk = text[self.emitted :]
        self.emitted = len(text)
        return chunk

    def flush(self) -> str:
        """Emit whatever is left at the end of a generation."""
        text = self.tokenizer.decode(self.ids)
        chunk = text[self.emitted :]
        self.emitted = len(text)
        return chunk

    def reset(self) -> None:
        self.ids.clear()
        self.emitted = 0
