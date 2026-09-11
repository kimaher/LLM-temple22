"""A byte-level BPE tokenizer, trained from scratch.

Why write our own instead of just importing one: when you pretrain from scratch
on a small corpus, a 50k general-purpose vocabulary wastes most of the embedding
matrix on tokens the model will never see.  A 4k-8k vocab learned on your own
corpus gives a much better parameters-per-useful-token ratio.

The algorithm (Sennrich et al. 2016, byte-level variant from GPT-2):

  1. Split the text into "words" with a regex, so merges never straddle a word
     boundary (this is what stops the tokenizer learning "the cat" as one token).
  2. Represent each word as a sequence of raw UTF-8 bytes, so any string on
     earth is encodable and there is no <unk>.
  3. Repeatedly find the most frequent adjacent pair of symbols across the whole
     corpus, and replace it everywhere with a new symbol.  Each such merge adds
     one entry to the vocabulary.

Training is done over *unique words with counts* rather than the raw character
stream, and the pair counts are maintained incrementally with a lazy heap, which
is what makes a few thousand merges over a few MB of text take seconds instead
of minutes.
"""

from __future__ import annotations

import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# GPT-2's pre-tokenisation pattern: keeps contractions together, attaches a
# leading space to a word (" the" is one token), and never merges across the
# letter/digit/punctuation/whitespace boundaries.
GPT2_SPLIT_PATTERN = (
    r"'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+"
)

Pair = Tuple[int, int]


class BPETokenizer:
    """Byte-level BPE with a learned merge table and optional special tokens."""

    def __init__(
        self,
        merges: Optional[List[Pair]] = None,
        special_tokens: Optional[Dict[str, int]] = None,
        pattern: str = GPT2_SPLIT_PATTERN,
    ) -> None:
        self.pattern = pattern
        self._compiled = re.compile(pattern)
        # merges in learned order; merge i produces token id 256 + i
        self.merges: List[Pair] = list(merges or [])
        self.ranks: Dict[Pair, int] = {pair: i for i, pair in enumerate(self.merges)}
        self.special_tokens: Dict[str, int] = dict(special_tokens or {})
        self._special_inv: Dict[int, str] = {v: k for k, v in self.special_tokens.items()}
        self._special_re = self._build_special_re()
        self._vocab = self._build_vocab()

    # ------------------------------------------------------------------ #
    # vocabulary bookkeeping
    # ------------------------------------------------------------------ #
    def _build_vocab(self) -> Dict[int, bytes]:
        """id -> byte string.  Ids 0-255 are the raw bytes; the rest are merges."""
        vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            vocab[256 + i] = vocab[a] + vocab[b]
        return vocab

    def _build_special_re(self) -> Optional[re.Pattern]:
        if not self.special_tokens:
            return None
        # Longest-first so "<|assistant|>" wins over any prefix of itself.
        keys = sorted(self.special_tokens, key=len, reverse=True)
        return re.compile("(" + "|".join(re.escape(k) for k in keys) + ")")

    @property
    def n_base_tokens(self) -> int:
        """Vocabulary size before special tokens are added."""
        return 256 + len(self.merges)

    @property
    def vocab_size(self) -> int:
        if self.special_tokens:
            return max(max(self.special_tokens.values()) + 1, self.n_base_tokens)
        return self.n_base_tokens

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        special_tokens: Optional[Sequence[str]] = None,
        pattern: str = GPT2_SPLIT_PATTERN,
        verbose: bool = False,
    ) -> "BPETokenizer":
        """Learn `vocab_size - 256 - len(special_tokens)` merges from `text`."""
        specials = list(special_tokens or [])
        n_merges = vocab_size - 256 - len(specials)
        if n_merges < 0:
            raise ValueError(f"vocab_size must be at least {256 + len(specials)}")

        # 1. unique words + frequencies
        word_freqs = Counter(re.findall(pattern, text))
        words: List[List[int]] = []
        freqs: List[int] = []
        for word, freq in word_freqs.items():
            words.append(list(word.encode("utf-8")))
            freqs.append(freq)
        if verbose:
            print(f"  {len(word_freqs):,} unique words, learning {n_merges:,} merges")

        # 2. initial pair statistics + an index from pair -> words containing it
        pair_counts: Counter[Pair] = Counter() # how many times does this pair occur in the text
        pair_where: Dict[Pair, set] = defaultdict(set) # which word indices contain this pair
        for wi, symbols in enumerate(words):
            f = freqs[wi]
            for pair in zip(symbols, symbols[1:]):
                pair_counts[pair] += f
                pair_where[pair].add(wi)

        # Max-heap over counts.  Entries go stale as counts change, so we
        # re-check the popped entry against the live count and skip if it moved
        # ("lazy deletion") - much cheaper than keeping the heap exact.
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        merges: List[Pair] = []
        while len(merges) < n_merges:
            best: Optional[Pair] = None
            while heap:
                neg_count, pair = heapq.heappop(heap)
                if pair_counts.get(pair, 0) == -neg_count and -neg_count > 0:
                    best = pair
                    break
            if best is None:
                if verbose:
                    print(f"  corpus exhausted after {len(merges)} merges")
                break

            new_id = 256 + len(merges)
            merges.append(best)

            # 3. apply the merge only inside the words that actually contain it,
            #    updating the statistics as we go.
            for wi in list(pair_where[best]):
                symbols = words[wi]
                f = freqs[wi]
                merged = _merge_symbols(symbols, best, new_id)
                if merged == symbols:
                    continue
                for pair in zip(symbols, symbols[1:]):  # retire old pairs
                    pair_counts[pair] -= f
                    if pair_counts[pair] <= 0:
                        pair_counts.pop(pair, None)
                    pair_where[pair].discard(wi)
                words[wi] = merged
                for pair in zip(merged, merged[1:]):  # register new ones
                    pair_counts[pair] += f
                    pair_where[pair].add(wi)
                    heapq.heappush(heap, (-pair_counts[pair], pair))
            pair_counts.pop(best, None)
            pair_where.pop(best, None)

            if verbose and (len(merges) % 500 == 0):
                print(f"  merge {len(merges):,}/{n_merges:,}")

        special_ids = {tok: 256 + len(merges) + i for i, tok in enumerate(specials)}
        return cls(merges=merges, special_tokens=special_ids, pattern=pattern)

    # ------------------------------------------------------------------ #
    # encoding / decoding
    # ------------------------------------------------------------------ #
    def _encode_chunk(self, text_bytes: bytes) -> List[int]:
        """Apply merges to one pre-token, always taking the lowest rank first.

        Order matters: BPE must replay merges in the order they were learned, or
        encoding and training disagree.
        """
        ids = list(text_bytes)
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            pair = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            ids = _merge_symbols(ids, pair, 256 + self.ranks[pair])
        return ids

    def encode_ordinary(self, text: str) -> List[int]:
        """Encode, treating any special-token text as ordinary characters."""
        out: List[int] = []
        for piece in self._compiled.findall(text):
            out.extend(self._encode_chunk(piece.encode("utf-8")))
        return out

    def encode(self, text: str, allowed_special: bool = True) -> List[int]:
        """Encode text; when `allowed_special`, `<|...|>` markers become ids."""
        if not allowed_special or self._special_re is None:
            return self.encode_ordinary(text)
        out: List[int] = []
        for part in self._special_re.split(text):
            if not part:
                continue
            if part in self.special_tokens:
                out.append(self.special_tokens[part])
            else:
                out.extend(self.encode_ordinary(part))
        return out

    def decode(self, ids: Iterable[int]) -> str:
        parts: List[bytes] = []
        for i in ids:
            i = int(i)
            if i in self._special_inv:
                parts.append(self._special_inv[i].encode("utf-8"))
            elif i in self._vocab:
                parts.append(self._vocab[i])
            # ids past the vocabulary (padding of the embedding matrix) decode
            # to nothing rather than crashing a live chat session.
        # A single token can end mid-codepoint, so decode the joined bytes.
        return b"".join(parts).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "pattern": self.pattern,
            "merges": [list(p) for p in self.merges],
            "special_tokens": self.special_tokens,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            merges=[tuple(p) for p in payload["merges"]],
            special_tokens=payload.get("special_tokens", {}),
            pattern=payload.get("pattern", GPT2_SPLIT_PATTERN),
        )


def _merge_symbols(ids: List[int], pair: Pair, new_id: int) -> List[int]:
    """Replace every non-overlapping occurrence of `pair` in `ids` with `new_id`."""
    out: List[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
