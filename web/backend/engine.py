"""Inference engines behind the HTTP API.

Two implementations share one interface:

* `ModelEngine` - loads a checkpoint and streams real tokens.
* `MockEngine`  - streams canned text, word by word, with a small delay.

The mock is not a toy: it means the entire frontend (streaming rendering,
cancellation, error states) can be built and tested before the first training
run finishes, and it keeps the web tests independent of any checkpoint.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Iterator, List, Optional, Protocol, Sequence

from inference.generate import SamplingConfig, generate_stream
from inference.stream import TokenStreamDecoder
from tokenizer.tokenizer import load_tokenizer
from train.utils import load_checkpoint, pick_device

Message = dict  # {"role": "user" | "assistant" | "system", "content": str}


@dataclass
class GenerationParams:
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    seed: Optional[int] = None


class ChatEngine(Protocol):
    name: str
    info: dict

    def stream(self, messages: Sequence[Message], params: GenerationParams) -> Iterator[str]:
        """Yield text chunks (not tokens: already detokenised) as they are produced."""
        ...


class MockEngine:
    """Deterministic fake responses, so the UI can be developed independently."""

    name = "mock"

    def __init__(self, delay: float = 0.03) -> None:
        self.delay = delay
        self.info = {"engine": "mock", "params": 0, "note": "no checkpoint loaded"}

    def stream(self, messages: Sequence[Message], params: GenerationParams) -> Iterator[str]:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        reply = (
            f'This is a mock response. You said: "{last_user}". '
            "Train a model and restart the server without LLM_MOCK to get real output."
        )
        for word in reply.split(" "):
            time.sleep(self.delay)
            yield word + " "


class ModelEngine:
    """Wraps a trained checkpoint + tokenizer for streaming chat."""

    name = "model"

    def __init__(
        self,
        checkpoint: str,
        tokenizer_spec: Optional[str] = None,
        device: str = "auto",
    ) -> None:
        self.device = pick_device(device)
        self.model, ckpt = load_checkpoint(checkpoint, device=self.device)
        self.model.eval()
        self.tokenizer = load_tokenizer(tokenizer_spec)
        # One model, many requests: generation mutates a KV cache, so serialise
        # requests with a lock.  (Real batching would mean a scheduler that
        # merges concurrent requests into one forward pass - out of scope here.)
        self._lock = threading.Lock()
        self.info = {
            "engine": "model",
            "checkpoint": checkpoint,
            "step": ckpt.get("step", 0),
            "params": self.model.num_params(),
            "device": self.device,
            "max_seq_len": self.model.config.max_seq_len,
            "vocab_size": self.model.config.vocab_size,
        }

    def stream(self, messages: Sequence[Message], params: GenerationParams) -> Iterator[str]:
        prompt_ids = self.tokenizer.render_chat(list(messages), add_generation_prompt=True)
        sampling = SamplingConfig(
            max_new_tokens=params.max_new_tokens,
            temperature=params.temperature,
            top_k=params.top_k,
            top_p=params.top_p,
            repetition_penalty=params.repetition_penalty,
            stop_ids=self.tokenizer.stop_ids,
            seed=params.seed,
        )
        decoder = TokenStreamDecoder(self.tokenizer)
        with self._lock:
            for token in generate_stream(self.model, prompt_ids, sampling, device=self.device):
                chunk = decoder.push(token)
                if chunk:
                    yield chunk
            tail = decoder.flush()
            if tail:
                yield tail


def build_engine() -> ChatEngine:
    """Pick an engine from the environment.

    LLM_MOCK=1            force the mock engine
    LLM_CHECKPOINT=path   checkpoint to serve (default: checkpoints/sft/best.pt)
    LLM_TOKENIZER=spec    tokenizer spec, must match training
    LLM_DEVICE=cuda|cpu   device override
    """
    if os.environ.get("LLM_MOCK", "").lower() in ("1", "true", "yes"):
        print("[engine] LLM_MOCK set - serving mock responses")
        return MockEngine()

    checkpoint = os.environ.get("LLM_CHECKPOINT", "checkpoints/sft/best.pt")
    if not os.path.exists(checkpoint):
        print(f"[engine] no checkpoint at {checkpoint} - falling back to the mock engine")
        return MockEngine()

    try:
        engine = ModelEngine(
            checkpoint,
            tokenizer_spec=os.environ.get("LLM_TOKENIZER"),
            device=os.environ.get("LLM_DEVICE", "auto"),
        )
        print(f"[engine] serving {checkpoint} on {engine.device}")
        return engine
    except Exception as exc:  # a broken checkpoint shouldn't stop the server booting
        print(f"[engine] failed to load {checkpoint} ({exc}) - falling back to the mock engine")
        return MockEngine()
