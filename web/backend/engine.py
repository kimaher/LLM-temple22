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


class HFEngine:
    """Serves a pretrained Hugging Face model (e.g. Qwen/Qwen3.5-2B) for comparison.

    Requires `pip install transformers accelerate`.  Qwen3.5 is a
    vision-language family and needs a recent transformers release; we only
    ever feed it text.
    """

    name = "hf"

    def __init__(self, model_id: str, device: str = "auto", thinking: bool = False) -> None:
        import torch
        import transformers
        from transformers import AutoTokenizer

        self.device = pick_device(device)
        self.thinking = thinking
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Some multimodal checkpoints keep the chat template on the processor
        # rather than the tokenizer.
        self._template = self.tokenizer
        if getattr(self.tokenizer, "chat_template", None) is None:
            from transformers import AutoProcessor

            self._template = AutoProcessor.from_pretrained(model_id)

        # Text-only models load as CausalLM; Qwen3.5 and other VLMs need one of
        # the multimodal auto classes (whose name depends on the transformers version).
        loaders = [
            getattr(transformers, cls)
            for cls in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForMultimodalLM")
            if hasattr(transformers, cls)
        ]
        errors = []
        for loader in loaders:
            try:
                self.model = loader.from_pretrained(model_id, dtype=dtype).to(self.device)
                break
            except (ValueError, KeyError) as exc:  # "unrecognized configuration class"
                errors.append(f"{loader.__name__}: {exc}")
        else:
            raise RuntimeError(f"could not load {model_id}:\n" + "\n".join(errors))
        self.model.eval()

        self._lock = threading.Lock()
        self.info = {
            "engine": "hf",
            "model": model_id,
            "params": sum(p.numel() for p in self.model.parameters()),
            "device": self.device,
            "thinking": thinking,
        }

    def _prompt_ids(self, messages: Sequence[Message]):
        msgs = list(messages)
        if self._template is not self.tokenizer:
            # processors expect structured content parts
            msgs = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in msgs]
        out = self._template.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=self.thinking,  # Qwen3-style templates; ignored by others
        )
        return out["input_ids"].to(self.device)

    def stream(self, messages: Sequence[Message], params: GenerationParams) -> Iterator[str]:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        input_ids = self._prompt_ids(messages)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        cancelled = threading.Event()

        class _Cancel(StoppingCriteria):
            def __call__(self, *args, **kwargs) -> bool:
                return cancelled.is_set()

        sample = params.temperature > 0
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=params.max_new_tokens,
            do_sample=sample,
            repetition_penalty=params.repetition_penalty,
            streamer=streamer,
            stopping_criteria=StoppingCriteriaList([_Cancel()]),
        )
        if sample:
            gen_kwargs.update(temperature=params.temperature, top_k=params.top_k, top_p=params.top_p)

        with self._lock:
            if params.seed is not None:
                torch.manual_seed(params.seed)
            # generate() blocks, so run it in a thread and drain the streamer here.
            worker = threading.Thread(target=self._generate, args=(gen_kwargs,), daemon=True)
            worker.start()
            try:
                for chunk in streamer:
                    if chunk:
                        yield chunk
            finally:
                # Client disconnected (generator closed) or finished: stop the
                # worker so it doesn't keep burning GPU behind our back.
                cancelled.set()
                worker.join()

    def _generate(self, gen_kwargs: dict) -> None:
        import torch

        with torch.inference_mode():
            self.model.generate(**gen_kwargs)


def build_engine() -> ChatEngine:
    """Pick an engine from the environment.

    LLM_MOCK=1            force the mock engine
    LLM_HF_MODEL=id       serve a pretrained Hugging Face model instead of a checkpoint
    LLM_HF_THINKING=1     enable thinking mode for Qwen3-style chat templates
    LLM_CHECKPOINT=path   checkpoint to serve (default: checkpoints/sft/best.pt)
    LLM_TOKENIZER=spec    tokenizer spec, must match training
    LLM_DEVICE=cuda|cpu   device override
    """
    if os.environ.get("LLM_MOCK", "").lower() in ("1", "true", "yes"):
        print("[engine] LLM_MOCK set - serving mock responses")
        return MockEngine()

    hf_model = os.environ.get("LLM_HF_MODEL")
    if hf_model:
        try:
            engine = HFEngine(
                hf_model,
                device=os.environ.get("LLM_DEVICE", "auto"),
                thinking=os.environ.get("LLM_HF_THINKING", "").lower() in ("1", "true", "yes"),
            )
            print(f"[engine] serving Hugging Face model {hf_model} on {engine.device}")
            return engine
        except Exception as exc:
            print(f"[engine] failed to load {hf_model} ({exc}) - falling back to the mock engine")
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
