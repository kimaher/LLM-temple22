# LLM-temple22

A decoder-only transformer language model built and trained **completely from
scratch** in plain PyTorch - no pretrained weights, no `transformers` model
classes - and served through a streaming web chat interface.

Every stage is its own runnable script, and each one works on a tiny dev-sized
input before you scale it up:

```
corpus -> BPE tokenizer -> token shards -> pretraining -> SFT -> sampling -> FastAPI (SSE) -> React UI
```

## What's implemented

| Layer | Details |
| --- | --- |
| **Model** ([model/](model/)) | RoPE positional embeddings, RMSNorm pre-norm blocks, causal multi-head attention (grouped-query capable), SwiGLU feed-forward, weight tying, static KV cache |
| **Tokenizer** ([tokenizer/](tokenizer/)) | Byte-level BPE trained from scratch (default) with a `tiktoken` backend as a configurable alternative; chat special tokens |
| **Data** ([data/](data/)) | Corpus download, cleaning, tokenization into flat `uint16` shards, memory-mapped random-window batching, SFT conversation batching with loss masks |
| **Training** ([train/](train/)) | Gradient accumulation, cosine LR with warmup, gradient clipping, mixed precision (bf16/fp16), atomic checkpointing + resume, periodic eval and sample previews |
| **Inference** ([inference/](inference/)) | KV-cached generation with greedy / temperature / top-k / top-p sampling, repetition penalty, incremental detokenization, terminal chat client |
| **Web** ([web/](web/)) | FastAPI `POST /chat` streaming Server-Sent Events, mock engine for frontend development, React + TypeScript + Vite chat SPA |

## Quickstart

```bash
pip install -r requirements.txt          # install torch first for your CUDA version

python scripts/smoke_train.py            # 1. does the architecture learn? (~10s)
python scripts/smoke_e2e.py              # 2. does the whole pipeline run? (~15s)
python -m pytest                         # 3. 47 unit tests, no checkpoint needed
```

`smoke_e2e.py` runs every stage end to end at toy scale (tokenizer -> data ->
pretrain -> SFT -> generate -> HTTP API) and writes to `runs/smoke/`, so it
never touches your real checkpoints.

## Full pipeline

```bash
# 1. corpus
python -m data.download --corpus tinyshakespeare      # or --list

# 2. tokenizer (trained on your own corpus; ~1s for 4k merges on 1 MB)
python -m tokenizer.train_tokenizer \
    --input data/raw/tinyshakespeare.txt --vocab-size 4096

# 3. tokenize + shard
python -m data.prepare --input data/raw/tinyshakespeare.txt --name shakespeare

# 4. pretrain
python -m train.pretrain --data data/processed/shakespeare \
    --model-config configs/tiny.json --max-steps 2000 --lr 1e-3

# 5. instruction-tune
python scripts/make_sft_sample.py                      # placeholder instruction set
python -m train.sft --init checkpoints/pretrain/best.pt --data data/sft_sample.jsonl

# 6. talk to it
python -m inference.chat --checkpoint checkpoints/sft/best.pt
```

Model presets live in [configs/](configs/): `dev.json` (0.5M params, for smoke
tests), `tiny.json` (4M), `small.json` (28M, GQA). They are plain JSON dumps of
[`ModelConfig`](model/config.py), and every checkpoint stores the config that
produced it, so a `.pt` file is always self-describing.

## Web app

```bash
# backend - mock engine, no checkpoint required
LLM_MOCK=1 uvicorn web.backend.main:app --reload --port 8000

# backend - real checkpoint
LLM_CHECKPOINT=checkpoints/sft/best.pt uvicorn web.backend.main:app --port 8000

# frontend (proxies /chat and /health to :8000)
cd web/frontend && npm install && npm run dev     # http://localhost:5173
```

Environment variables: `LLM_MOCK`, `LLM_CHECKPOINT`, `LLM_TOKENIZER`,
`LLM_DEVICE`. If the checkpoint is missing or fails to load, the server logs why
and falls back to the mock engine rather than refusing to boot.

`npm run build` writes `web/frontend/dist/`, which the backend then serves at
`/` - so a single `uvicorn` process is a complete demo.

Check the API without a browser:

```bash
python scripts/check_api.py                                  # mock engine
python scripts/check_api.py checkpoints/sft/best.pt          # real checkpoint
```

## Design notes

- **Why a custom BPE by default.** Training from scratch on a small corpus, a
  general-purpose 50k vocabulary would put most of the parameter budget in
  embeddings for tokens the model never sees. A 4k corpus-specific vocab gets
  ~3.3 bytes/token on Shakespeare. Swap it for `--tokenizer tiktoken:gpt2` any
  time; the chat special tokens work identically on both backends.
- **Why flat `.bin` shards.** Pretraining has no "examples", just one long token
  stream sliced into random windows. A flat file memory-maps cleanly, so the OS
  page cache does the data loading and the corpus never has to fit in RAM.
- **Why the KV cache is tested for equivalence.** It's an optimization, so the
  test that matters is that token-by-token decoding produces exactly the logits
  of one full forward pass (`tests/test_model.py`).
- **Why SSE, not WebSockets.** Generation is a one-way stream of text over
  ordinary HTTP. SSE gives that with no protocol upgrade and no client library.
- **Why a mock engine.** The entire frontend - streaming rendering, stop button,
  error states - can be built and tested before the first training run finishes,
  and the API tests need no checkpoint.

## Repo layout

```
configs/          model presets (dev / tiny / small)
model/            config.py, transformer.py, kv_cache.py
tokenizer/        bpe.py, tokenizer.py, train_tokenizer.py
data/             download.py, prepare.py, dataset.py, sft_dataset.py
train/            pretrain.py, sft.py, utils.py
inference/        generate.py, stream.py, chat.py
web/backend/      main.py (FastAPI + SSE), engine.py (model + mock)
web/frontend/     Vite + React + TypeScript SPA
scripts/          smoke_train.py, smoke_e2e.py, check_api.py, make_sft_sample.py
tests/            model, tokenizer, inference, web API
```

## Next steps

Not in this first pass, in rough priority order: scale to a larger corpus
(TinyStories is wired up in `data.download`), distributed training (DDP),
`torch.compile` + flash-attention benchmarking, request batching in the serving
engine, a proper eval harness (held-out perplexity + a few task probes), and
preference tuning (DPO) on top of SFT.
