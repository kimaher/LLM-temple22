"""FastAPI service: POST /chat, streaming tokens over Server-Sent Events.

    # serve the mock engine (no checkpoint needed)
    LLM_MOCK=1 uvicorn web.backend.main:app --reload --port 8000

    # serve a real checkpoint
    LLM_CHECKPOINT=checkpoints/sft/best.pt uvicorn web.backend.main:app --port 8000

Why SSE and not WebSockets: generation is a one-way stream of text from server
to client over ordinary HTTP.  SSE gives us that with automatic reconnection and
no protocol upgrade, and `fetch` can read it in the browser without a library.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator, List, Literal, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool

from web.backend.engine import ChatEngine, GenerationParams, build_engine

app = FastAPI(title="LLM-temple22", description="A transformer trained from scratch, served over HTTP")

# The Vite dev server runs on a different origin, so the browser needs CORS.
# In production the frontend is served from this same app (see below) and this
# is unnecessary - but harmless.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: ChatEngine = build_engine()


# --------------------------------------------------------------------------- #
# schemas
# --------------------------------------------------------------------------- #
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(min_length=1)
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: int = Field(default=50, ge=0, le=1000)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    seed: Optional[int] = None


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "engine": engine.name, "info": engine.info})


def sse(data: dict, event: str | None = None) -> str:
    """Format one Server-Sent Event frame."""
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/chat")
async def chat(request: Request, body: ChatRequest) -> StreamingResponse:
    """Stream a reply as SSE frames.

    Frames:
        data: {"token": "..."}      one chunk of text
        event: done / data: {...}   end of stream, with token count
        event: error / data: {...}  generation failed
    """
    params = GenerationParams(
        max_new_tokens=body.max_new_tokens,
        temperature=body.temperature,
        top_k=body.top_k,
        top_p=body.top_p,
        repetition_penalty=body.repetition_penalty,
        seed=body.seed,
    )
    messages = [m.model_dump() for m in body.messages]

    async def event_stream() -> AsyncIterator[str]:
        count = 0
        try:
            # Generation is synchronous and CPU/GPU-bound: running it in a
            # worker thread keeps the event loop free to serve other requests
            # and to notice disconnects.
            sync_iter = engine.stream(messages, params)
            async for chunk in iterate_in_threadpool(sync_iter):
                if await request.is_disconnected():
                    break  # user hit stop or closed the tab; stop burning GPU
                count += 1
                yield sse({"token": chunk})
            yield sse({"chunks": count}, event="done")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            yield sse({"message": str(exc)}, event="error")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # tell nginx not to buffer the stream
        },
    )


# --------------------------------------------------------------------------- #
# optionally serve the built frontend, so `uvicorn` alone is a full demo
# --------------------------------------------------------------------------- #
_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if _dist.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")
