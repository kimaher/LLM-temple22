"""Exercise the FastAPI SSE endpoint in-process, without starting a server.

    python scripts/check_api.py [checkpoint] [tokenizer-spec]

With no arguments it checks the mock engine, which is the version to run when
no model has been trained yet.
"""

from __future__ import annotations

import json
import os
import sys


def main() -> int:
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else None
    tokenizer = sys.argv[2] if len(sys.argv) > 2 else None

    if checkpoint:
        os.environ["LLM_CHECKPOINT"] = checkpoint
        if tokenizer:
            os.environ["LLM_TOKENIZER"] = tokenizer
        os.environ.pop("LLM_MOCK", None)
    else:
        os.environ["LLM_MOCK"] = "1"

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from fastapi.testclient import TestClient

    from web.backend.main import app

    client = TestClient(app)
    health = client.get("/health").json()
    print("health:", json.dumps(health))
    assert health["status"] == "ok"
    if checkpoint:
        assert health["engine"] == "model", f"expected the model engine, got {health['engine']}"

    with client.stream(
        "POST",
        "/chat",
        json={"messages": [{"role": "user", "content": "Who wrote Romeo and Juliet?"}],
              "max_new_tokens": 32, "seed": 0},
    ) as response:
        assert response.status_code == 200, response.status_code
        chunks, done = [], False
        event = "message"
        for line in response.iter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                payload = json.loads(line[5:])
                if event == "error":
                    raise SystemExit(f"stream error: {payload}")
                if event == "done":
                    done = True
                else:
                    chunks.append(payload["token"])
                event = "message"

    print(f"streamed {len(chunks)} chunks")
    print("reply:", repr("".join(chunks)))
    assert done, "stream ended without a done event"
    assert chunks, "no tokens were streamed"
    print("API OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
