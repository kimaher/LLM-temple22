"""API tests against the mock engine - no checkpoint required."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(monkeypatch_module=None):
    """A TestClient wired to the mock engine, with no artificial delay."""
    import web.backend.main as main
    from web.backend.engine import MockEngine

    main.engine = MockEngine(delay=0.0)
    return TestClient(main.app)


def read_events(response) -> list[tuple[str, dict]]:
    """Parse an SSE response body into (event_name, payload) pairs."""
    events = []
    for frame in response.text.split("\n\n"):
        if not frame.strip():
            continue
        name = "message"
        data = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
        if data:
            events.append((name, json.loads("\n".join(data))))
    return events


def test_health(client: TestClient):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["engine"] == "mock"


def test_chat_streams_tokens_then_done(client: TestClient):
    res = client.post("/chat", json={"messages": [{"role": "user", "content": "hello"}]})
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")

    events = read_events(res)
    assert events[-1][0] == "done"
    tokens = [payload["token"] for name, payload in events if name == "message"]
    assert len(tokens) > 1  # streamed in pieces, not one blob
    assert "hello" in "".join(tokens)  # the mock echoes the prompt


def test_chat_rejects_empty_messages(client: TestClient):
    assert client.post("/chat", json={"messages": []}).status_code == 422


def test_chat_validates_sampling_params(client: TestClient):
    bad = {"messages": [{"role": "user", "content": "hi"}], "temperature": 99}
    assert client.post("/chat", json=bad).status_code == 422


def test_chat_rejects_unknown_role(client: TestClient):
    bad = {"messages": [{"role": "root", "content": "hi"}]}
    assert client.post("/chat", json=bad).status_code == 422


def test_engine_errors_become_error_events(client: TestClient):
    """A failure mid-generation is reported in-stream, not as a 500."""
    import web.backend.main as main

    class Broken:
        name = "broken"
        info = {}

        def stream(self, messages, params):
            yield "partial "
            raise RuntimeError("boom")

    original, main.engine = main.engine, Broken()
    try:
        res = client.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})
        events = read_events(res)
        assert events[-1][0] == "error"
        assert "boom" in events[-1][1]["message"]
    finally:
        main.engine = original
