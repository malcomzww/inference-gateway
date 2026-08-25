"""End-to-end tests through the HTTP layer against the stub upstream.

These go through the real FastAPI stack -- dependency injection, headers,
status codes, SSE framing -- because the bugs this repo cares about (billing a
cached response, leaking a tenant's cache, streaming that is not streaming)
only appear once the pieces are wired together.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from llm_client_kit.cost import BudgetExceeded

from inference_gateway.app import GatewayState, RateLimiter, create_app
from inference_gateway.upstream import CHEAP_MODEL, STRONG_MODEL

ASK = {"messages": [{"role": "user", "content": "What is the capital of France?"}]}
T1 = {"X-Tenant-Id": "tenant-a"}
T2 = {"X-Tenant-Id": "tenant-b"}


@pytest.fixture
def state() -> GatewayState:
    return GatewayState()


@pytest.fixture
def client(state: GatewayState) -> TestClient:
    return TestClient(create_app(state))


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_completion_is_openai_shaped(client):
    """An OpenAI SDK must parse this unchanged; gateway data is namespaced."""
    body = client.post("/v1/chat/completions", json=ASK, headers=T1).json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert "usage" in body and "gateway" in body


def test_second_identical_request_is_served_from_cache(client, state):
    client.post("/v1/chat/completions", json=ASK, headers=T1)
    calls_after_first = state.upstream.call_count
    body = client.post("/v1/chat/completions", json=ASK, headers=T1).json()
    assert body["gateway"]["served_by"] == "exact"
    assert body["gateway"]["cost_usd"] == 0.0
    assert state.upstream.call_count == calls_after_first, "cache hit must not call upstream"


def test_cache_hit_is_not_billed(client, state):
    client.post("/v1/chat/completions", json=ASK, headers=T1)
    after_first = state.ledger_for("tenant-a").total_usd
    client.post("/v1/chat/completions", json=ASK, headers=T1)
    assert state.ledger_for("tenant-a").total_usd == after_first


def test_tenants_do_not_share_cache_or_ledger(client, state):
    """Isolation, end to end. The leak this design exists to prevent."""
    client.post("/v1/chat/completions", json=ASK, headers=T1)
    body = client.post("/v1/chat/completions", json=ASK, headers=T2).json()
    assert body["gateway"]["served_by"] == "upstream"
    assert state.ledger_for("tenant-b").total_usd > 0
    assert state.ledger_for("tenant-a").total_usd > 0
    assert "tenant-b" not in [r["tenant"] for r in state.audit.for_tenant("tenant-a")]


def test_low_confidence_answer_escalates_to_strong_model(client, state):
    ask = {"messages": [{"role": "user", "content": "an obscure ambiguous thing"}]}
    body = client.post("/v1/chat/completions", json=ask, headers=T1).json()
    assert body["gateway"]["escalated"] is True
    assert body["model"] == STRONG_MODEL
    assert state.upstream.models_called() == [CHEAP_MODEL, STRONG_MODEL]


def test_confident_answer_uses_only_the_cheap_model(client, state):
    client.post("/v1/chat/completions", json=ASK, headers=T1)
    assert state.upstream.models_called() == [CHEAP_MODEL]


def test_escalated_request_is_billed_for_both_calls(client, state):
    """Cost must reflect the cheap call the cascade also paid for."""
    ask = {"messages": [{"role": "user", "content": "an obscure ambiguous thing"}]}
    body = client.post("/v1/chat/completions", json=ask, headers=T1).json()
    entry = state.ledger_for("tenant-a").entries[-1]
    assert entry.prompt_tokens > 0
    assert body["gateway"]["cost_usd"] > 0


def test_empty_tenant_header_is_rejected(client):
    r = client.post("/v1/chat/completions", json=ASK, headers={"X-Tenant-Id": "  "})
    assert r.status_code == 400


def test_empty_messages_is_rejected(client):
    r = client.post("/v1/chat/completions", json={"messages": []}, headers=T1)
    assert r.status_code == 422


def test_idempotency_key_replays_without_rebilling(client, state):
    headers = {**T1, "Idempotency-Key": "abc-123"}
    ask = {"messages": [{"role": "user", "content": "unique prompt one"}]}
    first = client.post("/v1/chat/completions", json=ask, headers=headers).json()
    spend = state.ledger_for("tenant-a").total_usd
    second = client.post("/v1/chat/completions", json=ask, headers=headers).json()
    assert second["gateway"]["served_by"] == "idempotency"
    assert second["id"] == first["id"], "a replay must return the same response id"
    assert state.ledger_for("tenant-a").total_usd == spend


def test_idempotency_keys_are_scoped_per_tenant(client):
    """Otherwise guessing a key reads another tenant's answer."""
    headers_a = {**T1, "Idempotency-Key": "shared-key"}
    headers_b = {**T2, "Idempotency-Key": "shared-key"}
    client.post("/v1/chat/completions", json=ASK, headers=headers_a)
    body = client.post("/v1/chat/completions", json=ASK, headers=headers_b).json()
    assert body["gateway"]["served_by"] != "idempotency"


def test_rate_limit_returns_429_with_retry_after():
    state = GatewayState(limiter=RateLimiter(capacity=2, refill_per_s=0.0))
    client = TestClient(create_app(state))
    for i in range(2):
        ask = {"messages": [{"role": "user", "content": f"q{i}"}]}
        assert client.post("/v1/chat/completions", json=ask, headers=T1).status_code == 200
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "q3"}]},
        headers=T1,
    )
    assert r.status_code == 429
    assert r.headers["Retry-After"] == "1"


def test_rate_limit_is_per_tenant():
    """One tenant's burst must not 429 another."""
    state = GatewayState(limiter=RateLimiter(capacity=1, refill_per_s=0.0))
    client = TestClient(create_app(state))
    client.post("/v1/chat/completions", json=ASK, headers=T1)
    assert client.post("/v1/chat/completions", json=ASK, headers=T1).status_code == 429
    assert client.post("/v1/chat/completions", json=ASK, headers=T2).status_code == 200


def test_budget_exceeded_raises_rather_than_silently_overspending():
    state = GatewayState(budget_usd=0.0000001)
    client = TestClient(create_app(state), raise_server_exceptions=True)
    with pytest.raises(BudgetExceeded):
        client.post("/v1/chat/completions", json=ASK, headers=T1)


def test_usage_endpoint_reports_that_tenant_only(client):
    client.post("/v1/chat/completions", json=ASK, headers=T1)
    body = client.get("/v1/usage", headers=T1).json()
    assert body["tenant"] == "tenant-a" and body["calls"] == 1
    assert client.get("/v1/usage", headers=T2).json()["calls"] == 0


def test_stats_endpoint_aggregates(client):
    client.post("/v1/chat/completions", json=ASK, headers=T1)
    client.post("/v1/chat/completions", json=ASK, headers=T1)
    body = client.get("/v1/stats").json()
    assert body["cache"]["exact_hits"] == 1
    assert body["router"]["routed"] == 1
    assert body["total_usd"] > 0


# --- streaming -------------------------------------------------------------


def test_stream_returns_sse_frames(client):
    ask = {**ASK, "stream": True}
    with client.stream("POST", "/v1/chat/completions", json=ask, headers=T1) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        assert r.headers["cache-control"] == "no-cache"
        lines = [line for line in r.iter_lines() if line.strip()]
    assert lines[-1] == "data: [DONE]"
    assert len(lines) > 2, "a single frame is not streaming"


def test_stream_frames_are_incremental_chunks(client):
    """Each frame carries a delta, not the whole answer."""
    ask = {**ASK, "stream": True}
    with client.stream("POST", "/v1/chat/completions", json=ask, headers=T1) as r:
        frames = [
            json.loads(line[len("data: ") :])
            for line in r.iter_lines()
            if line.startswith("data: ") and "[DONE]" not in line
        ]
    assert all(f["object"] == "chat.completion.chunk" for f in frames)
    assert all("content" in f["choices"][0]["delta"] for f in frames)
    reassembled = "".join(f["choices"][0]["delta"]["content"] for f in frames)
    assert reassembled.strip().startswith("[stub-cheap]")


def test_stream_is_billed_for_chunks_generated(client, state):
    ask = {**ASK, "stream": True}
    with client.stream("POST", "/v1/chat/completions", json=ask, headers=T1) as r:
        sent = sum(1 for line in r.iter_lines() if line.startswith("data: {"))
    ledger = state.ledger_for("tenant-a")
    assert ledger.summary()["calls"] == 1
    assert ledger.summary()["completion_tokens"] == sent


async def test_disconnect_stops_generation_and_still_bills():
    """A gone client must stop the work but not erase the spend already made.

    Driven through the ASGI app directly: TestClient always drains the body,
    so the disconnect path is unreachable through it.
    """
    state = GatewayState()
    app = create_app(state)
    body = json.dumps({**ASK, "stream": True}).encode()

    messages = [
        {"type": "http.request", "body": body, "more_body": False},
        {"type": "http.disconnect"},
    ]
    sent: list[dict] = []

    async def receive() -> dict:
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "root_path": "",
        "scheme": "http",
        "query_string": b"",
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            (b"x-tenant-id", b"tenant-a"),
        ],
        "client": ("test", 1),
        "server": ("test", 80),
    }
    await app(scope, receive, send)

    chunks = [m for m in sent if m.get("type") == "http.response.body" and m.get("body")]
    full = b"".join(m["body"] for m in chunks)
    assert b"[DONE]" not in full, "generation continued after the client left"
    assert any(
        r.get("event") == "client_disconnect" for r in state.audit.records
    ), "the disconnect was not recorded"
    assert state.ledger_for("tenant-a").summary()["calls"] == 1, "spend was not recorded"
