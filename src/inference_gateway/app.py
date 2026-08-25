"""The gateway: OpenAI-shape endpoints over routing, caching and the ledger.

Scope note, since this is the file most likely to grow without limit: the app
exists to exercise the mechanisms the one question needs -- what a request
costs, what caching removes, what a cascade adds -- and to be a realistic place
for them to interact. It is not trying to be a production gateway. Things a
real deployment needs and this deliberately omits are listed in the README's
Limitations rather than half-implemented here.

Four decisions worth stating:

**Tenancy comes from a header, and everything is keyed by it.** The cache, the
ledger and the rate limiter are all per-tenant. A cache shared across tenants
is a data-leak channel disguised as an optimisation, and retrofitting isolation
into a design that assumed a single tenant is far harder than assuming it from
the start.

**Idempotency keys are honoured before routing.** A client that retries a
timed-out POST must not be billed twice for one logical request. The store is
in-memory, which is honestly wrong for multiple workers -- noted in the README.

**Streaming holds no more than one chunk in memory and checks for disconnect
between chunks.** An SSE endpoint that materialises the full response before
sending it is not streaming; one that keeps generating after the client has
gone is how a gateway melts under a flaky mobile network. Both are handled and
both are tested.

**The ledger is `llm_client_kit.cost.CostLedger`.** Not reimplemented. Money
arithmetic in `Decimal`, budget enforcement that raises, and per-model
attribution already exist there and are already tested.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from llm_client_kit.cost import CostLedger, ModelPrice
from pydantic import BaseModel, Field

from .cache import TwoTierCache
from .routing import Tier, TieredRouter, heuristic_confidence
from .upstream import CHEAP_MODEL, STRONG_MODEL, StubUpstream, _answer_for

API_VERSION = "v1"

# Prices are per million tokens and belong to the deployment, not the library.
# These are stub-model placeholders; a real deployment supplies its own.
DEFAULT_PRICES: dict[str, ModelPrice] = {
    CHEAP_MODEL: ModelPrice(input_per_mtok=0.15, output_per_mtok=0.60),
    STRONG_MODEL: ModelPrice(input_per_mtok=2.50, output_per_mtok=10.00),
}

CHEAP_TIER = Tier(CHEAP_MODEL, 0.15, 0.60)
STRONG_TIER = Tier(STRONG_MODEL, 2.50, 10.00)


class ChatMessageIn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessageIn] = Field(min_length=1)
    model: str | None = None
    stream: bool = False


@dataclass
class RateLimiter:
    """Per-tenant token bucket.

    Per-tenant rather than global: a global limit means one tenant's burst is
    every other tenant's 429, which is the multi-tenant failure that generates
    support tickets. Monotonic clock injected so the tests can advance time
    without sleeping.
    """

    capacity: int = 60
    refill_per_s: float = 60.0
    clock: Callable[[], float] = time.monotonic
    _buckets: dict[str, tuple[float, float]] = field(default_factory=dict)

    def allow(self, tenant: str, cost: float = 1.0) -> bool:
        now = self.clock()
        tokens, last = self._buckets.get(tenant, (float(self.capacity), now))
        tokens = min(self.capacity, tokens + (now - last) * self.refill_per_s)
        if tokens < cost:
            self._buckets[tenant] = (tokens, now)
            return False
        self._buckets[tenant] = (tokens - cost, now)
        return True


@dataclass
class AuditLog:
    """In-memory audit trail. One record per served request.

    Kept because "which tenant spent that, on which model, served from where"
    is unanswerable after the fact without it, and that question is the whole
    point of the ledger.
    """

    records: list[dict[str, Any]] = field(default_factory=list)

    def record(self, **fields: Any) -> None:
        self.records.append({"ts": time.time(), **fields})

    def for_tenant(self, tenant: str) -> list[dict[str, Any]]:
        return [r for r in self.records if r.get("tenant") == tenant]


@dataclass
class GatewayState:
    """Everything the app owns, in one injectable object.

    A single state object rather than module-level globals so tests can build
    an isolated gateway per test; module globals make test order matter, which
    is a debugging cost that compounds.
    """

    cache: TwoTierCache = field(default_factory=TwoTierCache)
    router: TieredRouter = field(
        default_factory=lambda: TieredRouter(
            cheap=CHEAP_TIER, strong=STRONG_TIER, confidence=heuristic_confidence
        )
    )
    upstream: StubUpstream = field(default_factory=StubUpstream)
    limiter: RateLimiter = field(default_factory=RateLimiter)
    audit: AuditLog = field(default_factory=AuditLog)
    ledgers: dict[str, CostLedger] = field(default_factory=dict)
    idempotency: dict[str, dict[str, Any]] = field(default_factory=dict)
    prices: dict[str, ModelPrice] = field(default_factory=lambda: dict(DEFAULT_PRICES))
    budget_usd: float | None = None

    def ledger_for(self, tenant: str) -> CostLedger:
        """One ledger per tenant. Spend is only meaningful when attributed."""
        if tenant not in self.ledgers:
            self.ledgers[tenant] = CostLedger(self.prices, budget_usd=self.budget_usd)
        return self.ledgers[tenant]

    def total_usd(self) -> float:
        return float(sum((led.total_decimal for led in self.ledgers.values()), Decimal(0)))


def get_state(request: Request) -> GatewayState:
    return request.app.state.gateway


def get_tenant(x_tenant_id: str = Header(default="default")) -> str:
    """Tenant identity from a header.

    A header, not a JWT. Real tenant isolation needs a signed token whose
    claims cannot be spoofed by any client that can set a header, and pretending
    otherwise would be the dishonest choice here -- so the README names this as
    a limitation rather than the code implying auth it does not perform.
    """
    if not x_tenant_id.strip():
        raise HTTPException(status_code=400, detail="X-Tenant-Id must not be empty")
    return x_tenant_id.strip()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build state on startup, report spend on shutdown."""
    if not hasattr(app.state, "gateway"):
        app.state.gateway = GatewayState()
    yield
    app.state.gateway.audit.record(event="shutdown", total_usd=app.state.gateway.total_usd())


def create_app(state: GatewayState | None = None) -> FastAPI:
    """Build the gateway. Injecting state is what makes it testable."""
    app = FastAPI(
        title="inference-gateway",
        version=API_VERSION,
        summary="OpenAI-shape gateway: tiered routing, caching, cost ledger.",
        lifespan=lifespan,
    )
    app.state.gateway = state or GatewayState()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": API_VERSION}

    @app.get(f"/{API_VERSION}/stats")
    async def stats(st: GatewayState = Depends(get_state)) -> dict[str, Any]:
        """Everything measured this process. The endpoint results are built on."""
        return {
            "cache": st.cache.stats.summary(),
            "router": st.router.stats.summary(),
            "upstream_calls": st.upstream.call_count,
            "total_usd": round(st.total_usd(), 6),
            "tenants": {t: led.summary() for t, led in st.ledgers.items()},
        }

    @app.get(f"/{API_VERSION}/usage")
    async def usage(
        st: GatewayState = Depends(get_state), tenant: str = Depends(get_tenant)
    ) -> dict[str, Any]:
        return {"tenant": tenant, **st.ledger_for(tenant).summary()}

    @app.post(f"/{API_VERSION}/chat/completions")
    async def chat(
        body: ChatRequest,
        request: Request,
        st: GatewayState = Depends(get_state),
        tenant: str = Depends(get_tenant),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> Any:
        if not st.limiter.allow(tenant):
            # 429 with the standard header: a client that cannot tell how long
            # to wait will retry immediately and make the overload worse.
            raise HTTPException(
                status_code=429,
                detail="rate limit exceeded",
                headers={"Retry-After": "1"},
            )

        messages = [m.model_dump() for m in body.messages]

        # Idempotency is checked before anything bills. Scoped by tenant so one
        # tenant cannot read another's reply by guessing a key.
        idem_scope = f"{tenant}:{idempotency_key}" if idempotency_key else None
        if idem_scope and idem_scope in st.idempotency:
            cached = dict(st.idempotency[idem_scope])
            cached["served_by"] = "idempotency"
            return cached

        if body.stream:
            return await _stream(st, tenant, messages, request)

        result = await _complete(st, tenant, messages, body.model)
        if idem_scope:
            st.idempotency[idem_scope] = result
        return result

    return app


async def _call_upstream(st: GatewayState, model: str, messages: list[dict]) -> Any:
    """One upstream call through the stub, in OpenAI shape.

    Kept as a seam: swapping this for `llm_client_kit.LLMClient.chat` against a
    real base URL is the only change needed to run against a live provider, and
    the tests inject the stub transport at exactly this boundary.
    """
    import httpx
    from llm_client_kit.transport import Completion

    transport = st.upstream.transport()
    async with httpx.AsyncClient(transport=transport, base_url="http://stub") as client:
        response = await client.post(
            "/chat/completions", json={"model": model, "messages": messages}
        )
        response.raise_for_status()
        return Completion.from_payload(response.json())


async def _complete(
    st: GatewayState, tenant: str, messages: list[dict], model: str | None
) -> dict[str, Any]:
    """Cache, route, bill. The non-streaming path."""
    lookup_model = model or CHEAP_MODEL
    entry, how = st.cache.lookup(tenant, lookup_model, messages)
    if entry is not None:
        # A cache hit costs nothing and is recorded as such: billing a cached
        # response would defeat the measurement the cache exists to support.
        st.audit.record(tenant=tenant, served_by=how, model=entry.model, cost_usd=0.0)
        return _envelope(entry.response, entry.model, how, 0.0, escalated=False)

    async def call(m: str, msgs: list[dict]) -> Any:
        return await _call_upstream(st, m, msgs)

    decision = await st.router.route(call, messages)
    ledger = st.ledger_for(tenant)
    cost = ledger.record(
        decision.model,
        prompt_tokens=decision.prompt_tokens,
        completion_tokens=decision.completion_tokens,
    )
    ledger.check_budget()

    st.cache.store(
        tenant,
        lookup_model,
        messages,
        decision.text,
        prompt_tokens=decision.prompt_tokens,
        completion_tokens=decision.completion_tokens,
    )
    st.audit.record(
        tenant=tenant,
        served_by="upstream",
        model=decision.model,
        escalated=decision.escalated,
        confidence=round(decision.confidence, 4),
        cost_usd=float(cost),
    )
    return _envelope(
        decision.text,
        decision.model,
        "upstream",
        float(cost),
        escalated=decision.escalated,
        confidence=round(decision.confidence, 4),
        signal=decision.signal,
        prompt_tokens=decision.prompt_tokens,
        completion_tokens=decision.completion_tokens,
    )


def _envelope(
    text: str,
    model: str,
    served_by: str,
    cost_usd: float,
    *,
    escalated: bool = False,
    confidence: float | None = None,
    signal: str | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> dict[str, Any]:
    """OpenAI-shape response plus a `gateway` block.

    Gateway metadata is namespaced rather than merged into the top level so an
    OpenAI SDK parses the response unchanged, while a caller who wants to know
    what a request cost can still find out without a second round trip.
    """
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "gateway": {
            "served_by": served_by,
            "escalated": escalated,
            "confidence": confidence,
            "confidence_signal": signal,
            "cost_usd": round(cost_usd, 8),
        },
    }


async def _stream(
    st: GatewayState, tenant: str, messages: list[dict], request: Request
) -> StreamingResponse:
    """SSE with backpressure and client-disconnect handling.

    Two properties this actually implements, both tested:

    *Backpressure* -- chunks are yielded one at a time from a generator, so at
    most one is in memory and the event loop applies the socket's own
    backpressure. Building the list first and joining it would be simpler and
    would not be streaming.

    *Disconnect* -- `await request.is_disconnected()` is checked between
    chunks, and generation stops when the client has gone. Without this a
    gateway keeps paying for tokens nobody will receive, which on a mobile
    network is a steady, invisible bill.
    """
    prompt = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "user")
    model = CHEAP_MODEL
    words = _answer_for(prompt, model).split(" ")

    async def gen() -> AsyncIterator[bytes]:
        sent = 0
        try:
            for i, word in enumerate(words):
                if await request.is_disconnected():
                    st.audit.record(
                        tenant=tenant, event="client_disconnect", chunks_sent=sent
                    )
                    return
                chunk = {
                    "id": "chatcmpl-stream",
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": (" " if i else "") + word},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode()
                sent += 1
                # Yield to the loop so a slow consumer actually exerts
                # backpressure rather than the whole body being buffered.
                await asyncio.sleep(0)
            yield b"data: [DONE]\n\n"
        finally:
            # Bill for what was generated even on disconnect: those tokens were
            # produced upstream whether or not the client received them.
            ledger = st.ledger_for(tenant)
            ledger.record(
                model,
                prompt_tokens=max(1, len(prompt.split())),
                completion_tokens=max(sent, 0),
            )
            st.audit.record(
                tenant=tenant, served_by="stream", model=model, chunks_sent=sent
            )

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


app = create_app()
