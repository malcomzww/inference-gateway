"""The stub upstream. There is no API key here, and that is the design.

CI must never make a live LLM call: a test suite whose result depends on a
third party's availability, rate limits and pricing is not a test suite. So
every path in this repo runs against `StubUpstream`, which is an
`httpx.MockTransport` handler speaking the OpenAI `/chat/completions` shape --
including streaming chunks, `usage`, and the error statuses the retry policy
needs to see.

Injecting at the *transport* layer rather than stubbing the client is the
point. `llm_client_kit.LLMClient` accepts an `httpx.AsyncBaseTransport`, so
the real retry loop, deadline propagation, timeout handling and connection
pooling all stay under test; only the socket is replaced. Stubbing the client
object instead would test nothing but the stub.

The canned answers are deterministic and keyed by prompt content so that
routing tests can reliably provoke a low-confidence answer and force
escalation without any randomness in the suite.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

import httpx

# Prompts containing these substrings get a deliberately hedged, low-confidence
# answer, so the escalation path is reachable from a test without monkeypatching
# the confidence function.
UNSURE_TRIGGERS = ("obscure", "ambiguous", "unsure")

CHEAP_MODEL = "stub-cheap"
STRONG_MODEL = "stub-strong"


def _answer_for(prompt: str, model: str) -> str:
    lowered = prompt.lower()
    if any(t in lowered for t in UNSURE_TRIGGERS) and model == CHEAP_MODEL:
        # Hedged and truncated: scores low under heuristic_confidence.
        return "I'm not sure, it depends"
    return f"[{model}] answer to: {prompt.strip()[:80]}"


def _tokens(text: str) -> int:
    """Word count as a token proxy. Adequate because every cost claim in this
    repo is per-token arithmetic that a proxy scales linearly; a real tokeniser
    would change the absolute counts, not any ratio being asserted."""
    return max(1, len(text.split()))


@dataclass
class StubUpstream:
    """Records requests and answers them in OpenAI shape.

    `fail_times` makes the first N calls return a retryable 503, which is how
    the retry integration is exercised without a flaky network.
    """

    fail_times: int = 0
    latency_s: float = 0.0
    requests: list[dict] = field(default_factory=list)
    _failed: int = 0

    def reset(self) -> None:
        self.requests.clear()
        self._failed = 0

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def models_called(self) -> list[str]:
        return [r.get("model", "") for r in self.requests]

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        self.requests.append(body)

        if self._failed < self.fail_times:
            self._failed += 1
            # 503 is in llm_client_kit's retryable set; Retry-After is short so
            # the suite does not spend real seconds asleep.
            return httpx.Response(503, headers={"Retry-After": "0"}, json={"error": "busy"})

        model = str(body.get("model", CHEAP_MODEL))
        messages: Sequence[dict] = body.get("messages") or []
        prompt = "\n".join(
            str(m.get("content", "")) for m in messages if m.get("role") == "user"
        )
        text = _answer_for(prompt, model)
        prompt_tokens = sum(_tokens(str(m.get("content", ""))) for m in messages)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-stub",
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
                    "completion_tokens": _tokens(text),
                    "total_tokens": prompt_tokens + _tokens(text),
                },
            },
        )

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    def chunks_for(self, prompt: str, model: str = CHEAP_MODEL) -> list[str]:
        """Token-ish chunks for the streaming path, deterministic."""
        return _answer_for(prompt, model).split(" ")
