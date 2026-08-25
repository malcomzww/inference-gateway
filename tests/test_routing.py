"""Tests for the cascade: when it escalates, and when it stops paying."""

from __future__ import annotations

import math

import pytest

from inference_gateway.routing import (
    Tier,
    TieredRouter,
    blended_cost_per_request,
    escalation_breakeven_rate,
    heuristic_confidence,
    logprob_confidence,
)
from inference_gateway.upstream import CHEAP_MODEL, STRONG_MODEL, StubUpstream

CHEAP = Tier(CHEAP_MODEL, 0.15, 0.60)
STRONG = Tier(STRONG_MODEL, 2.50, 10.00)


class _Reply:
    def __init__(self, text: str, p: int = 10, c: int = 5) -> None:
        self.text = text

        class U:
            prompt_tokens = p
            completion_tokens = c

        self.usage = U()


def test_confident_answer_does_not_escalate():
    router = TieredRouter(CHEAP, STRONG)
    escalate, score = router.decide("Paris is the capital of France.")
    assert not escalate and score == 1.0


@pytest.mark.parametrize(
    "text",
    ["I'm not sure, it depends", "", "unclear", "it depends"],
)
def test_hedged_or_empty_answers_escalate(text):
    router = TieredRouter(CHEAP, STRONG)
    escalate, _ = router.decide(text)
    assert escalate


def test_truncated_answer_scores_below_complete_one():
    complete = heuristic_confidence("The answer is four.")
    truncated = heuristic_confidence("The answer is fo")
    assert truncated < complete


def test_logprob_confidence_is_geometric_mean_probability():
    assert logprob_confidence("x", logprobs=[0.0, 0.0]) == pytest.approx(1.0)
    assert logprob_confidence("x", logprobs=[math.log(0.5)]) == pytest.approx(0.5)


def test_logprob_falls_back_to_heuristic_when_absent():
    """Degrading beats crashing on a server that omits an optional field."""
    assert logprob_confidence("I'm not sure, it depends", logprobs=None) < 1.0


async def test_route_escalates_and_charges_for_both_calls():
    """The double charge is the subtlety a cascade's cost model must not lose."""
    router = TieredRouter(CHEAP, STRONG)
    seen: list[str] = []

    async def call(model: str, messages: list[dict]) -> _Reply:
        seen.append(model)
        return _Reply("I'm not sure, it depends" if model == CHEAP_MODEL else "Definitely four.")

    decision = await router.route(call, [{"role": "user", "content": "q"}])
    assert seen == [CHEAP_MODEL, STRONG_MODEL]
    assert decision.escalated and decision.model == STRONG_MODEL
    assert decision.attempts == 2
    assert decision.prompt_tokens == 20, "tokens from both calls must accumulate"


async def test_route_stops_at_cheap_tier_when_confident():
    router = TieredRouter(CHEAP, STRONG)
    seen: list[str] = []

    async def call(model: str, messages: list[dict]) -> _Reply:
        seen.append(model)
        return _Reply("Four, definitively.")

    decision = await router.route(call, [{"role": "user", "content": "q"}])
    assert seen == [CHEAP_MODEL] and not decision.escalated
    assert router.stats.escalation_rate == 0.0


def test_blended_cost_sits_between_the_tiers():
    cheap_only = blended_cost_per_request(CHEAP, STRONG, 0.0, 500, 200)
    half = blended_cost_per_request(CHEAP, STRONG, 0.5, 500, 200)
    always = blended_cost_per_request(CHEAP, STRONG, 1.0, 500, 200)
    assert cheap_only < half < always
    assert cheap_only == CHEAP.cost(500, 200)


def test_full_escalation_costs_more_than_strong_alone():
    """Escalating everything means paying the cheap tier as a pure tax."""
    assert blended_cost_per_request(CHEAP, STRONG, 1.0, 500, 200) > STRONG.cost(500, 200)


def test_escalation_breakeven_rate_is_where_cascade_equals_strong_only():
    rate = escalation_breakeven_rate(CHEAP, STRONG, 500, 200)
    at = blended_cost_per_request(CHEAP, STRONG, rate, 500, 200)
    assert float(at) == pytest.approx(float(STRONG.cost(500, 200)), rel=1e-9)
    assert 0.0 < rate < 1.0


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_escalation_rate_must_be_a_fraction(bad):
    with pytest.raises(ValueError):
        blended_cost_per_request(CHEAP, STRONG, bad, 500, 200)


async def test_stub_upstream_provokes_escalation_without_patching():
    """The stub is keyed on content so the escalation path is reachable."""
    stub = StubUpstream()
    router = TieredRouter(CHEAP, STRONG)

    async def call(model: str, messages: list[dict]) -> _Reply:
        prompt = messages[0]["content"]
        from inference_gateway.upstream import _answer_for

        return _Reply(_answer_for(prompt, model))

    decision = await router.route(call, [{"role": "user", "content": "an obscure question"}])
    assert decision.escalated
    assert stub.call_count == 0  # routed through the callable, not the transport
