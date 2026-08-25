"""Tiered routing: try the cheap model, escalate only when it looks unsure.

The economic case is simple. If a cheap model handles fraction `p` of traffic
acceptably, the blended cost is `p * cheap + (1 - p) * (cheap + strong)` --
note the escalated requests pay for *both* calls. That second term is the part
people forget, and it is why escalation stops paying off long before the
escalation rate reaches 100%. `blended_cost_per_request` computes it honestly,
and `escalation_breakeven_rate` solves for the rate above which the cascade
costs more than simply always calling the strong model.

**The hard part is the confidence signal, not the routing.** Deciding *whether*
to escalate requires knowing whether an answer is good, which is very close to
the original problem. This module is explicit about that: `ConfidenceSignal` is
an interface with deliberately weak built-in implementations, and their
weakness is documented rather than papered over.

`logprob_confidence` is the honest one: mean token logprob is a real signal a
server can return. `heuristic_confidence` is a *stand-in* -- it detects hedging
phrases and truncation, which correlate with low quality but are trivially
gamed and say nothing about factual correctness. It exists because there is no
live model here to produce logprobs, and it is labelled as a stand-in
everywhere it appears so no reader mistakes it for a validated classifier.

What this repo does NOT establish: that escalation improves answer quality.
That needs a labelled quality set and a real model, neither of which exists
here. What it does establish is the cost arithmetic of a cascade, which is
what the one question needs.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Protocol

# Phrases that tend to precede an answer the model is not confident in.
# A stand-in signal, not a validated one -- see the module docstring.
_HEDGES = (
    "i'm not sure",
    "i am not sure",
    "i don't know",
    "i do not know",
    "not certain",
    "unclear",
    "cannot determine",
    "can't determine",
    "as an ai",
    "it depends",
    "i cannot answer",
)


class ConfidenceSignal(Protocol):
    """Maps a candidate answer to [0, 1]. Higher means more trustworthy.

    A Protocol rather than a base class so a caller can supply a real quality
    classifier -- which is what a production deployment should do -- without
    inheriting anything from this module.
    """

    def __call__(self, text: str, *, logprobs: Sequence[float] | None = ...) -> float: ...


def logprob_confidence(text: str, *, logprobs: Sequence[float] | None = None) -> float:
    """Mean token probability, from server-returned logprobs.

    The defensible signal of the two: it reflects the model's own token-level
    uncertainty rather than surface features of the string. Falls back to the
    heuristic when the server returns no logprobs, because a router that
    crashes on a server that omits an optional field is worse than one that
    degrades to a weaker signal -- but the fallback is why callers should log
    which signal actually fired.

    Note the known limitation: token probability measures fluency-confidence,
    not correctness. Models are routinely confident and wrong, so a high score
    here is evidence, not proof.
    """
    if not logprobs:
        return heuristic_confidence(text)
    mean_lp = sum(logprobs) / len(logprobs)
    # exp of a mean logprob is the geometric-mean token probability, already
    # in [0, 1]; no squashing needed and none applied, so the number keeps a
    # meaning a reader can check.
    return max(0.0, min(1.0, math.exp(mean_lp)))


def heuristic_confidence(text: str, *, logprobs: Sequence[float] | None = None) -> float:
    """**Stand-in signal.** Penalises hedging, emptiness and truncation.

    Not a quality classifier. It cannot detect a confidently-stated wrong
    answer, which is the failure mode that matters most, and any model can
    score 1.0 by never hedging. It is here so the escalation path is
    exercisable and testable without a live model, and every result derived
    from it is labelled as using a stand-in.
    """
    stripped = text.strip()
    if not stripped:
        return 0.0
    score = 1.0
    lowered = stripped.lower()
    for hedge in _HEDGES:
        if hedge in lowered:
            score -= 0.4
            break
    # A response ending mid-sentence usually means truncation.
    if stripped[-1] not in ".!?\"')]}`":
        score -= 0.2
    # Very short answers to anything are weak evidence of a non-answer.
    if len(re.findall(r"\w+", stripped)) < 4:
        score -= 0.3
    return max(0.0, min(1.0, score))


@dataclass(frozen=True)
class Tier:
    """One rung of the cascade."""

    model: str
    input_per_mtok: float
    output_per_mtok: float

    def cost(self, prompt_tokens: int, completion_tokens: int) -> Decimal:
        return (
            Decimal(str(self.input_per_mtok)) * Decimal(prompt_tokens)
            + Decimal(str(self.output_per_mtok)) * Decimal(completion_tokens)
        ) / Decimal(1_000_000)


@dataclass(frozen=True)
class RouteDecision:
    """What happened, and why. Returned so the caller can audit the routing.

    `escalated` and `confidence` are on the result rather than logged because
    "what fraction of traffic escalated, and at what confidence" is the
    question that decides whether the cascade is worth keeping -- and it is
    asked long after the logs have rotated.
    """

    model: str
    text: str
    escalated: bool
    confidence: float
    attempts: int
    prompt_tokens: int
    completion_tokens: int
    signal: str = "heuristic (stand-in)"


@dataclass
class RouterStats:
    routed: int = 0
    escalated: int = 0

    @property
    def escalation_rate(self) -> float:
        return 0.0 if not self.routed else self.escalated / self.routed

    def summary(self) -> dict[str, float | int]:
        return {
            "routed": self.routed,
            "escalated": self.escalated,
            "escalation_rate": round(self.escalation_rate, 4),
        }


@dataclass
class TieredRouter:
    """Cheap-first cascade with confidence escalation.

    `threshold` is the confidence below which the answer is retried on the
    strong tier. Like the cache threshold it is a cost/quality dial with no
    universally right value; unlike the cache threshold, getting it wrong
    costs money rather than correctness, which is the easier failure to live
    with. That asymmetry is why escalation defaults on and the semantic cache
    defaults off.
    """

    cheap: Tier
    strong: Tier
    threshold: float = 0.6
    confidence: Callable[..., float] = heuristic_confidence
    stats: RouterStats = field(default_factory=RouterStats)

    def decide(
        self,
        cheap_text: str,
        *,
        logprobs: Sequence[float] | None = None,
    ) -> tuple[bool, float]:
        """Would this cheap answer be escalated? Returns `(escalate, score)`.

        Split out from `route` so the escalation policy can be tested against
        fixed strings without any upstream at all.
        """
        score = self.confidence(cheap_text, logprobs=logprobs)
        return score < self.threshold, score

    async def route(
        self,
        call: Callable[..., object],
        messages: Sequence[dict[str, object]],
        *,
        logprobs: Sequence[float] | None = None,
    ) -> RouteDecision:
        """Run the cascade. `call(model, messages)` returns an upstream reply.

        Takes the upstream as a callable rather than owning a client so the
        routing logic is testable against a stub without patching sockets, and
        so retry/pooling stay the responsibility of `llm_client_kit`.
        """
        self.stats.routed += 1
        first = await call(self.cheap.model, messages)  # type: ignore[misc]
        text = str(getattr(first, "text", "") or "")
        usage = getattr(first, "usage", None)
        p_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
        c_tok = int(getattr(usage, "completion_tokens", 0) or 0)

        escalate, score = self.decide(text, logprobs=logprobs)
        signal = "logprob" if logprobs else "heuristic (stand-in)"
        if not escalate:
            return RouteDecision(
                self.cheap.model, text, False, score, 1, p_tok, c_tok, signal
            )

        self.stats.escalated += 1
        second = await call(self.strong.model, messages)  # type: ignore[misc]
        s_text = str(getattr(second, "text", "") or "")
        s_usage = getattr(second, "usage", None)
        # Tokens accumulate across both calls. Reporting only the second
        # would understate the cascade's cost by exactly the amount that
        # makes cascades look better than they are.
        p_tok += int(getattr(s_usage, "prompt_tokens", 0) or 0)
        c_tok += int(getattr(s_usage, "completion_tokens", 0) or 0)
        return RouteDecision(
            self.strong.model, s_text, True, score, 2, p_tok, c_tok, signal
        )


def blended_cost_per_request(
    cheap: Tier,
    strong: Tier,
    escalation_rate: float,
    prompt_tokens: int,
    completion_tokens: int,
) -> Decimal:
    """Average cost of one cascaded request, counting both calls on escalation.

    The double-charge on escalated requests is the whole subtlety here.
    """
    if not 0.0 <= escalation_rate <= 1.0:
        raise ValueError("escalation_rate must be in [0, 1]")
    c = cheap.cost(prompt_tokens, completion_tokens)
    s = strong.cost(prompt_tokens, completion_tokens)
    rate = Decimal(str(escalation_rate))
    return c + rate * s


def escalation_breakeven_rate(
    cheap: Tier,
    strong: Tier,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Escalation rate above which the cascade costs more than strong-only.

    Solving `cheap + r*strong = strong` gives `r = 1 - cheap/strong`. Above
    this rate you are paying the cheap model as a tax on every request for a
    filter that rarely filters. It is a genuinely useful number: it converts
    "is our cascade worth it" from an opinion into a threshold to compare the
    measured escalation rate against.
    """
    c = cheap.cost(prompt_tokens, completion_tokens)
    s = strong.cost(prompt_tokens, completion_tokens)
    if s <= 0:
        return 1.0
    return float(max(Decimal(0), Decimal(1) - c / s))
