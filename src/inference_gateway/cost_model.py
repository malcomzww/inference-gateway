"""The breakeven model: at what monthly volume does self-hosting beat the API?

This module is the reason the repo exists. Everything else -- routing, caching,
the ledger, streaming -- exists to feed realistic inputs into the arithmetic
here or to reduce the API side of the comparison.

**Read this before quoting any number out of this file.**

The API side is arithmetic over published, caller-supplied prices: given a
request mix and a price table, the monthly bill is not in dispute.

The self-hosted side is *not* measured on this machine. There is no GPU here.
Every throughput figure is a stated **assumption**, carried through the code in
a type called `ThroughputAssumption` so it cannot be mistaken for a
measurement at any call site. That naming is deliberate and load-bearing: a
modelled crossover presented as a measured one is the single most misleading
thing a cost analysis can do, and renaming the type is the cheapest way to make
the distinction survive a copy-paste into a slide.

The honest output is therefore not "the crossover is N requests/month". It is
"the crossover is N requests/month **if** the box sustains T tokens/sec at U
utilisation, and here is how N moves when T and U move". A single crossover
number hides the fact that the answer is dominated by an input nobody has
verified. `sweep_throughput` and `sensitivity` exist to expose that, and the
generated results lead with the sweep rather than the point estimate.

Costs are `Decimal` throughout, for the reason `llm_client_kit.cost` states:
money that drifts by fractions of a cent per call is money nobody trusts after
a million calls.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal

TOKENS_PER_MTOK = Decimal(1_000_000)
HOURS_PER_MONTH = Decimal(730)  # 365.25 * 24 / 12, the standard billing month


def _dec(value: float | int | str | Decimal) -> Decimal:
    """Convert to Decimal via str, so 0.1 stays 0.1 rather than 0.1000...055."""
    return value if isinstance(value, Decimal) else Decimal(str(value))


@dataclass(frozen=True)
class RequestMix:
    """The shape of an average request, and how often the gateway avoids it.

    Prompt and completion lengths are separate because they are priced
    separately -- output tokens are typically 3-5x input on hosted APIs, so a
    mix stated as a single "average tokens" number can be wrong by more than
    the effect being measured.

    `cache_hit_rate` is the fraction of requests the gateway serves without
    touching any model at all. It cuts the API bill and the self-hosted load
    by the same factor, so it moves the crossover far less than people expect
    -- see `sensitivity`, which shows exactly that. It is included anyway
    because leaving it out invites the reader to assume it was forgotten.
    """

    prompt_tokens: int = 500
    completion_tokens: int = 200
    cache_hit_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0 or self.completion_tokens < 0:
            raise ValueError("token counts must be non-negative")
        if not 0.0 <= self.cache_hit_rate < 1.0:
            # 1.0 is excluded, not clamped: a 100% hit rate means no model is
            # ever called, the crossover is undefined rather than infinite,
            # and silently clamping it would produce a confident wrong answer.
            raise ValueError("cache_hit_rate must be in [0, 1)")

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def miss_fraction(self) -> Decimal:
        """Fraction of requests that actually reach a model."""
        return Decimal(1) - _dec(self.cache_hit_rate)


@dataclass(frozen=True)
class ApiPricing:
    """USD per million tokens for the hosted API being compared against.

    Supplied by the caller for the same reason `llm_client_kit.cost` does not
    ship a price table: a hardcoded price is stale the day a provider
    reprices, and a stale price silently biases the crossover.
    """

    name: str
    input_per_mtok: float
    output_per_mtok: float

    def cost_per_request(self, mix: RequestMix) -> Decimal:
        """USD for one request that actually reaches the API."""
        return (
            _dec(self.input_per_mtok) * Decimal(mix.prompt_tokens)
            + _dec(self.output_per_mtok) * Decimal(mix.completion_tokens)
        ) / TOKENS_PER_MTOK

    def monthly_cost(self, mix: RequestMix, requests_per_month: int) -> Decimal:
        """USD/month. Cache hits cost nothing because they never leave the box."""
        billable = _dec(requests_per_month) * mix.miss_fraction
        return self.cost_per_request(mix) * billable


@dataclass(frozen=True)
class ThroughputAssumption:
    """An **assumed** self-hosted serving rate. NOT measured in this repo.

    Named for what it is. Every field that feeds the self-hosted side of the
    comparison lives on this type, so a reader who greps for
    "ThroughputAssumption" finds the complete list of things nobody here
    verified. There is no GPU on the machine that produced these results; the
    numbers come from published vLLM/TGI benchmarks and vendor list prices,
    and they are inputs to be replaced, not findings to be cited.

    `output_tokens_per_s` is deliberately the *output* rate. Prefill and
    decode have very different costs per token -- prefill is compute-bound and
    batches well, decode is memory-bandwidth-bound and does not -- and decode
    is what bounds a real serving deployment. Modelling on total tokens would
    flatter self-hosting by counting cheap prefill at the decode rate, which
    biases the answer in the direction the author might want it to go. That is
    exactly the bias to design against.

    `utilisation` is the fraction of the hour the box is actually serving.
    A GPU billed by the hour costs the same idle, so this is the single input
    that most often turns a promising self-host case into a bad one: real
    traffic is diurnal, and a box sized for peak sits idle at night.
    """

    name: str
    gpu_cost_per_hour: float
    output_tokens_per_s: float
    utilisation: float = 0.5
    source: str = "assumption, not measured on this machine"

    def __post_init__(self) -> None:
        if self.gpu_cost_per_hour < 0:
            raise ValueError("gpu_cost_per_hour must be non-negative")
        if self.output_tokens_per_s <= 0:
            raise ValueError("output_tokens_per_s must be positive")
        if not 0.0 < self.utilisation <= 1.0:
            raise ValueError("utilisation must be in (0, 1]")

    @property
    def effective_output_tokens_per_hour(self) -> Decimal:
        """Output tokens the box actually delivers per billed hour."""
        return _dec(self.output_tokens_per_s) * Decimal(3600) * _dec(self.utilisation)

    @property
    def cost_per_output_mtok(self) -> Decimal:
        """USD per million output tokens, at the assumed rate and utilisation.

        This is the number to compare against an API's output price, and the
        comparison is only as good as the two assumptions behind it.
        """
        per_hour = self.effective_output_tokens_per_hour
        return _dec(self.gpu_cost_per_hour) / per_hour * TOKENS_PER_MTOK

    def monthly_cost(self, replicas: int = 1) -> Decimal:
        """USD/month to keep `replicas` boxes running.

        Flat in volume by construction: that flatness against a linear API
        bill is the entire shape of the breakeven, and why one exists at all.
        """
        return _dec(self.gpu_cost_per_hour) * HOURS_PER_MONTH * Decimal(replicas)

    def max_requests_per_month(self, mix: RequestMix, replicas: int = 1) -> Decimal:
        """Capacity ceiling: requests/month one deployment can actually serve.

        The crossover is meaningless above this line -- you would be comparing
        an API bill against a box that cannot carry the load. `breakeven`
        reports both, and the results file prints the ceiling next to the
        crossover for exactly that reason.
        """
        if mix.completion_tokens == 0:
            return Decimal("Infinity")
        per_month = self.effective_output_tokens_per_hour * HOURS_PER_MONTH * Decimal(replicas)
        served = per_month / Decimal(mix.completion_tokens)
        if mix.miss_fraction == 0:
            return Decimal("Infinity")
        return served / mix.miss_fraction


@dataclass(frozen=True)
class Breakeven:
    """The answer, with enough context that it cannot be quoted bare."""

    requests_per_month: Decimal | None
    api_cost_per_request: Decimal
    selfhost_monthly_cost: Decimal
    capacity_requests_per_month: Decimal
    feasible: bool
    api: str
    assumption: ThroughputAssumption

    @property
    def crossover_int(self) -> int | None:
        if self.requests_per_month is None:
            return None
        return int(self.requests_per_month.to_integral_value(rounding="ROUND_CEILING"))

    def summary(self) -> str:
        if self.requests_per_month is None:
            return "no crossover: the API is cheaper at every volume"
        if not self.feasible:
            return (
                f"crossover at {self.crossover_int:,} req/month is ABOVE the "
                f"{int(self.capacity_requests_per_month):,} req/month this "
                f"deployment can serve -- it needs more replicas first"
            )
        return f"crossover at {self.crossover_int:,} req/month"


def breakeven(
    api: ApiPricing,
    assumption: ThroughputAssumption,
    mix: RequestMix,
    *,
    replicas: int = 1,
) -> Breakeven:
    """Monthly request volume at which self-hosting stops costing more.

    The arithmetic is a one-liner -- fixed monthly cost divided by per-request
    API cost -- and that simplicity is the point. The difficulty in this
    question was never the algebra; it is that the divisor depends on
    `ThroughputAssumption` values nobody here has measured. Read
    `sweep_throughput` before believing any single output of this function.
    """
    per_request = api.cost_per_request(mix) * mix.miss_fraction
    fixed = assumption.monthly_cost(replicas)
    capacity = assumption.max_requests_per_month(mix, replicas)

    if per_request <= 0:
        # A free API is never beaten by a paid box. Returning None rather
        # than infinity keeps callers from formatting "inf req/month".
        return Breakeven(None, per_request, fixed, capacity, False, api.name, assumption)

    crossover = fixed / per_request
    return Breakeven(
        requests_per_month=crossover,
        api_cost_per_request=api.cost_per_request(mix),
        selfhost_monthly_cost=fixed,
        capacity_requests_per_month=capacity,
        feasible=crossover <= capacity,
        api=api.name,
        assumption=assumption,
    )


def sweep_throughput(
    api: ApiPricing,
    assumption: ThroughputAssumption,
    mix: RequestMix,
    factors: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
    *,
    replicas: int = 1,
) -> list[tuple[float, Breakeven]]:
    """Recompute the crossover across a range of assumed throughputs.

    This is the honest presentation of the result. The crossover volume is
    *invariant* to throughput -- fixed monthly cost divided by per-request API
    cost has no throughput term -- but the *feasibility* of that crossover is
    not: a slower box cannot reach the volume where it would have paid off.
    The sweep makes that separation visible, and the results script asserts
    it, so the reader is never handed a crossover the hardware cannot serve.
    """
    out: list[tuple[float, Breakeven]] = []
    for f in factors:
        scaled = replace(assumption, output_tokens_per_s=assumption.output_tokens_per_s * f)
        out.append((f, breakeven(api, scaled, mix, replicas=replicas)))
    return out


def sensitivity(
    api: ApiPricing,
    assumption: ThroughputAssumption,
    mix: RequestMix,
    *,
    replicas: int = 1,
) -> list[tuple[str, Decimal | None, Decimal | None]]:
    """How far each input moves the crossover, one input at a time.

    Returns `(label, crossover, delta_vs_base)`. The ordering of magnitudes
    here is the actually useful finding in this file: GPU hourly rate and the
    API's output price move the crossover proportionally, while throughput
    does not move it at all. Anyone optimising serving speed to reach
    breakeven sooner is optimising the wrong variable -- speed buys headroom,
    not a lower crossover.
    """
    base = breakeven(api, assumption, mix, replicas=replicas).requests_per_month

    def delta(x: Decimal | None) -> Decimal | None:
        if x is None or base is None or base == 0:
            return None
        return (x - base) / base

    rows: list[tuple[str, Decimal | None, Decimal | None]] = []

    def add(label: str, be: Breakeven) -> None:
        rows.append((label, be.requests_per_month, delta(be.requests_per_month)))

    add("baseline", breakeven(api, assumption, mix, replicas=replicas))
    add(
        "GPU $/hr x2",
        breakeven(
            api,
            replace(assumption, gpu_cost_per_hour=assumption.gpu_cost_per_hour * 2),
            mix,
            replicas=replicas,
        ),
    )
    add(
        "throughput x2 (assumed)",
        breakeven(
            api,
            replace(assumption, output_tokens_per_s=assumption.output_tokens_per_s * 2),
            mix,
            replicas=replicas,
        ),
    )
    add(
        "utilisation 0.5 -> 1.0",
        breakeven(api, replace(assumption, utilisation=1.0), mix, replicas=replicas),
    )
    add(
        "API output price x0.5",
        breakeven(
            replace(api, output_per_mtok=api.output_per_mtok * 0.5),
            assumption,
            mix,
            replicas=replicas,
        ),
    )
    add(
        "cache hit rate 0 -> 0.5",
        breakeven(api, assumption, replace(mix, cache_hit_rate=0.5), replicas=replicas),
    )
    return rows
