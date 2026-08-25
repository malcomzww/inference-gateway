"""Tests for the breakeven arithmetic.

The properties asserted here are the ones the README's claims rest on. The
most important is `test_crossover_is_invariant_to_throughput`: it pins the
counter-intuitive result that faster serving does not lower the crossover, only
raises the ceiling. That is the finding most likely to be misread, so it gets a
test that fails loudly if the model ever stops implying it.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from inference_gateway.cost_model import (
    ApiPricing,
    RequestMix,
    ThroughputAssumption,
    breakeven,
    sensitivity,
    sweep_throughput,
)

API = ApiPricing("gpt-4o-mini-ish", input_per_mtok=0.15, output_per_mtok=0.60)
GPU = ThroughputAssumption("a10g-ish", gpu_cost_per_hour=1.00, output_tokens_per_s=400.0)
MIX = RequestMix(prompt_tokens=500, completion_tokens=200)


def test_api_cost_per_request_is_plain_arithmetic():
    # 500 in @ 0.15/M + 200 out @ 0.60/M = 0.000075 + 0.00012
    assert API.cost_per_request(MIX) == Decimal("0.000195")


def test_monthly_api_cost_scales_linearly():
    a = API.monthly_cost(MIX, 1_000)
    b = API.monthly_cost(MIX, 2_000)
    assert b == a * 2


def test_selfhost_cost_is_flat_in_volume():
    """The flat-vs-linear shape is why a crossover exists at all."""
    assert GPU.monthly_cost() == Decimal("730.00")


def test_crossover_is_where_the_bills_are_equal():
    be = breakeven(API, GPU, MIX)
    assert be.requests_per_month is not None
    at = be.requests_per_month
    api_bill = API.monthly_cost(MIX, int(at))
    # Within a cent: integer truncation of the volume, not model error.
    assert abs(api_bill - GPU.monthly_cost()) < Decimal("0.01")


def test_crossover_is_invariant_to_throughput():
    """Throughput does not appear in the crossover -- only in the capacity.

    The finding most likely to be misquoted. Fixed monthly cost over
    per-request API cost has no throughput term; a faster box reaches the
    crossover volume, it does not move it.
    """
    rows = sweep_throughput(API, GPU, MIX, factors=(0.25, 1.0, 4.0))
    crossovers = {be.crossover_int for _, be in rows}
    assert len(crossovers) == 1, f"crossover moved with throughput: {crossovers}"

    capacities = [be.capacity_requests_per_month for _, be in rows]
    assert capacities[0] < capacities[1] < capacities[2]


def test_slow_box_cannot_reach_its_own_crossover():
    """A crossover above capacity is not an achievable saving."""
    slow = ThroughputAssumption("slow", gpu_cost_per_hour=1.00, output_tokens_per_s=5.0)
    be = breakeven(API, slow, MIX)
    assert not be.feasible
    assert "ABOVE" in be.summary()


def test_utilisation_changes_capacity_not_crossover():
    busy = ThroughputAssumption(
        "busy", gpu_cost_per_hour=1.00, output_tokens_per_s=400.0, utilisation=1.0
    )
    assert breakeven(API, busy, MIX).crossover_int == breakeven(API, GPU, MIX).crossover_int
    assert busy.max_requests_per_month(MIX) > GPU.max_requests_per_month(MIX)


def test_doubling_gpu_price_doubles_the_crossover():
    dear = ThroughputAssumption("dear", gpu_cost_per_hour=2.00, output_tokens_per_s=400.0)
    base = breakeven(API, GPU, MIX).requests_per_month
    doubled = breakeven(API, dear, MIX).requests_per_month
    # Relative comparison, not exact Decimal equality: both values come from a
    # division carried to 28 significant digits, so the last digit differs
    # while the ratio is exact to any precision anyone acts on.
    assert doubled / base == pytest.approx(2.0, rel=1e-20)


def test_cache_hit_rate_raises_the_crossover():
    """Caching cuts the API bill, so self-hosting has to wait longer to win."""
    cached = RequestMix(500, 200, cache_hit_rate=0.5)
    assert (
        breakeven(API, GPU, cached).requests_per_month
        > breakeven(API, GPU, MIX).requests_per_month
    )


def test_free_api_has_no_crossover():
    free = ApiPricing("free", 0.0, 0.0)
    be = breakeven(free, GPU, MIX)
    assert be.requests_per_month is None
    assert "no crossover" in be.summary()


def test_sensitivity_ranks_gpu_price_above_throughput():
    """The actionable ordering: price moves the answer, speed does not."""
    rows = {label: delta for label, _, delta in sensitivity(API, GPU, MIX)}
    assert rows["throughput x2 (assumed)"] == 0, "throughput must not move the crossover"
    assert rows["utilisation 0.5 -> 1.0"] == 0
    assert float(rows["GPU $/hr x2"]) == pytest.approx(1.0, rel=1e-20)
    # Halving the API's output price makes self-hosting wait longer to win.
    assert float(rows["API output price x0.5"]) > 0.4


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_cache_hit_rate_must_be_a_proper_fraction(bad):
    """1.0 is rejected, not clamped: no model is called, so nothing crosses."""
    with pytest.raises(ValueError):
        RequestMix(500, 200, cache_hit_rate=bad)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"output_tokens_per_s": 0.0},
        {"output_tokens_per_s": -1.0},
        {"utilisation": 0.0},
        {"utilisation": 1.5},
        {"gpu_cost_per_hour": -1.0},
    ],
)
def test_impossible_assumptions_are_rejected(kwargs):
    base = {"gpu_cost_per_hour": 1.0, "output_tokens_per_s": 100.0}
    with pytest.raises(ValueError):
        ThroughputAssumption("bad", **{**base, **kwargs})
