"""Tests for the cache, including the measurement that disqualified it.

`test_embedder_separation_is_inverted` is the load-bearing test in this file.
It asserts the finding the ADR is built on -- that this embedder ranks hard
negatives *above* paraphrases -- and it is written to fail if that ever stops
being true, because the ADR's conclusion (ship the tier disabled) would then
need revisiting rather than silently standing on a stale measurement.
"""

from __future__ import annotations

import pytest

from inference_gateway.cache import (
    HashingEmbedder,
    LabelledPair,
    TwoTierCache,
    best_threshold_gain,
    exact_key,
    measure_false_hit_rate,
    normalise,
    separation_auc,
    sweep_thresholds,
)
from inference_gateway.pairs import counts, labelled_pairs

MSG = [{"role": "user", "content": "What is the capital of France?"}]


def test_exact_hit_after_store():
    cache = TwoTierCache()
    cache.store("t1", "m", MSG, "Paris")
    entry, how = cache.lookup("t1", "m", MSG)
    assert how == "exact"
    assert entry is not None and entry.response == "Paris"


def test_whitespace_and_case_do_not_split_the_key():
    """Otherwise the same prompt is billed twice for a cosmetic difference."""
    a = [{"role": "user", "content": "Hello   World"}]
    b = [{"role": "user", "content": "hello world"}]
    assert exact_key("t", "m", a) == exact_key("t", "m", b)
    assert normalise("  A  B ") == "a b"


def test_tenants_do_not_share_cache_entries():
    """The isolation property. A shared cache is a data-leak channel."""
    cache = TwoTierCache()
    cache.store("tenant-a", "m", MSG, "secret-a")
    entry, how = cache.lookup("tenant-b", "m", MSG)
    assert entry is None and how == "miss"


def test_models_do_not_share_cache_entries():
    """Tiered routing is pointless if a cheap answer is served for a strong one."""
    cache = TwoTierCache(semantic_enabled=True, threshold=0.0)
    cache.store("t", "cheap", MSG, "cheap answer")
    entry, how = cache.lookup("t", "strong", MSG)
    assert entry is None and how == "miss"


def test_entries_expire():
    now = [1000.0]
    cache = TwoTierCache(ttl_s=10.0, clock=lambda: now[0])
    cache.store("t", "m", MSG, "Paris")
    now[0] += 11.0
    entry, how = cache.lookup("t", "m", MSG)
    assert entry is None and how == "miss"


def test_eviction_is_per_tenant():
    """A noisy tenant must not flush a quiet tenant's entries."""
    cache = TwoTierCache(max_entries=2)
    cache.store("quiet", "m", [{"role": "user", "content": "keep me"}], "kept")
    for i in range(5):
        cache.store("noisy", "m", [{"role": "user", "content": f"q{i}"}], f"a{i}")
    entry, _ = cache.lookup("quiet", "m", [{"role": "user", "content": "keep me"}])
    assert entry is not None


def test_semantic_tier_is_off_by_default():
    """Because it was measured to be incapable of helping. See ADR 0001."""
    assert TwoTierCache().semantic_enabled is False


def test_semantic_hit_when_enabled_and_similar():
    cache = TwoTierCache(semantic_enabled=True, threshold=0.5)
    cache.store("t", "m", [{"role": "user", "content": "how do I reverse a list"}], "use [::-1]")
    entry, how = cache.lookup(
        "t", "m", [{"role": "user", "content": "how do I reverse a list?"}]
    )
    assert how == "semantic" and entry is not None


def test_embedder_is_deterministic():
    """A rate that moves between runs cannot be committed or defended."""
    e = HashingEmbedder()
    assert e.embed("hello world") == HashingEmbedder().embed("hello world")


def test_identical_text_scores_one():
    assert TwoTierCache().similarity("abc def", "abc def") == pytest.approx(1.0)


def test_pair_set_is_balanced_and_nonempty():
    eq, neq = counts()
    assert eq >= 25 and neq >= 25
    assert eq == neq, "an unbalanced set makes the two rates hard to compare"


def test_embedder_separation_is_inverted():
    """The measured finding behind ADR 0001.

    AUC below 0.5 means the embedder ranks pairs that must not be merged above
    pairs that should be. Not a tuning problem: no threshold fixes an inverted
    ranking.
    """
    auc = separation_auc(labelled_pairs())
    assert auc < 0.5, f"AUC {auc:.3f} is no longer inverted -- revisit ADR 0001"


def test_no_threshold_beats_never_caching():
    """The decision rule. Gain <= 0 means the tier cannot pay for itself."""
    gain, _ = best_threshold_gain(labelled_pairs())
    assert gain <= 0.0, f"a threshold now gains {gain:.3f} -- ADR 0001 needs revisiting"


def test_false_hit_rate_falls_as_threshold_rises():
    """Monotonicity: a stricter threshold cannot admit more false hits."""
    reports = sweep_thresholds(labelled_pairs(), [0.5, 0.7, 0.9, 0.95])
    rates = [r.false_hit_rate for r in reports]
    assert rates == sorted(rates, reverse=True)


def test_low_threshold_produces_many_false_hits():
    """Concretely: a permissive cache serves wrong answers, not just stale ones."""
    report = measure_false_hit_rate(labelled_pairs(), 0.5)
    assert report.false_hit_rate > 0.5


def test_false_hit_rate_ignores_the_equivalent_pairs():
    """It is a rate over non-equivalent pairs only; mixing them hides the risk."""
    pairs = [
        LabelledPair("a b c", "a b c", False),  # identical, will hit -> false hit
        LabelledPair("x y z", "q r s", False),  # unrelated, will miss -> true miss
        LabelledPair("m n o", "m n o", True),  # equivalent, hits; must not count
    ]
    report = measure_false_hit_rate(pairs, 0.99)
    assert report.false_hits == 1
    assert report.false_hit_rate == pytest.approx(0.5)


def test_auc_needs_both_labels():
    with pytest.raises(ValueError):
        separation_auc([LabelledPair("a", "b", True)])
