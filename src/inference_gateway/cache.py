"""Exact and semantic caching, and the false-hit rate that makes one risky.

Two caches with very different risk profiles, deliberately kept in one file so
the asymmetry is impossible to miss:

**Exact cache.** Keyed on a hash of (tenant, model, normalised messages). A hit
is a hit; the only correctness risk is a stale entry, which TTL handles. It is
boring, and boring is the correct amount of interesting for a cache.

**Semantic cache.** Keyed on embedding similarity. A hit is a *guess* that two
differently-worded prompts want the same answer. When that guess is wrong the
gateway returns a confident, fluent, wrong answer that no downstream error
handler will ever flag -- there is no exception, no 500, no retry. That is the
interesting failure in this repo, and the reason `measure_false_hit_rate`
exists and the ADR was written.

The threshold is therefore not a tuning knob to be set by taste. It is a
decision with a measured error rate attached, and this module's job is to make
that rate measurable rather than to hide it behind a default.

**On the embedding.** `HashingEmbedder` is a deterministic bag-of-character-
n-grams projection, not a learned sentence encoder. That choice is a
constraint, honestly stated: it needs no model download, no GPU and no network,
so CI can measure a false-hit rate on every commit instead of skipping the test
that matters. The cost is that it captures lexical overlap, not meaning -- it
will happily score "the capital of France" and "the capital of Finland" as very
similar, because they differ in one word. A real deployment must re-measure
with its own encoder; the *method* here transfers, the specific rate does not.
That limitation is stated in the ADR and the README rather than buried here.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

_WORD_RE = re.compile(r"\w+")


def normalise(text: str) -> str:
    """Casefold and collapse whitespace.

    Applied before exact hashing so that a prompt differing only in trailing
    whitespace is not billed twice. It is *not* applied before embedding --
    the embedder does its own tokenisation, and normalising twice quietly
    changes what the measured false-hit rate is a rate of.
    """
    return " ".join(text.split()).casefold()


def exact_key(tenant: str, model: str, messages: Sequence[dict[str, Any]]) -> str:
    """Stable cache key for an exact match.

    Tenant is part of the key, not a prefix bolted on later. A cache shared
    across tenants is a data-leak channel that looks like a performance
    optimisation, and it is far easier to never build than to retrofit.
    """
    h = hashlib.sha256()
    h.update(tenant.encode())
    h.update(b"\x00")
    h.update(model.encode())
    for m in messages:
        h.update(b"\x00")
        h.update(str(m.get("role", "")).encode())
        h.update(b"\x01")
        h.update(normalise(str(m.get("content", ""))).encode())
    return h.hexdigest()


class HashingEmbedder:
    """Deterministic character-n-gram hashing embedder. No model, no network.

    Character n-grams rather than whole words so that typos and morphology
    ("colour"/"color", "run"/"running") stay near each other -- near-duplicate
    phrasing is the case a semantic cache is meant to catch, and a pure
    bag-of-words embedder misses most of it.

    Deterministic by construction: the same text yields the same vector on
    every machine and every run, which is what lets the false-hit measurement
    be committed as a fixed number rather than a flaky one.
    """

    def __init__(self, dim: int = 256, ngram: int = 4) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if ngram <= 0:
            raise ValueError("ngram must be positive")
        self.dim = dim
        self.ngram = ngram

    def _grams(self, text: str) -> Iterable[str]:
        words = _WORD_RE.findall(text.casefold())
        # Whole words carry the topical signal; character n-grams carry the
        # near-miss signal. Both, because either alone measurably underfits.
        yield from words
        padded = " ".join(words)
        n = self.ngram
        for i in range(max(0, len(padded) - n + 1)):
            yield padded[i : i + n]

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for gram in self._grams(text):
            digest = hashlib.blake2b(gram.encode(), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            # Signed hashing: without the sign bit, unrelated features only
            # ever add, so every pair of long texts drifts toward similar.
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two already-normalised vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return sum(x * y for x, y in zip(a, b, strict=True))


@dataclass
class CacheEntry:
    key: str
    tenant: str
    model: str
    prompt: str
    response: str
    vector: list[float]
    created_at: float
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class CacheStats:
    exact_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0

    @property
    def lookups(self) -> int:
        return self.exact_hits + self.semantic_hits + self.misses

    @property
    def hit_rate(self) -> float:
        return 0.0 if not self.lookups else (self.exact_hits + self.semantic_hits) / self.lookups

    def summary(self) -> dict[str, float | int]:
        return {
            "lookups": self.lookups,
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
        }


@dataclass
class TwoTierCache:
    """Exact lookup first, semantic fallback, both scoped per tenant.

    Exact is tried first because it is free and cannot be wrong. The semantic
    tier only ever sees prompts the exact tier missed, which keeps the risky
    path off the majority of traffic in any workload with real repetition.

    **`semantic_enabled` defaults to False, because of a measurement.** On the
    pair set in `pairs.py`, this embedder's separation AUC is 0.03 -- far
    *below* the 0.5 of random guessing, meaning it reliably scores the pairs
    that must not be merged above the pairs that should be. The best gain any
    threshold achieves is 0.00. Shipping the tier on by default would mean
    shipping a component measured to be incapable of helping, which is the
    failure the measurement existed to catch. See
    `docs/adr/0001-semantic-cache-false-hit-rate.md`.

    The code path is kept, not deleted: it is what the measurement runs
    against, and swapping in a real sentence encoder is a one-argument change
    for anyone who re-measures and finds an AUC that justifies enabling it.

    `threshold` likewise has no safe default -- the right value depends on the
    encoder and on how costly a wrong answer is in the caller's domain. The
    0.95 here is the lowest value at which *this* embedder produced no false
    hits on *this* synthetic set; it is not transferable evidence.
    """

    threshold: float = 0.95
    ttl_s: float = 3600.0
    max_entries: int = 4096
    embedder: HashingEmbedder = field(default_factory=HashingEmbedder)
    clock: Callable[[], float] = time.monotonic
    # Off by default: see the class docstring and ADR 0001. Opt in only after
    # re-measuring with your own encoder.
    semantic_enabled: bool = False
    stats: CacheStats = field(default_factory=CacheStats)
    _exact: dict[str, CacheEntry] = field(default_factory=dict)
    _by_tenant: dict[str, list[CacheEntry]] = field(default_factory=dict)

    def _expired(self, entry: CacheEntry) -> bool:
        return self.clock() - entry.created_at > self.ttl_s

    def _evict(self, tenant: str) -> None:
        """Oldest-first eviction within a tenant.

        Per-tenant rather than global so a noisy tenant cannot flush everyone
        else's entries -- a global LRU makes one tenant's burst into every
        other tenant's latency regression.
        """
        bucket = self._by_tenant.get(tenant)
        if not bucket or len(bucket) <= self.max_entries:
            return
        overflow = len(bucket) - self.max_entries
        for entry in bucket[:overflow]:
            self._exact.pop(entry.key, None)
        del bucket[:overflow]

    def lookup(
        self, tenant: str, model: str, messages: Sequence[dict[str, Any]]
    ) -> tuple[CacheEntry | None, str]:
        """Return `(entry, how)` where `how` is exact, semantic or miss."""
        key = exact_key(tenant, model, messages)
        entry = self._exact.get(key)
        if entry is not None and not self._expired(entry):
            self.stats.exact_hits += 1
            return entry, "exact"
        if entry is not None:
            self._forget(entry)

        if not self.semantic_enabled:
            self.stats.misses += 1
            return None, "miss"

        prompt = self._prompt_text(messages)
        best, score = self._nearest(tenant, model, prompt)
        if best is not None and score >= self.threshold:
            self.stats.semantic_hits += 1
            return best, "semantic"

        self.stats.misses += 1
        return None, "miss"

    def _nearest(self, tenant: str, model: str, prompt: str) -> tuple[CacheEntry | None, float]:
        bucket = self._by_tenant.get(tenant, [])
        if not bucket:
            return None, 0.0
        vec = self.embedder.embed(prompt)
        best: CacheEntry | None = None
        best_score = -1.0
        for entry in list(bucket):
            if entry.model != model:
                # Never serve one model's answer for another's request: the
                # whole point of tiered routing is that the models differ.
                continue
            if self._expired(entry):
                self._forget(entry)
                continue
            score = cosine(vec, entry.vector)
            if score > best_score:
                best, best_score = entry, score
        return best, max(best_score, 0.0)

    def similarity(self, a: str, b: str) -> float:
        """Similarity of two raw prompts, for measurement and tests."""
        return cosine(self.embedder.embed(a), self.embedder.embed(b))

    def _forget(self, entry: CacheEntry) -> None:
        self._exact.pop(entry.key, None)
        bucket = self._by_tenant.get(entry.tenant)
        if bucket and entry in bucket:
            bucket.remove(entry)

    @staticmethod
    def _prompt_text(messages: Sequence[dict[str, Any]]) -> str:
        """Embed the user turns only.

        System prompts are usually identical across a tenant's traffic, so
        including them drags every pair's similarity upward and makes a fixed
        threshold mean different things for different system prompts.
        """
        parts = [
            str(m.get("content", "")) for m in messages if str(m.get("role", "")) == "user"
        ]
        if not parts:
            parts = [str(m.get("content", "")) for m in messages]
        return "\n".join(parts)

    def store(
        self,
        tenant: str,
        model: str,
        messages: Sequence[dict[str, Any]],
        response: str,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> CacheEntry:
        prompt = self._prompt_text(messages)
        entry = CacheEntry(
            key=exact_key(tenant, model, messages),
            tenant=tenant,
            model=model,
            prompt=prompt,
            response=response,
            vector=self.embedder.embed(prompt),
            created_at=self.clock(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        self._exact[entry.key] = entry
        self._by_tenant.setdefault(tenant, []).append(entry)
        self._evict(tenant)
        return entry

    def clear(self) -> None:
        self._exact.clear()
        self._by_tenant.clear()


@dataclass(frozen=True)
class LabelledPair:
    """Two prompts and the ground truth about whether one answer serves both.

    `equivalent=False` pairs are the ones that matter: they are the traps a
    semantic cache falls into. A pair set of only paraphrases measures
    nothing, because a cache that returns a hit for everything scores
    perfectly on it.
    """

    a: str
    b: str
    equivalent: bool


@dataclass(frozen=True)
class FalseHitReport:
    """Confusion counts at one threshold, plus the two rates that matter."""

    threshold: float
    true_hits: int
    false_hits: int
    true_misses: int
    false_misses: int

    @property
    def false_hit_rate(self) -> float:
        """Of the non-equivalent pairs, the fraction wrongly served from cache.

        This is the number the ADR is about. A false hit is a silently wrong
        answer; a false miss is only a wasted API call. They are not
        symmetric, and a single accuracy figure that averages them is
        precisely the summary that hides the risk.
        """
        denom = self.false_hits + self.true_misses
        return 0.0 if denom == 0 else self.false_hits / denom

    @property
    def true_hit_rate(self) -> float:
        """Of the equivalent pairs, the fraction correctly served from cache."""
        denom = self.true_hits + self.false_misses
        return 0.0 if denom == 0 else self.true_hits / denom

    def summary(self) -> dict[str, float]:
        return {
            "threshold": round(self.threshold, 4),
            "false_hit_rate": round(self.false_hit_rate, 4),
            "true_hit_rate": round(self.true_hit_rate, 4),
            "false_hits": self.false_hits,
            "true_hits": self.true_hits,
        }


def measure_false_hit_rate(
    pairs: Sequence[LabelledPair],
    threshold: float,
    embedder: HashingEmbedder | None = None,
) -> FalseHitReport:
    """Score a labelled pair set at one threshold.

    Deliberately a free function over a labelled set rather than a method on
    the cache: the measurement is about the *embedder and threshold*, and
    tying it to a live cache instance would make it depend on insertion order
    and eviction, which are irrelevant to the question being asked.
    """
    emb = embedder or HashingEmbedder()
    true_hits = false_hits = true_misses = false_misses = 0
    for pair in pairs:
        score = cosine(emb.embed(pair.a), emb.embed(pair.b))
        hit = score >= threshold
        if pair.equivalent and hit:
            true_hits += 1
        elif pair.equivalent:
            false_misses += 1
        elif hit:
            false_hits += 1
        else:
            true_misses += 1
    return FalseHitReport(threshold, true_hits, false_hits, true_misses, false_misses)


def separation_auc(
    pairs: Sequence[LabelledPair],
    embedder: HashingEmbedder | None = None,
) -> float:
    """P(equivalent pair scores above non-equivalent pair). Threshold-free.

    The single most useful diagnostic in this module, because it answers a
    question no individual threshold can: *does this embedder rank the right
    pairs higher at all?* Sweeping thresholds on an embedder that cannot
    separate the classes just enumerates bad options and invites picking the
    least-bad-looking one.

    Read it as: 1.0 perfect, 0.5 indistinguishable from coin-flipping, and
    below 0.5 **inverted** -- the embedder systematically scores the pairs it
    must not merge *above* the pairs it should. An inverted score is not a
    tuning problem, and no threshold repairs it. That is what this repo
    measured, and why the semantic tier ships disabled.
    """
    emb = embedder or HashingEmbedder()
    pos = [cosine(emb.embed(p.a), emb.embed(p.b)) for p in pairs if p.equivalent]
    neg = [cosine(emb.embed(p.a), emb.embed(p.b)) for p in pairs if not p.equivalent]
    if not pos or not neg:
        raise ValueError("need at least one pair of each label to compute AUC")
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def best_threshold_gain(
    pairs: Sequence[LabelledPair],
    embedder: HashingEmbedder | None = None,
) -> tuple[float, float]:
    """Best achievable `(true_hit_rate - false_hit_rate)` and where it occurs.

    The decision rule for whether to enable the semantic tier at all. A gain
    of <= 0 means every threshold trades away at least as much correctness as
    it buys in hit rate, so the cache cannot pay for itself at any setting --
    a conclusion that a threshold sweep alone presents as merely a bad row in
    a table, rather than as the disqualifying result it is.
    """
    emb = embedder or HashingEmbedder()
    scored = [(cosine(emb.embed(p.a), emb.embed(p.b)), p.equivalent) for p in pairs]
    n_pos = sum(1 for _, eq in scored if eq)
    n_neg = len(scored) - n_pos
    if not n_pos or not n_neg:
        raise ValueError("need at least one pair of each label")
    best = (0.0, 1.0)
    # Candidate thresholds are the observed scores: the step function only
    # changes at a datum, so scanning a fixed grid can miss the true optimum.
    for t in sorted({s for s, _ in scored} | {1.0}):
        true_hit = sum(1 for s, eq in scored if eq and s >= t) / n_pos
        false_hit = sum(1 for s, eq in scored if not eq and s >= t) / n_neg
        if true_hit - false_hit > best[0]:
            best = (true_hit - false_hit, t)
    return best


def sweep_thresholds(
    pairs: Sequence[LabelledPair],
    thresholds: Sequence[float],
    embedder: HashingEmbedder | None = None,
) -> list[FalseHitReport]:
    """Measure across thresholds so the tradeoff curve is visible.

    Choosing a threshold from a single measurement is guesswork. Choosing it
    from a curve is a decision, and the curve is what the ADR argues over.
    """
    emb = embedder or HashingEmbedder()
    return [measure_false_hit_rate(pairs, t, emb) for t in thresholds]
