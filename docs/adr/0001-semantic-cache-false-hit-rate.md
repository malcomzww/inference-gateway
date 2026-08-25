# ADR 0001 — The semantic cache ships disabled

- **Status:** Accepted
- **Date:** 2026-08-25
- **Decision owner:** Malcom Mudhungwaza
- **Supersedes:** nothing
- **Evidence:** `results/cache.md` (generated), `src/inference_gateway/pairs.py` (corpus)

## Context

The gateway has two cache tiers. The exact-match tier keys on a hash of
(tenant, model, normalised messages); a hit is a hit, and the only failure mode
is staleness, which TTL handles.

The semantic tier is different in kind. It keys on embedding similarity, so a
hit is a *guess* that two differently-worded prompts want the same answer. When
that guess is wrong the gateway returns a confident, fluent, wrong answer.
There is no exception, no 5xx, no retry, nothing for a downstream error handler
to catch. The request looks successful in every metric the gateway emits.

That asymmetry is the whole issue. A cache miss costs an API call. A false
cache hit costs correctness, silently, and the blast radius grows with the hit
rate — the better the cache appears to perform, the more wrong answers it
serves. Choosing a similarity threshold by taste, or by copying a number from a
blog post, is therefore not an acceptable engineering process here.

So before enabling the tier, I measured the thing that actually matters: **the
false-hit rate**, meaning the fraction of non-equivalent prompt pairs that the
cache would wrongly serve from a previous answer.

## What I measured

A 60-pair labelled corpus (`src/inference_gateway/pairs.py`): 30 paraphrase
pairs that a cache *should* merge, and 30 hard negatives it *must not*.

The corpus design is the load-bearing part. A pair set of randomly chosen
unrelated prompts measures nothing, because any threshold above zero scores
perfectly on it. The negatives here are all *lexically close and semantically
different* — the edits most likely to slip past a similarity check while
completely changing the correct answer:

| category | example |
|---|---|
| single-entity swap | "capital of France" vs "capital of Finland" |
| negation | "enable logging" vs "disable logging" |
| unit / magnitude | "50 km in miles" vs "50 miles in km" |
| direction reversal | "encrypt with GPG" vs "decrypt with GPG" |
| version | "new in Python 3.11" vs "new in Python 3.12" |
| adjacent concept | "time complexity" vs "space complexity" |

The embedder is `HashingEmbedder`, a deterministic bag of words plus character
4-grams. It needs no model download, no GPU and no network, which is what lets
this measurement run in CI on every commit rather than being the test that gets
skipped.

## Findings

The measured numbers are in `results/cache.md`, regenerated and asserted by
`scripts/generate_results.py`. Three things came out of it.

**1. Every threshold is either unsafe or useless.**

| threshold | false-hit rate | true-hit rate |
|---|---|---|
| 0.50 | 97% | 37% |
| 0.70 | 90% | 7% |
| 0.85 | 37% | 0% |
| 0.92 | 7% | 0% |
| 0.95 | 0% | 0% |

There is no row where the cache both helps and is safe. By the time the
false-hit rate is tolerable, the true-hit rate is zero — the cache never fires,
so it is an elaborate no-op.

**2. The ranking is inverted, which no threshold can fix.**

Separation AUC — the probability that an equivalent pair scores above a
non-equivalent one — is **0.031**. Random guessing is 0.5. This is not a
weak signal; it is a signal pointing the wrong way. The best achievable
`true-hit − false-hit` gain across all thresholds is **0.00**: never caching is
as good as the best possible threshold.

I want to be precise about why this is a stronger result than the sweep alone.
A threshold table invites you to pick the least-bad row and ship it. An
inverted AUC says the rows are not a tradeoff curve at all — the component is
ranking backwards, and tuning it is choosing where on a bad curve to sit.

**3. The mechanism is that surface similarity is anti-correlated with meaning
on exactly the cases that matter.**

- A hard negative changes **one word**, so it stays lexically near-identical
  (Python 3.11 vs 3.12 scores 0.944) while the right answer changes entirely.
- A paraphrase changes **most words** ("How do I merge two dictionaries?" vs
  "What's the way to combine two dicts?" scores 0.111) while meaning the same.

A lexical embedder is not merely imprecise here. It is measuring the quantity
that most reliably distinguishes the pairs it must keep apart from the pairs it
should merge — in the wrong direction.

## Decision

**The semantic cache tier ships disabled by default**
(`TwoTierCache.semantic_enabled = False`). The exact-match tier, which cannot
be wrong, stays enabled.

The threshold constant is set to **0.95** — the lowest value at which this
embedder produced zero false hits on this corpus — so that anyone who enables
the tier without re-measuring gets the safe-but-useless configuration rather
than the unsafe-but-useful one. If the tier is going to be a no-op, it should
be a no-op that cannot return a wrong answer.

The code path is **kept, not deleted**. It is what the measurement runs
against, and it is the seam where a real sentence encoder drops in as a
one-argument change.

## Why not the alternatives

**Ship it enabled at a "reasonable" threshold like 0.85.** This is what the
component would have done without the measurement. It would have served a wrong
answer for roughly a third of near-miss prompts while never once producing a
correct hit — strictly worse than being switched off, and invisible in every
dashboard.

**Delete the semantic tier entirely.** Tempting, and defensible. I kept it
because the measurement apparatus is the valuable part: the corpus, the AUC
diagnostic and the assertion in CI are what let a future encoder be evaluated
in an afternoon instead of argued about. Deleting the code would delete the
harness that justifies re-enabling it.

**Use a real sentence-transformer and re-measure.** This is the right next
step, and I did not take it here. It would add a ~90 MB model download and a
torch dependency to a repo whose CI must run offline and fast, in exchange for
improving a component that the breakeven analysis shows is not on the critical
path. The cost model says exact-match caching moves the crossover; it says
nothing that requires the semantic tier to work. I would rather ship a measured
"off" than an unmeasured "on".

## Consequences

- The gateway's cache hit rate is limited to exact repeats. For workloads with
  literal repetition (retries, popular prompts, idempotent replays) this is
  most of the available benefit anyway.
- `results/cache.md` is regenerated in CI and its assertions fail the build if
  the AUC stops being inverted or a threshold becomes usable. The decision
  cannot silently outlive the evidence for it.
- Anyone swapping in a different encoder gets an immediate verdict by running
  one script, with a decision rule (`gain > 0`) rather than a judgement call.

## What this does not establish

- **Nothing about real sentence encoders.** A learned embedder would very
  likely score far better than 0.031 AUC. This repo never ran one, so it
  claims nothing about how much better, and the "off by default" decision is
  specific to the embedder actually shipped.
- **Nothing about real traffic.** The corpus is 60 synthetic pairs, written
  adversarially by one person. Real workloads have a different and unmeasured
  mix of near-misses; the false-hit rates here are deliberately pessimistic and
  should not be quoted as production expectations.
- **Nothing about the corpus being representative.** The categories of hard
  negative reflect one view of what a hard case is. A category I did not think
  of is, by construction, not in the measurement.
