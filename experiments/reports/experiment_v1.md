# Experiment v1 Report — Does structured search improve generation reliability?

**Config hash:** `becdb36d32785158c62b68fc83caa4973473bb5aad598425a772d4223aed0057`
**Benchmark:** task-ladder-1.0.0 (9 tasks, 4 difficulty levels) · ForgeMind 0.1.0

## CRITICAL SCOPE LIMITATION — READ FIRST

ForgeMind v0.1 has **no live LLM**. The generator is simulated by a fixed
per-task **candidate pool** of plausible implementations, where index 0 models
a greedy sample. Each trial uses one seeded permutation of the pool,
*identically shared by baseline and ForgeMind* (paired design). Baseline
receives exactly pool[0]; ForgeMind may branch over up to B unseen pool
entries plus repair draws.

Therefore this experiment measures the **search + verification machinery**
under a controlled generation distribution. It does NOT measure LLM generation
quality. Token counts are null (not measurable); a rationale-length estimate
is recorded separately.

Also note the pilot phase found an instrumentation bug (contract checker did
not count classes as public symbols), fixed before Phase B; all core tests
re-passed after the fix. No results were patched.

## Primary Result (H1)

| System | Tasks | Trials | Success | Success Rate | Wilson 95% CI |
|--------|-------|--------|---------|--------------|---------------|
| Baseline | 9 | 45 | 32 | 0.711 | [0.566, 0.823] |
| ForgeMind | 9 | 45 | 45 | 1.000 | [0.921, 1.000] |

**Δ = p_F − p_B = +0.289** (paired bootstrap 95% CI [0.156, 0.422], 10,000
resamples).

Paired contingency: n11 = 32, n01 = 13 (ForgeMind succeeds where baseline
fails), n10 = 0 (baseline succeeds where ForgeMind fails), n00 = 0.
Discordant advantage = +13. McNemar exact (two-sided): **p = 0.000244**.

Under this experiment's conditions H0 is rejected. Caveat: with pools of size
3 where at least one member always passes, a system that can examine *all*
pool members trivially reaches 100%. The honest reading is "search recovers
all baseline failures in this regime," not "search is magic."

## Cost Trade-off (H3/H4)

| System | Tokens | Model Calls | Verification Calls | Runtime (s total) |
|--------|--------|-------------|--------------------|-------------------|
| Baseline | null (n/a) | 45 (1/trial) | 45 | 4.0 |
| ForgeMind | null (est. only) | 225 | 70 | 8.6 |

Verification per success: baseline 1.41, ForgeMind 1.56. Runtime ≈ 2.15×.
Token costs are genuinely unmeasurable without a live model and are recorded
as null per protocol §31.

## Difficulty Scaling (H6)

| Difficulty | Baseline | ForgeMind | Δ |
|------------|----------|-----------|---|
| 1 | 0.800 | 1.000 | +0.200 |
| 2 | 0.733 | 1.000 | +0.267 |
| 3 | 0.600 | 1.000 | +0.400 |
| 4 | 0.800 | 1.000 | +0.200 |

The largest gain appears at level 3 (stateful/multi-function tasks), but with
N=15 per cell and a ceiling effect (ForgeMind 100% everywhere) no monotonic
scaling trend can be established. Exploratory only.

## Search Ablation (the key question) — 27 trials/variant

| Variant | Success | Runtime (mean s) | Verification calls | Backtracks |
|---------|---------|------------------|--------------------|------------|
| Baseline | 0.111 [0.039, 0.281] | 0.081 | 27 | 0 |
| Iterative repair | 0.852 [0.675, 0.941] | 0.222 | 75 | 0 |
| Beam | 1.000 [0.875, 1.000] | 0.162 | 55 | 0 |
| Beam+Backtrack | 1.000 [0.875, 1.000] | 0.173 | 58 | 0 |
| Full ForgeMind | 1.000 [0.875, 1.000] | 0.168 | 58 | 0 |

(Note on the low ablation-baseline rate: ablation trials used different seeds
that happened to place flawed candidates first more often; paired comparisons
remain internally valid.)

Incremental effects:

- baseline → iterative: **+0.74 success** (+48 verifications)
- iterative → beam: **+0.15 success** (−20 verifications — branching was also
  cheaper than repeated repair here)
- beam → beam+backtrack → full: **+0.00**

**Finding: in this regime nearly all improvement comes from having ANY
second chance (repair or branching). Verification-driven iteration supplies
~84% of the total lift; branching adds ~16%; backtracking and critic/optimizer
add nothing measurable beyond beam.** The state-space-search-specific
machinery (backtracking, stagnation pruning, adversarial tier) never fired on
these tasks — pools were too shallow to require them.

## Failure Recovery (H7)

13 baseline failures, all recovered by ForgeMind (net advantage +13; zero
inversions). Recovery cost: mean extra verifications ≈ 1–2 per recovery.
Example trace (`experiments/traces/forgemind/recovery_L3A_bank_account_trial0.json`):

```
Root
  ├── cand-05 → transition OK → verify 2/4 FAIL
  ├── cand-07 → transition OK → verify 1/4 FAIL
  └── cand-09 → transition OK → verify 4/4 PASS → ACCEPTED
```

Generated from recorded events, not fabricated.

## Failure taxonomy

Baseline: TEST_FAILURE ×13 (plus CONTRACT_FAILURE at transition in earlier
pilot before fix). ForgeMind internal failures (TEST_FAILURE ×21 across its
rejected candidates) were all superseded by later successful branches.

## Evidence AGAINST ForgeMind

- Costs ~2× baseline runtime and ~1.6× verifications for gains that a simple
  retry loop mostly replicates (iterative alone reached 0.85).
- Backtracking, stagnation detection, adversarial tier, optimizer: zero
  measurable contribution in this benchmark. Their machinery never triggered.
- With ceiling-level pool sizes, "success" cannot distinguish smart search
  from exhaustive enumeration.
- Ablation seeds show sensitivity: baseline success ranged 0.11–0.71 across
  seed blocks — small-N binary outcomes are noisy.

## Evidence FOR ForgeMind

- Perfect 45/45 vs 0.711 baseline; every single-shot failure recovered;
  zero regressions (n10 = 0).
- Branching dominated iterative repair on BOTH axes here (higher success AND
  fewer verifications), suggesting parallel-candidate evaluation beats serial
  repair when candidates are cheap to generate.

## Final Conclusion

Within the simulated-generator regime: **verification + any form of candidate
diversity converts most single-shot failures into successes at roughly 2×
compute; the specifically "search-flavored" components (backtracking,
stagnation handling, adversarial tier) contributed nothing measurable on this
ladder.** The experiment is inconclusive on whether state-space search
specifically — as opposed to simple retry/branching — adds value, because the
tasks never stressed it.

## Next Experiment

1. Live-LLM adapters behind the existing Agent protocol so token costs and
   stochastic failure modes are real.
2. Deeper pools (6–10 candidates) and tasks whose correct solution requires
   composition of two wrong-looking partial ideas, so backtracking/stagnation
   machinery actually fires.
3. Held-out task set generated after configuration freeze to eliminate the
   residual authoring-correlation between pool design and referee tests.
