"""Scoring policy: guidance score always; acceptance score only on hard pass."""

from __future__ import annotations


from forgemind.core.config import ScoringWeights
from forgemind.verification.results import ComponentScores, HardGateResult, VerificationResult


class ScoringPolicy:
    """Guidance scoring. `guidance_mode` is an opt-in experiment
    switch (default "pass_rate"): in "bayesian" mode the correctness
    term is the Beta posterior mean over the EXACT acceptance counts,
    (alpha0 + passes) / (alpha0 + beta0 + total), instead of the raw
    pass rate. Priors default to Beta(1, 1) (uniform) and must be > 0.
    Everything else in the formula — weights, penalty, total — is
    identical in both modes."""

    def __init__(
        self,
        weights: ScoringWeights,
        *,
        guidance_mode: str = "pass_rate",
        bayesian_prior_alpha: float = 1.0,
        bayesian_prior_beta: float = 1.0,
    ) -> None:
        if guidance_mode not in ("pass_rate", "bayesian"):
            raise ValueError("guidance_mode must be 'pass_rate' or 'bayesian'")
        if bayesian_prior_alpha <= 0 or bayesian_prior_beta <= 0:
            raise ValueError("bayesian priors must be > 0")
        self._w = weights
        self._guidance_mode = guidance_mode
        self._prior_alpha = bayesian_prior_alpha
        self._prior_beta = bayesian_prior_beta

    def score(
        self,
        verification: VerificationResult,
        gate: HardGateResult,
        *,
        depth: int,
        repeated_failures: int = 0,
        stagnated: bool = False,
    ) -> ComponentScores:
        w = self._w
        # M3: correctness is the ACCEPTANCE rate. Adversarial outcomes
        # answer "can we break it?", not "how correct is it?" — they
        # have their own gate and must not dilute correctness.
        if self._guidance_mode == "bayesian":
            # Beta posterior mean: prior Beta(alpha0, beta0) updated by
            # (acceptance_passes, acceptance_total - passes). With no
            # tests this is the prior mean alpha0 / (alpha0 + beta0).
            # ComponentScores.correctness records this posterior mean,
            # not the raw pass rate, while the mode is active.
            correctness = (
                (self._prior_alpha + verification.acceptance_passes)
                / (self._prior_alpha + self._prior_beta
                   + verification.acceptance_total)
            )
        else:
            correctness = verification.acceptance_pass_rate
        # Deterministic soft components: measured wall-time/memory vary run to
        # run (OS jitter), so they are excluded from scoring in v0.1 and
        # recorded as raw evidence only. Neutral 0.5 keeps weights stable.
        performance = 0.5
        memory = 0.5
        complexity = max(0.0, 1.0 - depth / 10.0)
        maintainability = 0.5  # placeholder metric until quality analyzers exist
        search_cost = max(0.0, 1.0 - depth * 0.05)

        penalty = (
            min(repeated_failures, 4) / 4.0 * w.repeated_failure_penalty
            + (w.stagnation_penalty if stagnated else 0.0)
        )

        guidance = (
            w.test_pass_rate * correctness
            + w.performance * performance
            + w.memory * memory
            + w.complexity * complexity
            + w.depth_cost * search_cost
        ) / (
            w.test_pass_rate + w.performance + w.memory + w.complexity + w.depth_cost
        )
        guidance = max(0.0, guidance - penalty)

        total = guidance if gate.passed else 0.0
        return ComponentScores(
            hard_pass=gate.passed,
            correctness=correctness,
            performance=performance,
            memory=memory,
            complexity=complexity,
            maintainability=maintainability,
            search_cost=search_cost,
            guidance=guidance,
            total=total,
        )


