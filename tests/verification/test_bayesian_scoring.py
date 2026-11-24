"""Bayesian guidance mode: the correctness term is the Beta posterior
mean over the EXACT acceptance counts, (alpha0 + passes) /
(alpha0 + beta0 + total), instead of the raw pass rate.

Negative control: every test below that asserts a Bayesian value fails
if the mode is reverted to pass_rate (e.g. 0/5 gives 1/7 Bayesian
vs 0.0 pass-rate; 5/5 gives 6/7 vs 1.0).
"""

import pytest

from forgemind.core.config import ForgeMindConfig, ScoringWeights
from forgemind.verification.results import HardGateResult, VerificationResult
from forgemind.verification.scoring import ScoringPolicy


def _result(*, total, passes):
    return VerificationResult(
        static_gate_passed=True,
        acceptance_total=total, acceptance_passes=passes)


def _gate_passed():
    return HardGateResult(gates={"requiredTests": True})


def _bayesian(*, alpha=1.0, beta=1.0):
    return ScoringPolicy(
        ScoringWeights(), guidance_mode="bayesian",
        bayesian_prior_alpha=alpha, bayesian_prior_beta=beta)


def test_bayesian_zero_tests_is_prior_mean():
    # No evidence: the posterior is the prior. Beta(1, 1) -> 0.5.
    # (The raw pass rate would be 0.0 — this is the sharpest
    # difference between the modes.)
    scores = _bayesian().score(_result(total=0, passes=0), _gate_passed(),
                               depth=1)
    assert scores.correctness == pytest.approx(0.5)


def test_bayesian_five_of_five():
    scores = _bayesian().score(_result(total=5, passes=5), _gate_passed(),
                               depth=1)
    assert scores.correctness == pytest.approx(6 / 7)


def test_bayesian_zero_of_five():
    scores = _bayesian().score(_result(total=5, passes=0), _gate_passed(),
                               depth=1)
    assert scores.correctness == pytest.approx(1 / 7)


def test_bayesian_custom_prior_shifts_result():
    # A stronger prior pulls the same evidence toward its mean.
    strong = _bayesian(alpha=2.0, beta=2.0)
    scores = strong.score(_result(total=5, passes=5), _gate_passed(), depth=1)
    assert scores.correctness == pytest.approx(7 / 9)

    skewed = _bayesian(alpha=10.0, beta=1.0)
    scores = skewed.score(_result(total=5, passes=0), _gate_passed(), depth=1)
    assert scores.correctness == pytest.approx(10 / 16)
    # Same evidence, different prior -> different posterior.
    assert scores.correctness != pytest.approx(1 / 7)


def test_bayesian_flows_into_guidance_and_total():
    # The Bayesian correctness is what the guidance (and total, when
    # the gate passes) is built from — not just a recorded field.
    low = _bayesian().score(_result(total=5, passes=0), _gate_passed(),
                            depth=1)
    default = ScoringPolicy(ScoringWeights()).score(
        _result(total=5, passes=0), _gate_passed(), depth=1)
    assert low.correctness == pytest.approx(1 / 7)
    assert default.correctness == pytest.approx(0.0)
    assert low.guidance > default.guidance
    assert low.total == pytest.approx(low.guidance)


def test_bayesian_total_still_zero_when_gate_fails():
    # Only `total` is gated, exactly as in pass-rate mode; the
    # posterior correctness is still recorded.
    gate_fail = HardGateResult(gates={"requiredTests": False})
    scores = _bayesian().score(_result(total=5, passes=5), gate_fail, depth=1)
    assert scores.correctness == pytest.approx(6 / 7)
    assert scores.total == 0.0
    assert not scores.hard_pass


@pytest.mark.parametrize("total,passes", [(0, 0), (10, 9), (5, 5), (5, 0)])
def test_pass_rate_default_reproduces_old_correctness(total, passes):
    # Default mode is byte-identical to the old formula: correctness
    # is the raw acceptance pass rate, including 0.0 with no tests.
    scores = ScoringPolicy(ScoringWeights()).score(
        _result(total=total, passes=passes), _gate_passed(), depth=1)
    expected = passes / total if total else 0.0
    assert scores.correctness == pytest.approx(expected)


def test_bayesian_matches_pass_rate_with_many_tests():
    # With overwhelming evidence the posterior converges on the raw
    # rate; the prior is a small-sample correction, not a new signal.
    scores = _bayesian().score(
        _result(total=1000, passes=900), _gate_passed(), depth=1)
    assert scores.correctness == pytest.approx(0.9, abs=0.01)


def test_invalid_guidance_mode_raises():
    with pytest.raises(ValueError):
        ScoringPolicy(ScoringWeights(), guidance_mode="bogus")
    with pytest.raises(ValueError):
        ForgeMindConfig(guidance_mode="bogus")


def test_nonpositive_priors_raise():
    with pytest.raises(ValueError):
        ScoringPolicy(ScoringWeights(), guidance_mode="bayesian",
                      bayesian_prior_alpha=0.0)
    with pytest.raises(ValueError):
        ScoringPolicy(ScoringWeights(), guidance_mode="bayesian",
                      bayesian_prior_beta=-1.0)
    with pytest.raises(ValueError):
        ForgeMindConfig(bayesian_prior_alpha=0.0)
    with pytest.raises(ValueError):
        ForgeMindConfig(bayesian_prior_beta=-2.0)


def test_config_defaults():
    cfg = ForgeMindConfig()
    assert cfg.guidance_mode == "pass_rate"
    assert cfg.bayesian_prior_alpha == 1.0
    assert cfg.bayesian_prior_beta == 1.0
    bayes = ForgeMindConfig(guidance_mode="bayesian",
                            bayesian_prior_alpha=2.0, bayesian_prior_beta=3.0)
    assert bayes.guidance_mode == "bayesian"
    assert bayes.bayesian_prior_alpha == 2.0
    assert bayes.bayesian_prior_beta == 3.0
