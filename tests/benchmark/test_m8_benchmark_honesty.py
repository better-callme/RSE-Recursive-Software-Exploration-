"""M8: baseline counts are measured, deterministic totals are compared,
and builder instances are never shared across problems or arms."""

import forgemind.benchmark.kits as kits_mod
from forgemind.agents.mocks import MockBuilder, MockCritic
from forgemind.benchmark.experiment import ExperimentRunner
from forgemind.benchmark.problems import problem_spec
from forgemind.core.config import ForgeMindConfig
from forgemind.core.tokens import (
    estimate_tokens,
    estimate_tokens_for_proposals,
)


def _runner():
    return ExperimentRunner(ForgeMindConfig(random_seed=7))


def test_token_helper():
    assert estimate_tokens() == 0
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcdefgh") == 2
    assert estimate_tokens("abcd", "efgh") == 2

    class NoRationale:
        pass

    assert estimate_tokens_for_proposals([NoRationale()]) == 0


def test_baseline_counts_are_measured():
    """Pre-fix the baseline record hardcoded 1/1/0. The stats baseline
    really runs builder + critic (2 agent calls) and verify + adversarial
    verify (2 verifications); its 3 builder proposals carry 3 x 8 = 24
    estimated tokens (critic proposals carry no rationale). The trap
    baseline's gate fails, so no critic and no second verification —
    but its builder tokens are still 24, not 0."""
    suite = _runner().run_suite(["stats", "stats-trap"])
    base = {r.extra["suite_problem"]: r for r in suite.baseline.records}
    stats = base["stats"]
    assert stats.agent_calls == 2
    assert stats.verifications == 2
    assert stats.token_estimate == 24
    trap = base["stats-trap"]
    assert trap.agent_calls == 1
    assert trap.verifications == 1
    assert trap.token_estimate == 24


def test_aggregate_deterministic_totals():
    suite = _runner().run_suite(["stats", "stats-trap"])
    agg = suite.aggregate()
    for field, record_field in (
        ("agent_calls", "agent_calls"),
        ("verifications", "verifications"),
        ("total_tokens", "token_estimate"),
        ("nodes_created", "candidates"),
        ("nodes_explored", "nodes_explored"),
        ("nodes_pruned", "nodes_pruned"),
    ):
        assert agg[f"forgemind_{field}"] == sum(
            getattr(r, record_field) for r in suite.forgemind.records)
        assert agg[f"baseline_{field}"] == sum(
            getattr(r, record_field) for r in suite.baseline.records)
    mean = lambda rs: (sum(r.wall_clock_seconds for r in rs) / len(rs)
                       if rs else 0.0)
    assert agg["cost_ratio_noisy"] == (
        mean(suite.forgemind.records) < 1.0
        or mean(suite.baseline.records) < 1.0)


def test_factories_give_fresh_instances():
    runner = _runner()
    problems = [problem_spec("stats"), problem_spec("stats-trap")]
    built, critics = [], []

    def builder_factory():
        b = MockBuilder()
        built.append(b)
        return b

    def critic_factory():
        c = MockCritic()
        critics.append(c)
        return c

    runner.run_forgemind(problems, builder_factory=builder_factory)
    assert len(built) == 2
    runner.run_baseline(problems, builder_factory=builder_factory,
                        critic_factory=critic_factory)
    assert len(built) == 4
    assert len({id(b) for b in built}) == 4
    assert len(critics) == 2
    assert len({id(c) for c in critics}) == 2


def test_suite_arms_get_distinct_builder_instances(monkeypatch):
    """run_suite already constructs a fresh builder per arm (M8 pins
    it): builder() is called once per arm per problem, with distinct
    objects."""
    built = {}
    real_kit_for = kits_mod.mock_kit_for

    class RecordingKit:
        def __init__(self, kit, name):
            self._kit = kit
            self.name = name

        def architect(self):
            return self._kit.architect()

        def builder(self):
            b = self._kit.builder()
            built.setdefault(self.name, []).append(b)
            return b

        def critic(self):
            return self._kit.critic()

        def optimizer(self):
            return self._kit.optimizer()

    def fake_kit_for(name):
        return RecordingKit(real_kit_for(name), name)

    monkeypatch.setattr(kits_mod, "mock_kit_for", fake_kit_for)
    _runner().run_suite(["stats", "stats-trap"])
    for name, instances in built.items():
        assert len(instances) == 2
        assert instances[0] is not instances[1]
