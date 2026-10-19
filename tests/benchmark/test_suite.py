"""Tests for the strengthened benchmark: problem suite, trap discrimination,
aligned verification tiers, and aggregate reporting."""

from __future__ import annotations

import pytest

from forgemind.benchmark.experiment import ExperimentRunner
from forgemind.benchmark.problems import (
    list_problem_names,
    problem_builder,
    problem_spec,
)
from forgemind.core.config import ForgeMindConfig


def test_registry_lists_stats_and_trap():
    names = list_problem_names()
    assert names == ["stats", "stats-trap"]


def test_unknown_problem_raises_keyerror():
    with pytest.raises(KeyError):
        problem_spec("nope")


def test_trap_builder_emits_wrong_variant_first():
    from forgemind.agents.projections import build_builder_projection
    from forgemind.core.state import CodebaseState
    spec = problem_spec("stats-trap")
    builder = problem_builder("stats-trap")
    proj = build_builder_projection(
        spec, CodebaseState(spec_hash=spec.spec_hash, files={}))
    proposals = builder.propose(proj, {"source_node_id": "root"})
    # variant 0 must NOT be the correct implementation (mean truncation)
    assert "//" in proposals[0].created_files["solution.py"]


def test_trap_discriminates_between_arms():
    """ForgeMind recovers from the wrong-first builder; baseline cannot."""
    runner = ExperimentRunner(ForgeMindConfig())
    suite = runner.run_suite(["stats-trap"])
    row = suite.per_problem[0]
    assert row["problem"] == "stats-trap"
    assert row["forgemind"]["success"] is True
    assert row["baseline"]["success"] is False
    assert suite.aggregate()["delta_success"] == pytest.approx(1.0)


def test_baseline_faces_adversarial_tier():
    """With aligned tiers, the correct solution still passes baseline."""
    runner = ExperimentRunner(ForgeMindConfig())
    suite = runner.run_suite(["stats"])
    row = suite.per_problem[0]
    assert row["forgemind"]["success"] is True
    assert row["baseline"]["success"] is True


def test_suite_aggregate_across_problems():
    runner = ExperimentRunner(ForgeMindConfig())
    suite = runner.run_suite(["stats", "stats-trap"])
    agg = suite.aggregate()
    assert agg["problems"] == 2.0
    assert agg["forgemind_success_rate"] == pytest.approx(1.0)
    assert agg["baseline_success_rate"] == pytest.approx(0.5)
    assert agg["delta_success"] == pytest.approx(0.5)
    assert agg["cost_ratio"] > 1.0
    d = suite.to_json_dict()
    assert d["problem_names"] == ["stats", "stats-trap"]
    assert len(d["per_problem"]) == 2
    assert "forgemind_version" in d


def test_suite_report_roundtrip(tmp_path):
    runner = ExperimentRunner(ForgeMindConfig())
    suite = runner.run_suite(["stats"])
    out = runner.save_suite_report(suite, tmp_path / "suite.json")
    assert out.exists()
    import json
    data = json.loads(out.read_text())
    assert data["aggregate"]["problems"] == 1.0
