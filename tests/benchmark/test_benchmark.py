"""Backtracking, duplicate convergence, benchmark harness tests."""

from forgemind.agents.mocks import MockBuilder
from forgemind.benchmark.experiment import ExperimentRunner
from forgemind.benchmark.problems.stats_problem import StatsProblem
from forgemind.core.config import ForgeMindConfig
from forgemind.search.node import NodeStatus

GOOD_X = (
    "def compute_stats(data):\n"
    "    if not data:\n"
    "        return {'count': 0, 'sum': 0, 'min': None, 'max': None, 'mean': None}\n"
    "    t = sum(data)\n"
    "    return {'count': len(data), 'sum': t, 'min': min(data),\n"
    "            'max': max(data), 'mean': t / len(data)}\n"
)


def test_recovery_via_later_branch(spec, referee):
    """L4 rewrite of test_backtracking_via_real_frontier, which hid
    behind xfail: it passed whenever the run succeeded (asserting only
    nodes_explored >= 1) and xfailed instead of failing when the run
    failed — proving nothing in either branch, with a docstring
    claiming a forced first-candidate failure the code never staged.
    This version stages it for real with a wrong-first builder: the
    first variant fails acceptance and leaves a TEST_FAILURE record,
    search continues past it, and a later variant succeeds. No xfail:
    it fails loudly if recovery regresses."""
    from forgemind.agents.mocks import MockBuilder
    from forgemind.recovery.failure import FailureType, RecoveryPolicy
    from forgemind.search.engine import SearchEngine
    config = ForgeMindConfig(max_verifications=60)
    engine = SearchEngine(spec, config,
                          builder=MockBuilder(correct_first=False),
                          referee=referee, recovery=RecoveryPolicy())
    result = engine.run()
    assert result.success
    # Search did not stop at the first candidate.
    assert result.metrics.candidates_created >= 2
    # The wrong variant failed and was recorded, not silently skipped.
    assert any(r.failure_type == FailureType.TEST_FAILURE
               for r in engine._all_records)


def test_duplicate_convergence_pruned(spec, referee):
    """A and B independently produce state X: second X is hash-detected."""
    from forgemind.core.proposal import CodePatchProposal
    from forgemind.search.engine import SearchEngine
    from forgemind.recovery.failure import RecoveryPolicy

    engine = SearchEngine(spec, ForgeMindConfig(), referee=referee,
                          recovery=RecoveryPolicy())
    from forgemind.core.transition import TransitionEngine, make_root_state
    eng = TransitionEngine(spec)
    root = make_root_state(spec)
    patch_a = CodePatchProposal(created_files={"solution.py": GOOD_X})
    patch_b = CodePatchProposal(created_files={"solution.py": GOOD_X})  # identical!
    ra = eng.apply(root, patch_a)
    rb = eng.apply(root, patch_b)
    assert ra.ok and rb.ok
    assert ra.new_state.content_hash == rb.new_state.content_hash

    # Now through the frontier: second insert must be refused as duplicate.
    from forgemind.search.node import SearchNode
    n1 = SearchNode(node_id="na", state=ra.new_state, parent_node_id="root",
                    candidate_id="ca", depth=1, creation_index=1)
    n2 = SearchNode(node_id="nb", state=rb.new_state, parent_node_id="root",
                    candidate_id="cb", depth=1, creation_index=2)
    assert engine.frontier.insert(n1, 0.5) is True
    assert engine.frontier.insert(n2, 0.9) is False
    assert engine.metrics.duplicates == 0  # counter increments in engine loop only
    assert len(engine.frontier) == 1


def test_benchmark_harness_runs_both_runners(spec, tmp_path):
    config = ForgeMindConfig(max_verifications=40)
    runner = ExperimentRunner(config)
    runner._test_provider = StatsProblem().test_files
    problems = [spec]
    fm = runner.run_forgemind(problems)
    base = runner.run_baseline(problems)
    comparison = fm.compare(base)

    assert fm.records[0].runner == "forgemind"
    assert base.records[0].runner == "baseline"
    assert fm.success_rate in (0.0, 1.0)
    for key in ("delta_success", "cost_ratio", "forgemind_success_rate",
                "baseline_success_rate"):
        assert key in comparison

    data = fm.to_json_dict()
    assert data["records"][0]["problem_id"] == spec.problem_id
    assert "forgemind_version" in data and "platform" in data

    out = runner.save_report(fm, (tmp_path or __import__("pathlib").Path("/tmp")) /
                             "fm_test_report.json")
    assert out.exists()


def test_stats_problem_edge_cases_declared(spec):
    ids = set(spec.test_ids)
    assert {"test_normal", "test_empty", "test_single", "test_negative",
            "test_duplicates", "test_large_values"} <= ids
