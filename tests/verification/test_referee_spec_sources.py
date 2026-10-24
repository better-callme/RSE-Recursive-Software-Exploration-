"""H2: the referee's verdict is a function of (spec, state).

No attached provider and no hidden attributes: sources come from
spec.test_sources, which spec_hash covers. An explicitly attached
provider remains a visible override.
"""

import pytest

from forgemind.benchmark.problems import problem_spec
from forgemind.core.config import ForgeMindConfig
from forgemind.core.problem import AcceptanceTest, ProblemSpec
from forgemind.core.state import CodebaseState
from forgemind.verification.environment import SubprocessExecutionEnvironment
from forgemind.verification.referee import Referee

GOOD = (
    "def compute_stats(data):\n"
    "    if not data:\n"
    "        return {'count': 0, 'sum': 0, 'min': None, 'max': None, 'mean': None}\n"
    "    t = sum(data)\n"
    "    return {'count': len(data), 'sum': t, 'min': min(data),\n"
    "            'max': max(data), 'mean': t / len(data)}\n"
)
BAD_MEAN = GOOD.replace("t / len(data)", "t // len(data)")


def state_with(code: str) -> CodebaseState:
    return CodebaseState(spec_hash="a" * 64, files={"solution.py": code})


def bare_referee(spec) -> Referee:
    config = ForgeMindConfig()
    return Referee(spec, SubprocessExecutionEnvironment(config.resources),
                   config)


def test_referee_verifies_from_spec_sources_alone():
    # This is the CLI/API construction path: no attach_test_provider.
    referee = bare_referee(problem_spec("stats"))
    assert referee.verify(state_with(GOOD)).acceptance_passed
    v = referee.verify(state_with(BAD_MEAN))
    assert not v.acceptance_passed
    assert v.tests_failed > 0


def test_spec_without_sources_and_no_provider_raises_clearly():
    spec = ProblemSpec(
        problem_id="bare",
        description="no sources",
        requirements=("r",),
        acceptance_tests=(AcceptanceTest("t1", "test_t1"),),
    )
    referee = bare_referee(spec)
    with pytest.raises(ValueError, match="test_sources"):
        referee.verify(state_with(GOOD))


def test_attached_provider_is_an_explicit_override():
    referee = bare_referee(problem_spec("stats"))
    # M2: the override source must still cover the spec's declared
    # contract (all six stats tests) — the override is not exempt from
    # the declared ⊆ executed check; it only replaces the source.
    referee.attach_test_provider(
        lambda: {"anything.py": "".join(
            f"def {name}():\n    assert True\n"
            for name in ("test_normal", "test_empty", "test_single",
                         "test_negative", "test_duplicates",
                         "test_large_values"))})
    # With the override, even the bad-mean solution "passes": precedence
    # is deliberate and visible at the construction site.
    v = referee.verify(state_with(BAD_MEAN))
    assert v.acceptance_passed
    assert v.tests_total == 6
