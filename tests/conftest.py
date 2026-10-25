"""Shared fixtures: stats problem + referee with the real subprocess
environment (L4: the old 'in-memory env' claim was stale — L1 deleted
that environment, and the fixture always built the subprocess one)."""

from __future__ import annotations

import pytest

from forgemind.benchmark.problems.stats_problem import StatsProblem
from forgemind.core.config import ForgeMindConfig
from forgemind.verification.environment import SubprocessExecutionEnvironment
from forgemind.verification.referee import Referee


@pytest.fixture()
def spec():
    return StatsProblem().spec


@pytest.fixture()
def config():
    return ForgeMindConfig()


@pytest.fixture()
def referee(spec, config):
    r = Referee(spec, SubprocessExecutionEnvironment(config.resources), config)
    r.attach_test_provider(StatsProblem().test_files)
    return r
