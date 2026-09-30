"""Experiment v1: paired baseline-vs-ForgeMind evaluation with ablations.

Central design note (documented limitation): ForgeMind v0.1 uses deterministic
mock agents; there is no live LLM. The stochastic "model" is simulated by a
per-task CANDIDATE POOL: an ordered list of plausible implementations where
index 0 models a greedy (temperature-0) sample. Each trial applies one seeded
shuffle of the pool, IDENTICALLY SHARED by baseline and ForgeMind (paired
design). Baseline receives exactly pool[0]; ForgeMind may branch over up to B
pool entries and issue repair proposals drawn from the same pool.

Therefore this experiment measures the SEARCH AND VERIFICATION MACHINERY
under a controlled generation distribution -- NOT LLM generation quality.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import forgemind
from forgemind.agents import projections
from forgemind.benchmark.experiment import ExperimentRunner
from experiments.tasks import ALL_TASKS, TaskDef  # noqa: F401
from forgemind.core.config import ForgeMindConfig
from forgemind.core.history import EventType, History
from forgemind.core.proposal import CodePatchProposal
from forgemind.core.state import CodebaseState
from forgemind.core.transition import TransitionEngine, make_root_state
from forgemind.recovery.failure import RecoveryPolicy
from forgemind.search.engine import SearchEngine
from forgemind.verification.referee import Referee, hard_gate

EXPERIMENT_ID = "experiment_v1"
BENCHMARK_VERSION = "task-ladder-1.0.0"


# --------------------------------------------------------------------------
# Configuration freeze
# --------------------------------------------------------------------------

def default_config() -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "forgemind_version": forgemind.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        # Treatment parameters (FROZEN before final evaluation)
        "beam_width": 3,
        "branching_factor": 3,
        "max_depth": 4,
        "max_nodes": 60,
        "max_verifications": 40,
        "max_agent_calls": 120,
        "iterative_max_repairs": 2,
        "n_trials": 5,
        "ablation_trials": 3,
        "resources": {
            "wall_timeout_seconds": 10.0,
            "cpu_limit_seconds": 8,
            "memory_limit_mb": 512,
            "process_limit": 64,
            "max_output_bytes": 1000000,
        },
        # Model metadata: no live model in v0.1; the generator is the seeded
        # candidate-pool simulation described in the module docstring.
        "model": "mock://candidate-pool-v1",
        "provider": "deterministic-simulation",
        "temperature": None,   # not applicable; recorded as null per protocol
        "max_tokens": None,    # token counts unavailable; estimates only
        "random_seed_base": 20260822,
        "hard_limits": {
            "max_tasks": 16,
            "max_trials_total": 600,
            "max_wall_time_seconds": 3600,
        },
    }


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def freeze_config(path: Path) -> tuple[dict, str]:
    cfg = default_config()
    path.parent.mkdir(parents=True, exist_ok=True)
    chash = config_hash(cfg)
    payload = {"config": cfg, "config_hash": chash}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return cfg, chash


def load_frozen_config(path: Path) -> tuple[dict, str]:
    payload = json.loads(path.read_text())
    expected = payload["config_hash"]
    actual = config_hash(payload["config"])
    if expected != actual:
        raise RuntimeError("frozen experiment config was modified")
    return payload["config"], expected


# --------------------------------------------------------------------------
# Seeded candidate-pool "model"
# --------------------------------------------------------------------------

def shuffled_pool(task: TaskDef, seed: int) -> list[str]:
    """One seeded permutation of the task's candidate pool.

    The SAME permutation is handed to both treatments of a trial: this is the
    pairing mechanism. Index 0 after shuffle models the greedy sample.
    """
    pool = list(task.candidate_pool)
    random.Random(seed).shuffle(pool)
    return pool


# --------------------------------------------------------------------------
# Generic agents driven by the candidate pool (experiment infrastructure)
# --------------------------------------------------------------------------

class PoolBuilder:
    """Deterministic builder proposing unseen pool entries.

    propose(): up to B not-yet-used pool sources (branching).
    repair():  the next unseen entry (recovery without privileged knowledge).
    """

    role = "builder"

    def __init__(self, task: TaskDef, pool: Sequence[str]) -> None:
        self.task = task
        self._pool = list(pool)
        self._used: set[int] = set()

    def _unused(self, count: int) -> list[tuple[int, str]]:
        out = [(i, s) for i, s in enumerate(self._pool) if i not in self._used]
        return out[:count]

    def propose(
        self, projection: projections.BuilderProjection,
        request_context: Mapping[str, str],
    ) -> tuple[CodePatchProposal, ...]:
        picks = self._unused(3)
        props = []
        for idx, (pool_i, src) in enumerate(picks):
            self._used.add(pool_i)
            props.append(CodePatchProposal(
                agent_role=self.role,
                intent=f"pool variant #{pool_i}",
                rationale=f"deterministic pool draw {pool_i}",
                estimated_cost=1.0,
                created_files={self.task.module_path: src},
            ))
        return tuple(props)

    def repair(
        self, projection: projections.BuilderProjection, failure_evidence: str
    ) -> tuple[CodePatchProposal, ...]:
        picks = self._unused(1)
        if not picks:
            return ()
        pool_i, src = picks[0]
        self._used.add(pool_i)
        return (CodePatchProposal(
            agent_role=self.role,
            intent="repair: next unseen pool variant",
            rationale=f"repair draw {pool_i}",
            estimated_cost=2.0,
            created_files={self.task.module_path: src},
        ),)


class PoolCritic:
    """Generic static-analysis critic; hypotheses verified by Referee only."""

    role = "critic"

    def propose(
        self, projection: projections.CriticProjection,
        request_context: Mapping[str, str],
    ):
        from forgemind.core.proposal import AttackTestProposal, CritiqueProposal

        code = "\n".join(projection.relevant_code.values())
        attacks = []
        if "//" in code:
            attacks.append(AttackTestProposal(
                agent_role=self.role, test_name="critic_int_division",
                hypothesis="integer division used where true division expected",
                test_code=(
                    "def critic_int_division():\n"
                    "    pass  # generic probe: exercised via acceptance suite\n"
                ),
            ))
        critique = CritiqueProposal(
            agent_role=self.role, intent="generic review",
            issues=(), suggested_repair_directions=(), confidence=0.5,
        )
        return (critique,), tuple(attacks)


class NullOptimizer:
    role = "optimizer"

    def propose(self, projection, request_context):
        return ()


# --------------------------------------------------------------------------
# Trial execution (shared Referee per task; fresh state per trial)
# --------------------------------------------------------------------------

def _make_referee(task: TaskDef, env, cfg: dict) -> Referee:
    fm_cfg = _fm_config(cfg)
    ref = Referee(task.spec, env, fm_cfg)
    ref.attach_test_provider(task.test_files)
    return ref


def _fm_config(cfg: dict, *, disable_adversarial=False,
               disable_optimizer=False) -> ForgeMindConfig:
    return ForgeMindConfig(
        beam_width=cfg["beam_width"],
        branching_factor=cfg["branching_factor"],
        max_depth=cfg["max_depth"],
        max_nodes=cfg["max_nodes"],
        max_verifications=cfg["max_verifications"],
        max_agent_calls=cfg["max_agent_calls"],
        enable_adversarial_tier=not disable_adversarial,
        enable_optimizer=not disable_optimizer,
    )


def run_baseline_trial(task: TaskDef, pool: Sequence[str], env, cfg: dict) -> dict:
    """A: one greedy draw, one referee verdict. No branching/repair."""
    started = time.monotonic()
    ref = _make_referee(task, env, cfg)
    eng = TransitionEngine(task.spec)
    root = make_root_state(task.spec)
    t = eng.apply(root, CodePatchProposal(
        created_files={task.module_path: pool[0]}))
    if not t.ok:
        rec = _record(task, success=False, started=started,
                      failure_categories=[_map_violation(t.violation)],
                      termination_reason="SINGLE_SHOT_FAILED",
                      initial=root.content_hash, final=None,
                      verifications=0, extra={"transition_error": t.reason})
        return rec
    node_state = t.new_state
    verification = ref.verify(node_state)
    gate = hard_gate(verification)
    cats = [] if gate.passed else [_classify_verification(verification)]
    return _record(task, success=gate.passed, started=started,
                   failure_categories=cats,
                   termination_reason="SUCCESS" if gate.passed else "VERIFICATION_FAILURE",
                   initial=root.content_hash, final=node_state.content_hash,
                   verifications=1, verification_detail=verification)


def run_iterative_trial(task: TaskDef, pool: Sequence[str], env, cfg: dict) -> dict:
    """B: single branch + iterative repair (no branching beyond repairs)."""
    started = time.monotonic()
    ref = _make_referee(task, env, cfg)
    eng = TransitionEngine(task.spec)
    root = make_root_state(task.spec)
    builder = PoolBuilder(task, pool)

    current_src = pool[0]
    used = {0}
    verifications = 0
    failures: list[str] = []
    final_hash = None
    success = False
    last_verification = None

    for attempt in range(1 + int(cfg["iterative_max_repairs"])):
        t = eng.apply(root, CodePatchProposal(
            created_files={task.module_path: current_src}))
        if not t.ok:
            failures.append("CONTRACT_FAILURE")
            break
        verification = ref.verify(t.new_state)
        last_verification = verification
        verifications += 1
        final_hash = t.new_state.content_hash
        if hard_gate(verification).passed:
            success = True
            break
        failures.append(_classify_verification(verification))
        nxt = builder.repair(projections.build_builder_projection(
            task.spec, root), "evidence")
        if not nxt:
            break
        proposal = nxt[0]
        # find which pool index got consumed
        src = proposal.created_files[task.module_path]
        for i, s in enumerate(pool):
            if s == src:
                used.add(i)
        current_src = src

    return _record(task, success=success, started=started,
                   failure_categories=failures[:5],
                   termination_reason="SUCCESS" if success else "REPAIRS_EXHAUSTED",
                   initial=root.content_hash, final=final_hash,
                   verifications=verifications,
                   verification_detail=last_verification)


def run_search_trial(task: TaskDef, pool: Sequence[str], env, cfg: dict, *,
                     variant: str) -> dict:
    """C/D/E: SearchEngine variants."""
    started = time.monotonic()
    fm_cfg = _fm_config(
        cfg,
        disable_adversarial=(variant in ("beam",)),
        disable_optimizer=(variant in ("beam", "beam_backtrack")),
    )
    ref = Referee(task.spec, env, fm_cfg)
    ref.attach_test_provider(task.test_files)

    reset_ids()
    engine = SearchEngine(
        task.spec, fm_cfg,
        architect=_NullArchitect(),
        builder=PoolBuilder(task, pool),
        critic=PoolCritic(),
        optimizer=NullOptimizer(),
        referee=ref,
        recovery=RecoveryPolicy(),
        history=History(),
    )
    result = engine.run()
    m = result.metrics
    failures = [e.metadata.get("type") for e in
                engine.history.of_type(EventType.FAILURE_RECORDED)]
    accepted = result.accepted_node
    return _record(
        task, success=result.success, started=started,
        failure_categories=[str(f) for f in failures[:8]],
        termination_reason=result.termination.reason.upper(),
        initial=make_root_state(task.spec).content_hash,
        final=accepted.state.content_hash if accepted else None,
        verifications=m.verifications,
        verification_detail=accepted.verification if accepted else None,
        extra={
            "candidates": m.candidates_created,
            "nodes_explored": m.nodes_explored,
            "duplicates": m.duplicates,
            "pruned": m.pruned,
            "backtracks": m.backtracks,
            "stagnation_events": m.stagnation_events,
            "agent_calls": m.agent_calls,
            "max_depth": m.max_depth_reached,
            "token_estimate": m.token_estimate,
        },
    )


class _NullArchitect:
    role = "architect"

    def propose(self, projection, request_context):
        from forgemind.core.proposal import ArchitectureProposal

        return (ArchitectureProposal(
            agent_role=self.role, intent="generic",
            components=("module",), interfaces={}, dependency_graph={},
            implementation_steps=(), rationale="n/a"),)


def reset_ids() -> None:
    from forgemind.core.proposal import reset_id_counter

    reset_id_counter(0)


# --------------------------------------------------------------------------
# Record construction
# --------------------------------------------------------------------------

def _classify_verification(v) -> str:
    name = getattr(v, "failure_type", "") or ""
    mapping = {
        "SYNTAX_ERROR": "SYNTAX_FAILURE",
        "IMPORT_ERROR": "IMPORT_FAILURE",
        "TEST_FAILURE": "TEST_FAILURE",
        "ADVERSARIAL_FAILURE": "EDGE_CASE_FAILURE",
        "TIMEOUT": "TIMEOUT",
        "REGRESSION_FAILURE": "REGRESSION_FAILURE",
    }
    return mapping.get(name, name or "UNKNOWN")


def _map_violation(violation: str) -> str:
    return {
        "SYNTAX": "SYNTAX_FAILURE",
        "PROBLEM_SPEC_MUTATION": "CONTRACT_FAILURE",
        "INTERFACE_CONSTRAINT": "CONTRACT_FAILURE",
        "DEPENDENCY_POLICY": "DEPENDENCY_FAILURE",
    }.get(violation, "MALFORMED_PROPOSAL")


def _record(task, *, success, started, failure_categories,
            termination_reason, initial, final, verifications,
            verification_detail=None, extra=None, seed=None,
            system=None, trial_id=None, config_h=None) -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "task_id": task.task_id,
        "task_version": task.version,
        "difficulty": task.difficulty,
        "trial_id": trial_id,
        "system": system,
        "model": "mock://candidate-pool-v1",
        "provider": "deterministic-simulation",
        "seed": seed,
        "success": bool(success),
        "input_tokens": None,       # not measurable without live LLM
        "output_tokens": None,
        "total_tokens": None,
        "token_estimate": (extra or {}).get("token_estimate"),
        "wall_time_seconds": round(time.monotonic() - started, 4),
        "model_calls": (extra or {}).get("agent_calls", 1),
        "verification_calls": verifications,
        "candidates": (extra or {}).get("candidates", 1),
        "nodes_generated": (extra or {}).get("candidates", 1),
        "nodes_expanded": (extra or {}).get("nodes_explored", 1),
        "nodes_pruned": (extra or {}).get("pruned", 0),
        "duplicates": (extra or {}).get("duplicates", 0),
        "backtracks": (extra or {}).get("backtracks", 0),
        "retries": (extra or {}).get("candidates", 1) - 1,
        "max_depth": (extra or {}).get("max_depth", 1),
        "stagnation_events": (extra or {}).get("stagnation_events", 0),
        "test_pass_rate": (verification_detail.tests_pass_rate
                           if verification_detail is not None else None),
        "failure_categories": failure_categories,
        "recovered": False,   # computed in analysis
        "termination_reason": termination_reason,
        "initial_state_hash": initial[:16] if initial else None,
        "final_state_hash": final[:16] if final else None,
        "config_hash": config_h,
        "git_commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _git_commit() -> str | None:
    import subprocess

    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


# --------------------------------------------------------------------------
# Experiment driver
# --------------------------------------------------------------------------

SYSTEMS = ("baseline", "forgemind")
ABLATIONS = ("baseline", "iterative", "beam", "beam_backtrack", "full")


def run_pilot(tasks: Sequence[TaskDef], cfg: dict, chash: str,
              raw_path: Path) -> list[dict]:
    """Phase A: 2 trials/task x {baseline, forgemind}. Instrumentation shakeout."""
    records: list[dict] = []
    env = SubprocessExecutionEnvironment(_res(cfg))
    for trial in range(2):
        seed = int(cfg["random_seed_base"]) + trial * 977
        for task in tasks:
            pool = shuffled_pool(task, seed)
            for system in SYSTEMS:
                if system == "baseline":
                    rec = run_baseline_trial(task, pool, env, cfg)
                else:
                    rec = run_search_trial(task, pool, env, cfg, variant="full")
                rec.update(system=system, seed=seed, trial_id=trial,
                           config_hash=chash)
                records.append(rec)
                _append_jsonl(raw_path, rec)
    return records


def run_final(tasks: Sequence[TaskDef], cfg: dict, chash: str,
              raw_path: Path) -> list[dict]:
    records: list[dict] = []
    env = SubprocessExecutionEnvironment(_res(cfg))
    n_trials = int(cfg["n_trials"])

    order = list(tasks)
    random.Random(cfg["random_seed_base"]).shuffle(order)  # ordering-bias control

    for trial in range(n_trials):
        seed = int(cfg["random_seed_base"]) + trial * 977
        for task in order:
            pool = shuffled_pool(task, seed)
            # alternate treatment order across trials (§14)
            systems = SYSTEMS if trial % 2 == 0 else (SYSTEMS[1], SYSTEMS[0])
            for system in systems:
                if system == "baseline":
                    rec = run_baseline_trial(task, pool, env, cfg)
                else:
                    rec = run_search_trial(task, pool, env, cfg, variant="full")
                rec.update(system=system, seed=seed, trial_id=trial,
                           config_hash=chash)
                records.append(rec)
                _append_jsonl(raw_path, rec)
    return records


def run_ablations(tasks: Sequence[TaskDef], cfg: dict, chash: str,
                  raw_path: Path) -> list[dict]:
    records: list[dict] = []
    env = SubprocessExecutionEnvironment(_res(cfg))
    n = int(cfg["ablation_trials"])
    for trial in range(n):
        seed = int(cfg["random_seed_base"]) + 100_000 + trial * 977
        for task in sorted(tasks, key=lambda t: t.task_id):
            pool = shuffled_pool(task, seed)
            for variant in ABLATIONS:
                if variant == "baseline":
                    rec = run_baseline_trial(task, pool, env, cfg)
                elif variant == "iterative":
                    rec = run_iterative_trial(task, pool, env, cfg)
                else:
                    rec = run_search_trial(task, pool, env, cfg,
                                           variant="full" if variant == "full"
                                           else variant)
                rec.update(system=f"abl_{variant}", seed=seed,
                           trial_id=trial, config_hash=chash)
                records.append(rec)
                _append_jsonl(raw_path, rec)
    return records


def _res(cfg: dict):
    from forgemind.core.config import ResourceLimits

    return ResourceLimits(**cfg["resources"])


def _append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")


# re-export for harness CLI
from forgemind.verification.environment import SubprocessExecutionEnvironment  # noqa: E402
