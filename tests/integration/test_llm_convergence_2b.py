"""Experiment 2B — Atomic-capability LLM convergence (Phase 2B-1: canonical only).

Approved design: ~/workspace/rse/llm-convergence-2b-atomic-design.md
(Kashi, 2026-10-27), with Amendment A (engine starts from an EMPTY root,
so the stub is established by the first admitted proposal; max_depth 6).

TEST-ONLY. No production code is modified. The 2A harness is reused via
an importlib load of test_llm_convergence.py (its tests are NOT collected
from this module — they live on the loaded module object, not in this
namespace). The 2A builder gained one backward-compatible extension for
this file: a prompt_builder parameter (default preserves 2A behaviour).

Phase 2B-1 runs ONLY the canonical RSE + Greedy pair. The append control
arms are NOT run here; they are Phase 2B-2, gated on the canonical
landscape being viable (H >= 3, >= 1 convergence point, duplicates > 0,
multiple verifications). The live test asserts well-formedness only —
viability is Kashi's judgement from the raw report, not an assertion.

Layout rule (prompt-only, as in 2A): canonical = alphabetical order,
exactly one blank line between top-level defs, no extra top-level code.
The atomic policy does NOT enforce layout; layout disobedience shows up
honestly as non-convergence.
"""

import ast
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass

import pytest

from forgemind.agents.mocks import MockArchitect
from forgemind.agents.projections import build_architect_projection
from forgemind.agents.projections import build_builder_projection
from forgemind.core.config import ForgeMindConfig
from forgemind.core.config import ScoringWeights
from forgemind.core.problem import AcceptanceTest
from forgemind.core.problem import ProblemSpec
from forgemind.core.proposal import AttackTestProposal
from forgemind.core.proposal import CodePatchProposal
from forgemind.core.proposal import CritiqueProposal
from forgemind.core.state import CodebaseState
from forgemind.core.transition import TransitionEngine
from forgemind.recovery.failure import FailureType
from forgemind.recovery.failure import RecoveryPolicy
from forgemind.search.engine import SearchEngine
from forgemind.verification.environment import SubprocessExecutionEnvironment
from forgemind.verification.referee import Referee
from forgemind.verification.referee import hard_gate
from forgemind.verification.scoring import ScoringPolicy

# ----------------------------------------------------------------------
# Reuse of the 2A harness (test-only)
# ----------------------------------------------------------------------

_TLC_PATH = os.path.join(os.path.dirname(__file__),
                         "test_llm_convergence.py")
_tlc_spec = importlib.util.spec_from_file_location("tlc_2a", _TLC_PATH)
tlc = importlib.util.module_from_spec(_tlc_spec)
sys.modules["tlc_2a"] = tlc
_tlc_spec.loader.exec_module(tlc)

CallBudget = tlc.CallBudget
OpenRouterBuilder = tlc.OpenRouterBuilder

# ----------------------------------------------------------------------
# Phase 2B-1 configuration (env-overridable, defaults = approved design)
# ----------------------------------------------------------------------

LIVE_2B = (os.environ.get("FORGEMIND_LLM_2B_LIVE") == "1"
           and tlc._CONNECTOR_AVAILABLE)
MAX_CALLS_2B = int(os.environ.get("FORGEMIND_LLM_2B_MAX_CALLS", "32"))
MAX_TOKENS_2B = int(os.environ.get("FORGEMIND_LLM_2B_MAX_TOKENS", "16384"))
REPORT_PATH_2B = os.environ.get(
    "LLM_CONVERGENCE_2B_REPORT", "llm_convergence_2b1_report.json")
CACHE_PATH_2B = os.environ.get(
    "LLM_CONVERGENCE_2B_CACHE", "llm_convergence_2b1_cache.json")

# Phase 2B-2 append control (env-overridable; own live flag so the
# canonical live test is unaffected). Variant "append" selects the
# append layout sentence in the shared 2A builder; everything else —
# problem, policy, engine config, greedy, report shape — is identical
# to phase 1. The append cache is separate by default: variant is part
# of the canonical key, so append generations could never collide with
# canonical entries, but a dedicated file keeps the artifacts clean.
LIVE_2B_APPEND = (os.environ.get("FORGEMIND_LLM_2B_APPEND_LIVE") == "1"
                  and tlc._CONNECTOR_AVAILABLE)
REPORT_PATH_2B_APPEND = os.environ.get(
    "LLM_CONVERGENCE_2B_APPEND_REPORT",
    "llm_convergence_2b2_append_report.json")
CACHE_PATH_2B_APPEND = os.environ.get(
    "LLM_CONVERGENCE_2B_APPEND_CACHE",
    "llm_convergence_2b2_append_cache.json")

STUB_SOURCE = "def summarize(text):\n    return {}\n"

# Capability names. F5 is summarize itself (replacing the stub, or the
# first move from the empty root). F1-F4 are the independent helpers.
CAPABILITIES = ("count_words", "count_chars", "longest_token",
                "mean_length", "summarize")
HELPERS = ("count_words", "count_chars", "longest_token", "mean_length")

CAPABILITY_BEHAVIOUR = {
    "count_words": "count_words(text) -> int: number of whitespace-separated "
                   "tokens; empty text gives 0",
    "count_chars": "count_chars(text) -> int: len(text), including whitespace",
    "longest_token": "longest_token(text) -> str: longest token, '' when there "
                     "are no tokens; on ties the first longest wins",
    "mean_length": "mean_length(text) -> float: mean token length using true "
                   "division; 0.0 when there are no tokens",
    "summarize": "summarize(text) -> dict: calls the four helpers and returns "
                 "{words, chars, longest, average}",
}

_ACCEPTANCE_2B_SOURCE = '''\n\ndef test_imports():\n    from solution import summarize\n\n\ndef test_summarize_callable():\n    from solution import summarize\n    assert callable(summarize)\n'''


# ----------------------------------------------------------------------
# Problem: wordstats-atomic-v1
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class AtomicProblem:
    """ProblemSpec + structural acceptance-test sources (Referee materializes)."""

    spec: ProblemSpec = ProblemSpec(
        problem_id="wordstats-atomic-v1",
        description=(
            "Build summarize(text) in solution.py incrementally, one "
            "atomic capability per proposal: four independent helper "
            "functions plus summarize itself, which composes them."
        ),
        requirements=(
            "solution.py defines summarize(text) returning a dict with "
            "keys words, chars, longest, average",
            "count_words(text) counts whitespace-separated tokens; "
            "empty text gives 0",
            "count_chars(text) is len(text), including whitespace",
            "longest_token(text) is the longest token, or '' when there "
            "are no tokens; on ties the first longest token wins",
            "mean_length(text) is the mean token length using true "
            "division; 0.0 when there are no tokens",
            "summarize(text) calls the four helpers and returns words = "
            "count_words(text), chars = count_chars(text), longest = "
            "longest_token(text), average = mean_length(text)",
            "empty text gives words 0, chars 0, longest '', average 0.0",
            "whitespace-only text gives words 0, longest '', "
            "average 0.0 without raising",
        ),
        acceptance_tests=(
            AcceptanceTest(
                "imports", "test_imports",
                "solution.py imports and defines summarize"),
            AcceptanceTest(
                "summarize_callable", "test_summarize_callable",
                "summarize is callable (checked without calling it)"),
        ),
        interface_constraints={"summarize": "function: summarize(text)"},
        performance_constraints={"max_runtime_ms": 2000.0},
        test_sources={"__fm_acceptance_tests.py": _ACCEPTANCE_2B_SOURCE},
    )

    def test_files(self) -> dict[str, str]:
        return dict(self.spec.test_sources)


# Attack suite: the 2A behavioral suite moved wholesale to attacks.
# T2 folds 2A's empty test together with 2A's empty-average attack
# (both check the same input; the merged test asserts both).
_ATTACKS_2B = (
    AttackTestProposal(
        agent_role="critic",
        test_name="attack_normal",
        hypothesis="full statistics on a normal input",
        test_code=(
            "def attack_normal():\n"
            "    from solution import summarize\n"
            "    r = summarize('hi there you')\n"
            "    assert r['words'] == 3 and r['chars'] == 12\n"
            "    assert r['longest'] == 'there'\n"
            "    assert abs(r['average'] - 10 / 3) < 1e-9\n"
        ),
    ),
    AttackTestProposal(
        agent_role="critic",
        test_name="attack_empty",
        hypothesis="empty text gives zeros without raising",
        test_code=(
            "def attack_empty():\n"
            "    from solution import summarize\n"
            "    r = summarize('')\n"
            "    assert r['words'] == 0 and r['chars'] == 0\n"
            "    assert r['longest'] == '' and r['average'] == 0.0\n"
        ),
    ),
    AttackTestProposal(
        agent_role="critic",
        test_name="attack_single",
        hypothesis="single token statistics",
        test_code=(
            "def attack_single():\n"
            "    from solution import summarize\n"
            "    r = summarize('hello')\n"
            "    assert r['words'] == 1 and r['chars'] == 5\n"
            "    assert r['longest'] == 'hello'\n"
            "    assert abs(r['average'] - 5.0) < 1e-9\n"
        ),
    ),
    AttackTestProposal(
        agent_role="critic",
        test_name="attack_whitespace",
        hypothesis="irregular whitespace statistics",
        test_code=(
            "def attack_whitespace():\n"
            "    from solution import summarize\n"
            "    r = summarize('  aa   bbb  ')\n"
            "    assert r['words'] == 2 and r['chars'] == 12\n"
            "    assert r['longest'] == 'bbb'\n"
            "    assert abs(r['average'] - 2.5) < 1e-9\n"
        ),
    ),
    AttackTestProposal(
        agent_role="critic",
        test_name="attack_keys",
        hypothesis="result is a dict with all four keys",
        test_code=(
            "def attack_keys():\n"
            "    from solution import summarize\n"
            "    r = summarize('a b')\n"
            "    assert isinstance(r, dict)\n"
            "    for key in ('words', 'chars', 'longest', 'average'):\n"
            "        assert key in r, f'missing {key}'\n"
        ),
    ),
    AttackTestProposal(
        agent_role="critic",
        test_name="attack_mean_precision",
        hypothesis="integer division breaks the mean",
        test_code=(
            "def attack_mean_precision():\n"
            "    from solution import summarize\n"
            "    r = summarize('a bb ccc dddd')\n"
            "    assert abs(r['average'] - 2.5) < 1e-9\n"
        ),
    ),
    AttackTestProposal(
        agent_role="critic",
        test_name="attack_whitespace_only",
        hypothesis="whitespace-only text must not raise",
        test_code=(
            "def attack_whitespace_only():\n"
            "    from solution import summarize\n"
            "    r = summarize('   ')\n"
            "    assert r['words'] == 0 and r['longest'] == ''\n"
            "    assert r['average'] == 0.0\n"
        ),
    ),
)


class FixedAttackCritic2B:
    """Test-only critic returning the fixed 2B attack set, both arms.

    Return shape mirrors 2A's FixedAttackCritic: (critiques, attacks).
    """

    role = "critic"

    def propose(self, projection, request_context):
        critique = CritiqueProposal(
            agent_role=self.role, intent="adversarial review", issues=(),
            suggested_repair_directions=(), confidence=0.8)
        return (critique,), _ATTACKS_2B


# ----------------------------------------------------------------------
# Atomic-move policy (test-only; lives inside the shared builder)
# ----------------------------------------------------------------------

def _top_defs(source: str) -> dict[str, str]:
    """Top-level def name -> AST dump (formatting-insensitive).

    Raises ValueError on unparseable source; the caller turns that into
    a policy rejection, never an exception escaping propose().
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"unparseable: {exc}") from exc
    return {node.name: ast.dump(node, include_attributes=False)
            for node in tree.body if isinstance(node, ast.FunctionDef)}


def _is_stub(source: str) -> bool:
    """True iff the whole file is exactly the stub summarize."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return False
    body = tree.body[0].body
    return (len(body) == 1 and isinstance(body[0], ast.Return)
            and isinstance(body[0].value, ast.Dict)
            and not body[0].value.keys and not body[0].value.values)


def _summarize_is_stub(source: str) -> bool:
    """True iff the summarize FUNCTION in source is the stub, whatever
    else the file contains. This is the check the policy and the subset
    measurement need: F5 replaces the stub function even when helpers
    are already present, and a stub summarize is scaffold, not F5."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "summarize":
            body = node.body
            return (len(body) == 1 and isinstance(body[0], ast.Return)
                    and isinstance(body[0].value, ast.Dict)
                    and not body[0].value.keys
                    and not body[0].value.values)
    return False


def _proposed_solution(current_files, proposal) -> str | None:
    """Proposed solution.py content, or None if the proposal leaves it
    untouched. Any other file touched makes the proposal non-atomic."""
    touched = set(proposal.created_files) | set(proposal.modified_files)
    if touched - {"solution.py"}:
        return None  # sentinel: caller rejects before diffing
    if "solution.py" in proposal.modified_files:
        return proposal.modified_files["solution.py"]
    if "solution.py" in proposal.created_files:
        return proposal.created_files["solution.py"]
    return current_files.get("solution.py")


def check_atomic(current_files, proposal) -> tuple[bool, str]:
    """Admit-or-drop decision for one proposal. Pure function.

    Admits exactly: (a) one helper F1-F4 added, nothing else changed;
    (b) summarize replaced while the current one is the stub (F5);
    (c) from a state WITHOUT summarize: exactly the stub added
    (scaffold) or exactly a real summarize added (F5 as first move).
    Everything else is dropped. Never rewrites, merges, splits,
    reorders, or normalizes content.
    """
    touched = set(proposal.created_files) | set(proposal.modified_files)
    if touched - {"solution.py"}:
        return False, "touches files other than solution.py"
    current_source = current_files.get("solution.py", "")
    proposed_source = _proposed_solution(current_files, proposal)
    if proposed_source is None or proposed_source == current_source:
        return False, "no change to solution.py"
    try:
        current_defs = _top_defs(current_source) if current_source else {}
        proposed_defs = _top_defs(proposed_source)
    except ValueError as exc:
        return False, str(exc)
    added = set(proposed_defs) - set(current_defs)
    removed = set(current_defs) - set(proposed_defs)
    modified = {n for n in current_defs.keys() & proposed_defs.keys()
                if current_defs[n] != proposed_defs[n]}
    if len(added) == 1 and not removed and not modified:
        only = next(iter(added))
        if only in HELPERS:
            return True, f"capability {only}"
        if only == "summarize":
            if _summarize_is_stub(proposed_source):
                return True, "scaffold"
            return True, "capability summarize (F5 first)"
        return False, f"added non-capability function {only}"
    if not added and not removed and modified == {"summarize"}:
        if _summarize_is_stub(current_source):
            return True, "capability summarize (F5 replaces stub)"
        return False, "summarize modified but current is not the stub"
    return False, (f"added={sorted(added)} removed={sorted(removed)} "
                   f"modified={sorted(modified)}")


def _state_key(files) -> str:
    blob = json.dumps(dict(sorted(files.items())), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class AtomicPolicyBuilder:
    """Test-only wrapper: admits atomic proposals, drops the rest.

    The policy_log records every drop with the parent state key, the
    proposal intent, and the reason. Cache-hit re-filtering can log the
    same (state, intent) twice; the report counts distinct pairs as
    well as raw events, so re-filtering never inflates the rejection
    count silently.
    """

    def __init__(self, inner):
        self.inner = inner
        self.role = inner.role
        self.policy_log: list[dict] = []

    def propose(self, projection, request_context):
        current_files = dict(projection.target_files)
        state_key = _state_key(current_files)
        kept = []
        for proposal in self.inner.propose(projection, request_context):
            ok, reason = check_atomic(current_files, proposal)
            if ok:
                kept.append(proposal)
            else:
                self.policy_log.append({
                    "event": "policy_rejected", "state_key": state_key,
                    "intent": proposal.intent, "reason": reason,
                })
        return tuple(kept)


# ----------------------------------------------------------------------
# 2B prompt (capability list + atomic instruction + root note)
# ----------------------------------------------------------------------

_RESPONSE_SCHEMA_HINT = (
    "Respond with JSON only, no other text, matching this shape:\n"
    '{"intent": "short description of the change",\n'
    ' "files": {"solution.py": "<complete file content>"}}\n'
    "The files map must contain the COMPLETE new content of solution.py, "
    "not a diff."
)


def build_2b_prompt(projection, *, layout_sentence: str) -> str:
    """2B prompt: same skeleton as 2A, with the capability list and the
    atomic-move instruction replacing the one-incremental-change line.
    Deliberately OMITS critiques and attempt_number (arm-varying, as
    in 2A). Prior failures ARE included (repair context, in the key).
    The root note appears ONLY when the current state lacks summarize:
    the contract requires it in every state, so the first proposal must
    establish it (stub or full F5). That is a factual constraint
    disclosure; the atomic policy, not the prompt, is the guarantee.
    """
    spec = projection.spec
    lines = [
        f"Problem: {spec.problem_id}",
        f"Description: {spec.description}",
        "Requirements:",
        *[f"  - {r}" for r in spec.requirements],
        "Interface constraints:",
        *[f"  - {k}: {v}" for k, v in spec.interface_constraints.items()],
        "Acceptance tests (your code must pass these):",
        *[f"  - {t.name}: {t.description}"
          for t in spec.acceptance_tests],
        "Capabilities (implement EXACTLY ONE per proposal):",
        *[f"  - {CAPABILITY_BEHAVIOUR[c]}" for c in CAPABILITIES],
    ]
    if projection.architecture is not None:
        lines += [
            "Architecture guidance:",
            f"  intent: {projection.architecture.intent}",
            *[f"  step: {s}"
              for s in projection.architecture.implementation_steps],
        ]
    if projection.target_files:
        lines += ["Existing files you may build on:"]
        for path, content in sorted(projection.target_files.items()):
            lines += [f"--- {path} ---", content]
    if projection.prior_failures:
        # 2B-2 prep framing (test-side prompt only; the atomic rule
        # below is unchanged). GATE_PASSED intermediates carry their own
        # ADVERSARIAL_FAILURE records: attacks fail because the state is
        # INCOMPLETE, not broken. In 2B-1, framing those as "mistakes to
        # repair" drove the scaffold expansion to over-reach (a full
        # solution in one proposal -> policy drop -> branch death),
        # while the same state with a clean prompt got atomic proposals.
        # So adversarial-failure records are described as expected
        # incompleteness; every other failure keeps the FAILED framing.
        expected = [f for f in projection.prior_failures
                    if f.failure_type is FailureType.ADVERSARIAL_FAILURE]
        genuine = [f for f in projection.prior_failures
                   if f.failure_type is not FailureType.ADVERSARIAL_FAILURE]
        if expected:
            lines += [
                "Your current state is a VALID intermediate: it passes "
                "the structural checks, but the behavioral attacks fail "
                "because it is INCOMPLETE, not because it is broken. "
                "This is expected — do not try to fix everything at "
                "once.",
            ]
            for f in expected[-3:]:
                lines += [f"  - incomplete (attacks pending): "
                          f"{f.failure_type} at stage '{f.stage}': "
                          f"{f.evidence}"]
        if genuine:
            lines += ["Previous attempts FAILED. Do not repeat these "
                      "mistakes:"]
            for f in genuine[-3:]:
                lines += [f"  - {f.failure_type} at stage '{f.stage}': "
                          f"{f.evidence}"]
    lines += [
        "ATOMIC MOVE RULE: your proposal must implement EXACTLY ONE "
        "capability from the list above and change nothing else. Do "
        "NOT write the complete solution in one proposal. Do NOT add "
        "extra helper functions, and do NOT modify functions that "
        "already exist.",
    ]
    _has_summarize = False
    _sol = (projection.target_files or {}).get("solution.py", "")
    if _sol:
        try:
            _has_summarize = "summarize" in _top_defs(_sol)
        except ValueError:
            _has_summarize = False
    if not _has_summarize:
        lines += [
            "ROOT NOTE: the current state has no summarize function, "
            "but the contract requires summarize in every state. Your "
            "first proposal must therefore establish summarize: either "
            "the stub (def summarize(text): return {}) or the full "
            "summarize function (capability F5).",
        ]
    lines += [layout_sentence, "", _RESPONSE_SCHEMA_HINT]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Arms (canonical only in Phase 2B-1)
# ----------------------------------------------------------------------

def _rse_run_2b(builder):
    """Real SearchEngine on the 2B problem. max_depth 6 per Amendment A
    (scaffold level + five capabilities); everything else as in 2A,
    except the wall-clock budget: 1500s (2B-2 prep — the 2B-1 run died
    on the 300s production default after 4 expansions, while a
    convergence diamond needs ~6+; 1500s fits ~15-20 expansions at the
    observed ~95s/expansion latency). Test-only config; the production
    default is untouched."""
    spec = AtomicProblem().spec
    cfg = ForgeMindConfig(
        random_seed=0, enable_optimizer=False, beam_width=3,
        branching_factor=2, max_depth=6, max_nodes=32,
        max_wall_clock_seconds=1500)
    env = SubprocessExecutionEnvironment()
    referee = Referee(spec, env, cfg)
    referee.attach_test_provider(AtomicProblem().test_files)
    engine = SearchEngine(
        spec, cfg, builder=builder, critic=FixedAttackCritic2B(),
        referee=referee, recovery=RecoveryPolicy())
    return engine, engine.run()


def run_greedy_2b(builder, *, max_steps: int = 20) -> dict:
    """Greedy on the same builder/verifier, mirroring 2A's run_greedy.

    Starts from the same empty root as the RSE arm (the engine forces
    it), so root cache keys match. Commits the single best child per
    step by production guidance; never retains a frontier.
    """
    spec = AtomicProblem().spec
    cfg = ForgeMindConfig(random_seed=0)
    env = SubprocessExecutionEnvironment()
    referee = Referee(spec, env, cfg)
    referee.attach_test_provider(AtomicProblem().test_files)
    scorer = ScoringPolicy(ScoringWeights())
    trans = TransitionEngine(spec)
    architect = MockArchitect()
    state = CodebaseState(spec_hash=spec.spec_hash, files={})
    arch = architect.propose(build_architect_projection(spec, state), {})[0]
    committed: list[str] = []
    verifications = 0
    depth = 0
    for _ in range(max_steps):
        proj = build_builder_projection(
            spec, state, architecture=arch, failures=(), critiques=(),
            attempt_number=0)
        proposals = builder.propose(proj, {})
        if not proposals:
            return {"success": False, "committed": committed,
                    "verifications": verifications, "depth": depth}
        ranked = []
        for order, proposal in enumerate(proposals):
            tresult = trans.apply(state, proposal)
            if not tresult.ok:
                continue
            verification = referee.verify(tresult.new_state,
                                          attack_tests=_ATTACKS_2B)
            verifications += 1
            gate = hard_gate(verification, adversarial_expected=True)
            scores = scorer.score(verification, gate, depth=depth + 1)
            if gate.passed:
                committed.append(tresult.new_state.content_hash)
                return {"success": True, "committed": committed,
                        "verifications": verifications, "depth": depth + 1}
            ranked.append((-scores.guidance, order, tresult.new_state))
        if not ranked:
            return {"success": False, "committed": committed,
                    "verifications": verifications, "depth": depth}
        state = ranked[0][2]
        committed.append(state.content_hash)
        depth += 1
    return {"success": False, "committed": committed,
            "verifications": verifications, "depth": depth}


# ----------------------------------------------------------------------
# Measurement additions (subset coverage)
# ----------------------------------------------------------------------

def capability_subset(files) -> list[str]:
    """Capability names present in a state's solution.py.

    summarize counts as present only when it is NOT the stub; the stub
    is the scaffold, not the F5 capability.
    """
    source = files.get("solution.py", "")
    if not source:
        return []
    try:
        defs = _top_defs(source)
    except ValueError:
        return []
    present = [c for c in HELPERS if c in defs]
    if "summarize" in defs and not _summarize_is_stub(source):
        present.append("summarize")
    return sorted(present)


def subset_coverage(node_dicts) -> dict:
    """How many nodes carry each capability subset (subset -> count)."""
    counts: dict[str, int] = {}
    for node in node_dicts.values():
        subset = ",".join(capability_subset(node["state"]["files"]))
        counts[subset] = counts.get(subset, 0) + 1
    return counts


# ----------------------------------------------------------------------
# Keyless unit tests (no network, no connector)
# ----------------------------------------------------------------------

def _proposal(*, created=None, modified=None, intent="test"):
    return CodePatchProposal(
        agent_role="builder", intent=intent, rationale="unit",
        estimated_cost=1.0, created_files=created or {},
        modified_files=modified or {})


def _helper_source(name: str) -> str:
    bodies = {
        "count_words": "def count_words(text):\n    return len(text.split())\n",
        "count_chars": "def count_chars(text):\n    return len(text)\n",
        "longest_token": "def longest_token(text):\n"
                         "    toks = text.split()\n"
                         "    return max(toks, key=len) if toks else ''\n",
        "mean_length": "def mean_length(text):\n"
                       "    toks = text.split()\n"
                       "    return sum(map(len, toks)) / len(toks) "
                       "if toks else 0.0\n",
    }
    return bodies[name]


def test_policy_admits_single_helper_from_stub():
    ok, reason = check_atomic(
        {"solution.py": STUB_SOURCE},
        _proposal(modified={"solution.py":
                            STUB_SOURCE + _helper_source("count_words")}))
    assert ok and reason == "capability count_words"


def test_policy_admits_second_helper():
    current = STUB_SOURCE + _helper_source("count_words")
    ok, reason = check_atomic(
        {"solution.py": current},
        _proposal(modified={"solution.py":
                            current + _helper_source("count_chars")}))
    assert ok and reason == "capability count_chars"


def test_policy_admits_f5_replace_stub():
    real = ("def summarize(text):\n"
            "    return {'words': count_words(text)}\n")
    ok, reason = check_atomic(
        {"solution.py": STUB_SOURCE},
        _proposal(modified={"solution.py": real}))
    assert ok and "F5 replaces stub" in reason


def test_policy_admits_f5_replace_stub_with_helpers_present():
    current = STUB_SOURCE + _helper_source("count_words")
    real = ("def summarize(text):\n"
            "    return {'words': count_words(text)}\n")
    ok, reason = check_atomic(
        {"solution.py": current},
        _proposal(modified={"solution.py":
                            current.replace(STUB_SOURCE, real)}))
    assert ok and "F5 replaces stub" in reason


def test_policy_admits_scaffold_from_empty_root():
    ok, reason = check_atomic(
        {}, _proposal(created={"solution.py": STUB_SOURCE}))
    assert ok and reason == "scaffold"


def test_policy_admits_f5_first_from_empty_root():
    real = ("def summarize(text):\n"
            "    return {'words': count_words(text)}\n")
    ok, reason = check_atomic(
        {}, _proposal(created={"solution.py": real}))
    assert ok and "F5 first" in reason


def test_policy_rejects_two_capabilities():
    current = STUB_SOURCE
    both = (current + _helper_source("count_words")
            + _helper_source("count_chars"))
    ok, reason = check_atomic(
        {"solution.py": current},
        _proposal(modified={"solution.py": both}))
    assert not ok and "count_chars" in reason and "count_words" in reason


def test_policy_rejects_extra_helper():
    current = STUB_SOURCE + _helper_source("count_words")
    extra = current + "def _private(text):\n    return text\n"
    ok, _ = check_atomic(
        {"solution.py": current},
        _proposal(modified={"solution.py": extra}))
    assert not ok


def test_policy_rejects_modify_existing():
    current = STUB_SOURCE + _helper_source("count_words")
    changed = current.replace("len(text.split())", "len(text.split(' '))")
    ok, _ = check_atomic(
        {"solution.py": current},
        _proposal(modified={"solution.py": changed}))
    assert not ok


def test_policy_rejects_remove_function():
    current = STUB_SOURCE + _helper_source("count_words")
    ok, _ = check_atomic(
        {"solution.py": current},
        _proposal(modified={"solution.py": STUB_SOURCE}))
    assert not ok


def test_policy_rejects_new_file():
    ok, _ = check_atomic(
        {"solution.py": STUB_SOURCE},
        _proposal(modified={"solution.py": STUB_SOURCE},
                  created={"helper.py": "x = 1\n"}))
    assert not ok


def test_policy_rejects_unparseable():
    ok, reason = check_atomic(
        {"solution.py": STUB_SOURCE},
        _proposal(modified={"solution.py": "def broken(:\n"}))
    assert not ok and "unparseable" in reason


def test_policy_rejects_no_change():
    ok, _ = check_atomic(
        {"solution.py": STUB_SOURCE},
        _proposal(modified={"solution.py": STUB_SOURCE}))
    assert not ok


def test_policy_wrapper_filters_and_logs():
    class StubInner:
        role = "builder"

        def propose(self, projection, ctx):
            current = projection.target_files["solution.py"]
            return (
                _proposal(modified={"solution.py":
                                    current + _helper_source("count_words")},
                          intent="good"),
                _proposal(modified={"solution.py":
                                    current + _helper_source("count_words")
                                    + _helper_source("count_chars")},
                          intent="greedy"),
            )

    class FakeProjection:
        target_files = {"solution.py": STUB_SOURCE}

    wrapper = AtomicPolicyBuilder(StubInner())
    kept = wrapper.propose(FakeProjection(), {})
    assert len(kept) == 1 and kept[0].intent == "good"
    assert len(wrapper.policy_log) == 1
    assert wrapper.policy_log[0]["intent"] == "greedy"


def test_problem_spec_resolves_and_compiles():
    spec = AtomicProblem().spec
    assert spec.interface_constraints == {"summarize":
                                          "function: summarize(text)"}
    assert [t.test_id for t in spec.acceptance_tests] == [
        "test_imports", "test_summarize_callable"]
    for source in AtomicProblem().test_files().values():
        compile(source, "<acceptance>", "exec")
    for attack in _ATTACKS_2B:
        compile(attack.test_code, "<attack>", "exec")
    assert len(_ATTACKS_2B) == 7


def test_stub_detection():
    assert _is_stub(STUB_SOURCE)
    assert not _is_stub("def summarize(text):\n    return {'a': 1}\n")
    assert not _is_stub("def summarize(text):\n    return {}\n"
                       "def count_words(text):\n    return 0\n")


def test_prompt_contains_capabilities_and_atomic_rule():
    class FakeProjection:
        spec = AtomicProblem().spec
        architecture = None
        target_files = {"solution.py": STUB_SOURCE}
        prior_failures = ()

    prompt = build_2b_prompt(
        FakeProjection(), layout_sentence="LAYOUT-CANONICAL")
    for name in CAPABILITIES:
        assert name in prompt
    assert "EXACTLY ONE" in prompt
    assert "LAYOUT-CANONICAL" in prompt
    assert "ROOT NOTE" not in prompt  # summarize present in this state


def test_prompt_root_note_only_when_summarize_absent():
    class FakeProjection:
        spec = AtomicProblem().spec
        architecture = None
        target_files = {}
        prior_failures = ()

    prompt = build_2b_prompt(
        FakeProjection(), layout_sentence="LAYOUT-CANONICAL")
    assert "ROOT NOTE" in prompt


def test_prompt_variants_differ_only_by_layout():
    class FakeProjection:
        spec = AtomicProblem().spec
        architecture = None
        target_files = {"solution.py": STUB_SOURCE}
        prior_failures = ()

    main = build_2b_prompt(
        FakeProjection(), layout_sentence="LAYOUT-CANONICAL")
    control = build_2b_prompt(
        FakeProjection(), layout_sentence="LAYOUT-APPEND")
    assert main.replace("LAYOUT-CANONICAL", "X") == control.replace(
        "LAYOUT-APPEND", "X")


def test_subset_coverage_counts():
    nodes = {
        "n1": {"state": {"files": {"solution.py": STUB_SOURCE}}},
        "n2": {"state": {"files": {"solution.py": STUB_SOURCE
                                     + _helper_source("count_words")}}},
        "n3": {"state": {"files": {"solution.py": STUB_SOURCE
                                     + _helper_source("count_words")
                                     + _helper_source("count_chars")}}},
    }
    coverage = subset_coverage(nodes)
    assert coverage[""] == 1
    assert coverage["count_words"] == 1
    assert coverage["count_chars,count_words"] == 1


def _framing_projection(failures):
    class FakeProjection:
        spec = AtomicProblem().spec
        architecture = None
        target_files = {"solution.py": STUB_SOURCE}
        prior_failures = tuple(failures)

    return FakeProjection()


def _record(failure_type, stage, evidence):
    return RecoveryPolicy().record(
        state_hash="h", candidate_id="c", failure_type=failure_type,
        stage=stage, evidence=evidence)


def test_prompt_frames_adversarial_failure_as_expected_incompleteness():
    adv = _record(FailureType.ADVERSARIAL_FAILURE, "adversarial",
                  "ADV-EVIDENCE")
    prompt = build_2b_prompt(_framing_projection([adv]),
                             layout_sentence="LAYOUT-CANONICAL")
    assert "VALID intermediate" in prompt
    assert "INCOMPLETE" in prompt
    assert "ADV-EVIDENCE" in prompt
    assert "Previous attempts FAILED" not in prompt
    assert "EXACTLY ONE" in prompt  # atomic rule preserved


def test_prompt_keeps_failed_framing_for_genuine_failures():
    gen = _record(FailureType.SYNTAX_ERROR, "transition", "GEN-EVIDENCE")
    prompt = build_2b_prompt(_framing_projection([gen]),
                             layout_sentence="LAYOUT-CANONICAL")
    assert "Previous attempts FAILED" in prompt
    assert "GEN-EVIDENCE" in prompt
    assert "VALID intermediate" not in prompt


def test_prompt_splits_mixed_failures_by_kind():
    adv = _record(FailureType.ADVERSARIAL_FAILURE, "adversarial",
                  "ADV-EVIDENCE")
    gen = _record(FailureType.SYNTAX_ERROR, "transition", "GEN-EVIDENCE")
    prompt = build_2b_prompt(_framing_projection([adv, gen]),
                             layout_sentence="LAYOUT-CANONICAL")
    assert (prompt.index("VALID intermediate")
            < prompt.index("ADV-EVIDENCE")
            < prompt.index("Previous attempts FAILED")
            < prompt.index("GEN-EVIDENCE"))


def test_prompt_no_failure_blocks_without_failures():
    prompt = build_2b_prompt(_framing_projection([]),
                             layout_sentence="LAYOUT-CANONICAL")
    assert "VALID intermediate" not in prompt
    assert "Previous attempts FAILED" not in prompt
    assert "EXACTLY ONE" in prompt


def test_report_records_termination_reason_and_wall_clock():
    class StubBuilder:
        role = "builder"

        def propose(self, projection, request_context):
            return ()

    class StubInner:
        log = []
        _provider = "test"
        _model = "test"
        _temperature = 0.0
        _max_tokens = 0
        _seed = 0

    builder = AtomicPolicyBuilder(StubBuilder())
    engine, result = _rse_run_2b(builder)
    # No proposals: root expands to nothing, frontier empties.
    assert result.termination.reason == "frontier_empty"
    assert engine.config.max_wall_clock_seconds == 1500
    report = build_phase_report(
        experiment_id="2B-TEST", phase="test", inner=StubInner(),
        builder=builder, result=result,
        measurement=tlc.rse_arm_measurement(engine, result),
        greedy={"success": False, "committed": [], "verifications": 0,
                "depth": 0},
        max_calls=32)
    assert report["termination"] == {"reason": "frontier_empty",
                                     "accepted_node_id": None}
    assert report["config"]["max_wall_clock_seconds"] == 1500


def test_append_variant_prompt_uses_append_sentence():
    prompt = build_2b_prompt(_framing_projection([]),
                             layout_sentence=tlc.LAYOUT_SENTENCES["append"])
    assert tlc.LAYOUT_SENTENCES["append"] in prompt
    assert tlc.LAYOUT_SENTENCES["canonical"] not in prompt
    assert "EXACTLY ONE" in prompt  # atomic rule intact


def test_append_and_canonical_cache_keys_differ():
    # Same projection, different variant -> different canonical keys,
    # so append generations can never be served from canonical cache
    # entries (or vice versa). Budget 0: propose logs budget_exhausted
    # with the key and returns () without touching the network.
    proj = _framing_projection([])
    keys = []
    for variant in ("canonical", "append"):
        builder = OpenRouterBuilder(
            variant=variant, max_proposals=2, budget=CallBudget(0),
            temperature=tlc.TEMPERATURE, max_tokens=MAX_TOKENS_2B,
            seed=tlc.SEED, cache_path=None,
            prompt_builder=build_2b_prompt)
        assert builder.propose(proj, {"source_node_id": "n1"}) == ()
        assert builder.log[-1]["event"] == "budget_exhausted"
        keys.append(builder.log[-1]["canonical_key"])
    assert keys[0] != keys[1]


# ----------------------------------------------------------------------
# Report assembly (shared by the live test and keyless shape tests)
# ----------------------------------------------------------------------

def build_phase_report(*, experiment_id, phase, inner, builder, result,
                       measurement, greedy, max_calls):
    """Assemble a 2B phase report dict (pure assembly, no I/O).

    The live test writes the returned dict to REPORT_PATH_2B; keyless
    tests call it with stub runs to verify the shape, including the
    termination recording added in 2B-2 prep (2B-1's postmortem had to
    prove wall_clock termination by elimination + timestamps because
    the 2B-1 report did not persist it).
    """
    policy_events = builder.policy_log
    distinct = {(e["state_key"], e["intent"]) for e in policy_events}
    by_reason: dict[str, int] = {}
    for e in policy_events:
        by_reason[e["reason"]] = by_reason.get(e["reason"], 0) + 1
    # Nodes keyed by node id for coverage; convergence state_ids are
    # content hashes, so a separate hash -> files map joins them.
    node_dicts = {node.node_id: {"state": {"files": dict(node.state.files)}}
                  for node in result.graph.nodes.values()}
    hash_to_files = {node.state.content_hash: dict(node.state.files)
                     for node in result.graph.nodes.values()}
    coverage = subset_coverage(node_dicts)
    convergence = measurement.get("convergence_points", [])
    conv_subsets = []
    for point in convergence:
        state_id = point.get("state_id")
        files = hash_to_files.get(state_id, {})
        conv_subsets.append({"state_id": state_id,
                             "subset": capability_subset(files),
                             "histories": point.get("histories")})

    fresh_calls = sum(1 for e in inner.log if e.get("event") == "call_ok")
    cache_hits = sum(1 for e in inner.log if e.get("event") == "cache_hit")
    failed = sum(1 for e in inner.log if e.get("event") == "call_failed")
    malformed = sum(1 for e in inner.log
                    if e.get("event") == "call_malformed")
    return {
        "experiment_id": experiment_id,
        "phase": phase,
        "problem": "wordstats-atomic-v1",
        "provider": inner._provider,
        "selected_model": inner._model,
        "model_id": inner._model,
        "generation_parameters": {
            "temperature": inner._temperature,
            "max_tokens": inner._max_tokens,
            "seed": inner._seed,
            "seed_note": "seed recorded but NOT sent to the endpoint",
        },
        "config": {
            "beam_width": 3, "branching_factor": 2, "max_depth": 6,
            "max_nodes": 32, "max_wall_clock_seconds": 1500,
            "optimizer": "disabled",
            "amendment": "Amendment A: engine starts empty, scaffold rule, "
                          "max_depth 6 (not 5); 2B-2 prep: wall clock "
                          "1500s (was 300s production default)",
        },
        "termination": {
            "reason": result.termination.reason,
            "accepted_node_id": result.termination.accepted_node_id,
        },
        "budget": {"max_calls": max_calls, "fresh_calls": fresh_calls,
                   "unused": max_calls - fresh_calls},
        "generation": {"fresh_calls": fresh_calls, "cache_hits": cache_hits,
                       "failed_calls": failed, "malformed_calls": malformed,
                       "call_events": inner.log},
        "policy": {"rejected_events": len(policy_events),
                   "rejected_distinct": len(distinct),
                   "by_reason": by_reason, "log": policy_events},
        "arms": {
            "canonical": {
                "rse": measurement,
                "greedy": greedy,
                "subset_coverage": coverage,
                "convergence_points_with_subsets": conv_subsets,
            },
        },
        "viability": {
            "histories_ge_3": measurement.get("histories", 0) >= 3,
            "convergence_point_ge_1": len(convergence) >= 1,
            "duplicates_gt_0":
                measurement.get("duplicate_encounters", 0) > 0,
            "multiple_verifications":
                measurement.get("verification_metric", 0) > 1,
        },
    }


# ----------------------------------------------------------------------
# Live test — Phase 2B-1 (canonical RSE + Greedy only)
# ----------------------------------------------------------------------

@pytest.mark.skipif(
    not LIVE_2B,
    reason="live 2B run needs FORGEMIND_LLM_2B_LIVE=1 and the connector")
def test_llm_convergence_2b_phase1():
    budget = CallBudget(MAX_CALLS_2B)
    inner = OpenRouterBuilder(
        variant="canonical", max_proposals=2, budget=budget,
        temperature=tlc.TEMPERATURE, max_tokens=MAX_TOKENS_2B,
        seed=tlc.SEED, cache_path=CACHE_PATH_2B,
        prompt_builder=build_2b_prompt)
    builder = AtomicPolicyBuilder(inner)
    engine, result = _rse_run_2b(builder)
    measurement = tlc.rse_arm_measurement(engine, result)
    greedy = run_greedy_2b(builder)
    report = build_phase_report(
        experiment_id=os.environ.get("LLM_CONVERGENCE_2B_EXPERIMENT_ID",
                                     "2B-1"),
        phase=os.environ.get(
            "LLM_CONVERGENCE_2B_PHASE",
            "canonical only; append control NOT run (Phase 2B-2 gated "
            "on viability)"),
        inner=inner, builder=builder, result=result,
        measurement=measurement, greedy=greedy, max_calls=MAX_CALLS_2B)
    with open(REPORT_PATH_2B, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
    # Well-formedness only — viability is Kashi's judgement, not an
    # assertion.
    fresh_calls = report["generation"]["fresh_calls"]
    assert fresh_calls <= MAX_CALLS_2B
    assert measurement["histories"] >= measurement["unique_states"] >= 1
    assert set(report["viability"]) == {
        "histories_ge_3", "convergence_point_ge_1", "duplicates_gt_0",
        "multiple_verifications"}
    assert isinstance(report["termination"]["reason"], str) \
        and report["termination"]["reason"]


# ----------------------------------------------------------------------
# Live test — Phase 2B-2 append control (append RSE + Greedy only)
# ----------------------------------------------------------------------

@pytest.mark.skipif(
    not LIVE_2B_APPEND,
    reason="live 2B append run needs FORGEMIND_LLM_2B_APPEND_LIVE=1 and "
           "the connector")
def test_llm_convergence_2b_phase2_append():
    budget = CallBudget(MAX_CALLS_2B)
    inner = OpenRouterBuilder(
        variant="append", max_proposals=2, budget=budget,
        temperature=tlc.TEMPERATURE, max_tokens=MAX_TOKENS_2B,
        seed=tlc.SEED, cache_path=CACHE_PATH_2B_APPEND,
        prompt_builder=build_2b_prompt)
    builder = AtomicPolicyBuilder(inner)
    engine, result = _rse_run_2b(builder)
    measurement = tlc.rse_arm_measurement(engine, result)
    greedy = run_greedy_2b(builder)
    report = build_phase_report(
        experiment_id=os.environ.get(
            "LLM_CONVERGENCE_2B_APPEND_EXPERIMENT_ID", "2B-2-append"),
        phase=os.environ.get(
            "LLM_CONVERGENCE_2B_APPEND_PHASE",
            "2B-2 append control only (canonical arms from 2B-2 report)"),
        inner=inner, builder=builder, result=result,
        measurement=measurement, greedy=greedy, max_calls=MAX_CALLS_2B)
    with open(REPORT_PATH_2B_APPEND, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
    # Well-formedness only — viability and the append comparison are
    # Kashi's judgement, not assertions.
    fresh_calls = report["generation"]["fresh_calls"]
    assert fresh_calls <= MAX_CALLS_2B
    assert measurement["histories"] >= measurement["unique_states"] >= 1
    assert set(report["viability"]) == {
        "histories_ge_3", "convergence_point_ge_1", "duplicates_gt_0",
        "multiple_verifications"}
    assert isinstance(report["termination"]["reason"], str) \
        and report["termination"]["reason"]
