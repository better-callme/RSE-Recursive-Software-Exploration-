# ForgeMind v0.1 — Deterministic Search Foundation

ForgeMind is an experimental framework for testing whether structured search
and deterministic verification improve software-generation reliability.

It treats software generation as **constrained state-space exploration**:
probabilistic agents (LLMs or deterministic mocks) *propose* moves; a
deterministic engine *decides* — validating proposals, transforming immutable
states, hashing content, deduplicating, executing sandboxed verification,
scoring results, pruning, backtracking, and accepting.

## Central architectural law

> Agents propose. The deterministic engine decides.

```
Agent → Proposal → Candidate → TransitionEngine → CodebaseState'
                                                       │
                                     Referee (sandboxed execution)
                                                       │
                                        HardGate → Scoring → Frontier
```

Agents never mutate canonical state, never see internal search objects (only
deterministic *projections*), and can never mark anything accepted. Every
acceptance requires objective verification evidence produced by the Referee.

## Research hypothesis

Compare:

- **Baseline**: one deterministic proposal, verified once (including the
  same Tier-3 adversarial tests the engine faces, so "success" means the
  same thing in both arms).
- **ForgeMind**: bounded beam search over multiple genuine alternatives with
  failure-driven recovery, duplicate pruning, stagnation detection, and
  adversarial (critic-generated) test tiers.

Measured: success rate, test pass rate, wall-clock cost, agent calls,
verifications, candidates, nodes explored/pruned, duplicates, backtracks,
stagnation events, token estimates.

### Benchmark methodology

```bash
python -m forgemind benchmark                 # full suite (default)
python -m forgemind benchmark --problem stats # one problem
```

The suite runs every registered problem in `forgemind/benchmark/problems`
(`stats`, plus `stats-trap`) and aggregates per-problem success rates and
mean wall-clock cost ratio into one report.

`stats-trap` is the discriminating case: the same `stats-v1` spec, but the
mock builder emits wrong variants first (truncated mean, then broken
empty-list handling). The baseline takes variant 0 and fails; ForgeMind
must recover through search. With deterministic mocks there is no other
source of variation, so the trap is what makes the comparison informative.
Both arms face identical verification tiers — the measured delta is
attributable to search, not to a different bar.

Honest limits: two problems sharing one spec is still a thin evidence base,
and deterministic mocks stand in for probabilistic LLM proposers (see MVP
limitations). Add problems and an LLM agent adapter to harden the claim.

## Architecture

```
src/forgemind/
├── core/          ProblemSpec, CodebaseState, StateIdentity (SHA-256),
│                  Proposals, Candidate, TransitionEngine, History, Config
├── search/        SearchNode, SearchGraph, SearchFrontier, BeamSearchStrategy,
│                  StagnationDetector, SearchEngine (the loop)
├── agents/        Deterministic projections + MockArchitect/Builder/Critic/
│                  Optimizer (replaceable by LLM adapters)
├── verification/  ExecutionEnvironment protocol, SubprocessExecutionEnvironment,
│                  StaticVerifier (Tier 1), Referee (Tiers 1–3), HardGate,
│                  ScoringPolicy (guidance vs acceptance)
├── recovery/      FailureRecord types + deterministic RecoveryPolicy mapping
└── benchmark/     StatsProblem, Baseline runner, ExperimentRunner, metrics, CLI
```

### Multi-tier verification

- **Tier 1 — static**: AST parsing, syntax validity, forbidden constructs
  (`eval`/`exec`/`compile`, dangerous imports). Cheap early rejection.
- **Tier 2 — dynamic**: acceptance tests executed in an isolated subprocess
  under CPU/memory/wall-clock limits with sanitized environment.
- **Tier 3 — adversarial/regression**: critic-generated attack tests and
  regression suites. Critic opinions are hypotheses; only Referee execution is
  evidence.

### Acceptance vs guidance

Acceptance is binary hard gating (`syntax ∧ contract ∧ requiredTests ∧
security ∧ regression`). A state at 99/100 tests is still rejected — but its
pass rate feeds a separate **guidance score** used to rank repair targets.
Broken code can never be compensated by soft scores.

## Installation

```bash
cd forgemind
pip install -e ".[dev]"     # Python 3.12+, stdlib-only runtime deps
```

## Running tests

```bash
PYTHONPATH=src python -m pytest tests -q
```

Suites: `tests/unit`, `tests/search`, `tests/verification`,
`tests/security`, `tests/integration`, `tests/benchmark`.

## Running

```bash
PYTHONPATH=src python -m forgemind run --problem stats       # one search
PYTHONPATH=src python -m forgemind trace --problem stats     # full event log
PYTHONPATH=src python -m forgemind benchmark --problem stats # vs baseline
```

Example output (`run`):

```
SEARCH COMPLETE
outcome = ACCEPTED
nodes explored = 1
candidates = 1
verifications = 3
elapsed = 0.12s
accepted file: solution.py
```

The `trace` command prints the append-only event log (SEARCH_STARTED,
NODE_SELECTED, AGENT_CALLED, PROPOSAL_GENERATED, CANDIDATE_CREATED,
TRANSITION_APPLIED/REJECTED, DUPLICATE_DETECTED, VERIFICATION_*, FAILURE_,
STAGNATION_DETECTED, BACKTRACK, STATE_ACCEPTED, SEARCH_TERMINATED) so any run
is fully explainable after the fact.

## HTTP API

ForgeMind also ships a stdlib-only HTTP API (no third-party runtime
dependencies):

```bash
python -m forgemind serve --port 8000
```

| Method | Endpoint        | Description                                    |
|--------|-----------------|------------------------------------------------|
| GET    | /api/health     | `{"status": "ok", "version": ...}`             |
| GET    | /api/problems   | list of registered problem ids                 |
| POST   | /api/run        | run ForgeMind search, return metrics + summary |
| POST   | /api/trace      | run search, return the full event trace         |
| POST   | /api/benchmark  | baseline vs ForgeMind suite comparison     |

POST bodies are JSON: `/api/run` and `/api/trace` take `{"problem":
"stats", "seed": 0, "beam_width": 3, "max_depth": 6}`; `/api/benchmark`
takes `{"problems": ["stats", "stats-trap"], "seed": 0}` (defaults to the
full suite). Unknown problems return 404; malformed JSON returns 400.

## Security limitations (read this)

Candidate code is untrusted. The v0.1 subprocess environment provides:

- fresh temporary working directory per verification,
- wall-clock timeout with process-group SIGKILL,
- RLIMIT_CPU / RLIMIT_AS / RLIMIT_FSIZE / RLIMIT_CORE applied inside the child
  via an exec bootstrap (never `preexec_fn`, which is unsafe under CPython's
  vfork-based Popen),
- new session/process group (`start_new_session=True`) with guard against
  killing our own group,
- allowlisted environment variables; no sensitive inheritance,
- output truncation.

**This is resource isolation / damage reduction — not hostile-code isolation.**
Generated code runs with your user's privileges and can read files you can
read (there is no filesystem jail). AST filtering is defense-in-depth, not a
sandbox boundary. Production deployments should execute candidates in
container/gVisor/VM-grade isolation.

Known platform notes observed during development:

- Setting `RLIMIT_NPROC` interacts badly with some container runtimes' cgroup
  `pids.max` (fork fails outright); process-count containment therefore relies
  on timeout + group kill instead.
- `setsid()` may be forbidden in restricted containers; kill logic falls back
  to per-pid SIGKILL in that case.

## MVP limitations

- Mock agents are deterministic stand-ins; no live LLM integration yet (by
  design — the Agent protocol accepts adapters without engine changes).
- Scoring excludes measured runtime/memory from ranking (OS jitter breaks
  determinism); they're recorded as raw evidence. Weights live in config.
- Optimizer's "maintainability" component is a neutral placeholder.
- Beam search is bounded heuristic search, not A*.
- One bundled benchmark problem (descriptive statistics); the engine is
  problem-agnostic.

## Future LLM integration

Implement the `Agent` protocol (`propose(projection, ctx) -> tuple[Proposal, ...]`)
with prompt serialization + structured-output parsing. The engine, frontier,
referee, and scoring require zero changes — that is the architectural test for
the deterministic/probabilistic boundary.
