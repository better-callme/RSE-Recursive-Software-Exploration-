"""ForgeMind CLI.

Usage:
    python -m forgemind run --problem stats
    python -m forgemind benchmark --problem stats [--output report.json]
"""

from __future__ import annotations

import argparse
import sys

from forgemind.benchmark.experiment import ExperimentRunner
from forgemind.benchmark.problems.stats_problem import StatsProblem
from forgemind.core.config import ForgeMindConfig


_PROBLEMS = {
    "stats": lambda: StatsProblem(),
}


def _problem_spec(name: str) -> ProblemSpec:
    problem = _PROBLEMS[name]()
    spec = problem.spec
    if hasattr(problem, "test_files"):
        object.__setattr__(spec, "_test_provider_ref", problem.test_files)
    return spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forgemind")
    parser.add_argument("--seed", type=int, default=0)
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run ForgeMind search on one problem")
    run_p.add_argument("--problem", choices=sorted(_PROBLEMS), default="stats")
    run_p.add_argument("--beam-width", type=int, default=3)
    run_p.add_argument("--max-depth", type=int, default=6)

    bench_p = sub.add_parser("benchmark", help="compare ForgeMind vs baseline")
    bench_p.add_argument("--problem", choices=sorted(_PROBLEMS), default="stats")
    bench_p.add_argument("--output", default="forgemind_report.json")

    trace_p = sub.add_parser("trace", help="print the event log of a run")
    trace_p.add_argument("--problem", choices=sorted(_PROBLEMS), default="stats")
    return parser


def _engine_for(args) -> tuple:
    spec = _problem_spec(args.problem)
    config = ForgeMindConfig(
        random_seed=args.seed,
        beam_width=getattr(args, "beam_width", 3),
        max_depth=getattr(args, "max_depth", 6),
    )
    runner = ExperimentRunner(config)
    from forgemind.search.engine import SearchEngine
    from forgemind.recovery.failure import RecoveryPolicy
    from forgemind.core.history import History

    engine = SearchEngine(spec, config, referee=runner._referee(spec),
                          recovery=RecoveryPolicy(), history=History())
    return runner, engine


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = _problem_spec(args.problem)

    if args.command == "run":
        _, engine = _engine_for(args)
        result = engine.run()
        print(result.summary())
        node = result.accepted_node
        if node is not None and node.state.files:
            for path in sorted(node.state.files):
                print(f"accepted file: {path}")
            return 0
        return 1

    if args.command == "trace":
        _, engine = _engine_for(args)
        result = engine.run()
        print(engine.history.render_trace())
        print()
        print(result.summary())
        return 0 if result.success else 1

    if args.command == "benchmark":
        runner = ExperimentRunner(ForgeMindConfig(random_seed=args.seed))
        print("running baseline ...")
        baseline = runner.run_baseline([spec])
        print("running forgemind ...")
        fm = runner.run_forgemind([spec])
        comparison = fm.compare(baseline)
        out = runner.save_report(fm, args.output)
        baseline_out = str(args.output).replace(".json", "_baseline.json")
        runner.save_report(baseline, baseline_out)
        print(f"forgemind success rate : {fm.success_rate:.2f}")
        print(f"baseline success rate  : {baseline.success_rate:.2f}")
        print(f"delta success          : {comparison['delta_success']:+.2f}")
        print(f"cost ratio (fm/base)   : {comparison['cost_ratio']:.2f}x")
        print(f"report written to      : {out}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
