"""ForgeMind CLI.

Usage:
    python -m forgemind run --problem stats
    python -m forgemind benchmark --problem stats [--output report.json]
"""

from __future__ import annotations

import argparse
import sys

from forgemind.benchmark.experiment import ExperimentRunner
from forgemind.benchmark.problems import (
    PROBLEMS,
    list_problem_names,
    problem_builder,
    problem_spec,
)
from forgemind.core.config import ForgeMindConfig


# Backwards-compatible re-exports (used by forgemind.api.server).
__all__ = ["build_parser", "main", "PROBLEMS", "problem_spec", "problem_builder",
           "list_problem_names"]

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forgemind")
    parser.add_argument("--seed", type=int, default=0)
    sub = parser.add_subparsers(dest="command", required=True)

    problem_choices = list_problem_names()

    run_p = sub.add_parser("run", help="run ForgeMind search on one problem")
    run_p.add_argument("--problem", choices=problem_choices, default="stats")
    run_p.add_argument("--beam-width", type=int, default=3)
    run_p.add_argument("--max-depth", type=int, default=6)

    bench_p = sub.add_parser("benchmark", help="compare ForgeMind vs baseline (deterministic suite)")
    bench_p.add_argument("--problem", choices=["all"] + problem_choices,
                         default="all",
                         help="one problem id, or 'all' for the full suite")
    bench_p.add_argument("--output", default="forgemind_report.json")

    trace_p = sub.add_parser("trace", help="print the event log of a run")
    trace_p.add_argument("--problem", choices=problem_choices, default="stats")

    serve_p = sub.add_parser("serve", help="start the ForgeMind HTTP API server")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)
    return parser


def _engine_for(args) -> tuple:
    spec = problem_spec(args.problem)
    config = ForgeMindConfig(
        random_seed=args.seed,
        beam_width=getattr(args, "beam_width", 3),
        max_depth=getattr(args, "max_depth", 6),
    )
    runner = ExperimentRunner(config)
    from forgemind.benchmark.kits import mock_kit_for
    from forgemind.search.engine import SearchEngine
    from forgemind.recovery.failure import RecoveryPolicy
    from forgemind.core.history import History

    # H5: the call site resolves the kit and passes explicit agents; the
    # engine itself never selects a problem-specific agent set.
    kit = mock_kit_for(args.problem)
    engine = SearchEngine(spec, config, referee=runner._referee(spec),
                          architect=kit.architect(), builder=kit.builder(),
                          critic=kit.critic(), optimizer=kit.optimizer(),
                          recovery=RecoveryPolicy(), history=History())
    return runner, engine


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "serve":
        from forgemind.api.server import serve

        serve(host=args.host, port=args.port)
        return 0

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
        names = (list_problem_names() if args.problem == "all"
                 else [args.problem])
        suite = runner.run_suite(names)
        for row in suite.per_problem:
            fm, base = row["forgemind"], row["baseline"]
            print(f"[{row['problem']}] "
                  f"forgemind={'OK' if fm['success'] else 'FAIL'} "
                  f"baseline={'OK' if base['success'] else 'FAIL'} "
                  f"(fm {fm['wall_clock_seconds']:.2f}s / "
                  f"base {base['wall_clock_seconds']:.2f}s)")
        agg = suite.aggregate()
        print(f"forgemind success rate : {agg['forgemind_success_rate']:.2f}")
        print(f"baseline success rate  : {agg['baseline_success_rate']:.2f}")
        print(f"delta success          : {agg['delta_success']:+.2f}")
        # M8: deterministic totals are the primary cost comparison;
        # the wall ratio is environment-sensitive, not pass/fail.
        print(f"fm verifications       : {agg['forgemind_verifications']:.0f} / base {agg['baseline_verifications']:.0f}")
        print(f"fm agent calls         : {agg['forgemind_agent_calls']:.0f} / base {agg['baseline_agent_calls']:.0f}")
        print(f"fm tokens (est)        : {agg['forgemind_total_tokens']:.0f} / base {agg['baseline_total_tokens']:.0f}")
        print(f"observed wall ratio    : {agg['cost_ratio']:.2f}x (environment-sensitive)")
        if agg["cost_ratio_noisy"]:
            print("note: sub-second runs — wall ratio is noise-dominated, not a pass/fail criterion")
        out = runner.save_suite_report(suite, args.output)
        print(f"report written to      : {out}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
