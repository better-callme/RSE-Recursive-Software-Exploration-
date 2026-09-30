"""Analysis: recompute all statistics from raw results.jsonl.

The raw JSONL is the source of truth; this module must never be given data
that didn't come from it.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((center - spread) / denom, (center + spread) / denom)


def paired_bootstrap(records: list[dict], n_boot: int = 10000,
                     seed: int = 12345) -> dict:
    """Bootstrap CI for Delta = p_F - p_B over paired (task,trial) units."""
    import random

    pairs: dict[tuple[str, int], dict[str, int]] = defaultdict(dict)
    for r in records:
        pairs[(r["task_id"], r["trial_id"])][r["system"]] = 1 if r["success"] else 0
    units = [(k, v.get("baseline", 0), v.get("forgemind", 0))
             for k, v in sorted(pairs.items()) if "baseline" in v and "forgemind" in v]
    if not units:
        return {"error": "no paired observations"}
    deltas = []
    rng = random.Random(seed)
    for _ in range(n_boot):
        sample = [units[rng.randrange(len(units))] for _ in range(len(units))]
        pb = sum(u[1] for u in sample) / len(sample)
        pf = sum(u[2] for u in sample) / len(sample)
        deltas.append(pf - pb)
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[int(0.975 * len(deltas)) - 1]
    return {
        "n_pairs": len(units),
        "delta_point": sum(u[2] for u in units) / len(units)
                       - sum(u[1] for u in units) / len(units),
        "ci95_lo": lo,
        "ci95_hi": hi,
        "method": f"paired bootstrap over task/trial units, {n_boot} resamples",
    }


def mcnemar_exact(records: list[dict]) -> dict:
    """Exact binomial McNemar test on discordant pairs."""
    from math import comb

    table: dict[tuple[str, int], dict[str, int]] = defaultdict(dict)
    for r in records:
        table[(r["task_id"], r["trial_id"])][r["system"]] = 1 if r["success"] else 0
    n01 = n10 = 0
    for v in table.values():
        if "baseline" not in v or "forgemind" not in v:
            continue
        if v["baseline"] == 0 and v["forgemind"] == 1:
            n01 += 1
        elif v["baseline"] == 1 and v["forgemind"] == 0:
            n10 += 1
    total = n01 + n10
    if total == 0:
        return {"n01": n01, "n10": n10, "discordant_advantage": 0,
                "p_value": 1.0, "test": "McNemar exact"}
    # two-sided exact binomial test on min(n01,n10) vs B(total, 0.5)
    k = min(n01, n10)
    p_two = sum(comb(total, i) for i in range(0, k + 1)) / 2 ** total * 2
    p_two = min(1.0, p_two)
    return {"n01": n01, "n10": n10,
            "discordant_advantage": n01 - n10,
            "p_value": p_two, "test": "McNemar exact (two-sided)"}


def summarize_system(records: list[dict], system_key: str) -> dict:
    rs = [r for r in records if r["system"] == system_key]
    n = len(rs)
    wins = sum(1 for r in rs if r["success"])
    lo, hi = wilson_interval(wins, n)
    wall = [r["wall_time_seconds"] for r in rs]
    verif = sum(r["verification_calls"] for r in rs)
    return {
        "system": system_key,
        "trials": n,
        "successes": wins,
        "success_rate": round(wins / n, 4) if n else None,
        "wilson_ci95": [round(lo, 4), round(hi, 4)],
        "wall_time_total_s": round(sum(wall), 3),
        "wall_time_mean_s": round(sum(wall) / n, 4) if n else None,
        "verification_calls_total": verif,
        "candidates_total": sum(r["candidates"] for r in rs),
        "nodes_expanded_total": sum(r["nodes_expanded"] for r in rs),
        "pruned_total": sum(r["nodes_pruned"] for r in rs),
        "duplicates_total": sum(r["duplicates"] for r in rs),
        "backtracks_total": sum(r["backtracks"] for r in rs),
        "stagnation_events_total": sum(r["stagnation_events"] for r in rs),
        "token_estimate_total": sum((r.get("token_estimate") or 0) for r in rs),
        # cost per success (undefined when no successes)
        "verifications_per_success":
            round(verif / wins, 3) if wins else None,
        "wall_time_per_success_s":
            round(sum(wall) / wins, 4) if wins else None,
    }


def by_difficulty(records: list[dict]) -> list[dict]:
    out = []
    levels = sorted({r["difficulty"] for r in records})
    for lvl in levels:
        row = {"difficulty": lvl}
        for sysname in ("baseline", "forgemind"):
            rs = [r for r in records
                  if r["difficulty"] == lvl and r["system"] == sysname]
            row[f"{sysname}_success_rate"] = (
                round(sum(r["success"] for r in rs) / len(rs), 4) if rs else None)
            row[f"{sysname}_trials"] = len(rs)
        b = row.get("baseline_success_rate")
        f = row.get("forgemind_success_rate")
        row["delta"] = (round(f - b, 4)
                        if b is not None and f is not None else None)
        out.append(row)
    return out


def failure_taxonomy(records: list[dict]) -> dict:
    cats: dict[str, dict[str, int]] = {}
    for r in records:
        bucket = cats.setdefault(r["system"], defaultdict(int))
        for c in r["failure_categories"]:
            bucket[c] += 1
    return {s: dict(sorted(c.items(), key=lambda kv: -kv[1]))
            for s, c in cats.items()}


def recovery_analysis(all_records: list[dict]) -> dict:
    """For each (task,trial): baseline fail -> forgemind success = recovered."""
    pairs: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for r in all_records:
        if r["system"] in ("baseline", "forgemind"):
            pairs[(r["task_id"], r["trial_id"])][r["system"]] = r
    recoverable = recovered = lost = 0
    examples = []
    for key, v in sorted(pairs.items()):
        if "baseline" not in v or "forgemind" not in v:
            continue
        b, f = v["baseline"], v["forgemind"]
        if not b["success"] and f["success"]:
            recovered += 1
            examples.append({
                "task": key[0], "trial": key[1],
                "baseline_failure_categories": b["failure_categories"],
                "forgemind_termination": f["termination_reason"],
                "extra_verifications": f["verification_calls"] - 1,
                "extra_backtracks": f["backtracks"],
                "recovered": True,
            })
        elif b["success"] and not f["success"]:
            lost += 1
            examples.append({
                "task": key[0], "trial": key[1],
                "note": "BASELINE SUCCEEDED WHERE FORGEMIND FAILED",
                "forgemind_failures": f["failure_categories"],
            })
    recoverable_count = recovered + lost  # baseline failures ForgeMind could flip
    return {
        "baseline_failures_recovered_by_forgemind": recovered,
        "baseline_successes_lost_by_forgemind": lost,
        "recovery_examples_or_counterexamples": examples[:20],
        "net_recovery_advantage": recovered - lost,
    }


def search_efficiency(records: list[dict]) -> dict:
    fm = [r for r in records if r["system"] == "forgemind"]
    candidates = sum(r["candidates"] for r in fm)
    expanded = sum(r["nodes_expanded"] for r in fm)
    generated = candidates
    pruned = sum(r["nodes_pruned"] for r in fm)
    dups = sum(r["duplicates"] for r in fm)
    wins = sum(r["success"] for r in fm)
    verifs = sum(r["verification_calls"] for r in fm)
    return {
        "branching_factor_observed": round(candidates / expanded, 3) if expanded else None,
        "pruning_rate": round(pruned / generated, 3) if generated else None,
        "duplicate_rate": round(dups / generated, 3) if generated else None,
        "acceptance_efficiency": round(wins / candidates, 3) if candidates else None,
        "verification_efficiency": round(wins / verifs, 3) if verifs else None,
        "denominators": {"candidates": candidates, "expanded_nodes": expanded,
                         "pruned": pruned, "duplicates": dups, "wins": wins,
                         "verifications": verifs},
    }


def ablation_table(records: list[dict]) -> list[dict]:
    variants = ["abl_baseline", "abl_iterative", "abl_beam", "abl_beam_backtrack",
                "abl_full"]
    rows = []
    for v in variants:
        rs = [r for r in records if r["system"] == v]
        if not rs:
            continue
        n = len(rs)
        wins = sum(r["success"] for r in rs)
        lo, hi = wilson_interval(wins, n)
        rows.append({
            "variant": v.removeprefix("abl_"),
            "trials": n,
            "success_rate": round(wins / n, 4),
            "wilson_ci95": [round(lo, 4), round(hi, 4)],
            "mean_wall_time_s": round(sum(r["wall_time_seconds"] for r in rs) / n, 4),
            "verification_calls_total": sum(r["verification_calls"] for r in rs),
            "backtracks_total": sum(r["backtracks"] for r in rs),
            "nodes_expanded_total": sum(r["nodes_expanded"] for r in rs),
            "token_estimate_total": sum((r.get("token_estimate") or 0) for r in rs),
        })
    return rows


def incremental_analysis(rows: list[dict]) -> list[dict]:
    out = []
    for prev, cur in zip(rows, rows[1:]):
        out.append({
            "step": f"{prev['variant']} -> {cur['variant']}",
            "delta_success": round(cur["success_rate"] - prev["success_rate"], 4),
            "delta_wall_time_s": round(cur["mean_wall_time_s"] - prev["mean_wall_time_s"], 4),
            "delta_verifications": cur["verification_calls_total"] - prev["verification_calls_total"],
        })
    return out
