"""Token estimation: a chars//4 heuristic, canonicalized (M8).

ForgeMind is stdlib-only by design, so there is no real tokenizer.
This is an ESTIMATE for cost comparison, not a billing count. One
canonical rule used by the engine and the baseline alike, so the two
arms' token_estimate fields are comparable. A future LLM-backed agent
that reports real usage should replace the estimate with measured
input/output tokens rather than extending this heuristic.
"""

from __future__ import annotations

from typing import Any


def estimate_tokens(*rationales: str) -> int:
    """Rough token estimate for agent rationale text: chars // 4 each."""
    return sum(len(r) // 4 for r in rationales)


def estimate_tokens_for_proposals(proposals: Any) -> int:
    """Sum estimate_tokens over proposals' rationale attrs ('' if absent)."""
    return estimate_tokens(*(getattr(p, "rationale", "") for p in proposals))
