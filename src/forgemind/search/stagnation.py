"""Stagnation detection: flat guidance deltas and structural oscillation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class StagnationDetector:
    window: int = 4
    epsilon: float = 1e-3

    _guidance_history: deque[float] = field(default_factory=deque)
    _hash_history: deque[str] = field(default_factory=deque)
    stagnation_count: int = 0

    def record(self, *, guidance: float, content_hash: str) -> bool:
        """Record a transition outcome; returns True when stagnation detected."""
        self._guidance_history.append(guidance)
        self._hash_history.append(content_hash)
        if len(self._guidance_history) > self.window + 1:
            self._guidance_history.popleft()
        if len(self._hash_history) > 2 * self.window:
            self._hash_history.popleft()

        flat = (
            len(self._guidance_history) >= min(2, self.window)
            and abs(self._guidance_history[-1] - self._guidance_history[-2]) < self.epsilon
        )
        oscillating = self._detect_oscillation()

        if flat or oscillating:
            self.stagnation_count += 1
            return True
        return False

    def _detect_oscillation(self) -> bool:
        h = list(self._hash_history)
        n = len(h)
        if n < 4:
            return False
        for period in (2, 3):
            chunk = h[-period:]
            repeats = n // period
            if repeats >= 2 and all(h[i * period:(i + 1) * period] == chunk
                                    for i in range(repeats)):
                return True
        return False

    def reset_branch(self) -> None:
        self._guidance_history.clear()
