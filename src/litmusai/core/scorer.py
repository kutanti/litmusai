"""Scoring engine — evaluate agent outputs against expected results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from litmusai.core.agent import AgentResponse


@dataclass
class ScoreResult:
    """Result of scoring an agent's output."""
    passed: bool
    score: float  # 0.0 to 1.0
    reason: str = ""
    details: dict[str, Any] | None = None


class Scorer:
    """Score agent outputs against expected results."""

    def score(self, case: Any, response: AgentResponse) -> ScoreResult:
        """Score an agent response against a test case."""
        if not response.success:
            return ScoreResult(
                passed=False,
                score=0.0,
                reason=f"Agent error: {response.error}",
            )

        checks = []

        # Exact match
        if case.expected:
            match = case.expected.strip().lower() in response.output.strip().lower()
            checks.append(("exact_match", match))

        # Contains check
        for term in case.expected_contains:
            found = term.lower() in response.output.lower()
            checks.append((f"contains '{term}'", found))

        # Not contains check
        for term in case.expected_not_contains:
            not_found = term.lower() not in response.output.lower()
            checks.append((f"not_contains '{term}'", not_found))

        if not checks:
            # No assertions — pass if agent returned non-empty output
            passed = bool(response.output.strip())
            return ScoreResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                reason="Non-empty output" if passed else "Empty output",
            )

        passed_checks = sum(1 for _, ok in checks if ok)
        total_checks = len(checks)
        score = passed_checks / total_checks if total_checks > 0 else 0.0
        all_passed = all(ok for _, ok in checks)

        failed = [name for name, ok in checks if not ok]
        reason = "All checks passed" if all_passed else f"Failed: {', '.join(failed)}"

        return ScoreResult(
            passed=all_passed,
            score=score,
            reason=reason,
            details={"checks": checks},
        )
