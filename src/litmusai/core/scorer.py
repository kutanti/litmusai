"""Scoring engine — evaluate agent outputs against expected results.

Supports two modes:
    1. **Legacy**: ``expected_contains`` / ``expected_not_contains``
       on :class:`TestCase` — simple substring matching.
    2. **Assertions**: ``assertions`` list on :class:`TestCase` —
       multi-strategy scoring via the assertion engine.

When ``assertions`` is non-empty, it takes precedence over legacy
fields.  Use :meth:`Scorer.ascore` from async contexts to avoid
blocking the event loop with async assertions.
"""

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
    """Score agent outputs against expected results.

    Automatically uses the assertion engine when a test case has
    ``assertions`` defined, falling back to legacy substring
    matching otherwise.
    """

    def score(self, case: Any, response: AgentResponse) -> ScoreResult:
        """Score an agent response (sync).

        For async assertions use :meth:`ascore` instead to avoid
        blocking the event loop.
        """
        if not response.success:
            return ScoreResult(
                passed=False, score=0.0,
                reason=f"Agent error: {response.error}",
            )

        assertions = getattr(case, "assertions", [])
        if assertions:
            return self._score_assertions_sync(
                assertions, response, case,
            )

        return self._score_legacy(case, response)

    async def ascore(
        self, case: Any, response: AgentResponse,
    ) -> ScoreResult:
        """Score an agent response (async).

        Awaits :class:`AsyncAssertion` children natively so the
        event loop is never blocked.  Sync assertions are called
        normally.
        """
        if not response.success:
            return ScoreResult(
                passed=False, score=0.0,
                reason=f"Agent error: {response.error}",
            )

        assertions = getattr(case, "assertions", [])
        if not assertions:
            return self._score_legacy(case, response)

        from litmusai.assertions import AssertionResult, AsyncAssertion

        context = self._make_context(case)
        results: list[AssertionResult] = []

        for assertion in assertions:
            try:
                if isinstance(assertion, AsyncAssertion):
                    result = await assertion.acheck(
                        response.output, context=context,
                    )
                else:
                    result = assertion.check(
                        response.output, context=context,
                    )
            except Exception as e:
                result = AssertionResult(
                    passed=False, score=0.0,
                    reason=f"Assertion error: {e}",
                    assertion_type=type(assertion).__name__,
                )
            results.append(result)

        return self._compile(results)

    # ── internals ────────────────────────────────────────────

    def _score_assertions_sync(
        self,
        assertions: list[Any],
        response: AgentResponse,
        case: Any,
    ) -> ScoreResult:
        """Score using assertions (sync — uses ThreadPool fallback)."""
        from litmusai.assertions import AssertionResult

        context = self._make_context(case)
        results: list[AssertionResult] = []

        for assertion in assertions:
            try:
                result = assertion.check(
                    response.output, context=context,
                )
            except Exception as e:
                result = AssertionResult(
                    passed=False, score=0.0,
                    reason=f"Assertion error: {e}",
                    assertion_type=type(assertion).__name__,
                )
            results.append(result)

        return self._compile(results)

    @staticmethod
    def _make_context(case: Any) -> dict[str, Any]:
        return {
            "task": getattr(case, "task", ""),
            "expected": getattr(case, "expected", ""),
            "case_id": getattr(case, "id", ""),
        }

    @staticmethod
    def _compile(results: list[Any]) -> ScoreResult:
        """Convert AssertionResult list → ScoreResult."""
        all_passed = all(r.passed for r in results)
        avg_score = sum(r.score for r in results) / len(results)

        failed = [r for r in results if not r.passed]
        if all_passed:
            reason = f"All {len(results)} assertions passed"
        else:
            reasons = [r.reason for r in failed[:3]]
            reason = (
                f"{len(failed)}/{len(results)} failed: "
                f"{'; '.join(reasons)}"
            )

        return ScoreResult(
            passed=all_passed,
            score=avg_score,
            reason=reason,
            details={
                "assertions": [
                    {
                        "type": r.assertion_type,
                        "passed": r.passed,
                        "score": r.score,
                        "reason": r.reason,
                    }
                    for r in results
                ],
            },
        )

    def _score_legacy(
        self, case: Any, response: AgentResponse,
    ) -> ScoreResult:
        """Legacy scoring — substring matching."""
        checks: list[tuple[str, bool]] = []

        if case.expected:
            match = (
                case.expected.strip().lower()
                in response.output.strip().lower()
            )
            checks.append(("exact_match", match))

        for term in case.expected_contains:
            found = term.lower() in response.output.lower()
            checks.append((f"contains '{term}'", found))

        for term in case.expected_not_contains:
            not_found = term.lower() not in response.output.lower()
            checks.append((f"not_contains '{term}'", not_found))

        if not checks:
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
        reason = (
            "All checks passed"
            if all_passed
            else f"Failed: {', '.join(failed)}"
        )

        return ScoreResult(
            passed=all_passed,
            score=score,
            reason=reason,
            details={"checks": checks},
        )
