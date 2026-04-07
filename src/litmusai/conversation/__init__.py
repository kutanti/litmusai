"""Multi-turn conversation evaluation.

Provides :class:`Step`, :class:`MultiTurnCase`, and
:class:`ConversationRunner` for testing multi-step agent workflows.

Detects error compounding — distinguishes cascade failures
(caused by earlier errors) from independent failures.

Example::

    case = MultiTurnCase(
        id="refund",
        name="Refund flow",
        steps=[
            Step(user="I want to return my shoes",
                 assertions=[Contains(["return", "refund"], mode="any")]),
            Step(user="Order #12345",
                 assertions=[Contains(["order"])]),
        ],
    )
    result = await ConversationRunner(agent).run(case)
    print(result.summary())
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from litmusai.assertions import Assertion
    from litmusai.core.agent import Agent


@dataclass
class Step:
    """A single step in a multi-turn conversation.

    Attributes:
        user: The user message to send.
        assertions: Assertions to check on the agent's response.
        name: Optional step label for display.
        max_tokens: Optional per-step token limit hint.
    """

    user: str
    assertions: list[Assertion] = field(default_factory=list)
    name: str = ""
    max_tokens: int | None = None


@dataclass
class StepResult:
    """Result of a single conversation step.

    Attributes:
        step_index: Zero-based position in the conversation.
        user: The user message sent.
        response: The agent's response text.
        passed: Whether all assertions passed.
        score: Aggregate score (0.0–1.0).
        reason: Summary reason.
        latency_ms: Response time in milliseconds.
        cost: Cost of this step.
        is_cascade: True if failure was caused by earlier error.
        assertion_details: Per-assertion results.
    """

    step_index: int
    user: str
    response: str
    passed: bool
    score: float
    reason: str = ""
    latency_ms: float = 0.0
    cost: float = 0.0
    is_cascade: bool = False
    assertion_details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        d: dict[str, Any] = {
            "step": self.step_index,
            "user": self.user,
            "response": self.response[:500],
            "passed": self.passed,
            "score": self.score,
            "reason": self.reason,
            "latency_ms": round(self.latency_ms, 1),
            "cost": self.cost,
        }
        if self.is_cascade:
            d["is_cascade"] = True
        if self.assertion_details:
            d["assertion_details"] = self.assertion_details
        return d


@dataclass
class MultiTurnCase:
    """A multi-turn conversation test case.

    Attributes:
        id: Unique identifier.
        name: Display name.
        steps: Ordered list of conversation steps.
        system_prompt: Optional system prompt for the conversation.
        tags: Tags for filtering.
        metadata: Arbitrary metadata.
    """

    id: str
    name: str
    steps: list[Step] = field(default_factory=list)
    system_prompt: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationResult:
    """Result of running a multi-turn conversation test.

    Attributes:
        case: The test case that was run.
        steps: Results for each step.
        total_steps: Number of steps.
        passed_steps: Number of passing steps.
        failed_steps: Number of failing steps.
        first_failure: Index of the first failing step (None if all pass).
        cascade_failures: Steps that failed due to earlier errors.
        independent_failures: Steps that failed independently.
        context_maintained: Whether the agent maintained context throughout.
        total_cost: Sum of step costs.
        total_latency_ms: Sum of step latencies.
        passed: Whether the whole conversation passed.
    """

    case: MultiTurnCase
    steps: list[StepResult] = field(default_factory=list)
    total_steps: int = 0
    passed_steps: int = 0
    failed_steps: int = 0
    first_failure: int | None = None
    cascade_failures: int = 0
    independent_failures: int = 0
    context_maintained: bool = True
    total_cost: float = 0.0
    total_latency_ms: float = 0.0
    passed: bool = True

    @property
    def pass_rate(self) -> float:
        """Fraction of steps that passed."""
        if self.total_steps == 0:
            return 0.0
        return self.passed_steps / self.total_steps

    def summary(self) -> str:
        """One-line summary string."""
        parts = [
            f"{'✅' if self.passed else '❌'} {self.passed_steps}/{self.total_steps} steps",
        ]
        if self.cascade_failures > 0:
            parts.append(f"🔗 {self.cascade_failures} cascade")
        if self.independent_failures > 0:
            parts.append(f"💥 {self.independent_failures} independent")
        parts.append(f"💰 ${self.total_cost:.4f}")
        parts.append(f"⏱️ {self.total_latency_ms:.0f}ms")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "case_id": self.case.id,
            "case_name": self.case.name,
            "total_steps": self.total_steps,
            "passed_steps": self.passed_steps,
            "failed_steps": self.failed_steps,
            "pass_rate": round(self.pass_rate, 4),
            "first_failure": self.first_failure,
            "cascade_failures": self.cascade_failures,
            "independent_failures": self.independent_failures,
            "context_maintained": self.context_maintained,
            "total_cost": round(self.total_cost, 6),
            "total_latency_ms": round(self.total_latency_ms, 1),
            "passed": self.passed,
            "steps": [s.to_dict() for s in self.steps],
        }


class Conversation:
    """Stateful conversation context for an agent.

    Maintains message history across multiple turns.

    Usage::

        async with Conversation(agent) as conv:
            r1 = await conv.send("Hello")
            r2 = await conv.send("What did I just say?")
    """

    def __init__(
        self,
        agent: Agent,
        system_prompt: str | None = None,
    ):
        self._agent = agent
        self._history: list[dict[str, str]] = []
        if system_prompt:
            self._history.append(
                {"role": "system", "content": system_prompt},
            )

    @property
    def history(self) -> list[dict[str, str]]:
        """Current conversation history."""
        return list(self._history)

    @property
    def turn_count(self) -> int:
        """Number of user turns so far."""
        return sum(1 for m in self._history if m["role"] == "user")

    async def send(self, message: str) -> Any:
        """Send a message and get a response.

        Appends both user message and assistant response to history.

        Args:
            message: User message text.

        Returns:
            :class:`~litmusai.core.agent.AgentResponse`.
        """
        from litmusai.core.agent import AgentResponse

        self._history.append({"role": "user", "content": message})

        start = time.monotonic()
        response = await self._agent.run(
            message, history=self._history[:-1],
        )
        elapsed = (time.monotonic() - start) * 1000

        if isinstance(response, AgentResponse):
            response.latency_ms = elapsed
            self._history.append(
                {"role": "assistant", "content": response.output},
            )
        else:
            self._history.append(
                {"role": "assistant", "content": str(response)},
            )

        return response

    async def __aenter__(self) -> Conversation:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


class ConversationRunner:
    """Run multi-turn test cases against an agent.

    Args:
        agent: The agent to test.
        stop_on_failure: If True, stop the conversation after first failure.
    """

    def __init__(
        self,
        agent: Agent,
        stop_on_failure: bool = False,
    ):
        self._agent = agent
        self._stop_on_failure = stop_on_failure

    async def run(self, case: MultiTurnCase) -> ConversationResult:
        """Execute a multi-turn test case.

        Args:
            case: The test case to run.

        Returns:
            :class:`ConversationResult` with per-step analysis.
        """
        from litmusai.core.scorer import Scorer, ScoreResult

        result = ConversationResult(
            case=case,
            total_steps=len(case.steps),
        )

        first_failure_idx: int | None = None
        conv = Conversation(self._agent, system_prompt=case.system_prompt)

        async with conv:
            for i, step in enumerate(case.steps):
                response = await conv.send(step.user)

                # Score this step
                if step.assertions:
                    scorer = Scorer()
                    from litmusai.core.suite import TestCase

                    temp_case = TestCase(
                        id=f"{case.id}_step_{i}",
                        name=step.name or f"Step {i + 1}",
                        task=step.user,
                        assertions=step.assertions,
                    )
                    score = await scorer.ascore(temp_case, response)
                else:
                    # No assertions = auto-pass
                    score = ScoreResult(passed=True, score=1.0, reason="No assertions")

                # Determine if cascade failure
                is_cascade = False
                if not score.passed and first_failure_idx is not None:
                    is_cascade = True

                if not score.passed and first_failure_idx is None:
                    first_failure_idx = i

                step_result = StepResult(
                    step_index=i,
                    user=step.user,
                    response=response.output,
                    passed=score.passed,
                    score=score.score,
                    reason=score.reason,
                    latency_ms=response.latency_ms,
                    cost=response.cost,
                    is_cascade=is_cascade,
                    assertion_details=score.details.get("assertions", [])
                    if isinstance(score.details, dict)
                    else [],
                )

                result.steps.append(step_result)
                result.total_cost += response.cost
                result.total_latency_ms += response.latency_ms

                if score.passed:
                    result.passed_steps += 1
                else:
                    result.failed_steps += 1
                    if is_cascade:
                        result.cascade_failures += 1
                    else:
                        result.independent_failures += 1

                if not score.passed and self._stop_on_failure:
                    # Mark remaining steps as cascade failures
                    for j in range(i + 1, len(case.steps)):
                        remaining = StepResult(
                            step_index=j,
                            user=case.steps[j].user,
                            response="",
                            passed=False,
                            score=0.0,
                            reason="Skipped — earlier step failed",
                            is_cascade=True,
                        )
                        result.steps.append(remaining)
                        result.failed_steps += 1
                        result.cascade_failures += 1
                    break

        result.first_failure = first_failure_idx
        result.passed = result.failed_steps == 0

        # Check context maintenance — did the agent reference earlier turns?
        result.context_maintained = _check_context_maintenance(
            result.steps, case.steps,
        )

        return result

    async def run_suite(
        self,
        cases: list[MultiTurnCase],
        concurrency: int = 1,
    ) -> list[ConversationResult]:
        """Run multiple multi-turn test cases.

        Args:
            cases: List of test cases.
            concurrency: Max parallel conversations (default 1 for stateful).

        Returns:
            List of :class:`ConversationResult`.
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def _run_one(case: MultiTurnCase) -> ConversationResult:
            async with semaphore:
                return await self.run(case)

        return await asyncio.gather(*[_run_one(c) for c in cases])


def _check_context_maintenance(
    step_results: list[StepResult],
    steps: list[Step],
) -> bool:
    """Heuristic check for context maintenance.

    Returns False if the agent appears to have lost context
    (e.g., asks "what?" or "I don't understand" after context
    was established).
    """
    confusion_markers = [
        "i don't understand",
        "could you clarify",
        "what do you mean",
        "i'm not sure what",
        "can you repeat",
        "what are you referring to",
    ]

    for i, sr in enumerate(step_results):
        if i == 0:
            continue  # First turn can't lose context
        response_lower = sr.response.lower()
        for marker in confusion_markers:
            if marker in response_lower:
                return False

    return True


def load_multi_turn_suite(
    path: str | Path,
) -> list[MultiTurnCase]:
    """Load multi-turn test cases from a YAML file.

    Format::

        type: multi_turn
        cases:
          - id: refund_flow
            name: Refund conversation
            steps:
              - user: "I want to return my shoes"
                assertions:
                  - type: contains
                    patterns: ["return", "refund"]
              - user: "Order #12345"
                assertions:
                  - type: contains
                    patterns: ["order"]

    Args:
        path: Path to YAML file.

    Returns:
        List of :class:`MultiTurnCase`.
    """
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)

    cases: list[MultiTurnCase] = []

    for case_data in data.get("cases", []):
        steps: list[Step] = []
        for step_data in case_data.get("steps", []):
            raw_assertions = step_data.get("assertions", [])
            assertions: list[Any] = []
            if raw_assertions:
                from litmusai.core.suite import _parse_yaml_assertions
                assertions = _parse_yaml_assertions(raw_assertions)

            steps.append(Step(
                user=step_data["user"],
                assertions=assertions,
                name=step_data.get("name", ""),
            ))

        cases.append(MultiTurnCase(
            id=case_data["id"],
            name=case_data.get("name", case_data["id"]),
            steps=steps,
            system_prompt=case_data.get("system_prompt"),
            tags=case_data.get("tags", []),
        ))

    return cases
