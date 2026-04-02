"""Evaluation runner — execute agents against test suites.

Connects all LitmusAI modules into a unified scoring pipeline:

    Task → Agent → Response → [Assertions] → Score → Result

Example:
    >>> results = await evaluate(
    ...     agent=my_agent,
    ...     suite=suite,
    ...     log_dir="./results/",
    ... )
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from litmusai.core.agent import Agent, AgentResponse
from litmusai.core.scorer import Scorer, ScoreResult
from litmusai.core.suite import TestCase, TestSuite

console = Console()

# Safe filename characters
_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_filename(name: str) -> str:
    """Sanitize a string for use as a filename."""
    return _SAFE_RE.sub("_", name)[:100]


@dataclass
class TestResult:
    """Result of a single test case execution."""

    case: TestCase
    response: AgentResponse
    score: ScoreResult
    passed: bool
    latency_ms: float
    cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary for JSON logging."""
        return {
            "case_id": self.case.id,
            "case_name": self.case.name,
            "task": self.case.task,
            "response": self.response.output[:2000],
            "passed": self.passed,
            "score": self.score.score,
            "score_reason": self.score.reason,
            "score_details": self.score.details,
            "latency_ms": round(self.latency_ms, 1),
            "cost": self.cost,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model": self.response.model,
            "success": self.response.success,
            "error": self.response.error,
        }


@dataclass
class EvalResults:
    """Aggregated results from an evaluation run."""

    agent_name: str
    suite_name: str
    results: list[TestResult] = field(default_factory=list)
    total_cost: float = 0.0
    total_time_ms: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    timestamp: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def avg_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency_ms for r in self.results) / len(self.results)

    @property
    def avg_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score.score for r in self.results) / len(self.results)

    def summary(self) -> str:
        parts = [
            f"✅ {self.passed}/{len(self.results)} passed",
            f"❌ {self.failed} failed",
            f"💰 ${self.total_cost:.4f}",
            f"⚡ {self.avg_latency_ms:.0f}ms avg",
        ]
        if self.total_input_tokens > 0:
            total_tok = self.total_input_tokens + self.total_output_tokens
            parts.append(f"🔤 {total_tok} tokens")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary for JSON logging."""
        return {
            "agent_name": self.agent_name,
            "suite_name": self.suite_name,
            "timestamp": self.timestamp,
            "config": self.config,
            "summary": {
                "total": len(self.results),
                "passed": self.passed,
                "failed": self.failed,
                "pass_rate": round(self.pass_rate, 4),
                "avg_score": round(self.avg_score, 4),
                "avg_latency_ms": round(self.avg_latency_ms, 1),
                "total_cost": round(self.total_cost, 6),
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
            },
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, path: str | Path) -> Path:
        """Save results to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path

    def __repr__(self) -> str:
        return f"EvalResults({self.summary()})"


async def evaluate(
    agent: Agent,
    suite: TestSuite,
    scorer: Scorer | None = None,
    concurrency: int = 5,
    verbose: bool = True,
    *,
    log_dir: str | Path | None = None,
) -> EvalResults:
    """Run an agent against a test suite and return results.

    Args:
        agent: The agent to evaluate.
        suite: Test suite with cases to run.
        scorer: Custom scorer (default: auto-detect assertions).
        concurrency: Max parallel evaluations.
        verbose: Show progress bar.
        log_dir: Directory to save full result logs (JSON).

    Returns:
        :class:`EvalResults` with per-case scores and aggregates.
    """
    scorer = scorer or Scorer()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    config: dict[str, Any] = {
        "concurrency": concurrency,
    }

    results = EvalResults(
        agent_name=agent.name,
        suite_name=suite.name,
        timestamp=timestamp,
        config=config,
    )

    semaphore = asyncio.Semaphore(concurrency)

    async def run_case(case: TestCase) -> TestResult:
        async with semaphore:
            response = await agent.run(case.task)
            # Use async scoring to avoid blocking the event loop
            score = await scorer.ascore(case, response)

            return TestResult(
                case=case,
                response=response,
                score=score,
                passed=score.passed,
                latency_ms=response.latency_ms,
                cost=response.cost,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )

    def _accumulate(result: TestResult) -> None:
        results.results.append(result)
        results.total_cost += result.cost
        results.total_time_ms += result.latency_ms
        # Use split tokens when available, fall back to tokens_used
        inp = result.input_tokens
        out = result.output_tokens
        if inp == 0 and out == 0 and result.response.tokens_used > 0:
            # Provider only gave total — attribute to input
            inp = result.response.tokens_used
        results.total_input_tokens += inp
        results.total_output_tokens += out

    if verbose:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            prog_task = progress.add_task(
                f"Evaluating {agent.name} on {suite.name}...",
                total=len(suite),
            )
            # Queue-based pattern: feed cases through a queue
            # so only `concurrency` tasks are scheduled at a time.
            queue: asyncio.Queue[TestCase] = asyncio.Queue()
            for case in suite.cases:
                queue.put_nowait(case)

            async def worker() -> None:
                while not queue.empty():
                    try:
                        case = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    result = await run_case(case)
                    _accumulate(result)
                    progress.advance(prog_task)

            workers = [
                asyncio.ensure_future(worker())
                for _ in range(min(concurrency, len(suite)))
            ]
            await asyncio.gather(*workers)
    else:
        tasks = [run_case(case) for case in suite.cases]
        test_results = await asyncio.gather(*tasks)
        for result in test_results:
            _accumulate(result)

    if verbose:
        console.print(f"\n{results.summary()}\n")

    # ── Save logs ────────────────────────────────────────────
    if log_dir:
        log_path = Path(log_dir)
        safe_agent = _safe_filename(agent.name)
        safe_suite = _safe_filename(suite.name)
        safe_time = _safe_filename(timestamp)
        filename = f"{safe_agent}_{safe_suite}_{safe_time}.json"
        saved = results.save(log_path / filename)
        if verbose:
            console.print(f"📝 Results saved to {saved}")

    return results


async def compare(
    agents: dict[str, Agent],
    suite: TestSuite | str,
    scorer: Scorer | None = None,
    verbose: bool = True,
    *,
    log_dir: str | Path | None = None,
) -> dict[str, EvalResults]:
    """Compare multiple agents on the same test suite."""
    if isinstance(suite, str):
        suite = TestSuite.load(suite)

    all_results = {}
    for name, agent in agents.items():
        if verbose:
            console.print(f"\n[bold]Running: {name}[/bold]")
        all_results[name] = await evaluate(
            agent, suite, scorer, verbose=verbose, log_dir=log_dir,
        )

    if verbose:
        console.print("\n[bold]📊 Comparison Results:[/bold]")
        from rich.table import Table

        table = Table()
        table.add_column("Agent", style="bold")
        table.add_column("Pass Rate", justify="center")
        table.add_column("Avg Score", justify="center")
        table.add_column("Cost", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Avg Latency", justify="right")

        for name, result in all_results.items():
            total_tok = (
                result.total_input_tokens + result.total_output_tokens
            )
            table.add_row(
                name,
                f"{result.pass_rate:.0%}",
                f"{result.avg_score:.2f}",
                f"${result.total_cost:.4f}",
                str(total_tok) if total_tok > 0 else "-",
                f"{result.avg_latency_ms:.0f}ms",
            )

        console.print(table)

    return all_results
