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


# ─── Multi-Run Statistical Results ──────────────────────────────


@dataclass
class CaseStats:
    """Statistical summary for a single test case across N runs."""

    case_id: str
    case_name: str
    task: str
    n_runs: int = 0
    n_passed: int = 0
    scores: list[float] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.n_passed / self.n_runs if self.n_runs > 0 else 0.0

    @property
    def mean_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def std_score(self) -> float:
        if len(self.scores) < 2:
            return 0.0
        mean = self.mean_score
        variance = sum((s - mean) ** 2 for s in self.scores) / (
            len(self.scores) - 1
        )
        return float(variance ** 0.5)

    @property
    def mean_latency(self) -> float:
        return (
            sum(self.latencies) / len(self.latencies)
            if self.latencies else 0.0
        )

    @property
    def std_latency(self) -> float:
        if len(self.latencies) < 2:
            return 0.0
        mean = self.mean_latency
        variance = sum(
            (lat - mean) ** 2 for lat in self.latencies
        ) / (
            len(self.latencies) - 1
        )
        return float(variance ** 0.5)

    @property
    def reliability(self) -> str:
        """How reliable is this test? Based on pass consistency."""
        if self.n_runs == 0:
            return "unknown"
        rate = self.pass_rate
        if rate == 1.0:
            return "stable-pass"
        if rate == 0.0:
            return "stable-fail"
        if rate >= 0.8:
            return "mostly-pass"
        if rate <= 0.2:
            return "mostly-fail"
        return "flaky"


@dataclass
class MultiRunResults:
    """Statistical summary across multiple evaluation runs."""

    agent_name: str
    suite_name: str
    n_runs: int
    case_stats: dict[str, CaseStats] = field(default_factory=dict)
    run_results: list[EvalResults] = field(default_factory=list)
    timestamp: str = ""

    @property
    def mean_pass_rate(self) -> float:
        if not self.run_results:
            return 0.0
        return (
            sum(r.pass_rate for r in self.run_results)
            / len(self.run_results)
        )

    @property
    def std_pass_rate(self) -> float:
        if len(self.run_results) < 2:
            return 0.0
        mean = self.mean_pass_rate
        variance = sum(
            (r.pass_rate - mean) ** 2 for r in self.run_results
        ) / (len(self.run_results) - 1)
        return float(variance ** 0.5)

    @property
    def total_cost(self) -> float:
        return sum(r.total_cost for r in self.run_results)

    @property
    def flaky_tests(self) -> list[CaseStats]:
        return [
            s for s in self.case_stats.values()
            if s.reliability == "flaky"
        ]

    def summary(self) -> str:
        parts = [
            f"📊 {self.n_runs} runs",
            f"Pass rate: {self.mean_pass_rate:.0%} ±{self.std_pass_rate:.1%}",
            f"💰 ${self.total_cost:.4f} total",
        ]
        if self.flaky_tests:
            parts.append(f"⚠️ {len(self.flaky_tests)} flaky")
        return " | ".join(parts)

    def to_table(self) -> str:
        """Per-case statistical table."""
        lines: list[str] = []
        header = (
            f"{'Test':<25} {'Pass Rate':>10} "
            f"{'Mean Score':>12} {'Std Dev':>9} "
            f"{'Mean Lat':>10} {'Reliability':>12}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        for stats in sorted(
            self.case_stats.values(), key=lambda s: s.case_id,
        ):
            name = stats.case_name[:25]
            prate = f"{stats.n_passed}/{stats.n_runs}"
            mscore = f"{stats.mean_score:.3f}"
            std = f"±{stats.std_score:.3f}"
            mlat = f"{stats.mean_latency:.0f}ms"
            rel = stats.reliability
            lines.append(
                f"{name:<25} {prate:>10} "
                f"{mscore:>12} {std:>9} "
                f"{mlat:>10} {rel:>12}"
            )

        lines.append("-" * len(header))
        lines.append(self.summary())
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "suite_name": self.suite_name,
            "n_runs": self.n_runs,
            "timestamp": self.timestamp,
            "summary": {
                "mean_pass_rate": round(self.mean_pass_rate, 4),
                "std_pass_rate": round(self.std_pass_rate, 4),
                "total_cost": round(self.total_cost, 6),
                "flaky_count": len(self.flaky_tests),
            },
            "case_stats": {
                cid: {
                    "case_name": s.case_name,
                    "n_runs": s.n_runs,
                    "n_passed": s.n_passed,
                    "pass_rate": round(s.pass_rate, 4),
                    "mean_score": round(s.mean_score, 4),
                    "std_score": round(s.std_score, 4),
                    "mean_latency_ms": round(s.mean_latency, 1),
                    "std_latency_ms": round(s.std_latency, 1),
                    "reliability": s.reliability,
                }
                for cid, s in self.case_stats.items()
            },
        }

    def __repr__(self) -> str:
        return f"MultiRunResults({self.summary()})"


async def multi_evaluate(
    agent: Agent,
    suite: TestSuite,
    runs: int = 3,
    scorer: Scorer | None = None,
    concurrency: int = 5,
    verbose: bool = True,
) -> MultiRunResults:
    """Run evaluation N times and return statistical summary.

    Args:
        agent: The agent to evaluate.
        suite: Test suite to run.
        runs: Number of times to run each test (default 3).
        scorer: Custom scorer.
        concurrency: Max parallel evaluations per run.
        verbose: Show progress.

    Returns:
        :class:`MultiRunResults` with per-case statistics.

    Raises:
        ValueError: If ``runs`` is less than 1.
    """
    if runs < 1:
        msg = f"runs must be >= 1, got {runs}"
        raise ValueError(msg)

    all_runs: list[EvalResults] = []

    for i in range(runs):
        if verbose:
            console.print(
                f"\n[bold]Run {i + 1}/{runs}[/bold]"
            )
        result = await evaluate(
            agent, suite, scorer=scorer,
            concurrency=concurrency, verbose=verbose,
        )
        all_runs.append(result)

    # Build per-case statistics
    case_stats: dict[str, CaseStats] = {}
    for run in all_runs:
        for tr in run.results:
            cid = tr.case.id
            if cid not in case_stats:
                case_stats[cid] = CaseStats(
                    case_id=cid,
                    case_name=tr.case.name,
                    task=tr.case.task,
                )
            stats = case_stats[cid]
            stats.n_runs += 1
            if tr.passed:
                stats.n_passed += 1
            stats.scores.append(tr.score.score)
            stats.latencies.append(tr.latency_ms)
            stats.costs.append(tr.cost)

    multi = MultiRunResults(
        agent_name=agent.name,
        suite_name=suite.name,
        n_runs=runs,
        case_stats=case_stats,
        run_results=all_runs,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    if verbose:
        console.print(f"\n{multi.to_table()}\n")

    return multi


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
