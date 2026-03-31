"""Evaluation runner — execute agents against test suites."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from litmusai.core.agent import Agent, AgentResponse
from litmusai.core.suite import TestSuite, TestCase
from litmusai.core.scorer import Scorer, ScoreResult


console = Console()


@dataclass
class TestResult:
    """Result of a single test case execution."""
    case: TestCase
    response: AgentResponse
    score: ScoreResult
    passed: bool
    latency_ms: float
    cost: float = 0.0


@dataclass
class EvalResults:
    """Aggregated results from an evaluation run."""
    agent_name: str
    suite_name: str
    results: list[TestResult] = field(default_factory=list)
    total_cost: float = 0.0
    total_time_ms: float = 0.0
    timestamp: str = ""

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

    def summary(self) -> str:
        return (
            f"✅ {self.passed}/{len(self.results)} passed "
            f"| ❌ {self.failed} failed "
            f"| 💰 ${self.total_cost:.4f} "
            f"| ⚡ {self.avg_latency_ms:.0f}ms avg"
        )

    def __repr__(self) -> str:
        return f"EvalResults({self.summary()})"


async def evaluate(
    agent: Agent,
    suite: TestSuite,
    scorer: Scorer | None = None,
    concurrency: int = 5,
    verbose: bool = True,
) -> EvalResults:
    """Run an agent against a test suite and return results."""
    scorer = scorer or Scorer()
    results = EvalResults(
        agent_name=agent.name,
        suite_name=suite.name,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    semaphore = asyncio.Semaphore(concurrency)

    async def run_case(case: TestCase) -> TestResult:
        async with semaphore:
            response = await agent.run(case.task)
            score = scorer.score(case, response)
            return TestResult(
                case=case,
                response=response,
                score=score,
                passed=score.passed,
                latency_ms=response.latency_ms,
                cost=response.cost,
            )

    if verbose:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Evaluating {agent.name} on {suite.name}...",
                total=len(suite),
            )
            for case in suite.cases:
                result = await run_case(case)
                results.results.append(result)
                results.total_cost += result.cost
                results.total_time_ms += result.latency_ms
                progress.advance(task)
    else:
        tasks = [run_case(case) for case in suite.cases]
        test_results = await asyncio.gather(*tasks)
        for result in test_results:
            results.results.append(result)
            results.total_cost += result.cost
            results.total_time_ms += result.latency_ms

    if verbose:
        console.print(f"\n{results.summary()}\n")

    return results


async def compare(
    agents: dict[str, Agent],
    suite: TestSuite | str,
    scorer: Scorer | None = None,
    verbose: bool = True,
) -> dict[str, EvalResults]:
    """Compare multiple agents on the same test suite."""
    if isinstance(suite, str):
        suite = TestSuite.load(suite)

    all_results = {}
    for name, agent in agents.items():
        if verbose:
            console.print(f"\n[bold]Running: {name}[/bold]")
        all_results[name] = await evaluate(agent, suite, scorer, verbose=verbose)

    if verbose:
        console.print("\n[bold]📊 Comparison Results:[/bold]")
        from rich.table import Table
        table = Table()
        table.add_column("Agent", style="bold")
        table.add_column("Pass Rate", justify="center")
        table.add_column("Cost", justify="right")
        table.add_column("Avg Latency", justify="right")

        for name, result in all_results.items():
            rate = f"{result.pass_rate:.0%}"
            cost = f"${result.total_cost:.4f}"
            latency = f"{result.avg_latency_ms:.0f}ms"
            table.add_row(name, rate, cost, latency)

        console.print(table)

    return all_results
