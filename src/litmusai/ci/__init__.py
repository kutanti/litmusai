"""CI/CD integration engine for LitmusAI.

Provides the core logic for running evaluations in CI pipelines,
comparing against baselines, detecting regressions, and posting
results as GitHub PR comments.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from litmusai.core.agent import Agent
from litmusai.core.runner import EvalResults, evaluate
from litmusai.core.scorer import Scorer
from litmusai.core.suite import TestSuite

console = Console()


# ─── Agent Loading ─────────────────────────────────────────────────


def load_agent(agent_path: str) -> Agent:
    """Load an agent from a module path.

    Formats:
        - "module:attribute"  →  import module, get attribute
        - "path/to/file.py:attribute"  →  load file directly
        - "module:attribute"  where attribute is a function → Agent.from_function()

    Args:
        agent_path: Module path in format "module:attribute" or
            "path/file.py:attribute".

    Returns:
        An Agent instance.
    """
    if ":" not in agent_path:
        raise ValueError(
            f"Agent path must be in format 'module:attribute', "
            f"got '{agent_path}'"
        )

    module_part, attr_name = agent_path.rsplit(":", 1)

    # Handle file paths — load directly without mutating sys.path
    if module_part.endswith(".py"):
        file_path = Path(module_part).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Agent file not found: {module_part}")
        module_name = file_path.stem
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for '{file_path}'")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module_name = module_part
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise ImportError(
                f"Cannot import agent module '{module_name}': {e}"
            ) from e

    if not hasattr(module, attr_name):
        raise AttributeError(
            f"Module '{module_name}' has no attribute '{attr_name}'"
        )

    obj = getattr(module, attr_name)

    if isinstance(obj, Agent):
        return obj
    if callable(obj):
        return Agent.from_function(obj, name=attr_name)
    raise TypeError(
        f"'{attr_name}' is not an Agent or callable, got {type(obj)}"
    )


# ─── Baseline Comparison ──────────────────────────────────────────


def load_baseline(path: str | Path) -> dict[str, Any] | None:
    """Load baseline results from a JSON file."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return dict(json.loads(p.read_text()))
    except (json.JSONDecodeError, TypeError):
        return None


def save_baseline(data: dict[str, Any], path: str | Path | None = None) -> Path:
    """Save evaluation results as baseline."""
    p = Path(path or ".litmus/baseline.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))
    return p


def compare_with_baseline(
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Compare current results with baseline and detect regressions.

    Returns:
        Dict with comparison details and regression flags.
    """
    curr_summary = current.get("summary", {})
    base_summary = baseline.get("summary", {})

    curr_rate = curr_summary.get("pass_rate", 0)
    base_rate = base_summary.get("pass_rate", 0)
    rate_delta = curr_rate - base_rate

    curr_cost = curr_summary.get("total_cost", 0)
    base_cost = base_summary.get("total_cost", 0)
    cost_delta = curr_cost - base_cost

    curr_latency = curr_summary.get("avg_latency_ms", 0)
    base_latency = base_summary.get("avg_latency_ms", 0)
    latency_delta = curr_latency - base_latency

    # Detect regressions
    regressions: list[str] = []
    if rate_delta < -0.05:  # >5% drop
        regressions.append(
            f"Pass rate dropped {abs(rate_delta):.1%} "
            f"({base_rate:.1%} → {curr_rate:.1%})"
        )
    if base_cost > 0 and cost_delta > base_cost * 0.5:  # >50% cost increase
        regressions.append(
            f"Cost increased {cost_delta / base_cost:.0%} "
            f"(${base_cost:.4f} → ${curr_cost:.4f})"
        )
    if base_latency > 0 and latency_delta > base_latency * 0.5:
        regressions.append(
            f"Latency increased {latency_delta / base_latency:.0%} "
            f"({base_latency:.0f}ms → {curr_latency:.0f}ms)"
        )

    return {
        "pass_rate": {"current": curr_rate, "baseline": base_rate, "delta": rate_delta},
        "cost": {"current": curr_cost, "baseline": base_cost, "delta": cost_delta},
        "latency": {
            "current": curr_latency,
            "baseline": base_latency,
            "delta": latency_delta,
        },
        "regressions": regressions,
        "has_regression": len(regressions) > 0,
    }


# ─── Result Serialization ─────────────────────────────────────────


def results_to_dict(results: EvalResults) -> dict[str, Any]:
    """Convert EvalResults to a serializable dict."""
    return {
        "agent": results.agent_name,
        "suite": results.suite_name,
        "timestamp": results.timestamp,
        "summary": {
            "total": len(results.results),
            "passed": results.passed,
            "failed": results.failed,
            "pass_rate": round(results.pass_rate, 4),
            "total_cost": round(results.total_cost, 6),
            "avg_latency_ms": round(results.avg_latency_ms, 1),
        },
        "results": [
            {
                "test": r.case.name,
                "task": r.case.task,
                "passed": r.passed,
                "score": r.score.score,
                "reason": r.score.reason,
                "latency_ms": round(r.latency_ms, 1),
                "cost": round(r.cost, 6),
                "output": r.response.output[:500],
            }
            for r in results.results
        ],
    }


# ─── Report Formatting ────────────────────────────────────────────


def _delta_str(val: float, fmt: str = ".1%", invert: bool = False) -> str:
    """Format a delta value with arrow indicator."""
    if val == 0:
        return "→ (no change)"
    # For cost/latency, positive = bad; for pass rate, positive = good
    is_good = val > 0 if not invert else val < 0
    arrow = "🟢 ↑" if is_good else "🔴 ↓"
    return f"{arrow} {abs(val):{fmt}}"


def format_report(
    data: dict[str, Any],
    baseline: dict[str, Any] | None = None,
    fmt: str = "markdown",
    threshold: float = 0.7,
) -> str:
    """Format evaluation results as a report string.

    Args:
        data: Current evaluation results dict.
        baseline: Optional baseline results for comparison.
        fmt: Output format ("markdown", "github", "json").
        threshold: Pass rate threshold for verdict (default 0.7).

    Returns:
        Formatted report string.
    """
    if fmt == "json":
        output: dict[str, Any] = {"results": data}
        if baseline:
            output["comparison"] = compare_with_baseline(data, baseline)
        return json.dumps(output, indent=2)

    summary = data.get("summary", {})
    agent = data.get("agent", "unknown")
    suite_name = data.get("suite", "unknown")
    timestamp = data.get("timestamp", "")

    lines: list[str] = []
    lines.append(f"## 🧪 LitmusAI Report — {suite_name}")
    lines.append(f"**Agent:** {agent} | **Date:** {timestamp}")
    lines.append("")

    # Summary
    total = summary.get("total", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    pass_rate = summary.get("pass_rate", 0)
    cost = summary.get("total_cost", 0)
    latency = summary.get("avg_latency_ms", 0)

    verdict = "✅ PASSED" if pass_rate >= threshold else "❌ FAILED"
    lines.append(f"### {verdict}")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Pass Rate | {pass_rate:.0%} ({passed}/{total}) |")
    lines.append(f"| Failed | {failed} |")
    lines.append(f"| Total Cost | ${cost:.4f} |")
    lines.append(f"| Avg Latency | {latency:.0f}ms |")

    # Baseline comparison
    if baseline:
        comparison = compare_with_baseline(data, baseline)
        lines.append("")
        lines.append("### 📊 vs Baseline")
        lines.append("")
        lines.append("| Metric | Current | Baseline | Delta |")
        lines.append("|--------|---------|----------|-------|")

        pr = comparison["pass_rate"]
        lines.append(
            f"| Pass Rate | {pr['current']:.0%} | {pr['baseline']:.0%} "
            f"| {_delta_str(pr['delta'])} |"
        )

        c = comparison["cost"]
        lines.append(
            f"| Cost | ${c['current']:.4f} | ${c['baseline']:.4f} "
            f"| {_delta_str(c['delta'], '.4f', invert=True)} |"
        )

        lt = comparison["latency"]
        lines.append(
            f"| Latency | {lt['current']:.0f}ms | {lt['baseline']:.0f}ms "
            f"| {_delta_str(lt['delta'], '.0f', invert=True)} |"
        )

        if comparison["regressions"]:
            lines.append("")
            lines.append("### ⚠️ Regressions Detected")
            for r in comparison["regressions"]:
                lines.append(f"- {r}")

    # Per-test results
    results_list = data.get("results", [])
    if results_list:
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>📋 Detailed Results</summary>")
        lines.append("")
        lines.append("| # | Test | Status | Latency | Cost |")
        lines.append("|---|------|--------|---------|------|")
        for i, r in enumerate(results_list, 1):
            status = "✅" if r.get("passed") else "❌"
            lines.append(
                f"| {i} | {r.get('test', '')} | {status} "
                f"| {r.get('latency_ms', 0):.0f}ms "
                f"| ${r.get('cost', 0):.4f} |"
            )
        lines.append("")
        lines.append("</details>")

    return "\n".join(lines)


def format_table(data: dict[str, Any]) -> None:
    """Print results as a rich table to console."""
    summary = data.get("summary", {})

    table = Table(title=f"🧪 LitmusAI — {data.get('suite', 'Results')}")
    table.add_column("#", style="dim", width=4)
    table.add_column("Test", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Score", justify="center")
    table.add_column("Latency", justify="right")
    table.add_column("Cost", justify="right")

    for i, r in enumerate(data.get("results", []), 1):
        status = "✅" if r.get("passed") else "❌"
        table.add_row(
            str(i),
            r.get("test", ""),
            status,
            f"{r.get('score', 0):.2f}",
            f"{r.get('latency_ms', 0):.0f}ms",
            f"${r.get('cost', 0):.4f}",
        )

    console.print(table)

    pass_rate = summary.get("pass_rate", 0)
    console.print(
        f"\n✅ {summary.get('passed', 0)}/{summary.get('total', 0)} passed "
        f"| ❌ {summary.get('failed', 0)} failed "
        f"| 💰 ${summary.get('total_cost', 0):.4f} "
        f"| ⚡ {summary.get('avg_latency_ms', 0):.0f}ms avg"
        f"| Pass rate: {pass_rate:.0%}"
    )


# ─── Main Entry Point ─────────────────────────────────────────────


async def run_evaluation(
    suite: str,
    agent_path: str,
    concurrency: int = 5,
    output_path: str | None = None,
    fmt: str = "table",
    baseline_path: str | None = None,
    do_save_baseline: bool = False,
    budget: float | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Run a full evaluation and return results.

    This is the main entry point called by the CLI.

    Returns:
        Dict with 'success' bool and result data.
    """
    success = True
    effective_threshold = threshold if threshold is not None else 0.7

    # Load agent
    try:
        agent = load_agent(agent_path)
    except (ValueError, FileNotFoundError, ImportError, AttributeError, TypeError) as e:
        console.print(f"[red]Error loading agent: {e}[/red]")
        return {"success": False, "error": str(e)}

    # Load suite — support both names and file paths
    try:
        suite_path = Path(suite)
        if suite_path.exists() and suite_path.suffix in (".yaml", ".yml"):
            test_suite = TestSuite.from_yaml(suite_path)
        else:
            test_suite = TestSuite.load(suite)
    except Exception as e:
        console.print(f"[red]Error loading suite '{suite}': {e}[/red]")
        return {"success": False, "error": str(e)}

    # Run evaluation
    console.print(
        f"🧪 Running [bold]{test_suite.name}[/bold] "
        f"with [bold]{agent.name}[/bold]..."
    )

    results = await evaluate(
        agent=agent,
        suite=test_suite,
        scorer=Scorer(),
        concurrency=concurrency,
        verbose=fmt == "table",
    )

    data = results_to_dict(results)

    # Load baseline for comparison
    baseline = None
    if baseline_path:
        baseline = load_baseline(baseline_path)
        if baseline is None:
            console.print(
                f"[yellow]Warning: baseline not found at {baseline_path}[/yellow]"
            )

    # Check threshold
    if threshold is not None and results.pass_rate < threshold:
        console.print(
            f"[red]❌ Pass rate {results.pass_rate:.0%} "
            f"below threshold {threshold:.0%}[/red]"
        )
        success = False

    # Check budget
    if budget is not None and results.total_cost > budget:
        console.print(
            f"[red]❌ Total cost ${results.total_cost:.4f} "
            f"exceeds budget ${budget:.4f}[/red]"
        )
        success = False

    # Check for regressions
    has_regression = False
    if baseline:
        comparison = compare_with_baseline(data, baseline)
        has_regression = comparison["has_regression"]
        if has_regression:
            console.print("[red]⚠️ Regressions detected:[/red]")
            for r in comparison["regressions"]:
                console.print(f"  [red]• {r}[/red]")
            success = False

    # Build full output payload (consistent for stdout and file)
    output_payload: dict[str, Any] = {"results": data}
    if baseline:
        output_payload["comparison"] = compare_with_baseline(data, baseline)
    output_payload["success"] = success
    output_payload["has_regression"] = has_regression

    # Output results
    if fmt == "table":
        format_table(data)
    elif fmt == "json":
        console.print(json.dumps(output_payload, indent=2))
    elif fmt in ("markdown", "github"):
        md = format_report(data, baseline, fmt="markdown",
                           threshold=effective_threshold)
        console.print(md)

    # Save output file
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if output_path.endswith(".json"):
            Path(output_path).write_text(json.dumps(output_payload, indent=2))
        else:
            Path(output_path).write_text(
                format_report(data, baseline, fmt="markdown",
                              threshold=effective_threshold)
            )
        console.print(f"📄 Results saved to {output_path}")

    # Save baseline
    if do_save_baseline:
        bp = save_baseline(data)
        console.print(f"📊 Baseline saved to {bp}")

    return {"success": success, "data": data, "has_regression": has_regression}
