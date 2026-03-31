"""Reporter — format and export evaluation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table


console = Console()


class Reporter:
    """Generate reports from evaluation results."""

    @staticmethod
    def to_table(results: Any) -> None:
        """Print a rich table of results to the console."""
        table = Table(title=f"🧪 LitmusAI Results — {results.suite_name}")
        table.add_column("#", style="dim", width=4)
        table.add_column("Test", style="bold")
        table.add_column("Status", justify="center")
        table.add_column("Latency", justify="right")
        table.add_column("Cost", justify="right")
        table.add_column("Reason")

        for i, r in enumerate(results.results, 1):
            status = "✅" if r.passed else "❌"
            table.add_row(
                str(i),
                r.case.name,
                status,
                f"{r.latency_ms:.0f}ms",
                f"${r.cost:.4f}",
                r.score.reason,
            )

        console.print(table)
        console.print(f"\n{results.summary()}")

    @staticmethod
    def to_json(results: Any, path: str | Path | None = None) -> str:
        """Export results as JSON."""
        data = {
            "agent": results.agent_name,
            "suite": results.suite_name,
            "timestamp": results.timestamp,
            "summary": {
                "total": len(results.results),
                "passed": results.passed,
                "failed": results.failed,
                "pass_rate": results.pass_rate,
                "total_cost": results.total_cost,
                "avg_latency_ms": results.avg_latency_ms,
            },
            "results": [
                {
                    "test": r.case.name,
                    "task": r.case.task,
                    "passed": r.passed,
                    "score": r.score.score,
                    "reason": r.score.reason,
                    "latency_ms": r.latency_ms,
                    "cost": r.cost,
                    "output": r.response.output[:500],
                }
                for r in results.results
            ],
        }

        json_str = json.dumps(data, indent=2)

        if path:
            Path(path).write_text(json_str)

        return json_str

    @staticmethod
    def to_markdown(results: Any, path: str | Path | None = None) -> str:
        """Export results as Markdown."""
        lines = [
            f"# 🧪 LitmusAI Report — {results.suite_name}",
            f"\n**Agent:** {results.agent_name}",
            f"**Date:** {results.timestamp}",
            f"\n## Summary",
            f"- ✅ Passed: {results.passed}/{len(results.results)}",
            f"- ❌ Failed: {results.failed}",
            f"- 📊 Pass Rate: {results.pass_rate:.0%}",
            f"- 💰 Total Cost: ${results.total_cost:.4f}",
            f"- ⚡ Avg Latency: {results.avg_latency_ms:.0f}ms",
            f"\n## Results\n",
            "| # | Test | Status | Latency | Cost |",
            "|---|------|--------|---------|------|",
        ]

        for i, r in enumerate(results.results, 1):
            status = "✅" if r.passed else "❌"
            lines.append(
                f"| {i} | {r.case.name} | {status} | {r.latency_ms:.0f}ms | ${r.cost:.4f} |"
            )

        md = "\n".join(lines)
        if path:
            Path(path).write_text(md)
        return md
