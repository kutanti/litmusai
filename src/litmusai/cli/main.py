"""LitmusAI CLI — Test your AI agents from the command line."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="litmusai")
def cli() -> None:
    """🧪 LitmusAI — The open-source evaluation framework for AI agents."""


@cli.command()
def init() -> None:
    """Initialize a new LitmusAI project."""
    litmus_dir = Path(".litmus")
    litmus_dir.mkdir(exist_ok=True)
    (litmus_dir / "config.yaml").write_text(
        "# LitmusAI Configuration\n"
        "version: 1\n\n"
        "# Default settings\n"
        "defaults:\n"
        "  concurrency: 5\n"
        "  timeout: 60\n"
        "  verbose: true\n"
    )

    suites_dir = Path("suites")
    suites_dir.mkdir(exist_ok=True)
    (suites_dir / "example.yaml").write_text(
        "name: example\n"
        "description: Example test suite\n\n"
        "cases:\n"
        '  - id: test_001\n'
        '    name: "Simple greeting"\n'
        '    task: "Say hello"\n'
        '    expected_contains:\n'
        '      - "hello"\n'
        '  - id: test_002\n'
        '    name: "Math question"\n'
        '    task: "What is 2 + 2?"\n'
        '    expected_contains:\n'
        '      - "4"\n'
    )

    console.print("🧪 [bold green]LitmusAI project initialized![/bold green]")
    console.print("  📁 .litmus/config.yaml — configuration")
    console.print("  📁 suites/example.yaml — example test suite")
    console.print("\nNext steps:")
    console.print("  1. Define your agent in a Python file")
    console.print("  2. Run: [bold]litmus run --agent my_agent.py[/bold]")


@cli.command()
@click.option("--suite", "-s", required=True, help="Test suite name or YAML path")
@click.option("--agent", "-a", required=True, help="Agent module path (e.g. my_agent:agent)")
@click.option("--concurrency", "-c", default=5, help="Max parallel evaluations")
@click.option("--output", "-o", default=None, help="Output file for results")
@click.option(
    "--format", "fmt", default="table",
    type=click.Choice(["table", "json", "markdown", "github"]),
)
@click.option("--baseline", "-b", default=None, help="Baseline results JSON to compare against")
@click.option("--save-baseline", is_flag=True, help="Save results as new baseline")
@click.option("--budget", default=None, type=float, help="Max total cost budget ($)")
@click.option("--threshold", default=None, type=float, help="Min pass rate to succeed (0.0-1.0)")
@click.option("--model", "-m", default=None, help="Model name for cost tracking")
def run(
    suite: str,
    agent: str,
    concurrency: int,
    output: str | None,
    fmt: str,
    baseline: str | None,
    save_baseline: bool,
    budget: float | None,
    threshold: float | None,
    model: str | None,
) -> None:
    """Run evaluation against a test suite.

    Examples:

        litmus run -s research -a my_agent:agent

        litmus run -s suites/custom.yaml -a my_agent:agent --format json

        litmus run -s research -a my_agent:agent --baseline .litmus/baseline.json

        litmus run -s research -a my_agent:agent --threshold 0.8 --budget 1.0
    """
    from litmusai.ci import run_evaluation

    result = asyncio.run(run_evaluation(
        suite=suite,
        agent_path=agent,
        concurrency=concurrency,
        output_path=output,
        fmt=fmt,
        baseline_path=baseline,
        save_baseline=save_baseline,
        budget=budget,
        threshold=threshold,
        model=model,
    ))

    # Exit with non-zero if evaluation failed
    if not result.get("success", True):
        sys.exit(1)


@cli.command()
@click.argument("task")
@click.option("--suite", "-s", default="custom", help="Suite to add to")
def create_test(task: str, suite: str) -> None:
    """Create a new test case."""
    console.print(f"✅ Test added to suite [bold]{suite}[/bold]: {task}")


@cli.command()
def suites() -> None:
    """List available test suites."""
    from litmusai.core.suite import TestSuite

    available = TestSuite.available()
    if available:
        console.print("[bold]Available test suites:[/bold]")
        for s in available:
            console.print(f"  📋 {s}")
    else:
        console.print("No test suites found. Run [bold]litmus init[/bold] first.")


@cli.command()
@click.option("--results", "-r", required=True, help="Results JSON file")
@click.option("--baseline", "-b", default=None, help="Baseline JSON to compare")
def report(results: str, baseline: str | None) -> None:
    """Generate a report from saved results."""
    results_path = Path(results)
    if not results_path.exists():
        console.print(f"[red]Results file not found: {results}[/red]")
        sys.exit(1)

    data = json.loads(results_path.read_text())

    baseline_data = None
    if baseline:
        bp = Path(baseline)
        if bp.exists():
            baseline_data = json.loads(bp.read_text())

    from litmusai.ci import format_report

    output = format_report(data, baseline_data, fmt="markdown")
    console.print(output)


@cli.command()
@click.option("--port", "-p", default=3000, help="Dashboard port")
def dashboard(port: int) -> None:
    """Launch the results dashboard."""
    console.print(f"🌐 Dashboard coming soon! (port {port})")


@cli.command()
def badges() -> None:
    """Generate README badges from latest results."""
    litmus_dir = Path(".litmus")
    baseline_path = litmus_dir / "baseline.json"

    if not baseline_path.exists():
        console.print("[red]No baseline found. Run with --save-baseline first.[/red]")
        sys.exit(1)

    data = json.loads(baseline_path.read_text())
    summary = data.get("summary", {})
    pass_rate = summary.get("pass_rate", 0)
    pct = f"{pass_rate:.0%}"

    color = "brightgreen" if pass_rate >= 0.9 else "yellow" if pass_rate >= 0.7 else "red"

    badge_url = (
        f"https://img.shields.io/badge/LitmusAI-{pct}%20pass-{color}"
    )

    table = Table(title="📛 README Badges")
    table.add_column("Badge", style="bold")
    table.add_column("Markdown")
    table.add_row(
        "Pass Rate",
        f"[![LitmusAI]({badge_url})](https://github.com/kutanti/litmusai)",
    )
    console.print(table)


if __name__ == "__main__":
    cli()
