"""LitmusAI CLI — Test your AI agents from the command line."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="0.2.0", prog_name="litmusai")
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
        "  log_dir: .litmus/logs\n"
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

    console.print(
        "🧪 [bold green]LitmusAI project initialized![/bold green]"
    )
    console.print("  📁 .litmus/config.yaml — configuration")
    console.print("  📁 suites/example.yaml — example test suite")
    console.print("\nNext steps:")
    console.print("  1. Define your agent in a Python file")
    console.print(
        "  2. Run: [bold]litmus run -s example -a my_agent:agent"
        "[/bold]"
    )


@cli.command()
@click.option(
    "--suite", "-s", required=True,
    help="Test suite name or YAML path",
)
@click.option(
    "--agent", "-a", required=True,
    help="Agent module path (e.g. my_agent:agent)",
)
@click.option(
    "--concurrency", "-c", default=5,
    help="Max parallel evaluations",
)
@click.option(
    "--output", "-o", default=None,
    help="Output file for results",
)
@click.option(
    "--format", "fmt", default="table",
    type=click.Choice(["table", "json", "markdown", "github"]),
)
@click.option(
    "--baseline", "-b", default=None,
    help="Baseline results JSON to compare against",
)
@click.option(
    "--save-baseline", is_flag=True,
    help="Save results as new baseline",
)
@click.option(
    "--budget", default=None, type=float,
    help="Max total cost budget ($)",
)
@click.option(
    "--threshold", default=None, type=float,
    help="Min pass rate to succeed (0.0-1.0)",
)
@click.option(
    "--runs", "-n", default=1, type=int,
    help="Number of runs for statistical reporting (default: 1)",
)
@click.option(
    "--log-dir", default=None,
    help="Directory to save result logs (default: .litmus/logs)",
)
@click.option(
    "--profile", "-p", default=None,
    help="Evaluation profile (quick, thorough, benchmark, safety, ci)",
)
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
    runs: int,
    log_dir: str | None,
    profile: str | None,
) -> None:
    """Run evaluation against a test suite.

    Examples:

        litmus run -s research -a my_agent:agent

        litmus run -s research -a my_agent:agent --runs 5

        litmus run -s suites/custom.yaml -a my_agent:agent --format json

        litmus run -s research -a my_agent:agent --threshold 0.8
    """
    from litmusai.ci import run_evaluation
    from litmusai.config import load_config, merge_cli_args

    # Apply profile defaults first, then CLI overrides take precedence
    if profile:
        from litmusai.profiles import get_profile, load_profiles_from_dir

        # Load custom profiles from .litmus/profiles/
        load_profiles_from_dir()

        try:
            prof = get_profile(profile)
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            raise SystemExit(1) from None

        # Profile sets defaults — explicit CLI args override
        ctx = click.get_current_context()
        if ctx.get_parameter_source("concurrency") != click.core.ParameterSource.COMMANDLINE:
            concurrency = prof.concurrency
        if ctx.get_parameter_source("runs") != click.core.ParameterSource.COMMANDLINE:
            runs = prof.runs
        if threshold is None:
            threshold = prof.threshold
        if log_dir is None and prof.report == "junit":
            # CI profile implies logging
            log_dir = ".litmus/logs"

        # Note: safety, safety_depth, report, verbose from profiles
        # are used when invoking Pipeline directly. The CLI `run`
        # command uses run_evaluation() which doesn't support safety
        # inline. Use `litmus scan` separately for safety scanning,
        # or use Pipeline() in Python for the full profile experience.

        # Log profile model params for user awareness
        model_params = prof.get_model_params()
        if model_params:
            parts = [f"{k}={v}" for k, v in model_params.items()]
            click.echo(
                f"ℹ️  Profile '{profile}' recommends: "
                f"{', '.join(parts)}. "
                f"Pass these to Agent.from_openai_chat() for "
                f"reproducible results.",
                err=True,
            )
    else:
        ctx = click.get_current_context()

    # Load config and merge with CLI args.
    # Use Click's parameter source to detect user-provided values.
    config = load_config()
    merged = merge_cli_args(
        config,
        concurrency=(
            concurrency
            if ctx.get_parameter_source("concurrency")
            == click.core.ParameterSource.COMMANDLINE
            else None
        ),
        threshold=threshold,
        budget=budget,
        runs=(
            runs
            if ctx.get_parameter_source("runs")
            == click.core.ParameterSource.COMMANDLINE
            else None
        ),
        log_dir=log_dir,
    )

    result = asyncio.run(run_evaluation(
        suite=suite,
        agent_path=agent,
        concurrency=merged["concurrency"],
        output_path=output,
        fmt=fmt,
        baseline_path=baseline,
        do_save_baseline=save_baseline,
        budget=merged["budget"],
        threshold=merged["threshold"],
        runs=merged["runs"],
        log_dir=merged["log_dir"],
    ))

    # Exit with non-zero if evaluation failed
    if not result.get("success", True):
        sys.exit(1)


@cli.command()
@click.argument("task")
@click.option(
    "--suite", "-s", default="custom",
    help="Suite to add to",
)
def create_test(task: str, suite: str) -> None:
    """Create a new test case."""
    console.print(
        f"✅ Test added to suite [bold]{suite}[/bold]: {task}"
    )


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
        console.print(
            "No test suites found. "
            "Run [bold]litmus init[/bold] first."
        )


# ─── litmus profiles ─────────────────────────────────────────────


@cli.command()
def profiles() -> None:
    """List available evaluation profiles."""
    from litmusai.profiles import list_profiles, load_profiles_from_dir

    # Load any custom profiles
    load_profiles_from_dir()

    all_profiles = list_profiles()
    if all_profiles:
        console.print("[bold]Available evaluation profiles:[/bold]\n")
        for p in all_profiles:
            console.print(f"  [bold cyan]{p.name}[/bold cyan]")
            console.print(f"    {p.description}")
            details = []
            details.append(f"concurrency={p.concurrency}")
            details.append(f"runs={p.runs}")
            if p.safety:
                details.append(f"safety={p.safety_depth}")
            details.append(f"threshold={p.threshold}")
            if p.temperature is not None:
                details.append(f"temperature={p.temperature}")
            if p.seed is not None:
                details.append(f"seed={p.seed}")
            if p.report:
                details.append(f"report={p.report}")
            console.print(f"    [dim]{' · '.join(details)}[/dim]\n")
    else:
        console.print("No profiles found.")


# ─── litmus history ──────────────────────────────────────────────


@cli.command()
@click.option(
    "--log-dir", "-d", default=".litmus/logs",
    help="Directory with saved result logs",
)
@click.option(
    "--limit", "-n", default=20, type=int,
    help="Max number of runs to show",
)
def history(log_dir: str, limit: int) -> None:
    """Show past evaluation runs.

    Examples:

        litmus history

        litmus history --log-dir ./logs --limit 10
    """
    from litmusai.results import list_results

    entries = list_results(log_dir, limit=limit)

    if not entries:
        console.print(
            f"[yellow]No results found in {log_dir}[/yellow]"
        )
        console.print(
            "Run an evaluation with --log-dir first:\n"
            "  litmus run -s my_suite -a my_agent:agent "
            f"--log-dir {log_dir}"
        )
        return

    table = Table(
        title=f"📊 Evaluation History ({len(entries)} runs)",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Agent", style="bold")
    table.add_column("Suite")
    table.add_column("Timestamp")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Passed", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("File", style="dim")

    for i, e in enumerate(entries, 1):
        rate = e["pass_rate"]
        rate_str = f"{rate:.0%}" if isinstance(rate, float) else str(rate)
        color = (
            "green" if rate >= 0.9
            else "yellow" if rate >= 0.7
            else "red"
        )
        table.add_row(
            str(i),
            e["agent_name"],
            e["suite_name"],
            e["timestamp"],
            f"[{color}]{rate_str}[/{color}]",
            f"{e['passed']}/{e['total']}",
            f"${e['total_cost']:.4f}",
            e["filename"],
        )

    console.print(table)


# ─── litmus diff ─────────────────────────────────────────────────


@cli.command()
@click.argument("baseline_path")
@click.argument("current_path")
@click.option(
    "--format", "fmt", default="table",
    type=click.Choice(["table", "markdown"]),
    help="Output format",
)
@click.option(
    "--fail-on-regression", is_flag=True,
    help="Exit with code 1 if any regressions found",
)
def diff(
    baseline_path: str,
    current_path: str,
    fmt: str,
    fail_on_regression: bool,
) -> None:
    """Compare two evaluation runs and show regressions.

    Examples:

        litmus diff baseline.json current.json

        litmus diff baseline.json current.json --format markdown

        litmus diff baseline.json current.json --fail-on-regression
    """
    from litmusai.results import diff_results, load_results

    # Validate paths
    for path_str, label in [
        (baseline_path, "Baseline"),
        (current_path, "Current"),
    ]:
        if not Path(path_str).exists():
            console.print(
                f"[red]{label} file not found: {path_str}[/red]"
            )
            sys.exit(1)

    try:
        baseline = load_results(baseline_path)
        current = load_results(current_path)
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON: {e}[/red]")
        sys.exit(1)

    result = diff_results(baseline, current)

    if fmt == "markdown":
        console.print(result.to_markdown())
    else:
        # Rich table output
        _print_diff_table(result)

    # Summary
    n_reg = len(result.regressions)
    n_imp = len(result.improvements)
    n_new = len(result.new_tests)

    if n_reg:
        console.print(
            f"\n[bold red]🔴 {n_reg} regression(s) found[/bold red]"
        )
    if n_imp:
        console.print(
            f"[bold green]🟢 {n_imp} improvement(s)[/bold green]"
        )
    if n_new:
        console.print(f"[bold blue]🆕 {n_new} new test(s)[/bold blue]")

    if not n_reg and not n_imp and not n_new:
        console.print(
            "\n[green]✅ No changes detected[/green]"
        )

    if fail_on_regression and result.has_regressions:
        sys.exit(1)


def _print_diff_table(result: object) -> None:
    """Print a rich table for diff results."""
    from litmusai.results import DiffSummary

    if not isinstance(result, DiffSummary):
        return

    change = result.pass_rate_change
    arrow = "📈" if change > 0 else "📉" if change < 0 else "➡️"

    console.print(
        f"\n[bold]Baseline:[/bold] {result.baseline_name} "
        f"({result.baseline_timestamp})"
    )
    console.print(
        f"[bold]Current:[/bold]  {result.current_name} "
        f"({result.current_timestamp})"
    )
    console.print(
        f"[bold]Pass Rate:[/bold] "
        f"{result.baseline_pass_rate:.0%} → "
        f"{result.current_pass_rate:.0%} "
        f"{arrow} ({change:+.1%})\n"
    )

    table = Table(title="Per-Case Comparison")
    table.add_column("", width=3)
    table.add_column("Test", style="bold")
    table.add_column("Baseline", justify="center")
    table.add_column("Current", justify="center")
    table.add_column("Score Δ", justify="right")
    table.add_column("Latency Δ", justify="right")

    for c in result.cases:
        b_str = (
            "[green]PASS[/green]" if c.baseline_passed
            else "[red]FAIL[/red]" if c.baseline_passed is not None
            else "—"
        )
        c_str = (
            "[green]PASS[/green]" if c.current_passed
            else "[red]FAIL[/red]" if c.current_passed is not None
            else "—"
        )
        sc = (
            f"{c.score_change:+.2f}" if c.score_change is not None
            else "—"
        )
        lc = (
            f"{c.latency_change_pct:+.0f}%"
            if c.latency_change_pct is not None
            else "—"
        )
        table.add_row(
            c.status_icon, c.case_name, b_str, c_str, sc, lc,
        )

    console.print(table)


# ─── litmus scan ─────────────────────────────────────────────────


@cli.command()
@click.option(
    "--agent", "-a", required=True,
    help="Agent module path (e.g. my_agent:agent)",
)
@click.option(
    "--level", "-l", default="standard",
    type=click.Choice(["basic", "standard", "thorough"]),
    help="Scan thoroughness level",
)
@click.option(
    "--categories", default=None,
    help="Comma-separated categories to test (e.g. injection,jailbreak)",
)
@click.option(
    "--output", "-o", default=None,
    help="Save scan results to JSON file",
)
@click.option(
    "--fail-on-unsafe", is_flag=True,
    help="Exit with code 1 if agent is not safe",
)
def scan(
    agent: str,
    level: str,
    categories: str | None,
    output: str | None,
    fail_on_unsafe: bool,
) -> None:
    """Run safety scan against an agent.

    Examples:

        litmus scan -a my_agent:agent

        litmus scan -a my_agent:agent --level thorough

        litmus scan -a my_agent:agent --categories injection,jailbreak

        litmus scan -a my_agent:agent --fail-on-unsafe
    """
    from litmusai.safety import Category, SafetyScanner

    # Load agent
    agent_obj = _load_agent(agent)
    if agent_obj is None:
        return

    # Parse categories
    cats: list[str | Category] | None = None
    if categories:
        cat_names = [c.strip().upper() for c in categories.split(",")]
        cats = []
        for name in cat_names:
            try:
                cats.append(Category[name])
            except KeyError:
                valid = ", ".join(c.name.lower() for c in Category)
                console.print(
                    f"[red]Unknown category: {name}[/red]\n"
                    f"Valid: {valid}"
                )
                sys.exit(1)

    console.print(
        f"🔍 [bold]Scanning agent[/bold] "
        f"(level={level}"
        f"{f', categories={categories}' if categories else ''})"
    )
    console.print()

    scanner = SafetyScanner(depth=level, categories=cats)
    report = asyncio.run(scanner.scan(agent_obj))

    # Display results
    _print_scan_report(report)

    # Save if requested
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report.to_dict(), indent=2))
        console.print(f"\n💾 Results saved to {output}")

    if fail_on_unsafe and not report.is_safe:
        sys.exit(1)


def _print_scan_report(report: object) -> None:
    """Print a rich table for safety scan results."""
    from litmusai.safety import SafetyReport

    if not isinstance(report, SafetyReport):
        return

    # Overall status
    status = (
        "[bold green]✅ SAFE[/bold green]"
        if report.is_safe
        else "[bold red]❌ UNSAFE[/bold red]"
    )
    console.print(
        f"Overall: {status} (score: {report.safety_score:.0f}/100)"
    )
    console.print()

    # Category breakdown
    table = Table(title="Safety Scan Results")
    table.add_column("Category", style="bold")
    table.add_column("Attacks", justify="right")
    table.add_column("Failures", justify="right")
    table.add_column("Status")

    cat_results: dict[str, dict[str, int]] = {}
    for r in report.findings:
        cat = r.category.value
        if cat not in cat_results:
            cat_results[cat] = {"total": 0, "failed": 0}
        cat_results[cat]["total"] += 1
        if not r.passed:
            cat_results[cat]["failed"] += 1

    for cat, data in sorted(cat_results.items()):
        status_str = (
            "[green]PASS[/green]" if data["failed"] == 0
            else "[red]FAIL[/red]"
        )
        table.add_row(
            cat,
            str(data["total"]),
            str(data["failed"]),
            status_str,
        )

    console.print(table)

    # Show failures
    failures = [r for r in report.findings if not r.passed]
    if failures:
        console.print(
            f"\n[bold red]Failures ({len(failures)}):[/bold red]"
        )
        for f in failures:
            sev = f.severity.value if f.severity else "?"
            console.print(
                f"  🔴 [{sev}] {f.category.value}: "
                f"{f.attack_id}"
            )
            if f.description:
                console.print(
                    f"     └─ {f.description[:100]}"
                )


# ─── litmus report ───────────────────────────────────────────────


@cli.command()
@click.option(
    "--results", "-r", required=True,
    help="Results JSON file",
)
@click.option(
    "--baseline", "-b", default=None,
    help="Baseline JSON to compare",
)
@click.option(
    "--html", default=None,
    help="Generate HTML report at this path",
)
@click.option(
    "--junit", default=None,
    help="Generate JUnit XML report at this path",
)
@click.option(
    "--csv", "csv_path", default=None,
    help="Generate CSV report at this path",
)
def report(
    results: str,
    baseline: str | None,
    html: str | None,
    junit: str | None,
    csv_path: str | None,
) -> None:
    """Generate a report from saved results."""
    from litmusai.ci import format_report
    from litmusai.ci import load_baseline as load_bl

    results_path = Path(results)
    if not results_path.exists():
        console.print(
            f"[red]Results file not found: {results}[/red]"
        )
        sys.exit(1)

    try:
        data = json.loads(results_path.read_text())
    except json.JSONDecodeError as e:
        console.print(
            f"[red]Invalid JSON in results file: {e}[/red]"
        )
        sys.exit(1)

    # HTML report
    if html:
        from litmusai.reports import render_html

        if baseline:
            console.print(
                "[yellow]Note: --baseline is ignored "
                "with --html[/yellow]"
            )
        path = render_html(data, html)
        console.print(
            f"📊 [bold green]HTML report saved to {path}[/bold green]"
        )

    # JUnit XML
    if junit:
        from litmusai.exports import to_junit_xml

        path = to_junit_xml(data, junit)
        console.print(
            f"📋 [bold green]JUnit XML saved to {path}[/bold green]"
        )

    # CSV
    if csv_path:
        from litmusai.exports import to_csv

        path = to_csv(data, csv_path)
        console.print(
            f"📄 [bold green]CSV saved to {path}[/bold green]"
        )

    # If only export flags were given (no markdown needed), return
    if (html or junit or csv_path) and not baseline:
        return

    baseline_data = None
    if baseline:
        baseline_data = load_bl(baseline)
        if baseline_data is None:
            console.print(
                f"[yellow]Warning: baseline not found "
                f"at {baseline}[/yellow]"
            )

    output = format_report(data, baseline_data, fmt="markdown")
    console.print(output)


# ─── litmus badges ───────────────────────────────────────────────


@cli.command()
def badges() -> None:
    """Generate README badges from latest results."""
    litmus_dir = Path(".litmus")
    baseline_path = litmus_dir / "baseline.json"

    if not baseline_path.exists():
        console.print(
            "[red]No baseline found. "
            "Run with --save-baseline first.[/red]"
        )
        sys.exit(1)

    try:
        data = json.loads(baseline_path.read_text())
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid baseline JSON: {e}[/red]")
        sys.exit(1)

    summary = data.get("summary", {})
    pass_rate = summary.get("pass_rate", 0)
    pct = f"{pass_rate:.0%}"

    color = (
        "brightgreen" if pass_rate >= 0.9
        else "yellow" if pass_rate >= 0.7
        else "red"
    )

    pct_encoded = quote(pct, safe="")
    badge_url = (
        f"https://img.shields.io/badge/"
        f"LitmusAI-{pct_encoded}%20pass-{color}"
    )

    table = Table(title="📛 README Badges")
    table.add_column("Badge", style="bold")
    table.add_column("Markdown")
    table.add_row(
        "Pass Rate",
        f"[![LitmusAI]({badge_url})]"
        f"(https://github.com/kutanti/litmusai)",
    )
    console.print(table)


@cli.command()
def dashboard() -> None:
    """Launch the results dashboard."""
    console.print("🌐 Dashboard coming soon!")


# ─── Helpers ─────────────────────────────────────────────────────


def _load_agent(agent_path: str) -> Any:
    """Load an agent from a module:attribute path."""
    import importlib
    import importlib.util

    if ":" not in agent_path:
        console.print(
            "[red]Agent path must be module:attribute "
            f"(got '{agent_path}')[/red]\n"
            "Example: my_agent:agent"
        )
        sys.exit(1)

    module_path, attr_name = agent_path.rsplit(":", 1)

    try:
        # Handle file paths (e.g. path/to/file.py:agent)
        if module_path.endswith(".py"):
            file_path = Path(module_path).resolve()
            if not file_path.exists():
                console.print(
                    f"[red]Agent file not found: "
                    f"{module_path}[/red]"
                )
                sys.exit(1)
            spec = importlib.util.spec_from_file_location(
                file_path.stem, file_path,
            )
            if spec is None or spec.loader is None:
                console.print(
                    f"[red]Cannot load module from "
                    f"{module_path}[/red]"
                )
                sys.exit(1)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            # Add cwd to path for local modules
            if "." not in sys.path:
                sys.path.insert(0, ".")
            module = importlib.import_module(module_path)
    except ImportError as e:
        console.print(
            f"[red]Cannot import module "
            f"'{module_path}': {e}[/red]"
        )
        sys.exit(1)

    agent = getattr(module, attr_name, None)
    if agent is None:
        console.print(
            f"[red]Attribute '{attr_name}' not found "
            f"in module '{module_path}'[/red]"
        )
        sys.exit(1)

    return agent


if __name__ == "__main__":
    cli()
