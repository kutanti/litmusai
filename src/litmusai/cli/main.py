"""LitmusAI CLI — Test your AI agents from the command line."""

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="litmusai")
def cli() -> None:
    """🧪 LitmusAI — The open-source evaluation framework for AI agents."""
    pass


@cli.command()
def init() -> None:
    """Initialize a new LitmusAI project."""
    from pathlib import Path

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
@click.option("--suite", "-s", default="example", help="Test suite to run")
@click.option("--agent", "-a", required=True, help="Path to agent module")
@click.option("--concurrency", "-c", default=5, help="Max parallel evaluations")
@click.option("--output", "-o", default=None, help="Output file for results")
@click.option("--format", "fmt", default="table", type=click.Choice(["table", "json", "markdown"]))
def run(suite: str, agent: str, concurrency: int, output: str | None, fmt: str) -> None:
    """Run evaluation against a test suite."""

    console.print(f"🧪 Running suite [bold]{suite}[/bold] with agent [bold]{agent}[/bold]...")
    console.print("[dim]This is a preview — full evaluation engine coming soon![/dim]")


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
@click.option("--port", "-p", default=3000, help="Dashboard port")
def dashboard(port: int) -> None:
    """Launch the results dashboard."""
    console.print(f"🌐 Dashboard coming soon! (port {port})")


if __name__ == "__main__":
    cli()
