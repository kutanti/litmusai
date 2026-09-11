"""Run the composite action without evaluating input values as shell code."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def main() -> int:
    results_path = Path(".litmus/results.json")
    comment_path = Path(".litmus/pr-comment.md")
    # A failed invocation must not publish results left by a previous run.
    results_path.unlink(missing_ok=True)
    comment_path.unlink(missing_ok=True)

    args = [
        "litmus", "run",
        "--suite", os.environ["INPUT_SUITE"],
        "--agent", os.environ["INPUT_AGENT"],
        "--format", "json", "--output", str(results_path),
    ]
    for flag, variable in (
        ("--concurrency", "INPUT_CONCURRENCY"),
        ("--threshold", "INPUT_THRESHOLD"),
        ("--budget", "INPUT_BUDGET"),
        ("--runs", "INPUT_RUNS"),
        ("--log-dir", "INPUT_LOG_DIR"),
    ):
        value = os.environ.get(variable, "")
        if value:
            args.extend([flag, value])

    baseline_path = os.environ.get("INPUT_BASELINE", ".litmus/baseline.json")
    if Path(baseline_path).is_file():
        args.extend(["--baseline", baseline_path])
    if os.environ.get("INPUT_SAVE_BASELINE") == "true":
        args.append("--save-baseline")

    completed = subprocess.run(args, check=False)
    if not results_path.is_file():
        return completed.returncode or 1

    data = json.loads(results_path.read_text(encoding="utf-8"))
    results = data.get("results", data)
    # Keep action outputs consistent with the CLI's last-run checks and reports.
    if "run_results" in results:
        results = results["run_results"][-1]
    summary = results["summary"]
    outputs = {
        "results-path": results_path.as_posix(),
        "pass-rate": summary["pass_rate"],
        "total-cost": summary["total_cost"],
        "passed": summary["passed"],
        "failed": summary["failed"],
        "has-regression": str(data.get("has_regression", False)).lower(),
    }
    if (
        os.environ.get("INPUT_POST_COMMENT") == "true"
        and os.environ.get("GITHUB_EVENT_NAME") == "pull_request"
    ):
        from litmusai.ci import format_report, load_baseline

        report = format_report(
            results, load_baseline(baseline_path), fmt="markdown",
            threshold=float(os.environ.get("INPUT_THRESHOLD") or "0.7"),
        )
        comment_path.write_text(report, encoding="utf-8")
        outputs["comment-path"] = comment_path.as_posix()

    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        for name, value in outputs.items():
            output.write(f"{name}={value}\n")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
