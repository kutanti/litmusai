"""Result logging — save, load, compare, and diff evaluation results.

Provides persistence and comparison for evaluation runs:

    # Save
    results.save("results/run-001.json")

    # Load
    loaded = load_results("results/run-001.json")

    # Diff two runs
    diff = diff_results(baseline, current)
    print(diff.to_markdown())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CaseDiff:
    """Diff for a single test case between two runs."""

    case_id: str
    case_name: str
    task: str
    # Baseline values
    baseline_passed: bool | None = None
    baseline_score: float | None = None
    baseline_latency_ms: float | None = None
    baseline_cost: float | None = None
    baseline_tokens: int = 0
    # Current values
    current_passed: bool | None = None
    current_score: float | None = None
    current_latency_ms: float | None = None
    current_cost: float | None = None
    current_tokens: int = 0

    @property
    def is_regression(self) -> bool:
        """Test went from pass → fail."""
        return (
            self.baseline_passed is True
            and self.current_passed is False
        )

    @property
    def is_improvement(self) -> bool:
        """Test went from fail → pass."""
        return (
            self.baseline_passed is False
            and self.current_passed is True
        )

    @property
    def is_new(self) -> bool:
        """Test exists only in current run."""
        return self.baseline_passed is None

    @property
    def is_removed(self) -> bool:
        """Test exists only in baseline."""
        return self.current_passed is None

    @property
    def score_change(self) -> float | None:
        if self.baseline_score is not None and self.current_score is not None:
            return self.current_score - self.baseline_score
        return None

    @property
    def latency_change_pct(self) -> float | None:
        if (
            self.baseline_latency_ms
            and self.current_latency_ms
            and self.baseline_latency_ms > 0
        ):
            return (
                (self.current_latency_ms - self.baseline_latency_ms)
                / self.baseline_latency_ms
                * 100
            )
        return None

    @property
    def cost_change_pct(self) -> float | None:
        if (
            self.baseline_cost is not None
            and self.current_cost is not None
            and self.baseline_cost > 0
        ):
            return (
                (self.current_cost - self.baseline_cost)
                / self.baseline_cost
                * 100
            )
        return None

    @property
    def status_icon(self) -> str:
        if self.is_regression:
            return "🔴"
        if self.is_improvement:
            return "🟢"
        if self.is_new:
            return "🆕"
        if self.is_removed:
            return "⚪"
        if self.current_passed:
            return "✅"
        return "❌"


@dataclass
class DiffSummary:
    """Comparison between two evaluation runs."""

    baseline_name: str
    current_name: str
    baseline_suite: str
    current_suite: str
    baseline_timestamp: str
    current_timestamp: str
    cases: list[CaseDiff] = field(default_factory=list)

    @property
    def regressions(self) -> list[CaseDiff]:
        return [c for c in self.cases if c.is_regression]

    @property
    def improvements(self) -> list[CaseDiff]:
        return [c for c in self.cases if c.is_improvement]

    @property
    def new_tests(self) -> list[CaseDiff]:
        return [c for c in self.cases if c.is_new]

    @property
    def removed_tests(self) -> list[CaseDiff]:
        return [c for c in self.cases if c.is_removed]

    @property
    def baseline_pass_rate(self) -> float:
        baseline_cases = [
            c for c in self.cases if c.baseline_passed is not None
        ]
        if not baseline_cases:
            return 0.0
        return (
            sum(1 for c in baseline_cases if c.baseline_passed)
            / len(baseline_cases)
        )

    @property
    def current_pass_rate(self) -> float:
        current_cases = [
            c for c in self.cases if c.current_passed is not None
        ]
        if not current_cases:
            return 0.0
        return (
            sum(1 for c in current_cases if c.current_passed)
            / len(current_cases)
        )

    @property
    def pass_rate_change(self) -> float:
        return self.current_pass_rate - self.baseline_pass_rate

    @property
    def has_regressions(self) -> bool:
        return len(self.regressions) > 0

    def to_markdown(self) -> str:
        """Format as markdown report."""
        lines: list[str] = []
        lines.append("## 📊 Evaluation Diff")
        lines.append("")
        lines.append(
            f"**Baseline:** {self.baseline_name} "
            f"({self.baseline_timestamp})"
        )
        lines.append(
            f"**Current:** {self.current_name} "
            f"({self.current_timestamp})"
        )
        lines.append("")

        # Summary
        b_rate = f"{self.baseline_pass_rate:.0%}"
        c_rate = f"{self.current_pass_rate:.0%}"
        change = self.pass_rate_change
        arrow = "📈" if change > 0 else "📉" if change < 0 else "➡️"
        lines.append(
            f"**Pass Rate:** {b_rate} → {c_rate} "
            f"{arrow} ({change:+.1%})"
        )
        lines.append("")

        # Regressions
        if self.regressions:
            lines.append(
                f"### 🔴 Regressions ({len(self.regressions)})"
            )
            lines.append("")
            for c in self.regressions:
                score_info = ""
                if c.score_change is not None:
                    score_info = f" (score: {c.score_change:+.2f})"
                lines.append(
                    f"- **{c.case_name}** — was passing, now failing"
                    f"{score_info}"
                )
            lines.append("")

        # Improvements
        if self.improvements:
            lines.append(
                f"### 🟢 Improvements ({len(self.improvements)})"
            )
            lines.append("")
            for c in self.improvements:
                lines.append(
                    f"- **{c.case_name}** — was failing, now passing"
                )
            lines.append("")

        # New tests
        if self.new_tests:
            lines.append(f"### 🆕 New Tests ({len(self.new_tests)})")
            lines.append("")
            for c in self.new_tests:
                status = "✅" if c.current_passed else "❌"
                lines.append(f"- {status} **{c.case_name}**")
            lines.append("")

        # Per-case table
        lines.append("### Per-Case Details")
        lines.append("")
        lines.append(
            "| Status | Test | Baseline | Current "
            "| Score Δ | Latency Δ |"
        )
        lines.append(
            "|--------|------|----------|---------|"
            "---------|-----------|"
        )

        for c in self.cases:
            b_status = (
                "✅" if c.baseline_passed
                else "❌" if c.baseline_passed is not None
                else "—"
            )
            c_status = (
                "✅" if c.current_passed
                else "❌" if c.current_passed is not None
                else "—"
            )
            score_d = (
                f"{c.score_change:+.2f}" if c.score_change is not None
                else "—"
            )
            lat_d = (
                f"{c.latency_change_pct:+.0f}%"
                if c.latency_change_pct is not None
                else "—"
            )
            lines.append(
                f"| {c.status_icon} | {c.case_name} "
                f"| {b_status} | {c_status} "
                f"| {score_d} | {lat_d} |"
            )

        return "\n".join(lines)

    def to_table(self) -> str:
        """Format as terminal-friendly table."""
        lines: list[str] = []
        header = (
            f"{'':3} {'Test':<30} {'Baseline':>10} "
            f"{'Current':>10} {'Score Δ':>10} {'Latency Δ':>10}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        for c in self.cases:
            b = "PASS" if c.baseline_passed else (
                "FAIL" if c.baseline_passed is not None else "—"
            )
            cur = "PASS" if c.current_passed else (
                "FAIL" if c.current_passed is not None else "—"
            )
            sd = (
                f"{c.score_change:+.2f}" if c.score_change is not None
                else "—"
            )
            ld = (
                f"{c.latency_change_pct:+.0f}%"
                if c.latency_change_pct is not None
                else "—"
            )
            name = c.case_name[:30]
            lines.append(
                f"{c.status_icon:3} {name:<30} {b:>10} "
                f"{cur:>10} {sd:>10} {ld:>10}"
            )

        return "\n".join(lines)


def load_results(path: str | Path) -> dict[str, Any]:
    """Load evaluation results from a JSON file."""
    path = Path(path)
    with open(path) as f:
        data: dict[str, Any] = json.load(f)
    return data


def list_results(
    log_dir: str | Path,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """List saved evaluation results in a directory.

    Returns a list of summary dicts sorted by timestamp (newest first).
    """
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return []

    entries: list[dict[str, Any]] = []
    for f in sorted(log_dir.glob("*.json"), reverse=True):
        try:
            data = load_results(f)
            entries.append({
                "file": str(f),
                "filename": f.name,
                "agent_name": data.get("agent_name", "?"),
                "suite_name": data.get("suite_name", "?"),
                "timestamp": data.get("timestamp", "?"),
                "pass_rate": data.get("summary", {}).get(
                    "pass_rate", 0,
                ),
                "total": data.get("summary", {}).get("total", 0),
                "passed": data.get("summary", {}).get("passed", 0),
                "total_cost": data.get("summary", {}).get(
                    "total_cost", 0,
                ),
            })
        except (json.JSONDecodeError, KeyError):
            continue

        if len(entries) >= limit:
            break

    return entries


def diff_results(
    baseline: dict[str, Any],
    current: dict[str, Any],
) -> DiffSummary:
    """Compare two evaluation results and produce a diff.

    Args:
        baseline: Result dict from an earlier run (via :func:`load_results`).
        current: Result dict from the current run.

    Returns:
        A :class:`DiffSummary` with regressions, improvements, and
        per-case diffs.
    """
    # Index baseline results by case_id
    baseline_cases: dict[str, dict[str, Any]] = {}
    for r in baseline.get("results", []):
        baseline_cases[r["case_id"]] = r

    # Index current results
    current_cases: dict[str, dict[str, Any]] = {}
    for r in current.get("results", []):
        current_cases[r["case_id"]] = r

    # Union of all case IDs
    all_ids = list(dict.fromkeys(
        list(baseline_cases.keys()) + list(current_cases.keys())
    ))

    diffs: list[CaseDiff] = []
    for case_id in all_ids:
        b = baseline_cases.get(case_id)
        c = current_cases.get(case_id)

        diff = CaseDiff(
            case_id=case_id,
            case_name=(
                (c or b or {}).get("case_name", case_id)
            ),
            task=(c or b or {}).get("task", ""),
        )

        if b:
            diff.baseline_passed = b.get("passed")
            diff.baseline_score = b.get("score")
            diff.baseline_latency_ms = b.get("latency_ms")
            diff.baseline_cost = b.get("cost")
            diff.baseline_tokens = (
                b.get("input_tokens", 0) + b.get("output_tokens", 0)
            )

        if c:
            diff.current_passed = c.get("passed")
            diff.current_score = c.get("score")
            diff.current_latency_ms = c.get("latency_ms")
            diff.current_cost = c.get("cost")
            diff.current_tokens = (
                c.get("input_tokens", 0) + c.get("output_tokens", 0)
            )

        diffs.append(diff)

    return DiffSummary(
        baseline_name=baseline.get("agent_name", "baseline"),
        current_name=current.get("agent_name", "current"),
        baseline_suite=baseline.get("suite_name", "?"),
        current_suite=current.get("suite_name", "?"),
        baseline_timestamp=baseline.get("timestamp", "?"),
        current_timestamp=current.get("timestamp", "?"),
        cases=diffs,
    )
