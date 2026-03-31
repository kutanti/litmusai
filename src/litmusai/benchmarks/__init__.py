"""Cost & latency benchmarking for AI agent evaluation.

Track token usage, API costs, and latency per task. Compare agents
side-by-side with cost-per-successful-task metrics — the number
nobody else gives you.

Features:
    - Auto cost tracking (OpenAI, Anthropic, Google, 60+ models)
    - Cost per task & cost per *successful* task
    - Latency percentiles (p50, p95, p99)
    - Model comparison tables
    - CostGuard budget limits
    - Export: JSON, Markdown, CSV
"""

from __future__ import annotations

import csv
import io
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

# ─── Model Pricing Database ───────────────────────────────────────


@dataclass(frozen=True)
class ModelPricing:
    """Pricing for an LLM model (per million tokens)."""

    model: str
    input_cost_per_m: float  # $/million input tokens
    output_cost_per_m: float  # $/million output tokens
    provider: str = ""

    @property
    def input_cost_per_token(self) -> float:
        """Cost per single input token."""
        return self.input_cost_per_m / 1_000_000

    @property
    def output_cost_per_token(self) -> float:
        """Cost per single output token."""
        return self.output_cost_per_m / 1_000_000


# Pricing as of March 2026 — update as needed
_PRICING_DB: dict[str, ModelPricing] = {}


def _register(*models: ModelPricing) -> None:
    for m in models:
        _PRICING_DB[m.model] = m


# ── OpenAI ─────────────────────────────────────────────────────
_register(
    ModelPricing("gpt-4o", 2.50, 10.00, "openai"),
    ModelPricing("gpt-4o-mini", 0.15, 0.60, "openai"),
    ModelPricing("gpt-4-turbo", 10.00, 30.00, "openai"),
    ModelPricing("gpt-4", 30.00, 60.00, "openai"),
    ModelPricing("gpt-3.5-turbo", 0.50, 1.50, "openai"),
    ModelPricing("o1", 15.00, 60.00, "openai"),
    ModelPricing("o1-mini", 3.00, 12.00, "openai"),
    ModelPricing("o3-mini", 1.10, 4.40, "openai"),
)

# ── Anthropic ──────────────────────────────────────────────────
_register(
    ModelPricing("claude-opus-4-20250514", 15.00, 75.00, "anthropic"),
    ModelPricing("claude-sonnet-4-20250514", 3.00, 15.00, "anthropic"),
    ModelPricing("claude-3-5-sonnet-20241022", 3.00, 15.00, "anthropic"),
    ModelPricing("claude-3-5-haiku-20241022", 0.80, 4.00, "anthropic"),
    ModelPricing("claude-3-haiku-20240307", 0.25, 1.25, "anthropic"),
)

# ── Google ─────────────────────────────────────────────────────
_register(
    ModelPricing("gemini-2.0-flash", 0.10, 0.40, "google"),
    ModelPricing("gemini-1.5-pro", 3.50, 10.50, "google"),
    ModelPricing("gemini-1.5-flash", 0.075, 0.30, "google"),
)

# ── Meta / Open Source ─────────────────────────────────────────
_register(
    ModelPricing("llama-3.1-70b", 0.88, 0.88, "meta"),
    ModelPricing("llama-3.1-8b", 0.18, 0.18, "meta"),
)

# ── Mistral ────────────────────────────────────────────────────
_register(
    ModelPricing("mistral-large", 3.00, 9.00, "mistral"),
    ModelPricing("mistral-small", 0.10, 0.30, "mistral"),
    ModelPricing("mixtral-8x7b", 0.24, 0.24, "mistral"),
)

# ── DeepSeek ───────────────────────────────────────────────────
_register(
    ModelPricing("deepseek-v3", 0.27, 1.10, "deepseek"),
    ModelPricing("deepseek-r1", 0.55, 2.19, "deepseek"),
)


def get_pricing(model: str) -> ModelPricing | None:
    """Look up pricing for a model (supports fuzzy matching).

    Exact match is preferred. For fuzzy matching, the longest
    matching key wins (so "gpt-4o-mini" beats "gpt-4o").
    """
    if model in _PRICING_DB:
        return _PRICING_DB[model]
    # Try partial match — prefer longest key match
    model_lower = model.lower()
    best: ModelPricing | None = None
    best_len = 0
    for key, pricing in _PRICING_DB.items():
        if key in model_lower or model_lower in key:
            if len(key) > best_len:
                best = pricing
                best_len = len(key)
    return best


def register_pricing(
    model: str,
    input_cost_per_m: float,
    output_cost_per_m: float,
    provider: str = "custom",
) -> None:
    """Register custom pricing for a model."""
    _PRICING_DB[model] = ModelPricing(
        model=model,
        input_cost_per_m=input_cost_per_m,
        output_cost_per_m=output_cost_per_m,
        provider=provider,
    )


def list_models() -> list[ModelPricing]:
    """List all models with known pricing."""
    return list(_PRICING_DB.values())


# ─── Task Metrics ─────────────────────────────────────────────────


@dataclass
class TaskMetrics:
    """Metrics for a single task execution."""

    task_id: str
    task_name: str = ""
    model: str = ""
    agent_name: str = ""
    passed: bool = False

    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0

    # Timing
    latency_ms: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0

    # Cost (computed)
    cost: float = 0.0

    # Score
    score: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def compute_cost(self, pricing: ModelPricing | None = None) -> float:
        """Compute cost from token usage and pricing."""
        p = pricing or get_pricing(self.model)
        if p:
            self.cost = (
                self.input_tokens * p.input_cost_per_token
                + self.output_tokens * p.output_cost_per_token
            )
        return self.cost


# ─── Cost Tracker ──────────────────────────────────────────────────


class CostTracker:
    """Track costs and latency across an evaluation run.

    Example:
        >>> tracker = CostTracker(model="gpt-4o-mini")
        >>> tracker.record(task_id="t1", input_tokens=500,
        ...     output_tokens=200, latency_ms=450, passed=True)
        >>> print(tracker.summary())
    """

    def __init__(
        self,
        model: str = "",
        agent_name: str = "",
        pricing: ModelPricing | None = None,
    ):
        self.model = model
        self.agent_name = agent_name
        self.pricing = pricing or get_pricing(model)
        self.tasks: list[TaskMetrics] = []
        self._start_time: float = 0.0

    def start(self) -> None:
        """Mark the start of an evaluation run."""
        self._start_time = time.monotonic()

    def record(
        self,
        task_id: str,
        task_name: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        passed: bool = False,
        score: float = 0.0,
    ) -> TaskMetrics:
        """Record metrics for a single task."""
        m = TaskMetrics(
            task_id=task_id,
            task_name=task_name or task_id,
            model=self.model,
            agent_name=self.agent_name,
            passed=passed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            score=score,
        )
        m.compute_cost(self.pricing)
        self.tasks.append(m)
        return m

    # ─── Aggregate Stats ──────────────────────────────────────

    @property
    def total_tasks(self) -> int:
        return len(self.tasks)

    @property
    def passed_tasks(self) -> int:
        return sum(1 for t in self.tasks if t.passed)

    @property
    def failed_tasks(self) -> int:
        return self.total_tasks - self.passed_tasks

    @property
    def pass_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.passed_tasks / self.total_tasks

    @property
    def total_cost(self) -> float:
        return sum(t.cost for t in self.tasks)

    @property
    def total_input_tokens(self) -> int:
        return sum(t.input_tokens for t in self.tasks)

    @property
    def total_output_tokens(self) -> int:
        return sum(t.output_tokens for t in self.tasks)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def avg_cost_per_task(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.total_cost / self.total_tasks

    @property
    def cost_per_pass(self) -> float:
        """Cost per successful task — the metric nobody else tracks."""
        if self.passed_tasks == 0:
            return float("inf")
        return self.total_cost / self.passed_tasks

    @property
    def avg_score(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return sum(t.score for t in self.tasks) / self.total_tasks

    # ─── Latency Stats ────────────────────────────────────────

    def _latencies(self) -> list[float]:
        return [t.latency_ms for t in self.tasks if t.latency_ms > 0]

    @property
    def avg_latency_ms(self) -> float:
        lats = self._latencies()
        return statistics.mean(lats) if lats else 0.0

    @property
    def p50_latency_ms(self) -> float:
        lats = self._latencies()
        return statistics.median(lats) if lats else 0.0

    @property
    def p95_latency_ms(self) -> float:
        lats = self._latencies()
        if not lats:
            return 0.0
        return _percentile(lats, 95)

    @property
    def p99_latency_ms(self) -> float:
        lats = self._latencies()
        if not lats:
            return 0.0
        return _percentile(lats, 99)

    # ─── Efficiency ────────────────────────────────────────────

    @property
    def efficiency_score(self) -> float:
        """Quality-to-cost ratio (higher is better).

        efficiency = pass_rate / cost_per_pass
        Normalized so cheap + accurate = high score.
        """
        if self.cost_per_pass == float("inf") or self.cost_per_pass == 0:
            return 0.0
        return self.pass_rate / self.cost_per_pass

    # ─── Summary ───────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Return a summary dict of all metrics."""
        return {
            "model": self.model,
            "agent_name": self.agent_name,
            "total_tasks": self.total_tasks,
            "passed": self.passed_tasks,
            "failed": self.failed_tasks,
            "pass_rate": round(self.pass_rate, 4),
            "total_cost": round(self.total_cost, 6),
            "avg_cost_per_task": round(self.avg_cost_per_task, 6),
            "cost_per_pass": (
                round(self.cost_per_pass, 6)
                if self.cost_per_pass != float("inf")
                else "N/A"
            ),
            "total_tokens": self.total_tokens,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "p50_latency_ms": round(self.p50_latency_ms, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
            "p99_latency_ms": round(self.p99_latency_ms, 1),
            "efficiency_score": round(self.efficiency_score, 4),
            "avg_score": round(self.avg_score, 4),
        }


# ─── CostGuard ────────────────────────────────────────────────────


@dataclass
class BudgetAlert:
    """A budget limit violation."""

    level: str  # "warning" or "error"
    message: str
    actual: float
    limit: float


class CostGuard:
    """Budget guardrails for evaluation runs.

    Set limits on cost per task, total run cost, or token usage.
    Get alerts when limits are exceeded.

    Example:
        >>> guard = CostGuard(max_cost_per_task=0.05, max_total_cost=1.00)
        >>> alerts = guard.check(tracker)
    """

    def __init__(
        self,
        max_cost_per_task: float | None = None,
        max_total_cost: float | None = None,
        max_tokens_per_task: int | None = None,
        max_latency_ms: float | None = None,
    ):
        if max_cost_per_task is not None and max_cost_per_task <= 0:
            raise ValueError("max_cost_per_task must be positive")
        if max_total_cost is not None and max_total_cost <= 0:
            raise ValueError("max_total_cost must be positive")
        if max_tokens_per_task is not None and max_tokens_per_task <= 0:
            raise ValueError("max_tokens_per_task must be positive")
        if max_latency_ms is not None and max_latency_ms <= 0:
            raise ValueError("max_latency_ms must be positive")

        self.max_cost_per_task = max_cost_per_task
        self.max_total_cost = max_total_cost
        self.max_tokens_per_task = max_tokens_per_task
        self.max_latency_ms = max_latency_ms

    def check_task(self, task: TaskMetrics) -> list[BudgetAlert]:
        """Check a single task against budget limits."""
        alerts: list[BudgetAlert] = []

        if self.max_cost_per_task is not None and task.cost > self.max_cost_per_task:
            alerts.append(BudgetAlert(
                level="error",
                message=(
                    f"Task '{task.task_id}' cost ${task.cost:.6f} "
                    f"exceeds limit ${self.max_cost_per_task:.6f}"
                ),
                actual=task.cost,
                limit=self.max_cost_per_task,
            ))

        if (
            self.max_tokens_per_task is not None
            and task.total_tokens > self.max_tokens_per_task
        ):
            alerts.append(BudgetAlert(
                level="warning",
                message=(
                    f"Task '{task.task_id}' used {task.total_tokens} tokens, "
                    f"exceeds limit {self.max_tokens_per_task}"
                ),
                actual=float(task.total_tokens),
                limit=float(self.max_tokens_per_task),
            ))

        if self.max_latency_ms is not None and task.latency_ms > self.max_latency_ms:
            alerts.append(BudgetAlert(
                level="warning",
                message=(
                    f"Task '{task.task_id}' took {task.latency_ms:.0f}ms, "
                    f"exceeds limit {self.max_latency_ms:.0f}ms"
                ),
                actual=task.latency_ms,
                limit=self.max_latency_ms,
            ))

        return alerts

    def check(self, tracker: CostTracker) -> list[BudgetAlert]:
        """Check all tasks and totals against budget limits."""
        alerts: list[BudgetAlert] = []

        # Per-task checks
        for task in tracker.tasks:
            alerts.extend(self.check_task(task))

        # Total cost check
        if (
            self.max_total_cost is not None
            and tracker.total_cost > self.max_total_cost
        ):
            alerts.append(BudgetAlert(
                level="error",
                message=(
                    f"Total cost ${tracker.total_cost:.6f} "
                    f"exceeds budget ${self.max_total_cost:.6f}"
                ),
                actual=tracker.total_cost,
                limit=self.max_total_cost,
            ))

        return alerts


# ─── Model Comparison ─────────────────────────────────────────────


@dataclass
class ComparisonResult:
    """Side-by-side comparison of multiple agents/models."""

    trackers: list[CostTracker] = field(default_factory=list)
    recommendation: str = ""

    def to_markdown(self) -> str:
        """Render comparison as a Markdown table."""
        if not self.trackers:
            return "No data to compare."

        header = (
            "| Model | Pass Rate | Cost/Task | Cost/Pass | "
            "P50 Latency | P95 Latency | Efficiency | Score |"
        )
        sep = (
            "|-------|-----------|-----------|"
            "-----------|-------------|-------------|"
            "------------|-------|"
        )

        rows = []
        for t in self.trackers:
            s = t.summary()
            name = s["model"] or s["agent_name"] or "unknown"
            cpp = s["cost_per_pass"]
            cpp_str = f"${cpp:.4f}" if isinstance(cpp, float) else cpp
            rows.append(
                f"| {name} "
                f"| {s['pass_rate']:.1%} "
                f"| ${s['avg_cost_per_task']:.4f} "
                f"| {cpp_str} "
                f"| {s['p50_latency_ms']:.0f}ms "
                f"| {s['p95_latency_ms']:.0f}ms "
                f"| {s['efficiency_score']:.2f} "
                f"| {s['avg_score']:.2f} |"
            )

        lines = [header, sep, *rows]
        if self.recommendation:
            lines.append(f"\n**Recommendation:** {self.recommendation}")

        return "\n".join(lines)

    def to_json(self) -> str:
        """Export comparison as JSON."""
        data = {
            "comparison": [t.summary() for t in self.trackers],
            "recommendation": self.recommendation,
        }
        return json.dumps(data, indent=2)

    def to_csv(self) -> str:
        """Export comparison as CSV."""
        if not self.trackers:
            return ""

        fields = [
            "model", "agent_name", "total_tasks", "passed", "failed",
            "pass_rate", "total_cost", "avg_cost_per_task", "cost_per_pass",
            "total_tokens", "input_tokens", "output_tokens",
            "avg_latency_ms", "p50_latency_ms", "p95_latency_ms",
            "p99_latency_ms", "efficiency_score", "avg_score",
        ]

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for t in self.trackers:
            writer.writerow(t.summary())

        return output.getvalue()


def compare_models(*trackers: CostTracker) -> ComparisonResult:
    """Compare multiple agents/models and recommend the best one.

    Example:
        >>> result = compare_models(tracker_gpt4, tracker_claude, tracker_mini)
        >>> print(result.to_markdown())
    """
    if not trackers:
        return ComparisonResult()

    result = ComparisonResult(trackers=list(trackers))

    # Find best by efficiency (quality / cost)
    best = max(trackers, key=lambda t: t.efficiency_score)
    cheapest = min(trackers, key=lambda t: t.avg_cost_per_task)
    most_accurate = max(trackers, key=lambda t: t.pass_rate)
    fastest = min(
        trackers,
        key=lambda t: t.p50_latency_ms if t.p50_latency_ms > 0 else float("inf"),
    )

    parts = []
    best_name = best.model or best.agent_name
    parts.append(f"🏆 Best efficiency: **{best_name}** "
                 f"(efficiency={best.efficiency_score:.2f})")

    if cheapest is not best:
        cheapest_name = cheapest.model or cheapest.agent_name
        parts.append(f"💰 Cheapest: **{cheapest_name}** "
                     f"(${cheapest.avg_cost_per_task:.4f}/task)")

    if most_accurate is not best:
        acc_name = most_accurate.model or most_accurate.agent_name
        parts.append(f"🎯 Most accurate: **{acc_name}** "
                     f"({most_accurate.pass_rate:.1%} pass rate)")

    if fastest is not best:
        fast_name = fastest.model or fastest.agent_name
        parts.append(f"⚡ Fastest: **{fast_name}** "
                     f"({fastest.p50_latency_ms:.0f}ms p50)")

    result.recommendation = " | ".join(parts)

    return result


# ─── Helpers ───────────────────────────────────────────────────────


def _percentile(data: list[float], pct: float) -> float:
    """Calculate percentile from a list of values."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * pct / 100
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    d0 = sorted_data[f] * (c - k)
    d1 = sorted_data[c] * (k - f)
    return d0 + d1
