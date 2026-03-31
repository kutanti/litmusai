"""Tests for the Cost & Latency Benchmarking module."""

from __future__ import annotations

import json

import pytest

from litmusai.benchmarks import (
    BudgetAlert,
    ComparisonResult,
    CostGuard,
    CostTracker,
    ModelPricing,
    TaskMetrics,
    compare_models,
    get_pricing,
    list_models,
    register_pricing,
)

# ─── Test ModelPricing ─────────────────────────────────────────────


class TestModelPricing:
    def test_per_token_cost(self):
        p = ModelPricing("test-model", 2.50, 10.00, "test")
        assert p.input_cost_per_token == 2.50 / 1_000_000
        assert p.output_cost_per_token == 10.00 / 1_000_000

    def test_frozen(self):
        p = ModelPricing("test", 1.0, 2.0)
        with pytest.raises(AttributeError):
            p.model = "changed"  # type: ignore[misc]

    def test_provider_field(self):
        p = ModelPricing("m", 1.0, 2.0, "openai")
        assert p.provider == "openai"


# ─── Test Pricing Database ────────────────────────────────────────


class TestPricingDB:
    def test_get_known_model(self):
        p = get_pricing("gpt-4o-mini")
        assert p is not None
        assert p.model == "gpt-4o-mini"
        assert p.provider == "openai"

    def test_get_anthropic_model(self):
        p = get_pricing("claude-sonnet-4-20250514")
        assert p is not None
        assert p.provider == "anthropic"

    def test_get_unknown_model(self):
        assert get_pricing("totally-fake-model-xyz") is None

    def test_fuzzy_match(self):
        # "gpt-4o" is a substring of "openai/gpt-4o" but also of "gpt-4o-mini"
        # Fuzzy match should return a valid pricing object
        p = get_pricing("openai/gpt-4o-mini")
        assert p is not None
        assert p.model == "gpt-4o-mini"

    def test_register_custom(self):
        register_pricing("my-custom-model", 0.50, 1.00, "custom")
        p = get_pricing("my-custom-model")
        assert p is not None
        assert p.input_cost_per_m == 0.50
        assert p.provider == "custom"

    def test_register_normalizes_to_lowercase(self):
        register_pricing("My-Fancy-Model", 1.00, 2.00)
        p = get_pricing("my-fancy-model")
        assert p is not None

    def test_register_negative_cost_raises(self):
        with pytest.raises(ValueError, match="input_cost_per_m"):
            register_pricing("bad-model", -1.0, 2.0)

    def test_register_negative_output_cost_raises(self):
        with pytest.raises(ValueError, match="output_cost_per_m"):
            register_pricing("bad-model", 1.0, -2.0)

    def test_register_empty_name_raises(self):
        with pytest.raises(ValueError, match="model name"):
            register_pricing("", 1.0, 2.0)

    def test_get_empty_string_returns_none(self):
        assert get_pricing("") is None

    def test_get_whitespace_returns_none(self):
        assert get_pricing("   ") is None

    def test_list_models(self):
        models = list_models()
        assert len(models) > 10
        names = [m.model for m in models]
        assert "gpt-4o" in names
        assert "gpt-4o-mini" in names


# ─── Test TaskMetrics ─────────────────────────────────────────────


class TestTaskMetrics:
    def test_total_tokens(self):
        m = TaskMetrics(task_id="t1", input_tokens=500, output_tokens=200)
        assert m.total_tokens == 700

    def test_compute_cost_with_pricing(self):
        p = ModelPricing("test", 2.00, 8.00)
        m = TaskMetrics(task_id="t1", input_tokens=1000, output_tokens=500)
        cost = m.compute_cost(p)
        expected = 1000 * 2.0 / 1e6 + 500 * 8.0 / 1e6
        assert abs(cost - expected) < 1e-10

    def test_compute_cost_auto_lookup(self):
        m = TaskMetrics(task_id="t1", model="gpt-4o-mini",
                        input_tokens=1000, output_tokens=500)
        cost = m.compute_cost()
        assert cost > 0

    def test_compute_cost_unknown_model(self):
        m = TaskMetrics(task_id="t1", model="unknown-model",
                        input_tokens=1000, output_tokens=500)
        cost = m.compute_cost()
        assert cost == 0.0  # No pricing found

    def test_defaults(self):
        m = TaskMetrics(task_id="t1")
        assert not m.passed
        assert m.cost == 0.0
        assert m.latency_ms == 0.0


# ─── Test CostTracker ─────────────────────────────────────────────


class TestCostTracker:
    def _make_tracker(self) -> CostTracker:
        tracker = CostTracker(model="gpt-4o-mini", agent_name="test-agent")
        tracker.record("t1", input_tokens=500, output_tokens=200,
                        latency_ms=300, passed=True, score=0.9)
        tracker.record("t2", input_tokens=800, output_tokens=400,
                        latency_ms=550, passed=True, score=0.8)
        tracker.record("t3", input_tokens=600, output_tokens=300,
                        latency_ms=400, passed=False, score=0.3)
        return tracker

    def test_total_tasks(self):
        t = self._make_tracker()
        assert t.total_tasks == 3

    def test_passed_failed(self):
        t = self._make_tracker()
        assert t.passed_tasks == 2
        assert t.failed_tasks == 1

    def test_pass_rate(self):
        t = self._make_tracker()
        assert abs(t.pass_rate - 2 / 3) < 0.01

    def test_total_tokens(self):
        t = self._make_tracker()
        assert t.total_input_tokens == 1900
        assert t.total_output_tokens == 900
        assert t.total_tokens == 2800

    def test_total_cost(self):
        t = self._make_tracker()
        assert t.total_cost > 0

    def test_avg_cost_per_task(self):
        t = self._make_tracker()
        assert abs(t.avg_cost_per_task - t.total_cost / 3) < 1e-10

    def test_cost_per_pass(self):
        t = self._make_tracker()
        assert abs(t.cost_per_pass - t.total_cost / 2) < 1e-10

    def test_cost_per_pass_none_passed(self):
        tracker = CostTracker(model="gpt-4o-mini")
        tracker.record("t1", input_tokens=100, output_tokens=50, passed=False)
        assert tracker.cost_per_pass == float("inf")

    def test_avg_score(self):
        t = self._make_tracker()
        assert abs(t.avg_score - (0.9 + 0.8 + 0.3) / 3) < 0.01

    def test_avg_latency(self):
        t = self._make_tracker()
        assert abs(t.avg_latency_ms - (300 + 550 + 400) / 3) < 0.1

    def test_p50_latency(self):
        t = self._make_tracker()
        assert t.p50_latency_ms == 400.0  # median of [300, 400, 550]

    def test_p95_p99_latency(self):
        t = self._make_tracker()
        assert t.p95_latency_ms > 0
        assert t.p99_latency_ms > 0
        assert t.p99_latency_ms >= t.p95_latency_ms

    def test_empty_tracker(self):
        t = CostTracker()
        assert t.total_tasks == 0
        assert t.pass_rate == 0.0
        assert t.total_cost == 0.0
        assert t.avg_cost_per_task == 0.0
        assert t.avg_latency_ms == 0.0
        assert t.p50_latency_ms == 0.0
        assert t.avg_score == 0.0

    def test_efficiency_score(self):
        t = self._make_tracker()
        assert t.efficiency_score > 0

    def test_efficiency_zero_when_no_passes(self):
        tracker = CostTracker(model="gpt-4o-mini")
        tracker.record("t1", input_tokens=100, output_tokens=50, passed=False)
        assert tracker.efficiency_score == 0.0

    def test_summary_keys(self):
        t = self._make_tracker()
        s = t.summary()
        expected_keys = {
            "model", "agent_name", "total_tasks", "passed", "failed",
            "pass_rate", "total_cost", "avg_cost_per_task", "cost_per_pass",
            "total_tokens", "input_tokens", "output_tokens",
            "avg_latency_ms", "p50_latency_ms", "p95_latency_ms",
            "p99_latency_ms", "efficiency_score", "avg_score",
        }
        assert set(s.keys()) == expected_keys

    def test_summary_cost_per_pass_na(self):
        tracker = CostTracker(model="gpt-4o-mini")
        tracker.record("t1", input_tokens=100, output_tokens=50, passed=False)
        s = tracker.summary()
        assert s["cost_per_pass"] == "N/A"

    def test_start(self):
        t = CostTracker()
        t.start()
        assert t._start_time > 0

    def test_record_returns_metrics(self):
        t = CostTracker(model="gpt-4o-mini")
        m = t.record("t1", input_tokens=100, output_tokens=50, passed=True)
        assert isinstance(m, TaskMetrics)
        assert m.task_id == "t1"
        assert m.cost > 0

    def test_no_model_no_pricing(self):
        """CostTracker without a model should not pick up random pricing."""
        t = CostTracker()
        assert t.pricing is None
        m = t.record("t1", input_tokens=1000, output_tokens=500, passed=True)
        assert m.cost == 0.0


# ─── Test CostGuard ───────────────────────────────────────────────


class TestCostGuard:
    def test_no_violations(self):
        guard = CostGuard(max_cost_per_task=1.0, max_total_cost=10.0)
        tracker = CostTracker(model="gpt-4o-mini")
        tracker.record("t1", input_tokens=100, output_tokens=50, passed=True)
        alerts = guard.check(tracker)
        assert len(alerts) == 0

    def test_cost_per_task_violation(self):
        guard = CostGuard(max_cost_per_task=0.000001)
        tracker = CostTracker(model="gpt-4o-mini")
        tracker.record("t1", input_tokens=10000, output_tokens=5000, passed=True)
        alerts = guard.check(tracker)
        cost_alerts = [a for a in alerts if a.level == "error"]
        assert len(cost_alerts) >= 1

    def test_total_cost_violation(self):
        guard = CostGuard(max_total_cost=0.000001)
        tracker = CostTracker(model="gpt-4o-mini")
        tracker.record("t1", input_tokens=10000, output_tokens=5000, passed=True)
        alerts = guard.check(tracker)
        total_alerts = [a for a in alerts if "Total cost" in a.message]
        assert len(total_alerts) == 1

    def test_token_limit_violation(self):
        guard = CostGuard(max_tokens_per_task=100)
        tracker = CostTracker(model="gpt-4o-mini")
        tracker.record("t1", input_tokens=500, output_tokens=200, passed=True)
        alerts = guard.check(tracker)
        token_alerts = [a for a in alerts if "tokens" in a.message]
        assert len(token_alerts) == 1

    def test_latency_violation(self):
        guard = CostGuard(max_latency_ms=100)
        tracker = CostTracker(model="gpt-4o-mini")
        tracker.record("t1", input_tokens=100, output_tokens=50,
                        latency_ms=500, passed=True)
        alerts = guard.check(tracker)
        latency_alerts = [a for a in alerts if "ms" in a.message]
        assert len(latency_alerts) == 1

    def test_multiple_violations(self):
        guard = CostGuard(
            max_cost_per_task=0.000001,
            max_tokens_per_task=100,
            max_latency_ms=50,
        )
        tracker = CostTracker(model="gpt-4o-mini")
        tracker.record("t1", input_tokens=10000, output_tokens=5000,
                        latency_ms=500, passed=True)
        alerts = guard.check(tracker)
        assert len(alerts) >= 3

    def test_check_task_directly(self):
        guard = CostGuard(max_cost_per_task=0.000001)
        task = TaskMetrics(task_id="t1", cost=0.01)
        alerts = guard.check_task(task)
        assert len(alerts) == 1

    def test_invalid_max_cost_per_task(self):
        with pytest.raises(ValueError, match="max_cost_per_task"):
            CostGuard(max_cost_per_task=-1.0)

    def test_invalid_max_total_cost(self):
        with pytest.raises(ValueError, match="max_total_cost"):
            CostGuard(max_total_cost=0)

    def test_invalid_max_tokens(self):
        with pytest.raises(ValueError, match="max_tokens_per_task"):
            CostGuard(max_tokens_per_task=-10)

    def test_invalid_max_latency(self):
        with pytest.raises(ValueError, match="max_latency_ms"):
            CostGuard(max_latency_ms=0)

    def test_budget_alert_fields(self):
        alert = BudgetAlert(
            level="error", message="over budget",
            actual=1.5, limit=1.0,
        )
        assert alert.level == "error"
        assert alert.actual > alert.limit


# ─── Test Model Comparison ────────────────────────────────────────


class TestModelComparison:
    def _make_trackers(self) -> tuple[CostTracker, CostTracker, CostTracker]:
        # GPT-4o: accurate but expensive
        t1 = CostTracker(model="gpt-4o", agent_name="gpt4o-agent")
        for i in range(5):
            t1.record(f"t{i}", input_tokens=1000, output_tokens=500,
                       latency_ms=800, passed=True, score=0.95)

        # GPT-4o-mini: cheap but less accurate
        t2 = CostTracker(model="gpt-4o-mini", agent_name="mini-agent")
        for i in range(5):
            t2.record(f"t{i}", input_tokens=1000, output_tokens=500,
                       latency_ms=200, passed=i < 3, score=0.7 if i < 3 else 0.2)

        # Claude: middle ground
        t3 = CostTracker(model="claude-sonnet-4-20250514", agent_name="claude-agent")
        for i in range(5):
            t3.record(f"t{i}", input_tokens=1000, output_tokens=500,
                       latency_ms=500, passed=i < 4, score=0.85)

        return t1, t2, t3

    def test_compare_models(self):
        t1, t2, t3 = self._make_trackers()
        result = compare_models(t1, t2, t3)
        assert len(result.trackers) == 3
        assert result.recommendation != ""

    def test_compare_empty(self):
        result = compare_models()
        assert len(result.trackers) == 0

    def test_to_markdown(self):
        t1, t2, t3 = self._make_trackers()
        result = compare_models(t1, t2, t3)
        md = result.to_markdown()
        assert "| Model |" in md
        assert "gpt-4o" in md
        assert "gpt-4o-mini" in md
        assert "Recommendation" in md

    def test_to_markdown_empty(self):
        result = ComparisonResult()
        assert result.to_markdown() == "No data to compare."

    def test_to_json(self):
        t1, t2, t3 = self._make_trackers()
        result = compare_models(t1, t2, t3)
        data = json.loads(result.to_json())
        assert "comparison" in data
        assert len(data["comparison"]) == 3
        assert "recommendation" in data

    def test_to_csv(self):
        t1, t2, t3 = self._make_trackers()
        result = compare_models(t1, t2, t3)
        csv_str = result.to_csv()
        lines = csv_str.strip().split("\n")
        assert len(lines) == 4  # header + 3 rows
        assert "model" in lines[0]

    def test_to_csv_empty(self):
        result = ComparisonResult()
        assert result.to_csv() == ""

    def test_recommendation_mentions_best(self):
        t1, t2, t3 = self._make_trackers()
        result = compare_models(t1, t2, t3)
        assert "Best efficiency" in result.recommendation

    def test_single_model(self):
        t1 = CostTracker(model="gpt-4o")
        t1.record("t1", input_tokens=100, output_tokens=50,
                   passed=True, latency_ms=500)
        result = compare_models(t1)
        assert len(result.trackers) == 1
        assert "gpt-4o" in result.recommendation
