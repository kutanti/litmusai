"""Unavailable prices must stay unknown through evaluation and budget checks."""

import csv
import json
import xml.etree.ElementTree as ET
from copy import deepcopy

import pytest
from click.testing import CliRunner

from litmusai import Agent, AgentResponse
from litmusai import TestCase as Case
from litmusai import TestSuite as Suite
from litmusai.benchmarks import CostGuard, CostTracker, ModelPricing, compare_models
from litmusai.ci import compare_with_baseline, format_report, format_table
from litmusai.cli.main import cli
from litmusai.conversation import ConversationRunner, MultiTurnCase, Step
from litmusai.core.reporter import Reporter
from litmusai.core.runner import multi_evaluate
from litmusai.core.scorer import ScoreResult
from litmusai.exports import to_csv, to_junit_xml
from litmusai.reports import render_html
from litmusai.results import diff_results, load_results, normalize_results
from litmusai.scoring import aggregate_vectors, build_score_vector

MODEL = "unregistered-live-alias"
USAGE = {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500}


@pytest.fixture
def register_rate(monkeypatch):
    from litmusai.benchmarks import _PRICING_DB

    def register(input_rate, output_rate):
        monkeypatch.setitem(_PRICING_DB, MODEL, ModelPricing(MODEL, input_rate, output_rate))

    return register


def make_adapter(adapter):
    if adapter == "azure":
        return Agent.from_azure(resource="test", deployment=MODEL, api_key="test")
    return Agent.from_openai_chat(model=MODEL, base_url="http://proxy.test/v1", api_key="test")


def completion(usage=USAGE):
    return {"choices": [{"message": {"content": "ok"}}], "model": MODEL,
            "usage": usage, "copilot_usage": {"total_nano_aiu": 123456789}}


@pytest.mark.parametrize("adapter", ["openai", "azure"])
@pytest.mark.parametrize("rate,expected", [(None, None), ((3, 15), .0105), ((0, 0), 0)])
async def test_provider_cost_requires_pricing(adapter, rate, expected, register_rate, httpx_mock):
    if rate is not None:
        register_rate(*rate)
    httpx_mock.add_response(json=completion())
    response = await make_adapter(adapter).run("hello")
    assert response.success
    assert (response.input_tokens, response.output_tokens) == (1000, 500)
    assert response.total_tokens == 1500
    assert response.cost == (pytest.approx(expected) if expected is not None else None)


@pytest.mark.parametrize("adapter", ["openai", "azure"])
@pytest.mark.parametrize("usage", [
    None, {}, {"total_tokens": 1500}, {"prompt_tokens": 1000}, {"completion_tokens": 500},
    {"prompt_tokens": None, "completion_tokens": 0},
    {"prompt_tokens": -1, "completion_tokens": 0},
    {"prompt_tokens": "1000", "completion_tokens": 500},
    {"prompt_tokens": True, "completion_tokens": 500},
])
async def test_missing_or_invalid_usage_cannot_be_priced(
    adapter, usage, register_rate, httpx_mock,
):
    register_rate(3, 15)
    httpx_mock.add_response(json=completion(usage))
    response = await make_adapter(adapter).run("hello")
    assert response.success
    assert response.cost is None


async def test_explicit_zero_tokens_are_a_known_zero(register_rate, httpx_mock):
    register_rate(3, 15)
    httpx_mock.add_response(json=completion({"prompt_tokens": 0, "completion_tokens": 0}))
    response = await make_adapter("openai").run("hello")
    assert response.success and response.cost == 0


@pytest.mark.parametrize("adapter", ["openai", "azure"])
async def test_failed_provider_call_has_unknown_cost(adapter, httpx_mock):
    httpx_mock.add_response(status_code=500, json={"error": "failure"})
    response = await make_adapter(adapter).run("hello")
    assert not response.success and response.cost is None


@pytest.mark.parametrize("value,expected", [(None, None), (0, 0), (.02, .02),
                                             (float("nan"), None), (-1, None), (True, None)])
async def test_custom_response_cost_validation(value, expected, httpx_mock):
    data = {"output": "ok", "cost": value}
    if value != value:  # HTTP JSON cannot encode NaN.
        data["cost"] = "NaN"
    httpx_mock.add_response(json=data)
    responses = [AgentResponse(output="ok", cost=value),
                 await Agent.from_function(lambda _: data).run("hello"),
                 await Agent.from_url("http://proxy.test/agent").run("hello")]
    assert all(r.success and r.cost == expected for r in responses)


async def test_omitted_function_cost_is_unknown_but_explicit_zero_is_free():
    for result in ("ok", {"output": "ok"}, AgentResponse(output="ok")):
        response = await Agent.from_function(lambda _, result=result: result).run("hello")
        assert response.cost is None
    assert (await Agent.from_function(lambda _: {"output": "ok", "cost": 0}).run("hello")).cost == 0


async def test_mixed_costs_remain_unknown_through_repetitions_and_all_reports(tmp_path, capsys):
    suite = Suite("mixed", [Case(id="known", task="known"), Case(id="unknown", task="unknown")])
    agent = Agent.from_function(lambda task: AgentResponse(
        output="ok", cost=.02 if task == "known" else None))
    result = await multi_evaluate(agent, suite, runs=2, verbose=False)
    data = result.to_dict()
    assert result.total_cost is None and result.combined.total_cost is None
    assert "Unknown" in result.summary()
    assert data["summary"]["total_cost"] is None
    assert all(run["summary"]["total_cost"] is None for run in data["run_results"])
    assert [row["cost"] for row in data["results"]] == [.02, None, .02, None]
    assert data["dimensions"]["cost"] is None
    assert load_results(result.save(tmp_path / "result.json")) == data
    assert "Unknown" in Reporter.to_markdown(result.combined)
    assert "Unknown" in format_report(data)
    assert "Unknown" in render_html(data, tmp_path / "report.html").read_text(encoding="utf-8")
    format_table(data, show_dimensions=True)
    Reporter.to_table(result.combined)
    assert "Unknown" in capsys.readouterr().out
    with to_csv(data, tmp_path / "result.csv").open(encoding="utf-8", newline="") as file:
        assert [row["cost"] for row in csv.DictReader(file)] == ["0.02", "unknown"] * 2
    xml = ET.parse(to_junit_xml(data, tmp_path / "result.xml"))
    total_cost = xml.find(".//testsuite/properties/property[@name='total_cost']")
    assert total_cost.get("value") == "unknown"
    assert [p.get("value") for p in xml.findall(
        ".//testcase/properties/property[@name='cost']")] == ["0.02", "unknown"] * 2
    history = CliRunner().invoke(cli, ["history", "--log-dir", str(tmp_path)])
    assert history.exit_code == 0, history.output
    assert "Unknown" in history.output


@pytest.mark.parametrize("cost,expected", [(None, False), (0, True), (.02, False)])
@pytest.mark.parametrize("runs", [1, 2])
def test_cli_budget_requires_complete_costs(tmp_path, monkeypatch, cost, expected, runs):
    agent = Agent.from_function(lambda _: AgentResponse(output="ok", cost=cost))
    monkeypatch.setattr("litmusai.ci.load_agent", lambda _: agent)
    suite = tmp_path / "suite.yaml"
    suite.write_text('name: costs\ncases:\n  - id: one\n    task: hello\n', encoding="utf-8")
    result = CliRunner().invoke(cli, [
        "run", "-s", str(suite), "-a", "local:agent", "--format", "json", "--budget", ".01",
        "--runs", str(runs),
    ])
    payload = json.loads(result.stdout)
    assert payload["success"] is expected
    assert payload["budget_check"]["passed"] is expected
    assert result.exit_code == (0 if expected else 1), result.output
    if cost is None:
        assert "unknown" in payload["budget_check"]["reason"]
        assert "register model pricing" in result.stderr


def test_unknown_cost_is_excluded_from_scores_and_comparisons():
    score = ScoreResult(passed=False, score=.4, reason="partial")
    vectors = [build_score_vector(score_result=score, response=AgentResponse(output="ok", cost=c))
               for c in (None, 0, .05)]
    assert vectors[0].cost is None and "excluded" in vectors[0].details["cost"]
    assert vectors[1].cost == 1
    assert aggregate_vectors(vectors).cost is None
    base = {"summary": {"total_cost": .02}, "results": [{"case_id": "a", "cost": .02}]}
    current = {"summary": {"total_cost": None}, "results": [{"case_id": "a", "cost": None}]}
    for left, right in ((base, current), (current, base), (current, current)):
        comparison = compare_with_baseline(left, right)
        assert comparison["cost"]["delta"] is None
        assert comparison["cost"]["comparable"] is False
        assert diff_results(left, right).cases[0].cost_change_pct is None
        assert "Unavailable" in format_report(left, right)


def test_legacy_numbers_remain_readable_and_null_costs_override_stale_totals():
    legacy = {"summary": {"total_cost": 0}, "results": [{"case_id": "a", "cost": 0}]}
    assert normalize_results(legacy) == legacy
    mixed = deepcopy(legacy)
    mixed["results"].append({"case_id": "b", "cost": None})
    original = deepcopy(mixed)
    assert normalize_results(mixed)["summary"]["total_cost"] is None
    assert mixed == original


async def test_conversation_total_is_unknown_if_any_step_cost_is_unknown():
    costs = iter([.01, None, .01])
    agent = Agent.from_function(lambda task, **kw: AgentResponse(output="ok", cost=next(costs)))
    result = await ConversationRunner(agent).run(MultiTurnCase(
        id="costs", name="costs", steps=[Step(user="one"), Step(user="two"), Step(user="three")]))
    assert result.total_cost is None
    assert result.to_dict()["total_cost"] is None
    assert "Unknown" in result.summary()


def test_cost_tracker_and_guard_do_not_treat_unpriced_models_as_free():
    tracker = CostTracker(model=MODEL)
    task = tracker.record("unknown", input_tokens=1000, output_tokens=500, passed=True)
    assert task.cost is None and tracker.total_cost is None
    assert tracker.avg_cost_per_task is None and tracker.cost_per_pass is None
    assert tracker.efficiency_score is None
    alerts = CostGuard(max_cost_per_task=.1, max_total_cost=1).check(tracker)
    assert len(alerts) == 2 and all(a.level == "error" and a.actual is None for a in alerts)
    assert CostGuard(max_tokens_per_task=2000).check(tracker) == []
    comparison = compare_models(tracker)
    assert "Unknown" in comparison.to_markdown()
    assert "unknown" in comparison.to_csv()
    assert json.loads(comparison.to_json())["comparison"][0]["total_cost"] is None
    assert "Best efficiency" not in comparison.recommendation
