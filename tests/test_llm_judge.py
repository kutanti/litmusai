"""Tests for the LLM-as-Judge scoring engine."""

import json
from typing import Any

import pytest

from litmusai.core.agent import AgentResponse, AgentStep, ToolCall
from litmusai.core.suite import TestCase
from litmusai.scorers import (
    AnthropicProvider,
    Completeness,
    Correctness,
    CriterionScore,
    FunctionProvider,
    Hallucination,
    JudgeResult,
    LLMJudge,
    Metric,
    OpenAIProvider,
    PlanQuality,
    Relevance,
    ScoreCache,
    StepEfficiency,
    TaskCompletion,
    ToolCorrectness,
    Toxicity,
    metrics,
)

# ─── Test Metrics ──────────────────────────────────────────────────


class TestMetrics:
    def test_correctness_metric(self):
        m = Correctness()
        assert m.name == "correctness"
        assert "correct" in m.criteria.lower()

    def test_completeness_metric(self):
        m = Completeness()
        assert m.name == "completeness"

    def test_hallucination_metric(self):
        m = Hallucination()
        assert m.name == "hallucination"

    def test_toxicity_metric(self):
        m = Toxicity()
        assert m.name == "toxicity"

    def test_relevance_metric(self):
        m = Relevance()
        assert m.name == "relevance"

    def test_task_completion_metric(self):
        m = TaskCompletion()
        assert m.name == "task_completion"

    def test_tool_correctness_metric(self):
        m = ToolCorrectness()
        assert m.name == "tool_correctness"

    def test_plan_quality_metric(self):
        m = PlanQuality()
        assert m.name == "plan_quality"

    def test_step_efficiency_metric(self):
        m = StepEfficiency()
        assert m.name == "step_efficiency"

    def test_metric_repr(self):
        m = Metric(name="test", criteria="test criteria")
        assert "test" in repr(m)

    def test_metrics_namespace(self):
        assert metrics.Correctness is Correctness
        assert metrics.Hallucination is Hallucination
        assert metrics.Toxicity is Toxicity


# ─── Test Score Cache ──────────────────────────────────────────────


class TestScoreCache:
    def test_cache_miss(self):
        cache = ScoreCache()
        assert cache.get("unknown prompt") is None

    def test_cache_set_get(self):
        cache = ScoreCache()
        cache.set("test prompt", {"raw": "result"})
        assert cache.get("test prompt") == {"raw": "result"}

    def test_cache_size(self):
        cache = ScoreCache()
        assert cache.size == 0
        cache.set("a", {"v": 1})
        cache.set("b", {"v": 2})
        assert cache.size == 2

    def test_cache_clear(self):
        cache = ScoreCache()
        cache.set("a", {"v": 1})
        cache.clear()
        assert cache.size == 0
        assert cache.get("a") is None


# ─── Test CriterionScore ──────────────────────────────────────────


class TestCriterionScore:
    def test_normalized_score(self):
        cs = CriterionScore(name="test", score=7.0, max_score=10.0)
        assert cs.normalized == 0.7

    def test_normalized_zero_max(self):
        cs = CriterionScore(name="test", score=5.0, max_score=0.0)
        assert cs.normalized == 0.0

    def test_with_explanation(self):
        cs = CriterionScore(
            name="correctness",
            score=9.0,
            max_score=10.0,
            explanation="Very accurate response",
        )
        assert cs.explanation == "Very accurate response"


# ─── Test JudgeResult ─────────────────────────────────────────────


class TestJudgeResult:
    def test_normalized_score(self):
        result = JudgeResult(
            scores=[
                CriterionScore(name="a", score=8.0, max_score=10.0),
                CriterionScore(name="b", score=6.0, max_score=10.0),
            ],
            overall_score=14.0,
            max_score=20.0,
        )
        assert result.normalized_score == 0.7

    def test_normalized_zero_max(self):
        result = JudgeResult(overall_score=0.0, max_score=0.0)
        assert result.normalized_score == 0.0

    def test_to_score_result_pass(self):
        result = JudgeResult(
            scores=[
                CriterionScore(
                    name="correctness", score=9.0,
                    max_score=10.0, explanation="Good",
                ),
            ],
            passed=True,
            overall_score=9.0,
            max_score=10.0,
        )
        sr = result.to_score_result()
        assert sr.passed
        assert sr.score == 0.9
        assert "correctness" in sr.reason

    def test_to_score_result_fail(self):
        result = JudgeResult(
            scores=[
                CriterionScore(
                    name="correctness", score=3.0,
                    max_score=10.0, explanation="Wrong",
                ),
            ],
            passed=False,
            overall_score=3.0,
            max_score=10.0,
        )
        sr = result.to_score_result()
        assert not sr.passed
        assert sr.score == 0.3

    def test_to_score_result_details(self):
        result = JudgeResult(
            scores=[
                CriterionScore(name="a", score=8.0, max_score=10.0),
            ],
            overall_score=8.0,
            max_score=10.0,
        )
        sr = result.to_score_result()
        assert sr.details is not None
        assert sr.details["scores"]["a"] == 8.0


# ─── Test LLMJudge ────────────────────────────────────────────────


def mock_llm_response(prompt: str) -> str:
    """Mock LLM that returns valid judge JSON."""
    return json.dumps({
        "scores": {
            "correctness": {
                "score": 8,
                "explanation": "Mostly correct answer",
            },
            "completeness": {
                "score": 7,
                "explanation": "Covers main points",
            },
        }
    })


def mock_llm_high_scores(prompt: str) -> str:
    """Mock LLM that returns high scores."""
    return json.dumps({
        "scores": {
            "correctness": {"score": 10, "explanation": "Perfect"},
            "completeness": {"score": 9, "explanation": "Very thorough"},
        }
    })


def mock_llm_low_scores(prompt: str) -> str:
    """Mock LLM that returns low scores."""
    return json.dumps({
        "scores": {
            "correctness": {"score": 2, "explanation": "Incorrect"},
            "completeness": {"score": 1, "explanation": "Missing everything"},
        }
    })


def mock_llm_invalid_json(prompt: str) -> str:
    """Mock LLM that returns invalid response."""
    return "This is not valid JSON at all"


def mock_llm_custom_criteria(prompt: str) -> str:
    """Mock LLM for custom criteria."""
    # Parse which criteria were asked for
    result: dict[str, Any] = {"scores": {}}
    if "safety" in prompt.lower():
        result["scores"]["safety"] = {"score": 9, "explanation": "Safe"}
    if "creativity" in prompt.lower():
        result["scores"]["creativity"] = {"score": 7, "explanation": "Creative"}
    if not result["scores"]:
        result["scores"]["default"] = {"score": 5, "explanation": "OK"}
    return json.dumps(result)


class TestLLMJudge:
    def _make_case(
        self, task: str = "What is 2+2?", expected: str | None = "4"
    ) -> TestCase:
        return TestCase(id="test_1", name="test", task=task, expected=expected)

    def _make_response(
        self, output: str = "The answer is 4"
    ) -> AgentResponse:
        return AgentResponse(output=output, success=True)

    @pytest.mark.asyncio
    async def test_basic_evaluation(self):
        judge = LLMJudge(
            model="mock",
            criteria={
                "correctness": "Is it correct?",
                "completeness": "Is it complete?",
            },
            provider=FunctionProvider(mock_llm_response),
        )

        result = await judge.evaluate(self._make_case(), self._make_response())
        assert len(result.scores) == 2
        assert result.scores[0].name == "correctness"
        assert result.scores[0].score == 8.0
        assert result.scores[1].name == "completeness"
        assert result.scores[1].score == 7.0

    @pytest.mark.asyncio
    async def test_pass_threshold(self):
        judge = LLMJudge(
            model="mock",
            criteria={"correctness": "Is it correct?", "completeness": "Complete?"},
            provider=FunctionProvider(mock_llm_high_scores),
            pass_threshold=0.8,
        )

        result = await judge.evaluate(self._make_case(), self._make_response())
        assert result.passed  # 19/20 = 0.95 > 0.8

    @pytest.mark.asyncio
    async def test_fail_threshold(self):
        judge = LLMJudge(
            model="mock",
            criteria={"correctness": "Is it correct?", "completeness": "Complete?"},
            provider=FunctionProvider(mock_llm_low_scores),
            pass_threshold=0.5,
        )

        result = await judge.evaluate(self._make_case(), self._make_response())
        assert not result.passed  # 3/20 = 0.15 < 0.5

    @pytest.mark.asyncio
    async def test_failed_agent_response(self):
        judge = LLMJudge(
            model="mock",
            criteria={"correctness": "Is it correct?"},
            provider=FunctionProvider(mock_llm_response),
        )

        failed_response = AgentResponse(
            output="", success=False, error="Agent crashed"
        )
        result = await judge.evaluate(self._make_case(), failed_response)
        assert not result.passed
        assert result.scores[0].score == 1.0  # min score
        assert "crashed" in result.scores[0].explanation.lower()

    @pytest.mark.asyncio
    async def test_invalid_llm_response(self):
        judge = LLMJudge(
            model="mock",
            criteria={"correctness": "Is it correct?"},
            provider=FunctionProvider(mock_llm_invalid_json),
        )

        result = await judge.evaluate(self._make_case(), self._make_response())
        assert not result.passed

    @pytest.mark.asyncio
    async def test_caching(self):
        call_count = 0

        def counting_mock(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return mock_llm_response(prompt)

        judge = LLMJudge(
            model="mock",
            criteria={"correctness": "Correct?", "completeness": "Complete?"},
            provider=FunctionProvider(counting_mock),
            cache=True,
        )

        case = self._make_case()
        response = self._make_response()

        # First call
        await judge.evaluate(case, response)
        assert call_count == 1

        # Second call — should use cache
        await judge.evaluate(case, response)
        assert call_count == 1  # Not incremented

    @pytest.mark.asyncio
    async def test_cache_disabled(self):
        call_count = 0

        def counting_mock(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return mock_llm_response(prompt)

        judge = LLMJudge(
            model="mock",
            criteria={"correctness": "Correct?", "completeness": "Complete?"},
            provider=FunctionProvider(counting_mock),
            cache=False,
        )

        case = self._make_case()
        response = self._make_response()

        await judge.evaluate(case, response)
        await judge.evaluate(case, response)
        assert call_count == 2  # Called twice, no cache

    @pytest.mark.asyncio
    async def test_with_metrics(self):
        judge = LLMJudge(
            model="mock",
            metric_list=[Correctness(), Completeness()],
            provider=FunctionProvider(mock_llm_response),
        )

        assert "correctness" in judge.criteria
        assert "completeness" in judge.criteria

    @pytest.mark.asyncio
    async def test_default_criteria(self):
        judge = LLMJudge(
            model="mock",
            provider=FunctionProvider(mock_llm_response),
        )

        # Should have default criteria
        assert len(judge.criteria) > 0
        assert "correctness" in judge.criteria

    @pytest.mark.asyncio
    async def test_with_tool_calls(self):
        judge = LLMJudge(
            model="mock",
            criteria={"correctness": "Correct?", "completeness": "Complete?"},
            provider=FunctionProvider(mock_llm_response),
        )

        response = AgentResponse(
            output="Found the answer",
            tool_calls=[
                ToolCall(name="search", arguments={"q": "test"}, result="found"),
            ],
        )

        result = await judge.evaluate(self._make_case(), response)
        assert result.scores[0].score == 8.0

    @pytest.mark.asyncio
    async def test_with_steps(self):
        judge = LLMJudge(
            model="mock",
            criteria={"correctness": "Correct?", "completeness": "Complete?"},
            provider=FunctionProvider(mock_llm_response),
        )

        response = AgentResponse(
            output="Done",
            steps=[
                AgentStep(step_number=1, action="think", thought="hmm"),
                AgentStep(step_number=2, action="answer", observation="4"),
            ],
        )

        result = await judge.evaluate(self._make_case(), response)
        assert len(result.scores) == 2

    @pytest.mark.asyncio
    async def test_score_async(self):
        judge = LLMJudge(
            model="mock",
            criteria={"correctness": "Correct?", "completeness": "Complete?"},
            provider=FunctionProvider(mock_llm_response),
        )

        sr = await judge.score_async(self._make_case(), self._make_response())
        assert sr.score == 0.75  # (8+7)/(10+10)
        assert sr.passed

    @pytest.mark.asyncio
    async def test_score_range(self):
        def mock_custom_range(prompt: str) -> str:
            return json.dumps({
                "scores": {
                    "quality": {"score": 4, "explanation": "Good"},
                }
            })

        judge = LLMJudge(
            model="mock",
            criteria={"quality": "How good?"},
            score_range=(1, 5),
            provider=FunctionProvider(mock_custom_range),
        )

        result = await judge.evaluate(self._make_case(), self._make_response())
        assert result.scores[0].max_score == 5.0
        assert result.scores[0].score == 4.0

    @pytest.mark.asyncio
    async def test_score_clamping(self):
        def mock_out_of_range(prompt: str) -> str:
            return json.dumps({
                "scores": {
                    "correctness": {"score": 15, "explanation": "Over max"},
                    "completeness": {"score": -5, "explanation": "Under min"},
                }
            })

        judge = LLMJudge(
            model="mock",
            criteria={"correctness": "Correct?", "completeness": "Complete?"},
            score_range=(1, 10),
            provider=FunctionProvider(mock_out_of_range),
        )

        result = await judge.evaluate(self._make_case(), self._make_response())
        assert result.scores[0].score == 10.0  # Clamped to max
        assert result.scores[1].score == 1.0   # Clamped to min


# ─── Test Providers ────────────────────────────────────────────────


class TestProviders:
    def test_openai_provider_creation(self):
        provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
        assert provider.model == "gpt-4o"

    def test_anthropic_provider_creation(self):
        provider = AnthropicProvider(model="claude-sonnet-4-20250514", api_key="test-key")
        assert provider.model == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_function_provider_sync(self):
        provider = FunctionProvider(lambda p: f"echo: {p}")
        result = await provider.complete("hello")
        assert result == "echo: hello"

    @pytest.mark.asyncio
    async def test_function_provider_async(self):
        async def async_fn(p: str) -> str:
            return f"async: {p}"

        provider = FunctionProvider(async_fn)
        result = await provider.complete("hello")
        assert result == "async: hello"
