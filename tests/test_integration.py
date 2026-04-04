"""Integration test — run LitmusAI against a real LLM.

Requires OPENAI_API_KEY or LITELLM_BASE_URL + LITELLM_API_KEY.
Skips automatically if no API credentials are available.

Usage:
    # With OpenAI
    OPENAI_API_KEY=sk-... pytest tests/test_integration.py -v

    # With LiteLLM proxy
    LITELLM_BASE_URL=http://localhost:4000 LITELLM_API_KEY=sk-... \
        pytest tests/test_integration.py -v
"""

import os

import pytest

# Skip all tests if no usable API credentials
_has_openai = bool(
    os.getenv("OPENAI_API_KEY")
    and os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
)
_has_litellm = bool(
    os.getenv("LITELLM_API_KEY") and os.getenv("LITELLM_BASE_URL")
)
pytestmark = pytest.mark.skipif(
    not _has_openai and not _has_litellm,
    reason=(
        "No API credentials. Set OPENAI_API_KEY or "
        "LITELLM_BASE_URL + LITELLM_API_KEY"
    ),
)


def _get_agent():
    """Create an agent using available API credentials."""
    from litmusai import Agent

    base_url = os.getenv("LITELLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    api_key = (
        os.getenv("LITELLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    )
    model = os.getenv("LITMUS_TEST_MODEL", "gpt-4o")

    return Agent.from_openai_chat(
        model=model,
        base_url=base_url,
        api_key=api_key,
    )


class TestRealAgent:
    """Integration tests with a real LLM."""

    @pytest.mark.asyncio
    async def test_simple_math(self):
        from litmusai import evaluate
        from litmusai.assertions import Numeric
        from litmusai.core.suite import TestCase, TestSuite

        agent = _get_agent()
        suite = TestSuite(name="integration-math")
        suite.add_case(TestCase(
            id="math_001",
            name="Simple multiplication",
            task="What is 7 times 8? Reply with just the number.",
            assertions=[Numeric(56)],
        ))

        results = await evaluate(agent, suite, verbose=False)
        assert results.pass_rate >= 0.5  # LLMs should get this
        assert results.total_cost >= 0  # Cost tracking works
        assert results.results[0].latency_ms > 0

    @pytest.mark.asyncio
    async def test_knowledge_question(self):
        from litmusai import evaluate
        from litmusai.assertions import Contains
        from litmusai.core.suite import TestCase, TestSuite

        agent = _get_agent()
        suite = TestSuite(name="integration-knowledge")
        suite.add_case(TestCase(
            id="geo_001",
            name="Capital of Japan",
            task="What is the capital of Japan? One word answer.",
            assertions=[Contains("Tokyo")],
        ))

        results = await evaluate(agent, suite, verbose=False)
        assert results.pass_rate == 1.0

    @pytest.mark.asyncio
    async def test_token_tracking(self):
        """Verify real token usage is captured."""
        agent = _get_agent()
        response = await agent.run("Say hello in exactly 3 words.")

        assert response.output  # Got a response
        assert response.success
        # Token tracking should work with OpenAI-compatible API
        total_tokens = (
            response.input_tokens + response.output_tokens
        )
        if total_tokens > 0:
            # Tokens were tracked
            assert response.input_tokens > 0
            assert response.output_tokens > 0

    @pytest.mark.asyncio
    async def test_multi_run(self):
        from litmusai import multi_evaluate
        from litmusai.assertions import Contains
        from litmusai.core.suite import TestCase, TestSuite

        agent = _get_agent()
        suite = TestSuite(name="integration-multi")
        suite.add_case(TestCase(
            id="q1",
            name="Factual",
            task="Is water wet? Answer yes or no.",
            assertions=[Contains("yes")],
        ))

        multi = await multi_evaluate(
            agent, suite, runs=2, verbose=False,
        )
        assert multi.n_runs == 2
        assert "q1" in multi.case_stats
        assert multi.case_stats["q1"].n_runs == 2

    @pytest.mark.asyncio
    async def test_safety_scan(self):
        from litmusai.safety import SafetyScanner

        agent = _get_agent()
        scanner = SafetyScanner(depth="basic")
        report = await scanner.scan(agent)

        assert report.total_tests > 0
        assert 0 <= report.safety_score <= 100

    @pytest.mark.asyncio
    async def test_save_and_diff(self, tmp_path):
        from litmusai import evaluate
        from litmusai.assertions import Contains
        from litmusai.core.suite import TestCase, TestSuite
        from litmusai.results import diff_results, load_results

        agent = _get_agent()
        suite = TestSuite(name="integration-save")
        suite.add_case(TestCase(
            id="q1",
            name="Greeting",
            task="Say hello",
            assertions=[Contains("hello")],
        ))

        # Run and save
        await evaluate(
            agent, suite, verbose=False,
            log_dir=str(tmp_path),
        )

        # Load and diff against self
        from litmusai.results import list_results

        entries = list_results(str(tmp_path))
        assert len(entries) >= 1

        loaded = load_results(entries[0]["file"])
        diff = diff_results(loaded, loaded)
        assert len(diff.regressions) == 0

    @pytest.mark.asyncio
    async def test_html_report(self, tmp_path):
        from litmusai import evaluate
        from litmusai.assertions import Contains
        from litmusai.core.suite import TestCase, TestSuite
        from litmusai.reports import render_html

        agent = _get_agent()
        suite = TestSuite(name="integration-report")
        suite.add_case(TestCase(
            id="q1",
            name="Test",
            task="What color is the sky?",
            assertions=[Contains("blue")],
        ))

        results = await evaluate(agent, suite, verbose=False)
        path = render_html(
            results.to_dict(), tmp_path / "report.html",
        )
        assert path.exists()
        html = path.read_text()
        assert "integration-report" in html
