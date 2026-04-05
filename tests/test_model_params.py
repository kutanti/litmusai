"""Tests for model params logging and standardization (#26)."""

from __future__ import annotations

import pytest

from litmusai.core.agent import Agent, AgentResponse


class TestModelParams:
    def test_from_openai_chat_stores_params(self):
        agent = Agent.from_openai_chat(
            model="gpt-4o",
            api_key="test",
            temperature=0.0,
            max_tokens=512,
            seed=42,
        )
        assert agent.model_params["temperature"] == 0.0
        assert agent.model_params["max_tokens"] == 512
        assert agent.model_params["seed"] == 42

    def test_from_openai_chat_defaults(self):
        agent = Agent.from_openai_chat(model="gpt-4o", api_key="test")
        assert agent.model_params["max_tokens"] == 1024
        assert "temperature" not in agent.model_params
        assert "seed" not in agent.model_params

    def test_from_azure_stores_params(self):
        agent = Agent.from_azure(
            resource="r", deployment="d", api_key="test",
            temperature=0.5, max_tokens=256,
        )
        assert agent.model_params["temperature"] == 0.5
        assert agent.model_params["max_tokens"] == 256

    def test_manual_agent_model_params(self):
        async def fn(task, **kw):
            return AgentResponse(output="hi", model="test")

        agent = Agent(
            fn=fn, name="test", model="test",
            model_params={"temperature": 0.7, "custom": True},
        )
        assert agent.model_params["temperature"] == 0.7
        assert agent.model_params["custom"] is True

    def test_model_params_default_empty(self):
        async def fn(task, **kw):
            return AgentResponse(output="hi", model="test")

        agent = Agent(fn=fn, name="test")
        assert agent.model_params == {}


class TestEvalResultsConfig:
    @pytest.mark.asyncio
    async def test_config_contains_model_params(self):
        from litmusai import TestCase, TestSuite, evaluate

        async def fn(task, **kw):
            return AgentResponse(output="42", model="test")

        agent = Agent(
            fn=fn, name="test", model="test",
            model_params={"temperature": 0.0, "seed": 42},
        )

        suite = TestSuite(name="test")
        suite.add_case(TestCase(
            id="q1", name="Q", task="What?",
            expected_contains=["42"],
        ))

        results = await evaluate(agent, suite, verbose=False)
        assert results.config["model"] == "test"
        assert results.config["model_params"]["temperature"] == 0.0
        assert results.config["model_params"]["seed"] == 42

    @pytest.mark.asyncio
    async def test_config_model_params_is_copy(self):
        """Config stores a defensive copy, not a reference."""
        from litmusai import TestCase, TestSuite, evaluate

        async def fn(task, **kw):
            return AgentResponse(output="42", model="test")

        params = {"temperature": 0.0}
        agent = Agent(
            fn=fn, name="test", model="test",
            model_params=params,
        )

        suite = TestSuite(name="test")
        suite.add_case(TestCase(
            id="q1", name="Q", task="What?",
            expected_contains=["42"],
        ))

        results = await evaluate(agent, suite, verbose=False)
        # Mutating the original dict should not affect results
        params["temperature"] = 999
        assert results.config["model_params"]["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_config_in_to_dict(self):
        from litmusai import TestCase, TestSuite, evaluate

        async def fn(task, **kw):
            return AgentResponse(output="42", model="test")

        agent = Agent(
            fn=fn, name="test", model="test",
            model_params={"temperature": 0.0},
        )

        suite = TestSuite(name="test")
        suite.add_case(TestCase(
            id="q1", name="Q", task="What?",
            expected_contains=["42"],
        ))

        results = await evaluate(agent, suite, verbose=False)
        d = results.to_dict()
        assert d["config"]["model"] == "test"
        assert d["config"]["model_params"]["temperature"] == 0.0


class TestBenchmarkProfile:
    def test_benchmark_has_temperature_zero(self):
        from litmusai import get_profile

        p = get_profile("benchmark")
        assert p.temperature == 0.0
        assert p.seed == 42

    def test_benchmark_get_model_params(self):
        from litmusai import get_profile

        p = get_profile("benchmark")
        params = p.get_model_params()
        assert params["temperature"] == 0.0
        assert params["seed"] == 42

    def test_quick_no_model_params(self):
        from litmusai import get_profile

        p = get_profile("quick")
        params = p.get_model_params()
        assert params == {}

    def test_profiles_display_temperature(self):
        from click.testing import CliRunner

        from litmusai.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["profiles"])
        assert result.exit_code == 0
        assert "temperature=0.0" in result.output
        assert "seed=42" in result.output


class TestSeedParam:
    def test_seed_stored_on_agent(self):
        agent = Agent.from_openai_chat(
            model="gpt-4o", api_key="test", seed=123,
        )
        assert agent.model_params["seed"] == 123

    def test_no_seed_by_default(self):
        agent = Agent.from_openai_chat(model="gpt-4o", api_key="test")
        assert "seed" not in agent.model_params

    def test_extra_body_overrides_logged(self):
        agent = Agent.from_openai_chat(
            model="gpt-4o", api_key="test",
            temperature=0.5,
            extra_body={"temperature": 0.0, "seed": 99},
        )
        # extra_body overrides should be reflected in model_params
        assert agent.model_params["temperature"] == 0.0
        assert agent.model_params["seed"] == 99
