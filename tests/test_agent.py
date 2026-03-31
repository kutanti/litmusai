"""Tests for the Agent class."""

import pytest

from litmusai.core.agent import Agent


def simple_agent(task: str) -> str:
    return f"Response to: {task}"


async def async_agent(task: str) -> str:
    return f"Async response to: {task}"


def dict_agent(task: str) -> dict:
    return {"output": f"Dict response to: {task}", "cost": 0.01, "tokens_used": 100}


class TestAgent:
    @pytest.mark.asyncio
    async def test_from_function(self):
        agent = Agent.from_function(simple_agent, name="test")
        response = await agent.run("hello")
        assert response.success
        assert "hello" in response.output
        assert response.latency_ms > 0

    @pytest.mark.asyncio
    async def test_async_agent(self):
        agent = Agent.from_function(async_agent, name="async-test")
        response = await agent.run("hello")
        assert response.success
        assert "Async" in response.output

    @pytest.mark.asyncio
    async def test_dict_response(self):
        agent = Agent.from_function(dict_agent, name="dict-test")
        response = await agent.run("hello")
        assert response.success
        assert response.cost == 0.01
        assert response.tokens_used == 100

    @pytest.mark.asyncio
    async def test_error_handling(self):
        def failing_agent(task: str) -> str:
            raise ValueError("Intentional error")

        agent = Agent.from_function(failing_agent, name="fail-test")
        response = await agent.run("hello")
        assert not response.success
        assert "Intentional error" in response.error

    def test_repr(self):
        agent = Agent.from_function(simple_agent, name="test", model="gpt-4")
        assert "test" in repr(agent)
        assert "gpt-4" in repr(agent)
