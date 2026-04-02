"""LitmusAI — Universal Agent Adapter Examples.

Demonstrates how to use LitmusAI with different agent frameworks.
"""

import asyncio

from litmusai import Agent, AgentResponse, AgentStep, TestSuite, ToolCall, evaluate

# ─── Example 1: Simple Function Agent ─────────────────────────────


def simple_agent(task: str) -> str:
    """A basic agent that just echoes the task."""
    return f"I received your task: {task}. Here is my response."


# ─── Example 2: Agent with Rich Response ──────────────────────────


def rich_agent(task: str) -> dict:
    """An agent that returns structured data with cost and tool usage."""
    return {
        "output": f"Processed: {task}",
        "cost": 0.003,
        "tokens_used": 150,
        "input_tokens": 100,
        "output_tokens": 50,
        "model": "gpt-4o",
        "tool_calls": [
            {"name": "search", "arguments": {"query": task}, "result": "found it"},
        ],
        "steps": [
            {"step_number": 1, "action": "search", "thought": "Let me search for this"},
        ],
    }


# ─── Example 3: Agent returning AgentResponse directly ────────────


def direct_response_agent(task: str) -> AgentResponse:
    """An agent that returns a full AgentResponse for maximum control."""
    return AgentResponse(
        output=f"Direct: {task}",
        cost=0.005,
        tokens_used=200,
        model="claude-sonnet-4",
        tool_calls=[
            ToolCall(name="calculator", arguments={"expr": "2+2"}, result="4"),
        ],
        steps=[
            AgentStep(step_number=1, action="think", thought="Processing..."),
            AgentStep(step_number=2, action="calculate", observation="4"),
        ],
    )


# ─── Example 4: HTTP Endpoint Agent ───────────────────────────────

# agent = Agent.from_url(
#     "http://localhost:8000/agent",
#     name="api-agent",
#     headers={"Authorization": "Bearer sk-xxx"},
#     request_field="prompt",
#     response_field="reply",
# )


# ─── Example 5: CLI Agent ─────────────────────────────────────────

# agent = Agent.from_cli("python my_cli_agent.py", name="cli-agent")


# ─── Example 6: Custom Object Agent ───────────────────────────────


class MyCustomAgent:
    """Example of a custom agent class."""

    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    def run(self, task: str) -> str:
        return f"[{self.model}] {task}"


# ─── Main: Run evaluation with multiple adapters ──────────────────


async def main():
    # Create agents from different sources
    agents = {
        "simple": Agent.from_function(simple_agent, name="simple"),
        "rich": Agent.from_function(rich_agent, name="rich", model="gpt-4o"),
        "direct": Agent.from_function(direct_response_agent, name="direct"),
        "custom": Agent.from_callable(MyCustomAgent(), method="run", name="custom"),
        "echo-cli": Agent.from_cli("cat", name="echo-cli"),
    }

    # Create a test suite
    suite = TestSuite(name="adapter-demo", description="Test all adapter types")
    suite.add(task="Say hello", expected_contains=["hello"], name="Greeting")
    suite.add(task="What is 2+2?", name="Math")
    suite.add(task="Write a poem", name="Creative")

    # Evaluate each agent
    for label, agent in agents.items():
        print(f"\n{'='*50}")
        print(f"Testing: {label} ({agent.name})")
        print(f"{'='*50}")

        results = await evaluate(agent, suite, verbose=True)
        print(f"Pass rate: {results.pass_rate:.0%}")
        print(f"Total cost: ${results.total_cost:.4f}")
        print(f"Avg latency: {results.avg_latency_ms:.0f}ms")


if __name__ == "__main__":
    asyncio.run(main())
