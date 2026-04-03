"""Example 1: Basic evaluation with assertions.

Run any function as an agent and test it with assertions.

Usage:
    python examples/01_basic_eval.py
"""

import asyncio

from litmusai import Agent, evaluate
from litmusai.assertions import Contains, NotContains, Numeric
from litmusai.core.suite import TestCase, TestSuite


# ─── Define your agent ───────────────────────────────────────────
# Any async or sync function works. Here's a simple one:


async def my_agent(task: str) -> str:
    """A mock agent that answers questions."""
    task_lower = task.lower()
    if "capital" in task_lower and "france" in task_lower:
        return "The capital of France is Paris."
    if "2+2" in task_lower or "2 + 2" in task_lower:
        return "2 + 2 = 4"
    if "python" in task_lower:
        return "Python is a programming language created by Guido van Rossum."
    return "I don't know the answer to that."


# ─── Wrap it as an Agent ─────────────────────────────────────────

agent = Agent.from_function(my_agent, name="my-agent")


# ─── Build a test suite ──────────────────────────────────────────

suite = TestSuite(name="basic-knowledge")

suite.add_case(TestCase(
    id="geo_001",
    name="Capital of France",
    task="What is the capital of France?",
    assertions=[
        Contains("Paris"),
        NotContains(["London"]),
    ],
))

suite.add_case(TestCase(
    id="math_001",
    name="Simple addition",
    task="What is 2 + 2?",
    assertions=[Numeric(4)],
))

suite.add_case(TestCase(
    id="prog_001",
    name="Python language",
    task="Tell me about Python programming language",
    assertions=[
        Contains("Python"),
        Contains("programming"),
    ],
))


# ─── Run evaluation ──────────────────────────────────────────────

async def main():
    results = await evaluate(agent, suite)
    print(f"\n{results.summary()}")
    print(f"Pass rate: {results.pass_rate:.0%}")

    # Access individual results
    for r in results.results:
        status = "✅" if r.passed else "❌"
        print(f"  {status} {r.case.name}: {r.score.reason}")


if __name__ == "__main__":
    asyncio.run(main())
