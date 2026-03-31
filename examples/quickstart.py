"""LitmusAI Quick Start Example."""

import asyncio
from litmusai import Agent, TestSuite, evaluate


# Define a simple agent
def my_agent(task: str) -> str:
    """A simple echo agent for demonstration."""
    return f"I received your task: {task}. Here is my response with the answer."


async def main():
    # Create an agent
    agent = Agent.from_function(my_agent, name="demo-agent", model="echo")

    # Create a custom test suite
    suite = TestSuite(name="demo", description="Demo test suite")
    suite.add(task="Say hello", expected_contains=["hello"], name="Greeting test")
    suite.add(task="What is 2+2?", expected_contains=["4"], name="Math test")
    suite.add(task="Write code", expected_contains=["def"], name="Code test")

    # Run evaluation
    results = await evaluate(agent, suite)

    # Print results
    print(f"\nPass rate: {results.pass_rate:.0%}")
    print(f"Total cost: ${results.total_cost:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
