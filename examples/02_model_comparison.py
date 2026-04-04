"""Example 2: Compare multiple models side-by-side.

Uses LiteLLM or OpenAI-compatible APIs to compare models.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/02_model_comparison.py

    # Or with a custom base URL (LiteLLM, Ollama, etc.):
    export OPENAI_BASE_URL=http://localhost:4000
    export OPENAI_API_KEY=sk-anything
    python examples/02_model_comparison.py
"""

import asyncio
import os

from litmusai import Agent, TestSuite, compare
from litmusai.assertions import Contains, Numeric
from litmusai.core.suite import TestCase


# ─── Build test suite ─────────────────────────────────────────────

suite = TestSuite(name="model-comparison")

suite.add_case(TestCase(
    id="math_001",
    name="Arithmetic",
    task="What is 15 * 23? Give just the number.",
    assertions=[Numeric(345)],
))

suite.add_case(TestCase(
    id="geo_001",
    name="Geography",
    task="What is the largest country by area?",
    assertions=[Contains("Russia")],
))

suite.add_case(TestCase(
    id="code_001",
    name="Code generation",
    task="Write a Python function to check if a number is prime.",
    assertions=[
        Contains("def"),
        Contains("prime"),
    ],
))


# ─── Create agents for different models ──────────────────────────

def make_agent(model: str) -> Agent:
    """Create an agent using OpenAI-compatible API."""
    return Agent.from_openai_chat(
        model=model,
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY", ""),
    )


async def main():
    models = ["gpt-4o", "gpt-4.1"]

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Set OPENAI_API_KEY to run this example")
        print("   export OPENAI_API_KEY=sk-...")
        return

    agents = [make_agent(m) for m in models]
    comparison = await compare(agents, suite)
    print(comparison)


if __name__ == "__main__":
    asyncio.run(main())
