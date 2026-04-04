"""Example 5: JSON output validation.

Test that your agent returns valid structured data.

Usage:
    python examples/05_json_validation.py
"""

import asyncio
import json

from litmusai import Agent, evaluate
from litmusai.assertions import Contains, JsonPath, JsonSchema, JsonValid
from litmusai.core.suite import TestCase, TestSuite


# ─── An agent that returns JSON ──────────────────────────────────


async def json_agent(task: str) -> str:
    """An agent that extracts structured data."""
    if "weather" in task.lower():
        return json.dumps({
            "location": "San Francisco",
            "temperature": 68,
            "unit": "fahrenheit",
            "conditions": "sunny",
        })
    if "user" in task.lower():
        return json.dumps({
            "name": "Alice",
            "age": 30,
            "email": "alice@example.com",
            "roles": ["admin", "user"],
        })
    return json.dumps({"error": "Unknown request"})


agent = Agent.from_function(json_agent, name="json-agent")


# ─── Test JSON structure ─────────────────────────────────────────

suite = TestSuite(name="json-validation")

# Test that output is valid JSON with specific fields
suite.add_case(TestCase(
    id="weather_001",
    name="Weather data extraction",
    task="Get the current weather in San Francisco",
    assertions=[
        JsonValid(),  # Must be valid JSON
        JsonPath("$.location", expected="San Francisco"),
        JsonPath("$.unit", expected="fahrenheit"),
    ],
))

# Test against JSON Schema
suite.add_case(TestCase(
    id="user_001",
    name="User profile schema",
    task="Get user profile for Alice",
    assertions=[
        JsonSchema({
            "type": "object",
            "required": ["name", "age", "email"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
                "email": {"type": "string"},
            },
        }),
        Contains("alice"),
    ],
))


# ─── Run ──────────────────────────────────────────────────────────

async def main():
    results = await evaluate(agent, suite)
    print(f"\n{results.summary()}")

    for r in results.results:
        status = "✅" if r.passed else "❌"
        print(f"  {status} {r.case.name}: {r.score.reason}")


if __name__ == "__main__":
    asyncio.run(main())
