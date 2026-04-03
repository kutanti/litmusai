"""Example 4: Multi-run evaluation with flaky test detection.

Run each test multiple times to find unreliable behavior.

Usage:
    python examples/04_multi_run.py
"""

import asyncio
import random

from litmusai import Agent, multi_evaluate
from litmusai.assertions import Contains, Numeric
from litmusai.core.suite import TestCase, TestSuite


# ─── A deliberately flaky agent ──────────────────────────────────
# Sometimes it gets the answer right, sometimes not.


async def flaky_agent(task: str) -> str:
    """An agent that's unreliable on some questions."""
    if "2+2" in task.lower() or "2 + 2" in task.lower():
        # Always correct on easy math
        return "The answer is 4."

    if "capital" in task.lower():
        # 70% chance of correct answer
        if random.random() < 0.7:
            return "The capital of France is Paris."
        return "The capital of France is Lyon."

    return "I'm not sure about that."


agent = Agent.from_function(flaky_agent, name="flaky-agent")


# ─── Build suite ─────────────────────────────────────────────────

suite = TestSuite(name="reliability-test")

suite.add_case(TestCase(
    id="stable_001",
    name="Easy math (should be stable)",
    task="What is 2 + 2?",
    assertions=[Numeric(4)],
))

suite.add_case(TestCase(
    id="flaky_001",
    name="Geography (might be flaky)",
    task="What is the capital of France?",
    assertions=[Contains("Paris")],
))


# ─── Run multiple times ──────────────────────────────────────────

async def main():
    multi = await multi_evaluate(
        agent, suite,
        runs=5,       # Run each test 5 times
        verbose=True,
    )

    print(f"\n📊 {multi.summary()}")
    print(f"\nPer-case reliability:")

    for stats in multi.case_stats.values():
        print(
            f"  {stats.case_name}: "
            f"{stats.pass_rate:.0%} pass rate, "
            f"reliability={stats.reliability}"
        )

    if multi.flaky_tests:
        print(f"\n⚠️  Flaky tests detected:")
        for ft in multi.flaky_tests:
            print(
                f"  - {ft.case_name}: "
                f"{ft.n_passed}/{ft.n_runs} passed"
            )


if __name__ == "__main__":
    asyncio.run(main())
