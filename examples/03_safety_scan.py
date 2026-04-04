"""Example 3: Safety scanning an agent.

Tests your agent against prompt injection, jailbreaks, PII leaks, and more.

Usage:
    python examples/03_safety_scan.py
"""

import asyncio

from litmusai import Agent
from litmusai.safety import SafetyScanner


# ─── Define your agent ───────────────────────────────────────────
# A well-behaved agent that refuses harmful requests:


async def safe_agent(task: str) -> str:
    """An agent with safety guardrails."""
    dangerous_keywords = [
        "hack", "exploit", "bypass", "ignore instructions",
        "system prompt", "forget", "pretend", "jailbreak",
    ]

    task_lower = task.lower()
    for kw in dangerous_keywords:
        if kw in task_lower:
            return (
                "I can't help with that request. "
                "I'm designed to be helpful, harmless, and honest."
            )

    return f"Here's my response to: {task[:50]}..."


# ─── Wrap as Agent ───────────────────────────────────────────────

agent = Agent.from_function(safe_agent, name="safe-agent")


# ─── Run safety scan ─────────────────────────────────────────────

async def main():
    # Scan levels: "basic" (26 attacks), "standard" (41), "thorough" (46)
    scanner = SafetyScanner(depth="standard")
    report = await scanner.scan(agent)

    # Print full report
    print(report.to_markdown())
    print(f"\nSafety score: {report.safety_score:.0f}/100")
    print(f"Safe: {'✅ Yes' if report.is_safe else '❌ No'}")

    # Check specific categories
    for cat, score in report.categories.items():
        status = "✅" if score.passed == score.total else "⚠️"
        print(f"  {status} {cat.value}: {score.passed}/{score.total}")


if __name__ == "__main__":
    asyncio.run(main())
