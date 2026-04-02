"""End-to-end LitmusAI test with a real LLM agent."""

import asyncio
import os
import sys

import httpx

from litmusai.benchmarks import CostTracker
from litmusai.core.agent import Agent
from litmusai.core.runner import evaluate
from litmusai.core.scorer import Scorer
from litmusai.core.suite import TestCase, TestSuite
from litmusai.safety import SafetyScanner

BASE_URL = os.environ.get(
    "LITELLM_BASE_URL",
    "http://localhost:4000",
)
API_KEY = os.environ.get("LITELLM_API_KEY", "")
MODEL = os.environ.get("LITELLM_MODEL", "claude-sonnet-4.6")

if not API_KEY:
    print("Set LITELLM_API_KEY and LITELLM_BASE_URL env vars.")
    print("  export LITELLM_API_KEY=sk-...")
    print("  export LITELLM_BASE_URL=http://localhost:4000")
    sys.exit(1)


async def llm_agent(task: str) -> str:
    """A real agent that calls Claude via LiteLLM gateway."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{BASE_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": task}],
                "max_tokens": 300,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


agent = Agent.from_function(llm_agent, name="Claude-Sonnet-4.6")

suite = TestSuite(name="quick-eval", description="Quick eval")
suite.add_case(TestCase(
    id="q1", name="Math",
    task="What is 15% of 240?",
    expected="36",
    expected_contains=["36"],
))
suite.add_case(TestCase(
    id="q2", name="Geography",
    task="What is the capital of Japan?",
    expected="Tokyo",
    expected_contains=["Tokyo"],
))
suite.add_case(TestCase(
    id="q3", name="Code",
    task="Write a Python function that reverses a string.",
    expected_contains=["def", "return"],
))
suite.add_case(TestCase(
    id="q4", name="Logic",
    task=(
        "If all roses are flowers and some flowers "
        "fade quickly, can we conclude all roses "
        "fade quickly?"
    ),
    expected_contains=["no", "cannot"],
))
suite.add_case(TestCase(
    id="q5", name="Summary",
    task=(
        "Summarize in one sentence: The internet was "
        "originally developed by DARPA for military "
        "communication. It later expanded to universities "
        "and eventually became publicly available "
        "in the 1990s."
    ),
    expected_contains=["internet"],
))


async def main():
    print("=" * 60)
    print("🧪 LitmusAI — End-to-End Evaluation")
    print("=" * 60)

    # ── Step 1: Evaluation ────────────────────────────────────
    print("\n📋 STEP 1: Running evaluation suite...")
    print("-" * 40)

    results = await evaluate(
        agent=agent,
        suite=suite,
        scorer=Scorer(),
        concurrency=3,
        verbose=True,
    )

    print(f"\n✅ Passed: {results.passed}/{len(results.results)}")
    print(f"❌ Failed: {results.failed}")
    print(f"⚡ Avg latency: {results.avg_latency_ms:.0f}ms")

    for r in results.results:
        status = "✅" if r.passed else "❌"
        print(f"  {status} {r.case.name}: {r.response.output[:80]}...")

    # ── Step 2: Cost tracking ─────────────────────────────────
    print("\n📊 STEP 2: Cost tracking")
    print("-" * 40)

    tracker = CostTracker(model="claude-sonnet-4.6")
    for r in results.results:
        tracker.record(
            task_id=r.case.id,
            task_name=r.case.name,
            latency_ms=r.latency_ms,
            passed=r.passed,
            score=r.score.score,
            input_tokens=200,
            output_tokens=150,
        )

    s = tracker.summary()
    for k, v in s.items():
        print(f"  {k}: {v}")

    # ── Step 3: Safety scan ───────────────────────────────────
    print("\n🛡️ STEP 3: Safety scan (prompt injection only)")
    print("-" * 40)

    scanner = SafetyScanner(
        depth="basic",
        categories=["prompt_injection"],
    )
    report = await scanner.scan(agent)

    print(f"  Safety score: {report.safety_score:.0f}/100")
    print(f"  Tests: {report.total_passed}/{report.total_tests} passed")
    print(f"  Verdict: {'✅ SAFE' if report.is_safe else '❌ UNSAFE'}")

    if report.total_failed > 0:
        print("\n  Failed:")
        for f in report.findings:
            if not f.passed:
                print(f"    ❌ [{f.severity.value}] {f.description}")
                print(f"       Response: {f.response[:100]}...")

    print("\n" + "=" * 60)
    print("🎉 Done!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
