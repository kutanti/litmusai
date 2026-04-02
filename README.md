<div align="center">

# 🧪 LitmusAI

### Eval framework for AI agents that actually works

[![CI](https://github.com/kutanti/litmusai/actions/workflows/ci.yml/badge.svg)](https://github.com/kutanti/litmusai/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-404%20passing-brightgreen)](https://github.com/kutanti/litmusai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**5,900+ lines of code · 404 tests · 11 assertion types · 46 safety attacks · 5-model benchmarks**

[Quickstart](#-quickstart) · [Features](#-whats-inside) · [Assertions](#-assertion-engine) · [Examples](#-real-examples) · [Benchmark Results](#-real-benchmark-data)

</div>

---

## Why?

You deployed an agent. It works in demos. But:

- Does it still work after your last commit?
- Is Claude actually better than GPT for your use case — or just more expensive?
- Did your prompt change introduce a safety regression?
- What's your actual **cost per correct answer**?

LitmusAI answers these with **one function call**.

---

## ⚡ Quickstart

```bash
pip install litmuseval
```

### 30-second eval

```python
import asyncio
from litmusai import Agent, TestSuite, TestCase, Numeric, evaluate

# 1. Connect your agent (works with any OpenAI-compatible API)
agent = Agent.from_openai_chat(
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    model="gpt-4o",
)

# 2. Define tests with real assertions (not just substring matching)
suite = TestSuite(name="math-eval")
suite.add_case(TestCase(
    id="q1", name="Percentage",
    task="What is 15% of 240?",
    assertions=[Numeric(36, tolerance=0.01)],
))

# 3. Run
results = asyncio.run(evaluate(agent, suite))
print(results)
# ✅ 1/1 passed | 💰 $0.0003 | ⚡ 1200ms avg | 🔤 47 tokens
```

**That `$0.0003` is real** — not an estimate. `from_openai_chat` captures actual token usage from the API response.

---

## 🔧 What's Inside

### 1. Universal Agent Adapter

Connect **any** agent — function, API, or framework:

```python
# Any OpenAI-compatible API (OpenAI, Anthropic via proxy, LiteLLM, Ollama, vLLM)
agent = Agent.from_openai_chat(base_url="...", api_key="...", model="gpt-4o")

# Plain async function
agent = Agent.from_function(my_async_fn, name="my-agent")

# HTTP endpoint
agent = Agent.from_url("http://localhost:8000/chat")

# LangChain
agent = Agent.from_langchain(my_chain)

# CrewAI
agent = Agent.from_crewai(my_crew)

# Any callable
agent = Agent.from_callable(obj_with_call_method)
```

`from_openai_chat` is the recommended path — it automatically captures **real token counts and cost** from the API response. No guessing.

### 2. Assertion Engine (11 Types)

The heart of the framework. **Composable, type-safe scoring** that goes way beyond substring matching:

```python
from litmusai import (
    Numeric, Contains, NotContains, Exact, RegexMatch,
    JsonSchema, JsonPath, JsonValid,
    Semantic, LLMGrade, Custom,
    All, AnyOf, AtLeast, Weighted,
)

# ── String assertions ──────────────────────────────
Exact("Paris")                            # exact match
Contains(["Orwell", "1949"], mode="all")  # all must appear
NotContains(["hack", "exploit"])          # must NOT appear
RegexMatch(r"\b\d{4}-\d{2}-\d{2}\b")     # date format

# ── Numeric ────────────────────────────────────────
Numeric(36, tolerance=0.01)               # extracts numbers from text
Numeric(36)                               # handles "thirty-six" too

# ── Structured ─────────────────────────────────────
JsonValid()                               # valid JSON (handles ```json fences)
JsonSchema({"type": "object", "required": ["name", "age"]})
JsonPath("user.name", "Alice")            # check nested values
JsonPath("items[0].price", 10, operator="gt")

# ── AI-powered ─────────────────────────────────────
Semantic("The answer is 36", threshold=0.85)  # embedding similarity
LLMGrade("Is the math correct?", model="gpt-4o-mini")

# ── Custom ─────────────────────────────────────────
Custom(lambda r: len(r.split()) >= 10, name="min_words")
Custom(lambda r: "def " in r and "return" in r, name="has_function")
```

#### Compose them

```python
# All must pass
All(Numeric(36), NotContains(["sorry", "I can't"]))

# Any one is enough
AnyOf(Exact("36"), Numeric(36), Contains(["thirty-six"]))

# At least 2 of 3
AtLeast(2, [Exact("36"), Numeric(36), Contains(["36"])])

# Weighted scoring (no hard pass/fail)
Weighted([
    (Numeric(36), 0.6),           # 60% weight on correctness
    (NotContains(["sorry"]), 0.2), # 20% on confidence
    (Custom(lambda r: len(r) > 10, name="detail"), 0.2),
], threshold=0.7)
```

### 3. Safety Scanner (46 Attacks)

Red-team your agent automatically:

```python
from litmusai.safety import SafetyScanner

scanner = SafetyScanner(depth="standard")  # "basic" | "standard" | "thorough"
report = await scanner.scan(agent)

print(f"Score: {report.safety_score}/100")
print(f"Verdict: {'✅ SAFE' if report.is_safe else '❌ UNSAFE'}")

# Category breakdown
for cat, stats in report.categories.items():
    print(f"  {cat}: {stats.passed}/{stats.total}")
```

**Attack categories:** Prompt injection (8), Jailbreak (6), PII leak (5), Harmful content (5), Hallucination (5), Bias (5), Data exfiltration (3), Over-reliance (5)

Refusal detection built in — an agent that says *"I can't reveal my system prompt"* won't be flagged as a failure.

### 4. Cost & Latency Benchmarking

Real token-level cost tracking, not estimates:

```python
from litmusai.benchmarks import CostTracker, compare_models, register_pricing

# Register pricing ($/million tokens)
register_pricing("gpt-4o", input_cost=2.50, output_cost=10.0)
register_pricing("claude-sonnet-4", input_cost=3.0, output_cost=15.0)

# Track across runs
tracker = CostTracker(model="gpt-4o", agent_name="GPT-4o")
tracker.record("q1", task_name="Math", latency_ms=1200,
               passed=True, score=1.0, input_tokens=22, output_tokens=15)

# Compare models
comparison = compare_models(tracker_gpt, tracker_claude)
print(comparison.to_markdown())
```

### 5. LLM-as-Judge

Use an LLM to grade responses on criteria you define:

```python
from litmusai.scorers import LLMJudge

judge = LLMJudge(
    provider="openai",
    model="gpt-4o-mini",
    criteria=["correctness", "helpfulness", "safety"],
)
result = await judge.score(
    task="Explain quantum computing",
    response=agent_response,
)
print(f"Score: {result.score}/5 — {result.reason}")
```

### 6. CI/CD Integration

Catch regressions before they ship:

```yaml
# .github/workflows/eval.yml
- uses: kutanti/litmusai@v1
  with:
    agent: my_agent:agent
    suite: coding
    threshold: 0.8
    format: github
```

Regression detection: flags >5% pass rate drops, >50% cost increases, >50% latency spikes.

### 7. Result Logging

Full audit trail of every evaluation:

```python
results = await evaluate(agent, suite, log_dir="./eval-logs/")
# Saves: agent_name, task, response, tokens, cost, score, timestamp
# as structured JSON for reproducibility
```

---

## 🎯 Real Examples

### Example 1: Compare 5 models

```python
import asyncio
from litmusai import Agent, TestSuite, TestCase, Numeric, Contains, evaluate
from litmusai.benchmarks import register_pricing

# Register pricing
register_pricing("gpt-4o", 2.50, 10.0)
register_pricing("gpt-4.1", 2.0, 8.0)
register_pricing("claude-sonnet-4", 3.0, 15.0)

# Create agents
models = {
    "GPT-4o": Agent.from_openai_chat(
        base_url="https://api.openai.com/v1",
        api_key="sk-...", model="gpt-4o",
    ),
    "GPT-4.1": Agent.from_openai_chat(
        base_url="https://api.openai.com/v1",
        api_key="sk-...", model="gpt-4.1",
    ),
    "Claude Sonnet": Agent.from_openai_chat(
        base_url="https://api.anthropic.com/v1",
        api_key="sk-...", model="claude-sonnet-4-20250514",
    ),
}

# Test suite
suite = TestSuite(name="benchmark")
suite.add_case(TestCase(
    id="math", name="Math", task="What is 15% of 240?",
    assertions=[Numeric(36, tolerance=0.01)],
))
suite.add_case(TestCase(
    id="fact", name="Factual", task="Who wrote 1984?",
    assertions=[Contains(["Orwell", "1949"], mode="all")],
))

# Run all
async def main():
    for name, agent in models.items():
        results = await evaluate(agent, suite, verbose=True)
        print(f"{name}: {results.pass_rate:.0%} | ${results.total_cost:.4f}")

asyncio.run(main())
```

### Example 2: Safety scan before deploy

```python
from litmusai import Agent
from litmusai.safety import SafetyScanner

agent = Agent.from_openai_chat(
    base_url="https://api.openai.com/v1",
    api_key="sk-...", model="gpt-4o",
    system_prompt="You are a helpful customer service agent.",
)

scanner = SafetyScanner(depth="thorough")
report = await scanner.scan(agent)

assert report.is_safe, f"Safety score: {report.safety_score}/100"
assert len(report.critical_failures) == 0, "Critical vulnerabilities found!"
```

### Example 3: JSON API validation

```python
from litmusai import (
    Agent, TestSuite, TestCase,
    All, JsonValid, JsonSchema, JsonPath, evaluate,
)

suite = TestSuite(name="api-tests")
suite.add_case(TestCase(
    id="planets", name="Structured output",
    task="Return the 3 largest planets as a JSON array with name and diameter_km",
    assertions=[All(
        JsonValid(),
        JsonSchema({
            "type": "array", "minItems": 3,
            "items": {
                "type": "object",
                "required": ["name", "diameter_km"],
            },
        }),
        JsonPath("0.name", "Jupiter"),
    )],
))

results = await evaluate(agent, suite)
```

### Example 4: Weighted scoring for nuance

```python
from litmusai import TestCase, Numeric, NotContains, Custom, Weighted

TestCase(
    id="explain", name="Explain well",
    task="Explain why the sky is blue",
    assertions=[Weighted([
        # 50% — mentions Rayleigh scattering
        (Contains(["scatter", "rayleigh", "wavelength"], mode="any"), 0.5),
        # 30% — doesn't refuse or hedge
        (NotContains(["I'm not sure", "I can't"]), 0.3),
        # 20% — substantive answer (>50 words)
        (Custom(lambda r: len(r.split()) >= 50, name="length"), 0.2),
    ], threshold=0.7)],
)
```

---

## 📊 Real Benchmark Data

We benchmarked 5 models on 6 tasks with **real token counts** (not estimates):

| Model | Pass Rate | Real Tokens | Real Cost | Cost/Pass | Avg Latency |
|-------|-----------|------------|-----------|-----------|-------------|
| **GPT-4.1** | **100%** | 501 | $0.0031 | **$0.0005** 🏆 | 2,269ms |
| **GPT-4o** | **100%** | 575 | $0.0046 | $0.0008 | **1,616ms** ⚡ |
| Claude Sonnet 4 | 100% | 848 | $0.0109 | $0.0018 | 3,794ms |
| Claude Opus 4 | 83% | 691 | $0.0427 | $0.0085 | 2,888ms |
| Gemini 2.5 Pro | 50% | 166 | $0.0010 | $0.0003* | 7,263ms |

<sup>*Gemini is cheap but only 50% pass rate on our test suite.</sup>

**Key finding:** Claude Opus costs **14x more than GPT-4.1** per correct answer, with lower accuracy. This is the kind of insight that saves real money.

*Generated with `examples/e2e_test.py` — run it yourself with your own API keys.*

---

## 🏗️ Architecture

```
litmusai/
├── assertions/      # 11 assertion types + 4 composites (1,400 lines)
├── core/
│   ├── agent.py     # Universal agent adapter — 7 factory methods (750 lines)
│   ├── runner.py     # Async eval runner with concurrency + logging (300 lines)
│   ├── scorer.py     # Assertion-aware scoring engine (220 lines)
│   └── suite.py      # Test suite management + YAML (130 lines)
├── safety/          # Red-team scanner — 46 attacks (800 lines)
├── scorers/         # LLM-as-Judge engine (640 lines)
├── benchmarks/      # Cost tracking + model comparison (620 lines)
├── ci/              # CI/CD regression detection (500 lines)
├── cli/             # CLI commands (350 lines)
└── suites/          # Built-in test suites (YAML)
    ├── coding/
    ├── research/
    ├── planning/
    └── safety/
```

**5,900+ lines of code · 404 tests · 22 source files**

---

## 🚀 Getting Started

### Install

```bash
pip install litmuseval
```

### Dev setup

```bash
git clone https://github.com/kutanti/litmusai.git
cd litmusai
pip install -e ".[dev]"
pytest                    # 404 tests
ruff check src/ tests/    # lint
mypy src/litmusai/        # type check
```

### CLI

```bash
litmus run --suite coding --agent my_agent:agent
litmus run --suite research --agent my_agent:agent --format markdown
```

---

## 🗺️ Roadmap

- [x] Universal agent adapter (7 factory methods)
- [x] Assertion engine (11 types + 4 composites)
- [x] Scoring pipeline (assertions wired into runner)
- [x] Safety scanner (46 attacks, 3 depths)
- [x] Cost & latency benchmarking (real tokens)
- [x] LLM-as-Judge scoring
- [x] CI/CD regression detection
- [x] Result logging (JSON)
- [ ] PyPI publish (`pip install litmuseval`)
- [ ] Multiple runs with statistical reporting
- [ ] HTML reports
- [ ] Expanded test suites (50+ per domain)

---

## 🤝 Contributing

PRs welcome! See the [open issues](https://github.com/kutanti/litmusai/issues).

```bash
git clone https://github.com/kutanti/litmusai.git
cd litmusai
pip install -e ".[dev]"
pytest && ruff check src/ tests/ && mypy src/litmusai/
```

All changes go through PR review with automated CI (ruff, pytest, mypy across Python 3.10-3.12).

---

## 📜 License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

**Built by [Kunal Tanti](https://github.com/kutanti)**

If this helps you ship better agents, give it a ⭐

</div>
