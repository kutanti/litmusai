<div align="center">

# 🧪 LitmusAI

### The eval framework your AI agents deserve

[![CI](https://github.com/kutanti/litmusai/actions/workflows/ci.yml/badge.svg)](https://github.com/kutanti/litmusai/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/litmuseval)](https://pypi.org/project/litmuseval/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**15 assertion types · 46 safety attacks · 8 built-in suites · real token costs · one `pip install`**

[Install](#install) · [Quick Start](#quick-start) · [Features](#features) · [CLI](#cli) · [CI/CD](#cicd)

</div>

---

## The Problem

Your agent works in demos. But you have no idea if it still works after your last commit. You don't know if Claude is actually better than GPT for your use case — or just 5x more expensive. You can't tell if your prompt change introduced a safety regression. And your "eval" is you manually typing questions into a chat window.

**LitmusAI fixes this.** One function call gives you pass rates, real token costs, latency, safety scores, and regression detection — across any model, any agent framework, any deployment.

```python
results = await evaluate(agent, suite)
# ✅ 12/12 passed | 💰 $0.003 | ⚡ 1100ms avg | 🔤 850 tokens
```

That cost is real — captured from API responses, not estimated from tiktoken.

---

## Install

```bash
pip install litmuseval
```

## Quick Start

```python
import asyncio
import litmusai
from litmusai import Agent, TestSuite, TestCase, Numeric, Contains, evaluate

litmusai.configure(api_key="sk-...")

agent = Agent.from_openai_chat(model="gpt-4o")

suite = TestSuite(name="basics")
suite.add_case(TestCase(
    id="math", name="Percentage",
    task="What is 15% of 240?",
    assertions=[Numeric(36, tolerance=0.01)],
))
suite.add_case(TestCase(
    id="fact", name="Factual",
    task="Who wrote 1984?",
    assertions=[Contains(["Orwell"])],
))

results = asyncio.run(evaluate(agent, suite))
# ✅ 2/2 passed | 💰 $0.0005 | ⚡ 1100ms avg
```

Real token counts. Real cost. Not estimates.

---

## Why LitmusAI?

| | LitmusAI | promptfoo | Manual testing |
|---|---------|-----------|---------------|
| Agent-native (not just LLM) | ✅ | ❌ | ❌ |
| Real token cost tracking | ✅ | ❌ | ❌ |
| Safety red-teaming built in | ✅ | ❌ | ❌ |
| Works with any framework | ✅ | Partial | ✅ |
| One-line agent connection | ✅ | ❌ | N/A |
| Python-native | ✅ | JS/TS | N/A |
| Open source | ✅ | ⚠️ Acquired | N/A |

---

## Features

### Agent Adapters

Connect any agent with one line:

```python
Agent.from_openai_chat(model="gpt-4o")           # OpenAI / compatible APIs
Agent.from_azure(resource="r", deployment="d")     # Azure OpenAI
Agent.from_function(my_fn)                         # Async function
Agent.from_url("http://localhost:8000/chat")        # HTTP endpoint
Agent.from_langchain(chain)                        # LangChain
Agent.from_crewai(crew)                            # CrewAI
```

### Assertions (15 types)

```python
# Basics
Numeric(36, tolerance=0.01)              # extracts numbers, handles "thirty-six"
Contains(["Paris", "France"], mode="all")
NotContains(["hack", "exploit"])
Exact("yes")
RegexMatch(r"\d{4}-\d{2}-\d{2}")

# Structured
JsonValid()                              # handles ```json fences
JsonSchema({"type": "object", "required": ["name"]})
JsonPath("items[0].price", 10, operator="gt")

# AI-powered
Semantic("The capital of France", threshold=0.85)
LLMGrade("Is this factually correct?")

# Compose
All(Numeric(36), NotContains(["sorry"]))
AnyOf(Exact("36"), Numeric(36))
Weighted([(Numeric(36), 0.7), (Contains(["because"]), 0.3)], threshold=0.6)
```

### Safety Scanner

46 attacks across 7 categories — prompt injection, jailbreak, PII leak, bias, hallucination, toxicity, data exfiltration:

```python
from litmusai.safety import SafetyScanner

report = await SafetyScanner(depth="standard").scan(agent)
print(f"{report.safety_score}/100 — {'SAFE' if report.is_safe else 'UNSAFE'}")
```

### Cost Tracking

```python
from litmusai.benchmarks import register_pricing

register_pricing("gpt-4o", input_cost=2.50, output_cost=10.0)
results = await evaluate(agent, suite)
print(f"${results.total_cost:.4f}")  # real token-level cost
```

### Multi-Run Statistics

```python
from litmusai import multi_evaluate

stats = await multi_evaluate(agent, suite, runs=5)
# mean ± std dev per case, flaky test detection
```

### HTML Reports

```bash
litmus report --html report.html --log-dir ./eval-logs/
```

Interactive dark-theme report with sorting, filtering, and drill-down details.

### Global Config

```python
import litmusai

# Set once — all assertions and agents use these defaults
litmusai.configure(
    api_key="sk-...",
    base_url="https://api.openai.com/v1",
)

# Azure
litmusai.configure(api_key="azure-key", auth_style="azure")
agent = Agent.from_azure(resource="my-resource", deployment="gpt-4o")
```

### Retry & Tracing

```python
from litmusai.retry import with_retry, RetryConfig
from litmusai.tracing import Tracer

result = await with_retry(agent.run, task, config=RetryConfig(max_retries=3))

tracer = Tracer()
with tracer.span("eval") as s:
    result = await agent.run(task)
tracer.save("trace.json")
```

---

## CLI

```bash
litmus run --suite coding --agent my_agent:agent    # run eval
litmus scan --agent my_agent:agent --depth thorough  # safety scan
litmus diff --before run1.json --after run2.json     # compare runs
litmus history --log-dir ./eval-logs/                # list past runs
litmus report --html report.html --junit results.xml --csv results.csv
litmus init                                          # scaffold config
litmus suites                                        # list built-in suites
```

**8 built-in suites** (50 cases): coding, research, safety, planning, customer_support, summarization, instruction_following, tool_use.

---

## CI/CD

```yaml
# .github/workflows/eval.yml
- uses: kutanti/litmusai@v1
  with:
    agent: my_agent:agent
    suite: coding
    threshold: 0.8
```

Regression detection: flags pass rate drops, cost spikes, latency increases.

---

## Config File

```yaml
# .litmus/config.yaml
suite: coding
agent: my_agent:agent
threshold: 0.8
runs: 3
log-dir: ./eval-logs
```

---

## Development

```bash
git clone https://github.com/kutanti/litmusai.git
cd litmusai && pip install -e ".[dev]"
pytest                    # 641 tests
ruff check src/ tests/    # lint
mypy src/litmusai/        # types
```

---

## License

MIT

<div align="center">

**[kutanti/litmusai](https://github.com/kutanti/litmusai)** · If this helps you ship better agents, give it a ⭐

</div>
