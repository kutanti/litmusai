# LitmusAI

[![CI](https://github.com/kutanti/litmusai/actions/workflows/ci.yml/badge.svg)](https://github.com/kutanti/litmusai/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/litmuseval)](https://pypi.org/project/litmuseval/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Test framework for AI agents. Think pytest for LLMs — assertions, cost tracking, safety scanning, multi-turn conversations.

```bash
pip install litmuseval
```

## Why I built this

Evaluating AI agents by hand doesn't scale. I needed something that could run the same tests across any model and agent workflows and tell me what actually changed — cost, accuracy, safety. This is that.

## Quick start

```python
import asyncio
import litmusai
from litmusai import Agent, TestCase, Numeric, Contains, evaluate

litmusai.configure(api_key="sk-...")

agent = Agent.from_openai_chat(model="gpt-4.1")

results = asyncio.run(evaluate(agent, [
    TestCase(id="math", task="What is 15% of 240?", assertions=[Numeric(36)]),
    TestCase(id="fact", task="Who wrote 1984?", assertions=[Contains(["Orwell"])]),
]))
# 2/2 passed | $0.0003 | 937ms avg
```

Or write tests in YAML:

```yaml
name: my-tests
cases:
  - id: refund
    task: "I want to return my shoes"
    assertions:
      - type: contains
        patterns: ["return", "refund"]
        mode: any
      - type: not_contains
        patterns: ["sorry, I can't help"]
```

```bash
litmus run --suite my-tests.yaml --agent my_agent:agent
```

## What it actually does

**Assertions** — 16 types, not just substring matching. Numeric extraction (handles "thirty-six"), JSON schema validation, embedding similarity, LLM-as-judge, regex, composable logic (All, AnyOf, AtLeast).

**Safety scanning** — 46 attack prompts across prompt injection, jailbreaks, PII leaks, bias, hallucination. My GPT-4.1 agent scored 78/100 — it literally replied "HACKED" to one of the injection tests.

**Memory poisoning detection** — 18 multi-turn attacks that test whether your agent can be manipulated through conversational context. Plants false facts, fake credentials, exfiltration URLs in early turns and checks if the agent follows them later.

**Multi-turn conversations** — test multi-step workflows where context matters. Does your agent remember order #12345 from three turns ago? Does a poisoned instruction in turn 1 corrupt behavior in turn 5?

**Ground truth management** — define verified answers with provenance tracking (who verified it, when, confidence level). Assertions are auto-generated from ground truth entries.

**Real cost tracking** — costs come from actual API responses, not tiktoken estimates. Tiktoken can be off by 10-20%.

**Multi-run stats** — run the same test 5 times. Some models pass a test 3 out of 5 times. You don't catch that with a single run.

**Multi-dimensional scoring** — score responses across 7 dimensions (correctness, completeness, format, relevance, safety, latency, cost) with configurable weights.

## Multi-turn evaluation

Test conversations where each turn depends on the last:

```python
from litmusai import Agent, MultiTurnCase, Step, ConversationRunner, Contains, Numeric

agent = Agent.from_openai_chat(model="gpt-4.1", api_key="sk-...")

case = MultiTurnCase(
    id="refund_flow",
    name="Refund conversation",
    steps=[
        Step(user="I want to return my shoes",
             assertions=[Contains(["return", "help"], mode="any")]),
        Step(user="Order #12345",
             assertions=[Contains(["12345"])]),
        Step(user="Process the refund",
             assertions=[Contains(["refund", "confirm"], mode="any")]),
    ],
)

result = await ConversationRunner(agent).run(case)
# 3/3 steps passed | context maintained | $0.003
```

Error compounding detection tells you whether a failure in step 3 was caused by a mistake in step 1 (cascade) or happened independently.

## Memory poisoning scanner

Test whether your agent can be manipulated through conversational context injection:

```python
from litmusai import Agent, MemoryPoisonScanner

agent = Agent.from_openai_chat(model="gpt-4.1", api_key="sk-...")

scanner = MemoryPoisonScanner(depth="standard")
report = await scanner.scan(agent)

print(report.resistance_score)  # 0-100
print(report.summary())
# RESISTANT: 12/13 attacks resisted | score 92/100
```

18 attacks across 6 categories: instruction injection, false fact persistence, identity override, exfiltration setup, authority spoofing, and delayed triggers. Based on attack patterns from DeepMind's "AI Agent Traps" research.

## Ground truth

Define verified answers with provenance:

```yaml
cases:
  - id: boiling_point
    task: "At what temperature does water boil at sea level?"
    ground_truth:
      answer: 100
      answer_type: numeric
      source: "physics textbook"
      verified_by: "kunal"
      confidence: 1.0
```

Assertions are auto-generated — numeric tolerance, contains checks, JSON validation — based on the answer type. Explicit assertions in the YAML take precedence.

```bash
litmus validate-ground-truth ground_truth.yaml
litmus ground-truth-stats suite.yaml
```

## Some numbers I found interesting

I ran the same suite across models:

| Model | Pass Rate | Cost | Cost/Correct |
|-------|-----------|------|-------------|
| GPT-4.1 | 100% | $0.017 | $0.0034 |
| Claude Sonnet 4 | 100% | $0.011 | $0.0018 |
| Claude Opus 4 | 83% | $0.043 | $0.0085 |

Opus costs 14x more per correct answer than GPT-4.1 and scores lower. We were using it for months.

## Connect your agent

```python
Agent.from_openai_chat(model="gpt-4.1")           # OpenAI / compatible
Agent.from_azure(resource="r", deployment="d")     # Azure
Agent.from_function(my_fn)                         # any async function
Agent.from_url("http://localhost:8000/chat")       # HTTP endpoint
Agent.from_langchain(chain)                        # LangChain
Agent.from_crewai(crew)                            # CrewAI
```

## Pipeline

Run eval + safety + report in one call:

```python
import asyncio
from litmusai import Agent, Pipeline

agent = Agent.from_openai_chat(model="gpt-4.1", api_key="sk-...")

async def main():
    result = await Pipeline(
        agent, "coding",
        safety=True,
        runs=3,
        report="html",
    ).run()

asyncio.run(main())
```

## Profiles

Presets for common scenarios:

```bash
litmus run -s coding -a agent:fn --profile quick       # fast iteration
litmus run -s coding -a agent:fn --profile thorough    # 3 runs, strict threshold
litmus run -s coding -a agent:fn --profile benchmark   # 5 runs, temp=0
litmus run -s coding -a agent:fn --profile ci          # strict threshold
litmus profiles                                         # see all
```

Custom profiles in YAML:

```yaml
# .litmus/profiles/production.yaml
name: production
runs: 5
safety: true
safety_depth: thorough
threshold: 0.9
report: html
```

## Built-in suites

8 suites, 50 test cases to start with. Not meant to be comprehensive — they're a starting point. Write your own for your domain.

```bash
litmus suites                                    # list them
litmus run --suite coding --agent my_agent:agent  # run one
```

`coding` · `research` · `safety` · `planning` · `customer_support` · `summarization` · `instruction_following` · `tool_use`

## Custom assertions

```python
from litmusai.assertions import Assertion, AssertionResult, register_assertion

class MaxWords(Assertion):
    def __init__(self, limit: int):
        self.limit = limit

    def check(self, response: str, **kwargs) -> AssertionResult:
        count = len(response.split())
        return AssertionResult(
            passed=count <= self.limit,
            score=min(1.0, self.limit / max(count, 1)),
            reason=f"{count} words (max {self.limit})",
            assertion_type="MaxWords",
        )

register_assertion("max_words", MaxWords)
# Now usable in YAML: { type: max_words, limit: 100 }
```

## CLI

```bash
litmus run --suite coding --agent my_agent:agent    # evaluate
litmus run --suite tests.yaml --runs 5              # multi-run
litmus scan --agent my_agent:agent --level thorough  # safety scan
litmus diff --before run1.json --after run2.json     # compare runs
litmus report -r results.json --html report.html     # generate report
litmus init                                          # scaffold project
```

## CI/CD

```yaml
# .github/workflows/eval.yml
- uses: kutanti/litmusai@v1
  with:
    agent: my_agent:agent
    suite: coding
    threshold: 0.8
```

## Development

```bash
git clone https://github.com/kutanti/litmusai.git
cd litmusai && pip install -e ".[dev]"
pytest                    # 839 tests
ruff check src/ tests/    # lint
mypy src/litmusai/        # types
```

~11K lines of code, 38 source files. MIT licensed.

## License

MIT — [Kunal Tanti](https://github.com/kutanti)
