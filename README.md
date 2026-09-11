# LitmusAI

[![CI](https://github.com/kutanti/litmusai/actions/workflows/ci.yml/badge.svg)](https://github.com/kutanti/litmusai/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/litmuseval)](https://pypi.org/project/litmuseval/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

LitmusAI runs test cases against AI agents and records assertion results, latency, token usage, and estimated cost. Use it to compare model or prompt changes on tasks from your application.

Install the `litmuseval` package and import it as `litmusai`:

```bash
pip install litmuseval
```

## Quick start

This example runs locally without an API key:

```python
import asyncio
from litmusai import Agent, Contains, Numeric, TestCase, evaluate


def answer(task: str) -> str:
    answers = {
        "What is 15% of 240?": "36",
        "Who wrote 1984?": "George Orwell",
    }
    return answers.get(task, "I don't know")


agent = Agent.from_function(answer, name="example")
results = asyncio.run(evaluate(agent, [
    TestCase(id="math", task="What is 15% of 240?", assertions=[Numeric(36)]),
    TestCase(id="fact", task="Who wrote 1984?", assertions=[Contains(["Orwell"])]),
]))
assert results.passed == 2
results.save("results.json")
```

To call an OpenAI-compatible chat endpoint, replace `agent` with:

```python
import os

agent = Agent.from_openai_chat(
    model="gpt-4.1",
    api_key=os.environ["OPENAI_API_KEY"],
)
```

Pass the API key explicitly to this adapter. See [agent adapters](docs/adapters.md) for functions, HTTP endpoints, CLI programs, and framework integrations.

## YAML suites

Save this as `tests.yaml`:

```yaml
name: refund-tests
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

Define an agent or function in `my_agent.py`, then run:

```bash
litmus run --suite tests.yaml --agent my_agent:agent
litmus run --suite tests.yaml --agent my_agent:agent --runs 5
```

Assertions cover strings, numbers, regular expressions, JSON, semantic similarity, and LLM grading. `All`, `AnyOf`, `AtLeast`, and `Weighted` combine checks. Semantic and LLM assertions make additional API calls. JSON Schema validation uses the optional `jsonschema` package; install it for full schema support.

## Results and cost

Python results include per-case scores, responses, latency, and token counts. JSON, CSV, JUnit XML, Markdown, and HTML outputs support inspection and CI reporting.

Chat adapters read token counts from provider responses and calculate cost using the bundled pricing table. These are estimates, not billing records: prices can become outdated, and cached tokens or other provider charges may differ. An unrecognized model can report zero cost when no pricing is available.

```bash
litmus run -s tests.yaml -a my_agent:agent --format json --output run.json
litmus report -r run.json --html report.html
litmus diff --before earlier.json --after later.json
```

When comparing models, save the suite, model parameters, run count, raw results, and pricing assumptions. Small example suites do not establish a general model ranking.

## Conversations

Run this inside an async function or a notebook that supports `await`:

```python
from litmusai import ConversationRunner, MultiTurnCase, Step

case = MultiTurnCase(
    id="refund", name="Refund conversation",
    steps=[
        Step(user="I want to return my shoes",
             assertions=[Contains(["return", "help"], mode="any")]),
        Step(user="Order #12345", assertions=[Contains(["12345"])]),
        Step(user="Process the refund",
             assertions=[Contains(["refund", "confirm"], mode="any")]),
    ],
)
result = await ConversationRunner(agent).run(case)
print(result.summary())
```

The runner passes conversation history to the agent. Custom functions must accept and use the `history` keyword argument. The `is_cascade` flag marks failures after the first failure; it does not establish that an earlier mistake caused a later one. Context maintenance uses phrase matching and can misclassify legitimate clarification requests.

## Safety and memory poisoning

```python
from litmusai import MemoryPoisonScanner, SafetyScanner

safety = await SafetyScanner(depth="standard").scan(agent)
poisoning = await MemoryPoisonScanner(depth="standard").scan(agent)
print(safety.to_markdown())
print(poisoning.summary())
```

Safety scans use attack prompts and response patterns. Memory scans inject instructions or false facts into earlier turns and check later responses. The depth setting selects a subset of the attack library. These scores describe the selected checks; they do not prove that an agent is safe or resistant to other attacks.

```bash
litmus scan --agent my_agent:agent --level thorough --fail-on-unsafe
```

## Ground truth

Define an expected answer and record its source:

```yaml
name: science
cases:
  - id: boiling_point
    task: "At what temperature does water boil at sea level, in Celsius?"
    ground_truth:
      answer: 100
      answer_type: numeric
      source: "physics textbook"
      verified_by: "reviewer"
      confidence: 1.0
```

The loader generates assertions from the answer type. Explicit assertions take precedence. Provenance fields record what you supply; LitmusAI does not independently verify the answer.

## Pipelines and profiles

`Pipeline` combines evaluation, optional safety scanning, and report generation:

```python
from litmusai import Pipeline

result = await Pipeline(
    agent, "coding", safety=True, runs=3, report="html",
).run()
```

`litmus profiles` lists presets. The CLI applies a subset of profile settings; it does not run inline safety scans or set model temperature and seed. Set model parameters on the agent and run safety scans explicitly.

## Built-in suites

`litmus suites` lists `coding`, `research`, `safety`, `planning`, `customer_support`, `summarization`, `instruction_following`, and `tool_use`. Use these as examples, then add cases and assertions for your own application.

## Development

```bash
git clone https://github.com/kutanti/litmusai.git
cd litmusai
pip install -e ".[dev]"
pytest
ruff check src/ tests/
mypy src/litmusai/ --ignore-missing-imports
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the review process and commit style. Licensed under [MIT](LICENSE).
