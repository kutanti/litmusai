# LitmusAI

[![CI](https://github.com/kutanti/litmusai/actions/workflows/ci.yml/badge.svg)](https://github.com/kutanti/litmusai/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/litmuseval)](https://pypi.org/project/litmuseval/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

LitmusAI runs test cases against AI agents and records assertion results, labeled task metrics, latency, token usage, and estimated cost. Use it to compare model or prompt changes on tasks from your application.

## Installation

Requires Python 3.10 or newer. Install the `litmuseval` package and import it as `litmusai`:

```bash
pip install litmuseval
```

This README describes the current repository. To install that version, including the latest merged fixes, use Git:

```bash
pip install "git+https://github.com/kutanti/litmusai.git@main"
litmus --version
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

Save this local example agent as `my_agent.py`:

```python
def agent(task: str) -> str:
    if "return" in task.lower():
        return "I can help with your return and refund."
    return "Please share your order number."
```

Run the suite using the file path and function name:

```bash
litmus run --suite tests.yaml --agent my_agent.py:agent
litmus run --suite tests.yaml --agent my_agent.py:agent --runs 5
```

Replace the example function with your application's agent. An importable module can also be loaded as `my_agent:agent`.

For a new project, `litmus init` writes `.litmus/config.yaml` and `suites/example.yaml`. The starter suite checks for "hello" and "4"; replace it with cases for your agent.

Assertions cover strings, numbers, regular expressions, JSON, semantic similarity, and LLM grading. `All`, `AnyOf`, `AtLeast`, and `Weighted` combine checks. Semantic and LLM assertions make additional API calls. JSON Schema validation uses the optional `jsonschema` package; install it for full schema support.

## Classification and extraction metrics

Labeled suites report precision, recall and F1 from ground truth without an LLM judge. Classification includes accuracy, per-class counts, micro/macro/weighted averages and a confusion matrix. Extraction matches field values or entity occurrences one-to-one, with optional whitespace normalization and case folding.

From a checkout, run the local examples:

```bash
litmus run -s examples/routing.yaml -a examples/labeled_agents.py:route --runs 3 -o routing.json
litmus run -s examples/extraction.yaml -a examples/labeled_agents.py:extract -o extraction.json
litmus report -r routing.json --html routing.html
```

Failed calls and malformed predictions remain visible through error counts, prediction coverage and missed labels/items. Metrics pool counts across all repetitions and remain separate from assertion pass rates. See [labeled metrics](docs/labeled-metrics.md) for the Python API, matching rules, undefined values and result schema.

## Results and cost

Python results include per-case scores, responses, latency, and token counts. Use the `results.json` saved in the quick start to generate HTML, JUnit XML, or CSV reports:

```bash
litmus report --results results.json --html report.html
litmus report --results results.json --junit results.xml --csv results.csv
```

Save two Python evaluations with `results.save("earlier.json")` and `results.save("later.json")`, then compare them:

```bash
litmus diff earlier.json later.json --fail-on-regression
```

`litmus diff` matches cases by their IDs and exits with code 1 for a pass-to-fail regression when this flag is set. Keep case IDs stable between runs.

The CLI also writes JSON or Markdown evaluation summaries:

```bash
litmus run -s tests.yaml -a my_agent.py:agent --format json --output run.json
litmus run -s tests.yaml -a my_agent.py:agent --format markdown --output run.md
```

CLI JSON wraps the same versioned payload as `results.save()` in a status envelope. Both can be loaded by `litmus report` and `litmus diff`. Multi-run files retain every repetition; select individual `run_results` entries for a case-level diff.

Chat adapters read token counts from provider responses and calculate cost using the bundled pricing table. These are estimates, not billing records: prices can become outdated, and cached tokens or other provider charges may differ. An unrecognized model can report zero cost when no pricing is available.

When comparing models, save the suite, model parameters, run count, raw results, and pricing assumptions. Small example suites do not establish a general model ranking.

## Conversations

The following Python examples use `await`; run them inside an async function or a notebook that supports it. For conversations, configure an agent that accepts conversation history:

```python
from litmusai import Contains, ConversationRunner, MultiTurnCase, Step

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

The runner passes conversation history to the agent. Custom functions must accept and use the `history` keyword argument. A failed agent call fails its conversation step, including a step with no assertions. The `is_cascade` flag marks failures after the first failure; it does not establish that an earlier mistake caused a later one. Context maintenance uses phrase matching and can misclassify legitimate clarification requests.

## Safety and memory poisoning

```python
from litmusai import MemoryPoisonScanner
from litmusai.safety import SafetyScanner

safety = await SafetyScanner(depth="standard").scan(agent)
poisoning = await MemoryPoisonScanner(depth="standard").scan(agent)
print(safety.to_markdown())
print(poisoning.summary())
print(safety.verdict, poisoning.verdict)
```

Safety scans use attack prompts and response patterns. Memory scans inject instructions or false facts into earlier turns and check later responses. The depth setting selects a subset of the attack library. These scores describe the selected checks; they do not prove that an agent is safe or resistant to other attacks.

Failed agent calls retain their error details and count as failed checks. If any finding contains an agent error, the scan verdict is `INCONCLUSIVE`; `safety.is_safe` and `poisoning.is_resistant` are false. Pipeline summaries preserve this verdict. The CLI flag below exits with code 1 for an unsafe or inconclusive safety scan:

```bash
litmus scan --agent my_agent.py:agent --level thorough --fail-on-unsafe
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

`litmus profiles` lists the `quick`, `thorough`, `benchmark`, `safety`, and `ci` presets. For run count, concurrency, and pass-rate threshold, explicit CLI options override the selected profile, which overrides `.litmus/config.yaml` defaults:

```bash
litmus run -s tests.yaml -a my_agent.py:agent --profile thorough
litmus run -s tests.yaml -a my_agent.py:agent --profile thorough --runs 2 --concurrency 4
```

The first command uses three runs and concurrency of three. The second uses two runs and concurrency of four. CLI run counts and concurrency must be at least one.

The CLI applies a subset of profile settings; it does not run inline safety scans, generate the profile's report format, or set model temperature and seed. Set model parameters on the agent and run safety scans explicitly. To apply a profile's evaluation, safety, and reporting settings in Python:

```python
from litmusai.profiles import get_profile

profile = get_profile("thorough")
result = await Pipeline(agent, "coding", **profile.to_kwargs()).run()
```

With multiple runs, CLI summaries, threshold checks, and budget checks currently use the last run. `Pipeline` uses the first run for its primary evaluation and threshold. Use one run for CI gates that need these values to describe the entire evaluation; repeated-run statistics do not yet drive aggregate gates or budgets.

## GitHub Actions

Commit `tests.yaml` and `my_agent.py` from the YAML example, then save this as `.github/workflows/evaluate.yml`:

```yaml
name: Evaluate agent
on: [push, pull_request]

permissions:
  contents: read

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: kutanti/litmusai@v0.4.0
        id: evaluation
        with:
          suite: tests.yaml
          agent: my_agent.py:agent
          threshold: "0.8"
          runs: "1"
          post-comment: "false"
      - uses: actions/upload-artifact@v4
        if: always() && steps.evaluation.outputs.results-path != ''
        with:
          name: litmus-results
          path: ${{ steps.evaluation.outputs.results-path }}
```

The action installs LitmusAI from its selected revision and fails when the evaluation fails a threshold, budget, or baseline comparison. It exposes `pass-rate`, `total-cost`, `passed`, `failed`, `has-regression`, and `results-path` as step outputs. Results remain available after a failed evaluation when a results file was produced.

The local example needs no provider credentials or extra dependencies. For your own agent, install any additional dependencies before the evaluation step and pass its credentials through that step's `env`. If installing packages in an earlier step, use `actions/setup-python` with the same version as the action's `python-version` input (default `3.11`).

To enable comments on pull requests with write access, set `post-comment: "true"` and grant `pull-requests: write`. See [action.yml](action.yml) for baseline, budget, concurrency, and other inputs.

## Built-in suites

`litmus suites` lists `coding`, `research`, `safety`, `planning`, `customer_support`, `summarization`, `instruction_following`, and `tool_use`. Use these as examples, then add cases and assertions for your own application.

## Development

CI tests Python 3.10, 3.11, and 3.12 on Linux, plus Python 3.12 on Windows. It also runs lint, strict type checking, and source and wheel package builds.

```bash
git clone https://github.com/kutanti/litmusai.git
cd litmusai
pip install -e ".[dev]"
pytest
ruff check src/ tests/ scripts/
mypy src/litmusai/ --ignore-missing-imports
```

Provider integration tests are skipped when their credentials are absent. New PR titles and commit messages must use plain text without emojis; existing Git history is preserved.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the review process and [the project review](docs/project-review.md) for remaining work on multi-run gates, result formats, and adapter configuration. Licensed under [MIT](LICENSE).
