# LitmusAI

**Test AI agents before deployment. Monitor them while they run.**

[![CI](https://github.com/kutanti/litmusai/actions/workflows/ci.yml/badge.svg)](https://github.com/kutanti/litmusai/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/litmuseval)](https://pypi.org/project/litmuseval/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Turn tasks from your application into repeatable checks for your AI agent. LitmusAI measures answer quality, consistency across runs, latency, token usage, and estimated cost so you can evaluate a model, prompt, or workflow change before shipping it. Its experimental runtime monitor observes live conversations and tool activity, evaluates configured policies, and emits risk events to your own systems.

Start with a Python function and local assertions; connect a model or an existing agent when you're ready. The same suites run from Python, the command line, and GitHub Actions.

[Quick start](#quick-start) · [Runtime monitoring](#runtime-monitoring) · [Agent adapters](docs/adapters.md) · [Usage guide](docs/usage.md) · [Labeled metrics](docs/labeled-metrics.md) · [Feature validation](docs/feature-validation.md)

| What you need to check | What LitmusAI provides |
|---|---|
| Does the agent return the right answer? | Python and YAML suites, string/number/JSON assertions, semantic checks, and LLM grading |
| Does routing or extraction match your labels? | Accuracy, precision, recall, F1, confusion matrices, and field/entity matching |
| Is the result consistent? | Repeated evaluations, per-case pass rates, score variation, and flaky-case detection |
| Did a model or prompt change help? | Agent comparison, saved baselines, and case-level regression diffs |
| Can it follow a conversation? | History-aware conversations, assertions on each turn, and memory-poisoning probes |
| What does it cost to run? | Token accounting, estimated cost, latency, and configurable score budgets |
| Can it fit into your delivery workflow? | CLI gates, GitHub Actions, JSON/Markdown/HTML reports, JUnit XML, and CSV |
| Is a running agent requesting a forbidden action? | Async capture, tool/destination allowlists, supported-secret exposure checks, and optional prompt-injection classification |
| Is a conversation exceeding tool limits or violating a policy? | Conversation tool-call windows, versioned rubrics, evidence requirements, and optional deeper evaluation |
| Can my application respond to a runtime finding? | CloudEvents through signed HTTPS webhooks, Kafka, or Azure Event Grid; your consumer decides the response |

## Installation

Requires Python 3.10 or newer. Install the `litmuseval` package and import it as `litmusai`:

```bash
pip install litmuseval
```

For the 1.1.0 release, use `pip install --upgrade litmuseval==1.1.0`.
See the [release notes and migration guidance](CHANGELOG.md#110---2026-09-19)
for the new runtime features and upgrade guidance from 1.0.0.

The package is **`litmuseval`**, the Python import is **`litmusai`**, and the command is **`litmus`**. Local assertions and HTTP/chat adapters work with the base installation. For full JSON Schema validation, also run `pip install jsonschema`. Framework integrations need their framework's dependencies; see [adapters](docs/adapters.md).

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
cases = [
    TestCase(id="math", task="What is 15% of 240?", assertions=[Numeric(36)]),
    TestCase(id="fact", task="Who wrote 1984?", assertions=[Contains(["Orwell"])]),
]
results = asyncio.run(evaluate(agent, cases))
assert results.passed == 2
results.save("results.json")
```

Expected result: **2/2 cases pass**, and `results.json` contains both responses, scores, and timings. This local function supplies no usage metadata: token counters are zero and cost is unknown. Return `AgentResponse(output=..., cost=0.0)` to explicitly declare a free run.

To call an OpenAI-compatible chat endpoint, replace `agent` with:

```python
import os

agent = Agent.from_openai_chat(
    model="gpt-4.1",
    api_key=os.environ["OPENAI_API_KEY"],
)
```

Pass the API key explicitly to this adapter. See [agent adapters](docs/adapters.md) for functions, HTTP endpoints, CLI programs, and framework integrations.

## Runtime monitoring

A support agent passes its tests. During a live conversation, someone asks it to send customer records to an unapproved address. With the tool instrumented and a destination allowlist configured, LitmusAI detects the forbidden request and emits an event with the policy, conversation, and supporting evidence. Your system can alert an operator, escalate the conversation, or decide what the agent may do next.

```text
Customer <-> Your agent
                 |
          messages, context, tool activity
                 |
                 v
         LitmusAI Runtime (async)
         rules + conversation history
         optional external evaluators
                 |
                 v
          Risk event (CloudEvent)
                 |
         Webhook / Kafka / Event Grid
                 |
                 v
          Your response handler
```

Capture uses a bounded background queue, and evaluation runs outside the agent's response path. LitmusAI reports events; it does not block tool execution or enforce a response. An alert about a requested action does not establish that the action completed or that data was exfiltrated.

| Runtime feature | What you configure |
|---|---|
| [Threat detection](docs/runtime-threat-alerting.md) | Exact tool and destination allowlists, supported-secret patterns, and an optional Lakera or compatible HTTP prompt-injection classifier |
| [Conversation tool limits](docs/runtime-tool-usage.md) | A maximum number of distinct tool requests per conversation and time window; for example, alert on the 21st request within 60 seconds when the limit is 20 |
| [Two-stage evaluation](docs/runtime-conditional-review.md) | A first-stage injection classifier and an optional deeper HTTP evaluator for uncertain results, with separate workers and persistent call budgets |
| [Conversation policies](docs/runtime-conversation-policies.md) | Versioned rubrics, scope, history requirements, severity, and optional score thresholds for sensitive-data requests, abuse, business-policy violations, suspicious patterns, out-of-scope replies, and responses unsupported by approved sources |
| [Event delivery](docs/runtime-threat-alerting.md) | Signed HTTPS webhooks, Kafka, Azure Event Grid, or fan-out to all three; durable collector jobs, delivery retries, and replay |

Semantic policies require a compatible, customer-selected evaluator. The policy framework does not bundle validated detectors for every category. Grounding checks assess support against configured source-tool results; they do not verify universal truth. Delivery can repeat, so consumers should deduplicate CloudEvents IDs.

### Try the local demo

Runtime monitoring is an **experimental, opt-in, single-instance pilot**, included starting with 1.1.0. Install the collector and optional transports from PyPI:

```bash
pip install --upgrade "litmuseval[runtime]==1.1.0"
# Add transport dependencies when needed:
pip install --upgrade "litmuseval[runtime,runtime-kafka,runtime-azure]==1.1.0"
```

The runnable demo also needs the example files from the repository. Check out the matching release:

```bash
git clone --branch v1.1.0 https://github.com/kutanti/litmusai.git
cd litmusai
pip install -e ".[runtime]"
# Optional transport dependencies:
pip install -e ".[runtime-kafka,runtime-azure]"
```

Set `LITMUS_RUNTIME_API_KEY` and `LITMUS_WEBHOOK_SECRET` to separate random secrets of at least 32 characters. Use the same values in each terminal. Then run these commands in separate terminals from the repository root:

```bash
python -m uvicorn examples.runtime.webhook_receiver:app --port 8766 --no-access-log
litmus runtime serve --config examples/runtime/config.yaml
python -m examples.runtime.agent
```

The demo email tool is simulated: it sends no mail. The webhook receiver prints the resulting alerts. Run `python -m examples.runtime.tool_usage` to exercise conversation tool limits. See the [runtime guide](docs/runtime-threat-alerting.md) for secrets, TLS, transport configuration, delivery behavior, and operating limits.

### Connect your own agent

Add capture hooks where your application receives messages, supplies retrieved context, runs tools, and returns responses. The following is an integration sketch inside your application's async request handler; replace the application variables and functions with your own:

```python
from litmusai.runtime import RuntimeClient

async with RuntimeClient(
    endpoint=collector_url, api_key=runtime_api_key, project_id="support"
) as monitor:
    async with monitor.session(agent_id="support-agent", session_id=request_id) as session:
        session.emit_message(user_message)
        session.emit_context(retrieved_document)
        monitored_send_email = session.wrap_tool(
            send_email, name="send_email", destination_argument="recipient"
        )
        # Register this wrapper as the tool your agent actually executes.
        response = await run_existing_agent(
            user_message, tools={"send_email": monitored_send_email}
        )
        session.emit_response(response, destination="customer-channel")
        await monitor.aflush(timeout=5)
```

The SDK supports synchronous and asynchronous tools. Other languages can post events to the authenticated `/v1/events` endpoint. Instrumentation must capture the actual activity; the SDK cannot discover tools hidden inside an arbitrary agent endpoint. Flushing can wait, and a client crash can lose unacknowledged queued events. Inspect SDK health as well as collector coverage.

Kafka delivery has been exercised against a real broker in CI. Live Azure downstream receipt, semantic evaluator quality, and production latency still need deployment-specific validation; see [the remaining runtime validation work](https://github.com/kutanti/litmusai/issues/116).

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

For a new project, `litmus init` writes `.litmus/config.yaml` and `suites/example.yaml`. Run that suite with `litmus run -s suites/example.yaml -a my_agent.py:agent`. The starter suite checks for "hello" and "4"; adapt your agent or replace it with cases for your application. Run `init` in a new directory because it overwrites those starter files.

Assertions cover strings, numbers, regular expressions, JSON, semantic similarity, and LLM grading. `All`, `AnyOf`, `AtLeast`, and `Weighted` combine checks; `Custom` accepts a Python predicate. Semantic and LLM assertions make additional API calls. See the [assertion reference and extension examples](docs/usage.md#assertions) for Python names, YAML types, custom registration, and judge configuration.

## Compare changes and repeat runs

Reuse `agent` and `cases` from the quick start to compare two implementations:

```python
from litmusai import TestSuite, compare, multi_evaluate

suite = TestSuite(name="quickstart", cases=cases)
candidate = Agent.from_function(lambda task: "36", name="candidate")
comparison = asyncio.run(compare({"baseline": agent, "candidate": candidate}, suite))
assert comparison["baseline"].passed == 2
assert comparison["candidate"].passed == 1

repeated = asyncio.run(multi_evaluate(agent, suite, runs=5))
print(repeated.to_table())
repeated.save("repeated.json")
```

`multi_evaluate()` reports mean and standard deviation, each case's reliability, and `flaky_tests` for cases that both pass and fail across repetitions. Results retain every run. Use enough representative cases and repetitions for the decision you are making; a small example is not a model ranking.

## Classification and extraction metrics

Labeled suites report precision, recall and F1 from ground truth without an LLM judge. Classification includes accuracy, per-class counts, micro/macro/weighted averages and a confusion matrix. Extraction matches field values or entity occurrences one-to-one, with optional whitespace normalization and case folding.

From a checkout, run the local examples:

```bash
litmus run -s examples/routing.yaml -a examples/labeled_agents.py:route --runs 3 -o routing.json
litmus run -s examples/extraction.yaml -a examples/labeled_agents.py:extract -o extraction.json
litmus report -r routing.json --html routing.html
```

The routing example deliberately misses one billing request: expected accuracy is 75%, with macro F1 of about 0.733. The extraction example includes a duplicate and a missed address: expected micro precision and recall are both 75%.

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

For run-level gates and history:

```bash
litmus run -s tests.yaml -a my_agent.py:agent --threshold 0.9 --budget 1.00 --save-baseline --log-dir .litmus/logs
litmus run -s tests.yaml -a my_agent.py:agent --baseline .litmus/baseline.json --log-dir .litmus/logs
litmus history --log-dir .litmus/logs --limit 10
litmus badges
```

`--threshold` sets the minimum assertion pass rate. `--budget` checks recorded cost after evaluation; it does not stop spending mid-run. `--baseline` flags aggregate pass-rate drops greater than five percentage points, or cost/latency increases greater than 50%. Use matching suites and run counts when comparing totals. Without an explicit threshold, a profile/config threshold, or a baseline, failed assertions alone do not make `litmus run` exit nonzero.

The CLI also writes JSON or Markdown evaluation summaries:

```bash
litmus run -s tests.yaml -a my_agent.py:agent --format json --output run.json
litmus run -s tests.yaml -a my_agent.py:agent --format markdown --output run.md
```

CLI JSON wraps the same versioned payload as `results.save()` in a status envelope. Both can be loaded by `litmus report` and `litmus diff`. Multi-run files retain every repetition; select individual `run_results` entries for a case-level diff.

Chat adapters calculate cost when both model pricing and complete input/output token counts are available. Missing pricing or usage produces `None` in Python, `null` in JSON, and “Unknown” in reports. Explicit zero costs remain zero. If any execution has an unknown cost, the total is unknown and `--budget` fails with a diagnostic; unknown costs are also excluded from cost comparisons and the overall quality score. These estimates use registered rates; cached tokens and other provider charges may differ from your bill.

When comparing models, save the suite, model parameters, run count, raw results, and pricing assumptions. The [usage guide](docs/usage.md#cost-and-quality-dimensions) covers custom pricing, `CostTracker`, `CostGuard`, and the seven scoring dimensions available through `--dimensions` and `DimensionBudget`.

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

The runner passes conversation history to the agent. Custom functions must accept and use the `history` keyword argument. Use `agent.conversation()` for an interactive Python session, or `load_multi_turn_suite(path)` and `ConversationRunner.run_suite(cases)` for YAML conversations. Multi-turn suites use the Python runner, not `litmus run`. A failed agent call fails its conversation step, including a step with no assertions. The `is_cascade` flag marks failures after the first failure; it does not establish that an earlier mistake caused a later one. Context maintenance uses phrase matching and can misclassify legitimate clarification requests.

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

Memory-poisoning scans are available through the Python API and need a history-aware agent. `litmus scan` runs the safety scanner; use `--categories prompt_injection,jailbreak` to select attack categories and `--output safety.json` to save findings.

Failed agent calls retain their error details and count as failed checks. If any finding contains an agent error, the scan verdict is `INCONCLUSIVE`; `safety.is_safe` and `poisoning.is_resistant` are false. Pipeline summaries preserve this verdict. The CLI flag below exits with code 1 for an unsafe or inconclusive safety scan:

```bash
litmus scan --agent my_agent.py:agent --level thorough --fail-on-unsafe
```

## Ground truth

Define an expected answer and record its source. Save this as `science.yaml`:

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

```bash
litmus validate-ground-truth science.yaml
litmus ground-truth-stats --suite science.yaml --ground-truth science.yaml
```

Ground truth can also live in a separate file with the same case IDs. Load it with `load_ground_truth()` and attach it using `apply_ground_truth()`. Supported answer types are text, numeric, JSON, boolean, list, and subjective; subjective checks use an LLM grader.

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

With multiple runs, CLI summaries, threshold checks, and budget checks pool all repetitions. `PipelineResult.eval`, pipeline thresholds, and generated reports also use pooled results. Pipeline case-level baseline diffs compare repetition 1 on each side; `PipelineResult.passed` reflects the threshold and optional safety verdict, so inspect `baseline_diff.regressions` separately when using a pipeline baseline as a gate. Precision, recall, and F1 are informational until named metric gates are implemented.

Custom profiles live in `.litmus/profiles/`. See [configuration, retries, and tracing](docs/usage.md#configuration-and-profiles) for examples and the settings each entry point actually applies.

Datasets can retain an ID, revision, content fingerprint, structured inputs, and source example/trace IDs. These fields survive Python and CLI evaluation, saved JSON, and report export. See the [dataset contract and local example](docs/datasets.md).

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
      - uses: kutanti/litmusai@v1.1.0
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

## CLI reference

Run `litmus --help` or `litmus <command> --help` for available options.

| Command | Purpose |
|---|---|
| `init` | Write starter configuration and a YAML suite |
| `run` | Evaluate, repeat runs, apply gates, save results or a baseline |
| `suites`, `profiles` | List built-in suites and available evaluation presets |
| `report` | Render saved results as HTML, JUnit XML, CSV, or Markdown |
| `diff` | Compare individual cases in two saved evaluations |
| `history` | List saved evaluations from a log directory |
| `scan` | Run heuristic safety checks |
| `validate-ground-truth`, `ground-truth-stats` | Check answer files and label coverage |
| `badges` | Print a pass-rate badge from `.litmus/baseline.json` |
| `runtime serve` | Start the experimental runtime collector and workers from a config file |
| `runtime validate` | Validate runtime policy and delivery configuration |
| `runtime status` | Inspect project coverage, queue lag, delivery state, and latency |
| `runtime alerts` | List the project's runtime alerts |

`dashboard` and `create-test` are placeholders: the former prints report guidance; the latter only prints a message and does not save a test. Use HTML reports and edit YAML suites directly. The OpenAI Agents SDK adapter needs compatibility work; use a tested function wrapper as described in [adapters](docs/adapters.md#6-openai-agents-sdk-from_openai_agent).

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

See [CONTRIBUTING.md](CONTRIBUTING.md) for the review process and [feature validation](docs/feature-validation.md) for the tested workflows and remaining limits. The [earlier project review](docs/project-review.md) records historical findings, some of which have since been fixed. Licensed under [MIT](LICENSE).
