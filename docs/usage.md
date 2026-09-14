# LitmusAI usage guide

Start with the [README quick start](../README.md#quick-start). This guide covers the APIs behind the feature overview. Examples with `await` belong inside an async function or a notebook that supports it.

## Assertions

Import these checks from `litmusai`:

| Python API | YAML `type` | Checks |
|---|---|---|
| `Exact("ready")` | `exact` with `value` | Text equality; case-insensitive and stripped by default |
| `Contains(["return", "refund"], mode="any")` | `contains` with `patterns`, `mode` | All or any expected strings |
| `NotContains(["secret"])` | `not_contains` with `patterns` | Absence of strings |
| `Numeric(36, tolerance=0.01)` | `numeric` with `value`, `tolerance` | A number extracted from the response |
| `RegexMatch(r"ORD-\d+")` | `regex` with `pattern` | Regular expression match |
| `JsonValid()` | `json_valid` | Parseable JSON, including supported fenced output |
| `JsonSchema({"type": "object"})` | `json_schema` with `schema` | JSON Schema; install `jsonschema` for full validation |
| `JsonPath("$.status", expected="ok")` | `json_path` with `path`, `expected` | A value at a supported JSON path |
| `All(check1, check2)` | `all` with nested `assertions` | Every child passes |
| `AnyOf(check1, check2)` | `any_of` with nested `assertions` | At least one child passes |
| `AtLeast(2, [check1, check2, check3])` | Python only | At least N children pass |
| `Weighted([(check1, 0.7), (check2, 0.3)])` | Python only | Weighted child scores reach a threshold |
| `Custom(lambda text: len(text) < 200)` | Python only | Your own predicate or scoring function |
| `Semantic("expected answer", api_key=...)` | Python only | Embedding similarity |
| `LLMGrade("Explain the refund policy", api_key=...)` | Python only | An LLM grades against a rubric |

Semantic checks and LLM grading require a configured provider and make additional requests. The built-in YAML registry supports the types listed above; it does not automatically expose every Python assertion. `TestSuite.to_yaml()` omits executable assertion objects, so retain your original YAML or Python suite as the test definition.

For a reusable YAML check, subclass `Assertion` and register it before loading the suite:

```python
from litmusai import Assertion, AssertionResult
from litmusai.assertions import register_assertion


class MaxLength(Assertion):
    def __init__(self, limit: int = 200):
        self.limit = limit

    def check(self, response: str, *, context=None):
        passed = len(response) <= self.limit
        return AssertionResult(
            passed=passed,
            score=float(passed),
            reason=f"{len(response)} characters; limit {self.limit}",
        )


register_assertion("max_length", MaxLength)
```

```yaml
assertions:
  - type: max_length
    limit: 200
```

For CLI use, place the registration in your agent module or import it there. The CLI imports that module before parsing the suite.

## LLM judges

Use `LLMGrade` for a rubric assertion on a case. For multiple criteria, `litmusai.scorers.LLMJudge` supports custom criteria, predefined metrics, a configurable score range and pass threshold, and an in-memory prompt cache:

```python
import os
from litmusai.scorers import LLMJudge

judge = LLMJudge(
    model="gpt-4o-mini",
    api_key=os.environ["OPENAI_API_KEY"],
    criteria={
        "correctness": "Does the response agree with the expected answer?",
        "completeness": "Does it address every part of the task?",
    },
    score_range=(1, 5),
    pass_threshold=0.75,
)
# case is a TestCase; response is an AgentResponse.
judgment = await judge.evaluate(case, response)
print(judgment.normalized_score)
```

`FunctionProvider` lets you supply your own judge callable. Judge scores depend on the chosen model and rubric. Judge and embedding request costs are not included in the evaluated agent's token and cost totals.

## Cost and quality dimensions

`evaluate()` computes correctness, completeness, format, relevance, safety, latency, and cost dimensions. `results.avg_dimensions` contains their averages; each result's score vector includes explanations. These dimensions combine assertion results and heuristics; a high safety dimension is not a safety-scan verdict.

```python
from litmusai import DimensionBudget, evaluate

results = await evaluate(
    agent, suite,
    dimension_budget=DimensionBudget(
        latency_ms=1000, latency_max_ms=5000,
        cost_usd=0.005, cost_max_usd=0.05,
    ),
)
print(results.avg_dimensions.to_dict())
```

Values at or below a target score 1; values at or above its maximum score 0; intermediate values interpolate linearly. Budgets here affect scores, not request scheduling or CLI gates. `ScoreVector.compute_overall(weights=...)` supports custom weights. For a terminal summary, run:

```bash
litmus run -s tests.yaml -a my_agent.py:agent --dimensions
```

An unavailable cost is `None` in Python and `null` in JSON. The cost dimension is
also `None` and is excluded when weighting the overall score. If any repetition
has an unknown cost, the pooled cost and cost dimension are unknown. Reports show
“Unknown”; CSV, JUnit, and the GitHub Action's `total-cost` output use `unknown`.
`--budget` fails when a total cannot be estimated, and its JSON status envelope
includes a `budget_check` object with the limit, total, verdict, and reason.

Custom functions and HTTP endpoints must supply a cost to claim a priced run.
Use `AgentResponse(output="...", cost=0.0)` or `{"output": "...", "cost": 0.0}` for
an explicitly free execution. Chat adapters require registered pricing and both
input and output token counts; absent, partial, or invalid usage is unknown.
Explicit zero token counts with registered pricing remain a known zero cost.

Code that formats or adds costs must now handle `None`. Saved numeric costs from
older versions remain readable, including zero; their historical availability
cannot be reconstructed. Re-run evaluations to obtain the corrected estimates.

Custom deployment names may not match bundled pricing. Register your verified rates before evaluating, in USD per million tokens:

```python
from litmusai.benchmarks import register_pricing

# Illustrative rates; replace these with your provider's applicable prices.
register_pricing("my-deployment", input_cost_per_m=1.0, output_cost_per_m=4.0)
```

`CostTracker` records per-task usage, cost per successful task, and latency percentiles. `CostGuard` checks a tracker against cost, token, and latency limits and returns alerts. `compare_models(*trackers)` produces Markdown, JSON, and CSV comparisons. These are explicit Python utilities; registering a guard does not automatically attach it to `evaluate()`.

`CostGuard` returns an error when a configured cost limit cannot be checked due
to an unknown estimate. Cost-based model recommendations exclude unpriced runs;
pass rate and latency remain available. Baseline comparisons report an unavailable
cost delta when either total is unknown, while still comparing pass rate and latency.

## Configuration and profiles

The CLI searches the working directory and parents for `.litmus/config.yaml`, `.litmus/config.yml`, `litmus.yaml`, or `litmus.yml`:

```yaml
version: 1
defaults:
  concurrency: 5
  runs: 1
  threshold: 0.8
  budget: 1.0
  log_dir: .litmus/logs
```

Explicit CLI options override profiles, which override configuration defaults for run count, concurrency, and threshold. Built-in profiles are `quick`, `thorough`, `benchmark`, `safety`, and `ci`. Custom profile YAML files in `.litmus/profiles/` use fields such as:

```yaml
name: release
description: Repeated release evaluation with safety checks
runs: 3
concurrency: 2
threshold: 0.9
safety: true
safety_depth: thorough
report: html
```

Run `litmus run --profile release ...` for its CLI evaluation settings. To apply safety and report settings as well, explicitly load profiles and use a pipeline:

```python
from litmusai import Pipeline, get_profile
from litmusai.profiles import load_profiles_from_dir

load_profiles_from_dir()
profile = get_profile("release")
result = await Pipeline(agent, suite, **profile.to_kwargs()).run()
```

Profiles may recommend temperature and seed, but do not apply them to an agent. Configure those on the adapter. Likewise, the current CLI does not apply configuration-file `timeout`, `verbose`, `safety`, or `pricing` sections to evaluation; set agent timeouts, run scans, and register pricing explicitly.

`litmusai.configure()` supplies defaults to semantic/LLM assertions, and the Azure adapter can use its API key as a fallback. `Agent.from_openai_chat()` requires explicit `api_key` and `base_url` arguments for non-default settings. Global configuration does not configure every adapter or the standalone `LLMJudge`.

## Retries and tracing

Retries and spans are optional Python utilities. Wrap the underlying function that raises errors before adapting it:

```python
from litmusai import Agent
from litmusai.retry import RetryConfig, with_retry
from litmusai.tracing import Tracer

tracer = Tracer("refund-agent")
retry = RetryConfig(max_retries=2, retry_on=(TimeoutError,))


async def resilient_answer(task: str, **kwargs):
    with tracer.span("answer") as span:
        # call_provider is your sync or async function; it raises on failure.
        response = await with_retry(call_provider, task, config=retry, **kwargs)
        span.set_attribute("completed", True)
        return response


agent = Agent.from_function(resilient_answer, name="refund-agent")
# After calling the agent:
# tracer.save("trace.json")
```

`with_retry()` retries matching raised exceptions using exponential backoff and optional jitter. `Agent.run()` converts exceptions into failed `AgentResponse` objects, so wrapping `agent.run` itself does not retry those failures. Tracing supports nested spans, attributes, error capture, JSON export, and summaries. Use separate tracers for concurrent tasks; a tracer's nesting stack is shared within that instance.
