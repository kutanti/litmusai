# Labeled evaluation data

Labeled suites use `ground_truth.answer` for expected values and a suite-level
`metrics` configuration for task type and matching rules. Explicit assertions
remain independent of labeled metrics. Ground truth is retained even when a case
has assertions. Python cases accept `GroundTruth` directly; the older
`metadata["ground_truth"]` representation remains readable.

```yaml
schema_version: "1.0"
name: routing
metrics:
  task_type: classification
  labels: [billing, support]
  prediction_field: /label
cases:
  - id: invoice
    task: Where is my invoice?
    ground_truth:
      answer: billing
```

`prediction_field` is a JSON Pointer: `/label` selects a property, `/items/0`
selects an array element, and `""` selects the JSON root. Escape literal `/` and
`~` in property names as `~1` and `~0`. Classification without a pointer reads the
literal response text. Extraction always reads JSON.

Saved evaluation results add `schema_version: "1.0"`, `evaluation_id` and
`repetition` to the existing result structure. Repetition numbers start at one;
all runs from `multi_evaluate` share an evaluation ID. Multi-run exports retain
each individual evaluation in `run_results`.

The CLI uses the same serializer for JSON output and saved files. Its status
envelope contains `results`, `success`, and `has_regression`; `results` holds the
versioned Python API payload. Each case result carries its evaluation ID,
repetition, and case ID. Multi-run exports include all case results at the top
level as well as each run in `run_results`. Their summary pools pass/fail counts,
cost, and token usage across repetitions; the pooled `repetition` is `null`.
CLI threshold/budget checks and GitHub Action outputs use this pooled summary.
`--log-dir` saves the same complete result payload without the status envelope.

Reports accept both the canonical `agent_name`, `suite_name`, and `case_name`
fields and the earlier CLI aliases `agent`, `suite`, and `test`. Legacy `reason`
and `output` fields map to `score_reason` and `response`. Normalization preserves
the input and gives canonical fields precedence, including empty values. It also
normalizes entries in `run_results`. Missing IDs in older files remain absent;
case-level diffs require stable `case_id` values and cannot infer them from names.

`Reporter.to_json()` uses the same payload as `EvalResults.to_dict()` and
`MultiRunResults.to_dict()`. `load_results()`, report readers, and Python
HTML/JUnit/CSV exporters accept both
raw result payloads and CLI status envelopes. Unsupported result versions are
rejected. Multi-run HTML reports give every row its own details panel. Case-level
diffs require one repetition from each result's `run_results`; passing pooled
rows with repeated case IDs raises an error instead of discarding earlier runs.
CSV includes full `evaluation_id`, `case_id`, and `repetition` columns without
truncating identities. JUnit testcase names include available case IDs and
repetitions, with all three identity fields also stored as testcase properties.
Markdown/GitHub reports show the case ID and run number alongside each result.

Suite loading and evaluation require nonempty, unique string case IDs, including
for legacy suites and Python case lists. Invalid IDs fail before agent calls.
When exporting a manually constructed `EvalResults`, missing row evaluation IDs
and repetitions inherit the parent values without mutating the row objects.
Explicit conflicting IDs/repetitions, duplicate result identities, and observation
identities that disagree with their result are rejected. Pooled rows require
explicit repetition numbers. Manually assembled `MultiRunResults` must use a
shared evaluation ID and a unique positive repetition for each child run.

`litmusai.metrics.Observation` is the versioned record used by task metrics. It
retains case ID, evaluation ID, repetition, task type, expected and predicted
values, execution/validation status, error, and per-case evidence. Its JSON
representation can be read using `Observation.model_validate_json(...)`.
The identity tuple is `(evaluation_id, repetition, case_id)`. Observation IDs must
contain a non-whitespace character; valid IDs are preserved verbatim.

Observation values must be JSON scalars, lists, or objects with string keys at
every nesting level. Non-string keys are rejected before serialization so keys
such as `1` and `"1"` cannot collide. Tuples, other Python-only values, cycles,
and non-finite numbers are also rejected.

Legacy suites without a version remain accepted. Unsupported explicit versions
are rejected. `TestSuite.to_yaml()` preserves the metric configuration and ground
truth; it still does not serialize Python assertion objects.

An explicit `ground_truth` entry must be a mapping. Non-subjective entries must
have a non-null answer, even when the case has assertions. Valid false, zero,
empty-string, and empty-collection answers are retained. Subjective entries may
omit the answer. `apply_ground_truth()` retains truth and legacy metadata for
every matching case without replacing explicit assertions; its return value
counts cases receiving truth. Metric-enabled suites do not generate implicit
ground-truth assertions.

The same required-answer rule applies to `GroundTruth.from_dict()`,
`load_ground_truth()`, `apply_ground_truth()`, and assertion generation. The
standalone loader rejects explicit malformed truth and duplicate case IDs while
allowing cases that omit `ground_truth`. Applying truth validates all matching
entries before changing any case. Use a `JsonValid` assertion directly when only
JSON syntax matters and there is no expected answer.

Legacy JSON ground truth checks JSON syntax and expected object keys, without
requiring exact JSON equality. Empty object or array answers generate only
`JsonValid`, so a correctly formatted empty response passes and malformed JSON
fails. Other valid JSON responses also pass because there are no content
constraints; alternatives do not add a constraint in this case. Use explicit
assertions or labeled extraction metrics when content equality matters.

This contract is the prerequisite for labeled metrics in issue #100. Dataset
revisions, content fingerprints, external dataset identities, migration of older
CLI result files, and a complete dataset/result interchange format remain in
issue #99.

## Classification rules

Classification is single-label. Labels are explicit, unique, nonempty strings.
The expected value must be a declared label. Responses match exactly; extra
whitespace or different case is a different label. For JSON responses, configure
`prediction_field` (use `""` for a JSON string at the root).

Each correct prediction adds one true positive (TP). A wrong declared label adds
a false positive (FP) to the predicted class and a false negative (FN) to the
expected class. Missing labels, unknown labels, invalid JSON, and execution
failures add an FN to the expected class. They appear in the confusion matrix's
final column, whose predicted label is `null`; they do not invent another class.
Raw invalid values and errors remain in the observation.

Precision is `TP / (TP + FP)`, recall is `TP / (TP + FN)`, and F1 is
`2*TP / (2*TP + FP + FN)`. A zero denominator returns `value: 0` and
`defined: false`. Accuracy includes every attempted case. Prediction coverage,
invalid prediction counts, and execution error counts accompany the scores.
Precision can remain high when an agent abstains; use coverage and recall too.

Micro averages sum class counts before division. Macro averages include every
declared label equally, including classes absent from the dataset. Weighted
averages weight each class by its expected support. Undefined class values
contribute zero; averages list the contributing `undefined_labels` and set
`defined: false` if any contributing class is undefined. An empty observation
list returns no aggregate (`None`).

For offline use, `classification_observation(...)` parses and scores one full
response, and `classification_metrics(observations, config)` recomputes all
counts from observations. Repetitions are pooled before division. Duplicate
observation identities and mixed task types are rejected.

## Extraction rules

Set `metrics.task_type: extraction`. Ground truth and selected JSON predictions
must both be entity lists or both be field mappings:

- Entity lists match whole JSON values, including every property of an entity
  object. The order of entities and object keys does not matter.
- Field mappings match each `(field name, value)` pair. A list-valued field
  contributes one occurrence per list item. An empty list contributes no items.
  Nested objects are matched as whole values; array order inside an entity or
  nested object remains significant.

Each expected occurrence can match only one predicted occurrence. Unmatched
expected items are FN; unmatched predicted items (including extra duplicates)
are FP. A missing field contributes FN and an extra field contributes FP.
`null` is a literal value, distinct from an absent field. Matching preserves JSON
types: `1`, `1.0`, `true`, and `"1"` are distinct.

Matching is exact by default. `normalize_whitespace: true` trims strings and
collapses whitespace runs to one space. `casefold: true` applies Unicode case
folding. These options affect string values recursively, never field names or
object keys. Original values remain available in each observation's `matched`,
`missing`, and `extra` evidence.

Invalid JSON, a missing prediction pointer, the wrong root shape, and execution
errors are invalid predictions. They contribute FN for all expected items and
appear in coverage/error counts. A valid empty list or mapping has full prediction
coverage even if it misses expected items. Non-finite JSON numbers are invalid.

Extraction aggregates pool TP/FP/FN before calculating precision, recall and F1.
`exact_match_accuracy` additionally counts cases with no missing/extra items and
a valid prediction. If expected and predicted are both empty, that case is an
exact match but its precision/recall/F1 denominators are zero and undefined.
Use `extraction_observation(...)` and `extraction_metrics(observations, config)`
for offline scoring.

## Run locally

From a checkout with `pip install -e ".[dev]"`:

```bash
litmus run --suite examples/routing.yaml --agent examples/labeled_agents.py:route --runs 3 --output routing.json
litmus run --suite examples/extraction.yaml --agent examples/labeled_agents.py:extract --output extraction.json
litmus report -r routing.json --html routing.html
```

The examples use ordinary Python functions and make no provider calls. The
keyword router deliberately misses a refund request; the email extractor returns
a duplicate and misses an obfuscated address. Both produce non-perfect metrics.

The Python API uses the same suite configuration:

```python
import asyncio
from litmusai import Agent, GroundTruth, MetricConfig, TestCase, TestSuite, evaluate

suite = TestSuite(
    "routing",
    [TestCase(id="invoice", task="Invoice help", ground_truth=GroundTruth(answer="billing"))],
    metrics=MetricConfig(task_type="classification", labels=["billing", "support"]),
)
result = asyncio.run(evaluate(Agent.from_function(lambda task: "billing"), suite))
print(result.metrics["accuracy"])
print(result.results[0].observation.evidence)
result.save("routing.json")
```

The runner validates all labeled ground truth and unique, nonempty case IDs before
calling the agent. It reads `AgentResponse.output` in full before presentation
truncation. Agents returning structured predictions should return JSON text, or
`AgentResponse(output=json.dumps(prediction))`.

`EvalResults.metrics` and `MultiRunResults.metrics` expose aggregate values and
counts. `TestResult.observation` holds the complete selected prediction and
per-case evidence. Expected values and metric configuration are copied for each
evaluation, so later edits to source ground truth or labels do not change
completed metrics. Automatic log filenames include a unique suffix to retain
evaluations started within the same second. `MultiRunResults.combined` provides a
pooled evaluation, while
`run_results` keeps each repetition. Combined result payloads use `repetition:
null`; individual observations and per-run payloads retain their one-based number.

CLI and pipeline reports use all repetitions. Existing assertion pass-rate and
cost checks also use these pooled results. Metrics are informational: precision,
recall and F1 thresholds, LangSmith connectors, and retrieval metrics are separate
issues. Legacy assertion scores and the `correctness` dimension keep their
existing meaning; neither is classification accuracy. With no assertions or
legacy checks, the existing score checks only whether the output is nonempty.

CLI JSON files retain the outer status envelope (`results`, `success`,
`has_regression`); the inner `results` is the same versioned payload returned by
the Python API. Canonical names are `agent_name`, `suite_name`, `case_name`,
`score_reason`, and `response` in place of the earlier CLI aliases. The response
preview remains truncated; the metric observation's selected values are complete.
`load_results()` reads both Python files and CLI envelopes and rejects unsupported
explicit versions. It does not invent case IDs for older CLI files.
With `--format json`, stdout contains one JSON document; progress messages,
warnings, and save confirmations go to stderr. Agent/suite loading failures emit
`{"success": false, "error": "..."}` and exit with code 1.

To recalculate metrics after a JSON round trip:

```python
from litmusai import MetricConfig, Observation, aggregate_metrics
from litmusai.results import load_results

data = load_results("routing.json")
observations = [Observation.model_validate(row["observation"]) for row in data["results"]]
metrics = aggregate_metrics(observations, MetricConfig.model_validate(data["metric_config"]))
assert metrics == data["metrics"]
```

HTML and CLI summaries include coverage, errors, undefined flags, underlying
counts and class averages; HTML case details include metric evidence. Existing
JUnit/CSV exporters continue to report assertion outcomes. For a case-level diff
of multi-run results, select one entry from each payload's `run_results` first.
The diff rejects repeated case IDs instead of silently selecting one prediction;
statistical baseline comparisons remain follow-up work.

`Pipeline(..., baseline=...)` compares repetition 1 of the current evaluation
with repetition 1 of a multi-run baseline. A single-run baseline is used as
supplied, including legacy result files. Multi-run baselines can be saved logs,
CLI JSON envelopes, or pooled results saved with `PipelineResult.eval.save()`.
Repetition numbers determine the selection even if saved runs are reordered;
missing or ambiguous repetition 1 is rejected. This case-level comparison does
not summarize variation across runs: `PipelineResult.eval`, metrics, pass-rate
thresholds, logs, and reports continue to include all repetitions.
