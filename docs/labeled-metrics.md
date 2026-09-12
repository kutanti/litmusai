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
