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
fields and the earlier CLI aliases `agent`, `suite`, and `test`.

`litmusai.metrics.Observation` is the versioned record used by task metrics. It
retains case ID, evaluation ID, repetition, task type, expected and predicted
values, execution/validation status, error, and per-case evidence. Its JSON
representation can be read using `Observation.model_validate_json(...)`.
The identity tuple is `(evaluation_id, repetition, case_id)`.

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

This contract is the prerequisite for labeled metrics in issue #100. Dataset
revisions, content fingerprints, external dataset identities, migration of older
CLI result files, and a complete dataset/result interchange format remain in
issue #99.
