# Labeled evaluation data

Labeled suites use `ground_truth.answer` for expected values and a suite-level
`metrics` configuration for task type and matching rules. Explicit assertions
remain independent of labeled metrics. Ground truth is retained even when a case
has assertions. Python cases accept `GroundTruth` directly; the older
`metadata["ground_truth"]` representation remains readable.
YAML ground truth requires a non-null answer unless its type is `subjective`.
`apply_ground_truth` retains labels and metadata even when explicit assertions
already exist; its return value counts only cases with newly generated assertions.

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
CLI JSON output (`--format json` or a `.json` `--output` path) retains the same
versioned fields inside its existing `results` wrapper, including every repetition
for `--runs`. Legacy CLI aliases such as `agent`, `suite`, `test`, `reason`, and
`output` remain available alongside canonical fields. Human-readable reports,
baseline comparisons, and threshold/budget checks still use the last run.

`litmusai.metrics.Observation` is the versioned record used by task metrics. It
retains case ID, evaluation ID, repetition, task type, expected and predicted
values, execution/validation status, error, and per-case evidence. Its JSON
representation can be read using `Observation.model_validate_json(...)`.
The identity tuple is `(evaluation_id, repetition, case_id)`.

Legacy suites without a version remain accepted. Unsupported explicit versions
are rejected. `TestSuite.to_yaml()` preserves the metric configuration and ground
truth; it still does not serialize Python assertion objects.

This contract is the prerequisite for labeled metrics in issue #100. Dataset
revisions, content fingerprints, external dataset identities, migration of older
CLI result files, and a complete dataset/result interchange format remain in
issue #99.
