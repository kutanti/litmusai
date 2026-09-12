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
