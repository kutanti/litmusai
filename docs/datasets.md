# Dataset and result contract

Python suites, YAML/JSON datasets, CLI evaluations, and saved results share
schema version `1.0`. Dataset provenance fields are additive: older files remain
readable, and absent external identities remain absent. No provider SDK is
required.

## Run a local example

```bash
litmus run -s examples/provenance.yaml -a examples/provenance_agent.py:route --runs 3 --format json -o results.json
litmus report -r results.json --html report.html --junit results.xml --csv results.csv
```

This runs six case executions without an API key. The results retain both case
IDs, all repetition IDs, the dataset revision, a content fingerprint, structured
inputs, expected labels, and source references.

Datasets can also be converted for offline use:

```python
from litmusai import TestSuite

suite = TestSuite.from_yaml("examples/provenance.yaml")
suite.to_json("dataset.json")
restored = TestSuite.from_json("dataset.json")
assert restored.to_dict() == suite.to_dict()
```

`litmus run -s dataset.json ...` accepts the same exported dataset. Loading
rejects duplicate or blank case IDs, malformed labels, unsupported schema
versions, invalid source references, and values that cannot round-trip through
JSON. JSON mappings require string keys. NaN, infinity, cycles, tuples, and
arbitrary Python objects are rejected with a field or case diagnostic.

## Create a dataset in Python

```python
from litmusai import DatasetInfo, GroundTruth, MetricConfig, SourceReference, TestCase, TestSuite

suite = TestSuite(
    "support",
    cases=[TestCase(
        id="example-1",
        task="Route the support request",
        inputs={"text": "Please refund my purchase", "priority": 0},
        ground_truth=GroundTruth(answer="billing"),
        metadata={"split": "test"},
        source=SourceReference(provider="local", example_id="example-1"),
    )],
    metrics=MetricConfig(task_type="classification", labels=["billing", "technical"]),
    dataset=DatasetInfo(id="support", revision="v2", metadata={"owner": "support-team"}),
)
```

When `inputs` is a mapping, the runner calls `agent.run(task, inputs=...)` with a
copy. The agent function must accept the `inputs` keyword. An empty mapping is
still an explicit structured input. With `inputs=None`, or an older case that
omits it, the runner calls `agent.run(task)` as before. Keys inside `inputs` are
not expanded into function arguments.

The chat completions, Azure, OpenAI Agents SDK, and CLI adapters accept text
tasks. They reject structured `inputs`, including an empty mapping, before any
external call. Use a function wrapper to define how the case data reaches the
agent. See [adapter input support](adapters.md#structured-inputs).

## Dataset fields

The serialized dataset has `schema_version`, `name`, `description`, `task_type`,
`dataset`, `cases`, and optional `metrics`. `TestSuite.to_dict()` and `from_dict()` use this
same representation as YAML and JSON.

| Field | Meaning |
| --- | --- |
| `task_type` | `classification` or `extraction` with matching metric configuration; `assertion` otherwise. Inferred when absent in older suites. |
| `dataset.id` | Explicit dataset identity, or `null` when none is known. The suite name is a display name. |
| `dataset.revision` | Explicit revision from the producer, retained verbatim, or `null`. |
| `dataset.fingerprint` | SHA-256 of the local case data and metric configuration. Computed when exporting or evaluating. |
| `dataset.metadata` | JSON-compatible dataset metadata. |
| `dataset.source` | Optional original provider reference. |
| `cases[].id` | Stable, nonempty case ID, unique within the dataset. |
| `cases[].task` | Text prompt or instruction. |
| `cases[].inputs` | Structured input mapping, or `null`. |
| `cases[].ground_truth` | Expected answer/label and its ground-truth provenance. Existing legacy expected fields are also retained. |
| `cases[].metadata` | JSON-compatible per-case metadata, including empty or false values. |
| `cases[].source` | Original provider, dataset, example, and source trace identities. |
| `metrics` | Dataset task type and labeled metric configuration. Absent for an assertion-only suite. |

`SourceReference` reserves `provider`, `dataset_id`, `example_id`, and `trace_id`.
Only `provider` is required. A source trace identifies the original observation;
it is separate from the new local `evaluation_id`. Connectors should retain
these references when translating cases and should not treat a historical model
response as ground truth unless labels have been supplied.

## Fingerprint rules

The fingerprint uses SHA-256 over UTF-8 JSON with sorted mapping keys, compact
separators, unescaped Unicode, and finite JSON numbers. Cases are sorted by their
IDs before hashing, so changing case order or mapping key order does not change
the fingerprint. Array ordering inside an input or label remains significant.

The hashed content contains case IDs, names, tasks, structured inputs, expected
values, ground truth, metadata, source references, tags, and metric configuration.
The legacy metadata ground-truth alias is normalized to the explicit ground-truth
representation. It excludes the dataset name/description, dataset-level
identity/metadata, runtime assertions, scorer selection, and timeout settings.

This is an input-data fingerprint, not a hash of Python evaluator code. Changing
a callable assertion or its configuration does not change the fingerprint;
version evaluator code separately, for example with a Git revision in agent
metadata. Portable expected values should be recorded as ground truth or legacy
expected fields. A supplied external revision stays distinct from the fingerprint.

An exported dataset contains its computed fingerprint. Loading checks that it
matches the content. After intentionally editing an exported file, remove its
old fingerprint and export again to compute the new value. Programmatic edits
recompute the fingerprint on the next export or evaluation.

The runner snapshots dataset data before executing the agent, including across
repetitions. Mutating the original suite or the `inputs` mapping passed to an
agent does not rewrite the saved dataset evidence. Combining results from
different dataset descriptions/revisions/fingerprints raises an error.

## Assertions and portable datasets

YAML and JSON exports preserve `Exact`, `Contains`, `NotContains`, `Numeric`,
`RegexMatch`, `JsonValid`, `JsonSchema`, `JsonPath`, `All`, `AnyOf`, `AtLeast`, and
`Weighted`, including their options. This replaces the older YAML exporter that
silently omitted assertions.

Callable, plugin, and provider-backed assertions remain usable in Python.
Exporting such an assertion as a dataset definition raises an explicit error
instead of saving a different test. Their evaluation outcomes and details still
appear in saved results. Credentials and executable objects are not serialized
from assertion internals.

## Evaluation results and compatibility

Each new evaluation includes a `dataset` description alongside its existing
`evaluation_id`, `repetition`, `config`, `metric_config`, `metrics`, `summary`, and
`results`. The configuration captures model parameters and agent metadata. Every
result row retains case ID, task, structured inputs, expected fields, metadata,
source reference, full response, response metadata, assertion scores/details,
execution status/errors, latency, tokens, and estimated cost. Labeled observations
continue to hold selected predictions and metric evidence.

Multi-run JSON includes pooled rows and each individual evaluation in
`run_results`, all with the same dataset description. CLI JSON retains its status
envelope: `{"results": ..., "success": ..., "has_regression": ...}`. Load either
form with `litmusai.results.load_results()`.

The result reader also accepts older raw Python results and CLI aliases such as
`agent`, `suite`, `test`, `output`, and `reason`. Canonical fields take precedence.
It does not invent missing case, evaluation, repetition, dataset, or source IDs.
In particular, a case-level diff rejects missing case IDs instead of matching
display names. Older response text that was already truncated cannot be restored.
HTML reports explicitly indicate when dataset provenance is unavailable.

HTML displays dataset provenance and per-case inputs/source metadata. CSV adds
dataset ID/revision/fingerprint and JSON-encoded provenance fields.
`dataset_metadata` contains only the dataset's metadata mapping; `dataset_source`
holds its source reference separately from the per-case `source` column. An
explicit dataset with empty metadata exports `{}`; absent dataset descriptions
and absent source references leave their CSV fields blank. JUnit stores the
dataset and case provenance as properties. Saved JSON, CSV response cells,
and JUnit `system-out` preserve full response text; HTML uses a short preview.
Missing or `null` per-case provenance fields also export as blank CSV cells;
explicitly empty mappings, such as `inputs: {}` or `metadata: {}`, export as `{}`.

For a case-level diff of multi-run evaluations, select one run from each
payload's `run_results`. `Pipeline(baseline=...)` selects repetition 1 for
multi-run inputs. Dataset compatibility policies and named metric release gates
remain the scope of issue #103.
