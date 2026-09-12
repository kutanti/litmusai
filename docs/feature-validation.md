# Feature validation

This audit covers the repository on September 11, 2026, using Windows and Python 3.11.4. It checks implementation behavior with deterministic agents and mocked provider responses. It is not a benchmark of a model's quality.

## Scope and results

The final suite passed **1,140 tests**, with **7 live-provider tests skipped** because neither OpenAI nor LiteLLM credentials were configured. Statement coverage is **92.4%** across `litmusai` with full JSON Schema support installed. The initial base-dependency run passed 1,132 tests; eight regression cases were added during this audit. Ruff, strict mypy checks across 44 source files, and source/wheel builds passed. A clean wheel installation loaded all eight suites and 50 packaged cases, ran an evaluation, exposed `litmus --version`, and passed `pip check`.

The audit also passed **44 CLI invocations**, the README's local examples, all eight built-in suites, five local example scripts, and the usage guide's local Python/YAML examples. The table distinguishes local execution from mocked integrations and unfinished features. Deliberately simple agents can fail suite assertions; these workflow checks verify that LitmusAI records and reports those outcomes correctly.

| Feature | Validation |
|---|---|
| Agent execution | Sync and async functions, structured responses, errors, object methods, subprocess stdin/stdout, and metadata normalization |
| HTTP adapters | Mocked POST request/body/header handling, response fields, usage, tool calls, and HTTP failures; no external endpoint used |
| Chat and Azure adapters | Mocked request parameters, authentication, history, token parsing, cost estimates, and provider failures |
| Framework adapters | LangChain and CrewAI behavior tested with stand-in objects; actual installed framework releases not certified |
| Assertions | String, number, regex, JSON, composite, custom, semantic, and rubric checks; both lightweight and installed-`jsonschema` paths exercised across audit runs |
| LLM judges | Criteria, normalization, parsing, caching, and failures checked using fake providers; no live judge calls |
| Suites and ground truth | All eight built-in suites loaded and evaluated; YAML, stable IDs, provenance, automatic assertions, validation, and coverage commands |
| Labeled metrics | Classification and extraction examples, pooling, malformed predictions, error counts, confusion matrices, normalization, and persistence |
| Repeated runs and comparison | README comparison example, repeated-run statistics, saved repetitions, flaky detection, and regression diffs |
| Scoring and cost | Seven dimensions, budgets, pricing registration, cost trackers, guards, and model-comparison exports |
| Conversations | History, per-step assertions, failure propagation, stop-on-failure, and multi-turn suite loading |
| Safety and memory poisoning | All three scan depths with local agents; failing agents retain errors and produce inconclusive verdicts |
| Pipelines and profiles | Evaluation, safety, HTML/JUnit/CSV reporting, pooled results, profile defaults, and custom profile loading |
| CLI workflow | Initialization, evaluation formats, thresholds, budget checks, baselines, history, badges, reports, diffs, and ground-truth commands |
| Persistence and reporting | JSON round trips, legacy result readers, multi-run identities, Markdown, HTML, CSV, and parsed JUnit XML; browser checks confirmed filtering, sorting, and expandable prediction evidence |
| Extensions | Assertion registration, retry behavior, nested tracing, error capture, and trace export |
| GitHub Action | Local tests of input handling, subprocess invocation, output forwarding, failures, and baseline behavior; no hosted workflow or PR comment sent |
| Distribution | Source archive and wheel build; packaged suites and the CLI checked separately from the editable checkout |

## Fixes found during the audit

- `litmus init` printed a command using `-s example`, although custom suites require a file path. The command now points to `suites/example.yaml` and `my_agent.py:agent`; a regression test runs the printed command against the generated suite.
- Saving a Markdown report with a baseline comparison failed on Windows when arrows or non-Western text could not be encoded by the default code page. CLI Markdown files now explicitly use UTF-8, with a Unicode regression test.
- The model-comparison example passed a list to `compare()`, which expects a mapping of names to agents. It also passed `None` as the base URL when the environment variable was absent. Both are corrected and exercised against mocked chat responses.
- The adapter demo used Unix `cat`. It now runs a Python subprocess using the current interpreter, with a test that verifies every echoed task succeeds.

The scan help text also now uses the accepted `prompt_injection` category name. Additional HTTP tests check actual request and response handling, beyond adapter construction.

The README now documents comparison, repeated runs, CLI gates, ground-truth commands, feature boundaries, and extension APIs. Its old warning that multi-run gates inspect only one run was stale: current CLI and pipeline thresholds pool repetitions, and CLI cost checks include their combined recorded cost.

## Remaining limits

- **Live services:** seven provider integration tests remain unexecuted without credentials. Request mocks do not establish compatibility with every deployed model, endpoint, or SDK version.
- **Unfinished commands:** `create-test` only prints a message and writes no test. `dashboard` prints guidance for HTML reports. Neither is advertised as implemented functionality.
- **OpenAI Agents SDK:** `from_openai_agent` still assumes an incompatible import/response shape. Use a tested function wrapper; see [adapters](adapters.md#6-openai-agents-sdk-from_openai_agent).
- **Gates:** CLI baseline comparisons use aggregate pass rate, cost, and latency. `litmus diff` compares case IDs and requires selecting a repetition for multi-run data. Pipeline baseline diffs compare repetition 1, and `PipelineResult.passed` does not include baseline regressions.
- **Costs:** estimates depend on available pricing and reported usage. Judge/embedding costs are not included in agent totals. Budget gates run after evaluation and do not enforce a spending cap during requests.
- **Configuration:** CLI profile/config support is partial; model parameters, timeout, scan settings, and pricing may require explicit API configuration. See the [usage guide](usage.md#configuration-and-profiles).
- **Heuristics:** safety, memory-poisoning, context, and cascade checks report behavior under selected checks; they do not establish general safety or causal explanations.

## Reproduce

From a checkout, use an isolated environment and install the development dependencies:

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/ -q -ra
python -m ruff check src/ tests/ scripts/
python -m mypy src/litmusai/ --ignore-missing-imports
python -m pip install pytest-cov jsonschema build
python -m pytest tests/ -q --cov=litmusai --cov-report=term-missing
python -m build
```

The base run exercises the built-in JSON Schema fallback if `jsonschema` is absent. The second run exercises full schema validation. Follow the local examples in the [README](../README.md), and run the numbered examples `01`, `03`, `04`, and `05` plus `examples/quickstart.py` without credentials. Example `02` requires provider credentials for real model calls; its automated regression test uses HTTP mocks.

The existing pytest warnings concern imported data classes with `Test` prefixes and deprecated Click test helpers; they are separate from evaluation failures. Linux and other Python versions are covered by the repository's CI matrix, not by this local Windows audit.
