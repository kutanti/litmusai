# Changelog

All notable changes to LitmusAI will be documented in this file.

## [Unreleased]

### Fixed

- The command printed by `litmus init` now uses the generated suite's file path and a file-based agent reference.
- CLI Markdown exports use UTF-8, including baseline comparison arrows and Unicode suite names on Windows.
- The model-comparison example supplies a name-to-agent mapping and a default chat API URL.
- The adapter demo uses a portable Python echo subprocess instead of Unix `cat`.

### Documentation

- Added a feature overview, comparison examples, a CLI reference, an extended usage guide, and a feature-validation report.
- Corrected stale README descriptions of multi-run gates and documented unfinished commands and adapter limits.

## [0.4.0] - 2026-09-10

### Added

- Memory-poisoning scans with 18 conversation attacks across six categories. `MemoryPoisonScanner` supports basic, standard, and thorough depths and returns findings with per-attack results.
- Windows CI coverage alongside Linux tests for Python 3.10, 3.11, and 3.12.
- CI checks for emojis in new PR titles and commit messages. Existing commit history is preserved.

### Changed

- `evaluate()` accepts a list of test cases, and test-case names default to their IDs.
- Documentation uses plain language, runnable local examples, and explicit limits for cost estimates, scan heuristics, result formats, and multi-run gates.

### Fixed

- Failed agent calls remain failed in safety scans, memory scans, and conversation steps, including steps without assertions. Scan reports and pipeline summaries show `INCONCLUSIVE` when agent errors prevent a verdict.
- Standard and thorough memory scans no longer crash on Python 3.11 and newer because of repeated inline regex flags.
- GitHub Action inputs are passed as literal arguments, outputs are exposed to later steps, and stale result files are cleared. The action installs the selected action revision and preserves evaluation exit codes.
- CLI run count, concurrency, and threshold follow explicit options, then profile settings, then configuration defaults. CLI run counts and concurrency reject values below one.
- Starter assertions check meaningful answers, and the CLI version matches the package version.
- HTML report encoding, elapsed-time measurements, and platform-dependent tests work on Windows.

### Known limitations

- Multi-run CLI gates and cost checks use the last run; `Pipeline` uses the first run for its primary result and threshold.
- CLI JSON summaries and Python-saved results have different schemas. Use Python-saved results for report exports and case-level diffs.

## [0.3.0] - 2026-04-07

### Added
- **Multi-turn conversation evaluation** — `MultiTurnCase`, `Step`, `ConversationRunner` for testing multi-step agent workflows. Error compounding detection distinguishes cascade vs independent failures. YAML support for multi-turn suites.
- **Conversation context manager** — `agent.conversation()` / `Conversation(agent)` maintains message history across turns. Works with OpenAI, Azure, and custom agents.
- **Ground truth management** — `GroundTruth` dataclass with 6 answer types (text, numeric, json, boolean, list, subjective). Provenance metadata (source, verified_by, date, confidence). Auto-generates assertions from ground truth entries.
- **Ground truth CLI** — `litmus validate-ground-truth` and `litmus ground-truth-stats` commands
- **YAML ground_truth key** — define ground truth inline in suite YAML files; assertions auto-generated
- **Model params logging** — `Agent.model_params` tracks temperature, max_tokens, seed. Logged in evaluation results JSON.
- **Seed parameter** — `Agent.from_openai_chat(seed=42)` for reproducible outputs
- **Benchmark profile defaults** — temperature=0.0, seed=42 for reproducibility
- **Multi-dimensional scoring** — 7 quality dimensions (correctness, completeness, format, relevance, safety, latency, cost) with configurable weights. SVG radar chart in HTML reports.
- **`--dimensions` CLI flag** — show per-dimension scores in table output
- **`DimensionBudget`** — configurable latency/cost thresholds for scoring
- **`evaluate(dimension_budget=...)` kwarg** — custom budgets per evaluation

### Changed
- Updated README examples for the public API
- OpenAI and Azure adapters support conversation history via `history` kwarg
- Duplicate system messages prevented when using conversation history with agent-level system prompts
- Profile display shows temperature/seed when set

### Fixed
- CLI version display
- Profile field validation (concurrency≥1, runs≥1, etc.)

## [0.2.1] - 2026-04-06

### Added
- **Pipeline class** — `Pipeline(agent, suite, safety=True, runs=3, report="html").run()` chains eval + safety + report in one call. Returns `PipelineResult` with `summary()`, threshold checking, and combined results
- **Evaluation Profiles** — 5 built-in presets: `quick`, `thorough`, `benchmark`, `safety`, `ci`. CLI: `litmus run --profile thorough`. Custom YAML profiles in `.litmus/profiles/`
- **`litmus profiles`** CLI command — list all available profiles with descriptions and settings
- **`EvalProfile` dataclass** — frozen, validated, with `to_kwargs()` for Pipeline integration

### Changed
- Added README examples for pipelines and evaluation profiles

### Fixed
- CLI version display now correctly shows package version
- Profile fields validated on load (concurrency≥1, runs≥1, threshold 0-1, valid safety_depth/report)

## [0.2.0] - 2026-04-04

### Added
- **Azure OpenAI Support** — `Agent.from_azure(resource, deployment, api_key)` with correct deployment-scoped URLs, `api-version` query params, and `api-key` header auth
- **Global Configuration** — `litmusai.configure(api_key, base_url, auth_style)` sets defaults for all assertions and agents. Key resolution: instance param → global config → env var
- **Retry Logic** — `RetryConfig` + `with_retry()` with exponential backoff, jitter, configurable max retries and retriable exceptions
- **Tracing** — `Tracer` with context-manager spans, nested span support, timing, attributes, JSON export, and summary reports
- **Assertion Plugins** — `register_assertion("name", MyAssertion)` for custom assertion types, auto-discovered in YAML suite parsing
- **YAML Assertions** — Define assertions directly in suite YAML files (`contains`, `not_contains`, `exact`, `numeric`, `regex`, `json_valid`, `json_schema`, `json_path`, `all`, `any_of`)
- **JUnit XML Export** — `litmus report --junit results.xml` for CI integration
- **CSV Export** — `litmus report --csv results.csv` for spreadsheet analysis
- **HTML Report** — Interactive dark-theme report with sorting, filtering, expandable details, XSS protection
- **Config File Support** — `.litmus/config.yaml` with CLI override detection via `click.ParameterSource`
- **Integration Tests** — Real API tests (skipped without credentials)
- **4 New Built-in Suites** — `customer_support` (8 cases), `summarization` (5 cases), `instruction_following` (9 cases), `tool_use` (6 cases) — 50 total cases across 8 suites

### Changed
- **Better Error Messages** — Semantic and LLMGrade now validate API keys at construction with actionable messages mentioning `litmusai.configure()`, direct `api_key=`, and env vars
- **Semantic/LLMGrade Azure Support** — Both support `auth_style="azure"`, `api_version` param, and merge global `extra_headers`
- **Contains assertion** — YAML parser wraps single `value` in list, supports `patterns` key
- **Exact assertion** — Default `case_sensitive=False` matches class behavior

### Fixed
- Non-string assertion `type` in YAML now raises clear `ValueError` instead of `AttributeError`
- Non-list `assertions` field in YAML now raises `ValueError` instead of silent skip
- JUnit XML now includes `<?xml?>` declaration with UTF-8 encoding
- CLI `--baseline` flag works correctly with export flags

### Infrastructure
- 641+ tests (up from 477), mypy strict, ruff clean
- 32 source files, ~10,500 LOC
- Python 3.10, 3.11, 3.12 support

## [0.1.0] - 2026-04-03

### Added
- **Agent Adapter** — 7 factory methods (`from_function`, `from_openai_chat`, `from_langchain`, `from_litellm`, `from_anthropic`, `from_crewai`, `from_http`) normalizing to `AgentResponse`
- **Assertion Engine** — 11 assertion types: `Contains`, `NotContains`, `Equals`, `Numeric`, `Regex`, `JsonMatch`, `JsonSchema`, `Semantic`, `LLMGrade`, `AnyOf`, `AllOf`
- **LLM-as-Judge Scorer** — Configurable judge with custom rubrics and 1-5 scale grading
- **Cost Benchmarking** — Token tracking, model pricing registry, cost-per-eval with fuzzy model matching
- **Safety Scanner** — 46 attack prompts across 7 categories (injection, jailbreak, PII leak, hallucination, bias, overreliance, toxicity) with severity-weighted scoring
- **CI/CD Integration** — GitHub Action, PR comment posting, baseline comparison, regression detection
- **Scoring Pipeline** — Assertions + legacy scoring wired into `evaluate()` with async support
- **Result Logging** — Save, load, and diff evaluation runs with regression/improvement tracking
- **Multi-Run Statistics** — `multi_evaluate()` with mean ± std dev, flaky test detection, reliability classification
- **CLI** — `litmus run`, `litmus diff`, `litmus history`, `litmus scan`, `litmus init`, `litmus suites`, `litmus badges`
- **Built-in Test Suites** — Coding, research, safety, planning

### Infrastructure
- 477+ tests, mypy strict, ruff lint
- Python 3.10, 3.11, 3.12 support
- GitHub Actions CI (lint + test + type check)
- Copilot auto-review on PRs

[0.4.0]: https://github.com/kutanti/litmusai/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/kutanti/litmusai/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/kutanti/litmusai/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/kutanti/litmusai/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/kutanti/litmusai/releases/tag/v0.1.0
