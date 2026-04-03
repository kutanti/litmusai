# Changelog

All notable changes to LitmusAI will be documented in this file.

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

[0.1.0]: https://github.com/kutanti/litmusai/releases/tag/v0.1.0
