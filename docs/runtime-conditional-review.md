# Conditional runtime evaluation

An optional second stage evaluates uncertain prompt-injection screens and a sample
of clear screens. Local tool/destination/sensitive-data rules and usage policies
continue independently. The first-stage classifier should be a cheap, evaluated
screen; the second endpoint can use a customer-selected deeper LLM evaluator.
Litmus orchestrates both stages but does not bundle a general LLM model.

```yaml
projects:
  - project_id: support
    api_key_env: LITMUS_RUNTIME_API_KEY
    classifier:
      provider: http
      endpoint: https://screen.example/evaluate
      api_key_env: SCREEN_KEY
      version: screen-v1
      calls_per_minute: 600
    review:
      version: gate-v1
      on_outcomes: [needs_review, insufficient_context]
      clear_sample_rate: 0.01
      evaluator:
        provider: http
        endpoint: https://review.example/evaluate
        api_key_env: REVIEW_KEY
        version: review-v1
        timeout_seconds: 2
        calls_per_minute: 30
        context_events: 20
        context_chars: 32000
```

Both endpoints implement the injection-classifier HTTP contract in the
[runtime guide](runtime-threat-alerting.md). Versions must identify the deployed
policy/model/prompt behavior. Change the gate version when selection or evaluator
configuration changes. Trusted configuration supplies these choices; captured
conversation text cannot select providers, budgets, or sampling rules.

## Decisions and failure behavior

- `needs_review` means the screen cannot make a definite decision. Configured
  uncertainty outcomes create a durable second-stage job. `error` may be explicitly
  included for provider-failure fallback; it is not selected by default.
- `clear` results are selected using a stable hash of project, event ID, and gate
  version. A rate of 0 disables audit sampling; 1 selects all clears. Retries and
  restart do not change selection. Sampling measures possible screening misses;
  a clear screen is not a claim that the conversation is safe.
- Definite `detected` screens emit immediately. This feature does not retract those
  alerts. To require deeper evaluation for a tentative match, the screening endpoint
  must return `needs_review`. Lakera's binary positive results remain immediate
  detections; its incomplete results can enter review.
- Screening and review have separate persisted call budgets and independent workers.
  Slow review cannot block local findings, screening, or delivery. Budget skips,
  errors, missing context, and timeouts stay explicit; they never become clear.
- A deeper `needs_review` result becomes `insufficient_context`; there is no recursive
  escalation. A clear result with incomplete context also remains insufficient.
- Selection, the screening finding, and the next job commit atomically. Pending
  review continues after restart using the saved configuration and counts toward
  the pending-event quota. Crashes may repeat a provider call before its result is
  committed; provider calls are not exactly once.
- Both stages receive bounded, source-labelled, redacted context. A second stage
  can use a larger history budget, but cannot recover activity that was never captured.

## Telemetry and event compatibility

The optional response fields below report provider usage. Omit unknown values;
zero means measured zero. Litmus does not infer prices or invent confidence scores.

```json
{
  "outcome": "needs_review",
  "reason": "The supplied instructions have ambiguous intent.",
  "context_incomplete": false,
  "input_tokens": 120,
  "output_tokens": 15,
  "reported_cost_usd": 0.001
}
```

Findings record the selection reason, stage, screening/evaluator/gate versions,
provider-call flag, elapsed time, and optional reported cost/tokens. Alerts from
this configured pipeline have `data.schema_version: "1.2"` and an `evaluation`
object. Existing unconfigured alerts retain their earlier payloads; upgrade strict
consumers before enabling this feature. All transports use the same saved envelope.

`GET /v1/status` exposes cumulative, project-scoped `evaluation_metrics` grouped by
stage, decision, and outcome. `provider_calls` and `cost_samples` make missing cost
coverage explicit; `reported_cost_usd` is only the sum of reported samples, not total
billing. Metrics cover committed findings and exclude calls lost before commit.
They persist across restarts. Per-finding elapsed time measures the evaluation call
path; existing publication latency includes queue and delivery time.

Before production use, evaluate customer-relevant benign and risky fixtures and
report first-stage misses, audit discoveries, escalation rate, provider cost
coverage, and end-to-end latency under the intended load. Contract tests and mocked
verdicts verify orchestration, not the accuracy of either model. General business,
abuse, scope, and grounding policies remain separate features.

Local validation covers 24 new cases, including atomic selection/recovery, saved
configuration after restart, independent budgets, pending-event quotas, continued
local alerts during slow review, worker execution, redaction, incomplete context,
and reported/missing cost. Runtime suite: 115 passed / 3 external checks skipped.
Full suite: 1,407 passed / 10 skipped. Ruff and strict mypy passed.
