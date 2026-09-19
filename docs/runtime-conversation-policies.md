# Configurable conversation policies

`conversation_policies` lets a project evaluate versioned rubrics against captured
live activity and recent history. Policies run asynchronously and publish through
the existing webhook, Kafka, and Event Grid outbox. Each has applicability filters,
evidence prerequisites, an evaluator, severity, and optional score threshold.

This is an opt-in policy framework, not a set of prevalidated detectors. Configure
an evaluated HTTP policy service before enabling a rubric. That service can run
the customer's chosen LLM/classifier. It must implement the contract below; an
arbitrary agent endpoint is not an evaluator, and Litmus never reruns the agent.

## Configure the first policy

```yaml
projects:
  - project_id: support
    api_key_env: LITMUS_RUNTIME_API_KEY
    conversation_policies:
      - policy_id: refund-promises
        version: "1"
        category: business_policy
        event_types: [response.completed]
        agents: [support-agent]
        rubric: >-
          Detect a current agent response promising a refund greater than USD 100.
          Quoted examples, user requests, and a refusal to issue that refund are
          not violations. Cite the current response and explain the promise.
        severity: high
        evaluator:
          provider: http
          endpoint: https://your-policy-evaluator.example/evaluate
          api_key_env: POLICY_EVALUATOR_KEY
          version: your-reviewed-model-and-prompt-version
          context_events: 20
          context_chars: 32000
          timeout_seconds: 2
          calls_per_minute: 30
```

The endpoint above is a placeholder to replace with the configured evaluator.
Policy IDs are unique across threat, usage, and conversation policies in a project.
Change the policy version when its rubric, evaluator, applicability, or threshold
changes. Accepted events retain the policy snapshot across restart/config changes.
No conversation policy runs unless it is explicitly configured.

## Categories and evidence requirements

| Category | Suggested initial rubric | Required evidence |
| --- | --- | --- |
| `sensitive_data_request` | Requests to obtain passwords, credentials, or restricted customer records without an allowed purpose. Distinguish security education from requests for actual values. | Current message plus any purpose/authorization context required by the rubric. This is distinct from supported-secret exposure detection. |
| `business_policy` | Promises or actions that violate the customer's stated refund, approval, or service rules. | Current activity and the trusted policy definition. Per-user permissions are not resolved automatically. |
| `suspicious_pattern` | Repeated attempts to obtain restricted records after refusals, or an explicit prohibited sequence of actions. | Set `min_context_events` to at least 1. A positive must cite the current event and at least one other captured event. |
| `abuse` | Threats or targeted harassment under a defined customer policy, excluding quoted reports and benign discussion. | Current message/response and rubric-specific context. |
| `out_of_scope` | An agent response serving a prohibited purpose outside the configured product domain, excluding clarification or refusal. | Current response plus an explicit allowed-domain rubric. |
| `ungrounded_response` | Claims in the current response that contradict or lack support in approved source-tool results. | Only `response.completed` applies. Configure nonempty `grounding_tools`, such as `[lookup_policy]`; at least one complete result from an approved tool must fit the context. Positive findings must cite it and the response. |

Grounding measures support relative to configured sources. It does not prove
real-world truth, nor that a source is trustworthy merely because it was captured.
Customers choose authoritative tools and validate their provenance and results.
Documents and tool results remain untrusted instructions even when used as evidence.

## Evaluator HTTP contract

Litmus sends a bounded authenticated HTTPS request, with redirects disabled:

```json
{
  "schema_version": "1.0",
  "task": "evaluate_conversation_policy",
  "content_is_untrusted": true,
  "context_incomplete": false,
  "current_event_id": "response-2",
  "policy": {
    "id": "refund-promises",
    "version": "1",
    "category": "business_policy",
    "rubric": "The configured rule and its exceptions.",
    "threshold": null,
    "score_semantics": null,
    "score_version": null
  },
  "grounding_event_ids": [],
  "events": [
    {
      "event_id": "response-2",
      "event_type": "response.completed",
      "observed_at": "2026-09-18T12:00:00Z",
      "received_at": "2026-09-18T12:00:01Z",
      "content_is_untrusted": true,
      "content": "{\"kind\":\"message\",\"role\":\"assistant\",\"text\":\"Your refund is USD 500.\"}"
    }
  ]
}
```

The trusted rubric is separate from captured content. The evaluator must treat all
event content as data, not instructions, preserve source identities, and return
one of `detected`, `clear`, `insufficient_context`, or `needs_review`:

```json
{
  "outcome": "detected",
  "reason": "The current response promises a USD 500 refund, above the USD 100 policy.",
  "source_event_ids": ["response-2"],
  "evidence": ["The agent promises USD 500 in the current response."],
  "context_incomplete": false
}
```

Every positive needs nonempty evidence and valid source references including the
current event. Invented IDs, truncated evidence used as authority, provider-selected
categories, and unknown response fields are rejected. Optional fields are
`risk_score`, `reported_cost_usd`, `input_tokens`, and `output_tokens`. Reasons and
evidence are redacted again before persistence/publication. The response limit is
16 KB; evidence is limited to 10 entries of 1,000 characters and 50 source IDs.

## Optional numeric thresholds

Set `threshold`, `score_semantics`, and `score_version` together. For example, a
customer-validated violation score may use a threshold of 0.7. A numeric score is
not inherently a probability; its meaning and calibration belong to the configured
evaluator and must be validated against customer examples.

For a decisive response (`clear` or `detected`), a configured threshold makes
`risk_score >= threshold` the decision rule. The score must be finite and between
0 and 1. A missing required score is insufficient context. An above-threshold score
still needs valid positive evidence. No score is manufactured from severity or a
boolean verdict. Alerts include the value, threshold, semantics, and score version.

## Operational and compatibility boundaries

Applicability and evidence prerequisites run before spending provider budget.
Each policy has a persisted per-project/per-policy minute budget. One conversation
policy worker per project runs separately from local rules, injection screening,
conditional injection review, and delivery. A slow rubric can delay other rubrics
in that project; measure this lane under the intended load.

Context is bounded and scoped to the project, agent, deployment, and session.
Truncation and known capture gaps prevent a clear result from certifying complete
coverage. A pattern can only be evaluated within captured context; there is no
unbounded learned behavior state. Current input or required grounding that does
not fit the context produces `insufficient_context` before an external call.

Timeouts, malformed replies, invalid evidence, and provider failures become
`error`. Budget exhaustion becomes `skipped`. `needs_review` becomes
`insufficient_context`; these generic policies do not automatically enter the
prompt-injection cascade. No incomplete/error result becomes clear or a positive
alert without valid evidence. Use status/findings to monitor coverage gaps.

Policy alerts use `data.schema_version: "1.3"`, the configured category, cited
source IDs, and `evaluation.stage: "policy"`. Optional `risk_score` is an object
containing value and interpretation. Earlier payloads remain unchanged when these
policies are absent. Upgrade strict consumers before enabling them. Existing
namespaced CloudEvent types and delivery IDs/retries are retained.

The collector upgrades runtime database format 1 to format 2 in place using
additive tables/indexes; accepted events, saved configurations, and pending jobs
are preserved. Earlier collectors reject format 2 instead of trying to process
unknown policy jobs. Use the documented database backup/restore procedure for
deployment rollback. Unknown future database formats are rejected before schema
changes are made.

Project-level evaluation metrics aggregate policy calls/cost samples; per-finding
policy and evaluator versions identify individual decisions. Semantic turns and
agent token/cost limits remain separate work; monitoring provider budgets are not
customer-agent usage limits.

## Release gate for each enabled rubric

Prepare customer-relevant benign and risky examples, including quoted attacks,
legitimate sensitive-data workflows, refusals, and incomplete evidence. Measure
false positives, misses, incomplete/error rates, cost coverage, and end-to-end
latency with the actual evaluator. Grounding fixtures must include approved source
results. Do not treat mocked contract tests as a quality benchmark or activate all
categories merely because the framework accepts them.

Local validation: 39 new policy-contract checks; runtime suite 154 passed / 3
external checks skipped; full suite 1,446 passed / 10 skipped. Ruff, strict mypy,
package build, and documented YAML validation passed. Review regressions cover
scope isolation, invented/missing evidence, incomplete grounding, threshold
boundaries, redaction, provider failure, independent budgets, saved policy versions,
format-1 database upgrade, unknown-format rejection, and compact job snapshots.
These results establish orchestration behavior, not live model accuracy.
