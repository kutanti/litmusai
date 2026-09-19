# Conversation tool-call limits

Configure a project to emit a runtime event when a conversation exceeds a number of
distinct tool requests within a rolling window. This is an observed usage violation;
it does not by itself establish malicious intent. The customer's subscriber owns the
response, as with other runtime events.

```yaml
projects:
  - project_id: support
    api_key_env: LITMUS_RUNTIME_API_KEY
    usage_policies:
      - policy_id: conversation-tool-limit
        version: "1"
        max_calls: 20
        window_seconds: 60
        severity: high
        # Optional exact filters; omitted/empty means all.
        # tools: [send_email]
        # agents: [support-agent]
        # deployments: [production]
```

The 21st request crosses this limit. Exactly 20 requests does not. Separate
policies can set different limits or windows; policy IDs must be unique within a
project. Configuration changes require a new policy version. Each accepted event
retains its configuration, so pending evaluation is unaffected by later changes.

## Counting and notification semantics

- Count `tool.requested` only. `tool.completed`, `tool.failed`, messages, and responses
  do not add calls. SDK wrappers already supply the request boundary and a stable
  tool-call ID. Other integrations must do the same.
- Count each tool-call ID once within its project, agent, deployment, and session.
  Duplicate source events and request retries with the same call ID do not inflate
  counts. IDs must be unique for new calls within that scope.
- The window is `(collector receipt time - window_seconds, collector receipt time]`.
  Requests at the left boundary have expired. Equal timestamps follow acceptance
  order. Client timestamps cannot move a request into or out of a window. Delayed
  or batched capture can therefore differ from actual execution-time usage.
- Emit once when the count crosses from within the limit to above it. Later
  above-limit findings remain `detected` but their notifications are suppressed.
  A later crossing after calls expire creates a new logical alert. Duplicate
  delivery of an existing alert remains possible; subscribers must deduplicate.
  Enabling, re-enabling after unmonitored calls, or changing a policy version also
  emits on its first applicable request if the conversation is already above the
  configured limit. The collector does not wait for another exact crossing.
- Counts and supporting observations survive collector restart. Evaluation uses
  the window at acceptance, so worker backlog and processing order do not change
  the result. Observations and evaluation jobs are committed together.
- Known capture gaps in the window mark the finding incomplete. An observed count
  above the limit still proves an excess; a lower count with a gap is
  `insufficient_context`. Counts reflect captured requests, not uninstrumented
  activity or proof of successful tool execution.
- Request observations begin when this collector version is installed. Existing
  older events are not backfilled. Observation/deduplication history follows the
  configured event retention period. Windows are limited to 1 hour, below the
  minimum 1-day retention. Source references are bounded to the latest 50 events;
  a separate flag identifies truncated references without changing the full count.

## Event contract

The existing CloudEvents envelope and delivery/replay behavior are retained. Usage
alerts use category `excessive_tool_usage` and `data.schema_version: "1.1"`. Upgrade
strict consumers to this runtime contract before enabling these policies. Existing
threat alerts retain data version `1.0` and their existing fields.

```json
{
  "category": "excessive_tool_usage",
  "session_id": "customer-conversation-123",
  "policy_id": "conversation-tool-limit",
  "policy_version": "1",
  "usage": {
    "observed_count": 21,
    "limit": 20,
    "window_seconds": 60,
    "window_start": "2026-09-18T12:00:00Z",
    "window_end": "2026-09-18T12:01:00Z",
    "counting_basis": "collector_received_at",
    "threshold_crossed": true,
    "source_events_truncated": false
  }
}
```

This is an excerpt of `data`; complete alerts also include identity, severity,
evidence, source event IDs, timestamps, and context completeness. No confidence
score is invented for this deterministic rule. Webhook, Kafka, and Event Grid
routing can select the new category.

With the collector and receiver from the [runtime guide](runtime-threat-alerting.md)
running, use `python -m examples.runtime.tool_usage`. The example makes 21 simulated
authorized tool calls in one conversation, causing one usage alert under the example
configuration. It performs no real email or other external tool action.

Token/cost limits, semantic turn numbers, generalized behavior patterns, and
conditional LLM evaluation remain separate features.

## Review and validation

The review exercised threshold boundaries, duplicate requests/call IDs, ignored
completion events, equal receipt timestamps, reversed worker order, restart,
rolling expiry, project/agent/deployment/session isolation, multiple policy versions,
capture gaps, outbox rollback, bounded evidence, and retention. Regression tests
also cover enabling/lowering a policy on an already busy conversation and preserving
the existing alert payload when no usage evidence is present.

Local validation: 24 new usage checks; runtime suite 91 passed / 3 external checks
skipped; full suite 1,383 passed / 10 skipped; Ruff and strict mypy passed. A signed
notification reached a real loopback HTTP receiver through authenticated ingestion
and background workers. Actual customer instrumentation and production load still
require validation in the intended deployment.
