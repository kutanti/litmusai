# Live agent threat alerts

The experimental `litmusai.runtime` package observes a running agent and publishes
CloudEvents to Kafka, Azure Event Grid, signed HTTPS webhooks, or all three. It does
not execute, replay, or block the agent. A customer-owned consumer decides what to
do with an alert. An asynchronous notification cannot guarantee prevention.

## Run the local demonstration

Install from this checkout:

```sh
pip install -e ".[runtime]"
# Add these extras when using their publishers:
pip install -e ".[runtime-kafka,runtime-azure]"
```

Set `LITMUS_RUNTIME_API_KEY` and `LITMUS_WEBHOOK_SECRET` to separate random secrets
of at least 32 characters. Share the API key between agent and collector, and the
webhook secret between collector and receiver. Keep the same values in each terminal;
do not commit them. For example, generate a value with
`python -c "import secrets; print(secrets.token_hex(32))"` and store it in your secret manager.

Run these commands in separate terminals from the repository root:

```sh
python -m uvicorn examples.runtime.webhook_receiver:app --port 8766 --no-access-log
litmus runtime serve --config examples/runtime/config.yaml
python -m examples.runtime.agent
```

The sample's `send_email` tool is simulated. It produces forbidden-destination and
sensitive-data alerts without sending email. The receiver prints the category,
severity, and alert revision. The sample uses loopback HTTP only; deploy both the
collector and receivers behind TLS.

Inspect the project:

```sh
litmus runtime status --endpoint http://127.0.0.1:8765 --allow-local-http
litmus runtime alerts --endpoint http://127.0.0.1:8765 --allow-local-http
litmus runtime validate --config examples/runtime/config.yaml
```

## Connect an existing agent

```python
from litmusai.runtime import RuntimeClient

async with RuntimeClient(endpoint, api_key, project_id="support") as monitor:
    async with monitor.session(agent_id="support", session_id=request_id) as session:
        session.emit_message(user_message)
        session.emit_context(retrieved_document)
        monitored_send_email = session.wrap_tool(
            send_email, name="send_email", destination_argument="recipient"
        )
        # Your application must register this wrapper as the actual executable tool.
        response = await run_existing_agent(
            user_message, tools={"send_email": monitored_send_email}
        )
        session.emit_response(response, destination="customer-channel")
        await monitor.aflush(timeout=5)
```

Synchronous applications use ordinary `with` and `monitor.flush()`. Wrappers support
normal synchronous functions and `async def` functions. Arguments and results must
be JSON-compatible. Unsupported objects, oversized events, full queues, and capture
failures increment the client's `health["dropped"]`; the original tool still runs
once and its return value or exception is preserved. Async generators and opaque
framework tools require explicit capture hooks. The SDK does not discover hidden
tool calls inside an arbitrary HTTP endpoint or agent function.

`session.emit(event_type, Message(...))` and
`session.emit(event_type, ToolActivity(...), tool_call_id=...)` support explicit
integration at other boundaries. Capture each external document before passing it
to the agent, and actual tool requests, completions, and failures separately.

`emit` returns **locally queued**, not durable acceptance. The SDK uses a bounded
background sender, stable event IDs, and bounded retries. `flush` returns false if
its deadline expires or any event has failed/dropped during that client lifetime.
Inspect `health` to distinguish those cases. A process crash can lose queued events;
the SDK has no durable local spool. Detection starts when the collector accepts the
event, without waiting for `session.ended`.

## HTTP ingestion and inspection

Other languages can POST one `RuntimeEvent` or a bounded array to `/v1/events` with
`Authorization: Bearer <project-key>`. The key binds the project; changing a payload's
`project_id` cannot select another project. API keys currently authorize both capture
and inspection for that project; issue them to trusted application infrastructure.

```json
{
  "schema_version": "1.0",
  "event_id": "evt-1",
  "project_id": "support",
  "agent_id": "support-agent",
  "session_id": "session-1",
  "producer_id": "process-unique-id",
  "sequence": 1,
  "event_type": "tool.requested",
  "tool_call_id": "call-1",
  "observed_at": "2026-09-18T12:00:00Z",
  "payload": {
    "kind": "tool",
    "name": "send_email",
    "arguments": {"body": "sample"},
    "destination": "outside.example"
  }
}
```

Use a unique producer ID per process and an increasing sequence across that
producer's sessions. Event and tool IDs must be stable across retries. Tool IDs must
be unique per invocation within an agent/deployment/session. IDs permit letters,
numbers, underscores, periods, colons, and hyphens, up to 128 characters. Do not put
secrets in identifiers. Timestamps require timezones; they do not establish ordering
across producers. Sequence gaps and late arrivals mark context as incomplete.

A 200 response includes a result for every item: `accepted`, `duplicate`, `conflict`,
`busy`, or `rejected`. Only accepted/duplicate acknowledge durable receipt. Invalid
items do not prevent valid items in the same batch from being committed. A reused ID
with different sanitized content conflicts. Default limits are 50 events per request,
256 KiB per request, 64 KiB per event, 32 KiB per message, and 16 levels of nested payload.

Authenticated endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /v1/alerts?limit=100&offset=0` | Latest logical alerts |
| `GET /v1/alerts/{id}` | Alert and up to 200 recent delivery records |
| `GET /v1/findings?limit=100&offset=0` | Detector outcomes, evidence, and suppression |
| `GET /v1/status` | Coverage, capture gaps, queue lag, delivery states, latency, retention |
| `POST /v1/deliveries/{id}/replay` | Retry a failed delivery using its saved destination |

The maximum page size is 200. `/health/live` and `/health/ready` expose only service
readiness. Collector ingestion counters cover the current process; durable findings
and deliveries cover retained history. SDK-local failures cannot appear at the
collector until a later event reveals a sequence gap, so also collect SDK health.

## Detection and evidence

`allowed_tools` and `allowed_destinations` are **exact allowlists** from trusted
configuration. `null` means authorization context is unavailable; `[]` forbids all.
There is no implicit hostname, path-prefix, wildcard, or user-conversation permission
matching. List tools requiring destination context in `destination_tools`. Application
instrumentation must supply the actual destination in its canonical business format.

The initial local sensitive formats are GitHub token prefixes (`ghp_`, `gho_`, `ghu_`,
`ghs_`, `ghr_`, `github_pat_`), AWS access-key IDs, and PEM private-key blocks. Add
bounded custom token formats using policy `protected_patterns`, for example:

```yaml
protected_patterns:
  - name: customer_token
    prefix: cust_
    alphabet: ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789
    min_suffix: 16
    max_suffix: 128
```

Patterns use a literal prefix and alphabet, not arbitrary regular expressions. This
is not comprehensive PII or arbitrary-secret recognition. Ingest only data permitted
by your retention policy. Supported values are detected locally, then redacted from
payload values and keys before durable storage or external classification. Raw values
exist transiently at capture/ingestion. Retaining raw evidence is unsupported.

A secret in a tool **result** is inbound content, not evidence that it was sent to
that tool's destination. Exposure findings require outbound tool arguments or final
responses together with destination authorization. Missing authorization remains
`insufficient_context`. Findings distinguish `attempt`, `requested`, and `observed`;
even tool completion does not independently prove successful exfiltration.

### Prompt injection

Configure a classifier explicitly; otherwise its coverage is reported as `skipped`.
The supplied `lakera` adapter calls Guard v2 using your provider project and credential
(see the commented configuration in the example). Enable Prompt Defense on
`user::content` and `tool::content` in that provider project. Local rules continue even
if the classifier times out or exceeds its persisted per-project call budget.

The adapter requests a breakdown and reads `prompt_attack` detections for the current
message. It works in Detect mode, where the provider's top-level `flagged` is forced
false. Missing prompt coverage becomes `insufficient_context`; malformed responses
and network failures become `error`. This behavior follows the provider's
[Guard API](https://docs.lakera.ai/docs/api/guard) and
[screening roles](https://docs.lakera.ai/docs/api/screening-roles) documentation checked
September 18, 2026. Provider calls use sanitized content and never rerun an agent.

For another provider, implement `InjectionClassifier` when creating the app, or select
`provider: http` and supply an endpoint accepting the documented adapter request:
`schema_version`, `task: detect_prompt_injection`, `content_is_untrusted`,
`context_incomplete`, and `events` (current event first, then recent context). Each
item contains `event_id`, `event_type`, `source_trust: untrusted`, and a bounded JSON
payload string in `content`. Return
`{"outcome":"detected","reason":"...","context_incomplete":false}`;
`clear` and `insufficient_context` are also valid outcomes. Response size is limited
to 16 KiB. All captured content is untrusted; it cannot change operator policy.

Default semantic limits are 8 context events, 16,000 content characters, a 2-second
timeout, one concurrent call per project, and 60 calls per UTC minute. No score is
represented as a probability. Increment the classifier version when changing the
provider policy/model/configuration. Evaluate false positives and misses with real,
reviewed customer examples before enabling production response actions.

An opt-in quality check uses `tests/runtime/fixtures/prompt_injection.json`, including
direct/indirect attacks, ordinary requests, and quoted security discussion. Set
`LITMUS_TEST_LAKERA_KEY` and `LITMUS_TEST_LAKERA_PROJECT`, then run
`pytest tests/runtime/test_lakera.py -k live -s` against a test provider project. It
reports true/false positives, misses, incomplete coverage, precision, and recall for
these eight samples. These small fixtures are a behavioral check, not a calibrated
accuracy estimate for customer traffic.

## Destinations and delivery guarantees

See `examples/runtime/config.yaml` for all three destination configurations.
`categories`, `severities`, `agents`, and `deployments` filter routes; empty lists mean
all. A destination belongs to one project. URLs must use HTTPS without embedded
credentials, query strings, or fragments. Local HTTP requires an explicit loopback
development option; plaintext Kafka requires a separate development opt-in. Operators
control outbound hosts/topics and should enforce their network policy at deployment.

Every revision is an immutable CloudEvent with a stable event `id`, namespaced `type`,
project `source`, alert `subject`, timestamp, and versioned `data`. `data.alert_id` and
`data.revision` identify the logical alert. A separate delivery ID identifies each
destination attempt sequence. The same event ID/body is reused across destinations
and retries. Kafka uses project/session as its partition key; there is no global
ordering guarantee. Receivers must deduplicate event IDs and ignore older revisions.

The delivery contract is **at least once within the configured retry/retention
budgets**. Broker acknowledgement, Event Grid publish acceptance, and webhook 2xx do
not prove completion of the downstream response action. Kafka producer idempotence
does not remove duplicates across process restarts. Example consumers demonstrate
durable deduplication and revision handling; integrate actual actions through your
own durable, idempotent work queue.

Webhooks sign `timestamp + "." + exact_body` using HMAC-SHA256. Headers are
`X-Litmus-Timestamp`, `X-Litmus-Signature` (`v1=<hex>`), and `X-Litmus-Delivery-ID`.
Verify signature and a bounded clock window before parsing; deduplication is still
necessary. Redirects are never followed. The example receiver uses a five-minute window.

Kafka uses TLS or SASL over TLS, `acks=all`, bounded buffering and delivery timeout,
and broker callbacks. Configure your broker replication/minimum in-sync replicas to
match the required durability. The Event Grid publisher uses the Azure SDK with
`DefaultAzureCredential` (managed identity where available), or an access-key environment
reference. It disables nested SDK retries. Clients own Event Grid subscriptions,
dead-letter storage, and the subscriber's authentication. Configure both the topic's
input schema and the subscription's event delivery schema as `CloudEventSchemaV1_0`.
The example subscriber handles the CloudEvents `OPTIONS /events` validation handshake
without a delivery token; subsequent `POST /events` deliveries require an Azure
subscription delivery header `X-Litmus-Subscription-Token` matching
`EVENT_GRID_SUBSCRIPTION_TOKEN`. The handshake grants delivery permission, not
authentication. It advertises no rate limit; configure throttling at your HTTPS ingress
for deployment. The native `EventGridSchema` format and its
`Microsoft.EventGrid.SubscriptionValidationEvent` POST handshake are not supported by
this CloudEvents receiver. See Microsoft's
[CloudEvents endpoint validation documentation](https://learn.microsoft.com/en-us/azure/event-grid/end-point-validation-cloud-events-schema).

Each destination has its own worker and retry state. Failures cannot block other
destinations. Permanent failures and exhausted attempts remain queryable/replayable.
Policy/classifier/destination versions are immutable; bump the version for changes.
Outbox records retain their original destination configuration. Removing or changing
a destination does not reroute pending/replayed events. Webhook and Event Grid access
keys and Kafka SASL username/password references are resolved at publication. Kafka
refreshes the cached producer's PLAIN/SCRAM credentials for its next authentication;
existing authenticated connections remain open. TLS certificate/key files and private
key passwords are loaded when the Kafka producer is created, so restart the collector
after rotating those values. Environment changes made outside the running process also
require a restart to become visible. Outbox records store references, never secret
values. After correcting credentials, explicitly replay any deliveries already marked
as permanent authentication failures.

## Deployment, recovery, and limits

Use one service instance, one persistent local SQLite volume, and one worker process.
Do not share this database across replicas or put it on a network filesystem. SQLite
WAL with synchronous FULL commits events/jobs together, and findings/alerts/outbox
together. Interrupted leases recover after at most 60 seconds; consumers may receive
a duplicate when a crash occurs after publication but before acknowledgement is saved.

Defaults cap each project at 10,000 pending events, 100,000 stored events, and 20,000
unacknowledged deliveries. Full projects return `busy`. Increase limits according to
measured traffic and disk capacity; a prolonged destination outage requires operator
attention. Retention defaults to seven days and applies to events, findings, revisions,
and delivery attempts, including unresolved work. Expired unresolved work is counted
in status. Deduplication ends when retained IDs expire. Alert episodes correlate by
project, agent/deployment/session, policy/detector/category, and tool-call or parent/event
reference. Repeated equivalent findings within the cooldown remain inspectable even
when notification is suppressed. Different tool invocations remain separate episodes.

Drain work before removing a project. Stop the service and copy its SQLite database
and any WAL/SHM files together for an offline backup, or use SQLite's online backup API.
Restore into a single stopped instance with matching configuration versions and secret
references. Backups contain sanitized event content and need the same access controls
and retention policy as the primary volume.

Run `python scripts/benchmark_runtime.py` for the repeatable 200-event local experiment
(10 events/second, 20 interleaved sessions, real loopback webhook receiver). It reports
accepted/rejected counts, expected/received notifications, latency, and detector
coverage, including skipped semantic checks. This is not a production performance
guarantee. Kafka/Event Grid and classifier latency must be measured with the intended
infrastructure. Optional smoke tests use `LITMUS_TEST_KAFKA_BOOTSTRAP` or
`LITMUS_TEST_EVENT_GRID_ENDPOINT` plus `LITMUS_TEST_EVENT_GRID_KEY`; use disposable
infrastructure. The Kafka integration workflow provisions a disposable broker in CI.

Framework callbacks, durable SDK spooling, streaming token inspection, general behavior
anomaly detection, per-user authorization resolvers, dashboards, distributed workers,
and guaranteed inline blocking remain outside this pilot.
