# Runtime threat alerting: implementation review

Reviewed September 18, 2026 against issue #116. The single-instance runtime pilot is
implemented on `codex/runtime-threat-alerts`. The review fixes below are complete;
live deployment validation remains before closing the issue. This is the first PR
in the runtime monitoring roadmap; conversation usage policies and conditional
evaluation will be reviewed separately.

## Review approach and fixes

After implementing the workflow, reread the issue and code as a separate review pass.
Trace activity capture, authenticated ingestion, detector outcomes, transactional
recovery, each publisher, and consumer deduplication. Add regression cases for the
findings instead of relying only on the original successful-path tests.

| Finding | Correction and verification |
| --- | --- |
| A prompt injection found in a completed tool's result could be labelled as observed attack execution. | Prompt-injection findings always describe an attempt. Tool-returned hostile text alone does not establish that the agent followed it. A regression verifies the stage. |
| Historical context could contain a value newly covered by an updated redaction policy. | Reapply the job's protected-data policy before sending stored context externally. Test captures an old unrecognized token, changes the policy, and checks the classifier sees a redacted value. |
| A negative classifier verdict could hide character-budget truncation. | Classifiers propagate incomplete context; a clear verdict becomes insufficient context. Provider/parser failures remain errors. |
| Replaying a failed delivery after removing its destination and restarting could leave it without a worker. | Include retained failed destinations when starting workers. Replay uses the saved endpoint, event ID, and delivery ID. |
| Pending-event quotas counted detector jobs, allowing more events when only semantic jobs remained. | Count distinct pending events; test local completion with a stalled semantic backlog. Also cap retained events and unresolved deliveries. |
| Readiness only checked whether worker tasks were alive. | Check worker failures and storage connectivity. Storage errors return a safe unavailable response. |
| SDK ingestion acknowledgements could be unbounded. | Stream acknowledgements with size/time limits and preserve bounded attempts. An oversized reply is a visible capture failure. |
| The declared Pydantic 2.0 minimum did not export the new JSON value type. | Require Pydantic 2.5 or later and verify with an isolated 2.5.3 environment. |

The provider integration review also verified Lakera's documented Detect-mode behavior:
its top-level `flagged` is forced false. The adapter uses the current message's
`prompt_attack` breakdown, preserves actual tool-call metadata alongside tool results,
and treats absent coverage as incomplete. Untrusted captured roles never become
trusted system instructions. The adapter is optional and disabled unless configured.

Other checks cover project isolation, protected values in keys/metadata, missing
destination permissions, duplicate ingestion, transactional rollback, expired leases,
out-of-order activity, immutable configuration versions, independent fan-out,
signature verification, and duplicate/out-of-order consumer revisions. Current tests
exercise identical canonical event IDs across all three destination adapters.

## Validation completed

Environment: Windows build 26200, Python 3.11.4, 12 logical processors.

- Full repository suite after merging the 1.0.0 release baseline: **1,359 passed,
  10 skipped**. Seven skips are pre-existing
  provider-dependent checks; three are the new live Kafka, Event Grid, and Lakera checks.
- Runtime suite: **67 passed, 3 skipped**.
- Ruff passed for source, tests, runtime examples, and the benchmark script.
- Strict mypy passed for all 58 source files.
- Source distribution and wheel built successfully.
- Installed the wheel into a clean environment with base dependencies only. Existing
  evaluation, CLI, and runtime SDK imports worked with FastAPI, Uvicorn, Kafka, and
  Azure libraries absent.
- With Pydantic 2.5.3 and FastAPI 0.115.0, the runtime suite passed **60 tests**;
  10 checks requiring absent optional SDKs or external infrastructure were skipped.
- Kafka's real producer/callback/consumer path completed a loopback round trip using
  librdkafka's disposable protocol mock. This is distinct from an Apache Kafka broker.
- Azure contract checks used the real CloudEvent model and a mocked SDK client;
  they verified identity, retry configuration, and safe failure classification.
- Example YAML validation and `git diff --check` passed.
- PR #117 CI passed the disposable Apache Kafka publish-and-consume test using
  Apache Kafka 3.9.1, plus the Python 3.10-3.12 Linux and Python 3.12 Windows checks.

## Repeatable local performance experiment

Command: `python scripts/benchmark_runtime.py --events 200 --rate 10 --sessions 20`.

The 20-second workload interleaved 20 session IDs at 10 events/second. Events were
1,483–1,605 bytes, split equally among benign messages, allowed tool requests,
forbidden destinations, and outbound protected-data tool requests. SQLite used WAL
and synchronous FULL. Ingestion used an in-process ASGI client; notification delivery
used an actual loopback HTTP receiver with signature verification.

| Measurement | Result |
| --- | --- |
| Accepted / rejected source events | 200 / 0 |
| Expected / received unique notifications | 150 / 150 |
| Missing notifications | 0 |
| Observed event to receiver p95 | 133 ms |
| Collector receipt to publication acknowledgement p95 | 131 ms |
| Maximum receiver latency | 157 ms |
| Capture gaps / unfinished jobs | 0 / 0 |

Semantic detection was disabled in this experiment and correctly reported as skipped.
The result establishes a local baseline, not a production or cross-region guarantee.
The generated detailed report is `.litmus/runtime-benchmark.json`.

## Remaining release gates

- Supply a test Azure topic/credential and confirm publication plus downstream subscriber
  receipt, authentication, throttling, and dead-letter behavior in that deployment.
- Supply a test classifier project/credential and run the reviewed injection fixtures;
  measure semantic latency and customer-specific false positives/misses. No live model
  accuracy or calibrated confidence claim has been made.

The pilot provides alerting and client-owned response hooks. It does not provide
pre-execution blocking, a durable client spool, general PII recognition, or distributed
service replicas. These boundaries are documented in the usage guide.
