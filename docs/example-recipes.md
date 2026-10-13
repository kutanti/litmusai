# Example recipes

These recipes are deliberately small. They show where LitmusAI fits in a real agent repository: define behavior as cases, run the cases locally, then make the same checks a pull-request gate.

## Customer support regression

**Failure to catch:** a prompt change stops answering refund questions or starts returning an unhelpful fallback.

```yaml
name: support-regression
cases:
  - id: refund-policy
    task: "Can I return shoes after 30 days?"
    assertions:
      - type: contains
        patterns: ["return", "refund"]
        mode: any
  - id: escalation
    task: "I was charged twice"
    assertions:
      - type: contains
        patterns: ["support", "review", "charge"]
        mode: any
```

Run it with a wrapper around the support agent:

```bash
litmus run -s support.yaml -a support_agent.py:agent \
  --threshold 0.8 --runs 3 --save-baseline
```

## RAG answer and citation checks

**Failure to catch:** retrieval still finds the right document, but the answer omits a required fact or cites an unrelated source.

Keep retrieved source labels in the agent response or structured output and assert on both the answer and citation:

```yaml
name: knowledge-assistant
cases:
  - id: cancellation
    task: "How do I cancel my subscription?"
    assertions:
      - type: contains
        patterns: ["cancel"]
      - type: contains
        patterns: ["billing-policy"]
```

For stricter contracts, return JSON and use `json_schema` or `json_path` assertions. Ground truth can record the expected answer and its source; LitmusAI records the supplied provenance but does not independently verify it.

## Tool-use safety and cost regression

**Failure to catch:** a model starts calling an unnecessary tool, sends a request to an unapproved destination, or exceeds an evaluation budget.

Test tool intent in the agent's structured response:

```yaml
name: tool-use
cases:
  - id: account-lookup
    task: "Show the status of order 123"
    assertions:
      - type: json_path
        path: "$.tool"
        expected: "lookup_order"
      - type: json_path
        path: "$.order_id"
        expected: "123"
  - id: no-external-send
    task: "Summarize my account history"
    assertions:
      - type: not_contains
        patterns: ["send_email", "external_webhook"]
```

Add cost and latency gates once the adapter reports usage:

```bash
litmus run -s tool-use.yaml -a agent.py:agent \
  --threshold 0.9 --budget 1.00 --baseline .litmus/baseline.json
```

These checks describe the scenarios you selected. They do not prove that an agent is safe in every context. For live tool activity, use the opt-in [runtime monitoring](runtime-threat-alerting.md) features and validate them against your own traffic and policies.

## Turning a recipe into proof

For a public demonstration, keep the suite, agent wrapper, baseline, failed change, report, and workflow together. Show the pull request where one case changes from PASS to FAIL. That is more useful to a prospective user than a feature list, and it gives contributors a concrete pattern to copy.
