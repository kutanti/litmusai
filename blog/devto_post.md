---
title: Testing AI agents with LitmusAI
published: true
tags: ai, python, testing, opensource
---

[LitmusAI](https://github.com/kutanti/litmusai) runs test cases against agents and records assertion results, latency, token usage, and estimated cost.

## Start with a testable requirement

A refund assistant might need to explain the return process and retain an order number across turns. Write separate checks for those requirements so a failure points to a specific behavior.

```yaml
name: refunds
cases:
  - id: return_request
    task: "I want to return the shoes I bought three days ago"
    assertions:
      - type: contains
        patterns: ["return", "refund", "policy"]
        mode: any
```

This check only verifies that the response contains one of those words. It does not establish that the policy is correct or that a refund was processed. Add structured assertions or checks against application state for those requirements.

## Run and inspect

```bash
pip install litmuseval
litmus run --suite refunds.yaml --agent my_agent:agent --runs 5
litmus scan --agent my_agent:agent --level thorough
```

Repeated runs help reveal inconsistent results. Safety scans exercise a fixed library of attack prompts; passing them does not prove an agent is safe.

## Compare costs carefully

Chat adapters read token usage from provider responses and multiply it by a bundled pricing table. The resulting cost is an estimate. Check the table against provider prices, and retain the suite, raw outputs, model parameters, and run count before publishing a comparison.

The earlier version of this article included model rankings without the corresponding result files or a reproducible benchmark setup. Those rankings have been removed. The listed cost-per-correct values also did not support the claimed 14-fold difference.

See the [quick start](https://github.com/kutanti/litmusai#quick-start) for a runnable local example and the available report formats.
