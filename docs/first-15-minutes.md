# First 15 minutes with LitmusAI

This walkthrough is for a small Python team that wants a regression check in a pull request. It uses a local function, so it does not need an API key.

## 1. Install and initialize

```bash
pip install litmuseval
litmus init
```

Replace the generated `suites/example.yaml` with:

```yaml
name: support-agent
cases:
  - id: refund
    task: "I want to return my shoes"
    assertions:
      - type: contains
        patterns: ["return", "refund"]
        mode: any
  - id: order-status
    task: "Where is order 123?"
    assertions:
      - type: contains
        patterns: ["123"]
  - id: greeting
    task: "Hello"
    assertions:
      - type: contains
        patterns: ["hello", "help"]
        mode: any
```

Create `my_agent.py`:

```python
def agent(task: str) -> str:
    task = task.lower()
    if "return" in task:
        return "I can help with your refund."
    if "order" in task:
        return "Order 123 is being prepared."
    return "Hello, how can I help?"
```

## 2. Run locally and inspect a report

```bash
litmus run -s suites/example.yaml -a my_agent.py:agent \
  --threshold 0.8 --save-baseline --log-dir .litmus/logs \
  --format json --output .litmus/results.json
litmus report --results .litmus/results.json --html .litmus/report.html
```

Open `.litmus/report.html`. Commit the suite and agent, but keep `.litmus/` out of version control unless you intentionally want to share a baseline.

## 3. Add the pull-request gate

Create `.github/workflows/evaluate.yml`:

```yaml
name: Evaluate agent
on: [push, pull_request]

permissions:
  contents: read

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: kutanti/litmusai@v1.1.0
        with:
          suite: suites/example.yaml
          agent: my_agent.py:agent
          threshold: "0.8"
          post-comment: "false"
```

The action fails when the pass-rate gate, budget, or configured baseline comparison fails. Replace the sample function with your application agent, then add cases for the failures your users actually see. Start with ten representative cases rather than trying to test every possible prompt.

## What to do next

- Run with `--runs 3` or more when behavior is nondeterministic.
- Add `--budget` and a baseline after recording real provider usage.
- Keep case IDs stable so `litmus diff` can explain regressions.
- Use the [example recipes](example-recipes.md) for support, RAG, and tool-use workflows.
- Treat evaluation results as evidence from a chosen dataset, not proof that an agent is universally safe or correct.
