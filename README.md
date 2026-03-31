<div align="center">

# 🧪 LitmusAI

### The open-source evaluation framework for AI agents

Test. Compare. Ship with confidence.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/badge/pypi-coming%20soon-orange)](https://pypi.org/project/litmusai/)

[Getting Started](#-getting-started) · [Features](#-features) · [Documentation](#-documentation) · [Contributing](#-contributing)

</div>

---

## 🤔 Why LitmusAI?

AI agents are everywhere — but **how do you know if yours actually works?**

- ✅ Does it complete tasks correctly?
- 💰 How much does each run cost?
- ⚡ Is it fast enough?
- 🔄 Did your last change break something?
- 🤖 Is Claude better than GPT for your use case?

**LitmusAI answers all of these.** One command, real answers.

```bash
pip install litmusai
litmus init
litmus run --agent my_agent.py
```

---

## ✨ Features

### 🎯 Agent Evaluation
Run your agents against standardized test suites and custom scenarios.

```python
from litmusai import TestSuite, Agent

# Define your agent
agent = Agent.from_function(my_agent_fn)

# Run evaluation
suite = TestSuite.load("coding")
results = suite.run(agent)

print(results.summary())
# ✅ 12/15 passed | ⚠️ 2 slow | ❌ 1 failed
# 💰 Total cost: $0.47 | ⚡ Avg: 3.2s
```

### 🔀 Agent Comparison
Compare models and configurations side-by-side.

```python
from litmusai import compare

results = compare(
    agents={"claude": claude_agent, "gpt": gpt_agent},
    suite="coding",
)
results.to_table()
# ┌─────────┬──────────┬───────┬──────┐
# │ Agent   │ Pass Rate│ Cost  │ Time │
# ├─────────┼──────────┼───────┼──────┤
# │ claude  │ 93%      │ $0.47 │ 3.2s │
# │ gpt     │ 87%      │ $0.31 │ 2.8s │
# └─────────┴──────────┴───────┴──────┘
```

### 📊 Built-in Test Suites

| Suite | Tasks | What it tests |
|-------|-------|---------------|
| `coding` | 25 | Code generation, debugging, refactoring |
| `research` | 20 | Web search, synthesis, fact-checking |
| `tool-use` | 30 | API calls, file ops, multi-step tool chains |
| `planning` | 15 | Multi-step reasoning, task decomposition |
| `safety` | 20 | Prompt injection, harmful outputs, guardrails |
| `custom` | ∞ | Define your own test cases |

### 🔌 Framework Agnostic
Works with **any** agent framework:

```python
# LangChain
agent = Agent.from_langchain(my_langchain_agent)

# CrewAI
agent = Agent.from_crewai(my_crew)

# OpenClaw
agent = Agent.from_openclaw(config)

# Custom function
agent = Agent.from_function(my_fn)

# Any HTTP endpoint
agent = Agent.from_url("http://localhost:8000/agent")
```

### 🔄 CI/CD Integration

```yaml
# .github/workflows/agent-eval.yml
name: Agent Evaluation
on: [push, pull_request]

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: kutanti/litmusai-action@v1
        with:
          agent: ./my_agent.py
          suite: coding
          min-pass-rate: 90
```

### 📈 Dashboard
Beautiful local dashboard to visualize results over time.

```bash
litmus dashboard
# 🌐 Dashboard running at http://localhost:3000
```

---

## 🚀 Getting Started

### Installation

```bash
pip install litmusai
```

### Quick Start

```bash
# Initialize a new project
litmus init

# Create your first test
litmus create-test "Can the agent write a Python fibonacci function?"

# Run against your agent
litmus run --agent my_agent.py

# View results
litmus report
```

### Your First Evaluation

```python
from litmusai import Agent, TestSuite, evaluate

# 1. Wrap your agent
agent = Agent.from_function(
    fn=my_agent_function,
    name="my-agent",
    model="claude-sonnet-4"
)

# 2. Pick a test suite
suite = TestSuite.load("coding")

# 3. Run evaluation
results = evaluate(agent, suite)

# 4. Check results
print(results)
assert results.pass_rate >= 0.9, "Agent performance dropped!"
```

---

## 📖 Documentation

| Guide | Description |
|-------|-------------|
| [Installation](docs/installation.md) | Setup and requirements |
| [Quick Start](docs/quickstart.md) | Your first evaluation in 5 minutes |
| [Test Suites](docs/test-suites.md) | Built-in and custom test suites |
| [Agent Adapters](docs/adapters.md) | Connect any agent framework |
| [CI/CD](docs/ci-cd.md) | Automate evaluations in your pipeline |
| [Dashboard](docs/dashboard.md) | Visualize and track results |
| [API Reference](docs/api.md) | Full Python API docs |

---

## 🏗️ Architecture

```
litmusai/
├── core/           # Core evaluation engine
│   ├── agent.py    # Agent abstraction & adapters
│   ├── suite.py    # Test suite management
│   ├── runner.py   # Evaluation runner
│   ├── scorer.py   # Scoring & metrics
│   └── reporter.py # Results & reporting
├── suites/         # Built-in test suites
│   ├── coding/     # Code generation tasks
│   ├── research/   # Research & synthesis tasks
│   ├── tool_use/   # Tool usage tasks
│   ├── planning/   # Multi-step planning tasks
│   └── safety/     # Safety & guardrail tests
├── adapters/       # Framework integrations
│   ├── langchain.py
│   ├── crewai.py
│   ├── openclaw.py
│   └── http.py
├── dashboard/      # Web dashboard
├── cli/            # CLI commands
└── utils/          # Helpers & utilities
```

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Clone the repo
git clone https://github.com/kutanti/litmusai.git
cd litmusai

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🌟 Star History

If LitmusAI helps you ship better agents, give us a ⭐!

---

<div align="center">

**Built with ❤️ by [Kunal Tanti](https://github.com/kutanti)**

*Staff Software Engineer @ LinkedIn*

</div>
