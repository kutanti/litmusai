# Agent Adapters

LitmusAI provides universal adapters to evaluate agents from **any** framework using a single, consistent API.

## Quick Reference

| Adapter | Use Case | Example |
|---------|----------|---------|
| `from_function` | Simple Python function | `Agent.from_function(my_fn)` |
| `from_url` | HTTP/REST endpoint | `Agent.from_url("http://...")` |
| `from_cli` | CLI subprocess | `Agent.from_cli("python agent.py")` |
| `from_langchain` | LangChain agent/chain | `Agent.from_langchain(lc_agent)` |
| `from_crewai` | CrewAI crew | `Agent.from_crewai(crew)` |
| `from_openai_agent` | OpenAI Agents SDK | `Agent.from_openai_agent(oai)` |
| `from_callable` | Any object with a method | `Agent.from_callable(obj)` |

## AgentResponse

All adapters normalize outputs to `AgentResponse`:

```python
@dataclass
class AgentResponse:
    output: str              # The agent's final text output
    cost: float              # Estimated cost in USD
    latency_ms: float        # Execution time in milliseconds
    tokens_used: int         # Total tokens consumed
    input_tokens: int        # Input/prompt tokens
    output_tokens: int       # Output/completion tokens
    model: str               # The LLM model used
    tool_calls: list[ToolCall]   # Tools/functions called
    steps: list[AgentStep]       # Reasoning steps taken
    success: bool            # Whether execution succeeded
    error: str | None        # Error message if failed
```

## Adapters in Detail

### 1. Simple Function (`from_function`)

The simplest adapter — wrap any Python function:

```python
from litmusai import Agent

# Sync function
def my_agent(task: str) -> str:
    return f"Answer: {task}"

agent = Agent.from_function(my_agent, name="my-agent")

# Async function
async def async_agent(task: str) -> str:
    return await some_api_call(task)

agent = Agent.from_function(async_agent, name="async-agent")

# Return a dict for richer data
def rich_agent(task: str) -> dict:
    return {
        "output": "response",
        "cost": 0.01,
        "tokens_used": 100,
        "tool_calls": [{"name": "search", "arguments": {"q": task}}],
    }

agent = Agent.from_function(rich_agent, name="rich-agent")
```

### 2. HTTP Endpoint (`from_url`)

Evaluate any agent exposed as an API:

```python
agent = Agent.from_url(
    "http://localhost:8000/agent",
    name="api-agent",
    headers={"Authorization": "Bearer sk-xxx"},
    request_field="prompt",      # JSON field for the task
    response_field="reply",      # JSON field for the output
    timeout=60,
)
```

### 3. CLI Agent (`from_cli`)

Evaluate agents that run as command-line programs:

```python
agent = Agent.from_cli(
    "python my_agent.py",
    name="cli-agent",
    timeout=120,
)
# Task is sent via stdin, output captured from stdout
```

### 4. LangChain (`from_langchain`)

```python
from langchain.agents import AgentExecutor
from litmusai import Agent

lc_agent = AgentExecutor(agent=..., tools=[...])
agent = Agent.from_langchain(lc_agent, name="langchain-agent")

# Automatically extracts:
# - Final output
# - Tool calls from intermediate_steps
# - Reasoning steps
```

### 5. CrewAI (`from_crewai`)

```python
from crewai import Crew
from litmusai import Agent

crew = Crew(agents=[...], tasks=[...])
agent = Agent.from_crewai(crew, name="my-crew")

# Automatically extracts:
# - Final output
# - Token usage
# - Task outputs as steps
```

### 6. OpenAI Agents SDK (`from_openai_agent`)

```python
from openai.agents import Agent as OAIAgent
from litmusai import Agent

oai = OAIAgent(name="helper", model="gpt-4o", tools=[...])
agent = Agent.from_openai_agent(oai, name="openai-agent")

# Automatically extracts:
# - Final output
# - Tool calls
# - Token usage
```

### 7. Custom Object (`from_callable`)

For any custom agent implementation:

```python
class MyAgent:
    def predict(self, task: str) -> str:
        return "response"

agent = Agent.from_callable(MyAgent(), method="predict", name="custom")
```

## Building Custom Adapters

If your framework isn't listed, you can easily create a custom adapter:

```python
from litmusai import Agent, AgentResponse, ToolCall, AgentStep

def my_framework_adapter(framework_agent, name="custom"):
    async def run_fn(task: str) -> AgentResponse:
        # Call your framework
        result = framework_agent.execute(task)

        # Normalize the response
        return AgentResponse(
            output=result.text,
            cost=result.usage.cost,
            tokens_used=result.usage.tokens,
            tool_calls=[
                ToolCall(name=tc.name, arguments=tc.args)
                for tc in result.tool_calls
            ],
            steps=[
                AgentStep(step_number=i, action=s.action, observation=s.result)
                for i, s in enumerate(result.steps, 1)
            ],
        )

    return Agent.from_function(run_fn, name=name)
```
