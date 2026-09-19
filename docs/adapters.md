# Agent Adapters

Adapters normalize supported agent responses to `AgentResponse`. Check which metadata and conversation features your adapter supports.

## Quick Reference

| Adapter | Use Case | Example |
|---------|----------|---------|
| `from_function` | Simple Python function | `Agent.from_function(my_fn)` |
| `from_openai_chat` | OpenAI-compatible chat completions | `Agent.from_openai_chat(model="gpt-4.1", api_key=key)` |
| `from_azure` | Azure OpenAI deployment | `Agent.from_azure(resource="my-resource", deployment="my-deployment", api_key=key)` |
| `from_url` | HTTP/REST endpoint | `Agent.from_url("http://...")` |
| `from_cli` | CLI subprocess | `Agent.from_cli("python agent.py")` |
| `from_langchain` | LangChain agent/chain | `Agent.from_langchain(lc_agent)` |
| `from_crewai` | CrewAI crew | `Agent.from_crewai(crew)` |
| `from_openai_agent` | Needs SDK compatibility work | Use a tested `from_function` wrapper; see below |
| `from_callable` | Any object with a method | `Agent.from_callable(obj)` |

## Structured inputs

Cases with an `inputs` mapping pass it as one keyword argument to the adapter.
An empty mapping is still a structured input; omitted or `None` inputs use the
existing task-only call.

| Adapter | Where the mapping is sent |
| --- | --- |
| `from_function`, `from_callable` | The callable receives `inputs=...` and must handle that keyword. |
| `from_url` | The JSON body contains `inputs` beside the configured task field. |
| `from_langchain` | The invocation mapping contains `inputs` beside `input` (the task). |
| `from_crewai` | The mapping passed to `kickoff(inputs=...)` contains nested `inputs` beside `task`. |

The chat completions, Azure, OpenAI Agents SDK, and CLI adapters reject structured
inputs before sending a request or starting a subprocess. `Agent.run()` returns
a failed response with an explanation, and evaluation records a failed case.
Use `from_function` to define the prompt, SDK request, or stdin format that your
agent needs. For example, if a text agent expects JSON after its instruction:

```python
import json
from litmusai import Agent

text_agent = Agent.from_cli("python agent.py")

async def run_structured(task: str, *, inputs: dict):
    prompt = f"{task}\nInputs: {json.dumps(inputs, ensure_ascii=False)}"
    return await text_agent.run(prompt)

agent = Agent.from_function(run_structured)
```

## AgentResponse

All adapters normalize outputs to `AgentResponse`:

```python
@dataclass
class AgentResponse:
    output: str                  # The agent's final text output
    metadata: dict[str, Any]     # Arbitrary metadata from the run
    cost: float | None           # Estimated USD; None means unavailable, 0 means free
    latency_ms: float            # Execution time in milliseconds
    tokens_used: int             # Total tokens consumed
    input_tokens: int            # Input/prompt tokens
    output_tokens: int           # Output/completion tokens
    model: str                   # The LLM model used
    tool_calls: list[ToolCall]   # Tools/functions called
    steps: list[AgentStep]       # Reasoning steps taken
    success: bool                # Whether execution succeeded
    error: str | None            # Error message if failed

    # Properties
    total_tokens: int            # input_tokens + output_tokens
    num_steps: int               # len(steps)
    num_tool_calls: int          # Total tool calls across all steps
```

## Adapters in Detail

### Chat completions and Azure

Both adapters use `httpx` from the base installation; the OpenAI Python SDK is not required.

```python
import os
from litmusai import Agent

agent = Agent.from_openai_chat(
    model="gpt-4.1",
    api_key=os.environ["OPENAI_API_KEY"],
    system_prompt="Answer concisely.",
    temperature=0.0,
    timeout=60,
)
```

For a compatible local server or proxy, supply its `base_url` and model name. API keys and URLs must be passed explicitly to `from_openai_chat`; it does not read `configure()` or environment variables itself. This adapter targets chat completions, so provider-specific request parameters must be supported by your endpoint. Use `extra_headers` and `extra_body` for endpoint options.

For Azure, use `Agent.from_azure(resource=..., deployment=..., api_key=...)`. It builds the deployment URL and uses the `api-key` header. Its key resolution is explicit argument, then `AZURE_OPENAI_API_KEY`, then global configuration. Set `api_version` to a version supported by your deployment.

Both adapters accept conversation history, parse provider token usage and tool calls, and estimate cost using registered model pricing. A custom Azure deployment name may need `register_pricing()`. The adapters capture returned tool-call metadata; they do not execute tools on your behalf.

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

Install the dependencies your chain needs, or use `pip install "litmuseval[langchain]"`. The adapter accepts objects exposing `ainvoke()` or `invoke()`. Framework tests use stand-ins; compatibility with every framework release is not established.

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

Install `crewai` separately; there is no CrewAI package extra. The adapter expects a crew exposing `kickoff()`.

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

This adapter needs compatibility work before use with the current SDK. It assumes imports and response fields that may not exist in your installed version. Use `from_function` to wrap a runner you have tested with your SDK, and return `AgentResponse` with the output and usage fields you need.

### 7. Custom Object (`from_callable`)

For any custom agent implementation:

```python
class MyAgent:
    def predict(self, task: str) -> str:
        return "response"

agent = Agent.from_callable(MyAgent(), method="predict", name="custom")
```

## Building Custom Adapters

For another framework, wrap its result in `AgentResponse`:

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
