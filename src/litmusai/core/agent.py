"""Agent abstraction layer — wrap any agent framework.

This module provides a universal Agent class that normalizes different
agent frameworks into a single interface for evaluation.

Supported frameworks:
    - Simple Python functions (sync/async)
    - HTTP endpoints (REST APIs)
    - LangChain agents
    - CrewAI crews
    - OpenAI Agents SDK
    - CLI-based agents (subprocess)
    - Custom adapters via Agent protocol
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class ToolCall:
    """Record of a tool/function call made by an agent."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str = ""
    duration_ms: float = 0.0

    def __repr__(self) -> str:
        return f"ToolCall(name='{self.name}', args={self.arguments})"


@dataclass
class AgentStep:
    """A single step in an agent's execution."""

    step_number: int
    action: str
    observation: str = ""
    thought: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    duration_ms: float = 0.0

    def __repr__(self) -> str:
        return f"AgentStep(#{self.step_number}, action='{self.action}')"


@dataclass
class AgentResponse:
    """Normalized response from any agent execution.

    All agent adapters produce this standard response format,
    making it possible to evaluate agents from different frameworks
    using the same metrics and scoring.

    Attributes:
        output: The agent's final text output.
        metadata: Arbitrary metadata from the agent run.
        cost: Estimated cost in USD for this run.
        latency_ms: Total execution time in milliseconds.
        tokens_used: Total tokens consumed (input + output).
        input_tokens: Input/prompt tokens consumed.
        output_tokens: Output/completion tokens consumed.
        model: The LLM model used (e.g., "claude-sonnet-4").
        tool_calls: List of tools/functions called during execution.
        steps: List of reasoning steps taken by the agent.
        success: Whether the execution completed without errors.
        error: Error message if execution failed.
    """

    output: str
    metadata: dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0
    latency_ms: float = 0.0
    tokens_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    success: bool = True
    error: str | None = None

    @property
    def num_steps(self) -> int:
        """Number of steps taken by the agent."""
        return len(self.steps)

    @property
    def num_tool_calls(self) -> int:
        """Total number of tool calls across all steps."""
        step_tools = sum(len(s.tool_calls) for s in self.steps)
        return len(self.tool_calls) + step_tools

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output), falls back to tokens_used."""
        if self.input_tokens or self.output_tokens:
            return self.input_tokens + self.output_tokens
        return self.tokens_used


class Agent:
    """Universal agent wrapper — connect any AI agent to LitmusAI.

    The Agent class provides a unified interface for evaluating AI agents
    from any framework. Use the class methods (from_function, from_url, etc.)
    to wrap your agent, then pass it to evaluate() or compare().

    Example:
        >>> agent = Agent.from_function(my_fn, name="my-agent", model="gpt-4o")
        >>> results = await evaluate(agent, suite)
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        name: str = "agent",
        model: str = "",
        metadata: dict[str, Any] | None = None,
    ):
        self.fn = fn
        self.name = name
        self.model = model
        self.metadata = metadata or {}

    async def run(self, task: str, **kwargs: Any) -> AgentResponse:
        """Execute the agent on a given task.

        Args:
            task: The task/prompt to send to the agent.
            **kwargs: Additional arguments passed to the agent function.

        Returns:
            AgentResponse with normalized output, cost, latency, etc.
        """
        start = time.monotonic()

        try:
            if asyncio.iscoroutinefunction(self.fn):
                result = await self.fn(task, **kwargs)
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.fn(task, **kwargs)
                )

            elapsed = (time.monotonic() - start) * 1000

            if isinstance(result, AgentResponse):
                result.latency_ms = elapsed
                return result
            elif isinstance(result, str):
                return AgentResponse(output=result, latency_ms=elapsed, model=self.model)
            elif isinstance(result, dict):
                return self._parse_dict_response(result, elapsed)
            else:
                return AgentResponse(output=str(result), latency_ms=elapsed, model=self.model)

        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            return AgentResponse(
                output="",
                success=False,
                error=str(e),
                latency_ms=elapsed,
                model=self.model,
            )

    def _parse_dict_response(self, result: dict[str, Any], elapsed: float) -> AgentResponse:
        """Parse a dictionary response into an AgentResponse."""
        # Extract tool calls if present
        tool_calls = []
        raw_tools = result.get("tool_calls", [])
        for tc in raw_tools:
            if isinstance(tc, ToolCall):
                tool_calls.append(tc)
            elif isinstance(tc, dict):
                tool_calls.append(ToolCall(
                    name=str(tc.get("name", "")),
                    arguments=tc.get("arguments", tc.get("args", {})) or {},
                    result=str(tc.get("result", "")),
                ))

        # Extract steps if present
        steps = []
        raw_steps = result.get("steps", [])
        for i, step in enumerate(raw_steps):
            if isinstance(step, AgentStep):
                steps.append(step)
            elif isinstance(step, dict):
                steps.append(AgentStep(
                    step_number=step.get("step_number", i + 1),
                    action=str(step.get("action", "")),
                    observation=str(step.get("observation", "")),
                    thought=str(step.get("thought", "")),
                ))

        return AgentResponse(
            output=str(result.get("output", str(result))),
            cost=float(result.get("cost", 0.0)),
            tokens_used=int(result.get("tokens_used", 0)),
            input_tokens=int(result.get("input_tokens", 0)),
            output_tokens=int(result.get("output_tokens", 0)),
            model=str(result.get("model", self.model)),
            latency_ms=elapsed,
            metadata={k: v for k, v in result.items()
                      if k not in {"output", "cost", "tokens_used", "input_tokens",
                                   "output_tokens", "model", "tool_calls", "steps"}},
            tool_calls=tool_calls,
            steps=steps,
        )

    # ─── Factory Methods ──────────────────────────────────────────────

    @classmethod
    def from_function(
        cls,
        fn: Callable[..., Any],
        name: str = "agent",
        model: str = "",
    ) -> Agent:
        """Create an agent from a simple Python function.

        The function can be sync or async and should accept a task string
        as its first argument. It can return a string, dict, or AgentResponse.

        Args:
            fn: A callable that takes a task string and returns a response.
            name: Display name for the agent.
            model: The model being used (for reporting).

        Example:
            >>> def my_agent(task: str) -> str:
            ...     return f"Answer: {task}"
            >>> agent = Agent.from_function(my_agent, name="simple")
        """
        return cls(fn=fn, name=name, model=model)

    @classmethod
    def from_url(
        cls,
        url: str,
        name: str = "http-agent",
        model: str = "",
        headers: dict[str, str] | None = None,
        timeout: float = 120,
        request_field: str = "task",
        response_field: str = "output",
    ) -> Agent:
        """Create an agent from an HTTP endpoint.

        Sends POST requests with the task and parses the response.

        Args:
            url: The HTTP endpoint URL.
            name: Display name for the agent.
            model: The model being used (for reporting).
            headers: Additional HTTP headers.
            timeout: Request timeout in seconds.
            request_field: JSON field name for the task in the request body.
            response_field: JSON field name for the output in the response body.

        Example:
            >>> agent = Agent.from_url("http://localhost:8000/agent")
            >>> agent = Agent.from_url(
            ...     "https://api.example.com/chat",
            ...     headers={"Authorization": "Bearer xxx"},
            ...     request_field="message",
            ...     response_field="reply",
            ... )
        """
        req_headers = headers or {}

        async def http_fn(task: str, **kwargs: Any) -> dict[str, Any]:
            async with httpx.AsyncClient(timeout=timeout) as client:
                payload = {request_field: task, **kwargs}
                response = await client.post(url, json=payload, headers=req_headers)
                response.raise_for_status()
                data = response.json()

                return {
                    "output": str(data.get(response_field, data.get("response", str(data)))),
                    "cost": float(data.get("cost", 0.0)),
                    "tokens_used": int(
                        data.get("tokens_used", data.get("usage", {}).get("total_tokens", 0))
                    ),
                    "input_tokens": int(
                        data.get("input_tokens", data.get("usage", {}).get("prompt_tokens", 0))
                    ),
                    "output_tokens": int(
                        data.get("output_tokens", data.get("usage", {}).get("completion_tokens", 0))
                    ),
                    "model": str(data.get("model", model)),
                    "tool_calls": data.get("tool_calls", []),
                    "steps": data.get("steps", []),
                }

        return cls(fn=http_fn, name=name, model=model)

    @classmethod
    def from_cli(
        cls,
        command: str,
        name: str = "cli-agent",
        model: str = "",
        timeout: float = 120,
        shell: bool = True,
    ) -> Agent:
        """Create an agent from a CLI command.

        Runs the command as a subprocess, passing the task via stdin,
        and captures stdout as the output.

        Args:
            command: The CLI command to run (e.g., "python my_agent.py").
            name: Display name for the agent.
            model: The model being used (for reporting).
            timeout: Command timeout in seconds.
            shell: Whether to run via shell.

        Example:
            >>> agent = Agent.from_cli("python my_agent.py")
            >>> agent = Agent.from_cli("node agent.js", name="js-agent")
        """
        async def cli_fn(task: str, **kwargs: Any) -> dict[str, Any]:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=task.encode()),
                timeout=timeout,
            )

            if proc.returncode != 0:
                error_msg = stderr.decode().strip() or f"Process exited with code {proc.returncode}"
                return {
                    "output": "",
                    "success": False,
                    "error": error_msg,
                }

            return {"output": stdout.decode().strip()}

        return cls(fn=cli_fn, name=name, model=model)

    @classmethod
    def from_langchain(
        cls,
        agent: Any,
        name: str = "langchain-agent",
        model: str = "",
    ) -> Agent:
        """Create a LitmusAI agent from a LangChain agent/chain.

        Supports LangChain AgentExecutor, RunnableSequence, and other
        invokable objects. Extracts tool calls and intermediate steps
        when available.

        Args:
            agent: A LangChain agent, chain, or runnable.
            name: Display name for the agent.
            model: The model being used (for reporting).

        Example:
            >>> from langchain.agents import AgentExecutor
            >>> lc_agent = AgentExecutor(agent=..., tools=...)
            >>> agent = Agent.from_langchain(lc_agent)
        """
        async def langchain_fn(task: str, **kwargs: Any) -> dict[str, Any]:
            # Support both ainvoke and invoke
            if hasattr(agent, "ainvoke"):
                result = await agent.ainvoke({"input": task, **kwargs})
            elif hasattr(agent, "invoke"):
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: agent.invoke({"input": task, **kwargs})
                )
            else:
                raise TypeError(
                    f"LangChain agent must have 'invoke' or 'ainvoke' method, "
                    f"got {type(agent).__name__}"
                )

            if isinstance(result, str):
                return {"output": result}

            if isinstance(result, dict):
                output = str(result.get("output", result.get("result", str(result))))

                # Extract tool calls from intermediate steps
                tool_calls = []
                for step in result.get("intermediate_steps", []):
                    if isinstance(step, tuple) and len(step) >= 2:
                        action, observation = step[0], step[1]
                        tool_calls.append({
                            "name": getattr(action, "tool", str(action)),
                            "arguments": getattr(action, "tool_input", {}),
                            "result": str(observation),
                        })

                # Extract steps
                steps = []
                for i, step in enumerate(result.get("intermediate_steps", [])):
                    if isinstance(step, tuple) and len(step) >= 2:
                        action = step[0]
                        steps.append({
                            "step_number": i + 1,
                            "action": getattr(action, "tool", str(action)),
                            "thought": getattr(action, "log", ""),
                            "observation": str(step[1]),
                        })

                return {
                    "output": output,
                    "tool_calls": tool_calls,
                    "steps": steps,
                }

            return {"output": str(result)}

        return cls(fn=langchain_fn, name=name, model=model)

    @classmethod
    def from_crewai(
        cls,
        crew: Any,
        name: str = "crewai-agent",
        model: str = "",
    ) -> Agent:
        """Create a LitmusAI agent from a CrewAI crew.

        Args:
            crew: A CrewAI Crew instance.
            name: Display name for the agent.
            model: The model being used (for reporting).

        Example:
            >>> from crewai import Crew
            >>> crew = Crew(agents=[...], tasks=[...])
            >>> agent = Agent.from_crewai(crew)
        """
        def crewai_fn(task: str, **kwargs: Any) -> dict[str, Any]:
            result = crew.kickoff(inputs={"task": task, **kwargs})

            output = str(result)

            # Extract token usage if available
            token_usage = {}
            if hasattr(result, "token_usage"):
                token_usage = {
                    "tokens_used": getattr(result.token_usage, "total_tokens", 0),
                    "input_tokens": getattr(result.token_usage, "prompt_tokens", 0),
                    "output_tokens": getattr(result.token_usage, "completion_tokens", 0),
                }

            # Extract task results as steps
            steps = []
            if hasattr(result, "tasks_output"):
                for i, task_output in enumerate(result.tasks_output):
                    steps.append({
                        "step_number": i + 1,
                        "action": getattr(task_output, "name", f"Task {i + 1}"),
                        "observation": str(getattr(task_output, "raw", "")),
                    })

            return {
                "output": output,
                "steps": steps,
                **token_usage,
            }

        return cls(fn=crewai_fn, name=name, model=model)

    @classmethod
    def from_openai_agent(
        cls,
        agent: Any,
        name: str = "openai-agent",
        model: str = "",
    ) -> Agent:
        """Create a LitmusAI agent from an OpenAI Agents SDK agent.

        Args:
            agent: An OpenAI Agent instance.
            name: Display name for the agent.
            model: The model being used (for reporting).

        Example:
            >>> from openai import Agent as OAIAgent
            >>> oai_agent = OAIAgent(name="helper", model="gpt-4o")
            >>> agent = Agent.from_openai_agent(oai_agent)
        """
        async def openai_fn(task: str, **kwargs: Any) -> dict[str, Any]:
            try:
                from openai.agents import Runner
            except ImportError:
                raise ImportError(
                    "OpenAI Agents SDK not installed. "
                    "Install with: pip install 'openai[agents]'"
                )

            result = await Runner.run(agent, input=task)

            output = str(result.final_output) if hasattr(result, "final_output") else str(result)

            # Extract tool calls
            tool_calls = []
            if hasattr(result, "raw_responses"):
                for resp in result.raw_responses:
                    for choice in getattr(resp, "choices", []):
                        msg = getattr(choice, "message", None)
                        if msg and hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                tool_calls.append({
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                })

            # Extract usage
            usage = {}
            if hasattr(result, "raw_responses") and result.raw_responses:
                last_resp = result.raw_responses[-1]
                if hasattr(last_resp, "usage") and last_resp.usage:
                    usage = {
                        "input_tokens": last_resp.usage.prompt_tokens,
                        "output_tokens": last_resp.usage.completion_tokens,
                        "tokens_used": last_resp.usage.total_tokens,
                    }

            return {
                "output": output,
                "model": getattr(agent, "model", model),
                "tool_calls": tool_calls,
                **usage,
            }

        return cls(fn=openai_fn, name=name, model=model)

    @classmethod
    def from_callable(
        cls,
        obj: Any,
        name: str = "callable-agent",
        model: str = "",
        method: str = "run",
    ) -> Agent:
        """Create an agent from any object with a callable method.

        This is a generic adapter for custom agent implementations.

        Args:
            obj: Any object with a method that accepts a task string.
            name: Display name for the agent.
            model: The model being used (for reporting).
            method: The method name to call on the object.

        Example:
            >>> class MyAgent:
            ...     def run(self, task: str) -> str:
            ...         return "response"
            >>> agent = Agent.from_callable(MyAgent(), method="run")
        """
        fn = getattr(obj, method, None)
        if fn is None or not callable(fn):
            raise TypeError(f"'{method}' on {type(obj).__name__} is not callable")
        return cls(fn=fn, name=name, model=model)

    def __repr__(self) -> str:
        return f"Agent(name='{self.name}', model='{self.model}')"
