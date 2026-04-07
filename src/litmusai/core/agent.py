"""Agent abstraction layer — wrap any agent framework.

This module provides a universal Agent class that normalizes different
agent frameworks into a single interface for evaluation.

Built-in adapters:
    - Simple Python functions (sync/async)
    - HTTP endpoints (REST APIs)
    - CLI-based agents (subprocess)
    - LangChain agents/chains
    - CrewAI crews
    - OpenAI Agents SDK
    - Any callable object

Additional adapters (AutoGen, OpenClaw, etc.) can be added using
the from_function() or from_callable() escape hatches, or by
contributing new adapters to the project.
"""

from __future__ import annotations

import asyncio
import json as _json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx


def _safe_int(value: Any) -> int:
    """Convert a value to int safely, returning 0 for None/invalid."""
    if value is None:
        return 0
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


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
        model_params: dict[str, Any] | None = None,
    ):
        self.fn = fn
        self.name = name
        self.model = model
        self.metadata = metadata or {}
        self.model_params = model_params or {}

    async def run(self, task: str, **kwargs: Any) -> AgentResponse:
        """Execute the agent on a given task.

        Args:
            task: The task/prompt to send to the agent.
            **kwargs: Additional arguments passed to the agent function.
                Includes ``history`` (list of message dicts) for
                multi-turn conversations.

        Returns:
            AgentResponse with normalized output, cost, latency, etc.
        """
        start = time.monotonic()

        try:
            if asyncio.iscoroutinefunction(self.fn):
                result = await self.fn(task, **kwargs)
            else:
                result = await asyncio.to_thread(self.fn, task, **kwargs)

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

    def conversation(
        self, system_prompt: str | None = None,
    ) -> Any:
        """Create a stateful conversation context.

        Usage::

            async with agent.conversation() as conv:
                r1 = await conv.send("Hello")
                r2 = await conv.send("What did I just say?")

        Args:
            system_prompt: Optional system prompt.

        Returns:
            :class:`~litmusai.conversation.Conversation` context manager.
        """
        from litmusai.conversation import Conversation

        return Conversation(self, system_prompt=system_prompt)

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

        excluded_keys = {
            "output", "cost", "tokens_used", "input_tokens",
            "output_tokens", "model", "tool_calls", "steps",
            "success", "error",
        }

        return AgentResponse(
            output=str(result.get("output", str(result))),
            cost=float(result.get("cost", 0.0)),
            tokens_used=int(result.get("tokens_used", 0)),
            input_tokens=int(result.get("input_tokens", 0)),
            output_tokens=int(result.get("output_tokens", 0)),
            model=str(result.get("model", self.model)),
            latency_ms=elapsed,
            success=bool(result.get("success", True)),
            error=result.get("error"),
            metadata={k: v for k, v in result.items()
                      if k not in excluded_keys},
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
            shell: Whether to run via shell. If False, command is split
                   using shlex and executed directly.

        Example:
            >>> agent = Agent.from_cli("python my_agent.py")
            >>> agent = Agent.from_cli("node agent.js", name="js-agent")
        """
        async def cli_fn(task: str, **kwargs: Any) -> dict[str, Any]:
            if shell:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            else:
                import shlex
                args = shlex.split(command)
                proc = await asyncio.create_subprocess_exec(
                    *args,
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
                result = await asyncio.to_thread(
                    agent.invoke, {"input": task, **kwargs}
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
    def from_openai_chat(
        cls,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = "gpt-4o",
        name: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 1024,
        seed: int | None = None,
        timeout: float = 120,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> Agent:
        """Create an agent from any OpenAI-compatible chat API.

        Works with OpenAI, Anthropic (via proxy), LiteLLM,
        Ollama, vLLM, Together AI, Fireworks, and any provider that
        implements the ``/v1/chat/completions`` endpoint.

        For Azure OpenAI, use the Azure-specific endpoint directly
        via :meth:`from_url` or pass the full Azure URL as
        ``base_url`` with ``extra_headers`` for the api-key.

        **Automatically captures real token usage** from the API
        response — no guessing.

        Args:
            base_url: API base URL (e.g. ``https://api.openai.com/v1``).
            api_key: API key / bearer token.
            model: Model identifier (e.g. ``gpt-4o``, ``claude-sonnet-4``).
            name: Display name (defaults to model name).
            system_prompt: Optional system message prepended to every call.
            temperature: Sampling temperature (``None`` = provider default).
            max_tokens: Maximum tokens in the completion.
            seed: Random seed for reproducible outputs (provider support varies).
            timeout: HTTP timeout in seconds.
            extra_headers: Additional HTTP headers.
            extra_body: Extra fields merged into the request body.

        Returns:
            An :class:`Agent` that produces :class:`AgentResponse` with
            real ``input_tokens``, ``output_tokens``, ``tokens_used``,
            and computed ``cost`` when pricing is registered.

        Example:
            >>> agent = Agent.from_openai_chat(
            ...     base_url="https://api.openai.com/v1",
            ...     api_key="sk-...",
            ...     model="gpt-4o",
            ... )
            >>> resp = await agent.run("What is 2+2?")
            >>> print(resp.input_tokens, resp.output_tokens)
        """
        display_name = name or model
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            **(extra_headers or {}),
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Normalise base_url — strip trailing /v1 so we can add our own
        clean_url = base_url.rstrip("/")
        if not clean_url.endswith("/v1"):
            endpoint = f"{clean_url}/v1/chat/completions"
        else:
            endpoint = f"{clean_url}/chat/completions"

        async def openai_chat_fn(task: str, **kwargs: Any) -> AgentResponse:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Support conversation history for multi-turn
            history = kwargs.pop("history", None)
            if history:
                messages.extend(history)

            messages.append({"role": "user", "content": task})

            body: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            if temperature is not None:
                body["temperature"] = temperature
            if seed is not None:
                body["seed"] = seed
            if extra_body:
                body.update(extra_body)

            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(endpoint, headers=headers,
                                      json=body)
                r.raise_for_status()
                data = r.json()

            # ── Parse response ─────────────────────────────
            content = ""
            choices = data.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                content = msg.get("content", "") or ""

            # ── Extract real token usage ───────────────────
            usage = data.get("usage") or {}
            inp_tok = _safe_int(usage.get("prompt_tokens"))
            out_tok = _safe_int(usage.get("completion_tokens"))
            total_tok = _safe_int(
                usage.get("total_tokens", inp_tok + out_tok),
            )

            # ── Compute cost from pricing DB ───────────────
            # Try the model name returned by the API first (e.g.
            # "gpt-4o-2026-03-15"), then fall back to the requested
            # model name. The pricing DB supports fuzzy matching.
            resp_model = data.get("model", model)
            cost = 0.0
            try:
                from litmusai.benchmarks import get_pricing
                pricing = (
                    get_pricing(resp_model) or get_pricing(model)
                )
                if pricing and (inp_tok or out_tok):
                    cost = (
                        inp_tok * pricing.input_cost_per_token
                        + out_tok * pricing.output_cost_per_token
                    )
            except ImportError:
                pass

            # ── Extract tool calls if present ──────────────
            tool_calls: list[ToolCall] = []
            if choices:
                msg = choices[0].get("message", {})
                for tc in msg.get("tool_calls", []):
                    fn = tc.get("function", {})
                    raw_args = fn.get("arguments", "{}")
                    # Some providers return arguments as a dict
                    # already; others return a JSON string.
                    if isinstance(raw_args, dict):
                        args = raw_args
                    else:
                        try:
                            args = _json.loads(str(raw_args))
                        except (ValueError, TypeError):
                            args = {"raw": raw_args}
                    tool_calls.append(ToolCall(
                        name=fn.get("name", ""),
                        arguments=args if isinstance(args, dict) else {"raw": args},
                    ))

            return AgentResponse(
                output=content,
                input_tokens=inp_tok,
                output_tokens=out_tok,
                tokens_used=total_tok,
                cost=cost,
                model=resp_model,
                tool_calls=tool_calls,
            )

        # Build model_params for logging
        params: dict[str, Any] = {"max_tokens": max_tokens}
        if temperature is not None:
            params["temperature"] = temperature
        if seed is not None:
            params["seed"] = seed
        # Reflect extra_body overrides in logged params
        if extra_body:
            for key in ("temperature", "max_tokens", "seed"):
                if key in extra_body:
                    params[key] = extra_body[key]

        return cls(
            fn=openai_chat_fn,
            name=display_name,
            model=model,
            model_params=params,
        )

    @classmethod
    def from_azure(
        cls,
        *,
        resource: str,
        deployment: str,
        api_key: str = "",
        api_version: str = "2024-08-01-preview",
        name: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 1024,
        timeout: float = 120,
    ) -> Agent:
        """Create an agent from Azure OpenAI.

        Builds the correct Azure URL and uses ``api-key`` header
        authentication automatically.

        Args:
            resource: Azure resource name (e.g. ``"my-resource"``).
            deployment: Model deployment name (e.g. ``"gpt-4o"``).
            api_key: Azure API key. Falls back to ``AZURE_OPENAI_API_KEY``
                env var, then global config.
            api_version: Azure API version string.
            name: Display name (defaults to deployment name).
            system_prompt: Optional system message.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the completion.
            timeout: HTTP timeout in seconds.

        Returns:
            An :class:`Agent` configured for Azure OpenAI.

        Example:
            >>> agent = Agent.from_azure(
            ...     resource="my-resource",
            ...     deployment="gpt-4o",
            ...     api_key="your-azure-key",
            ... )
        """
        import os

        resolved_key = (
            api_key
            or os.getenv("AZURE_OPENAI_API_KEY", "")
        )
        if not resolved_key:
            from litmusai.globals import get_config
            resolved_key = get_config().api_key

        if not resolved_key:
            msg = (
                "No Azure API key provided. Pass api_key='...', "
                "set AZURE_OPENAI_API_KEY, or call "
                "litmusai.configure(api_key='...')."
            )
            raise ValueError(msg)

        # Azure uses deployment-scoped URLs with api-version query param
        endpoint = (
            f"https://{resource}.openai.azure.com"
            f"/openai/deployments/{deployment}"
            f"/chat/completions?api-version={api_version}"
        )
        display_name = name or deployment

        az_headers: dict[str, str] = {
            "Content-Type": "application/json",
            "api-key": resolved_key,
        }

        import httpx

        async def azure_chat_fn(task: str, **kwargs: Any) -> AgentResponse:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Support conversation history for multi-turn
            history = kwargs.pop("history", None)
            if history:
                messages.extend(history)

            messages.append({"role": "user", "content": task})

            body: dict[str, Any] = {
                "model": deployment,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            if temperature is not None:
                body["temperature"] = temperature

            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(
                    endpoint, headers=az_headers, json=body,
                )
                r.raise_for_status()
                data = r.json()

            choice = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            input_tok = usage.get("prompt_tokens", 0)
            output_tok = usage.get("completion_tokens", 0)

            cost = 0.0
            try:
                from litmusai.benchmarks import get_pricing
                pricing = get_pricing(deployment)
                if pricing:
                    cost = (
                        input_tok * pricing.input_cost_per_token
                        + output_tok * pricing.output_cost_per_token
                    )
            except ImportError:
                pass

            return AgentResponse(
                output=choice,
                model=deployment,
                input_tokens=input_tok,
                output_tokens=output_tok,
                tokens_used=input_tok + output_tok,
                cost=cost,
            )

        params: dict[str, Any] = {"max_tokens": max_tokens}
        if temperature is not None:
            params["temperature"] = temperature

        return cls(
            fn=azure_chat_fn,
            name=display_name,
            model=deployment,
            model_params=params,
        )

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
