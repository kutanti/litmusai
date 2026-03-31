"""Agent abstraction layer — wrap any agent framework."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class AgentResponse:
    """Response from an agent execution."""
    output: str
    metadata: dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0
    latency_ms: float = 0.0
    tokens_used: int = 0
    model: str = ""
    success: bool = True
    error: str | None = None


class Agent:
    """Universal agent wrapper — connect any AI agent to LitmusAI."""

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
        """Execute the agent on a given task."""
        import time
        start = time.monotonic()

        try:
            if asyncio.iscoroutinefunction(self.fn):
                result = await self.fn(task, **kwargs)
            else:
                result = self.fn(task, **kwargs)

            elapsed = (time.monotonic() - start) * 1000

            if isinstance(result, AgentResponse):
                result.latency_ms = elapsed
                return result
            elif isinstance(result, str):
                return AgentResponse(output=result, latency_ms=elapsed, model=self.model)
            elif isinstance(result, dict):
                return AgentResponse(
                    output=str(result.get("output", str(result))),
                    cost=float(result.get("cost", 0.0)),
                    tokens_used=int(result.get("tokens_used", 0)),
                    model=str(result.get("model", self.model)),
                    latency_ms=elapsed,
                    metadata=result,
                )
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

    @classmethod
    def from_function(cls, fn: Callable[..., Any], name: str = "agent", model: str = "") -> Agent:
        """Create an agent from a simple function."""
        return cls(fn=fn, name=name, model=model)

    @classmethod
    def from_url(cls, url: str, name: str = "http-agent", model: str = "") -> Agent:
        """Create an agent from an HTTP endpoint."""
        async def http_fn(task: str, **kwargs: Any) -> str:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(url, json={"task": task, **kwargs})
                response.raise_for_status()
                data = response.json()
                return str(data.get("output", data.get("response", str(data))))

        return cls(fn=http_fn, name=name, model=model)

    @classmethod
    def from_langchain(cls, agent: Any, name: str = "langchain-agent") -> Agent:
        """Create a LitmusAI agent from a LangChain agent."""
        async def langchain_fn(task: str, **kwargs: Any) -> str:
            result = await agent.ainvoke({"input": task, **kwargs})
            return str(result.get("output", str(result)))

        return cls(fn=langchain_fn, name=name)

    @classmethod
    def from_crewai(cls, crew: Any, name: str = "crewai-agent") -> Agent:
        """Create a LitmusAI agent from a CrewAI crew."""
        def crewai_fn(task: str, **kwargs: Any) -> str:
            result = crew.kickoff(inputs={"task": task, **kwargs})
            return str(result)

        return cls(fn=crewai_fn, name=name)

    def __repr__(self) -> str:
        return f"Agent(name='{self.name}', model='{self.model}')"
