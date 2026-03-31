"""Framework adapters for LitmusAI.

Each adapter provides a bridge between a specific agent framework
and LitmusAI's universal Agent interface.

Quick reference:
    - Agent.from_function(fn)          → Simple Python function
    - Agent.from_url(url)              → HTTP endpoint
    - Agent.from_cli(cmd)              → CLI subprocess
    - Agent.from_langchain(agent)      → LangChain
    - Agent.from_crewai(crew)          → CrewAI
    - Agent.from_openai_agent(agent)   → OpenAI Agents SDK
    - Agent.from_callable(obj)         → Any callable object
"""

from litmusai.core.agent import Agent, AgentResponse, AgentStep, ToolCall

__all__ = [
    "Agent",
    "AgentResponse",
    "AgentStep",
    "ToolCall",
]
