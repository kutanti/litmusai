"""LitmusAI — The open-source evaluation framework for AI agents."""

__version__ = "0.1.0"

from litmusai.assertions import (
    All,
    AnyOf,
    Assertion,
    AssertionResult,
    AtLeast,
    Contains,
    Custom,
    Exact,
    JsonPath,
    JsonSchema,
    JsonValid,
    LLMGrade,
    NotContains,
    Numeric,
    RegexMatch,
    Semantic,
    Weighted,
)
from litmusai.core.agent import Agent, AgentResponse, AgentStep, ToolCall
from litmusai.core.reporter import Reporter
from litmusai.core.runner import compare, evaluate, multi_evaluate
from litmusai.core.scorer import Scorer
from litmusai.core.suite import TestCase, TestSuite
from litmusai.globals import configure, get_config, reset_config

__all__ = [
    # Core
    "Agent",
    "AgentResponse",
    "AgentStep",
    "ToolCall",
    "TestCase",
    "TestSuite",
    "evaluate",
    "multi_evaluate",
    "compare",
    "Scorer",
    "Reporter",
    # Assertions
    "All",
    "AnyOf",
    "Assertion",
    "AssertionResult",
    "AtLeast",
    "Contains",
    "Custom",
    "Exact",
    "JsonPath",
    "JsonSchema",
    "JsonValid",
    "LLMGrade",
    "NotContains",
    "Numeric",
    "RegexMatch",
    "Semantic",
    "Weighted",
    # Global config
    "configure",
    "get_config",
    "reset_config",
]
