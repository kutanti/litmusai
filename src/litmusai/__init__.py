"""LitmusAI — The open-source evaluation framework for AI agents."""

__version__ = "0.2.1"

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
from litmusai.pipeline import Pipeline, PipelineResult, run_pipeline
from litmusai.profiles import (
    EvalProfile,
    get_profile,
    list_profiles,
    register_profile,
)

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
    # Pipeline
    "Pipeline",
    "PipelineResult",
    "run_pipeline",
    # Profiles
    "EvalProfile",
    "get_profile",
    "list_profiles",
    "register_profile",
]
