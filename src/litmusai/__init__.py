"""LitmusAI — The open-source evaluation framework for AI agents."""

__version__ = "0.1.0"

from litmusai.core.agent import Agent
from litmusai.core.suite import TestSuite
from litmusai.core.runner import evaluate, compare
from litmusai.core.scorer import Scorer
from litmusai.core.reporter import Reporter

__all__ = [
    "Agent",
    "TestSuite",
    "evaluate",
    "compare",
    "Scorer",
    "Reporter",
]
