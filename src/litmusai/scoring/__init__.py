"""Multi-dimensional scoring — Beyond pass/fail.

Provides :class:`ScoreVector` with 7 quality dimensions:
correctness, completeness, format, relevance, safety, latency, cost.

Each dimension is a 0.0–1.0 float.  A configurable weighted
composite produces the ``overall`` score.

Example::

    vector = ScoreVector(correctness=1.0, latency=0.7)
    vector.compute_overall()  # uses default weights
    print(vector.overall)     # ~0.435 (only 2 of 7 dimensions set)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Default dimension weights — correctness-heavy, cost/latency secondary.
DEFAULT_WEIGHTS: dict[str, float] = {
    "correctness": 0.40,
    "completeness": 0.20,
    "format": 0.10,
    "relevance": 0.10,
    "safety": 0.10,
    "latency": 0.05,
    "cost": 0.05,
}

DIMENSIONS = list(DEFAULT_WEIGHTS.keys())


@dataclass
class DimensionBudget:
    """Budget thresholds for latency/cost dimensions.

    Values below the budget get score 1.0.
    Values above the max get score 0.0.
    Values in between are linearly interpolated.

    Attributes:
        latency_ms: Target latency in milliseconds.
        latency_max_ms: Max acceptable latency (score = 0).
        cost_usd: Target cost in USD.
        cost_max_usd: Max acceptable cost (score = 0).
    """

    latency_ms: float = 2000.0
    latency_max_ms: float = 10000.0
    cost_usd: float = 0.01
    cost_max_usd: float = 0.10

    def score_latency(self, actual_ms: float) -> float:
        """Score latency on a 0-1 scale."""
        if actual_ms <= self.latency_ms:
            return 1.0
        if (
            actual_ms >= self.latency_max_ms
            or self.latency_max_ms <= self.latency_ms
        ):
            return 0.0
        # Linear interpolation
        return 1.0 - (
            (actual_ms - self.latency_ms)
            / (self.latency_max_ms - self.latency_ms)
        )

    def score_cost(self, actual_usd: float) -> float:
        """Score cost on a 0-1 scale."""
        if actual_usd <= self.cost_usd:
            return 1.0
        if (
            actual_usd >= self.cost_max_usd
            or self.cost_max_usd <= self.cost_usd
        ):
            return 0.0
        return 1.0 - (
            (actual_usd - self.cost_usd)
            / (self.cost_max_usd - self.cost_usd)
        )


@dataclass
class ScoreVector:
    """Multi-dimensional quality score.

    Each dimension is a 0.0–1.0 float. Call :meth:`compute_overall`
    to produce a weighted composite.

    Attributes:
        correctness: Is the answer right?
        completeness: Did it cover everything asked?
        format: Right structure/format?
        relevance: On-topic?
        safety: No harmful/leaked content?
        latency: Within time budget?
        cost: Within cost budget?
        overall: Weighted composite (set by :meth:`compute_overall`).
        details: Per-dimension explanations.
    """

    correctness: float = 0.0
    completeness: float = 0.0
    format: float = 0.0
    relevance: float = 0.0
    safety: float = 0.0
    latency: float = 0.0
    cost: float = 0.0
    overall: float = 0.0
    details: dict[str, str] = field(default_factory=dict)

    def compute_overall(
        self, weights: dict[str, float] | None = None,
    ) -> float:
        """Compute weighted composite from all dimensions.

        Args:
            weights: Custom weights per dimension.
                     Missing keys use defaults. Values are
                     normalized so they sum to 1.0.

        Returns:
            The ``overall`` score (also stored on ``self.overall``).
        """
        w = dict(DEFAULT_WEIGHTS)
        if weights:
            w.update(weights)

        # Normalize weights to sum to 1.0
        total = sum(w.get(d, 0) for d in DIMENSIONS)
        if total <= 0:
            self.overall = 0.0
            return 0.0

        self.overall = sum(
            getattr(self, dim) * (w.get(dim, 0) / total)
            for dim in DIMENSIONS
        )
        return self.overall

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        d = asdict(self)
        # Remove details if empty
        if not d.get("details"):
            d.pop("details", None)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScoreVector:
        """Deserialize from a dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{
            k: v for k, v in data.items()
            if k in valid_fields
        })

    def dimension_table(self) -> list[tuple[str, float, str]]:
        """Return list of (name, score, detail) tuples for display."""
        return [
            (dim, getattr(self, dim), self.details.get(dim, ""))
            for dim in DIMENSIONS
        ]


def build_score_vector(
    *,
    score_result: Any,
    response: Any,
    budget: DimensionBudget | None = None,
) -> ScoreVector:
    """Build a :class:`ScoreVector` from scoring + response data.

    Populates dimensions automatically:

    - **correctness**: from ``score_result.score``
    - **completeness**: from assertion details if available
    - **format**: from format-related assertions (JsonValid, etc.)
    - **relevance**: from relevance assertions (Semantic, etc.)
    - **safety**: 1.0 unless safety assertions failed
    - **latency**: from response latency vs budget
    - **cost**: from response cost vs budget

    Args:
        score_result: A :class:`ScoreResult` from the scorer.
        response: An :class:`AgentResponse` from the agent.
        budget: Optional :class:`DimensionBudget` for latency/cost.

    Returns:
        Populated :class:`ScoreVector` with ``overall`` computed.
    """
    budget = budget or DimensionBudget()
    details: dict[str, str] = {}

    # Start with correctness = overall assertion score
    correctness = getattr(score_result, "score", 0.0)
    details["correctness"] = getattr(score_result, "reason", "")

    # Parse assertion details if available
    assertion_details = _get_assertion_details(score_result)

    # Categorize assertions into dimensions
    format_scores: list[float] = []
    relevance_scores: list[float] = []
    safety_scores: list[float] = []
    correctness_scores: list[float] = []

    for a in assertion_details:
        atype = a.get("type", "").lower()
        ascore = a.get("score", 0.0)

        if atype in (
            "jsonvalid", "jsonschema", "jsonpath",
            "json_valid", "json_schema", "json_path",
            "regex", "regexmatch", "regex_match",
        ):
            format_scores.append(ascore)
        elif atype in ("semantic", "llmgrade", "llm_grade"):
            relevance_scores.append(ascore)
        elif atype in ("notcontains", "not_contains"):
            # NotContains can indicate safety or correctness
            safety_scores.append(ascore)
        elif atype in ("contains", "exact", "numeric"):
            correctness_scores.append(ascore)
        else:
            # Unknown type → correctness
            correctness_scores.append(ascore)

    # Override correctness if specific correctness assertions exist
    if correctness_scores:
        correctness = sum(correctness_scores) / len(correctness_scores)
        details["correctness"] = (
            f"{sum(1 for s in correctness_scores if s >= 1.0)}"
            f"/{len(correctness_scores)} passed"
        )

    # Completeness — heuristic based on output length
    output = getattr(response, "output", "") or ""
    if len(output.strip()) > 0:
        completeness = min(1.0, len(output.strip()) / 50)
    else:
        completeness = 0.0
    if completeness > 0:
        details["completeness"] = f"{len(output)} chars"

    # Format
    if format_scores:
        fmt = sum(format_scores) / len(format_scores)
        details["format"] = (
            f"{sum(1 for s in format_scores if s >= 1.0)}"
            f"/{len(format_scores)} format checks passed"
        )
    else:
        fmt = 1.0 if correctness >= 0.5 else 0.5
        details["format"] = "No format assertions"

    # Relevance
    if relevance_scores:
        relevance = sum(relevance_scores) / len(relevance_scores)
        details["relevance"] = (
            f"{sum(1 for s in relevance_scores if s >= 1.0)}"
            f"/{len(relevance_scores)} relevance checks"
        )
    else:
        relevance = correctness  # proxy
        details["relevance"] = "Inferred from correctness"

    # Safety
    if safety_scores:
        safety = sum(safety_scores) / len(safety_scores)
        details["safety"] = (
            f"{sum(1 for s in safety_scores if s >= 1.0)}"
            f"/{len(safety_scores)} safety checks"
        )
    else:
        safety = 1.0
        details["safety"] = "No safety assertions"

    # Latency
    latency_ms = getattr(response, "latency_ms", 0.0) or 0.0
    latency = budget.score_latency(latency_ms)
    details["latency"] = f"{latency_ms:.0f}ms (budget: {budget.latency_ms:.0f}ms)"

    # Cost
    cost_val = getattr(response, "cost", 0.0) or 0.0
    cost = budget.score_cost(cost_val)
    if cost_val > 0:
        details["cost"] = f"${cost_val:.4f} (budget: ${budget.cost_usd:.4f})"
    else:
        cost = 1.0
        details["cost"] = "No cost data"

    vector = ScoreVector(
        correctness=correctness,
        completeness=min(1.0, completeness),
        format=fmt,
        relevance=relevance,
        safety=safety,
        latency=latency,
        cost=cost,
        details=details,
    )
    vector.compute_overall()
    return vector


def aggregate_vectors(
    vectors: list[ScoreVector],
    weights: dict[str, float] | None = None,
) -> ScoreVector:
    """Compute mean :class:`ScoreVector` from multiple results.

    Args:
        vectors: Individual score vectors.
        weights: Optional custom weights for the overall score.

    Returns:
        Averaged :class:`ScoreVector`.
    """
    if not vectors:
        return ScoreVector()

    n = len(vectors)
    avg = ScoreVector(
        correctness=sum(v.correctness for v in vectors) / n,
        completeness=sum(v.completeness for v in vectors) / n,
        format=sum(v.format for v in vectors) / n,
        relevance=sum(v.relevance for v in vectors) / n,
        safety=sum(v.safety for v in vectors) / n,
        latency=sum(v.latency for v in vectors) / n,
        cost=sum(v.cost for v in vectors) / n,
    )
    avg.compute_overall(weights)
    return avg


def _get_assertion_details(score_result: Any) -> list[dict[str, Any]]:
    """Extract assertion details from a ScoreResult."""
    details = getattr(score_result, "details", None)
    if not details or not isinstance(details, dict):
        return []
    assertions: list[dict[str, Any]] = details.get("assertions", [])
    return assertions
