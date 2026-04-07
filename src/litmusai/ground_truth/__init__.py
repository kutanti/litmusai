"""Ground truth management — verified answer database.

Provides :class:`GroundTruth` for structured, verifiable expected
answers with provenance metadata (source, verified_by, date).

Supports auto-generation of assertions from ground truth entries
and YAML loading/validation.

Example::

    gt = GroundTruth(answer=36, answer_type="numeric", tolerance=0.01)
    assertions = gt.to_assertions()
    # [Numeric(36, tolerance=0.01)]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GroundTruth:
    """Verified expected answer with provenance metadata.

    Attributes:
        answer: The correct answer (text, number, dict, list, or None for subjective).
        answer_type: One of ``"text"``, ``"numeric"``, ``"json"``,
            ``"boolean"``, ``"list"``, ``"subjective"``.
        tolerance: Acceptable error margin for numeric answers.
        alternatives: Acceptable alternative phrasings.
        source: Where this answer comes from.
        verified_by: Who verified it.
        verified_date: When verified (ISO format string).
        confidence: How confident we are (0.0–1.0).
        notes: Any caveats or context.
    """

    answer: str | float | dict[str, Any] | list[Any] | None = None
    answer_type: str = "text"
    tolerance: float | None = None
    alternatives: list[str] = field(default_factory=list)
    source: str | None = None
    verified_by: str | None = None
    verified_date: str | None = None
    confidence: float = 1.0
    notes: str | None = None

    VALID_TYPES = ("text", "numeric", "json", "boolean", "list", "subjective")

    def __post_init__(self) -> None:
        if self.answer_type not in self.VALID_TYPES:
            msg = (
                f"answer_type must be one of {self.VALID_TYPES}, "
                f"got '{self.answer_type}'"
            )
            raise ValueError(msg)
        if self.answer_type == "numeric" and self.answer is not None:
            self.answer = float(self.answer)  # type: ignore[arg-type]
        if self.confidence < 0.0 or self.confidence > 1.0:
            msg = f"confidence must be 0.0–1.0, got {self.confidence}"
            raise ValueError(msg)

    def to_assertions(self) -> list[Any]:
        """Generate assertions from this ground truth entry.

        Returns:
            List of :class:`~litmusai.assertions.Assertion` objects.
        """
        from litmusai.assertions import (
            AnyOf,
            Contains,
            JsonValid,
            LLMGrade,
            Numeric,
        )

        if self.answer_type == "subjective":
            note = self.notes or "Evaluate reasoning quality"
            return [LLMGrade(note)]

        assertions: list[Any] = []

        if self.answer_type == "numeric" and self.answer is not None:
            if isinstance(self.answer, (dict, list)):
                msg = (
                    f"answer_type is 'numeric' but answer is "
                    f"{type(self.answer).__name__}, expected a number"
                )
                raise ValueError(msg)
            num_val = float(self.answer)
            assertions.append(
                Numeric(num_val, tolerance=self.tolerance or 0.01),
            )
        elif self.answer_type == "json":
            assertions.append(JsonValid())
            if self.answer is not None:
                # Check for expected keys in dict answers
                if isinstance(self.answer, dict):
                    assertions.append(Contains([str(k) for k in self.answer]))
                else:
                    assertions.append(Contains([str(self.answer)]))
        elif self.answer_type == "boolean" and self.answer is not None:
            from litmusai.assertions import RegexMatch
            # Word-boundary match to avoid "untrue" matching "true"
            pattern = rf"\b{str(self.answer).lower()}\b"
            assertions.append(RegexMatch(pattern))
        elif self.answer_type == "list" and isinstance(self.answer, list):
            for item in self.answer:
                assertions.append(Contains([str(item)]))
        elif self.answer is not None:
            # text type
            assertions.append(Contains([str(self.answer)]))

        # Handle alternatives — wrap the last content assertion in AnyOf
        if self.alternatives and assertions:
            alt_assertions = [
                Contains([alt]) for alt in self.alternatives
            ]
            # Find the last content assertion (skip structural like JsonValid)
            for idx in range(len(assertions) - 1, -1, -1):
                if isinstance(assertions[idx], (Contains, Numeric)):
                    primary = assertions[idx]
                    assertions[idx] = AnyOf(primary, *alt_assertions)
                    break
            else:
                # No content assertion found, append alternatives separately
                assertions.append(AnyOf(*alt_assertions))

        return assertions

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        d: dict[str, Any] = {"answer_type": self.answer_type}
        if self.answer is not None:
            d["answer"] = self.answer
        if self.tolerance is not None:
            d["tolerance"] = self.tolerance
        if self.alternatives:
            d["alternatives"] = self.alternatives
        if self.source:
            d["source"] = self.source
        if self.verified_by:
            d["verified_by"] = self.verified_by
        if self.verified_date:
            d["verified_date"] = self.verified_date
        if self.confidence != 1.0:
            d["confidence"] = self.confidence
        if self.notes:
            d["notes"] = self.notes
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GroundTruth:
        """Deserialize from a dictionary."""
        return cls(
            answer=data.get("answer"),
            answer_type=data.get("answer_type", "text"),
            tolerance=data.get("tolerance"),
            alternatives=data.get("alternatives", []),
            source=data.get("source"),
            verified_by=data.get("verified_by"),
            verified_date=data.get("verified_date"),
            confidence=data.get("confidence", 1.0),
            notes=data.get("notes"),
        )


def load_ground_truth(path: str | Path) -> dict[str, GroundTruth]:
    """Load ground truth entries from a YAML file.

    The file should have a ``cases`` list with ``id`` and
    ``ground_truth`` keys::

        cases:
          - id: math_001
            ground_truth:
              answer: 36
              answer_type: numeric

    Args:
        path: Path to YAML file.

    Returns:
        Dict mapping case ID to :class:`GroundTruth`.
    """
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        return {}

    entries: dict[str, GroundTruth] = {}

    for item in data.get("cases", []):
        case_id = item.get("id")
        if not case_id:
            continue
        gt_data = item.get("ground_truth")
        if not gt_data:
            continue
        entries[case_id] = GroundTruth.from_dict(gt_data)

    return entries


def validate_ground_truth(
    path: str | Path,
) -> list[str]:
    """Validate a ground truth YAML file.

    Returns:
        List of error/warning messages. Empty = valid.
    """
    path = Path(path)
    errors: list[str] = []

    if not path.exists():
        errors.append(f"File not found: {path}")
        return errors

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        errors.append(f"YAML parse error: {e}")
        return errors

    if not isinstance(data, dict):
        errors.append("Root must be a mapping")
        return errors

    cases = data.get("cases", [])
    if not isinstance(cases, list):
        errors.append("'cases' must be a list")
        return errors

    seen_ids: set[str] = set()
    for i, item in enumerate(cases):
        if not isinstance(item, dict):
            errors.append(f"Case {i}: must be a mapping")
            continue

        case_id = item.get("id")
        if not case_id:
            errors.append(f"Case {i}: missing 'id'")
            continue

        if case_id in seen_ids:
            errors.append(f"Case '{case_id}': duplicate ID")
        seen_ids.add(case_id)

        gt_data = item.get("ground_truth")
        if not gt_data:
            errors.append(f"Case '{case_id}': missing 'ground_truth'")
            continue

        if not isinstance(gt_data, dict):
            errors.append(
                f"Case '{case_id}': 'ground_truth' must be a mapping, "
                f"got {type(gt_data).__name__}"
            )
            continue

        try:
            gt = GroundTruth.from_dict(gt_data)
            # Validate answer is present for non-subjective types
            if gt.answer_type != "subjective" and gt.answer is None:
                errors.append(
                    f"Case '{case_id}': non-subjective type "
                    f"'{gt.answer_type}' requires an answer"
                )
        except (ValueError, TypeError) as e:
            errors.append(f"Case '{case_id}': {e}")

    return errors


def apply_ground_truth(
    suite: Any,
    ground_truth: dict[str, GroundTruth],
) -> int:
    """Apply ground truth entries to a test suite's cases.

    For each case that has a matching ground truth entry and no
    existing assertions, generates assertions from the ground truth.

    Args:
        suite: A :class:`~litmusai.core.suite.TestSuite`.
        ground_truth: Dict mapping case ID to :class:`GroundTruth`.

    Returns:
        Number of cases updated.
    """
    updated = 0
    for case in suite.cases:
        gt = ground_truth.get(case.id)
        if gt is None:
            continue
        # Only apply if case has no assertions already
        if case.assertions:
            continue
        assertions = gt.to_assertions()
        if assertions:
            case.assertions = assertions
            # Store ground truth metadata
            case.metadata["ground_truth"] = gt.to_dict()
            updated += 1
    return updated


def ground_truth_stats(
    suite: Any,
    ground_truth: dict[str, GroundTruth],
) -> dict[str, Any]:
    """Compute ground truth coverage stats for a suite.

    Returns:
        Dict with coverage statistics.
    """
    total = len(suite.cases)
    covered = 0
    subjective = 0
    missing: list[str] = []

    for case in suite.cases:
        gt = ground_truth.get(case.id)
        if gt is None:
            missing.append(case.id)
        elif gt.answer_type == "subjective":
            subjective += 1
            covered += 1
        else:
            covered += 1

    return {
        "total": total,
        "covered": covered,
        "subjective": subjective,
        "missing": len(missing),
        "missing_ids": missing,
        "coverage_pct": (covered / total * 100) if total > 0 else 0,
    }
