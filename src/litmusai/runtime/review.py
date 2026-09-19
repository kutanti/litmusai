"""Deterministic review selection, independent of untrusted conversation text."""

import hashlib
import json

from litmusai.runtime.config import ReviewConfig
from litmusai.runtime.models import DetectionResult, RuntimeEvent


def select_review(event: RuntimeEvent, finding: DetectionResult, config: ReviewConfig) -> str:
    """Return a recorded selection reason without spending provider budget."""
    if finding.outcome in config.on_outcomes:
        return "uncertain_screen"
    if finding.outcome == "clear":
        key = json.dumps([event.project_id, event.event_id, config.version]).encode()
        bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        return (
            "audit_sample"
            if bucket < int(config.clear_sample_rate * 2**64)
            else "clear_not_sampled"
        )
    if finding.outcome == "detected":
        return "screen_detected"
    return "coverage_gap"
