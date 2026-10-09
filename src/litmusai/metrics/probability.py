"""Metrics for detectors that return a probability instead of a hard label.

Every function takes parallel ``scores`` and ``labels`` sequences. A score is
the detector's probability that the example is positive, in ``[0, 1]``. A
score of ``None`` records a failed prediction: it is excluded from the metric
and reported through :func:`score_coverage`, never coerced to ``0.0``,
because a coerced failure would read as a confident negative.

The false-alarm rate is measured per scored example, which is the unit the
labels describe. It is not a per-event or per-session alert rate.
"""

from __future__ import annotations

import math
import random
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterable, Sequence
from itertools import groupby
from operator import itemgetter
from statistics import NormalDist
from typing import Any

__all__ = [
    "apply_temperature", "bootstrap_interval", "brier_score", "confusion_at",
    "cost_per_1000_events", "estimated_cost_usd", "expected_calibration_error",
    "fit_temperature", "latency_summary", "percentile", "probability_report",
    "recall_at_false_alarm_rate", "roc_auc", "score_coverage", "threshold_sweep",
    "wilson_interval",
]

_EPSILON = 1e-6


def _pairs(
    scores: Sequence[float | None], labels: Sequence[bool | int],
) -> list[tuple[float, bool]]:
    """Validate inputs and return the scored (probability, label) pairs."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    pairs: list[tuple[float, bool]] = []
    for score, label in zip(scores, labels):
        if label not in (True, False, 0, 1):
            raise ValueError("labels must be booleans or 0/1")
        if score is None:
            continue
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("scores must be numbers or None")
        value = float(score)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("scores must be finite probabilities between 0 and 1")
        pairs.append((value, bool(label)))
    return pairs


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def score_coverage(scores: Sequence[float | None]) -> dict[str, Any]:
    """Count scored and failed predictions so partial runs cannot look complete."""
    scored = sum(score is not None for score in scores)
    return {
        "total": len(scores),
        "scored": scored,
        "failed": len(scores) - scored,
        "coverage": _rate(scored, len(scores)),
    }


def confusion_at(
    scores: Sequence[float | None], labels: Sequence[bool | int], threshold: float,
) -> dict[str, Any]:
    """Confusion counts when ``score >= threshold`` is treated as positive."""
    pairs = _pairs(scores, labels)
    tp = sum(1 for score, label in pairs if label and score >= threshold)
    fp = sum(1 for score, label in pairs if not label and score >= threshold)
    fn = sum(1 for score, label in pairs if label and score < threshold)
    tn = sum(1 for score, label in pairs if not label and score < threshold)
    return _counts(threshold, tp, fp, tn, fn)


def _counts(threshold: float | None, tp: int, fp: int, tn: int, fn: int) -> dict[str, Any]:
    precision = _rate(tp, tp + fp)
    recall = _rate(tp, tp + fn)
    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "recall": recall,
        "false_alarm_rate": _rate(fp, fp + tn),
        "precision": precision,
        "f1": _rate(2 * tp, 2 * tp + fp + fn),
    }


def threshold_sweep(
    scores: Sequence[float | None],
    labels: Sequence[bool | int],
    thresholds: Iterable[float] | None = None,
) -> list[dict[str, Any]]:
    """Confusion counts at each threshold, in ascending threshold order.

    Without explicit thresholds every distinct score is used, which gives the
    exact operating points the detector can reach.
    """
    pairs = _pairs(scores, labels)
    grid = sorted(set(thresholds)) if thresholds is not None else sorted({s for s, _ in pairs})
    for threshold in grid:
        if not math.isfinite(threshold):
            raise ValueError("thresholds must be finite")
    positives = sorted(score for score, label in pairs if label)
    negatives = sorted(score for score, label in pairs if not label)
    points = []
    for threshold in grid:
        fn = bisect_left(positives, threshold)
        tn = bisect_left(negatives, threshold)
        points.append(_counts(threshold, len(positives) - fn, len(negatives) - tn, tn, fn))
    return points


def recall_at_false_alarm_rate(
    scores: Sequence[float | None],
    labels: Sequence[bool | int],
    max_false_alarm_rate: float,
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Highest recall whose false-alarm rate stays at or below the target.

    Returns the lowest threshold meeting the target, since recall can only
    fall as the threshold rises. When no reachable threshold meets the target,
    the result is the flag-nothing point: ``threshold`` is ``None`` and recall
    is zero.
    """
    if not 0.0 <= max_false_alarm_rate <= 1.0:
        raise ValueError("max_false_alarm_rate must be between 0 and 1")
    pairs = _pairs(scores, labels)
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        raise ValueError("recall at a false-alarm rate needs positive and negative examples")
    point = next(
        (
            candidate
            for candidate in threshold_sweep([s for s, _ in pairs], [y for _, y in pairs])
            if candidate["false_alarm_rate"] <= max_false_alarm_rate
        ),
        _counts(None, 0, 0, negatives, positives),
    )
    return {
        **point,
        "target_false_alarm_rate": max_false_alarm_rate,
        "recall_interval": list(wilson_interval(point["tp"], positives, confidence)),
        "confidence": confidence,
    }


def roc_auc(scores: Sequence[float | None], labels: Sequence[bool | int]) -> float | None:
    """Area under the ROC curve, counting tied scores as half-correct."""
    pairs = _pairs(scores, labels)
    positives = [score for score, label in pairs if label]
    negatives = [score for score, label in pairs if not label]
    if not positives or not negatives:
        return None
    ordered = sorted(pairs, key=lambda pair: pair[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index
        while end + 1 < len(ordered) and ordered[end + 1][0] == ordered[index][0]:
            end += 1
        average_rank = (index + end) / 2 + 1
        rank_sum += average_rank * sum(1 for _, label in ordered[index:end + 1] if label)
        index = end + 1
    n_pos, n_neg = len(positives), len(negatives)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def brier_score(scores: Sequence[float | None], labels: Sequence[bool | int]) -> float | None:
    """Mean squared difference between the probability and the 0/1 label."""
    pairs = _pairs(scores, labels)
    if not pairs:
        return None
    return sum((score - float(label)) ** 2 for score, label in pairs) / len(pairs)


def expected_calibration_error(
    scores: Sequence[float | None],
    labels: Sequence[bool | int],
    *,
    bins: int = 10,
    strategy: str = "uniform",
) -> dict[str, Any]:
    """Weighted gap between predicted probability and observed positive rate.

    ``uniform`` uses equal-width probability bins; ``quantile`` uses bins with
    roughly equal numbers of examples, which is steadier when scores cluster
    near 0 or 1. Equal scores always share a quantile bin, so the result does not
    depend on the order of tied examples. The maximum per-bin gap is reported
    alongside.
    """
    if bins < 1:
        raise ValueError("bins must be at least 1")
    if strategy not in {"uniform", "quantile"}:
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    pairs = _pairs(scores, labels)
    if not pairs:
        return {"ece": None, "max_calibration_error": None, "bins": []}
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    if strategy == "uniform":
        for score, label in pairs:
            buckets[min(int(score * bins), bins - 1)].append((score, label))
    else:
        # Positions where an equal-count split would start a new bin. A run of equal
        # scores goes whole into the bin where it starts instead of being split.
        edges = [len(pairs) * i // bins for i in range(1, bins)]
        position = 0
        for _, group in groupby(sorted(pairs, key=itemgetter(0)), key=itemgetter(0)):
            tied = list(group)
            buckets[bisect_right(edges, position)].extend(tied)
            position += len(tied)
    total = len(pairs)
    ece = 0.0
    worst = 0.0
    details = []
    for bucket in buckets:
        if not bucket:
            continue
        mean_score = sum(score for score, _ in bucket) / len(bucket)
        observed = sum(label for _, label in bucket) / len(bucket)
        gap = abs(mean_score - observed)
        ece += len(bucket) / total * gap
        worst = max(worst, gap)
        details.append({
            "count": len(bucket),
            "min_score": min(score for score, _ in bucket),
            "max_score": max(score for score, _ in bucket),
            "mean_score": mean_score,
            "observed_rate": observed,
            "gap": gap,
        })
    return {"ece": ece, "max_calibration_error": worst, "bins": details, "strategy": strategy}


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion; ``(0.0, 1.0)`` when ``total`` is zero."""
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("successes must be between 0 and total")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if total == 0:
        return (0.0, 1.0)
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
    margin /= denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def bootstrap_interval(
    scores: Sequence[float | None],
    labels: Sequence[bool | int],
    metric: Callable[[list[float], list[bool]], float | None],
    *,
    resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Percentile bootstrap interval for any metric of the scored pairs.

    Resamples that leave the metric undefined are skipped. ``None`` means no
    resample produced a value.
    """
    if resamples < 1:
        raise ValueError("resamples must be at least 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    pairs = _pairs(scores, labels)
    if not pairs:
        return None
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(resamples):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        value = metric([s for s, _ in sample], [y for _, y in sample])
        if value is not None:
            values.append(float(value))
    if not values:
        return None
    alpha = (1 - confidence) / 2
    return (percentile(values, 100 * alpha), percentile(values, 100 * (1 - alpha)))


def percentile(values: Sequence[float], q: float) -> float:
    """Linearly interpolated percentile, matching the common default method."""
    if not values:
        raise ValueError("percentile needs at least one value")
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def latency_summary(latencies_ms: Sequence[float]) -> dict[str, Any]:
    """Count, mean, p50, p95 and maximum latency in milliseconds."""
    if not latencies_ms:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None}
    return {
        "count": len(latencies_ms),
        "mean_ms": sum(latencies_ms) / len(latencies_ms),
        "p50_ms": percentile(latencies_ms, 50),
        "p95_ms": percentile(latencies_ms, 95),
        "max_ms": max(latencies_ms),
    }


def estimated_cost_usd(
    input_tokens: int,
    output_tokens: int,
    *,
    input_usd_per_million: float,
    output_usd_per_million: float = 0.0,
) -> float:
    """Cost calculated from token counts and a configured price, not a provider bill."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts must not be negative")
    if input_usd_per_million < 0 or output_usd_per_million < 0:
        raise ValueError("prices must not be negative")
    return (input_tokens * input_usd_per_million + output_tokens * output_usd_per_million) / 1e6


def cost_per_1000_events(total_cost_usd: float, events: int) -> float | None:
    """Scale a total cost to 1,000 events; ``None`` when no events were measured."""
    if total_cost_usd < 0 or events < 0:
        raise ValueError("cost and events must not be negative")
    return total_cost_usd / events * 1000 if events else None


def _logit(probability: float) -> float:
    clipped = min(max(probability, _EPSILON), 1 - _EPSILON)
    return math.log(clipped / (1 - clipped))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1 + exponent)


def apply_temperature(score: float, temperature: float) -> float:
    """Rescale a probability's log-odds by ``1 / temperature``."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a positive finite number")
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("score must be a probability between 0 and 1")
    return _sigmoid(_logit(score) / temperature)


def fit_temperature(
    scores: Sequence[float | None],
    labels: Sequence[bool | int],
    *,
    minimum: float = 0.05,
    maximum: float = 20.0,
) -> float:
    """Temperature minimising log loss on a calibration set.

    The log loss is convex in ``1 / temperature``, so its derivative is found
    by bisection. The result is clamped to ``[minimum, maximum]``; fit on a
    calibration split, never on the locked test set.
    """
    pairs = _pairs(scores, labels)
    positives = sum(label for _, label in pairs)
    if not positives or positives == len(pairs):
        raise ValueError("temperature fitting needs positive and negative examples")
    if not 0 < minimum < maximum:
        raise ValueError("temperature bounds must satisfy 0 < minimum < maximum")
    logits = [(_logit(score), float(label)) for score, label in pairs]

    def gradient(inverse: float) -> float:
        return sum((_sigmoid(inverse * z) - y) * z for z, y in logits)

    low, high = 1 / maximum, 1 / minimum
    if gradient(low) >= 0:
        return maximum
    if gradient(high) <= 0:
        return minimum
    for _ in range(100):
        middle = (low + high) / 2
        if gradient(middle) > 0:
            high = middle
        else:
            low = middle
    return 1 / ((low + high) / 2)


def probability_report(
    scores: Sequence[float | None],
    labels: Sequence[bool | int],
    *,
    false_alarm_targets: Sequence[float] = (0.01, 0.05),
    bins: int = 10,
    confidence: float = 0.95,
    resamples: int = 1000,
    latencies_ms: Sequence[float] | None = None,
    total_cost_usd: float | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Combine coverage, ranking, calibration, cost and latency into one record."""
    pairs = _pairs(scores, labels)
    scored = [s for s, _ in pairs]
    truth = [y for _, y in pairs]
    has_both = 0 < sum(truth) < len(truth)
    ece = expected_calibration_error(scored, truth, bins=bins)

    def ece_value(s: list[float], y: list[bool]) -> float | None:
        value = expected_calibration_error(s, y, bins=bins)["ece"]
        return float(value) if value is not None else None

    brier_ci = bootstrap_interval(
        scored, truth, lambda s, y: brier_score(s, y),
        resamples=resamples, confidence=confidence, seed=seed,
    ) if pairs else None
    ece_ci = bootstrap_interval(
        scored, truth, ece_value, resamples=resamples, confidence=confidence, seed=seed,
    ) if pairs else None
    return {
        "coverage": score_coverage(scores),
        "positives": sum(truth),
        "negatives": len(truth) - sum(truth),
        "roc_auc": roc_auc(scored, truth),
        "brier": brier_score(scored, truth),
        "brier_interval": list(brier_ci) if brier_ci else None,
        "ece": ece["ece"],
        "ece_interval": list(ece_ci) if ece_ci else None,
        "max_calibration_error": ece["max_calibration_error"],
        "calibration_bins": ece["bins"],
        "recall_at_false_alarm_rate": [
            recall_at_false_alarm_rate(scored, truth, target, confidence=confidence)
            for target in false_alarm_targets
        ] if has_both else [],
        "latency": latency_summary(latencies_ms or []),
        "total_cost_usd": total_cost_usd,
        "cost_per_1000_events": (
            cost_per_1000_events(total_cost_usd, len(scores))
            if total_cost_usd is not None else None
        ),
    }
