import math

import pytest

from litmusai.metrics.probability import (
    apply_temperature,
    bootstrap_interval,
    brier_score,
    confusion_at,
    cost_per_1000_events,
    estimated_cost_usd,
    expected_calibration_error,
    fit_temperature,
    latency_summary,
    percentile,
    probability_report,
    recall_at_false_alarm_rate,
    roc_auc,
    score_coverage,
    threshold_sweep,
    wilson_interval,
)

POSITIVE = [0.9, 0.8, 0.3]
NEGATIVE = [0.7, 0.2, 0.1, 0.05]
SCORES = POSITIVE + NEGATIVE
LABELS = [True] * len(POSITIVE) + [False] * len(NEGATIVE)


def test_brier_and_calibration_error_match_hand_calculations():
    assert brier_score([0.9, 0.1], [1, 0]) == pytest.approx(0.01)
    result = expected_calibration_error([0.9, 0.9, 0.1, 0.1], [1, 0, 0, 0], bins=10)
    assert result["ece"] == pytest.approx(0.25)
    assert result["max_calibration_error"] == pytest.approx(0.4)
    assert [bucket["count"] for bucket in result["bins"]] == [2, 2]


def test_quantile_bins_split_by_count():
    result = expected_calibration_error(
        [0.01, 0.02, 0.03, 0.99], [0, 0, 1, 1], bins=2, strategy="quantile",
    )
    assert [bucket["count"] for bucket in result["bins"]] == [2, 2]


def test_wilson_interval_matches_reference_values():
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.4902, abs=1e-4)
    assert high == pytest.approx(0.9433, abs=1e-4)
    assert wilson_interval(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        wilson_interval(3, 2)


def test_roc_auc_counts_ties_as_half():
    assert roc_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)
    assert roc_auc([0.5, 0.5], [0, 1]) == pytest.approx(0.5)
    assert roc_auc([0.5], [1]) is None


def test_threshold_sweep_matches_direct_confusion_counts():
    points = threshold_sweep(SCORES, LABELS)
    assert [point["threshold"] for point in points] == sorted(set(SCORES))
    for point in points:
        direct = confusion_at(SCORES, LABELS, point["threshold"])
        assert point == direct
    explicit = threshold_sweep(SCORES, LABELS, thresholds=[0.5, 0.0])
    assert [point["threshold"] for point in explicit] == [0.0, 0.5]
    assert explicit[1]["tp"] == 2 and explicit[1]["fp"] == 1


def test_recall_at_false_alarm_rate_picks_lowest_qualifying_threshold():
    strict = recall_at_false_alarm_rate(SCORES, LABELS, 0.0)
    assert strict["threshold"] == 0.8
    assert strict["recall"] == pytest.approx(2 / 3)
    assert strict["false_alarm_rate"] == 0.0
    low, high = strict["recall_interval"]
    assert low < strict["recall"] < high

    relaxed = recall_at_false_alarm_rate(SCORES, LABELS, 0.25)
    assert relaxed["threshold"] == 0.3
    assert relaxed["recall"] == 1.0


def test_recall_at_false_alarm_rate_reports_flag_nothing_point():
    result = recall_at_false_alarm_rate([0.9, 1.0], [1, 0], 0.0)
    assert result["threshold"] is None
    assert result["recall"] == 0.0
    assert result["false_alarm_rate"] == 0.0
    with pytest.raises(ValueError):
        recall_at_false_alarm_rate([0.9], [1], 0.1)


def test_failed_predictions_are_excluded_and_counted():
    scores = [0.9, None, 0.1]
    labels = [1, 1, 0]
    assert score_coverage(scores) == {"total": 3, "scored": 2, "failed": 1, "coverage": 2 / 3}
    assert brier_score(scores, labels) == pytest.approx(0.01)


@pytest.mark.parametrize("score", [1.5, -0.1, math.nan, math.inf, True, "0.5"])
def test_invalid_scores_are_rejected(score):
    with pytest.raises(ValueError):
        brier_score([score], [1])


def test_invalid_labels_and_lengths_are_rejected():
    with pytest.raises(ValueError):
        brier_score([0.5], [2])
    with pytest.raises(ValueError):
        brier_score([0.5, 0.5], [1])


def test_temperature_fit_recovers_overconfidence():
    scores = [0.95] * 100 + [0.05] * 100
    labels = [True] * 70 + [False] * 30 + [True] * 30 + [False] * 70
    temperature = fit_temperature(scores, labels)
    expected = math.log(0.95 / 0.05) / math.log(0.7 / 0.3)
    assert temperature == pytest.approx(expected, rel=1e-3)
    assert apply_temperature(0.95, temperature) == pytest.approx(0.7, abs=1e-3)
    with pytest.raises(ValueError):
        fit_temperature([0.2, 0.3], [1, 1])
    with pytest.raises(ValueError):
        apply_temperature(0.5, 0.0)


def test_temperature_fit_is_clamped():
    assert fit_temperature([0.99, 0.01], [0, 1], maximum=5.0) == 5.0


def test_percentiles_latency_and_cost():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, 2, 3, 4], 95) == pytest.approx(3.85)
    summary = latency_summary([100.0, 200.0, 300.0])
    assert summary["p50_ms"] == 200.0
    assert summary["max_ms"] == 300.0
    assert latency_summary([])["p95_ms"] is None
    assert estimated_cost_usd(1_000_000, 10, input_usd_per_million=0.042) == pytest.approx(0.042)
    assert cost_per_1000_events(0.5, 2000) == pytest.approx(0.25)
    assert cost_per_1000_events(0.5, 0) is None
    with pytest.raises(ValueError):
        estimated_cost_usd(-1, 0, input_usd_per_million=1.0)


def test_bootstrap_interval_is_deterministic_and_brackets_estimate():
    first = bootstrap_interval(SCORES, LABELS, lambda s, y: brier_score(s, y), resamples=200)
    second = bootstrap_interval(SCORES, LABELS, lambda s, y: brier_score(s, y), resamples=200)
    assert first == second
    assert first is not None
    assert first[0] <= brier_score(SCORES, LABELS) <= first[1]
    assert bootstrap_interval([], [], lambda s, y: 0.0) is None


def test_probability_report_combines_metrics():
    report = probability_report(
        SCORES + [None], LABELS + [True],
        latencies_ms=[10.0, 20.0], total_cost_usd=0.07, resamples=50,
    )
    assert report["coverage"]["failed"] == 1
    assert report["positives"] == 3
    assert report["negatives"] == 4
    assert report["roc_auc"] == pytest.approx(11 / 12)
    assert len(report["recall_at_false_alarm_rate"]) == 2
    assert report["latency"]["p50_ms"] == 15.0
    assert report["cost_per_1000_events"] == pytest.approx(0.07 / 8 * 1000)
    assert report["brier_interval"] is not None
