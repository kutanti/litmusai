"""Hand-calculated classification counts, parsing, and averaging policies."""

import json

import pytest

from litmusai.metrics import MetricConfig, Observation, classification_metrics
from litmusai.metrics.classification import classification_observation
from litmusai.metrics.common import prf


def observations(expected, predicted, config):
    return [classification_observation(e, p, config=config, case_id=str(i), evaluation_id="test")
            for i, (e, p) in enumerate(zip(expected, predicted, strict=True))]


def test_binary_counts_and_class_imbalance():
    config = MetricConfig(task_type="classification", labels=["yes", "no"])
    records = observations(["yes"] * 5 + ["no"] * 3,
                           ["yes"] * 3 + ["no"] * 2 + ["yes", "no", "no"], config)
    metrics = classification_metrics(records, config)
    yes = metrics["per_class"]["yes"]
    assert (yes["tp"], yes["fp"], yes["fn"], yes["support"]) == (3, 1, 2, 5)
    assert yes["precision"]["value"] == .75
    assert yes["recall"]["value"] == .6
    assert yes["f1"]["value"] == pytest.approx(2 / 3)
    assert metrics["accuracy"]["value"] == 5 / 8
    assert metrics["micro"]["f1"]["value"] == 5 / 8
    assert metrics["macro"]["f1"]["value"] == pytest.approx((2 / 3 + 4 / 7) / 2)
    assert metrics["weighted"]["f1"]["value"] == pytest.approx((5 * 2 / 3 + 3 * 4 / 7) / 8)
    assert metrics["confusion_matrix"]["counts"] == [[3, 2, 0], [1, 2, 0]]


def test_multiclass_and_absent_class_zero_denominators():
    config = MetricConfig(task_type="classification", labels=["a", "b", "c", "absent"])
    metrics = classification_metrics(observations(["a", "a", "b", "c"],
                                                   ["a", "b", "c", "c"], config), config)
    assert metrics["accuracy"]["value"] == .5
    assert metrics["per_class"]["c"]["precision"]["value"] == .5
    assert metrics["per_class"]["absent"]["precision"] == {
        "value": 0, "defined": False, "numerator": 0, "denominator": 0,
    }
    assert metrics["macro"]["f1"]["defined"] is False
    assert metrics["macro"]["f1"]["undefined_labels"] == ["absent"]
    assert metrics["weighted"]["f1"]["defined"] is True
    assert classification_metrics([], config) is None


def test_errors_unknown_labels_and_abstentions_remain_in_denominators():
    config = MetricConfig(task_type="classification", labels=["a", "b"])
    records = observations(["a"] * 3, ["a", "unknown", ""], config)
    records.append(classification_observation(
        "b", "a", config=config, case_id="error", evaluation_id="test", success=False,
        error="timeout",
    ))
    metrics = classification_metrics(records, config)
    assert metrics["accuracy"]["value"] == .25
    assert metrics["prediction_coverage"]["value"] == .25
    assert metrics["invalid_predictions"] == 2
    assert metrics["execution_errors"] == 1
    assert metrics["micro"]["precision"]["value"] == 1
    assert metrics["micro"]["recall"]["value"] == .25
    assert metrics["confusion_matrix"]["counts"] == [[1, 0, 2], [0, 0, 1]]
    assert records[-1].error == "timeout"
    assert records[1].predicted == "unknown"
    assert records[1].evidence["fn"] == 1


@pytest.mark.parametrize("output", ["not json", "{}", '{"label":null}', '{"label":42}',
                                   '{"label":"unknown"}', '{"label":NaN}',
                                   '{"label":1e999}'])
def test_malformed_structured_predictions(output):
    config = MetricConfig(task_type="classification", labels=["a"], prediction_field="/label")
    record = classification_observation("a", output, config=config, case_id="c", evaluation_id="e")
    assert record.status == "invalid_prediction"
    assert record.error
    assert record.evidence["fn"] == 1
    assert classification_metrics([record], config)["accuracy"]["value"] == 0


def test_json_pointer_and_full_response_are_used():
    config = MetricConfig(task_type="classification", labels=["a"], prediction_field="/a~1b/0/~0")
    output = json.dumps({"padding": "x" * 3000, "a/b": [{"~": "a"}]})
    record = classification_observation("a", output, config=config, case_id="c", evaluation_id="e")
    assert record.evidence["correct"] is True
    assert Observation.model_validate_json(record.model_dump_json()) == record


@pytest.mark.parametrize("truth", [None, "unknown", ["a"], {"label": "a"}, 1])
def test_invalid_ground_truth_is_rejected(truth):
    with pytest.raises(ValueError, match="expected label"):
        classification_observation(truth, "a", config=MetricConfig(
            task_type="classification", labels=["a"]), case_id="c", evaluation_id="e")


def test_aggregation_uses_counts_across_repetitions_and_is_order_independent():
    config = MetricConfig(task_type="classification", labels=["a", "b"])
    first = observations(["a", "b"], ["a", "b"], config)
    second = observations(["a", "b"], ["b", ""], config)
    for record in second:
        record.repetition = 2
    combined = classification_metrics(first + second, config)
    assert combined == classification_metrics(list(reversed(second + first)), config)
    assert combined["micro"]["f1"]["value"] == pytest.approx(4 / 7)
    assert combined["micro"]["f1"]["value"] != .5  # Mean of the two run F1s.
    with pytest.raises(ValueError, match="duplicate observation"):
        classification_metrics(first + first, config)


def test_count_formula_and_all_zero():
    assert prf(3, 1, 2)["f1"]["value"] == pytest.approx(2 / 3)
    assert prf(0, 0, 0)["f1"]["defined"] is False
