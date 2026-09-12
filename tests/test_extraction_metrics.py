"""Extraction matching fixtures with duplicates, fields and normalization."""

import json

import pytest

from litmusai.metrics import MetricConfig, Observation, extraction_metrics, extraction_observation


def observe(expected, predicted, **options):
    return extraction_observation(
        expected, json.dumps(predicted), config=MetricConfig(task_type="extraction", **options),
        case_id="case", evaluation_id="test",
    )


def test_entities_missing_extra_and_duplicate_occurrences():
    config = MetricConfig(task_type="extraction")
    record = observe(["a", "a", "b", "c", "d"], ["b", "a", "a", "a"])
    metrics = extraction_metrics([record], config)
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (3, 1, 2)
    assert metrics["precision"]["value"] == .75
    assert metrics["recall"]["value"] == .6
    assert metrics["f1"]["value"] == pytest.approx(2 / 3)
    assert record.evidence["missing"] == ["c", "d"]
    assert record.evidence["extra"] == ["a"]
    assert len(record.evidence["matched"]) == 3


def test_structured_entities_use_all_properties():
    record = observe(
        [{"type": "person", "text": "Alice"}, {"type": "city", "text": "Paris"}],
        [{"text": "Alice", "type": "person"}, {"type": "person", "text": "Paris"}],
    )
    assert (record.evidence["tp"], record.evidence["fp"], record.evidence["fn"]) == (1, 1, 1)


def test_fields_expand_lists_into_occurrences_and_keep_field_names():
    record = observe({"name": "Alice", "emails": ["a", "b"], "missing": "x"},
                     {"name": "Alice", "emails": ["a", "a"], "extra": "x"})
    assert (record.evidence["tp"], record.evidence["fp"], record.evidence["fn"]) == (2, 2, 2)
    assert {"field": "emails", "value": "a"} in record.evidence["extra"]
    assert {"field": "missing", "value": "x"} in record.evidence["missing"]


def test_normalization_is_opt_in_and_preserves_original_evidence():
    expected = [{"text": "Straße  café", "type": "place"}]
    predicted = [{"type": "place", "text": "  STRASSE\tCAFÉ "}]
    assert observe(expected, predicted).evidence["tp"] == 0
    record = observe(expected, predicted, normalize_whitespace=True, casefold=True)
    assert record.evidence["tp"] == 1
    assert record.evidence["matched"] == [{"expected": expected[0], "predicted": predicted[0]}]
    # Field names are never normalized.
    assert observe({"Name": "A"}, {"name": "a"}, casefold=True).evidence["tp"] == 0
    assert Observation.model_validate_json(record.model_dump_json()) == record


def test_normalized_duplicates_still_match_one_to_one():
    record = observe([" Alice "], ["ALICE", "alice"], normalize_whitespace=True, casefold=True)
    assert (record.evidence["tp"], record.evidence["fp"]) == (1, 1)


def test_json_value_types_do_not_coerce():
    assert observe([1, True, None], ["1", 1.0, "null"]).evidence["tp"] == 0
    assert observe({"a": None}, {}).evidence["fn"] == 1


@pytest.mark.parametrize("output", ["", "not json", "null", '"text"', "{}", "[NaN]", "[1e999]"])
def test_invalid_predictions_keep_all_expected_false_negatives(output):
    config = MetricConfig(task_type="extraction")
    record = extraction_observation(["a", "b"], output, config=config,
                                    case_id="case", evaluation_id="test")
    assert record.status == "invalid_prediction"
    assert record.evidence["missing"] == ["a", "b"]
    metrics = extraction_metrics([record], config)
    assert metrics["fn"] == 2
    assert metrics["invalid_predictions"] == 1
    assert metrics["prediction_coverage"]["value"] == 0


def test_execution_error_does_not_use_partial_response():
    config = MetricConfig(task_type="extraction")
    record = extraction_observation(["a"], '["a"]', config=config, case_id="c",
                                    evaluation_id="e", success=False, error="timeout")
    assert record.status == "execution_error"
    assert record.evidence["fn"] == 1
    assert extraction_metrics([record], config)["execution_errors"] == 1


@pytest.mark.parametrize("truth", [None, "text", 1, {1: "bad key"}, [float("nan")], [set()]])
def test_malformed_ground_truth_is_rejected(truth):
    with pytest.raises(ValueError):
        observe(truth, [])


def test_missing_prediction_pointer_and_empty_valid_predictions_differ():
    config = MetricConfig(task_type="extraction", prediction_field="/entities")
    missing = extraction_observation(["a"], '{}', config=config,
                                     case_id="missing", evaluation_id="e")
    empty = extraction_observation(["a"], '{"entities":[]}', config=config,
                                   case_id="empty", evaluation_id="e")
    assert missing.status == "invalid_prediction"
    assert empty.status == "ok"
    metrics = extraction_metrics([missing, empty], config)
    assert metrics["fn"] == 2
    assert metrics["prediction_coverage"]["value"] == .5


def test_empty_dataset_and_no_entities():
    config = MetricConfig(task_type="extraction")
    assert extraction_metrics([], config) is None
    metrics = extraction_metrics([observe([], [])], config)
    assert metrics["precision"]["defined"] is False
    assert metrics["recall"]["defined"] is False
    assert metrics["f1"]["defined"] is False
    assert metrics["exact_match_accuracy"]["value"] == 1


def test_pooled_repetition_counts_are_order_independent():
    config = MetricConfig(task_type="extraction")
    first = observe(["a"], ["a"])
    second = observe(["a", "b", "c"], [])
    second.repetition = 2
    metrics = extraction_metrics([first, second], config)
    assert metrics == extraction_metrics([second, first], config)
    assert metrics["f1"]["value"] == .4  # Not the mean of 1.0 and 0.0.
