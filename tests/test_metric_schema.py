"""Contract and compatibility tests for labeled observations."""

import json

import pytest
import yaml
from pydantic import ValidationError

from litmusai import Agent, GroundTruth, TestCase, TestSuite, multi_evaluate
from litmusai.metrics import MetricConfig, Observation


@pytest.mark.parametrize("truth", [
    {}, {"answer_type": "text"}, {"answer_type": "numeric", "answer": None},
    {"answer_type": "json"}, {"answer_type": "boolean"}, {"answer_type": "list"},
])
@pytest.mark.parametrize("explicit_assertions", [False, True])
def test_suite_rejects_missing_ground_truth_answers(tmp_path, truth, explicit_assertions):
    case = {"id": "missing", "task": "anything", "ground_truth": truth}
    if explicit_assertions:
        case["assertions"] = [{"type": "contains", "value": "anything"}]
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump({"name": "invalid", "cases": [case]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Case 'missing'.*requires an answer"):
        TestSuite.from_yaml(path)


@pytest.mark.parametrize("truth", [None, False, [], "answer"])
def test_explicit_ground_truth_requires_a_mapping(tmp_path, truth):
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump({"name": "invalid", "cases": [
        {"id": "case", "task": "anything", "ground_truth": truth},
    ]}), encoding="utf-8")
    with pytest.raises(ValueError, match="ground_truth.*must be a mapping"):
        TestSuite.from_yaml(path)


@pytest.mark.parametrize("answer_type,answer", [
    ("text", ""), ("numeric", 0), ("boolean", False), ("json", {}), ("list", []),
    ("subjective", None),
])
def test_suite_retains_valid_empty_answers_and_subjective_truth(tmp_path, answer_type, answer):
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump({"name": "valid", "cases": [
        {"id": "case", "ground_truth": {"answer_type": answer_type, "answer": answer},
         "assertions": [{"type": "contains", "value": "explicit check"}]},
    ]}), encoding="utf-8")
    case = TestSuite.from_yaml(path).cases[0]
    assert case.expected_value == answer
    assert case.ground_truth.answer_type == answer_type


@pytest.mark.parametrize("field", ["expected", "predicted", "evidence"])
@pytest.mark.parametrize("value", [
    {1: "x", "1": "y"}, {"nested": [{False: "x"}]}, {"nested": {None: "x"}},
])
def test_observation_rejects_non_string_keys_recursively(field, value):
    with pytest.raises(ValidationError, match="keys must be strings|string_type"):
        Observation.model_validate({
            "evaluation_id": "eval", "case_id": "case", "task_type": "extraction",
            "expected": [], field: value,
        })


@pytest.mark.parametrize("value", [(1, 2), {"nested": (1, 2)}, {1, 2}, float("inf"), float("nan")])
def test_observation_rejects_values_that_cannot_round_trip(value):
    with pytest.raises(ValidationError):
        Observation(evaluation_id="eval", case_id="case", task_type="extraction", expected=value)


def test_observation_rejects_cycles():
    value = []
    value.append(value)
    with pytest.raises(ValidationError, match="acyclic"):
        Observation(evaluation_id="eval", case_id="case", task_type="extraction", expected=value)


@pytest.mark.parametrize("field", ["evaluation_id", "case_id"])
@pytest.mark.parametrize("value", [" ", "\t\r\n", "\u2003"])
def test_observation_rejects_whitespace_only_ids(field, value):
    data = {"evaluation_id": "eval", "case_id": "case", "task_type": "extraction",
            "expected": [], field: value}
    with pytest.raises(ValidationError, match="non-whitespace"):
        Observation.model_validate(data)
    with pytest.raises(ValidationError, match="non-whitespace"):
        Observation.model_validate_json(json.dumps(data))


def test_observation_preserves_valid_ids_verbatim():
    observation = Observation(evaluation_id=" eval ", case_id=" café ",
                              task_type="extraction", expected=[])
    restored = Observation.model_validate_json(observation.model_dump_json())
    assert restored.evaluation_id == " eval "
    assert restored.case_id == " café "


def test_nested_json_values_round_trip_without_coercion():
    value = {"1": [0, False, None, "", 1.5, {"café": "Zoë"}], "empty": {}}
    original = Observation(evaluation_id="eval", case_id="case", task_type="extraction",
                           expected=value, predicted=value, evidence={"matched": [value]})
    assert Observation.model_validate_json(original.model_dump_json()) == original


def test_apply_truth_retains_empty_labeled_answers_without_generating_assertions():
    from litmusai import apply_ground_truth

    suite = TestSuite("extraction", [TestCase(id="empty")],
                      metrics=MetricConfig(task_type="extraction"))
    truth = GroundTruth(answer=[], answer_type="list")
    assert apply_ground_truth(suite, {"empty": truth}) == 1
    case = suite.cases[0]
    assert case.ground_truth is truth
    assert case.expected_value == []
    assert case.metadata["ground_truth"] == truth.to_dict()
    assert case.assertions == []


def test_ground_truth_survives_explicit_assertions(tmp_path):
    path = tmp_path / "suite.yaml"
    path.write_text('''
name: routing
cases:
  - id: invoice
    task: Où est la facture?
    ground_truth:
      answer: billing
    assertions:
      - type: contains
        value: bill
''', encoding="utf-8")
    case = TestSuite.from_yaml(path).cases[0]
    assert case.expected_value == "billing"
    assert case.metadata["ground_truth"]["answer"] == "billing"
    assert len(case.assertions) == 1


def test_labeled_suite_yaml_round_trip(tmp_path):
    suite = TestSuite("routing", [TestCase(
        id="facture", task="Où?", ground_truth=GroundTruth(answer="billing"),
    )], metrics=MetricConfig(task_type="classification", labels=["billing", "support"]))
    path = tmp_path / "suite.yaml"
    suite.to_yaml(path)
    loaded = TestSuite.from_yaml(path)
    assert loaded.metrics == suite.metrics
    assert loaded.cases[0].expected_value == "billing"
    assert loaded.cases[0].task == "Où?"
    assert loaded.cases[0].assertions == []


@pytest.mark.parametrize("options", [
    {"task_type": "classification"},
    {"task_type": "classification", "labels": ["a", "a"]},
    {"task_type": "classification", "labels": [1]},
    {"task_type": "extraction", "labels": ["a"]},
    {"task_type": "extraction", "prediction_field": "output.label"},
    {"task_type": "extraction", "prediction_field": "/bad~2"},
    {"task_type": "extraction", "unknown_option": True},
])
def test_invalid_rules_are_rejected(options):
    with pytest.raises(ValidationError):
        MetricConfig.model_validate(options)


def test_unsupported_suite_version(tmp_path):
    path = tmp_path / "suite.yaml"
    path.write_text('schema_version: "9.0"\ncases: []', encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported suite schema_version"):
        TestSuite.from_yaml(path)


def test_observation_round_trip_preserves_evidence_and_error():
    observation = Observation(
        evaluation_id="eval", repetition=2, case_id="café", task_type="extraction",
        expected=[{"name": "Zoë"}], predicted=None, status="execution_error",
        error="timeout", evidence={"missing": [{"name": "Zoë"}]},
    )
    assert Observation.model_validate_json(observation.model_dump_json()) == observation
    with pytest.raises(ValidationError):
        Observation.model_validate({**observation.model_dump(), "schema_version": "9"})


async def test_all_repetitions_are_serialized():
    suite = TestSuite("legacy", [TestCase(id="one", task="hello")])
    multi = await multi_evaluate(Agent.from_function(lambda _: "hello"), suite,
                                 runs=2, verbose=False)
    data = json.loads(json.dumps(multi.to_dict()))
    assert len(data["run_results"]) == 2
    assert [r["repetition"] for r in data["run_results"]] == [1, 2]
    assert all(r["evaluation_id"] == data["evaluation_id"] for r in data["run_results"])
    assert all(r["results"][0]["case_id"] == "one" for r in data["run_results"])
