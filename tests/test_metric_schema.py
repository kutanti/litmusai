"""Contract and compatibility tests for labeled observations."""

import json

import pytest
from pydantic import ValidationError

from litmusai import Agent, GroundTruth, TestCase, TestSuite, multi_evaluate
from litmusai.metrics import MetricConfig, Observation


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
