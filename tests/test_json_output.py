"""Regression coverage for the JSON formats observed through LiteLLM."""

import json
import sys

import pytest

from litmusai import Agent, GroundTruth, MetricConfig, evaluate
from litmusai import TestCase as Case
from litmusai import TestSuite as Suite
from litmusai._json import parse_json_output
from litmusai.assertions import JsonPath, JsonSchema, JsonValid
from litmusai.metrics import classification_observation, extraction_observation


@pytest.mark.parametrize("fence", ["{}", "```json\n{}\n```", "```\n{}\n```",
                                    " \n```JSON\r\n{}\r\n```\n "])
def test_assertions_and_extraction_accept_the_same_complete_answer(fence):
    expected = {"emails": ["alice@example.com", "bob@example.com"]}
    output = fence.format(json.dumps(expected))
    assert JsonValid().check(output).passed
    assert JsonSchema({"type": "object", "required": ["emails"]}).check(output).passed
    assert JsonPath("emails", expected["emails"]).check(output).passed
    observation = extraction_observation(expected, output, config=MetricConfig(
        task_type="extraction"), case_id="email", evaluation_id="live")
    assert observation.status == "ok"
    assert observation.predicted == expected
    assert observation.evidence["tp"] == 2


@pytest.mark.parametrize("output", [
    'Here is the answer: {"emails": []}',
    '```json\n{"emails": []}\n```\nThat is the answer.',
    '```json\n{"emails": []}\n```\n```json\n{"emails": ["a"]}\n```',
    '```json\n{"emails": []}\n```\nActually, reconsidering: {"emails": ["a"]}',
    '{"emails": []} {"emails": ["a"]}',
    '```python\n{"emails": []}\n```',
    '```json\n{"emails": []}',
    '{"emails": [NaN]}', '{"emails": [Infinity]}', '{"emails": [-Infinity]}',
    '{"emails": [1e999]}', '```json\n{"emails": [1e999]}\n```',
])
def test_ambiguous_and_nonfinite_answers_fail_assertions_and_metrics(output):
    for assertion in (JsonValid(), JsonSchema({"type": "object"}), JsonPath("emails", [])):
        result = assertion.check(output)
        assert not result.passed
        assert "not valid JSON" in result.reason
    observation = extraction_observation({"emails": ["a"]}, output, config=MetricConfig(
        task_type="extraction"), case_id="email", evaluation_id="live")
    assert observation.status == "invalid_prediction"
    assert observation.predicted == output
    assert observation.evidence["fn"] == 1


@pytest.mark.parametrize("output", ['{"label": "yes"}', '```json\n{"label": "yes"}\n```'])
def test_classification_json_pointer_uses_the_same_parser(output):
    observation = classification_observation("yes", output, config=MetricConfig(
        task_type="classification", labels=["yes", "no"], prediction_field="/label"),
        case_id="label", evaluation_id="live")
    assert observation.status == "ok"
    assert observation.predicted == "yes"


@pytest.mark.parametrize("source,expected", [("null", None), ("false", False), ("0", 0),
                                             ('"text"', "text"), ("[]", [])])
def test_complete_json_values_are_valid_including_null(source, expected):
    assert parse_json_output(source) == expected
    assert JsonValid().check(source).passed
    assert JsonValid().check(f"```json\n{source}\n```").passed


@pytest.mark.parametrize("fallback", [False, True])
def test_null_schema_and_path_are_distinct_from_parse_failure(monkeypatch, fallback):
    if fallback:
        monkeypatch.setitem(sys.modules, "jsonschema", None)
    assert JsonSchema({"type": "null"}).check("null").passed
    assert not JsonSchema({"type": "null"}).check("{}").passed
    assert not JsonSchema({"type": "object"}).check("null").passed
    assert JsonPath("$", None).check("null").passed
    assert JsonPath("a", None).check('{"a":null}').passed
    assert not JsonPath("a", None).check("{}").passed


def test_parser_recursion_error_is_a_failed_assertion(monkeypatch):
    # CPython versions have different JSON recursion limits. Exercise the
    # failure without relying on a particular interpreter's supported depth.
    def exhausted_parser(*args, **kwargs):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr("litmusai._json.json.loads", exhausted_parser)
    assert not JsonValid().check("[]").passed


async def test_evaluation_keeps_original_fenced_response_and_dataset_identity():
    output = '```json\n{"emails": ["a@example.com"]}\n```'
    suite = Suite("json", [Case(id="email", task="Find emails", ground_truth=GroundTruth(
        answer={"emails": ["a@example.com"]}, answer_type="json"), assertions=[JsonValid()])],
        metrics=MetricConfig(task_type="extraction"))
    fingerprint = suite.dataset_info.fingerprint
    result = await evaluate(Agent.from_function(lambda _: output), suite, verbose=False)
    assert result.pass_rate == 1
    assert result.metrics["f1"]["value"] == 1
    assert result.to_dict()["results"][0]["response"] == output
    assert result.dataset.fingerprint == fingerprint
