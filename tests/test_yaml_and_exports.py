"""Tests for YAML assertions and export formats."""

import json
import xml.etree.ElementTree as ET

import pytest

from litmusai.core.suite import TestSuite, _parse_yaml_assertions
from litmusai.exports import to_csv, to_junit_xml

# ─── YAML Assertion Parsing ─────────────────────────────────────


class TestYamlAssertions:
    def test_contains(self):
        specs = [{"type": "contains", "value": "hello"}]
        assertions = _parse_yaml_assertions(specs)
        assert len(assertions) == 1
        result = assertions[0].check("say hello world")
        assert result.passed

    def test_not_contains(self):
        specs = [{"type": "not_contains", "patterns": ["hack", "exploit"]}]
        assertions = _parse_yaml_assertions(specs)
        result = assertions[0].check("I can help with that")
        assert result.passed

    def test_not_contains_single_value(self):
        specs = [{"type": "not_contains", "value": "forbidden"}]
        assertions = _parse_yaml_assertions(specs)
        result = assertions[0].check("this is fine")
        assert result.passed

    def test_numeric(self):
        specs = [{"type": "numeric", "value": 42}]
        assertions = _parse_yaml_assertions(specs)
        result = assertions[0].check("the answer is 42")
        assert result.passed

    def test_numeric_with_tolerance(self):
        specs = [{"type": "numeric", "value": 42, "tolerance": 1.0}]
        assertions = _parse_yaml_assertions(specs)
        result = assertions[0].check("about 42.5")
        assert result.passed

    def test_exact(self):
        specs = [{"type": "exact", "value": "hello"}]
        assertions = _parse_yaml_assertions(specs)
        result = assertions[0].check("hello")
        assert result.passed

    def test_regex(self):
        specs = [{"type": "regex", "pattern": r"\d{3}-\d{4}"}]
        assertions = _parse_yaml_assertions(specs)
        result = assertions[0].check("call 555-1234")
        assert result.passed

    def test_json_valid(self):
        specs = [{"type": "json_valid"}]
        assertions = _parse_yaml_assertions(specs)
        result = assertions[0].check('{"name": "Alice"}')
        assert result.passed

    def test_json_schema(self):
        specs = [{"type": "json_schema", "schema": {
            "type": "object",
            "required": ["name"],
        }}]
        assertions = _parse_yaml_assertions(specs)
        result = assertions[0].check('{"name": "Alice"}')
        assert result.passed

    def test_json_path(self):
        specs = [{"type": "json_path", "path": "$.name", "expected": "Alice"}]
        assertions = _parse_yaml_assertions(specs)
        result = assertions[0].check('{"name": "Alice"}')
        assert result.passed

    def test_any_of_composite(self):
        specs = [{"type": "any_of", "assertions": [
            {"type": "contains", "value": "yes"},
            {"type": "contains", "value": "correct"},
        ]}]
        assertions = _parse_yaml_assertions(specs)
        result = assertions[0].check("yes that is correct")
        assert result.passed

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown assertion type"):
            _parse_yaml_assertions([{"type": "nonexistent"}])

    def test_missing_type_raises(self):
        with pytest.raises(ValueError, match="missing 'type'"):
            _parse_yaml_assertions([{"value": "hello"}])

    def test_case_insensitive_type(self):
        specs = [{"type": "CONTAINS", "value": "hello"}]
        assertions = _parse_yaml_assertions(specs)
        result = assertions[0].check("hello world")
        assert result.passed

    def test_multiple_assertions(self):
        specs = [
            {"type": "contains", "value": "hello"},
            {"type": "numeric", "value": 42},
        ]
        assertions = _parse_yaml_assertions(specs)
        assert len(assertions) == 2


class TestYamlSuiteWithAssertions:
    def test_load_suite_with_assertions(self, tmp_path):
        suite_yaml = tmp_path / "suite.yaml"
        suite_yaml.write_text(
            "name: test\n"
            "cases:\n"
            "  - id: q1\n"
            "    name: Math\n"
            '    task: "What is 6*7?"\n'
            "    assertions:\n"
            "      - type: numeric\n"
            "        value: 42\n"
            "      - type: contains\n"
            '        value: "42"\n'
        )
        suite = TestSuite.from_yaml(suite_yaml)
        assert len(suite.cases) == 1
        assert len(suite.cases[0].assertions) == 2

    def test_mixed_legacy_and_assertions(self, tmp_path):
        suite_yaml = tmp_path / "suite.yaml"
        suite_yaml.write_text(
            "name: mixed\n"
            "cases:\n"
            "  - id: q1\n"
            "    name: Legacy\n"
            '    task: "What is 2+2?"\n'
            "    expected_contains:\n"
            '      - "4"\n'
            "  - id: q2\n"
            "    name: Modern\n"
            '    task: "What is 6*7?"\n'
            "    assertions:\n"
            "      - type: numeric\n"
            "        value: 42\n"
        )
        suite = TestSuite.from_yaml(suite_yaml)
        assert len(suite.cases) == 2
        assert len(suite.cases[0].assertions) == 0
        assert len(suite.cases[0].expected_contains) == 1
        assert len(suite.cases[1].assertions) == 1

    def test_end_to_end_yaml_assertions(self, tmp_path):
        """Full flow: YAML → suite → evaluate."""
        import asyncio

        from litmusai import Agent, evaluate

        suite_yaml = tmp_path / "suite.yaml"
        suite_yaml.write_text(
            "name: e2e\n"
            "cases:\n"
            "  - id: q1\n"
            "    name: Math\n"
            '    task: "What is 6*7?"\n'
            "    assertions:\n"
            "      - type: numeric\n"
            "        value: 42\n"
        )
        suite = TestSuite.from_yaml(suite_yaml)

        async def mock_agent(task: str) -> str:
            return "The answer is 42."

        agent = Agent.from_function(mock_agent, name="mock")
        results = asyncio.run(evaluate(agent, suite, verbose=False))
        assert results.pass_rate == 1.0


# ─── JUnit XML Export ────────────────────────────────────────────


def _make_data(n_pass=2, n_fail=1):
    results = []
    for i in range(n_pass):
        results.append({
            "case_id": f"pass_{i}", "case_name": f"Pass {i}",
            "task": f"Task {i}", "response": f"Response {i}",
            "passed": True, "score": 1.0,
            "score_reason": "OK", "latency_ms": 100,
            "cost": 0.001, "input_tokens": 10,
            "output_tokens": 5, "model": "gpt-4o",
        })
    for i in range(n_fail):
        results.append({
            "case_id": f"fail_{i}", "case_name": f"Fail {i}",
            "task": f"Task fail {i}", "response": f"Wrong {i}",
            "passed": False, "score": 0.3,
            "score_reason": "Failed assertion", "latency_ms": 200,
            "cost": 0.002, "input_tokens": 15,
            "output_tokens": 10, "model": "gpt-4o",
        })
    total = n_pass + n_fail
    return {
        "agent_name": "test-agent",
        "suite_name": "test-suite",
        "timestamp": "2026-04-04T10:00:00",
        "summary": {
            "total": total, "passed": n_pass, "failed": n_fail,
            "pass_rate": n_pass / total if total else 0,
            "avg_score": 0.8, "avg_latency_ms": 150,
            "total_cost": 0.005,
            "total_input_tokens": 30, "total_output_tokens": 15,
        },
        "results": results,
    }


class TestJunitXml:
    def test_basic_export(self, tmp_path):
        data = _make_data()
        path = to_junit_xml(data, tmp_path / "results.xml")
        assert path.exists()
        tree = ET.parse(path)
        root = tree.getroot()
        assert root.tag == "testsuites"

    def test_correct_counts(self, tmp_path):
        data = _make_data(n_pass=3, n_fail=2)
        path = to_junit_xml(data, tmp_path / "results.xml")
        tree = ET.parse(path)
        ts = tree.find(".//testsuite")
        assert ts is not None
        assert ts.get("tests") == "5"
        assert ts.get("failures") == "2"

    def test_failure_elements(self, tmp_path):
        data = _make_data(n_pass=0, n_fail=1)
        path = to_junit_xml(data, tmp_path / "results.xml")
        tree = ET.parse(path)
        failure = tree.find(".//failure")
        assert failure is not None
        assert "Failed assertion" in (failure.get("message") or "")

    def test_properties(self, tmp_path):
        data = _make_data()
        path = to_junit_xml(data, tmp_path / "results.xml")
        tree = ET.parse(path)
        props = tree.findall(".//property")
        names = {p.get("name") for p in props}
        assert "agent" in names
        assert "suite" in names
        assert "pass_rate" in names

    def test_valid_xml(self, tmp_path):
        data = _make_data()
        path = to_junit_xml(data, tmp_path / "results.xml")
        content = path.read_text()
        assert content.startswith("<?xml")

    def test_creates_parent_dirs(self, tmp_path):
        data = _make_data()
        path = to_junit_xml(
            data, tmp_path / "deep" / "nested" / "results.xml",
        )
        assert path.exists()

    def test_system_out(self, tmp_path):
        data = _make_data(n_pass=1, n_fail=0)
        path = to_junit_xml(data, tmp_path / "results.xml")
        tree = ET.parse(path)
        stdout = tree.find(".//system-out")
        assert stdout is not None
        assert stdout.text is not None


# ─── CSV Export ──────────────────────────────────────────────────


class TestCsvExport:
    def test_basic_csv(self, tmp_path):
        data = _make_data()
        path = to_csv(data, tmp_path / "results.csv")
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 4  # header + 3 results

    def test_csv_header(self, tmp_path):
        data = _make_data()
        path = to_csv(data, tmp_path / "results.csv")
        header = path.read_text().split("\n")[0]
        assert "case_id" in header
        assert "passed" in header
        assert "score" in header

    def test_csv_creates_dirs(self, tmp_path):
        data = _make_data()
        path = to_csv(
            data, tmp_path / "sub" / "results.csv",
        )
        assert path.exists()


# ─── CLI Report Flags ────────────────────────────────────────────


class TestReportFlags:
    def test_report_has_junit_flag(self):
        from click.testing import CliRunner

        from litmusai.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["report", "--help"])
        assert "--junit" in result.output
        assert "--csv" in result.output

    def test_junit_via_cli(self, tmp_path):
        from click.testing import CliRunner

        from litmusai.cli.main import cli

        runner = CliRunner()
        data = _make_data()
        results_file = tmp_path / "results.json"
        results_file.write_text(json.dumps(data))
        xml_file = tmp_path / "results.xml"

        result = runner.invoke(
            cli,
            ["report", "-r", str(results_file),
             "--junit", str(xml_file)],
        )
        assert result.exit_code == 0
        assert xml_file.exists()

    def test_csv_via_cli(self, tmp_path):
        from click.testing import CliRunner

        from litmusai.cli.main import cli

        runner = CliRunner()
        data = _make_data()
        results_file = tmp_path / "results.json"
        results_file.write_text(json.dumps(data))
        csv_file = tmp_path / "results.csv"

        result = runner.invoke(
            cli,
            ["report", "-r", str(results_file),
             "--csv", str(csv_file)],
        )
        assert result.exit_code == 0
        assert csv_file.exists()
