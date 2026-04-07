"""Tests for ground truth management (#33)."""

from __future__ import annotations

import pytest
import yaml

from litmusai.ground_truth import (
    GroundTruth,
    apply_ground_truth,
    ground_truth_stats,
    load_ground_truth,
    validate_ground_truth,
)


class TestGroundTruth:
    def test_numeric(self):
        gt = GroundTruth(answer=36, answer_type="numeric", tolerance=0.01)
        assert gt.answer == 36.0
        assert gt.answer_type == "numeric"

    def test_text(self):
        gt = GroundTruth(answer="George Orwell", answer_type="text")
        assert gt.answer == "George Orwell"

    def test_subjective(self):
        gt = GroundTruth(
            answer_type="subjective",
            notes="No single correct answer",
            confidence=0.0,
        )
        assert gt.answer is None

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="answer_type must be"):
            GroundTruth(answer="x", answer_type="invalid")

    def test_invalid_confidence(self):
        with pytest.raises(ValueError, match="confidence must be"):
            GroundTruth(answer="x", confidence=1.5)

    def test_to_dict(self):
        gt = GroundTruth(
            answer=36, answer_type="numeric", tolerance=0.01,
            source="calculation", verified_by="kunal",
        )
        d = gt.to_dict()
        assert d["answer"] == 36.0
        assert d["answer_type"] == "numeric"
        assert d["source"] == "calculation"
        assert "confidence" not in d  # default 1.0 omitted

    def test_from_dict(self):
        d = {
            "answer": "Paris",
            "answer_type": "text",
            "alternatives": ["paris", "PARIS"],
            "source": "geography",
        }
        gt = GroundTruth.from_dict(d)
        assert gt.answer == "Paris"
        assert gt.alternatives == ["paris", "PARIS"]

    def test_roundtrip(self):
        gt = GroundTruth(
            answer=42, answer_type="numeric",
            tolerance=0.1, source="math",
            verified_by="test", confidence=0.9,
        )
        gt2 = GroundTruth.from_dict(gt.to_dict())
        assert gt2.answer == gt.answer
        assert gt2.tolerance == gt.tolerance
        assert gt2.confidence == gt.confidence


class TestToAssertions:
    def test_numeric_assertion(self):
        from litmusai.assertions import Numeric

        gt = GroundTruth(answer=36, answer_type="numeric", tolerance=0.01)
        assertions = gt.to_assertions()
        assert len(assertions) == 1
        assert isinstance(assertions[0], Numeric)

    def test_text_assertion(self):
        from litmusai.assertions import Contains

        gt = GroundTruth(answer="George Orwell", answer_type="text")
        assertions = gt.to_assertions()
        assert len(assertions) == 1
        assert isinstance(assertions[0], Contains)

    def test_text_with_alternatives(self):
        from litmusai.assertions import AnyOf

        gt = GroundTruth(
            answer="George Orwell",
            answer_type="text",
            alternatives=["Orwell", "Eric Arthur Blair"],
        )
        assertions = gt.to_assertions()
        assert len(assertions) == 1
        assert isinstance(assertions[0], AnyOf)

    def test_json_assertion(self):
        from litmusai.assertions import JsonValid

        gt = GroundTruth(answer_type="json")
        assertions = gt.to_assertions()
        assert len(assertions) == 1
        assert isinstance(assertions[0], JsonValid)

    def test_boolean_assertion(self):
        from litmusai.assertions import RegexMatch

        gt = GroundTruth(answer=True, answer_type="boolean")
        assertions = gt.to_assertions()
        assert len(assertions) == 1
        assert isinstance(assertions[0], RegexMatch)

    def test_list_assertion(self):
        gt = GroundTruth(
            answer=["Python", "JavaScript"],
            answer_type="list",
        )
        assertions = gt.to_assertions()
        assert len(assertions) == 2  # one Contains per item

    def test_subjective_assertion(self):
        import litmusai
        from litmusai.assertions import LLMGrade
        litmusai.configure(api_key="test-key")
        try:
            gt = GroundTruth(
                answer_type="subjective",
                notes="Check reasoning quality",
            )
            assertions = gt.to_assertions()
            assert len(assertions) == 1
            assert isinstance(assertions[0], LLMGrade)
        finally:
            litmusai.reset_config()


class TestLoadGroundTruth:
    def test_load_yaml(self, tmp_path):
        gt_file = tmp_path / "ground_truth.yaml"
        gt_file.write_text(yaml.dump({
            "cases": [
                {
                    "id": "math_001",
                    "ground_truth": {
                        "answer": 36,
                        "answer_type": "numeric",
                        "tolerance": 0.01,
                        "source": "calculation",
                    },
                },
                {
                    "id": "history_001",
                    "ground_truth": {
                        "answer": "George Orwell",
                        "answer_type": "text",
                        "alternatives": ["Orwell"],
                    },
                },
            ],
        }))

        entries = load_ground_truth(gt_file)
        assert len(entries) == 2
        assert entries["math_001"].answer == 36.0
        assert entries["history_001"].alternatives == ["Orwell"]

    def test_load_empty(self, tmp_path):
        gt_file = tmp_path / "empty.yaml"
        gt_file.write_text("cases: []")
        entries = load_ground_truth(gt_file)
        assert entries == {}


class TestValidateGroundTruth:
    def test_valid_file(self, tmp_path):
        gt_file = tmp_path / "gt.yaml"
        gt_file.write_text(yaml.dump({
            "cases": [{
                "id": "q1",
                "ground_truth": {
                    "answer": 42,
                    "answer_type": "numeric",
                },
            }],
        }))
        errors = validate_ground_truth(gt_file)
        assert errors == []

    def test_missing_file(self):
        errors = validate_ground_truth("/nonexistent.yaml")
        assert len(errors) == 1
        assert "not found" in errors[0]

    def test_duplicate_id(self, tmp_path):
        gt_file = tmp_path / "gt.yaml"
        gt_file.write_text(yaml.dump({
            "cases": [
                {"id": "q1", "ground_truth": {"answer": 1, "answer_type": "numeric"}},
                {"id": "q1", "ground_truth": {"answer": 2, "answer_type": "numeric"}},
            ],
        }))
        errors = validate_ground_truth(gt_file)
        assert any("duplicate" in e.lower() for e in errors)

    def test_missing_id(self, tmp_path):
        gt_file = tmp_path / "gt.yaml"
        gt_file.write_text(yaml.dump({
            "cases": [{"ground_truth": {"answer": 1, "answer_type": "numeric"}}],
        }))
        errors = validate_ground_truth(gt_file)
        assert any("missing 'id'" in e for e in errors)

    def test_missing_answer_for_non_subjective(self, tmp_path):
        gt_file = tmp_path / "gt.yaml"
        gt_file.write_text(yaml.dump({
            "cases": [{
                "id": "q1",
                "ground_truth": {"answer_type": "numeric"},
            }],
        }))
        errors = validate_ground_truth(gt_file)
        assert any("requires an answer" in e for e in errors)

    def test_invalid_type(self, tmp_path):
        gt_file = tmp_path / "gt.yaml"
        gt_file.write_text(yaml.dump({
            "cases": [{
                "id": "q1",
                "ground_truth": {"answer": "x", "answer_type": "bad"},
            }],
        }))
        errors = validate_ground_truth(gt_file)
        assert len(errors) >= 1


class TestApplyGroundTruth:
    def test_apply_to_suite(self):
        from litmusai import TestCase, TestSuite

        suite = TestSuite(name="test")
        suite.add_case(TestCase(id="q1", name="Q1", task="What is 6*7?"))
        suite.add_case(TestCase(id="q2", name="Q2", task="Who wrote 1984?"))

        gt = {
            "q1": GroundTruth(answer=42, answer_type="numeric"),
            "q2": GroundTruth(
                answer="George Orwell", answer_type="text",
                alternatives=["Orwell"],
            ),
        }

        updated = apply_ground_truth(suite, gt)
        assert updated == 2
        assert len(suite.cases[0].assertions) == 1  # Numeric
        assert len(suite.cases[1].assertions) == 1  # AnyOf
        assert "ground_truth" in suite.cases[0].metadata

    def test_skip_cases_with_existing_assertions(self):
        from litmusai import TestCase, TestSuite
        from litmusai.assertions import Contains

        suite = TestSuite(name="test")
        suite.add_case(TestCase(
            id="q1", name="Q1", task="Test",
            assertions=[Contains(["hello"])],
        ))

        gt = {"q1": GroundTruth(answer=42, answer_type="numeric")}
        updated = apply_ground_truth(suite, gt)
        assert updated == 0  # skipped — already has assertions

    def test_unmatched_cases(self):
        from litmusai import TestCase, TestSuite

        suite = TestSuite(name="test")
        suite.add_case(TestCase(id="q1", name="Q1", task="Test"))

        gt = {"q99": GroundTruth(answer=42, answer_type="numeric")}
        updated = apply_ground_truth(suite, gt)
        assert updated == 0


class TestGroundTruthStats:
    def test_stats(self):
        from litmusai import TestCase, TestSuite

        suite = TestSuite(name="test")
        suite.add_case(TestCase(id="q1", name="Q1", task="T"))
        suite.add_case(TestCase(id="q2", name="Q2", task="T"))
        suite.add_case(TestCase(id="q3", name="Q3", task="T"))

        gt = {
            "q1": GroundTruth(answer=42, answer_type="numeric"),
            "q2": GroundTruth(answer_type="subjective"),
        }

        stats = ground_truth_stats(suite, gt)
        assert stats["total"] == 3
        assert stats["covered"] == 2
        assert stats["subjective"] == 1
        assert stats["missing"] == 1
        assert stats["missing_ids"] == ["q3"]
        assert stats["coverage_pct"] == pytest.approx(66.67, abs=0.1)


class TestYAMLSuiteGroundTruth:
    def test_suite_with_ground_truth(self, tmp_path):
        """YAML suite with ground_truth auto-generates assertions."""
        from litmusai.core.suite import TestSuite

        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(yaml.dump({
            "name": "test",
            "cases": [
                {
                    "id": "q1",
                    "name": "Math",
                    "task": "What is 6*7?",
                    "ground_truth": {
                        "answer": 42,
                        "answer_type": "numeric",
                        "tolerance": 0.01,
                    },
                },
                {
                    "id": "q2",
                    "name": "Author",
                    "task": "Who wrote 1984?",
                    "ground_truth": {
                        "answer": "George Orwell",
                        "answer_type": "text",
                        "alternatives": ["Orwell"],
                    },
                },
            ],
        }))

        suite = TestSuite.from_yaml(suite_file)
        assert len(suite.cases) == 2
        # Assertions auto-generated from ground_truth
        assert len(suite.cases[0].assertions) == 1
        assert len(suite.cases[1].assertions) == 1
        assert "ground_truth" in suite.cases[0].metadata

    def test_explicit_assertions_take_precedence(self, tmp_path):
        """Explicit assertions prevent ground_truth auto-generation."""
        from litmusai.core.suite import TestSuite

        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(yaml.dump({
            "name": "test",
            "cases": [{
                "id": "q1",
                "name": "Math",
                "task": "What is 6*7?",
                "assertions": [
                    {"type": "contains", "value": "42"},
                ],
                "ground_truth": {
                    "answer": 42,
                    "answer_type": "numeric",
                },
            }],
        }))

        suite = TestSuite.from_yaml(suite_file)
        # Explicit assertions used, not ground_truth
        assert len(suite.cases[0].assertions) == 1
        assert "ground_truth" not in suite.cases[0].metadata


class TestCLI:
    def test_validate_command(self, tmp_path):
        from click.testing import CliRunner

        from litmusai.cli.main import cli

        gt_file = tmp_path / "gt.yaml"
        gt_file.write_text(yaml.dump({
            "cases": [{
                "id": "q1",
                "ground_truth": {"answer": 42, "answer_type": "numeric"},
            }],
        }))

        runner = CliRunner()
        result = runner.invoke(cli, ["validate-ground-truth", str(gt_file)])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_validate_invalid(self, tmp_path):
        from click.testing import CliRunner

        from litmusai.cli.main import cli

        gt_file = tmp_path / "gt.yaml"
        gt_file.write_text(yaml.dump({
            "cases": [{"ground_truth": {"answer": 1, "answer_type": "numeric"}}],
        }))

        runner = CliRunner()
        result = runner.invoke(cli, ["validate-ground-truth", str(gt_file)])
        assert result.exit_code == 1

    def test_stats_command_exists(self):
        from click.testing import CliRunner

        from litmusai.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["ground-truth-stats", "--help"])
        assert result.exit_code == 0
        assert "--suite" in result.output
        assert "--ground-truth" in result.output
