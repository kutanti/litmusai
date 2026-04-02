"""Tests for the assertion engine."""

import pytest

from litmusai.assertions import (
    All,
    AnyOf,
    AssertionResult,
    AtLeast,
    Contains,
    Custom,
    Exact,
    JsonPath,
    JsonSchema,
    JsonValid,
    NotContains,
    Numeric,
    RegexMatch,
    Weighted,
)

# ─── Exact ───────────────────────────────────────────────────────


class TestExact:
    def test_match(self):
        r = Exact("Paris").check("Paris")
        assert r.passed
        assert r.score == 1.0

    def test_case_insensitive(self):
        r = Exact("Paris").check("paris")
        assert r.passed

    def test_case_sensitive(self):
        r = Exact("Paris", case_sensitive=True).check("paris")
        assert not r.passed

    def test_strip(self):
        r = Exact("42").check("  42  ")
        assert r.passed

    def test_no_strip(self):
        r = Exact("42", strip=False).check("  42  ")
        assert not r.passed

    def test_no_match(self):
        r = Exact("Paris").check("London")
        assert not r.passed
        assert r.score == 0.0

    def test_long_response_truncated(self):
        r = Exact("Paris").check("x" * 200)
        assert not r.passed
        assert "..." in r.reason

    def test_repr(self):
        assert "Paris" in repr(Exact("Paris"))


# ─── Contains ────────────────────────────────────────────────────


class TestContains:
    def test_any_mode_found(self):
        r = Contains(["36", "thirty-six"], mode="any").check("It's 36.")
        assert r.passed

    def test_any_mode_none(self):
        r = Contains(["36", "thirty-six"], mode="any").check("hello")
        assert not r.passed

    def test_all_mode_found(self):
        r = Contains(["Orwell", "1949"], mode="all").check(
            "George Orwell wrote it in 1949"
        )
        assert r.passed

    def test_all_mode_partial(self):
        r = Contains(["Orwell", "1949"], mode="all").check(
            "George Orwell wrote it"
        )
        assert not r.passed
        assert r.score == 0.5

    def test_case_insensitive(self):
        r = Contains(["HELLO"]).check("hello world")
        assert r.passed

    def test_case_sensitive(self):
        r = Contains(["HELLO"], case_sensitive=True).check("hello world")
        assert not r.passed

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode must be"):
            Contains(["a"], mode="invalid")

    def test_details(self):
        r = Contains(["a", "b", "c"], mode="all").check("a and b")
        assert r.details["found"] == ["a", "b"]
        assert r.details["missing"] == ["c"]

    def test_repr(self):
        c = Contains(["hello"], mode="all")
        assert "all" in repr(c)


# ─── NotContains ─────────────────────────────────────────────────


class TestNotContains:
    def test_pass(self):
        r = NotContains(["hack", "exploit"]).check("I can't help with that.")
        assert r.passed

    def test_fail(self):
        r = NotContains(["hack"]).check("Here's how to hack it")
        assert not r.passed

    def test_multiple_violations(self):
        r = NotContains(["a", "b", "c"]).check("a b c")
        assert not r.passed
        assert len(r.details["violations"]) == 3
        assert r.score == 0.0

    def test_partial_violations(self):
        r = NotContains(["bad", "evil", "wrong"]).check("that's bad")
        assert not r.passed
        assert r.score == pytest.approx(2 / 3)

    def test_case_insensitive(self):
        r = NotContains(["HACK"]).check("hack it")
        assert not r.passed

    def test_repr(self):
        assert "hack" in repr(NotContains(["hack"]))


# ─── RegexMatch ──────────────────────────────────────────────────


class TestRegexMatch:
    def test_match(self):
        r = RegexMatch(r"\b36\.?0*\b").check("The answer is 36.")
        assert r.passed

    def test_no_match(self):
        r = RegexMatch(r"\b42\b").check("The answer is 36.")
        assert not r.passed

    def test_case_insensitive(self):
        r = RegexMatch(r"paris").check("PARIS is great")
        assert r.passed

    def test_full_match(self):
        r = RegexMatch(r"\d+", full_match=True).check("42")
        assert r.passed

    def test_full_match_fail(self):
        r = RegexMatch(r"\d+", full_match=True).check("answer: 42")
        assert not r.passed

    def test_details(self):
        r = RegexMatch(r"\d+").check("answer: 42")
        assert r.details["match"] == "42"

    def test_repr(self):
        assert "\\d+" in repr(RegexMatch(r"\d+"))


# ─── Numeric ─────────────────────────────────────────────────────


class TestNumeric:
    def test_exact(self):
        r = Numeric(36).check("36")
        assert r.passed

    def test_tolerance(self):
        r = Numeric(36, tolerance=0.5).check("35.8")
        assert r.passed

    def test_outside_tolerance(self):
        r = Numeric(36, tolerance=0.01).check("37")
        assert not r.passed

    def test_decimal(self):
        r = Numeric(3.14, tolerance=0.01).check("Pi is 3.14")
        assert r.passed

    def test_negative(self):
        r = Numeric(-5, tolerance=0.01).check("The result is -5")
        assert r.passed

    def test_comma_number(self):
        r = Numeric(1000, tolerance=1).check("There are 1,000 items")
        assert r.passed

    def test_word_number(self):
        r = Numeric(36, tolerance=1).check("thirty-six")
        assert r.passed

    def test_no_numbers(self):
        r = Numeric(36).check("I don't know the answer")
        assert not r.passed
        assert "No numbers" in r.reason

    def test_relative_tolerance(self):
        r = Numeric(100, relative_tolerance=0.1).check("The answer is 95")
        assert r.passed  # within 10%

    def test_relative_tolerance_fail(self):
        r = Numeric(100, relative_tolerance=0.01).check("The answer is 90")
        assert not r.passed

    def test_multiple_numbers_finds_closest(self):
        r = Numeric(42, tolerance=0.5).check("Between 10 and 42 and 100")
        assert r.passed

    def test_details(self):
        r = Numeric(36).check("36")
        assert r.details["found"] == 36.0
        assert r.details["expected"] == 36.0

    def test_repr(self):
        assert "36" in repr(Numeric(36))

    def test_word_compound(self):
        """thirty-six = 36."""
        r = Numeric(36, tolerance=0.5).check("About thirty-six items")
        assert r.passed

    def test_zero_tolerance_exact(self):
        r = Numeric(0, tolerance=0).check("0")
        assert r.passed

    def test_score_decreases_near_tolerance(self):
        exact = Numeric(100, tolerance=10).check("100")
        near = Numeric(100, tolerance=10).check("109")
        assert exact.score > near.score


# ─── JsonValid ───────────────────────────────────────────────────


class TestJsonValid:
    def test_valid_object(self):
        r = JsonValid().check('{"name": "Alice"}')
        assert r.passed
        assert r.details["type"] == "dict"

    def test_valid_array(self):
        r = JsonValid().check("[1, 2, 3]")
        assert r.passed

    def test_markdown_fence(self):
        r = JsonValid().check('```json\n{"a": 1}\n```')
        assert r.passed

    def test_invalid(self):
        r = JsonValid().check("This is not JSON")
        assert not r.passed

    def test_repr(self):
        assert "JsonValid" in repr(JsonValid())


# ─── JsonSchema ──────────────────────────────────────────────────


class TestJsonSchema:
    def test_valid(self):
        schema = {"type": "object", "required": ["name"]}
        r = JsonSchema(schema).check('{"name": "Alice"}')
        assert r.passed

    def test_missing_required(self):
        schema = {"type": "object", "required": ["name", "age"]}
        r = JsonSchema(schema).check('{"name": "Alice"}')
        assert not r.passed
        assert "age" in r.reason

    def test_wrong_type(self):
        schema = {"type": "array"}
        r = JsonSchema(schema).check('{"name": "Alice"}')
        assert not r.passed

    def test_array_min_items(self):
        schema = {"type": "array", "minItems": 3}
        r = JsonSchema(schema).check("[1, 2]")
        assert not r.passed

    def test_array_max_items(self):
        schema = {"type": "array", "maxItems": 2}
        r = JsonSchema(schema).check("[1, 2, 3]")
        assert not r.passed

    def test_nested_validation(self):
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string"},
            },
        }
        r = JsonSchema(schema).check('{"name": "Alice"}')
        assert r.passed

    def test_nested_type_error(self):
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer"},
            },
        }
        r = JsonSchema(schema).check('{"age": "not a number"}')
        assert not r.passed

    def test_array_items_validation(self):
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name"],
            },
        }
        r = JsonSchema(schema).check('[{"name": "A"}, {"name": "B"}]')
        assert r.passed

    def test_not_json(self):
        r = JsonSchema({"type": "object"}).check("not json")
        assert not r.passed

    def test_markdown_fence(self):
        schema = {"type": "object", "required": ["name"]}
        r = JsonSchema(schema).check('```json\n{"name": "A"}\n```')
        assert r.passed

    def test_repr(self):
        assert "JsonSchema" in repr(JsonSchema({"type": "object"}))


# ─── JsonPath ────────────────────────────────────────────────────


class TestJsonPath:
    def test_simple_path(self):
        r = JsonPath("name", "Jupiter").check('{"name": "Jupiter"}')
        assert r.passed

    def test_nested_path(self):
        r = JsonPath("user.name", "Alice").check(
            '{"user": {"name": "Alice"}}'
        )
        assert r.passed

    def test_array_index(self):
        r = JsonPath("items[0]", "first").check(
            '{"items": ["first", "second"]}'
        )
        assert r.passed

    def test_path_not_found(self):
        r = JsonPath("missing", "value").check('{"name": "Alice"}')
        assert not r.passed

    def test_operator_gt(self):
        r = JsonPath("age", 18, operator="gt").check('{"age": 25}')
        assert r.passed

    def test_operator_lt(self):
        r = JsonPath("age", 18, operator="lt").check('{"age": 10}')
        assert r.passed

    def test_operator_contains(self):
        r = JsonPath("bio", "Python", operator="contains").check(
            '{"bio": "I love Python programming"}'
        )
        assert r.passed

    def test_operator_exists(self):
        r = JsonPath("name", operator="exists").check('{"name": "Alice"}')
        assert r.passed

    def test_not_json(self):
        r = JsonPath("name", "Alice").check("not json")
        assert not r.passed

    def test_repr(self):
        assert "name" in repr(JsonPath("name", "Alice"))


# ─── Custom ──────────────────────────────────────────────────────


class TestCustom:
    def test_bool_true(self):
        r = Custom(lambda r: "def " in r, name="has_fn").check("def foo():")
        assert r.passed

    def test_bool_false(self):
        r = Custom(lambda r: "def " in r).check("no function here")
        assert not r.passed

    def test_float_score(self):
        r = Custom(lambda r: 0.8).check("anything")
        assert r.passed
        assert r.score == 0.8

    def test_float_below_threshold(self):
        r = Custom(lambda r: 0.3).check("anything")
        assert not r.passed

    def test_tuple_result(self):
        r = Custom(lambda r: (True, "looks good")).check("test")
        assert r.passed
        assert "looks good" in r.reason

    def test_assertion_result(self):
        r = Custom(lambda r: AssertionResult(
            passed=True, score=0.9, reason="custom"
        )).check("test")
        assert r.passed
        assert r.score == 0.9

    def test_exception_handling(self):
        r = Custom(lambda r: 1 / 0).check("test")
        assert not r.passed
        assert "error" in r.reason.lower()

    def test_name_in_type(self):
        r = Custom(lambda r: True, name="my_check").check("test")
        assert "my_check" in r.assertion_type

    def test_repr(self):
        assert "my_check" in repr(Custom(lambda r: True, name="my_check"))


# ─── All ─────────────────────────────────────────────────────────


class TestAll:
    def test_all_pass(self):
        r = All(
            Numeric(36), NotContains(["sorry"]),
        ).check("The answer is 36.")
        assert r.passed

    def test_one_fails(self):
        r = All(
            Numeric(36), NotContains(["answer"]),
        ).check("The answer is 36.")
        assert not r.passed

    def test_score_averaged(self):
        r = All(
            Numeric(36), Contains(["36", "result"], mode="all"),
        ).check("The answer is 36.")
        # Numeric passes (1.0), Contains partially (0.5)
        assert r.score == pytest.approx(0.75, abs=0.05)

    def test_details(self):
        r = All(Numeric(36)).check("36")
        assert len(r.details["results"]) == 1

    def test_repr(self):
        assert "All" in repr(All(Numeric(36)))


# ─── Any_ ────────────────────────────────────────────────────────


class TestAny:
    def test_one_passes(self):
        r = AnyOf(
            Exact("36"), Exact("thirty-six"),
        ).check("thirty-six")
        assert r.passed

    def test_none_pass(self):
        r = AnyOf(
            Exact("36"), Exact("thirty-six"),
        ).check("hello")
        assert not r.passed

    def test_best_score(self):
        r = AnyOf(
            Numeric(36), Exact("wrong"),
        ).check("The answer is 36.")
        assert r.passed
        assert r.score >= 0.8

    def test_repr(self):
        assert "AnyOf" in repr(AnyOf(Exact("a")))


# ─── AtLeast ─────────────────────────────────────────────────────


class TestAtLeast:
    def test_enough_pass(self):
        r = AtLeast(2, [
            Exact("36"), Numeric(36), Contains(["36"]),
        ]).check("36")
        assert r.passed

    def test_not_enough(self):
        r = AtLeast(3, [
            Exact("36"), Numeric(36), Contains(["xyz"]),
        ]).check("36")
        assert not r.passed

    def test_details(self):
        r = AtLeast(1, [Exact("a"), Exact("b")]).check("a")
        assert r.details["n_required"] == 1
        assert r.details["n_passed"] == 1

    def test_repr(self):
        assert "AtLeast" in repr(AtLeast(2, [Exact("a")]))


# ─── Weighted ────────────────────────────────────────────────────


class TestWeighted:
    def test_above_threshold(self):
        r = Weighted([
            (Numeric(36), 0.8),
            (NotContains(["sorry"]), 0.2),
        ], threshold=0.7).check("The answer is 36.")
        assert r.passed

    def test_below_threshold(self):
        r = Weighted([
            (Numeric(99), 0.8),
            (NotContains(["answer"]), 0.2),
        ], threshold=0.7).check("The answer is 36.")
        assert not r.passed

    def test_weights_normalized(self):
        # Weights sum to 2.0, should still work
        r = Weighted([
            (Numeric(36), 1.6),
            (NotContains(["sorry"]), 0.4),
        ], threshold=0.7).check("The answer is 36.")
        assert r.passed

    def test_zero_weight(self):
        r = Weighted([], threshold=0.5).check("anything")
        assert not r.passed

    def test_details(self):
        r = Weighted([
            (Numeric(36), 0.5),
            (Contains(["answer"]), 0.5),
        ]).check("The answer is 36.")
        assert len(r.details["results"]) == 2

    def test_repr(self):
        w = Weighted([(Exact("a"), 1.0)])
        assert "Weighted" in repr(w)


# ─── Integration: real-world scenarios ───────────────────────────


class TestRealWorldScenarios:
    def test_math_flexible(self):
        """Math answer should accept multiple formats."""
        check = AnyOf(
            Exact("36"),
            Numeric(36, tolerance=0.01),
            Contains(["thirty-six"]),
        )
        assert check.check("36").passed
        assert check.check("The answer is 36.").passed
        assert check.check("thirty-six").passed
        assert not check.check("42").passed

    def test_code_gen(self):
        """Code generation should have function + docstring."""
        check = All(
            Contains(["def", "return"], mode="all"),
            Custom(lambda r: '"""' in r or "'''" in r, name="docstring"),
        )
        code = 'def foo():\n    """My docstring."""\n    return 42'
        assert check.check(code).passed
        assert not check.check("def foo():\n    return 42").passed

    def test_json_api(self):
        """JSON API response validation."""
        check = All(
            JsonValid(),
            JsonSchema({
                "type": "object",
                "required": ["name", "diameter_km"],
            }),
            JsonPath("name", "Jupiter"),
        )
        r = check.check('{"name": "Jupiter", "diameter_km": 139820}')
        assert r.passed

    def test_safety_refusal(self):
        """Agent should refuse and NOT include harmful content."""
        check = All(
            NotContains(["hack", "exploit", "password"]),
            Contains(["can't", "unable", "won't"], mode="any"),
        )
        r = check.check("I can't help with hacking.")
        # Fails — contains "hack" in "hacking"
        assert not r.passed

        r = check.check("I can't help with that request.")
        assert r.passed

    def test_weighted_scoring(self):
        """Weighted scoring for balanced evaluation."""
        check = Weighted([
            (Numeric(36, tolerance=0.01), 0.6),  # correctness
            (NotContains(["I don't know"]), 0.2),  # confidence
            (Custom(lambda r: len(r.split()) >= 3, name="length"), 0.2),
        ], threshold=0.7)

        assert check.check("The answer is 36.").passed
        assert check.check("36").passed  # short but correct
        assert not check.check("I don't know").passed

    def test_complex_json_validation(self):
        """Complex nested JSON validation."""
        check = All(
            JsonValid(),
            JsonSchema({
                "type": "array",
                "minItems": 3,
                "items": {
                    "type": "object",
                    "required": ["name", "diameter_km"],
                    "properties": {
                        "name": {"type": "string"},
                        "diameter_km": {"type": "number"},
                    },
                },
            }),
        )
        valid = _json(
            '[{"name": "Jupiter", "diameter_km": 139820},'
            '{"name": "Saturn", "diameter_km": 116460},'
            '{"name": "Neptune", "diameter_km": 49528}]'
        )
        assert check.check(valid).passed

        too_few = '[{"name": "Jupiter", "diameter_km": 139820}]'
        assert not check.check(too_few).passed


def _json(s: str) -> str:
    """Identity — just for readability in tests."""
    return s
