"""Tests for retry logic, tracing, and assertion plugins."""


import pytest

from litmusai.assertions import (
    Assertion,
    AssertionResult,
    _clear_registered_assertions,
    get_registered_assertions,
    register_assertion,
)
from litmusai.retry import RetryConfig, with_retry
from litmusai.tracing import Span, Tracer

# ─── Retry Logic ─────────────────────────────────────────────────


class TestRetryConfig:
    def test_defaults(self):
        c = RetryConfig()
        assert c.max_retries == 3
        assert c.backoff_base == 1.0
        assert c.jitter is True

    def test_custom(self):
        c = RetryConfig(max_retries=5, backoff_base=0.5, jitter=False)
        assert c.max_retries == 5
        assert c.backoff_base == 0.5


class TestWithRetry:
    @pytest.mark.asyncio
    async def test_succeeds_first_try(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await with_retry(fn, config=RetryConfig(max_retries=3))
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("fail")
            return "ok"

        config = RetryConfig(
            max_retries=3, backoff_base=0.01, jitter=False,
        )
        result = await with_retry(fn, config=config)
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_exhausts_retries(self):
        async def fn():
            raise ValueError("always fail")

        config = RetryConfig(
            max_retries=2, backoff_base=0.01, jitter=False,
        )
        with pytest.raises(ValueError, match="always fail"):
            await with_retry(fn, config=config)

    @pytest.mark.asyncio
    async def test_no_retries(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        config = RetryConfig(max_retries=0)
        with pytest.raises(ValueError):
            await with_retry(fn, config=config)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_specific_exception(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("network")
            return "ok"

        config = RetryConfig(
            max_retries=3,
            backoff_base=0.01,
            jitter=False,
            retry_on=(ConnectionError,),
        )
        result = await with_retry(fn, config=config)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_no_retry_wrong_exception(self):
        async def fn():
            raise TypeError("wrong type")

        config = RetryConfig(
            max_retries=3,
            retry_on=(ConnectionError,),
        )
        with pytest.raises(TypeError):
            await with_retry(fn, config=config)

    @pytest.mark.asyncio
    async def test_on_retry_callback(self):
        retries = []

        def on_retry(attempt, error, delay):
            retries.append((attempt, str(error)))

        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("fail")
            return "ok"

        config = RetryConfig(
            max_retries=3,
            backoff_base=0.01,
            jitter=False,
            on_retry=on_retry,
        )
        await with_retry(fn, config=config)
        assert len(retries) == 2
        assert retries[0][0] == 1
        assert retries[1][0] == 2

    @pytest.mark.asyncio
    async def test_sync_function(self):
        def fn():
            return "sync ok"

        result = await with_retry(fn)
        assert result == "sync ok"

    @pytest.mark.asyncio
    async def test_passes_args(self):
        async def fn(a, b, c=10):
            return a + b + c

        result = await with_retry(fn, 1, 2, c=3)
        assert result == 6


# ─── Tracing ─────────────────────────────────────────────────────


class TestSpan:
    def test_basic_span(self):
        s = Span(name="test", start_time=1.0, end_time=1.5)
        assert s.duration_ms == 500.0
        assert s.status == "ok"

    def test_set_attribute(self):
        s = Span(name="test")
        s.set_attribute("model", "gpt-4o")
        assert s.attributes["model"] == "gpt-4o"

    def test_set_error(self):
        s = Span(name="test")
        s.set_error("timeout")
        assert s.status == "error"
        assert s.error == "timeout"

    def test_to_dict(self):
        s = Span(name="test", start_time=1.0, end_time=2.0)
        s.set_attribute("tokens", 100)
        d = s.to_dict()
        assert d["name"] == "test"
        assert d["duration_ms"] == 1000.0
        assert d["attributes"]["tokens"] == 100


class TestTracer:
    def test_basic_trace(self):
        tracer = Tracer(name="test")
        with tracer.span("step1") as s:
            s.set_attribute("key", "value")
        assert len(tracer.spans) == 1
        assert tracer.spans[0].duration_ms > 0

    def test_nested_spans(self):
        tracer = Tracer()
        with tracer.span("parent"):
            with tracer.span("child1"):
                pass
            with tracer.span("child2"):
                pass
        assert len(tracer.spans) == 1
        assert len(tracer.spans[0].children) == 2

    def test_error_span(self):
        tracer = Tracer()
        with pytest.raises(ValueError):
            with tracer.span("fail"):
                raise ValueError("boom")
        assert tracer.spans[0].status == "error"
        assert tracer.spans[0].error == "boom"

    def test_summary(self):
        tracer = Tracer(name="test")
        with tracer.span("step1") as s:
            s.set_attribute("model", "gpt-4o")
        summary = tracer.summary()
        assert "test" in summary
        assert "step1" in summary
        assert "model=gpt-4o" in summary

    def test_to_json(self):
        tracer = Tracer()
        with tracer.span("step"):
            pass
        j = tracer.to_json()
        import json
        data = json.loads(j)
        assert "spans" in data
        assert data["spans"][0]["name"] == "step"

    def test_save(self, tmp_path):
        tracer = Tracer()
        with tracer.span("step"):
            pass
        path = tracer.save(tmp_path / "trace.json")
        assert path.exists()

    def test_reset(self):
        tracer = Tracer()
        with tracer.span("step"):
            pass
        assert len(tracer.spans) == 1
        tracer.reset()
        assert len(tracer.spans) == 0

    def test_total_duration(self):
        tracer = Tracer()
        with tracer.span("s1"):
            pass
        with tracer.span("s2"):
            pass
        assert tracer.total_duration_ms > 0

    def test_to_dict(self):
        tracer = Tracer(name="mytest")
        with tracer.span("eval"):
            pass
        d = tracer.to_dict()
        assert d["name"] == "mytest"
        assert len(d["spans"]) == 1


# ─── Assertion Plugin Registry ──────────────────────────────────


class LengthCheck(Assertion):
    """Custom assertion: checks response length."""

    def __init__(self, min_length: int = 10):
        self.min_length = min_length

    def check(self, response, *, context=None):
        length = len(response)
        passed = length >= self.min_length
        return AssertionResult(
            passed=passed,
            score=min(1.0, length / self.min_length) if self.min_length else 1.0,
            reason=f"Length {length} >= {self.min_length}" if passed
                   else f"Length {length} < {self.min_length}",
        )


class TestPluginRegistry:
    def setup_method(self):
        _clear_registered_assertions()

    def teardown_method(self):
        _clear_registered_assertions()

    def test_register_and_get(self):
        register_assertion("length_check", LengthCheck)
        registered = get_registered_assertions()
        assert "length_check" in registered
        assert registered["length_check"] is LengthCheck

    def test_register_case_insensitive(self):
        register_assertion("LENGTH_CHECK", LengthCheck)
        registered = get_registered_assertions()
        assert "length_check" in registered

    def test_register_invalid_name(self):
        with pytest.raises(ValueError, match="non-empty"):
            register_assertion("", LengthCheck)

    def test_register_non_assertion_class(self):
        with pytest.raises(TypeError, match="Assertion subclass"):
            register_assertion("bad", str)  # type: ignore

    def test_use_in_yaml(self, tmp_path):
        """Registered assertions work in YAML suites."""
        register_assertion("length_check", LengthCheck)

        from litmusai.core.suite import TestSuite

        suite_yaml = tmp_path / "suite.yaml"
        suite_yaml.write_text(
            "name: plugin-test\n"
            "cases:\n"
            "  - id: q1\n"
            "    name: Length test\n"
            "    task: Say something long\n"
            "    assertions:\n"
            "      - type: length_check\n"
            "        min_length: 5\n"
        )
        suite = TestSuite.from_yaml(suite_yaml)
        assert len(suite.cases[0].assertions) == 1

        # Actually run the assertion
        result = suite.cases[0].assertions[0].check(
            "This is a long enough response",
        )
        assert result.passed

    def test_plugin_short_response(self):
        register_assertion("length_check", LengthCheck)
        a = LengthCheck(min_length=100)
        result = a.check("short")
        assert not result.passed

    def test_clear_registry(self):
        register_assertion("length_check", LengthCheck)
        assert len(get_registered_assertions()) == 1
        _clear_registered_assertions()
        assert len(get_registered_assertions()) == 0
