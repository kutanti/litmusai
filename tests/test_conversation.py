"""Tests for multi-turn conversation evaluation (#71)."""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from litmusai.conversation import (
    Conversation,
    ConversationResult,
    ConversationRunner,
    MultiTurnCase,
    Step,
    StepResult,
    load_multi_turn_suite,
)
from litmusai.core.agent import Agent, AgentResponse


def _make_echo_agent() -> Agent:
    """Agent that echoes back and references history."""
    call_count = {"n": 0}

    async def fn(task: str, **kwargs: Any) -> AgentResponse:
        call_count["n"] += 1
        history = kwargs.get("history", [])
        if history:
            prev = history[-1]["content"][:30]
            return AgentResponse(
                output=f"Turn {call_count['n']}: {task} (prev: {prev})",
                model="test",
            )
        return AgentResponse(
            output=f"Turn {call_count['n']}: {task}",
            model="test",
        )

    return Agent(fn=fn, name="echo", model="test")


def _make_failing_agent(fail_at: int = 2) -> Agent:
    """Agent that fails at a specific turn."""
    call_count = {"n": 0}

    async def fn(task: str, **kwargs: Any) -> AgentResponse:
        call_count["n"] += 1
        if call_count["n"] == fail_at:
            return AgentResponse(
                output="I don't understand what you're asking",
                model="test",
            )
        return AgentResponse(
            output=f"Turn {call_count['n']}: handled {task}",
            model="test",
        )

    return Agent(fn=fn, name="failing", model="test")


class TestStep:
    def test_basic(self):
        s = Step(user="hello")
        assert s.user == "hello"
        assert s.assertions == []

    def test_with_name(self):
        s = Step(user="hello", name="greeting")
        assert s.name == "greeting"


class TestMultiTurnCase:
    def test_basic(self):
        case = MultiTurnCase(
            id="test", name="Test",
            steps=[Step(user="hi"), Step(user="bye")],
        )
        assert len(case.steps) == 2
        assert case.id == "test"


class TestStepResult:
    def test_to_dict(self):
        sr = StepResult(
            step_index=0, user="hello", response="hi",
            passed=True, score=1.0, latency_ms=100, cost=0.001,
        )
        d = sr.to_dict()
        assert d["step"] == 0
        assert d["passed"] is True
        assert "is_cascade" not in d  # not set

    def test_cascade_in_dict(self):
        sr = StepResult(
            step_index=2, user="x", response="y",
            passed=False, score=0.0, is_cascade=True,
        )
        d = sr.to_dict()
        assert d["is_cascade"] is True


class TestConversationResult:
    def test_pass_rate(self):
        r = ConversationResult(
            case=MultiTurnCase(id="t", name="T"),
            total_steps=4, passed_steps=3, failed_steps=1,
        )
        assert r.pass_rate == 0.75

    def test_summary(self):
        r = ConversationResult(
            case=MultiTurnCase(id="t", name="T"),
            total_steps=4, passed_steps=2, failed_steps=2,
            cascade_failures=1, independent_failures=1,
            total_cost=0.01, total_latency_ms=500,
            passed=False,
        )
        s = r.summary()
        assert "2/4" in s
        assert "cascade" in s.lower()
        assert "independent" in s.lower()

    def test_to_dict(self):
        r = ConversationResult(
            case=MultiTurnCase(id="t", name="T"),
            total_steps=1, passed_steps=1, passed=True,
        )
        d = r.to_dict()
        assert d["case_id"] == "t"
        assert d["passed"] is True


class TestConversation:
    @pytest.mark.asyncio
    async def test_basic_turns(self):
        agent = _make_echo_agent()
        async with Conversation(agent) as conv:
            r1 = await conv.send("hello")
            assert "Turn 1" in r1.output
            assert conv.turn_count == 1

            r2 = await conv.send("world")
            assert "Turn 2" in r2.output
            assert conv.turn_count == 2

    @pytest.mark.asyncio
    async def test_history_maintained(self):
        agent = _make_echo_agent()
        async with Conversation(agent) as conv:
            await conv.send("first message")
            r2 = await conv.send("second message")
            # Agent should see history
            assert "prev:" in r2.output

    @pytest.mark.asyncio
    async def test_system_prompt(self):
        agent = _make_echo_agent()
        async with Conversation(agent, system_prompt="Be helpful") as conv:
            assert conv.history[0]["role"] == "system"
            await conv.send("hello")
            assert len(conv.history) == 3  # system + user + assistant

    @pytest.mark.asyncio
    async def test_agent_conversation_method(self):
        agent = _make_echo_agent()
        async with agent.conversation(system_prompt="test") as conv:
            r = await conv.send("hi")
            assert r.output


class TestConversationRunner:
    @pytest.mark.asyncio
    async def test_all_pass(self):
        from litmusai.assertions import Contains

        agent = _make_echo_agent()
        case = MultiTurnCase(
            id="t", name="Test",
            steps=[
                Step(user="hello", assertions=[Contains(["Turn"])]),
                Step(user="world", assertions=[Contains(["Turn"])]),
            ],
        )
        result = await ConversationRunner(agent).run(case)
        assert result.passed is True
        assert result.passed_steps == 2
        assert result.failed_steps == 0
        assert result.cascade_failures == 0

    @pytest.mark.asyncio
    async def test_independent_failure(self):
        from litmusai.assertions import Contains

        agent = _make_failing_agent(fail_at=2)
        case = MultiTurnCase(
            id="t", name="Test",
            steps=[
                Step(user="hello", assertions=[Contains(["handled"])]),
                Step(
                    user="world",
                    assertions=[Contains(["handled"])],
                ),  # fails — agent says "I don't understand"
                Step(user="bye", assertions=[Contains(["handled"])]),
            ],
        )
        result = await ConversationRunner(agent).run(case)
        assert result.passed is False
        assert result.independent_failures == 1
        assert result.cascade_failures >= 0
        assert result.first_failure == 1

    @pytest.mark.asyncio
    async def test_cascade_detection(self):
        from litmusai.assertions import Contains

        # Agent always says "I don't understand" — all turns fail
        call_count = {"n": 0}

        async def confused_fn(task: str, **kwargs: Any) -> AgentResponse:
            call_count["n"] += 1
            return AgentResponse(
                output="confused and lost",
                model="test",
            )

        agent = Agent(fn=confused_fn, name="confused", model="test")
        case = MultiTurnCase(
            id="t", name="Test",
            steps=[
                Step(user="hello", assertions=[Contains(["handled"])]),
                Step(user="world", assertions=[Contains(["handled"])]),
                Step(user="bye", assertions=[Contains(["handled"])]),
            ],
        )
        result = await ConversationRunner(agent).run(case)
        assert result.independent_failures == 1  # first failure
        assert result.cascade_failures == 2  # subsequent failures
        assert result.first_failure == 0

    @pytest.mark.asyncio
    async def test_stop_on_failure(self):
        from litmusai.assertions import Contains

        agent = _make_failing_agent(fail_at=1)
        case = MultiTurnCase(
            id="t", name="Test",
            steps=[
                Step(user="hello", assertions=[Contains(["handled"])]),
                Step(user="world", assertions=[Contains(["handled"])]),
                Step(user="bye", assertions=[Contains(["handled"])]),
            ],
        )
        result = await ConversationRunner(
            agent, stop_on_failure=True,
        ).run(case)
        assert result.total_steps == 3
        assert len(result.steps) == 3
        assert result.steps[1].reason == "Skipped — earlier step failed"
        assert result.steps[1].is_cascade is True

    @pytest.mark.asyncio
    async def test_no_assertions_auto_pass(self):
        agent = _make_echo_agent()
        case = MultiTurnCase(
            id="t", name="Test",
            steps=[
                Step(user="hello"),  # no assertions
                Step(user="world"),
            ],
        )
        result = await ConversationRunner(agent).run(case)
        assert result.passed is True
        assert result.passed_steps == 2

    @pytest.mark.asyncio
    async def test_run_suite(self):
        from litmusai.assertions import Contains

        agent = _make_echo_agent()
        cases = [
            MultiTurnCase(
                id="t1", name="Test 1",
                steps=[Step(user="hello", assertions=[Contains(["Turn"])])],
            ),
            MultiTurnCase(
                id="t2", name="Test 2",
                steps=[Step(user="world", assertions=[Contains(["Turn"])])],
            ),
        ]
        results = await ConversationRunner(agent).run_suite(cases)
        assert len(results) == 2
        assert all(r.passed for r in results)

    @pytest.mark.asyncio
    async def test_context_maintenance_detected(self):
        """Agent that says 'I don't understand' flags lost context."""
        async def confused_fn(task: str, **kwargs: Any) -> AgentResponse:
            return AgentResponse(
                output="I don't understand what you're asking",
                model="test",
            )

        agent = Agent(fn=confused_fn, name="confused", model="test")
        case = MultiTurnCase(
            id="t", name="Test",
            steps=[Step(user="hello"), Step(user="world")],
        )
        result = await ConversationRunner(agent).run(case)
        assert result.context_maintained is False


class TestLoadMultiTurnSuite:
    def test_load_yaml(self, tmp_path):
        suite_file = tmp_path / "multi.yaml"
        suite_file.write_text(yaml.dump({
            "type": "multi_turn",
            "cases": [{
                "id": "refund",
                "name": "Refund flow",
                "steps": [
                    {
                        "user": "I want to return my shoes",
                        "assertions": [
                            {"type": "contains", "patterns": ["return"]},
                        ],
                    },
                    {
                        "user": "Order #12345",
                        "assertions": [
                            {"type": "contains", "patterns": ["order"]},
                        ],
                    },
                ],
            }],
        }))

        cases = load_multi_turn_suite(suite_file)
        assert len(cases) == 1
        assert cases[0].id == "refund"
        assert len(cases[0].steps) == 2
        assert len(cases[0].steps[0].assertions) == 1

    def test_load_with_system_prompt(self, tmp_path):
        suite_file = tmp_path / "multi.yaml"
        suite_file.write_text(yaml.dump({
            "cases": [{
                "id": "t1",
                "name": "Test",
                "system_prompt": "You are a helpful agent",
                "steps": [{"user": "hello"}],
            }],
        }))

        cases = load_multi_turn_suite(suite_file)
        assert cases[0].system_prompt == "You are a helpful agent"

    def test_load_no_assertions(self, tmp_path):
        suite_file = tmp_path / "multi.yaml"
        suite_file.write_text(yaml.dump({
            "cases": [{
                "id": "t1",
                "steps": [{"user": "hello"}, {"user": "bye"}],
            }],
        }))

        cases = load_multi_turn_suite(suite_file)
        assert len(cases[0].steps) == 2
        assert cases[0].steps[0].assertions == []


class TestExports:
    def test_imports(self):
        from litmusai import (
            Conversation,
            ConversationRunner,
        )
        assert Conversation is not None
        assert ConversationRunner is not None
