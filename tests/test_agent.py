"""Tests for the Agent class and all adapters."""


import pytest

from litmusai.core.agent import Agent, AgentResponse, AgentStep, ToolCall

# ─── Test Fixtures ─────────────────────────────────────────────────


def sync_agent(task: str) -> str:
    return f"Response to: {task}"


async def async_agent(task: str) -> str:
    return f"Async response to: {task}"


def dict_agent(task: str) -> dict:
    return {
        "output": f"Dict response to: {task}",
        "cost": 0.01,
        "tokens_used": 100,
        "input_tokens": 60,
        "output_tokens": 40,
    }


def agent_with_tools(task: str) -> dict:
    return {
        "output": f"Used tools for: {task}",
        "tool_calls": [
            {"name": "search", "arguments": {"query": task}, "result": "found it"},
            {"name": "calculator", "arguments": {"expr": "2+2"}, "result": "4"},
        ],
        "steps": [
            {
                "step_number": 1, "action": "search",
                "thought": "need to search", "observation": "found it",
            },
            {
                "step_number": 2, "action": "calculator",
                "thought": "need to calculate", "observation": "4",
            },
        ],
    }


def agent_response_agent(task: str) -> AgentResponse:
    return AgentResponse(
        output=f"Direct response to: {task}",
        cost=0.05,
        tokens_used=200,
        model="test-model",
        tool_calls=[ToolCall(name="test_tool", arguments={"key": "value"}, result="done")],
        steps=[AgentStep(step_number=1, action="test", observation="ok")],
    )


def failing_agent(task: str) -> str:
    raise ValueError("Intentional error for testing")


def slow_agent(task: str) -> str:
    import time
    time.sleep(0.05)
    return f"Slow response to: {task}"


# ─── Test Agent Core ───────────────────────────────────────────────


class TestAgentCore:
    @pytest.mark.asyncio
    async def test_sync_function(self):
        agent = Agent.from_function(sync_agent, name="sync-test")
        response = await agent.run("hello")
        assert response.success
        assert "hello" in response.output
        assert response.latency_ms > 0

    @pytest.mark.asyncio
    async def test_async_function(self):
        agent = Agent.from_function(async_agent, name="async-test")
        response = await agent.run("hello")
        assert response.success
        assert "Async" in response.output

    @pytest.mark.asyncio
    async def test_dict_response(self):
        agent = Agent.from_function(dict_agent, name="dict-test")
        response = await agent.run("hello")
        assert response.success
        assert response.cost == 0.01
        assert response.tokens_used == 100
        assert response.input_tokens == 60
        assert response.output_tokens == 40
        assert response.total_tokens == 100

    @pytest.mark.asyncio
    async def test_agent_response_passthrough(self):
        agent = Agent.from_function(agent_response_agent, name="direct-test")
        response = await agent.run("hello")
        assert response.success
        assert response.cost == 0.05
        assert response.model == "test-model"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "test_tool"
        assert len(response.steps) == 1

    @pytest.mark.asyncio
    async def test_tool_calls_parsing(self):
        agent = Agent.from_function(agent_with_tools, name="tools-test")
        response = await agent.run("test task")
        assert response.success
        assert len(response.tool_calls) == 2
        assert response.tool_calls[0].name == "search"
        assert response.tool_calls[1].name == "calculator"
        assert response.num_tool_calls == 2  # 2 direct tool calls

    @pytest.mark.asyncio
    async def test_steps_parsing(self):
        agent = Agent.from_function(agent_with_tools, name="steps-test")
        response = await agent.run("test task")
        assert response.success
        assert len(response.steps) == 2
        assert response.steps[0].action == "search"
        assert response.steps[0].thought == "need to search"
        assert response.steps[1].observation == "4"
        assert response.num_steps == 2

    @pytest.mark.asyncio
    async def test_error_handling(self):
        agent = Agent.from_function(failing_agent, name="fail-test")
        response = await agent.run("hello")
        assert not response.success
        assert "Intentional error" in (response.error or "")
        assert response.output == ""
        assert response.latency_ms > 0

    @pytest.mark.asyncio
    async def test_latency_tracking(self):
        agent = Agent.from_function(slow_agent, name="slow-test")
        response = await agent.run("hello")
        assert response.success
        assert response.latency_ms >= 40  # at least 40ms (we sleep 50ms)

    def test_repr(self):
        agent = Agent.from_function(sync_agent, name="test", model="gpt-4")
        assert "test" in repr(agent)
        assert "gpt-4" in repr(agent)

    def test_metadata(self):
        agent = Agent(
            fn=sync_agent,
            name="test",
            metadata={"version": "1.0"},
        )
        assert agent.metadata["version"] == "1.0"


# ─── Test AgentResponse ───────────────────────────────────────────


class TestAgentResponse:
    def test_total_tokens_from_split(self):
        resp = AgentResponse(output="test", input_tokens=100, output_tokens=50)
        assert resp.total_tokens == 150

    def test_total_tokens_fallback(self):
        resp = AgentResponse(output="test", tokens_used=200)
        assert resp.total_tokens == 200

    def test_num_steps_empty(self):
        resp = AgentResponse(output="test")
        assert resp.num_steps == 0
        assert resp.num_tool_calls == 0

    def test_num_tool_calls_combined(self):
        resp = AgentResponse(
            output="test",
            tool_calls=[ToolCall(name="a"), ToolCall(name="b")],
            steps=[
                AgentStep(
                    step_number=1,
                    action="step1",
                    tool_calls=[ToolCall(name="c")],
                ),
            ],
        )
        assert resp.num_tool_calls == 3  # 2 direct + 1 in step


# ─── Test ToolCall & AgentStep ────────────────────────────────────


class TestDataClasses:
    def test_tool_call_repr(self):
        tc = ToolCall(name="search", arguments={"q": "test"})
        assert "search" in repr(tc)

    def test_agent_step_repr(self):
        step = AgentStep(step_number=1, action="search")
        assert "#1" in repr(step)
        assert "search" in repr(step)

    def test_tool_call_defaults(self):
        tc = ToolCall(name="test")
        assert tc.arguments == {}
        assert tc.result == ""
        assert tc.duration_ms == 0.0

    def test_agent_step_defaults(self):
        step = AgentStep(step_number=1, action="test")
        assert step.observation == ""
        assert step.thought == ""
        assert step.tool_calls == []


# ─── Test CLI Adapter ─────────────────────────────────────────────


class TestCLIAdapter:
    @pytest.mark.asyncio
    async def test_from_cli_echo(self):
        agent = Agent.from_cli("cat", name="echo-agent")
        response = await agent.run("hello world")
        assert response.success
        assert "hello world" in response.output

    @pytest.mark.asyncio
    async def test_from_cli_failure(self):
        agent = Agent.from_cli("false", name="fail-agent")
        response = await agent.run("test")
        # 'false' command returns exit code 1
        assert not response.success
        assert response.error
        assert response.output == ""

    @pytest.mark.asyncio
    async def test_from_cli_python(self):
        agent = Agent.from_cli(
            'python3 -c "import sys; print(f\'Got: {sys.stdin.read().strip()}\')"',
            name="python-agent",
        )
        response = await agent.run("test input")
        assert response.success
        assert "Got: test input" in response.output


# ─── Test HTTP Adapter ────────────────────────────────────────────


class TestHTTPAdapter:
    def test_from_url_creation(self):
        agent = Agent.from_url(
            "http://localhost:8000/agent",
            name="http-test",
            headers={"Authorization": "Bearer xxx"},
            request_field="prompt",
            response_field="reply",
        )
        assert agent.name == "http-test"

    def test_from_url_custom_fields(self):
        agent = Agent.from_url(
            "http://localhost:8000",
            request_field="message",
            response_field="answer",
        )
        assert agent.name == "http-agent"


# ─── Test Callable Adapter ────────────────────────────────────────


class TestCallableAdapter:
    @pytest.mark.asyncio
    async def test_from_callable(self):
        class MyAgent:
            def run(self, task: str) -> str:
                return f"MyAgent: {task}"

        agent = Agent.from_callable(MyAgent(), method="run", name="custom")
        response = await agent.run("hello")
        assert response.success
        assert "MyAgent: hello" in response.output

    @pytest.mark.asyncio
    async def test_from_callable_default_method(self):
        class RunAgent:
            def run(self, task: str) -> str:
                return f"RunAgent: {task}"

        agent = Agent.from_callable(RunAgent(), name="run-agent")
        response = await agent.run("test")
        assert response.success
        assert "RunAgent: test" in response.output

    def test_from_callable_invalid_method(self):
        class HasAttr:
            not_a_method = "just a string"

        with pytest.raises(TypeError, match="not callable"):
            Agent.from_callable(HasAttr(), method="not_a_method")

    def test_from_callable_missing_method(self):
        class NoMethod:
            pass

        with pytest.raises(TypeError, match="not callable"):
            Agent.from_callable(NoMethod(), method="nonexistent")


# ─── Test LangChain Adapter (Mocked) ─────────────────────────────


class TestLangChainAdapter:
    @pytest.mark.asyncio
    async def test_langchain_with_ainvoke(self):
        class MockLCAgent:
            async def ainvoke(self, inputs: dict) -> dict:
                return {
                    "output": f"LangChain: {inputs['input']}",
                    "intermediate_steps": [],
                }

        agent = Agent.from_langchain(MockLCAgent(), name="lc-test")
        response = await agent.run("hello")
        assert response.success
        assert "LangChain: hello" in response.output

    @pytest.mark.asyncio
    async def test_langchain_with_invoke(self):
        class MockLCSync:
            def invoke(self, inputs: dict) -> dict:
                return {"output": f"Sync LC: {inputs['input']}"}

        agent = Agent.from_langchain(MockLCSync(), name="lc-sync")
        response = await agent.run("hello")
        assert response.success
        assert "Sync LC: hello" in response.output

    @pytest.mark.asyncio
    async def test_langchain_with_tools(self):
        class MockAction:
            tool = "search"
            tool_input = {"query": "test"}
            log = "I should search"

        class MockLCWithTools:
            async def ainvoke(self, inputs: dict) -> dict:
                return {
                    "output": "Found the answer",
                    "intermediate_steps": [
                        (MockAction(), "search result here"),
                    ],
                }

        agent = Agent.from_langchain(MockLCWithTools(), name="lc-tools")
        response = await agent.run("search for something")
        assert response.success
        assert response.output == "Found the answer"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "search"
        assert len(response.steps) == 1
        assert response.steps[0].thought == "I should search"

    @pytest.mark.asyncio
    async def test_langchain_no_invoke_raises(self):
        class NotAnAgent:
            pass

        agent = Agent.from_langchain(NotAnAgent(), name="bad-agent")
        response = await agent.run("hello")
        assert not response.success
        assert "invoke" in (response.error or "").lower()


# ─── Test CrewAI Adapter (Mocked) ─────────────────────────────────


class TestCrewAIAdapter:
    @pytest.mark.asyncio
    async def test_crewai_basic(self):
        class MockCrew:
            def kickoff(self, inputs: dict) -> str:
                return f"CrewAI: {inputs['task']}"

        agent = Agent.from_crewai(MockCrew(), name="crew-test")
        response = await agent.run("hello")
        assert response.success
        assert "CrewAI: hello" in response.output

    @pytest.mark.asyncio
    async def test_crewai_with_token_usage(self):
        class MockTokenUsage:
            total_tokens = 500
            prompt_tokens = 300
            completion_tokens = 200

        class MockCrewResult:
            token_usage = MockTokenUsage()
            tasks_output = []

            def __str__(self):
                return "crew result"

        class MockCrewWithUsage:
            def kickoff(self, inputs: dict) -> MockCrewResult:
                return MockCrewResult()

        agent = Agent.from_crewai(MockCrewWithUsage(), name="crew-tokens")
        response = await agent.run("test")
        assert response.success
        assert response.tokens_used == 500
        assert response.input_tokens == 300
        assert response.output_tokens == 200


# ─── Test OpenAI Chat Adapter ──────────────────────────────────────


class TestOpenAIChatAdapter:
    def test_creation_defaults(self):
        agent = Agent.from_openai_chat(
            api_key="sk-test",
            model="gpt-4o",
        )
        assert agent.name == "gpt-4o"
        assert agent.model == "gpt-4o"

    def test_creation_custom_name(self):
        agent = Agent.from_openai_chat(
            api_key="sk-test",
            model="gpt-4o",
            name="my-agent",
        )
        assert agent.name == "my-agent"

    def test_creation_all_params(self):
        agent = Agent.from_openai_chat(
            base_url="http://localhost:4000",
            api_key="sk-test",
            model="claude-sonnet-4",
            name="claude",
            system_prompt="You are helpful.",
            temperature=0.0,
            max_tokens=500,
            timeout=30,
            extra_headers={"X-Custom": "yes"},
            extra_body={"top_p": 0.9},
        )
        assert agent.name == "claude"
        assert agent.model == "claude-sonnet-4"

    @pytest.mark.asyncio
    async def test_parses_token_usage(self, httpx_mock):
        """Real token usage is extracted from the API response."""
        agent = Agent.from_openai_chat(
            base_url="http://fake-api.test/v1",
            api_key="sk-test",
            model="gpt-4o",
        )

        httpx_mock.add_response(
            url="http://fake-api.test/v1/chat/completions",
            json={
                "choices": [
                    {"message": {"content": "The answer is 42.", "role": "assistant"}}
                ],
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 8,
                    "total_tokens": 23,
                },
                "model": "gpt-4o-2026-03-15",
            },
        )

        response = await agent.run("What is the meaning of life?")
        assert response.success
        assert response.output == "The answer is 42."
        assert response.input_tokens == 15
        assert response.output_tokens == 8
        assert response.tokens_used == 23
        assert response.model == "gpt-4o-2026-03-15"

    @pytest.mark.asyncio
    async def test_zero_tokens_when_missing(self, httpx_mock):
        """Handles responses without usage field gracefully."""
        agent = Agent.from_openai_chat(
            base_url="http://fake-api.test/v1",
            api_key="sk-test",
            model="gpt-4o",
        )

        httpx_mock.add_response(
            url="http://fake-api.test/v1/chat/completions",
            json={
                "choices": [
                    {"message": {"content": "Hello!", "role": "assistant"}}
                ],
            },
        )

        response = await agent.run("Hi")
        assert response.success
        assert response.output == "Hello!"
        assert response.input_tokens == 0
        assert response.output_tokens == 0
        assert response.tokens_used == 0

    @pytest.mark.asyncio
    async def test_extracts_tool_calls(self, httpx_mock):
        """Tool calls from the API response are captured."""
        agent = Agent.from_openai_chat(
            base_url="http://fake-api.test/v1",
            api_key="sk-test",
            model="gpt-4o",
        )

        httpx_mock.add_response(
            url="http://fake-api.test/v1/chat/completions",
            json={
                "choices": [{
                    "message": {
                        "content": "",
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }],
                    },
                }],
                "usage": {"prompt_tokens": 20, "completion_tokens": 15,
                          "total_tokens": 35},
            },
        )

        response = await agent.run("What's the weather in Paris?")
        assert response.success
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[0].arguments == {"city": "Paris"}

    @pytest.mark.asyncio
    async def test_computes_cost_with_pricing(self, httpx_mock):
        """Cost is computed when pricing is registered."""
        from litmusai.benchmarks import register_pricing

        register_pricing("cost-test-model", 3.0, 15.0)

        agent = Agent.from_openai_chat(
            base_url="http://fake-api.test/v1",
            api_key="sk-test",
            model="cost-test-model",
        )

        httpx_mock.add_response(
            url="http://fake-api.test/v1/chat/completions",
            json={
                "choices": [
                    {"message": {"content": "Hello", "role": "assistant"}}
                ],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 500,
                    "total_tokens": 1500,
                },
                "model": "cost-test-model",
            },
        )

        response = await agent.run("Hi")
        # 1000 * 3.0/1M + 500 * 15.0/1M = 0.003 + 0.0075 = 0.0105
        assert response.cost == pytest.approx(0.0105, abs=1e-6)

    @pytest.mark.asyncio
    async def test_system_prompt_included(self, httpx_mock):
        """System prompt is sent in the messages."""
        agent = Agent.from_openai_chat(
            base_url="http://fake-api.test/v1",
            api_key="sk-test",
            model="gpt-4o",
            system_prompt="You are a pirate.",
        )

        httpx_mock.add_response(
            url="http://fake-api.test/v1/chat/completions",
            json={
                "choices": [
                    {"message": {"content": "Arr!", "role": "assistant"}}
                ],
                "usage": {"prompt_tokens": 25, "completion_tokens": 5,
                          "total_tokens": 30},
            },
        )

        response = await agent.run("Hello")
        assert response.success
        assert response.output == "Arr!"

        # Verify the request included the system prompt
        request = httpx_mock.get_request()
        import json
        body = json.loads(request.content)
        assert len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][0]["content"] == "You are a pirate."
        assert body["messages"][1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_http_error_returns_failure(self, httpx_mock):
        """HTTP errors result in a failed AgentResponse."""
        agent = Agent.from_openai_chat(
            base_url="http://fake-api.test/v1",
            api_key="sk-test",
            model="gpt-4o",
        )

        httpx_mock.add_response(
            url="http://fake-api.test/v1/chat/completions",
            status_code=429,
            json={"error": {"message": "Rate limited"}},
        )

        response = await agent.run("Hi")
        assert not response.success
        assert response.error is not None

    @pytest.mark.asyncio
    async def test_base_url_without_v1(self, httpx_mock):
        """Base URL without /v1 suffix is handled correctly."""
        agent = Agent.from_openai_chat(
            base_url="http://localhost:4000",
            api_key="sk-test",
            model="gpt-4o",
        )

        httpx_mock.add_response(
            url="http://localhost:4000/v1/chat/completions",
            json={
                "choices": [
                    {"message": {"content": "OK", "role": "assistant"}}
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2,
                          "total_tokens": 7},
            },
        )

        response = await agent.run("Hi")
        assert response.success
        assert response.output == "OK"
        assert response.tokens_used == 7

    @pytest.mark.asyncio
    async def test_temperature_and_max_tokens(self, httpx_mock):
        """Temperature and max_tokens are sent in request body."""
        agent = Agent.from_openai_chat(
            base_url="http://fake-api.test/v1",
            api_key="sk-test",
            model="gpt-4o",
            temperature=0.0,
            max_tokens=256,
        )

        httpx_mock.add_response(
            url="http://fake-api.test/v1/chat/completions",
            json={
                "choices": [
                    {"message": {"content": "OK", "role": "assistant"}}
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2,
                          "total_tokens": 7},
            },
        )

        response = await agent.run("Hi")
        assert response.success

        import json
        request = httpx_mock.get_request()
        body = json.loads(request.content)
        assert body["temperature"] == 0.0
        assert body["max_tokens"] == 256

    @pytest.mark.asyncio
    async def test_extra_body_merged(self, httpx_mock):
        """Extra body params are merged into the request."""
        agent = Agent.from_openai_chat(
            base_url="http://fake-api.test/v1",
            api_key="sk-test",
            model="gpt-4o",
            extra_body={"top_p": 0.9, "seed": 42},
        )

        httpx_mock.add_response(
            url="http://fake-api.test/v1/chat/completions",
            json={
                "choices": [
                    {"message": {"content": "OK", "role": "assistant"}}
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2,
                          "total_tokens": 7},
            },
        )

        response = await agent.run("Hi")
        assert response.success

        import json
        request = httpx_mock.get_request()
        body = json.loads(request.content)
        assert body["top_p"] == 0.9
        assert body["seed"] == 42

    @pytest.mark.asyncio
    async def test_malformed_tool_call_args(self, httpx_mock):
        """Handles invalid JSON in tool call arguments gracefully."""
        agent = Agent.from_openai_chat(
            base_url="http://fake-api.test/v1",
            api_key="sk-test",
            model="gpt-4o",
        )

        httpx_mock.add_response(
            url="http://fake-api.test/v1/chat/completions",
            json={
                "choices": [{
                    "message": {
                        "content": "OK",
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "broken_fn",
                                "arguments": "{not valid json}",
                            },
                        }],
                    },
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                          "total_tokens": 15},
            },
        )

        response = await agent.run("Hi")
        assert response.success
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "broken_fn"
        assert "raw" in response.tool_calls[0].arguments

    @pytest.mark.asyncio
    async def test_no_auth_header_without_api_key(self, httpx_mock):
        """No Authorization header when api_key is empty."""
        agent = Agent.from_openai_chat(
            base_url="http://fake-api.test/v1",
            api_key="",
            model="llama3",
        )

        httpx_mock.add_response(
            url="http://fake-api.test/v1/chat/completions",
            json={
                "choices": [
                    {"message": {"content": "OK", "role": "assistant"}}
                ],
            },
        )

        response = await agent.run("Hi")
        assert response.success

        request = httpx_mock.get_request()
        assert "Authorization" not in request.headers

    @pytest.mark.asyncio
    async def test_null_token_usage(self, httpx_mock):
        """Handles None/null in token usage gracefully."""
        agent = Agent.from_openai_chat(
            base_url="http://fake-api.test/v1",
            api_key="sk-test",
            model="gpt-4o",
        )

        httpx_mock.add_response(
            url="http://fake-api.test/v1/chat/completions",
            json={
                "choices": [
                    {"message": {"content": "OK", "role": "assistant"}}
                ],
                "usage": {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                },
            },
        )

        response = await agent.run("Hi")
        assert response.success
        assert response.input_tokens == 0
        assert response.output_tokens == 0
        assert response.tokens_used == 0

    @pytest.mark.asyncio
    async def test_cost_uses_response_model(self, httpx_mock):
        """Cost lookup tries the API-returned model name first."""
        from litmusai.benchmarks import register_pricing

        register_pricing("gpt-4o-2026-03-15", 2.50, 10.0)

        agent = Agent.from_openai_chat(
            base_url="http://fake-api.test/v1",
            api_key="sk-test",
            model="gpt-4o",
        )

        httpx_mock.add_response(
            url="http://fake-api.test/v1/chat/completions",
            json={
                "choices": [
                    {"message": {"content": "OK", "role": "assistant"}}
                ],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 500,
                    "total_tokens": 1500,
                },
                "model": "gpt-4o-2026-03-15",
            },
        )

        response = await agent.run("Hi")
        assert response.cost > 0

    @pytest.mark.asyncio
    async def test_tool_call_args_already_dict(self, httpx_mock):
        """Handles tool call arguments that are already a dict."""
        agent = Agent.from_openai_chat(
            base_url="http://fake-api.test/v1",
            api_key="sk-test",
            model="gpt-4o",
        )

        httpx_mock.add_response(
            url="http://fake-api.test/v1/chat/completions",
            json={
                "choices": [{
                    "message": {
                        "content": "",
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "Paris"},
                            },
                        }],
                    },
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                          "total_tokens": 15},
            },
        )

        response = await agent.run("Weather?")
        assert response.success
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].arguments == {"city": "Paris"}
