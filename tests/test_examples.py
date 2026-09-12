"""Exercise example workflows without live provider requests."""

import importlib.util
import json
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def load_example(filename):
    spec = importlib.util.spec_from_file_location("litmus_example", EXAMPLES / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("base_url", [None, "", "https://proxy.example/v1"])
async def test_model_comparison_example(monkeypatch, httpx_mock, base_url):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    if base_url is None:
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    endpoint = (base_url or "https://api.openai.com/v1") + "/chat/completions"
    for _ in range(6):
        httpx_mock.add_response(url=endpoint, json={
            "choices": [{"message": {"content": "345 Russia def prime"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

    example = load_example("02_model_comparison.py")
    await example.main()

    requests = httpx_mock.get_requests()
    assert len(requests) == 6
    assert {json.loads(request.content)["model"] for request in requests} == {"gpt-4o", "gpt-4.1"}


async def test_adapter_demo_executes_echo_subprocess(monkeypatch):
    example = load_example("quickstart.py")
    actual_evaluate = example.evaluate
    captured = {}

    async def evaluate_and_capture(agent, suite, **kwargs):
        result = await actual_evaluate(agent, suite, verbose=False)
        captured[agent.name] = result
        return result

    monkeypatch.setattr(example, "evaluate", evaluate_and_capture)
    await example.main()

    responses = captured["echo-cli"].results
    assert len(responses) == 3
    assert all(result.response.success for result in responses)
    assert all(result.response.output == result.case.task for result in responses)
