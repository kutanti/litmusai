"""Exercise the composite action's inputs, outputs, and failure handling."""

import json
import runpy
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def action_run(monkeypatch, tmp_path):
    main = runpy.run_path(str(ROOT / "scripts/run_action.py"))["main"]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("INPUT_SUITE", "suites/customer support.yaml")
    monkeypatch.setenv("INPUT_AGENT", "my_agent:agent")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "outputs"))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("INPUT_POST_COMMENT", "false")
    for name in ("CONCURRENCY", "THRESHOLD", "BUDGET", "RUNS", "LOG_DIR", "BASELINE"):
        monkeypatch.delenv(f"INPUT_{name}", raising=False)
    return main


def write_results():
    path = Path(".litmus/results.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({
        "results": {"summary": {
            "pass_rate": 0.5, "total_cost": 0.03, "passed": 1, "failed": 1,
        }, "results": []},
        "has_regression": True,
    }), encoding="utf-8")


def test_inputs_remain_literal_arguments(action_run, monkeypatch):
    agent = "agent file.py:agent$(touch injected); echo unwanted"
    baseline = Path("baseline 'quoted'.json")
    baseline.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("INPUT_AGENT", agent)
    monkeypatch.setenv("INPUT_BASELINE", str(baseline))
    monkeypatch.setenv("INPUT_LOG_DIR", "results with spaces")
    monkeypatch.setenv("INPUT_RUNS", "3")
    monkeypatch.setenv("INPUT_BUDGET", "0.02")
    monkeypatch.setenv("INPUT_SAVE_BASELINE", "true")

    def run(args, *, check):
        assert check is False
        assert args[args.index("--suite") + 1] == "suites/customer support.yaml"
        assert args[args.index("--agent") + 1] == agent
        assert args[args.index("--baseline") + 1] == str(baseline)
        assert args[args.index("--log-dir") + 1] == "results with spaces"
        assert args[args.index("--runs") + 1] == "3"
        assert args[args.index("--budget") + 1] == "0.02"
        assert "--save-baseline" in args
        write_results()
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", run)
    assert action_run() == 0


def test_failed_evaluation_keeps_outputs_and_comment(action_run, monkeypatch):
    monkeypatch.setenv("INPUT_POST_COMMENT", "true")

    def run(args, *, check):
        write_results()
        return subprocess.CompletedProcess(args, 7)

    monkeypatch.setattr(subprocess, "run", run)
    assert action_run() == 7
    outputs = dict(line.split("=", 1) for line in Path("outputs").read_text().splitlines())
    assert outputs == {
        "results-path": ".litmus/results.json", "pass-rate": "0.5",
        "total-cost": "0.03", "passed": "1", "failed": "1",
        "has-regression": "true", "comment-path": ".litmus/pr-comment.md",
    }
    assert "FAILED" in Path(outputs["comment-path"]).read_text(encoding="utf-8")


@pytest.mark.parametrize("exit_code", [0, 1])
async def test_multi_run_exports_keep_action_outputs_and_comment(
    action_run, monkeypatch, exit_code,
):
    from litmusai import Agent, TestCase, TestSuite, multi_evaluate
    from litmusai.ci import results_to_dict

    predictions = iter(["wrong", "hello"])
    multi = await multi_evaluate(
        Agent.from_function(lambda _: next(predictions)),
        TestSuite("repeated", [TestCase(
            id="q1", task="greet", expected_contains=["hello"],
        )]),
        runs=2, verbose=False,
    )
    payload = {
        "results": results_to_dict(multi),
        "success": exit_code == 0,
        "has_regression": False,
    }
    monkeypatch.setenv("INPUT_RUNS", "2")
    monkeypatch.setenv("INPUT_POST_COMMENT", "true")

    def run(args, *, check):
        path = Path(".litmus/results.json")
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(args, exit_code)

    monkeypatch.setattr(subprocess, "run", run)
    assert action_run() == exit_code
    outputs = dict(line.split("=", 1) for line in Path("outputs").read_text().splitlines())
    assert outputs["pass-rate"] == "1.0"
    assert outputs["passed"] == "1"
    assert outputs["failed"] == "0"
    assert outputs["total-cost"] == "0.0"
    report = Path(outputs["comment-path"]).read_text(encoding="utf-8")
    assert "PASSED" in report
    assert "repeated" in report
    assert "q1" in report
    assert Path(outputs["results-path"]).read_text(encoding="utf-8") == json.dumps(payload)


@pytest.mark.parametrize("exit_code", [0, 1])
def test_missing_results_do_not_reuse_stale_files(action_run, monkeypatch, exit_code):
    write_results()
    Path(".litmus/pr-comment.md").write_text("stale")
    monkeypatch.setattr(subprocess, "run", lambda args, **kw: subprocess.CompletedProcess(
        args, exit_code,
    ))
    assert action_run() != 0
    assert not Path("outputs").exists()
    assert not Path(".litmus/pr-comment.md").exists()
    assert not Path(".litmus/results.json").exists()


def test_action_exposes_outputs_and_uses_checked_out_source():
    action = yaml.safe_load((ROOT / "action.yml").read_text(encoding="utf-8"))
    for name, metadata in action["outputs"].items():
        assert metadata["value"] == "${{ steps.eval.outputs." + name + " }}"
    commands = [step["run"] for step in action["runs"]["steps"] if "run" in step]
    assert any('pip install "$GITHUB_ACTION_PATH"' in command for command in commands)
    assert all("${{ inputs." not in command for command in commands)
