"""Regression tests for the one-time, post-merge 1.0.0 release."""

import json
import runpy
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40
OTHER_SHA = "b" * 40
REPO = "kutanti/litmusai"


@pytest.fixture
def release():
    return runpy.run_path(str(ROOT / "scripts/release_1_0_0.py"))["main"].__globals__


def test_release_notes_and_versions(release, tmp_path):
    pytest.importorskip("tomllib")
    notes = release["release_notes"](ROOT)
    assert "### Migration from 0.x" in notes
    assert "### Known limitations" in notes
    assert "## [0.5.0]" not in notes
    assert "## [Unreleased]" not in notes
    (tmp_path / "src/litmusai").mkdir(parents=True)
    for path in ["pyproject.toml", "src/litmusai/__init__.py", "CHANGELOG.md"]:
        (tmp_path / path).write_text((ROOT / path).read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "src/litmusai/__init__.py").write_text('__version__ = "0.5.0"\n')
    with pytest.raises(ValueError, match="versions"):
        release["release_notes"](tmp_path)
    (tmp_path / "src/litmusai/__init__.py").write_text('__version__ = "1.0.0"\n')
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "1.1.0"\n')
    with pytest.raises(ValueError, match="versions"):
        release["release_notes"](tmp_path)
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "1.0.0"\n')
    for changelog in [
        "## [Unreleased]\n\nNotes\n",
        "## [1.0.0] - 2026-09-17\n\n## [0.5.0]\nOld notes\n",
        "## [1.0.0] - 2026-09-17\nNotes\n## [1.0.0] - 2026-09-17\nDuplicate\n",
    ]:
        (tmp_path / "CHANGELOG.md").write_text(changelog)
        with pytest.raises(ValueError, match="changelog"):
            release["release_notes"](tmp_path)


@pytest.mark.parametrize("existing_tag", ["absent", "lightweight", "annotated", "nested"])
@pytest.mark.parametrize("existing_release", [False, True])
def test_create_and_retry(release, existing_tag, existing_release):
    ref = None if existing_tag == "absent" else {
        "object": {"type": "commit", "sha": SHA},
    }
    if existing_tag in {"annotated", "nested"}:
        ref["object"] = {"type": "tag", "sha": "annotation"}
    calls = []

    def api(path, data=None):
        nonlocal ref
        calls.append((path, data))
        if path == "git/ref/tags/v1.0.0":
            return ref
        if path == "git/refs":
            ref = {"object": {"type": "commit", "sha": data["sha"]}}
            return ref
        if path.startswith("git/tags/"):
            if existing_tag == "nested" and path.endswith("annotation"):
                return {"object": {"type": "tag", "sha": "inner"}}
            return {"object": {"type": "commit", "sha": SHA}}
        if path == "releases/tags/v1.0.0":
            return {"draft": False, "prerelease": False} if existing_release else None
        assert path == "releases"
        return data

    release["api"] = api
    release["ensure_release"](SHA, "Release notes")
    writes = [(path, data) for path, data in calls if data is not None]
    assert len(writes) == (existing_tag == "absent") + (not existing_release)
    if not existing_release:
        assert writes[-1][1] == {
            "tag_name": "v1.0.0", "target_commitish": SHA, "name": "v1.0.0",
            "body": "Release notes", "draft": False, "prerelease": False,
        }


@pytest.mark.parametrize("target", [
    {"type": "commit", "sha": OTHER_SHA},
    {"type": "tree", "sha": SHA},
])
def test_mismatched_tag_never_writes(release, target):
    calls = []

    def api(path, data=None):
        calls.append((path, data))
        return {"object": target}

    release["api"] = api
    with pytest.raises(ValueError, match="refusing to move"):
        release["ensure_release"](SHA, "notes")
    assert calls == [("git/ref/tags/v1.0.0", None)]


@pytest.mark.parametrize("state", [{"draft": True, "prerelease": False},
                                  {"draft": False, "prerelease": True}])
def test_unpublished_or_prerelease_fails(release, state):
    release["api"] = lambda path: (
        {"object": {"type": "commit", "sha": SHA}} if path.startswith("git/") else state
    )
    with pytest.raises(ValueError, match="published stable release"):
        release["ensure_release"](SHA, "notes")


@pytest.mark.parametrize("case", ["merged", "open", "other-commit", "other-base", "other-repo"])
def test_only_release_pr_merge_is_eligible(release, monkeypatch, tmp_path, case):
    pr = {
        "merged": True,
        "base": {"ref": "main", "repo": {"full_name": REPO}},
        "merge_commit_sha": SHA,
    }
    if case == "open":
        pr["merged"] = False
    elif case == "other-commit":
        pr["merge_commit_sha"] = OTHER_SHA
    elif case == "other-base":
        pr["base"]["ref"] = "release"
    elif case == "other-repo":
        pr["base"]["repo"]["full_name"] = "fork/litmusai"
    monkeypatch.setenv("RELEASE_SHA", SHA)
    monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: SHA)
    release["api"] = lambda path: pr if path == "pulls/115" else pytest.fail(path)
    release["release_notes"] = lambda root: "notes"
    writes = []
    release["ensure_release"] = lambda *args: writes.append(args)
    release["main"]()
    assert writes == ([(SHA, "notes")] if case == "merged" else [])
    assert output.exists() == (case == "merged")
    if output.exists():
        assert output.read_text() == f"sha={SHA}\n"
        monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: OTHER_SHA)
        with pytest.raises(ValueError, match="Checkout"):
            release["main"]()


@pytest.mark.parametrize("status,data,absent", [
    (404, None, True), (403, None, False), (500, None, False), (404, {"sha": SHA}, False),
])
def test_api_only_get_404_is_absent(release, monkeypatch, status, data, absent):
    monkeypatch.setenv("GITHUB_API_URL", "https://api.github.com")
    monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
    monkeypatch.setenv("GH_TOKEN", "test-token")

    def request(req, **kwargs):
        assert req.full_url == f"https://api.github.com/repos/{REPO}/git/refs"
        assert req.get_header("Authorization") == "Bearer " + "test-token"
        assert req.data == (json.dumps(data).encode() if data is not None else None)
        raise urllib.error.HTTPError(req.full_url, status, "failure", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", request)
    if absent:
        assert release["api"]("git/refs", data) is None
    else:
        with pytest.raises(urllib.error.HTTPError):
            release["api"]("git/refs", data)


def test_workflow_security_and_manual_paths():
    workflow = yaml.safe_load((ROOT / ".github/workflows/publish.yml").read_text())
    # PyYAML's YAML 1.1 loader interprets the Actions key "on" as True.
    triggers = workflow[True]
    assert triggers["workflow_run"] == {
        "workflows": ["CI"], "types": ["completed"], "branches": ["main"],
    }
    assert triggers["release"] == {"types": ["published"]}
    assert "workflow_dispatch" in triggers
    jobs = workflow["jobs"]
    release = jobs["release"]
    for condition in [
        "github.event_name == 'workflow_run'",
        "github.event.workflow_run.conclusion == 'success'",
        "github.event.workflow_run.event == 'push'",
        "github.event.workflow_run.head_branch == 'main'",
        "github.event.workflow_run.head_repository.full_name == github.repository",
    ]:
        assert condition in release["if"]
    checkout = release["steps"][0]["with"]
    assert checkout["ref"] == "${{ github.event.workflow_run.head_sha }}"
    assert checkout["persist-credentials"] is False
    assert release["permissions"] == {"contents": "write", "pull-requests": "read"}
    publish = jobs["publish"]
    assert publish["needs"] == "release"
    assert "!cancelled()" in publish["if"]
    assert "github.event_name != 'workflow_run'" in publish["if"]
    assert "needs.release.result == 'success'" in publish["if"]
    assert "needs.release.outputs.sha != ''" in publish["if"]
    assert publish["permissions"] == {"contents": "read", "id-token": "write"}
    assert publish["steps"][0]["with"]["ref"] == "${{ needs.release.outputs.sha || github.sha }}"
    assert publish["steps"][-1]["with"]["skip-existing"] is True
