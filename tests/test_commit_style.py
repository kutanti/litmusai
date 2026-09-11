"""Check new commit messages without rejecting inherited history."""

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

CHECKER = Path(__file__).resolve().parents[1] / "scripts/check_commit_messages.py"


@pytest.mark.parametrize("title, message, expected", [
    ("Fix output", "Fix output\n\nRetain UTF-8 text.", 0),
    ("Fix \U0001f680 output", "Fix output", 1),
    ("Fix output", "Fix output\n\n\U0001f680", 1),
    ("Fix output", "Add \U0001f1fa\U0001f1f8 locale", 1),
    ("Fix output", "Fix 1\ufe0f\u20e3 case", 1),
    ("Fix output", "Support caf\u00e9 names and \u6771\u4eac", 0),
])
def test_title_and_full_message(monkeypatch, title, message, expected):
    main = runpy.run_path(str(CHECKER))["main"]
    monkeypatch.setattr(sys, "argv", [str(CHECKER), "--base", "main", "--title", title])
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: f"abc\0{message}\0\n")
    assert main() == expected


def test_existing_emoji_commit_is_excluded(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, encoding="utf-8").strip()

    git("init")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    git("-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", "Old \U0001f680 commit")
    base = git("rev-parse", "HEAD")
    git("-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", "Add a plain commit")
    checked = subprocess.run(
        [sys.executable, str(CHECKER), "--base", base, "--title", "Plain title"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
