"""Reject emojis in a PR title and the commits introduced by its branch."""

from __future__ import annotations

import argparse
import re
import subprocess

EMOJI = re.compile(
    "[\U0001f000-\U0001faff\u2139\u23e9-\u23fa\u2600-\u27bf\u20e3\ufe0f]"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--title", required=True)
    args = parser.parse_args()
    failed = False
    if EMOJI.search(args.title):
        print("PR title contains an emoji. Use a plain description of the change.")
        failed = True

    # The range excludes commits already reachable from the target branch.
    commits = subprocess.check_output(
        ["git", "log", "--format=%H%x00%B%x00", f"{args.base}..{args.head}", "--"],
        encoding="utf-8",
    ).split("\0")
    for index in range(0, len(commits) - 1, 2):
        sha, message = commits[index].strip(), commits[index + 1]
        if EMOJI.search(message):
            print(f"Commit {sha[:12]} contains an emoji. Use a plain commit message.")
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
