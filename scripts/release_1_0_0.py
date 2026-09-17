"""Release only PR #115's CI-tested merge; never move an existing tag."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

VERSION = "1.0.0"
TAG = f"v{VERSION}"


def api(path: str, data: dict | None = None) -> dict | None:
    """Call the repository API, treating only GET 404 responses as absent."""
    url = f"{os.environ['GITHUB_API_URL']}/repos/{os.environ['GITHUB_REPOSITORY']}/{path}"
    request = urllib.request.Request(
        url,
        data=json.dumps(data).encode() if data is not None else None,
        headers={
            "Authorization": "Bearer " + os.environ["GH_TOKEN"],
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if data is None and exc.code == 404:
            return None
        raise


def release_notes(root: Path) -> str:
    """Validate package/runtime versions without importing package code."""
    import tomllib

    package = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    runtime = ast.parse((root / "src/litmusai/__init__.py").read_text(encoding="utf-8"))
    versions = [
        ast.literal_eval(node.value)
        for node in runtime.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__version__"
                for target in node.targets)
    ]
    if package["project"]["version"] != VERSION or versions != [VERSION]:
        raise ValueError("Package and runtime versions must both be 1.0.0")

    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    sections = re.split(r"^## ", changelog, flags=re.MULTILINE)
    notes = [
        section.split("\n", 1)[1].strip()
        for section in sections[1:]
        if re.match(r"\[1\.0\.0\] - \d{4}-\d{2}-\d{2}\n", section)
    ]
    if len(notes) != 1 or not notes[0]:
        raise ValueError("Expected one nonempty, dated 1.0.0 changelog section")
    return notes[0]


def ensure_release(sha: str, notes: str) -> None:
    """Create missing resources and verify existing lightweight or annotated tags."""
    ref = api(f"git/ref/tags/{TAG}")
    if ref is None:
        api("git/refs", {"ref": f"refs/tags/{TAG}", "sha": sha})
        ref = api(f"git/ref/tags/{TAG}")
    if ref is None:
        raise ValueError("Release tag was not created")
    target = ref["object"]
    while target["type"] == "tag":
        tag = api(f"git/tags/{target['sha']}")
        if tag is None:
            raise ValueError("Annotated tag target is missing")
        target = tag["object"]
    if target["type"] != "commit" or target["sha"] != sha:
        raise ValueError(
            f"Existing {TAG} tag does not target the tested merge; refusing to move it"
        )

    release = api(f"releases/tags/{TAG}")
    if release is None:
        api("releases", {
            "tag_name": TAG,
            "target_commitish": sha,
            "name": TAG,
            "body": notes,
            "draft": False,
            "prerelease": False,
        })
    elif release["draft"] or release["prerelease"]:
        raise ValueError(f"Existing {TAG} release must be a published stable release")


def main() -> None:
    sha = os.environ["RELEASE_SHA"]
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Expected a full tested commit SHA")
    pr = api("pulls/115")
    if not (
        pr
        and pr["merged"]
        and pr["base"]["ref"] == "main"
        and pr["base"]["repo"]["full_name"] == os.environ["GITHUB_REPOSITORY"]
        and pr["merge_commit_sha"] == sha
    ):
        print("Not PR #115's merged main commit; no automatic release.")
        return
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != sha:
        raise ValueError("Checkout does not match the CI-tested merge")
    notes = release_notes(Path.cwd())
    ensure_release(sha, notes)
    # GITHUB_TOKEN-created release events do not start another workflow.
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        output.write(f"sha={sha}\n")


if __name__ == "__main__":
    main()
