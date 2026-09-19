# Contributing to LitmusAI

Bug fixes, test cases, adapters, and documentation changes are welcome. For a larger API change, open an issue with the use case before implementing it.

## Development

1. Fork and clone the repository.
2. Create a virtual environment and install dependencies with `pip install -e ".[dev]"`.
3. Create a branch for the change.
4. Run `pytest`, `ruff check src/ tests/`, and `mypy src/litmusai/ --ignore-missing-imports`.
5. Open a pull request that explains the problem, changed behavior, and validation.

Add regression tests for bug fixes. Public functions need type hints and docstrings. API tests that need provider credentials are skipped when those credentials are absent.

## Writing and commit style

Use plain language in documentation, CLI output, PR titles, and commit messages. Describe what the code does. Avoid decorative emojis, slogans, unsupported benchmark claims, and claims of guaranteed safety or reproducibility.

Use a short imperative subject, such as `Fix safety scores for failed API calls` or `Document cost estimation limits`. Prefixes such as `fix:` and `docs:` are optional. Explain non-obvious behavior and validation in the body. PR titles and commit messages must contain no emojis; CI checks the title and commits introduced by each PR.

To check a branch locally:

```bash
python scripts/check_commit_messages.py --base origin/main --head HEAD --title "Your PR title"
```

The check applies to new commits. Existing commit messages and dates are preserved.

## Releases

Keep the version in `pyproject.toml` and `src/litmusai/__init__.py` identical.
Update the README's pinned installation and GitHub Action tag, and add dated
release notes, migration guidance, and comparison links to `CHANGELOG.md`.
Package versions and result/dataset schema versions are independent.

For 1.1.0, merge release PR #121. After `CI` succeeds for its push to `main`,
`.github/workflows/publish.yml` checks out that exact tested merge commit,
checks the package/runtime versions, and creates `v1.1.0` and its GitHub release
using the dated 1.1.0 section of `CHANGELOG.md`. It then builds, checks, and
publishes to PyPI through trusted publishing in the same workflow; releases
created with `GITHUB_TOKEN` do not trigger another workflow.

The release constants in `scripts/release.py` limit this automation to PR #121
and version 1.1.0. It does not run for
PR CI, failed CI, or later commits. Existing tags are never moved: a tag pointing
elsewhere fails the release. To recover from a partial failure, rerun the
workflow for the tested merge; matching tags and published releases are reused,
and already uploaded PyPI files are skipped. Repository token permissions and
the existing PyPI trusted publisher for `publish.yml` must be configured.

Future releases remain manual: merge and wait for CI, create a new version tag
at the tested commit, then publish a GitHub release with the changelog notes.
The `release: published` trigger and manual `workflow_dispatch` remain available;
when dispatching manually, select the release tag, not a moving branch.
Do not move existing release tags or tag the pre-merge PR branch.
Verify that the release tag, `litmuseval` package version, and `litmus --version`
all identify 1.1.0 before announcing the release.
