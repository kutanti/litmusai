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
