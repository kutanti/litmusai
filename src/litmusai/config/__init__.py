"""Configuration file support for LitmusAI.

Reads `.litmus/config.yaml` and provides defaults for CLI commands.

Config file format::

    # .litmus/config.yaml
    version: 1

    defaults:
      concurrency: 5
      timeout: 60
      verbose: true
      log_dir: .litmus/logs
      threshold: 0.8
      budget: 5.0
      runs: 1

    safety:
      level: standard
      fail_on_unsafe: false

    pricing:
      gpt-4o:
        input: 2.50
        output: 10.00
      claude-sonnet-4.6:
        input: 3.00
        output: 15.00
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATHS = [
    Path(".litmus/config.yaml"),
    Path(".litmus/config.yml"),
    Path("litmus.yaml"),
    Path("litmus.yml"),
]


def find_config(start_dir: Path | None = None) -> Path | None:
    """Find a config file by walking up from start_dir.

    Searches for ``.litmus/config.yaml``, ``.litmus/config.yml``,
    ``litmus.yaml``, or ``litmus.yml`` in the given directory
    and its parents.

    Returns:
        Path to config file, or None if not found.
    """
    current = (start_dir or Path.cwd()).resolve()

    for _ in range(20):  # max depth to prevent infinite loop
        for candidate in _DEFAULT_CONFIG_PATHS:
            path = current / candidate
            if path.exists():
                return path

        parent = current.parent
        if parent == current:
            break
        current = parent

    return None


def load_config(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Load configuration from a YAML file.

    If no path is given, searches for a config file using
    :func:`find_config`.

    Returns:
        Configuration dict. Empty dict if no config found.
    """
    if path is None:
        found = find_config()
        if found is None:
            return {}
        path = found
    else:
        path = Path(path)

    if not path.exists():
        return {}

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    return data


def get_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Extract default settings from config."""
    result: dict[str, Any] = config.get("defaults", {})
    return result


def get_safety_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract safety scanner settings from config."""
    result: dict[str, Any] = config.get("safety", {})
    return result


def get_pricing(config: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Extract model pricing from config."""
    result: dict[str, dict[str, float]] = config.get("pricing", {})
    return result


def merge_cli_args(
    config: dict[str, Any],
    *,
    concurrency: int | None = None,
    threshold: float | None = None,
    budget: float | None = None,
    runs: int | None = None,
    log_dir: str | None = None,
    verbose: bool | None = None,
) -> dict[str, Any]:
    """Merge CLI arguments with config defaults.

    CLI args take precedence over config file values.

    Returns:
        Merged settings dict.
    """
    defaults = get_defaults(config)

    return {
        "concurrency": (
            concurrency if concurrency is not None
            else defaults.get("concurrency", 5)
        ),
        "threshold": (
            threshold if threshold is not None
            else defaults.get("threshold")
        ),
        "budget": (
            budget if budget is not None
            else defaults.get("budget")
        ),
        "runs": (
            runs if runs is not None
            else defaults.get("runs", 1)
        ),
        "log_dir": (
            log_dir if log_dir is not None
            else defaults.get("log_dir")
        ),
        "verbose": (
            verbose if verbose is not None
            else defaults.get("verbose", True)
        ),
        "timeout": defaults.get("timeout", 60),
    }
