"""Evaluation profiles — preset configurations for common use cases.

Profiles are named presets that configure Pipeline settings for
different evaluation scenarios. Use them via CLI or Python:

CLI::

    litmus run -s coding -a my_agent:agent --profile quick
    litmus run -s coding -a my_agent:agent --profile thorough
    litmus run -s coding -a my_agent:agent --profile benchmark
    litmus run -s coding -a my_agent:agent --profile safety
    litmus run -s coding -a my_agent:agent --profile ci

Python::

    from litmusai.profiles import get_profile, list_profiles

    profile = get_profile("thorough")
    pipeline = Pipeline(agent, suite, **profile.to_kwargs())
    result = await pipeline.run()

Custom profiles via YAML::

    # .litmus/profiles/my_profile.yaml
    name: my-custom-profile
    description: Custom eval for production agents
    concurrency: 3
    runs: 5
    safety: true
    safety_depth: thorough
    threshold: 0.9
    report: html
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EvalProfile:
    """A named preset of Pipeline configuration.

    Attributes:
        name: Profile identifier (e.g. ``"quick"``, ``"thorough"``).
        description: Human-readable description.
        concurrency: Max parallel evaluations.
        runs: Number of evaluation runs.
        safety: Whether to run safety scanning.
        safety_depth: Safety scan depth (``"basic"``/``"standard"``/``"thorough"``).
        threshold: Minimum pass rate (0.0–1.0).
        report: Report format (``"html"``/``"junit"``/``"csv"``/``None``).
        verbose: Show progress output.
    """

    name: str
    description: str = ""
    concurrency: int = 5
    runs: int = 1
    safety: bool = False
    safety_depth: str = "standard"
    threshold: float = 0.5
    report: str | None = None
    verbose: bool = True

    def to_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs suitable for :class:`~litmusai.pipeline.Pipeline`.

        Returns:
            Dictionary of Pipeline constructor arguments.
        """
        return {
            "concurrency": self.concurrency,
            "runs": self.runs,
            "safety": self.safety,
            "safety_depth": self.safety_depth,
            "threshold": self.threshold,
            "report": self.report,
            "verbose": self.verbose,
        }


# ─── Built-in profiles ──────────────────────────────────────────

_BUILTIN_PROFILES: dict[str, EvalProfile] = {
    "quick": EvalProfile(
        name="quick",
        description="Fast feedback loop — no safety, no multi-run, high concurrency",
        concurrency=10,
        runs=1,
        safety=False,
        threshold=0.5,
        report=None,
        verbose=True,
    ),
    "thorough": EvalProfile(
        name="thorough",
        description="Full evaluation — safety scan, 3 runs, HTML report",
        concurrency=3,
        runs=3,
        safety=True,
        safety_depth="standard",
        threshold=0.7,
        report="html",
        verbose=True,
    ),
    "benchmark": EvalProfile(
        name="benchmark",
        description="Reproducible benchmarking — 5 runs, strict threshold, full logging",
        concurrency=3,
        runs=5,
        safety=False,
        threshold=0.8,
        report="html",
        verbose=True,
    ),
    "safety": EvalProfile(
        name="safety",
        description="Safety-focused — thorough safety scan, low concurrency",
        concurrency=2,
        runs=1,
        safety=True,
        safety_depth="thorough",
        threshold=0.5,
        report="html",
        verbose=True,
    ),
    "ci": EvalProfile(
        name="ci",
        description="CI/CD pipeline — strict threshold, JUnit output, no verbose",
        concurrency=5,
        runs=1,
        safety=False,
        threshold=0.8,
        report="junit",
        verbose=False,
    ),
}

# Custom profiles loaded from YAML
_custom_profiles: dict[str, EvalProfile] = {}


def get_profile(name: str) -> EvalProfile:
    """Get a profile by name.

    Checks custom profiles first, then built-in profiles.

    Args:
        name: Profile name.

    Returns:
        The :class:`EvalProfile`.

    Raises:
        ValueError: If the profile doesn't exist.
    """
    if name in _custom_profiles:
        return _custom_profiles[name]
    if name in _BUILTIN_PROFILES:
        return _BUILTIN_PROFILES[name]

    available = sorted(set(_BUILTIN_PROFILES) | set(_custom_profiles))
    msg = f"Unknown profile '{name}'. Available: {', '.join(available)}"
    raise ValueError(msg)


def list_profiles() -> list[EvalProfile]:
    """List all available profiles (built-in + custom).

    Returns:
        List of :class:`EvalProfile` objects.
    """
    all_profiles = {**_BUILTIN_PROFILES, **_custom_profiles}
    return sorted(all_profiles.values(), key=lambda p: p.name)


def register_profile(profile: EvalProfile) -> None:
    """Register a custom profile.

    Args:
        profile: The profile to register.
    """
    _custom_profiles[profile.name] = profile


def load_profile_yaml(path: str | Path) -> EvalProfile:
    """Load a profile from a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        The loaded :class:`EvalProfile`.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the YAML is invalid or missing ``name``.
    """
    import yaml

    path = Path(path)
    if not path.exists():
        msg = f"Profile file not found: {path}"
        raise FileNotFoundError(msg)

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "name" not in data:
        msg = f"Profile YAML must have a 'name' field: {path}"
        raise ValueError(msg)

    profile = EvalProfile(
        name=data["name"],
        description=data.get("description", ""),
        concurrency=data.get("concurrency", 5),
        runs=data.get("runs", 1),
        safety=data.get("safety", False),
        safety_depth=data.get("safety_depth", "standard"),
        threshold=data.get("threshold", 0.5),
        report=data.get("report"),
        verbose=data.get("verbose", True),
    )

    _custom_profiles[profile.name] = profile
    return profile


def load_profiles_from_dir(directory: str | Path = ".litmus/profiles") -> int:
    """Load all YAML profiles from a directory.

    Args:
        directory: Path to scan for ``*.yaml`` / ``*.yml`` files.

    Returns:
        Number of profiles loaded.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return 0

    count = 0
    for f in sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml")):
        try:
            load_profile_yaml(f)
            count += 1
        except (ValueError, FileNotFoundError):
            continue
    return count


def clear_custom_profiles() -> None:
    """Clear all custom profiles (for testing)."""
    _custom_profiles.clear()
