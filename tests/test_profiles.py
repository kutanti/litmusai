"""Tests for evaluation profiles."""

from __future__ import annotations

import pytest

from litmusai.profiles import (
    EvalProfile,
    clear_custom_profiles,
    get_profile,
    list_profiles,
    load_profile_yaml,
    load_profiles_from_dir,
    register_profile,
)


class TestBuiltinProfiles:
    def test_quick_profile(self):
        p = get_profile("quick")
        assert p.name == "quick"
        assert p.concurrency == 10
        assert p.runs == 1
        assert p.safety is False
        assert p.report is None

    def test_thorough_profile(self):
        p = get_profile("thorough")
        assert p.name == "thorough"
        assert p.concurrency == 3
        assert p.runs == 3
        assert p.safety is True
        assert p.safety_depth == "standard"
        assert p.threshold == 0.7
        assert p.report == "html"

    def test_benchmark_profile(self):
        p = get_profile("benchmark")
        assert p.runs == 5
        assert p.threshold == 0.8
        assert p.report == "html"

    def test_safety_profile(self):
        p = get_profile("safety")
        assert p.safety is True
        assert p.safety_depth == "thorough"
        assert p.concurrency == 2

    def test_ci_profile(self):
        p = get_profile("ci")
        assert p.threshold == 0.8
        assert p.report == "junit"
        assert p.verbose is False

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            get_profile("nonexistent")

    def test_error_lists_available(self):
        with pytest.raises(ValueError, match="quick"):
            get_profile("nonexistent")

    def test_list_profiles(self):
        profiles = list_profiles()
        names = [p.name for p in profiles]
        assert "quick" in names
        assert "thorough" in names
        assert "benchmark" in names
        assert "safety" in names
        assert "ci" in names
        assert len(profiles) >= 5


class TestToKwargs:
    def test_to_kwargs_contains_all_fields(self):
        p = get_profile("thorough")
        kwargs = p.to_kwargs()
        assert "concurrency" in kwargs
        assert "runs" in kwargs
        assert "safety" in kwargs
        assert "safety_depth" in kwargs
        assert "threshold" in kwargs
        assert "report" in kwargs
        assert "verbose" in kwargs

    def test_to_kwargs_values_match(self):
        p = get_profile("ci")
        kwargs = p.to_kwargs()
        assert kwargs["concurrency"] == 5
        assert kwargs["runs"] == 1
        assert kwargs["threshold"] == 0.8
        assert kwargs["report"] == "junit"
        assert kwargs["verbose"] is False


class TestCustomProfiles:
    def setup_method(self):
        clear_custom_profiles()

    def teardown_method(self):
        clear_custom_profiles()

    def test_register_profile(self):
        p = EvalProfile(name="custom1", description="My custom profile")
        register_profile(p)
        result = get_profile("custom1")
        assert result.name == "custom1"

    def test_custom_overrides_builtin(self):
        """Custom profile with same name as builtin takes precedence."""
        p = EvalProfile(name="quick", concurrency=99)
        register_profile(p)
        result = get_profile("quick")
        assert result.concurrency == 99

    def test_profile_is_frozen(self):
        """Profiles are immutable."""
        p = get_profile("quick")
        with pytest.raises(AttributeError):
            p.concurrency = 1  # type: ignore[misc]

    def test_list_includes_custom(self):
        register_profile(EvalProfile(name="zzz_custom"))
        profiles = list_profiles()
        names = [p.name for p in profiles]
        assert "zzz_custom" in names


class TestYamlProfiles:
    def setup_method(self):
        clear_custom_profiles()

    def teardown_method(self):
        clear_custom_profiles()

    def test_load_profile_yaml(self, tmp_path):
        yaml_content = """
name: my-profile
description: A test profile
concurrency: 8
runs: 3
safety: true
safety_depth: thorough
threshold: 0.9
report: html
"""
        f = tmp_path / "my_profile.yaml"
        f.write_text(yaml_content)

        p = load_profile_yaml(f)
        assert p.name == "my-profile"
        assert p.concurrency == 8
        assert p.runs == 3
        assert p.safety is True
        assert p.safety_depth == "thorough"
        assert p.threshold == 0.9
        assert p.report == "html"

        # Should be registered
        assert get_profile("my-profile").concurrency == 8

    def test_load_yaml_minimal(self, tmp_path):
        f = tmp_path / "minimal.yaml"
        f.write_text("name: minimal\n")

        p = load_profile_yaml(f)
        assert p.name == "minimal"
        assert p.concurrency == 5  # default
        assert p.runs == 1  # default

    def test_load_yaml_missing_name(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text("concurrency: 5\n")

        with pytest.raises(ValueError, match="must have a 'name' field"):
            load_profile_yaml(f)

    def test_load_yaml_invalid_concurrency(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text("name: bad\nconcurrency: 0\n")

        with pytest.raises(ValueError, match="concurrency must be >= 1"):
            load_profile_yaml(f)

    def test_load_yaml_invalid_runs(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text("name: bad\nruns: -1\n")

        with pytest.raises(ValueError, match="runs must be >= 1"):
            load_profile_yaml(f)

    def test_load_yaml_invalid_threshold(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text("name: bad\nthreshold: 1.5\n")

        with pytest.raises(ValueError, match="threshold must be 0.0-1.0"):
            load_profile_yaml(f)

    def test_load_yaml_invalid_safety_depth(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text("name: bad\nsafety_depth: extreme\n")

        with pytest.raises(ValueError, match="safety_depth must be"):
            load_profile_yaml(f)

    def test_load_yaml_invalid_report(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text("name: bad\nreport: pdf\n")

        with pytest.raises(ValueError, match="report must be"):
            load_profile_yaml(f)

    def test_load_yaml_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_profile_yaml("/nonexistent/path.yaml")

    def test_load_profiles_from_dir(self, tmp_path):
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        (profiles_dir / "fast.yaml").write_text(
            "name: fast\nconcurrency: 20\n"
        )
        (profiles_dir / "slow.yaml").write_text(
            "name: slow\nconcurrency: 1\nruns: 10\n"
        )

        count = load_profiles_from_dir(profiles_dir)
        assert count == 2
        assert get_profile("fast").concurrency == 20
        assert get_profile("slow").runs == 10

    def test_load_profiles_from_nonexistent_dir(self):
        count = load_profiles_from_dir("/nonexistent/dir")
        assert count == 0

    def test_load_profiles_skips_invalid(self, tmp_path):
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        (profiles_dir / "good.yaml").write_text("name: good\n")
        (profiles_dir / "bad.yaml").write_text("no_name_field: true\n")

        count = load_profiles_from_dir(profiles_dir)
        assert count == 1


class TestPipelineIntegration:
    """Test profiles work with Pipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_with_profile(self):
        from litmusai import Pipeline
        from litmusai.core.agent import Agent, AgentResponse

        async def fn(task, **kw):
            return AgentResponse(output="42", model="test")

        agent = Agent(fn=fn, name="test", model="test")

        profile = get_profile("quick")
        p = Pipeline(agent, "coding", **profile.to_kwargs())
        assert p.concurrency == 10
        assert p.runs == 1

    @pytest.mark.asyncio
    async def test_profile_kwargs_override(self):
        from litmusai import Pipeline
        from litmusai.core.agent import Agent, AgentResponse

        async def fn(task, **kw):
            return AgentResponse(output="42", model="test")

        agent = Agent(fn=fn, name="test", model="test")

        profile = get_profile("thorough")
        kwargs = profile.to_kwargs()
        kwargs["runs"] = 1  # override multi-run
        p = Pipeline(agent, "coding", **kwargs)
        assert p.runs == 1  # overridden
        assert p.safety is True  # from profile


class TestImports:
    def test_top_level_imports(self):
        import litmusai
        assert hasattr(litmusai, "EvalProfile")
        assert hasattr(litmusai, "get_profile")
        assert hasattr(litmusai, "list_profiles")
        assert hasattr(litmusai, "register_profile")


class TestCLIProfiles:
    def test_profiles_command(self):
        from click.testing import CliRunner

        from litmusai.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["profiles"])
        assert result.exit_code == 0
        assert "quick" in result.output
        assert "thorough" in result.output
        assert "benchmark" in result.output
        assert "safety" in result.output
        assert "ci" in result.output
