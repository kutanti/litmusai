"""Tests for config file support."""



from litmusai.config import (
    find_config,
    get_defaults,
    get_pricing,
    get_safety_config,
    load_config,
    merge_cli_args,
)


class TestFindConfig:
    def test_find_in_current_dir(self, tmp_path):
        config_dir = tmp_path / ".litmus"
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        config_file.write_text("version: 1\n")

        result = find_config(tmp_path)
        assert result == config_file

    def test_find_litmus_yaml(self, tmp_path):
        config_file = tmp_path / "litmus.yaml"
        config_file.write_text("version: 1\n")

        result = find_config(tmp_path)
        assert result == config_file

    def test_find_in_parent(self, tmp_path):
        config_dir = tmp_path / ".litmus"
        config_dir.mkdir()
        (config_dir / "config.yaml").write_text("version: 1\n")

        child = tmp_path / "sub" / "dir"
        child.mkdir(parents=True)

        result = find_config(child)
        assert result is not None
        assert result.name == "config.yaml"

    def test_not_found(self, tmp_path):
        child = tmp_path / "empty"
        child.mkdir()
        result = find_config(child)
        assert result is None

    def test_yml_extension(self, tmp_path):
        config_dir = tmp_path / ".litmus"
        config_dir.mkdir()
        (config_dir / "config.yml").write_text("version: 1\n")

        result = find_config(tmp_path)
        assert result is not None


class TestLoadConfig:
    def test_load_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "version: 1\n"
            "defaults:\n"
            "  concurrency: 10\n"
            "  threshold: 0.9\n"
        )
        config = load_config(config_file)
        assert config["version"] == 1
        assert config["defaults"]["concurrency"] == 10

    def test_load_nonexistent(self):
        config = load_config("/nonexistent/config.yaml")
        assert config == {}

    def test_load_empty(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        config = load_config(config_file)
        assert config == {}

    def test_load_auto_find(self, tmp_path, monkeypatch):
        config_dir = tmp_path / ".litmus"
        config_dir.mkdir()
        (config_dir / "config.yaml").write_text(
            "version: 1\ndefaults:\n  runs: 3\n"
        )
        monkeypatch.chdir(tmp_path)
        config = load_config()
        assert config.get("defaults", {}).get("runs") == 3


class TestGetDefaults:
    def test_with_defaults(self):
        config = {"defaults": {"concurrency": 10, "runs": 5}}
        defaults = get_defaults(config)
        assert defaults["concurrency"] == 10
        assert defaults["runs"] == 5

    def test_no_defaults(self):
        config = {"version": 1}
        defaults = get_defaults(config)
        assert defaults == {}

    def test_empty_config(self):
        assert get_defaults({}) == {}


class TestGetSafetyConfig:
    def test_with_safety(self):
        config = {"safety": {"level": "thorough"}}
        safety = get_safety_config(config)
        assert safety["level"] == "thorough"

    def test_no_safety(self):
        assert get_safety_config({}) == {}


class TestGetPricing:
    def test_with_pricing(self):
        config = {
            "pricing": {
                "gpt-4o": {"input": 2.50, "output": 10.0},
            },
        }
        pricing = get_pricing(config)
        assert pricing["gpt-4o"]["input"] == 2.50

    def test_no_pricing(self):
        assert get_pricing({}) == {}


class TestMergeCliArgs:
    def test_cli_overrides_config(self):
        config = {"defaults": {"concurrency": 10, "threshold": 0.9}}
        merged = merge_cli_args(config, concurrency=20)
        assert merged["concurrency"] == 20
        assert merged["threshold"] == 0.9

    def test_config_fills_missing(self):
        config = {
            "defaults": {
                "concurrency": 10,
                "threshold": 0.85,
                "budget": 5.0,
                "log_dir": "./logs",
            },
        }
        merged = merge_cli_args(config)
        assert merged["concurrency"] == 10
        assert merged["threshold"] == 0.85
        assert merged["budget"] == 5.0
        assert merged["log_dir"] == "./logs"

    def test_defaults_when_no_config(self):
        merged = merge_cli_args({})
        assert merged["concurrency"] == 5
        assert merged["threshold"] is None
        assert merged["budget"] is None
        assert merged["runs"] == 1
        assert merged["verbose"] is True
        assert merged["timeout"] == 60

    def test_runs_from_config(self):
        config = {"defaults": {"runs": 3}}
        merged = merge_cli_args(config)
        assert merged["runs"] == 3

    def test_runs_cli_overrides(self):
        config = {"defaults": {"runs": 3}}
        merged = merge_cli_args(config, runs=5)
        assert merged["runs"] == 5

    def test_verbose_override(self):
        config = {"defaults": {"verbose": False}}
        merged = merge_cli_args(config, verbose=True)
        assert merged["verbose"] is True

    def test_all_cli_args(self):
        config = {"defaults": {"concurrency": 10}}
        merged = merge_cli_args(
            config,
            concurrency=20,
            threshold=0.95,
            budget=1.0,
            runs=5,
            log_dir="./custom",
            verbose=False,
        )
        assert merged["concurrency"] == 20
        assert merged["threshold"] == 0.95
        assert merged["budget"] == 1.0
        assert merged["runs"] == 5
        assert merged["log_dir"] == "./custom"
        assert merged["verbose"] is False
