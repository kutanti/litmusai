"""Tests for global config, Azure support, and error messages."""


import pytest

import litmusai
from litmusai.globals import configure, get_config, reset_config

# ─── Global Config ──────────────────────────────────────────────


class TestGlobalConfig:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_configure_basic(self):
        configure(api_key="sk-test", base_url="https://example.com/v1")
        cfg = get_config()
        assert cfg.api_key == "sk-test"
        assert cfg.base_url == "https://example.com/v1"
        assert cfg.is_configured

    def test_configure_azure_style(self):
        configure(
            api_key="azure-key",
            base_url="https://myresource.openai.azure.com",
            auth_style="azure",
        )
        cfg = get_config()
        assert cfg.auth_style == "azure"
        headers = cfg.get_auth_headers()
        assert "api-key" in headers
        assert headers["api-key"] == "azure-key"
        assert "Authorization" not in headers

    def test_configure_bearer_style(self):
        configure(api_key="sk-test", auth_style="bearer")
        cfg = get_config()
        headers = cfg.get_auth_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer sk-test"

    def test_configure_invalid_auth_style(self):
        with pytest.raises(ValueError, match="auth_style"):
            configure(auth_style="invalid")

    def test_configure_model(self):
        configure(model="gpt-4o", embedding_model="text-embedding-3-large")
        cfg = get_config()
        assert cfg.model == "gpt-4o"
        assert cfg.embedding_model == "text-embedding-3-large"

    def test_configure_extra_headers(self):
        configure(extra_headers={"X-Custom": "value"})
        cfg = get_config()
        headers = cfg.get_auth_headers()
        assert headers["X-Custom"] == "value"

    def test_reset(self):
        configure(api_key="sk-test")
        assert get_config().api_key == "sk-test"
        reset_config()
        assert get_config().api_key == ""
        assert not get_config().is_configured

    def test_configure_strips_trailing_slash(self):
        configure(base_url="https://api.example.com/v1/")
        assert get_config().base_url == "https://api.example.com/v1"

    def test_partial_configure(self):
        configure(api_key="key1")
        configure(model="gpt-4o")  # Should keep api_key
        cfg = get_config()
        assert cfg.api_key == "key1"
        assert cfg.model == "gpt-4o"

    def test_litmusai_configure_exported(self):
        """configure() is accessible from top-level litmusai."""
        assert hasattr(litmusai, "configure")
        assert hasattr(litmusai, "get_config")
        assert hasattr(litmusai, "reset_config")


# ─── Semantic with global config ─────────────────────────────────


class TestSemanticGlobalConfig:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_semantic_uses_global_config(self):
        configure(api_key="sk-global", base_url="https://custom.api.com/v1")
        from litmusai.assertions import Semantic

        sem = Semantic("reference text")
        assert sem.api_key == "sk-global"
        assert sem.base_url == "https://custom.api.com/v1"

    def test_semantic_instance_overrides_global(self):
        configure(api_key="sk-global")
        from litmusai.assertions import Semantic

        sem = Semantic("reference", api_key="sk-instance")
        assert sem.api_key == "sk-instance"

    def test_semantic_azure_auth(self):
        configure(
            api_key="azure-key",
            base_url="https://myresource.openai.azure.com",
            auth_style="azure",
        )
        from litmusai.assertions import Semantic

        sem = Semantic("reference")
        assert sem.auth_style == "azure"
        assert sem.api_key == "azure-key"

    def test_semantic_no_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        reset_config()
        from litmusai.assertions import Semantic

        with pytest.raises(ValueError, match="No API key"):
            Semantic("reference")

    def test_semantic_uses_env_var(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        reset_config()
        from litmusai.assertions import Semantic

        sem = Semantic("reference")
        assert sem.api_key == "sk-from-env"


# ─── LLMGrade with global config ────────────────────────────────


class TestLLMGradeGlobalConfig:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_llmgrade_uses_global_config(self):
        configure(api_key="sk-global", model="gpt-4o")
        from litmusai.assertions import LLMGrade

        judge = LLMGrade("Is this correct?")
        assert judge.api_key == "sk-global"
        assert judge.model == "gpt-4o"

    def test_llmgrade_instance_overrides(self):
        configure(api_key="sk-global")
        from litmusai.assertions import LLMGrade

        judge = LLMGrade("criteria", api_key="sk-instance", model="claude")
        assert judge.api_key == "sk-instance"
        assert judge.model == "claude"

    def test_llmgrade_no_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        reset_config()
        from litmusai.assertions import LLMGrade

        with pytest.raises(ValueError, match="No API key"):
            LLMGrade("criteria")

    def test_llmgrade_azure_auth(self):
        configure(
            api_key="azure-key",
            auth_style="azure",
        )
        from litmusai.assertions import LLMGrade

        judge = LLMGrade("criteria")
        assert judge.auth_style == "azure"


# ─── Agent.from_azure ────────────────────────────────────────────


class TestAgentFromAzure:
    def test_from_azure_basic(self):
        from litmusai import Agent

        agent = Agent.from_azure(
            resource="my-resource",
            deployment="gpt-4o",
            api_key="azure-key-123",
        )
        assert agent.name == "gpt-4o"
        assert agent.model == "gpt-4o"

    def test_from_azure_custom_name(self):
        from litmusai import Agent

        agent = Agent.from_azure(
            resource="my-resource",
            deployment="gpt-4o",
            api_key="azure-key",
            name="my-azure-agent",
        )
        assert agent.name == "my-azure-agent"

    def test_from_azure_env_var(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-azure-key")
        from litmusai import Agent

        agent = Agent.from_azure(
            resource="my-resource",
            deployment="gpt-4o",
        )
        assert agent.name == "gpt-4o"

    def test_from_azure_global_config(self, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        configure(api_key="global-azure-key")
        from litmusai import Agent

        agent = Agent.from_azure(
            resource="my-resource",
            deployment="gpt-4o",
        )
        assert agent.name == "gpt-4o"
        reset_config()

    def test_from_azure_no_key_raises(self, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        reset_config()
        from litmusai import Agent

        with pytest.raises(ValueError, match="No Azure API key"):
            Agent.from_azure(
                resource="my-resource",
                deployment="gpt-4o",
            )


# ─── Error Messages ─────────────────────────────────────────────


class TestErrorMessages:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_semantic_error_mentions_configure(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from litmusai.assertions import Semantic

        with pytest.raises(ValueError) as exc_info:
            Semantic("ref")
        msg = str(exc_info.value)
        assert "litmusai.configure" in msg
        assert "OPENAI_API_KEY" in msg
        assert "api_key" in msg

    def test_llmgrade_error_mentions_configure(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from litmusai.assertions import LLMGrade

        with pytest.raises(ValueError) as exc_info:
            LLMGrade("criteria")
        msg = str(exc_info.value)
        assert "litmusai.configure" in msg

    def test_azure_agent_error_mentions_env_var(self, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        reset_config()
        from litmusai import Agent

        with pytest.raises(ValueError) as exc_info:
            Agent.from_azure(resource="r", deployment="d")
        msg = str(exc_info.value)
        assert "AZURE_OPENAI_API_KEY" in msg
        assert "litmusai.configure" in msg
