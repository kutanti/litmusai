"""Global configuration for LitmusAI.

Set default API credentials and base URLs once, used by all
assertions and agents automatically.

Usage::

    import litmusai

    # Set once at startup
    litmusai.configure(
        api_key="sk-...",
        base_url="https://api.openai.com/v1",
    )

    # All assertions pick up defaults automatically
    from litmusai.assertions import Semantic, LLMGrade
    sem = Semantic(threshold=0.85)  # uses global api_key
    judge = LLMGrade(rubric="...")  # uses global api_key

    # Or override per-instance
    sem = Semantic(api_key="sk-other", threshold=0.85)

Azure::

    litmusai.configure(
        api_key="your-azure-key",
        base_url="https://your-resource.openai.azure.com",
        auth_style="azure",  # uses api-key header instead of Bearer
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _GlobalConfig:
    """Internal global configuration store."""

    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    auth_style: str = "bearer"  # "bearer" or "azure"
    model: str = ""
    embedding_model: str = ""
    extra_headers: dict[str, str] = field(default_factory=dict)
    _configured: bool = False

    def get_auth_headers(self) -> dict[str, str]:
        """Build auth headers based on auth_style."""
        headers: dict[str, str] = {}
        if self.api_key:
            if self.auth_style == "azure":
                headers["api-key"] = self.api_key
            else:
                headers["Authorization"] = f"Bearer {self.api_key}"
        if self.extra_headers:
            headers.update(self.extra_headers)
        return headers

    @property
    def is_configured(self) -> bool:
        return self._configured


# Singleton
_config = _GlobalConfig()


def configure(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    auth_style: str | None = None,
    model: str | None = None,
    embedding_model: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> None:
    """Set global defaults for LitmusAI.

    Values set here are used by :class:`~litmusai.assertions.Semantic`,
    :class:`~litmusai.assertions.LLMGrade`, and agent factories when
    no per-instance override is provided.

    Args:
        api_key: Default API key for all assertions and agents.
        base_url: Default API base URL.
        auth_style: Authentication style — ``"bearer"`` (default)
            or ``"azure"`` (uses ``api-key`` header).
        model: Default model for LLM-based assertions.
        embedding_model: Default model for embedding assertions.
        extra_headers: Additional headers applied to all API calls.

    Example::

        import litmusai
        litmusai.configure(
            api_key="sk-...",
            base_url="https://api.openai.com/v1",
        )

        # Azure:
        litmusai.configure(
            api_key="your-azure-key",
            base_url="https://myresource.openai.azure.com",
            auth_style="azure",
        )
    """
    if api_key is not None:
        _config.api_key = api_key
    if base_url is not None:
        _config.base_url = base_url.rstrip("/")
    if auth_style is not None:
        if auth_style not in ("bearer", "azure"):
            msg = f"auth_style must be 'bearer' or 'azure', got '{auth_style}'"
            raise ValueError(msg)
        _config.auth_style = auth_style
    if model is not None:
        _config.model = model
    if embedding_model is not None:
        _config.embedding_model = embedding_model
    if extra_headers is not None:
        _config.extra_headers = extra_headers
    _config._configured = True


def get_config() -> _GlobalConfig:
    """Return the current global configuration (read-only)."""
    return _config


def reset_config() -> None:
    """Reset global configuration to defaults (for testing)."""
    global _config
    _config = _GlobalConfig()
