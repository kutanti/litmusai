"""CLI for the optional live runtime service."""

from __future__ import annotations

import json

import click
import httpx

from litmusai.runtime.config import load_config, secret, validate_url


@click.group()
def runtime() -> None:
    """Observe live agent activity and publish threat events."""


@runtime.command("validate")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def validate(config_path: str) -> None:
    """Validate trusted configuration without printing resolved secrets."""
    try:
        config = load_config(config_path)
    except Exception:
        raise click.ClickException(
            "Invalid runtime configuration; check the documented schema."
        ) from None
    click.echo(f"Valid: {len(config.projects)} projects, {len(config.destinations)} destinations")


@runtime.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8765, type=click.IntRange(1, 65535), show_default=True)
def serve(config_path: str, host: str, port: int) -> None:
    """Start a single-instance collector; deploy behind TLS."""
    try:
        import uvicorn

        from litmusai.runtime.service import create_app
    except ImportError:
        raise click.ClickException(
            'Install the server extra: pip install "litmuseval[runtime]"'
        ) from None
    try:
        app = create_app(load_config(config_path))
    except Exception:
        raise click.ClickException(
            "Runtime configuration, secrets, or database are invalid."
        ) from None
    # Avoid leaking user-supplied URL identifiers through access logs.
    uvicorn.run(app, host=host, port=port, access_log=False)


def _query(endpoint: str, api_key_env: str, path: str, local: bool) -> None:
    try:
        endpoint = validate_url(endpoint, local)
        with httpx.Client(timeout=10, follow_redirects=False, trust_env=False) as client:
            response = client.get(
                endpoint + path, headers={"Authorization": "Bearer " + secret(api_key_env)}
            )
            response.raise_for_status()
            click.echo(json.dumps(response.json(), indent=2))
    except Exception:
        raise click.ClickException(
            "Runtime request failed; check credentials and service health."
        ) from None


@runtime.command()
@click.option("--endpoint", required=True)
@click.option("--api-key-env", default="LITMUS_RUNTIME_API_KEY", show_default=True)
@click.option("--allow-local-http", is_flag=True)
def alerts(endpoint: str, api_key_env: str, allow_local_http: bool) -> None:
    """List the authenticated project's latest alerts."""
    _query(endpoint, api_key_env, "/v1/alerts", allow_local_http)


@runtime.command()
@click.option("--endpoint", required=True)
@click.option("--api-key-env", default="LITMUS_RUNTIME_API_KEY", show_default=True)
@click.option("--allow-local-http", is_flag=True)
def status(endpoint: str, api_key_env: str, allow_local_http: bool) -> None:
    """Inspect capture gaps, detector coverage, and destination delivery states."""
    _query(endpoint, api_key_env, "/v1/status", allow_local_http)
