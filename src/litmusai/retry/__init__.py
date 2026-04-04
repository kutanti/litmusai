"""Retry logic for flaky API calls.

Wraps agent calls with configurable retry behavior — exponential
backoff, jitter, and retry-on-specific-errors.

Usage::

    from litmusai.retry import with_retry, RetryConfig

    config = RetryConfig(max_retries=3, backoff_base=1.0)
    result = await with_retry(agent.run, "What is 2+2?", config=config)
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Args:
        max_retries: Maximum number of retry attempts (0 = no retries).
        backoff_base: Base delay in seconds for exponential backoff.
        backoff_max: Maximum delay in seconds.
        jitter: Add random jitter to avoid thundering herd.
        retry_on: Exception types to retry on. Default: all exceptions.
        on_retry: Optional callback ``(attempt, error, delay)`` called
            before each retry.
    """

    max_retries: int = 3
    backoff_base: float = 1.0
    backoff_max: float = 30.0
    jitter: bool = True
    retry_on: tuple[type[BaseException], ...] = (Exception,)
    on_retry: Callable[[int, BaseException, float], None] | None = None


def _compute_delay(config: RetryConfig, attempt: int) -> float:
    """Compute delay for the given attempt number."""
    delay = min(
        config.backoff_base * (2 ** attempt),
        config.backoff_max,
    )
    if config.jitter:
        delay = delay * (0.5 + random.random() * 0.5)  # noqa: S311
    return float(delay)


async def with_retry(
    fn: Callable[..., Any],
    *args: Any,
    config: RetryConfig | None = None,
    **kwargs: Any,
) -> Any:
    """Call an async function with retry logic.

    Args:
        fn: Async callable to retry.
        *args: Positional args for ``fn``.
        config: Retry configuration. Uses defaults if None.
        **kwargs: Keyword args for ``fn``.

    Returns:
        The return value of ``fn``.

    Raises:
        The last exception if all retries are exhausted.
    """
    config = config or RetryConfig()
    last_error: BaseException | None = None

    for attempt in range(config.max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(fn):
                return await fn(*args, **kwargs)
            return fn(*args, **kwargs)
        except config.retry_on as e:
            last_error = e

            if attempt >= config.max_retries:
                break

            delay = _compute_delay(config, attempt)
            logger.warning(
                "Retry %d/%d after %.1fs: %s",
                attempt + 1, config.max_retries, delay, e,
            )

            if config.on_retry:
                config.on_retry(attempt + 1, e, delay)

            await asyncio.sleep(delay)

    raise last_error  # type: ignore[misc]
