"""Retry policy shared by Axis provider adapters."""

from asyncio import sleep

from axis_agent.types import JSONValue
from axis_ai.events import ProviderRetryEvent
from axis_ai.provider import CancellationToken

RETRY_BASE_DELAY_SECONDS = 0.25
RETRY_POLL_SECONDS = 0.05


def retry_delay_seconds(attempt: int, *, max_delay_seconds: float) -> float:
    """Return capped exponential backoff for a zero-based failed attempt."""
    if max_delay_seconds <= 0:
        return 0.0
    base_delay = min(RETRY_BASE_DELAY_SECONDS, max_delay_seconds)
    return float(min(max_delay_seconds, base_delay * (2**attempt)))


def make_retry_event(
    *,
    attempt: int,
    max_retries: int,
    delay_seconds: float,
    reason: str,
    data: dict[str, JSONValue] | None = None,
) -> ProviderRetryEvent:
    """Describe the next request attempt after a transient failure."""
    next_attempt = attempt + 2
    max_attempts = max_retries + 1
    delay_suffix = f" in {delay_seconds:g}s" if delay_seconds else ""
    return ProviderRetryEvent(
        attempt=next_attempt,
        max_attempts=max_attempts,
        delay_seconds=delay_seconds,
        message=(
            f"Retrying provider request {next_attempt}/{max_attempts} after {reason}{delay_suffix}."
        ),
        data=data,
    )


async def wait_for_retry(
    delay_seconds: float,
    *,
    signal: CancellationToken | None,
) -> bool:
    """Wait for backoff while polling the provider cancellation token."""
    remaining = delay_seconds
    while remaining > 0:
        if signal is not None and signal.is_cancelled():
            return False
        step = min(RETRY_POLL_SECONDS, remaining)
        await sleep(step)
        remaining -= step
    return signal is None or not signal.is_cancelled()
