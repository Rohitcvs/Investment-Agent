"""
Production helpers: retries with backoff, and a small time-based cache.

- Retries: data sources like Yahoo Finance fail now and then. Retrying with a
  growing wait (backoff) turns most of those blips into successes.
- Cache: the research and technical agents both need the same price history.
  Caching it means one download instead of two, and repeat runs within the
  cache window cost nothing.
"""
import functools
import time
from typing import Callable

from observability import log_event


def with_retry(attempts: int = 3, base_delay: float = 0.5, sleep: Callable[[float], None] = time.sleep):
    """Retry a function on any exception, waiting base_delay, 2x, 4x... between tries."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    log_event("retry", function=fn.__name__, attempt=attempt, error=repr(exc))
                    if attempt == attempts:
                        raise
                    sleep(base_delay * 2 ** (attempt - 1))
        return wrapper

    return decorator


class TTLCache:
    """A dictionary whose entries expire after ttl_seconds."""

    def __init__(self, ttl_seconds: int, clock: Callable[[], float] = time.monotonic):
        self.ttl = ttl_seconds
        self.clock = clock
        self._data: dict = {}
        self.hits = 0
        self.misses = 0

    def get_or_compute(self, key, compute: Callable):
        now = self.clock()
        if key in self._data:
            value, stored_at = self._data[key]
            if now - stored_at < self.ttl:
                self.hits += 1
                return value
        self.misses += 1
        value = compute()
        self._data[key] = (value, now)
        return value

    def clear(self):
        self._data.clear()
