"""
Security: who is calling, what they may see, and how often they may call.

Authentication: each caller sends an API key in the X-API-Key header. A key
maps to exactly one customer, so the customer ID comes from the key, never
from the request body. That is what keeps one customer from reading or
changing another customer's runs and context (tenant isolation).

Admin keys can see everything, including the audit log and metrics.

If no keys are configured, auth is off (local development). Production would
store hashed keys in a database, or use OAuth/SSO, instead of an env variable.

Rate limiting: a token bucket per customer on the expensive endpoint
(starting an analysis), so one client can't run up the LLM bill or starve
everyone else. Production would keep the buckets in Redis so every server
shares them.
"""
import hmac
import threading
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Principal:
    customer_id: str | None  # None = admin or dev mode: not limited to one customer
    is_admin: bool

    def can_access(self, customer_id: str | None) -> bool:
        return self.is_admin or self.customer_id == customer_id


DEV_PRINCIPAL = Principal(customer_id=None, is_admin=True)


class ApiKeyAuth:
    def __init__(self, customer_keys: dict[str, str] | None = None, admin_keys: set[str] | None = None):
        self.customer_keys = customer_keys or {}  # api key -> customer id
        self.admin_keys = admin_keys or set()

    @property
    def enabled(self) -> bool:
        return bool(self.customer_keys or self.admin_keys)

    def authenticate(self, api_key: str | None) -> Principal | None:
        """The caller for this key, or None if the key is missing or wrong."""
        if not self.enabled:
            return DEV_PRINCIPAL
        if not api_key:
            return None
        # compare_digest takes the same time whether or not the key matches,
        # so an attacker can't guess a key one character at a time from response times.
        for key in self.admin_keys:
            if hmac.compare_digest(key, api_key):
                return Principal(customer_id=None, is_admin=True)
        for key, customer_id in self.customer_keys.items():
            if hmac.compare_digest(key, api_key):
                return Principal(customer_id=customer_id, is_admin=False)
        return None


class RateLimiter:
    """
    Token bucket: each caller has a bucket of `per_minute` tokens that refills
    steadily. Each request takes one token; an empty bucket means "try later".
    Allows short bursts while capping the average rate.
    """

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic):
        self.capacity = per_minute
        self.refill_per_second = per_minute / 60
        self.clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last update time)
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, float]:
        """(allowed, seconds to wait before retrying). A limit of 0 turns rate limiting off."""
        if self.capacity <= 0:
            return True, 0.0
        with self._lock:
            now = self.clock()
            tokens, last = self._buckets.get(key, (float(self.capacity), now))
            tokens = min(self.capacity, tokens + (now - last) * self.refill_per_second)
            if tokens >= 1:
                self._buckets[key] = (tokens - 1, now)
                return True, 0.0
            self._buckets[key] = (tokens, now)
            return False, round((1 - tokens) / self.refill_per_second, 1)
