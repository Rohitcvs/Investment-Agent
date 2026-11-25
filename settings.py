"""
Central settings, read from environment variables (or a .env file).

Keeping every tunable value here means behaviour can change per environment
(dev, demo, production) without touching code.
"""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes"}


def _pins(name: str) -> dict:
    """Parse "writer=1.0.0,critic=1.0.0" into {"writer": "1.0.0", "critic": "1.0.0"}."""
    raw = os.getenv(name, "").strip()
    return dict(item.split("=", 1) for item in raw.split(",") if "=" in item) if raw else {}


@dataclass(frozen=True)
class Settings:
    # Optimization: each role can use a different model. The critic only has to
    # judge a draft, so a smaller, cheaper model is usually enough.
    main_model: str = os.getenv("MAIN_MODEL", "gpt-4o")
    critic_model: str = os.getenv("CRITIC_MODEL", "gpt-4o-mini")

    # Guardrails
    max_revisions: int = int(os.getenv("MAX_REVISIONS", "3"))
    agent_step_limit: int = int(os.getenv("AGENT_STEP_LIMIT", "12"))  # caps each sub-agent's tool loop
    require_approval: bool = _bool("REQUIRE_APPROVAL", True)          # human-in-the-loop before saving

    # Resilience and caching
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    retry_attempts: int = int(os.getenv("RETRY_ATTEMPTS", "3"))

    # Files
    memory_file: str = os.getenv("MEMORY_FILE", "report_memory.json")
    audit_file: str = os.getenv("AUDIT_FILE", "audit_log.jsonl")

    # Persistence: checkpoints, run records and customer context survive restarts
    database_file: str = os.getenv("DATABASE_FILE", "agent.db")

    # Async processing: how many analyses can run in the background at once
    worker_threads: int = int(os.getenv("WORKER_THREADS", "2"))

    # Security: API keys mapped to customers, e.g. "key-abc=acme,key-def=globex".
    # Leave both empty to turn auth off for local development.
    api_keys: dict = field(default_factory=lambda: _pins("API_KEYS"))
    admin_api_keys: set = field(default_factory=lambda: {
        k.strip() for k in os.getenv("ADMIN_API_KEYS", "").split(",") if k.strip()})
    # Max analyses each customer can start per minute (0 turns the limit off)
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))

    # Skill versions: pin a version to roll back a bad release, e.g. "writer=1.0.0"
    skill_versions: dict = field(default_factory=lambda: _pins("SKILL_VERSIONS"))

    # Cost estimates in USD per 1M tokens (input, output).
    # These are placeholders: check your provider's current pricing and update them.
    prices: dict = field(default_factory=lambda: {
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
    })


settings = Settings()
