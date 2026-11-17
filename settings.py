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

    # Cost estimates in USD per 1M tokens (input, output).
    # These are placeholders: check your provider's current pricing and update them.
    prices: dict = field(default_factory=lambda: {
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
    })


settings = Settings()
