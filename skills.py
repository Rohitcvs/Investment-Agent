"""
Versioned skill registry.

A skill is a packaged, reusable capability: a name, a semantic version, a short
description, and the prompt that drives it. The graph never hard-codes prompts.
It asks the registry for a skill, so:

- Anyone can reuse a skill in a new workflow instead of starting from a blank page.
- Changing a prompt means registering a new version, not editing the old one,
  so every run records exactly which versions produced its report.
- A bad release can be rolled back without a code change by pinning the old
  version, e.g. SKILL_VERSIONS="writer=1.0.0".
"""
from dataclasses import dataclass

from settings import settings


@dataclass(frozen=True)
class Skill:
    name: str
    version: str
    description: str
    prompt: str

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}"


_REGISTRY: dict[str, dict[str, Skill]] = {}


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def register(skill: Skill) -> Skill:
    versions = _REGISTRY.setdefault(skill.name, {})
    if skill.version in versions:
        raise ValueError(f"{skill.key} is already registered; publish a new version instead.")
    versions[skill.version] = skill
    return skill


def get_skill(name: str, version: str | None = None, pins: dict[str, str] | None = None) -> Skill:
    """The requested version, else a pinned version, else the latest one."""
    versions = _REGISTRY.get(name)
    if not versions:
        raise KeyError(f"Unknown skill: {name}")
    pins = settings.skill_versions if pins is None else pins
    wanted = version or pins.get(name)
    if wanted:
        if wanted not in versions:
            raise KeyError(f"Unknown version {name}@{wanted}. Available: {sorted(versions)}")
        return versions[wanted]
    return versions[max(versions, key=_version_tuple)]


def list_skills() -> list[dict]:
    return [
        {"name": s.name, "version": s.version, "description": s.description}
        for name in sorted(_REGISTRY)
        for s in sorted(_REGISTRY[name].values(), key=lambda s: _version_tuple(s.version))
    ]


# --- Built-in skills ---------------------------------------------------------

register(Skill(
    name="research",
    version="1.0.0",
    description="Gathers past reports, one year of price data, and recent news for a stock.",
    prompt=(
        "You are a research assistant gathering data on a stock. Use your tools to: "
        "1. Retrieve past reports from memory for long-term context (Memory_Tool, action='retrieve'). "
        "2. Fetch the last year of stock data. "
        "3. Research recent news and market sentiment, focusing strictly on sources from the last 6-12 months. "
        "Then write a consolidated summary of what you found. Only use facts your tools returned."
    ),
))

register(Skill(
    name="technical",
    version="1.0.0",
    description="Calculates and explains RSI, MACD, SMA-20 and Bollinger Bands.",
    prompt=(
        "You are a technical analyst. Use your tool to calculate technical indicators for the stock, "
        "then explain what RSI, MACD, SMA-20 and the Bollinger Bands say about its recent trend and momentum. "
        "Only use the numbers your tool returned."
    ),
))

register(Skill(
    name="writer",
    version="1.0.0",
    description="Writes a Buy/Hold/Sell report from the research and technical data.",
    prompt=(
        "You are a senior financial analyst. Write a clear analysis using ONLY the research and technical "
        "data provided, with a strong emphasis on the last 6-12 months. Synthesize both into a coherent "
        "narrative, highlight recent trends and news, and end with a recommendation: Buy, Hold, or Sell, "
        "with the reasons. Write in clean Markdown."
    ),
))

register(Skill(
    name="writer",
    version="1.1.0",
    description="Adds a short section on fit with the customer's risk tolerance and time horizon.",
    prompt=(
        "You are a senior financial analyst. Write a clear analysis using ONLY the research and technical "
        "data provided, with a strong emphasis on the last 6-12 months. Synthesize both into a coherent "
        "narrative, highlight recent trends and news, and end with a recommendation: Buy, Hold, or Sell, "
        "with the reasons. If customer context is provided, add a short 'Fit for this customer' section "
        "that relates the recommendation to their risk tolerance and time horizon. Never invent customer "
        "details that weren't provided. Write in clean Markdown."
    ),
))

register(Skill(
    name="critic",
    version="1.0.0",
    description="Reviews a draft for recency, grounding and clarity; approves or requests changes.",
    prompt=(
        "You are a meticulous financial editor reviewing a draft report. Check: "
        "1. Recency: does it focus on data and events from the last 6-12 months? "
        "2. Grounding: is every claim supported by the research and technical data? "
        "3. Clarity: is the reasoning clear, with a definite Buy/Hold/Sell recommendation? "
        "If the report needs no changes, reply with only the word APPROVED. "
        "Otherwise, give specific, actionable feedback for the writer."
    ),
))
