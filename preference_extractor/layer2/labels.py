from __future__ import annotations

from enum import Enum


class PreferenceDimension(str, Enum):
    COST = "cost"
    POWER = "power"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"


DIMENSIONS = tuple(PreferenceDimension)


class PreferenceLevel(str, Enum):
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


LEVELS = tuple(PreferenceLevel)

LEVEL_TO_ID = {
    PreferenceLevel.VERY_LOW: 0,
    PreferenceLevel.LOW: 1,
    PreferenceLevel.MEDIUM: 2,
    PreferenceLevel.HIGH: 3,
    PreferenceLevel.VERY_HIGH: 4,
}

ID_TO_LEVEL = {
    value: key
    for key, value in LEVEL_TO_ID.items()
}


class ResolutionStatus(str, Enum):
    RESOLVED = "RESOLVED"
    NO_SIGNAL = "NO_SIGNAL"
    NEEDS_FALLBACK = "NEEDS_FALLBACK"
    RELATIVE_ONLY = "RELATIVE_ONLY"
    UNRESOLVED = "UNRESOLVED"


class ResolutionSource(str, Enum):
    TRANSFORMER = "TRANSFORMER"
    LLM_FALLBACK = "LLM_FALLBACK"
    RELATION_RESOLVER = "RELATION_RESOLVER"
    DETERMINISTIC_GUARD = "DETERMINISTIC_GUARD"
    NONE = "NONE"
