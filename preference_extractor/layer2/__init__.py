from typing import TYPE_CHECKING

from .confidence import (
    ConfidencePolicy,
    FrozenCalibratedConfidencePolicy,
)
from .labels import (
    PreferenceDimension,
    PreferenceLevel,
    ResolutionSource,
    ResolutionStatus,
)
from .schemas import (
    DimensionPreferenceResult,
    PreferenceExtractionResult,
    PreferenceRelation,
)

if TYPE_CHECKING:
    from .llm_fallback import PreferenceLLMFallback
    from .model import XLMRPreferenceMultiTaskModel
    from .runtime import Layer2PreferenceExtractor


__all__ = [
    "ConfidencePolicy",
    "FrozenCalibratedConfidencePolicy",
    "DimensionPreferenceResult",
    "Layer2PreferenceExtractor",
    "PreferenceDimension",
    "PreferenceExtractionResult",
    "PreferenceLevel",
    "PreferenceLLMFallback",
    "PreferenceRelation",
    "ResolutionSource",
    "ResolutionStatus",
    "XLMRPreferenceMultiTaskModel",
]


def __getattr__(name):
    if name == "XLMRPreferenceMultiTaskModel":
        from .model import XLMRPreferenceMultiTaskModel
        return XLMRPreferenceMultiTaskModel

    if name == "Layer2PreferenceExtractor":
        from .runtime import Layer2PreferenceExtractor
        return Layer2PreferenceExtractor

    if name == "PreferenceLLMFallback":
        from .llm_fallback import PreferenceLLMFallback
        return PreferenceLLMFallback

    raise AttributeError(name)
