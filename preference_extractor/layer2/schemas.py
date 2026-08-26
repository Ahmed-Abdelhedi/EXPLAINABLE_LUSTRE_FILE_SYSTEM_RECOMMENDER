from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .labels import (
    PreferenceDimension,
    PreferenceLevel,
    ResolutionSource,
    ResolutionStatus,
)


@dataclass(frozen=True)
class PreferenceRelation:
    higher: PreferenceDimension
    lower: PreferenceDimension
    evidence: str
    relation_type: str = "MORE_IMPORTANT_THAN"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "higher": self.higher.value,
            "lower": self.lower.value,
            "relation_type": self.relation_type,
            "evidence": self.evidence,
        }


@dataclass
class DimensionPreferenceResult:
    dimension: PreferenceDimension
    status: ResolutionStatus
    source: ResolutionSource
    level: Optional[PreferenceLevel] = None
    presence_probability: Optional[float] = None
    intensity_confidence: Optional[float] = None
    evidence: Optional[str] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "status": self.status.value,
            "source": self.source.value,
            "level": (
                self.level.value
                if self.level is not None
                else None
            ),
            "presence_probability": self.presence_probability,
            "intensity_confidence": self.intensity_confidence,
            "evidence": self.evidence,
            "reason": self.reason,
        }


@dataclass
class PreferenceExtractionResult:
    text: str
    dimensions: Dict[
        PreferenceDimension,
        DimensionPreferenceResult,
    ]
    relations: List[PreferenceRelation] = field(
        default_factory=list
    )
    deterministic_guard_used: bool = False
    deterministic_guard_dimensions: List[
        PreferenceDimension
    ] = field(default_factory=list)
    llm_fallback_used: bool = False
    llm_fallback_dimensions: List[
        PreferenceDimension
    ] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "dimensions": {
                dimension.value: result.to_dict()
                for dimension, result
                in self.dimensions.items()
            },
            "relations": [
                relation.to_dict()
                for relation in self.relations
            ],
            "deterministic_guard_used":
                self.deterministic_guard_used,
            "deterministic_guard_dimensions": [
                dimension.value
                for dimension
                in self.deterministic_guard_dimensions
            ],
            "llm_fallback_used": self.llm_fallback_used,
            "llm_fallback_dimensions": [
                dimension.value
                for dimension
                in self.llm_fallback_dimensions
            ],
            "requirement_fields": self.to_requirement_fields(),
        }

    def to_requirement_fields(
        self,
    ) -> Dict[str, Optional[str]]:
        """
        Absolute preference fields only.

        NO_SIGNAL is explicit and valid.
        RELATIVE_ONLY / unresolved cases return None so later scoring does not
        invent an absolute intensity.
        """
        field_map = {
            PreferenceDimension.COST:
                "cost_priority",
            PreferenceDimension.POWER:
                "power_priority",
            PreferenceDimension.PERFORMANCE:
                "performance_priority",
            PreferenceDimension.RELIABILITY:
                "reliability_priority",
        }

        output: Dict[str, Optional[str]] = {}

        for dimension, field_name in field_map.items():
            result = self.dimensions[dimension]

            if result.status == ResolutionStatus.NO_SIGNAL:
                output[field_name] = "NO_SIGNAL"
            elif (
                result.status == ResolutionStatus.RESOLVED
                and result.level is not None
            ):
                output[field_name] = result.level.value
            else:
                output[field_name] = None

        return output
