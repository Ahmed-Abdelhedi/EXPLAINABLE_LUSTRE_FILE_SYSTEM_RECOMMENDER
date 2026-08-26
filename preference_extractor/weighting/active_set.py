from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from preference_extractor.layer2.labels import (
    PreferenceDimension,
    PreferenceLevel,
    ResolutionStatus,
)
from preference_extractor.layer2.schemas import (
    PreferenceExtractionResult,
    PreferenceRelation,
)


@dataclass(frozen=True)
class ActivePreferenceSet:
    active_dimensions: List[PreferenceDimension]
    absolute_levels: Dict[PreferenceDimension, PreferenceLevel]
    relations: List[PreferenceRelation]
    blocked_dimensions: List[PreferenceDimension] = field(
        default_factory=list
    )
    violations: List[str] = field(default_factory=list)


def build_active_set(
    extraction: PreferenceExtractionResult,
) -> ActivePreferenceSet:
    active: List[PreferenceDimension] = []
    levels: Dict[PreferenceDimension, PreferenceLevel] = {}
    blocked: List[PreferenceDimension] = []
    violations: List[str] = []

    for dimension in PreferenceDimension:
        result = extraction.dimensions[dimension]

        if result.status == ResolutionStatus.NO_SIGNAL:
            continue

        if (
            result.status == ResolutionStatus.RESOLVED
            and result.level is not None
        ):
            active.append(dimension)
            levels[dimension] = result.level
            continue

        if result.status == ResolutionStatus.RELATIVE_ONLY:
            active.append(dimension)
            continue

        if result.status in {
            ResolutionStatus.UNRESOLVED,
            ResolutionStatus.NEEDS_FALLBACK,
        }:
            blocked.append(dimension)
            continue

        violations.append(
            f"UNSUPPORTED_LAYER2_STATUS:{dimension.value}:{result.status.value}"
        )

    active_set = set(active)

    for relation in extraction.relations:
        if relation.higher not in active_set:
            violations.append(
                "RELATION_ENDPOINT_NOT_ACTIVE:"
                f"{relation.higher.value}"
            )

        if relation.lower not in active_set:
            violations.append(
                "RELATION_ENDPOINT_NOT_ACTIVE:"
                f"{relation.lower.value}"
            )

        if relation.higher == relation.lower:
            violations.append(
                "SELF_RELATION:"
                f"{relation.higher.value}"
            )

    return ActivePreferenceSet(
        active_dimensions=active,
        absolute_levels=levels,
        relations=list(extraction.relations),
        blocked_dimensions=blocked,
        violations=violations,
    )
