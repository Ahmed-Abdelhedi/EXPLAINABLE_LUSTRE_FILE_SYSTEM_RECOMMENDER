from __future__ import annotations

from typing import Dict, Iterable, List

from .labels import (
    DIMENSIONS,
    PreferenceDimension,
    ResolutionStatus,
)
from .schemas import (
    DimensionPreferenceResult,
    PreferenceRelation,
)


class PreferenceOutputValidator:
    """
    Deterministic safety boundary for Layer 2.
    """

    @staticmethod
    def evidence_is_supported(
        evidence: str | None,
        text: str,
    ) -> bool:
        if evidence is None:
            return True

        evidence = (
            evidence.strip()
        )

        if not evidence:
            return False

        return (
            evidence in text
            or " ".join(
                evidence.split()
            )
            in " ".join(
                text.split()
            )
        )

    def validate_dimension(
        self,
        result: DimensionPreferenceResult,
        text: str,
    ) -> None:
        if result.dimension not in DIMENSIONS:
            raise ValueError(
                f"Unsupported dimension: {result.dimension}"
            )

        if (
            result.status
            == ResolutionStatus.RESOLVED
            and result.level is None
        ):
            raise ValueError(
                "RESOLVED requires an absolute level."
            )

        if (
            result.status
            in {
                ResolutionStatus.NO_SIGNAL,
                ResolutionStatus.NEEDS_FALLBACK,
                ResolutionStatus.RELATIVE_ONLY,
                ResolutionStatus.UNRESOLVED,
            }
            and result.level is not None
        ):
            raise ValueError(
                f"{result.status.value} cannot carry an absolute level."
            )

        if not self.evidence_is_supported(
            result.evidence,
            text,
        ):
            raise ValueError(
                "Evidence is not supported by the current user message."
            )

    def validate_relations(
        self,
        relations: Iterable[
            PreferenceRelation
        ],
        text: str,
    ) -> None:
        for relation in relations:
            if (
                relation.higher
                == relation.lower
            ):
                raise ValueError(
                    "A relation cannot compare a dimension with itself."
                )

            if not self.evidence_is_supported(
                relation.evidence,
                text,
            ):
                raise ValueError(
                    "Relation evidence is unsupported."
                )

    def validate_all(
        self,
        dimensions: Dict[
            PreferenceDimension,
            DimensionPreferenceResult,
        ],
        relations: List[
            PreferenceRelation
        ],
        text: str,
    ) -> None:
        if set(
            dimensions
        ) != set(
            DIMENSIONS
        ):
            raise ValueError(
                "Layer 2 must return exactly four dimensions."
            )

        for result in dimensions.values():
            self.validate_dimension(
                result,
                text,
            )

        self.validate_relations(
            relations,
            text,
        )
