from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from .labels import PreferenceDimension
from .schemas import PreferenceRelation


def _normalize(
    text: str,
) -> str:
    value = unicodedata.normalize(
        "NFKD",
        text.casefold(),
    )

    value = "".join(
        character
        for character in value
        if not unicodedata.combining(
            character
        )
    )

    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


ALIASES: Dict[
    PreferenceDimension,
    Tuple[str, ...],
] = {
    PreferenceDimension.COST: (
        "cost",
        "costs",
        "budget",
        "price",
        "expense",
        "expenses",
        "cout",
        "couts",
        "prix",
        "depense",
        "depenses",
    ),
    PreferenceDimension.POWER: (
        "power",
        "energy",
        "electricity",
        "consumption",
        "wattage",
        "energie",
        "electricite",
        "consommation",
        "puissance",
    ),
    PreferenceDimension.PERFORMANCE: (
        "performance",
        "performances",
        "throughput",
        "bandwidth",
        "latency",
        "speed",
        "debit",
        "bande passante",
        "latence",
        "vitesse",
    ),
    PreferenceDimension.RELIABILITY: (
        "reliability",
        "availability",
        "resilience",
        "robustness",
        "fault tolerance",
        "fiabilite",
        "disponibilite",
        "resilience",
        "robustesse",
        "tolerance aux pannes",
    ),
}


ABSOLUTE_INTENSITY_CUES = (
    "absolutely critical",
    "critical",
    "essential",
    "non negotiable",
    "top priority",
    "highest priority",
    "very important",
    "moderately important",
    "secondary concern",
    "secondary priority",
    "slightly important",
    "almost irrelevant",
    "not important",
    "does not matter",
    "not a concern",
    "priorite absolue",
    "critique",
    "essentiel",
    "essentielle",
    "non negociable",
    "priorite maximale",
    "tres important",
    "tres importante",
    "importance moyenne",
    "secondaire",
    "peu important",
    "peu importante",
    "presque sans importance",
    "pas important",
    "pas importante",
    "ne compte pas",
)


@dataclass(frozen=True)
class ComparativeAnalysis:
    relations: List[PreferenceRelation]
    relative_only_dimensions: Set[
        PreferenceDimension
    ]


class ComparativeRelationResolver:
    """
    Preserve relative order without inventing absolute HIGH/LOW levels.
    """

    _CONNECTORS = (
        r"more important than",
        r"matters more than",
        r"more valuable than",
        r"should come before",
        r"takes precedence over",
        r"preferred over",
        r"prioritized over",
        r"prioritised over",
        r"plus importante? que",
        r"compte plus que",
        r"doit passer avant",
        r"prime sur",
        r"prioritaire sur",
    )

    def __init__(self) -> None:
        alias_pairs = []

        for dimension, aliases in ALIASES.items():
            for alias in aliases:
                alias_pairs.append(
                    (
                        dimension,
                        re.escape(
                            _normalize(
                                alias
                            )
                        ),
                    )
                )

        self._alias_pairs = tuple(
            alias_pairs
        )

    def _mentions(
        self,
        normalized_text: str,
    ) -> List[
        Tuple[
            PreferenceDimension,
            int,
            int,
        ]
    ]:
        mentions = []

        for dimension, alias_pattern in (
            self._alias_pairs
        ):
            for match in re.finditer(
                rf"\b{alias_pattern}\b",
                normalized_text,
            ):
                mentions.append(
                    (
                        dimension,
                        match.start(),
                        match.end(),
                    )
                )

        mentions.sort(
            key=lambda item:
                item[1]
        )

        return mentions

    @staticmethod
    def _has_absolute_cue(
        normalized_text: str,
    ) -> bool:
        return any(
            cue
            in normalized_text
            for cue in ABSOLUTE_INTENSITY_CUES
        )

    def resolve(
        self,
        text: str,
    ) -> ComparativeAnalysis:
        normalized = _normalize(
            text
        )

        mentions = self._mentions(
            normalized
        )

        relations: List[
            PreferenceRelation
        ] = []

        seen = set()

        for left_index in range(
            len(
                mentions
            )
        ):
            left_dimension, _, left_end = (
                mentions[
                    left_index
                ]
            )

            for right_index in range(
                left_index + 1,
                len(
                    mentions
                ),
            ):
                (
                    right_dimension,
                    right_start,
                    _,
                ) = mentions[
                    right_index
                ]

                if (
                    left_dimension
                    == right_dimension
                ):
                    continue

                between = normalized[
                    left_end:right_start
                ]

                if not any(
                    re.search(
                        connector,
                        between,
                    )
                    for connector
                    in self._CONNECTORS
                ):
                    continue

                key = (
                    left_dimension,
                    right_dimension,
                )

                if key in seen:
                    continue

                seen.add(
                    key
                )

                relations.append(
                    PreferenceRelation(
                        higher=left_dimension,
                        lower=right_dimension,
                        evidence=text,
                    )
                )

        relative_only_dimensions: Set[
            PreferenceDimension
        ] = set()

        if (
            relations
            and not self._has_absolute_cue(
                normalized
            )
        ):
            for relation in relations:
                relative_only_dimensions.add(
                    relation.higher
                )
                relative_only_dimensions.add(
                    relation.lower
                )

        return ComparativeAnalysis(
            relations=relations,
            relative_only_dimensions=(
                relative_only_dimensions
            ),
        )
