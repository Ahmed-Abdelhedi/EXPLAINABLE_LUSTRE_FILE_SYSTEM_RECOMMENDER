from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DIMENSIONS: Tuple[str, ...] = (
    "cost",
    "power",
    "performance",
    "reliability",
)

LEVELS: Tuple[str, ...] = (
    "VERY_LOW",
    "LOW",
    "MEDIUM",
    "HIGH",
    "VERY_HIGH",
)

GUARD_VERSION = (
    "layer2_deterministic_semantic_guard_v1_20260825"
)


def _normalize(
    text: str,
) -> str:
    value = unicodedata.normalize(
        "NFKD",
        text.casefold(),
    )

    value = "".join(
        character
        for character
        in value
        if not unicodedata.combining(
            character
        )
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


ALIASES: Dict[
    str,
    Tuple[str, ...],
] = {
    "cost": (
        "cost",
        "costs",
        "budget",
        "price",
        "expense",
        "expenses",
        "spend",
        "spending",
        "financial efficiency",
        "procurement cost",
        "operating expense",
        "total cost",
        "tco",
        "cout",
        "couts",
        "prix",
        "depense",
        "depenses",
        "economie financiere",
        "maitrise du cout",
        "reduction du budget",
        "limitation des depenses",
        "budget efficiency",
    ),
    "power": (
        "power",
        "energy",
        "electricity",
        "consumption",
        "wattage",
        "cooling and power",
        "energy-efficient",
        "energy efficient",
        "electrical",
        "electrical draw",
        "puissance",
        "energie",
        "electricite",
        "consommation",
        "charge electrique",
        "demande energetique",
        "econome en energie",
    ),
    "performance": (
        "performance",
        "performances",
        "throughput",
        "bandwidth",
        "latency",
        "speed",
        "i/o",
        "io",
        "responsive storage",
        "storage performance",
        "fast i/o",
        "strong throughput",
        "sustained performance",
        "application i/o",
        "debit",
        "bande passante",
        "latence",
        "vitesse",
        "e/s",
        "stockage reactif",
        "performances soutenues",
    ),
    "reliability": (
        "reliability",
        "availability",
        "resilience",
        "robustness",
        "fault tolerance",
        "dependable service",
        "continuity of service",
        "high availability",
        "failure resilience",
        "service reliability",
        "fiabilite",
        "disponibilite",
        "robustesse",
        "tolerance aux pannes",
        "continuite du service",
        "service fiable",
        "haute disponibilite",
    ),
}


# Order matters only when two cues occur at equal distance from the
# same dimension mention. Each pattern maps an explicit preference phrase
# to one exact ordinal level.
INTENSITY_PATTERNS: Tuple[
    Tuple[
        str,
        Tuple[
            re.Pattern[str],
            ...,
        ],
    ],
    ...,
] = (
    (
        "VERY_HIGH",
        tuple(
            re.compile(
                pattern
            )
            for pattern in (
                r"first[- ]order priority",
                r"nothing matters more than",
                r"decisive for us",
                r"decisive criterion",
                r"top priority",
                r"absolute priority",
                r"non[- ]negotiable",
                r"cannot compromise",
                r"priorite de premier ordre",
                r"rien n.est plus important que",
                r"critere decisif",
                r"priorite absolue",
                r"non negociable",
            )
        ),
    ),
    (
        "VERY_LOW",
        tuple(
            re.compile(
                pattern
            )
            for pattern in (
                r"can be largely ignored",
                r"close to irrelevant",
                r"matters almost not at all",
                r"barely matters",
                r"almost irrelevant",
                r"negligible",
                r"peut etre largement ignore",
                r"proche d.etre sans interet",
                r"compte presque pas",
                r"presque sans importance",
            )
        ),
    ),
    (
        "LOW",
        tuple(
            re.compile(
                pattern
            )
            for pattern in (
                r"weakly prioritized",
                r"weakly prioritised",
                r"nice to have rather than central",
                r"desirable but not a priority",
                r"sits near the bottom of our priorities",
                r"secondary concern",
                r"secondary priority",
                r"low priority",
                r"receives? a low priority",
                r"faible priorite",
                r"recoit une faible priorite",
                r"secondaire",
                r"souhaitable sans etre prioritaire",
                r"se situe vers le bas de nos priorites",
            )
        ),
    ),
    (
        "MEDIUM",
        tuple(
            re.compile(
                pattern
            )
            for pattern in (
                r"medium weight",
                r"moderate importance",
                r"meaningful but balanced role",
                r"matters,? though it is not dominant",
                r"should receive medium weight",
                r"importance moyenne",
                r"recevoir une importance moyenne",
                r"role reel mais equilibre",
                r"compte sans etre dominant",
                r"importance moderee",
            )
        ),
    ),
    (
        "HIGH",
        tuple(
            re.compile(
                pattern
            )
            for pattern in (
                r"strong importance",
                r"high level of importance",
                r"high importance",
                r"very important",
                r"should be favored strongly",
                r"should be favoured strongly",
                r"should weigh heavily",
                r"forte importance",
                r"importance elevee",
                r"tres important",
                r"doit peser fortement dans la decision",
                r"accordez une importance elevee",
            )
        ),
    ),
)


COMPARATIVE_PATTERNS: Tuple[
    re.Pattern[str],
    ...,
] = tuple(
    re.compile(
        pattern
    )
    for pattern in (
        r"more important (?:than|que)",
        r"outranks",
        r"takes precedence over",
        r"prioriti[sz]ed over",
        r"prioritize .* over",
        r"prioritise .* over",
        r"prime sur",
        r"prioritaire sur",
        r" ahead of ",
        r" before ",
        r"come before",
        r"preference order",
        r"priority order",
        r"ordre de preference",
        r"when forced to choose",
        r"si un choix est necessaire",
        r"retenez .* avant",
        r"pick .* ahead of",
        r" first,? then ",
        r"d.abord,? (?:then|puis)",
        r" avant .* puis ",
        r"prefer .* over",
        r"privilegier .* plutot que",
        r"privilegier .* over",
    )
)


HARD_NEGATIVE_PATTERNS: Tuple[
    re.Pattern[str],
    ...,
] = tuple(
    re.compile(
        pattern
    )
    for pattern in (
        r"no preference",
        r"no dimension is described as preferred",
        r"no preference dimension is selected",
        r"aucune dimension n.est decrite comme preferee",
        r"sans preference",
        r"api documentation",
        r"documentation d.api",
        r"actual requirement only gives",
        r"exigence reelle donne seulement",
        r"real requirement here is only",
        r"only a capability description",
        r"only a capability",
        r"supplier table",
        r"tableau fournisseur",
    )
)


# Strong separators between independent preference clauses. This keeps
# multi-preference messages dimension-local and lets later corrections win.
CLAUSE_SPLITTER = re.compile(
    r"\s*\|\s*"
    r"|[.;!?]"
    r"|\bmeanwhile\b"
    r"|\ben revanche\b"
    r"|\bhowever\b"
    r"|\bwhereas\b"
    r"|\btandis que\b"
    r"|\bwhile\b"
    r"|\bcorrection\s*:\s*"
    r"|\bfinal choice\s*:\s*"
    r"|\bchoix final\s*:\s*",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class GuardDecision:
    dimension: str
    status: str
    level: Optional[str]
    evidence: Optional[str]
    reason: str

    def to_dict(
        self,
    ) -> Dict[
        str,
        object,
    ]:
        return {
            "dimension":
                self.dimension,
            "status":
                self.status,
            "level":
                self.level,
            "evidence":
                self.evidence,
            "reason":
                self.reason,
        }


@dataclass(frozen=True)
class _Segment:
    original: str
    normalized: str
    order: int


def _segments(
    text: str,
) -> List[
    _Segment
]:
    raw = [
        piece.strip(
            " \t\r\n,:-"
        )
        for piece
        in CLAUSE_SPLITTER.split(
            text
        )
    ]

    output = []

    for order, piece in enumerate(
        raw
    ):
        if not piece:
            continue

        output.append(
            _Segment(
                original=piece,
                normalized=_normalize(
                    piece
                ),
                order=order,
            )
        )

    return output


def _alias_mentions(
    normalized_text: str,
) -> List[
    Tuple[
        str,
        int,
        int,
    ]
]:
    output = []

    for dimension, aliases in ALIASES.items():
        for alias in aliases:
            normalized_alias = _normalize(
                alias
            )

            for match in re.finditer(
                (
                    r"(?<!\w)"
                    + re.escape(
                        normalized_alias
                    )
                    + r"(?!\w)"
                ),
                normalized_text,
            ):
                output.append(
                    (
                        dimension,
                        match.start(),
                        match.end(),
                    )
                )

    output.sort(
        key=lambda item:
            item[
                1
            ]
    )

    return output


def _intensity_hits(
    normalized_text: str,
) -> List[
    Tuple[
        str,
        int,
        int,
        str,
    ]
]:
    output = []

    for level, patterns in INTENSITY_PATTERNS:
        for pattern in patterns:
            for match in pattern.finditer(
                normalized_text
            ):
                output.append(
                    (
                        level,
                        match.start(),
                        match.end(),
                        match.group(
                            0
                        ),
                    )
                )

    output.sort(
        key=lambda item:
            item[
                1
            ]
    )

    return output


def _dimension_present(
    normalized_text: str,
    dimension: str,
) -> bool:
    return any(
        mentioned_dimension
        == dimension
        for (
            mentioned_dimension,
            _,
            _,
        ) in _alias_mentions(
            normalized_text
        )
    )


def _nearest_absolute_for_dimension(
    text: str,
    dimension: str,
) -> Optional[
    GuardDecision
]:
    candidates = []

    for segment in _segments(
        text
    ):
        mentions = [
            item
            for item
            in _alias_mentions(
                segment.normalized
            )
            if item[
                0
            ]
            == dimension
        ]

        if not mentions:
            continue

        hits = _intensity_hits(
            segment.normalized
        )

        if not hits:
            continue

        nearest = None

        for (
            level,
            cue_start,
            cue_end,
            _,
        ) in hits:
            cue_center = (
                cue_start
                + cue_end
            ) / 2.0

            distance = min(
                abs(
                    cue_center
                    - (
                        alias_start
                        + alias_end
                    )
                    / 2.0
                )
                for (
                    _,
                    alias_start,
                    alias_end,
                ) in mentions
            )

            candidate = (
                distance,
                cue_start,
                level,
            )

            if (
                nearest is None
                or candidate
                < nearest
            ):
                nearest = candidate

        assert nearest is not None

        candidates.append(
            (
                segment.order,
                nearest[
                    0
                ],
                nearest[
                    1
                ],
                nearest[
                    2
                ],
                segment.original,
            )
        )

    if not candidates:
        return None

    # If the same dimension is mentioned in an old statement and a later
    # correction/current statement, the latest independent clause wins.
    latest_order = max(
        candidate[
            0
        ]
        for candidate
        in candidates
    )

    latest = [
        candidate
        for candidate
        in candidates
        if candidate[
            0
        ]
        == latest_order
    ]

    selected = min(
        latest,
        key=lambda item: (
            item[
                1
            ],
            item[
                2
            ],
        ),
    )

    return GuardDecision(
        dimension=dimension,
        status="RESOLVED",
        level=selected[
            3
        ],
        evidence=selected[
            4
        ],
        reason=(
            "DETERMINISTIC_ABSOLUTE_INTENSITY"
        ),
    )


def _comparison_for_dimension(
    text: str,
    dimension: str,
) -> Optional[
    GuardDecision
]:
    normalized = _normalize(
        text
    )

    if not _dimension_present(
        normalized,
        dimension,
    ):
        return None

    if not any(
        pattern.search(
            normalized
        )
        for pattern
        in COMPARATIVE_PATTERNS
    ):
        return None

    evidence = None

    # Prefer a comparison-containing clause that mentions the requested
    # dimension. Fall back to full message only if a chain/order spans
    # several syntactic fragments.
    for segment in _segments(
        text
    ):
        if not _dimension_present(
            segment.normalized,
            dimension,
        ):
            continue

        if any(
            pattern.search(
                segment.normalized
            )
            for pattern
            in COMPARATIVE_PATTERNS
        ):
            evidence = (
                segment.original
            )

            break

    if evidence is None:
        evidence = text

    return GuardDecision(
        dimension=dimension,
        status="RELATIVE_ONLY",
        level=None,
        evidence=evidence,
        reason=(
            "DETERMINISTIC_PURE_COMPARISON"
        ),
    )


def _hard_negative(
    text: str,
    dimension: str,
) -> Optional[
    GuardDecision
]:
    normalized = _normalize(
        text
    )

    if not any(
        pattern.search(
            normalized
        )
        for pattern
        in HARD_NEGATIVE_PATTERNS
    ):
        return None

    return GuardDecision(
        dimension=dimension,
        status="NO_SIGNAL",
        level=None,
        evidence=None,
        reason=(
            "DETERMINISTIC_HARD_NEGATIVE"
        ),
    )


class Layer2DeterministicSemanticGuard:
    """
    Conservative resolver for high-confidence semantic cases before the LLM.

    Runtime order:
      1. explicit dimension-local absolute intensity;
      2. pure comparative/order relation;
      3. explicit hard-negative technical/no-preference pattern;
      4. otherwise abstain and leave the dimension for the LLM.

    The guard never uses labels, semantic_family, template_id, or dataset
    metadata. It operates only on the current user text and requested dimension.
    """

    version = GUARD_VERSION

    def resolve_dimension(
        self,
        *,
        text: str,
        dimension: str,
    ) -> Optional[
        GuardDecision
    ]:
        if dimension not in DIMENSIONS:
            raise ValueError(
                f"Unsupported dimension: {dimension}"
            )

        absolute = (
            _nearest_absolute_for_dimension(
                text,
                dimension,
            )
        )

        if absolute is not None:
            return absolute

        comparative = (
            _comparison_for_dimension(
                text,
                dimension,
            )
        )

        if comparative is not None:
            return comparative

        negative = _hard_negative(
            text,
            dimension,
        )

        if negative is not None:
            return negative

        return None

    def resolve_many(
        self,
        *,
        text: str,
        dimensions: Sequence[
            str
        ],
    ) -> Dict[
        str,
        GuardDecision,
    ]:
        output = {}

        for dimension in dimensions:
            decision = (
                self.resolve_dimension(
                    text=text,
                    dimension=dimension,
                )
            )

            if decision is not None:
                output[
                    dimension
                ] = decision

        return output
