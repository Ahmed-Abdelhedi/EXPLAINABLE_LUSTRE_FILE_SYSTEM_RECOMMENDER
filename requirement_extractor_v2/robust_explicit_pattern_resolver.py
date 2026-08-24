from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from .explicit_pattern_resolver import (
    ExplicitPatternResolver,
    ExplicitResolutionResult,
)
from .models import (
    ParamName,
    Quantity,
    QuantityDimension,
    SemanticLink,
    SemanticRole,
)


@dataclass(frozen=True)
class _LocalStructuralRule:
    field: ParamName
    role: SemanticRole
    dimensions: frozenset[QuantityDimension]
    cues: Tuple[re.Pattern[str], ...]


def _compile(*patterns: str) -> Tuple[re.Pattern[str], ...]:
    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)


# Only high-precision labels are used here. This is not a replacement for the
# Semantic Linker; it merely prevents one quantity from stealing the explicit
# label of a neighboring quantity in multi-requirement messages.
_LOCAL_RULES: Tuple[_LocalStructuralRule, ...] = (
    _LocalStructuralRule(
        field=ParamName.target_read_gbps,
        role=SemanticRole.TARGET,
        dimensions=frozenset({QuantityDimension.THROUGHPUT}),
        cues=_compile(
            r"\bread\s+target\b",
            r"\btarget\s+read\b",
            r"\bread\s+throughput\b",
            r"\bread\s+bandwidth\b",
            r"\bd[ée]bit\s+de\s+lecture\b",
            r"\blecture\b",
        ),
    ),
    _LocalStructuralRule(
        field=ParamName.target_write_gbps,
        role=SemanticRole.TARGET,
        dimensions=frozenset({QuantityDimension.THROUGHPUT}),
        cues=_compile(
            r"\bwrite\s+target\b",
            r"\btarget\s+write\b",
            r"\bwrite\s+throughput\b",
            r"\bwrite\s+bandwidth\b",
            r"\bd[ée]bit\s+d['’]?[ée]criture\b",
            r"\b[ée]criture\b",
        ),
    ),
    _LocalStructuralRule(
        field=ParamName.average_file_size_gb,
        role=SemanticRole.AVERAGE_VALUE,
        dimensions=frozenset({QuantityDimension.FILE_SIZE}),
        cues=_compile(
            r"\baverage\s+file\s+size\b",
            r"\bavg\.?\s+file\s+size\b",
            r"\bmean\s+file\s+size\b",
            r"\btaille\s+moyenne\b",
        ),
    ),
    _LocalStructuralRule(
        field=ParamName.max_file_size_gb,
        role=SemanticRole.MAXIMUM_LIMIT,
        dimensions=frozenset({QuantityDimension.FILE_SIZE}),
        cues=_compile(
            r"\bmaximum\s+file\s+size\b",
            r"\bmax(?:imum)?\s+file\s+size\b",
            r"\btaille\s+maximale\b",
            r"\btaille\s+maximum\b",
        ),
    ),
)


# Clause separators intentionally exclude ``or/ou`` because alternatives such
# as ``20 or 30 GB/s`` belong to one field-level interpretation and must be
# handled later by CandidateRelationResolver.
_CLAUSE_SEPARATOR_RE = re.compile(
    r"(?:\b(?:and|et)\b|[;,])",
    re.IGNORECASE,
)


# =====================================================================
# STEP 2.1 — PLANNING HORIZON
# =====================================================================
#
# planning_horizon_years is intentionally NOT added to the XLM-R FIELD_LABELS
# in Step 2.1. Adding a new Transformer class would invalidate the output head
# of the already trained model. Instead, high-precision planning-horizon forms
# are resolved deterministically here and still pass through the relation-aware
# verifier.
#
# A quantity is considered a planning horizon only when:
#   - it is explicitly attached to a year marker; AND
#   - the same local clause contains a planning-horizon cue,
#     OR the same sentence expresses annual growth over/for/sur that duration.
#
# This intentionally rejects:
#   "We have 3 years of logs."
#   "The project started 3 years ago."
#
_YEAR_MARKER_RE = re.compile(
    r"(?:"
    r"years?|yrs?|"
    r"ans?|"
    r"ann[ée]es?"
    r")\b",
    re.IGNORECASE,
)

_DIRECT_HORIZON_CUES = _compile(
    r"\bplanning\s+horizon\b",
    r"\bplanning\s+period\b",
    r"\bprojection\s+horizon\b",
    r"\bdesign\s+horizon\b",
    r"\bcapacity\s+planning\s+horizon\b",
    r"\bhorizon\s+de\s+planification\b",
    r"\bhorizon\s+de\s+projection\b",
    r"\bp[ée]riode\s+de\s+planification\b",
    r"\bplan\s+for\b",
    r"\bplanning\s+for\b",
    r"\bplanifi(?:er|cation|ons|ez|e|ent)\b.{0,24}\b(?:sur|pour)\b",
    r"\bpr[ée]voir\b.{0,24}\b(?:sur|pour)\b",
)

_GROWTH_CUES = _compile(
    r"\bannual\s+growth\b",
    r"\byearly\s+growth\b",
    r"\bgrowth\s+per\s+year\b",
    r"\bannual\s+increase\b",
    r"\bcroissance\s+annuelle\b",
    r"\baugmentation\s+annuelle\b",
)

_DURATION_CONNECTOR_RE = re.compile(
    r"\b(?:over|for|sur|pour)\s+(?:the\s+)?(?:next\s+)?$",
    re.IGNORECASE,
)

_POST_HORIZON_CUE_RE = re.compile(
    r"^\s*(?:planning\s+horizon|horizon\s+de\s+planification)\b",
    re.IGNORECASE,
)


def _clause_bounds(text: str, quantity: Quantity) -> Tuple[int, int]:
    left = 0
    right = len(text)

    for match in _CLAUSE_SEPARATOR_RE.finditer(text):
        if match.end() <= quantity.start:
            left = match.end()
            continue

        if match.start() >= quantity.end:
            right = match.start()
            break

    return left, right


def _sentence_bounds(text: str, quantity: Quantity) -> Tuple[int, int]:
    left = max(
        text.rfind(".", 0, quantity.start),
        text.rfind("!", 0, quantity.start),
        text.rfind("?", 0, quantity.start),
        text.rfind("\n", 0, quantity.start),
    )
    left = 0 if left < 0 else left + 1

    candidates = [
        index
        for index in (
            text.find(".", quantity.end),
            text.find("!", quantity.end),
            text.find("?", quantity.end),
            text.find("\n", quantity.end),
        )
        if index >= 0
    ]
    right = min(candidates) if candidates else len(text)
    return left, right


def _matching_local_rules(
    text: str,
    quantity: Quantity,
) -> List[_LocalStructuralRule]:
    start, end = _clause_bounds(text, quantity)
    clause = text[start:end]

    matches: List[_LocalStructuralRule] = []

    for rule in _LOCAL_RULES:
        if quantity.dimension not in rule.dimensions:
            continue

        if any(pattern.search(clause) for pattern in rule.cues):
            matches.append(rule)

    # Deduplicate by field while preserving order.
    unique: List[_LocalStructuralRule] = []
    seen = set()

    for rule in matches:
        if rule.field in seen:
            continue

        seen.add(rule.field)
        unique.append(rule)

    return unique


def _clause_evidence(text: str, quantity: Quantity) -> str:
    start, end = _clause_bounds(text, quantity)
    return text[start:end].strip()


def _year_marker_span(
    text: str,
    quantity: Quantity,
) -> Optional[Tuple[int, int]]:
    """
    Return the exact year-marker span associated with this quantity.

    Supports both scanner behaviours:
    - Quantity.raw == "3" and the source suffix is " years";
    - Quantity.raw == "3 years" when an upstream parser consumed the unit.
    """
    raw_match = _YEAR_MARKER_RE.search(quantity.raw or "")

    if raw_match is not None:
        return (
            quantity.start + raw_match.start(),
            quantity.start + raw_match.end(),
        )

    suffix = text[quantity.end : min(len(text), quantity.end + 24)]
    suffix_match = re.match(
        r"^\s*[- ]?\s*(?P<unit>"
        r"(?:years?|yrs?|ans?|ann[ée]es?)"
        r")\b",
        suffix,
        flags=re.IGNORECASE,
    )

    if suffix_match is None:
        return None

    return (
        quantity.end + suffix_match.start("unit"),
        quantity.end + suffix_match.end("unit"),
    )


def _planning_horizon_link(
    text: str,
    quantity: Quantity,
) -> Optional[SemanticLink]:
    marker = _year_marker_span(text, quantity)

    if marker is None:
        return None

    marker_start, marker_end = marker
    clause_start, clause_end = _clause_bounds(text, quantity)
    clause = text[clause_start:clause_end]

    # 1. Explicit horizon wording in the same clause.
    if any(pattern.search(clause) for pattern in _DIRECT_HORIZON_CUES):
        return SemanticLink(
            quantity_id=quantity.id,
            field=ParamName.planning_horizon_years,
            role=SemanticRole.TARGET,
            evidence=clause.strip(),
            resolver="explicit_planning_horizon",
        )

    # 2. Post-positioned form: "3-year planning horizon".
    after_marker = text[marker_end : min(len(text), marker_end + 64)]
    if _POST_HORIZON_CUE_RE.search(after_marker):
        sentence_start, sentence_end = _sentence_bounds(text, quantity)
        return SemanticLink(
            quantity_id=quantity.id,
            field=ParamName.planning_horizon_years,
            role=SemanticRole.TARGET,
            evidence=text[sentence_start:sentence_end].strip(),
            resolver="explicit_planning_horizon",
        )

    # 3. Coupled growth duration:
    #       "annual growth 20% over 3 years"
    #       "croissance annuelle 20% sur 3 ans"
    sentence_start, sentence_end = _sentence_bounds(text, quantity)
    sentence = text[sentence_start:sentence_end]
    local_prefix = text[max(sentence_start, quantity.start - 32) : quantity.start]

    has_growth = any(pattern.search(sentence) for pattern in _GROWTH_CUES)
    has_duration_connector = _DURATION_CONNECTOR_RE.search(local_prefix) is not None

    if has_growth and has_duration_connector:
        return SemanticLink(
            quantity_id=quantity.id,
            field=ParamName.planning_horizon_years,
            role=SemanticRole.TARGET,
            evidence=sentence.strip(),
            resolver="explicit_planning_horizon",
        )

    return None


class RobustExplicitPatternResolver:
    """
    High-precision structural adapter for ExplicitPatternResolver.

    Step 1 behaviour
    ----------------
    The base resolver uses global nearest-cue distance, which is effective for
    ordinary one-requirement sentences. In a multi-requirement sentence a cue
    belonging to the next quantity can occasionally be closer than the cue
    introducing the current quantity. This adapter repairs only cases where
    the quantity's local clause contains exactly one compatible explicit label.

    Examples repaired deterministically:
        Read target: 70 GB/s et write target: 35 GB/s.
        Taille moyenne 3 GB et taille maximale 90 GB.

    Alternatives are deliberately not split:
        Write throughput could be 20 or 30 GB/s.
    CandidateRelationResolver remains responsible for that relation.

    Step 2.1 behaviour
    ------------------
    ``planning_horizon_years`` is resolved here through explicit, high-
    precision time-horizon wording without changing the Transformer's label
    space.
    """

    def __init__(
        self,
        base_resolver: Optional[ExplicitPatternResolver] = None,
    ) -> None:
        self.base_resolver = base_resolver or ExplicitPatternResolver()

    def resolve(
        self,
        text: str,
        quantities: Iterable[Quantity],
    ) -> ExplicitResolutionResult:
        items = list(quantities)
        base_result = self.base_resolver.resolve(text, items)

        link_by_quantity = {
            link.quantity_id: link
            for link in base_result.links
        }

        # -------------------------------------------------------------
        # Step 2.1: explicit planning horizon.
        # This runs even for a single quantity.
        # -------------------------------------------------------------
        planning_horizon_ids = set()

        for quantity in items:
            horizon_link = _planning_horizon_link(
                text=text,
                quantity=quantity,
            )

            if horizon_link is None:
                continue

            link_by_quantity[quantity.id] = horizon_link
            planning_horizon_ids.add(quantity.id)

        # -------------------------------------------------------------
        # Step 1 local-clause repair only matters when there are at least
        # two quantities. Never overwrite a planning-horizon decision.
        # -------------------------------------------------------------
        if len(items) >= 2:
            for quantity in items:
                if quantity.id in planning_horizon_ids:
                    continue

                matches = _matching_local_rules(
                    text,
                    quantity,
                )

                if len(matches) != 1:
                    continue

                rule = matches[0]
                current = link_by_quantity.get(
                    quantity.id
                )

                if (
                    current is not None
                    and current.field == rule.field
                    and current.role == rule.role
                ):
                    continue

                link_by_quantity[quantity.id] = SemanticLink(
                    quantity_id=quantity.id,
                    field=rule.field,
                    role=rule.role,
                    evidence=_clause_evidence(
                        text,
                        quantity,
                    ),
                    resolver="explicit_pattern_structural",
                )

        quantity_order = {
            quantity.id: index
            for index, quantity in enumerate(items)
        }

        links = sorted(
            link_by_quantity.values(),
            key=lambda link: quantity_order.get(
                link.quantity_id,
                10**9,
            ),
        )

        resolved_ids = {
            link.quantity_id
            for link in links
        }

        unresolved = [
            quantity.id
            for quantity in items
            if quantity.id not in resolved_ids
        ]

        return ExplicitResolutionResult(
            links=links,
            unresolved_quantity_ids=unresolved,
        )

    def info(self) -> dict:
        return {
            "name": "RobustExplicitPatternResolver",
            "base": type(self.base_resolver).__name__,
            "local_clause_binding": True,
            "alternative_connector_split": False,
            "planning_horizon_explicit_rules": True,
            "planning_horizon_transformer_label_added": False,
        }