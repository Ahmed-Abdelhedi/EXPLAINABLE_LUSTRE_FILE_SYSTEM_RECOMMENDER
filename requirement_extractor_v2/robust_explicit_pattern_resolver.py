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


# Only high-precision labels are used here.  This is not a replacement for the
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


class RobustExplicitPatternResolver:
    """
    High-precision structural adapter for ExplicitPatternResolver.

    The base resolver uses global nearest-cue distance, which is effective for
    ordinary one-requirement sentences.  In a multi-requirement sentence a cue
    belonging to the next quantity can occasionally be closer than the cue
    introducing the current quantity.  This adapter repairs only cases where
    the quantity's local clause contains exactly one compatible, explicit
    field label.

    Examples repaired deterministically:
        Read target: 70 GB/s et write target: 35 GB/s.
        Taille moyenne 3 GB et taille maximale 90 GB.

    Alternatives are deliberately not split:
        Write throughput could be 20 or 30 GB/s.
    CandidateRelationResolver remains responsible for that relation.
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

        # No cross-quantity stealing is possible with a single quantity.
        if len(items) < 2:
            return base_result

        link_by_quantity = {
            link.quantity_id: link
            for link in base_result.links
        }

        for quantity in items:
            matches = _matching_local_rules(text, quantity)
            if len(matches) != 1:
                continue

            rule = matches[0]
            current = link_by_quantity.get(quantity.id)

            if current is not None and current.field == rule.field and current.role == rule.role:
                continue

            link_by_quantity[quantity.id] = SemanticLink(
                quantity_id=quantity.id,
                field=rule.field,
                role=rule.role,
                evidence=_clause_evidence(text, quantity),
                resolver="explicit_pattern_structural",
            )

        quantity_order = {
            quantity.id: index
            for index, quantity in enumerate(items)
        }
        links = sorted(
            link_by_quantity.values(),
            key=lambda link: quantity_order.get(link.quantity_id, 10**9),
        )
        resolved_ids = {link.quantity_id for link in links}
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
        }
