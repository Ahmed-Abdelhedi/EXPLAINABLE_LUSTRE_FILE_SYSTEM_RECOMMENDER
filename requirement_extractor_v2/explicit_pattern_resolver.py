from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from .models import (
    ParamName,
    Quantity,
    QuantityDimension,
    SemanticLink,
    SemanticRole,
)


# =====================================================================
# RESULT MODEL
# =====================================================================


@dataclass
class ExplicitResolutionResult:
    """
    Result returned by ExplicitPatternResolver.

    links:
        Quantities that were resolved by clear deterministic patterns.

    unresolved_quantity_ids:
        Quantities that were intentionally left for the Semantic Linker
        or, later, the LLM fallback.
    """

    links: List[SemanticLink]
    unresolved_quantity_ids: List[str]

    def to_dict(self) -> dict:
        return {
            "links": [link.to_dict() for link in self.links],
            "unresolved_quantity_ids": list(self.unresolved_quantity_ids),
        }


# =====================================================================
# INTERNAL RULE MODEL
# =====================================================================


@dataclass(frozen=True)
class _ExplicitRule:
    field: ParamName
    role: SemanticRole
    dimensions: frozenset[QuantityDimension]
    cues: Tuple[re.Pattern[str], ...]
    max_distance: int = 48


def _compile(*patterns: str) -> Tuple[re.Pattern[str], ...]:
    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)


# =====================================================================
# EXPLICIT RULES
#
# These rules intentionally cover only clear formulations.
# They are NOT meant to understand every possible sentence.
# Difficult or unusual formulations must remain unresolved and continue
# to the Semantic Linker.
# =====================================================================


_RULES: Tuple[_ExplicitRule, ...] = (
    # -----------------------------------------------------------------
    # Power
    # -----------------------------------------------------------------
    _ExplicitRule(
        field=ParamName.max_power_w,
        role=SemanticRole.MAXIMUM_LIMIT,
        dimensions=frozenset(
            {
                QuantityDimension.POWER,
                QuantityDimension.UNKNOWN,
            }
        ),
        cues=_compile(
            r"\bmaximum\s+power\b",
            r"\bmax(?:imum)?\s+power\b",
            r"\bpower\s+(?:limit|ceiling)\b",
            r"\bpower\s+must\s+not\s+exceed\b",
            r"\bmust\s+not\s+exceed\b",
            r"\banything\s+above\b",
            r"\bpuissance\s+maximale\b",
            r"\blimite\s+de\s+puissance\b",
            r"\bne\s+doit\s+pas\s+d[ée]passer\b",
        ),
    ),
    _ExplicitRule(
        field=ParamName.max_power_w,
        role=SemanticRole.CURRENT_VALUE,
        dimensions=frozenset(
            {
                QuantityDimension.POWER,
                QuantityDimension.UNKNOWN,
            }
        ),
        cues=_compile(
            r"\bcurrent(?:ly)?\s+(?:power\s+)?consum(?:es?|ption)\b",
            r"\bcurrently\s+consumes?\b",
            r"\bconsommation\s+(?:de\s+puissance\s+)?actuelle\b",
            r"\bconsomme\s+actuellement\b",
        ),
    ),
    _ExplicitRule(
        field=ParamName.max_power_w,
        role=SemanticRole.EXPECTED_VALUE,
        dimensions=frozenset(
            {
                QuantityDimension.POWER,
                QuantityDimension.UNKNOWN,
            }
        ),
        cues=_compile(
            r"\bexpected\s+(?:power\s+)?consumption\b",
            r"\bexpected\s+to\s+consume\b",
            r"\bwill\s+consume\b",
            r"\bconsommation\s+(?:de\s+puissance\s+)?pr[ée]vue\b",
            r"\bdevrait\s+consommer\b",
        ),
    ),

    # -----------------------------------------------------------------
    # Budget
    # -----------------------------------------------------------------
    _ExplicitRule(
        field=ParamName.max_budget_usd,
        role=SemanticRole.MAXIMUM_LIMIT,
        dimensions=frozenset(
            {
                QuantityDimension.MONEY,
                QuantityDimension.UNKNOWN,
            }
        ),
        cues=_compile(
            r"\bmaximum\s+budget\b",
            r"\bmax(?:imum)?\s+budget\b",
            r"\bbudget\s+(?:limit|ceiling|maximum|max)\b",
            r"\bbudget\s+must\s+not\s+exceed\b",
            r"\bbudget\s+maximal\b",
            r"\bbudget\s+maximum\b",
            r"\blimite\s+(?:du|de)\s+budget\b",
        ),
    ),
    _ExplicitRule(
        field=ParamName.max_budget_usd,
        role=SemanticRole.CURRENT_VALUE,
        dimensions=frozenset(
            {
                QuantityDimension.MONEY,
                QuantityDimension.UNKNOWN,
            }
        ),
        cues=_compile(
            r"\bcurrently\s+spend(?:s|ing)?\b",
            r"\bcurrent\s+spend(?:ing)?\b",
            r"\bd[ée]pense\s+actuelle\b",
            r"\bd[ée]pensons\s+actuellement\b",
        ),
    ),

    # -----------------------------------------------------------------
    # Requested usable capacity
    #
    # IMPORTANT:
    # There are TWO families of capacity rules:
    #
    # 1. Explicit capacity labels such as "usable capacity" or
    #    "capacité utile". These can safely help even when a quantity
    #    arrives with UNKNOWN dimension.
    #
    # 2. Generic request verbs such as "need", "require" or "besoin de".
    #    These are accepted ONLY when the QuantityScanner has already
    #    identified the quantity as CAPACITY from its unit.
    #
    # This prevents:
    #     "We need 200 clients."
    # from being misclassified as requested capacity.
    # -----------------------------------------------------------------
    _ExplicitRule(
        field=ParamName.requested_usable_capacity_tib,
        role=SemanticRole.TARGET,
        dimensions=frozenset(
            {
                QuantityDimension.CAPACITY,
                QuantityDimension.UNKNOWN,
            }
        ),
        cues=_compile(
            r"\busable\s+capacity\b",
            r"\brequested\s+capacity\b",
            r"\brequired\s+capacity\b",
            r"\bcapacity\s+(?:needed|required|target)\b",
            r"\bcapacit[ée]\s+utile\b",
            r"\bcapacit[ée]\s+demand[ée]e?\b",
            r"\bcapacit[ée]\s+requise\b",
        ),
        max_distance=32,
    ),
    _ExplicitRule(
        field=ParamName.requested_usable_capacity_tib,
        role=SemanticRole.TARGET,
        dimensions=frozenset(
            {
                QuantityDimension.CAPACITY,
            }
        ),
        cues=_compile(
            r"\bneed(?:ed)?\b",
            r"\brequire(?:d)?\b",
            r"\bwe\s+need\b",
            r"\bwe\s+require\b",
            r"\bbesoin\s+de\b",
            r"\bil\s+faut\b",
        ),
        max_distance=24,
    ),
    _ExplicitRule(
        field=ParamName.requested_usable_capacity_tib,
        role=SemanticRole.CURRENT_VALUE,
        dimensions=frozenset(
            {
                QuantityDimension.CAPACITY,
                QuantityDimension.UNKNOWN,
            }
        ),
        cues=_compile(
            r"\bcurrent\s+capacity\b",
            r"\bcurrently\s+have\b",
            r"\bcapacit[ée]\s+actuelle\b",
            r"\bnous\s+avons\s+actuellement\b",
        ),
    ),

    # -----------------------------------------------------------------
    # File sizes
    # -----------------------------------------------------------------
    _ExplicitRule(
        field=ParamName.average_file_size_gb,
        role=SemanticRole.AVERAGE_VALUE,
        dimensions=frozenset(
            {
                QuantityDimension.FILE_SIZE,
                QuantityDimension.UNKNOWN,
            }
        ),
        cues=_compile(
            r"\baverage\s+file\s+size\b",
            r"\bavg\.?\s+file\s+size\b",
            r"\bmean\s+file\s+size\b",
            r"\baverage\s+file\b",
            r"\btaille\s+moyenne\s+(?:(?:du|des?)\s+)?fichiers?\b",
            r"\bfichiers?\s+(?:font|de)\s+.*?\ben\s+moyenne\b",
        ),
    ),
    _ExplicitRule(
        field=ParamName.max_file_size_gb,
        role=SemanticRole.MAXIMUM_LIMIT,
        dimensions=frozenset(
            {
                QuantityDimension.FILE_SIZE,
                QuantityDimension.UNKNOWN,
            }
        ),
        cues=_compile(
            r"\bmaximum\s+file\s+size\b",
            r"\bmax(?:imum)?\s+file\s+size\b",
            r"\bfile\s+size\s+(?:limit|maximum|max)\b",
            r"\btaille\s+maximale\s+(?:(?:du|des?)\s+)?fichiers?\b",
            r"\btaille\s+maximum\s+(?:(?:du|des?)\s+)?fichiers?\b",
        ),
    ),

    # -----------------------------------------------------------------
    # Client count
    # Deliberately covers explicit "clients" formulations only.
    # "200 hosts", "200 machines", etc. are left for Semantic Linker.
    # -----------------------------------------------------------------
    _ExplicitRule(
        field=ParamName.client_count,
        role=SemanticRole.TOTAL_COUNT,
        dimensions=frozenset({QuantityDimension.UNKNOWN}),
        cues=_compile(
            r"\bnumber\s+of\s+clients\b",
            r"\bclient\s+count\b",
            r"\bclients?\b",
            r"\bnombre\s+de\s+clients\b",
        ),
        max_distance=28,
    ),

    # -----------------------------------------------------------------
    # Total file count
    # -----------------------------------------------------------------
    _ExplicitRule(
        field=ParamName.total_file_count,
        role=SemanticRole.TOTAL_COUNT,
        dimensions=frozenset({QuantityDimension.UNKNOWN}),
        cues=_compile(
            r"\btotal\s+(?:number\s+of\s+)?files\b",
            r"\bfile\s+count\b",
            r"\bnumber\s+of\s+files\b",
            r"\bfiles\b",
            r"\bnombre\s+(?:total\s+)?de\s+fichiers\b",
            r"\bfichiers\b",
        ),
        max_distance=28,
    ),

    # -----------------------------------------------------------------
    # Throughput
    # -----------------------------------------------------------------
    _ExplicitRule(
        field=ParamName.target_read_gbps,
        role=SemanticRole.TARGET,
        dimensions=frozenset(
            {
                QuantityDimension.THROUGHPUT,
                QuantityDimension.UNKNOWN,
            }
        ),
        cues=_compile(
            r"\bread\s+throughput\b",
            r"\bread\s+bandwidth\b",
            r"\btarget\s+read\b",
            r"\bread\b",
            r"\bd[ée]bit\s+de\s+lecture\b",
            r"\blecture\b",
        ),
        max_distance=24,
    ),
    _ExplicitRule(
        field=ParamName.target_write_gbps,
        role=SemanticRole.TARGET,
        dimensions=frozenset(
            {
                QuantityDimension.THROUGHPUT,
                QuantityDimension.UNKNOWN,
            }
        ),
        cues=_compile(
            r"\bwrite\s+throughput\b",
            r"\bwrite\s+bandwidth\b",
            r"\btarget\s+write\b",
            r"\bwrite\b",
            r"\bd[ée]bit\s+d['’]?[ée]criture\b",
            r"\b[ée]criture\b",
        ),
        max_distance=24,
    ),

    # -----------------------------------------------------------------
    # Annual growth
    # -----------------------------------------------------------------
    _ExplicitRule(
        field=ParamName.annual_growth_percent,
        role=SemanticRole.GROWTH_RATE,
        dimensions=frozenset(
            {
                QuantityDimension.PERCENT,
                QuantityDimension.UNKNOWN,
            }
        ),
        cues=_compile(
            r"\bannual\s+growth\b",
            r"\byearly\s+growth\b",
            r"\bgrowth\s+per\s+year\b",
            r"\bannual\s+increase\b",
            r"\bcroissance\s+annuelle\b",
            r"\baugmentation\s+annuelle\b",
        ),
        max_distance=36,
    ),
)


_READ_RATIO_CUES = _compile(
    r"\bread\b",
    r"\breads\b",
    r"\blecture\b",
)

_WRITE_RATIO_CUES = _compile(
    r"\bwrite\b",
    r"\bwrites\b",
    r"\b[ée]criture\b",
)


# =====================================================================
# MATCHING HELPERS
# =====================================================================


def _span_distance(
    first_start: int,
    first_end: int,
    second_start: int,
    second_end: int,
) -> int:
    if first_end <= second_start:
        return second_start - first_end

    if second_end <= first_start:
        return first_start - second_end

    return 0


def _nearest_cue_distance(
    text: str,
    quantity: Quantity,
    cues: Sequence[re.Pattern[str]],
) -> Optional[int]:
    distances: List[int] = []

    for cue in cues:
        for match in cue.finditer(text):
            distances.append(
                _span_distance(
                    quantity.start,
                    quantity.end,
                    match.start(),
                    match.end(),
                )
            )

    if not distances:
        return None

    return min(distances)


def _nearest_cue(
    text: str,
    quantity: Quantity,
    cues: Sequence[re.Pattern[str]],
) -> Optional[Tuple[int, str]]:
    best: Optional[Tuple[int, str]] = None

    for cue in cues:
        for match in cue.finditer(text):
            distance = _span_distance(
                quantity.start,
                quantity.end,
                match.start(),
                match.end(),
            )

            candidate = (distance, match.group(0))

            if best is None or candidate[0] < best[0]:
                best = candidate

    return best


def _evidence_window(
    text: str,
    quantity: Quantity,
    radius: int = 72,
) -> str:
    start = max(0, quantity.start - radius)
    end = min(len(text), quantity.end + radius)
    return text[start:end].strip()


# =====================================================================
# PUBLIC RESOLVER
# =====================================================================


class ExplicitPatternResolver:
    """
    Deterministically resolve only clear quantity-to-field relations.

    This component follows the selective-cascade principle:
    - clear explicit structure -> resolve here;
    - unclear/unusual language -> leave unresolved;
    - do not use embeddings;
    - do not use an LLM;
    - do not validate the final business value here.

    Examples:
        "Maximum power: 800 W."
        -> max_power_w + maximum_limit

        "Currently consumes 800 W."
        -> max_power_w + current_value
           (later verifier must NOT accept it as max_power_w)

        "The filesystem will be mounted by 200 hosts."
        -> unresolved here
           (Semantic Linker handles "hosts")
    """

    def resolve(
        self,
        text: str,
        quantities: Iterable[Quantity],
    ) -> ExplicitResolutionResult:
        items = list(quantities)

        links: List[SemanticLink] = []
        resolved_ids: Set[str] = set()

        # -------------------------------------------------------------
        # 1. Special explicit read/write ratio handling
        # -------------------------------------------------------------
        ratio_links = self._resolve_read_write_ratio(text, items)

        for link in ratio_links:
            links.append(link)
            resolved_ids.add(link.quantity_id)

        # -------------------------------------------------------------
        # 2. General one-quantity / one-field explicit rules
        # -------------------------------------------------------------
        for quantity in items:
            if quantity.id in resolved_ids:
                continue

            best_match: Optional[
                Tuple[int, int, _ExplicitRule, str]
            ] = None

            for rule_index, rule in enumerate(_RULES):
                if quantity.dimension not in rule.dimensions:
                    continue

                nearest = _nearest_cue(
                    text,
                    quantity,
                    rule.cues,
                )

                if nearest is None:
                    continue

                distance, cue_text = nearest

                if distance > rule.max_distance:
                    continue

                # Prefer the nearest cue. If tied, preserve rule order.
                candidate = (
                    distance,
                    rule_index,
                    rule,
                    cue_text,
                )

                if best_match is None or candidate[:2] < best_match[:2]:
                    best_match = candidate

            if best_match is None:
                continue

            _, _, selected_rule, cue_text = best_match

            links.append(
                SemanticLink(
                    quantity_id=quantity.id,
                    field=selected_rule.field,
                    role=selected_rule.role,
                    evidence=_evidence_window(text, quantity),
                    resolver="explicit_pattern",
                )
            )
            resolved_ids.add(quantity.id)

        unresolved = [
            quantity.id
            for quantity in items
            if quantity.id not in resolved_ids
        ]

        return ExplicitResolutionResult(
            links=links,
            unresolved_quantity_ids=unresolved,
        )

    def _resolve_read_write_ratio(
        self,
        text: str,
        quantities: Sequence[Quantity],
    ) -> List[SemanticLink]:
        percent_quantities = [
            quantity
            for quantity in quantities
            if quantity.dimension == QuantityDimension.PERCENT
        ]

        if len(percent_quantities) < 2:
            return []

        links: List[SemanticLink] = []
        used: Set[str] = set()

        # For each percentage, determine whether the closest explicit
        # directional cue is read or write.
        for quantity in percent_quantities:
            read_distance = _nearest_cue_distance(
                text,
                quantity,
                _READ_RATIO_CUES,
            )

            write_distance = _nearest_cue_distance(
                text,
                quantity,
                _WRITE_RATIO_CUES,
            )

            if read_distance is None and write_distance is None:
                continue

            if (
                read_distance is not None
                and (
                    write_distance is None
                    or read_distance < write_distance
                )
                and read_distance <= 20
            ):
                links.append(
                    SemanticLink(
                        quantity_id=quantity.id,
                        field=ParamName.read_write_ratio,
                        role=SemanticRole.RATIO_COMPONENT,
                        evidence=_evidence_window(text, quantity),
                        resolver="explicit_pattern",
                    )
                )
                used.add(quantity.id)
                continue

            if (
                write_distance is not None
                and (
                    read_distance is None
                    or write_distance < read_distance
                )
                and write_distance <= 20
            ):
                links.append(
                    SemanticLink(
                        quantity_id=quantity.id,
                        field=ParamName.read_write_ratio,
                        role=SemanticRole.RATIO_COMPONENT,
                        evidence=_evidence_window(text, quantity),
                        resolver="explicit_pattern",
                    )
                )
                used.add(quantity.id)

        # Only treat the percentages as an explicit ratio when at least
        # two components were identified. A lone "70% read" remains
        # unresolved for later verification/linking.
        if len(used) < 2:
            return []

        return links