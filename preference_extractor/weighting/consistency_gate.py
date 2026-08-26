from __future__ import annotations

from typing import Mapping, Optional

from preference_extractor.layer2.labels import (
    PreferenceDimension,
    PreferenceLevel,
)
from preference_extractor.layer2.schemas import PreferenceRelation

from .best_worst_selector import LEVEL_RANK
from .models import ConsistencyReport, ConsistencyStatus


def check_weight_consistency(
    *,
    weights: Mapping[PreferenceDimension, float],
    absolute_levels: Mapping[PreferenceDimension, PreferenceLevel],
    relations,
    best: PreferenceDimension,
    worst: PreferenceDimension,
    xi_star: float,
    max_xi: Optional[float] = None,
    epsilon: float = 1e-9,
) -> ConsistencyReport:
    """
    Deterministic trust gate after BWM.

    No default xi threshold is invented. If a project-specific threshold is
    later calibrated on development data, pass it as max_xi.

    Qualitative labels are used only as order constraints:
      VERY_HIGH > HIGH > MEDIUM > LOW > VERY_LOW

    Equal labels do NOT force equal numerical weights.
    """
    violations = []

    total = float(sum(weights.values()))

    if abs(total - 1.0) > 1e-8:
        violations.append(
            f"WEIGHT_SUM_NOT_ONE:{total:.12f}"
        )

    for dimension, weight in weights.items():
        if weight < -epsilon:
            violations.append(
                f"NEGATIVE_WEIGHT:{dimension.value}:{weight}"
            )

    dimensions = list(weights)

    for i, left in enumerate(dimensions):
        for right in dimensions[i + 1:]:
            left_level = absolute_levels.get(left)
            right_level = absolute_levels.get(right)

            if left_level is None or right_level is None:
                continue

            left_rank = LEVEL_RANK[left_level]
            right_rank = LEVEL_RANK[right_level]

            if (
                left_rank > right_rank
                and not (
                    weights[left]
                    > weights[right] + epsilon
                )
            ):
                violations.append(
                    "ORDINAL_ORDER_VIOLATION:"
                    f"{left.value}>{right.value}"
                )

            if (
                right_rank > left_rank
                and not (
                    weights[right]
                    > weights[left] + epsilon
                )
            ):
                violations.append(
                    "ORDINAL_ORDER_VIOLATION:"
                    f"{right.value}>{left.value}"
                )

    for relation in relations:
        if (
            relation.higher in weights
            and relation.lower in weights
            and not (
                weights[relation.higher]
                > weights[relation.lower] + epsilon
            )
        ):
            violations.append(
                "RELATIVE_ORDER_VIOLATION:"
                f"{relation.higher.value}>{relation.lower.value}"
            )

    best_weight = weights[best]
    worst_weight = weights[worst]

    for dimension, weight in weights.items():
        if best_weight + epsilon < weight:
            violations.append(
                f"BEST_NOT_MAXIMAL:{best.value}<{dimension.value}"
            )

        if worst_weight > weight + epsilon:
            violations.append(
                f"WORST_NOT_MINIMAL:{worst.value}>{dimension.value}"
            )

    if max_xi is not None and xi_star > max_xi + epsilon:
        violations.append(
            "BWM_DEVIATION_ABOVE_CONFIGURED_THRESHOLD:"
            f"{xi_star:.12f}>{max_xi:.12f}"
        )

    return ConsistencyReport(
        status=(
            ConsistencyStatus.FAIL
            if violations
            else ConsistencyStatus.PASS
        ),
        violations=violations,
        deviation_threshold=max_xi,
    )
