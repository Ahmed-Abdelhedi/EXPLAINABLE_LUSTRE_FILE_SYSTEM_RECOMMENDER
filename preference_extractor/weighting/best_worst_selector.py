from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from preference_extractor.layer2.labels import (
    PreferenceDimension,
    PreferenceLevel,
)
from preference_extractor.layer2.schemas import PreferenceRelation

from .active_set import ActivePreferenceSet


LEVEL_RANK = {
    PreferenceLevel.VERY_LOW: 0,
    PreferenceLevel.LOW: 1,
    PreferenceLevel.MEDIUM: 2,
    PreferenceLevel.HIGH: 3,
    PreferenceLevel.VERY_HIGH: 4,
}


@dataclass(frozen=True)
class BestWorstSelection:
    best: Optional[PreferenceDimension]
    worst: Optional[PreferenceDimension]
    status: str
    violations: List[str]
    best_candidates: List[PreferenceDimension]
    worst_candidates: List[PreferenceDimension]


def _build_preference_edges(
    active: ActivePreferenceSet,
) -> Set[Tuple[PreferenceDimension, PreferenceDimension]]:
    """
    Directed edge A -> B means A is strictly more important than B.

    Absolute labels contribute only ordering, never numerical ratios.
    Comparative Layer-2 relations contribute the same strict ordering.
    """
    edges: Set[
        Tuple[PreferenceDimension, PreferenceDimension]
    ] = set()

    dimensions = list(active.active_dimensions)

    for index, left in enumerate(dimensions):
        for right in dimensions[index + 1:]:
            left_level = active.absolute_levels.get(left)
            right_level = active.absolute_levels.get(right)

            if left_level is None or right_level is None:
                continue

            left_rank = LEVEL_RANK[left_level]
            right_rank = LEVEL_RANK[right_level]

            if left_rank > right_rank:
                edges.add((left, right))
            elif right_rank > left_rank:
                edges.add((right, left))

    for relation in active.relations:
        edges.add((relation.higher, relation.lower))

    return edges


def _has_cycle(
    nodes: Iterable[PreferenceDimension],
    edges: Set[Tuple[PreferenceDimension, PreferenceDimension]],
) -> bool:
    adjacency: Dict[PreferenceDimension, List[PreferenceDimension]] = {
        node: [] for node in nodes
    }

    for higher, lower in edges:
        if higher in adjacency:
            adjacency[higher].append(lower)

    visiting: Set[PreferenceDimension] = set()
    visited: Set[PreferenceDimension] = set()

    def visit(node: PreferenceDimension) -> bool:
        if node in visiting:
            return True

        if node in visited:
            return False

        visiting.add(node)

        for neighbor in adjacency[node]:
            if visit(neighbor):
                return True

        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in adjacency)


def select_best_worst(
    active: ActivePreferenceSet,
    *,
    explicit_best: Optional[PreferenceDimension] = None,
    explicit_worst: Optional[PreferenceDimension] = None,
) -> BestWorstSelection:
    dimensions = list(active.active_dimensions)
    active_set = set(dimensions)
    violations: List[str] = list(active.violations)

    if not dimensions:
        return BestWorstSelection(
            best=None,
            worst=None,
            status="NO_ACTIVE",
            violations=violations,
            best_candidates=[],
            worst_candidates=[],
        )

    edges = _build_preference_edges(active)

    if _has_cycle(dimensions, edges):
        violations.append("PREFERENCE_ORDER_CYCLE")
        return BestWorstSelection(
            best=None,
            worst=None,
            status="INCONSISTENT",
            violations=violations,
            best_candidates=[],
            worst_candidates=[],
        )

    incoming = {
        dimension: 0
        for dimension in dimensions
    }
    outgoing = {
        dimension: 0
        for dimension in dimensions
    }

    for higher, lower in edges:
        if higher in active_set and lower in active_set:
            outgoing[higher] += 1
            incoming[lower] += 1

    best_candidates = [
        dimension
        for dimension in dimensions
        if incoming[dimension] == 0
    ]
    worst_candidates = [
        dimension
        for dimension in dimensions
        if outgoing[dimension] == 0
    ]

    if explicit_best is not None:
        if explicit_best not in active_set:
            violations.append(
                f"EXPLICIT_BEST_NOT_ACTIVE:{explicit_best.value}"
            )
        elif incoming[explicit_best] > 0:
            violations.append(
                f"EXPLICIT_BEST_CONTRADICTS_ORDER:{explicit_best.value}"
            )

    if explicit_worst is not None:
        if explicit_worst not in active_set:
            violations.append(
                f"EXPLICIT_WORST_NOT_ACTIVE:{explicit_worst.value}"
            )
        elif outgoing[explicit_worst] > 0:
            violations.append(
                f"EXPLICIT_WORST_CONTRADICTS_ORDER:{explicit_worst.value}"
            )

    if (
        explicit_best is not None
        and explicit_worst is not None
        and explicit_best == explicit_worst
        and len(dimensions) > 1
    ):
        violations.append("BEST_EQUALS_WORST")

    if violations:
        return BestWorstSelection(
            best=None,
            worst=None,
            status="INCONSISTENT",
            violations=violations,
            best_candidates=best_candidates,
            worst_candidates=worst_candidates,
        )

    best = explicit_best
    worst = explicit_worst

    if best is None and len(best_candidates) == 1:
        best = best_candidates[0]

    if worst is None and len(worst_candidates) == 1:
        worst = worst_candidates[0]

    if best is None or worst is None:
        return BestWorstSelection(
            best=best,
            worst=worst,
            status="NEEDS_BEST_WORST",
            violations=[],
            best_candidates=best_candidates,
            worst_candidates=worst_candidates,
        )

    return BestWorstSelection(
        best=best,
        worst=worst,
        status="READY",
        violations=[],
        best_candidates=best_candidates,
        worst_candidates=worst_candidates,
    )
