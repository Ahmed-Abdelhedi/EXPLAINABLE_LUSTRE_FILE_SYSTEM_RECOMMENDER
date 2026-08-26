from __future__ import annotations

from typing import Dict, Iterable, Mapping

import numpy as np
from scipy.optimize import linprog

from preference_extractor.layer2.labels import PreferenceDimension

from .models import BWMSolution


class BWMSolverError(RuntimeError):
    pass


def _check_vector(
    *,
    active_dimensions,
    values: Mapping[PreferenceDimension, int],
    name: str,
) -> None:
    for dimension in active_dimensions:
        if dimension not in values:
            raise ValueError(
                f"{name} is missing {dimension.value}."
            )

        value = values[dimension]

        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"{name}[{dimension.value}] must be an integer."
            )

        if not 1 <= value <= 9:
            raise ValueError(
                f"{name}[{dimension.value}] must be inside [1, 9]."
            )


def solve_linear_bwm(
    *,
    active_dimensions: Iterable[PreferenceDimension],
    best: PreferenceDimension,
    worst: PreferenceDimension,
    best_to_others: Mapping[PreferenceDimension, int],
    others_to_worst: Mapping[PreferenceDimension, int],
) -> BWMSolution:
    """
    Rezaei linear Best-Worst Method.

    Minimize xi subject to:
        |w_B - a_Bj * w_j| <= xi
        |w_j - a_jW * w_W| <= xi
        sum_j w_j = 1
        w_j >= 0
        xi >= 0

    The numerical weights are therefore determined by the optimization model,
    not by an arbitrary mapping from qualitative labels.
    """
    active = list(active_dimensions)

    if len(active) < 2:
        raise ValueError(
            "Linear BWM requires at least two active criteria."
        )

    if best not in active or worst not in active:
        raise ValueError(
            "Best and worst must belong to the active set."
        )

    if best == worst:
        raise ValueError(
            "Best and worst must be different for multi-criterion BWM."
        )

    _check_vector(
        active_dimensions=active,
        values=best_to_others,
        name="best_to_others",
    )
    _check_vector(
        active_dimensions=active,
        values=others_to_worst,
        name="others_to_worst",
    )

    if best_to_others[best] != 1:
        raise ValueError(
            "Best-to-best comparison must equal 1."
        )

    if others_to_worst[worst] != 1:
        raise ValueError(
            "Worst-to-worst comparison must equal 1."
        )

    if best_to_others[worst] != others_to_worst[best]:
        raise ValueError(
            "Best-over-worst comparison is inconsistent between vectors."
        )

    n = len(active)
    index = {
        dimension: position
        for position, dimension in enumerate(active)
    }
    xi_index = n

    objective = np.zeros(n + 1, dtype=float)
    objective[xi_index] = 1.0

    a_ub = []
    b_ub = []

    best_index = index[best]
    worst_index = index[worst]

    for dimension in active:
        j = index[dimension]
        a_bj = float(best_to_others[dimension])

        # w_B - a_Bj*w_j <= xi
        row = np.zeros(n + 1, dtype=float)
        row[best_index] += 1.0
        row[j] -= a_bj
        row[xi_index] -= 1.0
        a_ub.append(row)
        b_ub.append(0.0)

        # -(w_B - a_Bj*w_j) <= xi
        row = np.zeros(n + 1, dtype=float)
        row[best_index] -= 1.0
        row[j] += a_bj
        row[xi_index] -= 1.0
        a_ub.append(row)
        b_ub.append(0.0)

    for dimension in active:
        j = index[dimension]
        a_jw = float(others_to_worst[dimension])

        # w_j - a_jW*w_W <= xi
        row = np.zeros(n + 1, dtype=float)
        row[j] += 1.0
        row[worst_index] -= a_jw
        row[xi_index] -= 1.0
        a_ub.append(row)
        b_ub.append(0.0)

        # -(w_j - a_jW*w_W) <= xi
        row = np.zeros(n + 1, dtype=float)
        row[j] -= 1.0
        row[worst_index] += a_jw
        row[xi_index] -= 1.0
        a_ub.append(row)
        b_ub.append(0.0)

    a_eq = np.zeros((1, n + 1), dtype=float)
    a_eq[0, :n] = 1.0
    b_eq = np.array([1.0], dtype=float)

    bounds = [(0.0, None)] * (n + 1)

    result = linprog(
        c=objective,
        A_ub=np.array(a_ub),
        b_ub=np.array(b_ub),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        raise BWMSolverError(
            "Linear BWM optimization failed: "
            f"{result.status} {result.message}"
        )

    raw_weights = np.array(
        result.x[:n],
        dtype=float,
    )

    # Numerical cleanup only: HiGHS can return tiny floating residuals.
    raw_weights[
        np.abs(raw_weights) < 1e-12
    ] = 0.0

    total = float(raw_weights.sum())

    if total <= 0.0:
        raise BWMSolverError(
            "Linear BWM returned a non-positive weight sum."
        )

    normalized = raw_weights / total

    weights: Dict[PreferenceDimension, float] = {
        dimension: float(normalized[index[dimension]])
        for dimension in active
    }

    return BWMSolution(
        weights=weights,
        xi_star=float(result.x[xi_index]),
        solver="scipy.optimize.linprog(method='highs')",
        solver_status=str(result.message),
    )
