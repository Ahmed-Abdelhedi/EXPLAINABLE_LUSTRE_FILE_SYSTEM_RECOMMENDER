from __future__ import annotations

from typing import Mapping, Optional

from preference_extractor.layer2.labels import (
    PreferenceDimension,
    PreferenceLevel,
    ResolutionStatus,
)
from preference_extractor.layer2.schemas import PreferenceExtractionResult

from .active_set import build_active_set
from .best_worst_selector import select_best_worst
from .bwm_solver import BWMSolverError, solve_linear_bwm
from .consistency_gate import check_weight_consistency
from .elicitation import build_bwm_vectors, prepare_elicitation
from .models import (
    ConsistencyReport,
    ConsistencyStatus,
    METHOD_LINEAR_BWM,
    METHOD_SINGLE_ACTIVE,
    WeightingResult,
    WeightingStatus,
)


class FormalPreferenceWeightingLayer:
    """
    Final deterministic layer of the Preference Extractor.

    Layer-2 qualitative labels are NOT mapped directly to numbers.

    Multi-criterion flow:
        Layer-2 verified preferences
        -> active set
        -> Best/Worst selection from ordinal/relative evidence
        -> explicit user BWM judgments (1..9)
        -> linear BWM optimization
        -> deterministic ordinal/relative consistency gate
        -> normalized weights

    No LLM generates final numerical weights.
    """

    def __init__(
        self,
        *,
        max_xi: Optional[float] = None,
        consistency_epsilon: float = 1e-9,
    ) -> None:
        if max_xi is not None and max_xi < 0.0:
            raise ValueError("max_xi must be non-negative.")

        self.max_xi = max_xi
        self.consistency_epsilon = float(
            consistency_epsilon
        )

    @staticmethod
    def _zero_weight_dict():
        return {
            dimension: 0.0
            for dimension in PreferenceDimension
        }

    def run(
        self,
        extraction: PreferenceExtractionResult,
        *,
        explicit_best: Optional[PreferenceDimension] = None,
        explicit_worst: Optional[PreferenceDimension] = None,
        bwm_answers: Optional[Mapping[str, int]] = None,
        single_active_confirmed: bool = False,
    ) -> WeightingResult:
        active = build_active_set(extraction)

        if active.violations:
            return WeightingResult(
                status=WeightingStatus.INCONSISTENT_PREFERENCES,
                active_dimensions=active.active_dimensions,
                weights=self._zero_weight_dict(),
                violations=list(active.violations),
                notes=[
                    "Layer-2 preference state is structurally inconsistent."
                ],
            )

        if active.blocked_dimensions:
            return WeightingResult(
                status=WeightingStatus.BLOCKED_UNRESOLVED,
                active_dimensions=active.active_dimensions,
                weights=self._zero_weight_dict(),
                violations=[
                    "UNRESOLVED_DIMENSIONS:"
                    + ",".join(
                        dimension.value
                        for dimension
                        in active.blocked_dimensions
                    )
                ],
                notes=[
                    "Numerical weights are blocked until unresolved "
                    "preference dimensions are clarified."
                ],
            )

        if not active.active_dimensions:
            return WeightingResult(
                status=WeightingStatus.NO_ACTIVE_PREFERENCE,
                active_dimensions=[],
                weights=self._zero_weight_dict(),
                notes=[
                    "No neutral/equal weight vector is invented automatically."
                ],
            )

        if len(active.active_dimensions) == 1:
            dimension = active.active_dimensions[0]
            level = active.absolute_levels.get(dimension)

            auto_allowed = (
                level in {
                    PreferenceLevel.HIGH,
                    PreferenceLevel.VERY_HIGH,
                }
            )

            if not auto_allowed and not single_active_confirmed:
                return WeightingResult(
                    status=(
                        WeightingStatus
                        .NEEDS_SINGLE_CRITERION_CONFIRMATION
                    ),
                    active_dimensions=[dimension],
                    best=dimension,
                    worst=dimension,
                    weights=self._zero_weight_dict(),
                    notes=[
                        "A single LOW/MEDIUM/VERY_LOW preference is not "
                        "silently converted into 100% of the ranking objective.",
                        "Confirm that this should be the only ranking objective "
                        "before assigning weight 1.0.",
                    ],
                )

            weights = self._zero_weight_dict()
            weights[dimension] = 1.0

            return WeightingResult(
                status=WeightingStatus.WEIGHTS_READY,
                active_dimensions=[dimension],
                method=METHOD_SINGLE_ACTIVE,
                best=dimension,
                worst=dimension,
                weights=weights,
                xi_star=0.0,
                consistency=ConsistencyReport(
                    status=ConsistencyStatus.PASS
                ),
                source="CONFIRMED_SINGLE_ACTIVE_CRITERION",
                notes=[
                    "No BWM comparisons are required for one confirmed "
                    "active criterion."
                ],
            )

        selection = select_best_worst(
            active,
            explicit_best=explicit_best,
            explicit_worst=explicit_worst,
        )

        if selection.status == "INCONSISTENT":
            return WeightingResult(
                status=WeightingStatus.INCONSISTENT_PREFERENCES,
                active_dimensions=active.active_dimensions,
                weights=self._zero_weight_dict(),
                violations=selection.violations,
            )

        if selection.status == "NEEDS_BEST_WORST":
            return WeightingResult(
                status=WeightingStatus.NEEDS_BEST_WORST,
                active_dimensions=active.active_dimensions,
                best=selection.best,
                worst=selection.worst,
                weights=self._zero_weight_dict(),
                notes=[
                    "Best/Worst cannot be selected uniquely from verified "
                    "ordinal and comparative evidence.",
                    "Best candidates: "
                    + ",".join(
                        dimension.value
                        for dimension
                        in selection.best_candidates
                    ),
                    "Worst candidates: "
                    + ",".join(
                        dimension.value
                        for dimension
                        in selection.worst_candidates
                    ),
                ],
            )

        assert selection.best is not None
        assert selection.worst is not None

        plan = prepare_elicitation(
            active_dimensions=active.active_dimensions,
            best=selection.best,
            worst=selection.worst,
            supplied_answers=bwm_answers,
        )

        if plan.violations:
            return WeightingResult(
                status=WeightingStatus.INVALID_BWM_JUDGMENTS,
                active_dimensions=active.active_dimensions,
                best=selection.best,
                worst=selection.worst,
                weights=self._zero_weight_dict(),
                missing_questions=plan.missing_questions,
                violations=plan.violations,
            )

        if plan.missing_questions:
            return WeightingResult(
                status=WeightingStatus.NEEDS_BWM_COMPARISONS,
                active_dimensions=active.active_dimensions,
                best=selection.best,
                worst=selection.worst,
                weights=self._zero_weight_dict(),
                missing_questions=plan.missing_questions,
                source="USER_ELICITED_BWM",
                notes=[
                    "Qualitative labels determine ordering only; "
                    "numerical BWM ratios must come from explicit user judgments."
                ],
            )

        best_to_others, others_to_worst = (
            build_bwm_vectors(
                active_dimensions=active.active_dimensions,
                best=selection.best,
                worst=selection.worst,
                answers=plan.answers,
            )
        )

        try:
            solution = solve_linear_bwm(
                active_dimensions=active.active_dimensions,
                best=selection.best,
                worst=selection.worst,
                best_to_others=best_to_others,
                others_to_worst=others_to_worst,
            )
        except (ValueError, BWMSolverError) as exc:
            return WeightingResult(
                status=WeightingStatus.INVALID_BWM_JUDGMENTS,
                active_dimensions=active.active_dimensions,
                best=selection.best,
                worst=selection.worst,
                weights=self._zero_weight_dict(),
                violations=[
                    f"BWM_SOLVER_INPUT_ERROR:{exc}"
                ],
            )

        consistency = check_weight_consistency(
            weights=solution.weights,
            absolute_levels=active.absolute_levels,
            relations=active.relations,
            best=selection.best,
            worst=selection.worst,
            xi_star=solution.xi_star,
            max_xi=self.max_xi,
            epsilon=self.consistency_epsilon,
        )

        all_weights = self._zero_weight_dict()
        all_weights.update(solution.weights)

        if consistency.status == ConsistencyStatus.FAIL:
            return WeightingResult(
                status=WeightingStatus.INCONSISTENT_PREFERENCES,
                active_dimensions=active.active_dimensions,
                method=METHOD_LINEAR_BWM,
                best=selection.best,
                worst=selection.worst,
                weights=all_weights,
                xi_star=solution.xi_star,
                consistency=consistency,
                violations=list(consistency.violations),
                source="USER_ELICITED_BWM",
                notes=[
                    "The BWM solution is preserved for diagnosis but is not "
                    "trusted as a final preference vector."
                ],
            )

        return WeightingResult(
            status=WeightingStatus.WEIGHTS_READY,
            active_dimensions=active.active_dimensions,
            method=METHOD_LINEAR_BWM,
            best=selection.best,
            worst=selection.worst,
            weights=all_weights,
            xi_star=solution.xi_star,
            consistency=consistency,
            source="USER_ELICITED_BWM",
            notes=[
                "Weights are produced by linear BWM, not by direct ordinal "
                "label-to-number mapping."
            ],
        )
