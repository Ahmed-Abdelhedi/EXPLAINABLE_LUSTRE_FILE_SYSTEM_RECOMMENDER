from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from preference_extractor.layer2.labels import PreferenceDimension

from .models import BWMQuestion


B2O = "BEST_TO_OTHER"
O2W = "OTHER_TO_WORST"


@dataclass(frozen=True)
class ElicitationPlan:
    questions: List[BWMQuestion]
    answers: Dict[str, int]
    missing_questions: List[BWMQuestion]
    violations: List[str]

    @property
    def complete(self) -> bool:
        return not self.missing_questions and not self.violations


def b2o_id(
    best: PreferenceDimension,
    other: PreferenceDimension,
) -> str:
    return f"B2O:{best.value}:{other.value}"


def o2w_id(
    other: PreferenceDimension,
    worst: PreferenceDimension,
) -> str:
    return f"O2W:{other.value}:{worst.value}"


def build_questions(
    *,
    active_dimensions: Iterable[PreferenceDimension],
    best: PreferenceDimension,
    worst: PreferenceDimension,
) -> List[BWMQuestion]:
    """
    Standard BWM requires 2n-3 unique user judgments.

    We ask:
      - Best over every other criterion (n-1).
      - Every non-best/non-worst criterion over Worst (n-2).

    The Best-over-Worst judgment is shared by both comparison vectors and is
    therefore asked exactly once.
    """
    active = list(active_dimensions)

    questions: List[BWMQuestion] = []

    for other in active:
        if other == best:
            continue

        questions.append(
            BWMQuestion(
                comparison_id=b2o_id(best, other),
                kind=B2O,
                left=best,
                right=other,
                prompt=(
                    f"On the BWM 1-9 scale, how much more important is "
                    f"{best.value} than {other.value}?"
                ),
            )
        )

    for other in active:
        if other in {best, worst}:
            continue

        questions.append(
            BWMQuestion(
                comparison_id=o2w_id(other, worst),
                kind=O2W,
                left=other,
                right=worst,
                prompt=(
                    f"On the BWM 1-9 scale, how much more important is "
                    f"{other.value} than {worst.value}?"
                ),
            )
        )

    return questions


def _validate_answer(
    comparison_id: str,
    value,
) -> Optional[str]:
    if isinstance(value, bool) or not isinstance(value, int):
        return f"NON_INTEGER_BWM_JUDGMENT:{comparison_id}"

    if not 1 <= value <= 9:
        return f"BWM_JUDGMENT_OUT_OF_RANGE:{comparison_id}:{value}"

    return None


def prepare_elicitation(
    *,
    active_dimensions: Iterable[PreferenceDimension],
    best: PreferenceDimension,
    worst: PreferenceDimension,
    supplied_answers: Optional[Mapping[str, int]] = None,
) -> ElicitationPlan:
    questions = build_questions(
        active_dimensions=active_dimensions,
        best=best,
        worst=worst,
    )

    supplied = dict(supplied_answers or {})
    expected_ids = {
        question.comparison_id
        for question in questions
    }

    violations: List[str] = []

    for comparison_id in supplied:
        if comparison_id not in expected_ids:
            violations.append(
                f"UNEXPECTED_BWM_JUDGMENT:{comparison_id}"
            )

    accepted: Dict[str, int] = {}

    for question in questions:
        if question.comparison_id not in supplied:
            continue

        value = supplied[question.comparison_id]
        error = _validate_answer(
            question.comparison_id,
            value,
        )

        if error is not None:
            violations.append(error)
            continue

        accepted[question.comparison_id] = int(value)

    missing = [
        question
        for question in questions
        if question.comparison_id not in accepted
    ]

    return ElicitationPlan(
        questions=questions,
        answers=accepted,
        missing_questions=missing,
        violations=violations,
    )


def build_bwm_vectors(
    *,
    active_dimensions: Iterable[PreferenceDimension],
    best: PreferenceDimension,
    worst: PreferenceDimension,
    answers: Mapping[str, int],
) -> Tuple[
    Dict[PreferenceDimension, int],
    Dict[PreferenceDimension, int],
]:
    """
    Returns:
      a_Bj  = Best-to-Others comparison vector
      a_jW  = Others-to-Worst comparison vector
    """
    active = list(active_dimensions)

    best_to_others: Dict[PreferenceDimension, int] = {
        best: 1
    }

    for dimension in active:
        if dimension == best:
            continue

        best_to_others[dimension] = int(
            answers[b2o_id(best, dimension)]
        )

    others_to_worst: Dict[PreferenceDimension, int] = {
        worst: 1,
        best: best_to_others[worst],
    }

    for dimension in active:
        if dimension in {best, worst}:
            continue

        others_to_worst[dimension] = int(
            answers[o2w_id(dimension, worst)]
        )

    return best_to_others, others_to_worst
