from __future__ import annotations

import json

from requirement_extractor_v2.models import (
    ParamName,
    VerificationStatus,
)
from requirement_extractor_v2.selective_cascade import SelectiveCascade
from requirement_extractor_v2.verified_pipeline import VerifiedRequirementPipeline


class _NeverSemanticLinker:
    """Semantic fallback disabled only for this deterministic regression test."""

    class _Prediction:
        accepted = False
        link = None
        confidence = 0.0
        margin = 0.0
        raw_field = "__UNRESOLVED__"
        raw_role = "unspecified"

    def predict(self, *args, **kwargs):
        return self._Prediction()


class _NeverLLMFallback:
    def resolve_quantity(self, *args, **kwargs):
        return None

    def info(self):
        return {
            "enabled": False,
            "test_double": True,
        }


def _pipeline() -> VerifiedRequirementPipeline:
    cascade = SelectiveCascade(
        semantic_linker=_NeverSemanticLinker(),
        llm_fallback=_NeverLLMFallback(),
    )

    return VerifiedRequirementPipeline(
        cascade=cascade
    )


def _dump(result) -> str:
    return json.dumps(
        result.to_dict(),
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def _assert_verified(
    result,
    field: ParamName,
    expected,
    label: str,
) -> None:
    actual = result.verified_values().get(
        field.value
    )

    if actual != expected:
        raise AssertionError(
            f"{label}: expected {field.value}={expected!r}, "
            f"got {actual!r}\\nPIPELINE RESULT:\\n{_dump(result)}"
        )


def _assert_status(
    result,
    field: ParamName,
    expected_status: VerificationStatus,
    label: str,
) -> None:
    decisions = [
        decision
        for decision in result.decisions
        if decision.field == field
    ]

    if not decisions:
        raise AssertionError(
            f"{label}: no decision found for {field.value}\\n"
            f"PIPELINE RESULT:\\n{_dump(result)}"
        )

    if not any(
        decision.status == expected_status
        for decision in decisions
    ):
        raise AssertionError(
            f"{label}: expected status {expected_status.value}, got "
            f"{[d.status.value for d in decisions]}\\n"
            f"PIPELINE RESULT:\\n{_dump(result)}"
        )


def main() -> None:
    pipeline = _pipeline()

    result = pipeline.process(
        "Planning horizon is 3 years."
    )
    _assert_verified(
        result,
        ParamName.planning_horizon_years,
        3,
        "explicit English",
    )

    result = pipeline.process(
        "Plan for 5 years."
    )
    _assert_verified(
        result,
        ParamName.planning_horizon_years,
        5,
        "compact English",
    )

    result = pipeline.process(
        "Horizon de planification : 4 ans."
    )
    _assert_verified(
        result,
        ParamName.planning_horizon_years,
        4,
        "explicit French",
    )

    result = pipeline.process(
        "Annual growth is 20% over 3 years."
    )
    _assert_verified(
        result,
        ParamName.annual_growth_percent,
        20,
        "annual growth component",
    )
    _assert_verified(
        result,
        ParamName.planning_horizon_years,
        3,
        "growth-coupled horizon",
    )

    # Contextual answer with explicit natural-language unit.
    result = pipeline.process(
        "3 years",
        previous_question=(
            "What planning horizon should be used?"
        ),
        previous_question_field=(
            ParamName.planning_horizon_years
        ),
    )
    _assert_verified(
        result,
        ParamName.planning_horizon_years,
        3,
        "contextual answer with years",
    )

    # Contextual answer with unit inherited from field_defs.TARGET_UNITS.
    result = pipeline.process(
        "3",
        previous_question=(
            "What is the planning horizon in years?"
        ),
        previous_question_field=(
            ParamName.planning_horizon_years
        ),
    )
    _assert_verified(
        result,
        ParamName.planning_horizon_years,
        3,
        "contextual answer inherited years",
    )

    result = pipeline.process(
        "Planning horizon is 0 years."
    )
    _assert_status(
        result,
        ParamName.planning_horizon_years,
        VerificationStatus.INVALID,
        "zero horizon",
    )

    result = pipeline.process(
        "Planning horizon is -2 years."
    )
    _assert_status(
        result,
        ParamName.planning_horizon_years,
        VerificationStatus.INVALID,
        "negative horizon",
    )

    result = pipeline.process(
        "Planning horizon is 2.5 years."
    )
    _assert_status(
        result,
        ParamName.planning_horizon_years,
        VerificationStatus.INVALID,
        "fractional horizon",
    )

    result = pipeline.process(
        "We have 3 years of logs."
    )
    if (
        ParamName.planning_horizon_years.value
        in result.verified_values()
    ):
        raise AssertionError(
            "safety negative logs: duration was incorrectly accepted as "
            f"planning_horizon_years\\nPIPELINE RESULT:\\n{_dump(result)}"
        )

    result = pipeline.process(
        "The project started 3 years ago."
    )
    if (
        ParamName.planning_horizon_years.value
        in result.verified_values()
    ):
        raise AssertionError(
            "safety negative history: duration was incorrectly accepted as "
            f"planning_horizon_years\\nPIPELINE RESULT:\\n{_dump(result)}"
        )

    print(
        "Planning horizon regression test: PASS"
    )
    print(
        "  - explicit English: PASS"
    )
    print(
        "  - compact English: PASS"
    )
    print(
        "  - explicit French: PASS"
    )
    print(
        "  - annual growth + horizon: PASS"
    )
    print(
        "  - contextual answer with unit: PASS"
    )
    print(
        "  - contextual answer inherited unit: PASS"
    )
    print(
        "  - positive integer validation: PASS"
    )
    print(
        "  - safety negatives: PASS"
    )


if __name__ == "__main__":
    main()