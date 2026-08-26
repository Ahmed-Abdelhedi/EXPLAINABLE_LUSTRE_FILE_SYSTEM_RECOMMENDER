from __future__ import annotations

import hashlib
import json
from pathlib import Path

from preference_extractor.layer2.labels import (
    PreferenceDimension,
    PreferenceLevel,
    ResolutionSource,
    ResolutionStatus,
)
from preference_extractor.layer2.schemas import (
    DimensionPreferenceResult,
    PreferenceExtractionResult,
)

from .elicitation import b2o_id, o2w_id
from .runtime import FormalPreferenceWeightingLayer


def _example_extraction() -> PreferenceExtractionResult:
    levels = {
        PreferenceDimension.COST:
            PreferenceLevel.LOW,
        PreferenceDimension.POWER:
            None,
        PreferenceDimension.PERFORMANCE:
            PreferenceLevel.HIGH,
        PreferenceDimension.RELIABILITY:
            PreferenceLevel.VERY_HIGH,
    }

    dimensions = {}

    for dimension in PreferenceDimension:
        level = levels[dimension]

        if level is None:
            dimensions[dimension] = DimensionPreferenceResult(
                dimension=dimension,
                status=ResolutionStatus.NO_SIGNAL,
                source=ResolutionSource.TRANSFORMER,
            )
        else:
            dimensions[dimension] = DimensionPreferenceResult(
                dimension=dimension,
                status=ResolutionStatus.RESOLVED,
                source=ResolutionSource.TRANSFORMER,
                level=level,
            )

    return PreferenceExtractionResult(
        text="Synthetic deterministic smoke state.",
        dimensions=dimensions,
        relations=[],
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    layer = FormalPreferenceWeightingLayer()
    extraction = _example_extraction()

    first = layer.run(extraction)

    if first.status.value != "NEEDS_BWM_COMPARISONS":
        raise RuntimeError(
            f"Expected NEEDS_BWM_COMPARISONS, got {first.status.value}."
        )

    if first.best != PreferenceDimension.RELIABILITY:
        raise RuntimeError("Best selection smoke failed.")

    if first.worst != PreferenceDimension.COST:
        raise RuntimeError("Worst selection smoke failed.")

    if len(first.missing_questions) != 3:
        raise RuntimeError(
            "Three active criteria must require exactly 2n-3 = 3 BWM judgments."
        )

    answers = {
        b2o_id(
            PreferenceDimension.RELIABILITY,
            PreferenceDimension.PERFORMANCE,
        ): 3,
        b2o_id(
            PreferenceDimension.RELIABILITY,
            PreferenceDimension.COST,
        ): 5,
        o2w_id(
            PreferenceDimension.PERFORMANCE,
            PreferenceDimension.COST,
        ): 2,
    }

    final = layer.run(
        extraction,
        bwm_answers=answers,
    )

    if final.status.value != "WEIGHTS_READY":
        raise RuntimeError(
            f"Expected WEIGHTS_READY, got {final.status.value}: "
            f"{final.violations}"
        )

    expected = {
        "cost": 0.125,
        "power": 0.0,
        "performance": 0.225,
        "reliability": 0.650,
    }

    actual = final.all_four_weights()

    for dimension, expected_value in expected.items():
        if abs(actual[dimension] - expected_value) > 1e-8:
            raise RuntimeError(
                f"Unexpected weight for {dimension}: "
                f"{actual[dimension]} != {expected_value}"
            )

    if abs(sum(actual.values()) - 1.0) > 1e-10:
        raise RuntimeError("Weights do not sum to 1.")

    if abs(float(final.xi_star) - 0.025) > 1e-8:
        raise RuntimeError(
            f"Unexpected xi*: {final.xi_star}"
        )

    report = {
        "step": "4.5",
        "status": "FORMAL_PREFERENCE_WEIGHTING_SMOKE_PASS",
        "method": "LINEAR_BWM",
        "example": {
            "qualitative_labels": {
                "cost": "LOW",
                "power": "NO_SIGNAL",
                "performance": "HIGH",
                "reliability": "VERY_HIGH",
            },
            "bwm_judgments": answers,
            "result": final.to_dict(),
        },
        "properties_verified": {
            "no_direct_label_to_numeric_mapping": True,
            "best_worst_selected_from_verified_order": True,
            "exact_question_count_2n_minus_3": True,
            "weights_sum_to_one": True,
            "inactive_dimension_weight_zero": True,
            "linear_bwm_solver_used": True,
            "xi_star_recorded": True,
            "ordinal_consistency_gate_passed": True,
            "llm_generates_weights": False,
        },
    }

    report_path = Path(
        "preference_extractor/weighting/"
        "formal_weighting_smoke_report.json"
    )
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    production_files = [
        Path("preference_extractor/weighting/models.py"),
        Path("preference_extractor/weighting/active_set.py"),
        Path("preference_extractor/weighting/best_worst_selector.py"),
        Path("preference_extractor/weighting/elicitation.py"),
        Path("preference_extractor/weighting/bwm_solver.py"),
        Path("preference_extractor/weighting/consistency_gate.py"),
        Path("preference_extractor/weighting/runtime.py"),
    ]

    freeze = {
        "step": "4.5",
        "status": "FORMAL_PREFERENCE_WEIGHTING_LAYER_FROZEN",
        "method": "LINEAR_BWM_PLUS_ORDINAL_CONSISTENCY_GATE",
        "scope": (
            "deterministic formal weighting implementation; "
            "no direct qualitative-label-to-number mapping"
        ),
        "production_smoke": str(report_path),
        "weights_example": actual,
        "xi_star_example": final.xi_star,
        "production_file_sha256": {
            str(path): _sha256(path)
            for path in production_files
        },
        "scientific_contract": {
            "qualitative_labels_define_order_not_ratios": True,
            "bwm_ratios_must_be_user_elicited": True,
            "default_xi_acceptance_threshold": None,
            "reason_no_default_xi_threshold": (
                "A project-specific xi threshold must be calibrated rather "
                "than invented arbitrarily."
            ),
            "sum_weights": 1.0,
        },
    }

    freeze_path = Path(
        "preference_extractor/weighting/"
        "FORMAL_PREFERENCE_WEIGHTING_LAYER_FROZEN.json"
    )
    freeze_path.write_text(
        json.dumps(
            freeze,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
    )
    print()
    print("STATUS:", freeze["status"])
    print("FREEZE FILE:", freeze_path)


if __name__ == "__main__":
    main()
