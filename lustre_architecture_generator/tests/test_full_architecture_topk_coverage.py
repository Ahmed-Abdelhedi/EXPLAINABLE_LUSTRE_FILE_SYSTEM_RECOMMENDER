from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


import full_architecture.topk_coverage as topk  # noqa: E402
from full_architecture.topk_coverage import (  # noqa: E402
    TopKCoverageError,
    analyze_case_topk_coverage,
    normalize_top_k_values,
    unresolved_case_ids_from_h10c,
)


def test_normalize_top_k_values_sorts_and_deduplicates() -> None:
    assert normalize_top_k_values(
        [50, 20, 50],
        baseline_k=10,
    ) == [20, 50]


def test_normalize_top_k_values_rejects_baseline_or_lower() -> None:
    with pytest.raises(
        TopKCoverageError,
        match="doit être > baseline_k",
    ):
        normalize_top_k_values(
            [10, 20],
            baseline_k=10,
        )


def test_normalize_top_k_values_rejects_empty_schedule() -> None:
    with pytest.raises(
        TopKCoverageError,
        match="Au moins une valeur K",
    ):
        normalize_top_k_values(
            [],
            baseline_k=10,
        )


def test_unresolved_case_ids_from_h10c_filters_recovered() -> None:
    result = unresolved_case_ids_from_h10c(
        {
            "cases": [
                {
                    "case_id": "A",
                    "status": "OK",
                    "recovered_valid_architecture": True,
                },
                {
                    "case_id": "B",
                    "status": "OK",
                    "recovered_valid_architecture": False,
                },
                {
                    "case_id": "C",
                    "status": "FAILED",
                    "recovered_valid_architecture": False,
                },
            ]
        }
    )

    assert result == ["B"]


def _architecture() -> dict[str, Any]:
    return {
        "case_id": "REQ_TOPK_001",
    }


def _analysis(
    *,
    recovered: bool,
    classification: str,
    arch_id: str | None = None,
) -> dict[str, Any]:
    return {
        "recovered_valid_architecture": recovered,
        "recovered_architecture_id": arch_id,
        "coverage_interpretation": (
            "RECOVERED"
            if recovered
            else "UNRESOLVED"
        ),
        "search": {
            "classification": classification,
            "minimum_total_cost_usd": 10.0,
            "minimum_total_power_w": 20.0,
        },
        "limits": {
            "maximum_budget_usd": 100.0,
            "maximum_power_w": 100.0,
        },
        "option_counts": {
            "mdt_raw": 10,
            "ost_raw": 12,
            "mdt_pareto": 2,
            "ost_pareto": 3,
        },
        "selected_path_indexes": {
            "mdt_hardware_path_index": 1,
            "ost_hardware_path_index": 2,
        },
    }


def test_topk_coverage_stops_at_first_recovered_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_k: list[int] = []

    def fake_handoff(**kwargs: Any) -> dict[str, Any]:
        observed_k.append(kwargs["top_k"])
        return {
            "case_id": "REQ_TOPK_001",
            "top_k": kwargs["top_k"],
        }

    def fake_analysis(**kwargs: Any) -> dict[str, Any]:
        k_value = kwargs["handoff"]["top_k"]
        return _analysis(
            recovered=(k_value == 20),
            classification=(
                "FEASIBLE_PAIR_EXISTS"
                if k_value == 20
                else "POWER_LOWER_BOUND_EXCEEDS"
            ),
            arch_id="ARCH_20" if k_value == 20 else None,
        )

    monkeypatch.setattr(
        topk,
        "build_runtime_handoff",
        fake_handoff,
    )
    monkeypatch.setattr(
        topk,
        "analyze_case_full_path_domain",
        fake_analysis,
    )

    result = analyze_case_topk_coverage(
        architecture=_architecture(),
        drive_catalog=[],
        hardware_catalog={},
        top_k_values=[20, 50],
        baseline_k=10,
    )

    assert observed_k == [20]
    assert result["recovered_at_k"] == 20
    assert result["recovered_architecture_id"] == "ARCH_20"


def test_topk_coverage_reaches_k50_when_k20_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_k: list[int] = []

    def fake_handoff(**kwargs: Any) -> dict[str, Any]:
        observed_k.append(kwargs["top_k"])
        return {
            "case_id": "REQ_TOPK_001",
            "top_k": kwargs["top_k"],
        }

    def fake_analysis(**kwargs: Any) -> dict[str, Any]:
        k_value = kwargs["handoff"]["top_k"]
        return _analysis(
            recovered=(k_value == 50),
            classification=(
                "FEASIBLE_PAIR_EXISTS"
                if k_value == 50
                else "BUDGET_LOWER_BOUND_EXCEEDS"
            ),
            arch_id="ARCH_50" if k_value == 50 else None,
        )

    monkeypatch.setattr(
        topk,
        "build_runtime_handoff",
        fake_handoff,
    )
    monkeypatch.setattr(
        topk,
        "analyze_case_full_path_domain",
        fake_analysis,
    )

    result = analyze_case_topk_coverage(
        architecture=_architecture(),
        drive_catalog=[],
        hardware_catalog={},
        top_k_values=[20, 50],
        baseline_k=10,
    )

    assert observed_k == [20, 50]
    assert result["recovered_at_k"] == 50


def test_topk_coverage_can_remain_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        topk,
        "build_runtime_handoff",
        lambda **kwargs: {
            "case_id": "REQ_TOPK_001",
            "top_k": kwargs["top_k"],
        },
    )
    monkeypatch.setattr(
        topk,
        "analyze_case_full_path_domain",
        lambda **kwargs: _analysis(
            recovered=False,
            classification="JOINT_BUDGET_POWER_CONFLICT",
        ),
    )

    result = analyze_case_topk_coverage(
        architecture=_architecture(),
        drive_catalog=[],
        hardware_catalog={},
        top_k_values=[20, 50],
        baseline_k=10,
    )

    assert result["recovered_valid_architecture"] is False
    assert result["recovered_at_k"] is None
    assert "K_50" in result["coverage_interpretation"]


def test_topk_coverage_never_claims_global_infeasibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        topk,
        "build_runtime_handoff",
        lambda **kwargs: {
            "case_id": "REQ_TOPK_001",
            "top_k": kwargs["top_k"],
        },
    )
    monkeypatch.setattr(
        topk,
        "analyze_case_full_path_domain",
        lambda **kwargs: _analysis(
            recovered=False,
            classification="POWER_LOWER_BOUND_EXCEEDS",
        ),
    )

    result = analyze_case_topk_coverage(
        architecture=_architecture(),
        drive_catalog=[],
        hardware_catalog={},
        top_k_values=[20, 50],
        baseline_k=10,
    )

    assert result["global_infeasibility_claimed"] is False
    assert "below K=50" in result["remaining_uncertainty"]


def test_topk_coverage_does_not_use_beam_or_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        topk,
        "build_runtime_handoff",
        lambda **kwargs: {
            "case_id": "REQ_TOPK_001",
            "top_k": kwargs["top_k"],
        },
    )
    monkeypatch.setattr(
        topk,
        "analyze_case_full_path_domain",
        lambda **kwargs: _analysis(
            recovered=True,
            classification="FEASIBLE_PAIR_EXISTS",
            arch_id="ARCH_OK",
        ),
    )

    result = analyze_case_topk_coverage(
        architecture=_architecture(),
        drive_catalog=[],
        hardware_catalog={},
        top_k_values=[20],
        baseline_k=10,
    )

    assert result["beam_search_applied"] is False
    assert result["architecture_scoring_required"] is False
