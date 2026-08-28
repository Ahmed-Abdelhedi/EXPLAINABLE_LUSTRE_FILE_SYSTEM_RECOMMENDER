from pathlib import Path

import pytest

from e2e_pipeline.end_to_end_pipeline import (
    PipelineLimits,
    run_e2e,
)


def _requirement():
    return {
        "requested_usable_capacity_tib": 10,
        "client_count": 8,
        "average_file_size_gb": 1,
        "max_file_size_gb": 2,
        "total_file_count": 1000,
        "read_write_ratio": {"read_percent": 70.0, "write_percent": 30.0},
        "access_type": "sequential",
        "target_read_gbps": 1,
        "target_write_gbps": 0.5,
        "ha_required": False,
        "max_budget_usd": 100000,
        "max_power_w": 10000,
        "annual_growth_percent": 10,
        "planning_horizon_years": 3,
        "cost_priority": "MEDIUM",
        "power_priority": "LOW",
        "reliability_priority": "HIGH",
        "performance_priority": "HIGH",
        "preference_weights": {
            "cost": 0.2,
            "power": 0.1,
            "performance": 0.4,
            "reliability": 0.3,
        },
    }


class FakeBackend:
    def __init__(self, *, handoff_error=None, valid_ids=("A2",)):
        self.handoff_error = handoff_error
        self.valid_ids = set(valid_ids)
        self.calls = []

    def load_and_validate_config(self, project_root):
        self.calls.append("config")
        return {"version": "fake"}

    def analyze_workload(self, case, config):
        self.calls.append("workload")
        return {"case_id": case["case_id"], "capacity_planning": {"planned_usable_capacity_tib": 15.0}}

    def calculate_features(self, workload, config):
        self.calls.append("features")
        return {"case_id": workload["case_id"]}

    def generate_technical_architecture(self, features, config):
        self.calls.append("technical")
        return {
            "case_id": features["case_id"],
            "MDT_requirement": {"required_total_iops": 100},
            "OST_requirement": {"required_usable_capacity_tib": 15.0},
        }

    def load_drive_catalog(self, project_root):
        self.calls.append("drive_catalog")
        return [{"drive_id": "D1"}]

    def build_handoff(self, architecture, catalog, top_k):
        self.calls.append("handoff")
        if self.handoff_error is not None:
            raise RuntimeError(self.handoff_error)
        return {
            "case_id": architecture["case_id"],
            "requested_top_k": top_k,
            "actual_top_k": {"mdt": 1, "ost": 1},
            "ranking_provenance": {"mdt": {"model_family": "LightGBM"}, "ost": {"model_family": "LightGBM"}},
            "mdt_candidates": [{"identity": {"drive_id": "M1"}}],
            "ost_candidates": [{"identity": {"drive_id": "O1"}}],
        }

    def load_hardware_catalog(self):
        self.calls.append("hardware_catalog")
        return {"catalog_kind": "fake"}

    def generate_architectures(self, handoff, hardware_catalog, limits):
        self.calls.append("h8")
        return {
            "case_id": handoff["case_id"],
            "summary": {
                "mdt_role_options": 2,
                "ost_role_options": 2,
                "potential_pair_count": 4,
                "generated_architecture_count": 2,
                "truncated_by_max_architectures": False,
            },
            "architectures": [
                {"architecture_id": "A1", "case_id": handoff["case_id"], "state": {}},
                {"architecture_id": "A2", "case_id": handoff["case_id"], "state": {}},
            ],
        }

    def score_architectures(self, generated, handoff):
        self.calls.append("h9")
        return {
            "case_id": handoff["case_id"],
            "summary": {"architecture_count": 2},
            "architectures": [
                {"architecture_id": "A1", "rank": 1, "score": 0.95},
                {"architecture_id": "A2", "rank": 2, "score": 0.80},
            ],
        }

    def validate_architectures(self, generated, handoff, hardware_catalog):
        self.calls.append("h10")
        rows = []
        for architecture in generated["architectures"]:
            architecture_id = architecture["architecture_id"]
            valid = architecture_id in self.valid_ids
            rows.append(
                {
                    "architecture_id": architecture_id,
                    "valid": valid,
                    "decision": "VALID" if valid else "INVALID",
                    "violations": [] if valid else [{"code": "budget_exceeded"}],
                }
            )
        return {
            "case_id": handoff["case_id"],
            "summary": {
                "architecture_count": len(rows),
                "valid_architecture_count": sum(row["valid"] for row in rows),
                "invalid_architecture_count": sum(not row["valid"] for row in rows),
                "has_valid_architecture": any(row["valid"] for row in rows),
                "violation_code_counts": {"budget_exceeded": sum(not row["valid"] for row in rows)},
            },
            "architectures": rows,
        }


def test_e2e_selects_highest_scoring_architecture_that_h10_declares_valid(tmp_path):
    backend = FakeBackend(valid_ids=("A2",))
    output = tmp_path / "final_e2e_result.json"

    result = run_e2e(
        _requirement(),
        output_path=output,
        project_root=Path("."),
        backend=backend,
        limits=PipelineLimits(),
    )

    assert result["status"] == "SUCCESS"
    assert result["best_architecture"]["architecture_id"] == "A2"
    assert result["best_architecture"]["score"]["score"] == pytest.approx(0.80)
    assert result["best_architecture"]["validation"]["valid"] is True
    assert output.exists()
    assert backend.calls == [
        "config",
        "workload",
        "features",
        "technical",
        "drive_catalog",
        "handoff",
        "hardware_catalog",
        "h8",
        "h9",
        "h10",
    ]


def test_e2e_reports_no_feasible_mdt_without_hiding_the_stage(tmp_path):
    backend = FakeBackend(
        handoff_error=(
            "CASE: échec MDT Ranker : MDTInferenceError: "
            "Aucun candidat MDT ne respecte les contraintes déterministes."
        )
    )

    result = run_e2e(
        _requirement(),
        output_path=tmp_path / "result.json",
        project_root=Path("."),
        backend=backend,
    )

    assert result["status"] == "NO_FEASIBLE_MDT"
    assert result["failure_stage"] == "RANKING_MDT"
    assert "MDT" in result["message"]


def test_e2e_reports_no_valid_architecture_after_h10(tmp_path):
    backend = FakeBackend(valid_ids=())

    result = run_e2e(
        _requirement(),
        output_path=tmp_path / "result.json",
        project_root=Path("."),
        backend=backend,
    )

    assert result["status"] == "NO_VALID_ARCHITECTURE"
    assert result["failure_stage"] == "H10_FULL_VALIDATION"
    assert result["architecture_search"]["h10_summary"]["valid_architecture_count"] == 0


def test_e2e_reports_no_feasible_ost_without_hiding_the_stage(tmp_path):
    backend = FakeBackend(
        handoff_error=(
            "CASE: échec OST Ranker : OSTInferenceError: "
            "Aucun candidat OST ne respecte les contraintes déterministes."
        )
    )

    result = run_e2e(
        _requirement(),
        output_path=tmp_path / "result.json",
        project_root=Path("."),
        backend=backend,
    )

    assert result["status"] == "NO_FEASIBLE_OST"
    assert result["failure_stage"] == "RANKING_OST"
    assert "OST" in result["message"]
