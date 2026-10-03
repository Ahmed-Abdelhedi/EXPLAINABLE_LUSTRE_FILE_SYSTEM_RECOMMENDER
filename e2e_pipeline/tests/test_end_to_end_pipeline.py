import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from e2e_pipeline.end_to_end_pipeline import (
    FrozenRuntimeBackend,
    PipelineLimits,
    SearchOptions,
    run_e2e,
    run_e2e_from_file,
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
    def __init__(self, *, handoff_error=None, valid_ids=("A2",), beam_error=None):
        self.handoff_error = handoff_error
        self.valid_ids = set(valid_ids)
        self.calls = []
        self.beam_error = beam_error
        self.beam_parameters = None
        self.beam_result = None

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

    def run_beam_search(self, handoff, hardware_catalog, limits, search_options):
        self.calls.append("beam")
        self.beam_parameters = (handoff, hardware_catalog, limits, search_options)
        if self.beam_error is not None:
            raise self.beam_error
        architectures = []
        for rank, (architecture_id, score) in enumerate((("A1", 0.95), ("A2", 0.80)), 1):
            valid = architecture_id in self.valid_ids
            architectures.append({
                "architecture_id": architecture_id,
                "case_id": handoff["case_id"],
                "state": {"stage": "COMPLETE", "validation": {"is_valid": valid}},
                "h9": {"architecture_id": architecture_id, "rank": rank, "score": score},
                "h10": {
                    "architecture_id": architecture_id,
                    "valid": valid,
                    "decision": "VALID" if valid else "INVALID",
                    "beam_search_applied": False,
                },
                "search_provenance": {"beam_heuristic": 0.7, "trace": [{"depth": 6}]},
            })
        valid = [row for row in architectures if row["h10"]["valid"]]
        self.beam_result = {
            "status": "VALID_ARCHITECTURE_FOUND" if valid else "NO_VALID_ARCHITECTURE_FOUND_WITHIN_SEARCH_DOMAIN",
            "beam_search_applied": True,
            "global_infeasibility_claimed": False,
            "beam_width": search_options.beam_width,
            "summary": {"complete_architectures_produced": 2, "h10_calls": 2, "valid_architecture_count": len(valid)},
            "heuristic_policy": {"id": "fake_beam_policy"},
            "search_trace": [{"search_stage": "OST_HARDWARE", "retained": 2}],
            "lookahead_trace": [{"search_stage": "LOOKAHEAD_MDT", "branches_created": 3}],
            "architectures": architectures,
            "best_validated_architecture": valid[0] if valid else None,
        }
        return self.beam_result

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
        search_options=SearchOptions(strategy="exhaustive"),
    )

    assert result["status"] == "SUCCESS"
    assert result["best_architecture"]["architecture_id"] == "A2"
    assert result["best_architecture"]["score"]["score"] == pytest.approx(0.80)
    assert result["best_architecture"]["validation"]["valid"] is True
    assert output.exists()
    assert result["architecture_search"]["strategy"] == "exhaustive"
    assert result["architecture_search"]["beam_search_applied"] is False
    assert result["trace"]["beam_search_applied"] is False
    assert result["trace"]["h10_is_final_validity_authority"] is True
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
        search_options=SearchOptions(strategy="exhaustive"),
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


def test_beam_is_default_and_preserves_the_public_best_architecture_contract(tmp_path):
    backend = FakeBackend()
    output = tmp_path / "beam.json"
    requirement = _requirement()
    result = run_e2e(requirement, backend=backend, output_path=output)

    assert SearchOptions() == SearchOptions(strategy="beam", beam_width=8)
    assert result["status"] == "SUCCESS"
    assert backend.calls == [
        "config", "workload", "features", "technical", "drive_catalog", "handoff",
        "hardware_catalog", "beam",
    ]
    assert result["schema_version"] == "1.0"
    assert result["pipeline_version"] == "1.1"
    best = result["best_architecture"]
    beam_best = backend.beam_result["best_validated_architecture"]
    assert best["architecture_id"] == "A2"  # Higher H9 score on INVALID A1 cannot win.
    assert best["score"] == beam_best["h9"]
    assert best["validation"] == beam_best["h10"]
    assert best["validation"]["valid"] is True
    assert best["validation"]["decision"] == "VALID"
    assert best["architecture"] == {
        "architecture_id": "A2",
        "case_id": beam_best["case_id"],
        "state": beam_best["state"],
        "search_provenance": beam_best["search_provenance"],
    }
    assert result["trace"]["beam_search_applied"] is True
    assert result["trace"]["h10_is_final_validity_authority"] is True
    search = result["architecture_search"]
    assert search["strategy"] == "beam"
    assert search["beam_search_applied"] is True
    assert search["beam_width"] == 8
    assert search["beam_status"] == "VALID_ARCHITECTURE_FOUND"
    assert search["global_infeasibility_claimed"] is False
    for field in ("summary", "heuristic_policy", "search_trace", "lookahead_trace"):
        assert search[field] == backend.beam_result[field]
    assert "H10" in search["selection_rule"]
    assert "Beam search domain" in search["selection_rule"]
    assert "architectures" not in search  # Do not duplicate the complete Beam pool.
    assert json.loads(output.read_text(encoding="utf-8"))["best_architecture"] == best
    backend.beam_result["best_validated_architecture"]["search_provenance"]["trace"].clear()
    assert best["architecture"]["search_provenance"]["trace"] == [{"depth": 6}]
    assert requirement == _requirement()


def test_beam_propagates_top_k_width_and_shared_path_limit(tmp_path):
    backend = FakeBackend()
    limits = PipelineLimits(top_k=7, max_paths_per_variant=3, max_role_options_per_role=1, max_architectures=1)
    options = SearchOptions(beam_width=13)
    result = run_e2e(_requirement(), backend=backend, output_path=tmp_path / "result.json", limits=limits, search_options=options)

    handoff, catalog, received_limits, received_options = backend.beam_parameters
    assert handoff["requested_top_k"] == 7
    assert catalog == {"catalog_kind": "fake"}
    assert received_limits is limits
    assert received_options is options
    assert result["candidate_space"]["requested_top_k"] == 7
    assert result["architecture_search"]["actual_top_k"] == {"mdt": 1, "ost": 1}
    assert result["architecture_search"]["beam_width"] == 13
    assert result["architecture_search"]["max_paths_per_variant"] == 3
    assert result["trace"]["search_options"] == {"strategy": "beam", "beam_width": 13}


def test_frozen_backend_delegates_only_the_existing_beam_api_arguments():
    calls = []
    expected = {"status": "test result"}
    def beam_search(**kwargs):
        calls.append(kwargs)
        return expected
    backend = FrozenRuntimeBackend.__new__(FrozenRuntimeBackend)
    backend.beam_module = SimpleNamespace(beam_search_architectures=beam_search)
    handoff, catalog = {"requested_top_k": 5}, {"catalog": "test"}
    result = backend.run_beam_search(handoff, catalog, PipelineLimits(max_paths_per_variant=3), SearchOptions(beam_width=11))
    assert result is expected
    assert calls == [{"handoff": handoff, "hardware_catalog": catalog, "beam_width": 11, "max_paths_per_variant": 3}]


def test_beam_no_solution_is_a_bounded_engineering_outcome(tmp_path):
    backend = FakeBackend(valid_ids=())
    output = tmp_path / "no_solution.json"
    result = run_e2e(_requirement(), backend=backend, output_path=output, limits=PipelineLimits(top_k=5, max_paths_per_variant=1))
    assert result["status"] == "NO_VALID_ARCHITECTURE_WITHIN_SEARCH_DOMAIN"
    assert result["failure_stage"] == "BEAM_SEARCH"
    assert result["best_architecture"] is None
    assert result["architecture_search"]["beam_status"] == "NO_VALID_ARCHITECTURE_FOUND_WITHIN_SEARCH_DOMAIN"
    assert result["architecture_search"]["global_infeasibility_claimed"] is False
    assert "Top-K limité à 5" in result["message"]
    assert "beam_width=8" in result["message"]
    assert "max_paths_per_variant=1" in result["message"]
    assert "ne constitue pas une preuve d'infaisabilité globale" in result["message"]
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == result["status"]
    assert not {"h8", "h9", "h10"}.intersection(backend.calls)


def test_beam_runtime_exception_is_not_confused_with_no_solution(tmp_path):
    backend = FakeBackend(beam_error=RuntimeError("integration broke"))
    result = run_e2e(_requirement(), backend=backend, output_path=tmp_path / "error.json")
    assert result["status"] == "PIPELINE_ERROR"
    assert result["failure_stage"] == "BEAM_SEARCH"
    assert result["message"] == "RuntimeError: integration broke"
    assert result["best_architecture"] is None
    assert result["trace"]["beam_search_applied"] is True


@pytest.mark.parametrize("valid,decision", [(False, "INVALID"), (True, "INVALID"), (False, "VALID"), (1, "VALID"), ("true", "VALID")])
def test_beam_adapter_never_turns_an_invalid_h10_record_into_success(tmp_path, monkeypatch, valid, decision):
    backend = FakeBackend()
    original = backend.run_beam_search
    def bad_h10(*args):
        result = original(*args)
        result["best_validated_architecture"]["h10"].update(valid=valid, decision=decision)
        return result
    monkeypatch.setattr(backend, "run_beam_search", bad_h10)
    result = run_e2e(_requirement(), backend=backend, output_path=tmp_path / "bad.json")
    assert result["status"] == "PIPELINE_ERROR"
    assert result["failure_stage"] == "BEAM_SEARCH"
    assert "H10 VALID" in result["message"]
    assert result["best_architecture"] is None


@pytest.mark.parametrize("status,best_missing", [("VALID_ARCHITECTURE_FOUND", True), ("NO_VALID_ARCHITECTURE_FOUND_WITHIN_SEARCH_DOMAIN", False)])
def test_either_missing_best_or_no_solution_status_prevents_beam_success(tmp_path, monkeypatch, status, best_missing):
    backend = FakeBackend()
    original = backend.run_beam_search
    def no_solution(*args):
        result = original(*args)
        result["status"] = status
        if best_missing:
            result["best_validated_architecture"] = None
        return result
    monkeypatch.setattr(backend, "run_beam_search", no_solution)
    result = run_e2e(_requirement(), backend=backend, output_path=tmp_path / "none.json")
    assert result["status"] == "NO_VALID_ARCHITECTURE_WITHIN_SEARCH_DOMAIN"
    assert result["best_architecture"] is None
    assert result["architecture_search"]["global_infeasibility_claimed"] is False


@pytest.mark.parametrize("field", ["h9", "h10"])
def test_beam_adapter_refuses_a_score_or_validation_for_a_different_architecture(tmp_path, monkeypatch, field):
    backend = FakeBackend()
    original = backend.run_beam_search
    def different_id(*args):
        result = original(*args)
        result["best_validated_architecture"][field]["architecture_id"] = "OTHER"
        return result
    monkeypatch.setattr(backend, "run_beam_search", different_id)
    result = run_e2e(_requirement(), backend=backend, output_path=tmp_path / "ids.json")
    assert result["status"] == "PIPELINE_ERROR"
    assert result["failure_stage"] == "BEAM_SEARCH"
    assert result["best_architecture"] is None
    assert "identités Beam, H9 et H10" in result["message"]


def test_beam_adapter_exception_preserves_its_type_and_message(tmp_path, monkeypatch):
    backend = FakeBackend()
    original = backend.run_beam_search
    def missing_score(*args):
        result = original(*args)
        del result["best_validated_architecture"]["h9"]
        return result
    monkeypatch.setattr(backend, "run_beam_search", missing_score)
    result = run_e2e(_requirement(), backend=backend, output_path=tmp_path / "missing.json")
    assert result["status"] == "PIPELINE_ERROR"
    assert result["failure_stage"] == "BEAM_SEARCH"
    assert result["message"] == "KeyError: 'h9'"


@pytest.mark.parametrize("strategy", ["beam", "exhaustive"])
def test_file_api_writes_json_for_both_strategies(tmp_path, monkeypatch, strategy):
    backend = FakeBackend()
    monkeypatch.setattr("e2e_pipeline.end_to_end_pipeline.FrozenRuntimeBackend", lambda root: backend)
    requirement_path = tmp_path / "requirement.json"
    requirement_path.write_text(json.dumps(_requirement()), encoding="utf-8")
    output_path = tmp_path / "result.json"
    result = run_e2e_from_file(requirement_path, output_path=output_path, search_options=SearchOptions(strategy=strategy, beam_width=11))
    stored = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["status"] == stored["status"] == "SUCCESS"
    assert result["best_architecture"] == stored["best_architecture"]
    assert stored["architecture_search"]["strategy"] == strategy
    assert stored["trace"]["beam_search_applied"] is (strategy == "beam")
    assert stored["trace"]["h10_is_final_validity_authority"] is True
    assert stored["best_architecture"]["validation"]["valid"] is True


@pytest.mark.parametrize("strategy", ["other", "BEAM", "", None, True])
def test_search_options_rejects_unknown_strategy(strategy):
    with pytest.raises(ValueError, match="strategy"):
        SearchOptions(strategy=strategy)


@pytest.mark.parametrize("width", [0, -1, True, False, 8.0, "8", None])
def test_search_options_requires_positive_integer_width_without_bool_coercion(width):
    with pytest.raises(ValueError, match="beam_width"):
        SearchOptions(beam_width=width)
