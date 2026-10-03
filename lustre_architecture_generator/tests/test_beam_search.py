from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import full_architecture.beam_search as beam
from full_architecture.architecture_scoring import assert_scoring_result_valid, score_generated_architectures
from full_architecture.catalog_loader import load_reference_catalog
from full_architecture.compatibility_rules import find_compatible_hardware_paths
from full_architecture.full_architecture_generator import generate_full_architectures
from full_architecture.full_architecture_validator import validate_complete_architecture, validate_generated_architectures
from full_architecture.handoff_contract import ArchitectureHandoffError, assert_valid_architecture_handoff
from full_architecture.hardware_schema import HardwareSchemaError
from full_architecture.protection_arithmetic import assert_protection_result_valid, enumerate_candidate_protections
from test_full_architecture_validator import candidate, handoff as h10_handoff


def handoff(count: int = 2) -> dict:
    """Extend the existing H10 controlled candidates to the canonical H1 schema."""
    result = h10_handoff()
    result.update({
        "stage": "ranking_to_full_architecture", "requested_top_k": count,
        "actual_top_k": {"mdt": count, "ost": count},
        "unit_contract": {"throughput_rate": "GB/s"},
        "contract_invariants": {
            "pre_raid_counts_are_lower_bounds": True, "hard_feasibility_precedes_ml": True,
            "no_final_hardware_selected": True, "beam_search_not_applied": True,
        },
    })
    for role in ("MDT", "OST"):
        candidates = []
        for rank in range(1, count + 1):
            value = candidate(role, f"{role}_{rank:02}")
            value["selection_rank"] = rank
            value["ranking"]["ml_rank"] = rank
            checks = (["capacity", "read_iops", "write_iops"] if role == "MDT" else
                      ["capacity", "read_bandwidth", "write_bandwidth", "total_bandwidth"])
            value["deterministic_filter_evidence"] = {
                "status": "feasible",
                **{k: {"satisfied": True} for k in checks + ["raw_budget_lower_bound", "raw_power_lower_bound"]},
            }
            candidates.append(value)
        result[f"{role.lower()}_candidates"] = candidates
    assert_valid_architecture_handoff(result)
    return result


def catalog() -> dict:
    result = load_reference_catalog()
    keep = {
        "servers": {"SRV_MDS_DENSE_01", "SRV_OSS_BALANCED_01"},
        "controllers": {"CTRL_SAS12_16P_01"},
        "enclosures": {"ENC_SAS_24X25_01"},
        "networks": {"NET_ETH100_01"},
        "ha_profiles": {"HA_NONE", "HA_ACTIVE_PASSIVE"},
        "protection_profiles": {"PROT_RAID1_2", "PROT_RAID6_10"},
    }
    for section, ids in keep.items():
        result[section] = [item for item in result[section] if item["id"] in ids]
    return result


def search(h=None, c=None, width=8, paths=2):
    return beam.beam_search_architectures(
        handoff=h if h is not None else handoff(),
        hardware_catalog=c if c is not None else catalog(),
        beam_width=width, max_paths_per_variant=paths,
    )


@pytest.mark.parametrize("width", [0, -1, True, False, 1.5, "4", None, float("nan")])
def test_invalid_width_rejected(width):
    with pytest.raises(beam.BeamSearchError, match="beam_width"):
        search(width=width)


@pytest.mark.parametrize("paths", [0, -1, True, 1.5, "2", None])
def test_invalid_path_limit_rejected(paths):
    with pytest.raises(beam.BeamSearchError, match="max_paths_per_variant"):
        search(paths=paths)


def test_search_is_deterministic_except_elapsed_time():
    first, second = search(), search()
    first["summary"].pop("search_seconds")
    second["summary"].pop("search_seconds")
    assert first == second


def test_inputs_and_frozen_contracts_are_preserved():
    h, c = handoff(), catalog()
    before = copy.deepcopy((h, c))
    result = search(h, c)
    assert (h, c) == before
    assert h["contract_invariants"]["beam_search_not_applied"] is True
    assert result["beam_search_applied"] is True
    assert_scoring_result_valid(result["scoring_result"])
    assert all(row["h9"]["beam_search_applied"] is False for row in result["architectures"])
    assert all(row["h10"]["beam_search_applied"] is False for row in result["architectures"])
    exhaustive = generate_full_architectures(handoff=h, hardware_catalog=c, max_paths_per_variant=1)
    assert exhaustive["generation_contract"]["beam_search_applied"] is False


def test_final_recommendation_really_passes_h10():
    h, c = handoff(), catalog()
    result = search(h, c)
    selected = result["best_validated_architecture"]
    assert selected is not None
    decision = validate_complete_architecture(architecture=selected, handoff=h, hardware_catalog=c)
    assert decision["decision"] == "VALID"
    assert decision["valid"] is True
    assert decision["violations"] == []
    assert selected["h10"]["recomputed"] is not None
    assert result["summary"]["h10_calls"] == len(result["architectures"])


def test_no_partial_state_is_scored_or_validated(monkeypatch):
    real_score, real_validate = beam.score_generated_architectures, beam.validate_complete_architecture
    observed = []

    def score(**kwargs):
        for record in kwargs["generation_result"]["architectures"]:
            assert record["state"]["stage"] == "COMPLETE"
            assert record["state"]["validation"]["status"] == "PENDING_FULL_VALIDATOR"
        return real_score(**kwargs)

    def validate(**kwargs):
        state = kwargs["architecture"]["state"]
        assert state["stage"] == "COMPLETE"
        assert state["validation"]["is_valid"] is False
        observed.append(state)
        return real_validate(**kwargs)

    monkeypatch.setattr(beam, "score_generated_architectures", score)
    monkeypatch.setattr(beam, "validate_complete_architecture", validate)
    result = search()
    assert len(observed) == result["summary"]["h10_calls"] > 0
    for entry in result["best_validated_architecture"]["search_provenance"]["trace"]:
        assert entry["physical_validation_status"] == "PENDING_FULL_VALIDATOR"
    assert [s["physical_stage"] for s in result["best_validated_architecture"]["search_provenance"]["trace"]] == [
        "EMPTY", "EMPTY", "EMPTY", "DRIVES_SELECTED", "PROTECTION_SELECTED", "PROTECTION_SELECTED",
    ]


def test_all_final_geometry_and_paths_come_from_h5_h6():
    h, c = handoff(), catalog()
    result = search(h, c, width=64)
    for record in result["architectures"]:
        selected = record["state"]["selected"]
        for role in ("MDT", "OST"):
            drive = selected[f"{role.lower()}_drive"]["drive_id"]
            value = next(v for v in h[f"{role.lower()}_candidates"] if v["identity"]["drive_id"] == drive)
            protection = selected[f"{role.lower()}_protection"]
            assert_protection_result_valid(protection)
            assert protection in enumerate_candidate_protections(
                candidate=value, protection_profiles=c["protection_profiles"],
                requirement=h["requirements"][f"{role}_requirement"],
            )
            assert selected[f"{role.lower()}_hardware_path"] in find_compatible_hardware_paths(
                candidate=value, protection_result=protection, role=role,
                hardware_catalog=c, ha_required=False, max_paths=2,
            )


@pytest.mark.parametrize("field,limit,reason", [
    ("max_budget_usd", 1.0, "budget_lower_bound_exceeded"),
    ("max_power_w", 1.0, "power_lower_bound_exceeded"),
])
def test_hard_budget_power_prunes_before_hardware(field, limit, reason, monkeypatch):
    h = handoff()
    h["requirements"]["constraints"][field] = limit
    monkeypatch.setattr(beam, "find_compatible_hardware_paths", lambda **kw: pytest.fail("H6 should not be reached"))
    result = search(h)
    assert result["status"] == beam.NO_VALID_ARCHITECTURE
    assert result["best_validated_architecture"] is None
    assert result["search_trace"][0]["pruning_reason_counts"][reason] == 2
    assert result["summary"]["pruning_reason_counts"][reason] == 6  # four look-ahead + two search attempts
    assert result["summary"]["h10_calls"] == 0
    assert result["global_infeasibility_claimed"] is False


def test_hardware_cost_is_included_in_early_bound():
    h = handoff()
    h["requirements"]["constraints"]["max_budget_usd"] = 5000.0
    result = search(h)
    assert result["search_trace"][-1]["search_stage"] == "MDT_DRIVE"
    assert result["search_trace"][-1]["hard_pruned"] > 0
    assert result["lookahead_trace"][0]["pruning_reason_counts"]["budget_lower_bound_exceeded"] > 0


def test_joint_cost_bound_and_boundary_are_safe():
    h, c = handoff(1), catalog()
    g = generate_full_architectures(handoff=h, hardware_catalog=c, max_paths_per_variant=1)
    cost = min(r["state"]["cost_power"]["total_cost_usd"] for r in g["architectures"])
    h["requirements"]["constraints"]["max_budget_usd"] = cost
    assert search(h, c, width=64, paths=1)["best_validated_architecture"] is not None
    h["requirements"]["constraints"]["max_budget_usd"] = cost - 1
    result = search(h, c, width=64, paths=1)
    assert result["status"] == beam.NO_VALID_ARCHITECTURE
    assert result["summary"]["hard_pruned"] > 0


def test_future_resources_do_not_cause_partial_rejection():
    result = search(width=64)
    assert result["best_validated_architecture"] is not None
    assert result["search_trace"][0]["hard_pruned"] == 0


def test_no_compatible_path_is_explicit():
    c = catalog()
    c["controllers"][0]["supported_protocols"] = ["NVME"]
    result = search(c=c)
    assert result["status"] == beam.NO_VALID_ARCHITECTURE
    assert result["summary"]["pruning_reason_counts"]["h6_no_compatible_path"] > 0


def test_h5_impossible_branch_is_counted():
    h = handoff()
    for value in h["mdt_candidates"]:
        value["pre_raid"]["provided_capacity_tib"] = 0.0
    result = search(h)
    assert result["status"] == beam.NO_VALID_ARCHITECTURE
    assert result["summary"]["pruning_reason_counts"]["h5_protection_impossible"] == 4


def test_invalid_geometry_is_rejected_at_catalog_boundary():
    c = catalog()
    c["protection_profiles"][0]["minimum_drives_per_group"] = 3
    with pytest.raises(HardwareSchemaError):
        search(c=c)


def test_invalid_handoff_is_rejected():
    h = handoff()
    h["mdt_candidates"][1]["identity"]["drive_id"] = h["mdt_candidates"][0]["identity"]["drive_id"]
    with pytest.raises(ArchitectureHandoffError, match="dupliqué"):
        search(h)


def test_missing_numeric_input_is_error_not_search_failure():
    h = handoff()
    del h["mdt_candidates"][0]["pre_raid"]["provided_read_iops"]
    with pytest.raises(beam.BeamSearchError, match="provided_read_iops"):
        search(h)


def test_stable_ties_and_unique_architecture_ids():
    first, second = search(width=64), search(width=64)
    assert [r["architecture_id"] for r in first["architectures"]] == [r["architecture_id"] for r in second["architectures"]]
    ids = [r["architecture_id"] for r in first["architectures"]]
    assert len(ids) == len(set(ids))
    for left, right in zip(first["architectures"], first["architectures"][1:]):
        if left["h9"]["score"] == right["h9"]["score"]:
            assert left["architecture_id"] < right["architecture_id"]
    assert search(width=1)["best_validated_architecture"]["state"]["selected"]["mdt_drive"]["drive_id"] == "MDT_01"


def test_wide_beam_matches_full_exhaustive_pool_and_best():
    h, c = handoff(), catalog()
    g = generate_full_architectures(handoff=h, hardware_catalog=c, max_paths_per_variant=2)
    scored = score_generated_architectures(generation_result=g, handoff=h)
    decisions = validate_generated_architectures(generation_result=g, handoff=h, hardware_catalog=c)
    valid_ids = {d["architecture_id"] for d in decisions["architectures"] if d["decision"] == "VALID"}
    baseline = next(s for s in scored["architectures"] if s["architecture_id"] in valid_ids)
    result = search(h, c, width=64)
    assert set(r["architecture_id"] for r in result["architectures"]) == set(r["architecture_id"] for r in g["architectures"])
    assert result["best_validated_architecture"]["architecture_id"] == baseline["architecture_id"]
    assert result["summary"]["best_final_h9_score"] == baseline["score"]
    assert result["summary"]["width_pruned"] == 0


def test_width_reduces_upstream_expansion_not_only_final_pool(monkeypatch):
    h = handoff(8)
    observed_h5 = []
    real = beam.enumerate_candidate_protections

    def record(**kwargs):
        observed_h5.append(kwargs["candidate"]["identity"]["drive_id"])
        return real(**kwargs)

    monkeypatch.setattr(beam, "enumerate_candidate_protections", record)
    narrow = search(h, width=1)
    assert set(observed_h5) == {f"{role}_{rank:02}" for role in ("MDT", "OST") for rank in range(1, 9)}
    wide = search(h, width=64)
    assert narrow["summary"]["branches_explored"] < wide["summary"]["branches_explored"]
    # V2 prepares linear role envelopes once, independent of B; it still avoids
    # the Cartesian COMPLETE pool and bounds all six main expansion levels.
    assert narrow["summary"]["h5_calls"] == wide["summary"]["h5_calls"]
    assert narrow["summary"]["h6_calls"] == wide["summary"]["h6_calls"]
    assert narrow["summary"]["h10_calls"] == 1
    assert narrow["summary"]["max_active_states"] <= 1
    assert narrow["summary"]["max_selection_buffer"] <= 2


def test_high_soft_score_cannot_override_h10(monkeypatch):
    real = beam.validate_complete_architecture

    def reject(**kwargs):
        record = copy.deepcopy(kwargs["architecture"])
        record["state"]["cost_power"]["total_cost_usd"] += 100
        return real(**{**kwargs, "architecture": record})

    monkeypatch.setattr(beam, "validate_complete_architecture", reject)
    result = search()
    assert result["status"] == beam.NO_VALID_ARCHITECTURE
    assert result["best_validated_architecture"] is None
    assert result["summary"]["h10_calls"] > 0
    assert result["summary"]["valid_architecture_count"] == 0
    assert all(r["h10"]["decision"] == "INVALID" for r in result["architectures"])


def test_required_ha_is_preserved():
    h = handoff()
    h["requirements"]["constraints"]["ha_required"] = True
    result = search(h, width=64)
    assert result["best_validated_architecture"] is not None
    for record in result["architectures"]:
        assert record["h10"]["hard_requirements"]["ha"]["mdt_satisfied"]
        assert record["h10"]["hard_requirements"]["ha"]["ost_satisfied"]


def test_exploration_and_pruning_counters_reconcile_and_output_is_json():
    result = search(handoff(8), width=2)
    stages, summary = result["search_trace"], result["summary"]
    all_stages = stages + result["lookahead_trace"]
    assert sum(s["branches_created"] for s in all_stages) == summary["branches_created"]
    assert sum(s["hard_pruned"] for s in all_stages) == summary["hard_pruned"]
    assert sum(s["width_pruned"] for s in stages) == summary["width_pruned"]
    assert all(s["retained"] <= 2 for s in stages)
    assert summary["branches_pruned"] == summary["hard_pruned"] + summary["width_pruned"]
    json.dumps(result, allow_nan=False)


def test_heuristic_formula_is_exact_and_pool_independent():
    node = beam.BeamState({}, mdt={"candidate": handoff()["mdt_candidates"][0]})
    constraints = {"max_budget_usd": 900, "max_power_w": 30}
    weights = {"performance_priority": 0.4, "cost_priority": 0.2,
               "power_priority": 0.2, "reliability_priority": 0.2}
    info = beam.beam_heuristic(node, preference_weights=weights, constraints=constraints, ha_modes={})
    assert info["components"] == {"ranking": 1, "performance": 0, "cost": .5, "power": .5, "reliability": 0}
    assert info["score"] == pytest.approx(.25 + .75 * (.2 * .5 + .2 * .5))
