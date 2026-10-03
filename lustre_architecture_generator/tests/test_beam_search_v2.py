from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

from test_beam_search import beam, catalog, handoff, search
from test_beam_search_benchmark import benchmark, reference

from diagnose_beam_search import diagnose, load_v1
from full_architecture.full_architecture_generator import enumerate_role_options, generate_full_architectures
from full_architecture.full_architecture_validator import validate_generated_architectures
from full_architecture.architecture_scoring import score_generated_architectures
from full_architecture.handoff_contract import ArchitectureHandoffError


def future_hardware_fixture():
    h, c = handoff(), catalog()
    # The ML-first OST uses many small drives. Raw drive power fits, but its
    # controllers/enclosures exceed the case limit; the second drive fits.
    first = h["ost_candidates"][0]["pre_raid"]
    first["provided_capacity_tib"] = 4.0
    h["requirements"]["OST_requirement"]["required_usable_capacity_tib"] = 120.0
    c["protection_profiles"] = [p for p in c["protection_profiles"] if p["raid_level"] == "RAID6"]
    h["requirements"]["constraints"]["max_power_w"] = 7000.0
    h["requirements"]["constraints"]["max_budget_usd"] = 200000.0
    return h, c


def test_future_hardware_fixture_is_lost_by_v1_and_recovered_by_v2():
    h, c = future_hardware_fixture()
    old = load_v1().beam_search_architectures(handoff=h, hardware_catalog=c, beam_width=1, max_paths_per_variant=1)
    new = search(h, c, width=1, paths=1)
    assert old["best_validated_architecture"] is None
    assert old["search_trace"][-1]["search_stage"] == "OST_HARDWARE"
    selected = new["best_validated_architecture"]
    assert selected is not None
    assert selected["state"]["selected"]["ost_drive"]["drive_id"] == "OST_02"
    assert selected["h10"]["decision"] == "VALID"
    assert new["summary"]["h10_calls"] == 1


def test_mdt_first_budget_allocation_loses_pair_and_safe_future_bound_recovers_it():
    h, c = handoff(), catalog()
    c["protection_profiles"] = [p for p in c["protection_profiles"] if p["raid_level"] == "RAID6"]
    h["mdt_candidates"][0]["pre_raid"]["drive_level_cost_usd"] *= 3
    mdt = enumerate_role_options(handoff=h, hardware_catalog=c, role="MDT", max_paths_per_variant=1)
    ost = enumerate_role_options(handoff=h, hardware_catalog=c, role="OST", max_paths_per_variant=1)
    mdt_cost = {o["candidate"]["identity"]["drive_id"]: beam.role_option_cost_power(o)["cost_usd"] for o in mdt}
    ost_min = min(beam.role_option_cost_power(o)["cost_usd"] for o in ost)
    budget = (mdt_cost["MDT_01"]+mdt_cost["MDT_02"])/2+ost_min
    assert max(mdt_cost.values()) <= budget  # both MDT roles fit independently
    assert mdt_cost["MDT_02"]+ost_min < budget < mdt_cost["MDT_01"]+ost_min
    h["requirements"]["constraints"]["max_budget_usd"] = budget
    old = load_v1().beam_search_architectures(handoff=h, hardware_catalog=c, beam_width=1, max_paths_per_variant=1)
    new = search(h, c, width=1, paths=1)
    assert old["best_validated_architecture"] is None
    assert new["summary"]["pruning_reason_counts"]["completion_budget_lower_bound_exceeded"] > 0
    assert new["best_validated_architecture"]["state"]["selected"]["mdt_drive"]["drive_id"] == "MDT_02"
    assert new["best_validated_architecture"]["h10"]["decision"] == "VALID"


def test_renaming_case_does_not_change_search_choices():
    h, c = future_hardware_fixture()
    first = search(h, c, width=1)
    h["case_id"] = "ARBITRARY_OTHER_CASE"
    second = search(h, c, width=1)
    assert first["best_validated_architecture"]["state"]["selected"] == second["best_validated_architecture"]["state"]["selected"]
    assert first["search_trace"] == second["search_trace"]


def test_production_engine_has_no_exhaustive_or_benchmark_dependency():
    source = Path(beam.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
            assert not any("benchmark" in n or "diagnos" in n or "v1_reference" in n for n in names)
            assert "generate_full_architectures" not in names
            assert "enumerate_role_options" not in names
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert not node.value.startswith(("REQ_", "DRV_", "PROT_", "SRV_"))


def test_completion_heuristic_exact_formula_and_optimistic_components():
    envelopes = (
        {"cost_usd": 450, "power_w": 15, "performance_sum": .25, "metric_count": 1,
         "reliability_sum": 1, "ranking": 1, "option_count": 3},
        {"cost_usd": 450, "power_w": 15, "performance_sum": .5, "metric_count": 1,
         "reliability_sum": .5, "ranking": .5, "option_count": 4},
    )
    weights = {"performance_priority": .4, "cost_priority": .2, "power_priority": .2, "reliability_priority": .2}
    info = beam.beam_heuristic(beam.BeamState({}), preference_weights=weights,
                             constraints={"max_budget_usd": 900, "max_power_w": 30},
                             ha_modes={}, completion_envelopes=envelopes)
    assert info["components"] == {"ranking": .75, "performance": .375, "cost": .5, "power": .5, "reliability": .375}
    assert info["score"] == pytest.approx(.4*.375+.2*.5+.2*.5+.2*.375)
    assert info["completion_option_counts"] == [3, 4]


def test_prefix_envelopes_are_admissible_for_every_role_completion():
    h, c = handoff(), catalog()
    modes = {p["id"]: p["mode"] for p in c["ha_profiles"]}
    for role in ("MDT", "OST"):
        options = enumerate_role_options(handoff=h, hardware_catalog=c, role=role, max_paths_per_variant=2)
        index = {}
        choices = []
        for option in options:
            choice = {**option, "hardware_path_index": option["provenance"]["hardware_path_index"]}
            choices.append(choice)
            beam._add_completion(index, choice, modes)
        for choice in choices:
            exact = beam.role_option_cost_power(choice)
            full_key = beam._prefix_key(choice)
            for depth in range(4):
                envelope = index[full_key[:depth]]
                assert envelope["cost_usd"] <= exact["cost_usd"]
                assert envelope["power_w"] <= exact["power_w"]
                assert envelope["ranking"] >= 1/choice["candidate"]["selection_rank"]


def test_safe_completion_pruning_loses_no_exhaustive_valid_prefix():
    h, c = future_hardware_fixture()
    generated = generate_full_architectures(handoff=h, hardware_catalog=c, max_paths_per_variant=1)
    scored = score_generated_architectures(generation_result=generated, handoff=h)
    validated = validate_generated_architectures(generation_result=generated, handoff=h, hardware_catalog=c)
    report = diagnose(beam, h, c, 1, generated, scored, validated)
    assert report["exhaustive_valid_count"] > 0
    assert all(s.get("viable_hard_pruned", 0) == 0 for s in report["levels"])


def test_lookahead_never_builds_complete_cartesian_states(monkeypatch):
    h, c = handoff(8), catalog()
    observed = []
    real = beam.build_complete_state_from_choices
    def build(**kwargs):
        observed.append(kwargs)
        return real(**kwargs)
    monkeypatch.setattr(beam, "build_complete_state_from_choices", build)
    result = search(h, c, width=2)
    assert len(observed) == result["summary"]["complete_architectures_produced"] <= 2
    assert result["summary"]["lookahead_role_options"]["mdt"] > 2
    assert result["summary"]["lookahead_role_options"]["ost"] > 2
    assert result["summary"]["h5_calls"] == 8*2*2


def test_optimistic_completion_scores_do_not_increase_along_a_path():
    result = search(width=8)
    for architecture in result["architectures"]:
        scores = [s["heuristic"]["score"] for s in architecture["search_provenance"]["trace"]]
        assert all(child <= parent+1e-12 for parent, child in zip(scores, scores[1:]))


def test_reordered_noncanonical_handoff_is_rejected():
    h, c = handoff(5), catalog()
    h["mdt_candidates"].reverse()
    h["ost_candidates"].reverse()
    with pytest.raises(ArchitectureHandoffError, match="selection_rank"):
        search(h, c, width=4)


def test_quality_separates_search_regret_from_pool_normalization():
    ref = reference()
    result = search(width=4)
    quality = benchmark.compare_quality(result, ref)
    assert quality["common_score_regret"] == pytest.approx(quality["search_common_score_regret"]+quality["pool_normalization_common_score_regret"])
    assert quality["pool_normalization_common_score_regret"] >= -1e-12
    assert quality["search_common_score_regret"] >= -1e-12
    assert quality["did_beam_find_exhaustive_best"] == quality["best_id_matched"]
    assert quality["retained_exhaustive_valid_count"] == result["summary"]["valid_architecture_count"]


def test_wide_quality_reports_zero_search_loss_when_all_valid_survive():
    result = search(width=64)
    quality = benchmark.compare_quality(result, reference())
    assert quality["exhaustive_valid_recall"] == 1
    assert quality["search_common_score_regret"] == 0
    assert quality["pool_normalization_common_score_regret"] == 0
    assert quality["did_beam_find_exhaustive_best"] is True


def test_normalization_loss_is_not_mislabeled_as_missing_best_branch():
    result = search(width=64)
    ref = reference()
    worst_id = ref["valid_scores"][-1]["architecture_id"]
    result["best_validated_architecture"] = copy.deepcopy(next(r for r in result["architectures"] if r["architecture_id"] == worst_id))
    quality = benchmark.compare_quality(result, ref)
    assert quality["exhaustive_best_survived_search"] is True
    assert quality["did_beam_find_exhaustive_best"] is False
    assert quality["search_common_score_regret"] == 0
    assert quality["pool_normalization_common_score_regret"] > 0


def test_before_after_requires_matching_configurations():
    old = {"cases": [{"case_id": "A", "requested_top_k": 5, "beam_width": 8,
                      "valid_architecture_count": 0, "runtime_seconds": .1, "quality": {}}]}
    new = {**old["cases"][0], "valid_architecture_count": 1, "quality": {"relative_quality": .9}}
    row = benchmark.before_after_rows(old, [new])[0]
    assert row["old_valid"] is False and row["new_valid"] is True
    assert row["old_quality"] is None and row["new_quality"] == .9
    with pytest.raises(ValueError, match="Missing BEFORE"):
        benchmark.before_after_rows(old, [{**new, "case_id": "B"}])
