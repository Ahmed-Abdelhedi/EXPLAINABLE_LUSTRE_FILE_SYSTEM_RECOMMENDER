"""Check evaluation math and sampling independently of the production heuristic."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "evaluation/architecture/beam_search/v2_extended/run_extended_benchmark.py"
SPEC = importlib.util.spec_from_file_location("extended_beam_evaluation", RUNNER)
extended = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extended)


@pytest.mark.parametrize("values,fraction,expected", [([1, 2, 3, 4, 5], .95, 4.8), ([7], .9, 7), ([], .9, None)])
def test_quantiles_are_linear_and_explicit_when_empty(values, fraction, expected):
    assert extended.percentile(values, fraction) == (pytest.approx(expected) if expected is not None else None)


def test_sample_is_order_independent_excludes_development_and_includes_extremes():
    dataset = json.loads((extended.PROJECT_ROOT / "output/lustre_architecture_dataset.json").read_text(encoding="utf-8"))
    unchanged = copy.deepcopy(dataset)
    sample, info = extended.select_cases(dataset, 100)
    reversed_sample, reversed_info = extended.select_cases(list(reversed(dataset)), 100)
    assert sample == reversed_sample
    assert info == reversed_info
    assert dataset == unchanged
    assert len(sample) == len({c["case_id"] for c in sample}) == 100
    assert not ({c["case_id"] for c in sample} & extended.DEVELOPMENT_IDS)
    for name in info["tertile_thresholds"]:
        population = info["distributions"]["population"]["numeric"][name]
        selected = info["distributions"]["sample"]["numeric"][name]
        assert (selected["min"], selected["max"]) == (population["min"], population["max"])
        assert all(selected["bands"].get(label, 0) > 0 for label in ("low", "medium", "high"))
    assert set(info["distributions"]["sample"]["ha_required"]) == {"true", "false"}
    assert "critical" in info["distributions"]["sample"]["ost_load"]


def measured_row(k, b, *, valid=True, common_score=.9, search_regret=.1, normalization_regret=0):
    q = {
        "status": "COMPARED_IN_COMMON_POOL", "common_h9_score": common_score,
        "relative_quality": common_score, "search_common_score_regret": search_regret,
        "pool_normalization_common_score_regret": normalization_regret,
        "exhaustive_best_survived_search": search_regret == 0,
        "did_beam_find_exhaustive_best": search_regret == normalization_regret == 0,
        "selected_architecture_common_rank": 2,
    } if valid else {"status": "BEAM_FOUND_NO_VALID_ARCHITECTURE"}
    return {
        "case_id": "CASE", "requested_top_k": k, "beam_width": b,
        "status": "VALID_ARCHITECTURE_FOUND" if valid else "NO_VALID_ARCHITECTURE_FOUND_WITHIN_SEARCH_DOMAIN",
        "valid_found": valid, "runtime_total": float(b), "branches_generated": b*10,
        "hard_pruned": b*2, "beam_width_pruned": b*3, "complete_architectures": b,
        "h10_calls": b, "h10_valid_count": b if valid else 0,
        "selected_architecture_id": f"ARCH_{b}" if valid else None,
        "quality": q, "exhaustive_status": "COMPLETE", "exhaustive_valid_count": 50,
        "memory": dict.fromkeys(("max_cached_h5_entries", "max_cached_h6_entries", "max_beam_states",
                                 "role_completion_envelope_count", "max_selection_buffer"), 1),
    }


def all_rows():
    return [measured_row(k, b) for k in extended.KS for b in extended.WIDTHS]


def test_aggregate_uses_case_distribution_and_generated_branch_denominator():
    first = measured_row(5, 1)
    second = measured_row(5, 1, valid=False)
    second["runtime_total"] = 3.
    second["branches_generated"] = 90
    second["hard_pruned"] = 18
    second["beam_width_pruned"] = 27
    row = extended.aggregate([first, second])[0]
    assert row["valid_cases"] == 1
    assert row["total_cases"] == 2
    assert row["feasible_solution_rate"] == .5
    assert row["runtime_mean"] == row["runtime_median"] == 2
    assert row["runtime_p95"] == pytest.approx(2.9)
    assert row["branches_mean"] == row["branches_median"] == 50
    assert row["hard_pruning_pct"] == 20
    assert row["width_pruning_pct"] == 30


def test_validity_monotonicity_is_separate_from_quality_monotonicity():
    rows = all_rows()
    rows[1] = measured_row(5, 4, common_score=.8, search_regret=.2)
    summary, sensitivity, _ = extended.analyze(rows, [])
    case = sensitivity[0]
    assert case["validity_monotonicity"] is True
    assert case["selected_quality_monotonicity"] is False
    assert case["quality_counterexamples"][0] == {
        "lower_b": 1, "higher_b": 4, "lower_quality": .9, "higher_quality": .8,
    }
    assert summary["validity_monotone_groups"] == 4


def test_validity_failure_and_missing_quality_remain_explicit():
    rows = all_rows()
    rows[1] = measured_row(5, 4, valid=False)
    for row in rows[5:10]:
        row["quality"] = {"status": "NO_EXHAUSTIVE_REFERENCE"}
        row["exhaustive_status"] = "SKIPPED_DOMAIN_TOO_LARGE"
        row["exhaustive_valid_count"] = None
    summary, sensitivity, _ = extended.analyze(rows, [])
    assert sensitivity[0]["validity_monotonicity"] is False
    assert sensitivity[0]["validity_counterexamples"] == [{"lower_b": 1, "higher_b": 4}]
    assert sensitivity[1]["selected_quality_monotonicity"] is None
    assert summary["quality_comparisons"] == 14
    assert len(summary["feasible_exhaustive_but_beam_missing"]) == 1


def test_loss_attribution_uses_two_regrets_and_tolerance():
    rows = all_rows()
    rows[0] = measured_row(5, 1, common_score=.8, search_regret=.1, normalization_regret=.1)
    rows[1] = measured_row(5, 4, common_score=.9, search_regret=0, normalization_regret=.1)
    rows[2] = measured_row(5, 8, common_score=1, search_regret=1e-14, normalization_regret=0)
    summary, _, _ = extended.analyze(rows, [])
    losses = summary["loss_distribution"]
    assert losses["both"]["count"] == losses["normalization_only"]["count"] == losses["no_measurable_loss"]["count"] == 1
    assert losses["search_only"]["count"] == 17
    assert sum(v["fraction"] for v in losses.values()) == pytest.approx(1)


def test_quality_does_not_invent_scores_for_skipped_reference():
    result = {"best_validated_architecture": None}
    assert extended.quality(result, {"status": "SKIPPED_DOMAIN_TOO_LARGE"}) == {"status": "NO_EXHAUSTIVE_REFERENCE"}


def test_cache_observation_leaves_engine_and_trace_unchanged():
    import sys
    from test_beam_search import catalog, handoff
    previous = sys.gettrace()
    code = extended.existing.beam_search_architectures.__code__
    observed, result = extended.observe_caches(handoff(), catalog(), 2)
    assert sys.gettrace() is previous
    assert extended.existing.beam_search_architectures.__code__ is code
    assert observed["max_cached_h5_entries"] == result["summary"]["h5_calls"]
    assert observed["max_cached_h6_entries"] == result["summary"]["h6_calls"]
    assert observed["role_completion_envelope_count"] > 0


def test_repeat_fingerprint_detects_changes_beyond_selected_id():
    result = {"status": "VALID_ARCHITECTURE_FOUND", "summary": {"best_architecture_id": "A", "search_seconds": 1., "h10_calls": 1},
              "architectures": [{"architecture_id": "A", "h9": {"score": .8}, "h10": {"decision": "VALID"}}]}
    another = copy.deepcopy(result)
    another["summary"]["search_seconds"] = 99.
    assert extended.fingerprint(result) == extended.fingerprint(another)
    another["architectures"][0]["h9"]["score"] = .9
    assert extended.fingerprint(result) != extended.fingerprint(another)
