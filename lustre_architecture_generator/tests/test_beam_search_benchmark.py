from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "evaluation/architecture/beam_search"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import benchmark_beam_search as benchmark
from test_beam_search import catalog, handoff, search


def reference(max_pairs=10000):
    return benchmark.exhaustive_reference(
        handoff=handoff(), hardware_catalog=catalog(), max_paths_per_variant=2, max_pairs=max_pairs,
    )


def test_exhaustive_reference_is_complete_and_uncapped():
    result = reference()
    assert result["status"] == "COMPLETE"
    assert result["h10_calls"] == result["potential_pairs"]
    assert result["valid_count"] == result["potential_pairs"]


def test_baseline_pair_limit_skips_before_generation(monkeypatch):
    monkeypatch.setattr(benchmark, "generate_full_architectures", lambda **kw: pytest.fail("Baseline must be skipped"))
    result = reference(1)
    assert result["status"] == "SKIPPED_PAIR_LIMIT"
    assert "runtime_seconds" not in result
    assert result["potential_pairs"] > 1


def test_quality_uses_exhaustive_pool_and_actual_beam_winner():
    ref = reference()
    worst = ref["valid_scores"][-1]
    result = {"best_validated_architecture": {"architecture_id": worst["architecture_id"], "h9": {"score": 1.0}}}
    quality = benchmark.compare_quality(result, ref)
    assert quality["common_h9_score"] == worst["score"]
    assert quality["exhaustive_valid_rank"] == len(ref["valid_scores"])
    assert quality["common_score_regret"] == pytest.approx(ref["best_h9_score"] - worst["score"])
    assert quality["common_score_regret"] > 0


def test_quality_rejects_choice_outside_reference():
    with pytest.raises(RuntimeError, match="outside"):
        benchmark.compare_quality({"best_validated_architecture": {"architecture_id": "INVENTED"}}, reference())


def test_no_solution_and_skipped_quality_are_explicit():
    assert benchmark.compare_quality({"best_validated_architecture": None}, reference())["status"] == "BEAM_FOUND_NO_VALID_ARCHITECTURE"
    assert benchmark.compare_quality(search(), {"status": "SKIPPED_PAIR_LIMIT"})["status"] == "NO_EXHAUSTIVE_REFERENCE"


def test_benchmark_records_counts_quality_and_repeats(tmp_path):
    ref = reference()
    row, result = benchmark.benchmark_case(
        handoff=handoff(), hardware_catalog=catalog(), beam_width=8,
        max_paths_per_variant=2, reference=ref, repeats=2,
    )
    assert len(row["runtime_samples_seconds"]) == 2
    assert row["h10_calls"] == result["summary"]["h10_calls"]
    assert row["quality"]["status"] == "COMPARED_IN_COMMON_POOL"
    payload = {"config": {"case_ids": [row["case_id"]], "max_paths_per_variant": 2,
                          "repeats": 2, "exhaustive_max_pairs": 10000},
               "cases": [row], "aggregates": benchmark.aggregate([row]), "baselines": []}
    path = tmp_path / "results.json"
    benchmark.write_outputs(path, payload)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert path.with_suffix(".csv").exists()
    assert "common-pool" in path.with_suffix(".md").read_text(encoding="utf-8")


def test_aggregate_keeps_execution_errors_in_feasibility_denominator():
    ref = reference()
    row, _ = benchmark.benchmark_case(handoff=handoff(), hardware_catalog=catalog(),
                                    beam_width=8, max_paths_per_variant=2, reference=ref)
    failure = {"case_id": "ERROR", "requested_top_k": row["requested_top_k"],
               "beam_width": 8, "status": "EXECUTION_ERROR"}
    summary = benchmark.aggregate([row, failure])[0]
    assert summary["cases"] == 2
    assert summary["errors"] == 1
    assert summary["feasible_solution_rate"] == .5
