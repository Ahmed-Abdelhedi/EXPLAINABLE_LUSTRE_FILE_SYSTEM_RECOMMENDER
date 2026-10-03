"""Verify the completed campaign against source inputs, CSV and independent audit."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path

import run_extended_benchmark as extended


def main():
    directory = extended.HERE
    payload = json.loads((directory / "extended_benchmark.json").read_text(encoding="utf-8"))
    audit = json.loads((directory / "no_solution_audit.json").read_text(encoding="utf-8"))
    rows = payload["cases"]
    ids = {r["case_id"] for r in payload["dataset"]["selected_cases"]}
    expected = set(itertools.product(ids, extended.KS, extended.WIDTHS))
    measured = {(r["case_id"], r["requested_top_k"], r["beam_width"]) for r in rows}
    assert payload["completed"] and measured == expected and len(rows) == len(expected)
    assert not ids & extended.DEVELOPMENT_IDS
    assert len(payload["baselines"]) == len(ids)*len(extended.KS)
    checks = ["Complete unique case x K x B matrix; all development cases excluded"]
    for row in rows:
        assert len(row["runtime_samples_seconds"]) == payload["config"]["repeats"]
        assert row["deterministic_repeats"] is True
        assert row["beam_search_applied"] is True
        assert row["peak_beam_states"] <= row["beam_width"]
        assert row["memory"]["max_selection_buffer"] <= row["beam_width"]+1
        assert row["complete_architectures"] == row["h10_calls"] <= row["beam_width"]
        assert 0 <= row["h10_valid_count"] <= row["h10_calls"]
        assert row["valid_found"] == (row["h10_valid_count"] > 0) == (row["selected_architecture_id"] is not None)
        assert row["branches_generated"] == row["main_search_branches_created"]+row["lookahead_branches_created"]
        assert row["hard_pruned"] + row["beam_width_pruned"] <= row["branches_generated"]
        q = row["quality"]
        if q["status"] == "COMPARED_IN_COMMON_POOL":
            assert row["exhaustive_status"] == "COMPLETE"
            assert abs(q["total_common_score_regret"]-q["search_common_score_regret"]-q["pool_normalization_common_score_regret"]) <= extended.EPS
            assert q["best_surviving_architecture_common_rank"] <= q["selected_architecture_common_rank"]
            assert 0 <= q["relative_quality"] <= 1+extended.EPS
        elif row["exhaustive_status"] == "SKIPPED_DOMAIN_TOO_LARGE":
            assert "relative_quality" not in q and "total_common_score_regret" not in q
    checks.append("Repeat fingerprints, Beam/COMPLETE/buffer bounds, counters and common-pool regret identities")
    for baseline in payload["baselines"]:
        if baseline["status"] == "SKIPPED_DOMAIN_TOO_LARGE":
            assert baseline["potential_pairs"] > payload["config"]["exhaustive_max_pairs"]
        elif baseline["status"] == "COMPLETE":
            assert baseline["potential_pairs"] <= payload["config"]["exhaustive_max_pairs"]
            assert baseline["potential_pairs"] == baseline["architectures_explored"] == baseline["h10_calls"]
    checks.append("Uncapped exhaustive pools below threshold; explicit skips above threshold")
    assert extended.aggregate(rows) == payload["aggregates"]
    with (directory / "extended_benchmark.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    csv_keys = {(r["case_id"], int(r["requested_top_k"]), int(r["beam_width"])) for r in csv_rows}
    assert csv_keys == expected and len(csv_rows) == len(rows)
    for row, csv_row in zip(rows, csv_rows):
        assert float(csv_row["runtime_total"]) == row["runtime_total"]
        assert int(csv_row["branches_generated"]) == row["branches_generated"]
    checks.append("Aggregates independently recomputed; per-configuration CSV matches JSON")
    missing = {(r["case_id"], r["requested_top_k"], r["beam_width"]) for r in rows if not r["valid_found"]}
    audited = {(r["case_id"], r["requested_top_k"], b) for r in audit["cases"] for b in r["missing_widths"]}
    assert missing == audited and len(missing) == audit["missing_configurations"]
    checks.append("Every missing Beam solution included in the independent feasibility audit")
    manifest = json.loads((directory / "frozen_sources_initial.json").read_text(encoding="utf-8"))
    change = payload.get("documented_beam_change")
    for name, digest in manifest.items():
        if change and name == change["path"]:
            assert digest == change["before_sha256"]
            assert hashlib.sha256((directory/"before_joint_fix/beam_search_v2_reference.py").read_bytes()).hexdigest() == digest
            assert hashlib.sha256((extended.REPO_ROOT/name).read_bytes()).hexdigest() == change["after_sha256"]
        else:
            assert hashlib.sha256((extended.REPO_ROOT/name).read_bytes()).hexdigest() == digest, name
    for name, digest in payload["input_hashes"].items():
        if Path(name).name == "run_extended_benchmark.py":
            assert hashlib.sha256((directory/"campaign_runner_snapshot.py").read_bytes()).hexdigest() == digest
        else:
            assert hashlib.sha256((extended.REPO_ROOT/name).read_bytes()).hexdigest() == digest, name
    models = json.loads((directory / "official_ranker_manifest.json").read_text(encoding="utf-8"))
    for model in models.values():
        for name, digest in model["files"].items():
            assert hashlib.sha256((extended.REPO_ROOT/name).read_bytes()).hexdigest() == digest, name
    checks.append("Only the explicitly documented Beam patch changed production sources; H5-H10, ranking, inputs and earlier V1/V2 artifacts retain their hashes")
    report = (directory/"extended_benchmark.md").read_text(encoding="utf-8")
    for letter in "ABCDEFGHIJKLMNOPQR":
        assert f"## {letter}." in report
    assert "PENDING" not in report
    checks.append("Final report contains all A-R sections with no unfinished placeholders")
    extended.dump(directory/"artifact_validation.json", {"status": "PASS", "configurations_checked": len(rows), "checks": checks})
    print(f"PASS: {len(checks)} checks; {len(rows)} configurations")


if __name__ == "__main__":
    main()
