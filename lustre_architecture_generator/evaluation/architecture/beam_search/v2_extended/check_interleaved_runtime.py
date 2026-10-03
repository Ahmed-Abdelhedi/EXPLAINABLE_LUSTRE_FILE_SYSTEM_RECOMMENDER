"""Measure before/after in one process; cross-campaign wall times vary."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from statistics import fmean, median
from time import perf_counter

import run_extended_benchmark as extended


def main():
    directory = extended.HERE
    before_path = directory/"before_joint_fix/beam_search_v2_reference.py"
    name = "full_architecture._beam_v2_before_joint_runtime"
    spec = importlib.util.spec_from_file_location(name, before_path)
    before = importlib.util.module_from_spec(spec)
    sys.modules[name] = before
    spec.loader.exec_module(before)
    sample = json.loads((directory/"sample_manifest.json").read_text(encoding="utf-8"))["selected_cases"]
    case_ids = [r["case_id"] for r in sorted(sample, key=lambda r: hashlib.sha256(r["case_id"].encode()).hexdigest())][:9]
    case_ids = sorted(set(case_ids + ["REQ_000267"]))
    dataset = json.loads((extended.PROJECT_ROOT/"output/lustre_architecture_dataset.json").read_text(encoding="utf-8"))
    by_id = {c["case_id"]: c for c in dataset}
    drives = json.loads((extended.PROJECT_ROOT/"data/catalogue_drives_ready_final.json").read_text(encoding="utf-8"))
    hardware = extended.existing.load_reference_catalog()
    rows = []
    functions = {"before": before.beam_search_architectures, "after": extended.existing.beam_search_architectures}
    for case_id in case_ids:
        for k in extended.KS:
            handoff = extended.existing.build_runtime_handoff(architecture=by_id[case_id], catalog=drives, top_k=k)
            samples = {phase: [] for phase in functions}
            results = {}
            for repetition in range(2):
                for phase in (("before", "after") if repetition == 0 else ("after", "before")):
                    started = perf_counter()
                    result = functions[phase](handoff=handoff, hardware_catalog=hardware,
                                              beam_width=32, max_paths_per_variant=1)
                    samples[phase].append(perf_counter()-started)
                    results[phase] = result
            rows.append({"case_id": case_id, "k": k, "b": 32,
                         "runtime_samples": samples,
                         "before_median_seconds": median(samples["before"]),
                         "after_median_seconds": median(samples["after"]),
                         "ratio_after_over_before": median(samples["after"])/median(samples["before"]),
                         "before_valid": results["before"]["best_validated_architecture"] is not None,
                         "after_valid": results["after"]["best_validated_architecture"] is not None,
                         "before_branches": results["before"]["summary"]["branches_created"],
                         "after_branches": results["after"]["summary"]["branches_created"]})
        print(f"Interleaved runtime: {case_id}", flush=True)
    aggregates = [{"k": k, "pairs": sum(r["k"] == k for r in rows),
                   "mean_before_seconds": fmean(r["before_median_seconds"] for r in rows if r["k"] == k),
                   "mean_after_seconds": fmean(r["after_median_seconds"] for r in rows if r["k"] == k),
                   "median_ratio_after_over_before": median(r["ratio_after_over_before"] for r in rows if r["k"] == k)}
                  for k in extended.KS]
    extended.dump(directory/"interleaved_runtime.json", {
        "method": "One process, same handoff/catalog, B=32, two repetitions, alternating BEFORE/AFTER order. Nine SHA256-selected cases plus the reproduction case. No exhaustive reference or quality inference.",
        "before_source_sha256": extended.existing.sha256_file(before_path),
        "after_source_sha256": extended.existing.sha256_file(extended.PROJECT_ROOT/"src/full_architecture/beam_search.py"),
        "case_ids": case_ids, "rows": rows, "aggregates": aggregates,
    })
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
