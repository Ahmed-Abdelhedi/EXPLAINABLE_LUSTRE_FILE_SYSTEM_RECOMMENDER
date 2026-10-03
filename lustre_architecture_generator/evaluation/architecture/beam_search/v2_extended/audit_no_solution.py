"""Explain every missing Beam solution without exhaustive H9 materialization.

The frozen feasibility module enumerates per-role H5/H6 options, checks scalar
additive budget/power pairs and, if feasible, constructs one H7/H10 witness.
This provides feasibility evidence, never a common-pool quality reference.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from time import perf_counter

import run_extended_benchmark as extended
from full_architecture.feasibility_coverage import analyze_case_feasibility_domain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-directory", type=Path, default=extended.HERE)
    args = parser.parse_args()
    extended.HERE = args.benchmark_directory.resolve()
    payload = json.loads((extended.HERE / "extended_benchmark.json").read_text(encoding="utf-8"))
    dataset = json.loads((extended.PROJECT_ROOT / "output/lustre_architecture_dataset.json").read_text(encoding="utf-8"))
    by_id = {case["case_id"]: case for case in dataset}
    drives = json.loads((extended.PROJECT_ROOT / "data/catalogue_drives_ready_final.json").read_text(encoding="utf-8"))
    hardware = extended.existing.load_reference_catalog()
    missing = defaultdict(list)
    for row in payload["cases"]:
        if not row["valid_found"]:
            missing[(row["case_id"], row["requested_top_k"])].append(row)
    audited = []
    for (case_id, k), rows in sorted(missing.items()):
        started = perf_counter()
        handoff = extended.existing.build_runtime_handoff(architecture=by_id[case_id], catalog=drives, top_k=k)
        counts = {role.lower(): len(extended.existing.enumerate_role_options(
            handoff=handoff, hardware_catalog=hardware, role=role,
            max_paths_per_variant=payload["config"]["max_paths_per_variant"],
        )) for role in ("MDT", "OST")}
        if not all(counts.values()):
            audit = {"classification": "NO_COMPATIBLE_ROLE_OPTION", "option_counts": counts,
                     "domain_proven_infeasible": True, "recovered_valid_architecture": False,
                     "h10_calls": 0}
        else:
            detail = analyze_case_feasibility_domain(
                handoff=handoff, hardware_catalog=hardware,
                max_paths_per_variant=payload["config"]["max_paths_per_variant"],
            )
            audit = {"classification": detail["search"]["classification"],
                     "option_counts": detail["option_counts"],
                     "domain_proven_infeasible": not detail["search"]["found"],
                     "recovered_valid_architecture": detail["recovered_valid_architecture"],
                     "recovered_architecture_id": detail["recovered_architecture_id"],
                     "h10_calls": int(detail["recovered_valid_architecture"]), "detail": detail}
        audit.update({"case_id": case_id, "requested_top_k": k,
                      "missing_widths": [r["beam_width"] for r in rows],
                      "runtime_seconds": perf_counter()-started})
        audited.append(audit)
        print(f"{case_id} K={k}: {audit['classification']}", flush=True)
    result = {
        "method": "Frozen public per-role enumeration and feasibility_coverage.analyze_case_feasibility_domain; scalar cost/power check, at most one COMPLETE/H10 witness, no H9 pool or quality approximation.",
        "scope": "Same Top-K, same max_paths_per_variant as the campaign. Domain infeasibility does not imply global infeasibility.",
        "audited_case_k_groups": len(audited),
        "missing_configurations": sum(len(r["missing_widths"]) for r in audited),
        "classification_counts": dict(Counter(r["classification"] for r in audited)),
        "feasible_missed_groups": sum(r["recovered_valid_architecture"] for r in audited),
        "cases": audited,
    }
    extended.dump(extended.HERE / "no_solution_audit.json", result)
    lines = ["# Audit des absences de solution Beam", "", result["method"], "", result["scope"], "",
             f"{len(audited)} couples case×K ; {result['missing_configurations']} configurations sans solution ; {result['feasible_missed_groups']} témoins faisables manqués.", ""]
    lines += extended.table(["Case", "K", "Largeurs sans solution", "Classification", "Options MDT", "Options OST", "Infaisable dans le domaine"],
                            [[r["case_id"], r["requested_top_k"], r["missing_widths"], r["classification"],
                              r["option_counts"]["mdt"], r["option_counts"]["ost"], r["domain_proven_infeasible"]] for r in audited])
    (extended.HERE / "no_solution_audit.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
