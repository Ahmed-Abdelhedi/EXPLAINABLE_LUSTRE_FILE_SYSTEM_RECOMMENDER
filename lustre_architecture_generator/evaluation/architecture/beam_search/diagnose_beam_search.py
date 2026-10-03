"""Offline exhaustive-prefix diagnosis; never imported by the production Beam."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import benchmark_beam_search as benchmark
from full_architecture import beam_search as current
from full_architecture.full_architecture_generator import generate_full_architectures, _path_signature
from full_architecture.full_architecture_validator import validate_generated_architectures
from full_architecture.architecture_scoring import score_generated_architectures


def load_v1():
    name = "full_architecture._beam_v1_reference"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("beam_search_v1_reference.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def selected_prefix(selected, stage):
    keys = [selected["mdt_drive"]["drive_id"],
            selected["mdt_protection"]["protection_profile_id"],
            _path_signature(selected["mdt_hardware_path"]),
            selected["ost_drive"]["drive_id"],
            selected["ost_protection"]["protection_profile_id"],
            _path_signature(selected["ost_hardware_path"])]
    return tuple(keys[:current.SEARCH_STAGES.index(stage) + 1])


def node_prefix(node, stage):
    values = []
    for choice in (node.mdt, node.ost):
        if choice is None:
            break
        values.append(choice["candidate"]["identity"]["drive_id"])
        if "protection" not in choice:
            break
        values.append(choice["protection"]["protection_profile_id"])
        if "hardware_path" not in choice:
            break
        values.append(_path_signature(choice["hardware_path"]))
    return tuple(values[:current.SEARCH_STAGES.index(stage) + 1])


def diversity(nodes, constraints):
    choices = [n.mdt for n in nodes if n.mdt]
    bounds = [current._lower_bounds(n) for n in nodes]
    return {
        "mdt_drives": len({c["candidate"]["identity"]["drive_id"] for c in choices}),
        "mdt_protections": len({c["protection"]["protection_profile_id"] for c in choices if c.get("protection")}),
        "mdt_hardware_paths": len({_path_signature(c["hardware_path"]) for c in choices if c.get("hardware_path")}),
        "distinct_costs": len({b["cost_usd"] for b in bounds}),
        "distinct_powers": len({b["power_w"] for b in bounds}),
        "cost_deciles_of_limit": sorted({int(10*b["cost_usd"]/constraints["max_budget_usd"]) for b in bounds}) if constraints["max_budget_usd"] else [],
        "power_deciles_of_limit": sorted({int(10*b["power_w"]/constraints["max_power_w"]) for b in bounds}) if constraints["max_power_w"] else [],
    }


def diagnose(module, handoff, hardware, width, generated, scored, validated):
    valid_ids = {r["architecture_id"] for r in validated["architectures"] if r["decision"] == "VALID"}
    valid_records = [r for r in generated["architectures"] if r["architecture_id"] in valid_ids]
    best_id = next(r["architecture_id"] for r in scored["architectures"] if r["architecture_id"] in valid_ids)
    best = next(r for r in valid_records if r["architecture_id"] == best_id)
    attempted, eligible, kept = {}, {}, {}
    real_replace = module.replace

    def observe(node, **changes):
        value = real_replace(node, **changes)
        if "branch_id" in changes:
            stage = module.SEARCH_STAGES[len(node.trace)]
            attempted.setdefault(stage, []).append(value)
        elif "beam_score" in changes:
            eligible.setdefault(value.trace[-1]["search_stage"], []).append(value)
        elif "trace" in changes and value.trace and "beam_rank" in value.trace[-1]:
            kept.setdefault(value.trace[-1]["search_stage"], []).append(value)
        return value

    with patch.object(module, "replace", side_effect=observe):
        result = module.beam_search_architectures(handoff=handoff, hardware_catalog=hardware,
                                                beam_width=width, max_paths_per_variant=1)
    levels = []
    first_loss = None
    constraints = handoff["requirements"]["constraints"]
    for stage in module.SEARCH_STAGES:
        counters = next((s for s in result["search_trace"] if s["search_stage"] == stage),
                        {"search_stage": stage, "branches_created": 0, "hard_pruned": 0,
                         "width_pruned": 0, "retained": 0, "pruning_reason_counts": {}})
        all_nodes = sorted(eligible.get(stage, []), key=module._node_key)
        survivors = kept.get(stage, [])
        valid_prefixes = {selected_prefix(r["state"]["selected"], stage) for r in valid_records}
        retained_prefixes = {node_prefix(n, stage) for n in survivors}
        viable_generated = sum(node_prefix(n, stage) in valid_prefixes for n in attempted.get(stage, []))
        viable_eligible = sum(node_prefix(n, stage) in valid_prefixes for n in all_nodes)
        viable_kept = sum(node_prefix(n, stage) in valid_prefixes for n in survivors)
        target = selected_prefix(best["state"]["selected"], stage)
        ranked = next(((i, n) for i, n in enumerate(all_nodes, 1) if node_prefix(n, stage) == target), None)
        attempted_target = next((n for n in attempted.get(stage, []) if node_prefix(n, stage) == target), None)
        status = "KEPT" if target in retained_prefixes else (
            "WIDTH_PRUNED" if ranked else "HARD_PRUNED" if attempted_target else "PARENT_ALREADY_LOST")
        if status != "KEPT" and first_loss is None:
            first_loss = stage
        levels.append({**counters,
            "best_retained_score": max((n.beam_score for n in survivors), default=None),
            "worst_retained_score": min((n.beam_score for n in survivors), default=None),
            "viable_generated": viable_generated, "viable_hard_pruned": viable_generated-viable_eligible,
            "viable_width_pruned": viable_eligible-viable_kept, "viable_retained": viable_kept,
            "reachable_valid_architectures": sum(selected_prefix(r["state"]["selected"], stage) in retained_prefixes for r in valid_records),
            "diversity": diversity(survivors, constraints),
            "exhaustive_best_path": {"status": status, "rank_before_width": ranked[0] if ranked else None,
                                     "score": ranked[1].beam_score if ranked else None},
        })
    complete = result["summary"]["complete_architectures_produced"]
    complete_scores = [r["search_provenance"]["beam_heuristic"] for r in result["architectures"]]
    valid_scores = [r["search_provenance"]["beam_heuristic"] for r in result["architectures"] if r["h10"]["decision"] == "VALID"]
    levels.extend([
        {"search_stage": "COMPLETE", "branches_created": complete, "hard_pruned": 0, "width_pruned": 0,
         "retained": complete, "viable_retained": result["summary"]["valid_architecture_count"],
         "best_retained_score": max(complete_scores, default=None), "worst_retained_score": min(complete_scores, default=None),
         "pruning_reason_counts": {}},
        {"search_stage": "H10", "branches_created": result["summary"]["h10_calls"],
         "hard_pruned": complete-result["summary"]["valid_architecture_count"], "width_pruned": 0,
         "retained": result["summary"]["valid_architecture_count"], "viable_retained": result["summary"]["valid_architecture_count"],
         "best_retained_score": max(valid_scores, default=None), "worst_retained_score": min(valid_scores, default=None),
         "pruning_reason_counts": {"H10_INVALID": complete-result["summary"]["valid_architecture_count"]} if complete != result["summary"]["valid_architecture_count"] else {}},
    ])
    return {"beam_width": width, "status": result["status"], "levels": levels,
            "lookahead_trace": result.get("lookahead_trace", []),
            "summary": result["summary"], "exhaustive_valid_count": len(valid_records),
            "exhaustive_best_id": best_id, "best_path_first_loss": first_loss,
            "exhaustive_best_selected": best["state"]["selected"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beam-widths", nargs="+", type=int, default=[32, 128, 256])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("diagnostic_v1.json"))
    parser.add_argument("--strategy", choices=["v1", "v2"], default="v1")
    args = parser.parse_args()
    architectures = json.loads((benchmark.PROJECT_ROOT / "output/lustre_architecture_dataset.json").read_text(encoding="utf-8"))
    drives = json.loads((benchmark.PROJECT_ROOT / "data/catalogue_drives_ready_final.json").read_text(encoding="utf-8"))
    architecture = next(a for a in architectures if a["case_id"] == "REQ_000003")
    handoff = benchmark.build_runtime_handoff(architecture=architecture, catalog=drives, top_k=5)
    hardware = benchmark.load_reference_catalog()
    generated = generate_full_architectures(handoff=handoff, hardware_catalog=hardware, max_paths_per_variant=1)
    scored = score_generated_architectures(generation_result=generated, handoff=handoff)
    validated = validate_generated_architectures(generation_result=generated, handoff=handoff, hardware_catalog=hardware)
    module = load_v1() if args.strategy == "v1" else current
    runs = [diagnose(module, handoff, hardware, b, generated, scored, validated) for b in args.beam_widths]
    payload = {"purpose": f"offline_exhaustive_prefix_diagnosis_{args.strategy}", "case_id": handoff["case_id"],
               "requested_top_k": 5, "max_paths_per_variant": 1,
               "source_sha256": benchmark.sha256_file(Path(module.__file__)), "runs": runs}
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    lines = [f"# Offline {args.strategy.upper()} diagnosis: REQ_000003, K=5", "",
             "Exhaustive truth is used only here; instrumented runtime is not a benchmark runtime.", "",
             "| B | Stage | Generated | Hard | Width | Kept | Viable kept | Best score | Worst score | Best path rank | Best path status |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for run in runs:
        for row in run["levels"]:
            path = row.get("exhaustive_best_path", {})
            lines.append(f"| {run['beam_width']} | {row['search_stage']} | {row['branches_created']} | {row['hard_pruned']} | {row['width_pruned']} | {row['retained']} | {row.get('viable_retained','')} | {row.get('best_retained_score','')} | {row.get('worst_retained_score','')} | {path.get('rank_before_width','')} | {path.get('status','')} |")
    args.output.with_suffix(".md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    for run in runs:
        print(f"B={run['beam_width']}: {run['status']}; best path lost at {run['best_path_first_loss']}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
