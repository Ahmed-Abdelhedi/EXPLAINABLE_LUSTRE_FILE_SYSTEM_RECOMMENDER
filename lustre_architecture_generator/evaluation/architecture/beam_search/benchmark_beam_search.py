"""K x B evaluation on existing runtime inputs, with a common H9 reference pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
from statistics import fmean, median
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from full_architecture.architecture_scoring import score_generated_architectures
from full_architecture.beam_search import beam_search_architectures
from full_architecture.catalog_loader import DEFAULT_HARDWARE_CATALOG, load_reference_catalog, sha256_file
from full_architecture.full_architecture_generator import FullArchitectureGeneratorError, enumerate_role_options, generate_full_architectures
from full_architecture.full_architecture_validator import validate_generated_architectures
from full_architecture.runtime_adapter import build_runtime_handoff


def exhaustive_reference(*, handoff: dict, hardware_catalog: dict, max_paths_per_variant: int,
                         max_pairs: int) -> dict:
    """Preflight a configured domain, then run unmodified H8/H9/H10 in full.

    The preflight is timed separately and never caps a baseline prefix. An
    oversized domain is explicitly skipped rather than called exhaustive.
    """
    preflight_started = perf_counter()
    try:
        counts = [len(enumerate_role_options(
            handoff=handoff, hardware_catalog=hardware_catalog, role=role,
            max_paths_per_variant=max_paths_per_variant,
        )) for role in ("MDT", "OST")]
    except FullArchitectureGeneratorError as error:
        return {"status": "NO_ROLE_OPTIONS", "error": str(error), "potential_pairs": 0,
                "preflight_seconds": perf_counter() - preflight_started,
                "runtime_seconds": 0.0, "h10_calls": 0, "valid_count": 0,
                "best_architecture_id": None, "scores": {}, "valid_scores": []}
    pairs = counts[0] * counts[1]
    common = {"potential_pairs": pairs, "role_option_counts": {"mdt": counts[0], "ost": counts[1]},
              "preflight_seconds": perf_counter() - preflight_started}
    if pairs == 0:
        return {**common, "status": "NO_ROLE_OPTIONS", "runtime_seconds": 0.0,
                "h10_calls": 0, "valid_count": 0, "best_architecture_id": None,
                "scores": {}, "valid_scores": []}
    if pairs > max_pairs:
        return {**common, "status": "SKIPPED_PAIR_LIMIT", "pair_limit": max_pairs}
    started = perf_counter()
    generated = generate_full_architectures(
        handoff=handoff, hardware_catalog=hardware_catalog,
        max_paths_per_variant=max_paths_per_variant,
    )
    scored = score_generated_architectures(generation_result=generated, handoff=handoff)
    validated = validate_generated_architectures(
        generation_result=generated, handoff=handoff, hardware_catalog=hardware_catalog,
    )
    elapsed = perf_counter() - started
    valid_ids = {r["architecture_id"] for r in validated["architectures"] if r["decision"] == "VALID"}
    valid_scores = [r for r in scored["architectures"] if r["architecture_id"] in valid_ids]
    best = valid_scores[0] if valid_scores else None
    return {
        **common, "status": "COMPLETE", "runtime_seconds": elapsed,
        "architectures_explored": len(generated["architectures"]),
        "h10_calls": len(validated["architectures"]), "valid_count": len(valid_ids),
        "best_architecture_id": best["architecture_id"] if best else None,
        "best_h9_score": best["score"] if best else None,
        "scores": {r["architecture_id"]: r for r in scored["architectures"]},
        "valid_scores": valid_scores,
    }


def compare_quality(result: dict, reference: dict) -> dict:
    """Locate the *actual Beam winner* in the exhaustive H9 pool; never rescale it."""
    if reference["status"] != "COMPLETE":
        return {"status": "NO_EXHAUSTIVE_REFERENCE"}
    selected = result["best_validated_architecture"]
    if not reference["valid_scores"]:
        return {"status": "NO_VALID_EXHAUSTIVE_ARCHITECTURE"}
    if selected is None:
        return {"status": "BEAM_FOUND_NO_VALID_ARCHITECTURE", "did_beam_find_exhaustive_best": False,
                "exhaustive_best_survived_search": False, "retained_exhaustive_valid_count": 0,
                "exhaustive_valid_recall": 0.0}
    arch_id = selected["architecture_id"]
    common = reference["scores"].get(arch_id)
    if common is None:
        raise RuntimeError("Beam selected an architecture outside the exhaustive domain.")
    valid_rank = next((i for i, r in enumerate(reference["valid_scores"], start=1)
                       if r["architecture_id"] == arch_id), None)
    if valid_rank is None:
        raise RuntimeError("Beam winner was not VALID in the independent exhaustive run.")
    best = reference["valid_scores"][0]
    quality = {
        "status": "COMPARED_IN_COMMON_POOL", "common_h9_score": common["score"],
        "exhaustive_best_h9_score": best["score"],
        "common_score_regret": best["score"] - common["score"],
        "relative_quality": common["score"] / best["score"] if best["score"] > 0 else 1.0,
        "exhaustive_valid_rank": valid_rank,
        "exhaustive_pool_rank": common["rank"],
        "best_id_matched": arch_id == best["architecture_id"],
        "did_beam_find_exhaustive_best": arch_id == best["architecture_id"],
        "exhaustive_best_architecture_id": best["architecture_id"],
        "beam_architecture_id": arch_id,
        "best_score_matched": abs(common["score"] - best["score"]) <= 1e-12,
    }
    if "architectures" in result:
        valid_ids = {r["architecture_id"] for r in result["architectures"]
                     if r["h10"]["decision"] == "VALID" and r["h10"]["valid"] is True}
        retained = [r for r in reference["valid_scores"] if r["architecture_id"] in valid_ids]
        if len(retained) != len(valid_ids):
            raise RuntimeError("Beam VALID pool differs from independent exhaustive H10.")
        retained_best = retained[0]
        quality.update({
            "exhaustive_best_survived_search": best["architecture_id"] in valid_ids,
            "retained_exhaustive_valid_count": len(retained),
            "exhaustive_valid_recall": len(retained)/len(reference["valid_scores"]),
            "retained_best_in_common_pool": retained_best["architecture_id"],
            "search_common_score_regret": best["score"]-retained_best["score"],
            "pool_normalization_common_score_regret": retained_best["score"]-common["score"],
            "local_h9_changed_retained_winner": retained_best["architecture_id"] != arch_id,
            "exhaustive_normalization": {"cost": common["cost_power"]["cost_normalization"],
                                         "power": common["cost_power"]["power_normalization"],
                                         "fault_tolerance": common["reliability_explainability"]["fault_tolerance_normalizer"]},
            "local_normalization": {"cost": selected["h9"]["cost_power"]["cost_normalization"],
                                    "power": selected["h9"]["cost_power"]["power_normalization"],
                                    "fault_tolerance": selected["h9"]["reliability_explainability"]["fault_tolerance_normalizer"]},
        })
    return quality


def benchmark_case(*, handoff: dict, hardware_catalog: dict, beam_width: int,
                   max_paths_per_variant: int, reference: dict, repeats: int = 1) -> tuple[dict, dict]:
    runs = [beam_search_architectures(
        handoff=handoff, hardware_catalog=hardware_catalog, beam_width=beam_width,
        max_paths_per_variant=max_paths_per_variant,
    ) for _ in range(repeats)]
    result = runs[0]
    ids = [r["summary"]["best_architecture_id"] for r in runs]
    if len(set(ids)) > 1:
        raise RuntimeError("Nondeterministic Beam recommendation across repeats.")
    summary = result["summary"]
    runtime = median(r["summary"]["search_seconds"] for r in runs)
    pairs = reference["potential_pairs"]
    compared = compare_quality(result, reference)
    row = {
        "case_id": handoff["case_id"], "requested_top_k": handoff["requested_top_k"],
        "actual_top_k": handoff["actual_top_k"], "beam_width": beam_width,
        "status": result["status"], "runtime_seconds": runtime,
        "runtime_samples_seconds": [r["summary"]["search_seconds"] for r in runs],
        **{k: summary[k] for k in (
            "branches_created",
            "branches_explored", "branches_pruned", "hard_pruned", "width_pruned",
            "pruning_reason_counts", "h5_calls", "h6_calls", "max_active_states",
            "complete_architectures_produced", "h10_calls", "valid_architecture_count",
        )},
        "best_architecture_id": summary["best_architecture_id"],
        "local_final_h9_score": summary["best_final_h9_score"],
        "quality": compared, "exhaustive_status": reference["status"],
        "exhaustive_potential_pairs": pairs,
        "materialization_reduction": 1 - summary["complete_architectures_produced"] / pairs if pairs else None,
        "exhaustive_runtime_seconds": reference.get("runtime_seconds"),
        "speedup": (reference["runtime_seconds"] / runtime
                    if reference["status"] == "COMPLETE" and runtime > 0 else None),
    }
    row.update({key: summary[key] for key in ("main_search_branches_created", "lookahead_branches_created", "lookahead_role_options") if key in summary})
    row["search_trace"] = result["search_trace"]
    row["lookahead_trace"] = result.get("lookahead_trace", [])
    return row, result


def before_after_rows(before: dict, after_rows: list[dict]) -> list[dict]:
    old = {(r["case_id"], r["requested_top_k"], r["beam_width"]): r for r in before["cases"]}
    comparisons = []
    for new in after_rows:
        key = (new["case_id"], new["requested_top_k"], new["beam_width"])
        previous = old.get(key)
        if previous is None:
            raise ValueError(f"Missing BEFORE configuration: {key}")
        comparisons.append({"case_id": key[0], "requested_top_k": key[1], "beam_width": key[2],
                            **{f"{label}_{field}": row.get(field) for label, row in (("old", previous), ("new", new))
                               for field in ("best_architecture_id", "runtime_seconds", "branches_explored", "hard_pruned", "width_pruned", "complete_architectures_produced", "h10_calls")},
                            "old_valid": previous.get("valid_architecture_count", 0) > 0,
                            "new_valid": new.get("valid_architecture_count", 0) > 0,
                            "old_quality": previous.get("quality", {}).get("relative_quality"),
                            "new_quality": new.get("quality", {}).get("relative_quality")})
    return comparisons


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["requested_top_k"], row["beam_width"]), []).append(row)
    output = []
    for (k, width), values in sorted(groups.items()):
        successful = [r for r in values if "runtime_seconds" in r]
        quality = [r["quality"] for r in successful if r["quality"]["status"] == "COMPARED_IN_COMMON_POOL"]
        speedups = [r["speedup"] for r in successful if r["speedup"] is not None]
        output.append({
            "requested_top_k": k, "beam_width": width, "cases": len(values),
            "errors": len(values) - len(successful),
            "feasible_solution_rate": sum(r["valid_architecture_count"] > 0 for r in successful) / len(values),
            "mean_runtime_seconds": fmean(r["runtime_seconds"] for r in successful) if successful else None,
            **{f"total_{key}": sum(r[key] for r in successful) for key in (
                "branches_explored", "branches_pruned", "hard_pruned", "width_pruned",
                "h10_calls", "valid_architecture_count", "complete_architectures_produced",
            )},
            "quality_comparisons": len(quality),
            "mean_relative_quality": fmean(q["relative_quality"] for q in quality) if quality else None,
            "mean_common_score_regret": fmean(q["common_score_regret"] for q in quality) if quality else None,
            "best_id_matches": sum(q["best_id_matched"] for q in quality),
            "best_score_matches": sum(q["best_score_matched"] for q in quality),
            "mean_speedup": fmean(speedups) if speedups else None,
        })
    return output


def write_outputs(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rows = payload["aggregates"]
    with path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Beam Search: measured controlled runtime evaluation", "",
        "Data: existing architecture dataset and official LightGBM handoffs. "
        "These cases are controlled/synthetic, not production generalization evidence.", "",
        f"Cases: {', '.join(payload['config']['case_ids'])}. "
        f"Paths/variant: {payload['config']['max_paths_per_variant']}. "
        f"Repeats: {payload['config']['repeats']}. Baseline pair limit: {payload['config']['exhaustive_max_pairs']}.", "",
        "Time excludes ranking and exhaustive domain preflight; both are recorded separately in JSON. "
        "Exhaustive time includes full H8 generation, H9, and H10 for every pair. "
        "Beam time includes its input validation, all search stages, H9, and H10.", "",
        "Quality locates the actual Beam winner in the exhaustive H9 pool, including invalid architectures "
        "in normalization; the reference winner/rank uses only H10 VALID architectures. "
        "Quality = common-pool Beam score / common-pool best VALID score. "
        "Scores from different pools are never compared. Missing comparisons remain absent.", "",
        "Times are means of per-case medians. Counts are totals across cases. "
        "Speedup is the mean of per-case runtime ratios.", "",
        "| K | B | Feasible rate | Time/case (s) | Branches | Pruned | H10 | VALID | Quality | Speedup |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        quality = f"{row['mean_relative_quality']:.4f}" if row["mean_relative_quality"] is not None else "n/a"
        speedup = f"{row['mean_speedup']:.1f}x" if row["mean_speedup"] is not None else "n/a"
        runtime = f"{row['mean_runtime_seconds']:.6f}" if row["mean_runtime_seconds"] is not None else "n/a"
        lines.append(f"| {row['requested_top_k']} | {row['beam_width']} | {row['feasible_solution_rate']:.1%} | "
                     f"{runtime} | {row['total_branches_explored']} | {row['total_branches_pruned']} | "
                     f"{row['total_h10_calls']} | {row['total_valid_architecture_count']} | {quality} | {speedup} |")
    lines.extend(["", "## Exhaustive reference runs", "",
                  "| Case | K | Status | Pairs | Time (s) | H10 | VALID | Best architecture |",
                  "|---|---:|---|---:|---:|---:|---:|---|"])
    for row in payload["baselines"]:
        lines.append(f"| {row['case_id']} | {row['requested_top_k']} | {row['status']} | "
                     f"{row['potential_pairs']} | {row.get('runtime_seconds', 'n/a')} | "
                     f"{row.get('h10_calls', 'n/a')} | {row.get('valid_count', 'n/a')} | "
                     f"{row.get('best_architecture_id', 'n/a')} |")
    if any("search_common_score_regret" in r.get("quality", {}) for r in payload["cases"]):
        lines.extend(["", "## Common-pool quality and loss attribution", "",
                      "Search regret measures the best VALID missing from the retained pool. "
                      "Normalization regret measures a different local H9 winner inside that retained pool. "
                      "Their sum equals total common-pool regret.", "",
                      "| Case | K | B | VALID rank | Quality | Best survived | Best chosen | Search regret | Normalization regret |",
                      "|---|---:|---:|---:|---:|---|---|---:|---:|"])
        for row in payload["cases"]:
            q = row.get("quality", {})
            if "search_common_score_regret" not in q:
                continue
            lines.append(f"| {row['case_id']} | {row['requested_top_k']} | {row['beam_width']} | {q['exhaustive_valid_rank']} | {q['relative_quality']:.6f} | {q['exhaustive_best_survived_search']} | {q['did_beam_find_exhaustive_best']} | {q['search_common_score_regret']:.9f} | {q['pool_normalization_common_score_regret']:.9f} |")
    if payload.get("before_after"):
        comparisons = payload["before_after"]
        comparison_path = path.with_name(path.stem+"_before_after.csv")
        with comparison_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
        lines.extend(["", "## BEFORE versus AFTER", "",
                      "Old times are the archived V1 campaign, not a simultaneous timing experiment.", "",
                      "| Case | K | B | Old valid | New valid | Old quality | New quality | Old time (s) | New time (s) |",
                      "|---|---:|---:|---|---|---:|---:|---:|---:|"])
        for row in comparisons:
            oq = f"{row['old_quality']:.6f}" if row["old_quality"] is not None else "n/a"
            nq = f"{row['new_quality']:.6f}" if row["new_quality"] is not None else "n/a"
            lines.append(f"| {row['case_id']} | {row['requested_top_k']} | {row['beam_width']} | {row['old_valid']} | {row['new_valid']} | {oq} | {nq} | {row['old_runtime_seconds']:.6f} | {row['new_runtime_seconds']:.6f} |")
    lines.extend(["", "## Scope and limitations", "",
                  "`SKIPPED_PAIR_LIMIT` means no exhaustive quality/time comparison was executed for that domain. "
                  "The pair limit never truncates a baseline and never limits Beam. "
                  "A missing Beam solution does not prove global infeasibility. "
                  "An increased width need not monotonically improve quality or feasibility. "
                  "The physical catalog and per-variant path cap constrain both methods.", ""])
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architectures", type=Path, default=PROJECT_ROOT / "output/lustre_architecture_dataset.json")
    parser.add_argument("--drive-catalog", type=Path, default=PROJECT_ROOT / "data/catalogue_drives_ready_final.json")
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--limit", type=int, default=3, help="0: all cases (explicitly controlled subset recommended).")
    parser.add_argument("--top-k-values", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--beam-widths", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--max-paths-per-variant", type=int, default=1)
    parser.add_argument("--exhaustive-max-pairs", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "beam_search_benchmark.json")
    parser.add_argument("--before", type=Path, help="Archived V1 report; adds explicit BEFORE/AFTER comparison.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 0 or any(v <= 0 for v in [*args.top_k_values, *args.beam_widths,
                                            args.max_paths_per_variant, args.exhaustive_max_pairs, args.repeats]):
        raise ValueError("K, B, paths, pair limit and repeats must be positive; limit must be >= 0.")
    architectures = json.loads(args.architectures.read_text(encoding="utf-8"))
    drives = json.loads(args.drive_catalog.read_text(encoding="utf-8"))
    hardware = load_reference_catalog()
    if args.case_ids:
        by_id = {v["case_id"]: v for v in architectures}
        selected = [by_id[key] for key in args.case_ids]
    else:
        selected = architectures if args.limit == 0 else architectures[:args.limit]
    if not selected:
        raise ValueError("No selected cases.")
    rows, baselines, example = [], [], None
    ranking_times = []
    for architecture in selected:
        for k in sorted(set(args.top_k_values)):
            try:
                started = perf_counter()
                handoff = build_runtime_handoff(architecture=architecture, catalog=drives, top_k=k)
                ranking_times.append({"case_id": architecture["case_id"], "requested_top_k": k,
                                      "seconds": perf_counter() - started})
                reference = exhaustive_reference(
                    handoff=handoff, hardware_catalog=hardware,
                    max_paths_per_variant=args.max_paths_per_variant, max_pairs=args.exhaustive_max_pairs,
                )
                baseline = {key: value for key, value in reference.items() if key not in {"scores", "valid_scores"}}
                baselines.append({"case_id": handoff["case_id"], "requested_top_k": k, **baseline})
                print(f"{handoff['case_id']} K={k}: exhaustive={reference['status']} pairs={reference['potential_pairs']}", flush=True)
                for width in sorted(set(args.beam_widths)):
                    row, result = benchmark_case(
                        handoff=handoff, hardware_catalog=hardware, beam_width=width,
                        max_paths_per_variant=args.max_paths_per_variant, reference=reference, repeats=args.repeats,
                    )
                    rows.append(row)
                    if example is None and result["best_validated_architecture"] is not None:
                        example = {"case_id": handoff["case_id"], "requested_top_k": k,
                                   "beam_width": width, "requirements": handoff["requirements"],
                                   "architecture": result["best_validated_architecture"]}
                    print(f"  B={width}: {row['status']} branches={row['branches_explored']} H10={row['h10_calls']}", flush=True)
            except Exception as error:
                print(f"{architecture['case_id']} K={k}: ERROR {type(error).__name__}: {error}", flush=True)
                for width in sorted(set(args.beam_widths)):
                    if not any(r['case_id'] == architecture['case_id'] and r['requested_top_k'] == k and r['beam_width'] == width for r in rows):
                        rows.append({"case_id": architecture["case_id"], "requested_top_k": k, "beam_width": width,
                                     "status": "EXECUTION_ERROR", "error": f"{type(error).__name__}: {error}"})
    source_files = [args.architectures, args.drive_catalog, DEFAULT_HARDWARE_CATALOG,
                    Path(__file__), SRC_DIR / "full_architecture/beam_search.py"]
    payload = {
        "schema_version": "1.0", "purpose": "beam_search_k_b_benchmark",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "source_sha256": {str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else str(p): sha256_file(p)
                          for p in source_files},
        "config": {"case_ids": [v["case_id"] for v in selected], "top_k_values": sorted(set(args.top_k_values)),
                   "beam_widths": sorted(set(args.beam_widths)), "max_paths_per_variant": args.max_paths_per_variant,
                   "exhaustive_max_pairs": args.exhaustive_max_pairs, "repeats": args.repeats},
        "ranking_times": ranking_times, "baselines": baselines,
        "cases": rows, "aggregates": aggregate(rows), "validated_example": example,
    }
    if args.before:
        before = json.loads(args.before.read_text(encoding="utf-8"))
        for source in source_files[:3]:
            key = str(source.relative_to(PROJECT_ROOT)) if source.is_relative_to(PROJECT_ROOT) else str(source)
            if before["source_sha256"].get(key) != sha256_file(source):
                raise ValueError(f"BEFORE input hash mismatch: {key}")
        if before["config"]["max_paths_per_variant"] != args.max_paths_per_variant:
            raise ValueError("BEFORE path cap differs.")
        payload["before_source"] = {"path": str(args.before), "sha256": sha256_file(args.before)}
        payload["before_after"] = before_after_rows(before, rows)
    write_outputs(args.output, payload)
    print(f"Wrote {args.output} and companion CSV/Markdown.", flush=True)
    if any(row["status"] == "EXECUTION_ERROR" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
