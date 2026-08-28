from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"

DEFAULT_ARCHITECTURES = (
    PROJECT_ROOT
    / "output"
    / "lustre_architecture_dataset.json"
)
DEFAULT_DRIVE_CATALOG = (
    PROJECT_ROOT
    / "data"
    / "catalogue_drives_ready_final.json"
)
DEFAULT_BASELINE = (
    PROJECT_ROOT
    / "evaluation"
    / "architecture"
    / "validation"
    / "full_architecture_validation.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "evaluation"
    / "architecture"
    / "validation"
    / "feasibility_coverage_path2.json"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from full_architecture.catalog_loader import load_reference_catalog  # noqa: E402
from full_architecture.feasibility_coverage import (  # noqa: E402
    analyze_case_feasibility_domain,
    unresolved_case_ids_from_h10,
)
from full_architecture.runtime_adapter import build_runtime_handoff  # noqa: E402


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "H10-B: enlève le cap H8 max_role_options sur les cas sans "
            "architecture valide, en gardant Top-K et max_paths explicites."
        )
    )
    parser.add_argument(
        "--limit-unresolved",
        type=int,
        default=10,
        help="0 = tous les cas sans architecture valide du baseline H10.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-paths-per-variant", type=int, default=2)
    parser.add_argument(
        "--architectures",
        type=Path,
        default=DEFAULT_ARCHITECTURES,
    )
    parser.add_argument(
        "--drive-catalog",
        type=Path,
        default=DEFAULT_DRIVE_CATALOG,
    )
    parser.add_argument(
        "--baseline-h10",
        type=Path,
        default=DEFAULT_BASELINE,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.limit_unresolved < 0:
        raise ValueError("--limit-unresolved doit être >= 0.")
    if args.top_k <= 0:
        raise ValueError("--top-k doit être > 0.")
    if args.max_paths_per_variant <= 0:
        raise ValueError("--max-paths-per-variant doit être > 0.")

    architectures = load_json(args.architectures)
    drive_catalog = load_json(args.drive_catalog)
    baseline = load_json(args.baseline_h10)
    hardware_catalog = load_reference_catalog()

    if not isinstance(architectures, list):
        raise TypeError("architectures doit être une liste.")
    if not isinstance(drive_catalog, list):
        raise TypeError("drive_catalog doit être une liste.")

    unresolved_ids = unresolved_case_ids_from_h10(baseline)
    selected_ids = (
        unresolved_ids
        if args.limit_unresolved == 0
        else unresolved_ids[: args.limit_unresolved]
    )

    by_case = {
        str(item.get("case_id")): item
        for item in architectures
        if isinstance(item, dict)
    }

    rows: list[dict[str, Any]] = []
    classification_counts: Counter[str] = Counter()
    recovered = 0
    started = time.perf_counter()

    for index, case_id in enumerate(selected_ids, start=1):
        architecture = by_case.get(case_id)

        if architecture is None:
            row = {
                "case_id": case_id,
                "status": "FAILED",
                "recovered_valid_architecture": False,
                "classification": "CASE_NOT_FOUND",
                "error": "case_id absent du dataset architecture.",
            }
            rows.append(row)
            classification_counts.update(["CASE_NOT_FOUND"])
            continue

        try:
            handoff = build_runtime_handoff(
                architecture=architecture,
                catalog=drive_catalog,
                top_k=args.top_k,
            )

            analysis = analyze_case_feasibility_domain(
                handoff=handoff,
                hardware_catalog=hardware_catalog,
                max_paths_per_variant=args.max_paths_per_variant,
            )

            was_recovered = bool(
                analysis["recovered_valid_architecture"]
            )

            if was_recovered:
                recovered += 1

            classification = analysis["search"]["classification"]
            classification_counts.update([classification])

            row = {
                "case_id": case_id,
                "status": "OK",
                "recovered_valid_architecture": was_recovered,
                "coverage_interpretation": analysis[
                    "coverage_interpretation"
                ],
                "classification": classification,
                "mdt_options": analysis["option_counts"]["mdt"],
                "ost_options": analysis["option_counts"]["ost"],
                "potential_pairs": analysis["option_counts"][
                    "potential_pairs"
                ],
                "pairs_examined": analysis["search"]["pairs_examined"],
                "minimum_total_cost_usd": analysis["search"][
                    "minimum_total_cost_usd"
                ],
                "maximum_budget_usd": analysis["limits"][
                    "maximum_budget_usd"
                ],
                "minimum_total_power_w": analysis["search"][
                    "minimum_total_power_w"
                ],
                "maximum_power_w": analysis["limits"][
                    "maximum_power_w"
                ],
                "recovered_architecture_id": analysis[
                    "recovered_architecture_id"
                ],
                "error": "",
            }

        except Exception as error:
            row = {
                "case_id": case_id,
                "status": "FAILED",
                "recovered_valid_architecture": False,
                "classification": "EXECUTION_ERROR",
                "error": f"{type(error).__name__}: {error}",
            }
            classification_counts.update(["EXECUTION_ERROR"])

        rows.append(row)

        if (
            index == 1
            or index % 50 == 0
            or index == len(selected_ids)
        ):
            print(
                f"[{index}/{len(selected_ids)}] "
                f"{case_id} : {row['status']} "
                f"recovered={row['recovered_valid_architecture']}"
            )

    elapsed = time.perf_counter() - started
    failures = [row for row in rows if row["status"] != "OK"]

    baseline_summary = baseline.get("summary", {})
    baseline_cases = int(baseline_summary.get("cases", 0))
    baseline_with_valid = int(
        baseline_summary.get("cases_with_valid_architecture", 0)
    )
    baseline_without_valid = int(
        baseline_summary.get("cases_without_valid_architecture", 0)
    )

    full_unresolved_run = (
        args.limit_unresolved == 0
        or len(selected_ids) == len(unresolved_ids)
    )

    coverage_after = (
        baseline_with_valid + recovered
        if full_unresolved_run
        else None
    )

    payload = {
        "schema_version": "1.0",
        "purpose": "h10b_feasibility_coverage_role_cap_expansion",
        "semantics": {
            "baseline_h10_is_authoritative": True,
            "top_k_fixed": args.top_k,
            "max_paths_per_variant_fixed": args.max_paths_per_variant,
            "max_role_options_removed": True,
            "full_cartesian_cost_power_feasibility_checked": True,
            "recovered_pair_confirmed_by_h10": True,
            "beam_search_applied": False,
            "does_not_certify_global_infeasibility": True,
            "unresolved_scope": (
                "still unresolved within current Top-K and current "
                "max_paths_per_variant; path-cap expansion may still recover"
            ),
        },
        "baseline": {
            "cases": baseline_cases,
            "cases_with_valid_architecture": baseline_with_valid,
            "cases_without_valid_architecture": baseline_without_valid,
        },
        "limits": {
            "limit_unresolved": args.limit_unresolved,
            "top_k": args.top_k,
            "max_paths_per_variant": args.max_paths_per_variant,
        },
        "summary": {
            "status": "VALIDATED" if not failures else "FAILED",
            "baseline_unresolved_cases": len(unresolved_ids),
            "analyzed_unresolved_cases": len(rows),
            "successful_analysis_cases": len(rows) - len(failures),
            "failure_cases": len(failures),
            "recovered_by_role_option_expansion": recovered,
            "still_unresolved": (
                len(rows) - len(failures) - recovered
            ),
            "classification_counts": dict(
                classification_counts.most_common()
            ),
            "full_unresolved_run": full_unresolved_run,
            "overall_cases_with_valid_after_expansion": coverage_after,
            "overall_coverage_ratio_after_expansion": (
                coverage_after / baseline_cases
                if (
                    coverage_after is not None
                    and baseline_cases > 0
                )
                else None
            ),
            "elapsed_seconds": round(elapsed, 4),
        },
        "cases": rows,
    }

    save_json(args.output, payload)

    print()
    print("H10-B FEASIBILITY COVERAGE")
    print("==========================")
    print("Status                    :", payload["summary"]["status"])
    print(
        "Baseline unresolved       :",
        payload["summary"]["baseline_unresolved_cases"],
    )
    print(
        "Analyzed unresolved       :",
        payload["summary"]["analyzed_unresolved_cases"],
    )
    print(
        "Failures                  :",
        payload["summary"]["failure_cases"],
    )
    print(
        "Recovered by role expansion:",
        payload["summary"]["recovered_by_role_option_expansion"],
    )
    print(
        "Still unresolved          :",
        payload["summary"]["still_unresolved"],
    )

    if coverage_after is not None:
        print(
            "Overall cases with valid  :",
            f"{coverage_after}/{baseline_cases}",
        )
        print(
            "Overall coverage ratio    :",
            round(
                payload["summary"][
                    "overall_coverage_ratio_after_expansion"
                ],
                8,
            ),
        )

    print("Elapsed                   :", payload["summary"]["elapsed_seconds"], "s")
    print("Output                    :", args.output)

    if classification_counts:
        print("Classifications           :")
        for code, count in classification_counts.most_common():
            print(f"  - {code}: {count}")

    if failures:
        print()
        print("Premières erreurs :")
        for row in failures[:10]:
            print("-", row["case_id"], ":", row["error"])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
