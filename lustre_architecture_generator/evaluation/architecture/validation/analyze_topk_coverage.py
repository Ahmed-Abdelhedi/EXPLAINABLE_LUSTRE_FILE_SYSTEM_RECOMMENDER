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
DEFAULT_H10C = (
    PROJECT_ROOT
    / "evaluation"
    / "architecture"
    / "validation"
    / "feasibility_coverage_full_paths.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "evaluation"
    / "architecture"
    / "validation"
    / "feasibility_coverage_topk_20_50.json"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from full_architecture.catalog_loader import (  # noqa: E402
    load_reference_catalog,
)
from full_architecture.topk_coverage import (  # noqa: E402
    analyze_case_topk_coverage,
    normalize_top_k_values,
    unresolved_case_ids_from_h10c,
)


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
            "H10-D : couverture de faisabilité Top-K sur les cas restant "
            "non résolus après H10-C."
        )
    )

    parser.add_argument(
        "--limit-unresolved",
        type=int,
        default=10,
        help="0 = tous les cas non résolus de H10-C.",
    )
    parser.add_argument(
        "--baseline-k",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--top-k-values",
        type=int,
        nargs="+",
        default=[20, 50],
    )
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
        "--h10c-result",
        type=Path,
        default=DEFAULT_H10C,
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

    k_values = normalize_top_k_values(
        args.top_k_values,
        baseline_k=args.baseline_k,
    )

    architectures = load_json(args.architectures)
    drive_catalog = load_json(args.drive_catalog)
    h10c = load_json(args.h10c_result)
    hardware_catalog = load_reference_catalog()

    if not isinstance(architectures, list):
        raise TypeError("architectures doit être une liste.")
    if not isinstance(drive_catalog, list):
        raise TypeError("drive_catalog doit être une liste.")

    unresolved_ids = unresolved_case_ids_from_h10c(h10c)
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
    recovered_by_k: Counter[int] = Counter()
    final_classifications: Counter[str] = Counter()
    started = time.perf_counter()

    for index, case_id in enumerate(selected_ids, start=1):
        architecture = by_case.get(case_id)

        if architecture is None:
            row = {
                "case_id": case_id,
                "status": "FAILED",
                "recovered_valid_architecture": False,
                "recovered_at_k": None,
                "final_classification": "CASE_NOT_FOUND",
                "error": "case_id absent du dataset architecture.",
            }
            rows.append(row)
            final_classifications.update(["CASE_NOT_FOUND"])
            continue

        try:
            analysis = analyze_case_topk_coverage(
                architecture=architecture,
                drive_catalog=drive_catalog,
                hardware_catalog=hardware_catalog,
                top_k_values=k_values,
                baseline_k=args.baseline_k,
            )

            recovered = bool(
                analysis["recovered_valid_architecture"]
            )
            recovered_at_k = analysis["recovered_at_k"]
            final_classification = analysis[
                "final_classification"
            ]

            if recovered_at_k is not None:
                recovered_by_k.update([int(recovered_at_k)])

            final_classifications.update(
                [final_classification]
            )

            row = {
                "case_id": case_id,
                "status": "OK",
                "recovered_valid_architecture": recovered,
                "recovered_at_k": recovered_at_k,
                "recovered_architecture_id": analysis[
                    "recovered_architecture_id"
                ],
                "final_classification": final_classification,
                "coverage_interpretation": analysis[
                    "coverage_interpretation"
                ],
                "attempts": analysis["attempts"],
                "error": "",
            }

        except Exception as error:
            row = {
                "case_id": case_id,
                "status": "FAILED",
                "recovered_valid_architecture": False,
                "recovered_at_k": None,
                "final_classification": "EXECUTION_ERROR",
                "error": f"{type(error).__name__}: {error}",
            }
            final_classifications.update(["EXECUTION_ERROR"])

        rows.append(row)

        if (
            index == 1
            or index % 20 == 0
            or index == len(selected_ids)
        ):
            print(
                f"[{index}/{len(selected_ids)}] "
                f"{case_id} : {row['status']} "
                f"recovered_at_k={row['recovered_at_k']}"
            )

    elapsed = time.perf_counter() - started
    failures = [row for row in rows if row["status"] != "OK"]
    recovered_total = sum(recovered_by_k.values())

    h10c_summary = h10c.get("summary", {})
    baseline_cases = int(
        h10c.get("baseline", {}).get("cases", 0)
    )
    baseline_valid_after_h10c = int(
        h10c_summary.get(
            "overall_cases_with_valid_after_h10c",
            0,
        )
    )

    full_unresolved_run = (
        args.limit_unresolved == 0
        or len(selected_ids) == len(unresolved_ids)
    )

    overall_valid_after_h10d = (
        baseline_valid_after_h10c + recovered_total
        if full_unresolved_run
        else None
    )

    payload = {
        "schema_version": "1.0",
        "purpose": "h10d_topk_feasibility_coverage",
        "semantics": {
            "h10c_result_is_baseline": True,
            "baseline_k": args.baseline_k,
            "tested_top_k_values": k_values,
            "max_role_options_removed": True,
            "hardware_path_domain_exhaustive_for_current_catalog": True,
            "recovered_pair_confirmed_by_h10": True,
            "beam_search_applied": False,
            "architecture_scoring_required": False,
            "this_is_feasibility_coverage_not_k_optimality": True,
            "global_infeasibility_is_not_claimed": True,
        },
        "baseline": {
            "cases": baseline_cases,
            "cases_with_valid_after_h10c": (
                baseline_valid_after_h10c
            ),
            "unresolved_after_h10c": len(unresolved_ids),
        },
        "limits": {
            "limit_unresolved": args.limit_unresolved,
            "baseline_k": args.baseline_k,
            "top_k_values": k_values,
        },
        "summary": {
            "status": "VALIDATED" if not failures else "FAILED",
            "baseline_unresolved_cases": len(unresolved_ids),
            "analyzed_unresolved_cases": len(rows),
            "successful_analysis_cases": (
                len(rows) - len(failures)
            ),
            "failure_cases": len(failures),
            "recovered_total": recovered_total,
            "recovered_by_k": {
                str(k): recovered_by_k.get(k, 0)
                for k in k_values
            },
            "still_unresolved": (
                len(rows) - len(failures) - recovered_total
            ),
            "final_classification_counts": dict(
                final_classifications.most_common()
            ),
            "full_unresolved_run": full_unresolved_run,
            "overall_cases_with_valid_after_h10d": (
                overall_valid_after_h10d
            ),
            "overall_coverage_ratio_after_h10d": (
                overall_valid_after_h10d / baseline_cases
                if (
                    overall_valid_after_h10d is not None
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
    print("H10-D TOP-K FEASIBILITY COVERAGE")
    print("===============================")
    print("Status                    :", payload["summary"]["status"])
    print(
        "Baseline unresolved       :",
        payload["summary"]["baseline_unresolved_cases"],
    )
    print(
        "Analyzed unresolved       :",
        payload["summary"]["analyzed_unresolved_cases"],
    )
    print("Failures                  :", payload["summary"]["failure_cases"])
    for k_value in k_values:
        print(
            f"Recovered at K={k_value:<12}:",
            payload["summary"]["recovered_by_k"][str(k_value)],
        )
    print(
        "Recovered total           :",
        payload["summary"]["recovered_total"],
    )
    print(
        "Still unresolved          :",
        payload["summary"]["still_unresolved"],
    )

    if overall_valid_after_h10d is not None:
        print(
            "Overall cases with valid  :",
            f"{overall_valid_after_h10d}/{baseline_cases}",
        )
        print(
            "Overall coverage ratio    :",
            round(
                payload["summary"][
                    "overall_coverage_ratio_after_h10d"
                ],
                8,
            ),
        )

    print(
        "Elapsed                   :",
        payload["summary"]["elapsed_seconds"],
        "s",
    )
    print("Output                    :", args.output)

    if final_classifications:
        print("Final classifications     :")
        for code, count in final_classifications.most_common():
            print(f"  - {code}: {count}")

    if failures:
        print()
        print("Premières erreurs :")
        for row in failures[:10]:
            print("-", row["case_id"], ":", row["error"])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
