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
DEFAULT_H10B = (
    PROJECT_ROOT
    / "evaluation"
    / "architecture"
    / "validation"
    / "feasibility_coverage_path2.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "evaluation"
    / "architecture"
    / "validation"
    / "feasibility_coverage_full_paths.json"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from full_architecture.catalog_loader import (  # noqa: E402
    load_reference_catalog,
)
from full_architecture.feasibility_coverage import (  # noqa: E402
    analyze_case_full_path_domain,
    unresolved_case_ids_from_h10b,
)
from full_architecture.runtime_adapter import (  # noqa: E402
    build_runtime_handoff,
)


def load_json(path: Path) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def save_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
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
            "H10-C : analyse exhaustive du domaine hardware-path "
            "pour les cas restant non résolus après H10-B."
        )
    )

    parser.add_argument(
        "--limit-unresolved",
        type=int,
        default=10,
        help="0 = tous les cas non résolus du résultat H10-B.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
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
        "--h10b-result",
        type=Path,
        default=DEFAULT_H10B,
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
        raise ValueError(
            "--limit-unresolved doit être >= 0."
        )

    if args.top_k <= 0:
        raise ValueError(
            "--top-k doit être > 0."
        )

    architectures = load_json(
        args.architectures
    )
    drive_catalog = load_json(
        args.drive_catalog
    )
    h10b = load_json(
        args.h10b_result
    )
    hardware_catalog = (
        load_reference_catalog()
    )

    if not isinstance(
        architectures,
        list,
    ):
        raise TypeError(
            "architectures doit être une liste."
        )

    if not isinstance(
        drive_catalog,
        list,
    ):
        raise TypeError(
            "drive_catalog doit être une liste."
        )

    unresolved_ids = (
        unresolved_case_ids_from_h10b(
            h10b
        )
    )

    selected_ids = (
        unresolved_ids
        if args.limit_unresolved == 0
        else unresolved_ids[
            :args.limit_unresolved
        ]
    )

    by_case = {
        str(
            item.get(
                "case_id"
            )
        ): item
        for item in architectures
        if isinstance(
            item,
            dict,
        )
    }

    rows: list[
        dict[str, Any]
    ] = []

    classifications: Counter[
        str
    ] = Counter()

    recovered = 0
    started = (
        time.perf_counter()
    )

    for index, case_id in enumerate(
        selected_ids,
        start=1,
    ):
        architecture = by_case.get(
            case_id
        )

        if architecture is None:
            row = {
                "case_id": case_id,
                "status": "FAILED",
                "recovered_valid_architecture": False,
                "classification": "CASE_NOT_FOUND",
                "error": (
                    "case_id absent du dataset architecture."
                ),
            }

            rows.append(
                row
            )
            classifications.update(
                ["CASE_NOT_FOUND"]
            )
            continue

        try:
            handoff = build_runtime_handoff(
                architecture=architecture,
                catalog=drive_catalog,
                top_k=args.top_k,
            )

            analysis = (
                analyze_case_full_path_domain(
                    handoff=handoff,
                    hardware_catalog=(
                        hardware_catalog
                    ),
                )
            )

            was_recovered = bool(
                analysis[
                    "recovered_valid_architecture"
                ]
            )

            if was_recovered:
                recovered += 1

            classification = (
                analysis[
                    "search"
                ][
                    "classification"
                ]
            )

            classifications.update(
                [classification]
            )

            row = {
                "case_id": case_id,
                "status": "OK",
                "recovered_valid_architecture": (
                    was_recovered
                ),
                "coverage_interpretation": (
                    analysis[
                        "coverage_interpretation"
                    ]
                ),
                "classification": (
                    classification
                ),
                "hardware_path_upper_bound": (
                    analysis[
                        "domain"
                    ][
                        "hardware_path_upper_bound"
                    ]
                ),
                "mdt_raw_options": (
                    analysis[
                        "option_counts"
                    ][
                        "mdt_raw"
                    ]
                ),
                "ost_raw_options": (
                    analysis[
                        "option_counts"
                    ][
                        "ost_raw"
                    ]
                ),
                "mdt_pareto_options": (
                    analysis[
                        "option_counts"
                    ][
                        "mdt_pareto"
                    ]
                ),
                "ost_pareto_options": (
                    analysis[
                        "option_counts"
                    ][
                        "ost_pareto"
                    ]
                ),
                "minimum_total_cost_usd": (
                    analysis[
                        "search"
                    ][
                        "minimum_total_cost_usd"
                    ]
                ),
                "maximum_budget_usd": (
                    analysis[
                        "limits"
                    ][
                        "maximum_budget_usd"
                    ]
                ),
                "minimum_total_power_w": (
                    analysis[
                        "search"
                    ][
                        "minimum_total_power_w"
                    ]
                ),
                "maximum_power_w": (
                    analysis[
                        "limits"
                    ][
                        "maximum_power_w"
                    ]
                ),
                "selected_path_indexes": (
                    analysis[
                        "selected_path_indexes"
                    ]
                ),
                "recovered_architecture_id": (
                    analysis[
                        "recovered_architecture_id"
                    ]
                ),
                "error": "",
            }

        except Exception as error:
            row = {
                "case_id": case_id,
                "status": "FAILED",
                "recovered_valid_architecture": False,
                "classification": "EXECUTION_ERROR",
                "error": (
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
            }

            classifications.update(
                ["EXECUTION_ERROR"]
            )

        rows.append(
            row
        )

        if (
            index == 1
            or index % 25 == 0
            or index == len(
                selected_ids
            )
        ):
            print(
                f"[{index}/{len(selected_ids)}] "
                f"{case_id} : "
                f"{row['status']} "
                "recovered="
                f"{row['recovered_valid_architecture']}"
            )

    elapsed = (
        time.perf_counter()
        - started
    )

    failures = [
        row
        for row in rows
        if row[
            "status"
        ]
        != "OK"
    ]

    h10b_summary = h10b.get(
        "summary",
        {},
    )

    baseline_valid_after_h10b = int(
        h10b_summary.get(
            "overall_cases_with_valid_after_expansion",
            0,
        )
    )

    baseline_cases = int(
        h10b.get(
            "baseline",
            {},
        ).get(
            "cases",
            0,
        )
    )

    full_unresolved_run = (
        args.limit_unresolved == 0
        or len(
            selected_ids
        )
        == len(
            unresolved_ids
        )
    )

    overall_valid_after_h10c = (
        baseline_valid_after_h10b
        + recovered
        if full_unresolved_run
        else None
    )

    payload = {
        "schema_version": "1.0",
        "purpose": (
            "h10c_full_hardware_path_"
            "feasibility_coverage"
        ),
        "semantics": {
            "h10b_result_is_baseline": True,
            "top_k_fixed": args.top_k,
            "max_role_options_removed": True,
            "hardware_path_cap_removed": True,
            "hardware_path_domain_exhaustive_for_current_catalog": True,
            "pareto_reduction_preserves_cost_power_feasibility": True,
            "recovered_pair_confirmed_by_h10": True,
            "beam_search_applied": False,
            "global_infeasibility_is_not_claimed": True,
            "remaining_if_unresolved": (
                "Top-K candidate truncation and "
                "reference-catalog scope"
            ),
        },
        "baseline": {
            "cases": baseline_cases,
            "cases_with_valid_after_h10b": (
                baseline_valid_after_h10b
            ),
            "unresolved_after_h10b": len(
                unresolved_ids
            ),
        },
        "limits": {
            "limit_unresolved": (
                args.limit_unresolved
            ),
            "top_k": args.top_k,
        },
        "summary": {
            "status": (
                "VALIDATED"
                if not failures
                else "FAILED"
            ),
            "baseline_unresolved_cases": len(
                unresolved_ids
            ),
            "analyzed_unresolved_cases": len(
                rows
            ),
            "successful_analysis_cases": (
                len(
                    rows
                )
                - len(
                    failures
                )
            ),
            "failure_cases": len(
                failures
            ),
            "recovered_by_full_path_expansion": (
                recovered
            ),
            "still_unresolved": (
                len(
                    rows
                )
                - len(
                    failures
                )
                - recovered
            ),
            "classification_counts": dict(
                classifications.most_common()
            ),
            "full_unresolved_run": (
                full_unresolved_run
            ),
            "overall_cases_with_valid_after_h10c": (
                overall_valid_after_h10c
            ),
            "overall_coverage_ratio_after_h10c": (
                (
                    overall_valid_after_h10c
                    / baseline_cases
                )
                if (
                    overall_valid_after_h10c
                    is not None
                    and baseline_cases > 0
                )
                else None
            ),
            "elapsed_seconds": round(
                elapsed,
                4,
            ),
        },
        "cases": rows,
    }

    save_json(
        args.output,
        payload,
    )

    print()
    print(
        "H10-C FULL HARDWARE PATH COVERAGE"
    )
    print(
        "================================="
    )
    print(
        "Status                    :",
        payload[
            "summary"
        ][
            "status"
        ],
    )
    print(
        "Baseline unresolved       :",
        payload[
            "summary"
        ][
            "baseline_unresolved_cases"
        ],
    )
    print(
        "Analyzed unresolved       :",
        payload[
            "summary"
        ][
            "analyzed_unresolved_cases"
        ],
    )
    print(
        "Failures                  :",
        payload[
            "summary"
        ][
            "failure_cases"
        ],
    )
    print(
        "Recovered by path expansion:",
        payload[
            "summary"
        ][
            "recovered_by_full_path_expansion"
        ],
    )
    print(
        "Still unresolved          :",
        payload[
            "summary"
        ][
            "still_unresolved"
        ],
    )

    if (
        overall_valid_after_h10c
        is not None
    ):
        print(
            "Overall cases with valid  :",
            f"{overall_valid_after_h10c}/"
            f"{baseline_cases}",
        )
        print(
            "Overall coverage ratio    :",
            round(
                payload[
                    "summary"
                ][
                    "overall_coverage_ratio_after_h10c"
                ],
                8,
            ),
        )

    print(
        "Elapsed                   :",
        payload[
            "summary"
        ][
            "elapsed_seconds"
        ],
        "s",
    )
    print(
        "Output                    :",
        args.output,
    )

    if classifications:
        print(
            "Classifications           :"
        )

        for code, count in (
            classifications.most_common()
        ):
            print(
                f"  - {code}: {count}"
            )

    if failures:
        print()
        print(
            "Premières erreurs :"
        )

        for row in failures[
            :10
        ]:
            print(
                "-",
                row[
                    "case_id"
                ],
                ":",
                row[
                    "error"
                ],
            )

        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
