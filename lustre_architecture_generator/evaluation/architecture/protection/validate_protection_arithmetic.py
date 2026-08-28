from __future__ import annotations

import argparse
import json
import sys
import time
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

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "evaluation"
    / "architecture"
    / "protection"
    / "protection_arithmetic_validation.json"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


from full_architecture.catalog_loader import (  # noqa: E402
    load_reference_catalog,
)
from full_architecture.protection_arithmetic import (  # noqa: E402
    assert_protection_result_valid,
    enumerate_candidate_protections,
)
from full_architecture.runtime_adapter import (  # noqa: E402
    build_runtime_handoff,
)


def load_json(path: Path) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(
            handle
        )


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
            "Validation réelle de l'arithmétique post-RAID H5."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="0 = tous les cas.",
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
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.limit < 0:
        raise ValueError(
            "--limit doit être >= 0."
        )

    architectures = load_json(
        args.architectures
    )

    drives = load_json(
        args.drive_catalog
    )

    hardware = (
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
        drives,
        list,
    ):
        raise TypeError(
            "drive catalog doit être une liste."
        )

    selected = (
        architectures
        if args.limit == 0
        else architectures[
            :args.limit
        ]
    )

    profiles = hardware[
        "protection_profiles"
    ]

    rows: list[
        dict[str, Any]
    ] = []

    started = (
        time.perf_counter()
    )

    for index, architecture in enumerate(
        selected,
        start=1,
    ):
        case_id = str(
            architecture.get(
                "case_id",
                f"CASE_{index:06d}",
            )
        )

        try:
            handoff = (
                build_runtime_handoff(
                    architecture=architecture,
                    catalog=drives,
                    top_k=args.top_k,
                )
            )

            mdt_requirement = (
                handoff[
                    "requirements"
                ][
                    "MDT_requirement"
                ]
            )

            ost_requirement = (
                handoff[
                    "requirements"
                ][
                    "OST_requirement"
                ]
            )

            mdt_results_count = 0
            ost_results_count = 0

            max_mdt_physical = 0
            max_ost_physical = 0

            for candidate in handoff[
                "mdt_candidates"
            ]:
                results = (
                    enumerate_candidate_protections(
                        candidate=candidate,
                        protection_profiles=(
                            profiles
                        ),
                        requirement=(
                            mdt_requirement
                        ),
                    )
                )

                for result in results:
                    assert_protection_result_valid(
                        result
                    )

                mdt_results_count += len(
                    results
                )

                max_mdt_physical = max(
                    max_mdt_physical,
                    max(
                        int(
                            result[
                                "physical_drive_count"
                            ]
                        )
                        for result in results
                    ),
                )

            for candidate in handoff[
                "ost_candidates"
            ]:
                results = (
                    enumerate_candidate_protections(
                        candidate=candidate,
                        protection_profiles=(
                            profiles
                        ),
                        requirement=(
                            ost_requirement
                        ),
                    )
                )

                for result in results:
                    assert_protection_result_valid(
                        result
                    )

                ost_results_count += len(
                    results
                )

                max_ost_physical = max(
                    max_ost_physical,
                    max(
                        int(
                            result[
                                "physical_drive_count"
                            ]
                        )
                        for result in results
                    ),
                )

            row = {
                "case_id": case_id,
                "status": "OK",
                "mdt_variants": (
                    mdt_results_count
                ),
                "ost_variants": (
                    ost_results_count
                ),
                "max_mdt_physical_drives": (
                    max_mdt_physical
                ),
                "max_ost_physical_drives": (
                    max_ost_physical
                ),
                "error": "",
            }

        except Exception as error:
            row = {
                "case_id": case_id,
                "status": "FAILED",
                "mdt_variants": 0,
                "ost_variants": 0,
                "max_mdt_physical_drives": 0,
                "max_ost_physical_drives": 0,
                "error": (
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
            }

        rows.append(
            row
        )

        if (
            index == 1
            or index % 100 == 0
            or index == len(selected)
        ):
            print(
                f"[{index}/{len(selected)}] "
                f"{case_id} : "
                f"{row['status']}"
            )

    elapsed = (
        time.perf_counter()
        - started
    )

    failures = [
        row
        for row in rows
        if row["status"] != "OK"
    ]

    payload = {
        "schema_version": "1.0",
        "purpose": (
            "post_raid_protection_arithmetic_validation"
        ),
        "summary": {
            "status": (
                "VALIDATED"
                if not failures
                else "FAILED"
            ),
            "cases": len(
                rows
            ),
            "valid_cases": (
                len(rows)
                - len(failures)
            ),
            "failure_cases": (
                len(failures)
            ),
            "top_k": args.top_k,
            "protection_profiles": (
                len(
                    profiles
                )
            ),
            "elapsed_seconds": (
                round(
                    elapsed,
                    4,
                )
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
        "FULL ARCHITECTURE PROTECTION H5"
    )
    print(
        "==============================="
    )
    print(
        "Status              :",
        payload[
            "summary"
        ][
            "status"
        ],
    )
    print(
        "Cases               :",
        payload[
            "summary"
        ][
            "cases"
        ],
    )
    print(
        "Valid cases         :",
        payload[
            "summary"
        ][
            "valid_cases"
        ],
    )
    print(
        "Failures            :",
        payload[
            "summary"
        ][
            "failure_cases"
        ],
    )
    print(
        "Protection profiles :",
        payload[
            "summary"
        ][
            "protection_profiles"
        ],
    )
    print(
        "Elapsed             :",
        payload[
            "summary"
        ][
            "elapsed_seconds"
        ],
        "s",
    )
    print(
        "Output              :",
        args.output,
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

        raise SystemExit(1)


if __name__ == "__main__":
    main()
