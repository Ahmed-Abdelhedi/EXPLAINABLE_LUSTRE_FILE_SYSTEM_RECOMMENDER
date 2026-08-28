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
    / "compatibility"
    / "compatibility_runtime_validation.json"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


from full_architecture.catalog_loader import (  # noqa: E402
    load_reference_catalog,
)
from full_architecture.compatibility_rules import (  # noqa: E402
    find_compatible_hardware_paths,
)
from full_architecture.protection_arithmetic import (  # noqa: E402
    enumerate_candidate_protections,
)
from full_architecture.runtime_adapter import (  # noqa: E402
    build_runtime_handoff,
)


def load_json(
    path: Path,
) -> Any:
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
            "Validation H6 des chemins hardware compatibles."
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
        "--max-paths",
        type=int,
        default=1,
        help=(
            "Nombre de chemins compatibles à chercher par variante."
        ),
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

    if args.top_k <= 0:
        raise ValueError(
            "--top-k doit être > 0."
        )

    if args.max_paths <= 0:
        raise ValueError(
            "--max-paths doit être > 0."
        )

    architectures = load_json(
        args.architectures
    )

    drive_catalog = load_json(
        args.drive_catalog
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

    selected = (
        architectures
        if args.limit == 0
        else architectures[
            :args.limit
        ]
    )

    profiles = (
        hardware_catalog[
            "protection_profiles"
        ]
    )

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
                    catalog=drive_catalog,
                    top_k=args.top_k,
                )
            )

            requirements = (
                handoff[
                    "requirements"
                ]
            )

            constraints = (
                requirements[
                    "constraints"
                ]
            )

            ha_required = bool(
                constraints.get(
                    "ha_required",
                    False,
                )
            )

            total_variants = 0
            compatible_variants = 0

            role_candidate_counts = {
                "MDT": 0,
                "OST": 0,
            }

            role_candidates_with_path = {
                "MDT": 0,
                "OST": 0,
            }

            for role_key, candidate_key, requirement_key in (
                (
                    "MDT",
                    "mdt_candidates",
                    "MDT_requirement",
                ),
                (
                    "OST",
                    "ost_candidates",
                    "OST_requirement",
                ),
            ):
                requirement = (
                    requirements[
                        requirement_key
                    ]
                )

                for candidate in handoff[
                    candidate_key
                ]:
                    role_candidate_counts[
                        role_key
                    ] += 1

                    variants = (
                        enumerate_candidate_protections(
                            candidate=candidate,
                            protection_profiles=(
                                profiles
                            ),
                            requirement=(
                                requirement
                            ),
                        )
                    )

                    candidate_has_path = (
                        False
                    )

                    for variant in variants:
                        total_variants += 1

                        paths = (
                            find_compatible_hardware_paths(
                                candidate=candidate,
                                protection_result=(
                                    variant
                                ),
                                role=role_key,
                                hardware_catalog=(
                                    hardware_catalog
                                ),
                                ha_required=(
                                    ha_required
                                ),
                                max_paths=(
                                    args.max_paths
                                ),
                            )
                        )

                        if paths:
                            compatible_variants += 1
                            candidate_has_path = True

                    if candidate_has_path:
                        role_candidates_with_path[
                            role_key
                        ] += 1

            missing_roles = [
                role
                for role in (
                    "MDT",
                    "OST",
                )
                if role_candidates_with_path[
                    role
                ] <= 0
            ]

            if missing_roles:
                raise RuntimeError(
                    "Aucun candidat avec chemin hardware compatible pour : "
                    + ", ".join(
                        missing_roles
                    )
                )

            row = {
                "case_id": case_id,
                "status": "OK",
                "variants": total_variants,
                "compatible_variants": (
                    compatible_variants
                ),
                "pruned_variants": (
                    total_variants
                    - compatible_variants
                ),
                "mdt_candidates": (
                    role_candidate_counts[
                        "MDT"
                    ]
                ),
                "mdt_candidates_with_path": (
                    role_candidates_with_path[
                        "MDT"
                    ]
                ),
                "ost_candidates": (
                    role_candidate_counts[
                        "OST"
                    ]
                ),
                "ost_candidates_with_path": (
                    role_candidates_with_path[
                        "OST"
                    ]
                ),
                "ha_required": (
                    ha_required
                ),
                "error": "",
            }

        except Exception as error:
            row = {
                "case_id": case_id,
                "status": "FAILED",
                "variants": 0,
                "compatible_variants": 0,
                "pruned_variants": 0,
                "mdt_candidates": 0,
                "mdt_candidates_with_path": 0,
                "ost_candidates": 0,
                "ost_candidates_with_path": 0,
                "ha_required": False,
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
        if row[
            "status"
        ]
        != "OK"
    ]

    total_variants = sum(
        int(
            row[
                "variants"
            ]
        )
        for row in rows
        if row[
            "status"
        ]
        == "OK"
    )

    compatible_variants = sum(
        int(
            row[
                "compatible_variants"
            ]
        )
        for row in rows
        if row[
            "status"
        ]
        == "OK"
    )

    payload = {
        "schema_version": "1.1",
        "purpose": (
            "full_architecture_hardware_compatibility_validation"
        ),
        "validation_semantics": (
            "Compatibility is a hard-pruning layer. "
            "Individual protection/hardware combinations may be rejected. "
            "A case is valid when at least one MDT candidate and one OST "
            "candidate retain at least one compatible hardware path."
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
            "top_k": args.top_k,
            "protection_profiles": len(
                profiles
            ),
            "total_variants": (
                total_variants
            ),
            "compatible_variants": (
                compatible_variants
            ),
            "pruned_variants": (
                total_variants
                - compatible_variants
            ),
            "compatibility_ratio": (
                round(
                    compatible_variants
                    / total_variants,
                    8,
                )
                if total_variants
                else 0.0
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
        "FULL ARCHITECTURE COMPATIBILITY H6"
    )
    print(
        "=================================="
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
        "Compatible variants :",
        payload[
            "summary"
        ][
            "compatible_variants"
        ],
        "/",
        payload[
            "summary"
        ][
            "total_variants"
        ],
    )
    print(
        "Pruned variants     :",
        payload[
            "summary"
        ][
            "pruned_variants"
        ],
    )
    print(
        "Compatibility ratio :",
        payload[
            "summary"
        ][
            "compatibility_ratio"
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

        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
