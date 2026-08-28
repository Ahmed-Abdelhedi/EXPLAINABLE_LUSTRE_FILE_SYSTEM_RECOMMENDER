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
    PROJECT_ROOT / "output" / "lustre_architecture_dataset.json"
)
DEFAULT_DRIVE_CATALOG = (
    PROJECT_ROOT / "data" / "catalogue_drives_ready_final.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "evaluation"
    / "architecture"
    / "validation"
    / "full_architecture_validation.json"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from full_architecture.catalog_loader import load_reference_catalog  # noqa: E402
from full_architecture.full_architecture_generator import (  # noqa: E402
    generate_full_architectures,
)
from full_architecture.full_architecture_validator import (  # noqa: E402
    assert_full_validation_result_valid,
    validate_generated_architectures,
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
        description="Validation déterministe complète H10."
    )
    parser.add_argument("--limit", type=int, default=10, help="0 = tous les cas.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-paths-per-variant", type=int, default=2)
    parser.add_argument("--max-role-options", type=int, default=4)
    parser.add_argument("--max-architectures", type=int, default=16)
    parser.add_argument("--architectures", type=Path, default=DEFAULT_ARCHITECTURES)
    parser.add_argument("--drive-catalog", type=Path, default=DEFAULT_DRIVE_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.limit < 0:
        raise ValueError("--limit doit être >= 0.")

    architectures = load_json(args.architectures)
    drive_catalog = load_json(args.drive_catalog)
    hardware_catalog = load_reference_catalog()

    if not isinstance(architectures, list):
        raise TypeError("architectures doit être une liste.")
    if not isinstance(drive_catalog, list):
        raise TypeError("drive_catalog doit être une liste.")

    selected = architectures if args.limit == 0 else architectures[: args.limit]

    rows: list[dict[str, Any]] = []
    total_architectures = 0
    total_valid = 0
    total_invalid = 0
    violation_counts: Counter[str] = Counter()
    started = time.perf_counter()

    for index, architecture in enumerate(selected, start=1):
        case_id = str(architecture.get("case_id", f"CASE_{index:06d}"))

        try:
            handoff = build_runtime_handoff(
                architecture=architecture,
                catalog=drive_catalog,
                top_k=args.top_k,
            )
            generated = generate_full_architectures(
                handoff=handoff,
                hardware_catalog=hardware_catalog,
                max_paths_per_variant=args.max_paths_per_variant,
                max_role_options_per_role=args.max_role_options,
                max_architectures=args.max_architectures,
            )
            validated = validate_generated_architectures(
                generation_result=generated,
                handoff=handoff,
                hardware_catalog=hardware_catalog,
            )
            assert_full_validation_result_valid(validated)

            # H10 doit être strictement déterministe.
            repeated = validate_generated_architectures(
                generation_result=generated,
                handoff=handoff,
                hardware_catalog=hardware_catalog,
            )
            if validated != repeated:
                raise RuntimeError("Validation H10 non déterministe.")

            summary = validated["summary"]
            architecture_count = int(summary["architecture_count"])
            valid_count = int(summary["valid_architecture_count"])
            invalid_count = int(summary["invalid_architecture_count"])

            total_architectures += architecture_count
            total_valid += valid_count
            total_invalid += invalid_count
            violation_counts.update(summary["violation_code_counts"])

            row = {
                "case_id": case_id,
                "status": "OK",
                "architectures_validated": architecture_count,
                "valid_architectures": valid_count,
                "invalid_architectures": invalid_count,
                "has_valid_architecture": bool(summary["has_valid_architecture"]),
                "first_valid_architecture_id": summary[
                    "first_valid_architecture_id"
                ],
                "violation_code_counts": summary["violation_code_counts"],
                "error": "",
            }

        except Exception as error:
            row = {
                "case_id": case_id,
                "status": "FAILED",
                "architectures_validated": 0,
                "valid_architectures": 0,
                "invalid_architectures": 0,
                "has_valid_architecture": False,
                "first_valid_architecture_id": None,
                "violation_code_counts": {},
                "error": f"{type(error).__name__}: {error}",
            }

        rows.append(row)

        if index == 1 or index % 100 == 0 or index == len(selected):
            print(f"[{index}/{len(selected)}] {case_id} : {row['status']}")

    elapsed = time.perf_counter() - started
    failures = [row for row in rows if row["status"] != "OK"]
    cases_with_valid = sum(
        1
        for row in rows
        if row["status"] == "OK" and row["has_valid_architecture"]
    )

    payload = {
        "schema_version": "1.0",
        "purpose": "full_architecture_deterministic_validation",
        "validation_semantics": {
            "validator_is_independent_of_h9_score": True,
            "recomputes_h5_protection": True,
            "recomputes_h6_hardware_path": True,
            "checks_h7_aggregates": True,
            "checks_hard_requirements_budget_power_ha": True,
            "beam_search_applied": False,
            "case_without_valid_architecture_is_not_runtime_failure": True,
        },
        "limits": {
            "top_k": args.top_k,
            "max_paths_per_variant": args.max_paths_per_variant,
            "max_role_options": args.max_role_options,
            "max_architectures": args.max_architectures,
        },
        "summary": {
            "status": "VALIDATED" if not failures else "FAILED",
            "cases": len(rows),
            "valid_execution_cases": len(rows) - len(failures),
            "failure_cases": len(failures),
            "architectures_validated": total_architectures,
            "valid_architectures": total_valid,
            "invalid_architectures": total_invalid,
            "cases_with_valid_architecture": cases_with_valid,
            "cases_without_valid_architecture": (
                len(rows) - len(failures) - cases_with_valid
            ),
            "valid_architecture_ratio": (
                total_valid / total_architectures
                if total_architectures
                else 0.0
            ),
            "violation_code_counts": dict(violation_counts.most_common()),
            "elapsed_seconds": round(elapsed, 4),
        },
        "cases": rows,
    }

    save_json(args.output, payload)

    print()
    print("FULL ARCHITECTURE VALIDATOR H10")
    print("===============================")
    print("Status                    :", payload["summary"]["status"])
    print("Cases                     :", payload["summary"]["cases"])
    print(
        "Valid execution cases     :",
        payload["summary"]["valid_execution_cases"],
    )
    print("Failures                  :", payload["summary"]["failure_cases"])
    print(
        "Architectures validated   :",
        payload["summary"]["architectures_validated"],
    )
    print("Valid architectures       :", payload["summary"]["valid_architectures"])
    print("Invalid architectures     :", payload["summary"]["invalid_architectures"])
    print(
        "Cases with valid arch     :",
        payload["summary"]["cases_with_valid_architecture"],
    )
    print(
        "Cases without valid arch  :",
        payload["summary"]["cases_without_valid_architecture"],
    )
    print(
        "Valid architecture ratio  :",
        round(payload["summary"]["valid_architecture_ratio"], 8),
    )
    print("Elapsed                   :", payload["summary"]["elapsed_seconds"], "s")
    print("Output                    :", args.output)

    if violation_counts:
        print("Top violation codes       :")
        for code, count in violation_counts.most_common(8):
            print(f"  - {code}: {count}")

    if failures:
        print()
        print("Premières erreurs d'exécution :")
        for row in failures[:10]:
            print("-", row["case_id"], ":", row["error"])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
