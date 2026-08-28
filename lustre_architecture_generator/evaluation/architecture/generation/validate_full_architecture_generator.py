from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"

DEFAULT_ARCHITECTURES = PROJECT_ROOT / "output" / "lustre_architecture_dataset.json"
DEFAULT_DRIVE_CATALOG = PROJECT_ROOT / "data" / "catalogue_drives_ready_final.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "evaluation"
    / "architecture"
    / "generation"
    / "full_architecture_generation_validation.json"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from full_architecture.catalog_loader import load_reference_catalog  # noqa: E402
from full_architecture.full_architecture_generator import (  # noqa: E402
    generate_full_architectures,
)
from full_architecture.runtime_adapter import build_runtime_handoff  # noqa: E402


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation runtime H8 FullArchitectureGenerator."
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
    for name in (
        "top_k",
        "max_paths_per_variant",
        "max_role_options",
        "max_architectures",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} doit être > 0.")

    architectures = load_json(args.architectures)
    drive_catalog = load_json(args.drive_catalog)
    hardware_catalog = load_reference_catalog()

    if not isinstance(architectures, list):
        raise TypeError("architectures doit être une liste.")
    if not isinstance(drive_catalog, list):
        raise TypeError("drive_catalog doit être une liste.")

    selected = architectures if args.limit == 0 else architectures[: args.limit]
    rows: list[dict[str, Any]] = []
    sample_architecture: dict[str, Any] | None = None
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

            records = generated["architectures"]
            if not records:
                raise RuntimeError("Aucune architecture générée.")

            ids = [record["architecture_id"] for record in records]
            if len(ids) != len(set(ids)):
                raise RuntimeError("architecture_id dupliqué.")

            for record in records:
                state = record["state"]
                if state.get("stage") != "COMPLETE":
                    raise RuntimeError("State H8 non COMPLETE.")
                validation = state.get("validation", {})
                if validation.get("status") != "PENDING_FULL_VALIDATOR":
                    raise RuntimeError("State H8 ne doit pas être final-validé.")
                semantics = record.get("generation_semantics", {})
                if semantics.get("beam_search_applied") is not False:
                    raise RuntimeError("Beam Search ne doit pas être appliqué en H8.")
                if semantics.get("architecture_score_applied") is not False:
                    raise RuntimeError("Scoring architecture ne doit pas être appliqué en H8.")

            if sample_architecture is None:
                sample_architecture = records[0]

            summary = generated["summary"]
            row = {
                "case_id": case_id,
                "status": "OK",
                "mdt_role_options": summary["mdt_role_options"],
                "ost_role_options": summary["ost_role_options"],
                "potential_pair_count": summary["potential_pair_count"],
                "generated_architecture_count": summary["generated_architecture_count"],
                "truncated": summary["truncated_by_max_architectures"],
                "error": "",
            }
        except Exception as error:
            row = {
                "case_id": case_id,
                "status": "FAILED",
                "mdt_role_options": 0,
                "ost_role_options": 0,
                "potential_pair_count": 0,
                "generated_architecture_count": 0,
                "truncated": False,
                "error": f"{type(error).__name__}: {error}",
            }

        rows.append(row)

        if index == 1 or index % 100 == 0 or index == len(selected):
            print(f"[{index}/{len(selected)}] {case_id} : {row['status']}")

    elapsed = time.perf_counter() - started
    failures = [row for row in rows if row["status"] != "OK"]
    total_generated = sum(
        int(row["generated_architecture_count"])
        for row in rows
        if row["status"] == "OK"
    )

    payload = {
        "schema_version": "1.0",
        "purpose": "full_architecture_generator_runtime_validation",
        "validation_semantics": (
            "Controlled H8 generation domain. max_role_options and max_architectures "
            "are explicit validation bounds, not architecture scores or Beam Search."
        ),
        "configuration": {
            "top_k": args.top_k,
            "max_paths_per_variant": args.max_paths_per_variant,
            "max_role_options": args.max_role_options,
            "max_architectures": args.max_architectures,
        },
        "summary": {
            "status": "VALIDATED" if not failures else "FAILED",
            "cases": len(rows),
            "valid_cases": len(rows) - len(failures),
            "failure_cases": len(failures),
            "total_generated_architectures": total_generated,
            "mean_generated_per_case": (
                round(total_generated / len(rows), 6) if rows else 0.0
            ),
            "elapsed_seconds": round(elapsed, 4),
        },
        "sample_architecture": sample_architecture,
        "cases": rows,
    }

    save_json(args.output, payload)

    print()
    print("FULL ARCHITECTURE GENERATOR H8")
    print("==============================")
    print("Status                  :", payload["summary"]["status"])
    print("Cases                   :", payload["summary"]["cases"])
    print("Valid cases             :", payload["summary"]["valid_cases"])
    print("Failures                :", payload["summary"]["failure_cases"])
    print("Generated architectures :", payload["summary"]["total_generated_architectures"])
    print("Mean / case             :", payload["summary"]["mean_generated_per_case"])
    print("Elapsed                 :", payload["summary"]["elapsed_seconds"], "s")
    print("Output                  :", args.output)

    if failures:
        print()
        print("Premières erreurs :")
        for row in failures[:10]:
            print("-", row["case_id"], ":", row["error"])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
