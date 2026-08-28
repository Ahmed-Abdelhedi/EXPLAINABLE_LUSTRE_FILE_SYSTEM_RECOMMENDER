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
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "architecture" / "state" / "state_transition_validation.json"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from full_architecture.architecture_state import (  # noqa: E402
    build_complete_state_from_choices,
    validate_full_architecture_state,
)
from full_architecture.catalog_loader import load_reference_catalog  # noqa: E402
from full_architecture.compatibility_rules import find_compatible_hardware_paths  # noqa: E402
from full_architecture.protection_arithmetic import enumerate_candidate_protections  # noqa: E402
from full_architecture.runtime_adapter import build_runtime_handoff  # noqa: E402


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def first_choice_with_path(
    *,
    candidates: list[dict[str, Any]],
    protection_profiles: list[dict[str, Any]],
    requirement: dict[str, Any],
    role: str,
    hardware_catalog: dict[str, Any],
    ha_required: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for candidate in candidates:
        for variant in enumerate_candidate_protections(
            candidate=candidate,
            protection_profiles=protection_profiles,
            requirement=requirement,
        ):
            paths = find_compatible_hardware_paths(
                candidate=candidate,
                protection_result=variant,
                role=role,
                hardware_catalog=hardware_catalog,
                ha_required=ha_required,
                max_paths=1,
            )
            if paths:
                return candidate, variant, paths[0]
    raise RuntimeError(f"Aucun choix déterministe compatible pour {role}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation H7 ArchitectureState + transitions.")
    parser.add_argument("--limit", type=int, default=10, help="0 = tous les cas.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--architectures", type=Path, default=DEFAULT_ARCHITECTURES)
    parser.add_argument("--drive-catalog", type=Path, default=DEFAULT_DRIVE_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("--limit doit être >= 0.")
    if args.top_k <= 0:
        raise ValueError("--top-k doit être > 0.")

    architectures = load_json(args.architectures)
    drives = load_json(args.drive_catalog)
    hardware = load_reference_catalog()
    if not isinstance(architectures, list) or not isinstance(drives, list):
        raise TypeError("Les datasets architectures/drives doivent être des listes.")

    cases = architectures if args.limit == 0 else architectures[: args.limit]
    profiles = hardware["protection_profiles"]
    rows: list[dict[str, Any]] = []
    sample_state: dict[str, Any] | None = None
    started = time.perf_counter()

    for index, architecture in enumerate(cases, start=1):
        case_id = str(architecture.get("case_id", f"CASE_{index:06d}"))
        try:
            handoff = build_runtime_handoff(
                architecture=architecture,
                catalog=drives,
                top_k=args.top_k,
            )
            reqs = handoff["requirements"]
            ha_required = bool(reqs["constraints"].get("ha_required", False))
            mdt_candidate, mdt_protection, mdt_path = first_choice_with_path(
                candidates=handoff["mdt_candidates"],
                protection_profiles=profiles,
                requirement=reqs["MDT_requirement"],
                role="MDT",
                hardware_catalog=hardware,
                ha_required=ha_required,
            )
            ost_candidate, ost_protection, ost_path = first_choice_with_path(
                candidates=handoff["ost_candidates"],
                protection_profiles=profiles,
                requirement=reqs["OST_requirement"],
                role="OST",
                hardware_catalog=hardware,
                ha_required=ha_required,
            )
            state = build_complete_state_from_choices(
                handoff=handoff,
                mdt_candidate=mdt_candidate,
                ost_candidate=ost_candidate,
                mdt_protection=mdt_protection,
                ost_protection=ost_protection,
                mdt_path=mdt_path,
                ost_path=ost_path,
            )
            validate_full_architecture_state(state)
            if sample_state is None:
                sample_state = state
            row = {
                "case_id": case_id,
                "status": "OK",
                "stage": state["stage"],
                "trace_length": len(state["trace"]),
                "mdt_drive_id": state["selected"]["mdt_drive"]["drive_id"],
                "ost_drive_id": state["selected"]["ost_drive"]["drive_id"],
                "mdt_physical_drives": state["counts"]["mdt_physical_drives"],
                "ost_physical_drives": state["counts"]["ost_physical_drives"],
                "mds_count": state["counts"]["mds_count"],
                "oss_count": state["counts"]["oss_count"],
                "total_cost_usd": round(float(state["cost_power"]["total_cost_usd"]), 4),
                "total_power_w": round(float(state["cost_power"]["total_power_w"]), 4),
                "error": "",
            }
        except Exception as error:
            row = {
                "case_id": case_id,
                "status": "FAILED",
                "stage": "",
                "trace_length": 0,
                "mdt_drive_id": "",
                "ost_drive_id": "",
                "mdt_physical_drives": 0,
                "ost_physical_drives": 0,
                "mds_count": 0,
                "oss_count": 0,
                "total_cost_usd": 0.0,
                "total_power_w": 0.0,
                "error": f"{type(error).__name__}: {error}",
            }
        rows.append(row)
        if index == 1 or index % 100 == 0 or index == len(cases):
            print(f"[{index}/{len(cases)}] {case_id} : {row['status']}")

    elapsed = time.perf_counter() - started
    failures = [row for row in rows if row["status"] != "OK"]
    payload = {
        "schema_version": "1.0",
        "purpose": "architecture_state_transition_validation",
        "selection_semantics": (
            "Validation-only canonical deterministic choice: first ranked candidate, "
            "first protection profile, first compatible hardware path. This is not "
            "an optimization or final recommendation policy."
        ),
        "summary": {
            "status": "VALIDATED" if not failures else "FAILED",
            "cases": len(rows),
            "valid_cases": len(rows) - len(failures),
            "failure_cases": len(failures),
            "top_k": args.top_k,
            "elapsed_seconds": round(elapsed, 4),
        },
        "sample_state": sample_state,
        "cases": rows,
    }
    save_json(args.output, payload)

    print()
    print("FULL ARCHITECTURE STATE H7")
    print("==========================")
    print("Status      :", payload["summary"]["status"])
    print("Cases       :", payload["summary"]["cases"])
    print("Valid cases :", payload["summary"]["valid_cases"])
    print("Failures    :", payload["summary"]["failure_cases"])
    print("Top-K       :", payload["summary"]["top_k"])
    print("Elapsed     :", payload["summary"]["elapsed_seconds"], "s")
    print("Output      :", args.output)

    if failures:
        print("\nPremières erreurs :")
        for row in failures[:10]:
            print("-", row["case_id"], ":", row["error"])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
