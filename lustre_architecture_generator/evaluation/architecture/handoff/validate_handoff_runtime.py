from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"

DEFAULT_ARCHITECTURES = (
    PROJECT_ROOT / "output" / "lustre_architecture_dataset.json"
)
DEFAULT_CATALOG = (
    PROJECT_ROOT / "data" / "catalogue_drives_ready_final.json"
)
DEFAULT_OUTPUT_JSON = (
    PROJECT_ROOT
    / "evaluation"
    / "architecture"
    / "handoff"
    / "handoff_runtime_validation.json"
)
DEFAULT_OUTPUT_CSV = (
    PROJECT_ROOT
    / "evaluation"
    / "architecture"
    / "handoff"
    / "handoff_runtime_cases.csv"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from full_architecture.handoff_contract import (  # noqa: E402
    validate_architecture_handoff,
)
from full_architecture.runtime_adapter import (  # noqa: E402
    build_runtime_handoff,
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


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation réelle Ranking -> Full Architecture Handoff."
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--architectures",
        type=Path,
        default=DEFAULT_ARCHITECTURES,
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.limit < 0:
        raise ValueError("--limit doit être >= 0.")

    if args.top_k <= 0:
        raise ValueError("--top-k doit être > 0.")

    architectures = load_json(args.architectures)
    catalog = load_json(args.catalog)

    if not isinstance(architectures, list):
        raise TypeError("Le dataset architectural doit être une liste.")

    if not isinstance(catalog, list):
        raise TypeError("Le catalogue doit être une liste.")

    selected = architectures if args.limit == 0 else architectures[: args.limit]

    if not selected:
        raise RuntimeError("Aucun cas à valider.")

    rows: list[dict[str, Any]] = []
    sample_handoff: dict[str, Any] | None = None
    started = time.perf_counter()

    for index, architecture in enumerate(selected, start=1):
        case_started = time.perf_counter()

        if isinstance(architecture, dict):
            case_id = str(
                architecture.get("case_id", f"CASE_{index:06d}")
            )
        else:
            case_id = f"CASE_{index:06d}"

        try:
            if not isinstance(architecture, dict):
                raise TypeError("Le cas architectural doit être un objet.")

            handoff = build_runtime_handoff(
                architecture=architecture,
                catalog=catalog,
                top_k=args.top_k,
            )

            errors = validate_architecture_handoff(handoff)
            if errors:
                raise RuntimeError(" | ".join(errors))

            if sample_handoff is None:
                sample_handoff = handoff

            row = {
                "case_id": case_id,
                "status": "OK",
                "mdt_candidate_count": len(handoff["mdt_candidates"]),
                "ost_candidate_count": len(handoff["ost_candidates"]),
                "duration_seconds": round(
                    time.perf_counter() - case_started,
                    6,
                ),
                "error": "",
            }

        except Exception as error:
            row = {
                "case_id": case_id,
                "status": "FAILED",
                "mdt_candidate_count": 0,
                "ost_candidate_count": 0,
                "duration_seconds": round(
                    time.perf_counter() - case_started,
                    6,
                ),
                "error": f"{type(error).__name__}: {error}",
            }

        rows.append(row)

        if index == 1 or index % 100 == 0 or index == len(selected):
            print(
                f"[{index}/{len(selected)}] "
                f"{case_id} : {row['status']}"
            )

    elapsed = time.perf_counter() - started
    failed = [row for row in rows if row["status"] != "OK"]
    valid = [row for row in rows if row["status"] == "OK"]
    durations = [
        float(row["duration_seconds"])
        for row in valid
    ]

    summary = {
        "status": "VALIDATED" if not failed else "FAILED",
        "cases": len(rows),
        "valid_cases": len(valid),
        "failure_cases": len(failed),
        "requested_top_k": args.top_k,
        "mean_case_seconds": (
            round(statistics.mean(durations), 6)
            if durations
            else 0.0
        ),
        "elapsed_seconds": round(elapsed, 4),
    }

    save_json(
        args.output_json,
        {
            "schema_version": "1.0",
            "purpose": (
                "real_ranking_to_full_architecture_handoff_validation"
            ),
            "summary": summary,
            "sample_handoff": sample_handoff,
            "cases": rows,
        },
    )
    save_csv(args.output_csv, rows)

    print()
    print("RANKING -> FULL ARCHITECTURE HANDOFF")
    print("====================================")
    print("Status      :", summary["status"])
    print("Cases       :", summary["cases"])
    print("Valid cases :", summary["valid_cases"])
    print("Failures    :", summary["failure_cases"])
    print("Top-K       :", summary["requested_top_k"])
    print("Mean/case   :", summary["mean_case_seconds"], "s")
    print("Elapsed     :", summary["elapsed_seconds"], "s")
    print("JSON        :", args.output_json)
    print("CSV         :", args.output_csv)

    if failed:
        print()
        print("Premières erreurs :")
        for row in failed[:10]:
            print("-", row["case_id"], ":", row["error"])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
