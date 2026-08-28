from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from statistics import fmean
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
    / "scoring"
    / "architecture_scoring_validation.json"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from full_architecture.architecture_scoring import (  # noqa: E402
    assert_scoring_result_valid,
    score_generated_architectures,
)
from full_architecture.catalog_loader import (  # noqa: E402
    load_reference_catalog,
)
from full_architecture.full_architecture_generator import (  # noqa: E402
    generate_full_architectures,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation runtime H9 du scoring architecture."
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
        "--max-paths-per-variant",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max-role-options",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--max-architectures",
        type=int,
        default=16,
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
        raise ValueError("--limit doit être >= 0.")

    architectures = load_json(args.architectures)
    drive_catalog = load_json(args.drive_catalog)
    hardware_catalog = load_reference_catalog()

    if not isinstance(architectures, list):
        raise TypeError("architectures doit être une liste.")
    if not isinstance(drive_catalog, list):
        raise TypeError("drive_catalog doit être une liste.")

    selected = (
        architectures
        if args.limit == 0
        else architectures[: args.limit]
    )

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    total_scored = 0
    total_snapshot_pass = 0
    best_scores: list[float] = []

    for index, architecture in enumerate(selected, start=1):
        case_id = str(
            architecture.get("case_id", f"CASE_{index:06d}")
        )

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

            scored = score_generated_architectures(
                generation_result=generated,
                handoff=handoff,
            )
            assert_scoring_result_valid(scored)

            repeated = score_generated_architectures(
                generation_result=generated,
                handoff=handoff,
            )

            if scored != repeated:
                raise RuntimeError("Scoring H9 non déterministe.")

            summary = scored["summary"]
            architecture_count = int(summary["architecture_count"])
            snapshot_pass_count = int(
                summary["pre_h10_hard_snapshot_pass_count"]
            )
            best_score = summary["best_pre_h10_score"]

            if architecture_count != generated["summary"][
                "generated_architecture_count"
            ]:
                raise RuntimeError(
                    "Nombre d'architectures scorées incohérent."
                )

            for item in scored["architectures"]:
                value = float(item["score"])
                if not math.isfinite(value) or not (0.0 <= value <= 1.0):
                    raise RuntimeError("Score H9 hors [0, 1].")

            total_scored += architecture_count
            total_snapshot_pass += snapshot_pass_count

            if best_score is not None:
                best_scores.append(float(best_score))

            row = {
                "case_id": case_id,
                "status": "OK",
                "architectures_scored": architecture_count,
                "pre_h10_hard_snapshot_pass": snapshot_pass_count,
                "pre_h10_hard_snapshot_fail": (
                    architecture_count - snapshot_pass_count
                ),
                "best_pre_h10_architecture_id": summary[
                    "best_pre_h10_architecture_id"
                ],
                "best_pre_h10_score": best_score,
                "score_min": summary["score_min"],
                "score_mean": summary["score_mean"],
                "score_max": summary["score_max"],
                "error": "",
            }

        except Exception as error:
            row = {
                "case_id": case_id,
                "status": "FAILED",
                "architectures_scored": 0,
                "pre_h10_hard_snapshot_pass": 0,
                "pre_h10_hard_snapshot_fail": 0,
                "best_pre_h10_architecture_id": None,
                "best_pre_h10_score": None,
                "score_min": None,
                "score_mean": None,
                "score_max": None,
                "error": f"{type(error).__name__}: {error}",
            }

        rows.append(row)

        if (
            index == 1
            or index % 100 == 0
            or index == len(selected)
        ):
            print(
                f"[{index}/{len(selected)}] "
                f"{case_id} : {row['status']}"
            )

    elapsed = time.perf_counter() - started
    failures = [
        row
        for row in rows
        if row["status"] != "OK"
    ]

    payload = {
        "schema_version": "1.0",
        "purpose": "full_architecture_scoring_validation",
        "scoring_semantics": {
            "soft_score_only": True,
            "hard_constraints_are_not_soft_penalties": True,
            "pre_score_gate_is_not_full_H10_validation": True,
            "beam_search_applied": False,
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
            "valid_cases": len(rows) - len(failures),
            "failure_cases": len(failures),
            "architectures_scored": total_scored,
            "pre_h10_hard_snapshot_pass": total_snapshot_pass,
            "pre_h10_hard_snapshot_fail": (
                total_scored - total_snapshot_pass
            ),
            "mean_architectures_per_case": (
                total_scored / len(rows)
                if rows
                else 0.0
            ),
            "mean_best_eligible_score": (
                fmean(best_scores)
                if best_scores
                else None
            ),
            "cases_without_snapshot_pass": sum(
                1
                for row in rows
                if (
                    row["status"] == "OK"
                    and row["pre_h10_hard_snapshot_pass"] == 0
                )
            ),
            "elapsed_seconds": round(elapsed, 4),
        },
        "cases": rows,
    }

    save_json(args.output, payload)

    print()
    print("FULL ARCHITECTURE SCORING H9")
    print("============================")
    print("Status                    :", payload["summary"]["status"])
    print("Cases                     :", payload["summary"]["cases"])
    print("Valid cases               :", payload["summary"]["valid_cases"])
    print("Failures                  :", payload["summary"]["failure_cases"])
    print("Architectures scored      :", payload["summary"]["architectures_scored"])
    print(
        "Hard snapshot pass        :",
        payload["summary"]["pre_h10_hard_snapshot_pass"],
    )
    print(
        "Hard snapshot fail        :",
        payload["summary"]["pre_h10_hard_snapshot_fail"],
    )
    print(
        "Cases without snapshot pass:",
        payload["summary"]["cases_without_snapshot_pass"],
    )
    print(
        "Mean / case               :",
        round(payload["summary"]["mean_architectures_per_case"], 4),
    )
    print(
        "Mean best pre-H10 score   :",
        (
            round(payload["summary"]["mean_best_eligible_score"], 6)
            if payload["summary"]["mean_best_eligible_score"] is not None
            else None
        ),
    )
    print("Elapsed                   :", payload["summary"]["elapsed_seconds"], "s")
    print("Output                    :", args.output)

    if failures:
        print()
        print("Premières erreurs :")
        for row in failures[:10]:
            print("-", row["case_id"], ":", row["error"])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
