from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
RANKING_DIR = PROJECT_ROOT / "src" / "ranking"

if str(RANKING_DIR) not in sys.path:
    sys.path.insert(0, str(RANKING_DIR))


from feature_builder import (  # noqa: E402
    DEFAULT_ARCHITECTURES_PATH,
    DEFAULT_CATALOG_PATH,
    load_json,
)
from mdt_ranker_inference import rank_all_mdt_candidates  # noqa: E402
from ost_ranker_inference import rank_all_ost_candidates  # noqa: E402
from ranker_loader import load_ranker_bundle  # noqa: E402


class RuntimeValidationError(RuntimeError):
    """Erreur pendant la validation end-to-end du runtime ranking."""


def validate_ranking_result(
    result: dict[str, Any],
    role: str,
) -> None:
    candidates = result["ranked_candidates"]

    if not candidates:
        raise RuntimeValidationError(
            f"{result['case_id']} {role}: classement vide."
        )

    if result["feasible_candidate_count"] != len(candidates):
        raise RuntimeValidationError(
            f"{result['case_id']} {role}: candidate_count incohérent."
        )

    drive_ids = [candidate["drive_id"] for candidate in candidates]
    if len(drive_ids) != len(set(drive_ids)):
        raise RuntimeValidationError(
            f"{result['case_id']} {role}: drive_id dupliqué."
        )

    ranks = [candidate["ml_rank"] for candidate in candidates]
    if ranks != list(range(1, len(candidates) + 1)):
        raise RuntimeValidationError(
            f"{result['case_id']} {role}: rangs non contigus."
        )

    scores = [float(candidate["ml_score"]) for candidate in candidates]
    if not all(math.isfinite(score) for score in scores):
        raise RuntimeValidationError(
            f"{result['case_id']} {role}: score non fini."
        )

    expected_order = sorted(
        candidates,
        key=lambda item: (-float(item["ml_score"]), item["drive_id"]),
    )
    expected_ids = [item["drive_id"] for item in expected_order]
    if drive_ids != expected_ids:
        raise RuntimeValidationError(
            f"{result['case_id']} {role}: tri non déterministe."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Valide l'intégration LightGBM officielle sur les cas architecturaux."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = tous les cas; sinon limite au N premiers cas.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CURRENT_DIR / "lightgbm_runtime_validation.json",
    )
    args = parser.parse_args()

    architectures = load_json(DEFAULT_ARCHITECTURES_PATH)
    catalog = load_json(DEFAULT_CATALOG_PATH)

    if not isinstance(architectures, list) or not architectures:
        raise RuntimeValidationError("Dataset architectural invalide ou vide.")
    if not isinstance(catalog, list) or not catalog:
        raise RuntimeValidationError("Catalogue de drives invalide ou vide.")

    if args.limit < 0:
        raise ValueError("--limit doit être >= 0.")

    selected_architectures = (
        architectures if args.limit == 0 else architectures[: args.limit]
    )

    # Charge et valide les deux modèles une fois avant la campagne.
    mdt_model, mdt_metadata = load_ranker_bundle("mdt")
    ost_model, ost_metadata = load_ranker_bundle("ost")

    started = time.perf_counter()
    mdt_candidate_counts: list[int] = []
    ost_candidate_counts: list[int] = []
    mdt_top1 = Counter()
    ost_top1 = Counter()

    for index, architecture in enumerate(selected_architectures, start=1):
        mdt_result = rank_all_mdt_candidates(architecture, catalog)
        ost_result = rank_all_ost_candidates(architecture, catalog)

        validate_ranking_result(mdt_result, "MDT")
        validate_ranking_result(ost_result, "OST")

        mdt_candidate_counts.append(mdt_result["feasible_candidate_count"])
        ost_candidate_counts.append(ost_result["feasible_candidate_count"])
        mdt_top1[mdt_result["ranked_candidates"][0]["drive_id"]] += 1
        ost_top1[ost_result["ranked_candidates"][0]["drive_id"]] += 1

        if index == 1 or index % 100 == 0 or index == len(selected_architectures):
            print(
                f"[{index}/{len(selected_architectures)}] "
                f"{architecture['case_id']} : MDT/OST OK"
            )

    elapsed = time.perf_counter() - started

    result = {
        "schema_version": "1.0",
        "status": "VALIDATED",
        "case_count": len(selected_architectures),
        "elapsed_seconds": elapsed,
        "mdt": {
            "model_family": mdt_metadata["model_family"],
            "seed": mdt_metadata["selected_seed"],
            "candidate_count_min": min(mdt_candidate_counts),
            "candidate_count_max": max(mdt_candidate_counts),
            "top1_drive_distribution": dict(mdt_top1),
        },
        "ost": {
            "model_family": ost_metadata["model_family"],
            "seed": ost_metadata["selected_seed"],
            "candidate_count_min": min(ost_candidate_counts),
            "candidate_count_max": max(ost_candidate_counts),
            "top1_drive_distribution": dict(ost_top1),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nLIGHTGBM RUNTIME INTEGRATION : VALIDATED")
    print("Cases   :", len(selected_architectures))
    print("Elapsed :", round(elapsed, 2), "s")
    print("Output  :", args.output)


if __name__ == "__main__":
    main()
