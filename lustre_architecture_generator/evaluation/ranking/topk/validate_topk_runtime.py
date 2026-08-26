from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean
from types import ModuleType
from typing import Any


# ============================================================
# Chemins du projet
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
RANKING_DIR = SRC_DIR / "ranking"

DEFAULT_ARCHITECTURES_PATH = (
    PROJECT_ROOT
    / "output"
    / "lustre_architecture_dataset.json"
)

DEFAULT_CATALOG_PATH = (
    PROJECT_ROOT
    / "data"
    / "catalogue_drives_ready_final.json"
)

DEFAULT_OUTPUT_JSON = (
    PROJECT_ROOT
    / "evaluation"
    / "ranking"
    / "topk"
    / "topk_runtime_validation.json"
)

DEFAULT_OUTPUT_CSV = (
    PROJECT_ROOT
    / "evaluation"
    / "ranking"
    / "topk"
    / "topk_runtime_cases.csv"
)


# ============================================================
# Exceptions
# ============================================================

class TopKValidationError(RuntimeError):
    """Erreur pendant la validation globale du Top-K."""


# ============================================================
# Chargement explicite des modules runtime
# ============================================================

def _load_module(
    module_name: str,
    module_path: Path,
) -> ModuleType:
    """
    Charge un module Python depuis son chemin exact.

    Le chargement explicite évite les faux positifs Pylance
    tout en exécutant exactement les fichiers runtime locaux.
    """

    if not module_path.exists():
        raise FileNotFoundError(
            f"Module introuvable : {module_path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Impossible de créer la spec pour {module_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def load_runtime_modules() -> tuple[ModuleType, ModuleType, ModuleType]:
    """
    Charge les trois modules utilisés pendant la validation :
    MDT inference, OST inference et diversification OST.
    """

    for directory in (RANKING_DIR, SRC_DIR):
        directory_text = str(directory)
        if directory_text not in sys.path:
            sys.path.insert(0, directory_text)

    mdt_module = _load_module(
        "_topk_validation_mdt_ranker_inference",
        RANKING_DIR / "mdt_ranker_inference.py",
    )

    ost_module = _load_module(
        "_topk_validation_ost_ranker_inference",
        RANKING_DIR / "ost_ranker_inference.py",
    )

    diversified_module = _load_module(
        "_topk_validation_diversified_topk",
        RANKING_DIR / "diversified_topk.py",
    )

    return mdt_module, ost_module, diversified_module


# ============================================================
# I/O
# ============================================================

def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier JSON introuvable : {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_json(
    path: Path,
    payload: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def save_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )
        return

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Helpers de validation
# ============================================================

def candidate_ids(
    candidates: list[dict[str, Any]],
) -> list[str]:
    return [
        str(candidate["drive_id"])
        for candidate in candidates
    ]


def unique_count(
    values: list[str],
) -> int:
    return len(set(values))


def finite_ml_scores(
    candidates: list[dict[str, Any]],
) -> bool:
    for candidate in candidates:
        try:
            value = float(candidate["ml_score"])
        except (TypeError, ValueError, KeyError):
            return False

        if not math.isfinite(value):
            return False

    return True


def candidate_signature(
    candidates: list[dict[str, Any]],
) -> list[tuple[str, int, tuple[str, ...]]]:
    signature: list[
        tuple[str, int, tuple[str, ...]]
    ] = []

    for candidate in candidates:
        reasons = tuple(
            str(reason)
            for reason in candidate.get(
                "selection_reasons",
                [],
            )
        )

        signature.append(
            (
                str(candidate["drive_id"]),
                int(candidate["ml_rank"]),
                reasons,
            )
        )

    return signature


def is_specialized_candidate(
    candidate: dict[str, Any],
) -> bool:
    neutral_reasons = {
        "global_ml_top",
        "ml_fill",
    }

    return any(
        str(reason) not in neutral_reasons
        for reason in candidate.get(
            "selection_reasons",
            [],
        )
    )


def media_present(
    candidates: list[dict[str, Any]],
    media_type: str,
) -> bool:
    expected = media_type.strip().upper()

    return any(
        str(
            candidate.get(
                "media_type",
                "",
            )
        ).strip().upper() == expected
        for candidate in candidates
    )


def validate_full_ranking(
    *,
    label: str,
    ranked_candidates: list[dict[str, Any]],
    reported_count: int,
) -> list[str]:
    failures: list[str] = []

    if len(ranked_candidates) != reported_count:
        failures.append(
            f"{label}_reported_count_mismatch"
        )

    ids = candidate_ids(
        ranked_candidates
    )

    if unique_count(ids) != len(ids):
        failures.append(
            f"{label}_duplicate_drive_ids"
        )

    expected_ranks = list(
        range(
            1,
            len(ranked_candidates) + 1,
        )
    )

    actual_ranks = [
        int(candidate["ml_rank"])
        for candidate in ranked_candidates
    ]

    if actual_ranks != expected_ranks:
        failures.append(
            f"{label}_ml_ranks_not_contiguous"
        )

    if not finite_ml_scores(
        ranked_candidates
    ):
        failures.append(
            f"{label}_non_finite_ml_score"
        )

    return failures


# ============================================================
# Validation d'un cas
# ============================================================

def validate_case(
    *,
    architecture: dict[str, Any],
    catalog: list[dict[str, Any]],
    mdt_module: ModuleType,
    ost_module: ModuleType,
    diversified_module: ModuleType,
    top_k: int,
    global_top_count: int,
    diversification_multiplier: int,
    minimum_diversification_pool_size: int,
) -> dict[str, Any]:
    case_id = str(
        architecture.get(
            "case_id",
            "UNKNOWN",
        )
    )

    failures: list[str] = []

    mdt_result = (
        mdt_module.rank_all_mdt_candidates(
            architecture=architecture,
            catalog=catalog,
        )
    )

    ost_result = (
        ost_module.rank_all_ost_candidates(
            architecture=architecture,
            catalog=catalog,
        )
    )

    mdt_ranked = mdt_result[
        "ranked_candidates"
    ]

    ost_ranked = ost_result[
        "ranked_candidates"
    ]

    failures.extend(
        validate_full_ranking(
            label="mdt",
            ranked_candidates=mdt_ranked,
            reported_count=int(
                mdt_result[
                    "feasible_candidate_count"
                ]
            ),
        )
    )

    failures.extend(
        validate_full_ranking(
            label="ost",
            ranked_candidates=ost_ranked,
            reported_count=int(
                ost_result[
                    "feasible_candidate_count"
                ]
            ),
        )
    )

    # --------------------------------------------------------
    # MDT Top-K pur ML
    # --------------------------------------------------------

    expected_mdt_top_count = min(
        top_k,
        len(mdt_ranked),
    )

    mdt_top = mdt_ranked[
        :expected_mdt_top_count
    ]

    mdt_top_ids = candidate_ids(
        mdt_top
    )

    if len(mdt_top) != expected_mdt_top_count:
        failures.append(
            "mdt_topk_wrong_size"
        )

    if unique_count(mdt_top_ids) != len(
        mdt_top_ids
    ):
        failures.append(
            "mdt_topk_duplicates"
        )

    # --------------------------------------------------------
    # OST Top-K diversifié
    # --------------------------------------------------------

    diversified_a = (
        diversified_module
        .select_diversified_ost_top_k(
            ranking_result=ost_result,
            top_k=top_k,
            global_top_count=global_top_count,
            diversification_multiplier=(
                diversification_multiplier
            ),
            minimum_diversification_pool_size=(
                minimum_diversification_pool_size
            ),
        )
    )

    diversified_b = (
        diversified_module
        .select_diversified_ost_top_k(
            ranking_result=ost_result,
            top_k=top_k,
            global_top_count=global_top_count,
            diversification_multiplier=(
                diversification_multiplier
            ),
            minimum_diversification_pool_size=(
                minimum_diversification_pool_size
            ),
        )
    )

    selected = diversified_a[
        "diversified_candidates"
    ]

    selected_again = diversified_b[
        "diversified_candidates"
    ]

    selected_ids = candidate_ids(
        selected
    )

    ost_source_ids = candidate_ids(
        ost_ranked
    )

    expected_selected_count = min(
        top_k,
        len(ost_ranked),
    )

    if len(selected) != expected_selected_count:
        failures.append(
            "ost_diversified_wrong_size"
        )

    if int(
        diversified_a["selected_count"]
    ) != len(selected):
        failures.append(
            "ost_diversified_reported_count_mismatch"
        )

    if unique_count(selected_ids) != len(
        selected_ids
    ):
        failures.append(
            "ost_diversified_duplicates"
        )

    source_id_set = set(
        ost_source_ids
    )

    if any(
        drive_id not in source_id_set
        for drive_id in selected_ids
    ):
        failures.append(
            "ost_diversified_candidate_not_in_source"
        )

    expected_diversified_ranks = list(
        range(
            1,
            len(selected) + 1,
        )
    )

    actual_diversified_ranks = [
        int(
            candidate["diversified_rank"]
        )
        for candidate in selected
    ]

    if (
        actual_diversified_ranks
        != expected_diversified_ranks
    ):
        failures.append(
            "ost_diversified_ranks_not_contiguous"
        )

    selected_ml_ranks = [
        int(candidate["ml_rank"])
        for candidate in selected
    ]

    if (
        selected_ml_ranks
        != sorted(selected_ml_ranks)
    ):
        failures.append(
            "ost_diversified_not_sorted_by_ml_rank"
        )

    if not finite_ml_scores(
        selected
    ):
        failures.append(
            "ost_diversified_non_finite_ml_score"
        )

    # --------------------------------------------------------
    # Conservation du Top ML global
    # --------------------------------------------------------

    required_global_count = min(
        global_top_count,
        expected_selected_count,
        len(ost_ranked),
    )

    required_global_ids = set(
        candidate_ids(
            ost_ranked[
                :required_global_count
            ]
        )
    )

    if not required_global_ids.issubset(
        set(selected_ids)
    ):
        failures.append(
            "ost_global_ml_top_not_preserved"
        )

    # --------------------------------------------------------
    # Garde-fou du pool de diversification
    # --------------------------------------------------------

    pool_size = int(
        diversified_a[
            "diversification_pool_size"
        ]
    )

    specialized_candidates = [
        candidate
        for candidate in selected
        if is_specialized_candidate(
            candidate
        )
    ]

    specialized_out_of_pool = [
        candidate
        for candidate in specialized_candidates
        if int(candidate["ml_rank"]) > pool_size
    ]

    if specialized_out_of_pool:
        failures.append(
            "ost_specialized_candidate_out_of_pool"
        )

    maximum_specialized_ml_rank = (
        max(
            (
                int(candidate["ml_rank"])
                for candidate
                in specialized_candidates
            ),
            default=0,
        )
    )

    # --------------------------------------------------------
    # Déterminisme du Top-K
    # --------------------------------------------------------

    deterministic = (
        candidate_signature(selected)
        == candidate_signature(
            selected_again
        )
    )

    if not deterministic:
        failures.append(
            "ost_diversified_nondeterministic"
        )

    # --------------------------------------------------------
    # Diversité média
    # --------------------------------------------------------

    diversification_pool = ost_ranked[
        :pool_size
    ]

    pool_has_hdd = media_present(
        diversification_pool,
        "HDD",
    )

    pool_has_ssd = media_present(
        diversification_pool,
        "SSD",
    )

    selected_has_hdd = media_present(
        selected,
        "HDD",
    )

    selected_has_ssd = media_present(
        selected,
        "SSD",
    )

    if (
        top_k >= 2
        and pool_has_hdd
        and not selected_has_hdd
    ):
        failures.append(
            "ost_hdd_not_represented_despite_pool"
        )

    if (
        top_k >= 2
        and pool_has_ssd
        and not selected_has_ssd
    ):
        failures.append(
            "ost_ssd_not_represented_despite_pool"
        )

    # --------------------------------------------------------
    # Overlap avec le Top-K ML pur
    # --------------------------------------------------------

    pure_ost_top = ost_ranked[
        :expected_selected_count
    ]

    pure_ost_ids = set(
        candidate_ids(
            pure_ost_top
        )
    )

    selected_id_set = set(
        selected_ids
    )

    overlap_count = len(
        pure_ost_ids
        & selected_id_set
    )

    union_count = len(
        pure_ost_ids
        | selected_id_set
    )

    overlap_ratio = (
        overlap_count
        / expected_selected_count
        if expected_selected_count > 0
        else 1.0
    )

    jaccard = (
        overlap_count
        / union_count
        if union_count > 0
        else 1.0
    )

    return {
        "case_id": case_id,
        "mdt_feasible_count": len(
            mdt_ranked
        ),
        "ost_feasible_count": len(
            ost_ranked
        ),
        "mdt_topk_count": len(
            mdt_top
        ),
        "ost_topk_count": len(
            selected
        ),
        "ost_pure_ml_overlap_count": (
            overlap_count
        ),
        "ost_pure_ml_overlap_ratio": (
            round(
                overlap_ratio,
                8,
            )
        ),
        "ost_pure_ml_jaccard": round(
            jaccard,
            8,
        ),
        "diversification_pool_size": (
            pool_size
        ),
        "maximum_specialized_ml_rank": (
            maximum_specialized_ml_rank
        ),
        "pool_has_hdd": pool_has_hdd,
        "selected_has_hdd": (
            selected_has_hdd
        ),
        "pool_has_ssd": pool_has_ssd,
        "selected_has_ssd": (
            selected_has_ssd
        ),
        "deterministic": deterministic,
        "failure_count": len(
            failures
        ),
        "failures": failures,
    }


# ============================================================
# Agrégation
# ============================================================

def build_summary(
    rows: list[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    total_cases = len(rows)

    failure_cases = [
        row
        for row in rows
        if int(row["failure_count"]) > 0
    ]

    overlap_ratios = [
        float(
            row[
                "ost_pure_ml_overlap_ratio"
            ]
        )
        for row in rows
    ]

    jaccards = [
        float(
            row["ost_pure_ml_jaccard"]
        )
        for row in rows
    ]

    max_specialized_ranks = [
        int(
            row[
                "maximum_specialized_ml_rank"
            ]
        )
        for row in rows
    ]

    hdd_opportunities = sum(
        bool(row["pool_has_hdd"])
        for row in rows
    )

    hdd_preserved = sum(
        bool(row["selected_has_hdd"])
        for row in rows
        if bool(row["pool_has_hdd"])
    )

    ssd_opportunities = sum(
        bool(row["pool_has_ssd"])
        for row in rows
    )

    ssd_preserved = sum(
        bool(row["selected_has_ssd"])
        for row in rows
        if bool(row["pool_has_ssd"])
    )

    determinism_failures = sum(
        not bool(row["deterministic"])
        for row in rows
    )

    return {
        "status": (
            "VALIDATED"
            if not failure_cases
            else "FAILED"
        ),
        "cases": total_cases,
        "valid_cases": (
            total_cases
            - len(failure_cases)
        ),
        "failure_cases": len(
            failure_cases
        ),
        "total_failures": sum(
            int(row["failure_count"])
            for row in rows
        ),
        "determinism_failures": (
            determinism_failures
        ),
        "mean_ost_pure_ml_overlap_ratio": (
            round(
                mean(overlap_ratios),
                8,
            )
            if overlap_ratios
            else 0.0
        ),
        "min_ost_pure_ml_overlap_ratio": (
            round(
                min(overlap_ratios),
                8,
            )
            if overlap_ratios
            else 0.0
        ),
        "mean_ost_pure_ml_jaccard": (
            round(
                mean(jaccards),
                8,
            )
            if jaccards
            else 0.0
        ),
        "maximum_specialized_ml_rank": (
            max(
                max_specialized_ranks,
                default=0,
            )
        ),
        "hdd_pool_opportunity_cases": (
            hdd_opportunities
        ),
        "hdd_represented_cases": (
            hdd_preserved
        ),
        "ssd_pool_opportunity_cases": (
            ssd_opportunities
        ),
        "ssd_represented_cases": (
            ssd_preserved
        ),
        "elapsed_seconds": round(
            elapsed_seconds,
            4,
        ),
    }


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Valide le contrat Top-K MDT et le Top-K OST "
            "diversifié sur tout le dataset architectural."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Nombre de cas à valider. "
            "0 = tous les cas."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--global-top-count",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--diversification-multiplier",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--minimum-diversification-pool-size",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--architectures",
        type=Path,
        default=DEFAULT_ARCHITECTURES_PATH,
    )

    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG_PATH,
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


# ============================================================
# Main
# ============================================================

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

    architectures = load_json(
        args.architectures
    )

    catalog = load_json(
        args.catalog
    )

    if not isinstance(
        architectures,
        list,
    ):
        raise TypeError(
            "Le dataset architectural doit être une liste."
        )

    if not architectures:
        raise TopKValidationError(
            "Le dataset architectural est vide."
        )

    if not isinstance(
        catalog,
        list,
    ):
        raise TypeError(
            "Le catalogue doit être une liste."
        )

    if not catalog:
        raise TopKValidationError(
            "Le catalogue de drives est vide."
        )

    selected_architectures = (
        architectures
        if args.limit == 0
        else architectures[
            :args.limit
        ]
    )

    (
        mdt_module,
        ost_module,
        diversified_module,
    ) = load_runtime_modules()

    rows: list[
        dict[str, Any]
    ] = []

    started = time.perf_counter()

    total = len(
        selected_architectures
    )

    for index, architecture in enumerate(
        selected_architectures,
        start=1,
    ):
        case_id = str(
            architecture.get(
                "case_id",
                f"CASE_{index:06d}",
            )
        )

        try:
            row = validate_case(
                architecture=architecture,
                catalog=catalog,
                mdt_module=mdt_module,
                ost_module=ost_module,
                diversified_module=(
                    diversified_module
                ),
                top_k=args.top_k,
                global_top_count=(
                    args.global_top_count
                ),
                diversification_multiplier=(
                    args
                    .diversification_multiplier
                ),
                minimum_diversification_pool_size=(
                    args
                    .minimum_diversification_pool_size
                ),
            )
        except Exception as error:
            row = {
                "case_id": case_id,
                "mdt_feasible_count": 0,
                "ost_feasible_count": 0,
                "mdt_topk_count": 0,
                "ost_topk_count": 0,
                "ost_pure_ml_overlap_count": 0,
                "ost_pure_ml_overlap_ratio": 0.0,
                "ost_pure_ml_jaccard": 0.0,
                "diversification_pool_size": 0,
                "maximum_specialized_ml_rank": 0,
                "pool_has_hdd": False,
                "selected_has_hdd": False,
                "pool_has_ssd": False,
                "selected_has_ssd": False,
                "deterministic": False,
                "failure_count": 1,
                "failures": [
                    (
                        "runtime_exception: "
                        f"{type(error).__name__}: "
                        f"{error}"
                    )
                ],
            }

        rows.append(
            row
        )

        if (
            index == 1
            or index % 100 == 0
            or index == total
        ):
            status = (
                "OK"
                if int(
                    row[
                        "failure_count"
                    ]
                ) == 0
                else "FAILED"
            )

            print(
                f"[{index}/{total}] "
                f"{case_id} : {status}"
            )

    elapsed = (
        time.perf_counter()
        - started
    )

    summary = build_summary(
        rows,
        elapsed,
    )

    payload = {
        "schema_version": "1.0",
        "purpose": (
            "ranking_topk_runtime_validation"
        ),
        "configuration": {
            "top_k": args.top_k,
            "global_top_count": (
                args.global_top_count
            ),
            "diversification_multiplier": (
                args
                .diversification_multiplier
            ),
            "minimum_diversification_pool_size": (
                args
                .minimum_diversification_pool_size
            ),
            "limit": args.limit,
            "architectures_path": str(
                args.architectures
            ),
            "catalog_path": str(
                args.catalog
            ),
        },
        "summary": summary,
        "cases": rows,
    }

    save_json(
        args.output_json,
        payload,
    )

    csv_rows: list[
        dict[str, Any]
    ] = []

    for row in rows:
        csv_row = dict(
            row
        )

        csv_row["failures"] = (
            " | ".join(
                str(value)
                for value in row[
                    "failures"
                ]
            )
        )

        csv_rows.append(
            csv_row
        )

    save_csv(
        args.output_csv,
        csv_rows,
    )

    print()
    print(
        "TOP-K RUNTIME VALIDATION"
    )
    print(
        "========================"
    )
    print(
        "Status                         :",
        summary["status"],
    )
    print(
        "Cases                          :",
        summary["cases"],
    )
    print(
        "Valid cases                    :",
        summary["valid_cases"],
    )
    print(
        "Failure cases                  :",
        summary["failure_cases"],
    )
    print(
        "Determinism failures           :",
        summary[
            "determinism_failures"
        ],
    )
    print(
        "Mean OST pure-ML overlap       :",
        summary[
            "mean_ost_pure_ml_overlap_ratio"
        ],
    )
    print(
        "Min OST pure-ML overlap        :",
        summary[
            "min_ost_pure_ml_overlap_ratio"
        ],
    )
    print(
        "Mean OST pure-ML Jaccard       :",
        summary[
            "mean_ost_pure_ml_jaccard"
        ],
    )
    print(
        "Maximum specialized ML rank    :",
        summary[
            "maximum_specialized_ml_rank"
        ],
    )
    print(
        "HDD representation             :",
        (
            f"{summary['hdd_represented_cases']}/"
            f"{summary['hdd_pool_opportunity_cases']}"
        ),
    )
    print(
        "SSD representation             :",
        (
            f"{summary['ssd_represented_cases']}/"
            f"{summary['ssd_pool_opportunity_cases']}"
        ),
    )
    print(
        "Elapsed                         :",
        f"{summary['elapsed_seconds']} s",
    )
    print(
        "JSON                            :",
        args.output_json,
    )
    print(
        "CSV                             :",
        args.output_csv,
    )

    if summary["status"] != "VALIDATED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
