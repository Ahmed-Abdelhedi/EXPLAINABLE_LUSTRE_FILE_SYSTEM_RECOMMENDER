from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# Gestion des imports
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


import mdt_candidate_generator as mdt_generator  # noqa: E402

from feature_builder import (  # noqa: E402
    DEFAULT_ARCHITECTURES_PATH,
    DEFAULT_CATALOG_PATH,
    build_mdt_feature_row,
    load_json,
)

from ranker_loader import (  # noqa: E402
    load_ranker_bundle,
    prepare_lightgbm_dataframe,
)


# ============================================================
# Exception
# ============================================================

class MDTInferenceError(RuntimeError):
    """Erreur pendant l'inférence du MDT Ranker."""


# ============================================================
# Génération des candidats MDT faisables
# ============================================================

def generate_all_feasible_mdt_candidates(
    architecture: dict[str, Any],
    catalog: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Applique les filtres déterministes à tous les drives MDT.

    Aucun Top-K teacher n'est appliqué ici. Tous les candidats faisables
    sont transmis au modèle LightGBM officiel.
    """

    requirement = architecture["MDT_requirement"]
    constraints = architecture["constraints"]
    preferences = architecture["preferences"]

    feasible_candidates: list[dict[str, Any]] = []

    for drive in catalog:
        if not bool(drive.get("mdt_eligible", False)):
            continue

        candidate, rejection_reasons = mdt_generator.evaluate_drive(
            drive,
            requirement,
            constraints,
            preferences,
        )

        if rejection_reasons:
            continue

        feasible_candidates.append(
            {
                "drive": drive,
                "candidate": candidate,
            }
        )

    if not feasible_candidates:
        raise MDTInferenceError(
            "Aucun candidat MDT ne respecte les contraintes déterministes."
        )

    return feasible_candidates


# ============================================================
# Préparation LightGBM
# ============================================================

def prepare_mdt_dataframe(
    feature_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> pd.DataFrame:
    """Prépare le DataFrame MDT selon le contrat LightGBM officiel."""

    try:
        return prepare_lightgbm_dataframe(
            feature_rows=feature_rows,
            metadata=metadata,
        )
    except Exception as error:
        raise MDTInferenceError(
            f"Impossible de préparer les features MDT : {error}"
        ) from error


def prepare_mdt_pool(
    feature_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> pd.DataFrame:
    """
    Alias de compatibilité avec l'ancien runtime CatBoost.

    Le retour est désormais un DataFrame pandas et non un catboost.Pool.
    """

    return prepare_mdt_dataframe(feature_rows, metadata)


# ============================================================
# Classement MDT complet
# ============================================================

def rank_all_mdt_candidates(
    architecture: dict[str, Any],
    catalog: list[dict[str, Any]],
) -> dict[str, Any]:
    """Classe tous les drives MDT faisables avec LightGBM."""

    model, metadata = load_ranker_bundle("mdt")

    feasible_candidates = generate_all_feasible_mdt_candidates(
        architecture=architecture,
        catalog=catalog,
    )

    feature_rows: list[dict[str, Any]] = []

    for item in feasible_candidates:
        feature_rows.append(
            build_mdt_feature_row(
                architecture=architecture,
                drive=item["drive"],
                candidate=item["candidate"],
            )
        )

    prediction_frame = prepare_mdt_dataframe(
        feature_rows=feature_rows,
        metadata=metadata,
    )

    try:
        raw_predictions: Any = model.predict(
            prediction_frame,
            num_iteration=model.num_trees(),
            validate_features=True,
        )

        predictions = np.asarray(
            raw_predictions,
            dtype=np.float64,
        ).reshape(-1)
    except Exception as error:
        raise MDTInferenceError(
            f"Échec de prédiction LightGBM MDT : {error}"
        ) from error

    if len(predictions) != len(feasible_candidates):
        raise MDTInferenceError(
            "Le nombre de prédictions ne correspond pas au nombre "
            "de candidats MDT faisables."
        )

    ranked_candidates: list[dict[str, Any]] = []

    for item, prediction in zip(feasible_candidates, predictions):
        drive = item["drive"]
        candidate = item["candidate"]
        ml_score = float(prediction)

        if not math.isfinite(ml_score):
            raise MDTInferenceError(
                f"Score ML invalide pour {drive['drive_id']} : {ml_score}"
            )

        ranked_candidates.append(
            {
                "drive_id": drive["drive_id"],
                "drive_name": drive["name"],
                "manufacturer": drive.get("manufacturer"),
                "series": drive.get("series"),
                "media_type": drive["media_type"],
                "protocol": drive["protocol"],
                "capacity_tib": drive["capacity_tib"],
                "random_read_iops_4k": drive["random_read_iops_4k"],
                "random_write_iops_4k": drive["random_write_iops_4k"],
                "endurance_dwpd": drive["endurance_dwpd_numeric"],
                "price_usd": drive["price_en_dollars"],
                "power_w": drive["power_consumption_en_w"],
                "ml_score": ml_score,
                "raw_minimum_drive_count": candidate[
                    "raw_minimum_drive_count"
                ],
                "raw_provided_capacity_tib": candidate[
                    "raw_provided_capacity_tib"
                ],
                "raw_provided_read_iops": candidate[
                    "raw_provided_read_iops"
                ],
                "raw_provided_write_iops": candidate[
                    "raw_provided_write_iops"
                ],
                "raw_drive_cost_usd": candidate["raw_drive_cost_usd"],
                "raw_drive_power_w": candidate["raw_drive_power_w"],
            }
        )

    ranked_candidates.sort(
        key=lambda item: (-item["ml_score"], item["drive_id"])
    )

    for rank, candidate in enumerate(ranked_candidates, start=1):
        candidate["ml_rank"] = rank

    return {
        "case_id": architecture["case_id"],
        "model_role": metadata["model_role"],
        "model_family": metadata["model_family"],
        "model_type": metadata["model_type"],
        "model_seed": metadata["selected_seed"],
        "feature_count": len(metadata["feature_columns"]),
        "feasible_candidate_count": len(ranked_candidates),
        "ranked_candidates": ranked_candidates,
    }


# ============================================================
# Affichage du Top-K
# ============================================================

def print_mdt_top_k(
    ranking_result: dict[str, Any],
    top_k: int = 10,
) -> None:
    """Affiche les meilleurs candidats MDT."""

    candidates = ranking_result["ranked_candidates"][:top_k]

    print("=" * 100)
    print("CLASSEMENT MDT RANKER OFFICIEL")
    print("Case ID             :", ranking_result["case_id"])
    print("Modèle              :", ranking_result["model_family"])
    print("Seed                :", ranking_result["model_seed"])
    print("Candidats faisables :", ranking_result["feasible_candidate_count"])
    print("Features            :", ranking_result["feature_count"])
    print("Top-K affiché       :", len(candidates))
    print("=" * 100)

    header = (
        f"{'Rang':<6}"
        f"{'Drive ID':<14}"
        f"{'Nom':<28}"
        f"{'Protocole':<12}"
        f"{'Cap. TiB':>10}"
        f"{'Drives':>9}"
        f"{'Score ML':>14}"
    )

    print(header)
    print("-" * 100)

    for candidate in candidates:
        print(
            f"{candidate['ml_rank']:<6}"
            f"{candidate['drive_id']:<14}"
            f"{candidate['drive_name'][:26]:<28}"
            f"{candidate['protocol']:<12}"
            f"{float(candidate['capacity_tib']):>10.2f}"
            f"{candidate['raw_minimum_drive_count']:>9}"
            f"{candidate['ml_score']:>14.8f}"
        )

    print("-" * 100)
    print("Classement de tous les candidats MDT : VALIDÉ")


# ============================================================
# Test local
# ============================================================

def main() -> None:
    """Classe les candidats MDT du premier cas architectural."""

    architectures = load_json(DEFAULT_ARCHITECTURES_PATH)
    catalog = load_json(DEFAULT_CATALOG_PATH)

    if not isinstance(architectures, list):
        raise TypeError("Le dataset architectural doit être une liste.")
    if not architectures:
        raise MDTInferenceError("Le dataset architectural est vide.")
    if not isinstance(catalog, list):
        raise TypeError("Le catalogue doit être une liste.")
    if not catalog:
        raise MDTInferenceError("Le catalogue est vide.")

    ranking_result = rank_all_mdt_candidates(
        architecture=architectures[0],
        catalog=catalog,
    )
    print_mdt_top_k(ranking_result=ranking_result, top_k=10)


if __name__ == "__main__":
    main()
