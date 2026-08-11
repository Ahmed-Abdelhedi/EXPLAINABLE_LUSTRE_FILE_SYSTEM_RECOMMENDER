from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from catboost import CatBoostRanker


# Dossier lustre_architecture_generator/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RANKERS_ROOT = PROJECT_ROOT / "artifacts" / "rankers"


def normalize_ranker_type(ranker_type: str) -> str:
    """Valide et normalise le type du ranker."""

    normalized_type = ranker_type.strip().lower()

    if normalized_type not in {"mdt", "ost"}:
        raise ValueError(
            f"Type de ranker invalide : {ranker_type!r}. "
            "Valeurs autorisées : 'mdt' ou 'ost'."
        )

    return normalized_type


def get_ranker_paths(
    ranker_type: str,
) -> tuple[Path, Path]:
    """
    Retourne les chemins du modèle et des metadata.
    """

    normalized_type = normalize_ranker_type(
        ranker_type
    )

    ranker_directory = (
        RANKERS_ROOT / normalized_type
    )

    model_path = (
        ranker_directory
        / f"{normalized_type}_ranker.cbm"
    )

    metadata_path = (
        ranker_directory
        / f"{normalized_type}_ranker_metadata.json"
    )

    return model_path, metadata_path


def load_ranker_metadata(
    ranker_type: str,
) -> dict[str, Any]:
    """Charge et valide les metadata d'un ranker."""

    _, metadata_path = get_ranker_paths(
        ranker_type
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata introuvable : {metadata_path}"
        )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    required_fields = {
        "model_role",
        "model_type",
        "feature_columns",
        "categorical_features",
        "numeric_features",
    }

    missing_fields = (
        required_fields - metadata.keys()
    )

    if missing_fields:
        raise ValueError(
            f"Metadata incomplète : {metadata_path}. "
            f"Champs manquants : "
            f"{sorted(missing_fields)}"
        )

    feature_columns = metadata["feature_columns"]
    categorical_features = metadata[
        "categorical_features"
    ]
    numeric_features = metadata[
        "numeric_features"
    ]

    if not isinstance(feature_columns, list):
        raise TypeError(
            "'feature_columns' doit être une liste."
        )

    if len(feature_columns) != (
        len(categorical_features)
        + len(numeric_features)
    ):
        raise ValueError(
            "Le nombre de features catégorielles et "
            "numériques ne correspond pas au nombre "
            "total de features."
        )

    return metadata


def load_ranker_model(
    ranker_type: str,
) -> CatBoostRanker:
    """Charge le modèle CatBoost au format CBM."""

    model_path, _ = get_ranker_paths(
        ranker_type
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}"
        )

    model = CatBoostRanker()

    try:
        model.load_model(str(model_path))
    except Exception as error:
        raise RuntimeError(
            f"Impossible de charger le modèle "
            f"{model_path}: {error}"
        ) from error

    return model


def validate_model_and_metadata(
    ranker_type: str,
    model: CatBoostRanker,
    metadata: dict[str, Any],
) -> None:
    """
    Vérifie que le modèle CBM correspond à ses metadata.
    """

    normalized_type = normalize_ranker_type(
        ranker_type
    )

    expected_role = (
        f"{normalized_type.upper()}"
        "_pre_raid_drive_ranker"
    )

    actual_role = metadata["model_role"]

    if actual_role != expected_role:
        raise ValueError(
            f"Rôle incorrect pour {normalized_type.upper()} : "
            f"{actual_role!r} au lieu de "
            f"{expected_role!r}."
        )

    if metadata["model_type"] != "CatBoostRanker":
        raise ValueError(
            f"Type de modèle incorrect : "
            f"{metadata['model_type']!r}"
        )

    metadata_features = metadata[
        "feature_columns"
    ]

    model_features = list(
        model.feature_names_
    )

    if not model_features:
        raise ValueError(
            f"Le modèle {normalized_type.upper()} "
            "ne contient pas de noms de features."
        )

    if len(model_features) != len(
        metadata_features
    ):
        raise ValueError(
            f"Nombre de features différent pour "
            f"{normalized_type.upper()} : "
            f"modèle={len(model_features)}, "
            f"metadata={len(metadata_features)}."
        )

    if model_features != metadata_features:
        differences = []

        for index, (
            model_feature,
            metadata_feature,
        ) in enumerate(
            zip(model_features, metadata_features)
        ):
            if model_feature != metadata_feature:
                differences.append(
                    {
                        "position": index,
                        "model": model_feature,
                        "metadata": metadata_feature,
                    }
                )

        raise ValueError(
            f"L'ordre des features du modèle "
            f"{normalized_type.upper()} ne correspond "
            f"pas aux metadata. "
            f"Premières différences : "
            f"{differences[:5]}"
        )


def load_ranker_bundle(
    ranker_type: str,
) -> tuple[CatBoostRanker, dict[str, Any]]:
    """
    Charge le modèle et ses metadata, puis vérifie
    leur cohérence.
    """

    metadata = load_ranker_metadata(
        ranker_type
    )

    model = load_ranker_model(
        ranker_type
    )

    validate_model_and_metadata(
        ranker_type=ranker_type,
        model=model,
        metadata=metadata,
    )

    return model, metadata


def print_ranker_summary(
    ranker_type: str,
    model: CatBoostRanker,
    metadata: dict[str, Any],
) -> None:
    """Affiche un résumé du modèle chargé."""

    print("=" * 60)
    print(f"RANKER : {ranker_type.upper()}")
    print(f"Rôle : {metadata['model_role']}")
    print(f"Type : {metadata['model_type']}")
    print(
        "Nombre de features :",
        len(metadata["feature_columns"]),
    )
    print(
        "Nombre d'arbres :",
        model.tree_count_,
    )
    print(
        "Modèle chargé :",
        model.is_fitted(),
    )
    print(
        "Correspondance modèle/metadata : VALIDÉE"
    )


if __name__ == "__main__":
    for current_ranker in ("mdt", "ost"):
        current_model, current_metadata = (
            load_ranker_bundle(
                current_ranker
            )
        )

        print_ranker_summary(
            ranker_type=current_ranker,
            model=current_model,
            metadata=current_metadata,
        )

    print(
        "\nChargement des deux modèles CatBoost : VALIDÉ"
    )