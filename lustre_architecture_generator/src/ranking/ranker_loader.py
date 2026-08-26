from __future__ import annotations

import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import lightgbm as lgb
import pandas as pd


# ============================================================
# Chemins
# ============================================================

# lustre_architecture_generator/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Les anciens artefacts CatBoost peuvent rester dans artifacts/rankers/mdt
# et artifacts/rankers/ost pour traçabilité. Le runtime officiel charge
# exclusivement les modèles placés sous artifacts/rankers/official/.
RANKERS_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "rankers"
    / "official"
)


# ============================================================
# Exceptions
# ============================================================

class RankerLoaderError(RuntimeError):
    """Erreur de chargement ou de validation d'un ranker officiel."""


# ============================================================
# Utilitaires
# ============================================================

def normalize_ranker_type(ranker_type: str) -> str:
    """Valide et normalise le rôle du ranker."""

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
    """Retourne les chemins du modèle LightGBM officiel et de ses metadata."""

    normalized_type = normalize_ranker_type(ranker_type)
    ranker_directory = RANKERS_ROOT / normalized_type

    model_path = ranker_directory / f"{normalized_type}_ranker.txt"
    metadata_path = (
        ranker_directory
        / f"{normalized_type}_ranker_metadata.json"
    )

    return model_path, metadata_path


def file_sha256(path: Path) -> str:
    """Calcule le SHA256 d'un fichier sans le charger entièrement en mémoire."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _normalize_category_value(value: Any) -> str:
    """Normalise une valeur catégorielle avant application du mapping appris."""

    if value is None:
        return "NONE"

    try:
        if bool(pd.isna(value)):
            return "NONE"
    except (TypeError, ValueError):
        pass

    text = str(value)
    if text == "":
        return "NONE"

    return text


# ============================================================
# Metadata
# ============================================================

def load_ranker_metadata(
    ranker_type: str,
) -> dict[str, Any]:
    """Charge et valide les metadata du ranker LightGBM officiel."""

    normalized_type = normalize_ranker_type(ranker_type)
    _, metadata_path = get_ranker_paths(normalized_type)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata officielle introuvable : {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    required_fields = {
        "schema_version",
        "status",
        "model_role",
        "model_family",
        "model_type",
        "model_file",
        "model_sha256",
        "selected_seed",
        "feature_columns",
        "categorical_features",
        "numeric_features",
        "category_mappings",
        "dataset_sha256_uncompressed",
        "runtime_contract",
    }

    missing_fields = required_fields - metadata.keys()
    if missing_fields:
        raise ValueError(
            f"Metadata incomplète : {metadata_path}. "
            f"Champs manquants : {sorted(missing_fields)}"
        )

    if metadata["status"] != "OFFICIAL":
        raise ValueError(
            f"Le ranker {normalized_type.upper()} n'est pas marqué OFFICIAL."
        )

    if metadata["model_family"] != "LightGBM":
        raise ValueError(
            f"Famille de modèle incorrecte : {metadata['model_family']!r}"
        )

    if metadata["model_type"] != "LGBMRanker":
        raise ValueError(
            f"Type de modèle incorrect : {metadata['model_type']!r}"
        )

    feature_columns = metadata["feature_columns"]
    categorical_features = metadata["categorical_features"]
    numeric_features = metadata["numeric_features"]
    category_mappings = metadata["category_mappings"]

    for field_name, value in (
        ("feature_columns", feature_columns),
        ("categorical_features", categorical_features),
        ("numeric_features", numeric_features),
    ):
        if not isinstance(value, list):
            raise TypeError(f"'{field_name}' doit être une liste.")

    if feature_columns != categorical_features + numeric_features:
        raise ValueError(
            "L'ordre feature_columns doit être exactement "
            "categorical_features + numeric_features."
        )

    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("Des features dupliquées sont présentes dans les metadata.")

    if not isinstance(category_mappings, dict):
        raise TypeError("'category_mappings' doit être un dictionnaire.")

    if set(category_mappings) != set(categorical_features):
        raise ValueError(
            "Les clés de category_mappings ne correspondent pas exactement "
            "aux features catégorielles."
        )

    for column in categorical_features:
        categories = category_mappings[column]
        if not isinstance(categories, list) or not categories:
            raise ValueError(
                f"Mapping catégoriel invalide pour {column!r}."
            )
        if "NONE" not in categories:
            raise ValueError(
                f"Le mapping {column!r} ne contient pas 'NONE'."
            )
        if "__UNKNOWN__" not in categories:
            raise ValueError(
                f"Le mapping {column!r} ne contient pas '__UNKNOWN__'."
            )

    expected_role = f"{normalized_type.upper()}_pre_raid_drive_ranker"
    if metadata["model_role"] != expected_role:
        raise ValueError(
            f"Rôle incorrect pour {normalized_type.upper()} : "
            f"{metadata['model_role']!r} au lieu de {expected_role!r}."
        )

    model_path, _ = get_ranker_paths(normalized_type)
    if metadata["model_file"] != model_path.name:
        raise ValueError(
            f"model_file incohérent : {metadata['model_file']!r} "
            f"au lieu de {model_path.name!r}."
        )

    return metadata


# ============================================================
# Prétraitement LightGBM
# ============================================================

def prepare_lightgbm_dataframe(
    feature_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> pd.DataFrame:
    """
    Reproduit le prétraitement utilisé pendant l'entraînement LightGBM.

    - ordre exact des features ;
    - valeurs catégorielles absentes -> NONE ;
    - catégories non vues au train -> __UNKNOWN__ ;
    - pd.Categorical avec les catégories apprises sur le train uniquement ;
    - colonnes numériques converties sans masquer les valeurs invalides.
    """

    if not feature_rows:
        raise RankerLoaderError("Aucune ligne de features reçue.")

    feature_columns = list(metadata["feature_columns"])
    categorical_features = list(metadata["categorical_features"])
    numeric_features = list(metadata["numeric_features"])
    category_mappings = metadata["category_mappings"]

    normalized_rows: list[dict[str, Any]] = []

    for row_index, feature_row in enumerate(feature_rows):
        missing_features = [
            column for column in feature_columns
            if column not in feature_row
        ]
        extra_features = [
            column for column in feature_row
            if column not in feature_columns
        ]

        if missing_features:
            raise RankerLoaderError(
                f"Ligne {row_index} : features manquantes : {missing_features}"
            )

        if extra_features:
            raise RankerLoaderError(
                f"Ligne {row_index} : features supplémentaires : {extra_features}"
            )

        normalized_rows.append(
            {column: feature_row[column] for column in feature_columns}
        )

    dataframe = pd.DataFrame(normalized_rows, columns=feature_columns)

    for column in categorical_features:
        categories = [str(value) for value in category_mappings[column]]
        known_categories = set(categories)

        normalized_values = [
            _normalize_category_value(value)
            for value in dataframe[column].tolist()
        ]

        normalized_values = [
            value if value in known_categories else "__UNKNOWN__"
            for value in normalized_values
        ]

        dataframe[column] = pd.Categorical(
            normalized_values,
            categories=categories,
        )

    for column in numeric_features:
        try:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="raise",
            )
        except (TypeError, ValueError) as error:
            raise RankerLoaderError(
                f"Feature numérique invalide dans {column!r}: {error}"
            ) from error

    return dataframe.loc[:, feature_columns]


# ============================================================
# Chargement et validation du modèle
# ============================================================

def load_ranker_model(
    ranker_type: str,
) -> lgb.Booster:
    """Charge le modèle LightGBM officiel au format texte."""

    model_path, _ = get_ranker_paths(ranker_type)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle officiel introuvable : {model_path}"
        )

    try:
        return lgb.Booster(model_file=str(model_path))
    except Exception as error:
        raise RankerLoaderError(
            f"Impossible de charger le modèle {model_path}: {error}"
        ) from error


def validate_model_and_metadata(
    ranker_type: str,
    model: lgb.Booster,
    metadata: dict[str, Any],
) -> None:
    """Vérifie le contrat modèle/metadata de l'artefact officiel."""

    normalized_type = normalize_ranker_type(ranker_type)
    model_path, _ = get_ranker_paths(normalized_type)

    actual_sha256 = file_sha256(model_path)
    if actual_sha256 != metadata["model_sha256"]:
        raise RankerLoaderError(
            f"SHA256 incorrect pour {normalized_type.upper()} : "
            f"{actual_sha256} != {metadata['model_sha256']}"
        )

    model_features = list(model.feature_name())
    metadata_features = list(metadata["feature_columns"])

    if model_features != metadata_features:
        differences: list[dict[str, Any]] = []
        maximum_length = max(len(model_features), len(metadata_features))

        for index in range(maximum_length):
            model_feature = (
                model_features[index]
                if index < len(model_features)
                else None
            )
            metadata_feature = (
                metadata_features[index]
                if index < len(metadata_features)
                else None
            )
            if model_feature != metadata_feature:
                differences.append(
                    {
                        "position": index,
                        "model": model_feature,
                        "metadata": metadata_feature,
                    }
                )

        raise RankerLoaderError(
            f"Ordre des features incorrect pour {normalized_type.upper()}. "
            f"Premières différences : {differences[:5]}"
        )

    expected_tree_count = int(metadata["tree_count"])
    actual_tree_count = int(model.num_trees())
    if actual_tree_count != expected_tree_count:
        raise RankerLoaderError(
            f"Nombre d'arbres incorrect pour {normalized_type.upper()} : "
            f"{actual_tree_count} != {expected_tree_count}"
        )

    expected_categories = [
        list(metadata["category_mappings"][column])
        for column in metadata["categorical_features"]
    ]
    model_categories = model.pandas_categorical

    if model_categories is None:
        raise RankerLoaderError(
            f"Le modèle {normalized_type.upper()} ne contient pas "
            "les métadonnées pandas_categorical LightGBM."
        )

    actual_categories = [list(values) for values in model_categories]
    if actual_categories != expected_categories:
        raise RankerLoaderError(
            f"Mappings catégoriels différents pour {normalized_type.upper()}."
        )


@lru_cache(maxsize=2)
def _load_ranker_bundle_cached(
    normalized_type: str,
) -> tuple[lgb.Booster, dict[str, Any]]:
    metadata = load_ranker_metadata(normalized_type)
    model = load_ranker_model(normalized_type)
    validate_model_and_metadata(
        ranker_type=normalized_type,
        model=model,
        metadata=metadata,
    )
    return model, metadata


def load_ranker_bundle(
    ranker_type: str,
) -> tuple[lgb.Booster, dict[str, Any]]:
    """
    Charge et valide un bundle officiel.

    Le Booster est mis en cache pour éviter de recharger le modèle pour
    chaque cas architectural. Les metadata sont recopiées avant retour.
    """

    normalized_type = normalize_ranker_type(ranker_type)
    model, metadata = _load_ranker_bundle_cached(normalized_type)
    return model, copy.deepcopy(metadata)


def clear_ranker_cache() -> None:
    """Vide le cache des modèles officiels, utile pour tests/développement."""

    _load_ranker_bundle_cached.cache_clear()


def print_ranker_summary(
    ranker_type: str,
    model: lgb.Booster,
    metadata: dict[str, Any],
) -> None:
    """Affiche un résumé de l'artefact officiel chargé."""

    print("=" * 72)
    print(f"RANKER OFFICIEL : {ranker_type.upper()}")
    print(f"Rôle             : {metadata['model_role']}")
    print(f"Famille          : {metadata['model_family']}")
    print(f"Type             : {metadata['model_type']}")
    print(f"Seed             : {metadata['selected_seed']}")
    print(f"Features         : {len(metadata['feature_columns'])}")
    print(f"Arbres           : {model.num_trees()}")
    print(f"Dataset SHA256   : {metadata['dataset_sha256_uncompressed']}")
    print(f"Model SHA256     : {metadata['model_sha256']}")
    print("Contrat modèle/metadata : VALIDÉ")


if __name__ == "__main__":
    for current_ranker in ("mdt", "ost"):
        current_model, current_metadata = load_ranker_bundle(current_ranker)
        print_ranker_summary(
            ranker_type=current_ranker,
            model=current_model,
            metadata=current_metadata,
        )

    print("\nChargement des deux modèles LightGBM officiels : VALIDÉ")
