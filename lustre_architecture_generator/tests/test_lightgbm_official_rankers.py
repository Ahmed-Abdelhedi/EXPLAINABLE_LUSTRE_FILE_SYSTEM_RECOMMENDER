from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# Chemins
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANKING_DIR = PROJECT_ROOT / "src" / "ranking"
RANKER_LOADER_PATH = RANKING_DIR / "ranker_loader.py"


# ============================================================
# Chargement explicite du module testé
# ============================================================

def _load_ranker_loader_module() -> Any:
    """
    Charge ranker_loader.py directement depuis son chemin.

    Cette méthode évite un faux positif Pylance lié à l'ajout
    dynamique de src/ranking dans sys.path tout en testant
    exactement le fichier runtime du projet.
    """

    if not RANKER_LOADER_PATH.exists():
        raise FileNotFoundError(
            f"ranker_loader.py introuvable : {RANKER_LOADER_PATH}"
        )

    module_name = "_official_ranker_loader_under_test"

    spec = importlib.util.spec_from_file_location(
        module_name,
        RANKER_LOADER_PATH,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Impossible de créer la spec pour {RANKER_LOADER_PATH}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


ranker_loader = _load_ranker_loader_module()

file_sha256 = ranker_loader.file_sha256
get_ranker_paths = ranker_loader.get_ranker_paths
load_ranker_bundle = ranker_loader.load_ranker_bundle
load_ranker_metadata = ranker_loader.load_ranker_metadata
prepare_lightgbm_dataframe = ranker_loader.prepare_lightgbm_dataframe


# ============================================================
# Contrat attendu des modèles officiels
# ============================================================

EXPECTED = {
    "mdt": {
        "seed": 168,
        "features": 49,
        "trees": 449,
        "dataset_sha256": (
            "a6dbcb1ae8c446f626a05d1f8393500"
            "a8ee77292770baf6e6ce10dc5824b273c"
        ),
    },
    "ost": {
        "seed": 84,
        "features": 52,
        "trees": 709,
        "dataset_sha256": (
            "28380ba8e4ae5d988da834b5d74bce6"
            "bd3062d2a948cdece3710227ae51fd2b2"
        ),
    },
}


def synthetic_row(metadata: dict[str, Any]) -> dict[str, Any]:
    """Construit une ligne synthétique conforme au contrat de features."""

    row: dict[str, Any] = {}

    for column in metadata["categorical_features"]:
        row[column] = metadata["category_mappings"][column][0]

    for column in metadata["numeric_features"]:
        row[column] = 0.0

    return {
        column: row[column]
        for column in metadata["feature_columns"]
    }


def test_official_ranker_metadata_contract() -> None:
    for role, expected in EXPECTED.items():
        metadata = load_ranker_metadata(role)

        assert metadata["status"] == "OFFICIAL"
        assert metadata["model_family"] == "LightGBM"
        assert metadata["model_type"] == "LGBMRanker"
        assert metadata["selected_seed"] == expected["seed"]
        assert len(metadata["feature_columns"]) == expected["features"]

        assert (
            metadata["dataset_sha256_uncompressed"]
            == expected["dataset_sha256"]
        )

        assert metadata["feature_columns"] == (
            metadata["categorical_features"]
            + metadata["numeric_features"]
        )


def test_official_model_file_hashes_match_metadata() -> None:
    for role in EXPECTED:
        model_path, _ = get_ranker_paths(role)
        metadata = load_ranker_metadata(role)

        assert model_path.exists()
        assert file_sha256(model_path) == metadata["model_sha256"]


def test_official_lightgbm_models_load_and_match_contract() -> None:
    for role, expected in EXPECTED.items():
        model, metadata = load_ranker_bundle(role)

        assert list(model.feature_name()) == metadata["feature_columns"]
        assert model.num_trees() == expected["trees"]

        expected_categories = [
            metadata["category_mappings"][column]
            for column in metadata["categorical_features"]
        ]

        assert model.pandas_categorical == expected_categories


def test_unknown_and_missing_categories_are_mapped_deterministically() -> None:
    for role in EXPECTED:
        metadata = load_ranker_metadata(role)
        row = synthetic_row(metadata)

        first_category = metadata["categorical_features"][0]

        unknown_row = dict(row)
        unknown_row[first_category] = (
            "VALUE_NEVER_SEEN_DURING_TRAINING"
        )

        missing_row = dict(row)
        missing_row[first_category] = None

        frame = prepare_lightgbm_dataframe(
            [unknown_row, missing_row],
            metadata,
        )

        values = frame[first_category].astype(str).tolist()

        assert values == ["__UNKNOWN__", "NONE"]
        assert list(frame.columns) == metadata["feature_columns"]

        assert isinstance(
            frame[first_category].dtype,
            pd.CategoricalDtype,
        )


def test_official_predictions_are_finite_and_repeatable() -> None:
    for role in EXPECTED:
        model, metadata = load_ranker_bundle(role)
        row = synthetic_row(metadata)

        frame = prepare_lightgbm_dataframe(
            [row, row],
            metadata,
        )

        raw_prediction_a: Any = model.predict(
            frame,
            num_iteration=model.num_trees(),
            validate_features=True,
        )

        raw_prediction_b: Any = model.predict(
            frame,
            num_iteration=model.num_trees(),
            validate_features=True,
        )

        prediction_a = np.asarray(
            raw_prediction_a,
            dtype=np.float64,
        ).reshape(-1)

        prediction_b = np.asarray(
            raw_prediction_b,
            dtype=np.float64,
        ).reshape(-1)

        assert prediction_a.size == 2
        assert prediction_b.size == 2

        assert np.isfinite(prediction_a).all()
        assert np.isfinite(prediction_b).all()

        assert np.allclose(
            prediction_a,
            prediction_b,
            rtol=0.0,
            atol=0.0,
        )

        assert math.isfinite(float(prediction_a[0]))