from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


# ============================================================
# Chemins du projet
# ============================================================

# lustre_architecture_generator/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# lustre_architecture_generator/src/
SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# Imports des générateurs déterministes
# ============================================================

import mdt_candidate_generator as mdt_generator  # noqa: E402
import ost_candidate_generator as ost_generator  # noqa: E402


try:
    # Import utilisé lorsque le package est exécuté avec python -m
    from .ranker_loader import load_ranker_metadata
except ImportError:
    # Import utilisé lorsque le fichier est exécuté directement
    from ranker_loader import load_ranker_metadata


# ============================================================
# Fichiers par défaut
# ============================================================

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


# ============================================================
# Exceptions
# ============================================================

class FeatureBuilderError(ValueError):
    """Erreur de construction ou de validation des features ML."""


# ============================================================
# Fonctions communes
# ============================================================

def load_json(path: Path) -> Any:
    """Charge un fichier JSON."""

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def safe_ratio(
    numerator: float,
    denominator: float,
    *,
    zero_denominator_value: float = 0.0,
) -> float:
    """
    Effectue une division sécurisée.

    Cette fonction reprend le comportement utilisé pendant
    la création des datasets d'entraînement.
    """

    if denominator == 0:
        return zero_denominator_value

    return numerator / denominator


def categorical(value: Any) -> str:
    """
    Normalise une valeur catégorielle pour CatBoost.

    Les valeurs absentes sont remplacées par la catégorie NONE.
    """

    if value is None or value == "":
        return "NONE"

    return str(value)


def numeric(value: Any) -> float | int:
    """
    Normalise une valeur numérique pour CatBoost.

    Dans le CSV d'entraînement, une valeur vide était relue par
    pandas comme NaN. Pour l'inférence, on utilise donc float("nan").
    """

    if value is None or value == "":
        return float("nan")

    if isinstance(value, bool):
        return int(value)

    return value


def validate_required_keys(
    source: dict[str, Any],
    required_keys: set[str],
    source_name: str,
) -> None:
    """Vérifie la présence des champs indispensables."""

    missing_keys = required_keys - set(source.keys())

    if missing_keys:
        raise FeatureBuilderError(
            f"{source_name} incomplet. "
            f"Champs manquants : {sorted(missing_keys)}"
        )


def validate_feature_order(
    ranker_type: str,
    features: dict[str, Any],
) -> None:
    """
    Vérifie que les features ont exactement les mêmes noms
    et le même ordre que dans les metadata du modèle.
    """

    metadata = load_ranker_metadata(ranker_type)

    expected_columns = metadata["feature_columns"]
    actual_columns = list(features.keys())

    if actual_columns == expected_columns:
        return

    differences: list[dict[str, Any]] = []

    maximum_length = max(
        len(actual_columns),
        len(expected_columns),
    )

    for index in range(maximum_length):
        actual = (
            actual_columns[index]
            if index < len(actual_columns)
            else None
        )

        expected = (
            expected_columns[index]
            if index < len(expected_columns)
            else None
        )

        if actual != expected:
            differences.append(
                {
                    "position": index,
                    "actual": actual,
                    "expected": expected,
                }
            )

    raise FeatureBuilderError(
        f"Les features {ranker_type.upper()} ne correspondent "
        "pas aux metadata. Premières différences : "
        f"{differences[:10]}"
    )


def validate_no_infinite_values(
    feature_row: dict[str, Any],
    ranker_type: str,
) -> None:
    """Bloque les valeurs numériques infinies."""

    infinite_features = [
        feature_name
        for feature_name, value in feature_row.items()
        if isinstance(value, float)
        and math.isinf(value)
    ]

    if infinite_features:
        raise FeatureBuilderError(
            f"Valeurs infinies détectées dans les features "
            f"{ranker_type.upper()} : {infinite_features}"
        )


# ============================================================
# Feature Builder MDT
# ============================================================

def build_mdt_feature_row(
    architecture: dict[str, Any],
    drive: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """
    Construit les 49 features attendues par le MDT Ranker.

    Parameters
    ----------
    architecture:
        Cas produit par architecture_generator.py.

    drive:
        Drive provenant du catalogue canonique.

    candidate:
        Résultat déterministe produit par
        mdt_candidate_generator.evaluate_drive().

    Returns
    -------
    dict
        Features MDT ordonnées conformément aux metadata.
    """

    validate_required_keys(
        architecture,
        {
            "MDT_requirement",
            "constraints",
            "preferences",
        },
        "Architecture",
    )

    requirement = architecture["MDT_requirement"]
    constraints = architecture["constraints"]
    preferences = architecture["preferences"]

    validate_required_keys(
        requirement,
        {
            "priority",
            "required_total_iops",
            "required_read_iops",
            "required_write_iops",
            "required_metadata_capacity_tib",
            "latency_requirement",
            "endurance_requirement",
            "reliability_requirement",
        },
        "MDT_requirement",
    )

    validate_required_keys(
        constraints,
        {
            "ha_required",
            "max_budget_usd",
            "max_power_w",
        },
        "constraints",
    )

    validate_required_keys(
        preferences,
        {
            "performance_priority",
            "cost_priority",
            "power_priority",
            "reliability_priority",
        },
        "preferences",
    )

    validate_required_keys(
        drive,
        {
            "media_type",
            "protocol",
            "drive_form_factor_standard",
            "pcie_gen_required",
            "pcie_lanes_required",
            "latency_class",
            "capacity_tib",
            "random_read_iops_4k",
            "random_write_iops_4k",
            "endurance_dwpd_numeric",
            "mtbf_hours",
            "warranty_years",
            "price_en_dollars",
            "power_consumption_en_w",
        },
        "Drive MDT",
    )

    validate_required_keys(
        candidate,
        {
            "count_by_capacity",
            "count_by_read_iops",
            "count_by_write_iops",
            "raw_minimum_drive_count",
            "raw_provided_capacity_tib",
            "raw_provided_read_iops",
            "raw_provided_write_iops",
            "raw_drive_cost_usd",
            "raw_drive_power_w",
        },
        "Candidat MDT",
    )

    required_capacity = float(
        requirement["required_metadata_capacity_tib"]
    )

    required_read_iops = float(
        requirement["required_read_iops"]
    )

    required_write_iops = float(
        requirement["required_write_iops"]
    )

    minimum_dwpd = (
        mdt_generator.ENDURANCE_MIN_DWPD[
            requirement["endurance_requirement"]
        ]
    )

    minimum_mtbf = (
        mdt_generator.RELIABILITY_MIN_MTBF_HOURS[
            requirement["reliability_requirement"]
        ]
    )

    required_latency_order = (
        mdt_generator.LATENCY_ORDER[
            requirement["latency_requirement"]
        ]
    )

    features: dict[str, Any] = {
        # ====================================================
        # Features catégorielles MDT : 9
        # ====================================================

        "mdt_priority": categorical(
            requirement["priority"]
        ),

        "latency_requirement": categorical(
            requirement["latency_requirement"]
        ),

        "endurance_requirement": categorical(
            requirement["endurance_requirement"]
        ),

        "reliability_requirement": categorical(
            requirement["reliability_requirement"]
        ),

        "drive_media_type": categorical(
            drive["media_type"]
        ),

        "drive_protocol": categorical(
            drive["protocol"]
        ),

        "drive_form_factor": categorical(
            drive["drive_form_factor_standard"]
        ),

        "drive_pcie_gen": categorical(
            drive["pcie_gen_required"]
        ),

        "drive_latency_class": categorical(
            drive["latency_class"]
        ),

        # ====================================================
        # Features numériques MDT : besoins et préférences
        # ====================================================

        "required_total_iops": numeric(
            requirement["required_total_iops"]
        ),

        "required_read_iops": required_read_iops,

        "required_write_iops": required_write_iops,

        "required_metadata_capacity_tib":
            required_capacity,

        "ha_required": int(
            bool(constraints["ha_required"])
        ),

        "max_budget_usd": numeric(
            constraints["max_budget_usd"]
        ),

        "max_power_w": numeric(
            constraints["max_power_w"]
        ),

        "performance_priority": numeric(
            preferences["performance_priority"]
        ),

        "cost_priority": numeric(
            preferences["cost_priority"]
        ),

        "power_priority": numeric(
            preferences["power_priority"]
        ),

        "reliability_priority": numeric(
            preferences["reliability_priority"]
        ),

        # ====================================================
        # Caractéristiques du drive MDT
        # ====================================================

        "drive_pcie_lanes": numeric(
            drive["pcie_lanes_required"]
        ),

        "drive_capacity_tib": numeric(
            drive["capacity_tib"]
        ),

        "drive_random_read_iops_4k": numeric(
            drive["random_read_iops_4k"]
        ),

        "drive_random_write_iops_4k": numeric(
            drive["random_write_iops_4k"]
        ),

        "drive_endurance_dwpd": numeric(
            drive["endurance_dwpd_numeric"]
        ),

        "drive_mtbf_hours": numeric(
            drive["mtbf_hours"]
        ),

        "drive_warranty_years": numeric(
            drive["warranty_years"]
        ),

        "drive_price_usd": numeric(
            drive["price_en_dollars"]
        ),

        "drive_power_w": numeric(
            drive["power_consumption_en_w"]
        ),

        # ====================================================
        # Dimensionnement pré-RAID MDT
        # ====================================================

        "count_by_capacity": numeric(
            candidate["count_by_capacity"]
        ),

        "count_by_read_iops": numeric(
            candidate["count_by_read_iops"]
        ),

        "count_by_write_iops": numeric(
            candidate["count_by_write_iops"]
        ),

        "raw_minimum_drive_count": numeric(
            candidate["raw_minimum_drive_count"]
        ),

        "raw_provided_capacity_tib": numeric(
            candidate["raw_provided_capacity_tib"]
        ),

        "raw_provided_read_iops": numeric(
            candidate["raw_provided_read_iops"]
        ),

        "raw_provided_write_iops": numeric(
            candidate["raw_provided_write_iops"]
        ),

        "raw_drive_cost_usd": numeric(
            candidate["raw_drive_cost_usd"]
        ),

        "raw_drive_power_w": numeric(
            candidate["raw_drive_power_w"]
        ),

        # ====================================================
        # Ratios et interactions MDT
        # ====================================================

        "capacity_requirement_to_drive_ratio": round(
            safe_ratio(
                required_capacity,
                float(drive["capacity_tib"]),
            ),
            8,
        ),

        "read_requirement_to_drive_ratio": round(
            safe_ratio(
                required_read_iops,
                float(drive["random_read_iops_4k"]),
            ),
            8,
        ),

        "write_requirement_to_drive_ratio": round(
            safe_ratio(
                required_write_iops,
                float(drive["random_write_iops_4k"]),
            ),
            8,
        ),

        "capacity_headroom_ratio": round(
            safe_ratio(
                float(
                    candidate[
                        "raw_provided_capacity_tib"
                    ]
                ),
                required_capacity,
                zero_denominator_value=1.0,
            ),
            8,
        ),

        "read_iops_headroom_ratio": round(
            safe_ratio(
                float(
                    candidate[
                        "raw_provided_read_iops"
                    ]
                ),
                required_read_iops,
                zero_denominator_value=1.0,
            ),
            8,
        ),

        "write_iops_headroom_ratio": round(
            safe_ratio(
                float(
                    candidate[
                        "raw_provided_write_iops"
                    ]
                ),
                required_write_iops,
                zero_denominator_value=1.0,
            ),
            8,
        ),

        "budget_fraction": round(
            safe_ratio(
                float(
                    candidate["raw_drive_cost_usd"]
                ),
                float(constraints["max_budget_usd"]),
            ),
            8,
        ),

        "power_fraction": round(
            safe_ratio(
                float(
                    candidate["raw_drive_power_w"]
                ),
                float(constraints["max_power_w"]),
            ),
            8,
        ),

        "endurance_margin_ratio": round(
            safe_ratio(
                float(
                    drive["endurance_dwpd_numeric"]
                ),
                float(minimum_dwpd),
                zero_denominator_value=1.0,
            ),
            8,
        ),

        "reliability_margin_ratio": round(
            safe_ratio(
                float(drive["mtbf_hours"]),
                float(
                    minimum_mtbf
                    if minimum_mtbf > 0
                    else 2_500_000
                ),
            ),
            8,
        ),

        "latency_margin": (
            mdt_generator.LATENCY_ORDER[
                drive["latency_class"]
            ]
            - required_latency_order
        ),
    }

    validate_feature_order(
        ranker_type="mdt",
        features=features,
    )

    validate_no_infinite_values(
        feature_row=features,
        ranker_type="mdt",
    )

    return features


# ============================================================
# Feature Builder OST
# ============================================================

def build_ost_feature_row(
    architecture: dict[str, Any],
    drive: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """
    Construit les 52 features attendues par le OST Ranker.

    Parameters
    ----------
    architecture:
        Cas produit par architecture_generator.py.

    drive:
        Drive provenant du catalogue canonique.

    candidate:
        Résultat déterministe produit par
        ost_candidate_generator.evaluate_drive().

    Returns
    -------
    dict
        Features OST ordonnées conformément aux metadata.
    """

    validate_required_keys(
        architecture,
        {
            "OST_requirement",
            "constraints",
            "preferences",
        },
        "Architecture",
    )

    requirement = architecture["OST_requirement"]
    constraints = architecture["constraints"]
    preferences = architecture["preferences"]

    validate_required_keys(
        requirement,
        {
            "priority",
            "throughput_requirement",
            "access_pattern",
            "file_size_class",
            "reliability_requirement",
            "required_usable_capacity_tib",
            "required_read_bandwidth_gbps",
            "required_write_bandwidth_gbps",
            "required_total_bandwidth_gbps",
        },
        "OST_requirement",
    )

    validate_required_keys(
        constraints,
        {
            "ha_required",
            "max_budget_usd",
            "max_power_w",
        },
        "constraints",
    )

    validate_required_keys(
        preferences,
        {
            "performance_priority",
            "cost_priority",
            "power_priority",
            "reliability_priority",
        },
        "preferences",
    )

    validate_required_keys(
        drive,
        {
            "media_type",
            "protocol",
            "drive_form_factor_standard",
            "pcie_gen_required",
            "pcie_lanes_required",
            "recording_technology",
            "capacity_tib",
            "seq_read_mb_s",
            "seq_write_mb_s",
            "mtbf_hours",
            "warranty_years",
            "rpm",
            "workload_rating_tb_per_year",
            "price_en_dollars",
            "power_consumption_en_w",
        },
        "Drive OST",
    )

    validate_required_keys(
        candidate,
        {
            "drive_read_bandwidth_gbps",
            "drive_write_bandwidth_gbps",
            "count_by_capacity",
            "count_by_read_bandwidth",
            "count_by_write_bandwidth",
            "raw_minimum_drive_count",
            "raw_provided_capacity_tib",
            "raw_provided_read_bandwidth_gbps",
            "raw_provided_write_bandwidth_gbps",
            "raw_provided_total_bandwidth_gbps",
            "raw_drive_cost_usd",
            "raw_drive_power_w",
        },
        "Candidat OST",
    )

    required_capacity = float(
        requirement["required_usable_capacity_tib"]
    )

    required_read_bandwidth = float(
        requirement["required_read_bandwidth_gbps"]
    )

    required_write_bandwidth = float(
        requirement["required_write_bandwidth_gbps"]
    )

    minimum_mtbf = (
        ost_generator.RELIABILITY_MIN_MTBF_HOURS[
            requirement["reliability_requirement"]
        ]
    )

    features: dict[str, Any] = {
        # ====================================================
        # Features catégorielles OST : 10
        # ====================================================

        "ost_priority": categorical(
            requirement["priority"]
        ),

        "throughput_requirement": categorical(
            requirement["throughput_requirement"]
        ),

        "access_pattern": categorical(
            requirement["access_pattern"]
        ),

        "file_size_class": categorical(
            requirement["file_size_class"]
        ),

        "reliability_requirement": categorical(
            requirement["reliability_requirement"]
        ),

        "drive_media_type": categorical(
            drive["media_type"]
        ),

        "drive_protocol": categorical(
            drive["protocol"]
        ),

        "drive_form_factor": categorical(
            drive["drive_form_factor_standard"]
        ),

        "drive_pcie_gen": categorical(
            drive["pcie_gen_required"]
        ),

        "drive_recording_technology": categorical(
            drive["recording_technology"]
        ),

        # ====================================================
        # Features numériques OST : besoins et préférences
        # ====================================================

        "required_usable_capacity_tib":
            required_capacity,

        "required_read_bandwidth_gbps":
            required_read_bandwidth,

        "required_write_bandwidth_gbps":
            required_write_bandwidth,

        "required_total_bandwidth_gbps": numeric(
            requirement[
                "required_total_bandwidth_gbps"
            ]
        ),

        "ha_required": int(
            bool(constraints["ha_required"])
        ),

        "max_budget_usd": numeric(
            constraints["max_budget_usd"]
        ),

        "max_power_w": numeric(
            constraints["max_power_w"]
        ),

        "performance_priority": numeric(
            preferences["performance_priority"]
        ),

        "cost_priority": numeric(
            preferences["cost_priority"]
        ),

        "power_priority": numeric(
            preferences["power_priority"]
        ),

        "reliability_priority": numeric(
            preferences["reliability_priority"]
        ),

        # ====================================================
        # Caractéristiques du drive OST
        # ====================================================

        "drive_pcie_lanes": numeric(
            drive["pcie_lanes_required"]
        ),

        "drive_capacity_tib": numeric(
            drive["capacity_tib"]
        ),

        "drive_seq_read_mb_s": numeric(
            drive["seq_read_mb_s"]
        ),

        "drive_seq_write_mb_s": numeric(
            drive["seq_write_mb_s"]
        ),

        "drive_read_bandwidth_gbps": numeric(
            candidate["drive_read_bandwidth_gbps"]
        ),

        "drive_write_bandwidth_gbps": numeric(
            candidate["drive_write_bandwidth_gbps"]
        ),

        "drive_mtbf_hours": numeric(
            drive["mtbf_hours"]
        ),

        "drive_warranty_years": numeric(
            drive["warranty_years"]
        ),

        "drive_rpm": numeric(
            drive["rpm"]
        ),

        "drive_workload_rating_tb_per_year": numeric(
            drive["workload_rating_tb_per_year"]
        ),

        "drive_price_usd": numeric(
            drive["price_en_dollars"]
        ),

        "drive_power_w": numeric(
            drive["power_consumption_en_w"]
        ),

        # ====================================================
        # Dimensionnement pré-RAID OST
        # ====================================================

        "count_by_capacity": numeric(
            candidate["count_by_capacity"]
        ),

        "count_by_read_bandwidth": numeric(
            candidate["count_by_read_bandwidth"]
        ),

        "count_by_write_bandwidth": numeric(
            candidate["count_by_write_bandwidth"]
        ),

        "raw_minimum_drive_count": numeric(
            candidate["raw_minimum_drive_count"]
        ),

        "raw_provided_capacity_tib": numeric(
            candidate["raw_provided_capacity_tib"]
        ),

        "raw_provided_read_bandwidth_gbps": numeric(
            candidate[
                "raw_provided_read_bandwidth_gbps"
            ]
        ),

        "raw_provided_write_bandwidth_gbps": numeric(
            candidate[
                "raw_provided_write_bandwidth_gbps"
            ]
        ),

        "raw_provided_total_bandwidth_gbps": numeric(
            candidate[
                "raw_provided_total_bandwidth_gbps"
            ]
        ),

        "raw_drive_cost_usd": numeric(
            candidate["raw_drive_cost_usd"]
        ),

        "raw_drive_power_w": numeric(
            candidate["raw_drive_power_w"]
        ),

        # ====================================================
        # Ratios et interactions OST
        # ====================================================

        "capacity_requirement_to_drive_ratio": round(
            safe_ratio(
                required_capacity,
                float(drive["capacity_tib"]),
            ),
            8,
        ),

        "read_requirement_to_drive_ratio": round(
            safe_ratio(
                required_read_bandwidth,
                float(
                    candidate[
                        "drive_read_bandwidth_gbps"
                    ]
                ),
            ),
            8,
        ),

        "write_requirement_to_drive_ratio": round(
            safe_ratio(
                required_write_bandwidth,
                float(
                    candidate[
                        "drive_write_bandwidth_gbps"
                    ]
                ),
            ),
            8,
        ),

        "capacity_headroom_ratio": round(
            safe_ratio(
                float(
                    candidate[
                        "raw_provided_capacity_tib"
                    ]
                ),
                required_capacity,
            ),
            8,
        ),

        "read_bandwidth_headroom_ratio": round(
            safe_ratio(
                float(
                    candidate[
                        "raw_provided_read_bandwidth_gbps"
                    ]
                ),
                required_read_bandwidth,
                zero_denominator_value=1.0,
            ),
            8,
        ),

        "write_bandwidth_headroom_ratio": round(
            safe_ratio(
                float(
                    candidate[
                        "raw_provided_write_bandwidth_gbps"
                    ]
                ),
                required_write_bandwidth,
                zero_denominator_value=1.0,
            ),
            8,
        ),

        "budget_fraction": round(
            safe_ratio(
                float(
                    candidate["raw_drive_cost_usd"]
                ),
                float(constraints["max_budget_usd"]),
            ),
            8,
        ),

        "power_fraction": round(
            safe_ratio(
                float(
                    candidate["raw_drive_power_w"]
                ),
                float(constraints["max_power_w"]),
            ),
            8,
        ),

        "reliability_margin_ratio": round(
            safe_ratio(
                float(drive["mtbf_hours"]),
                float(
                    minimum_mtbf
                    if minimum_mtbf > 0
                    else 2_500_000
                ),
            ),
            8,
        ),
    }

    validate_feature_order(
        ranker_type="ost",
        features=features,
    )

    validate_no_infinite_values(
        feature_row=features,
        ranker_type="ost",
    )

    return features


# ============================================================
# Recherche de candidats faisables pour les tests locaux
# ============================================================

def find_first_feasible_mdt_candidate(
    architecture: dict[str, Any],
    catalog: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Trouve le premier candidat MDT qui respecte toutes
    les contraintes déterministes.
    """

    requirement = architecture["MDT_requirement"]
    constraints = architecture["constraints"]
    preferences = architecture["preferences"]

    for drive in catalog:
        # Même comportement que pendant la création
        # du dataset d'entraînement MDT.
        if not bool(drive.get("mdt_eligible", False)):
            continue

        candidate, rejection_reasons = (
            mdt_generator.evaluate_drive(
                drive,
                requirement,
                constraints,
                preferences,
            )
        )

        if not rejection_reasons:
            return drive, candidate

    raise FeatureBuilderError(
        "Aucun candidat MDT faisable pour le cas testé."
    )


def find_first_feasible_ost_candidate(
    architecture: dict[str, Any],
    catalog: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Trouve le premier candidat OST qui respecte toutes
    les contraintes déterministes.
    """

    requirement = architecture["OST_requirement"]
    constraints = architecture["constraints"]
    preferences = architecture["preferences"]

    for drive in catalog:
        # Pendant l'entraînement OST, evaluate_drive()
        # était appelé sur tous les drives du catalogue.
        candidate, rejection_reasons = (
            ost_generator.evaluate_drive(
                drive,
                requirement,
                constraints,
                preferences,
            )
        )

        if not rejection_reasons:
            return drive, candidate

    raise FeatureBuilderError(
        "Aucun candidat OST faisable pour le cas testé."
    )


# ============================================================
# Affichage du résumé de test
# ============================================================

def print_feature_summary(
    ranker_type: str,
    architecture: dict[str, Any],
    drive: dict[str, Any],
    feature_row: dict[str, Any],
) -> None:
    """Affiche un résumé du test du Feature Builder."""

    metadata = load_ranker_metadata(ranker_type)

    print("=" * 60)
    print(
        f"TEST FEATURE BUILDER "
        f"{ranker_type.upper()}"
    )
    print(
        "Case ID :",
        architecture["case_id"],
    )
    print(
        "Drive   :",
        drive["drive_id"],
    )
    print(
        "Nom     :",
        drive["name"],
    )
    print(
        "Nombre de features :",
        len(feature_row),
    )
    print(
        "Features catégorielles :",
        len(metadata["categorical_features"]),
    )
    print(
        "Features numériques :",
        len(metadata["numeric_features"]),
    )
    print(
        "Ordre conforme aux metadata : VALIDÉ"
    )
    print(
        f"Construction feature "
        f"{ranker_type.upper()}    : VALIDÉE"
    )


# ============================================================
# Test local
# ============================================================

def main() -> None:
    """Teste la construction des features MDT et OST."""

    architectures = load_json(
        DEFAULT_ARCHITECTURES_PATH
    )

    catalog = load_json(
        DEFAULT_CATALOG_PATH
    )

    if not isinstance(architectures, list):
        raise TypeError(
            "Le dataset architectural doit être une liste."
        )

    if not architectures:
        raise FeatureBuilderError(
            "Le dataset architectural est vide."
        )

    if not isinstance(catalog, list):
        raise TypeError(
            "Le catalogue de drives doit être une liste."
        )

    if not catalog:
        raise FeatureBuilderError(
            "Le catalogue de drives est vide."
        )

    test_architecture = architectures[0]

    # --------------------------------------------------------
    # Test MDT
    # --------------------------------------------------------

    mdt_drive, mdt_candidate = (
        find_first_feasible_mdt_candidate(
            architecture=test_architecture,
            catalog=catalog,
        )
    )

    mdt_feature_row = build_mdt_feature_row(
        architecture=test_architecture,
        drive=mdt_drive,
        candidate=mdt_candidate,
    )

    print_feature_summary(
        ranker_type="mdt",
        architecture=test_architecture,
        drive=mdt_drive,
        feature_row=mdt_feature_row,
    )

    print()

    # --------------------------------------------------------
    # Test OST
    # --------------------------------------------------------

    ost_drive, ost_candidate = (
        find_first_feasible_ost_candidate(
            architecture=test_architecture,
            catalog=catalog,
        )
    )

    ost_feature_row = build_ost_feature_row(
        architecture=test_architecture,
        drive=ost_drive,
        candidate=ost_candidate,
    )

    print_feature_summary(
        ranker_type="ost",
        architecture=test_architecture,
        drive=ost_drive,
        feature_row=ost_feature_row,
    )

    print()
    print(
        "Construction des features MDT et OST : VALIDÉE"
    )


if __name__ == "__main__":
    main()