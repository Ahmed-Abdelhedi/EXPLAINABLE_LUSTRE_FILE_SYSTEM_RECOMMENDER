"""Calcul des features métier pour le Lustre Architecture Generator.

Ce module transforme la sortie validée de ``workload_analyzer.py`` en un
contrat de features explicite pour les plans MDT et OST.

Il ne choisit aucun disque, niveau RAID, nombre de targets ou paramètre de
striping. Ces décisions appartiennent aux modules suivants du pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable: Any, **_: Any) -> Any:  # type: ignore[misc]
        return iterable


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_FILE = BASE_DIR / "output" / "workload_analysis_dataset.json"
DEFAULT_CONFIG_FILE = BASE_DIR / "config" / "architecture_rules.json"
DEFAULT_OUTPUT_FILE = BASE_DIR / "output" / "workload_features_dataset.json"


class FeatureCalculationError(ValueError):
    """Erreur de validation ou de calcul des features."""


def load_json(path: Path) -> Any:
    """Charge un fichier JSON et produit des erreurs lisibles."""

    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise FeatureCalculationError(
            f"JSON invalide dans {path} : ligne {exc.lineno}, "
            f"colonne {exc.colno}."
        ) from exc


def require_mapping(data: dict[str, Any], field: str, case_id: str) -> dict[str, Any]:
    """Lit une section objet obligatoire."""

    value = data.get(field)
    if not isinstance(value, dict):
        raise FeatureCalculationError(
            f"{case_id} : la section '{field}' doit être un objet JSON."
        )
    return value


def require_number(
    data: dict[str, Any],
    field: str,
    case_id: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Lit et valide une valeur numérique finie."""

    value = data.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FeatureCalculationError(
            f"{case_id} : '{field}' doit être numérique."
        )

    numeric = float(value)
    if not math.isfinite(numeric):
        raise FeatureCalculationError(
            f"{case_id} : '{field}' doit être fini."
        )
    if minimum is not None and numeric < minimum:
        raise FeatureCalculationError(
            f"{case_id} : '{field}' doit être >= {minimum}."
        )
    if maximum is not None and numeric > maximum:
        raise FeatureCalculationError(
            f"{case_id} : '{field}' doit être <= {maximum}."
        )
    return numeric


def validate_threshold_order(
    values: list[tuple[str, float]],
    *,
    strictly_increasing: bool = True,
) -> None:
    """Vérifie l'ordre d'une série de seuils."""

    for (previous_name, previous), (current_name, current) in zip(
        values,
        values[1:],
    ):
        valid = current > previous if strictly_increasing else current >= previous
        if not valid:
            operator = ">" if strictly_increasing else ">="
            raise FeatureCalculationError(
                f"Configuration invalide : {current_name} doit être "
                f"{operator} {previous_name}."
            )


def validate_config(config: dict[str, Any]) -> None:
    """Valide les règles nécessaires au calcul des features."""

    required_sections = {
        "version",
        "importance_thresholds",
        "file_size_thresholds_gb",
        "parallelism_thresholds",
        "bandwidth_thresholds_gbps",
        "metadata_pressure",
        "data_pressure",
    }
    missing = sorted(required_sections - set(config))
    if missing:
        raise FeatureCalculationError(
            "Sections manquantes dans architecture_rules.json : "
            + ", ".join(missing)
        )

    importance = config["importance_thresholds"]
    if not isinstance(importance, dict):
        raise FeatureCalculationError(
            "importance_thresholds doit être un objet JSON."
        )
    for key in ("medium", "high", "critical"):
        if key not in importance:
            raise FeatureCalculationError(
                f"Seuil manquant : importance_thresholds.{key}."
            )
    validate_threshold_order(
        [
            ("medium", float(importance["medium"])),
            ("high", float(importance["high"])),
            ("critical", float(importance["critical"])),
        ]
    )

    file_sizes = config["file_size_thresholds_gb"]
    validate_threshold_order(
        [
            ("small_max", float(file_sizes["small_max"])),
            ("medium_max", float(file_sizes["medium_max"])),
        ]
    )

    parallelism = config["parallelism_thresholds"]
    validate_threshold_order(
        [
            ("medium", float(parallelism["medium"])),
            ("high", float(parallelism["high"])),
            ("massive", float(parallelism["massive"])),
        ]
    )

    bandwidth = config["bandwidth_thresholds_gbps"]
    validate_threshold_order(
        [
            ("medium", float(bandwidth["medium"])),
            ("high", float(bandwidth["high"])),
            ("very_high", float(bandwidth["very_high"])),
        ]
    )

    metadata_pressure = config["metadata_pressure"]
    validate_threshold_order(
        [
            ("medium_score", float(metadata_pressure["medium_score"])),
            ("high_score", float(metadata_pressure["high_score"])),
        ]
    )
    validate_threshold_order(
        [
            ("medium_file_count", float(
                metadata_pressure["medium_file_count"])),
            ("high_file_count", float(metadata_pressure["high_file_count"])),
        ]
    )

    data_pressure = config["data_pressure"]
    validate_threshold_order(
        [
            ("medium_score", float(data_pressure["medium_score"])),
            ("high_score", float(data_pressure["high_score"])),
        ]
    )


def classify_file_size(size_gb: float, config: dict[str, Any]) -> str:
    """Classe la taille moyenne des fichiers."""

    thresholds = config["file_size_thresholds_gb"]
    if size_gb < float(thresholds["small_max"]):
        return "small_files"
    if size_gb < float(thresholds["medium_max"]):
        return "medium_files"
    return "large_files"


def classify_parallelism(client_count: int, config: dict[str, Any]) -> str:
    """Classe le niveau de parallélisme utilisateur."""

    thresholds = config["parallelism_thresholds"]
    if client_count >= int(thresholds["massive"]):
        return "massive"
    if client_count >= int(thresholds["high"]):
        return "high"
    if client_count >= int(thresholds["medium"]):
        return "medium"
    return "low"


def classify_bandwidth(total_gbps: float, config: dict[str, Any]) -> str:
    """Classe le débit agrégé demandé."""

    thresholds = config["bandwidth_thresholds_gbps"]
    if total_gbps >= float(thresholds["very_high"]):
        return "very_high"
    if total_gbps >= float(thresholds["high"]):
        return "high"
    if total_gbps >= float(thresholds["medium"]):
        return "medium"
    return "low"


def classify_importance(score: float, config: dict[str, Any]) -> str:
    """Transforme un score absolu en importance indépendante de l'autre plan."""

    thresholds = config["importance_thresholds"]
    if score >= float(thresholds["critical"]):
        return "critical"
    if score >= float(thresholds["high"]):
        return "high"
    if score >= float(thresholds["medium"]):
        return "medium"
    return "low"


def classify_metadata_pressure(
    metadata_score: float,
    file_count: int,
    average_file_size_gb: float,
    config: dict[str, Any],
) -> str:
    """Classe la pression MDT avec score, volumétrie et petits fichiers.

    Un faible nombre de petits fichiers n'est pas suffisant pour produire une
    pression élevée. La combinaison petits fichiers + volumétrie moyenne est
    en revanche considérée comme une forte pression metadata.
    """

    rules = config["metadata_pressure"]
    high_score = float(rules["high_score"])
    medium_score = float(rules["medium_score"])
    high_count = int(rules["high_file_count"])
    medium_count = int(rules["medium_file_count"])
    small_size = float(rules["small_file_size_gb"])

    is_small_file_workload = average_file_size_gb < small_size

    if (
        metadata_score >= high_score
        or file_count >= high_count
        or (is_small_file_workload and file_count >= medium_count)
    ):
        return "high"

    if (
        metadata_score >= medium_score
        or file_count >= medium_count
        or is_small_file_workload
    ):
        return "medium"

    return "low"


def classify_data_pressure(data_score: float, config: dict[str, Any]) -> str:
    """Classe la pression OST selon le score absolu."""

    rules = config["data_pressure"]
    if data_score >= float(rules["high_score"]):
        return "high"
    if data_score >= float(rules["medium_score"]):
        return "medium"
    return "low"


def classify_read_write_profile(read_percent: float, write_percent: float) -> str:
    """Décrit le profil lecture/écriture sans remplacer les pourcentages."""

    difference = read_percent - write_percent
    if difference >= 20:
        return "read_dominant"
    if difference <= -20:
        return "write_dominant"
    return "balanced_rw"


def role_from_workload(workload_type: str) -> dict[str, str]:
    """Convertit le workload dominant en indication de rôle relative."""

    mapping = {
        "metadata_heavy": {
            "primary_role": "MDT",
            "secondary_role": "metadata",
        },
        "data_heavy": {
            "primary_role": "OST",
            "secondary_role": "data",
        },
        "balanced": {
            "primary_role": "BALANCED",
            "secondary_role": "mixed",
        },
    }

    if workload_type not in mapping:
        raise FeatureCalculationError(
            f"workload_type non supporté : '{workload_type}'."
        )
    return mapping[workload_type]


def calculate_features(case: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Calcule les features MDT/OST d'un cas analysé."""

    case_id = case.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise FeatureCalculationError(
            "Chaque cas doit contenir un case_id non vide."
        )

    source = require_mapping(case, "source_requirement", case_id)
    planning = require_mapping(case, "capacity_planning", case_id)
    normalized = require_mapping(case, "normalized_factors", case_id)
    scores = require_mapping(case, "scores", case_id)
    metadata_indicators = require_mapping(case, "metadata_indicators", case_id)
    data_indicators = require_mapping(case, "data_indicators", case_id)
    constraints = require_mapping(case, "constraints", case_id)
    preferences = require_mapping(case, "preferences", case_id)

    workload_type = case.get("workload_type")
    if not isinstance(workload_type, str):
        raise FeatureCalculationError(
            f"{case_id} : workload_type doit être une chaîne."
        )

    metadata_score = require_number(
        scores,
        "metadata_score",
        case_id,
        minimum=0.0,
        maximum=1.0,
    )
    data_score = require_number(
        scores,
        "data_score",
        case_id,
        minimum=0.0,
        maximum=1.0,
    )
    score_difference = require_number(
        scores,
        "score_difference",
        case_id,
        minimum=-1.0,
        maximum=1.0,
    )

    if not math.isclose(
        score_difference,
        metadata_score - data_score,
        abs_tol=2e-6,
    ):
        raise FeatureCalculationError(
            f"{case_id} : score_difference incohérent."
        )

    file_count = int(require_number(
        metadata_indicators,
        "file_count",
        case_id,
        minimum=1,
    ))
    average_file_size_gb = require_number(
        metadata_indicators,
        "average_file_size_gb",
        case_id,
        minimum=0,
    )
    client_count = int(require_number(
        metadata_indicators,
        "client_count",
        case_id,
        minimum=1,
    ))

    planned_capacity_tib = require_number(
        data_indicators,
        "planned_usable_capacity_tib",
        case_id,
        minimum=0,
    )
    requested_capacity_tib = require_number(
        data_indicators,
        "requested_capacity_tib",
        case_id,
        minimum=0,
    )
    read_gbps = require_number(
        data_indicators,
        "target_read_gbps",
        case_id,
        minimum=0,
    )
    write_gbps = require_number(
        data_indicators,
        "target_write_gbps",
        case_id,
        minimum=0,
    )
    total_bandwidth_gbps = require_number(
        data_indicators,
        "total_bandwidth_gbps",
        case_id,
        minimum=0,
    )
    max_file_size_gb = require_number(
        data_indicators,
        "max_file_size_gb",
        case_id,
        minimum=average_file_size_gb,
    )

    if not math.isclose(
        total_bandwidth_gbps,
        read_gbps + write_gbps,
        abs_tol=1e-6,
    ):
        raise FeatureCalculationError(
            f"{case_id} : total_bandwidth_gbps doit être égal à "
            "target_read_gbps + target_write_gbps."
        )

    read_write_ratio = require_mapping(source, "read_write_ratio", case_id)
    read_percent = require_number(
        read_write_ratio,
        "read_percent",
        case_id,
        minimum=0,
        maximum=100,
    )
    write_percent = require_number(
        read_write_ratio,
        "write_percent",
        case_id,
        minimum=0,
        maximum=100,
    )

    access_type = data_indicators.get("access_type")
    if not isinstance(access_type, str) or not access_type:
        raise FeatureCalculationError(
            f"{case_id} : access_type manquant ou invalide."
        )

    # Ratios explicables et numériquement plus lisibles que bandwidth/file.
    bandwidth_per_capacity = (
        total_bandwidth_gbps / planned_capacity_tib
        if planned_capacity_tib > 0
        else 0.0
    )
    bandwidth_per_million_files = (
        total_bandwidth_gbps / (file_count / 1_000_000)
        if file_count > 0
        else 0.0
    )
    files_per_tib = (
        file_count / planned_capacity_tib
        if planned_capacity_tib > 0
        else 0.0
    )
    clients_per_tib = (
        client_count / planned_capacity_tib
        if planned_capacity_tib > 0
        else 0.0
    )

    file_size_class = classify_file_size(average_file_size_gb, config)
    parallelism_level = classify_parallelism(client_count, config)
    bandwidth_level = classify_bandwidth(total_bandwidth_gbps, config)
    metadata_pressure = classify_metadata_pressure(
        metadata_score,
        file_count,
        average_file_size_gb,
        config,
    )
    data_pressure = classify_data_pressure(data_score, config)

    role_dominance = role_from_workload(workload_type)
    absolute_importance = {
        "mdt_importance": classify_importance(metadata_score, config),
        "ost_importance": classify_importance(data_score, config),
    }

    return {
        "case_id": case_id,
        "workload_summary": {
            "workload_type": workload_type,
            "read_write_profile": classify_read_write_profile(
                read_percent,
                write_percent,
            ),
            "access_type": access_type,
            "file_size_class": file_size_class,
            "parallelism_level": parallelism_level,
            "bandwidth_level": bandwidth_level,
        },
        "scores": {
            "metadata_score": round(metadata_score, 6),
            "data_score": round(data_score, 6),
            "score_difference": round(score_difference, 6),
        },
        "role_analysis": {
            "role_dominance": role_dominance,
            "absolute_importance": absolute_importance,
            "interpretation": (
                "La dominance compare MDT et OST ; l'importance mesure "
                "chaque plan indépendamment."
            ),
        },
        "mdt_features": {
            "file_count": file_count,
            "average_file_size_gb": round(average_file_size_gb, 9),
            "client_count": client_count,
            "small_file_factor": round(
                require_number(
                    normalized,
                    "small_file_factor",
                    case_id,
                    minimum=0,
                    maximum=1,
                ),
                6,
            ),
            "file_count_score": round(
                require_number(
                    normalized,
                    "file_count_score",
                    case_id,
                    minimum=0,
                    maximum=1,
                ),
                6,
            ),
            "client_count_score": round(
                require_number(
                    normalized,
                    "client_count_score",
                    case_id,
                    minimum=0,
                    maximum=1,
                ),
                6,
            ),
            "metadata_pressure": metadata_pressure,
            "files_per_planned_tib": round(files_per_tib, 6),
            "clients_per_planned_tib": round(clients_per_tib, 9),
        },
        "ost_features": {
            "requested_usable_capacity_tib": round(requested_capacity_tib, 6),
            "planned_usable_capacity_tib": round(planned_capacity_tib, 6),
            "target_read_gbps": round(read_gbps, 6),
            "target_write_gbps": round(write_gbps, 6),
            "total_bandwidth_gbps": round(total_bandwidth_gbps, 6),
            "bandwidth_per_planned_tib": round(
                bandwidth_per_capacity,
                9,
            ),
            "bandwidth_per_million_files": round(
                bandwidth_per_million_files,
                9,
            ),
            "large_file_factor": round(
                require_number(
                    normalized,
                    "large_file_factor",
                    case_id,
                    minimum=0,
                    maximum=1,
                ),
                6,
            ),
            "capacity_score": round(
                require_number(
                    normalized,
                    "capacity_score",
                    case_id,
                    minimum=0,
                    maximum=1,
                ),
                6,
            ),
            "bandwidth_score": round(
                require_number(
                    normalized,
                    "bandwidth_score",
                    case_id,
                    minimum=0,
                    maximum=1,
                ),
                6,
            ),
            "data_pressure": data_pressure,
            "average_file_size_gb": round(average_file_size_gb, 9),
            "max_file_size_gb": round(max_file_size_gb, 9),
            "access_type": access_type,
        },
        "capacity_planning": planning,
        "read_write_ratio": {
            "read_percent": round(read_percent, 6),
            "write_percent": round(write_percent, 6),
        },
        "constraints": constraints,
        "preferences": preferences,
        "trace": {
            "feature_calculator_version": "3.0",
            "rules_version": str(config["version"]),
            "input_analyzer_version": str(
                require_mapping(case, "trace", case_id).get(
                    "analyzer_version",
                    "unknown",
                )
            ),
            "ratio_capacity_basis": "planned_usable_capacity_tib",
            "role_semantics": "relative_dominance_and_absolute_importance",
        },
    }


def generate_dataset(
    cases: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Calcule les features de tous les cas et vérifie les doublons."""

    results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for case in tqdm(cases, desc="Calcul des features"):
        if not isinstance(case, dict):
            raise FeatureCalculationError(
                "Chaque entrée du dataset doit être un objet JSON."
            )

        result = calculate_features(case, config)
        case_id = result["case_id"]
        if case_id in seen_ids:
            raise FeatureCalculationError(
                f"case_id dupliqué : {case_id}."
            )
        seen_ids.add(case_id)
        results.append(result)

    return results


def save_json(data: Any, path: Path) -> None:
    """Sauvegarde un JSON UTF-8 avec création du dossier parent."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def print_summary(results: list[dict[str, Any]], output_path: Path) -> None:
    """Affiche un résumé métier de la génération."""

    workload_counts = Counter(
        result["workload_summary"]["workload_type"]
        for result in results
    )
    mdt_importance_counts = Counter(
        result["role_analysis"]["absolute_importance"]["mdt_importance"]
        for result in results
    )
    ost_importance_counts = Counter(
        result["role_analysis"]["absolute_importance"]["ost_importance"]
        for result in results
    )
    metadata_pressure_counts = Counter(
        result["mdt_features"]["metadata_pressure"]
        for result in results
    )
    data_pressure_counts = Counter(
        result["ost_features"]["data_pressure"]
        for result in results
    )

    print(f"Cas chargés : {len(results)}")
    print(f"Cas générés : {len(results)}")

    print("Distribution workload :")
    for name in ("metadata_heavy", "balanced", "data_heavy"):
        print(f"  - {name}: {workload_counts.get(name, 0)}")

    print("Importance MDT :")
    for name in ("low", "medium", "high", "critical"):
        print(f"  - {name}: {mdt_importance_counts.get(name, 0)}")

    print("Importance OST :")
    for name in ("low", "medium", "high", "critical"):
        print(f"  - {name}: {ost_importance_counts.get(name, 0)}")

    print("Pression MDT :")
    for name in ("low", "medium", "high"):
        print(f"  - {name}: {metadata_pressure_counts.get(name, 0)}")

    print("Pression OST :")
    for name in ("low", "medium", "high"):
        print(f"  - {name}: {data_pressure_counts.get(name, 0)}")

    print(f"Fichier sauvegardé : {output_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcule les features MDT/OST à partir du workload analysé."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_json(args.config)
    cases = load_json(args.input)

    if not isinstance(config, dict):
        raise FeatureCalculationError(
            "architecture_rules.json doit contenir un objet JSON."
        )
    if not isinstance(cases, list):
        raise FeatureCalculationError(
            "Le dataset d'analyse doit contenir une liste JSON."
        )

    validate_config(config)
    results = generate_dataset(cases, config)
    save_json(results, args.output)
    print_summary(results, args.output)


if __name__ == "__main__":
    try:
        main()
    except (FeatureCalculationError, FileNotFoundError) as exc:
        print(f"Erreur : {exc}")
        raise SystemExit(1) from exc
