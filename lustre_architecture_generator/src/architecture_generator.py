"""Génération des exigences techniques MDT et OST pour Lustre.

Ce module transforme ``workload_features_dataset.json`` en
``lustre_architecture_dataset.json``.

Il produit uniquement des exigences techniques indépendantes du matériel.
Il ne choisit aucun disque, niveau RAID, nombre de targets, serveur ou
paramètre de striping.
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
DEFAULT_INPUT_FILE = BASE_DIR / "output" / "workload_features_dataset.json"
DEFAULT_CONFIG_FILE = BASE_DIR / "config" / "architecture_rules.json"
DEFAULT_OUTPUT_FILE = BASE_DIR / "output" / "lustre_architecture_dataset.json"


class ArchitectureGenerationError(ValueError):
    """Erreur de validation ou de génération des exigences Lustre."""


def load_json(path: Path) -> Any:
    """Charge un fichier JSON en produisant des erreurs lisibles."""

    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise ArchitectureGenerationError(
            f"JSON invalide dans {path} : ligne {exc.lineno}, "
            f"colonne {exc.colno}."
        ) from exc


def save_json(data: Any, path: Path) -> None:
    """Sauvegarde un JSON UTF-8 avec création du dossier parent."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def require_mapping(
    data: dict[str, Any],
    field: str,
    case_id: str,
) -> dict[str, Any]:
    """Lit une section objet obligatoire."""

    value = data.get(field)
    if not isinstance(value, dict):
        raise ArchitectureGenerationError(
            f"{case_id} : la section '{field}' doit être un objet JSON."
        )
    return value


def require_string(
    data: dict[str, Any],
    field: str,
    case_id: str,
) -> str:
    """Lit une chaîne obligatoire non vide."""

    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ArchitectureGenerationError(
            f"{case_id} : '{field}' doit être une chaîne non vide."
        )
    return value.strip()


def require_bool(
    data: dict[str, Any],
    field: str,
    case_id: str,
) -> bool:
    """Lit un booléen obligatoire."""

    value = data.get(field)
    if not isinstance(value, bool):
        raise ArchitectureGenerationError(
            f"{case_id} : '{field}' doit être booléen."
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
    """Lit une valeur numérique finie."""

    value = data.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArchitectureGenerationError(
            f"{case_id} : '{field}' doit être numérique."
        )

    number = float(value)
    if not math.isfinite(number):
        raise ArchitectureGenerationError(
            f"{case_id} : '{field}' doit être fini."
        )
    if minimum is not None and number < minimum:
        raise ArchitectureGenerationError(
            f"{case_id} : '{field}' doit être >= {minimum}."
        )
    if maximum is not None and number > maximum:
        raise ArchitectureGenerationError(
            f"{case_id} : '{field}' doit être <= {maximum}."
        )
    return number


def validate_positive_config_number(
    section: dict[str, Any],
    field: str,
    section_name: str,
    *,
    allow_zero: bool = False,
) -> float:
    """Valide une constante numérique positive de configuration."""

    value = section.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArchitectureGenerationError(
            f"Configuration invalide : {section_name}.{field} doit être numérique."
        )

    number = float(value)
    minimum_valid = number >= 0 if allow_zero else number > 0
    if not math.isfinite(number) or not minimum_valid:
        operator = ">= 0" if allow_zero else "> 0"
        raise ArchitectureGenerationError(
            f"Configuration invalide : {section_name}.{field} doit être {operator}."
        )
    return number


def validate_config(config: dict[str, Any]) -> None:
    """Valide les règles nécessaires au générateur d'architecture."""

    required_sections = {
        "version",
        "mdt_estimation",
        "ost_estimation",
        "reliability_rules",
        "bandwidth_thresholds_gbps",
    }
    missing = sorted(required_sections - set(config))
    if missing:
        raise ArchitectureGenerationError(
            "Sections manquantes dans architecture_rules.json : "
            + ", ".join(missing)
        )

    mdt = config["mdt_estimation"]
    if not isinstance(mdt, dict):
        raise ArchitectureGenerationError(
            "mdt_estimation doit être un objet JSON."
        )

    required_mdt_fields = {
        "base_iops_per_client",
        "small_file_multiplier",
        "medium_file_multiplier",
        "large_file_multiplier",
        "random_access_multiplier",
        "mixed_access_multiplier",
        "sequential_access_multiplier",
        "high_metadata_multiplier",
        "medium_metadata_multiplier",
        "low_metadata_multiplier",
        "iops_safety_factor",
        "metadata_bytes_per_file",
        "metadata_capacity_safety_factor",
    }
    missing_mdt = sorted(required_mdt_fields - set(mdt))
    if missing_mdt:
        raise ArchitectureGenerationError(
            "Champs manquants dans mdt_estimation : "
            + ", ".join(missing_mdt)
        )

    for field in required_mdt_fields:
        validate_positive_config_number(mdt, field, "mdt_estimation")

    ost = config["ost_estimation"]
    if not isinstance(ost, dict):
        raise ArchitectureGenerationError(
            "ost_estimation doit être un objet JSON."
        )

    for field in ("bandwidth_safety_factor", "capacity_safety_factor"):
        validate_positive_config_number(ost, field, "ost_estimation")

    reliability = config["reliability_rules"]
    if not isinstance(reliability, dict):
        raise ArchitectureGenerationError(
            "reliability_rules doit être un objet JSON."
        )

    allowed_reliability = {"low", "medium", "high", "critical"}
    for field in ("ha_required_level", "default_level"):
        value = reliability.get(field)
        if value not in allowed_reliability:
            raise ArchitectureGenerationError(
                f"Configuration invalide : reliability_rules.{field} "
                f"doit appartenir à {sorted(allowed_reliability)}."
            )

    bandwidth_thresholds = config["bandwidth_thresholds_gbps"]
    if not isinstance(bandwidth_thresholds, dict):
        raise ArchitectureGenerationError(
            "bandwidth_thresholds_gbps doit être un objet JSON."
        )

    threshold_names = ("medium", "high", "very_high")
    threshold_values = [
        validate_positive_config_number(
            bandwidth_thresholds,
            field,
            "bandwidth_thresholds_gbps",
        )
        for field in threshold_names
    ]
    if not threshold_values[0] < threshold_values[1] < threshold_values[2]:
        raise ArchitectureGenerationError(
            "Les seuils de bande passante doivent respecter : "
            "medium < high < very_high."
        )


def classify_planned_throughput_requirement(
    required_total_bandwidth_gbps: float,
    config: dict[str, Any],
) -> str:
    """Classe le débit après application de la marge de sécurité OST."""

    thresholds = config["bandwidth_thresholds_gbps"]
    medium = float(thresholds["medium"])
    high = float(thresholds["high"])
    very_high = float(thresholds["very_high"])

    if required_total_bandwidth_gbps < medium:
        return "low"
    if required_total_bandwidth_gbps < high:
        return "medium"
    if required_total_bandwidth_gbps < very_high:
        return "high"
    return "very_high"


def classify_latency_requirement(
    metadata_pressure: str,
    mdt_importance: str,
) -> str:
    """Déduit une classe de latence MDT à partir de la pression metadata."""

    if mdt_importance == "critical" or metadata_pressure == "high":
        return "very_low"
    if mdt_importance == "high" or metadata_pressure == "medium":
        return "low"
    return "moderate"


def classify_endurance_requirement(
    metadata_pressure: str,
    write_percent: float,
    mdt_importance: str,
) -> str:
    """Déduit l'endurance MDT à partir des écritures et de la pression."""

    if (
        mdt_importance == "critical"
        or metadata_pressure == "high"
        or write_percent >= 60
    ):
        return "high"
    if (
        mdt_importance in {"high", "medium"}
        or metadata_pressure == "medium"
        or write_percent >= 40
    ):
        return "medium"
    return "standard"


def classify_reliability_requirement(
    *,
    ha_required: bool,
    importance: str,
    config: dict[str, Any],
) -> str:
    """Déduit le niveau de fiabilité sans choisir la protection."""

    rules = config["reliability_rules"]
    if ha_required:
        return str(rules["ha_required_level"])
    if importance in {"critical", "high"}:
        return "high"
    return str(rules["default_level"])


def file_multiplier(file_size_class: str, config: dict[str, Any]) -> float:
    """Retourne le multiplicateur IOPS lié à la taille moyenne des fichiers."""

    rules = config["mdt_estimation"]
    mapping = {
        "small_files": "small_file_multiplier",
        "medium_files": "medium_file_multiplier",
        "large_files": "large_file_multiplier",
    }
    try:
        return float(rules[mapping[file_size_class]])
    except KeyError as exc:
        raise ArchitectureGenerationError(
            f"file_size_class non supportée : '{file_size_class}'."
        ) from exc


def access_multiplier(access_type: str, config: dict[str, Any]) -> float:
    """Retourne le multiplicateur IOPS lié au pattern d'accès."""

    rules = config["mdt_estimation"]
    normalized = access_type.strip().lower()

    if "random" in normalized:
        return float(rules["random_access_multiplier"])
    if "sequential" in normalized or "sequent" in normalized:
        return float(rules["sequential_access_multiplier"])
    return float(rules["mixed_access_multiplier"])


def metadata_pressure_multiplier(
    metadata_pressure: str,
    config: dict[str, Any],
) -> float:
    """Retourne le multiplicateur IOPS lié à la pression MDT."""

    rules = config["mdt_estimation"]
    mapping = {
        "high": "high_metadata_multiplier",
        "medium": "medium_metadata_multiplier",
        "low": "low_metadata_multiplier",
    }
    try:
        return float(rules[mapping[metadata_pressure]])
    except KeyError as exc:
        raise ArchitectureGenerationError(
            f"metadata_pressure non supportée : '{metadata_pressure}'."
        ) from exc


def generate_mdt_requirement(
    case_id: str,
    mdt_features: dict[str, Any],
    workload_summary: dict[str, Any],
    read_write_ratio: dict[str, Any],
    constraints: dict[str, Any],
    role_analysis: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Génère l'exigence technique MDT et sa trace de calcul."""

    file_count = int(require_number(
        mdt_features,
        "file_count",
        case_id,
        minimum=1,
    ))
    client_count = int(require_number(
        mdt_features,
        "client_count",
        case_id,
        minimum=1,
    ))
    metadata_pressure = require_string(
        mdt_features,
        "metadata_pressure",
        case_id,
    )
    file_size_class = require_string(
        workload_summary,
        "file_size_class",
        case_id,
    )
    access_type = require_string(
        workload_summary,
        "access_type",
        case_id,
    )

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
    if not math.isclose(read_percent + write_percent, 100.0, abs_tol=1e-6):
        raise ArchitectureGenerationError(
            f"{case_id} : read_percent + write_percent doit être égal à 100."
        )

    absolute_importance = require_mapping(
        role_analysis,
        "absolute_importance",
        case_id,
    )
    mdt_importance = require_string(
        absolute_importance,
        "mdt_importance",
        case_id,
    )
    ha_required = require_bool(constraints, "ha_required", case_id)

    rules = config["mdt_estimation"]
    base_iops_per_client = float(rules["base_iops_per_client"])
    size_factor = file_multiplier(file_size_class, config)
    access_factor = access_multiplier(access_type, config)
    pressure_factor = metadata_pressure_multiplier(metadata_pressure, config)
    safety_factor = float(rules["iops_safety_factor"])

    raw_iops = (
        base_iops_per_client
        * client_count
        * size_factor
        * access_factor
        * pressure_factor
    )
    required_total_iops = max(1, math.ceil(raw_iops * safety_factor))
    required_read_iops = int(round(required_total_iops * read_percent / 100.0))
    required_write_iops = required_total_iops - required_read_iops

    metadata_bytes_per_file = float(rules["metadata_bytes_per_file"])
    metadata_capacity_safety_factor = float(
        rules["metadata_capacity_safety_factor"]
    )
    raw_metadata_capacity_bytes = file_count * metadata_bytes_per_file
    required_metadata_capacity_tib = (
        raw_metadata_capacity_bytes
        * metadata_capacity_safety_factor
        / (1024.0 ** 4)
    )

    reliability = classify_reliability_requirement(
        ha_required=ha_required,
        importance=mdt_importance,
        config=config,
    )

    requirement = {
        "priority": mdt_importance,
        "priority_basis": "normalized_metadata_intensity",
        "required_total_iops": required_total_iops,
        "required_read_iops": required_read_iops,
        "required_write_iops": required_write_iops,
        "latency_requirement": classify_latency_requirement(
            metadata_pressure,
            mdt_importance,
        ),
        "required_metadata_capacity_tib": round(
            required_metadata_capacity_tib,
            9,
        ),
        "endurance_requirement": classify_endurance_requirement(
            metadata_pressure,
            write_percent,
            mdt_importance,
        ),
        "reliability_requirement": reliability,
        "ha_required": ha_required,
    }

    trace = {
        "method": "deterministic_proxy_v1",
        "confidence": "medium",
        "base_iops_per_client": base_iops_per_client,
        "client_count": client_count,
        "file_size_multiplier": size_factor,
        "access_multiplier": access_factor,
        "metadata_pressure_multiplier": pressure_factor,
        "iops_safety_factor": safety_factor,
        "metadata_bytes_per_file": metadata_bytes_per_file,
        "metadata_capacity_safety_factor": metadata_capacity_safety_factor,
        "notes": [
            "Les IOPS MDT sont une estimation proxy faute d'un taux "
            "d'opérations metadata fourni par l'utilisateur.",
            "La capacité MDT couvre les métadonnées estimées et une marge "
            "de sécurité ; elle ne fixe pas le nombre de MDT.",
            "La répartition read/write des IOPS MDT réutilise le ratio I/O "
            "utilisateur comme proxy, faute d'un ratio metadata dédié.",
        ],
    }

    return requirement, trace


def generate_ost_requirement(
    case_id: str,
    ost_features: dict[str, Any],
    workload_summary: dict[str, Any],
    constraints: dict[str, Any],
    role_analysis: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Génère l'exigence technique OST et sa trace de calcul."""

    planned_capacity_tib = require_number(
        ost_features,
        "planned_usable_capacity_tib",
        case_id,
        minimum=0,
    )
    target_read_gbps = require_number(
        ost_features,
        "target_read_gbps",
        case_id,
        minimum=0,
    )
    target_write_gbps = require_number(
        ost_features,
        "target_write_gbps",
        case_id,
        minimum=0,
    )

    file_size_class = require_string(
        workload_summary,
        "file_size_class",
        case_id,
    )
    access_type = require_string(
        workload_summary,
        "access_type",
        case_id,
    )
    bandwidth_level = require_string(
        workload_summary,
        "bandwidth_level",
        case_id,
    )
    data_pressure = require_string(
        ost_features,
        "data_pressure",
        case_id,
    )

    absolute_importance = require_mapping(
        role_analysis,
        "absolute_importance",
        case_id,
    )
    ost_importance = require_string(
        absolute_importance,
        "ost_importance",
        case_id,
    )
    ha_required = require_bool(constraints, "ha_required", case_id)

    rules = config["ost_estimation"]
    bandwidth_safety_factor = float(rules["bandwidth_safety_factor"])
    capacity_safety_factor = float(rules["capacity_safety_factor"])

    required_usable_capacity_tib = (
        planned_capacity_tib * capacity_safety_factor
    )
    required_read_bandwidth_gbps = (
        target_read_gbps * bandwidth_safety_factor
    )
    required_write_bandwidth_gbps = (
        target_write_gbps * bandwidth_safety_factor
    )
    required_total_bandwidth_gbps = (
        required_read_bandwidth_gbps
        + required_write_bandwidth_gbps
    )

    reliability = classify_reliability_requirement(
        ha_required=ha_required,
        importance=ost_importance,
        config=config,
    )

    requirement = {
        "priority": ost_importance,
        "priority_basis": "normalized_data_intensity",
        "required_usable_capacity_tib": round(
            required_usable_capacity_tib,
            6,
        ),
        "required_read_bandwidth_gbps": round(
            required_read_bandwidth_gbps,
            6,
        ),
        "required_write_bandwidth_gbps": round(
            required_write_bandwidth_gbps,
            6,
        ),
        "required_total_bandwidth_gbps": round(
            required_total_bandwidth_gbps,
            6,
        ),
        "throughput_requirement": classify_planned_throughput_requirement(
            required_total_bandwidth_gbps,
            config,
        ),
        "input_bandwidth_level": bandwidth_level,
        "data_pressure": data_pressure,
        "access_pattern": access_type,
        "file_size_class": file_size_class,
        "reliability_requirement": reliability,
        "ha_required": ha_required,
    }

    trace = {
        "method": "deterministic_planning_v1",
        "confidence": "high",
        "capacity_basis": "planned_usable_capacity_tib",
        "capacity_safety_factor": capacity_safety_factor,
        "bandwidth_safety_factor": bandwidth_safety_factor,
        "notes": [
            "La capacité OST reste indépendante du RAID ; la capacité brute "
            "sera calculée pendant la génération des groupes de protection.",
            "Les débits lecture et écriture sont conservés séparément.",
        ],
    }

    return requirement, trace


def generate_architecture_case(
    case: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Transforme un cas de features en contrat d'architecture MDT/OST."""

    case_id = case.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ArchitectureGenerationError(
            "Chaque cas doit contenir un case_id non vide."
        )

    workload_summary = require_mapping(
        case,
        "workload_summary",
        case_id,
    )
    scores = require_mapping(case, "scores", case_id)
    role_analysis = require_mapping(
        case,
        "role_analysis",
        case_id,
    )
    mdt_features = require_mapping(case, "mdt_features", case_id)
    ost_features = require_mapping(case, "ost_features", case_id)
    capacity_planning = require_mapping(
        case,
        "capacity_planning",
        case_id,
    )
    read_write_ratio = require_mapping(
        case,
        "read_write_ratio",
        case_id,
    )
    constraints = require_mapping(
        case,
        "constraints",
        case_id,
    )
    preferences = require_mapping(
        case,
        "preferences",
        case_id,
    )
    input_trace = require_mapping(case, "trace", case_id)

    metadata_score = require_number(
        scores,
        "metadata_score",
        case_id,
        minimum=0,
        maximum=1,
    )
    data_score = require_number(
        scores,
        "data_score",
        case_id,
        minimum=0,
        maximum=1,
    )
    score_difference = require_number(
        scores,
        "score_difference",
        case_id,
        minimum=-1,
        maximum=1,
    )
    if not math.isclose(
        score_difference,
        metadata_score - data_score,
        abs_tol=2e-6,
    ):
        raise ArchitectureGenerationError(
            f"{case_id} : score_difference incohérent."
        )

    mdt_requirement, mdt_trace = generate_mdt_requirement(
        case_id,
        mdt_features,
        workload_summary,
        read_write_ratio,
        constraints,
        role_analysis,
        config,
    )
    ost_requirement, ost_trace = generate_ost_requirement(
        case_id,
        ost_features,
        workload_summary,
        constraints,
        role_analysis,
        config,
    )

    return {
        "case_id": case_id,
        "source_requirement": {
            "capacity_planning": capacity_planning,
            "data_characteristics": {
                "total_file_count": int(require_number(
                    mdt_features,
                    "file_count",
                    case_id,
                    minimum=1,
                )),
                "average_file_size_gb": round(require_number(
                    mdt_features,
                    "average_file_size_gb",
                    case_id,
                    minimum=0,
                ), 9),
                "max_file_size_gb": round(require_number(
                    ost_features,
                    "max_file_size_gb",
                    case_id,
                    minimum=0,
                ), 9),
            },
            "io_profile": {
                "client_count": int(require_number(
                    mdt_features,
                    "client_count",
                    case_id,
                    minimum=1,
                )),
                "read_write_ratio": read_write_ratio,
                "access_type": require_string(
                    workload_summary,
                    "access_type",
                    case_id,
                ),
                "target_read_gbps": round(require_number(
                    ost_features,
                    "target_read_gbps",
                    case_id,
                    minimum=0,
                ), 6),
                "target_write_gbps": round(require_number(
                    ost_features,
                    "target_write_gbps",
                    case_id,
                    minimum=0,
                ), 6),
            },
            "constraints": constraints,
            "preferences": preferences,
        },
        "workload_analysis": {
            "workload_type": require_string(
                workload_summary,
                "workload_type",
                case_id,
            ),
            "metadata_score": round(metadata_score, 6),
            "data_score": round(data_score, 6),
            "score_difference": round(score_difference, 6),
            "file_size_class": require_string(
                workload_summary,
                "file_size_class",
                case_id,
            ),
            "access_type": require_string(
                workload_summary,
                "access_type",
                case_id,
            ),
            "parallelism_level": require_string(
                workload_summary,
                "parallelism_level",
                case_id,
            ),
            "bandwidth_level": require_string(
                workload_summary,
                "bandwidth_level",
                case_id,
            ),
        },
        "role_analysis": role_analysis,
        "MDT_requirement": mdt_requirement,
        "OST_requirement": ost_requirement,
        "constraints": constraints,
        "preferences": preferences,
        "trace": {
            "architecture_generator_version": "1.1",
            "rules_version": str(config["version"]),
            "input_feature_calculator_version": str(
                input_trace.get("feature_calculator_version", "unknown")
            ),
            "mdt_estimation": mdt_trace,
            "ost_estimation": ost_trace,
            "forbidden_decisions_not_generated": [
                "drive",
                "raid",
                "target_count",
                "stripe_count",
                "stripe_size",
            ],
        },
    }


def generate_dataset(
    cases: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Génère le dataset architectural et vérifie les doublons."""

    results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for case in tqdm(cases, desc="Génération des architectures"):
        if not isinstance(case, dict):
            raise ArchitectureGenerationError(
                "Chaque entrée du dataset doit être un objet JSON."
            )

        result = generate_architecture_case(case, config)
        case_id = result["case_id"]
        if case_id in seen_ids:
            raise ArchitectureGenerationError(
                f"case_id dupliqué : {case_id}."
            )
        seen_ids.add(case_id)
        results.append(result)

    return results


def print_summary(
    results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Affiche un résumé des exigences générées."""

    mdt_priority_counts = Counter(
        result["MDT_requirement"]["priority"]
        for result in results
    )
    ost_priority_counts = Counter(
        result["OST_requirement"]["priority"]
        for result in results
    )
    latency_counts = Counter(
        result["MDT_requirement"]["latency_requirement"]
        for result in results
    )
    reliability_counts = Counter(
        result["OST_requirement"]["reliability_requirement"]
        for result in results
    )

    total_iops_values = [
        int(result["MDT_requirement"]["required_total_iops"])
        for result in results
    ]
    total_bandwidth_values = [
        float(result["OST_requirement"]["required_total_bandwidth_gbps"])
        for result in results
    ]

    print(f"Cas chargés : {len(results)}")
    print(f"Architectures générées : {len(results)}")

    print("Priorité MDT :")
    for name in ("low", "medium", "high", "critical"):
        print(f"  - {name}: {mdt_priority_counts.get(name, 0)}")

    print("Priorité OST :")
    for name in ("low", "medium", "high", "critical"):
        print(f"  - {name}: {ost_priority_counts.get(name, 0)}")

    print("Latence MDT :")
    for name in ("moderate", "low", "very_low"):
        print(f"  - {name}: {latency_counts.get(name, 0)}")

    print("Fiabilité OST :")
    for name in ("low", "medium", "high", "critical"):
        print(f"  - {name}: {reliability_counts.get(name, 0)}")

    if total_iops_values:
        print(
            "IOPS MDT requis : "
            f"min={min(total_iops_values)}, "
            f"moyenne={sum(total_iops_values) / len(total_iops_values):.2f}, "
            f"max={max(total_iops_values)}"
        )

    if total_bandwidth_values:
        print(
            "Bande passante OST requise : "
            f"min={min(total_bandwidth_values):.2f}, "
            f"moyenne={sum(total_bandwidth_values) / len(total_bandwidth_values):.2f}, "
            f"max={max(total_bandwidth_values):.2f} Gbps"
        )

    print(f"Fichier sauvegardé : {output_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Génère les exigences techniques MDT/OST à partir des features."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_FILE,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_json(args.config)
    cases = load_json(args.input)

    if not isinstance(config, dict):
        raise ArchitectureGenerationError(
            "architecture_rules.json doit contenir un objet JSON."
        )
    if not isinstance(cases, list):
        raise ArchitectureGenerationError(
            "Le dataset de features doit contenir une liste JSON."
        )

    validate_config(config)
    results = generate_dataset(cases, config)
    save_json(results, args.output)
    print_summary(results, args.output)


if __name__ == "__main__":
    try:
        main()
    except (ArchitectureGenerationError, FileNotFoundError) as exc:
        print(f"Erreur : {exc}")
        raise SystemExit(1) from exc
