"""Analyse déterministe des workloads Lustre.

Ce module transforme chaque besoin utilisateur normalisé en :
- facteurs normalisés indépendants du dataset ;
- metadata_score et data_score ;
- classification metadata_heavy / data_heavy / balanced ;
- capacité planifiée ;
- contrat enrichi conservant contraintes et préférences.

Important :
- metadata_heavy ne signifie pas que les métadonnées occupent plus de volume ;
- la classification représente la pression ou le goulot d'étranglement dominant ;
- ce module ne choisit aucun disque, RAID, nombre de targets ou paramètre de striping.
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
except ImportError:  # Le script reste utilisable sans tqdm.

    def tqdm(iterable: Any, **_: Any) -> Any:  # type: ignore[misc]
        return iterable


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_FILE = BASE_DIR / "data" / "use_cases_lustre_1200_v4.json"
DEFAULT_CONFIG_FILE = BASE_DIR / "config" / "architecture_rules.json"
DEFAULT_OUTPUT_FILE = BASE_DIR / "output" / "workload_analysis_dataset.json"


class WorkloadAnalysisError(ValueError):
    """Erreur de validation ou de calcul du workload."""


def load_json(path: Path) -> Any:
    """Charge un fichier JSON avec des erreurs explicites."""

    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise WorkloadAnalysisError(
            f"JSON invalide dans {path} : ligne {exc.lineno}, colonne {exc.colno}"
        ) from exc


def require_number(
    data: dict[str, Any],
    field: str,
    *,
    minimum: float | None = None,
) -> float:
    """Lit et valide un champ numérique."""

    value = data.get(field)

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkloadAnalysisError(
            f"{data.get('case_id', '<case inconnu>')} : "
            f"'{field}' doit être numérique."
        )

    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise WorkloadAnalysisError(
            f"{data.get('case_id', '<case inconnu>')} : "
            f"'{field}' doit être fini."
        )

    if minimum is not None and numeric_value < minimum:
        raise WorkloadAnalysisError(
            f"{data.get('case_id', '<case inconnu>')} : "
            f"'{field}' doit être >= {minimum}."
        )

    return numeric_value


def require_boolean(data: dict[str, Any], field: str) -> bool:
    """Lit et valide un champ booléen."""

    value = data.get(field)
    if not isinstance(value, bool):
        raise WorkloadAnalysisError(
            f"{data.get('case_id', '<case inconnu>')} : "
            f"'{field}' doit être booléen."
        )
    return value


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Force une valeur dans un intervalle fermé."""

    return max(minimum, min(value, maximum))


def normalize_with_rule(value: float, rule: dict[str, Any], field: str) -> float:
    """Normalise avec des bornes métier fixes, indépendantes du dataset."""

    minimum = float(rule["minimum"])
    maximum = float(rule["maximum"])
    scale = str(rule.get("scale", "linear")).lower()

    if maximum <= minimum:
        raise WorkloadAnalysisError(
            f"Configuration invalide pour '{field}' : "
            "maximum doit être > minimum."
        )

    bounded_value = clamp(value, minimum, maximum)

    if scale == "log":
        if minimum <= 0:
            raise WorkloadAnalysisError(
                f"Configuration invalide pour '{field}' : "
                "minimum doit être > 0 avec scale='log'."
            )
        score = (
            math.log10(bounded_value) - math.log10(minimum)
        ) / (
            math.log10(maximum) - math.log10(minimum)
        )
    elif scale == "linear":
        score = (bounded_value - minimum) / (maximum - minimum)
    else:
        raise WorkloadAnalysisError(
            f"Configuration invalide pour '{field}' : "
            f"scale '{scale}' non supportée."
        )

    return clamp(score, 0.0, 1.0)


def calculate_file_size_factors(
    average_file_size_gb: float,
    config: dict[str, Any],
) -> tuple[float, float]:
    """Calcule les pressions relatives associées à la taille moyenne.

    Les deux facteurs sont continus :
    - les petits fichiers renforcent la pression metadata ;
    - les fichiers moyens constituent une zone de transition ;
    - les gros fichiers renforcent la pression data.

    Ils ne représentent pas des probabilités et leur somme n'est donc pas
    obligatoirement égale à 1.
    """

    thresholds = config["file_size_thresholds_gb"]
    small_max = float(thresholds["small_max"])
    medium_max = float(thresholds["medium_max"])
    maximum_size = float(
        config["normalization"]["average_file_size_gb"]["maximum"]
    )

    size = clamp(average_file_size_gb, 0.0, maximum_size)

    if size <= small_max:
        progress = size / small_max
        small_file_factor = 1.0 - 0.5 * progress
        large_file_factor = 0.0
    elif size < medium_max:
        progress = (size - small_max) / (medium_max - small_max)
        small_file_factor = 0.5 * (1.0 - progress)
        large_file_factor = 0.5 * progress
    else:
        progress = (size - medium_max) / (maximum_size - medium_max)
        small_file_factor = 0.0
        large_file_factor = 0.5 + 0.5 * progress

    return (
        clamp(small_file_factor, 0.0, 1.0),
        clamp(large_file_factor, 0.0, 1.0),
    )


def validate_weights(weights: dict[str, Any], group_name: str) -> None:
    """Vérifie que les poids sont numériques, positifs et totalisent 1."""

    total = 0.0
    for name, value in weights.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise WorkloadAnalysisError(
                f"Poids invalide '{group_name}.{name}' : "
                "valeur numérique attendue."
            )
        if value < 0:
            raise WorkloadAnalysisError(
                f"Poids invalide '{group_name}.{name}' : valeur négative."
            )
        total += float(value)

    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise WorkloadAnalysisError(
            f"La somme des poids '{group_name}' doit être égale à 1.0, "
            f"valeur actuelle : {total}."
        )


def validate_config(config: dict[str, Any]) -> None:
    """Valide toutes les sections utilisées par l'analyseur."""

    required_sections = {
        "version",
        "capacity_planning",
        "normalization",
        "score_weights",
        "workload_classification",
        "file_size_thresholds_gb",
    }
    missing = sorted(required_sections - config.keys())
    if missing:
        raise WorkloadAnalysisError(
            "Sections manquantes dans architecture_rules.json : "
            + ", ".join(missing)
        )

    normalization = config["normalization"]
    required_normalization_fields = (
        "file_count",
        "average_file_size_gb",
        "client_count",
        "capacity_tib",
        "read_gbps",
        "write_gbps",
    )

    for field in required_normalization_fields:
        if field not in normalization:
            raise WorkloadAnalysisError(
                f"Règle de normalisation manquante : '{field}'."
            )

        rule = normalization[field]
        if not isinstance(rule, dict):
            raise WorkloadAnalysisError(
                f"La règle de normalisation '{field}' doit être un objet."
            )

        for key in ("minimum", "maximum"):
            if key not in rule:
                raise WorkloadAnalysisError(
                    f"Clé manquante : normalization.{field}.{key}."
                )

        normalize_with_rule(
            float(rule["minimum"]),
            rule,
            field,
        )

    metadata_weights = config["score_weights"].get("metadata")
    data_weights = config["score_weights"].get("data")

    if not isinstance(metadata_weights, dict) or not isinstance(data_weights, dict):
        raise WorkloadAnalysisError(
            "score_weights doit contenir les objets 'metadata' et 'data'."
        )

    expected_metadata_weights = {
        "file_count",
        "small_file_factor",
        "client_count",
    }
    expected_data_weights = {
        "capacity",
        "bandwidth",
        "large_file_factor",
    }

    if set(metadata_weights) != expected_metadata_weights:
        raise WorkloadAnalysisError(
            "score_weights.metadata doit contenir exactement : "
            + ", ".join(sorted(expected_metadata_weights))
        )

    if set(data_weights) != expected_data_weights:
        raise WorkloadAnalysisError(
            "score_weights.data doit contenir exactement : "
            + ", ".join(sorted(expected_data_weights))
        )

    validate_weights(metadata_weights, "score_weights.metadata")
    validate_weights(data_weights, "score_weights.data")

    capacity_rules = config["capacity_planning"]
    fill_ratio = float(capacity_rules["default_target_fill_ratio"])
    fill_ratio_min = float(capacity_rules["minimum_target_fill_ratio"])
    fill_ratio_max = float(capacity_rules["maximum_target_fill_ratio"])

    if not (0 < fill_ratio_min <= fill_ratio <= fill_ratio_max <= 1):
        raise WorkloadAnalysisError(
            "Les valeurs de capacity_planning doivent respecter "
            "0 < minimum <= default <= maximum <= 1."
        )

    if "legacy_default_planning_horizon_years" in capacity_rules:
        raise WorkloadAnalysisError(
            "Configuration obsolète : "
            "capacity_planning.legacy_default_planning_horizon_years "
            "a été supprimé au freeze S10. "
            "planning_horizon_years doit être fourni explicitement "
            "dans chaque besoin utilisateur."
        )

    margin = float(config["workload_classification"]["dominance_margin"])
    if not 0 <= margin <= 1:
        raise WorkloadAnalysisError(
            "workload_classification.dominance_margin doit être compris "
            "entre 0 et 1."
        )

    file_size_thresholds = config["file_size_thresholds_gb"]
    small_max = float(file_size_thresholds["small_max"])
    medium_max = float(file_size_thresholds["medium_max"])
    maximum_size = float(normalization["average_file_size_gb"]["maximum"])

    if not 0 < small_max < medium_max < maximum_size:
        raise WorkloadAnalysisError(
            "Les seuils de taille doivent respecter : "
            "0 < small_max < medium_max < maximum normalisé."
        )


def extract_read_write_ratio(case: dict[str, Any]) -> tuple[float, float]:
    """Extrait et valide le ratio lecture/écriture."""

    ratio = case.get("read_write_ratio")
    if not isinstance(ratio, dict):
        raise WorkloadAnalysisError(
            f"{case.get('case_id', '<case inconnu>')} : "
            "'read_write_ratio' doit être un objet."
        )

    read_percent = require_number(ratio, "read_percent", minimum=0)
    write_percent = require_number(ratio, "write_percent", minimum=0)

    if not math.isclose(read_percent + write_percent, 100.0, abs_tol=0.5):
        raise WorkloadAnalysisError(
            f"{case.get('case_id', '<case inconnu>')} : "
            "read_percent + write_percent doit être proche de 100, "
            f"valeur actuelle : {read_percent + write_percent}."
        )

    return read_percent, write_percent


def extract_planning_horizon_years(
    case: dict[str, Any],
) -> tuple[float, str]:
    """Retourne l'horizon explicite de planification et sa provenance.

    Depuis le freeze S10, ``planning_horizon_years`` fait partie du contrat
    d'entrée obligatoire. Aucun horizon par défaut n'est appliqué : une
    absence doit être corrigée en amont plutôt que masquée dans le sizing.
    """

    if "planning_horizon_years" not in case:
        raise WorkloadAnalysisError(
            f"{case.get('case_id', '<case inconnu>')} : "
            "'planning_horizon_years' est obligatoire depuis le freeze S10 ; "
            "aucun fallback n'est autorisé."
        )

    horizon = require_number(
        case,
        "planning_horizon_years",
        minimum=0.0,
    )
    if horizon <= 0:
        raise WorkloadAnalysisError(
            f"{case.get('case_id', '<case inconnu>')} : "
            "'planning_horizon_years' doit être > 0."
        )
    return horizon, "input"


def classify_workload(
    metadata_score: float,
    data_score: float,
    dominance_margin: float,
) -> tuple[str, float]:
    """Classe le goulot d'étranglement dominant à partir des deux scores."""

    score_difference = metadata_score - data_score

    if score_difference >= dominance_margin:
        workload_type = "metadata_heavy"
    elif score_difference <= -dominance_margin:
        workload_type = "data_heavy"
    else:
        workload_type = "balanced"

    return workload_type, score_difference


def analyze_workload(case: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Analyse un cas utilisateur et retourne un contrat enrichi."""

    case_id = case.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise WorkloadAnalysisError(
            "Chaque cas doit contenir un 'case_id' non vide."
        )

    file_count = require_number(case, "total_file_count", minimum=1)
    average_file_size_gb = require_number(
        case,
        "average_file_size_gb",
        minimum=0,
    )
    max_file_size_gb = require_number(case, "max_file_size_gb", minimum=0)
    client_count = require_number(case, "client_count", minimum=1)
    requested_capacity_tib = require_number(
        case,
        "requested_usable_capacity_tib",
        minimum=0,
    )
    target_read_gbps = require_number(case, "target_read_gbps", minimum=0)
    target_write_gbps = require_number(case, "target_write_gbps", minimum=0)
    annual_growth_percent = require_number(
        case,
        "annual_growth_percent",
        minimum=0,
    )
    planning_horizon_years, planning_horizon_source = (
        extract_planning_horizon_years(case)
    )
    max_budget_usd = require_number(case, "max_budget_usd", minimum=0)
    max_power_w = require_number(case, "max_power_w", minimum=0)
    ha_required = require_boolean(case, "ha_required")
    read_percent, write_percent = extract_read_write_ratio(case)

    if max_file_size_gb < average_file_size_gb:
        raise WorkloadAnalysisError(
            f"{case_id} : max_file_size_gb doit être "
            ">= average_file_size_gb."
        )

    access_type = case.get("access_type")
    if not isinstance(access_type, str) or not access_type.strip():
        raise WorkloadAnalysisError(
            f"{case_id} : 'access_type' doit être une chaîne non vide."
        )

    normalized_access_type = access_type.strip().lower()
    allowed_access_types = {"random", "mixed", "sequential"}
    if normalized_access_type not in allowed_access_types:
        raise WorkloadAnalysisError(
            f"{case_id} : access_type '{access_type}' non supporté. "
            f"Valeurs autorisées : {sorted(allowed_access_types)}."
        )

    preferences: dict[str, float] = {}
    for field in (
        "performance_priority",
        "cost_priority",
        "power_priority",
        "reliability_priority",
    ):
        preferences[field] = require_number(case, field, minimum=0)

    preference_sum = sum(preferences.values())
    if not math.isclose(preference_sum, 1.0, abs_tol=0.01):
        raise WorkloadAnalysisError(
            f"{case_id} : la somme des priorités doit être proche de 1.0, "
            f"valeur actuelle : {preference_sum:.6f}."
        )

    # Capacity planning multi-années :
    # C_planned = C_requested * (1 + g)^n / target_fill_ratio
    capacity_rules = config["capacity_planning"]
    target_fill_ratio = float(capacity_rules["default_target_fill_ratio"])
    annual_growth_factor = 1.0 + annual_growth_percent / 100.0
    growth_factor = annual_growth_factor ** planning_horizon_years
    planned_usable_capacity_tib = (
        requested_capacity_tib * growth_factor / target_fill_ratio
    )

    normalization_rules = config["normalization"]

    file_count_score = normalize_with_rule(
        file_count,
        normalization_rules["file_count"],
        "file_count",
    )
    average_file_size_score = normalize_with_rule(
        average_file_size_gb,
        normalization_rules["average_file_size_gb"],
        "average_file_size_gb",
    )
    client_count_score = normalize_with_rule(
        client_count,
        normalization_rules["client_count"],
        "client_count",
    )
    capacity_score = normalize_with_rule(
        planned_usable_capacity_tib,
        normalization_rules["capacity_tib"],
        "capacity_tib",
    )
    read_score = normalize_with_rule(
        target_read_gbps,
        normalization_rules["read_gbps"],
        "read_gbps",
    )
    write_score = normalize_with_rule(
        target_write_gbps,
        normalization_rules["write_gbps"],
        "write_gbps",
    )

    small_file_factor, large_file_factor = calculate_file_size_factors(
        average_file_size_gb,
        config,
    )
    bandwidth_score = (read_score + write_score) / 2.0

    metadata_weights = config["score_weights"]["metadata"]
    metadata_score = (
        float(metadata_weights["file_count"]) * file_count_score
        + float(metadata_weights["small_file_factor"]) * small_file_factor
        + float(metadata_weights["client_count"]) * client_count_score
    )

    data_weights = config["score_weights"]["data"]
    data_score = (
        float(data_weights["capacity"]) * capacity_score
        + float(data_weights["bandwidth"]) * bandwidth_score
        + float(data_weights["large_file_factor"]) * large_file_factor
    )

    metadata_score = clamp(metadata_score, 0.0, 1.0)
    data_score = clamp(data_score, 0.0, 1.0)

    dominance_margin = float(
        config["workload_classification"]["dominance_margin"]
    )
    workload_type, score_difference = classify_workload(
        metadata_score,
        data_score,
        dominance_margin,
    )

    total_bandwidth_gbps = target_read_gbps + target_write_gbps

    return {
        "case_id": case_id,
        "source_requirement": {
            "requested_usable_capacity_tib": round(
                requested_capacity_tib,
                6,
            ),
            "client_count": int(client_count),
            "average_file_size_gb": round(average_file_size_gb, 9),
            "max_file_size_gb": round(max_file_size_gb, 9),
            "total_file_count": int(file_count),
            "read_write_ratio": {
                "read_percent": round(read_percent, 6),
                "write_percent": round(write_percent, 6),
            },
            "access_type": normalized_access_type,
            "target_read_gbps": round(target_read_gbps, 6),
            "target_write_gbps": round(target_write_gbps, 6),
            "ha_required": ha_required,
            "annual_growth_percent": round(annual_growth_percent, 6),
            "planning_horizon_years": round(planning_horizon_years, 6),
        },
        "capacity_planning": {
            "requested_usable_capacity_tib": round(
                requested_capacity_tib,
                6,
            ),
            "annual_growth_percent": round(annual_growth_percent, 6),
            "planning_horizon_years": round(planning_horizon_years, 6),
            "annual_growth_factor": round(annual_growth_factor, 9),
            "growth_factor": round(growth_factor, 9),
            "target_fill_ratio": round(target_fill_ratio, 6),
            "planned_usable_capacity_tib": round(
                planned_usable_capacity_tib,
                6,
            ),
        },
        "normalized_factors": {
            "file_count_score": round(file_count_score, 6),
            "average_file_size_score": round(
                average_file_size_score,
                6,
            ),
            "small_file_factor": round(small_file_factor, 6),
            "client_count_score": round(client_count_score, 6),
            "capacity_score": round(capacity_score, 6),
            "read_bandwidth_score": round(read_score, 6),
            "write_bandwidth_score": round(write_score, 6),
            "bandwidth_score": round(bandwidth_score, 6),
            "large_file_factor": round(large_file_factor, 6),
        },
        "scores": {
            "metadata_score": round(metadata_score, 6),
            "data_score": round(data_score, 6),
            "score_difference": round(score_difference, 6),
        },
        "workload_type": workload_type,
        "workload_semantics": {
            "classification_basis": "dominant_pressure_not_storage_volume",
            "metadata_heavy_meaning": (
                "La pression des opérations de métadonnées est dominante."
            ),
            "data_heavy_meaning": (
                "La pression de capacité ou de débit des données est dominante."
            ),
        },
        "metadata_indicators": {
            "file_count": int(file_count),
            "average_file_size_gb": round(average_file_size_gb, 9),
            "client_count": int(client_count),
        },
        "data_indicators": {
            "requested_capacity_tib": round(requested_capacity_tib, 6),
            "planned_usable_capacity_tib": round(
                planned_usable_capacity_tib,
                6,
            ),
            "target_read_gbps": round(target_read_gbps, 6),
            "target_write_gbps": round(target_write_gbps, 6),
            "total_bandwidth_gbps": round(total_bandwidth_gbps, 6),
            "average_file_size_gb": round(average_file_size_gb, 9),
            "max_file_size_gb": round(max_file_size_gb, 9),
            "access_type": normalized_access_type,
        },
        "constraints": {
            "ha_required": ha_required,
            "max_budget_usd": round(max_budget_usd, 6),
            "max_power_w": round(max_power_w, 6),
        },
        "preferences": {
            key: round(value, 6) for key, value in preferences.items()
        },
        "trace": {
            "analyzer_version": "3.0",
            "rules_version": str(config["version"]),
            "normalization_scope": "fixed_business_bounds",
            "capacity_score_basis": "planned_usable_capacity_tib",
            "capacity_formula": (
                "requested_capacity*(1+annual_growth)^planning_horizon/"
                "target_fill_ratio"
            ),
            "planning_horizon_source": planning_horizon_source,
            "file_size_factor_method": "piecewise_continuous_v1",
        },
    }


def generate_dataset(
    use_cases: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Analyse l'ensemble du dataset et interdit les case_id dupliqués."""

    if not isinstance(use_cases, list):
        raise WorkloadAnalysisError(
            "Le fichier d'entrée doit contenir une liste JSON."
        )

    results: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()

    for case in tqdm(use_cases, desc="Analyse des workloads"):
        if not isinstance(case, dict):
            raise WorkloadAnalysisError(
                "Chaque élément du dataset doit être un objet JSON."
            )

        result = analyze_workload(case, config)
        case_id = result["case_id"]

        if case_id in seen_case_ids:
            raise WorkloadAnalysisError(f"case_id dupliqué : {case_id}")

        seen_case_ids.add(case_id)
        results.append(result)

    return results


def parse_args() -> argparse.Namespace:
    """Déclare les arguments CLI."""

    parser = argparse.ArgumentParser(
        description="Génère le dataset d'analyse des workloads Lustre."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help=f"Dataset d'entrée (défaut : {DEFAULT_INPUT_FILE})",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_FILE,
        help=f"Règles d'architecture (défaut : {DEFAULT_CONFIG_FILE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Fichier de sortie (défaut : {DEFAULT_OUTPUT_FILE})",
    )
    return parser.parse_args()


def print_summary(
    use_cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Affiche le résumé de génération."""

    workload_counts = Counter(item["workload_type"] for item in results)
    metadata_scores = [item["scores"]["metadata_score"] for item in results]
    data_scores = [item["scores"]["data_score"] for item in results]

    print(f"Cas chargés : {len(use_cases)}")
    print(f"Cas générés : {len(results)}")
    print("Distribution :")

    for workload_type in ("metadata_heavy", "balanced", "data_heavy"):
        count = workload_counts.get(workload_type, 0)
        percentage = (count / len(results) * 100.0) if results else 0.0
        print(f"  - {workload_type}: {count} ({percentage:.2f} %)")

    if results:
        print(
            "Metadata score : "
            f"min={min(metadata_scores):.4f}, "
            f"moyenne={sum(metadata_scores) / len(metadata_scores):.4f}, "
            f"max={max(metadata_scores):.4f}"
        )
        print(
            "Data score : "
            f"min={min(data_scores):.4f}, "
            f"moyenne={sum(data_scores) / len(data_scores):.4f}, "
            f"max={max(data_scores):.4f}"
        )

    print(f"Fichier sauvegardé : {output_path}")


def main() -> None:
    """Point d'entrée CLI."""

    args = parse_args()

    config = load_json(args.config)
    if not isinstance(config, dict):
        raise WorkloadAnalysisError(
            "Le fichier de configuration doit contenir un objet JSON."
        )
    validate_config(config)

    use_cases = load_json(args.input)
    if not isinstance(use_cases, list):
        raise WorkloadAnalysisError(
            "Le fichier d'entrée doit contenir une liste JSON."
        )

    results = generate_dataset(use_cases, config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print_summary(use_cases, results, args.output)


if __name__ == "__main__":
    main()