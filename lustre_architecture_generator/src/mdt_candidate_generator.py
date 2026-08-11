r"""Générateur déterministe de candidats de disques MDT.

Entrées :
- output/lustre_architecture_dataset.json
- data/catalogue_drives_ready_final.json

Sortie :
- output/mdt_drive_candidates_dataset.json

Le générateur :
- conserve case_id, MDT_requirement, constraints et preferences ;
- évalue uniquement les disques avec mdt_eligible=true ;
- vérifie latence, endurance et fiabilité ;
- calcule une borne minimale avant RAID selon capacité et IOPS ;
- élimine les candidats dont la borne minimale dépasse déjà le budget
  ou la puissance globale ;
- classe les candidats faisables et conserve le Top-K ;
- ne choisit ni RAID, ni nombre final de disques, ni MDT count,
  ni serveur, ni striping.

Exécution depuis la racine du projet :

    python .\src\mdt_candidate_generator.py

Ou avec des chemins explicites :

    python .\src\mdt_candidate_generator.py ^
        --input .\output\lustre_architecture_dataset.json ^
        --catalog .\data\catalogue_drives_ready_final.json ^
        --output .\output\mdt_drive_candidates_dataset.json ^
        --top-k 10
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = BASE_DIR / "output" / "lustre_architecture_dataset.json"
DEFAULT_CATALOG = BASE_DIR / "data" / "catalogue_drives_ready_final.json"
DEFAULT_OUTPUT = BASE_DIR / "output" / "mdt_drive_candidates_dataset.json"


GENERATOR_VERSION = "1.0"

LATENCY_ORDER = {
    "moderate": 0,
    "low": 1,
    "very_low": 2,
}

ENDURANCE_MIN_DWPD = {
    "standard": 0.3,
    "medium": 1.0,
    "high": 3.0,
    "critical": 10.0,
}

RELIABILITY_MIN_MTBF_HOURS = {
    "low": 0,
    "medium": 1_500_000,
    "high": 2_000_000,
    "critical": 2_500_000,
}

PREFERENCE_FIELDS = (
    "performance_priority",
    "cost_priority",
    "power_priority",
    "reliability_priority",
)


class MDTCandidateGenerationError(ValueError):
    """Erreur de contrat ou de génération des candidats MDT."""


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")


def require_dict(
    parent: dict[str, Any],
    key: str,
    context: str,
) -> dict[str, Any]:
    value = parent.get(key)

    if not isinstance(value, dict):
        raise MDTCandidateGenerationError(
            f"{context}.{key}: objet JSON requis."
        )

    return value


def require_string(
    parent: dict[str, Any],
    key: str,
    context: str,
) -> str:
    value = parent.get(key)

    if not isinstance(value, str) or not value.strip():
        raise MDTCandidateGenerationError(
            f"{context}.{key}: chaîne non vide requise."
        )

    return value


def require_bool(
    parent: dict[str, Any],
    key: str,
    context: str,
) -> bool:
    value = parent.get(key)

    if not isinstance(value, bool):
        raise MDTCandidateGenerationError(
            f"{context}.{key}: booléen requis."
        )

    return value


def require_number(
    parent: dict[str, Any],
    key: str,
    context: str,
    *,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    value = parent.get(key)

    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise MDTCandidateGenerationError(
            f"{context}.{key}: nombre fini requis."
        )

    number = float(value)

    if strictly_positive and number <= 0:
        raise MDTCandidateGenerationError(
            f"{context}.{key}: valeur strictement positive requise."
        )

    if minimum is not None and number < minimum:
        raise MDTCandidateGenerationError(
            f"{context}.{key}: minimum autorisé={minimum}, "
            f"obtenu={number}."
        )

    return number


def ceil_ratio(required: float, available_per_drive: float) -> int:
    if available_per_drive <= 0:
        raise MDTCandidateGenerationError(
            "La ressource disponible par disque doit être positive."
        )

    if required <= 0:
        return 0

    return int(math.ceil(required / available_per_drive))


def validate_preferences(
    preferences: dict[str, Any],
    case_id: str,
) -> dict[str, float]:
    validated: dict[str, float] = {}

    for field in PREFERENCE_FIELDS:
        validated[field] = require_number(
            preferences,
            field,
            f"{case_id}.preferences",
            minimum=0,
        )

    total = sum(validated.values())

    if not math.isclose(total, 1.0, abs_tol=0.01):
        raise MDTCandidateGenerationError(
            f"{case_id}: la somme des préférences doit être proche de 1, "
            f"obtenu={total}."
        )

    return validated


def validate_architecture_case(
    case: dict[str, Any],
) -> tuple[
    str,
    dict[str, Any],
    dict[str, Any],
    dict[str, float],
]:
    case_id = require_string(case, "case_id", "$")
    mdt = require_dict(case, "MDT_requirement", case_id)
    constraints = require_dict(case, "constraints", case_id)
    preferences_raw = require_dict(case, "preferences", case_id)

    require_number(
        mdt,
        "required_total_iops",
        f"{case_id}.MDT_requirement",
        strictly_positive=True,
    )
    require_number(
        mdt,
        "required_read_iops",
        f"{case_id}.MDT_requirement",
        minimum=0,
    )
    require_number(
        mdt,
        "required_write_iops",
        f"{case_id}.MDT_requirement",
        minimum=0,
    )
    require_number(
        mdt,
        "required_metadata_capacity_tib",
        f"{case_id}.MDT_requirement",
        minimum=0,
    )

    latency_requirement = require_string(
        mdt,
        "latency_requirement",
        f"{case_id}.MDT_requirement",
    )
    endurance_requirement = require_string(
        mdt,
        "endurance_requirement",
        f"{case_id}.MDT_requirement",
    )
    reliability_requirement = require_string(
        mdt,
        "reliability_requirement",
        f"{case_id}.MDT_requirement",
    )

    if latency_requirement not in LATENCY_ORDER:
        raise MDTCandidateGenerationError(
            f"{case_id}: latency_requirement non supporté="
            f"{latency_requirement!r}."
        )

    if endurance_requirement not in ENDURANCE_MIN_DWPD:
        raise MDTCandidateGenerationError(
            f"{case_id}: endurance_requirement non supporté="
            f"{endurance_requirement!r}."
        )

    if reliability_requirement not in RELIABILITY_MIN_MTBF_HOURS:
        raise MDTCandidateGenerationError(
            f"{case_id}: reliability_requirement non supporté="
            f"{reliability_requirement!r}."
        )

    require_bool(
        constraints,
        "ha_required",
        f"{case_id}.constraints",
    )
    require_number(
        constraints,
        "max_budget_usd",
        f"{case_id}.constraints",
        strictly_positive=True,
    )
    require_number(
        constraints,
        "max_power_w",
        f"{case_id}.constraints",
        strictly_positive=True,
    )

    preferences = validate_preferences(preferences_raw, case_id)

    return case_id, mdt, constraints, preferences


def validate_drive(
    drive: dict[str, Any],
    index: int,
) -> None:
    context = f"catalog[{index}]"

    require_string(drive, "drive_id", context)
    require_string(drive, "name", context)
    require_string(drive, "manufacturer", context)
    require_string(drive, "series", context)
    require_string(drive, "media_type", context)
    require_string(drive, "protocol", context)
    require_string(drive, "drive_form_factor_standard", context)
    require_string(drive, "latency_class", context)
    require_bool(drive, "mdt_eligible", context)

    for field in (
        "capacity_tib",
        "random_read_iops_4k",
        "random_write_iops_4k",
        "mtbf_hours",
        "price_en_dollars",
        "power_consumption_en_w",
    ):
        require_number(
            drive,
            field,
            context,
            strictly_positive=True,
        )

    if bool(drive["mdt_eligible"]):
        require_number(
            drive,
            "endurance_dwpd_numeric",
            context,
            strictly_positive=True,
        )

    if drive["latency_class"] not in LATENCY_ORDER:
        raise MDTCandidateGenerationError(
            f"{context}.latency_class non supportée="
            f"{drive['latency_class']!r}."
        )


def validate_catalog(
    catalog: list[dict[str, Any]],
) -> None:
    if not catalog:
        raise MDTCandidateGenerationError(
            "Le catalogue ne doit pas être vide."
        )

    seen_ids: set[str] = set()

    for index, drive in enumerate(catalog):
        if not isinstance(drive, dict):
            raise MDTCandidateGenerationError(
                f"catalog[{index}]: objet JSON requis."
            )

        validate_drive(drive, index)
        drive_id = drive["drive_id"]

        if drive_id in seen_ids:
            raise MDTCandidateGenerationError(
                f"drive_id dupliqué dans le catalogue : {drive_id}."
            )

        seen_ids.add(drive_id)


def latency_is_compatible(
    candidate_class: str,
    required_class: str,
) -> bool:
    return (
        LATENCY_ORDER[candidate_class]
        >= LATENCY_ORDER[required_class]
    )


def latency_fit(
    candidate_class: str,
    required_class: str,
) -> float:
    difference = (
        LATENCY_ORDER[candidate_class]
        - LATENCY_ORDER[required_class]
    )

    if difference < 0:
        return 0.0

    return min(1.0, 0.85 + 0.075 * difference)


def headroom_fit(
    provided: float,
    required: float,
) -> float:
    if required <= 0:
        return 1.0

    ratio = provided / required

    if ratio < 1:
        return 0.0

    return min(1.0, 0.70 + 0.30 * min(ratio - 1.0, 1.0))


def endurance_fit(
    actual_dwpd: float,
    minimum_dwpd: float,
) -> float:
    if actual_dwpd < minimum_dwpd:
        return 0.0

    ratio = actual_dwpd / minimum_dwpd
    return min(1.0, 0.75 + 0.25 * min(ratio - 1.0, 1.0))


def reliability_fit(
    actual_mtbf: float,
    minimum_mtbf: float,
) -> float:
    if actual_mtbf < minimum_mtbf:
        return 0.0

    if minimum_mtbf <= 0:
        return min(1.0, actual_mtbf / 2_500_000.0)

    extra_ratio = (actual_mtbf - minimum_mtbf) / max(
        minimum_mtbf,
        1.0,
    )

    return min(1.0, 0.80 + 0.20 * min(extra_ratio, 1.0))


def lower_is_better_fit(
    used: float,
    maximum: float,
) -> float:
    if maximum <= 0:
        return 0.0

    ratio = used / maximum
    return max(0.0, 1.0 - min(ratio, 1.0))


def evaluate_drive(
    drive: dict[str, Any],
    mdt: dict[str, Any],
    constraints: dict[str, Any],
    preferences: dict[str, float],
) -> tuple[dict[str, Any], list[str]]:
    required_capacity = float(
        mdt["required_metadata_capacity_tib"]
    )
    required_read_iops = float(mdt["required_read_iops"])
    required_write_iops = float(mdt["required_write_iops"])

    drive_capacity = float(drive["capacity_tib"])
    drive_read_iops = float(drive["random_read_iops_4k"])
    drive_write_iops = float(drive["random_write_iops_4k"])
    drive_dwpd = float(drive["endurance_dwpd_numeric"])
    drive_mtbf = float(drive["mtbf_hours"])
    drive_price = float(drive["price_en_dollars"])
    drive_power = float(drive["power_consumption_en_w"])

    count_by_capacity = ceil_ratio(
        required_capacity,
        drive_capacity,
    )
    count_by_read_iops = ceil_ratio(
        required_read_iops,
        drive_read_iops,
    )
    count_by_write_iops = ceil_ratio(
        required_write_iops,
        drive_write_iops,
    )

    raw_minimum_count = max(
        1,
        count_by_capacity,
        count_by_read_iops,
        count_by_write_iops,
    )

    provided_capacity = raw_minimum_count * drive_capacity
    provided_read_iops = raw_minimum_count * drive_read_iops
    provided_write_iops = raw_minimum_count * drive_write_iops

    raw_cost = raw_minimum_count * drive_price
    raw_power = raw_minimum_count * drive_power

    required_latency = str(mdt["latency_requirement"])
    required_endurance = str(mdt["endurance_requirement"])
    required_reliability = str(mdt["reliability_requirement"])

    minimum_dwpd = ENDURANCE_MIN_DWPD[required_endurance]
    minimum_mtbf = RELIABILITY_MIN_MTBF_HOURS[
        required_reliability
    ]

    rejection_reasons: list[str] = []

    if not bool(drive["mdt_eligible"]):
        rejection_reasons.append("not_mdt_eligible")

    if not latency_is_compatible(
        str(drive["latency_class"]),
        required_latency,
    ):
        rejection_reasons.append("latency_requirement_not_met")

    if drive_dwpd < minimum_dwpd:
        rejection_reasons.append("endurance_requirement_not_met")

    if drive_mtbf < minimum_mtbf:
        rejection_reasons.append("reliability_requirement_not_met")

    max_budget = float(constraints["max_budget_usd"])
    max_power = float(constraints["max_power_w"])

    if raw_cost > max_budget:
        rejection_reasons.append(
            "raw_mdt_cost_exceeds_global_budget"
        )

    if raw_power > max_power:
        rejection_reasons.append(
            "raw_mdt_power_exceeds_global_power_limit"
        )

    count_fit = 1.0 / math.sqrt(raw_minimum_count)

    performance_component = (
        0.30 * count_fit
        + 0.25 * headroom_fit(
            provided_read_iops,
            required_read_iops,
        )
        + 0.25 * headroom_fit(
            provided_write_iops,
            required_write_iops,
        )
        + 0.10 * headroom_fit(
            provided_capacity,
            required_capacity,
        )
        + 0.10 * latency_fit(
            str(drive["latency_class"]),
            required_latency,
        )
    )

    reliability_component = (
        0.45 * endurance_fit(drive_dwpd, minimum_dwpd)
        + 0.55 * reliability_fit(drive_mtbf, minimum_mtbf)
    )

    cost_component = lower_is_better_fit(
        raw_cost,
        max_budget,
    )
    power_component = lower_is_better_fit(
        raw_power,
        max_power,
    )

    final_score = (
        preferences["performance_priority"]
        * performance_component
        + preferences["reliability_priority"]
        * reliability_component
        + preferences["cost_priority"]
        * cost_component
        + preferences["power_priority"]
        * power_component
    )

    candidate = {
        "drive_id": drive["drive_id"],
        "name": drive["name"],
        "manufacturer": drive["manufacturer"],
        "series": drive["series"],
        "media_type": drive["media_type"],
        "protocol": drive["protocol"],
        "drive_form_factor_standard": drive[
            "drive_form_factor_standard"
        ],
        "capacity_tib": round(drive_capacity, 6),
        "random_read_iops_4k": int(drive_read_iops),
        "random_write_iops_4k": int(drive_write_iops),
        "endurance_dwpd_numeric": round(drive_dwpd, 6),
        "mtbf_hours": int(drive_mtbf),
        "latency_class": drive["latency_class"],
        "count_by_capacity": count_by_capacity,
        "count_by_read_iops": count_by_read_iops,
        "count_by_write_iops": count_by_write_iops,
        "raw_minimum_drive_count": raw_minimum_count,
        "raw_provided_capacity_tib": round(
            provided_capacity,
            6,
        ),
        "raw_provided_read_iops": int(provided_read_iops),
        "raw_provided_write_iops": int(provided_write_iops),
        "raw_drive_cost_usd": round(raw_cost, 2),
        "raw_drive_power_w": round(raw_power, 3),
        "score_components": {
            "performance_fit": round(
                performance_component,
                6,
            ),
            "reliability_fit": round(
                reliability_component,
                6,
            ),
            "cost_efficiency": round(
                cost_component,
                6,
            ),
            "power_efficiency": round(
                power_component,
                6,
            ),
        },
        "score": round(final_score, 6),
        "pre_raid_feasible": not rejection_reasons,
    }

    return candidate, rejection_reasons


def candidate_sort_key(
    candidate: dict[str, Any],
) -> tuple[Any, ...]:
    return (
        -float(candidate["score"]),
        int(candidate["raw_minimum_drive_count"]),
        float(candidate["raw_drive_cost_usd"]),
        float(candidate["raw_drive_power_w"]),
        str(candidate["drive_id"]),
    )


def generate_case_candidates(
    architecture_case: dict[str, Any],
    catalog: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    (
        case_id,
        mdt,
        constraints,
        preferences,
    ) = validate_architecture_case(architecture_case)

    feasible: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()

    for drive in catalog:
        if not bool(drive["mdt_eligible"]):
            rejection_counts.update(["not_mdt_eligible"])
            continue

        candidate, reasons = evaluate_drive(
            drive,
            mdt,
            constraints,
            preferences,
        )

        if reasons:
            rejection_counts.update(reasons)
            continue

        feasible.append(candidate)

    feasible.sort(key=candidate_sort_key)
    selected = feasible[:top_k]

    for rank, candidate in enumerate(selected, start=1):
        candidate["rank"] = rank

    return {
        "case_id": case_id,
        "MDT_requirement": mdt,
        "constraints": constraints,
        "preferences": preferences,
        "candidate_summary": {
            "catalog_drive_count": len(catalog),
            "mdt_eligible_drive_count": sum(
                bool(drive["mdt_eligible"])
                for drive in catalog
            ),
            "pre_raid_feasible_count": len(feasible),
            "rejected_count": len(catalog) - len(feasible),
            "top_k_requested": top_k,
            "top_k_returned": len(selected),
            "rejection_counts": dict(
                sorted(rejection_counts.items())
            ),
        },
        "candidates": selected,
        "trace": {
            "mdt_candidate_generator_version":
                GENERATOR_VERSION,
            "candidate_stage": "pre_raid_drive_model_ranking",
            "raw_minimum_drive_count_is_lower_bound": True,
            "global_budget_and_power_are_not_allocated_per_role": True,
            "raid_not_selected": True,
            "final_drive_count_not_selected": True,
            "ha_preserved_for_beam_search": bool(
                constraints["ha_required"]
            ),
        },
    }


def generate_dataset(
    architecture_cases: list[dict[str, Any]],
    catalog: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        raise MDTCandidateGenerationError(
            "top_k doit être strictement positif."
        )

    validate_catalog(catalog)

    results: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()

    total = len(architecture_cases)

    for index, case in enumerate(architecture_cases, start=1):
        if not isinstance(case, dict):
            raise MDTCandidateGenerationError(
                f"architecture[{index - 1}]: objet JSON requis."
            )

        result = generate_case_candidates(
            case,
            catalog,
            top_k,
        )
        case_id = result["case_id"]

        if case_id in seen_case_ids:
            raise MDTCandidateGenerationError(
                f"case_id dupliqué : {case_id}."
            )

        seen_case_ids.add(case_id)
        results.append(result)

        if index % 100 == 0 or index == total:
            print(
                f"Génération MDT : {index}/{total}",
                end="\r" if index < total else "\n",
            )

    return results


def print_summary(
    results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    top1_protocols = Counter(
        case["candidates"][0]["protocol"]
        for case in results
        if case["candidates"]
    )
    top1_series = Counter(
        case["candidates"][0]["series"]
        for case in results
        if case["candidates"]
    )

    cases_without_candidate = sum(
        not case["candidates"]
        for case in results
    )

    feasible_counts = [
        case["candidate_summary"]["pre_raid_feasible_count"]
        for case in results
    ]

    print("Génération des candidats MDT")
    print("-----------------------------")
    print(f"Cas analysés               : {len(results)}")
    print(
        "Cas sans candidat          : "
        f"{cases_without_candidate}"
    )
    print(
        "Candidats faisables min    : "
        f"{min(feasible_counts) if feasible_counts else 0}"
    )
    print(
        "Candidats faisables max    : "
        f"{max(feasible_counts) if feasible_counts else 0}"
    )
    print(
        "Protocoles Top-1           : "
        f"{dict(top1_protocols)}"
    )
    print(
        "Séries Top-1 principales  : "
        f"{dict(top1_series.most_common(10))}"
    )
    print(f"Fichier sauvegardé         : {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Génère et classe les candidats de disques MDT "
            "avant le Beam Search RAID."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Dataset produit par architecture_generator.py.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
        help="Catalogue final des disques.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Fichier JSON des candidats MDT.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Nombre maximal de candidats conservés par cas.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    architecture_cases = load_json(args.input)
    catalog = load_json(args.catalog)

    if not isinstance(architecture_cases, list):
        raise MDTCandidateGenerationError(
            "Le dataset architectural doit être une liste JSON."
        )

    if not isinstance(catalog, list):
        raise MDTCandidateGenerationError(
            "Le catalogue doit être une liste JSON."
        )

    results = generate_dataset(
        architecture_cases,
        catalog,
        args.top_k,
    )
    save_json(args.output, results)
    print_summary(results, args.output)


if __name__ == "__main__":
    main()
