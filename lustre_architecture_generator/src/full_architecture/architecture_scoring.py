from __future__ import annotations

import copy
import math
from statistics import fmean
from typing import Any

from .architecture_state import validate_full_architecture_state


SCORING_SCHEMA_VERSION = "1.0"
SCORING_POLICY_ID = "ARCH_SCORE_POLICY_V1"

PREFERENCE_KEYS = (
    "performance_priority",
    "cost_priority",
    "power_priority",
    "reliability_priority",
)

RELIABILITY_FIELDS = (
    "endurance_dwpd",
    "mtbf_hours",
    "warranty_years",
    "workload_rating_tb_per_year",
)


class ArchitectureScoringError(RuntimeError):
    """Erreur du scoring soft H9 des architectures complètes."""


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArchitectureScoringError(f"{field}: objet JSON requis.")
    return value


def _list(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ArchitectureScoringError(f"{field}: liste JSON requise.")
    return value


def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArchitectureScoringError(f"{field}: chaîne non vide requise.")
    return value.strip()


def _finite(
    value: Any,
    *,
    field: str,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise ArchitectureScoringError(f"{field}: nombre requis.")

    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ArchitectureScoringError(f"{field}: nombre requis.") from error

    if not math.isfinite(number):
        raise ArchitectureScoringError(f"{field}: nombre fini requis.")

    if minimum is not None and number < minimum:
        raise ArchitectureScoringError(
            f"{field}: valeur >= {minimum} requise."
        )

    return number


def _optional_finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number) or number < 0.0:
        return None

    return number


def _mean(values: list[float], *, default: float = 0.0) -> float:
    if not values:
        return default
    return float(fmean(values))


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def normalize_preference_weights(
    preferences: dict[str, Any],
) -> dict[str, Any]:
    """
    Normalise les poids issus du Preference Extractor/BWM.

    H9 ne convertit aucun label qualitatif en poids. Il consomme uniquement les
    poids numériques déjà calculés en amont.
    """

    values: dict[str, float] = {}

    for key in PREFERENCE_KEYS:
        values[key] = _finite(
            preferences.get(key),
            field=f"preferences.{key}",
            minimum=0.0,
        )

    original_sum = sum(values.values())

    if original_sum <= 0.0:
        raise ArchitectureScoringError(
            "La somme des poids de préférence doit être > 0."
        )

    normalized = {
        key: value / original_sum
        for key, value in values.items()
    }

    return {
        "original": values,
        "original_sum": original_sum,
        "normalized": normalized,
        "normalized_sum": sum(normalized.values()),
    }


def _candidate_lookup(
    handoff: dict[str, Any],
    role: str,
) -> dict[str, dict[str, Any]]:
    key = "mdt_candidates" if role == "MDT" else "ost_candidates"
    candidates = _list(
        handoff.get(key),
        field=f"handoff.{key}",
    )

    result: dict[str, dict[str, Any]] = {}

    for index, candidate in enumerate(candidates):
        candidate_map = _mapping(
            candidate,
            field=f"handoff.{key}[{index}]",
        )
        identity = _mapping(
            candidate_map.get("identity"),
            field=f"handoff.{key}[{index}].identity",
        )
        drive_id = _string(
            identity.get("drive_id"),
            field=f"handoff.{key}[{index}].identity.drive_id",
        )

        if drive_id in result:
            raise ArchitectureScoringError(
                f"drive_id {drive_id!r} dupliqué dans {key}."
            )

        result[drive_id] = candidate_map

    if not result:
        raise ArchitectureScoringError(f"{key}: aucun candidat.")

    return result


def _performance_metric(
    *,
    required: float,
    provided: float,
) -> dict[str, Any]:
    """
    Score de headroom borné sans constante arbitraire.

    Pour required > 0 :
        score = max(0, 1 - required / provided)

    Donc :
    - juste à la limite -> 0;
    - 2x le besoin -> 0.5;
    - le score tend vers 1 lorsque le headroom augmente.

    Une métrique dont le besoin vaut 0 n'entre pas dans la moyenne.
    """

    if required <= 0.0:
        return {
            "active": False,
            "required": required,
            "provided": provided,
            "ratio": None,
            "headroom_score": None,
            "satisfied": provided >= required,
        }

    if provided <= 0.0:
        ratio = 0.0
        score = 0.0
    else:
        ratio = provided / required
        score = _clamp01(1.0 - (required / provided))

    return {
        "active": True,
        "required": required,
        "provided": provided,
        "ratio": ratio,
        "headroom_score": score,
        "satisfied": provided + 1e-12 >= required,
    }


def _performance_snapshot(
    *,
    state: dict[str, Any],
    requirements: dict[str, Any],
) -> dict[str, Any]:
    mdt_req = _mapping(
        requirements.get("MDT_requirement"),
        field="requirements.MDT_requirement",
    )
    ost_req = _mapping(
        requirements.get("OST_requirement"),
        field="requirements.OST_requirement",
    )
    perf = _mapping(
        state.get("performance"),
        field="state.performance",
    )

    metric_specs = (
        (
            "mdt_capacity",
            _finite(
                mdt_req.get("required_metadata_capacity_tib"),
                field="MDT.required_metadata_capacity_tib",
                minimum=0.0,
            ),
            _finite(
                perf.get("metadata_capacity_tib"),
                field="state.performance.metadata_capacity_tib",
                minimum=0.0,
            ),
        ),
        (
            "mdt_read_iops",
            _finite(
                mdt_req.get("required_read_iops"),
                field="MDT.required_read_iops",
                minimum=0.0,
            ),
            _finite(
                perf.get("mdt_read_iops"),
                field="state.performance.mdt_read_iops",
                minimum=0.0,
            ),
        ),
        (
            "mdt_write_iops",
            _finite(
                mdt_req.get("required_write_iops"),
                field="MDT.required_write_iops",
                minimum=0.0,
            ),
            _finite(
                perf.get("mdt_write_iops"),
                field="state.performance.mdt_write_iops",
                minimum=0.0,
            ),
        ),
        (
            "ost_capacity",
            _finite(
                ost_req.get("required_usable_capacity_tib"),
                field="OST.required_usable_capacity_tib",
                minimum=0.0,
            ),
            _finite(
                perf.get("ost_usable_capacity_tib"),
                field="state.performance.ost_usable_capacity_tib",
                minimum=0.0,
            ),
        ),
        (
            "ost_read_bandwidth",
            _finite(
                ost_req.get("required_read_bandwidth_gbps"),
                field="OST.required_read_bandwidth_gbps",
                minimum=0.0,
            ),
            _finite(
                perf.get("ost_read_bandwidth_gb_s"),
                field="state.performance.ost_read_bandwidth_gb_s",
                minimum=0.0,
            ),
        ),
        (
            "ost_write_bandwidth",
            _finite(
                ost_req.get("required_write_bandwidth_gbps"),
                field="OST.required_write_bandwidth_gbps",
                minimum=0.0,
            ),
            _finite(
                perf.get("ost_write_bandwidth_gb_s"),
                field="state.performance.ost_write_bandwidth_gb_s",
                minimum=0.0,
            ),
        ),
        (
            "ost_total_bandwidth",
            _finite(
                ost_req.get("required_total_bandwidth_gbps"),
                field="OST.required_total_bandwidth_gbps",
                minimum=0.0,
            ),
            _finite(
                perf.get("ost_total_bandwidth_gb_s"),
                field="state.performance.ost_total_bandwidth_gb_s",
                minimum=0.0,
            ),
        ),
    )

    metrics: dict[str, dict[str, Any]] = {}
    active_scores: list[float] = []
    satisfied_values: list[bool] = []

    for name, required, provided in metric_specs:
        metric = _performance_metric(
            required=required,
            provided=provided,
        )
        metrics[name] = metric
        satisfied_values.append(bool(metric["satisfied"]))

        if metric["active"]:
            active_scores.append(float(metric["headroom_score"]))

    return {
        "score": _mean(active_scores, default=1.0),
        "metrics": metrics,
        "all_requirements_satisfied": all(satisfied_values),
    }


def _inverse_minmax_score(
    value: float,
    values: list[float],
) -> float:
    if not values:
        return 1.0

    minimum = min(values)
    maximum = max(values)

    if math.isclose(minimum, maximum, rel_tol=0.0, abs_tol=1e-12):
        return 1.0

    return _clamp01((maximum - value) / (maximum - minimum))


def _numeric_reliability_pool(
    *,
    candidates: dict[str, dict[str, Any]],
    media_type: str,
    field: str,
) -> list[float]:
    values: list[float] = []

    for candidate in candidates.values():
        identity = candidate.get("identity")
        reliability = candidate.get("reliability")

        if not isinstance(identity, dict) or not isinstance(reliability, dict):
            continue

        if str(identity.get("media_type", "")).upper() != media_type:
            continue

        value = _optional_finite(reliability.get(field))
        if value is not None:
            values.append(value)

    return values


def _higher_is_better_minmax(
    value: float,
    pool: list[float],
) -> float:
    if not pool:
        return 0.0

    minimum = min(pool)
    maximum = max(pool)

    if math.isclose(minimum, maximum, rel_tol=0.0, abs_tol=1e-12):
        return 1.0

    return _clamp01((value - minimum) / (maximum - minimum))


def _drive_reliability_score(
    *,
    candidate: dict[str, Any],
    role_candidates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    identity = _mapping(
        candidate.get("identity"),
        field="candidate.identity",
    )
    reliability = _mapping(
        candidate.get("reliability"),
        field="candidate.reliability",
    )
    media_type = str(identity.get("media_type", "")).upper()

    evidence: dict[str, dict[str, Any]] = {}
    credits: list[float] = []
    applicable_count = 0
    known_count = 0

    for field in RELIABILITY_FIELDS:
        pool = _numeric_reliability_pool(
            candidates=role_candidates,
            media_type=media_type,
            field=field,
        )

        if not pool:
            evidence[field] = {
                "applicable": False,
                "known": False,
                "raw_value": None,
                "score": None,
            }
            continue

        applicable_count += 1
        raw_value = _optional_finite(reliability.get(field))

        if raw_value is None:
            evidence[field] = {
                "applicable": True,
                "known": False,
                "raw_value": None,
                "score": 0.0,
            }
            credits.append(0.0)
            continue

        known_count += 1
        score = _higher_is_better_minmax(raw_value, pool)
        evidence[field] = {
            "applicable": True,
            "known": True,
            "raw_value": raw_value,
            "score": score,
        }
        credits.append(score)

    coverage = (
        known_count / applicable_count
        if applicable_count > 0
        else 0.0
    )

    return {
        "score": _mean(credits, default=0.0),
        "coverage": coverage,
        "media_type": media_type,
        "evidence": evidence,
        "semantics": (
            "numeric evidence normalized within role + media family; "
            "missing applicable evidence receives no reliability credit"
        ),
    }


def _selected_candidate(
    *,
    state: dict[str, Any],
    role: str,
    lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    selected = _mapping(
        state.get("selected"),
        field="state.selected",
    )
    key = "mdt_drive" if role == "MDT" else "ost_drive"
    drive = _mapping(
        selected.get(key),
        field=f"state.selected.{key}",
    )
    drive_id = _string(
        drive.get("drive_id"),
        field=f"state.selected.{key}.drive_id",
    )

    candidate = lookup.get(drive_id)
    if candidate is None:
        raise ArchitectureScoringError(
            f"{role}: drive {drive_id!r} absent du handoff."
        )

    return candidate


def _raw_reliability_snapshot(
    *,
    state: dict[str, Any],
    mdt_lookup: dict[str, dict[str, Any]],
    ost_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    selected = _mapping(
        state.get("selected"),
        field="state.selected",
    )

    mdt_candidate = _selected_candidate(
        state=state,
        role="MDT",
        lookup=mdt_lookup,
    )
    ost_candidate = _selected_candidate(
        state=state,
        role="OST",
        lookup=ost_lookup,
    )

    mdt_drive = _drive_reliability_score(
        candidate=mdt_candidate,
        role_candidates=mdt_lookup,
    )
    ost_drive = _drive_reliability_score(
        candidate=ost_candidate,
        role_candidates=ost_lookup,
    )

    mdt_protection = _mapping(
        selected.get("mdt_protection"),
        field="state.selected.mdt_protection",
    )
    ost_protection = _mapping(
        selected.get("ost_protection"),
        field="state.selected.ost_protection",
    )
    mdt_path = _mapping(
        selected.get("mdt_hardware_path"),
        field="state.selected.mdt_hardware_path",
    )
    ost_path = _mapping(
        selected.get("ost_hardware_path"),
        field="state.selected.ost_hardware_path",
    )

    return {
        "drive_score": _mean(
            [
                float(mdt_drive["score"]),
                float(ost_drive["score"]),
            ]
        ),
        "drive_evidence": {
            "mdt": mdt_drive,
            "ost": ost_drive,
        },
        "mdt_fault_tolerance": _finite(
            mdt_protection.get("fault_tolerance_drives_per_group"),
            field="MDT.fault_tolerance_drives_per_group",
            minimum=0.0,
        ),
        "ost_fault_tolerance": _finite(
            ost_protection.get("fault_tolerance_drives_per_group"),
            field="OST.fault_tolerance_drives_per_group",
            minimum=0.0,
        ),
        "mdt_ha_enabled": (
            str(mdt_path.get("ha_profile_id", "")).upper()
            not in {"", "HA_NONE"}
        ),
        "ost_ha_enabled": (
            str(ost_path.get("ha_profile_id", "")).upper()
            not in {"", "HA_NONE"}
        ),
    }


def _hard_constraint_snapshot(
    *,
    state: dict[str, Any],
    requirements: dict[str, Any],
    performance_snapshot: dict[str, Any],
) -> dict[str, Any]:
    constraints = _mapping(
        requirements.get("constraints"),
        field="requirements.constraints",
    )
    cost_power = _mapping(
        state.get("cost_power"),
        field="state.cost_power",
    )

    total_cost = _finite(
        cost_power.get("total_cost_usd"),
        field="state.cost_power.total_cost_usd",
        minimum=0.0,
    )
    total_power = _finite(
        cost_power.get("total_power_w"),
        field="state.cost_power.total_power_w",
        minimum=0.0,
    )
    max_budget = _finite(
        constraints.get("max_budget_usd"),
        field="constraints.max_budget_usd",
        minimum=0.0,
    )
    max_power = _finite(
        constraints.get("max_power_w"),
        field="constraints.max_power_w",
        minimum=0.0,
    )

    checks = {
        "performance_requirements": bool(
            performance_snapshot["all_requirements_satisfied"]
        ),
        "budget_lower_bound": total_cost <= max_budget + 1e-12,
        "power_lower_bound": total_power <= max_power + 1e-12,
    }

    return {
        "checks": checks,
        "passed": all(checks.values()),
        "budget": {
            "maximum_usd": max_budget,
            "architecture_lower_bound_usd": total_cost,
            "satisfied": checks["budget_lower_bound"],
        },
        "power": {
            "maximum_w": max_power,
            "architecture_lower_bound_w": total_power,
            "satisfied": checks["power_lower_bound"],
        },
        "semantics": (
            "pre-score snapshot only; H10 remains the independent full "
            "deterministic validator"
        ),
    }


def _prepare_raw_record(
    *,
    architecture: dict[str, Any],
    handoff: dict[str, Any],
    mdt_lookup: dict[str, dict[str, Any]],
    ost_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    architecture_id = _string(
        architecture.get("architecture_id"),
        field="architecture.architecture_id",
    )
    case_id = _string(
        architecture.get("case_id"),
        field="architecture.case_id",
    )
    state = _mapping(
        architecture.get("state"),
        field="architecture.state",
    )

    validate_full_architecture_state(state)

    if state.get("stage") != "COMPLETE":
        raise ArchitectureScoringError(
            f"{architecture_id}: H9 exige un ArchitectureState COMPLETE."
        )

    if case_id != str(handoff.get("case_id")):
        raise ArchitectureScoringError(
            f"{architecture_id}: case_id incohérent avec handoff."
        )

    requirements = _mapping(
        handoff.get("requirements"),
        field="handoff.requirements",
    )
    performance = _performance_snapshot(
        state=state,
        requirements=requirements,
    )
    reliability = _raw_reliability_snapshot(
        state=state,
        mdt_lookup=mdt_lookup,
        ost_lookup=ost_lookup,
    )
    hard_snapshot = _hard_constraint_snapshot(
        state=state,
        requirements=requirements,
        performance_snapshot=performance,
    )

    cost_power = _mapping(
        state.get("cost_power"),
        field="state.cost_power",
    )

    return {
        "architecture_id": architecture_id,
        "case_id": case_id,
        "generation_index": architecture.get("generation_index"),
        "state": state,
        "performance": performance,
        "reliability_raw": reliability,
        "hard_constraint_snapshot": hard_snapshot,
        "total_cost_usd": _finite(
            cost_power.get("total_cost_usd"),
            field="state.cost_power.total_cost_usd",
            minimum=0.0,
        ),
        "total_power_w": _finite(
            cost_power.get("total_power_w"),
            field="state.cost_power.total_power_w",
            minimum=0.0,
        ),
    }


def score_generated_architectures(
    *,
    generation_result: dict[str, Any],
    handoff: dict[str, Any],
) -> dict[str, Any]:
    """
    Score toutes les architectures H8 d'un même case_id.

    Le score est soft et préférence-dépendant. Il ne remplace aucune
    contrainte dure et ne marque jamais un state comme validé. Le classement
    produit ici est pré-H10 et doit être filtré par le validateur complet avant
    toute recommandation finale.
    """

    generation_case = _string(
        generation_result.get("case_id"),
        field="generation_result.case_id",
    )
    handoff_case = _string(
        handoff.get("case_id"),
        field="handoff.case_id",
    )

    if generation_case != handoff_case:
        raise ArchitectureScoringError(
            "generation_result et handoff n'ont pas le même case_id."
        )

    architectures = _list(
        generation_result.get("architectures"),
        field="generation_result.architectures",
    )
    if not architectures:
        raise ArchitectureScoringError(
            f"{generation_case}: aucune architecture à scorer."
        )

    requirements = _mapping(
        handoff.get("requirements"),
        field="handoff.requirements",
    )
    preferences = _mapping(
        requirements.get("preferences"),
        field="requirements.preferences",
    )
    weight_info = normalize_preference_weights(preferences)
    weights = weight_info["normalized"]

    mdt_lookup = _candidate_lookup(handoff, "MDT")
    ost_lookup = _candidate_lookup(handoff, "OST")

    raw_records = [
        _prepare_raw_record(
            architecture=_mapping(
                architecture,
                field=f"architectures[{index}]",
            ),
            handoff=handoff,
            mdt_lookup=mdt_lookup,
            ost_lookup=ost_lookup,
        )
        for index, architecture in enumerate(architectures)
    ]

    # H9 définit uniquement la fonction de score.
    # La normalisation est donc calculée sur tout le pool H8 fourni.
    # Le snapshot de contraintes reste diagnostique : H10 est le seul
    # validateur complet autorisé à décider quelles architectures sont valides.
    cost_values = [
        float(record["total_cost_usd"])
        for record in raw_records
    ]
    power_values = [
        float(record["total_power_w"])
        for record in raw_records
    ]

    maximum_fault_tolerance = max(
        [
            float(record["reliability_raw"]["mdt_fault_tolerance"])
            for record in raw_records
        ]
        + [
            float(record["reliability_raw"]["ost_fault_tolerance"])
            for record in raw_records
        ]
        + [1.0]
    )

    scored: list[dict[str, Any]] = []

    for raw in raw_records:
        reliability_raw = raw["reliability_raw"]

        protection_score = _mean(
            [
                _clamp01(
                    float(reliability_raw["mdt_fault_tolerance"])
                    / maximum_fault_tolerance
                ),
                _clamp01(
                    float(reliability_raw["ost_fault_tolerance"])
                    / maximum_fault_tolerance
                ),
            ]
        )
        ha_score = _mean(
            [
                1.0 if reliability_raw["mdt_ha_enabled"] else 0.0,
                1.0 if reliability_raw["ost_ha_enabled"] else 0.0,
            ]
        )
        reliability_score = _mean(
            [
                float(reliability_raw["drive_score"]),
                protection_score,
                ha_score,
            ]
        )

        components = {
            "performance": _clamp01(float(raw["performance"]["score"])),
            "cost": _inverse_minmax_score(
                float(raw["total_cost_usd"]),
                cost_values,
            ),
            "power": _inverse_minmax_score(
                float(raw["total_power_w"]),
                power_values,
            ),
            "reliability": _clamp01(reliability_score),
        }

        contributions = {
            "performance": (
                weights["performance_priority"]
                * components["performance"]
            ),
            "cost": weights["cost_priority"] * components["cost"],
            "power": weights["power_priority"] * components["power"],
            "reliability": (
                weights["reliability_priority"]
                * components["reliability"]
            ),
        }

        total_score = _clamp01(sum(contributions.values()))

        scored.append(
            {
                "schema_version": SCORING_SCHEMA_VERSION,
                "scoring_policy_id": SCORING_POLICY_ID,
                "architecture_id": raw["architecture_id"],
                "case_id": raw["case_id"],
                "generation_index": raw["generation_index"],
                "pre_h10_hard_snapshot_passed": bool(
                    raw["hard_constraint_snapshot"]["passed"]
                ),
                "score": total_score,
                "score_direction": "higher_is_better",
                "components": components,
                "weighted_contributions": contributions,
                "preference_weights": copy.deepcopy(weights),
                "performance_explainability": copy.deepcopy(
                    raw["performance"]
                ),
                "reliability_explainability": {
                    "score": reliability_score,
                    "drive_score": reliability_raw["drive_score"],
                    "drive_evidence": copy.deepcopy(
                        reliability_raw["drive_evidence"]
                    ),
                    "protection_score": protection_score,
                    "ha_score": ha_score,
                    "fault_tolerance_normalizer": maximum_fault_tolerance,
                    "semantics": (
                        "reference proxy score; not a failure probability "
                        "or vendor reliability prediction"
                    ),
                },
                "cost_power": {
                    "total_cost_usd": raw["total_cost_usd"],
                    "total_power_w": raw["total_power_w"],
                    "cost_normalization": {
                        "minimum": min(cost_values),
                        "maximum": max(cost_values),
                        "higher_score_means": "lower_cost",
                    },
                    "power_normalization": {
                        "minimum": min(power_values),
                        "maximum": max(power_values),
                        "higher_score_means": "lower_power",
                    },
                },
                "hard_constraint_snapshot": copy.deepcopy(
                    raw["hard_constraint_snapshot"]
                ),
                "full_validator_applied": False,
                "beam_search_applied": False,
                "state_validation_status_preserved": (
                    raw["state"]["validation"]["status"]
                ),
            }
        )

    ranked = sorted(
        scored,
        key=lambda item: (
            -float(item["score"]),
            str(item["architecture_id"]),
        ),
    )

    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank

    snapshot_passed = [
        item
        for item in ranked
        if item["pre_h10_hard_snapshot_passed"]
    ]
    scores = [float(item["score"]) for item in ranked]

    return {
        "schema_version": SCORING_SCHEMA_VERSION,
        "stage": "full_architecture_scoring",
        "scoring_policy": {
            "id": SCORING_POLICY_ID,
            "type": "deterministic_preference_weighted_soft_score",
            "performance": (
                "mean bounded headroom score over active MDT/OST requirements"
            ),
            "cost": "inverse min-max within the generated H8 case pool",
            "power": "inverse min-max within the generated H8 case pool",
            "reliability": (
                "equal mean of drive numeric evidence proxy, protection "
                "fault-tolerance proxy and HA presence proxy"
            ),
            "hard_constraints_are_not_soft_penalties": True,
            "full_validator_applied": False,
            "beam_search_applied": False,
        },
        "case_id": generation_case,
        "preference_weights": weight_info,
        "summary": {
            "architecture_count": len(ranked),
            "pre_h10_hard_snapshot_pass_count": len(snapshot_passed),
            "pre_h10_hard_snapshot_fail_count": (
                len(ranked) - len(snapshot_passed)
            ),
            "best_pre_h10_architecture_id": ranked[0]["architecture_id"],
            "best_pre_h10_score": ranked[0]["score"],
            "score_min": min(scores),
            "score_mean": _mean(scores),
            "score_max": max(scores),
        },
        "architectures": ranked,
    }


def assert_scoring_result_valid(
    result: dict[str, Any],
) -> None:
    if result.get("schema_version") != SCORING_SCHEMA_VERSION:
        raise ArchitectureScoringError(
            "schema_version scoring incorrecte."
        )

    if result.get("stage") != "full_architecture_scoring":
        raise ArchitectureScoringError("stage scoring incorrect.")

    architectures = _list(
        result.get("architectures"),
        field="result.architectures",
    )

    if not architectures:
        raise ArchitectureScoringError("Aucun score architecture.")

    seen: set[str] = set()

    for expected_rank, item in enumerate(architectures, start=1):
        item_map = _mapping(
            item,
            field=f"result.architectures[{expected_rank - 1}]",
        )
        architecture_id = _string(
            item_map.get("architecture_id"),
            field="score.architecture_id",
        )

        if architecture_id in seen:
            raise ArchitectureScoringError(
                f"architecture_id dupliqué: {architecture_id}"
            )
        seen.add(architecture_id)

        if item_map.get("rank") != expected_rank:
            raise ArchitectureScoringError(
                "Ranks H9 non contigus."
            )

        score = _finite(
            item_map.get("score"),
            field="score.score",
            minimum=0.0,
        )
        if score > 1.0 + 1e-12:
            raise ArchitectureScoringError("score H9 > 1.")

        components = _mapping(
            item_map.get("components"),
            field="score.components",
        )
        for key in ("performance", "cost", "power", "reliability"):
            component = _finite(
                components.get(key),
                field=f"score.components.{key}",
                minimum=0.0,
            )
            if component > 1.0 + 1e-12:
                raise ArchitectureScoringError(
                    f"score.components.{key} > 1."
                )

        if item_map.get("full_validator_applied") is not False:
            raise ArchitectureScoringError(
                "H9 ne doit pas prétendre avoir appliqué H10."
            )

        if item_map.get("beam_search_applied") is not False:
            raise ArchitectureScoringError(
                "H9 ne doit pas appliquer Beam Search."
            )
