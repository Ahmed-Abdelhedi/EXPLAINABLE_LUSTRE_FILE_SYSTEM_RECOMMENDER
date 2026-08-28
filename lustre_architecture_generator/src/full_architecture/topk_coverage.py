from __future__ import annotations

from typing import Any, Iterable

from .feasibility_coverage import analyze_case_full_path_domain
from .runtime_adapter import build_runtime_handoff


TOPK_COVERAGE_SCHEMA_VERSION = "1.0"
TOPK_COVERAGE_POLICY_ID = "H10D_TOPK_FEASIBILITY_COVERAGE_V1"


class TopKCoverageError(RuntimeError):
    """Erreur de l'analyse H10-D de couverture Top-K."""


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TopKCoverageError(f"{field}: objet JSON requis.")
    return value


def _list(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise TopKCoverageError(f"{field}: liste JSON requise.")
    return value


def normalize_top_k_values(
    values: Iterable[int],
    *,
    baseline_k: int = 10,
) -> list[int]:
    """
    Normalise le calendrier H10-D.

    Les K doivent être strictement supérieurs au K déjà couvert par H10-C.
    L'ordre final est croissant et les doublons sont supprimés.
    """

    if isinstance(baseline_k, bool):
        raise TopKCoverageError("baseline_k: entier > 0 requis.")

    try:
        baseline = int(baseline_k)
    except (TypeError, ValueError) as error:
        raise TopKCoverageError("baseline_k: entier > 0 requis.") from error

    if baseline <= 0:
        raise TopKCoverageError("baseline_k: entier > 0 requis.")

    normalized: set[int] = set()

    for raw in values:
        if isinstance(raw, bool):
            raise TopKCoverageError("top_k_values: entiers requis.")

        try:
            value = int(raw)
        except (TypeError, ValueError) as error:
            raise TopKCoverageError(
                "top_k_values: entiers requis."
            ) from error

        if value <= baseline:
            raise TopKCoverageError(
                f"Chaque K H10-D doit être > baseline_k={baseline}."
            )

        normalized.add(value)

    if not normalized:
        raise TopKCoverageError(
            "Au moins une valeur K supérieure au baseline est requise."
        )

    return sorted(normalized)


def unresolved_case_ids_from_h10c(
    h10c_result: dict[str, Any],
) -> list[str]:
    """
    Extrait les cas encore non récupérés après H10-C.
    """

    rows = _list(
        h10c_result.get("cases"),
        field="h10c_result.cases",
    )

    result: list[str] = []

    for index, row in enumerate(rows):
        row_map = _mapping(
            row,
            field=f"h10c_result.cases[{index}]",
        )

        if row_map.get("status") != "OK":
            continue

        if row_map.get("recovered_valid_architecture") is False:
            case_id = row_map.get("case_id")

            if not isinstance(case_id, str) or not case_id.strip():
                raise TopKCoverageError(
                    f"h10c_result.cases[{index}].case_id invalide."
                )

            result.append(case_id.strip())

    return result


def analyze_case_topk_coverage(
    *,
    architecture: dict[str, Any],
    drive_catalog: list[dict[str, Any]],
    hardware_catalog: dict[str, Any],
    top_k_values: Iterable[int],
    baseline_k: int = 10,
) -> dict[str, Any]:
    """
    Cherche une solution valide aux K demandés en réutilisant le domaine H10-C.

    Pour chaque K :
    - H2 reconstruit le handoff Top-K;
    - H10-C enlève le cap `max_role_options`;
    - H10-C couvre tous les paths compatibles du catalogue courant;
    - toute solution retrouvée est confirmée par H10.

    Aucun score H9 et aucun Beam Search ne participent à la décision.
    """

    architecture_map = _mapping(
        architecture,
        field="architecture",
    )
    case_id = architecture_map.get("case_id")

    if not isinstance(case_id, str) or not case_id.strip():
        raise TopKCoverageError("architecture.case_id invalide.")

    k_values = normalize_top_k_values(
        top_k_values,
        baseline_k=baseline_k,
    )

    attempts: list[dict[str, Any]] = []
    recovered_at_k: int | None = None
    recovered_architecture_id: str | None = None

    for k_value in k_values:
        handoff = build_runtime_handoff(
            architecture=architecture_map,
            catalog=drive_catalog,
            top_k=k_value,
        )

        analysis = analyze_case_full_path_domain(
            handoff=handoff,
            hardware_catalog=hardware_catalog,
        )

        attempt = {
            "top_k": k_value,
            "recovered_valid_architecture": bool(
                analysis["recovered_valid_architecture"]
            ),
            "classification": analysis["search"]["classification"],
            "coverage_interpretation": analysis[
                "coverage_interpretation"
            ],
            "recovered_architecture_id": analysis[
                "recovered_architecture_id"
            ],
            "mdt_raw_options": analysis["option_counts"]["mdt_raw"],
            "ost_raw_options": analysis["option_counts"]["ost_raw"],
            "mdt_pareto_options": analysis["option_counts"]["mdt_pareto"],
            "ost_pareto_options": analysis["option_counts"]["ost_pareto"],
            "minimum_total_cost_usd": analysis["search"][
                "minimum_total_cost_usd"
            ],
            "maximum_budget_usd": analysis["limits"][
                "maximum_budget_usd"
            ],
            "minimum_total_power_w": analysis["search"][
                "minimum_total_power_w"
            ],
            "maximum_power_w": analysis["limits"][
                "maximum_power_w"
            ],
            "selected_path_indexes": analysis["selected_path_indexes"],
        }
        attempts.append(attempt)

        if attempt["recovered_valid_architecture"]:
            recovered_at_k = k_value
            recovered_architecture_id = attempt[
                "recovered_architecture_id"
            ]
            break

    last_attempt = attempts[-1]

    return {
        "schema_version": TOPK_COVERAGE_SCHEMA_VERSION,
        "policy_id": TOPK_COVERAGE_POLICY_ID,
        "case_id": case_id.strip(),
        "baseline_k": int(baseline_k),
        "tested_top_k_values": k_values,
        "attempts": attempts,
        "recovered_valid_architecture": recovered_at_k is not None,
        "recovered_at_k": recovered_at_k,
        "recovered_architecture_id": recovered_architecture_id,
        "final_classification": last_attempt["classification"],
        "coverage_interpretation": (
            f"RECOVERED_AT_K_{recovered_at_k}"
            if recovered_at_k is not None
            else (
                f"NO_FEASIBLE_PAIR_THROUGH_K_{k_values[-1]}_"
                "WITH_FULL_REFERENCE_HARDWARE_PATH_DOMAIN"
            )
        ),
        "beam_search_applied": False,
        "architecture_scoring_required": False,
        "global_infeasibility_claimed": False,
        "remaining_uncertainty": (
            None
            if recovered_at_k is not None
            else (
                f"drive candidates below K={k_values[-1]} and "
                "reference-catalog scope"
            )
        ),
    }
