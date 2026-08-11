from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Callable


# ============================================================
# Gestion des imports
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from feature_builder import (  # noqa: E402
    DEFAULT_ARCHITECTURES_PATH,
    DEFAULT_CATALOG_PATH,
    load_json,
)

from ost_ranker_inference import (  # noqa: E402
    rank_all_ost_candidates,
)


# ============================================================
# Exception
# ============================================================

class DiversifiedTopKError(RuntimeError):
    """
    Erreur pendant la construction du Top-K OST diversifié.
    """


# ============================================================
# Fonctions numériques utilitaires
# ============================================================

def safe_float(
    value: Any,
    *,
    default: float = 0.0,
) -> float:
    """
    Convertit une valeur en nombre flottant fini.

    Parameters
    ----------
    value:
        Valeur à convertir.

    default:
        Valeur utilisée lorsque la conversion échoue ou
        lorsque le résultat n'est pas fini.
    """

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(result):
        return default

    return result


def cost_per_tib(
    candidate: dict[str, Any],
) -> float:
    """
    Calcule le coût unitaire du drive par TiB.

    Formule
    -------
    coût par TiB = prix du drive / capacité du drive
    """

    capacity_tib = safe_float(
        candidate.get("capacity_tib")
    )

    price_usd = safe_float(
        candidate.get("price_usd")
    )

    if capacity_tib <= 0:
        return float("inf")

    if price_usd <= 0:
        return float("inf")

    return price_usd / capacity_tib


def power_per_tib(
    candidate: dict[str, Any],
) -> float:
    """
    Calcule la puissance consommée par TiB.

    Formule
    -------
    puissance par TiB = puissance du drive / capacité
    """

    capacity_tib = safe_float(
        candidate.get("capacity_tib")
    )

    power_w = safe_float(
        candidate.get("power_w")
    )

    if capacity_tib <= 0:
        return float("inf")

    if power_w <= 0:
        return float("inf")

    return power_w / capacity_tib


# ============================================================
# Validation du classement source
# ============================================================

def validate_ranking_result(
    ranking_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Valide le résultat produit par le OST Ranker.
    """

    required_result_fields = {
        "case_id",
        "feasible_candidate_count",
        "ranked_candidates",
    }

    missing_result_fields = (
        required_result_fields
        - set(ranking_result.keys())
    )

    if missing_result_fields:
        raise DiversifiedTopKError(
            "Résultat du OST Ranker incomplet. "
            f"Champs manquants : "
            f"{sorted(missing_result_fields)}"
        )

    ranked_candidates = ranking_result[
        "ranked_candidates"
    ]

    if not isinstance(ranked_candidates, list):
        raise TypeError(
            "'ranked_candidates' doit être une liste."
        )

    if not ranked_candidates:
        raise DiversifiedTopKError(
            "Le classement OST est vide."
        )

    required_candidate_fields = {
        "drive_id",
        "drive_name",
        "media_type",
        "capacity_tib",
        "seq_read_mb_s",
        "price_usd",
        "power_w",
        "ml_score",
        "ml_rank",
        "raw_minimum_drive_count",
    }

    for index, candidate in enumerate(
        ranked_candidates
    ):
        missing_candidate_fields = (
            required_candidate_fields
            - set(candidate.keys())
        )

        if missing_candidate_fields:
            raise DiversifiedTopKError(
                f"Candidat OST à la position {index} "
                "incomplet. Champs manquants : "
                f"{sorted(missing_candidate_fields)}"
            )

    return ranked_candidates


# ============================================================
# Gestion des candidats sélectionnés
# ============================================================

def add_candidate(
    selected_by_id: dict[str, dict[str, Any]],
    candidate: dict[str, Any],
    reason: str,
) -> None:
    """
    Ajoute un candidat au Top-K sans créer de doublon.

    Un même candidat peut recevoir plusieurs raisons de
    sélection. Cela permet de conserver une trace explicable
    de chaque décision.
    """

    drive_id = str(candidate["drive_id"])

    if drive_id not in selected_by_id:
        selected_candidate = dict(candidate)

        selected_candidate[
            "selection_reasons"
        ] = []

        selected_candidate[
            "cost_per_tib"
        ] = cost_per_tib(candidate)

        selected_candidate[
            "power_per_tib"
        ] = power_per_tib(candidate)

        selected_by_id[drive_id] = (
            selected_candidate
        )

    selection_reasons = selected_by_id[
        drive_id
    ]["selection_reasons"]

    if reason not in selection_reasons:
        selection_reasons.append(reason)


# ============================================================
# Sélection déterministe du meilleur candidat
# ============================================================

def select_minimum_candidate(
    candidates: list[dict[str, Any]],
    metric: Callable[
        [dict[str, Any]],
        float,
    ],
) -> dict[str, Any] | None:
    """
    Sélectionne le candidat avec la plus petite métrique.

    En cas d'égalité :
    1. le meilleur rang ML est préféré ;
    2. le drive_id est utilisé pour garantir un résultat
       déterministe.
    """

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda candidate: (
            metric(candidate),
            int(candidate["ml_rank"]),
            str(candidate["drive_id"]),
        ),
    )


def select_maximum_candidate(
    candidates: list[dict[str, Any]],
    metric: Callable[
        [dict[str, Any]],
        float,
    ],
) -> dict[str, Any] | None:
    """
    Sélectionne le candidat avec la plus grande métrique.

    En cas d'égalité :
    1. le meilleur rang ML est préféré ;
    2. le drive_id garantit un résultat déterministe.
    """

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda candidate: (
            -metric(candidate),
            int(candidate["ml_rank"]),
            str(candidate["drive_id"]),
        ),
    )


# ============================================================
# Construction du Top-K OST diversifié
# ============================================================

def select_diversified_ost_top_k(
    ranking_result: dict[str, Any],
    top_k: int = 10,
    global_top_count: int = 4,
    diversification_multiplier: int = 4,
    minimum_diversification_pool_size: int = 40,
) -> dict[str, Any]:
    """
    Construit un Top-K OST diversifié avec garde-fou ML.

    Ordre de sélection
    ------------------
    1. meilleurs candidats ML globaux ;
    2. meilleur HDD du pool de qualité ;
    3. meilleur SSD du pool de qualité ;
    4. meilleur coût par TiB du pool de qualité ;
    5. plus grande capacité du pool de qualité ;
    6. meilleur débit séquentiel du pool de qualité ;
    7. meilleure efficacité énergétique du pool de qualité ;
    8. remplissage selon le classement ML global.

    Le pool de diversification empêche un candidat ayant un
    rang ML très faible d'entrer uniquement grâce à une
    caractéristique spécialisée.
    """

    if top_k <= 0:
        raise ValueError(
            "top_k doit être strictement supérieur à zéro."
        )

    if global_top_count < 0:
        raise ValueError(
            "global_top_count ne peut pas être négatif."
        )

    if diversification_multiplier <= 0:
        raise ValueError(
            "diversification_multiplier doit être "
            "strictement supérieur à zéro."
        )

    if minimum_diversification_pool_size <= 0:
        raise ValueError(
            "minimum_diversification_pool_size doit être "
            "strictement supérieur à zéro."
        )

    ranked_candidates = validate_ranking_result(
        ranking_result
    )

    # --------------------------------------------------------
    # Pool limité aux candidats de qualité ML raisonnable
    # --------------------------------------------------------

    requested_pool_size = max(
        top_k * diversification_multiplier,
        minimum_diversification_pool_size,
    )

    diversification_pool_size = min(
        len(ranked_candidates),
        requested_pool_size,
    )

    diversification_pool = ranked_candidates[
        :diversification_pool_size
    ]

    selected_by_id: dict[
        str,
        dict[str, Any],
    ] = {}

    # --------------------------------------------------------
    # 1. Conservation des meilleurs candidats ML globaux
    # --------------------------------------------------------

    global_selection_count = min(
        global_top_count,
        top_k,
        len(ranked_candidates),
    )

    for candidate in ranked_candidates[
        :global_selection_count
    ]:
        add_candidate(
            selected_by_id=selected_by_id,
            candidate=candidate,
            reason="global_ml_top",
        )

    # --------------------------------------------------------
    # Séparation HDD / SSD dans le pool de qualité
    # --------------------------------------------------------

    hdd_candidates = [
        candidate
        for candidate in diversification_pool
        if str(
            candidate.get("media_type", "")
        ).strip().upper() == "HDD"
    ]

    ssd_candidates = [
        candidate
        for candidate in diversification_pool
        if str(
            candidate.get("media_type", "")
        ).strip().upper() == "SSD"
    ]

    # --------------------------------------------------------
    # 2. Meilleur HDD selon le score ML
    # --------------------------------------------------------

    if hdd_candidates:
        best_hdd_candidate = hdd_candidates[0]

        add_candidate(
            selected_by_id=selected_by_id,
            candidate=best_hdd_candidate,
            reason="best_hdd_ml",
        )

    # --------------------------------------------------------
    # 3. Meilleur SSD selon le score ML
    # --------------------------------------------------------

    if ssd_candidates:
        best_ssd_candidate = ssd_candidates[0]

        add_candidate(
            selected_by_id=selected_by_id,
            candidate=best_ssd_candidate,
            reason="best_ssd_ml",
        )

    # --------------------------------------------------------
    # 4. Meilleur coût par TiB
    # --------------------------------------------------------

    valid_cost_candidates = [
        candidate
        for candidate in diversification_pool
        if math.isfinite(
            cost_per_tib(candidate)
        )
    ]

    best_cost_candidate = (
        select_minimum_candidate(
            candidates=valid_cost_candidates,
            metric=cost_per_tib,
        )
    )

    if best_cost_candidate is not None:
        add_candidate(
            selected_by_id=selected_by_id,
            candidate=best_cost_candidate,
            reason="best_cost_per_tib",
        )

    # --------------------------------------------------------
    # 5. Plus grande capacité unitaire
    # --------------------------------------------------------

    highest_capacity_candidate = (
        select_maximum_candidate(
            candidates=diversification_pool,
            metric=lambda candidate: safe_float(
                candidate.get("capacity_tib")
            ),
        )
    )

    if highest_capacity_candidate is not None:
        add_candidate(
            selected_by_id=selected_by_id,
            candidate=highest_capacity_candidate,
            reason="highest_capacity",
        )

    # --------------------------------------------------------
    # 6. Meilleur débit séquentiel unitaire en lecture
    # --------------------------------------------------------

    highest_read_candidate = (
        select_maximum_candidate(
            candidates=diversification_pool,
            metric=lambda candidate: safe_float(
                candidate.get("seq_read_mb_s")
            ),
        )
    )

    if highest_read_candidate is not None:
        add_candidate(
            selected_by_id=selected_by_id,
            candidate=highest_read_candidate,
            reason="highest_sequential_read",
        )

    # --------------------------------------------------------
    # 7. Meilleure efficacité énergétique par TiB
    # --------------------------------------------------------

    valid_power_candidates = [
        candidate
        for candidate in diversification_pool
        if math.isfinite(
            power_per_tib(candidate)
        )
    ]

    best_power_candidate = (
        select_minimum_candidate(
            candidates=valid_power_candidates,
            metric=power_per_tib,
        )
    )

    if best_power_candidate is not None:
        add_candidate(
            selected_by_id=selected_by_id,
            candidate=best_power_candidate,
            reason="best_power_per_tib",
        )

    # --------------------------------------------------------
    # Limitation avant remplissage
    #
    # L'ordre d'insertion représente la priorité des règles :
    # global ML, média, coût, capacité, débit, puissance.
    # --------------------------------------------------------

    if len(selected_by_id) > top_k:
        prioritized_candidates = list(
            selected_by_id.items()
        )[:top_k]

        selected_by_id = dict(
            prioritized_candidates
        )

    # --------------------------------------------------------
    # 8. Remplissage avec le classement ML global
    # --------------------------------------------------------

    for candidate in ranked_candidates:
        if len(selected_by_id) >= top_k:
            break

        drive_id = str(candidate["drive_id"])

        # Ne pas ajouter ml_fill à un candidat déjà présent.
        if drive_id in selected_by_id:
            continue

        add_candidate(
            selected_by_id=selected_by_id,
            candidate=candidate,
            reason="ml_fill",
        )

    # --------------------------------------------------------
    # Construction de la liste finale
    # --------------------------------------------------------

    selected_candidates = list(
        selected_by_id.values()
    )

    # Affichage final ordonné selon le rang ML.
    selected_candidates.sort(
        key=lambda candidate: (
            int(candidate["ml_rank"]),
            str(candidate["drive_id"]),
        )
    )

    for diversified_rank, candidate in enumerate(
        selected_candidates,
        start=1,
    ):
        candidate[
            "diversified_rank"
        ] = diversified_rank

    # --------------------------------------------------------
    # Distribution des types de média
    # --------------------------------------------------------

    media_distribution: dict[str, int] = {}

    for candidate in selected_candidates:
        media_type = str(
            candidate.get(
                "media_type",
                "UNKNOWN",
            )
        ).strip().upper()

        media_distribution[media_type] = (
            media_distribution.get(
                media_type,
                0,
            )
            + 1
        )

    specialized_candidates = [
        candidate
        for candidate in selected_candidates
        if any(
            reason != "global_ml_top"
            and reason != "ml_fill"
            for reason in candidate[
                "selection_reasons"
            ]
        )
    ]

    maximum_specialized_ml_rank = (
        max(
            int(candidate["ml_rank"])
            for candidate in specialized_candidates
        )
        if specialized_candidates
        else None
    )

    return {
        "case_id": ranking_result["case_id"],

        "source_candidate_count":
            ranking_result[
                "feasible_candidate_count"
            ],

        "requested_top_k": top_k,

        "selected_count": len(
            selected_candidates
        ),

        "global_top_count":
            global_selection_count,

        "diversification_pool_size":
            diversification_pool_size,

        "maximum_specialized_ml_rank":
            maximum_specialized_ml_rank,

        "media_distribution":
            media_distribution,

        "diversified_candidates":
            selected_candidates,
    }


# ============================================================
# Affichage du Top-K diversifié
# ============================================================

def print_diversified_ost_top_k(
    diversified_result: dict[str, Any],
) -> None:
    """
    Affiche le Top-K OST diversifié et les raisons de sélection.
    """

    candidates = diversified_result[
        "diversified_candidates"
    ]

    print("=" * 150)
    print("TOP-K OST DIVERSIFIÉ AVEC GARDE-FOU ML")

    print(
        "Case ID                  :",
        diversified_result["case_id"],
    )

    print(
        "Candidats faisables      :",
        diversified_result[
            "source_candidate_count"
        ],
    )

    print(
        "Candidats sélectionnés   :",
        diversified_result[
            "selected_count"
        ],
    )

    print(
        "Taille pool diversification :",
        diversified_result[
            "diversification_pool_size"
        ],
    )

    print(
        "Rang ML spécialisé maximal  :",
        diversified_result[
            "maximum_specialized_ml_rank"
        ],
    )

    print(
        "Distribution médias      :",
        diversified_result[
            "media_distribution"
        ],
    )

    print("=" * 150)

    header = (
        f"{'Rang D.':<9}"
        f"{'Rang ML':<9}"
        f"{'Drive ID':<14}"
        f"{'Nom':<27}"
        f"{'Média':<8}"
        f"{'Cap. TiB':>10}"
        f"{'Drives':>9}"
        f"{'Score ML':>13}"
        f"  Raisons"
    )

    print(header)
    print("-" * 150)

    for candidate in candidates:
        reasons = ", ".join(
            candidate["selection_reasons"]
        )

        print(
            f"{candidate['diversified_rank']:<9}"
            f"{candidate['ml_rank']:<9}"
            f"{candidate['drive_id']:<14}"
            f"{candidate['drive_name'][:25]:<27}"
            f"{candidate['media_type']:<8}"
            f"{safe_float(candidate['capacity_tib']):>10.2f}"
            f"{candidate['raw_minimum_drive_count']:>9}"
            f"{candidate['ml_score']:>13.6f}"
            f"  {reasons}"
        )

    print("-" * 150)

    print(
        "Top-K OST diversifié avec garde-fou ML : VALIDÉ"
    )


# ============================================================
# Test local
# ============================================================

def main() -> None:
    """
    Teste le classement OST et la diversification sur
    le premier cas architectural.
    """

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
        raise DiversifiedTopKError(
            "Le dataset architectural est vide."
        )

    if not isinstance(catalog, list):
        raise TypeError(
            "Le catalogue doit être une liste."
        )

    if not catalog:
        raise DiversifiedTopKError(
            "Le catalogue de drives est vide."
        )

    architecture = architectures[0]

    ranking_result = rank_all_ost_candidates(
        architecture=architecture,
        catalog=catalog,
    )

    diversified_result = (
        select_diversified_ost_top_k(
            ranking_result=ranking_result,
            top_k=10,
            global_top_count=4,
            diversification_multiplier=4,
            minimum_diversification_pool_size=40,
        )
    )

    print_diversified_ost_top_k(
        diversified_result=diversified_result
    )


if __name__ == "__main__":
    main()