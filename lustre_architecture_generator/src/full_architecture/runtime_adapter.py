from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable

from .handoff_contract import (
    ArchitectureHandoffError,
    assemble_architecture_handoff,
    assert_valid_architecture_handoff,
)


PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
RANKING_DIR = SRC_DIR / "ranking"

RankingCallable = Callable[..., dict[str, Any]]


class ArchitectureRuntimeAdapterError(RuntimeError):
    """Erreur pendant l'adaptation Ranking -> Full Architecture."""


def _prepare_import_paths() -> None:
    for directory in (RANKING_DIR, SRC_DIR):
        text = str(directory)
        if text not in sys.path:
            sys.path.insert(0, text)


def load_official_ranking_functions() -> tuple[
    RankingCallable,
    RankingCallable,
    RankingCallable,
]:
    """Charge les fonctions officielles sans modifier le code Ranking frozen."""

    _prepare_import_paths()

    try:
        mdt_module = importlib.import_module("mdt_ranker_inference")
        ost_module = importlib.import_module("ost_ranker_inference")
        topk_module = importlib.import_module("diversified_topk")
    except Exception as error:
        raise ArchitectureRuntimeAdapterError(
            "Chargement du runtime Ranking impossible : "
            f"{type(error).__name__}: {error}"
        ) from error

    functions = (
        getattr(mdt_module, "rank_all_mdt_candidates", None),
        getattr(ost_module, "rank_all_ost_candidates", None),
        getattr(topk_module, "select_diversified_ost_top_k", None),
    )

    names = (
        "rank_all_mdt_candidates",
        "rank_all_ost_candidates",
        "select_diversified_ost_top_k",
    )

    for name, function in zip(names, functions):
        if not callable(function):
            raise ArchitectureRuntimeAdapterError(
                f"Fonction Ranking introuvable : {name}"
            )

    return functions  # type: ignore[return-value]


def _validate_case_id(
    expected: str,
    result: dict[str, Any],
    label: str,
) -> None:
    actual = result.get("case_id")
    if actual != expected:
        raise ArchitectureRuntimeAdapterError(
            f"{label}: case_id={actual!r}, attendu={expected!r}."
        )


def build_runtime_handoff(
    *,
    architecture: dict[str, Any],
    catalog: list[dict[str, Any]],
    top_k: int = 10,
    global_top_count: int = 4,
    diversification_multiplier: int = 4,
    minimum_diversification_pool_size: int = 40,
    mdt_ranker: RankingCallable | None = None,
    ost_ranker: RankingCallable | None = None,
    ost_topk_selector: RankingCallable | None = None,
) -> dict[str, Any]:
    """
    Exécute le Ranking réel puis produit le contrat H1.

    Aucun RAID, serveur, contrôleur, enclosure, réseau, BOM ou Beam Search
    n'est choisi ici.
    """

    if not isinstance(architecture, dict):
        raise TypeError("architecture doit être un objet JSON.")

    if not isinstance(catalog, list) or not catalog:
        raise ArchitectureRuntimeAdapterError(
            "catalog doit être une liste non vide."
        )

    if top_k <= 0:
        raise ValueError("top_k doit être > 0.")

    if global_top_count < 0:
        raise ValueError("global_top_count doit être >= 0.")

    if diversification_multiplier <= 0:
        raise ValueError("diversification_multiplier doit être > 0.")

    if minimum_diversification_pool_size <= 0:
        raise ValueError(
            "minimum_diversification_pool_size doit être > 0."
        )

    case_id = architecture.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ArchitectureRuntimeAdapterError(
            "architecture.case_id doit être une chaîne non vide."
        )
    case_id = case_id.strip()

    if (
        mdt_ranker is None
        or ost_ranker is None
        or ost_topk_selector is None
    ):
        official_mdt, official_ost, official_topk = (
            load_official_ranking_functions()
        )
        mdt_ranker = mdt_ranker or official_mdt
        ost_ranker = ost_ranker or official_ost
        ost_topk_selector = ost_topk_selector or official_topk

    try:
        mdt_result = mdt_ranker(
            architecture=architecture,
            catalog=catalog,
        )
    except Exception as error:
        raise ArchitectureRuntimeAdapterError(
            f"{case_id}: échec MDT Ranker : "
            f"{type(error).__name__}: {error}"
        ) from error

    if not isinstance(mdt_result, dict):
        raise ArchitectureRuntimeAdapterError(
            f"{case_id}: résultat MDT invalide."
        )
    _validate_case_id(case_id, mdt_result, "MDT Ranker")

    try:
        ost_result = ost_ranker(
            architecture=architecture,
            catalog=catalog,
        )
    except Exception as error:
        raise ArchitectureRuntimeAdapterError(
            f"{case_id}: échec OST Ranker : "
            f"{type(error).__name__}: {error}"
        ) from error

    if not isinstance(ost_result, dict):
        raise ArchitectureRuntimeAdapterError(
            f"{case_id}: résultat OST invalide."
        )
    _validate_case_id(case_id, ost_result, "OST Ranker")

    try:
        diversified_result = ost_topk_selector(
            ranking_result=ost_result,
            top_k=top_k,
            global_top_count=global_top_count,
            diversification_multiplier=diversification_multiplier,
            minimum_diversification_pool_size=(
                minimum_diversification_pool_size
            ),
        )
    except Exception as error:
        raise ArchitectureRuntimeAdapterError(
            f"{case_id}: échec Top-K OST diversifié : "
            f"{type(error).__name__}: {error}"
        ) from error

    if not isinstance(diversified_result, dict):
        raise ArchitectureRuntimeAdapterError(
            f"{case_id}: résultat Top-K OST invalide."
        )
    _validate_case_id(case_id, diversified_result, "OST Top-K")

    try:
        handoff = assemble_architecture_handoff(
            architecture=architecture,
            catalog=catalog,
            mdt_ranking_result=mdt_result,
            ost_ranking_result=ost_result,
            diversified_ost_result=diversified_result,
            top_k=top_k,
        )
        assert_valid_architecture_handoff(handoff)
    except ArchitectureHandoffError:
        raise
    except Exception as error:
        raise ArchitectureRuntimeAdapterError(
            f"{case_id}: construction du handoff impossible : "
            f"{type(error).__name__}: {error}"
        ) from error

    return handoff
