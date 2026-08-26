from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


# ============================================================
# Chemins
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
RANKING_DIR = SRC_DIR / "ranking"
DIVERSIFIED_TOPK_PATH = (
    RANKING_DIR
    / "diversified_topk.py"
)


# ============================================================
# Chargement explicite
# ============================================================

def _load_module(
    module_name: str,
    module_path: Path,
) -> ModuleType:
    if not module_path.exists():
        raise FileNotFoundError(
            f"Module introuvable : {module_path}"
        )

    for directory in (
        RANKING_DIR,
        SRC_DIR,
    ):
        directory_text = str(
            directory
        )
        if (
            directory_text
            not in sys.path
        ):
            sys.path.insert(
                0,
                directory_text,
            )

    spec = (
        importlib.util
        .spec_from_file_location(
            module_name,
            module_path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise ImportError(
            f"Spec impossible : {module_path}"
        )

    module = (
        importlib.util
        .module_from_spec(spec)
    )

    sys.modules[
        module_name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


topk_module = _load_module(
    "_test_diversified_topk_runtime",
    DIVERSIFIED_TOPK_PATH,
)

select_diversified_ost_top_k = (
    topk_module
    .select_diversified_ost_top_k
)

validate_ranking_result = (
    topk_module
    .validate_ranking_result
)


# ============================================================
# Fixtures synthétiques
# ============================================================

def make_candidate(
    rank: int,
) -> dict[str, Any]:
    """
    Crée 60 candidats :
    - rangs 1..30 : SSD
    - rangs 31..60 : HDD

    Le pool de diversification par défaut de taille 40
    contient donc les deux types de média.
    """

    is_ssd = rank <= 30

    media_type = (
        "SSD"
        if is_ssd
        else "HDD"
    )

    capacity_tib = (
        float(
            4 + rank
        )
        if is_ssd
        else float(
            8 + rank / 2
        )
    )

    price_usd = (
        1200.0
        + rank * 11.0
        if is_ssd
        else 240.0
        + rank * 2.0
    )

    power_w = (
        16.0
        + rank * 0.1
        if is_ssd
        else 7.0
        + rank * 0.05
    )

    seq_read_mb_s = (
        7000.0
        - rank * 15.0
        if is_ssd
        else 280.0
        + rank
    )

    return {
        "drive_id": (
            f"DRV_TEST_{rank:03d}"
        ),
        "drive_name": (
            f"Synthetic Drive {rank}"
        ),
        "media_type": media_type,
        "capacity_tib": capacity_tib,
        "seq_read_mb_s": (
            seq_read_mb_s
        ),
        "price_usd": price_usd,
        "power_w": power_w,
        "ml_score": (
            100.0 - rank
        ),
        "ml_rank": rank,
        "raw_minimum_drive_count": (
            max(
                1,
                100 - rank,
            )
        ),
    }


def make_ranking_result() -> dict[str, Any]:
    candidates = [
        make_candidate(rank)
        for rank in range(
            1,
            61,
        )
    ]

    return {
        "case_id": (
            "REQ_SYNTHETIC_TOPK"
        ),
        "feasible_candidate_count": (
            len(candidates)
        ),
        "ranked_candidates": (
            candidates
        ),
    }


# ============================================================
# Tests contrat
# ============================================================

def test_topk_returns_exactly_ten_unique_candidates() -> None:
    result = (
        select_diversified_ost_top_k(
            ranking_result=(
                make_ranking_result()
            ),
            top_k=10,
            global_top_count=4,
            diversification_multiplier=4,
            minimum_diversification_pool_size=40,
        )
    )

    selected = result[
        "diversified_candidates"
    ]

    ids = [
        candidate["drive_id"]
        for candidate in selected
    ]

    assert len(
        selected
    ) == 10

    assert len(
        ids
    ) == len(
        set(ids)
    )


def test_topk_preserves_global_ml_top_four() -> None:
    ranking = (
        make_ranking_result()
    )

    result = (
        select_diversified_ost_top_k(
            ranking_result=ranking,
            top_k=10,
            global_top_count=4,
            diversification_multiplier=4,
            minimum_diversification_pool_size=40,
        )
    )

    expected = {
        candidate["drive_id"]
        for candidate in ranking[
            "ranked_candidates"
        ][:4]
    }

    selected = {
        candidate["drive_id"]
        for candidate in result[
            "diversified_candidates"
        ]
    }

    assert expected.issubset(
        selected
    )


def test_specialized_candidates_stay_inside_quality_pool() -> None:
    result = (
        select_diversified_ost_top_k(
            ranking_result=(
                make_ranking_result()
            ),
            top_k=10,
            global_top_count=4,
            diversification_multiplier=4,
            minimum_diversification_pool_size=40,
        )
    )

    pool_size = int(
        result[
            "diversification_pool_size"
        ]
    )

    specialized = [
        candidate
        for candidate in result[
            "diversified_candidates"
        ]
        if any(
            reason not in {
                "global_ml_top",
                "ml_fill",
            }
            for reason in candidate[
                "selection_reasons"
            ]
        )
    ]

    assert specialized

    assert all(
        int(
            candidate["ml_rank"]
        ) <= pool_size
        for candidate in specialized
    )


def test_topk_is_deterministic() -> None:
    ranking = (
        make_ranking_result()
    )

    first = (
        select_diversified_ost_top_k(
            ranking_result=ranking,
            top_k=10,
            global_top_count=4,
            diversification_multiplier=4,
            minimum_diversification_pool_size=40,
        )
    )

    second = (
        select_diversified_ost_top_k(
            ranking_result=ranking,
            top_k=10,
            global_top_count=4,
            diversification_multiplier=4,
            minimum_diversification_pool_size=40,
        )
    )

    def signature(
        result: dict[str, Any],
    ) -> list[
        tuple[
            str,
            int,
            tuple[str, ...],
        ]
    ]:
        return [
            (
                str(
                    candidate[
                        "drive_id"
                    ]
                ),
                int(
                    candidate[
                        "ml_rank"
                    ]
                ),
                tuple(
                    str(reason)
                    for reason
                    in candidate[
                        "selection_reasons"
                    ]
                ),
            )
            for candidate
            in result[
                "diversified_candidates"
            ]
        ]

    assert signature(
        first
    ) == signature(
        second
    )


def test_hdd_and_ssd_are_represented_when_present_in_pool() -> None:
    result = (
        select_diversified_ost_top_k(
            ranking_result=(
                make_ranking_result()
            ),
            top_k=10,
            global_top_count=4,
            diversification_multiplier=4,
            minimum_diversification_pool_size=40,
        )
    )

    media = {
        str(
            candidate[
                "media_type"
            ]
        ).upper()
        for candidate
        in result[
            "diversified_candidates"
        ]
    }

    assert "SSD" in media
    assert "HDD" in media


def test_invalid_top_k_is_rejected() -> None:
    with pytest.raises(
        ValueError
    ):
        select_diversified_ost_top_k(
            ranking_result=(
                make_ranking_result()
            ),
            top_k=0,
        )


def test_incomplete_ranking_result_is_rejected() -> None:
    with pytest.raises(
        Exception
    ):
        validate_ranking_result(
            {
                "case_id": "BROKEN",
                "feasible_candidate_count": 1,
                "ranked_candidates": [
                    {
                        "drive_id": (
                            "ONLY_ID"
                        )
                    }
                ],
            }
        )
