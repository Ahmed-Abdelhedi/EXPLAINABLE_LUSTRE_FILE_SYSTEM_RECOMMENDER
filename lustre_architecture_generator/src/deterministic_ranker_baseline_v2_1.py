#!/usr/bin/env python3
"""
Baseline déterministe V2.1 pour le ranking MDT / OST.

But
---
Améliorer uniquement le MDT de la V2, sans toucher à la logique OST.

MDT V2.1
--------
Sépare explicitement :
    - performance IOPS
    - latence
    - endurance
    - fiabilité
    - coût
    - puissance

Les 4 priorités utilisateur restent la base :
    performance_priority
    reliability_priority
    cost_priority
    power_priority

La priorité performance est divisée entre :
    performance IOPS + latence

La priorité fiabilité est divisée entre :
    endurance + fiabilité

Cette division dépend uniquement des exigences métier :
    latency_requirement
    endurance_requirement

Les coefficients sont fixes et monotones, définis avant l'évaluation.
Ils ne sont ni appris ni optimisés sur le teacher.

OST
---
Conserve exactement la logique V2 :
    performance / reliability / cost / power

Aucun entraînement.
Aucune seed.
Aucune utilisation de teacher_score, teacher_rank ou relevance_grade
dans le calcul du score.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_DATA_DIR = BASE_DIR / "output" / "training"

DEFAULT_OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / "deterministic_baseline_v2_1"
)


# ============================================================
# CONTRAT
# ============================================================

GROUP_COLUMN = "case_id"
LABEL_COLUMN = "relevance_grade"

PRIORITY_COLUMNS = [
    "performance_priority",
    "reliability_priority",
    "cost_priority",
    "power_priority",
]

FORBIDDEN_SCORE_COLUMNS = {
    "teacher_score",
    "teacher_rank",
    "relevance_grade",
    "is_teacher_top1",
    "is_teacher_top3",
    "is_teacher_top5",
    "is_teacher_top10",
}


# ============================================================
# MDT V2.1 : REGLES METIER
# ============================================================
#
# La priorité "performance" est partagée entre :
# - IOPS
# - latence
#
# Plus l'exigence de latence est stricte,
# plus la part attribuée à la latence augmente.
#
# Ces valeurs sont FIXES et NON APPRENTES.
# ============================================================

MDT_LATENCY_SHARE = {
    "moderate": 0.20,
    "low": 0.35,
    "very_low": 0.50,
}


# ============================================================
# La priorité "reliability" est partagée entre :
# - endurance
# - fiabilité MTBF
#
# Plus l'endurance requise est stricte,
# plus la part attribuée à l'endurance augmente.
#
# Valeurs FIXES et NON APPRENTES.
# ============================================================

MDT_ENDURANCE_SHARE = {
    "standard": 0.25,
    "medium": 0.40,
    "high": 0.55,
    "critical": 0.70,
}


# ============================================================
# OST V2 : INCHANGE
# ============================================================

OST_BLOCKS = {

    "performance": [
        "capacity_headroom_ratio",
        "read_bandwidth_headroom_ratio",
        "write_bandwidth_headroom_ratio",
    ],

    "reliability": [
        "reliability_margin_ratio",
    ],

    "cost": [
        "raw_drive_cost_usd",
    ],

    "power": [
        "raw_drive_power_w",
    ],
}


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Baseline déterministe V2.1 MDT / OST. "
            "MDT amélioré, OST identique à V2."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


# ============================================================
# CHARGEMENT
# ============================================================

def load_manifest(data_dir: Path):

    path = (
        data_dir
        / "training_dataset_manifest.json"
    )

    if not path.exists():

        matches = list(
            data_dir.rglob(
                "training_dataset_manifest.json"
            )
        )

        if not matches:

            raise FileNotFoundError(
                "training_dataset_manifest.json "
                f"introuvable dans {data_dir}"
            )

        path = matches[0]


    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        manifest = json.load(handle)


    print(
        f"Manifest chargé : {path}"
    )

    return manifest


def find_dataset(
    data_dir: Path,
    declared_name: str,
):

    name = Path(
        declared_name
    ).name


    names = [
        name
    ]


    if name.endswith(
        ".csv.gz"
    ):

        names.append(
            name[:-3]
        )


    elif name.endswith(
        ".csv"
    ):

        names.append(
            name + ".gz"
        )


    for candidate in dict.fromkeys(
        names
    ):

        direct = (
            data_dir
            / candidate
        )

        if direct.exists():

            return direct


        matches = list(
            data_dir.rglob(
                candidate
            )
        )

        if matches:

            return matches[0]


    raise FileNotFoundError(
        f"{declared_name} introuvable "
        f"dans {data_dir}"
    )


# ============================================================
# COLONNES REQUISES
# ============================================================

def required_columns(
    role: str,
):

    common = {
        "split",
        "case_id",
        "drive_id",
        "drive_name",
        "teacher_rank",
        "teacher_score",
        LABEL_COLUMN,
        *PRIORITY_COLUMNS,
    }


    if role == "mdt":

        common.update({

            "latency_requirement",
            "endurance_requirement",

            "required_read_iops",
            "required_write_iops",

            "read_iops_headroom_ratio",
            "write_iops_headroom_ratio",

            "latency_margin",
            "endurance_margin_ratio",
            "reliability_margin_ratio",

            "raw_drive_cost_usd",
            "raw_drive_power_w",
        })


    elif role == "ost":

        for features in (
            OST_BLOCKS.values()
        ):

            common.update(
                features
            )


    else:

        raise ValueError(
            f"Rôle inconnu : {role}"
        )


    return common


def scoring_columns(
    role: str,
):

    columns = set(
        PRIORITY_COLUMNS
    )


    if role == "mdt":

        columns.update({

            "latency_requirement",
            "endurance_requirement",

            "required_read_iops",
            "required_write_iops",

            "read_iops_headroom_ratio",
            "write_iops_headroom_ratio",

            "latency_margin",
            "endurance_margin_ratio",
            "reliability_margin_ratio",

            "raw_drive_cost_usd",
            "raw_drive_power_w",
        })


    else:

        for features in (
            OST_BLOCKS.values()
        ):

            columns.update(
                features
            )


    return columns


def load_test_dataset(
    data_dir: Path,
    manifest: dict,
    role: str,
):

    path = find_dataset(
        data_dir,
        manifest[role]["file"],
    )


    print(
        f"\nChargement {role.upper()} : {path}"
    )


    df = pd.read_csv(
        path,
        compression="infer",
    )


    missing = sorted(
        required_columns(role)
        -
        set(df.columns)
    )


    if missing:

        raise ValueError(
            f"{role.upper()} : "
            f"colonnes absentes : {missing}"
        )


    leakage = (
        scoring_columns(role)
        &
        FORBIDDEN_SCORE_COLUMNS
    )


    if leakage:

        raise ValueError(
            "Fuite de données détectée : "
            f"{sorted(leakage)}"
        )


    # --------------------------------------------------------
    # Exactement le même split test.
    # --------------------------------------------------------

    test_df = (

        df.loc[
            df["split"] == "test"
        ]

        .copy()

        .reset_index(
            drop=True
        )
    )


    if test_df.empty:

        raise ValueError(
            f"{role.upper()} : split test vide."
        )


    split_counts = (

        df[
            [
                "case_id",
                "split",
            ]
        ]

        .drop_duplicates()

        .groupby(
            "case_id"
        )["split"]

        .nunique()
    )


    if int(
        split_counts.max()
    ) != 1:

        raise ValueError(
            f"{role.upper()} : "
            "un case_id apparaît dans plusieurs splits."
        )


    # --------------------------------------------------------
    # Colonnes numériques.
    # --------------------------------------------------------

    numeric_columns = set(
        PRIORITY_COLUMNS
    )


    if role == "mdt":

        numeric_columns.update({

            "required_read_iops",
            "required_write_iops",

            "read_iops_headroom_ratio",
            "write_iops_headroom_ratio",

            "latency_margin",
            "endurance_margin_ratio",
            "reliability_margin_ratio",

            "raw_drive_cost_usd",
            "raw_drive_power_w",
        })


    else:

        for features in (
            OST_BLOCKS.values()
        ):

            numeric_columns.update(
                features
            )


    for column in numeric_columns:

        test_df[column] = (
            pd.to_numeric(
                test_df[column],
                errors="coerce",
            )
        )


    test_df[
        "teacher_rank"
    ] = (

        pd.to_numeric(
            test_df[
                "teacher_rank"
            ],
            errors="raise",
        )

        .astype(int)
    )


    test_df[
        LABEL_COLUMN
    ] = (

        pd.to_numeric(
            test_df[
                LABEL_COLUMN
            ],
            errors="raise",
        )

        .astype(int)
    )


    print(
        f"{role.upper()} : "
        f"{len(test_df)} lignes test"
    )


    print(
        f"{role.upper()} : "
        f"{test_df['case_id'].nunique()} case_id"
    )


    return test_df


# ============================================================
# NORMALISATION MIN-MAX PAR CASE_ID
# ============================================================

def minmax_normalize(
    series: pd.Series,
    higher_is_better: bool,
):

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).astype(float)


    finite = values[
        np.isfinite(values)
    ]


    if finite.empty:

        normalized = pd.Series(
            0.5,
            index=series.index,
            dtype=float,
        )

        return normalized


    minimum = float(
        finite.min()
    )

    maximum = float(
        finite.max()
    )


    if math.isclose(
        minimum,
        maximum,
        abs_tol=1e-15,
    ):

        normalized = pd.Series(
            0.5,
            index=series.index,
            dtype=float,
        )


    else:

        normalized = (

            (values - minimum)

            /

            (maximum - minimum)
        )


        normalized = (

            normalized

            .fillna(0.5)

            .clip(
                0.0,
                1.0,
            )
        )


    if not higher_is_better:

        normalized = (
            1.0
            -
            normalized
        )


    return normalized


# ============================================================
# PRIORITES DE BASE
# ============================================================

def base_priority_weights(
    group: pd.DataFrame,
):

    first = group.iloc[0]


    raw = {

        "performance":
            float(
                first[
                    "performance_priority"
                ]
            ),

        "reliability":
            float(
                first[
                    "reliability_priority"
                ]
            ),

        "cost":
            float(
                first[
                    "cost_priority"
                ]
            ),

        "power":
            float(
                first[
                    "power_priority"
                ]
            ),
    }


    cleaned = {}


    for key, value in raw.items():

        if (
            not np.isfinite(value)
            or
            value < 0
        ):

            cleaned[key] = 0.0

        else:

            cleaned[key] = value


    total = sum(
        cleaned.values()
    )


    if total <= 0:

        weights = {
            "performance": 0.25,
            "reliability": 0.25,
            "cost": 0.25,
            "power": 0.25,
        }


    else:

        weights = {

            key:
                value
                /
                total

            for key, value
            in cleaned.items()
        }


    return (
        raw,
        weights,
    )


# ============================================================
# MDT V2.1 : POIDS 6 DIMENSIONS
# ============================================================

def mdt_v21_weights(
    group: pd.DataFrame,
):

    (
        raw_priorities,
        base_weights,
    ) = base_priority_weights(
        group
    )


    first = group.iloc[0]


    latency_requirement = str(
        first[
            "latency_requirement"
        ]
    ).strip()


    endurance_requirement = str(
        first[
            "endurance_requirement"
        ]
    ).strip()


    if (
        latency_requirement
        not in
        MDT_LATENCY_SHARE
    ):

        raise ValueError(
            "latency_requirement non supporté : "
            f"{latency_requirement!r}"
        )


    if (
        endurance_requirement
        not in
        MDT_ENDURANCE_SHARE
    ):

        raise ValueError(
            "endurance_requirement non supporté : "
            f"{endurance_requirement!r}"
        )


    latency_share = (
        MDT_LATENCY_SHARE[
            latency_requirement
        ]
    )


    endurance_share = (
        MDT_ENDURANCE_SHARE[
            endurance_requirement
        ]
    )


    performance_weight = (

        base_weights[
            "performance"
        ]

        *

        (
            1.0
            -
            latency_share
        )
    )


    latency_weight = (

        base_weights[
            "performance"
        ]

        *

        latency_share
    )


    endurance_weight = (

        base_weights[
            "reliability"
        ]

        *

        endurance_share
    )


    reliability_weight = (

        base_weights[
            "reliability"
        ]

        *

        (
            1.0
            -
            endurance_share
        )
    )


    weights = {

        "performance":
            performance_weight,

        "latency":
            latency_weight,

        "endurance":
            endurance_weight,

        "reliability":
            reliability_weight,

        "cost":
            base_weights[
                "cost"
            ],

        "power":
            base_weights[
                "power"
            ],
    }


    # Sécurité numérique.
    total = sum(
        weights.values()
    )


    weights = {

        key:
            value
            /
            total

        for key, value
        in weights.items()
    }


    return {

        "raw_priorities":
            raw_priorities,

        "base_weights":
            base_weights,

        "weights":
            weights,

        "latency_requirement":
            latency_requirement,

        "latency_share":
            latency_share,

        "endurance_requirement":
            endurance_requirement,

        "endurance_share":
            endurance_share,
    }


# ============================================================
# MDT V2.1 : SCORE D'UN CASE
# ============================================================

def score_mdt_case_v21(
    group: pd.DataFrame,
):

    result = group.copy()


    weight_info = (
        mdt_v21_weights(
            group
        )
    )


    weights = (
        weight_info[
            "weights"
        ]
    )


    # --------------------------------------------------------
    # IOPS lecture / écriture
    #
    # La composition du score performance suit
    # le ratio IOPS réellement demandé par le workload.
    # --------------------------------------------------------

    read_required = float(
        result.iloc[0][
            "required_read_iops"
        ]
    )


    write_required = float(
        result.iloc[0][
            "required_write_iops"
        ]
    )


    total_iops = (
        max(
            read_required,
            0.0,
        )

        +

        max(
            write_required,
            0.0,
        )
    )


    if total_iops <= 0:

        read_share = 0.5
        write_share = 0.5


    else:

        read_share = (
            max(
                read_required,
                0.0,
            )
            /
            total_iops
        )


        write_share = (
            max(
                write_required,
                0.0,
            )
            /
            total_iops
        )


    result[
        "v21_norm_read_iops"
    ] = minmax_normalize(

        result[
            "read_iops_headroom_ratio"
        ],

        higher_is_better=True,
    )


    result[
        "v21_norm_write_iops"
    ] = minmax_normalize(

        result[
            "write_iops_headroom_ratio"
        ],

        higher_is_better=True,
    )


    result[
        "v21_performance_score"
    ] = (

        read_share
        *
        result[
            "v21_norm_read_iops"
        ]

        +

        write_share
        *
        result[
            "v21_norm_write_iops"
        ]
    )


    # --------------------------------------------------------
    # Latence séparée
    # --------------------------------------------------------

    result[
        "v21_latency_score"
    ] = minmax_normalize(

        result[
            "latency_margin"
        ],

        higher_is_better=True,
    )


    # --------------------------------------------------------
    # Endurance séparée
    # --------------------------------------------------------

    result[
        "v21_endurance_score"
    ] = minmax_normalize(

        result[
            "endurance_margin_ratio"
        ],

        higher_is_better=True,
    )


    # --------------------------------------------------------
    # Fiabilité séparée
    # --------------------------------------------------------

    result[
        "v21_reliability_score"
    ] = minmax_normalize(

        result[
            "reliability_margin_ratio"
        ],

        higher_is_better=True,
    )


    # --------------------------------------------------------
    # Coût
    # --------------------------------------------------------

    result[
        "v21_cost_score"
    ] = minmax_normalize(

        result[
            "raw_drive_cost_usd"
        ],

        higher_is_better=False,
    )


    # --------------------------------------------------------
    # Puissance
    # --------------------------------------------------------

    result[
        "v21_power_score"
    ] = minmax_normalize(

        result[
            "raw_drive_power_w"
        ],

        higher_is_better=False,
    )


    # --------------------------------------------------------
    # Poids dans les prédictions
    # --------------------------------------------------------

    for key, value in (
        weights.items()
    ):

        result[
            f"v21_weight_{key}"
        ] = value


    result[
        "v21_read_iops_share"
    ] = read_share


    result[
        "v21_write_iops_share"
    ] = write_share


    # --------------------------------------------------------
    # SCORE FINAL MDT V2.1
    # --------------------------------------------------------

    result[
        "deterministic_v21_score"
    ] = (

        weights[
            "performance"
        ]
        *
        result[
            "v21_performance_score"
        ]

        +

        weights[
            "latency"
        ]
        *
        result[
            "v21_latency_score"
        ]

        +

        weights[
            "endurance"
        ]
        *
        result[
            "v21_endurance_score"
        ]

        +

        weights[
            "reliability"
        ]
        *
        result[
            "v21_reliability_score"
        ]

        +

        weights[
            "cost"
        ]
        *
        result[
            "v21_cost_score"
        ]

        +

        weights[
            "power"
        ]
        *
        result[
            "v21_power_score"
        ]
    )


    result = (

        result

        .sort_values(
            [
                "deterministic_v21_score",
                "drive_id",
            ],
            ascending=[
                False,
                True,
            ],
        )

        .reset_index(
            drop=True
        )
    )


    result[
        "deterministic_v21_rank"
    ] = np.arange(
        1,
        len(result) + 1,
    )


    audit = {

        "role":
            "mdt",

        "case_id":
            group.iloc[0][
                "case_id"
            ],

        "latency_requirement":
            weight_info[
                "latency_requirement"
            ],

        "latency_share_of_performance":
            weight_info[
                "latency_share"
            ],

        "endurance_requirement":
            weight_info[
                "endurance_requirement"
            ],

        "endurance_share_of_reliability":
            weight_info[
                "endurance_share"
            ],

        "read_iops_share":
            read_share,

        "write_iops_share":
            write_share,
    }


    for key, value in (
        weight_info[
            "raw_priorities"
        ].items()
    ):

        audit[
            f"{key}_priority_raw"
        ] = value


    for key, value in (
        weights.items()
    ):

        audit[
            f"{key}_weight"
        ] = value


    audit[
        "weight_sum"
    ] = sum(
        weights.values()
    )


    return (
        result,
        audit,
    )


# ============================================================
# OST V2 : LOGIQUE INCHANGEE
# ============================================================

def score_ost_case_v2(
    group: pd.DataFrame,
):

    result = group.copy()


    (
        raw_priorities,
        weights,
    ) = base_priority_weights(
        group
    )


    block_scores = {}


    for block_name, features in (
        OST_BLOCKS.items()
    ):

        normalized_columns = []


        for feature in features:

            column = (
                f"v21_norm_{feature}"
            )


            result[
                column
            ] = minmax_normalize(

                result[
                    feature
                ],

                higher_is_better=(
                    block_name
                    not in
                    {
                        "cost",
                        "power",
                    }
                ),
            )


            normalized_columns.append(
                column
            )


        score_column = (
            f"v21_{block_name}_score"
        )


        result[
            score_column
        ] = (

            result[
                normalized_columns
            ]

            .mean(
                axis=1
            )
        )


        block_scores[
            block_name
        ] = score_column


        result[
            f"v21_weight_{block_name}"
        ] = weights[
            block_name
        ]


    result[
        "deterministic_v21_score"
    ] = (

        weights[
            "performance"
        ]
        *
        result[
            block_scores[
                "performance"
            ]
        ]

        +

        weights[
            "reliability"
        ]
        *
        result[
            block_scores[
                "reliability"
            ]
        ]

        +

        weights[
            "cost"
        ]
        *
        result[
            block_scores[
                "cost"
            ]
        ]

        +

        weights[
            "power"
        ]
        *
        result[
            block_scores[
                "power"
            ]
        ]
    )


    result = (

        result

        .sort_values(
            [
                "deterministic_v21_score",
                "drive_id",
            ],
            ascending=[
                False,
                True,
            ],
        )

        .reset_index(
            drop=True
        )
    )


    result[
        "deterministic_v21_rank"
    ] = np.arange(
        1,
        len(result) + 1,
    )


    audit = {

        "role":
            "ost",

        "case_id":
            group.iloc[0][
                "case_id"
            ],
    }


    for key, value in (
        raw_priorities.items()
    ):

        audit[
            f"{key}_priority_raw"
        ] = value


    for key, value in (
        weights.items()
    ):

        audit[
            f"{key}_weight"
        ] = value


    audit[
        "weight_sum"
    ] = sum(
        weights.values()
    )


    return (
        result,
        audit,
    )


# ============================================================
# SCORE DATASET
# ============================================================

def score_dataset(
    test_df: pd.DataFrame,
    role: str,
):

    scored_frames = []

    audit_rows = []


    for _, group in (

        test_df.groupby(
            GROUP_COLUMN,
            sort=False,
        )

    ):

        if role == "mdt":

            (
                scored,
                audit,
            ) = score_mdt_case_v21(
                group
            )


        else:

            (
                scored,
                audit,
            ) = score_ost_case_v2(
                group
            )


        scored_frames.append(
            scored
        )


        audit_rows.append(
            audit
        )


    result = pd.concat(
        scored_frames,
        ignore_index=True,
    )


    result[
        "role"
    ] = role


    result[
        "model"
    ] = (
        "DeterministicBaselineV2.1"
    )


    return (
        result,
        pd.DataFrame(
            audit_rows
        ),
    )


# ============================================================
# METRIQUES
# ============================================================

def dcg_at_k(
    relevances,
    k: int,
):

    values = np.asarray(
        relevances,
        dtype=float,
    )[:k]


    if len(values) == 0:

        return 0.0


    gains = (

        np.power(
            2.0,
            values,
        )

        -

        1.0
    )


    discounts = np.log2(

        np.arange(
            2,
            len(values) + 2,
        )
    )


    return float(

        np.sum(
            gains
            /
            discounts
        )
    )


def ndcg_at_k(
    group: pd.DataFrame,
    k: int,
):

    predicted = (

        group

        .sort_values(
            [
                "deterministic_v21_score",
                "drive_id",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )


    actual = dcg_at_k(

        predicted[
            LABEL_COLUMN
        ].to_numpy(),

        k,
    )


    ideal = dcg_at_k(

        np.sort(
            group[
                LABEL_COLUMN
            ].to_numpy()
        )[::-1],

        k,
    )


    if ideal <= 0:

        return 1.0


    return (
        actual
        /
        ideal
    )


def evaluate_role(
    scored_df: pd.DataFrame,
    role: str,
):

    rows = []


    for case_id, group in (

        scored_df.groupby(
            GROUP_COLUMN,
            sort=False,
        )

    ):


        predicted = (

            group

            .sort_values(
                [
                    "deterministic_v21_score",
                    "drive_id",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
        )


        teacher = (

            group

            .sort_values(
                [
                    "teacher_rank",
                    "drive_id",
                ],
                ascending=[
                    True,
                    True,
                ],
            )
        )


        pred_top1 = (
            predicted.iloc[0][
                "drive_id"
            ]
        )


        teacher_top1 = (
            teacher.iloc[0][
                "drive_id"
            ]
        )


        pred3 = set(
            predicted.head(3)[
                "drive_id"
            ]
        )

        teacher3 = set(
            teacher.head(3)[
                "drive_id"
            ]
        )


        pred5 = set(
            predicted.head(5)[
                "drive_id"
            ]
        )

        teacher5 = set(
            teacher.head(5)[
                "drive_id"
            ]
        )


        pred10 = set(
            predicted.head(10)[
                "drive_id"
            ]
        )

        teacher10 = set(
            teacher.head(10)[
                "drive_id"
            ]
        )


        intersection10 = len(
            pred10
            &
            teacher10
        )


        union10 = len(
            pred10
            |
            teacher10
        )


        rows.append({

            "role":
                role,

            "case_id":
                case_id,

            "candidate_count":
                int(
                    len(group)
                ),

            "ndcg_at_5":
                ndcg_at_k(
                    group,
                    5,
                ),

            "ndcg_at_10":
                ndcg_at_k(
                    group,
                    10,
                ),

            "top1_agreement":
                int(
                    pred_top1
                    ==
                    teacher_top1
                ),

            "top3_overlap":
                (
                    len(
                        pred3
                        &
                        teacher3
                    )
                    /
                    3.0
                ),

            "recall_at_5":
                (
                    len(
                        pred5
                        &
                        teacher5
                    )
                    /
                    max(
                        1,
                        len(
                            teacher5
                        ),
                    )
                ),

            "recall_at_10":
                (
                    intersection10
                    /
                    max(
                        1,
                        len(
                            teacher10
                        ),
                    )
                ),

            "top10_jaccard":
                (
                    intersection10
                    /
                    union10
                )
                if union10
                else 1.0,

            "predicted_top1_teacher_rank":
                int(
                    predicted.iloc[0][
                        "teacher_rank"
                    ]
                ),

            "predicted_top1_drive_id":
                pred_top1,

            "teacher_top1_drive_id":
                teacher_top1,
        })


    case_metrics = pd.DataFrame(
        rows
    )


    summary = {

        "model":
            "DeterministicBaselineV2.1",

        "role":
            role,

        "case_count":
            int(
                case_metrics[
                    "case_id"
                ].nunique()
            ),

        "ndcg_at_5":
            float(
                case_metrics[
                    "ndcg_at_5"
                ].mean()
            ),

        "ndcg_at_10":
            float(
                case_metrics[
                    "ndcg_at_10"
                ].mean()
            ),

        "top1_agreement":
            float(
                case_metrics[
                    "top1_agreement"
                ].mean()
            ),

        "top3_overlap":
            float(
                case_metrics[
                    "top3_overlap"
                ].mean()
            ),

        "recall_at_5":
            float(
                case_metrics[
                    "recall_at_5"
                ].mean()
            ),

        "recall_at_10":
            float(
                case_metrics[
                    "recall_at_10"
                ].mean()
            ),

        "top10_jaccard":
            float(
                case_metrics[
                    "top10_jaccard"
                ].mean()
            ),

        "predicted_top1_teacher_rank":
            float(
                case_metrics[
                    "predicted_top1_teacher_rank"
                ].mean()
            ),
    }


    return (
        summary,
        case_metrics,
    )


# ============================================================
# COMPARAISON OPTIONNELLE V1 / V2 / V2.1
# ============================================================

def build_optional_version_comparison(
    output_dir: Path,
    current_metrics: pd.DataFrame,
):

    frames = []


    v1_path = (
        BASE_DIR
        / "output"
        / "deterministic_baseline"
        / "deterministic_global_metrics.csv"
    )


    v2_path = (
        BASE_DIR
        / "output"
        / "deterministic_baseline_v2"
        / "deterministic_v2_global_metrics.csv"
    )


    if v1_path.exists():

        v1 = pd.read_csv(
            v1_path
        )

        v1[
            "baseline_version"
        ] = "V1"

        frames.append(
            v1
        )


    if v2_path.exists():

        v2 = pd.read_csv(
            v2_path
        )

        v2[
            "baseline_version"
        ] = "V2"

        frames.append(
            v2
        )


    current = (
        current_metrics.copy()
    )


    current[
        "baseline_version"
    ] = "V2.1"


    frames.append(
        current
    )


    comparison = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


    path = (
        output_dir
        /
        "deterministic_versions_comparison.csv"
    )


    comparison.to_csv(
        path,
        index=False,
    )


    return path


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()


    data_dir = (
        args.data_dir
        .resolve()
    )


    output_dir = (
        args.output_dir
        .resolve()
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    print(
        "\n"
        "============================================"
    )

    print(
        " BASELINE DETERMINISTE V2.1 MDT / OST"
    )

    print(
        " MDT: performance/latence/endurance/"
        "fiabilite/cout/puissance"
    )

    print(
        " OST: logique V2 inchangee"
    )

    print(
        "============================================"
    )


    print(
        f"Data   : {data_dir}"
    )


    print(
        f"Output : {output_dir}"
    )


    manifest = load_manifest(
        data_dir
    )


    summaries = []

    case_frames = []

    prediction_frames = []

    audit_frames = []


    for role in [
        "mdt",
        "ost",
    ]:

        start = time.perf_counter()


        test_df = load_test_dataset(
            data_dir,
            manifest,
            role,
        )


        (
            scored_df,
            audit_df,
        ) = score_dataset(
            test_df,
            role,
        )


        (
            summary,
            case_metrics,
        ) = evaluate_role(
            scored_df,
            role,
        )


        elapsed = (
            time.perf_counter()
            -
            start
        )


        summary[
            "execution_seconds"
        ] = elapsed


        summaries.append(
            summary
        )


        case_frames.append(
            case_metrics
        )


        prediction_frames.append(
            scored_df
        )


        audit_frames.append(
            audit_df
        )


        print(
            "\n"
            "--------------------------------------------"
        )

        print(
            f"{role.upper()} V2.1 RESULTS"
        )

        print(
            "--------------------------------------------"
        )


        print(
            f"Cases       : "
            f"{summary['case_count']}"
        )

        print(
            f"NDCG@5      : "
            f"{summary['ndcg_at_5']:.6f}"
        )

        print(
            f"NDCG@10     : "
            f"{summary['ndcg_at_10']:.6f}"
        )

        print(
            f"Top-1       : "
            f"{summary['top1_agreement']:.4f}"
        )

        print(
            f"Top-3       : "
            f"{summary['top3_overlap']:.4f}"
        )

        print(
            f"Recall@5    : "
            f"{summary['recall_at_5']:.6f}"
        )

        print(
            f"Recall@10   : "
            f"{summary['recall_at_10']:.6f}"
        )

        print(
            f"Jaccard@10  : "
            f"{summary['top10_jaccard']:.6f}"
        )

        print(
            f"Mean rank   : "
            f"{summary['predicted_top1_teacher_rank']:.4f}"
        )

        print(
            f"Temps       : "
            f"{elapsed:.3f} s"
        )


    global_metrics = pd.DataFrame(
        summaries
    )


    case_metrics = pd.concat(
        case_frames,
        ignore_index=True,
    )


    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
        sort=False,
    )


    audit = pd.concat(
        audit_frames,
        ignore_index=True,
        sort=False,
    )


    # ========================================================
    # OUTPUTS
    # ========================================================

    global_path = (
        output_dir
        /
        "deterministic_v21_global_metrics.csv"
    )


    case_path = (
        output_dir
        /
        "deterministic_v21_case_metrics.csv"
    )


    predictions_path = (
        output_dir
        /
        "deterministic_v21_predictions.csv.gz"
    )


    audit_path = (
        output_dir
        /
        "deterministic_v21_weight_audit.csv"
    )


    config_path = (
        output_dir
        /
        "deterministic_v21_config.json"
    )


    zip_path = (
        output_dir
        /
        "deterministic_v21_artifacts.zip"
    )


    global_metrics.to_csv(
        global_path,
        index=False,
    )


    case_metrics.to_csv(
        case_path,
        index=False,
    )


    predictions.to_csv(
        predictions_path,
        index=False,
        compression="gzip",
    )


    audit.to_csv(
        audit_path,
        index=False,
    )


    comparison_path = (
        build_optional_version_comparison(
            output_dir,
            global_metrics,
        )
    )


    config = {

        "algorithm":
            "Deterministic Baseline V2.1",

        "training":
            False,

        "seed":
            None,

        "split":
            "test",

        "group_column":
            GROUP_COLUMN,

        "evaluation_label":
            LABEL_COLUMN,

        "mdt_logic": {

            "dimensions": [
                "performance",
                "latency",
                "endurance",
                "reliability",
                "cost",
                "power",
            ],

            "performance_score":
                (
                    "read/write IOPS headroom, "
                    "weighted by requested read/write IOPS share"
                ),

            "latency_score":
                "latency_margin",

            "endurance_score":
                "endurance_margin_ratio",

            "reliability_score":
                "reliability_margin_ratio",

            "cost_score":
                "inverse min-max raw_drive_cost_usd",

            "power_score":
                "inverse min-max raw_drive_power_w",

            "latency_share_rule":
                MDT_LATENCY_SHARE,

            "endurance_share_rule":
                MDT_ENDURANCE_SHARE,

            "note":
                (
                    "Fixed business rules defined before evaluation; "
                    "not learned from teacher labels."
                ),
        },

        "ost_logic":
            (
                "Unchanged from deterministic V2: "
                "performance/reliability/cost/power."
            ),

        "normalization":
            "Min-max independently inside each case_id",

        "forbidden_score_columns":
            sorted(
                FORBIDDEN_SCORE_COLUMNS
            ),
    }


    with config_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            config,
            handle,
            ensure_ascii=False,
            indent=2,
        )


    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:

        for path in [
            global_path,
            case_path,
            predictions_path,
            audit_path,
            comparison_path,
            config_path,
        ]:

            archive.write(
                path,
                arcname=path.name,
            )


    print(
        "\n"
        "============================================"
    )

    print(
        " RESULTATS GLOBAUX V2.1"
    )

    print(
        "============================================"
    )


    print(
        global_metrics.to_string(
            index=False
        )
    )


    print(
        "\nFichiers générés :"
    )


    for path in [
        global_path,
        case_path,
        predictions_path,
        audit_path,
        comparison_path,
        config_path,
        zip_path,
    ]:

        print(
            f"- {path}"
        )


if __name__ == "__main__":

    main()