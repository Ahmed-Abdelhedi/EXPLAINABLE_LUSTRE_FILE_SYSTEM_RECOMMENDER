#!/usr/bin/env python3
"""
Baseline déterministe V2 pour le ranking MDT / OST.

Différence avec V1
------------------
V1:
    - 6 critères
    - poids égaux

V2:
    - construit 4 blocs métier :
        performance
        reliability
        cost
        power
    - utilise les priorités utilisateur déjà présentes dans les datasets :
        performance_priority
        reliability_priority
        cost_priority
        power_priority
    - normalise ces 4 priorités par case_id
    - calcule un score final pondéré

Aucun entraînement.
Aucune seed.
Aucune utilisation de teacher_score / teacher_rank / relevance_grade
dans le calcul du score.

Le split utilisé pour l'évaluation reste exactement "test".
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
    / "deterministic_baseline_v2"
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
# DEFINITION DES BLOCS METIER
# ============================================================
#
# Important :
# ces blocs sont volontairement plus simples que les modèles ML.
#
# Ils ne recopient pas la formule du teacher.
# Ils regroupent seulement quelques features métier cohérentes.
#
# ============================================================

ROLE_BLOCKS = {

    # --------------------------------------------------------
    # MDT
    # --------------------------------------------------------
    "mdt": {

        "performance": [
            "capacity_headroom_ratio",
            "read_iops_headroom_ratio",
            "write_iops_headroom_ratio",
            "latency_margin",
        ],

        "reliability": [
            "endurance_margin_ratio",
            "reliability_margin_ratio",
        ],

        "cost": [
            "raw_drive_cost_usd",
        ],

        "power": [
            "raw_drive_power_w",
        ],
    },


    # --------------------------------------------------------
    # OST
    # --------------------------------------------------------
    "ost": {

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
    },
}


# Ces blocs sont "plus petit = meilleur".
LOWER_IS_BETTER_BLOCKS = {
    "cost",
    "power",
}


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Baseline déterministe V2 de ranking MDT / OST "
            "pondérée par les priorités utilisateur."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=(
            "Dossier contenant training_dataset_manifest.json "
            "et les datasets MDT/OST."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Dossier de sortie.",
    )

    return parser.parse_args()


# ============================================================
# CHARGEMENT
# ============================================================

def load_manifest(data_dir: Path):

    manifest_path = (
        data_dir
        / "training_dataset_manifest.json"
    )

    if not manifest_path.exists():

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

        manifest_path = matches[0]


    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        manifest = json.load(handle)


    print(
        f"Manifest chargé : {manifest_path}"
    )

    return manifest


def find_dataset(
    data_dir: Path,
    declared_name: str,
):

    declared_name = Path(
        declared_name
    ).name

    possible_names = [
        declared_name
    ]


    if declared_name.endswith(
        ".csv.gz"
    ):

        possible_names.append(
            declared_name[:-3]
        )


    elif declared_name.endswith(
        ".csv"
    ):

        possible_names.append(
            declared_name + ".gz"
        )


    for filename in dict.fromkeys(
        possible_names
    ):

        direct = (
            data_dir
            / filename
        )

        if direct.exists():

            return direct


        matches = list(
            data_dir.rglob(
                filename
            )
        )

        if matches:

            return matches[0]


    raise FileNotFoundError(
        f"Dataset introuvable : {declared_name}\n"
        f"Noms testés : {possible_names}"
    )


def get_required_columns(
    role: str,
):

    columns = {
        "split",
        "case_id",
        "drive_id",
        "drive_name",
        "teacher_rank",
        "teacher_score",
        LABEL_COLUMN,
        *PRIORITY_COLUMNS,
    }


    for feature_list in (
        ROLE_BLOCKS[role].values()
    ):

        columns.update(
            feature_list
        )


    return columns


def load_test_dataset(
    data_dir: Path,
    manifest: dict,
    role: str,
):

    dataset_path = find_dataset(
        data_dir,
        manifest[role]["file"],
    )


    print(
        f"\nChargement {role.upper()} : "
        f"{dataset_path}"
    )


    df = pd.read_csv(
        dataset_path,
        compression="infer",
    )


    required_columns = (
        get_required_columns(
            role
        )
    )


    missing = sorted(
        required_columns
        -
        set(df.columns)
    )


    if missing:

        raise ValueError(
            f"{role.upper()} : "
            f"colonnes absentes : "
            f"{missing}"
        )


    # --------------------------------------------------------
    # Anti-fuite
    # --------------------------------------------------------

    scoring_columns = set(
        PRIORITY_COLUMNS
    )


    for values in (
        ROLE_BLOCKS[role].values()
    ):

        scoring_columns.update(
            values
        )


    leakage = (
        scoring_columns
        &
        FORBIDDEN_SCORE_COLUMNS
    )


    if leakage:

        raise ValueError(
            "Fuite de données détectée : "
            f"{sorted(leakage)}"
        )


    # --------------------------------------------------------
    # Même split test que CatBoost / LightGBM
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
            f"{role.upper()} : "
            "split test vide."
        )


    # --------------------------------------------------------
    # Vérifier qu'un case_id appartient
    # à un seul split.
    # --------------------------------------------------------

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
            "un case_id apparaît "
            "dans plusieurs splits."
        )


    # --------------------------------------------------------
    # Conversion numérique
    # --------------------------------------------------------

    numeric_columns = (
        set(PRIORITY_COLUMNS)
    )


    for values in (
        ROLE_BLOCKS[role].values()
    ):

        numeric_columns.update(
            values
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
# NORMALISATION DES FEATURES PAR CASE_ID
# ============================================================

def minmax_normalize(
    series: pd.Series,
    higher_is_better: bool,
):

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).astype(float)


    finite_values = values[
        np.isfinite(values)
    ]


    if finite_values.empty:

        normalized = pd.Series(
            0.5,
            index=series.index,
            dtype=float,
        )

        return normalized


    minimum = float(
        finite_values.min()
    )

    maximum = float(
        finite_values.max()
    )


    # Critère constant :
    # il n'aide pas à départager les candidats.
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
# PRIORITES UTILISATEUR
# ============================================================

def normalized_priority_weights(
    group: pd.DataFrame,
):

    # Les priorités sont constantes pour toutes les lignes
    # d'un même case_id.
    raw = {

        "performance":
            float(
                group.iloc[0][
                    "performance_priority"
                ]
            ),

        "reliability":
            float(
                group.iloc[0][
                    "reliability_priority"
                ]
            ),

        "cost":
            float(
                group.iloc[0][
                    "cost_priority"
                ]
            ),

        "power":
            float(
                group.iloc[0][
                    "power_priority"
                ]
            ),
    }


    # Valeurs non finies ou négatives -> 0.
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


    # Sécurité :
    # si toutes les priorités valent 0,
    # on revient à 25 % par bloc.
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


    return raw, weights


# ============================================================
# CALCUL DES BLOCS
# ============================================================

def compute_block_score(
    result: pd.DataFrame,
    features: list[str],
    lower_is_better: bool,
    block_name: str,
):

    normalized_columns = []


    for feature in features:

        column_name = (
            f"v2_norm_{feature}"
        )


        result[
            column_name
        ] = minmax_normalize(

            result[
                feature
            ],

            higher_is_better=(
                not lower_is_better
            ),
        )


        normalized_columns.append(
            column_name
        )


    block_score_column = (
        f"v2_{block_name}_score"
    )


    result[
        block_score_column
    ] = (

        result[
            normalized_columns
        ]

        .mean(
            axis=1
        )
    )


    return block_score_column


# ============================================================
# SCORE V2 D'UN CASE_ID
# ============================================================

def score_one_case_v2(
    group: pd.DataFrame,
    role: str,
):

    result = group.copy()


    # --------------------------------------------------------
    # 1. Poids dynamiques du workload
    # --------------------------------------------------------

    raw_priorities, weights = (
        normalized_priority_weights(
            group
        )
    )


    for block_name in [
        "performance",
        "reliability",
        "cost",
        "power",
    ]:

        result[
            f"v2_weight_{block_name}"
        ] = weights[
            block_name
        ]


        result[
            f"v2_raw_priority_{block_name}"
        ] = raw_priorities[
            block_name
        ]


    # --------------------------------------------------------
    # 2. Scores des 4 blocs
    # --------------------------------------------------------

    block_score_columns = {}


    for block_name, features in (
        ROLE_BLOCKS[
            role
        ].items()
    ):

        block_score_columns[
            block_name
        ] = compute_block_score(

            result=result,

            features=features,

            lower_is_better=(
                block_name
                in
                LOWER_IS_BETTER_BLOCKS
            ),

            block_name=block_name,
        )


    # --------------------------------------------------------
    # 3. Score final
    # --------------------------------------------------------

    result[
        "deterministic_v2_score"
    ] = (

        result[
            block_score_columns[
                "performance"
            ]
        ]
        *
        weights[
            "performance"
        ]

        +

        result[
            block_score_columns[
                "reliability"
            ]
        ]
        *
        weights[
            "reliability"
        ]

        +

        result[
            block_score_columns[
                "cost"
            ]
        ]
        *
        weights[
            "cost"
        ]

        +

        result[
            block_score_columns[
                "power"
            ]
        ]
        *
        weights[
            "power"
        ]
    )


    # --------------------------------------------------------
    # 4. Ranking
    # --------------------------------------------------------

    result = (

        result

        .sort_values(
            [
                "deterministic_v2_score",
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
        "deterministic_v2_rank"
    ] = np.arange(
        1,
        len(result) + 1,
    )


    return (
        result,
        raw_priorities,
        weights,
    )


# ============================================================
# SCORE DU DATASET
# ============================================================

def score_dataset_v2(
    test_df: pd.DataFrame,
    role: str,
):

    scored_groups = []

    priority_audit_rows = []


    for case_id, group in (

        test_df.groupby(
            GROUP_COLUMN,
            sort=False,
        )

    ):

        (
            scored_case,
            raw_priorities,
            weights,
        ) = score_one_case_v2(
            group,
            role,
        )


        scored_groups.append(
            scored_case
        )


        priority_audit_rows.append({

            "role":
                role,

            "case_id":
                case_id,

            "performance_priority_raw":
                raw_priorities[
                    "performance"
                ],

            "reliability_priority_raw":
                raw_priorities[
                    "reliability"
                ],

            "cost_priority_raw":
                raw_priorities[
                    "cost"
                ],

            "power_priority_raw":
                raw_priorities[
                    "power"
                ],

            "performance_weight":
                weights[
                    "performance"
                ],

            "reliability_weight":
                weights[
                    "reliability"
                ],

            "cost_weight":
                weights[
                    "cost"
                ],

            "power_weight":
                weights[
                    "power"
                ],

            "weight_sum":
                sum(
                    weights.values()
                ),
        })


    scored = pd.concat(
        scored_groups,
        ignore_index=True,
    )


    scored[
        "role"
    ] = role


    scored[
        "model"
    ] = (
        "DeterministicBaselineV2"
    )


    priority_audit = pd.DataFrame(
        priority_audit_rows
    )


    return (
        scored,
        priority_audit,
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
                "deterministic_v2_score",
                "drive_id",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )


    actual_dcg = dcg_at_k(

        predicted[
            LABEL_COLUMN
        ].to_numpy(),

        k,
    )


    ideal_relevance = np.sort(

        group[
            LABEL_COLUMN
        ].to_numpy()

    )[::-1]


    ideal_dcg = dcg_at_k(
        ideal_relevance,
        k,
    )


    if ideal_dcg == 0:

        return 1.0


    return (
        actual_dcg
        /
        ideal_dcg
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
                    "deterministic_v2_score",
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


        predicted_top1 = (
            predicted.iloc[0][
                "drive_id"
            ]
        )


        teacher_top1 = (
            teacher.iloc[0][
                "drive_id"
            ]
        )


        predicted_top3 = set(
            predicted.head(3)[
                "drive_id"
            ]
        )

        teacher_top3 = set(
            teacher.head(3)[
                "drive_id"
            ]
        )


        predicted_top5 = set(
            predicted.head(5)[
                "drive_id"
            ]
        )

        teacher_top5 = set(
            teacher.head(5)[
                "drive_id"
            ]
        )


        predicted_top10 = set(
            predicted.head(10)[
                "drive_id"
            ]
        )

        teacher_top10 = set(
            teacher.head(10)[
                "drive_id"
            ]
        )


        intersection10 = len(
            predicted_top10
            &
            teacher_top10
        )


        union10 = len(
            predicted_top10
            |
            teacher_top10
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
                    predicted_top1
                    ==
                    teacher_top1
                ),

            "top3_overlap":
                (
                    len(
                        predicted_top3
                        &
                        teacher_top3
                    )
                    /
                    3.0
                ),

            "recall_at_5":
                (
                    len(
                        predicted_top5
                        &
                        teacher_top5
                    )
                    /
                    max(
                        1,
                        len(
                            teacher_top5
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
                            teacher_top10
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
                predicted_top1,

            "teacher_top1_drive_id":
                teacher_top1,
        })


    case_metrics = pd.DataFrame(
        rows
    )


    summary = {

        "model":
            "DeterministicBaselineV2",

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
        " BASELINE DETERMINISTE V2 MDT / OST"
    )

    print(
        " Priorités utilisateur dynamiques"
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

    case_metrics_frames = []

    predictions_frames = []

    priority_audit_frames = []


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
            priority_audit,
        ) = score_dataset_v2(
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


        execution_seconds = (
            time.perf_counter()
            -
            start
        )


        summary[
            "execution_seconds"
        ] = execution_seconds


        summaries.append(
            summary
        )


        case_metrics_frames.append(
            case_metrics
        )


        predictions_frames.append(
            scored_df
        )


        priority_audit_frames.append(
            priority_audit
        )


        print(
            "\n"
            "--------------------------------------------"
        )

        print(
            f"{role.upper()} V2 RESULTS"
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
            f"{execution_seconds:.3f} s"
        )


    # ========================================================
    # CONCATENER
    # ========================================================

    global_metrics = pd.DataFrame(
        summaries
    )


    case_metrics = pd.concat(
        case_metrics_frames,
        ignore_index=True,
    )


    predictions = pd.concat(
        predictions_frames,
        ignore_index=True,
    )


    priority_audit = pd.concat(
        priority_audit_frames,
        ignore_index=True,
    )


    # ========================================================
    # PATHS DE SORTIE
    # ========================================================

    global_metrics_path = (
        output_dir
        /
        "deterministic_v2_global_metrics.csv"
    )


    case_metrics_path = (
        output_dir
        /
        "deterministic_v2_case_metrics.csv"
    )


    predictions_path = (
        output_dir
        /
        "deterministic_v2_predictions.csv.gz"
    )


    priority_audit_path = (
        output_dir
        /
        "deterministic_v2_priority_audit.csv"
    )


    config_path = (
        output_dir
        /
        "deterministic_v2_config.json"
    )


    zip_path = (
        output_dir
        /
        "deterministic_v2_artifacts.zip"
    )


    # ========================================================
    # SAUVEGARDE
    # ========================================================

    global_metrics.to_csv(
        global_metrics_path,
        index=False,
    )


    case_metrics.to_csv(
        case_metrics_path,
        index=False,
    )


    predictions.to_csv(
        predictions_path,
        index=False,
        compression="gzip",
    )


    priority_audit.to_csv(
        priority_audit_path,
        index=False,
    )


    # ========================================================
    # CONFIG
    # ========================================================

    config = {

        "algorithm":
            (
                "Deterministic V2 dynamic "
                "preference-weighted ranking baseline"
            ),

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

        "priority_columns":
            PRIORITY_COLUMNS,

        "priority_rule":
            (
                "The four numeric user priorities are "
                "renormalized to sum to 1 inside each case_id."
            ),

        "role_blocks":
            ROLE_BLOCKS,

        "block_internal_weights":
            (
                "Equal weights between features belonging "
                "to the same block."
            ),

        "feature_normalization":
            (
                "Min-max independently inside each case_id."
            ),

        "lower_is_better_blocks":
            sorted(
                LOWER_IS_BETTER_BLOCKS
            ),

        "forbidden_score_columns":
            sorted(
                FORBIDDEN_SCORE_COLUMNS
            ),

        "important_note":
            (
                "teacher_score, teacher_rank and relevance_grade "
                "are used only after scoring for evaluation."
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


    # ========================================================
    # ZIP
    # ========================================================

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:

        for path in [
            global_metrics_path,
            case_metrics_path,
            predictions_path,
            priority_audit_path,
            config_path,
        ]:

            archive.write(
                path,
                arcname=path.name,
            )


    # ========================================================
    # RESULTATS FINAUX
    # ========================================================

    print(
        "\n"
        "============================================"
    )

    print(
        " RESULTATS GLOBAUX V2"
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

    print(
        f"- {global_metrics_path}"
    )

    print(
        f"- {case_metrics_path}"
    )

    print(
        f"- {predictions_path}"
    )

    print(
        f"- {priority_audit_path}"
    )

    print(
        f"- {config_path}"
    )

    print(
        f"- {zip_path}"
    )


if __name__ == "__main__":

    main()