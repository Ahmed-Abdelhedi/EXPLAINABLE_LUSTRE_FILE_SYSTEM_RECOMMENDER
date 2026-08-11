from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS PAR DEFAUT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_DATA_DIR = BASE_DIR / "output" / "training"

DEFAULT_OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / "deterministic_baseline"
)


# ============================================================
# COLONNES IMPORTANTES
# ============================================================

GROUP_COLUMN = "case_id"
LABEL_COLUMN = "relevance_grade"


# Ces colonnes sont INTERDITES dans le calcul du score.
# Elles servent uniquement à l'évaluation après le ranking.
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
# BASELINE DETERMINISTE
# ============================================================
#
# Objectif :
# créer volontairement une méthode SIMPLE.
#
# Aucun entraînement.
# Aucune seed.
# Aucun CatBoost.
# Aucun LightGBM.
#
# Chaque rôle utilise seulement 6 critères.
#
# Tous les critères ont le même poids :
#
#     1 / 6 = 16.67 %
#
# ============================================================

BASELINE_FEATURES = {

    # --------------------------------------------------------
    # MDT
    # --------------------------------------------------------
    "mdt": {

        # Plus grand = meilleur
        "higher_is_better": [

            # endurance disponible / endurance demandée
            "endurance_margin_ratio",

            # marge IOPS écriture
            "write_iops_headroom_ratio",

            # marge IOPS lecture
            "read_iops_headroom_ratio",

            # MTBF disponible / minimum demandé
            "reliability_margin_ratio",
        ],

        # Plus petit = meilleur
        "lower_is_better": [

            # coût brut de la solution pré-RAID
            "raw_drive_cost_usd",

            # puissance brute pré-RAID
            "raw_drive_power_w",
        ],
    },


    # --------------------------------------------------------
    # OST
    # --------------------------------------------------------
    "ost": {

        # Plus grand = meilleur
        "higher_is_better": [

            # marge de capacité
            "capacity_headroom_ratio",

            # marge de débit lecture
            "read_bandwidth_headroom_ratio",

            # marge de débit écriture
            "write_bandwidth_headroom_ratio",

            # marge de fiabilité
            "reliability_margin_ratio",
        ],

        # Plus petit = meilleur
        "lower_is_better": [

            # coût brut
            "raw_drive_cost_usd",

            # puissance brute
            "raw_drive_power_w",
        ],
    },
}


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Baseline déterministe de ranking MDT / OST."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=(
            "Dossier contenant les datasets de training "
            "MDT/OST et le manifest."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Dossier dans lequel écrire les résultats.",
    )

    return parser.parse_args()


# ============================================================
# MANIFEST
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


# ============================================================
# TROUVER DATASET CSV / CSV.GZ
# ============================================================

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

    # Le manifest peut déclarer CSV.GZ alors que
    # le fichier réel est CSV.
    if declared_name.endswith(".csv.gz"):

        possible_names.append(
            declared_name[:-3]
        )

    elif declared_name.endswith(".csv"):

        possible_names.append(
            declared_name + ".gz"
        )

    for filename in possible_names:

        direct_path = (
            data_dir
            / filename
        )

        if direct_path.exists():

            return direct_path

        recursive_matches = list(
            data_dir.rglob(filename)
        )

        if recursive_matches:

            return recursive_matches[0]

    raise FileNotFoundError(
        f"Dataset introuvable : {declared_name}\n"
        f"Noms testés : {possible_names}"
    )


# ============================================================
# CHARGEMENT DU SPLIT TEST
# ============================================================

def load_test_dataset(
    data_dir: Path,
    manifest: dict,
    role: str,
):

    info = manifest[role]

    dataset_path = find_dataset(
        data_dir,
        info["file"],
    )

    print(
        f"\nChargement {role.upper()} : "
        f"{dataset_path}"
    )

    df = pd.read_csv(
        dataset_path,
        compression="infer",
    )

    score_features = (

        BASELINE_FEATURES[
            role
        ]["higher_is_better"]

        +

        BASELINE_FEATURES[
            role
        ]["lower_is_better"]
    )


    # --------------------------------------------------------
    # Vérification des colonnes
    # --------------------------------------------------------

    required_columns = {

        "split",

        "case_id",

        "drive_id",

        "drive_name",

        "teacher_rank",

        "teacher_score",

        "relevance_grade",

        *score_features,
    }

    missing_columns = sorted(
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            f"{role.upper()} : "
            f"colonnes absentes : "
            f"{missing_columns}"
        )


    # --------------------------------------------------------
    # Vérification anti-fuite
    # --------------------------------------------------------

    leakage = (

        set(score_features)

        &

        FORBIDDEN_SCORE_COLUMNS
    )

    if leakage:

        raise ValueError(
            "Fuite de données détectée : "
            f"{sorted(leakage)}"
        )


    # --------------------------------------------------------
    # On utilise EXACTEMENT le split test existant.
    # --------------------------------------------------------

    test_df = (

        df.loc[
            df["split"] == "test"
        ]

        .copy()

        .reset_index(drop=True)
    )

    if test_df.empty:

        raise ValueError(
            f"{role.upper()} : split test vide."
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

    if split_counts.max() != 1:

        raise ValueError(
            f"{role.upper()} : "
            "un case_id apparaît "
            "dans plusieurs splits."
        )


    # --------------------------------------------------------
    # Conversion numérique
    # --------------------------------------------------------

    for feature in score_features:

        test_df[feature] = pd.to_numeric(
            test_df[feature],
            errors="coerce",
        )


    test_df["teacher_rank"] = (

        pd.to_numeric(
            test_df["teacher_rank"],
            errors="raise",
        )

        .astype(int)
    )


    test_df[
        LABEL_COLUMN
    ] = (

        pd.to_numeric(
            test_df[LABEL_COLUMN],
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
# NORMALISATION MIN-MAX
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


    # Si toutes les valeurs sont absentes
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


    # Si tous les drives ont exactement
    # la même valeur :
    # le critère n'aide pas à les départager.
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
            .clip(0.0, 1.0)
        )


    # Pour coût / puissance :
    # petit = meilleur.
    if not higher_is_better:

        normalized = (
            1.0
            -
            normalized
        )

    return normalized


# ============================================================
# CALCUL DU SCORE POUR UN CASE_ID
# ============================================================

def score_one_case(
    group: pd.DataFrame,
    role: str,
):

    result = group.copy()

    component_columns = []


    # --------------------------------------------------------
    # Critères où PLUS = MEILLEUR
    # --------------------------------------------------------

    for feature in (

        BASELINE_FEATURES[
            role
        ]["higher_is_better"]

    ):

        component_name = (
            f"component_{feature}"
        )

        result[
            component_name
        ] = minmax_normalize(

            result[feature],

            higher_is_better=True,
        )

        component_columns.append(
            component_name
        )


    # --------------------------------------------------------
    # Critères où MOINS = MEILLEUR
    # --------------------------------------------------------

    for feature in (

        BASELINE_FEATURES[
            role
        ]["lower_is_better"]

    ):

        component_name = (
            f"component_{feature}"
        )

        result[
            component_name
        ] = minmax_normalize(

            result[feature],

            higher_is_better=False,
        )

        component_columns.append(
            component_name
        )


    # --------------------------------------------------------
    # SCORE FINAL
    #
    # 6 critères
    # même poids
    #
    # donc score = moyenne
    # --------------------------------------------------------

    result[
        "deterministic_score"
    ] = (

        result[
            component_columns
        ]

        .mean(axis=1)
    )


    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    result = (

        result

        .sort_values(
            [
                "deterministic_score",
                "drive_id",
            ],
            ascending=[
                False,
                True,
            ],
        )

        .reset_index(drop=True)
    )


    result[
        "deterministic_rank"
    ] = np.arange(
        1,
        len(result) + 1,
    )


    return result


# ============================================================
# SCORE DE TOUT LE DATASET TEST
# ============================================================

def score_dataset(
    test_df: pd.DataFrame,
    role: str,
):

    scored_groups = []


    for case_id, group in (

        test_df.groupby(
            GROUP_COLUMN,
            sort=False,
        )

    ):

        scored_case = score_one_case(
            group,
            role,
        )

        scored_groups.append(
            scored_case
        )


    result = pd.concat(
        scored_groups,
        ignore_index=True,
    )


    result["role"] = role

    result["model"] = (
        "DeterministicBaseline"
    )


    return result


# ============================================================
# DCG
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


# ============================================================
# NDCG
# ============================================================

def ndcg_at_k(
    group: pd.DataFrame,
    k: int,
):

    predicted = (

        group

        .sort_values(
            [
                "deterministic_score",
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


# ============================================================
# EVALUATION
# ============================================================

def evaluate_role(
    scored_df: pd.DataFrame,
    role: str,
):

    case_rows = []


    for case_id, group in (

        scored_df.groupby(
            "case_id",
            sort=False,
        )

    ):


        # ----------------------------------------------------
        # Ranking déterministe produit
        # ----------------------------------------------------

        predicted = (

            group

            .sort_values(
                [
                    "deterministic_score",
                    "drive_id",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
        )


        # ----------------------------------------------------
        # Ranking teacher
        #
        # Seulement pour l'évaluation.
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Top-1
        # ----------------------------------------------------

        predicted_top1 = (
            predicted.iloc[0]["drive_id"]
        )

        teacher_top1 = (
            teacher.iloc[0]["drive_id"]
        )


        # ----------------------------------------------------
        # Top-K
        # ----------------------------------------------------

        predicted_top3 = set(
            predicted.head(3)["drive_id"]
        )

        teacher_top3 = set(
            teacher.head(3)["drive_id"]
        )


        predicted_top5 = set(
            predicted.head(5)["drive_id"]
        )

        teacher_top5 = set(
            teacher.head(5)["drive_id"]
        )


        predicted_top10 = set(
            predicted.head(10)["drive_id"]
        )

        teacher_top10 = set(
            teacher.head(10)["drive_id"]
        )


        top10_intersection = len(
            predicted_top10
            &
            teacher_top10
        )


        top10_union = len(
            predicted_top10
            |
            teacher_top10
        )


        # ----------------------------------------------------
        # Métriques du case_id
        # ----------------------------------------------------

        case_rows.append({

            "role":
                role,

            "case_id":
                case_id,

            "candidate_count":
                len(group),


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
                    len(
                        teacher_top5
                    )
                ),


            "recall_at_10":
                (
                    top10_intersection
                    /
                    len(
                        teacher_top10
                    )
                ),


            "top10_jaccard":
                (
                    top10_intersection
                    /
                    top10_union
                )
                if top10_union
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
        case_rows
    )


    # ========================================================
    # Moyennes sur les 180 case_id
    # ========================================================

    summary = {

        "model":
            "DeterministicBaseline",

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
        " BASELINE DETERMINISTE MDT / OST"
    )

    print(
        "============================================"
    )

    print(
        f"Data : {data_dir}"
    )

    print(
        f"Output : {output_dir}"
    )


    manifest = load_manifest(
        data_dir
    )


    summaries = []

    case_metrics_all = []

    predictions_all = []


    # ========================================================
    # MDT + OST
    # ========================================================

    for role in [
        "mdt",
        "ost",
    ]:

        start = time.perf_counter()


        # ----------------------------------------------------
        # Charger mêmes données test
        # ----------------------------------------------------

        test_df = load_test_dataset(
            data_dir,
            manifest,
            role,
        )


        # ----------------------------------------------------
        # Ranking déterministe
        # ----------------------------------------------------

        scored_df = score_dataset(
            test_df,
            role,
        )


        # ----------------------------------------------------
        # Métriques
        # ----------------------------------------------------

        summary, case_metrics = (
            evaluate_role(
                scored_df,
                role,
            )
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


        case_metrics_all.append(
            case_metrics
        )


        predictions_all.append(
            scored_df
        )


        # ----------------------------------------------------
        # Résultat terminal
        # ----------------------------------------------------

        print(
            "\n"
            "--------------------------------------------"
        )

        print(
            f"{role.upper()} RESULTS"
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
        case_metrics_all,
        ignore_index=True,
    )


    predictions = pd.concat(
        predictions_all,
        ignore_index=True,
    )


    # ========================================================
    # SAUVEGARDE
    # ========================================================

    global_metrics_path = (

        output_dir
        /
        "deterministic_global_metrics.csv"
    )


    case_metrics_path = (

        output_dir
        /
        "deterministic_case_metrics.csv"
    )


    predictions_path = (

        output_dir
        /
        "deterministic_predictions.csv.gz"
    )


    config_path = (

        output_dir
        /
        "deterministic_baseline_config.json"
    )


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


    # ========================================================
    # CONFIGURATION / TRAÇABILITÉ
    # ========================================================

    config = {

        "algorithm":
            "Deterministic weighted ranking baseline",

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

        "weights":
            "equal weights: 1/6",

        "normalization":
            "min-max normalization inside each case_id",

        "features":
            BASELINE_FEATURES,

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


    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n"
        "============================================"
    )

    print(
        " RESULTATS GLOBAUX"
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
        f"- {config_path}"
    )


if __name__ == "__main__":

    main()