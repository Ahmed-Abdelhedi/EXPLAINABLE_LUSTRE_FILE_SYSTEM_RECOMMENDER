from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import pandas as pd

import lightgbm as lgb
from catboost import CatBoostRanker, EFstrType, Pool


SCRIPT_VERSION = "2.2-pylance-clean"

DEFAULT_SEEDS = [21, 42, 84, 126, 168]
DEFAULT_ROLES = ["mdt", "ost"]
DEFAULT_MODELS = ["catboost", "lightgbm"]

IDENTIFIER_FALLBACK = [
    "split",
    "case_id",
    "drive_id",
    "drive_name",
    "manufacturer",
    "series",
]

LABEL_FALLBACK = [
    "group_size",
    "teacher_rank",
    "teacher_score",
    "relevance_grade",
    "is_teacher_top1",
    "is_teacher_top5",
    "is_teacher_top10",
]

LIGHTGBM_BASE_PARAMS: dict[str, Any] = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "boosting_type": "gbdt",
    "n_estimators": 1800,
    "learning_rate": 0.045,
    "max_depth": 8,
    "num_leaves": 255,
    "min_child_samples": 30,
    "reg_alpha": 0.0,
    "reg_lambda": 6.0,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "label_gain": [0, 1, 3, 7, 15],
    "n_jobs": -1,
    "verbosity": -1,
    "importance_type": "gain",
}

CATBOOST_BASE_PARAMS: dict[str, Any] = {
    "loss_function": "YetiRankPairwise",
    "eval_metric": "NDCG:top=10",
    "iterations": 1800,
    "learning_rate": 0.045,
    "depth": 8,
    "l2_leaf_reg": 6.0,
    "random_strength": 0.5,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.9,
    "verbose": False,
    "allow_writing_files": False,
}

EXPECTED_FINAL_DATASET_SHA256_UNCOMPRESSED = {
    "mdt": "a6dbcb1ae8c446f626a05d1f8393500a8ee77292770baf6e6ce10dc5824b273c",
    "ost": "28380ba8e4ae5d988da834b5d74bce6bd3062d2a948cdece3710227ae51fd2b2",
}

EXPECTED_FINAL_ROWS = {
    "mdt": 188_412,
    "ost": 116_572,
}


class ComparisonTrainingError(RuntimeError):
    """Erreur de contrat ou d'entraînement de la comparaison finale."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Comparaison finale CatBoostRanker vs LightGBM LGBMRanker sur les "
            "mêmes datasets, splits, features, seeds et métriques MDT/OST."
        )
    )
    parser.add_argument(
        "--training-dir",
        type=Path,
        default=Path("lustre_architecture_generator/output/training"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lustre_architecture_generator/artifacts/rankers/comparison_final"),
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=Path("lustre_architecture_generator/evaluation/ranking/comparison_final"),
    )
    parser.add_argument(
        "--roles",
        nargs="+",
        choices=DEFAULT_ROLES,
        default=DEFAULT_ROLES,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=DEFAULT_MODELS,
        default=DEFAULT_MODELS,
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
    )
    parser.add_argument(
        "--device-type",
        choices=["gpu", "cpu"],
        default="gpu",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=1800,
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=150,
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Smoke-test uniquement. Ne pas utiliser pour la campagne finale.",
    )
    parser.add_argument(
        "--skip-final-hash-check",
        action="store_true",
        help="Autorisé uniquement pour smoke-tests sur des datasets historiques.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_uncompressed_gzip(path: Path) -> str:
    h = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def load_manifest(training_dir: Path) -> dict[str, Any]:
    path = training_dir / "training_dataset_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Manifest introuvable: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ComparisonTrainingError("training_dataset_manifest.json invalide.")
    return value


def load_role_data(
    training_dir: Path,
    manifest: dict[str, Any],
    role: str,
    max_cases: int | None,
    enforce_final_hash: bool,
) -> tuple[pd.DataFrame, list[str], list[str], list[str], dict[str, Any]]:
    if role not in manifest or not isinstance(manifest[role], dict):
        raise ComparisonTrainingError(f"Manifest sans section {role!r}.")

    meta = manifest[role]
    dataset_path = training_dir / str(meta["file"])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset {role} introuvable: {dataset_path}")

    compressed_sha = sha256_file(dataset_path)
    uncompressed_sha = sha256_uncompressed_gzip(dataset_path)

    if enforce_final_hash:
        expected_sha = EXPECTED_FINAL_DATASET_SHA256_UNCOMPRESSED[role]
        if uncompressed_sha != expected_sha:
            raise ComparisonTrainingError(
                f"{role}: mauvais dataset final. SHA256 obtenu={uncompressed_sha}, "
                f"attendu={expected_sha}."
            )

    df = pd.read_csv(dataset_path, compression="gzip", low_memory=False)
    full_row_count = int(len(df))

    if enforce_final_hash and full_row_count != EXPECTED_FINAL_ROWS[role]:
        raise ComparisonTrainingError(
            f"{role}: row_count={full_row_count}, attendu={EXPECTED_FINAL_ROWS[role]}."
        )

    if max_cases is not None:
        all_ids = sorted(df["case_id"].astype(str).unique())
        keep_ids = all_ids[:max_cases]
        df = df[df["case_id"].astype(str).isin(keep_ids)].copy()

    identifier_columns = list(meta.get("identifier_columns", IDENTIFIER_FALLBACK))
    categorical_features = list(meta["categorical_features"])
    numeric_features = list(meta["numeric_features"])
    feature_columns = list(meta["model_feature_columns"])
    label_columns = list(meta.get("label_columns", LABEL_FALLBACK))

    expected_columns = set(
        identifier_columns + categorical_features + numeric_features + label_columns
    )
    missing = sorted(expected_columns - set(df.columns))
    if missing:
        raise ComparisonTrainingError(f"{role}: colonnes manquantes: {missing}")

    if feature_columns != categorical_features + numeric_features:
        raise ComparisonTrainingError(
            f"{role}: model_feature_columns != categorical_features + numeric_features."
        )

    forbidden_features = {
        "teacher_rank",
        "teacher_score",
        "relevance_grade",
        "is_teacher_top1",
        "is_teacher_top5",
        "is_teacher_top10",
        "group_size",
    }
    leaked = sorted(set(feature_columns) & forbidden_features)
    if leaked:
        raise ComparisonTrainingError(f"{role}: fuite de labels dans les features: {leaked}")

    for col in categorical_features:
        df[col] = df[col].fillna("NONE").astype(str)

    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["case_id"] = df["case_id"].astype(str)
    df["drive_id"] = df["drive_id"].astype(str)
    df["split"] = df["split"].astype(str)
    df["relevance_grade"] = pd.to_numeric(
        df["relevance_grade"], errors="raise"
    ).astype(int)
    df["teacher_rank"] = pd.to_numeric(df["teacher_rank"], errors="raise")

    valid_splits = {"train", "validation", "test"}
    observed = set(df["split"].unique())
    if observed != valid_splits:
        raise ComparisonTrainingError(
            f"{role}: splits attendus={sorted(valid_splits)}, obtenus={sorted(observed)}."
        )

    contract = {
        "dataset_file": str(dataset_path),
        "dataset_rows_full": full_row_count,
        "dataset_rows_used": int(len(df)),
        "dataset_sha256_compressed": compressed_sha,
        "dataset_sha256_uncompressed": uncompressed_sha,
        "feature_columns": feature_columns,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
    }

    return df, feature_columns, categorical_features, numeric_features, contract


def sort_for_ranking(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        ["case_id", "teacher_rank", "drive_id"],
        kind="stable",
    ).reset_index(drop=True)


def split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    seen: dict[str, str] = {}

    for split in ("train", "validation", "test"):
        part = sort_for_ranking(df[df["split"] == split].copy())
        if part.empty:
            raise ComparisonTrainingError(f"Split vide: {split}")
        frames[split] = part

        for case_id in part["case_id"].unique():
            prior = seen.get(case_id)
            if prior is not None:
                raise ComparisonTrainingError(
                    f"case_id {case_id} présent dans {prior} et {split}."
                )
            seen[case_id] = split

    return frames


def group_sizes(df: pd.DataFrame) -> list[int]:
    sizes = df.groupby("case_id", sort=False).size().astype(int).tolist()
    if sum(sizes) != len(df):
        raise AssertionError("Somme des tailles de groupes incohérente.")
    return sizes


def dcg(relevances: list[float], k: int) -> float:
    values = relevances[:k]
    return sum(
        (2.0 ** rel - 1.0) / math.log2(index + 2.0)
        for index, rel in enumerate(values)
    )


def ndcg_for_group(group: pd.DataFrame, pred_col: str, k: int) -> float:
    predicted = group.sort_values(
        [pred_col, "drive_id"], ascending=[False, True], kind="stable"
    )
    rel_pred = predicted["relevance_grade"].astype(float).tolist()
    rel_ideal = sorted(group["relevance_grade"].astype(float).tolist(), reverse=True)
    ideal = dcg(rel_ideal, k)
    return 1.0 if ideal <= 0 else dcg(rel_pred, k) / ideal


def evaluate_predictions(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    metrics: dict[str, list[float]] = {
        "ndcg_at_5": [],
        "ndcg_at_10": [],
        "top1_agreement": [],
        "top3_overlap": [],
        "recall_at_5": [],
        "recall_at_10": [],
        "top10_overlap_count": [],
        "top10_jaccard": [],
        "predicted_top1_teacher_rank": [],
    }

    for _, group in df.groupby("case_id", sort=False):
        teacher = group.sort_values(["teacher_rank", "drive_id"], kind="stable")
        predicted = group.sort_values(
            [pred_col, "drive_id"], ascending=[False, True], kind="stable"
        )

        metrics["ndcg_at_5"].append(ndcg_for_group(group, pred_col, 5))
        metrics["ndcg_at_10"].append(ndcg_for_group(group, pred_col, 10))

        teacher_ids = teacher["drive_id"].tolist()
        pred_ids = predicted["drive_id"].tolist()
        metrics["top1_agreement"].append(float(pred_ids[0] == teacher_ids[0]))

        for k, name in ((3, "top3_overlap"), (5, "recall_at_5"), (10, "recall_at_10")):
            t = set(teacher_ids[:k])
            p = set(pred_ids[:k])
            metrics[name].append(len(t & p) / max(1, min(k, len(t))))

        t10 = set(teacher_ids[:10])
        p10 = set(pred_ids[:10])
        overlap = len(t10 & p10)
        union = len(t10 | p10)
        metrics["top10_overlap_count"].append(float(overlap))
        metrics["top10_jaccard"].append(overlap / union if union else 1.0)

        pred_top_drive = pred_ids[0]
        teacher_rank_rows = group.loc[
            group["drive_id"] == pred_top_drive,
            ["teacher_rank"],
        ]
        if teacher_rank_rows.empty:
            raise ComparisonTrainingError(
                f"Aucun teacher_rank trouvé pour drive_id={pred_top_drive!r}."
            )
        teacher_rank = float(
            teacher_rank_rows.to_numpy(dtype=float)[0, 0]
        )
        metrics["predicted_top1_teacher_rank"].append(teacher_rank)

    return {key: mean(values) for key, values in metrics.items()}


def prediction_frame(
    source: pd.DataFrame,
    predictions: np.ndarray,
    model_family: str,
    role: str,
    seed: int,
    split: str,
) -> pd.DataFrame:
    result = source[
        ["case_id", "drive_id", "teacher_rank", "teacher_score", "relevance_grade"]
    ].copy()
    result["ml_score"] = np.asarray(predictions, dtype=float)
    result.insert(0, "split", split)
    result.insert(0, "seed", seed)
    result.insert(0, "role", role)
    result.insert(0, "model_family", model_family)
    return result


def build_train_category_mappings(
    train_df: pd.DataFrame,
    categorical_features: list[str],
) -> dict[str, list[str]]:
    mappings: dict[str, list[str]] = {}
    for col in categorical_features:
        values = set(train_df[col].fillna("NONE").astype(str))
        mappings[col] = sorted(values | {"NONE", "__UNKNOWN__"})
    return mappings


def prepare_lightgbm_frames(
    frames: dict[str, pd.DataFrame],
    categorical_features: list[str],
    category_mappings: dict[str, list[str]],
) -> dict[str, pd.DataFrame]:
    prepared: dict[str, pd.DataFrame] = {}
    for split, frame in frames.items():
        part = frame.copy()
        for col in categorical_features:
            allowed = set(category_mappings[col])
            values = part[col].fillna("NONE").astype(str)
            values = values.where(values.isin(allowed), "__UNKNOWN__")
            part[col] = pd.Categorical(
                values,
                categories=category_mappings[col],
            )
        prepared[split] = part
    return prepared


def train_lightgbm_one_seed(
    role: str,
    seed: int,
    frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
    categorical_features: list[str],
    category_mappings: dict[str, list[str]],
    model_root: Path,
    device_type: str,
    n_estimators: int,
    early_stopping_rounds: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    prepared = prepare_lightgbm_frames(frames, categorical_features, category_mappings)
    train_df, val_df, test_df = prepared["train"], prepared["validation"], prepared["test"]

    params = dict(LIGHTGBM_BASE_PARAMS)
    params.update(
        {
            "n_estimators": n_estimators,
            "random_state": seed,
            "bagging_seed": seed,
            "feature_fraction_seed": seed,
            "data_random_seed": seed,
            "device_type": device_type,
        }
    )

    model = lgb.LGBMRanker(**params)
    history: dict[str, dict[str, list[float]]] = {}
    callbacks: list[Any] = [lgb.record_evaluation(history)]
    if early_stopping_rounds > 0:
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))

    start = time.perf_counter()
    model.fit(
        train_df[feature_columns],
        train_df["relevance_grade"],
        group=group_sizes(train_df),
        eval_set=[(val_df[feature_columns], val_df["relevance_grade"])],
        eval_group=[group_sizes(val_df)],
        eval_at=[5, 10],
        categorical_feature=categorical_features,
        callbacks=callbacks,
    )
    training_seconds = time.perf_counter() - start

    pred_start = time.perf_counter()
    val_pred = np.asarray(
        model.predict(val_df[feature_columns]),
        dtype=float,
    ).reshape(-1)
    test_pred = np.asarray(
        model.predict(test_df[feature_columns]),
        dtype=float,
    ).reshape(-1)
    prediction_seconds = time.perf_counter() - pred_start

    val_eval = prediction_frame(
        val_df, val_pred, "LightGBM", role, seed, "validation"
    )
    test_eval = prediction_frame(
        test_df, test_pred, "LightGBM", role, seed, "test"
    )
    val_metrics = evaluate_predictions(val_eval, "ml_score")
    test_metrics = evaluate_predictions(test_eval, "ml_score")

    role_dir = model_root / "lightgbm" / role
    role_dir.mkdir(parents=True, exist_ok=True)
    model_path = role_dir / f"{role}_lightgbm_ranker_seed_{seed}.txt"
    model.booster_.save_model(str(model_path), num_iteration=model.best_iteration_)
    save_json(role_dir / f"{role}_evaluation_history_seed_{seed}.json", history)

    best_iteration = int(model.best_iteration_ or model.n_estimators_)
    metrics_row: dict[str, Any] = {
        "model_family": "LightGBM",
        "role": role,
        "seed": seed,
        "device_type": device_type,
        "best_iteration": best_iteration,
        "tree_count": int(model.booster_.num_trees()),
        "training_seconds": training_seconds,
        "prediction_seconds_validation_and_test": prediction_seconds,
        "model_size_mib": model_path.stat().st_size / (1024 * 1024),
    }
    metrics_row.update({f"validation_{k}": v for k, v in val_metrics.items()})
    metrics_row.update({f"test_{k}": v for k, v in test_metrics.items()})

    gain = model.booster_.feature_importance(importance_type="gain")
    split_imp = model.booster_.feature_importance(importance_type="split")
    importance = pd.DataFrame(
        {
            "model_family": "LightGBM",
            "role": role,
            "seed": seed,
            "feature": feature_columns,
            "importance": gain.astype(float),
            "secondary_importance": split_imp.astype(float),
            "importance_type": "gain",
        }
    )

    return metrics_row, pd.concat([val_eval, test_eval], ignore_index=True), importance


def make_catboost_pool(
    df: pd.DataFrame,
    feature_columns: list[str],
    categorical_features: list[str],
) -> Pool:
    data = df[feature_columns].copy()
    for col in categorical_features:
        data[col] = data[col].fillna("NONE").astype(str)
    return Pool(
        data=data,
        label=df["relevance_grade"].astype(float),
        group_id=df["case_id"].astype(str),
        cat_features=categorical_features,
        feature_names=feature_columns,
    )


def train_catboost_one_seed(
    role: str,
    seed: int,
    frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
    categorical_features: list[str],
    model_root: Path,
    device_type: str,
    n_estimators: int,
    early_stopping_rounds: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    train_df, val_df, test_df = frames["train"], frames["validation"], frames["test"]
    train_pool = make_catboost_pool(train_df, feature_columns, categorical_features)
    val_pool = make_catboost_pool(val_df, feature_columns, categorical_features)
    test_pool = make_catboost_pool(test_df, feature_columns, categorical_features)

    params = dict(CATBOOST_BASE_PARAMS)
    params.update(
        {
            "iterations": n_estimators,
            "random_seed": seed,
            "task_type": device_type.upper(),
            "od_type": "Iter",
            "od_wait": max(1, early_stopping_rounds),
            "use_best_model": True,
        }
    )
    if device_type == "gpu":
        params["devices"] = "0"

    model = CatBoostRanker(**params)

    start = time.perf_counter()
    model.fit(train_pool, eval_set=val_pool, verbose=False)
    training_seconds = time.perf_counter() - start

    pred_start = time.perf_counter()
    val_pred = model.predict(val_pool)
    test_pred = model.predict(test_pool)
    prediction_seconds = time.perf_counter() - pred_start

    val_eval = prediction_frame(val_df, val_pred, "CatBoost", role, seed, "validation")
    test_eval = prediction_frame(test_df, test_pred, "CatBoost", role, seed, "test")
    val_metrics = evaluate_predictions(val_eval, "ml_score")
    test_metrics = evaluate_predictions(test_eval, "ml_score")

    role_dir = model_root / "catboost" / role
    role_dir.mkdir(parents=True, exist_ok=True)
    model_path = role_dir / f"{role}_catboost_ranker_seed_{seed}.cbm"
    model.save_model(str(model_path))
    save_json(
        role_dir / f"{role}_evaluation_history_seed_{seed}.json",
        model.get_evals_result(),
    )

    tree_count_raw = model.tree_count_
    tree_count = (
        int(tree_count_raw)
        if tree_count_raw is not None
        else 0
    )

    best_iteration = int(model.get_best_iteration())
    if best_iteration < 0:
        best_iteration = max(0, tree_count - 1)

    metrics_row: dict[str, Any] = {
        "model_family": "CatBoost",
        "role": role,
        "seed": seed,
        "device_type": device_type,
        "best_iteration": best_iteration,
        "tree_count": tree_count,
        "training_seconds": training_seconds,
        "prediction_seconds_validation_and_test": prediction_seconds,
        "model_size_mib": model_path.stat().st_size / (1024 * 1024),
    }
    metrics_row.update({f"validation_{k}": v for k, v in val_metrics.items()})
    metrics_row.update({f"test_{k}": v for k, v in test_metrics.items()})

    importance_values = model.get_feature_importance(
        train_pool,
        type=EFstrType.PredictionValuesChange,
    )
    importance = pd.DataFrame(
        {
            "model_family": "CatBoost",
            "role": role,
            "seed": seed,
            "feature": feature_columns,
            "importance": np.asarray(importance_values, dtype=float),
            "secondary_importance": np.nan,
            "importance_type": "PredictionValuesChange",
        }
    )

    return metrics_row, pd.concat([val_eval, test_eval], ignore_index=True), importance


def aggregate_summary(run_metrics: pd.DataFrame) -> pd.DataFrame:
    base_metrics = [
        "ndcg_at_5",
        "ndcg_at_10",
        "top1_agreement",
        "top3_overlap",
        "recall_at_5",
        "recall_at_10",
        "top10_jaccard",
        "predicted_top1_teacher_rank",
    ]
    metric_columns = (
        [f"validation_{name}" for name in base_metrics]
        + [f"test_{name}" for name in base_metrics]
        + [
            "training_seconds",
            "prediction_seconds_validation_and_test",
            "model_size_mib",
            "best_iteration",
        ]
    )
    rows: list[dict[str, Any]] = []

    for (model_family, role), group in run_metrics.groupby(
        ["model_family", "role"], sort=True
    ):
        row: dict[str, Any] = {
            "model_family": model_family,
            "role": role,
            "seed_count": int(len(group)),
            "device_type": str(group["device_type"].to_numpy()[0]),
        }
        for col in metric_columns:
            values = [float(x) for x in group[col].tolist()]
            row[f"{col}_mean"] = mean(values)
            row[f"{col}_std"] = pstdev(values) if len(values) > 1 else 0.0
            row[f"{col}_min"] = min(values)
            row[f"{col}_max"] = max(values)
        rows.append(row)

    return pd.DataFrame(rows)


def feature_stability(feature_importance: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model_family, role, feature), group in feature_importance.groupby(
        ["model_family", "role", "feature"], sort=True
    ):
        values = group["importance"].astype(float).tolist()
        avg = mean(values)
        std = pstdev(values) if len(values) > 1 else 0.0
        rows.append(
            {
                "model_family": model_family,
                "role": role,
                "feature": feature,
                "importance_mean": avg,
                "importance_std": std,
                "importance_min": min(values),
                "importance_max": max(values),
                "coefficient_of_variation": std / avg if avg > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def model_selection_table(summary: pd.DataFrame) -> pd.DataFrame:
    # IMPORTANT: sélection uniquement sur VALIDATION. Le test reste une mesure finale
    # indépendante et n'intervient pas dans le classement des familles.
    rows: list[dict[str, Any]] = []
    for role, group in summary.groupby("role", sort=True):
        ranked = group.sort_values(
            [
                "validation_ndcg_at_5_mean",
                "validation_top1_agreement_mean",
                "validation_ndcg_at_10_mean",
                "validation_ndcg_at_5_std",
                "prediction_seconds_validation_and_test_mean",
            ],
            ascending=[False, False, False, True, True],
            kind="stable",
        ).reset_index(drop=True)
        for comparison_rank, (_, row) in enumerate(
            ranked.iterrows(),
            start=1,
        ):
            rows.append(
                {
                    "role": role,
                    "comparison_rank_on_validation": comparison_rank,
                    "model_family": row["model_family"],
                    "validation_ndcg_at_5_mean": row["validation_ndcg_at_5_mean"],
                    "validation_ndcg_at_5_std": row["validation_ndcg_at_5_std"],
                    "validation_ndcg_at_10_mean": row["validation_ndcg_at_10_mean"],
                    "validation_top1_agreement_mean": row["validation_top1_agreement_mean"],
                    "test_ndcg_at_5_mean": row["test_ndcg_at_5_mean"],
                    "test_ndcg_at_10_mean": row["test_ndcg_at_10_mean"],
                    "test_top1_agreement_mean": row["test_top1_agreement_mean"],
                    "test_recall_at_10_mean": row["test_recall_at_10_mean"],
                    "training_seconds_mean": row["training_seconds_mean"],
                    "prediction_seconds_validation_and_test_mean": row[
                        "prediction_seconds_validation_and_test_mean"
                    ],
                    "model_size_mib_mean": row["model_size_mib_mean"],
                }
            )
    return pd.DataFrame(rows)


def family_overall_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_family, group in summary.groupby("model_family", sort=True):
        rows.append(
            {
                "model_family": model_family,
                "roles": ",".join(sorted(group["role"].astype(str).tolist())),
                "validation_ndcg_at_5_role_mean": float(group["validation_ndcg_at_5_mean"].mean()),
                "validation_ndcg_at_10_role_mean": float(group["validation_ndcg_at_10_mean"].mean()),
                "validation_top1_role_mean": float(group["validation_top1_agreement_mean"].mean()),
                "test_ndcg_at_5_role_mean": float(group["test_ndcg_at_5_mean"].mean()),
                "test_ndcg_at_10_role_mean": float(group["test_ndcg_at_10_mean"].mean()),
                "test_top1_role_mean": float(group["test_top1_agreement_mean"].mean()),
                "training_seconds_role_sum": float(group["training_seconds_mean"].sum()),
                "model_size_mib_role_sum": float(group["model_size_mib_mean"].sum()),
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values(
        [
            "validation_ndcg_at_5_role_mean",
            "validation_top1_role_mean",
            "validation_ndcg_at_10_role_mean",
        ],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)


def build_markdown_report(
    summary: pd.DataFrame,
    selection: pd.DataFrame,
    overall: pd.DataFrame,
    contract: dict[str, Any],
) -> str:
    lines = [
        "# Final CatBoost vs LightGBM Ranker Comparison",
        "",
        "This report compares both ranker families using the exact same final MDT/OST datasets,",
        "case-level splits, feature schemas, labels, seeds, and evaluation metrics.",
        "",
        "## Dataset contract",
        "",
    ]
    for role in ("mdt", "ost"):
        if role in contract["roles"]:
            rc = contract["roles"][role]
            lines.extend(
                [
                    f"- **{role.upper()}**: {rc['dataset_rows_full']} rows, "
                    f"SHA256(uncompressed) `{rc['dataset_sha256_uncompressed']}`",
                    f"  - cases: {rc['split_case_counts']}",
                    f"  - rows: {rc['split_rows']}",
                ]
            )
    lines.extend(["", "## Multi-seed summary", "", summary.to_markdown(index=False), ""])
    lines.extend(["## Role-level decision table", "", selection.to_markdown(index=False), ""])
    lines.extend(["## Family-level overall table", "", overall.to_markdown(index=False), ""])
    lines.extend(
        [
            "## Decision rule",
            "",
            "Model-family selection is based on validation metrics only. The test split is kept "
            "independent and is reported only as final confirmation. Role-level ranking uses "
            "validation NDCG@5 first, then validation Top-1 agreement, validation NDCG@10, "
            "lower seed variability, and inference time. The family-level table summarizes "
            "both MDT and OST before the official ranker family is frozen.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.seeds:
        raise ComparisonTrainingError("Au moins une seed est requise.")
    if args.n_estimators <= 0:
        raise ComparisonTrainingError("--n-estimators doit être > 0.")
    if args.early_stopping_rounds < 0:
        raise ComparisonTrainingError("--early-stopping-rounds doit être >= 0.")

    enforce_final_hash = not args.skip_final_hash_check and args.max_cases is None
    manifest = load_manifest(args.training_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.evaluation_dir.mkdir(parents=True, exist_ok=True)

    all_run_metrics: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []
    all_importance: list[pd.DataFrame] = []
    role_contract: dict[str, Any] = {}

    print("Final CatBoost vs LightGBM ranking comparison")
    print("----------------------------------------------")
    print(f"Roles       : {args.roles}")
    print(f"Models      : {args.models}")
    print(f"Seeds       : {args.seeds}")
    print(f"Device      : {args.device_type}")
    print(f"Estimators  : {args.n_estimators}")
    print(f"Early stop  : {args.early_stopping_rounds}")
    if args.max_cases is not None:
        print(f"SMOKE TEST  : max_cases={args.max_cases}")

    for role in args.roles:
        (
            df,
            feature_columns,
            categorical_features,
            numeric_features,
            base_contract,
        ) = load_role_data(
            args.training_dir,
            manifest,
            role,
            args.max_cases,
            enforce_final_hash,
        )
        frames = split_frames(df)
        category_mappings = build_train_category_mappings(
            frames["train"], categorical_features
        )
        base_contract["lightgbm_category_mappings_train_only"] = category_mappings
        base_contract.update(
            {
                "split_rows": {k: int(len(v)) for k, v in frames.items()},
                "split_case_counts": {
                    k: int(v["case_id"].nunique()) for k, v in frames.items()
                },
            }
        )
        role_contract[role] = base_contract

        print(
            f"\n[{role.upper()}] rows={len(df)} features={len(feature_columns)} "
            f"cases={df['case_id'].nunique()}"
        )

        for model_name in args.models:
            for seed in args.seeds:
                print(f"  {model_name} seed={seed} ...", end="", flush=True)
                if model_name == "lightgbm":
                    metrics_row, predictions, importance = train_lightgbm_one_seed(
                        role,
                        seed,
                        frames,
                        feature_columns,
                        categorical_features,
                        category_mappings,
                        args.output_dir,
                        args.device_type,
                        args.n_estimators,
                        args.early_stopping_rounds,
                    )
                else:
                    metrics_row, predictions, importance = train_catboost_one_seed(
                        role,
                        seed,
                        frames,
                        feature_columns,
                        categorical_features,
                        args.output_dir,
                        args.device_type,
                        args.n_estimators,
                        args.early_stopping_rounds,
                    )

                all_run_metrics.append(metrics_row)
                all_predictions.append(predictions)
                all_importance.append(importance)
                print(
                    f" done | NDCG@5={metrics_row['test_ndcg_at_5']:.6f} "
                    f"Top1={metrics_row['test_top1_agreement']:.4f} "
                    f"iter={metrics_row['best_iteration']}"
                )

    run_metrics = pd.DataFrame(all_run_metrics)
    predictions = pd.concat(all_predictions, ignore_index=True)
    importance = pd.concat(all_importance, ignore_index=True)
    summary = aggregate_summary(run_metrics)
    stability = feature_stability(importance)
    selection = model_selection_table(summary)
    overall = family_overall_table(summary)

    run_metrics.to_csv(args.evaluation_dir / "ranker_comparison_run_metrics.csv", index=False)
    summary.to_csv(args.evaluation_dir / "ranker_comparison_multiseed_summary.csv", index=False)
    selection.to_csv(args.evaluation_dir / "ranker_comparison_decision_table.csv", index=False)
    overall.to_csv(args.evaluation_dir / "ranker_comparison_family_overall.csv", index=False)
    predictions.to_csv(
        args.evaluation_dir / "ranker_comparison_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    importance.to_csv(
        args.evaluation_dir / "ranker_comparison_feature_importance.csv",
        index=False,
    )
    stability.to_csv(
        args.evaluation_dir / "ranker_comparison_feature_stability.csv",
        index=False,
    )

    contract = {
        "schema_version": "1.0",
        "purpose": "final_catboost_vs_lightgbm_mdt_ost_comparison",
        "training_timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "library_versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "lightgbm": lgb.__version__,
            "catboost": __import__("catboost").__version__,
        },
        "models": args.models,
        "roles_requested": args.roles,
        "seeds": args.seeds,
        "device_type": args.device_type,
        "n_estimators": args.n_estimators,
        "early_stopping_rounds": args.early_stopping_rounds,
        "ranking_label": "relevance_grade",
        "group_column": "case_id",
        "teacher_columns_excluded_from_features": ["teacher_rank", "teacher_score"],
        "hard_constraints_applied_before_model": True,
        "split_contract": manifest.get("case_split", {}),
        "same_splits_for_both_algorithms": True,
        "same_features_for_both_algorithms": True,
        "same_seeds_for_both_algorithms": True,
        "same_metrics_for_both_algorithms": True,
        "selection_uses_validation_only": True,
        "test_split_reserved_for_final_confirmation": True,
        "expected_final_dataset_sha256_uncompressed": EXPECTED_FINAL_DATASET_SHA256_UNCOMPRESSED,
        "lightgbm_parameters_except_seed_device_estimators": LIGHTGBM_BASE_PARAMS,
        "catboost_parameters_except_seed_device_iterations": CATBOOST_BASE_PARAMS,
        "roles": role_contract,
    }
    save_json(args.evaluation_dir / "ranker_comparison_contract.json", contract)

    report = build_markdown_report(summary, selection, overall, contract)
    (args.evaluation_dir / "RANKER_COMPARISON_REPORT.md").write_text(
        report, encoding="utf-8"
    )

    print("\nFinal comparison completed")
    print("--------------------------")
    print(summary.to_string(index=False))
    print("\nRole-level decision table (validation-based)")
    print(selection.to_string(index=False))
    print("\nFamily-level overall table")
    print(overall.to_string(index=False))
    print(f"\nModels     : {args.output_dir}")
    print(f"Evaluation : {args.evaluation_dir}")
    print("STATUS     : VALIDATED")


if __name__ == "__main__":
    main()
