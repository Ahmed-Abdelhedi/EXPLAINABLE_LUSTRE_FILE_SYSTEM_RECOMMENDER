from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from preference_extractor.signal_detector.context_guard import (
    PreferenceGuardDecision,
)
from preference_extractor.signal_detector.runtime import (
    PreferenceSignalDetector,
)


TARGET_RECALL_DEFAULT = 0.985


@contextmanager
def artifact_root(path: Path):
    path = path.resolve()

    if path.is_dir():
        yield path
        return

    if path.suffix.lower() != ".zip":
        raise ValueError(
            f"Artifact must be a directory or .zip: {path}"
        )

    with zipfile.ZipFile(path, "r") as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(
                f"Corrupted ZIP member: {bad}"
            )

        with tempfile.TemporaryDirectory(
            prefix="pref_v2_guarded_calibrate_"
        ) as tmp:
            tmp_path = Path(tmp)
            archive.extractall(tmp_path)

            roots = [
                candidate
                for candidate in tmp_path.iterdir()
                if candidate.is_dir()
            ]

            yield (
                roots[0]
                if len(roots) == 1
                else tmp_path
            )


def read_jsonl(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def label_of(row: dict) -> int:
    if "label_id" in row:
        return int(
            row["label_id"]
        )

    return int(
        row["label"]
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
):
    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "f1": float(
            f1
        ),
        "macro_precision": float(
            macro_precision
        ),
        "macro_recall": float(
            macro_recall
        ),
        "macro_f1": float(
            macro_f1
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": (
            float(
                fp / (fp + tn)
            )
            if fp + tn
            else 0.0
        ),
        "fnr": (
            float(
                fn / (fn + tp)
            )
            if fn + tp
            else 0.0
        ),
    }


def choose_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    guard_codes: np.ndarray,
    target_recall: float,
):
    """
    Calibrate the FINAL production pipeline, not the raw Transformer alone.

    guard_codes:
        -1 -> FORCE_NO_SIGNAL
         0 -> PASS_TO_MODEL
        +1 -> FORCE_SIGNAL
    """
    candidates = np.unique(
        np.concatenate(
            [
                np.array(
                    [0.0, 1.0],
                    dtype=float,
                ),
                probabilities.astype(
                    float
                ),
            ]
        )
    )

    rows = []

    forced_negative = (
        guard_codes == -1
    )
    forced_positive = (
        guard_codes == 1
    )
    pass_to_model = (
        guard_codes == 0
    )

    for threshold in candidates:
        raw_prediction = (
            probabilities
            >= threshold
        ).astype(
            int
        )

        guarded_prediction = (
            raw_prediction.copy()
        )

        guarded_prediction[
            forced_negative
        ] = 0

        guarded_prediction[
            forced_positive
        ] = 1

        raw_metrics = compute_metrics(
            y_true,
            raw_prediction,
        )

        guarded_metrics = compute_metrics(
            y_true,
            guarded_prediction,
        )

        rows.append(
            {
                "threshold": float(
                    threshold
                ),
                "raw": raw_metrics,
                "guarded": guarded_metrics,
            }
        )

    eligible = [
        row
        for row in rows
        if (
            row["guarded"]["recall"]
            >= target_recall
        )
    ]

    if eligible:
        chosen = max(
            eligible,
            key=lambda row: (
                row["guarded"]["precision"],
                row["guarded"]["f1"],
                row["guarded"]["accuracy"],
                row["threshold"],
            ),
        )

        policy = (
            "guarded_pipeline_max_precision_subject_to_"
            f"recall>={target_recall}"
        )
    else:
        chosen = max(
            rows,
            key=lambda row: (
                row["guarded"]["f1"],
                row["guarded"]["macro_f1"],
                row["guarded"]["accuracy"],
                row["threshold"],
            ),
        )

        policy = (
            "guarded_pipeline_fallback_max_f1"
        )

    route_counts = {
        "FORCE_NO_SIGNAL": int(
            np.sum(
                forced_negative
            )
        ),
        "PASS_TO_MODEL": int(
            np.sum(
                pass_to_model
            )
        ),
        "FORCE_SIGNAL": int(
            np.sum(
                forced_positive
            )
        ),
    }

    return (
        chosen,
        rows,
        policy,
        route_counts,
    )


def quantiles(values: np.ndarray):
    return {
        str(q): float(
            np.quantile(
                values,
                q,
            )
        )
        for q in (
            0.00,
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            1.00,
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate V2 threshold on VALIDATION ONLY, using the "
            "FINAL production pipeline: Transformer + PreferenceContextGuard. "
            "The previously inspected holdout is never read."
        )
    )

    parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--validation",
        type=Path,
        default=Path(
            "preference_extractor/training/data/data_layer1_v2/"
            "preference_signal_val_v2.jsonl"
        ),
    )

    parser.add_argument(
        "--target-recall",
        type=float,
        default=TARGET_RECALL_DEFAULT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "preference_extractor/evaluation/results/"
            "v2_guarded_threshold_candidate_step3_1c.json"
        ),
    )

    args = parser.parse_args()

    if not (
        0.0
        < args.target_recall
        <= 1.0
    ):
        raise ValueError(
            "target recall must be in (0, 1]"
        )

    data = read_jsonl(
        args.validation
    )

    if len(data) != 5200:
        raise AssertionError(
            f"Expected 5200 validation rows, got {len(data)}"
        )

    y_true = np.array(
        [
            label_of(row)
            for row in data
        ],
        dtype=int,
    )

    with artifact_root(
        args.artifact
    ) as root:
        detector = PreferenceSignalDetector(
            artifact_path=str(root),
            use_context_guard=True,
        )

        stored_threshold = float(
            detector.threshold
        )

        probabilities = []
        guard_codes = []

        for index, row in enumerate(
            data,
            start=1,
        ):
            probability = (
                detector
                ._preference_probability(
                    row["text"]
                )
            )

            probabilities.append(
                probability
            )

            guard_result = (
                detector
                .context_guard
                .resolve(
                    row["text"]
                )
            )

            if (
                guard_result.decision
                == PreferenceGuardDecision.FORCE_NO_SIGNAL
            ):
                guard_codes.append(
                    -1
                )

            elif (
                guard_result.decision
                == PreferenceGuardDecision.FORCE_SIGNAL
            ):
                guard_codes.append(
                    1
                )

            else:
                guard_codes.append(
                    0
                )

            if (
                index % 250
                == 0
                or index
                == len(data)
            ):
                print(
                    f"[{index}/{len(data)}] "
                    "validation probabilities + guard decisions"
                )

    probabilities = np.array(
        probabilities,
        dtype=float,
    )

    guard_codes = np.array(
        guard_codes,
        dtype=int,
    )

    stored_raw = (
        probabilities
        >= stored_threshold
    ).astype(
        int
    )

    stored_guarded = (
        stored_raw.copy()
    )

    stored_guarded[
        guard_codes == -1
    ] = 0

    stored_guarded[
        guard_codes == 1
    ] = 1

    (
        chosen,
        sweep,
        policy,
        route_counts,
    ) = choose_threshold(
        y_true=y_true,
        probabilities=probabilities,
        guard_codes=guard_codes,
        target_recall=args.target_recall,
    )

    positives = probabilities[
        y_true == 1
    ]

    negatives = probabilities[
        y_true == 0
    ]

    payload = {
        "step": "3.1C_guarded_calibration_fix",
        "calibration_dataset": str(
            args.validation
        ),
        "calibration_dataset_size": len(
            data
        ),
        "critical_protocol_note": (
            "Threshold selection uses V2 VALIDATION ONLY and calibrates "
            "the FINAL production pipeline (Transformer + Context Guard). "
            "The previously inspected 1200-case V2 holdout is not read."
        ),
        "target_recall": args.target_recall,
        "stored_artifact_threshold": stored_threshold,
        "stored_threshold_metrics_recomputed": {
            "raw_transformer": compute_metrics(
                y_true,
                stored_raw,
            ),
            "guarded_pipeline": compute_metrics(
                y_true,
                stored_guarded,
            ),
        },
        "guard_route_counts_on_validation": route_counts,
        "candidate": {
            "threshold": chosen[
                "threshold"
            ],
            "policy": policy,
            "raw_transformer_metrics_on_validation": (
                chosen[
                    "raw"
                ]
            ),
            "guarded_pipeline_metrics_on_validation": (
                chosen[
                    "guarded"
                ]
            ),
        },
        "probability_quantiles": {
            "positive_class": quantiles(
                positives
            ),
            "negative_class": quantiles(
                negatives
            ),
        },
        "sweep_summary": {
            "candidate_count": len(
                sweep
            ),
            "minimum_threshold": float(
                min(
                    row[
                        "threshold"
                    ]
                    for row in sweep
                )
            ),
            "maximum_threshold": float(
                max(
                    row[
                        "threshold"
                    ]
                    for row in sweep
                )
            ),
        },
        "artifact_is_not_modified": True,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "V2 GUARDED PIPELINE — VALIDATION-ONLY THRESHOLD CANDIDATE"
    )
    print(
        "=" * 80
    )
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(
        f"\nSaved: {args.output}"
    )
    print(
        "\nIMPORTANT: do not generate/evaluate the fresh holdout "
        "until this guarded candidate is reviewed."
    )


if __name__ == "__main__":
    main()
