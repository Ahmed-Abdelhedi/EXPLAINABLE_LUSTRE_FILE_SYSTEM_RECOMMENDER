from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from collections import Counter
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
            prefix="pref_v21_guarded_calibrate_"
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
        return int(row["label_id"])

    raw = row.get("label")

    if raw == "PREFERENCE_SIGNAL":
        return 1

    if raw == "NO_PREFERENCE_SIGNAL":
        return 0

    return int(raw)


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
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
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
            float(fp / (fp + tn))
            if fp + tn
            else 0.0
        ),
        "fnr": (
            float(fn / (fn + tp))
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
    candidates = np.unique(
        np.concatenate(
            [
                np.array(
                    [0.0, 1.0],
                    dtype=float,
                ),
                probabilities.astype(float),
            ]
        )
    )

    forced_negative = (
        guard_codes == -1
    )
    forced_positive = (
        guard_codes == 1
    )

    rows = []

    for threshold in candidates:
        raw_prediction = (
            probabilities
            >= threshold
        ).astype(int)

        guarded_prediction = (
            raw_prediction.copy()
        )

        guarded_prediction[
            forced_negative
        ] = 0

        guarded_prediction[
            forced_positive
        ] = 1

        rows.append(
            {
                "threshold": float(
                    threshold
                ),
                "raw": compute_metrics(
                    y_true,
                    raw_prediction,
                ),
                "guarded": compute_metrics(
                    y_true,
                    guarded_prediction,
                ),
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

    return chosen, rows, policy


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Final V2.1 threshold calibration on V2.1 VALIDATION ONLY, "
            "using Transformer + PreferenceContextGuard. "
            "No holdout is read."
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
            "preference_extractor/training/data/data_layer1_v2_1/"
            "preference_signal_val_v2_1.jsonl"
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
            "v2_1_FINAL_guarded_threshold_candidate.json"
        ),
    )

    args = parser.parse_args()

    data = read_jsonl(
        args.validation
    )

    if len(data) != 5800:
        raise AssertionError(
            f"Expected 5800 V2.1 validation rows, got {len(data)}"
        )

    y_true = np.asarray(
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

        stored_raw_candidate = float(
            detector.threshold
        )

        probabilities = []
        guard_codes = []
        guard_reasons = Counter()

        for index, row in enumerate(
            data,
            start=1,
        ):
            probability = float(
                detector._preference_probability(
                    row["text"]
                )
            )
            probabilities.append(
                probability
            )

            guard_result = (
                detector.context_guard.resolve(
                    row["text"]
                )
            )

            if (
                guard_result.decision
                == PreferenceGuardDecision.FORCE_NO_SIGNAL
            ):
                guard_codes.append(-1)
                route = "FORCE_NO_SIGNAL"

            elif (
                guard_result.decision
                == PreferenceGuardDecision.FORCE_SIGNAL
            ):
                guard_codes.append(1)
                route = "FORCE_SIGNAL"

            else:
                guard_codes.append(0)
                route = "PASS_TO_MODEL"

            reason = getattr(
                guard_result,
                "reason",
                None,
            )

            guard_reasons[
                f"{route}:{reason}"
            ] += 1

            if (
                index % 250 == 0
                or index == len(data)
            ):
                print(
                    f"[{index}/{len(data)}] "
                    "V2.1 validation probabilities + guard decisions"
                )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )
    guard_codes = np.asarray(
        guard_codes,
        dtype=int,
    )

    stored_raw_prediction = (
        probabilities
        >= stored_raw_candidate
    ).astype(int)

    stored_guarded_prediction = (
        stored_raw_prediction.copy()
    )
    stored_guarded_prediction[
        guard_codes == -1
    ] = 0
    stored_guarded_prediction[
        guard_codes == 1
    ] = 1

    chosen, sweep, policy = choose_threshold(
        y_true=y_true,
        probabilities=probabilities,
        guard_codes=guard_codes,
        target_recall=args.target_recall,
    )

    payload = {
        "step": "3.1G",
        "model_version": "v2.1",
        "calibration_dataset": str(
            args.validation
        ),
        "calibration_dataset_size": len(
            data
        ),
        "critical_protocol_note": (
            "FINAL production threshold candidate is selected on "
            "V2.1 VALIDATION ONLY using Transformer + Context Guard. "
            "No old or fresh holdout is read."
        ),
        "target_recall": args.target_recall,
        "stored_raw_transformer_candidate": (
            stored_raw_candidate
        ),
        "stored_candidate_recomputed": {
            "raw_transformer": compute_metrics(
                y_true,
                stored_raw_prediction,
            ),
            "guarded_pipeline": compute_metrics(
                y_true,
                stored_guarded_prediction,
            ),
        },
        "guard_route_counts": {
            "FORCE_NO_SIGNAL": int(
                np.sum(
                    guard_codes == -1
                )
            ),
            "PASS_TO_MODEL": int(
                np.sum(
                    guard_codes == 0
                )
            ),
            "FORCE_SIGNAL": int(
                np.sum(
                    guard_codes == 1
                )
            ),
        },
        "guard_reason_counts": dict(
            guard_reasons
        ),
        "final_candidate": {
            "threshold": float(
                chosen["threshold"]
            ),
            "policy": policy,
            "raw_transformer_metrics_on_validation": (
                chosen["raw"]
            ),
            "guarded_pipeline_metrics_on_validation": (
                chosen["guarded"]
            ),
        },
        "sweep_summary": {
            "candidate_count": len(
                sweep
            ),
            "minimum_threshold": float(
                min(
                    row["threshold"]
                    for row in sweep
                )
            ),
            "maximum_threshold": float(
                max(
                    row["threshold"]
                    for row in sweep
                )
            ),
        },
        "artifact_is_not_modified": True,
        "fresh_final_holdout_v3_is_not_read": True,
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

    print("\n" + "=" * 80)
    print(
        "V2.1 FINAL GUARDED VALIDATION-ONLY THRESHOLD CANDIDATE"
    )
    print("=" * 80)
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\nSaved: {args.output}")
    print(
        "\nIMPORTANT: do not edit the artifact and do not generate "
        "the fresh holdout until this result is reviewed."
    )


if __name__ == "__main__":
    main()
