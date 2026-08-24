from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from preference_extractor.signal_detector.runtime import (
    PreferenceSignalDetector,
)
from preference_extractor.signal_detector.schemas import (
    PreferenceSignalResult,
)


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
            prefix="pref_v21_fresh_eval_"
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


def compute_metrics(
    y_true,
    y_pred,
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "FIRST_RUN evaluation of frozen V2.1 on the fresh final V3 holdout."
        )
    )

    parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--frozen-threshold",
        type=Path,
        default=Path(
            "preference_extractor/evaluation/results/"
            "v2_1_FINAL_threshold_FROZEN.json"
        ),
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "preference_extractor/evaluation/datasets/"
            "preference_signal_fresh_final_holdout_v3.jsonl"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "preference_extractor/evaluation/results/"
            "v2_1_fresh_final_holdout_v3_FIRST_RUN.json"
        ),
    )

    args = parser.parse_args()

    if args.output.exists():
        raise RuntimeError(
            f"FIRST_RUN output already exists: {args.output}. "
            "Do not overwrite independent evidence."
        )

    frozen = json.loads(
        args.frozen_threshold.read_text(encoding="utf-8")
    )

    if frozen.get("status") != "FROZEN_BEFORE_FRESH_HOLDOUT":
        raise AssertionError(
            "Threshold file is not frozen correctly."
        )

    threshold = float(
        frozen["threshold"]
    )

    data = read_jsonl(args.dataset)

    if len(data) != 1200:
        raise AssertionError(
            f"Expected 1200 fresh holdout rows, got {len(data)}"
        )

    y_true = []
    raw_pred = []
    guarded_pred = []

    routes = Counter()

    by_family = defaultdict(
        lambda: {
            "y": [],
            "raw": [],
            "guarded": [],
        }
    )

    by_language = defaultdict(
        lambda: {
            "y": [],
            "raw": [],
            "guarded": [],
        }
    )

    with artifact_root(args.artifact) as root:
        detector = PreferenceSignalDetector(
            artifact_path=str(root),
            use_context_guard=True,
        )

        detector.threshold = threshold

        for index, row in enumerate(
            data,
            start=1,
        ):
            y = int(row["label_id"])

            probability = float(
                detector._preference_probability(
                    row["text"]
                )
            )

            raw_has_signal = (
                probability >= threshold
            )

            raw_result = PreferenceSignalResult(
                has_preference_signal=raw_has_signal,
                label=detector.id2label[
                    1 if raw_has_signal else 0
                ],
                probability=probability,
                threshold=threshold,
                decision_source="transformer",
                transformer_has_preference_signal=raw_has_signal,
            )

            guarded = detector.apply_context_guard(
                text=row["text"],
                model_result=raw_result,
            )

            raw_label = int(raw_has_signal)
            guarded_label = int(
                guarded.has_preference_signal
            )

            y_true.append(y)
            raw_pred.append(raw_label)
            guarded_pred.append(
                guarded_label
            )

            routes[
                guarded.decision_source
            ] += 1

            family = row["stress_family"]
            language = row["language"]

            by_family[family]["y"].append(y)
            by_family[family]["raw"].append(
                raw_label
            )
            by_family[family]["guarded"].append(
                guarded_label
            )

            by_language[language]["y"].append(y)
            by_language[language]["raw"].append(
                raw_label
            )
            by_language[language]["guarded"].append(
                guarded_label
            )

            if (
                index % 100 == 0
                or index == len(data)
            ):
                print(
                    f"[{index}/{len(data)}] FIRST_RUN"
                )

    raw_metrics = compute_metrics(
        y_true,
        raw_pred,
    )

    guarded_metrics = compute_metrics(
        y_true,
        guarded_pred,
    )

    per_family = {}

    for family, values in sorted(
        by_family.items()
    ):
        per_family[family] = {
            "n": len(values["y"]),
            "raw": compute_metrics(
                values["y"],
                values["raw"],
            ),
            "guarded": compute_metrics(
                values["y"],
                values["guarded"],
            ),
        }

    per_language = {}

    for language, values in sorted(
        by_language.items()
    ):
        per_language[language] = {
            "n": len(values["y"]),
            "raw": compute_metrics(
                values["y"],
                values["raw"],
            ),
            "guarded": compute_metrics(
                values["y"],
                values["guarded"],
            ),
        }

    payload = {
        "step": "3.1H",
        "evaluation": (
            "V2.1 fresh final holdout V3 FIRST_RUN"
        ),
        "model_version": "v2.1",
        "threshold": threshold,
        "threshold_status": (
            "frozen before fresh-holdout generation"
        ),
        "samples": len(data),
        "raw_transformer": raw_metrics,
        "guarded_pipeline": guarded_metrics,
        "guard_route_counts": dict(routes),
        "per_family": per_family,
        "per_language": per_language,
        "protocol_note": (
            "This is FIRST_RUN independent evidence for the frozen V2.1 "
            "pipeline. Do not overwrite. Any later tuning makes V3 "
            "regression-only."
        ),
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
        "V2.1 FRESH FINAL HOLDOUT V3 — FIRST RUN"
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


if __name__ == "__main__":
    main()
