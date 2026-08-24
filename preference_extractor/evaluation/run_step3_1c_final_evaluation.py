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
            prefix="pref_v2_final_eval_"
        ) as tmp:
            tmp_path = Path(tmp)
            archive.extractall(
                tmp_path
            )

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
        labels=[
            0,
            1,
        ],
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "FIRST fresh holdout evaluation for Step 3.1C. "
            "Run only after the validation-only threshold candidate is frozen."
        )
    )

    parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--threshold-candidate",
        type=Path,
        default=Path(
            "preference_extractor/evaluation/results/"
            "v2_threshold_candidate_step3_1c.json"
        ),
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "preference_extractor/evaluation/datasets/"
            "preference_signal_final_holdout_v3.jsonl"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "preference_extractor/evaluation/results/"
            "step3_1c_final_holdout_FIRST_RUN.json"
        ),
    )

    args = parser.parse_args()

    if args.output.exists():
        raise RuntimeError(
            f"FIRST_RUN output already exists: {args.output}. "
            "Do not overwrite independent evidence."
        )

    threshold_payload = json.loads(
        args.threshold_candidate.read_text(
            encoding="utf-8"
        )
    )

    threshold = float(
        threshold_payload[
            "candidate"
        ][
            "threshold"
        ]
    )

    data = read_jsonl(
        args.dataset
    )

    if len(
        data
    ) != 1200:
        raise AssertionError(
            f"Expected 1200 final holdout rows, got {len(data)}"
        )

    with artifact_root(
        args.artifact
    ) as root:
        detector = PreferenceSignalDetector(
            artifact_path=str(
                root
            ),
            use_context_guard=True,
        )

        # Apply candidate in memory only.
        detector.threshold = threshold

        y_true = []
        raw_pred = []
        guarded_pred = []
        route_counts = Counter()

        family = defaultdict(
            lambda: {
                "y": [],
                "raw": [],
                "guarded": [],
            }
        )

        language = defaultdict(
            lambda: {
                "y": [],
                "raw": [],
                "guarded": [],
            }
        )

        for index, row in enumerate(
            data,
            start=1,
        ):
            y = int(
                row[
                    "label_id"
                ]
            )

            probability = (
                detector
                ._preference_probability(
                    row["text"]
                )
            )

            raw_has_signal = (
                probability
                >= threshold
            )

            raw_result = (
                PreferenceSignalResult(
                    has_preference_signal=raw_has_signal,
                    label=detector.id2label[
                        1
                        if raw_has_signal
                        else 0
                    ],
                    probability=float(
                        probability
                    ),
                    threshold=threshold,
                    decision_source="transformer",
                    transformer_has_preference_signal=raw_has_signal,
                )
            )

            guarded = (
                detector
                .apply_context_guard(
                    text=row["text"],
                    model_result=raw_result,
                )
            )

            raw_label = int(
                raw_has_signal
            )

            guarded_label = int(
                guarded.has_preference_signal
            )

            y_true.append(
                y
            )

            raw_pred.append(
                raw_label
            )

            guarded_pred.append(
                guarded_label
            )

            route_counts[
                guarded.decision_source
            ] += 1

            family[
                row[
                    "stress_family"
                ]
            ][
                "y"
            ].append(
                y
            )

            family[
                row[
                    "stress_family"
                ]
            ][
                "raw"
            ].append(
                raw_label
            )

            family[
                row[
                    "stress_family"
                ]
            ][
                "guarded"
            ].append(
                guarded_label
            )

            language[
                row[
                    "language"
                ]
            ][
                "y"
            ].append(
                y
            )

            language[
                row[
                    "language"
                ]
            ][
                "raw"
            ].append(
                raw_label
            )

            language[
                row[
                    "language"
                ]
            ][
                "guarded"
            ].append(
                guarded_label
            )

            if (
                index % 100
                == 0
                or index
                == len(
                    data
                )
            ):
                print(
                    f"[{index}/{len(data)}]"
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

    for name, values in sorted(
        family.items()
    ):
        per_family[
            name
        ] = {
            "n": len(
                values[
                    "y"
                ]
            ),
            "raw": compute_metrics(
                values[
                    "y"
                ],
                values[
                    "raw"
                ],
            ),
            "guarded": compute_metrics(
                values[
                    "y"
                ],
                values[
                    "guarded"
                ],
            ),
        }

    per_language = {}

    for name, values in sorted(
        language.items()
    ):
        per_language[
            name
        ] = {
            "n": len(
                values[
                    "y"
                ]
            ),
            "raw": compute_metrics(
                values[
                    "y"
                ],
                values[
                    "raw"
                ],
            ),
            "guarded": compute_metrics(
                values[
                    "y"
                ],
                values[
                    "guarded"
                ],
            ),
        }

    payload = {
        "step": "3.1C",
        "evaluation": (
            "fresh_final_holdout_FIRST_RUN"
        ),
        "dataset": str(
            args.dataset
        ),
        "samples": len(
            data
        ),
        "threshold_candidate": threshold,
        "threshold_source": (
            "V2 VALIDATION ONLY"
        ),
        "protocol_note": (
            "This file is FIRST_RUN evidence. "
            "Do not overwrite it. "
            "Any system modification after inspecting these results "
            "turns later evaluations on this same dataset into regression."
        ),
        "raw_transformer": raw_metrics,
        "guarded_pipeline": guarded_metrics,
        "guard_route_counts": dict(
            route_counts
        ),
        "per_family": per_family,
        "per_language": per_language,
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
        "STEP 3.1C FRESH HOLDOUT FIRST RUN"
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


if __name__ == "__main__":
    main()
