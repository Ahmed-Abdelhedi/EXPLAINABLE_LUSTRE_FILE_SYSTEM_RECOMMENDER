from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from preference_extractor.signal_detector.runtime import PreferenceSignalDetector


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate(detector, data):
    y_true, raw_pred, guarded_pred = [], [], []
    routes = Counter()

    for row in data:
        text = row["text"]
        y = int(row.get("label_id", row["label"]))
        raw = detector.predict_model_only(text)
        guarded = detector.apply_context_guard(text=text, model_result=raw)

        y_true.append(y)
        raw_pred.append(int(raw.has_preference_signal))
        guarded_pred.append(int(guarded.has_preference_signal))
        routes[guarded.decision_source] += 1

    return {
        "raw": compute(y_true, raw_pred),
        "guarded": compute(y_true, guarded_pred),
        "guard_routes": dict(routes),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-artifact", required=True, type=Path)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("preference_extractor/evaluation/datasets/preference_signal_test_v1.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("preference_extractor/evaluation/signal_detector_v1_v2_regression_comparison.json"),
    )
    args = parser.parse_args()

    data = read_jsonl(args.dataset)

    v1 = PreferenceSignalDetector(use_context_guard=True)
    v2 = PreferenceSignalDetector(
        artifact_path=str(args.v2_artifact),
        use_context_guard=True,
    )

    payload = {
        "benchmark": "known_500_case_regression_benchmark",
        "protocol_note": (
            "This 500-case dataset was inspected before V2 development. "
            "These numbers are regression evidence only, not independent evidence."
        ),
        "v1": evaluate(v1, data),
        "v2": evaluate(v2, data),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
