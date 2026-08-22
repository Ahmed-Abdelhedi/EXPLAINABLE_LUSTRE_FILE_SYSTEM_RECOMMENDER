import json
import time
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

from preference_extractor.signal_detector.runtime import (
    PreferenceSignalDetector
)


DATASET = Path(
    "preference_extractor/evaluation/datasets/"
    "preference_signal_test_v1.jsonl"
)


OUTPUT = Path(
    "preference_extractor/evaluation/"
    "signal_detector_metrics.json"
)



def load_dataset():

    with open(
        DATASET,
        encoding="utf-8"
    ) as f:

        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]



def main():

    detector = PreferenceSignalDetector()

    data = load_dataset()

    y_true = []
    y_pred = []

    latencies = []


    for sample in data:

        start = time.perf_counter()

        result = detector.predict(
            sample["text"]
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        latencies.append(
            elapsed * 1000
        )

        prediction = int(
            result.has_preference_signal
        )

        expected = sample["label"]

        if prediction != expected:
            print("\n================ ERROR ================")
            print("ID:", sample["id"])
            print("TEXT:", sample["text"])
            print("CATEGORY:", sample["category"])
            print("EXPECTED:", expected)
            print("PREDICTED:", prediction)
            print("PROBABILITY:", result.probability)
            print("========================================\n")

        y_true.append(expected)
        y_pred.append(prediction)


    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary"
        )
    )


    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()


    metrics = {

        "samples": len(data),

        "accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "false_positive":
            int(fp),

        "false_negative":
            int(fn),

        "avg_latency_ms":
            sum(latencies)
            /
            len(latencies)

    }


    OUTPUT.write_text(
        json.dumps(
            metrics,
            indent=2
        ),
        encoding="utf-8"
    )


    print(
        json.dumps(
            metrics,
            indent=2
        )
    )



if __name__ == "__main__":
    main()
