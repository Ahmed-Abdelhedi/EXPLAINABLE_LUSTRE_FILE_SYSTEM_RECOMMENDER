from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from preference_extractor.signal_detector.context_guard import (
    PreferenceGuardDecision,
)
from preference_extractor.signal_detector.runtime import (
    PreferenceSignalDetector,
)


DEFAULT_VALIDATION = Path(
    "preference_extractor/training/data/data_layer1_v2/"
    "preference_signal_val_v2.jsonl"
)

DEFAULT_JSON_OUTPUT = Path(
    "preference_extractor/evaluation/results/"
    "step3_1d_validation_false_negative_analysis.json"
)

DEFAULT_CSV_OUTPUT = Path(
    "preference_extractor/evaluation/results/"
    "step3_1d_validation_false_negatives.csv"
)


@contextmanager
def artifact_root(path: Path):
    path = path.resolve()

    if path.is_dir():
        yield path
        return

    if path.suffix.lower() != ".zip":
        raise ValueError(
            f"Artifact must be a directory or .zip file: {path}"
        )

    with zipfile.ZipFile(path, "r") as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(
                f"Corrupted ZIP member: {bad}"
            )

        with tempfile.TemporaryDirectory(
            prefix="preference_v2_fn_analysis_"
        ) as tmp:
            tmp_path = Path(tmp)
            archive.extractall(tmp_path)

            roots = [
                candidate
                for candidate in tmp_path.iterdir()
                if candidate.is_dir()
            ]

            if len(roots) == 1:
                yield roots[0]
            else:
                yield tmp_path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc

            rows.append(row)

    return rows


def label_of(row: dict[str, Any]) -> int:
    if "label_id" in row:
        return int(row["label_id"])

    raw = row.get("label")

    if isinstance(raw, int):
        return int(raw)

    if raw == "PREFERENCE_SIGNAL":
        return 1

    if raw == "NO_PREFERENCE_SIGNAL":
        return 0

    raise KeyError(
        "Could not resolve binary label from row. "
        "Expected label_id or label."
    )


def first_present(
    row: dict[str, Any],
    *keys: str,
    default: str = "unknown",
) -> str:
    for key in keys:
        value = row.get(key)

        if value is not None and str(value).strip():
            return str(value)

    return default


def family_of(row: dict[str, Any]) -> str:
    # The hard V2 generator may expose the family under one of these
    # names depending on the exact dataset revision.
    return first_present(
        row,
        "stress_family",
        "hard_family",
        "family",
        "category",
        default="legacy_or_unknown",
    )


def template_family_of(row: dict[str, Any]) -> str:
    return first_present(
        row,
        "template_family",
        default="unknown",
    )


def template_id_of(row: dict[str, Any]) -> str:
    return first_present(
        row,
        "template_id",
        default="unknown",
    )


def language_of(row: dict[str, Any]) -> str:
    return first_present(
        row,
        "language",
        "lang",
        default="unknown",
    )


def sample_id_of(row: dict[str, Any], index: int) -> str:
    return first_present(
        row,
        "sample_id",
        "id",
        default=f"row_{index:06d}",
    )


def probability_bucket(probability: float) -> str:
    if probability < 1e-5:
        return "<1e-5"
    if probability < 1e-4:
        return "[1e-5,1e-4)"
    if probability < 1e-3:
        return "[1e-4,1e-3)"
    if probability < 1e-2:
        return "[1e-3,1e-2)"
    if probability < 5e-2:
        return "[1e-2,0.05)"
    if probability < 0.5:
        return "[0.05,0.5)"
    if probability < 0.9:
        return "[0.5,0.9)"
    if probability < 0.99:
        return "[0.9,0.99)"
    if probability < 0.999:
        return "[0.99,0.999)"
    return ">=0.999"


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None

    return float(sum(values) / len(values))


def safe_median(values: list[float]) -> float | None:
    if not values:
        return None

    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2

    if n % 2:
        return float(ordered[mid])

    return float(
        (ordered[mid - 1] + ordered[mid]) / 2.0
    )


def safe_min(values: list[float]) -> float | None:
    return float(min(values)) if values else None


def safe_max(values: list[float]) -> float | None:
    return float(max(values)) if values else None


def summarize_group(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    probabilities = [
        float(record["probability"])
        for record in records
    ]

    return {
        "count": len(records),
        "probability_min": safe_min(probabilities),
        "probability_median": safe_median(probabilities),
        "probability_mean": safe_mean(probabilities),
        "probability_max": safe_max(probabilities),
        "languages": dict(
            Counter(
                record["language"]
                for record in records
            )
        ),
        "probability_buckets": dict(
            Counter(
                record["probability_bucket"]
                for record in records
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Step 3.1D: analyze V2 VALIDATION false negatives only. "
            "This script never reads the previously inspected holdout."
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
        default=DEFAULT_VALIDATION,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Optional threshold override. "
            "Default: use the threshold stored in the artifact."
        ),
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
    )

    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_CSV_OUTPUT,
    )

    args = parser.parse_args()

    rows = read_jsonl(args.validation)

    if len(rows) != 5200:
        raise AssertionError(
            f"Expected 5200 validation examples, got {len(rows)}."
        )

    with artifact_root(args.artifact) as root:
        detector = PreferenceSignalDetector(
            artifact_path=str(root),
            use_context_guard=True,
        )

        threshold = (
            float(args.threshold)
            if args.threshold is not None
            else float(detector.threshold)
        )

        positives_total = 0
        raw_false_negatives: list[dict[str, Any]] = []
        guarded_false_negatives: list[dict[str, Any]] = []

        raw_confusion = Counter()
        guarded_confusion = Counter()
        guard_routes = Counter()

        positive_probabilities: list[float] = []
        negative_probabilities: list[float] = []

        for index, row in enumerate(
            rows,
            start=1,
        ):
            y_true = label_of(row)

            probability = float(
                detector._preference_probability(
                    row["text"]
                )
            )

            raw_pred = int(
                probability >= threshold
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
                guarded_pred = 0
                guard_route = "FORCE_NO_SIGNAL"

            elif (
                guard_result.decision
                == PreferenceGuardDecision.FORCE_SIGNAL
            ):
                guarded_pred = 1
                guard_route = "FORCE_SIGNAL"

            else:
                guarded_pred = raw_pred
                guard_route = "PASS_TO_MODEL"

            guard_routes[guard_route] += 1

            if y_true == 1:
                positives_total += 1
                positive_probabilities.append(
                    probability
                )
            else:
                negative_probabilities.append(
                    probability
                )

            raw_confusion[
                (y_true, raw_pred)
            ] += 1

            guarded_confusion[
                (y_true, guarded_pred)
            ] += 1

            record = {
                "row_index": index,
                "sample_id": sample_id_of(
                    row,
                    index,
                ),
                "language": language_of(row),
                "family": family_of(row),
                "template_family": template_family_of(row),
                "template_id": template_id_of(row),
                "probability": probability,
                "threshold": threshold,
                "probability_bucket": probability_bucket(
                    probability
                ),
                "guard_route": guard_route,
                "text": str(row["text"]),
            }

            if (
                y_true == 1
                and raw_pred == 0
            ):
                raw_false_negatives.append(
                    record.copy()
                )

            if (
                y_true == 1
                and guarded_pred == 0
            ):
                guarded_false_negatives.append(
                    record.copy()
                )

            if (
                index % 250 == 0
                or index == len(rows)
            ):
                print(
                    f"[{index}/{len(rows)}] analyzed"
                )

    def cm_payload(counter: Counter) -> dict[str, int]:
        return {
            "tn": int(counter[(0, 0)]),
            "fp": int(counter[(0, 1)]),
            "fn": int(counter[(1, 0)]),
            "tp": int(counter[(1, 1)]),
        }

    raw_cm = cm_payload(raw_confusion)
    guarded_cm = cm_payload(guarded_confusion)

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_template_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_template_id: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in guarded_false_negatives:
        by_family[
            record["family"]
        ].append(record)

        by_language[
            record["language"]
        ].append(record)

        by_template_family[
            record["template_family"]
        ].append(record)

        by_template_id[
            record["template_id"]
        ].append(record)

    family_summary = {
        key: summarize_group(value)
        for key, value in sorted(
            by_family.items(),
            key=lambda item: (
                -len(item[1]),
                item[0],
            ),
        )
    }

    language_summary = {
        key: summarize_group(value)
        for key, value in sorted(
            by_language.items(),
            key=lambda item: (
                -len(item[1]),
                item[0],
            ),
        )
    }

    template_family_summary = {
        key: summarize_group(value)
        for key, value in sorted(
            by_template_family.items(),
            key=lambda item: (
                -len(item[1]),
                item[0],
            ),
        )
    }

    template_id_summary = {
        key: summarize_group(value)
        for key, value in sorted(
            by_template_id.items(),
            key=lambda item: (
                -len(item[1]),
                item[0],
            ),
        )
        if key != "unknown"
    }

    probability_bucket_summary = dict(
        Counter(
            record["probability_bucket"]
            for record in guarded_false_negatives
        )
    )

    guard_route_summary_on_fn = dict(
        Counter(
            record["guard_route"]
            for record in guarded_false_negatives
        )
    )

    fn_probabilities = [
        float(record["probability"])
        for record in guarded_false_negatives
    ]

    report = {
        "step": "3.1D",
        "scope": (
            "V2 VALIDATION false-negative analysis only"
        ),
        "protocol_note": (
            "This analysis intentionally reads only "
            "preference_signal_val_v2.jsonl. "
            "No holdout is read, generated or evaluated."
        ),
        "validation_dataset": str(args.validation),
        "validation_size": len(rows),
        "threshold": threshold,
        "positive_examples": positives_total,
        "raw_transformer_confusion": raw_cm,
        "guarded_pipeline_confusion": guarded_cm,
        "guard_routes_all_validation": dict(
            guard_routes
        ),
        "raw_false_negative_count": len(
            raw_false_negatives
        ),
        "guarded_false_negative_count": len(
            guarded_false_negatives
        ),
        "guarded_false_negative_rate_among_positives": (
            len(guarded_false_negatives)
            / positives_total
            if positives_total
            else 0.0
        ),
        "fn_probability_summary": {
            "min": safe_min(fn_probabilities),
            "median": safe_median(fn_probabilities),
            "mean": safe_mean(fn_probabilities),
            "max": safe_max(fn_probabilities),
            "buckets": probability_bucket_summary,
        },
        "positive_probability_summary": {
            "min": safe_min(positive_probabilities),
            "median": safe_median(positive_probabilities),
            "mean": safe_mean(positive_probabilities),
            "max": safe_max(positive_probabilities),
        },
        "negative_probability_summary": {
            "min": safe_min(negative_probabilities),
            "median": safe_median(negative_probabilities),
            "mean": safe_mean(negative_probabilities),
            "max": safe_max(negative_probabilities),
        },
        "false_negatives_by_family": family_summary,
        "false_negatives_by_language": language_summary,
        "false_negatives_by_template_family": template_family_summary,
        "false_negatives_by_template_id": template_id_summary,
        "guard_routes_on_false_negatives": guard_route_summary_on_fn,
        "all_guarded_false_negatives": guarded_false_negatives,
    }

    args.json_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.csv_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.json_output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    fieldnames = [
        "row_index",
        "sample_id",
        "language",
        "family",
        "template_family",
        "template_id",
        "probability",
        "threshold",
        "probability_bucket",
        "guard_route",
        "text",
    ]

    with args.csv_output.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in guarded_false_negatives:
            writer.writerow(
                {
                    key: record.get(key)
                    for key in fieldnames
                }
            )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STEP 3.1D — V2 VALIDATION FALSE NEGATIVE ANALYSIS"
    )

    print(
        "=" * 80
    )

    print(
        f"threshold: {threshold}"
    )

    print(
        f"raw FN: {len(raw_false_negatives)}"
    )

    print(
        f"guarded FN: {len(guarded_false_negatives)}"
    )

    print(
        f"guard routes: {dict(guard_routes)}"
    )

    print(
        "\nFALSE NEGATIVES BY FAMILY"
    )

    if family_summary:
        for family, metrics in family_summary.items():
            print(
                f"  {family}: "
                f"{metrics['count']} "
                f"(median p={metrics['probability_median']:.8g})"
            )
    else:
        print(
            "  none"
        )

    print(
        "\nFALSE NEGATIVES BY LANGUAGE"
    )

    if language_summary:
        for language, metrics in language_summary.items():
            print(
                f"  {language}: "
                f"{metrics['count']} "
                f"(median p={metrics['probability_median']:.8g})"
            )
    else:
        print(
            "  none"
        )

    print(
        "\nFALSE NEGATIVE PROBABILITY BUCKETS"
    )

    for bucket, count in sorted(
        probability_bucket_summary.items()
    ):
        print(
            f"  {bucket}: {count}"
        )

    print(
        f"\nSaved JSON: {args.json_output}"
    )

    print(
        f"Saved CSV:  {args.csv_output}"
    )

    print(
        "\nIMPORTANT: inspect/analyze this VALIDATION report only. "
        "Do not touch the fresh final holdout yet."
    )


if __name__ == "__main__":
    main()
