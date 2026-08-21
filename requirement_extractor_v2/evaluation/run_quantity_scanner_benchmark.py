from __future__ import annotations

import argparse
import json
import math
import re

from pathlib import Path
from typing import Any, Optional

from requirement_extractor_v2.quantity_scanner import QuantityScanner
from requirement_extractor_v2.models import ParamName
from requirement_extractor_v2.unit_normalizer import normalize_unit_value


# =====================================================================
# DATASET
# =====================================================================


def load_jsonl(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


# =====================================================================
# NUMERIC HELPERS
# =====================================================================


def num_equal(
    a: Any,
    b: Any,
    tol: float = 1e-9,
) -> bool:
    """
    Compare two numerical values safely.
    """

    try:
        return math.isclose(
            float(a),
            float(b),
            rel_tol=tol,
            abs_tol=tol,
        )

    except Exception:
        return a == b


def norm_unit(
    unit: Optional[str],
) -> Optional[str]:
    """
    Minimal textual normalization for unit comparison.
    """

    if unit is None:
        return None

    return (
        str(unit)
        .strip()
        .lower()
        .replace(" ", "")
    )


# =====================================================================
# EXPLICIT UNIT DETECTION IN USER TEXT
# =====================================================================


_TEXT_UNIT_RE = re.compile(
    r"(?:"
    r"GB/s|GBps|Gbps|"
    r"MB/s|MBps|Mbps|"
    r"TiB|TB|GiB|GB|MiB|MB|"
    r"MW|kW|W|"
    r"megawatts?|kilowatts?|watts?|"
    r"USD|dollars?|\$|"
    r"%|percent|pourcent"
    r")",
    flags=re.IGNORECASE,
)


def text_contains_explicit_unit(
    text: str,
) -> bool:
    """
    True only when the CURRENT user message contains an explicit unit.

    Important:
        A unit inherited from a previous clarification question must not be
        credited to QuantityScanner.
    """

    return bool(
        _TEXT_UNIT_RE.search(text or "")
    )


# =====================================================================
# CANONICALIZATION FOR EVALUATION
# =====================================================================


def canonicalize_prediction(
    pred,
    gold: dict,
):
    """
    Convert a QuantityScanner prediction to the final canonical unit
    ONLY FOR BENCHMARK COMPARISON.

    Example:

        Scanner:
            4 kW

        Gold end-to-end representation:
            4000 W

    This does NOT change the scanner itself.

    The actual V2 architecture performs this conversion later, after
    semantic FIELD resolution.
    """

    field = gold.get("field")

    if not field:
        return pred.value, pred.unit

    if pred.unit is None:
        return pred.value, pred.unit

    try:
        param = ParamName(field)

        normalized_value, normalized_unit = (
            normalize_unit_value(
                param,
                pred.value,
                pred.unit,
            )
        )

        return (
            normalized_value,
            normalized_unit,
        )

    except Exception:
        return (
            pred.value,
            pred.unit,
        )


# =====================================================================
# GOLD / PREDICTION MATCHING
# =====================================================================


def prediction_matches_gold(
    pred,
    gold: dict,
) -> bool:
    """
    Determine whether a scanner prediction corresponds to a gold quantity.

    Two possibilities:

    1. The numerical value is already identical.
       Example:
           200 clients -> 200

    2. The detected value/unit becomes identical after canonical unit
       conversion.
       Example:
           4 kW -> 4000 W

    Missing units are NOT invented here.
    """

    # -------------------------------------------------------------
    # Direct numerical match
    # -------------------------------------------------------------

    if num_equal(
        pred.value,
        gold["value"],
    ):
        return True

    # -------------------------------------------------------------
    # Canonical numerical match
    # -------------------------------------------------------------

    if pred.unit is None:
        return False

    normalized_value, _ = (
        canonicalize_prediction(
            pred,
            gold,
        )
    )

    return num_equal(
        normalized_value,
        gold["value"],
    )


def match_predictions(
    predictions,
    golds,
):
    """
    One-to-one matching between gold quantities and scanner predictions.

    Returns:
        matched pairs
        number of missed gold quantities
        number of extra predictions
    """

    used_prediction_indexes = set()

    matches = []

    for gold in golds:

        found_index = None

        for prediction_index, prediction in enumerate(
            predictions
        ):

            if (
                prediction_index
                in used_prediction_indexes
            ):
                continue

            if prediction_matches_gold(
                prediction,
                gold,
            ):
                found_index = (
                    prediction_index
                )
                break

        if found_index is not None:

            used_prediction_indexes.add(
                found_index
            )

            matches.append(
                (
                    gold,
                    predictions[
                        found_index
                    ],
                )
            )

    missed = (
        len(golds)
        - len(matches)
    )

    extra = (
        len(predictions)
        - len(used_prediction_indexes)
    )

    return (
        matches,
        missed,
        extra,
    )


# =====================================================================
# UNIT EVALUATION
# =====================================================================


def should_score_unit(
    row: dict,
    gold: dict,
) -> bool:
    """
    Decide whether unit detection belongs to QuantityScanner responsibility.

    Example that SHOULD be scored:

        "Maximum power is 4 kW."

    Example that MUST NOT be scored:

        Previous question:
            "Maximum power in watts?"

        User:
            "200"

    In the second case, W comes from ConversationScopeResolver rather than
    from QuantityScanner.
    """

    if gold.get("unit") is None:
        return False

    # Unit explicitly visible in current message:
    # QuantityScanner is responsible for detecting it.
    if text_contains_explicit_unit(
        row["text"]
    ):
        return True

    # No unit in current message.
    # If this is a contextual answer, the unit belongs to conversation scope.
    if (
        row.get("expected_scope")
        == "ANSWER_TO_PREVIOUS_QUESTION"
    ):
        return False

    # For non-contextual cases, the benchmark expected a unit despite the
    # absence of an explicit token. Keep the case scoreable so such problems
    # remain visible.
    return True


def unit_matches_gold(
    prediction,
    gold: dict,
) -> bool:
    """
    Compare prediction unit with gold canonical unit.

    Example:

        prediction = 4 kW
        gold       = 4000 W

    -> unit is correct because kW converts to canonical W.

    Important:
        pred.unit=None is never considered correct when unit scoring applies.
    """

    if prediction.unit is None:
        return False

    _, normalized_unit = (
        canonicalize_prediction(
            prediction,
            gold,
        )
    )

    return (
        norm_unit(normalized_unit)
        ==
        norm_unit(
            gold.get("unit")
        )
    )


# =====================================================================
# METRIC HELPERS
# =====================================================================


def safe_precision(
    tp: int,
    fp: int,
):
    denominator = tp + fp

    if denominator == 0:
        return None

    return tp / denominator


def safe_recall(
    tp: int,
    fn: int,
):
    denominator = tp + fn

    if denominator == 0:
        return None

    return tp / denominator


def safe_f1(
    precision,
    recall,
):
    if (
        precision is None
        or recall is None
    ):
        return None

    if (
        precision + recall
        == 0
    ):
        return 0.0

    return (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
    )


# =====================================================================
# BENCHMARK
# =====================================================================


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--output",
        default=(
            "quantity_scanner_metrics.json"
        ),
    )

    args = parser.parse_args()

    rows = load_jsonl(
        Path(args.dataset)
    )

    scanner = QuantityScanner()

    # =================================================================
    # GLOBAL COUNTERS
    # =================================================================

    tp = 0
    fp = 0
    fn = 0

    matched_total = 0

    normalized_value_ok = 0

    unit_ok = 0
    unit_total = 0

    negative_messages = 0
    negative_messages_with_false_positive = 0

    per_category = {}

    details = []

    # =================================================================
    # RUN
    # =================================================================

    for row in rows:

        golds = row.get(
            "gold_quantities",
            [],
        )

        predictions = scanner.scan(
            row["text"]
        )

        # -------------------------------------------------------------
        # Negative messages
        # -------------------------------------------------------------

        if not golds:

            negative_messages += 1

            if predictions:
                negative_messages_with_false_positive += 1

        # -------------------------------------------------------------
        # Match quantities
        # -------------------------------------------------------------

        matches, missed, extra = (
            match_predictions(
                predictions,
                golds,
            )
        )

        tp += len(matches)
        fn += missed
        fp += extra

        matched_total += len(matches)

        # -------------------------------------------------------------
        # Category counters
        # -------------------------------------------------------------

        category = row["category"]

        category_metrics = (
            per_category.setdefault(
                category,
                {
                    "messages": 0,

                    "tp": 0,
                    "fp": 0,
                    "fn": 0,

                    "matched": 0,

                    "value_ok": 0,

                    "unit_ok": 0,
                    "unit_total": 0,

                    "negative_messages": 0,
                    "negative_messages_with_fp": 0,
                },
            )
        )

        category_metrics[
            "messages"
        ] += 1

        category_metrics[
            "tp"
        ] += len(matches)

        category_metrics[
            "fp"
        ] += extra

        category_metrics[
            "fn"
        ] += missed

        category_metrics[
            "matched"
        ] += len(matches)

        if not golds:

            category_metrics[
                "negative_messages"
            ] += 1

            if predictions:
                category_metrics[
                    "negative_messages_with_fp"
                ] += 1

        # -------------------------------------------------------------
        # Pair-level evaluation
        # -------------------------------------------------------------

        pair_results = []

        for gold, prediction in matches:

            normalized_value, normalized_unit = (
                canonicalize_prediction(
                    prediction,
                    gold,
                )
            )

            value_is_correct = (
                num_equal(
                    normalized_value,
                    gold["value"],
                )
                or
                num_equal(
                    prediction.value,
                    gold["value"],
                )
            )

            normalized_value_ok += int(
                value_is_correct
            )

            category_metrics[
                "value_ok"
            ] += int(
                value_is_correct
            )

            score_unit = (
                should_score_unit(
                    row,
                    gold,
                )
            )

            unit_is_correct = None

            if score_unit:

                unit_total += 1

                category_metrics[
                    "unit_total"
                ] += 1

                unit_is_correct = (
                    unit_matches_gold(
                        prediction,
                        gold,
                    )
                )

                unit_ok += int(
                    unit_is_correct
                )

                category_metrics[
                    "unit_ok"
                ] += int(
                    unit_is_correct
                )

            pair_results.append(
                {
                    "gold":
                        gold,

                    "prediction":
                        prediction.to_dict(),

                    "normalized_prediction":
                        {
                            "value":
                                normalized_value,

                            "unit":
                                normalized_unit,
                        },

                    "value_ok":
                        value_is_correct,

                    "unit_scored":
                        score_unit,

                    "unit_ok":
                        unit_is_correct,
                }
            )

        # -------------------------------------------------------------
        # Detail row
        # -------------------------------------------------------------

        details.append(
            {
                "id":
                    row["id"],

                "category":
                    category,

                "language":
                    row.get(
                        "language"
                    ),

                "text":
                    row["text"],

                "expected_scope":
                    row.get(
                        "expected_scope"
                    ),

                "gold":
                    golds,

                "predicted":
                    [
                        prediction.to_dict()
                        for prediction
                        in predictions
                    ],

                "matched_pairs":
                    pair_results,

                "missed":
                    missed,

                "extra":
                    extra,

                "quantity_detection_ok":
                    (
                        missed == 0
                        and extra == 0
                    ),
            }
        )

    # =================================================================
    # GLOBAL METRICS
    # =================================================================

    precision = safe_precision(
        tp,
        fp,
    )

    recall = safe_recall(
        tp,
        fn,
    )

    f1 = safe_f1(
        precision,
        recall,
    )

    normalized_value_accuracy = (
        normalized_value_ok
        / matched_total
        if matched_total
        else None
    )

    unit_accuracy = (
        unit_ok
        / unit_total
        if unit_total
        else None
    )

    false_positive_rate_per_message = (
        fp
        / len(rows)
        if rows
        else 0.0
    )

    negative_message_false_positive_rate = (
        negative_messages_with_false_positive
        / negative_messages
        if negative_messages
        else None
    )

    # =================================================================
    # PER-CATEGORY METRICS
    # =================================================================

    for (
        category,
        metrics,
    ) in per_category.items():

        category_precision = (
            safe_precision(
                metrics["tp"],
                metrics["fp"],
            )
        )

        category_recall = (
            safe_recall(
                metrics["tp"],
                metrics["fn"],
            )
        )

        category_f1 = (
            safe_f1(
                category_precision,
                category_recall,
            )
        )

        metrics[
            "precision"
        ] = category_precision

        metrics[
            "recall"
        ] = category_recall

        metrics[
            "f1"
        ] = category_f1

        metrics[
            "normalized_value_accuracy"
        ] = (
            metrics["value_ok"]
            / metrics["matched"]
            if metrics["matched"]
            else None
        )

        metrics[
            "unit_accuracy"
        ] = (
            metrics["unit_ok"]
            / metrics["unit_total"]
            if metrics["unit_total"]
            else None
        )

        denominator = (
            metrics["tp"]
            + metrics["fp"]
            + metrics["fn"]
        )

        metrics[
            "failure_rate"
        ] = (
            (
                metrics["fp"]
                + metrics["fn"]
            )
            / denominator
            if denominator
            else 0.0
        )

        metrics[
            "negative_message_false_positive_rate"
        ] = (
            metrics[
                "negative_messages_with_fp"
            ]
            / metrics[
                "negative_messages"
            ]
            if metrics[
                "negative_messages"
            ]
            else None
        )

    # =================================================================
    # FINAL RESULT
    # =================================================================

    metrics = {
        "n_messages":
            len(rows),

        "tp":
            tp,

        "fp":
            fp,

        "fn":
            fn,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "normalized_value_accuracy":
            normalized_value_accuracy,

        "unit_accuracy":
            unit_accuracy,

        "false_positive_rate_per_message":
            false_positive_rate_per_message,

        "negative_messages":
            negative_messages,

        "negative_messages_with_false_positive":
            negative_messages_with_false_positive,

        "negative_message_false_positive_rate":
            negative_message_false_positive_rate,

        "per_category":
            per_category,
    }

    result = {
        "metrics":
            metrics,

        "details":
            details,
    }

    output_path = Path(
        args.output
    )

    output_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()