from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

from requirement_extractor_v2.explicit_pattern_resolver import (
    ExplicitPatternResolver,
)
from requirement_extractor_v2.models import (
    Quantity,
    QuantityDetection,
    QuantityDimension,
)


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def percentile(values, q):
    if not values:
        return None

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return values[lo]

    frac = pos - lo

    return (
        values[lo] * (1.0 - frac)
        + values[hi] * frac
    )


def safe_div(a, b):
    return a / b if b else 0.0


def f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0

    return (
        2.0
        * precision
        * recall
        / (precision + recall)
    )


def build_quantity(case, q):
    return Quantity(
        id=q["id"],
        raw=q["raw"],
        normalized=q.get(
            "normalized",
            q["raw"],
        ),
        value=q["value"],
        unit=q.get("unit"),
        dimension=QuantityDimension(
            q["dimension"]
        ),
        start=int(q["start"]),
        end=int(q["end"]),
        source_text=case["text"],
        detection=QuantityDetection.UNKNOWN,
        corrected=bool(
            q.get("corrected", False)
        ),
    )


def as_tuple(link):
    return (
        link.quantity_id,
        link.field.value,
        link.role.value,
    )


def gold_tuple(g):
    return (
        g["quantity_id"],
        g["field"],
        g["role"],
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Standalone benchmark for ExplicitPatternResolver. "
            "Gold quantities are injected so QuantityScanner errors "
            "do not confound this component evaluation."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--output",
        default=(
            "requirement_extractor_v2/evaluation/"
            "explicit_resolver_metrics.json"
        ),
    )

    args = parser.parse_args()

    cases = load_jsonl(
        Path(args.dataset)
    )

    resolver = (
        ExplicitPatternResolver()
    )

    tp = fp = fn = 0
    exact_cases = 0
    false_resolved_negative_quantities = 0
    negative_quantities = 0
    positive_quantities = 0
    covered_positive_quantities = 0
    field_correct = 0
    role_correct = 0
    wrong_mapping_count = 0
    latencies_ms = []

    details = []

    per_category = defaultdict(
        lambda: {
            "n_cases": 0,
            "exact_cases": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "gold_positive_quantities": 0,
            "covered_positive_quantities": 0,
            "negative_quantities": 0,
            "false_resolved_negative_quantities": 0,
        }
    )

    for case in cases:

        quantities = [
            build_quantity(
                case,
                q,
            )
            for q
            in case["quantities"]
        ]

        started = (
            time.perf_counter()
        )

        result = resolver.resolve(
            case["text"],
            quantities,
        )

        latency_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        latencies_ms.append(
            latency_ms
        )

        predicted = {
            as_tuple(link)
            for link
            in result.links
        }

        gold = {
            gold_tuple(g)
            for g
            in case["gold_links"]
        }

        case_tp = len(
            predicted & gold
        )
        case_fp = len(
            predicted - gold
        )
        case_fn = len(
            gold - predicted
        )

        tp += case_tp
        fp += case_fp
        fn += case_fn

        exact = (
            predicted == gold
        )

        exact_cases += int(
            exact
        )

        gold_by_qid = {
            qid: (field, role)
            for qid, field, role
            in gold
        }

        pred_by_qid = {
            qid: (field, role)
            for qid, field, role
            in predicted
        }

        positive_qids = set(
            gold_by_qid
        )

        all_qids = {
            q.id
            for q
            in quantities
        }

        negative_qids = (
            all_qids
            - positive_qids
        )

        positive_quantities += len(
            positive_qids
        )

        negative_quantities += len(
            negative_qids
        )

        covered = (
            positive_qids
            & set(pred_by_qid)
        )

        covered_positive_quantities += len(
            covered
        )

        false_negative_resolution_qids = (
            negative_qids
            & set(pred_by_qid)
        )

        false_resolved_negative_quantities += len(
            false_negative_resolution_qids
        )

        for qid in positive_qids:
            gold_field, gold_role = (
                gold_by_qid[qid]
            )

            pred = (
                pred_by_qid.get(qid)
            )

            if pred is None:
                continue

            pred_field, pred_role = pred

            field_correct += int(
                pred_field
                == gold_field
            )

            role_correct += int(
                pred_role
                == gold_role
            )

            if (
                pred_field != gold_field
                or pred_role != gold_role
            ):
                wrong_mapping_count += 1

        cat = per_category[
            case["category"]
        ]

        cat["n_cases"] += 1
        cat["exact_cases"] += int(
            exact
        )
        cat["tp"] += case_tp
        cat["fp"] += case_fp
        cat["fn"] += case_fn
        cat[
            "gold_positive_quantities"
        ] += len(positive_qids)
        cat[
            "covered_positive_quantities"
        ] += len(covered)
        cat[
            "negative_quantities"
        ] += len(negative_qids)
        cat[
            "false_resolved_negative_quantities"
        ] += len(
            false_negative_resolution_qids
        )

        details.append(
            {
                "id": case["id"],
                "category":
                    case["category"],
                "language":
                    case["language"],
                "text":
                    case["text"],
                "expected_behavior":
                    case[
                        "expected_behavior"
                    ],
                "gold_links": sorted(
                    list(gold)
                ),
                "predicted_links": sorted(
                    list(predicted)
                ),
                "unresolved_quantity_ids":
                    result
                    .unresolved_quantity_ids,
                "latency_ms":
                    latency_ms,
                "exact":
                    exact,
                "tp": case_tp,
                "fp": case_fp,
                "fn": case_fn,
                "notes":
                    case.get(
                        "notes",
                        "",
                    ),
            }
        )

    precision = safe_div(
        tp,
        tp + fp,
    )

    recall = safe_div(
        tp,
        tp + fn,
    )

    f1 = f1_score(
        precision,
        recall,
    )

    coverage = safe_div(
        covered_positive_quantities,
        positive_quantities,
    )

    false_resolution_rate = safe_div(
        false_resolved_negative_quantities,
        negative_quantities,
    )

    for cat, stats in (
        per_category.items()
    ):
        p = safe_div(
            stats["tp"],
            stats["tp"]
            + stats["fp"],
        )
        r = safe_div(
            stats["tp"],
            stats["tp"]
            + stats["fn"],
        )

        stats["precision"] = p
        stats["recall"] = r
        stats["f1"] = f1_score(
            p,
            r,
        )

        stats["coverage"] = safe_div(
            stats[
                "covered_positive_quantities"
            ],
            stats[
                "gold_positive_quantities"
            ],
        )

        stats[
            "false_resolution_rate"
        ] = safe_div(
            stats[
                "false_resolved_negative_quantities"
            ],
            stats[
                "negative_quantities"
            ],
        )

        stats["case_exact_accuracy"] = safe_div(
            stats["exact_cases"],
            stats["n_cases"],
        )

    metrics = {
        "n_cases": len(cases),
        "gold_links": (
            tp + fn
        ),
        "predicted_links": (
            tp + fp
        ),
        "tp": tp,
        "fp": fp,
        "fn": fn,

        "precision": precision,
        "recall": recall,
        "f1": f1,

        "positive_quantities":
            positive_quantities,

        "coverage_on_clear_quantities":
            coverage,

        "field_accuracy_on_gold_quantities":
            safe_div(
                field_correct,
                positive_quantities,
            ),

        "role_accuracy_on_gold_quantities":
            safe_div(
                role_correct,
                positive_quantities,
            ),

        "wrong_mapping_count":
            wrong_mapping_count,

        "negative_quantities":
            negative_quantities,

        "false_resolved_negative_quantities":
            false_resolved_negative_quantities,

        "false_resolution_rate_on_should_unresolved":
            false_resolution_rate,

        "case_exact_accuracy":
            safe_div(
                exact_cases,
                len(cases),
            ),

        "mean_latency_ms":
            statistics.mean(
                latencies_ms
            )
            if latencies_ms
            else None,

        "median_latency_ms":
            statistics.median(
                latencies_ms
            )
            if latencies_ms
            else None,

        "p95_latency_ms":
            percentile(
                latencies_ms,
                0.95,
            ),

        "max_latency_ms":
            max(latencies_ms)
            if latencies_ms
            else None,

        "per_category":
            dict(per_category),
    }

    output = {
        "metrics": metrics,
        "details": details,
        "benchmark_note": (
            "Diagnostic/component benchmark. Gold quantities are injected "
            "to isolate ExplicitPatternResolver. Because this set is now "
            "being used to inspect/fix the resolver, preserve it as a "
            "development/regression benchmark rather than claiming it as "
            "the final untouched end-to-end test."
        ),
    }

    Path(args.output).write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print(
        "EXPLICIT PATTERN RESOLVER METRICS"
    )
    print("=" * 80)
    print(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        )
    )

    failures = [
        item
        for item in details
        if not item["exact"]
    ]

    print()
    print(
        f"Failures: {len(failures)} / {len(cases)}"
    )

    if failures:
        print()
        print("=" * 80)
        print("FAILURES")
        print("=" * 80)

        for item in failures:
            print()
            print(
                item["id"],
                "|",
                item["category"],
                "|",
                item["language"],
            )
            print(
                "TEXT      :",
                item["text"],
            )
            print(
                "GOLD      :",
                item["gold_links"],
            )
            print(
                "PREDICTED :",
                item["predicted_links"],
            )
            print(
                "UNRESOLVED:",
                item[
                    "unresolved_quantity_ids"
                ],
            )
    else:
        print()
        print(
            "EXPLICIT PATTERN RESOLVER: "
            "ALL CASES PASSED"
        )


if __name__ == "__main__":
    main()