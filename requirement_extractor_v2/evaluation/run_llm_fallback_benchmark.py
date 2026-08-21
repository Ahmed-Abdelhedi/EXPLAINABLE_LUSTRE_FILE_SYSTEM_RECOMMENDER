from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from requirement_extractor_v2.llm_fallback_extractor import (
    LLMFallbackExtractor,
)
from requirement_extractor_v2.models import (
    Quantity,
    QuantityDetection,
    QuantityDimension,
)


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def percentile(values, q):
    if not values:
        return None

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return values[lower]

    fraction = position - lower

    return (
        values[lower] * (1.0 - fraction)
        + values[upper] * fraction
    )


def build_quantity(case):
    q = case["quantity"]

    return Quantity(
        id=q["id"],
        raw=q["raw"],
        normalized=q.get("normalized") or q["raw"],
        value=q["value"],
        unit=q.get("unit"),
        dimension=QuantityDimension(q["dimension"]),
        start=int(q["start"]),
        end=int(q["end"]),
        source_text=case["text"],
        detection=QuantityDetection.UNKNOWN,
        corrected=bool(q.get("corrected", False)),
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Standalone component benchmark for the guarded Ollama "
            "LLM fallback. The quantity is fixed; only FIELD + ROLE "
            "classification and safe abstention are evaluated."
        )
    )

    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--output",
        default=(
            "requirement_extractor_v2/evaluation/"
            "llm_fallback_metrics.json"
        ),
    )
    parser.add_argument(
        "--host",
        default="http://localhost:11434",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-coder:7b",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of unmeasured warm-up calls before the benchmark.",
    )

    args = parser.parse_args()

    cases = load_jsonl(
        Path(args.dataset)
    )

    fallback = LLMFallbackExtractor(
        enabled=True,
        host=args.host,
        model=args.model,
    )

    # ---------------------------------------------------------------
    # Warm-up
    # ---------------------------------------------------------------

    if args.warmup > 0 and cases:
        warm_case = next(
            (
                c for c in cases
                if c["expected"] == "ABSTAIN"
            ),
            cases[0],
        )

        print(
            f"Warming up {args.model} "
            f"with {args.warmup} call(s)..."
        )

        for _ in range(args.warmup):
            fallback.resolve_quantity(
                user_text=warm_case["text"],
                quantity=build_quantity(warm_case),
                previous_question=warm_case.get(
                    "previous_question"
                ),
            )

        # Keep benchmark metrics independent from warm-up calls.
        fallback.call_count = 0
        fallback.call_log.clear()

    details = []
    latencies = []

    field_correct = 0
    role_correct = 0
    pair_correct = 0
    expected_resolve_count = 0
    expected_abstain_count = 0
    correct_abstentions = 0
    false_resolutions = 0
    unresolved_on_resolvable = 0
    overall_correct = 0

    category_stats = defaultdict(
        lambda: {
            "n": 0,
            "correct": 0,
            "expected_resolve": 0,
            "resolved_correct": 0,
            "expected_abstain": 0,
            "correct_abstain": 0,
            "false_resolution": 0,
            "latencies_s": [],
        }
    )

    language_stats = defaultdict(
        lambda: {
            "n": 0,
            "correct": 0,
        }
    )

    print()
    print(
        f"Running {len(cases)} LLM fallback cases "
        f"with model={args.model}"
    )

    for index, case in enumerate(cases, start=1):
        quantity = build_quantity(case)

        before_log = len(
            fallback.call_log
        )

        started = time.perf_counter()

        link = fallback.resolve_quantity(
            user_text=case["text"],
            quantity=quantity,
            previous_question=case.get(
                "previous_question"
            ),
        )

        latency = (
            time.perf_counter()
            - started
        )

        latencies.append(
            latency
        )

        new_logs = fallback.call_log[
            before_log:
        ]

        last_log = (
            new_logs[-1]
            if new_logs
            else {}
        )

        status = last_log.get(
            "status",
            "unknown",
        )

        expected = case["expected"]
        predicted_field = (
            None
            if link is None
            else link.field.value
        )
        predicted_role = (
            None
            if link is None
            else link.role.value
        )

        case_correct = False
        field_ok = None
        role_ok = None
        pair_ok = None

        if expected == "RESOLVE":
            expected_resolve_count += 1

            field_ok = (
                predicted_field
                == case["expected_field"]
            )
            role_ok = (
                predicted_role
                == case["expected_role"]
            )
            pair_ok = (
                field_ok
                and role_ok
            )

            field_correct += int(
                field_ok
            )
            role_correct += int(
                role_ok
            )
            pair_correct += int(
                pair_ok
            )

            if link is None:
                unresolved_on_resolvable += 1

            case_correct = bool(
                pair_ok
            )

        elif expected == "ABSTAIN":
            expected_abstain_count += 1

            if link is None:
                correct_abstentions += 1
                case_correct = True
            else:
                false_resolutions += 1
                case_correct = False

        else:
            raise ValueError(
                f"Unknown expected={expected!r}"
            )

        overall_correct += int(
            case_correct
        )

        category = case["category"]
        language = case["language"]

        cat = category_stats[
            category
        ]
        cat["n"] += 1
        cat["correct"] += int(
            case_correct
        )
        cat["latencies_s"].append(
            latency
        )

        if expected == "RESOLVE":
            cat["expected_resolve"] += 1
            cat["resolved_correct"] += int(
                bool(pair_ok)
            )
        else:
            cat["expected_abstain"] += 1
            cat["correct_abstain"] += int(
                link is None
            )
            cat["false_resolution"] += int(
                link is not None
            )

        language_stats[
            language
        ]["n"] += 1
        language_stats[
            language
        ]["correct"] += int(
            case_correct
        )

        details.append(
            {
                "id": case["id"],
                "category": category,
                "language": language,
                "text": case["text"],
                "expected": expected,
                "expected_field": case.get(
                    "expected_field"
                ),
                "expected_role": case.get(
                    "expected_role"
                ),
                "predicted_field": predicted_field,
                "predicted_role": predicted_role,
                "llm_status": status,
                "latency_s": latency,
                "correct": case_correct,
                "field_correct": field_ok,
                "role_correct": role_ok,
                "pair_correct": pair_ok,
                "call_log": new_logs,
                "notes": case.get(
                    "notes",
                    "",
                ),
            }
        )

        print(
            f"[{index:02d}/{len(cases)}] "
            f"{case['id']} "
            f"{expected:<7} "
            f"status={status:<24} "
            f"latency={latency:.3f}s "
            f"{'PASS' if case_correct else 'FAIL'}"
        )

    status_distribution = Counter(
        entry.get(
            "status",
            "unknown",
        )
        for entry
        in fallback.call_log
    )

    for category, stats in category_stats.items():
        stats["accuracy"] = (
            stats["correct"]
            / stats["n"]
            if stats["n"]
            else None
        )
        stats["mean_latency_s"] = (
            statistics.mean(
                stats["latencies_s"]
            )
            if stats["latencies_s"]
            else None
        )
        stats["p95_latency_s"] = (
            percentile(
                stats["latencies_s"],
                0.95,
            )
        )
        del stats[
            "latencies_s"
        ]

    for language, stats in language_stats.items():
        stats["accuracy"] = (
            stats["correct"]
            / stats["n"]
            if stats["n"]
            else None
        )

    metrics = {
        "n_cases": len(cases),
        "model": args.model,
        "host": args.host,
        "llm_calls": fallback.call_count,

        "overall_accuracy": (
            overall_correct
            / len(cases)
            if cases
            else None
        ),

        "expected_resolve_count":
            expected_resolve_count,

        "field_accuracy_on_resolvable": (
            field_correct
            / expected_resolve_count
            if expected_resolve_count
            else None
        ),

        "role_accuracy_on_resolvable": (
            role_correct
            / expected_resolve_count
            if expected_resolve_count
            else None
        ),

        "pair_accuracy_on_resolvable": (
            pair_correct
            / expected_resolve_count
            if expected_resolve_count
            else None
        ),

        "unresolved_on_resolvable_count":
            unresolved_on_resolvable,

        "expected_abstain_count":
            expected_abstain_count,

        "abstention_accuracy": (
            correct_abstentions
            / expected_abstain_count
            if expected_abstain_count
            else None
        ),

        "false_resolution_count":
            false_resolutions,

        "false_resolution_rate_on_abstain_cases": (
            false_resolutions
            / expected_abstain_count
            if expected_abstain_count
            else None
        ),

        "mean_latency_s": (
            statistics.mean(latencies)
            if latencies
            else None
        ),

        "median_latency_s": (
            statistics.median(latencies)
            if latencies
            else None
        ),

        "p95_latency_s": (
            percentile(
                latencies,
                0.95,
            )
        ),

        "max_latency_s": (
            max(latencies)
            if latencies
            else None
        ),

        "status_distribution":
            dict(
                status_distribution
            ),

        "per_category":
            dict(
                category_stats
            ),

        "per_language":
            dict(
                language_stats
            ),
    }

    output = {
        "metrics": metrics,
        "details": details,
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
    print("LLM FALLBACK METRICS")
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
        if not item["correct"]
    ]

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
            )
            print(
                "TEXT      :",
                item["text"],
            )
            print(
                "EXPECTED  :",
                item["expected"],
                item["expected_field"],
                item["expected_role"],
            )
            print(
                "PREDICTED :",
                item["predicted_field"],
                item["predicted_role"],
            )
            print(
                "LLM STATUS:",
                item["llm_status"],
            )
            print(
                "LATENCY   :",
                round(
                    item["latency_s"],
                    3,
                ),
                "s",
            )
    else:
        print()
        print(
            "LLM FALLBACK BENCHMARK: "
            "ALL CASES PASSED"
        )


if __name__ == "__main__":
    main()