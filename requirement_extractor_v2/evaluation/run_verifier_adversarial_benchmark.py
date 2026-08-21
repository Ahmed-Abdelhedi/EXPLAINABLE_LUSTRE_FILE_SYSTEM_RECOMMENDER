from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from requirement_extractor_v2.deterministic_verifier import DeterministicVerifier
from requirement_extractor_v2.models import (
    ParamName,
    Quantity,
    QuantityDetection,
    QuantityDimension,
    SemanticLink,
    SemanticRole,
)


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_quantity(case):
    q = case["quantity"]
    text = case["source_text"]
    raw = q["raw"]
    start = text.find(raw)
    if start < 0:
        start = 0
    end = start + len(raw) if raw else start

    return Quantity(
        id=q["id"],
        raw=raw,
        normalized=q.get("normalized") or raw,
        value=q.get("value"),
        unit=q.get("unit"),
        dimension=QuantityDimension(q["dimension"]),
        start=start,
        end=end,
        source_text=text,
        detection=QuantityDetection.UNKNOWN,
        corrected=False,
    )


def build_link(case):
    raw = case.get("link")
    if raw is None:
        return None

    field = raw.get("field")

    return SemanticLink(
        quantity_id=raw["quantity_id"],
        field=None if field is None else ParamName(field),
        role=SemanticRole(raw["role"]),
        evidence=raw.get("evidence", ""),
        resolver=raw.get("resolver", "adversarial_benchmark"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--output",
        default="requirement_extractor_v2/evaluation/verifier_adversarial_metrics.json",
    )
    args = parser.parse_args()

    cases = load_jsonl(Path(args.dataset))
    verifier = DeterministicVerifier()

    details = []
    by_category = defaultdict(lambda: {"n": 0, "correct": 0, "false_acceptance": 0})

    correct = 0
    false_acceptance = 0
    expected_nonverified = 0
    status_confusion = defaultdict(Counter)

    for case in cases:
        quantity = build_quantity(case)
        link = build_link(case)

        decision = verifier.verify(
            quantity=quantity,
            link=link,
            source_text=case["source_text"],
        )

        expected = case["expected_status"]
        observed = decision.status.value
        reason_probe = case.get("expected_reason_contains")

        status_ok = observed == expected
        reason_ok = True

        if reason_probe:
            joined = " | ".join(decision.reasons).casefold()
            reason_ok = reason_probe.casefold() in joined

        case_ok = status_ok and reason_ok

        correct += int(case_ok)

        category = case["category"]
        by_category[category]["n"] += 1
        by_category[category]["correct"] += int(case_ok)

        if expected != "VERIFIED":
            expected_nonverified += 1

            if observed == "VERIFIED":
                false_acceptance += 1
                by_category[category]["false_acceptance"] += 1

        status_confusion[expected][observed] += 1

        details.append(
            {
                "id": case["id"],
                "category": category,
                "source_text": case["source_text"],
                "expected_status": expected,
                "observed_status": observed,
                "expected_reason_contains": reason_probe,
                "observed_reasons": decision.reasons,
                "correct": case_ok,
                "false_acceptance": (
                    expected != "VERIFIED"
                    and observed == "VERIFIED"
                ),
                "decision": decision.to_dict(),
                "notes": case.get("notes", ""),
            }
        )

    for category, stats in by_category.items():
        stats["accuracy"] = stats["correct"] / stats["n"] if stats["n"] else None
        stats["false_acceptance_rate"] = (
            stats["false_acceptance"] / stats["n"]
            if stats["n"]
            else None
        )

    metrics = {
        "n_cases": len(cases),
        "correct": correct,
        "accuracy": correct / len(cases) if cases else None,
        "expected_nonverified_cases": expected_nonverified,
        "false_acceptance_count": false_acceptance,
        "false_acceptance_rate_among_expected_nonverified": (
            false_acceptance / expected_nonverified
            if expected_nonverified
            else None
        ),
        "per_category": dict(by_category),
        "status_confusion": {
            gold: dict(pred)
            for gold, pred in status_confusion.items()
        },
    }

    output = {
        "metrics": metrics,
        "details": details,
    }

    Path(args.output).write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    failures = [x for x in details if not x["correct"]]

    if failures:
        print()
        print("=" * 80)
        print("FAILURES")
        print("=" * 80)

        for x in failures:
            print()
            print(x["id"], "|", x["category"])
            print("TEXT     :", x["source_text"])
            print("EXPECTED :", x["expected_status"])
            print("OBSERVED :", x["observed_status"])
            print("REASONS  :", x["observed_reasons"])
            print("FALSE ACCEPTANCE:", x["false_acceptance"])
    else:
        print()
        print("DETERMINISTIC VERIFIER ADVERSARIAL BENCHMARK: ALL CASES PASSED")


if __name__ == "__main__":
    main()