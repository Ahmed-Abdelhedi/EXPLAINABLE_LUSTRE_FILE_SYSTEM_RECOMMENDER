from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from requirement_extractor_v2.deterministic_verifier import (
    DeterministicVerifier,
)
from requirement_extractor_v2.models import (
    ParamName,
    Quantity,
    QuantityDetection,
    QuantityDimension,
    SemanticLink,
    SemanticRole,
)


def quantity(
    qid: str,
    text: str,
    raw: str,
    value,
    unit,
    dimension: QuantityDimension,
) -> Quantity:
    start = text.index(raw)

    return Quantity(
        id=qid,
        raw=raw,
        normalized=raw,
        value=value,
        unit=unit,
        dimension=dimension,
        start=start,
        end=start + len(raw),
        source_text=text,
        detection=QuantityDetection.UNKNOWN,
        corrected=False,
    )


def link(
    qid: str,
    field: ParamName,
    role: SemanticRole,
    evidence: str,
    resolver: str = "batch_test",
) -> SemanticLink:
    return SemanticLink(
        quantity_id=qid,
        field=field,
        role=role,
        evidence=evidence,
        resolver=resolver,
    )


def signature(decision):
    return {
        "quantity_id": decision.quantity_id,
        "status": decision.status.value,
        "field": (
            None
            if decision.field is None
            else decision.field.value
        ),
        "role": (
            None
            if decision.role is None
            else decision.role.value
        ),
        "reasons": list(decision.reasons),
    }


def statuses(decisions):
    return [
        decision.status.value
        for decision in decisions
    ]


def build_cases():
    cases = []

    # 1 — single valid count
    text = "The system will serve 200 clients."
    q1 = quantity(
        "q1", text, "200", 200, None,
        QuantityDimension.UNKNOWN,
    )
    l1 = link(
        "q1",
        ParamName.client_count,
        SemanticRole.TOTAL_COUNT,
        "200 clients",
    )
    cases.append({
        "id": "BATCH_001",
        "category": "single_valid",
        "quantities": [q1],
        "links": [l1],
        "source_text": text,
        "expected_statuses": ["VERIFIED"],
        "expected_reason_contains": [],
    })

    # 2 — single valid power
    text = "Maximum power is 12 kW."
    q1 = quantity(
        "q1", text, "12 kW", 12, "kW",
        QuantityDimension.POWER,
    )
    l1 = link(
        "q1",
        ParamName.max_power_w,
        SemanticRole.MAXIMUM_LIMIT,
        "Maximum power is 12 kW",
    )
    cases.append({
        "id": "BATCH_002",
        "category": "single_valid",
        "quantities": [q1],
        "links": [l1],
        "source_text": text,
        "expected_statuses": ["VERIFIED"],
        "expected_reason_contains": [],
    })

    # 3 — missing link
    text = "The requirement is around 320."
    q1 = quantity(
        "q1", text, "320", 320, None,
        QuantityDimension.UNKNOWN,
    )
    cases.append({
        "id": "BATCH_003",
        "category": "missing_link",
        "quantities": [q1],
        "links": [],
        "source_text": text,
        "expected_statuses": ["UNRESOLVED"],
        "expected_reason_contains": ["no_semantic_link"],
    })

    # 4 — duplicate identical links for one quantity
    text = "There will be 250 clients."
    q1 = quantity(
        "q1", text, "250", 250, None,
        QuantityDimension.UNKNOWN,
    )
    l1 = link(
        "q1",
        ParamName.client_count,
        SemanticRole.TOTAL_COUNT,
        "250 clients",
    )
    l2 = link(
        "q1",
        ParamName.client_count,
        SemanticRole.TOTAL_COUNT,
        "250 clients",
        resolver="duplicate_test",
    )
    cases.append({
        "id": "BATCH_004",
        "category": "duplicate_links",
        "quantities": [q1],
        "links": [l1, l2],
        "source_text": text,
        "expected_statuses": ["AMBIGUOUS"],
        "expected_reason_contains": [
            "multiple_semantic_links_for_same_quantity"
        ],
    })

    # 5 — duplicate conflicting links for one quantity
    text = "The namespace contains 9000000 files."
    q1 = quantity(
        "q1", text, "9000000", 9000000, None,
        QuantityDimension.UNKNOWN,
    )
    l1 = link(
        "q1",
        ParamName.total_file_count,
        SemanticRole.TOTAL_COUNT,
        "9000000 files",
    )
    l2 = link(
        "q1",
        ParamName.client_count,
        SemanticRole.TOTAL_COUNT,
        "9000000 files",
        resolver="conflicting_test",
    )
    cases.append({
        "id": "BATCH_005",
        "category": "duplicate_links",
        "quantities": [q1],
        "links": [l1, l2],
        "source_text": text,
        "expected_statuses": ["AMBIGUOUS"],
        "expected_reason_contains": [
            "multiple_semantic_links_for_same_quantity"
        ],
    })

    # 6 — orphan link
    orphan = link(
        "q404",
        ParamName.client_count,
        SemanticRole.TOTAL_COUNT,
        "200 clients",
    )
    cases.append({
        "id": "BATCH_006",
        "category": "orphan_link",
        "quantities": [],
        "links": [orphan],
        "source_text": "200 clients",
        "expected_statuses": ["INVALID"],
        "expected_reason_contains": [
            "semantic_link_references_unknown_quantity"
        ],
    })

    # 7 — valid + orphan
    text = "There will be 180 clients."
    q1 = quantity(
        "q1", text, "180", 180, None,
        QuantityDimension.UNKNOWN,
    )
    l1 = link(
        "q1",
        ParamName.client_count,
        SemanticRole.TOTAL_COUNT,
        "180 clients",
    )
    orphan = link(
        "ghost",
        ParamName.total_file_count,
        SemanticRole.TOTAL_COUNT,
        "999",
    )
    cases.append({
        "id": "BATCH_007",
        "category": "mixed_orphan",
        "quantities": [q1],
        "links": [l1, orphan],
        "source_text": text,
        "expected_statuses": ["VERIFIED", "INVALID"],
        "expected_reason_contains": [
            "semantic_link_references_unknown_quantity"
        ],
    })

    # 8 — two quantities, one unresolved
    text = "200 clients and around 500."
    q1 = quantity(
        "q1", text, "200", 200, None,
        QuantityDimension.UNKNOWN,
    )
    q2 = quantity(
        "q2", text, "500", 500, None,
        QuantityDimension.UNKNOWN,
    )
    l1 = link(
        "q1",
        ParamName.client_count,
        SemanticRole.TOTAL_COUNT,
        "200 clients",
    )
    cases.append({
        "id": "BATCH_008",
        "category": "mixed_missing",
        "quantities": [q1, q2],
        "links": [l1],
        "source_text": text,
        "expected_statuses": ["VERIFIED", "UNRESOLVED"],
        "expected_reason_contains": ["no_semantic_link"],
    })

    # 9 — two valid quantities
    text = "Maximum power is 10 kW and maximum budget is 90000 USD."
    q1 = quantity(
        "q1", text, "10 kW", 10, "kW",
        QuantityDimension.POWER,
    )
    q2 = quantity(
        "q2", text, "90000 USD", 90000, "USD",
        QuantityDimension.MONEY,
    )
    l1 = link(
        "q1",
        ParamName.max_power_w,
        SemanticRole.MAXIMUM_LIMIT,
        "Maximum power is 10 kW",
    )
    l2 = link(
        "q2",
        ParamName.max_budget_usd,
        SemanticRole.MAXIMUM_LIMIT,
        "maximum budget is 90000 USD",
    )
    cases.append({
        "id": "BATCH_009",
        "category": "multi_valid",
        "quantities": [q1, q2],
        "links": [l1, l2],
        "source_text": text,
        "expected_statuses": ["VERIFIED", "VERIFIED"],
        "expected_reason_contains": [],
    })

    # 10 — duplicate q1 + valid q2
    text = "200 clients and maximum power 8 kW."
    q1 = quantity(
        "q1", text, "200", 200, None,
        QuantityDimension.UNKNOWN,
    )
    q2 = quantity(
        "q2", text, "8 kW", 8, "kW",
        QuantityDimension.POWER,
    )
    l1 = link(
        "q1",
        ParamName.client_count,
        SemanticRole.TOTAL_COUNT,
        "200 clients",
    )
    l1b = link(
        "q1",
        ParamName.client_count,
        SemanticRole.TOTAL_COUNT,
        "200 clients",
        resolver="second",
    )
    l2 = link(
        "q2",
        ParamName.max_power_w,
        SemanticRole.MAXIMUM_LIMIT,
        "maximum power 8 kW",
    )
    cases.append({
        "id": "BATCH_010",
        "category": "duplicate_plus_valid",
        "quantities": [q1, q2],
        "links": [l1, l1b, l2],
        "source_text": text,
        "expected_statuses": ["AMBIGUOUS", "VERIFIED"],
        "expected_reason_contains": [
            "multiple_semantic_links_for_same_quantity"
        ],
    })

    # 11 — two orphan links
    o1 = link(
        "ghost1",
        ParamName.client_count,
        SemanticRole.TOTAL_COUNT,
        "100 clients",
    )
    o2 = link(
        "ghost2",
        ParamName.total_file_count,
        SemanticRole.TOTAL_COUNT,
        "1000000 files",
    )
    cases.append({
        "id": "BATCH_011",
        "category": "orphan_link",
        "quantities": [],
        "links": [o1, o2],
        "source_text": "",
        "expected_statuses": ["INVALID", "INVALID"],
        "expected_reason_contains": [
            "semantic_link_references_unknown_quantity"
        ],
    })

    # 12 — unresolved known quantity + orphan
    text = "The requirement is around 450."
    q1 = quantity(
        "q1", text, "450", 450, None,
        QuantityDimension.UNKNOWN,
    )
    orphan = link(
        "ghost",
        ParamName.client_count,
        SemanticRole.TOTAL_COUNT,
        "123",
    )
    cases.append({
        "id": "BATCH_012",
        "category": "missing_plus_orphan",
        "quantities": [q1],
        "links": [orphan],
        "source_text": text,
        "expected_statuses": ["UNRESOLVED", "INVALID"],
        "expected_reason_contains": [
            "no_semantic_link",
            "semantic_link_references_unknown_quantity",
        ],
    })

    # 13 — invalid role
    text = "There will be 220 clients."
    q1 = quantity(
        "q1", text, "220", 220, None,
        QuantityDimension.UNKNOWN,
    )
    l1 = link(
        "q1",
        ParamName.client_count,
        SemanticRole.MAXIMUM_LIMIT,
        "220 clients",
    )
    cases.append({
        "id": "BATCH_013",
        "category": "invalid_role",
        "quantities": [q1],
        "links": [l1],
        "source_text": text,
        "expected_statuses": ["INVALID"],
        "expected_reason_contains": ["invalid_field_role_pair"],
    })

    # 14 — field/dimension mismatch
    text = "Maximum power is 12 kW."
    q1 = quantity(
        "q1", text, "12 kW", 12, "kW",
        QuantityDimension.POWER,
    )
    l1 = link(
        "q1",
        ParamName.max_budget_usd,
        SemanticRole.MAXIMUM_LIMIT,
        "12 kW",
    )
    cases.append({
        "id": "BATCH_014",
        "category": "dimension_mismatch",
        "quantities": [q1],
        "links": [l1],
        "source_text": text,
        "expected_statuses": ["INVALID"],
        "expected_reason_contains": [
            "field_incompatible_with_dimension"
        ],
    })

    # 15 — unsupported evidence
    text = "There will be 205 clients."
    q1 = quantity(
        "q1", text, "205", 205, None,
        QuantityDimension.UNKNOWN,
    )
    l1 = link(
        "q1",
        ParamName.client_count,
        SemanticRole.TOTAL_COUNT,
        "205 machines",
    )
    cases.append({
        "id": "BATCH_015",
        "category": "bad_evidence",
        "quantities": [q1],
        "links": [l1],
        "source_text": text,
        "expected_statuses": ["AMBIGUOUS"],
        "expected_reason_contains": [
            "evidence_not_supported_by_source_text"
        ],
    })

    # 16 — non-integer client count
    text = "The system has 200.5 clients."
    q1 = quantity(
        "q1", text, "200.5", 200.5, None,
        QuantityDimension.UNKNOWN,
    )
    l1 = link(
        "q1",
        ParamName.client_count,
        SemanticRole.TOTAL_COUNT,
        "200.5 clients",
    )
    cases.append({
        "id": "BATCH_016",
        "category": "invalid_value",
        "quantities": [q1],
        "links": [l1],
        "source_text": text,
        "expected_statuses": ["INVALID"],
        "expected_reason_contains": [
            "count_value_must_be_integer"
        ],
    })

    # 17 — two ratio components valid independently
    text = "Workload is 70% read and 30% write."
    q1 = quantity(
        "q1", text, "70%", 70, "%",
        QuantityDimension.PERCENT,
    )
    q2 = quantity(
        "q2", text, "30%", 30, "%",
        QuantityDimension.PERCENT,
    )
    l1 = link(
        "q1",
        ParamName.read_write_ratio,
        SemanticRole.RATIO_COMPONENT,
        "70% read",
    )
    l2 = link(
        "q2",
        ParamName.read_write_ratio,
        SemanticRole.RATIO_COMPONENT,
        "30% write",
    )
    cases.append({
        "id": "BATCH_017",
        "category": "multi_valid_ratio",
        "quantities": [q1, q2],
        "links": [l1, l2],
        "source_text": text,
        "expected_statuses": ["VERIFIED", "VERIFIED"],
        "expected_reason_contains": [],
    })

    # 18 — duplicate ratio links for same quantity
    text = "Workload is 70% read and 30% write."
    q1 = quantity(
        "q1", text, "70%", 70, "%",
        QuantityDimension.PERCENT,
    )
    l1 = link(
        "q1",
        ParamName.read_write_ratio,
        SemanticRole.RATIO_COMPONENT,
        "70% read",
    )
    l2 = link(
        "q1",
        ParamName.annual_growth_percent,
        SemanticRole.GROWTH_RATE,
        "70% read",
        resolver="conflicting",
    )
    cases.append({
        "id": "BATCH_018",
        "category": "duplicate_links",
        "quantities": [q1],
        "links": [l1, l2],
        "source_text": text,
        "expected_statuses": ["AMBIGUOUS"],
        "expected_reason_contains": [
            "multiple_semantic_links_for_same_quantity"
        ],
    })

    # 19 — three links for one quantity
    text = "There will be 300 clients."
    q1 = quantity(
        "q1", text, "300", 300, None,
        QuantityDimension.UNKNOWN,
    )
    links = [
        link(
            "q1",
            ParamName.client_count,
            SemanticRole.TOTAL_COUNT,
            "300 clients",
            resolver=f"r{i}",
        )
        for i in range(3)
    ]
    cases.append({
        "id": "BATCH_019",
        "category": "duplicate_links",
        "quantities": [q1],
        "links": links,
        "source_text": text,
        "expected_statuses": ["AMBIGUOUS"],
        "expected_reason_contains": [
            "multiple_semantic_links_for_same_quantity"
        ],
    })

    # 20 — one valid, one invalid role, one unresolved
    text = "180 clients, maximum power 9 kW, and around 777."
    q1 = quantity(
        "q1", text, "180", 180, None,
        QuantityDimension.UNKNOWN,
    )
    q2 = quantity(
        "q2", text, "9 kW", 9, "kW",
        QuantityDimension.POWER,
    )
    q3 = quantity(
        "q3", text, "777", 777, None,
        QuantityDimension.UNKNOWN,
    )
    l1 = link(
        "q1",
        ParamName.client_count,
        SemanticRole.TOTAL_COUNT,
        "180 clients",
    )
    l2 = link(
        "q2",
        ParamName.max_power_w,
        SemanticRole.TOTAL_COUNT,
        "maximum power 9 kW",
    )
    cases.append({
        "id": "BATCH_020",
        "category": "mixed_statuses",
        "quantities": [q1, q2, q3],
        "links": [l1, l2],
        "source_text": text,
        "expected_statuses": [
            "VERIFIED",
            "INVALID",
            "UNRESOLVED",
        ],
        "expected_reason_contains": [
            "invalid_field_role_pair",
            "no_semantic_link",
        ],
    })

    return cases


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        default=(
            "requirement_extractor_v2/evaluation/"
            "deterministic_verifier_verify_many_metrics.json"
        ),
    )

    args = parser.parse_args()

    verifier = DeterministicVerifier()
    cases = build_cases()

    correct = 0
    false_acceptance_count = 0
    failures = []
    details = []
    status_counter = Counter()

    for case in cases:
        decisions = verifier.verify_many(
            quantities=case["quantities"],
            links=case["links"],
            source_text=case["source_text"],
        )

        actual_statuses = statuses(
            decisions
        )

        status_counter.update(
            actual_statuses
        )

        expected = case[
            "expected_statuses"
        ]

        status_match = (
            actual_statuses
            == expected
        )

        all_reasons = [
            reason
            for decision in decisions
            for reason in decision.reasons
        ]

        reason_match = all(
            any(
                expected_reason
                in actual_reason
                for actual_reason in all_reasons
            )
            for expected_reason
            in case[
                "expected_reason_contains"
            ]
        )

        passed = (
            status_match
            and reason_match
        )

        correct += int(
            passed
        )

        expected_has_no_verified = (
            "VERIFIED"
            not in expected
        )

        actual_has_verified = (
            "VERIFIED"
            in actual_statuses
        )

        if (
            expected_has_no_verified
            and actual_has_verified
        ):
            false_acceptance_count += 1

        detail = {
            "id": case["id"],
            "category": case["category"],
            "expected_statuses": expected,
            "actual_statuses": actual_statuses,
            "expected_reason_contains":
                case[
                    "expected_reason_contains"
                ],
            "decisions": [
                signature(decision)
                for decision in decisions
            ],
            "passed": passed,
        }

        details.append(
            detail
        )

        label = (
            "PASS"
            if passed
            else "FAIL"
        )

        print(
            f"[{label}] "
            f"{case['id']} "
            f"{case['category']}: "
            f"{actual_statuses}"
        )

        if not passed:
            failures.append(
                detail
            )

    metrics = {
        "n_cases": len(cases),
        "correct_cases": correct,
        "accuracy": (
            correct / len(cases)
        ),
        "false_acceptance_count":
            false_acceptance_count,
        "status_distribution":
            dict(status_counter),
        "failures": len(failures),
    }

    output = {
        "metrics": metrics,
        "details": details,
        "benchmark_note": (
            "Adversarial batch benchmark for DeterministicVerifier.verify_many. "
            "It specifically tests duplicate semantic links, orphan links, "
            "mixed verified/unresolved/invalid batches, evidence failures and "
            "conflicting batch inputs."
        ),
    }

    Path(args.output).write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print(
        "DETERMINISTIC VERIFIER VERIFY_MANY METRICS"
    )
    print("=" * 80)
    print(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        )
    )

    if failures:
        print()
        print("=" * 80)
        print("FAILURES")
        print("=" * 80)

        for failure in failures:
            print(
                json.dumps(
                    failure,
                    indent=2,
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    main()