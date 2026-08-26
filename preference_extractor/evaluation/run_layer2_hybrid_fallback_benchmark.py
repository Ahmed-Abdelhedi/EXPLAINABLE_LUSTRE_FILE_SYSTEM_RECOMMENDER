from __future__ import annotations

import argparse
import json
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .layer2_fallback_common import (
    latency_summary,
    load_jsonl,
    parse_llm_response,
    prediction_matches_gold,
)
from .layer2_hybrid_guard import (
    GUARD_VERSION,
    Layer2DeterministicSemanticGuard,
)
from .layer2_residual_validator import (
    RESIDUAL_VALIDATOR_VERSION,
    validate_residual_prediction,
)


HYBRID_PROMPT_VERSION = (
    "layer2_hybrid_residual_qwen_v1_20260825"
)


HYBRID_PROMPT = r"""
You are the residual fallback for Layer 2 of a multilingual preference
extractor.

A deterministic semantic guard has already resolved clear absolute intensity,
clear pure comparisons, and clear hard-negative/no-preference cases.

Resolve ONLY the dimensions listed in REQUESTED_DIMENSIONS.

Allowed dimensions:
cost, power, performance, reliability

Allowed statuses:
RESOLVED, NO_SIGNAL, RELATIVE_ONLY, UNRESOLVED

Allowed RESOLVED levels:
VERY_LOW, LOW, MEDIUM, HIGH, VERY_HIGH

Rules:
1. NO_SIGNAL is not VERY_LOW.
2. Pure comparison/order without an independent absolute cue is RELATIVE_ONLY.
3. Do not infer priority intensity from numeric limits, throughput values,
   system capabilities, API/log fields, or technical adjectives.
4. Current/final/latest user choice overrides superseded/history.
5. Vendor/third-party opinion is not the user's preference unless adopted.
6. If uncertain, return UNRESOLVED.
7. Evidence for RESOLVED or RELATIVE_ONLY must be an exact substring copied
   from CURRENT_USER_MESSAGE.
8. Return every requested dimension exactly once and no others.
9. JSON only.

Output:
{
  "dimensions": {
    "<requested dimension>": {
      "status": "RESOLVED|NO_SIGNAL|RELATIVE_ONLY|UNRESOLVED",
      "level": "VERY_LOW|LOW|MEDIUM|HIGH|VERY_HIGH|null",
      "evidence": "exact substring|null"
    }
  }
}
""".strip()


def _prompt(
    *,
    text: str,
    dimensions: Sequence[
        str
    ],
) -> str:
    return (
        HYBRID_PROMPT
        + "\n\nREQUESTED_DIMENSIONS\n"
        + json.dumps(
            list(
                dimensions
            ),
            ensure_ascii=False,
        )
        + "\n\nCURRENT_USER_MESSAGE\n"
        + text
    )


def _call_ollama(
    *,
    host: str,
    model: str,
    prompt: str,
    timeout_seconds: int,
) -> Tuple[
    str,
    float,
]:
    payload = json.dumps(
        {
            "model":
                model,
            "prompt":
                prompt,
            "stream":
                False,
            "format":
                "json",
            "keep_alive":
                "30m",
            "options": {
                "temperature":
                    0.0,
                "num_predict":
                    320,
            },
        }
    ).encode(
        "utf-8"
    )

    request = urllib.request.Request(
        host.rstrip(
            "/"
        )
        + "/api/generate",
        data=payload,
        headers={
            "Content-Type":
                "application/json",
        },
        method="POST",
    )

    started = (
        time.perf_counter()
    )

    with urllib.request.urlopen(
        request,
        timeout=
            timeout_seconds,
    ) as response:
        outer = json.loads(
            response
            .read()
            .decode(
                "utf-8"
            )
        )

    latency = (
        time.perf_counter()
        - started
    )

    return (
        str(
            outer.get(
                "response",
                "",
            )
        ),
        latency,
    )


def _gold_by_dimension(
    row: Mapping[
        str,
        Any
    ],
) -> Dict[
    str,
    Dict[
        str,
        Any
    ],
]:
    return {
        item[
            "dimension"
        ]:
            item[
                "gold_expected"
            ]
        for item
        in row[
            "fallback_dimensions"
        ]
    }


def _metrics(
    records: Sequence[
        Mapping[
            str,
            Any
        ]
    ],
) -> Dict[
    str,
    Any
]:
    cases = [
        item
        for record in records
        for item
        in record[
            "dimension_results"
        ]
    ]

    total = len(
        cases
    )

    exact = sum(
        bool(
            item[
                "exact_correct"
            ]
        )
        for item
        in cases
    )

    accepted = [
        item
        for item
        in cases
        if item[
            "prediction"
        ].get(
            "accepted",
            False,
        )
    ]

    accepted_correct = sum(
        bool(
            item[
                "exact_correct"
            ]
        )
        for item
        in accepted
    )

    wrong_accepted = (
        len(
            accepted
        )
        - accepted_correct
    )

    guard_cases = [
        item
        for item
        in cases
        if item[
            "source"
        ]
        == "DETERMINISTIC_GUARD"
    ]

    guard_correct = sum(
        bool(
            item[
                "exact_correct"
            ]
        )
        for item
        in guard_cases
    )

    residual_cases = [
        item
        for item
        in cases
        if item[
            "source"
        ]
        == "LLM_RESIDUAL"
    ]

    residual_correct = sum(
        bool(
            item[
                "exact_correct"
            ]
        )
        for item
        in residual_cases
    )

    residual_accepted = [
        item
        for item
        in residual_cases
        if item["prediction"].get("accepted", False)
    ]

    residual_accepted_correct = sum(
        bool(item["exact_correct"])
        for item in residual_accepted
    )

    validator_corrections = sum(
        str(item.get("residual_validation", {}).get("action", ""))
        == "CANONICALIZED_RESOLVED_LEVEL"
        for item in residual_cases
    )

    validator_rejections = sum(
        str(item.get("residual_validation", {}).get("action", ""))
        .startswith("REJECTED:")
        for item in residual_cases
    )

    global_contract_violation_rows = sum(
        bool(record.get("llm_called"))
        and record.get("llm_response_valid") is False
        for record in records
    )

    status_metrics = {}

    for status in (
        "RESOLVED",
        "NO_SIGNAL",
        "RELATIVE_ONLY",
    ):
        subset = [
            item
            for item
            in cases
            if item[
                "gold"
            ][
                "status"
            ]
            == status
        ]

        status_metrics[
            status
        ] = {
            "n":
                len(
                    subset
                ),
            "exact_accuracy":
                (
                    sum(
                        bool(
                            item[
                                "exact_correct"
                            ]
                        )
                        for item
                        in subset
                    )
                    / len(
                        subset
                    )
                    if subset
                    else None
                ),
        }

    latencies = [
        float(
            record[
                "llm_latency_s"
            ]
        )
        for record
        in records
        if record.get(
            "llm_latency_s"
        )
        is not None
    ]

    llm_calls = sum(
        bool(
            record[
                "llm_called"
            ]
        )
        for record
        in records
    )

    return {
        "rows":
            len(
                records
            ),
        "dimension_cases":
            total,
        "exact_correct":
            exact,
        "exact_accuracy":
            (
                exact
                / total
                if total
                else None
            ),
        "accepted_outputs":
            len(
                accepted
            ),
        "accepted_precision":
            (
                accepted_correct
                / len(
                    accepted
                )
                if accepted
                else None
            ),
        "wrong_accepted_count":
            wrong_accepted,
        "false_acceptance_rate_among_all_dimensions":
            (
                wrong_accepted
                / total
                if total
                else None
            ),
        "abstention_rate":
            (
                (
                    total
                    - len(
                        accepted
                    )
                )
                / total
                if total
                else None
            ),
        "guard_dimension_cases":
            len(
                guard_cases
            ),
        "guard_coverage":
            (
                len(
                    guard_cases
                )
                / total
                if total
                else None
            ),
        "guard_precision_on_covered_evaluation_only":
            (
                guard_correct
                / len(
                    guard_cases
                )
                if guard_cases
                else None
            ),
        "residual_llm_dimension_cases":
            len(
                residual_cases
            ),
        "residual_llm_exact_accuracy":
            (
                residual_correct
                / len(
                    residual_cases
                )
                if residual_cases
                else None
            ),
        "residual_accepted_outputs":
            len(residual_accepted),
        "residual_accepted_precision":
            (
                residual_accepted_correct
                / len(residual_accepted)
                if residual_accepted
                else None
            ),
        "residual_validator_corrections":
            validator_corrections,
        "residual_validator_rejections":
            validator_rejections,
        "llm_global_contract_violation_rows":
            global_contract_violation_rows,
        "llm_calls":
            llm_calls,
        "llm_row_call_rate":
            (
                llm_calls
                / len(
                    records
                )
                if records
                else None
            ),
        "llm_calls_saved_vs_one_call_per_input_row":
            (
                1.0
                - (
                    llm_calls
                    / len(
                        records
                    )
                )
                if records
                else None
            ),
        "gold_status_metrics":
            status_metrics,
        "llm_latency":
            latency_summary(
                latencies
            ),
    }


def run(
    *,
    dataset: Path,
    output: Path,
    model: str,
    host: str,
    timeout_seconds: int,
    guard_only: bool,
) -> Dict[
    str,
    Any
]:
    rows = load_jsonl(
        dataset
    )

    guard = (
        Layer2DeterministicSemanticGuard()
    )

    records: List[
        Dict[
            str,
            Any
        ]
    ] = []

    for index, row in enumerate(
        rows,
        start=1,
    ):
        text = row[
            "text"
        ]

        requested = [
            item[
                "dimension"
            ]
            for item
            in row[
                "fallback_dimensions"
            ]
        ]

        gold = _gold_by_dimension(
            row
        )

        guarded = (
            guard.resolve_many(
                text=text,
                dimensions=requested,
            )
        )

        unresolved = [
            dimension
            for dimension
            in requested
            if dimension
            not in guarded
        ]

        llm_predictions = {}

        raw_response = None

        llm_latency = None

        llm_error = None

        llm_response_valid = None
        llm_response_violations = []

        if (
            unresolved
            and not guard_only
        ):
            try:
                (
                    raw_response,
                    llm_latency,
                ) = _call_ollama(
                    host=host,
                    model=model,
                    prompt=_prompt(
                        text=text,
                        dimensions=
                            unresolved,
                    ),
                    timeout_seconds=
                        timeout_seconds,
                )

                parsed = (
                    parse_llm_response(
                        raw_text=
                            raw_response,
                        requested_dimensions=
                            unresolved,
                        user_text=text,
                    )
                )

                llm_predictions = (
                    parsed[
                        "dimensions"
                    ]
                )

                llm_response_valid = bool(
                    parsed.get("valid", False)
                )
                llm_response_violations = list(
                    parsed.get("violations", [])
                )

            except Exception as exc:
                llm_error = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

        dimension_results = []

        for dimension in requested:
            if dimension in guarded:
                decision = guarded[
                    dimension
                ]

                prediction = {
                    "status":
                        decision.status,
                    "level":
                        decision.level,
                    "evidence":
                        decision.evidence,
                    "accepted":
                        True,
                    "validation_error":
                        None,
                }

                source = (
                    "DETERMINISTIC_GUARD"
                )

                reason = (
                    decision.reason
                )

            elif dimension in llm_predictions:
                raw_llm_prediction = dict(
                    llm_predictions[dimension]
                )

                residual_validation = (
                    validate_residual_prediction(
                        text=text,
                        dimension=dimension,
                        prediction=raw_llm_prediction,
                    )
                )

                prediction = (
                    residual_validation.prediction
                )

                source = (
                    "LLM_RESIDUAL"
                )

                reason = (
                    "DETERMINISTIC_GUARD_ABSTAINED"
                )

            else:
                prediction = {
                    "status":
                        "UNRESOLVED",
                    "level":
                        None,
                    "evidence":
                        None,
                    "accepted":
                        False,
                    "validation_error":
                        (
                            "GUARD_ONLY"
                            if guard_only
                            else (
                                llm_error
                                or "NO_VALID_LLM_RESULT"
                            )
                        ),
                }

                source = (
                    "LLM_RESIDUAL"
                )

                reason = (
                    "DETERMINISTIC_GUARD_ABSTAINED"
                )

            gold_item = gold[
                dimension
            ]

            result_item = {
                "dimension": dimension,
                "source": source,
                "reason": reason,
                "gold": gold_item,
                "prediction": prediction,
                "exact_correct": prediction_matches_gold(
                    prediction,
                    gold_item,
                ),
            }

            if source == "LLM_RESIDUAL":
                if dimension in llm_predictions:
                    result_item["raw_llm_prediction"] = (
                        raw_llm_prediction
                    )
                    result_item["residual_validation"] = (
                        residual_validation.to_dict()
                    )
                else:
                    result_item["raw_llm_prediction"] = None
                    result_item["residual_validation"] = {
                        "action": "NO_LLM_PREDICTION",
                        "raw_status": None,
                        "raw_level": None,
                        "canonical_level": None,
                    }

            dimension_results.append(result_item)

        record = {
            "sample_id":
                row[
                    "sample_id"
                ],
            "language":
                row.get(
                    "language"
                ),
            "semantic_family":
                row.get(
                    "semantic_family"
                ),
            "requested_dimensions":
                requested,
            "guarded_dimensions":
                sorted(
                    guarded
                ),
            "residual_dimensions":
                unresolved,
            "llm_called":
                bool(
                    unresolved
                    and not guard_only
                ),
            "llm_latency_s":
                llm_latency,
            "llm_error":
                llm_error,
            "llm_response_valid":
                llm_response_valid,
            "llm_response_violations":
                llm_response_violations,
            "raw_response":
                raw_response,
            "dimension_results":
                dimension_results,
        }

        records.append(
            record
        )

        correct = sum(
            bool(
                item[
                    "exact_correct"
                ]
            )
            for item
            in dimension_results
        )

        print(
            f"[{index}/{len(rows)}] "
            f"{row['sample_id']} "
            f"{correct}/"
            f"{len(dimension_results)} "
            f"guard={len(guarded)} "
            f"residual={len(unresolved)} "
            f"llm={'YES' if record['llm_called'] else 'NO'}"
        )

    report = {
        "step":
            "4.2E",
        "benchmark":
            (
                "layer2_hybrid_guard_plus_residual_llm"
            ),
        "guard_version":
            GUARD_VERSION,
        "residual_validator_version":
            RESIDUAL_VALIDATOR_VERSION,
        "hybrid_prompt_version":
            HYBRID_PROMPT_VERSION,
        "model":
            model,
        "guard_only":
            guard_only,
        "protocol_note":
            (
                "The deterministic guard receives only current text and "
                "requested fallback dimensions. Gold labels are used only "
                "after resolution for scoring. The LLM is called only for "
                "dimensions where the guard abstains. Residual LLM outputs are then "
                "subject to a deterministic evidence validator before acceptance."
            ),
        "overall":
            _metrics(
                records
            ),
        "records":
            records,
    }

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return report


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--model",
        default="qwen2.5-coder:7b",
    )

    parser.add_argument(
        "--host",
        default="http://localhost:11434",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--guard-only",
        action="store_true",
    )

    args = parser.parse_args()

    report = run(
        dataset=
            args.dataset,
        output=
            args.output,
        model=
            args.model,
        host=
            args.host,
        timeout_seconds=
            args.timeout,
        guard_only=
            args.guard_only,
    )

    print(
        "\n"
        + "="
        * 80
    )

    print(
        "LAYER 2 HYBRID BENCHMARK"
    )

    print(
        "="
        * 80
    )

    print(
        json.dumps(
            report[
                "overall"
            ],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
