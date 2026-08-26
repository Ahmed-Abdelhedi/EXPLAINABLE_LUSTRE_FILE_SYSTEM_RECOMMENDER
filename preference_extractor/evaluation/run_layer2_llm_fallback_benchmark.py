from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .layer2_fallback_common import (
    PROMPT_VERSION,
    build_prompt,
    gold_expected_for_dimension,
    latency_summary,
    load_jsonl,
    parse_llm_response,
    prediction_matches_gold,
    prompt_policy_sha256,
)


def call_ollama(
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

    start = time.perf_counter()

    with urllib.request.urlopen(
        request,
        timeout=
            timeout_seconds,
    ) as response:
        outer = json.loads(
            response.read().decode(
                "utf-8"
            )
        )

    latency = (
        time.perf_counter()
        - start
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


def expected_map(
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
    output = {}

    for item in row[
        "fallback_dimensions"
    ]:
        output[
            item[
                "dimension"
            ]
        ] = item[
            "gold_expected"
        ]

    return output


def transformer_baseline_map(
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
    output = {}

    for item in row[
        "fallback_dimensions"
    ]:
        probability = float(
            item[
                "presence_probability"
            ]
        )

        if probability < 0.5:
            output[
                item[
                    "dimension"
                ]
            ] = {
                "status":
                    "NO_SIGNAL",
                "level":
                    None,
            }

        else:
            output[
                item[
                    "dimension"
                ]
            ] = {
                "status":
                    "RESOLVED",
                "level":
                    item[
                        "transformer_level"
                    ],
            }

    return output


def validate_prompt_freeze(
    path: Optional[
        Path
    ],
    model: str,
) -> None:
    if path is None:
        return

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    expected_hash = (
        prompt_policy_sha256()
    )

    if (
        data.get(
            "prompt_policy_sha256"
        )
        != expected_hash
    ):
        raise RuntimeError(
            "Prompt freeze hash does not match current benchmark code."
        )

    if (
        data.get(
            "model"
        )
        != model
    ):
        raise RuntimeError(
            "Prompt freeze model does not match requested Ollama model."
        )


def metric_block(
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
    dimension_cases = []

    for record in records:
        for item in record[
            "dimension_results"
        ]:
            dimension_cases.append(
                item
            )

    total = len(
        dimension_cases
    )

    exact_correct = sum(
        bool(
            item[
                "exact_correct"
            ]
        )
        for item
        in dimension_cases
    )

    accepted = [
        item
        for item
        in dimension_cases
        if item[
            "prediction"
        ][
            "accepted"
        ]
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

    unsupported = [
        item
        for item
        in dimension_cases
        if item[
            "prediction"
        ].get(
            "validation_error"
        )
    ]

    abstained = [
        item
        for item
        in dimension_cases
        if not item[
            "prediction"
        ][
            "accepted"
        ]
    ]

    gold_resolved = [
        item
        for item
        in dimension_cases
        if item[
            "gold"
        ][
            "status"
        ]
        == "RESOLVED"
    ]

    gold_no_signal = [
        item
        for item
        in dimension_cases
        if item[
            "gold"
        ][
            "status"
        ]
        == "NO_SIGNAL"
    ]

    gold_relative = [
        item
        for item
        in dimension_cases
        if item[
            "gold"
        ][
            "status"
        ]
        == "RELATIVE_ONLY"
    ]

    def exact_rate(
        items,
    ):
        if not items:
            return None

        return (
            sum(
                bool(
                    item[
                        "exact_correct"
                    ]
                )
                for item
                in items
            )
            / len(
                items
            )
        )

    baseline_correct = sum(
        bool(
            item[
                "transformer_baseline_correct"
            ]
        )
        for item
        in dimension_cases
    )

    latencies = [
        float(
            record[
                "latency_s"
            ]
        )
        for record
        in records
        if record.get(
            "latency_s"
        )
        is not None
    ]

    return {
        "rows":
            len(
                records
            ),
        "dimension_cases":
            total,
        "exact_correct":
            exact_correct,
        "exact_accuracy":
            (
                exact_correct
                / total
                if total
                else None
            ),
        "accepted_dimension_outputs":
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
        "false_acceptance_rate_among_all_fallback_dimensions":
            (
                wrong_accepted
                / total
                if total
                else None
            ),
        "abstained_or_rejected_count":
            len(
                abstained
            ),
        "abstention_rate":
            (
                len(
                    abstained
                )
                / total
                if total
                else None
            ),
        "unsupported_or_contract_rejected_count":
            len(
                unsupported
            ),
        "unsupported_extraction_rate":
            (
                len(
                    unsupported
                )
                / total
                if total
                else None
            ),
        "gold_resolved_exact_accuracy":
            exact_rate(
                gold_resolved
            ),
        "gold_no_signal_accuracy":
            exact_rate(
                gold_no_signal
            ),
        "gold_relative_only_accuracy":
            exact_rate(
                gold_relative
            ),
        "transformer_raw_baseline_exact_accuracy_on_same_fallback_dimensions":
            (
                baseline_correct
                / total
                if total
                else None
            ),
        "llm_absolute_gain_over_transformer_raw_baseline":
            (
                (
                    exact_correct
                    - baseline_correct
                )
                / total
                if total
                else None
            ),
        "latency":
            latency_summary(
                latencies
            ),
        "llm_calls":
            len(
                records
            ),
        "average_llm_calls_per_row":
            (
                1.0
                if records
                else 0.0
            ),
    }


def aggregate_by(
    records: Sequence[
        Mapping[
            str,
            Any
        ]
    ],
    key_name: str,
) -> Dict[
    str,
    Any
]:
    groups = defaultdict(
        list
    )

    for record in records:
        groups[
            str(
                record.get(
                    key_name,
                    "unknown",
                )
            )
        ].append(
            record
        )

    return {
        key:
            metric_block(
                values
            )
        for key, values
        in sorted(
            groups.items()
        )
    }


def per_dimension_metrics(
    records,
):
    groups = defaultdict(
        list
    )

    for record in records:
        for item in record[
            "dimension_results"
        ]:
            groups[
                item[
                    "dimension"
                ]
            ].append(
                item
            )

    output = {}

    for dimension, items in sorted(
        groups.items()
    ):
        total = len(
            items
        )

        correct = sum(
            item[
                "exact_correct"
            ]
            for item
            in items
        )

        accepted = [
            item
            for item
            in items
            if item[
                "prediction"
            ][
                "accepted"
            ]
        ]

        accepted_correct = sum(
            item[
                "exact_correct"
            ]
            for item
            in accepted
        )

        output[
            dimension
        ] = {
            "n":
                total,
            "exact_accuracy":
                (
                    correct
                    / total
                    if total
                    else None
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
        }

    return output


def run_benchmark(
    *,
    dataset: Path,
    output: Path,
    partial_output: Path,
    model: str,
    host: str,
    timeout_seconds: int,
    warmup_calls: int,
    prompt_freeze: Optional[
        Path
    ],
    resume: bool,
) -> Dict[str, Any]:
    validate_prompt_freeze(
        prompt_freeze,
        model,
    )

    rows = load_jsonl(
        dataset
    )

    completed: Dict[
        str,
        Dict[str, Any]
    ] = {}

    if (
        resume
        and partial_output.exists()
    ):
        for record in load_jsonl(
            partial_output
        ):
            completed[
                record[
                    "sample_id"
                ]
            ] = record

    if warmup_calls > 0:
        for _ in range(
            warmup_calls
        ):
            call_ollama(
                host=host,
                model=model,
                prompt=(
                    "Return JSON only: "
                    '{"dimensions":{}}'
                ),
                timeout_seconds=
                    timeout_seconds,
            )

    records: List[
        Dict[str, Any]
    ] = []

    with partial_output.open(
        "a"
        if (
            resume
            and partial_output.exists()
        )
        else "w",
        encoding="utf-8",
    ) as partial_handle:
        for index, row in enumerate(
            rows,
            start=1,
        ):
            sample_id = row[
                "sample_id"
            ]

            if sample_id in completed:
                records.append(
                    completed[
                        sample_id
                    ]
                )

                print(
                    f"[{index}/{len(rows)}] "
                    f"{sample_id} RESUME"
                )

                continue

            requested = [
                item[
                    "dimension"
                ]
                for item
                in row[
                    "fallback_dimensions"
                ]
            ]

            prompt = build_prompt(
                row[
                    "text"
                ],
                requested,
            )

            error = None

            raw = ""

            latency = None

            try:
                (
                    raw,
                    latency,
                ) = call_ollama(
                    host=host,
                    model=model,
                    prompt=prompt,
                    timeout_seconds=
                        timeout_seconds,
                )

                parsed = (
                    parse_llm_response(
                        raw_text=raw,
                        requested_dimensions=
                            requested,
                        user_text=
                            row[
                                "text"
                            ],
                    )
                )

            except Exception as exc:
                error = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                parsed = (
                    parse_llm_response(
                        raw_text="",
                        requested_dimensions=
                            requested,
                        user_text=
                            row[
                                "text"
                            ],
                    )
                )

            gold = expected_map(
                row
            )

            baseline = (
                transformer_baseline_map(
                    row
                )
            )

            dimension_results = []

            reason_by_dimension = {
                item[
                    "dimension"
                ]:
                    item[
                        "fallback_reason"
                    ]
                for item
                in row[
                    "fallback_dimensions"
                ]
            }

            for dimension in requested:
                prediction = (
                    parsed[
                        "dimensions"
                    ][
                        dimension
                    ]
                )

                gold_item = gold[
                    dimension
                ]

                baseline_item = (
                    baseline[
                        dimension
                    ]
                )

                dimension_results.append(
                    {
                        "dimension":
                            dimension,
                        "fallback_reason":
                            reason_by_dimension[
                                dimension
                            ],
                        "gold":
                            gold_item,
                        "prediction":
                            prediction,
                        "exact_correct":
                            prediction_matches_gold(
                                prediction,
                                gold_item,
                            ),
                        "transformer_raw_baseline":
                            baseline_item,
                        "transformer_baseline_correct":
                            prediction_matches_gold(
                                baseline_item,
                                gold_item,
                            ),
                    }
                )

            record = {
                "sample_id":
                    sample_id,
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
                "latency_s":
                    latency,
                "ollama_error":
                    error,
                "response_valid":
                    parsed[
                        "valid"
                    ],
                "response_violations":
                    parsed[
                        "violations"
                    ],
                "raw_response":
                    raw,
                "dimension_results":
                    dimension_results,
            }

            records.append(
                record
            )

            partial_handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            partial_handle.flush()

            exact = sum(
                item[
                    "exact_correct"
                ]
                for item
                in dimension_results
            )

            print(
                f"[{index}/{len(rows)}] "
                f"{sample_id} "
                f"{exact}/{len(dimension_results)} "
                f"latency="
                f"{latency if latency is not None else 'ERR'}"
            )

    overall = metric_block(
        records
    )

    report = {
        "step":
            "4.2A",
        "benchmark":
            "layer2_actual_llm_fallback_subset",
        "dataset":
            str(
                dataset
            ),
        "model":
            model,
        "host":
            host,
        "prompt_version":
            PROMPT_VERSION,
        "prompt_policy_sha256":
            prompt_policy_sha256(),
        "prompt_freeze_required":
            prompt_freeze is not None,
        "prompt_freeze":
            (
                str(
                    prompt_freeze
                )
                if prompt_freeze
                is not None
                else None
            ),
        "overall":
            overall,
        "per_dimension":
            per_dimension_metrics(
                records
            ),
        "per_language":
            aggregate_by(
                records,
                "language",
            ),
        "per_family":
            aggregate_by(
                records,
                "semantic_family",
            ),
        "protocol_note":
            (
                "The LLM sees only the current message and requested "
                "fallback dimensions. Gold labels in the fallback dataset "
                "are used only after inference for scoring."
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
        "--partial-output",
        type=Path,
        default=None,
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
        "--warmup-calls",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--prompt-freeze",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    args = parser.parse_args()

    partial = (
        args.partial_output
        if args.partial_output
        is not None
        else args.output
        .with_suffix(
            ".partial.jsonl"
        )
    )

    report = run_benchmark(
        dataset=
            args.dataset,
        output=
            args.output,
        partial_output=
            partial,
        model=
            args.model,
        host=
            args.host,
        timeout_seconds=
            args.timeout,
        warmup_calls=
            args.warmup_calls,
        prompt_freeze=
            args.prompt_freeze,
        resume=
            args.resume,
    )

    print(
        "\n"
        + "="
        * 80
    )

    print(
        "LAYER 2 LLM FALLBACK BENCHMARK"
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
