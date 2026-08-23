from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


KNOWN_DATASETS = {
    "v2_independent_end_to_end_benchmark_v1.jsonl": {
        "expected_count": 300,
        "evaluation_role": "broad_ablation_benchmark",
    },
    "quantity_e2e_holdout_v1.jsonl": {
        "expected_count": 96,
        "evaluation_role": "quantity_e2e_holdout",
    },
    "actual_fallback_subset_v1.jsonl": {
        "expected_count": 23,
        "evaluation_role": "actual_llm_fallback_subset",
    },
}


def load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return None


def jsonl_count(path: Path) -> int:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return sum(
            1
            for line in handle
            if line.strip()
        )


def recursive_strings(
    value: Any,
) -> Iterable[str]:
    if isinstance(
        value,
        str,
    ):
        yield value
    elif isinstance(
        value,
        dict,
    ):
        for child in value.values():
            yield from recursive_strings(
                child
            )
    elif isinstance(
        value,
        list,
    ):
        for child in value:
            yield from recursive_strings(
                child
            )


def first_number(
    data: Dict[str, Any],
    keys: List[str],
) -> Optional[int]:
    containers = [
        data,
    ]

    metrics = data.get(
        "metrics"
    )

    if isinstance(
        metrics,
        dict,
    ):
        containers.append(
            metrics
        )

    metadata = data.get(
        "metadata"
    )

    if isinstance(
        metadata,
        dict,
    ):
        containers.append(
            metadata
        )

    for container in containers:
        for key in keys:
            value = container.get(
                key
            )

            if isinstance(
                value,
                bool,
            ):
                continue

            if isinstance(
                value,
                int,
            ):
                return value

            if isinstance(
                value,
                float,
            ) and value.is_integer():
                return int(
                    value
                )

    return None


def detect_count_claims(
    strings: Iterable[str],
) -> List[int]:
    claims = set()

    patterns = (
        r"\b(96|300)\s*[- ]message\b",
        r"\b(96|300)\s+messages\b",
        r"\bbenchmark\s+(?:of|contains?)\s+(96|300)\b",
        r"\bfixed\s+(96|300)\b",
    )

    for text in strings:
        for pattern in patterns:
            for match in re.finditer(
                pattern,
                text,
                flags=re.IGNORECASE,
            ):
                claims.add(
                    int(
                        match.group(1)
                    )
                )

    return sorted(
        claims
    )


def infer_dataset_mentions(
    strings: Iterable[str],
) -> List[str]:
    found = set()

    for text in strings:
        for name in (
            KNOWN_DATASETS
        ):
            if name in text:
                found.add(
                    name
                )

    return sorted(
        found
    )


def audit_datasets(
    datasets_dir: Path,
) -> List[
    Dict[str, Any]
]:
    rows: List[
        Dict[str, Any]
    ] = []

    for name, spec in (
        KNOWN_DATASETS.items()
    ):
        path = (
            datasets_dir
            / name
        )

        if not path.exists():
            rows.append(
                {
                    "dataset":
                        name,
                    "status":
                        "MISSING",
                    "expected_count":
                        spec[
                            "expected_count"
                        ],
                    "actual_count":
                        None,
                    "evaluation_role":
                        spec[
                            "evaluation_role"
                        ],
                }
            )
            continue

        actual = jsonl_count(
            path
        )

        rows.append(
            {
                "dataset":
                    name,
                "status": (
                    "OK"
                    if actual
                    == spec[
                        "expected_count"
                    ]
                    else "COUNT_MISMATCH"
                ),
                "expected_count":
                    spec[
                        "expected_count"
                    ],
                "actual_count":
                    actual,
                "evaluation_role":
                    spec[
                        "evaluation_role"
                    ],
            }
        )

    return rows


def audit_result_file(
    path: Path,
) -> Dict[str, Any]:
    data = load_json(
        path
    )

    if not isinstance(
        data,
        dict,
    ):
        return {
            "file":
                str(path),
            "status":
                "SKIP_NOT_JSON_OBJECT",
            "n":
                None,
            "count_claims":
                [],
            "dataset_mentions":
                [],
            "issues":
                [],
        }

    n = first_number(
        data,
        [
            "n_messages",
            "n_cases",
            "source_messages",
        ],
    )

    all_strings = list(
        recursive_strings(
            data
        )
    )

    claims = (
        detect_count_claims(
            all_strings
        )
    )

    mentions = (
        infer_dataset_mentions(
            all_strings
        )
    )

    issues: List[str] = []

    if (
        n is not None
        and claims
        and n not in claims
    ):
        issues.append(
            (
                "numeric_count_vs_methodology_claim: "
                f"stored_count={n}, "
                f"text_claims={claims}"
            )
        )

    if (
        96 in claims
        and 300 in claims
    ):
        issues.append(
            "both_96_and_300_claimed_in_same_artifact"
        )

    for dataset_name in mentions:
        expected = (
            KNOWN_DATASETS[
                dataset_name
            ][
                "expected_count"
            ]
        )

        if (
            n is not None
            and dataset_name
            !=
            "actual_fallback_subset_v1.jsonl"
            and n != expected
        ):
            issues.append(
                (
                    "dataset_name_vs_count: "
                    f"{dataset_name} expects "
                    f"{expected}, stored_count={n}"
                )
            )

    filename_lower = (
        path.name.casefold()
    )

    if (
        "first_run"
        in filename_lower
        and n == 96
    ):
        # This is good and intentionally preserved.
        pass

    status = (
        "REVIEW"
        if issues
        else "OK"
    )

    return {
        "file":
            str(path),
        "status":
            status,
        "n":
            n,
        "count_claims":
            claims,
        "dataset_mentions":
            mentions,
        "issues":
            issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Requirement Extractor V2 evaluation metadata for "
            "96-vs-300 benchmark inconsistencies. No file is modified."
        )
    )

    parser.add_argument(
        "--evaluation-dir",
        default=(
            "requirement_extractor_v2/evaluation"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "requirement_extractor_v2/evaluation/results/"
            "evaluation_metadata_audit.json"
        ),
    )

    args = parser.parse_args()

    evaluation_dir = Path(
        args.evaluation_dir
    )

    datasets_dir = (
        evaluation_dir
        / "datasets"
    )

    results_dir = (
        evaluation_dir
        / "results"
    )

    dataset_rows = (
        audit_datasets(
            datasets_dir
        )
    )

    result_rows: List[
        Dict[str, Any]
    ] = []

    if results_dir.exists():
        for path in sorted(
            results_dir.rglob(
                "*.json"
            )
        ):
            result_rows.append(
                audit_result_file(
                    path
                )
            )

    # Also inspect legacy result JSON files directly under evaluation/.
    for path in sorted(
        evaluation_dir.glob(
            "*.json"
        )
    ):
        result_rows.append(
            audit_result_file(
                path
            )
        )

    review_rows = [
        row
        for row in result_rows
        if row[
            "status"
        ] == "REVIEW"
    ]

    output = {
        "protocol": {
            "300_message_dataset":
                (
                    "v2_independent_end_to_end_benchmark_v1.jsonl"
                ),
            "300_message_role":
                "broad_ablation_benchmark",
            "96_message_dataset":
                "quantity_e2e_holdout_v1.jsonl",
            "96_message_role":
                "quantity_e2e_holdout",
            "96_first_run_rule":
                (
                    "Only the preserved first run is called independent; "
                    "later runs after observed-failure fixes are regression."
                ),
        },
        "datasets":
            dataset_rows,
        "result_files_scanned":
            len(
                result_rows
            ),
        "review_count":
            len(
                review_rows
            ),
        "review_items":
            review_rows,
        "all_result_items":
            result_rows,
    }

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print(
        "EVALUATION METADATA AUDIT"
    )
    print("=" * 88)

    print(
        "\nDATASETS"
    )

    for row in dataset_rows:
        print(
            f"- {row['dataset']}: "
            f"{row['status']} "
            f"(actual={row['actual_count']}, "
            f"expected={row['expected_count']}, "
            f"role={row['evaluation_role']})"
        )

    print()
    print(
        "RESULT JSON FILES"
    )
    print(
        f"- scanned: {len(result_rows)}"
    )
    print(
        f"- need review: "
        f"{len(review_rows)}"
    )

    if review_rows:
        print(
            "\nFILES TO REVIEW"
        )

        for row in review_rows:
            print(
                f"\n- {row['file']}"
            )

            print(
                f"  stored count: "
                f"{row['n']}"
            )

            print(
                f"  textual claims: "
                f"{row['count_claims']}"
            )

            for issue in (
                row["issues"]
            ):
                print(
                    f"  ISSUE: {issue}"
                )

    print()
    print(
        "Audit JSON written to:"
    )
    print(
        output_path
    )

    print()
    print(
        "No metrics or result files were modified."
    )


if __name__ == "__main__":
    main()
