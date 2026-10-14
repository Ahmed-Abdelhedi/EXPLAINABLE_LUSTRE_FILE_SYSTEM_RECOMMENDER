from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DATA_DIR = Path(
    "preference_extractor/training/data/data_layer1_v2_1"
)

TRAIN = (
    DATA_DIR
    / "preference_signal_train_v2_1.jsonl"
)

VAL = (
    DATA_DIR
    / "preference_signal_val_v2_1.jsonl"
)

TARGETED_TRAIN = (
    DATA_DIR
    / "preference_signal_targeted_train_v2_1.jsonl"
)

TARGETED_VAL = (
    DATA_DIR
    / "preference_signal_targeted_val_v2_1.jsonl"
)

REGRESSION_VAL = (
    DATA_DIR
    / "preference_signal_regression_val_v2_5200.jsonl"
)

REGRESSION_HOLDOUT = (
    DATA_DIR
    / "preference_signal_regression_holdout_v2_1200.jsonl"
)

LEGACY_TEST = (
    DATA_DIR
    / "preference_signal_legacy_test_v1_4000.jsonl"
)

KNOWN_500 = Path(
    "preference_extractor/evaluation/datasets/"
    "preference_signal_test_v1.jsonl"
)

REPORT = (
    DATA_DIR
    / "preference_signal_v2_1_QA_REPORT.json"
)


def normalize(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(text).strip().casefold(),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def main() -> None:
    required_files = (
        TRAIN,
        VAL,
        TARGETED_TRAIN,
        TARGETED_VAL,
        REGRESSION_VAL,
        REGRESSION_HOLDOUT,
        LEGACY_TEST,
        KNOWN_500,
    )

    missing = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing files:\n- "
            + "\n- ".join(
                missing
            )
        )

    datasets = {
        "train_v2_1": read_jsonl(
            TRAIN
        ),
        "val_v2_1": read_jsonl(
            VAL
        ),
        "targeted_train": read_jsonl(
            TARGETED_TRAIN
        ),
        "targeted_val": read_jsonl(
            TARGETED_VAL
        ),
        "regression_val_v2": read_jsonl(
            REGRESSION_VAL
        ),
        "regression_holdout_v2": read_jsonl(
            REGRESSION_HOLDOUT
        ),
        "legacy_test_v1": read_jsonl(
            LEGACY_TEST
        ),
        "known_500": read_jsonl(
            KNOWN_500
        ),
    }

    checks = []

    def add_check(
        name: str,
        condition: bool,
        observed: Any,
        expected: Any,
    ) -> None:
        checks.append(
            {
                "name": name,
                "pass": bool(
                    condition
                ),
                "observed": observed,
                "expected": expected,
            }
        )

    expected_sizes = {
        "train_v2_1": 28400,
        "val_v2_1": 5800,
        "targeted_train": 2400,
        "targeted_val": 600,
        "regression_val_v2": 5200,
        "regression_holdout_v2": 1200,
        "legacy_test_v1": 4000,
        "known_500": 500,
    }

    for name, expected in (
        expected_sizes.items()
    ):
        observed = len(
            datasets[
                name
            ]
        )

        add_check(
            f"{name}_size",
            observed == expected,
            observed,
            expected,
        )

    required_fields = {
        "sample_id",
        "group_id",
        "text",
        "label_id",
        "language",
    }

    for name in (
        "train_v2_1",
        "val_v2_1",
        "targeted_train",
        "targeted_val",
    ):
        rows = datasets[
            name
        ]

        fields_ok = all(
            required_fields.issubset(
                row.keys()
            )
            for row in rows
        )

        add_check(
            f"{name}_required_fields",
            fields_ok,
            fields_ok,
            True,
        )

        unique_text = len(
            {
                normalize(
                    row[
                        "text"
                    ]
                )
                for row in rows
            }
        )

        add_check(
            f"{name}_unique_text",
            unique_text == len(
                rows
            ),
            unique_text,
            len(
                rows
            ),
        )

        unique_group = len(
            {
                str(
                    row[
                        "group_id"
                    ]
                )
                for row in rows
            }
        )

        add_check(
            f"{name}_unique_group_id",
            unique_group == len(
                rows
            ),
            unique_group,
            len(
                rows
            ),
        )

    expected_labels = {
        "train_v2_1": {
            0: 14200,
            1: 14200,
        },
        "val_v2_1": {
            0: 2900,
            1: 2900,
        },
        "targeted_train": {
            0: 1200,
            1: 1200,
        },
        "targeted_val": {
            0: 300,
            1: 300,
        },
    }

    for name, expected in (
        expected_labels.items()
    ):
        observed = dict(
            Counter(
                int(
                    row[
                        "label_id"
                    ]
                )
                for row in datasets[
                    name
                ]
            )
        )

        add_check(
            f"{name}_label_balance",
            observed == expected,
            observed,
            expected,
        )

    targeted_lang_expected = {
        "targeted_train": {
            "en": 1600,
            "mixed": 800,
        },
        "targeted_val": {
            "en": 400,
            "mixed": 200,
        },
    }

    for name, expected in (
        targeted_lang_expected.items()
    ):
        observed = dict(
            Counter(
                row[
                    "language"
                ]
                for row in datasets[
                    name
                ]
            )
        )

        add_check(
            f"{name}_language_targeting",
            observed == expected,
            observed,
            expected,
        )

    targeted_train_family = dict(
        Counter(
            row[
                "hard_family"
            ]
            for row in datasets[
                "targeted_train"
            ]
        )
    )

    targeted_val_family = dict(
        Counter(
            row[
                "hard_family"
            ]
            for row in datasets[
                "targeted_val"
            ]
        )
    )

    expected_train_family = {
        "v21_long_context_final_choice": 1200,
        "v21_long_context_rejected_choice": 300,
        "v21_long_context_third_party_choice": 300,
        "v21_long_context_capability_choice": 300,
        "v21_long_context_no_preference": 300,
    }

    expected_val_family = {
        "v21_long_context_final_choice": 300,
        "v21_long_context_rejected_choice": 75,
        "v21_long_context_third_party_choice": 75,
        "v21_long_context_capability_choice": 75,
        "v21_long_context_no_preference": 75,
    }

    add_check(
        "targeted_train_family_counts",
        targeted_train_family
        == expected_train_family,
        targeted_train_family,
        expected_train_family,
    )

    add_check(
        "targeted_val_family_counts",
        targeted_val_family
        == expected_val_family,
        targeted_val_family,
        expected_val_family,
    )

    train_text = {
        normalize(
            row[
                "text"
            ]
        )
        for row in datasets[
            "train_v2_1"
        ]
    }

    val_text = {
        normalize(
            row[
                "text"
            ]
        )
        for row in datasets[
            "val_v2_1"
        ]
    }

    train_val_overlap = len(
        train_text
        & val_text
    )

    add_check(
        "combined_train_val_text_overlap_zero",
        train_val_overlap == 0,
        train_val_overlap,
        0,
    )

    train_groups = {
        str(
            row[
                "group_id"
            ]
        )
        for row in datasets[
            "train_v2_1"
        ]
    }

    val_groups = {
        str(
            row[
                "group_id"
            ]
        )
        for row in datasets[
            "val_v2_1"
        ]
    }

    group_overlap = len(
        train_groups
        & val_groups
    )

    add_check(
        "combined_train_val_group_overlap_zero",
        group_overlap == 0,
        group_overlap,
        0,
    )

    ttrain = datasets[
        "targeted_train"
    ]

    tval = datasets[
        "targeted_val"
    ]

    targeted_template_family_overlap = len(
        {
            row[
                "template_family"
            ]
            for row in ttrain
        }
        & {
            row[
                "template_family"
            ]
            for row in tval
        }
    )

    targeted_template_id_overlap = len(
        {
            row[
                "template_id"
            ]
            for row in ttrain
        }
        & {
            row[
                "template_id"
            ]
            for row in tval
        }
    )

    add_check(
        "targeted_train_val_template_family_overlap_zero",
        targeted_template_family_overlap
        == 0,
        targeted_template_family_overlap,
        0,
    )

    add_check(
        "targeted_train_val_template_id_overlap_zero",
        targeted_template_id_overlap
        == 0,
        targeted_template_id_overlap,
        0,
    )

    targeted_train_text = {
        normalize(
            row[
                "text"
            ]
        )
        for row in ttrain
    }

    targeted_val_text = {
        normalize(
            row[
                "text"
            ]
        )
        for row in tval
    }

    for ref_name in (
        "regression_val_v2",
        "regression_holdout_v2",
        "known_500",
    ):
        ref_text = {
            normalize(
                row[
                    "text"
                ]
            )
            for row in datasets[
                ref_name
            ]
        }

        overlap_train = len(
            targeted_train_text
            & ref_text
        )

        overlap_val = len(
            targeted_val_text
            & ref_text
        )

        add_check(
            f"targeted_train_{ref_name}_text_overlap_zero",
            overlap_train == 0,
            overlap_train,
            0,
        )

        add_check(
            f"targeted_val_{ref_name}_text_overlap_zero",
            overlap_val == 0,
            overlap_val,
            0,
        )

    # Specifically protect the known 72 old validation FN:
    # targeted examples must not be copies of old validation messages.
    old_val_text = {
        normalize(
            row[
                "text"
            ]
        )
        for row in datasets[
            "regression_val_v2"
        ]
    }

    known72_copy_count = len(
        (
            targeted_train_text
            | targeted_val_text
        )
        & old_val_text
    )

    add_check(
        "known_v2_validation_messages_not_copied",
        known72_copy_count == 0,
        known72_copy_count,
        0,
    )

    all_pass = all(
        check[
            "pass"
        ]
        for check in checks
    )

    payload = {
        "dataset": (
            "preference_signal_layer1_v2_1"
        ),
        "all_checks_pass": all_pass,
        "checks_passed": sum(
            int(
                check[
                    "pass"
                ]
            )
            for check in checks
        ),
        "checks_total": len(
            checks
        ),
        "checks": checks,
        "protocol": {
            "old_v2_validation": (
                "development/regression only"
            ),
            "old_v2_holdout": (
                "regression only"
            ),
            "known_500": (
                "regression only"
            ),
            "fresh_final_holdout_v3": (
                "must remain untouched until V2.1 model "
                "and final threshold are frozen"
            ),
        },
    }

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        f"\nSaved: {REPORT}"
    )

    if not all_pass:
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
