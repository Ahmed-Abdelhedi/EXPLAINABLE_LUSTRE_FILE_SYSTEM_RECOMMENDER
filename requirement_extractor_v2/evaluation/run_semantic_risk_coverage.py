from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from requirement_extractor_v2.models import (
    Quantity,
    QuantityDetection,
    QuantityDimension,
)
from requirement_extractor_v2.semantic_linker.runtime import (
    SemanticLinkerRuntime,
)


DEFAULT_THRESHOLDS = [
    0.60,
    0.70,
    0.80,
    0.85,
    0.90,
    0.95,
    0.966,
]


# =====================================================================
# DATA STRUCTURES
# =====================================================================


@dataclass
class RawSemanticPrediction:
    sample_id: str
    group_id: Optional[str]
    language: Optional[str]
    source: Optional[str]

    gold_field: str
    gold_role: str
    dimension: str

    raw_field: str
    raw_role: str

    confidence: float
    margin: float

    pair_correct: bool
    field_correct: bool
    role_correct: bool

    production_safety_passed: bool

    span_warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# IO HELPERS
# =====================================================================


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(
                    json.loads(line)
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from exc

    return rows


def write_json(
    path: Path,
    payload: Any,
) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def write_jsonl(
    path: Path,
    rows: Iterable[dict],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


# =====================================================================
# DATASET VALIDATION
# =====================================================================


REQUIRED_KEYS = {
    "sample_id",
    "original_text",
    "marked_text",
    "quantity_id",
    "raw_quantity",
    "value",
    "unit",
    "dimension",
    "field",
    "role",
}


def validate_dataset(
    rows: Sequence[dict],
) -> None:
    if not rows:
        raise ValueError(
            "Dataset is empty."
        )

    seen_sample_ids = set()

    for index, row in enumerate(rows, start=1):
        missing = (
            REQUIRED_KEYS
            - set(row)
        )

        if missing:
            raise ValueError(
                f"Row {index} is missing keys: "
                f"{sorted(missing)}"
            )

        sample_id = row["sample_id"]

        if sample_id in seen_sample_ids:
            raise ValueError(
                f"Duplicate sample_id: {sample_id}"
            )

        seen_sample_ids.add(
            sample_id
        )

        try:
            QuantityDimension(
                row["dimension"]
            )
        except ValueError as exc:
            raise ValueError(
                f"Unsupported dimension "
                f"{row['dimension']!r} "
                f"for sample {sample_id}"
            ) from exc

        marked_text = row["marked_text"]

        if (
            "[Q]" not in marked_text
            or "[/Q]" not in marked_text
        ):
            raise ValueError(
                f"Missing [Q]/[/Q] markers "
                f"for sample {sample_id}"
            )


# =====================================================================
# TARGET QUANTITY RECONSTRUCTION
# =====================================================================


def _extract_marked_surface(
    marked_text: str,
) -> Optional[str]:
    start_marker = marked_text.find(
        "[Q]"
    )
    end_marker = marked_text.find(
        "[/Q]"
    )

    if (
        start_marker < 0
        or end_marker < 0
        or end_marker <= start_marker
    ):
        return None

    return marked_text[
        start_marker + 3:
        end_marker
    ]


def build_quantity_for_safety_gate(
    row: dict,
) -> tuple[Quantity, Optional[str]]:
    """
    Reconstruct only the Quantity information required by the current
    production safety gate.

    Model inference itself uses the stored `marked_text` directly so the
    benchmark reproduces the original Semantic Linker test input as closely
    as possible.

    The source span here is used only by the deterministic production safety
    gate, which inspects local context around UNKNOWN/unitless quantities.
    """

    text = row["original_text"]

    raw_quantity = (
        row.get("raw_quantity")
        or _extract_marked_surface(
            row["marked_text"]
        )
        or ""
    )

    span_warning = None

    occurrences: List[int] = []

    cursor = 0

    while raw_quantity:
        found = text.find(
            raw_quantity,
            cursor,
        )

        if found < 0:
            break

        occurrences.append(
            found
        )

        cursor = (
            found
            + max(
                1,
                len(raw_quantity),
            )
        )

    if len(occurrences) == 1:
        start = occurrences[0]
        end = (
            start
            + len(raw_quantity)
        )

    elif len(occurrences) > 1:
        # The curated Semantic Linker set normally has one target occurrence.
        # If the same surface appears several times, use the first one and
        # record the case in the audit output.
        start = occurrences[0]
        end = (
            start
            + len(raw_quantity)
        )

        span_warning = (
            f"raw_quantity occurs "
            f"{len(occurrences)} times; "
            f"first occurrence used "
            f"for production safety gate"
        )

    else:
        # This does NOT affect model-only risk/coverage because inference uses
        # stored marked_text. It only means the production safety gate cannot
        # be perfectly localized for this record.
        start = 0
        end = 0

        span_warning = (
            "raw_quantity not found in original_text; "
            "fallback zero-length span used "
            "for production safety gate"
        )

    quantity = Quantity(
        id=row["quantity_id"],
        raw=raw_quantity,
        normalized=(
            row.get(
                "normalized_quantity"
            )
            or raw_quantity
        ),
        value=row["value"],
        unit=row.get("unit"),
        dimension=QuantityDimension(
            row["dimension"]
        ),
        start=start,
        end=end,
        source_text=text,
        detection=QuantityDetection.UNKNOWN,
        corrected=bool(
            row.get(
                "corrected",
                False,
            )
        ),
    )

    return (
        quantity,
        span_warning,
    )


# =====================================================================
# RAW MODEL INFERENCE
# =====================================================================


@torch.no_grad()
def collect_raw_predictions(
    runtime: SemanticLinkerRuntime,
    rows: Sequence[dict],
    batch_size: int,
) -> List[RawSemanticPrediction]:
    """
    Run XLM-R once over the full test set and store raw constrained-decoder
    outputs.

    Important:
        We feed the dataset's original `marked_text` directly.

    This is intentional because the original Semantic Linker training/test
    notebook evaluated exactly the already-marked input. It avoids mixing the
    separate QuantityScanner evaluation into the Semantic Linker benchmark.
    """

    predictions: List[
        RawSemanticPrediction
    ] = []

    runtime.model.eval()

    for batch_start in range(
        0,
        len(rows),
        batch_size,
    ):
        batch = rows[
            batch_start:
            batch_start + batch_size
        ]

        marked_texts = [
            row["marked_text"]
            for row in batch
        ]

        encoded = runtime.tokenizer(
            marked_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=runtime.max_length,
        )

        input_ids = (
            encoded["input_ids"]
            .to(runtime.device)
        )

        attention_mask = (
            encoded["attention_mask"]
            .to(runtime.device)
        )

        field_logits, role_logits = (
            runtime.model(
                input_ids=input_ids,
                attention_mask=
                    attention_mask,
            )
        )

        field_logits = (
            field_logits
            .float()
            .cpu()
        )

        role_logits = (
            role_logits
            .float()
            .cpu()
        )

        for local_index, row in enumerate(batch):
            (
                field_id,
                role_id,
                confidence,
                margin,
            ) = runtime._decode_valid_pair(
                field_logits[
                    local_index:
                    local_index + 1
                ],
                role_logits[
                    local_index:
                    local_index + 1
                ],
                row["dimension"],
            )

            raw_field = (
                runtime.id_to_field[
                    field_id
                ]
            )

            raw_role = (
                runtime.id_to_role[
                    role_id
                ]
            )

            quantity, span_warning = (
                build_quantity_for_safety_gate(
                    row
                )
            )

            safety_passed = (
                runtime
                ._passes_production_safety_gate(
                    text=row[
                        "original_text"
                    ],
                    quantity=quantity,
                    predicted_field=
                        raw_field,
                )
            )

            gold_field = row["field"]
            gold_role = row["role"]

            predictions.append(
                RawSemanticPrediction(
                    sample_id=
                        row["sample_id"],
                    group_id=
                        row.get(
                            "group_id"
                        ),
                    language=
                        row.get(
                            "language"
                        ),
                    source=
                        row.get(
                            "source"
                        ),
                    gold_field=
                        gold_field,
                    gold_role=
                        gold_role,
                    dimension=
                        row["dimension"],
                    raw_field=
                        raw_field,
                    raw_role=
                        raw_role,
                    confidence=float(
                        confidence
                    ),
                    margin=float(
                        margin
                    ),
                    pair_correct=(
                        raw_field
                        == gold_field
                        and
                        raw_role
                        == gold_role
                    ),
                    field_correct=(
                        raw_field
                        == gold_field
                    ),
                    role_correct=(
                        raw_role
                        == gold_role
                    ),
                    production_safety_passed=
                        bool(
                            safety_passed
                        ),
                    span_warning=
                        span_warning,
                )
            )

        done = min(
            batch_start
            + len(batch),
            len(rows),
        )

        print(
            f"Inference: {done}/{len(rows)}"
        )

    return predictions


# =====================================================================
# METRICS
# =====================================================================


def safe_divide(
    numerator: int | float,
    denominator: int | float,
) -> Optional[float]:
    if denominator == 0:
        return None

    return (
        numerator
        / denominator
    )


def compute_raw_metrics(
    predictions: Sequence[
        RawSemanticPrediction
    ],
) -> dict:
    total = len(
        predictions
    )

    field_correct = sum(
        int(item.field_correct)
        for item in predictions
    )

    role_correct = sum(
        int(item.role_correct)
        for item in predictions
    )

    pair_correct = sum(
        int(item.pair_correct)
        for item in predictions
    )

    gold_unresolved = [
        item
        for item in predictions
        if item.gold_field
        == "__UNRESOLVED__"
    ]

    raw_unresolved_correct = sum(
        int(
            item.raw_field
            == "__UNRESOLVED__"
        )
        for item
        in gold_unresolved
    )

    return {
        "n_samples":
            total,

        "field_accuracy":
            safe_divide(
                field_correct,
                total,
            ),

        "role_accuracy":
            safe_divide(
                role_correct,
                total,
            ),

        "pair_accuracy":
            safe_divide(
                pair_correct,
                total,
            ),

        "gold_unresolved_count":
            len(
                gold_unresolved
            ),

        "raw_unresolved_recall":
            safe_divide(
                raw_unresolved_correct,
                len(
                    gold_unresolved
                ),
            ),
    }


def acceptance_metrics(
    predictions: Sequence[
        RawSemanticPrediction
    ],
    threshold: float,
    margin_threshold: float,
    production_gate: bool,
) -> dict:
    """
    Selective classification metrics.

    Coverage:
        accepted / all test examples

    Accepted precision:
        correct accepted pairs / accepted

    Risk:
        1 - accepted precision

    Unresolved recall:
        fraction of gold __UNRESOLVED__ examples that remain rejected

    Final pair accuracy:
        accepted examples use raw FIELD/ROLE;
        rejected examples become (__UNRESOLVED__, unspecified).
    """

    total = len(
        predictions
    )

    accepted_flags: List[
        bool
    ] = []

    for item in predictions:
        accepted = (
            item.raw_field
            != "__UNRESOLVED__"
            and
            item.confidence
            >= threshold
            and
            item.margin
            >= margin_threshold
        )

        if production_gate:
            accepted = (
                accepted
                and
                item.production_safety_passed
            )

        accepted_flags.append(
            bool(accepted)
        )

    accepted_count = sum(
        int(flag)
        for flag in accepted_flags
    )

    accepted_correct = sum(
        int(
            accepted
            and
            item.pair_correct
        )
        for item, accepted
        in zip(
            predictions,
            accepted_flags,
        )
    )

    accepted_precision = (
        safe_divide(
            accepted_correct,
            accepted_count,
        )
    )

    risk = (
        None
        if accepted_precision is None
        else
        1.0
        - accepted_precision
    )

    gold_unresolved_count = 0
    gold_unresolved_rejected = 0

    final_pair_correct = 0

    for item, accepted in zip(
        predictions,
        accepted_flags,
    ):
        gold_is_unresolved = (
            item.gold_field
            == "__UNRESOLVED__"
        )

        if gold_is_unresolved:
            gold_unresolved_count += 1

            if not accepted:
                gold_unresolved_rejected += 1

        if accepted:
            final_field = (
                item.raw_field
            )
            final_role = (
                item.raw_role
            )

        else:
            final_field = (
                "__UNRESOLVED__"
            )
            final_role = (
                "unspecified"
            )

        if (
            final_field
            == item.gold_field
            and
            final_role
            == item.gold_role
        ):
            final_pair_correct += 1

    return {
        "accepted_count":
            accepted_count,

        "accepted_correct":
            accepted_correct,

        "coverage":
            safe_divide(
                accepted_count,
                total,
            ),

        "accepted_precision":
            accepted_precision,

        "risk":
            risk,

        "gold_unresolved_count":
            gold_unresolved_count,

        "unresolved_recall":
            safe_divide(
                gold_unresolved_rejected,
                gold_unresolved_count,
            ),

        "final_pair_accuracy":
            safe_divide(
                final_pair_correct,
                total,
            ),
    }


def build_risk_coverage_table(
    predictions: Sequence[
        RawSemanticPrediction
    ],
    thresholds: Sequence[float],
    margin_threshold: float,
) -> List[dict]:
    rows: List[dict] = []

    for threshold in thresholds:
        model_only = (
            acceptance_metrics(
                predictions=
                    predictions,
                threshold=
                    threshold,
                margin_threshold=
                    margin_threshold,
                production_gate=False,
            )
        )

        production = (
            acceptance_metrics(
                predictions=
                    predictions,
                threshold=
                    threshold,
                margin_threshold=
                    margin_threshold,
                production_gate=True,
            )
        )

        rows.append(
            {
                "threshold":
                    threshold,

                "margin_threshold":
                    margin_threshold,

                "model_accepted_count":
                    model_only[
                        "accepted_count"
                    ],

                "model_accepted_correct":
                    model_only[
                        "accepted_correct"
                    ],

                "model_coverage":
                    model_only[
                        "coverage"
                    ],

                "model_accepted_precision":
                    model_only[
                        "accepted_precision"
                    ],

                "model_risk":
                    model_only[
                        "risk"
                    ],

                "model_unresolved_recall":
                    model_only[
                        "unresolved_recall"
                    ],

                "model_final_pair_accuracy":
                    model_only[
                        "final_pair_accuracy"
                    ],

                "production_accepted_count":
                    production[
                        "accepted_count"
                    ],

                "production_accepted_correct":
                    production[
                        "accepted_correct"
                    ],

                "production_coverage":
                    production[
                        "coverage"
                    ],

                "production_accepted_precision":
                    production[
                        "accepted_precision"
                    ],

                "production_risk":
                    production[
                        "risk"
                    ],

                "production_unresolved_recall":
                    production[
                        "unresolved_recall"
                    ],

                "production_final_pair_accuracy":
                    production[
                        "final_pair_accuracy"
                    ],
            }
        )

    return rows


# =====================================================================
# HISTORICAL CONSISTENCY CHECK
# =====================================================================


def load_historical_metrics(
    runtime: SemanticLinkerRuntime,
) -> Optional[dict]:
    path = (
        runtime.artifact_dir
        / "test_metrics.json"
    )

    if not path.is_file():
        return None

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(
            handle
        )


def nearest_threshold_row(
    table: Sequence[dict],
    target: float,
) -> Optional[dict]:
    if not table:
        return None

    return min(
        table,
        key=lambda row: abs(
            float(
                row["threshold"]
            )
            - target
        ),
    )


def compare_with_historical(
    raw_metrics: dict,
    table: Sequence[dict],
    historical: Optional[dict],
) -> dict:
    if historical is None:
        return {
            "available": False,
            "reason":
                "test_metrics.json not found",
        }

    result: Dict[str, Any] = {
        "available": True,
        "raw": {},
        "selective": {},
    }

    raw_historical = (
        historical.get(
            "raw_constrained",
            {}
        )
    )

    for key in (
        "field_accuracy",
        "role_accuracy",
        "pair_accuracy",
    ):
        observed = (
            raw_metrics.get(key)
        )

        expected = (
            raw_historical.get(key)
        )

        result["raw"][key] = {
            "observed":
                observed,
            "historical":
                expected,
            "absolute_difference":
                (
                    None
                    if (
                        observed is None
                        or expected is None
                    )
                    else
                    abs(
                        observed
                        - expected
                    )
                ),
        }

    selective_historical = (
        historical.get(
            "selective",
            {}
        )
    )

    historical_threshold = (
        selective_historical.get(
            "confidence_threshold"
        )
    )

    if historical_threshold is None:
        result["selective"] = {
            "available": False,
            "reason":
                "historical confidence threshold missing",
        }
        return result

    row = nearest_threshold_row(
        table,
        float(
            historical_threshold
        ),
    )

    if row is None:
        result["selective"] = {
            "available": False,
            "reason":
                "risk/coverage table is empty",
        }
        return result

    result["selective"] = {
        "available": True,
        "historical_threshold":
            historical_threshold,
        "evaluated_threshold":
            row["threshold"],
        "note":
            (
                "Historical selective metrics are compared "
                "against MODEL-ONLY acceptance because the "
                "production safety gate was added after the "
                "original trained-model evaluation."
            ),
    }

    mapping = {
        "accepted_count":
            "model_accepted_count",
        "accepted_precision":
            "model_accepted_precision",
        "coverage":
            "model_coverage",
        "unresolved_recall":
            "model_unresolved_recall",
        "final_pair_accuracy":
            "model_final_pair_accuracy",
    }

    for historical_key, observed_key in mapping.items():
        observed = row.get(
            observed_key
        )

        expected = (
            selective_historical.get(
                historical_key
            )
        )

        result["selective"][
            historical_key
        ] = {
            "observed":
                observed,
            "historical":
                expected,
            "absolute_difference":
                (
                    None
                    if (
                        observed is None
                        or expected is None
                    )
                    else
                    abs(
                        observed
                        - expected
                    )
                ),
        }

    return result


# =====================================================================
# REPORTING
# =====================================================================


def percent(
    value: Optional[float],
) -> str:
    if value is None:
        return "N/A"

    return (
        f"{100.0 * value:.2f}%"
    )


def print_table(
    table: Sequence[dict],
) -> None:
    print()
    print(
        "=" * 132
    )
    print(
        "SEMANTIC LINKER — RISK / COVERAGE"
    )
    print(
        "=" * 132
    )

    header = (
        f"{'Thr':>7} | "
        f"{'Model Acc':>9} | "
        f"{'Model Cov':>10} | "
        f"{'Model Prec':>11} | "
        f"{'Model Risk':>10} || "
        f"{'Prod Acc':>8} | "
        f"{'Prod Cov':>9} | "
        f"{'Prod Prec':>10} | "
        f"{'Prod Risk':>9}"
    )

    print(
        header
    )
    print(
        "-" * len(header)
    )

    for row in table:
        print(
            f"{row['threshold']:7.3f} | "
            f"{row['model_accepted_count']:9d} | "
            f"{percent(row['model_coverage']):>10} | "
            f"{percent(row['model_accepted_precision']):>11} | "
            f"{percent(row['model_risk']):>10} || "
            f"{row['production_accepted_count']:8d} | "
            f"{percent(row['production_coverage']):>9} | "
            f"{percent(row['production_accepted_precision']):>10} | "
            f"{percent(row['production_risk']):>9}"
        )

    print()


def write_csv(
    path: Path,
    table: Sequence[dict],
) -> None:
    if not table:
        return

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                table[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            table
        )


def maybe_plot(
    path: Path,
    table: Sequence[dict],
) -> dict:
    """
    Create a compact risk/coverage figure if matplotlib is installed.

    The benchmark remains valid even if plotting is unavailable.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {
            "created": False,
            "reason":
                "matplotlib is not installed",
        }

    model_coverage = [
        row["model_coverage"]
        for row in table
        if (
            row["model_coverage"]
            is not None
            and
            row["model_risk"]
            is not None
        )
    ]

    model_risk = [
        row["model_risk"]
        for row in table
        if (
            row["model_coverage"]
            is not None
            and
            row["model_risk"]
            is not None
        )
    ]

    production_coverage = [
        row["production_coverage"]
        for row in table
        if (
            row["production_coverage"]
            is not None
            and
            row["production_risk"]
            is not None
        )
    ]

    production_risk = [
        row["production_risk"]
        for row in table
        if (
            row["production_coverage"]
            is not None
            and
            row["production_risk"]
            is not None
        )
    ]

    fig, ax = plt.subplots(
        figsize=(7.5, 5.0)
    )

    ax.plot(
        model_coverage,
        model_risk,
        marker="o",
        label="Model-only selective",
    )

    ax.plot(
        production_coverage,
        production_risk,
        marker="o",
        label="Production + safety gate",
    )

    for row in table:
        if (
            row["model_coverage"]
            is not None
            and
            row["model_risk"]
            is not None
        ):
            ax.annotate(
                f"{row['threshold']:.3f}",
                (
                    row["model_coverage"],
                    row["model_risk"],
                ),
                fontsize=8,
            )

    ax.set_xlabel(
        "Coverage"
    )

    ax.set_ylabel(
        "Risk (1 - accepted precision)"
    )

    ax.set_title(
        "Semantic Linker Risk–Coverage"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return {
        "created": True,
        "path": str(path),
    }


# =====================================================================
# CLI
# =====================================================================


def parse_thresholds(
    raw_values: Sequence[str],
) -> List[float]:
    values: List[float] = []

    for raw in raw_values:
        value = float(
            raw
        )

        if not (
            0.0 <= value <= 1.0
        ):
            raise ValueError(
                f"Threshold must be in [0, 1], got {value}"
            )

        values.append(
            value
        )

    return sorted(
        set(
            values
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Semantic Linker selective classification "
            "with risk/coverage curves on the original 450-example "
            "test split."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "Path to semantic_linker_test_v1_1.jsonl"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "requirement_extractor_v2/"
            "evaluation/"
            "semantic_risk_coverage"
        ),
    )

    parser.add_argument(
        "--artifact-dir",
        default=None,
        help=(
            "Optional Semantic Linker artifact directory. "
            "When omitted, runtime auto-discovery is used."
        ),
    )

    parser.add_argument(
        "--device",
        default=None,
        help=(
            "cpu, cuda, etc. Default: runtime auto-detection."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--thresholds",
        nargs="+",
        default=[
            str(value)
            for value
            in DEFAULT_THRESHOLDS
        ],
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive"
        )

    dataset_path = Path(
        args.dataset
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    thresholds = parse_thresholds(
        args.thresholds
    )

    rows = load_jsonl(
        dataset_path
    )

    validate_dataset(
        rows
    )

    print(
        f"Dataset: {dataset_path}"
    )

    print(
        f"Samples: {len(rows)}"
    )

    print(
        "Loading Semantic Linker..."
    )

    runtime = SemanticLinkerRuntime(
        artifact_dir=
            args.artifact_dir,
        device=
            args.device,
    )

    print(
        f"Device: {runtime.device}"
    )

    print(
        f"Artifact: {runtime.artifact_dir}"
    )

    print(
        "Calibration revision: "
        f"{runtime.calibration_revision}"
    )

    print(
        "Role threshold example (client_count): "
        f"{runtime.role_thresholds.get('client_count', {})}"
    )

    # V3.3 no longer uses one global confidence/margin threshold.
    # Keep the evaluation grid explicit and independent.
    thresholds = sorted(
        set(thresholds)
    )

    predictions = (
        collect_raw_predictions(
            runtime=runtime,
            rows=rows,
            batch_size=
                args.batch_size,
        )
    )

    raw_metrics = (
        compute_raw_metrics(
            predictions
        )
    )

    table = (
        build_risk_coverage_table(
            predictions=
                predictions,
            thresholds=
                thresholds,
            margin_threshold=0.05,
        )
    )

    historical = (
        load_historical_metrics(
            runtime
        )
    )

    consistency = (
        compare_with_historical(
            raw_metrics=
                raw_metrics,
            table=
                table,
            historical=
                historical,
        )
    )

    span_warnings = [
        {
            "sample_id":
                item.sample_id,
            "warning":
                item.span_warning,
        }
        for item in predictions
        if item.span_warning
        is not None
    ]

    plot_path = (
        output_dir
        / "semantic_risk_coverage.png"
    )

    plot_status = (
        maybe_plot(
            path=plot_path,
            table=table,
        )
    )

    report = {
        "dataset":
            str(
                dataset_path
            ),

        "n_samples":
            len(rows),

        "runtime_info":
            runtime.info(),

        "thresholds":
            thresholds,

        "margin_threshold":
            0.05,


        "raw_constrained":
            raw_metrics,

        "risk_coverage":
            table,

        "historical_test_metrics":
            historical,

        "historical_consistency":
            consistency,

        "production_safety_gate_note": (
            "The model-only columns reproduce selective classification "
            "before the later deterministic production safety gate. "
            "The production columns add the current runtime safety gate."
        ),

        "span_warning_count":
            len(
                span_warnings
            ),

        "span_warnings":
            span_warnings,

        "plot":
            plot_status,
    }

    json_path = (
        output_dir
        / "semantic_risk_coverage_metrics.json"
    )

    csv_path = (
        output_dir
        / "semantic_risk_coverage.csv"
    )

    predictions_path = (
        output_dir
        / "semantic_test_raw_predictions.jsonl"
    )

    write_json(
        json_path,
        report,
    )

    write_csv(
        csv_path,
        table,
    )

    write_jsonl(
        predictions_path,
        (
            item.to_dict()
            for item in predictions
        ),
    )

    print()
    print(
        "RAW CONSTRAINED METRICS"
    )

    print(
        json.dumps(
            raw_metrics,
            ensure_ascii=False,
            indent=2,
        )
    )

    print_table(
        table
    )

    print(
        "HISTORICAL CONSISTENCY"
    )

    print(
        json.dumps(
            consistency,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print(
        "OUTPUTS"
    )

    print(
        f"- {json_path}"
    )

    print(
        f"- {csv_path}"
    )

    print(
        f"- {predictions_path}"
    )

    if plot_status.get(
        "created"
    ):
        print(
            f"- {plot_path}"
        )

    else:
        print(
            "- PNG not created: "
            f"{plot_status.get('reason')}"
        )


if __name__ == "__main__":
    main()