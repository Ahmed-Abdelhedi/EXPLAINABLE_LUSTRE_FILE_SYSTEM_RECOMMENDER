from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from requirement_extractor_v2.conversation_scope_resolver import (
    ConversationScopeResolver,
)
from requirement_extractor_v2.models import (
    ParamName,
    Quantity,
    ScopeIntent,
)
from requirement_extractor_v2.semantic_linker.runtime import (
    SemanticLinkerRuntime,
)

# Use the robustness components added during Step 1.3.
# A fallback to the original components is kept only so the script fails
# gracefully on older checkouts; the console will clearly report which path
# is active.
try:
    from requirement_extractor_v2.robust_quantity_scanner import (
        RobustQuantityScanner as QuantityScannerForEvaluation,
    )
    ROBUST_SCANNER_ACTIVE = True
except ImportError:
    from requirement_extractor_v2.quantity_scanner import (
        QuantityScanner as QuantityScannerForEvaluation,
    )
    ROBUST_SCANNER_ACTIVE = False

try:
    from requirement_extractor_v2.robust_explicit_pattern_resolver import (
        RobustExplicitPatternResolver as ExplicitResolverForEvaluation,
    )
    ROBUST_EXPLICIT_ACTIVE = True
except ImportError:
    from requirement_extractor_v2.explicit_pattern_resolver import (
        ExplicitPatternResolver as ExplicitResolverForEvaluation,
    )
    ROBUST_EXPLICIT_ACTIVE = False


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def same_value(a: Any, b: Any, tol: float = 1e-6) -> bool:
    try:
        af = float(a)
        bf = float(b)
        return abs(af - bf) <= tol * max(
            1.0,
            abs(bf),
        )
    except (TypeError, ValueError):
        return a == b


def normalize_unit(unit: Optional[str]) -> Optional[str]:
    if unit is None:
        return None

    normalized = (
        str(unit)
        .strip()
        .casefold()
        .replace(" ", "")
    )

    aliases = {
        "w": "w",
        "watt": "w",
        "watts": "w",
        "kw": "kw",
        "mw": "mw",
        "usd": "usd",
        "$": "usd",
        "%": "%",
        "percent": "%",
        "pourcent": "%",
        "gb": "gb",
        "gib": "gib",
        "tb": "tb",
        "tib": "tib",
        "gb/s": "gb/s",
        "gbps": "gbps",
        "mb/s": "mb/s",
        "mbps": "mbps",
    }

    return aliases.get(
        normalized,
        normalized,
    )


def extract_context(case: Dict[str, Any]) -> Tuple[
    Optional[str],
    Optional[str],
    Optional[str],
]:
    """
    Support both dataset layouts used in this repository.

    300-message benchmark:
        case["context"]["previous_question"]

    96-message holdout:
        case["previous_question"]
    """
    context = case.get("context") or {}

    previous_question = (
        context.get("previous_question")
        if "previous_question" in context
        else case.get("previous_question")
    )

    previous_question_field = (
        context.get("previous_question_field")
        if "previous_question_field" in context
        else case.get("previous_question_field")
    )

    requested_unit = (
        context.get("requested_unit")
        if "requested_unit" in context
        else case.get("requested_unit")
    )

    return (
        previous_question,
        previous_question_field,
        requested_unit,
    )


def extract_gold(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "gold_quantities" in case:
        return list(
            case.get("gold_quantities")
            or []
        )

    return list(
        case.get("expected_outputs")
        or []
    )


def expected_outcome(case: Dict[str, Any]) -> str:
    if "expected_outcome" in case:
        return str(
            case["expected_outcome"]
        ).upper()

    safety = str(
        case.get("safety", "normal")
    ).casefold()

    if safety == "ambiguity":
        return "AMBIGUOUS"

    if safety == "out_of_scope":
        return "OUT_OF_SCOPE"

    return "VERIFIED"


def quantity_payload(quantity: Quantity) -> Dict[str, Any]:
    return {
        "id": quantity.id,
        "raw": quantity.raw,
        "normalized": quantity.normalized,
        "value": quantity.value,
        "unit": quantity.unit,
        "dimension": quantity.dimension.value,
        "start": int(quantity.start),
        "end": int(quantity.end),
        "corrected": bool(
            getattr(
                quantity,
                "corrected",
                False,
            )
        ),
    }


def gold_match_score(
    quantity: Quantity,
    gold_item: Dict[str, Any],
) -> Optional[Tuple[int, int]]:
    """
    Return a sortable score for matching one detected Quantity to one gold
    quantity.

    Priority:
      1. numerical value must match;
      2. exact normalized unit match is preferred;
      3. a gold item with a concrete field is preferred.

    The second element exists only to make the tuple deterministic.
    """
    if not same_value(
        quantity.value,
        gold_item.get("value"),
    ):
        return None

    q_unit = normalize_unit(
        quantity.unit
    )
    g_unit = normalize_unit(
        gold_item.get("unit")
    )

    unit_score = 0

    if q_unit == g_unit:
        unit_score = 2
    elif q_unit is None or g_unit is None:
        unit_score = 1

    field_score = int(
        gold_item.get("field")
        is not None
    )

    return (
        unit_score * 10
        + field_score,
        0,
    )


def find_gold_match(
    quantity: Quantity,
    gold_items: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    candidates: List[
        Tuple[
            Tuple[int, int],
            int,
            Dict[str, Any],
        ]
    ] = []

    for index, item in enumerate(
        gold_items
    ):
        score = gold_match_score(
            quantity,
            item,
        )

        if score is None:
            continue

        candidates.append(
            (
                score,
                -index,
                item,
            )
        )

    if not candidates:
        return None

    candidates.sort(
        reverse=True,
        key=lambda row: (
            row[0],
            row[1],
        ),
    )

    return candidates[0][2]


def build_fallback_label(
    case: Dict[str, Any],
    quantity: Quantity,
) -> Tuple[
    str,
    Optional[str],
    Optional[str],
    str,
]:
    """
    Label the actual fallback case.

    Safety rule:
    - AMBIGUOUS / UNRESOLVED / INVALID messages are ABSTAIN cases even if
      their individual quantities have known fields in the annotation.
      The LLM must never turn a globally unsafe message into an automatic
      acceptance.
    """
    outcome = expected_outcome(
        case
    )

    if outcome in {
        "AMBIGUOUS",
        "UNRESOLVED",
        "INVALID",
        "OUT_OF_SCOPE",
    }:
        return (
            "ABSTAIN",
            None,
            None,
            (
                "Global expected outcome is "
                f"{outcome}; fallback must not "
                "convert this into an automatic "
                "field/role resolution."
            ),
        )

    gold_items = extract_gold(
        case
    )

    match = find_gold_match(
        quantity,
        gold_items,
    )

    if match is None:
        return (
            "ABSTAIN",
            None,
            None,
            (
                "No gold quantity could be matched "
                "to this detected quantity."
            ),
        )

    field = match.get(
        "field"
    )
    role = match.get(
        "role"
    )

    if not field or not role:
        return (
            "ABSTAIN",
            None,
            None,
            (
                "Matched gold quantity has no "
                "concrete field/role."
            ),
        )

    return (
        "RESOLVE",
        str(field),
        str(role),
        (
            "Gold field/role exists and the "
            "message-level expected outcome is "
            "VERIFIED."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the REAL LLM-fallback subset: only quantities that are "
            "unresolved by the current robust Explicit Resolver and then "
            "rejected/abstained by the current Semantic Linker."
        )
    )

    parser.add_argument(
        "--dataset",
        default=(
            "requirement_extractor_v2/evaluation/datasets/"
            "v2_independent_end_to_end_benchmark_v1.jsonl"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "requirement_extractor_v2/evaluation/datasets/"
            "actual_fallback_subset_v1.jsonl"
        ),
    )

    parser.add_argument(
        "--metadata-output",
        default=(
            "requirement_extractor_v2/evaluation/results/"
            "actual_fallback_subset_v1_metadata.json"
        ),
    )

    args = parser.parse_args()

    dataset_path = Path(
        args.dataset
    )

    cases = load_jsonl(
        dataset_path
    )

    scope_resolver = (
        ConversationScopeResolver()
    )

    scanner = (
        QuantityScannerForEvaluation()
    )

    explicit_resolver = (
        ExplicitResolverForEvaluation()
    )

    semantic_linker = (
        SemanticLinkerRuntime()
    )

    fallback_rows: List[
        Dict[str, Any]
    ] = []

    scope_counts: Dict[str, int] = {}
    semantic_abstention_count = 0
    explicit_unresolved_count = 0
    scanned_quantity_count = 0
    skipped_contextual = 0
    skipped_out_of_scope = 0
    labeling_warnings: List[
        Dict[str, Any]
    ] = []

    print(
        "Building actual fallback subset"
    )

    print(
        "  robust scanner active       :",
        ROBUST_SCANNER_ACTIVE,
    )

    print(
        "  robust explicit active      :",
        ROBUST_EXPLICIT_ACTIVE,
    )

    print(
        "  source dataset              :",
        dataset_path,
    )

    print(
        "  source messages             :",
        len(cases),
    )

    for message_index, case in enumerate(
        cases,
        start=1,
    ):
        text = str(
            case.get(
                "text",
                "",
            )
        )

        (
            previous_question,
            previous_question_field_raw,
            requested_unit,
        ) = extract_context(
            case
        )

        previous_question_field = None

        if previous_question_field_raw:
            try:
                previous_question_field = (
                    ParamName(
                        previous_question_field_raw
                    )
                )
            except ValueError:
                labeling_warnings.append(
                    {
                        "id": case.get(
                            "id"
                        ),
                        "warning": (
                            "Unknown previous_question_field: "
                            f"{previous_question_field_raw!r}"
                        ),
                    }
                )

        scope = scope_resolver.resolve(
            user_text=text,
            previous_question_field=
                previous_question_field,
            requested_unit=
                requested_unit,
            previous_question=
                previous_question,
        )

        scope_name = (
            scope.intent.value
        )

        scope_counts[
            scope_name
        ] = (
            scope_counts.get(
                scope_name,
                0,
            )
            + 1
        )

        if (
            scope.intent
            == ScopeIntent.OUT_OF_SCOPE
        ):
            skipped_out_of_scope += 1
            continue

        if (
            scope.intent
            == ScopeIntent
            .ANSWER_TO_PREVIOUS_QUESTION
        ):
            # Production resolves this branch from conversation state
            # without Semantic Linker / LLM.
            skipped_contextual += 1
            continue

        quantities = scanner.scan(
            text
        )

        scanned_quantity_count += len(
            quantities
        )

        if not quantities:
            continue

        explicit = (
            explicit_resolver.resolve(
                text,
                quantities,
            )
        )

        quantity_by_id = {
            quantity.id: quantity
            for quantity
            in quantities
        }

        for quantity_id in (
            explicit
            .unresolved_quantity_ids
        ):
            explicit_unresolved_count += 1

            quantity = (
                quantity_by_id[
                    quantity_id
                ]
            )

            # In the production VerifiedRequirementPipeline, a rich
            # NEW_REQUIREMENT/CORRECTION intentionally does NOT inherit
            # stale previous-question context.
            semantic_prediction = (
                semantic_linker.predict(
                    text=text,
                    quantity=quantity,
                    previous_question=None,
                )
            )

            if (
                semantic_prediction.accepted
                and
                semantic_prediction.link
                is not None
            ):
                continue

            semantic_abstention_count += 1

            (
                expected,
                expected_field,
                expected_role,
                label_reason,
            ) = build_fallback_label(
                case=case,
                quantity=quantity,
            )

            row = {
                "id": (
                    f"{case.get('id', 'CASE')}"
                    f"__{quantity.id}"
                ),
                "source_message_id":
                    case.get("id"),
                "source_message_index":
                    message_index,
                "category":
                    case.get(
                        "category",
                        "unknown",
                    ),
                "language":
                    case.get(
                        "language",
                        "unknown",
                    ),
                "text": text,
                "previous_question": None,
                "quantity":
                    quantity_payload(
                        quantity
                    ),
                "expected": expected,
                "expected_field":
                    expected_field,
                "expected_role":
                    expected_role,
                "source_expected_outcome":
                    expected_outcome(
                        case
                    ),
                "label_reason":
                    label_reason,
                "notes":
                    case.get(
                        "notes",
                        "",
                    ),
                "semantic_debug": {
                    "accepted":
                        semantic_prediction
                        .accepted,
                    "confidence":
                        semantic_prediction
                        .confidence,
                    "margin":
                        semantic_prediction
                        .margin,
                    "raw_field":
                        semantic_prediction
                        .raw_field,
                    "raw_role":
                        semantic_prediction
                        .raw_role,
                },
            }

            fallback_rows.append(
                row
            )

    write_jsonl(
        Path(
            args.output
        ),
        fallback_rows,
    )

    resolve_count = sum(
        1
        for row in fallback_rows
        if row["expected"]
        == "RESOLVE"
    )

    abstain_count = (
        len(fallback_rows)
        - resolve_count
    )

    unique_source_messages = len(
        {
            row[
                "source_message_id"
            ]
            for row in fallback_rows
        }
    )

    metadata = {
        "source_dataset":
            str(dataset_path),
        "source_messages":
            len(cases),
        "robust_scanner_active":
            ROBUST_SCANNER_ACTIVE,
        "robust_explicit_active":
            ROBUST_EXPLICIT_ACTIVE,
        "scope_counts":
            scope_counts,
        "skipped_out_of_scope_messages":
            skipped_out_of_scope,
        "skipped_contextual_answer_messages":
            skipped_contextual,
        "scanned_quantity_count":
            scanned_quantity_count,
        "explicit_unresolved_quantity_count":
            explicit_unresolved_count,
        "semantic_abstention_quantity_count":
            semantic_abstention_count,
        "fallback_subset_quantity_cases":
            len(fallback_rows),
        "fallback_subset_source_messages":
            unique_source_messages,
        "expected_resolve_count":
            resolve_count,
        "expected_abstain_count":
            abstain_count,
        "fallback_rate_per_source_message":
            (
                unique_source_messages
                / len(cases)
                if cases
                else 0.0
            ),
        "labeling_warnings":
            labeling_warnings,
        "methodology": (
            "Actual fallback subset generated from the current execution "
            "path. OUT_OF_SCOPE turns and short contextual answers are "
            "excluded because they do not reach Semantic Linker/LLM in "
            "production. For remaining turns, only quantities unresolved "
            "by Explicit Resolver and then abstained/rejected by Semantic "
            "Linker are emitted."
        ),
    }

    metadata_path = Path(
        args.metadata_output
    )

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print(
        "ACTUAL FALLBACK SUBSET SUMMARY"
    )
    print("=" * 80)
    print(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        )
    )
    print()
    print(
        "Fallback dataset written to:",
        args.output,
    )
    print(
        "Metadata written to:",
        args.metadata_output,
    )


if __name__ == "__main__":
    main()
