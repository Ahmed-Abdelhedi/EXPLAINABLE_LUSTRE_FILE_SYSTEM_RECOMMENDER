from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


VALIDATION_DIR = Path(__file__).resolve().parent
DEFAULT_REPORTS_ROOT = VALIDATION_DIR / "reports"

SUPPORTED_MODES = (
    "deterministic",
    "llm_fallback",
    "hybrid",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse les écarts entre les sorties attendues et réelles du "
            "Requirement Extractor."
        )
    )

    parser.add_argument(
        "--mode",
        choices=SUPPORTED_MODES,
        default="deterministic",
        help="Mode à analyser.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Chemin facultatif du fichier results_<mode>.json. "
            "Par défaut : reports/<mode>/results_<mode>.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Chemin facultatif du rapport d'erreurs. "
            "Par défaut : reports/<mode>/errors_<mode>.json"
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Nombre maximal d'exemples conservés par groupe.",
    )

    return parser.parse_args()


def values_equal(expected: Any, actual: Any) -> bool:
    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        return math.isclose(
            float(expected),
            float(actual),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )

    if isinstance(expected, dict) and isinstance(actual, dict):
        return (
            set(expected.keys()) == set(actual.keys())
            and all(
                values_equal(expected[key], actual[key])
                for key in expected
            )
        )

    if isinstance(expected, list) and isinstance(actual, list):
        return (
            len(expected) == len(actual)
            and all(
                values_equal(expected_item, actual_item)
                for expected_item, actual_item in zip(expected, actual)
            )
        )

    return expected == actual


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Le fichier doit contenir un objet JSON.")

    if not isinstance(data.get("scenarios"), list):
        raise ValueError(
            "Le fichier doit contenir une liste sous la clé 'scenarios'."
        )

    return data


def compare_sets(
    expected_values: Iterable[str],
    actual_values: Iterable[str],
) -> Dict[str, List[str]]:
    expected_set = set(expected_values or [])
    actual_set = set(actual_values or [])

    return {
        "missing": sorted(expected_set - actual_set),
        "unexpected": sorted(actual_set - expected_set),
        "correct": sorted(expected_set & actual_set),
    }


def compare_calculation(
    expected: Dict[str, Any],
    actual: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Compare uniquement les clés déclarées dans le calcul attendu.
    """

    errors: List[Dict[str, Any]] = []

    for key, expected_value in expected.items():
        if key not in actual:
            errors.append(
                {
                    "type": "missing_calculation_key",
                    "key": key,
                    "expected": expected_value,
                    "actual": None,
                }
            )
            continue

        actual_value = actual[key]

        if not values_equal(expected_value, actual_value):
            errors.append(
                {
                    "type": "wrong_calculation_value",
                    "key": key,
                    "expected": expected_value,
                    "actual": actual_value,
                }
            )

    return errors


def append_limited(
    mapping: Dict[str, List[Dict[str, Any]]],
    key: str,
    value: Dict[str, Any],
    limit: int,
) -> None:
    bucket = mapping[key]

    if len(bucket) < limit:
        bucket.append(value)


def analyze(results: Dict[str, Any], top: int) -> Dict[str, Any]:
    required_fields = results.get(
        "dataset_metadata",
        {},
    ).get("required_fields", [])

    if not required_fields:
        raise ValueError(
            "dataset_metadata.required_fields est absent ou vide."
        )

    summary = Counter()
    by_field = Counter()
    by_category = Counter()
    by_error_type = Counter()
    by_language = Counter()
    by_difficulty = Counter()

    examples_by_field: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    examples_by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    examples_by_error_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    failed_turns: List[Dict[str, Any]] = []
    technical_errors: List[Dict[str, Any]] = []

    for scenario in results["scenarios"]:
        scenario_id = scenario.get("id")
        category = scenario.get("category") or "unknown"
        language = scenario.get("language") or "unknown"
        difficulty = scenario.get("difficulty") or "unknown"

        for turn in scenario.get("turns", []):
            summary["turns"] += 1

            turn_index = turn.get("turn_index")
            user_text = turn.get("user_text")
            expected = turn.get("expected") or {}
            actual = turn.get("actual")
            technical_error = turn.get("technical_error")

            base_context = {
                "scenario_id": scenario_id,
                "turn_index": turn_index,
                "category": category,
                "language": language,
                "difficulty": difficulty,
                "user_text": user_text,
            }

            if technical_error is not None or actual is None:
                summary["technical_errors"] += 1

                technical_errors.append(
                    {
                        **base_context,
                        "technical_error": technical_error,
                    }
                )
                continue

            turn_errors: List[Dict[str, Any]] = []

            expected_fields = expected.get("fields") or {}
            actual_fields = actual.get("fields") or {}

            for field in required_fields:
                expected_value = expected_fields.get(field)
                actual_value = actual_fields.get(field)

                if values_equal(expected_value, actual_value):
                    continue

                if expected_value is not None and actual_value is None:
                    error_type = "missing_expected_field"

                elif expected_value is None and actual_value is not None:
                    error_type = "unexpected_extracted_field"

                else:
                    error_type = "wrong_field_value"

                error = {
                    "scope": "field",
                    "type": error_type,
                    "field": field,
                    "expected": expected_value,
                    "actual": actual_value,
                }

                turn_errors.append(error)
                by_field[field] += 1
                by_error_type[error_type] += 1

                example = {
                    **base_context,
                    **error,
                }

                append_limited(
                    examples_by_field,
                    field,
                    example,
                    top,
                )
                append_limited(
                    examples_by_error_type,
                    error_type,
                    example,
                    top,
                )

            scalar_checks = (
                ("status", expected.get("status"), actual.get("status")),
                ("stage", expected.get("stage"), actual.get("stage")),
                (
                    "question_field",
                    expected.get("question_field"),
                    actual.get("question_field"),
                ),
            )

            for key, expected_value, actual_value in scalar_checks:
                if values_equal(expected_value, actual_value):
                    continue

                error_type = f"wrong_{key}"

                error = {
                    "scope": key,
                    "type": error_type,
                    "expected": expected_value,
                    "actual": actual_value,
                }

                turn_errors.append(error)
                by_error_type[error_type] += 1

                append_limited(
                    examples_by_error_type,
                    error_type,
                    {
                        **base_context,
                        **error,
                    },
                    top,
                )

            list_fields = (
                "missing_fields",
                "conflicting_fields",
                "invalid_fields",
                "unsupported_fields",
            )

            for key in list_fields:
                set_comparison = compare_sets(
                    expected.get(key) or [],
                    actual.get(key) or [],
                )

                if (
                    not set_comparison["missing"]
                    and not set_comparison["unexpected"]
                ):
                    continue

                error_type = f"wrong_{key}"

                error = {
                    "scope": key,
                    "type": error_type,
                    "expected": sorted(expected.get(key) or []),
                    "actual": sorted(actual.get(key) or []),
                    "missing_from_actual": set_comparison["missing"],
                    "unexpected_in_actual": set_comparison["unexpected"],
                }

                turn_errors.append(error)
                by_error_type[error_type] += 1

                append_limited(
                    examples_by_error_type,
                    error_type,
                    {
                        **base_context,
                        **error,
                    },
                    top,
                )

            must_not_extract = expected.get("must_not_extract") or []

            for field in must_not_extract:
                actual_value = actual_fields.get(field)

                if actual_value is None:
                    continue

                error_type = "must_not_extract_violation"

                error = {
                    "scope": "hallucination",
                    "type": error_type,
                    "field": field,
                    "expected": None,
                    "actual": actual_value,
                }

                turn_errors.append(error)
                by_field[field] += 1
                by_error_type[error_type] += 1

                append_limited(
                    examples_by_error_type,
                    error_type,
                    {
                        **base_context,
                        **error,
                    },
                    top,
                )

            expected_calculation = expected.get("calculation") or {}

            if expected_calculation:
                for calculation_error in compare_calculation(
                    expected_calculation,
                    actual.get("calculation") or {},
                ):
                    turn_errors.append(
                        {
                            "scope": "calculation",
                            **calculation_error,
                        }
                    )
                    by_error_type[calculation_error["type"]] += 1

                    append_limited(
                        examples_by_error_type,
                        calculation_error["type"],
                        {
                            **base_context,
                            "scope": "calculation",
                            **calculation_error,
                        },
                        top,
                    )

            if not turn_errors:
                summary["successful_turns"] += 1
                continue

            summary["failed_turns"] += 1
            summary["total_errors"] += len(turn_errors)
            by_category[category] += 1
            by_language[language] += 1
            by_difficulty[difficulty] += 1

            failed_turn = {
                **base_context,
                "error_count": len(turn_errors),
                "errors": turn_errors,
            }

            failed_turns.append(failed_turn)

            append_limited(
                examples_by_category,
                category,
                failed_turn,
                top,
            )

    sorted_failed_turns = sorted(
        failed_turns,
        key=lambda item: (
            -item["error_count"],
            item["scenario_id"] or "",
            item["turn_index"] or 0,
        ),
    )

    return {
        "summary": {
            "turn_count": summary["turns"],
            "successful_turns": summary["successful_turns"],
            "failed_turns": summary["failed_turns"],
            "technical_errors": summary["technical_errors"],
            "total_detected_errors": summary["total_errors"],
            "turn_success_rate": (
                summary["successful_turns"] / summary["turns"]
                if summary["turns"]
                else 0.0
            ),
        },
        "error_counts": {
            "by_type": dict(by_error_type.most_common()),
            "by_field": dict(by_field.most_common()),
            "by_category": dict(by_category.most_common()),
            "by_language": dict(by_language.most_common()),
            "by_difficulty": dict(by_difficulty.most_common()),
        },
        "priority_fields": [
            {
                "field": field,
                "error_count": count,
            }
            for field, count in by_field.most_common()
        ],
        "priority_error_types": [
            {
                "error_type": error_type,
                "error_count": count,
            }
            for error_type, count in by_error_type.most_common()
        ],
        "worst_turns": sorted_failed_turns[:top],
        "examples": {
            "by_field": dict(examples_by_field),
            "by_category": dict(examples_by_category),
            "by_error_type": dict(examples_by_error_type),
        },
        "technical_error_details": technical_errors[:top],
    }


def print_summary(report: Dict[str, Any], output_path: Path) -> None:
    summary = report["summary"]

    print("=" * 72)
    print("ANALYSE DES ERREURS")
    print("=" * 72)
    print(f"Tours analysés           : {summary['turn_count']}")
    print(f"Tours totalement corrects: {summary['successful_turns']}")
    print(f"Tours avec écarts        : {summary['failed_turns']}")
    print(f"Erreurs techniques       : {summary['technical_errors']}")
    print(f"Écarts détectés          : {summary['total_detected_errors']}")
    print(
        f"Taux de réussite          : "
        f"{summary['turn_success_rate'] * 100:.2f}%"
    )

    print("-" * 72)
    print("Champs prioritaires :")

    for item in report["priority_fields"][:10]:
        print(
            f"  {item['field']:<34} "
            f"{item['error_count']:>5} erreur(s)"
        )

    print("-" * 72)
    print("Types d'erreur prioritaires :")

    for item in report["priority_error_types"][:10]:
        print(
            f"  {item['error_type']:<34} "
            f"{item['error_count']:>5}"
        )

    print(f"Rapport produit           : {output_path}")


def main() -> int:
    args = parse_args()

    default_input = (
        DEFAULT_REPORTS_ROOT
        / args.mode
        / f"results_{args.mode}.json"
    )
    default_output = (
        DEFAULT_REPORTS_ROOT
        / args.mode
        / f"errors_{args.mode}.json"
    )

    input_path = (args.input or default_input).resolve()
    output_path = (args.output or default_output).resolve()

    try:
        results = load_json(input_path)
        report = analyze(results, top=args.top)

        report["run"] = {
            "mode": args.mode,
            "input_path": str(input_path),
            "output_path": str(output_path),
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                report,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print_summary(report, output_path)
        return 0

    except (
        FileNotFoundError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
