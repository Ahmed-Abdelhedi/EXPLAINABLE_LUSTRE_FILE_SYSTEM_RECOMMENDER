from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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
            "Calcule les métriques de validation du Requirement Extractor "
            "à partir d'un fichier results_<mode>.json."
        )
    )

    parser.add_argument(
        "--mode",
        choices=SUPPORTED_MODES,
        default="deterministic",
        help="Mode dont les résultats doivent être analysés.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Chemin facultatif du fichier de résultats. "
            "Par défaut : reports/<mode>/results_<mode>.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Chemin facultatif du fichier de métriques. "
            "Par défaut : reports/<mode>/metrics_<mode>.json"
        ),
    )

    return parser.parse_args()


def values_equal(expected: Any, actual: Any) -> bool:
    """
    Compare récursivement deux valeurs.

    Les valeurs numériques utilisent une petite tolérance pour éviter
    qu'une différence de représentation flottante soit comptée comme erreur.
    """

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


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def precision_recall_f1(
    true_positive: int,
    false_positive: int,
    false_negative: int,
) -> Dict[str, float]:
    precision = safe_divide(
        true_positive,
        true_positive + false_positive,
    )
    recall = safe_divide(
        true_positive,
        true_positive + false_negative,
    )

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compare_sets(
    expected_values: Iterable[str],
    actual_values: Iterable[str],
) -> Tuple[int, int, int, bool]:
    expected_set = set(expected_values or [])
    actual_set = set(actual_values or [])

    true_positive = len(expected_set & actual_set)
    false_positive = len(actual_set - expected_set)
    false_negative = len(expected_set - actual_set)

    return (
        true_positive,
        false_positive,
        false_negative,
        expected_set == actual_set,
    )


def percentile(values: List[float], probability: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return ordered[lower_index]

    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    fraction = position - lower_index

    return lower_value + (upper_value - lower_value) * fraction


def load_results(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier de résultats introuvable : {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Le fichier de résultats doit être un objet JSON.")

    if not isinstance(data.get("scenarios"), list):
        raise ValueError(
            "Le fichier doit contenir une liste sous la clé 'scenarios'."
        )

    required_fields = (
        data.get("dataset_metadata", {}).get("required_fields")
    )

    if not isinstance(required_fields, list) or not required_fields:
        raise ValueError(
            "dataset_metadata.required_fields est absent ou invalide."
        )

    return data


def calculation_matches(
    expected: Dict[str, Any],
    actual: Dict[str, Any],
) -> bool:
    """
    Compare uniquement les clés présentes dans le calcul attendu.
    """

    for key, expected_value in expected.items():
        if key not in actual:
            return False

        if not values_equal(expected_value, actual[key]):
            return False

    return True


def calculate_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    required_fields: List[str] = results[
        "dataset_metadata"
    ]["required_fields"]

    global_counts = Counter()
    field_counts: Dict[str, Counter] = {
        field: Counter()
        for field in required_fields
    }
    category_counts: Dict[str, Counter] = defaultdict(Counter)
    language_counts: Dict[str, Counter] = defaultdict(Counter)
    difficulty_counts: Dict[str, Counter] = defaultdict(Counter)

    latencies: List[float] = []

    missing_detection = Counter()
    conflict_detection = Counter()
    invalid_detection = Counter()
    unsupported_detection = Counter()

    technical_errors = 0
    scenario_count = 0
    turn_count = 0
    scenario_full_success_count = 0

    for scenario in results["scenarios"]:
        scenario_count += 1
        category = scenario.get("category") or "unknown"
        language = scenario.get("language") or "unknown"
        difficulty = scenario.get("difficulty") or "unknown"

        scenario_all_turns_match = True

        for turn in scenario.get("turns", []):
            turn_count += 1
            category_counts[category]["turns"] += 1
            language_counts[language]["turns"] += 1
            difficulty_counts[difficulty]["turns"] += 1

            latency_ms = float(turn.get("latency_ms", 0.0))
            latencies.append(latency_ms)

            expected = turn.get("expected") or {}
            actual = turn.get("actual")
            technical_error = turn.get("technical_error")

            if technical_error is not None or actual is None:
                technical_errors += 1
                global_counts["technical_errors"] += 1
                category_counts[category]["technical_errors"] += 1
                language_counts[language]["technical_errors"] += 1
                difficulty_counts[difficulty]["technical_errors"] += 1
                scenario_all_turns_match = False
                continue

            expected_fields = expected.get("fields") or {}
            actual_fields = actual.get("fields") or {}

            all_fields_match = True

            for field in required_fields:
                expected_value = expected_fields.get(field)
                actual_value = actual_fields.get(field)

                expected_present = expected_value is not None
                actual_present = actual_value is not None
                value_match = values_equal(expected_value, actual_value)

                global_counts["field_slots"] += 1
                field_counts[field]["field_slots"] += 1
                category_counts[category]["field_slots"] += 1
                language_counts[language]["field_slots"] += 1
                difficulty_counts[difficulty]["field_slots"] += 1

                if value_match:
                    global_counts["field_slots_correct"] += 1
                    field_counts[field]["field_slots_correct"] += 1
                    category_counts[category]["field_slots_correct"] += 1
                    language_counts[language]["field_slots_correct"] += 1
                    difficulty_counts[difficulty]["field_slots_correct"] += 1
                else:
                    all_fields_match = False

                if expected_present:
                    global_counts["expected_non_null_fields"] += 1
                    field_counts[field]["expected_non_null_fields"] += 1
                    category_counts[category]["expected_non_null_fields"] += 1
                    language_counts[language]["expected_non_null_fields"] += 1
                    difficulty_counts[difficulty][
                        "expected_non_null_fields"
                    ] += 1

                    if actual_present and values_equal(
                        expected_value,
                        actual_value,
                    ):
                        global_counts["correct_expected_field_values"] += 1
                        field_counts[field][
                            "correct_expected_field_values"
                        ] += 1
                        category_counts[category][
                            "correct_expected_field_values"
                        ] += 1
                        language_counts[language][
                            "correct_expected_field_values"
                        ] += 1
                        difficulty_counts[difficulty][
                            "correct_expected_field_values"
                        ] += 1

                if expected_present and actual_present:
                    global_counts["field_presence_tp"] += 1
                    field_counts[field]["field_presence_tp"] += 1

                elif not expected_present and actual_present:
                    global_counts["field_presence_fp"] += 1
                    field_counts[field]["field_presence_fp"] += 1
                    field_counts[field]["hallucinated"] += 1

                elif expected_present and not actual_present:
                    global_counts["field_presence_fn"] += 1
                    field_counts[field]["field_presence_fn"] += 1
                    field_counts[field]["missing_when_expected"] += 1

            if all_fields_match:
                global_counts["field_exact_turns"] += 1
                category_counts[category]["field_exact_turns"] += 1
                language_counts[language]["field_exact_turns"] += 1
                difficulty_counts[difficulty]["field_exact_turns"] += 1

            status_match = expected.get("status") == actual.get("status")
            stage_match = expected.get("stage") == actual.get("stage")

            global_counts["status_correct"] += int(status_match)
            global_counts["stage_correct"] += int(stage_match)

            category_counts[category]["status_correct"] += int(status_match)
            category_counts[category]["stage_correct"] += int(stage_match)
            language_counts[language]["status_correct"] += int(status_match)
            language_counts[language]["stage_correct"] += int(stage_match)
            difficulty_counts[difficulty]["status_correct"] += int(
                status_match
            )
            difficulty_counts[difficulty]["stage_correct"] += int(
                stage_match
            )

            expected_question = expected.get("question_field")

            if expected_question is not None:
                global_counts["clarification_expected"] += 1
                category_counts[category]["clarification_expected"] += 1

                question_match = (
                    expected_question == actual.get("question_field")
                )

                global_counts["clarification_correct"] += int(
                    question_match
                )
                category_counts[category]["clarification_correct"] += int(
                    question_match
                )

            must_not_extract = expected.get("must_not_extract") or []

            for field in must_not_extract:
                global_counts["hallucination_opportunities"] += 1

                if actual_fields.get(field) is not None:
                    global_counts["hallucinations"] += 1

            set_metrics = (
                (
                    "missing_fields",
                    missing_detection,
                ),
                (
                    "conflicting_fields",
                    conflict_detection,
                ),
                (
                    "invalid_fields",
                    invalid_detection,
                ),
                (
                    "unsupported_fields",
                    unsupported_detection,
                ),
            )

            set_exact_matches: Dict[str, bool] = {}

            for key, counter in set_metrics:
                tp, fp, fn, exact_match = compare_sets(
                    expected.get(key) or [],
                    actual.get(key) or [],
                )

                counter["tp"] += tp
                counter["fp"] += fp
                counter["fn"] += fn
                counter["exact"] += int(exact_match)
                counter["turns"] += 1
                set_exact_matches[key] = exact_match

            expected_calculation = expected.get("calculation") or {}
            calculation_match = True

            if expected_calculation:
                global_counts["calculation_expected"] += 1
                calculation_match = calculation_matches(
                    expected_calculation,
                    actual.get("calculation") or {},
                )
                global_counts["calculation_correct"] += int(
                    calculation_match
                )

            full_turn_match = (
                all_fields_match
                and status_match
                and stage_match
                and all(set_exact_matches.values())
                and (
                    expected_question is None
                    or expected_question == actual.get("question_field")
                )
                and calculation_match
            )

            global_counts["full_turns"] += 1
            category_counts[category]["full_turns"] += 1
            language_counts[language]["full_turns"] += 1
            difficulty_counts[difficulty]["full_turns"] += 1

            if full_turn_match:
                global_counts["full_turns_correct"] += 1
                category_counts[category]["full_turns_correct"] += 1
                language_counts[language]["full_turns_correct"] += 1
                difficulty_counts[difficulty]["full_turns_correct"] += 1
            else:
                scenario_all_turns_match = False

        if scenario_all_turns_match:
            scenario_full_success_count += 1

    field_presence = precision_recall_f1(
        global_counts["field_presence_tp"],
        global_counts["field_presence_fp"],
        global_counts["field_presence_fn"],
    )

    def set_detection_result(counter: Counter) -> Dict[str, Any]:
        result = precision_recall_f1(
            counter["tp"],
            counter["fp"],
            counter["fn"],
        )

        result.update(
            {
                "true_positive": counter["tp"],
                "false_positive": counter["fp"],
                "false_negative": counter["fn"],
                "exact_set_accuracy": safe_divide(
                    counter["exact"],
                    counter["turns"],
                ),
            }
        )

        return result

    per_field: Dict[str, Any] = {}

    for field, counter in field_counts.items():
        presence = precision_recall_f1(
            counter["field_presence_tp"],
            counter["field_presence_fp"],
            counter["field_presence_fn"],
        )

        per_field[field] = {
            "presence_precision": presence["precision"],
            "presence_recall": presence["recall"],
            "presence_f1": presence["f1"],
            "field_slot_accuracy": safe_divide(
                counter["field_slots_correct"],
                counter["field_slots"],
            ),
            "value_accuracy_when_expected": safe_divide(
                counter["correct_expected_field_values"],
                counter["expected_non_null_fields"],
            ),
            "expected_non_null_count": counter[
                "expected_non_null_fields"
            ],
            "missing_when_expected": counter["missing_when_expected"],
            "hallucinated_when_not_expected": counter["hallucinated"],
        }

    def grouped_metrics(
        groups: Dict[str, Counter],
    ) -> Dict[str, Any]:
        output: Dict[str, Any] = {}

        for group_name, counter in sorted(groups.items()):
            output[group_name] = {
                "turn_count": counter["turns"],
                "field_slot_accuracy": safe_divide(
                    counter["field_slots_correct"],
                    counter["field_slots"],
                ),
                "field_value_accuracy": safe_divide(
                    counter["correct_expected_field_values"],
                    counter["expected_non_null_fields"],
                ),
                "field_exact_match_rate": safe_divide(
                    counter["field_exact_turns"],
                    counter["turns"],
                ),
                "status_accuracy": safe_divide(
                    counter["status_correct"],
                    counter["turns"],
                ),
                "stage_accuracy": safe_divide(
                    counter["stage_correct"],
                    counter["turns"],
                ),
                "full_turn_exact_match_rate": safe_divide(
                    counter["full_turns_correct"],
                    counter["full_turns"],
                ),
                "technical_errors": counter["technical_errors"],
            }

        return output

    normalization_counter = category_counts.get(
        "unit_normalization",
        Counter(),
    )
    multi_turn_counter = category_counts.get(
        "multi_turn",
        Counter(),
    )

    hallucination_rate = safe_divide(
        global_counts["hallucinations"],
        global_counts["hallucination_opportunities"],
    )

    metrics = {
        "summary": {
            "scenario_count": scenario_count,
            "turn_count": turn_count,
            "technical_errors": technical_errors,
            "scenario_full_success_rate": safe_divide(
                scenario_full_success_count,
                scenario_count,
            ),
            "full_turn_exact_match_rate": safe_divide(
                global_counts["full_turns_correct"],
                global_counts["full_turns"],
            ),
            "field_exact_match_rate": safe_divide(
                global_counts["field_exact_turns"],
                turn_count,
            ),
            "field_slot_accuracy": safe_divide(
                global_counts["field_slots_correct"],
                global_counts["field_slots"],
            ),
            "field_precision": field_presence["precision"],
            "field_recall": field_presence["recall"],
            "field_f1": field_presence["f1"],
            "field_value_accuracy": safe_divide(
                global_counts["correct_expected_field_values"],
                global_counts["expected_non_null_fields"],
            ),
            "status_accuracy": safe_divide(
                global_counts["status_correct"],
                turn_count,
            ),
            "stage_accuracy": safe_divide(
                global_counts["stage_correct"],
                turn_count,
            ),
            "clarification_target_accuracy": safe_divide(
                global_counts["clarification_correct"],
                global_counts["clarification_expected"],
            ),
            "hallucination_rate": hallucination_rate,
            "normalization_accuracy": safe_divide(
                normalization_counter["correct_expected_field_values"],
                normalization_counter["expected_non_null_fields"],
            ),
            "multi_turn_consistency": safe_divide(
                multi_turn_counter["correct_expected_field_values"],
                multi_turn_counter["expected_non_null_fields"],
            ),
            "calculation_accuracy": safe_divide(
                global_counts["calculation_correct"],
                global_counts["calculation_expected"],
            ),
            "average_latency_ms": (
                statistics.mean(latencies)
                if latencies
                else 0.0
            ),
            "median_latency_ms": (
                statistics.median(latencies)
                if latencies
                else 0.0
            ),
            "p95_latency_ms": percentile(latencies, 0.95),
            "maximum_latency_ms": max(latencies) if latencies else 0.0,
        },
        "detection_metrics": {
            "missing_fields": set_detection_result(missing_detection),
            "conflicting_fields": set_detection_result(
                conflict_detection
            ),
            "invalid_fields": set_detection_result(invalid_detection),
            "unsupported_fields": set_detection_result(
                unsupported_detection
            ),
        },
        "per_field": per_field,
        "per_category": grouped_metrics(category_counts),
        "per_language": grouped_metrics(language_counts),
        "per_difficulty": grouped_metrics(difficulty_counts),
        "counts": {
            "field_presence": {
                "true_positive": global_counts["field_presence_tp"],
                "false_positive": global_counts["field_presence_fp"],
                "false_negative": global_counts["field_presence_fn"],
            },
            "hallucinations": global_counts["hallucinations"],
            "hallucination_opportunities": global_counts[
                "hallucination_opportunities"
            ],
            "clarification_expected": global_counts[
                "clarification_expected"
            ],
            "clarification_correct": global_counts[
                "clarification_correct"
            ],
            "calculation_expected": global_counts[
                "calculation_expected"
            ],
            "calculation_correct": global_counts[
                "calculation_correct"
            ],
        },
        "metric_definitions": {
            "field_precision": (
                "Part des champs produits qui étaient attendus."
            ),
            "field_recall": (
                "Part des champs attendus effectivement produits."
            ),
            "field_f1": (
                "Moyenne harmonique de la précision et du rappel des champs."
            ),
            "field_value_accuracy": (
                "Part des valeurs attendues non nulles extraites avec "
                "la valeur exacte après normalisation."
            ),
            "field_slot_accuracy": (
                "Exactitude sur les 13 positions de champ, y compris "
                "les valeurs nulles attendues."
            ),
            "field_exact_match_rate": (
                "Part des tours où les 13 champs correspondent exactement."
            ),
            "full_turn_exact_match_rate": (
                "Part des tours où champs, statut, étape, listes de validation, "
                "clarification et calcul attendu correspondent."
            ),
            "scenario_full_success_rate": (
                "Part des scénarios dont tous les tours correspondent "
                "complètement."
            ),
            "normalization_accuracy": (
                "Exactitude des valeurs attendues pour la catégorie "
                "unit_normalization."
            ),
            "multi_turn_consistency": (
                "Exactitude cumulative des valeurs attendues dans les "
                "scénarios multi_turn."
            ),
            "hallucination_rate": (
                "Part des champs must_not_extract qui ont malgré tout "
                "été renseignés."
            ),
        },
    }

    return metrics


def print_summary(metrics: Dict[str, Any]) -> None:
    summary = metrics["summary"]

    print("=" * 72)
    print("MÉTRIQUES DE VALIDATION")
    print("=" * 72)
    print(f"Scénarios                         : {summary['scenario_count']}")
    print(f"Tours                             : {summary['turn_count']}")
    print(f"Erreurs techniques                : {summary['technical_errors']}")
    print(
        "Field F1                         : "
        f"{summary['field_f1'] * 100:.2f}%"
    )
    print(
        "Field value accuracy             : "
        f"{summary['field_value_accuracy'] * 100:.2f}%"
    )
    print(
        "Exact match des 13 champs        : "
        f"{summary['field_exact_match_rate'] * 100:.2f}%"
    )
    print(
        "Exact match complet par tour     : "
        f"{summary['full_turn_exact_match_rate'] * 100:.2f}%"
    )
    print(
        "Status accuracy                  : "
        f"{summary['status_accuracy'] * 100:.2f}%"
    )
    print(
        "Clarification target accuracy    : "
        f"{summary['clarification_target_accuracy'] * 100:.2f}%"
    )
    print(
        "Normalization accuracy           : "
        f"{summary['normalization_accuracy'] * 100:.2f}%"
    )
    print(
        "Multi-turn consistency           : "
        f"{summary['multi_turn_consistency'] * 100:.2f}%"
    )
    print(
        "Hallucination rate               : "
        f"{summary['hallucination_rate'] * 100:.2f}%"
    )
    print(
        "Latence moyenne                  : "
        f"{summary['average_latency_ms']:.3f} ms"
    )


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
        / f"metrics_{args.mode}.json"
    )

    input_path = (args.input or default_input).resolve()
    output_path = (args.output or default_output).resolve()

    try:
        results = load_results(input_path)
        metrics = calculate_metrics(results)

        metrics["run"] = {
            "mode": args.mode,
            "input_path": str(input_path),
            "output_path": str(output_path),
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                metrics,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print_summary(metrics)
        print(f"Fichier produit                   : {output_path}")

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
