from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List


# ============================================================
# PATH CONFIGURATION
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

DEFAULT_DATASET_PATH = CURRENT_DIR / "stress_dataset.json"
DEFAULT_RESULTS_DIR = CURRENT_DIR / "results"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run stress tests for the Lustre Requirement Chatbot v2."
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM fallback during tests.",
    )

    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="Ollama host URL.",
    )

    parser.add_argument(
        "--ollama-model",
        default="qwen2.5-coder:7b",
        help="Ollama model name.",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="Path to the stress dataset JSON file.",
    )

    parser.add_argument(
        "--results",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory where result files will be written.",
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop execution after the first failed turn.",
    )

    return parser.parse_args()


ARGS = parse_args()

DATASET_PATH = Path(ARGS.dataset)
RESULTS_DIR = Path(ARGS.results)

# Allow importing files from project root:
# requirement_chatbot.py, models.py, etc.
sys.path.insert(0, str(PROJECT_ROOT))

# Disable LLM fallback during deterministic tests
if ARGS.llm:
    os.environ["ENABLE_LLM_FALLBACK"] = "true"
    os.environ["OLLAMA_HOST"] = ARGS.ollama_host
    os.environ["OLLAMA_MODEL"] = ARGS.ollama_model
else:
    os.environ["ENABLE_LLM_FALLBACK"] = "false"

from requirement_chatbot import RequirementChatbot  # noqa: E402


# ============================================================
# UTILS
# ============================================================

def enum_value(value):
    """
    Convert Enum values to plain strings.
    """

    if value is None:
        return None

    if hasattr(value, "value"):
        return value.value

    return str(value)


def to_plain_json(final_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts state.final_json to a normal Python dict.

    In the chatbot, values can be:
    - None
    - FinalFieldValue(value=...)
    - direct primitive values
    """

    plain = {}

    for key, value in final_json.items():
        if value is None:
            plain[key] = None
        elif hasattr(value, "value"):
            plain[key] = value.value
        else:
            plain[key] = value

    return plain


def get_path(data: Dict[str, Any], path: str):
    """
    Reads nested values using dot notation.

    Example:
        read_write_ratio.read_percent
    """

    current = data

    for part in path.split("."):
        if current is None:
            return None

        if not isinstance(current, dict):
            return None

        current = current.get(part)

    return current


def values_equal(actual, expected) -> bool:
    """
    Robust comparison for numbers and normal values.
    """

    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected)) < 1e-6

    return actual == expected


def check_expected_fields(
    actual_json: Dict[str, Any],
    expected_fields: Dict[str, Any],
) -> List[str]:
    """
    Checks fields that must have exact expected values.
    """

    errors = []

    for path, expected_value in expected_fields.items():
        actual_value = get_path(actual_json, path)

        if not values_equal(actual_value, expected_value):
            errors.append(
                f"{path}: expected={expected_value!r}, actual={actual_value!r}"
            )

    return errors


def check_none_fields(
    actual_json: Dict[str, Any],
    none_fields: List[str],
) -> List[str]:
    """
    Checks fields that must remain None.
    Useful for anti-hallucination tests.
    """

    errors = []

    for path in none_fields:
        actual_value = get_path(actual_json, path)

        if actual_value is not None:
            errors.append(
                f"{path}: expected=None, actual={actual_value!r}"
            )

    return errors


def check_not_expected_fields(
    actual_json: Dict[str, Any],
    not_expected_fields: Dict[str, Any],
) -> List[str]:
    """
    Checks fields that must NOT take a forbidden value.

    Example:
        client_count must not become 3
        when the user says "3 salles machines".
    """

    errors = []

    for path, forbidden_value in not_expected_fields.items():
        actual_value = get_path(actual_json, path)

        if values_equal(actual_value, forbidden_value):
            errors.append(
                f"{path}: forbidden={forbidden_value!r}, actual={actual_value!r}"
            )

    return errors


def check_question_contains(
    questions: List[str],
    expected_parts: List[str],
) -> List[str]:
    """
    Checks if the chatbot question contains useful explanation terms.

    Example:
        ratio 70/40 should produce a question containing:
        70, 40, 110, 100
    """

    errors = []

    joined_questions = " ".join(questions).lower()

    for part in expected_parts:
        if part.lower() not in joined_questions:
            errors.append(
                f"question should contain {part!r}, actual={joined_questions!r}"
            )

    return errors


def load_dataset() -> List[Dict[str, Any]]:
    """
    Loads the JSON dataset.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError("Dataset must be a list of scenarios.")

    return dataset


# ============================================================
# EVALUATION LOGIC
# ============================================================

def evaluate_turn(
    scenario_name: str,
    turn_index: int,
    turn: Dict[str, Any],
    bot: RequirementChatbot,
) -> Dict[str, Any]:
    """
    Runs one user turn and compares the chatbot output with expectations.
    """

    user_text = turn["user"]

    state = bot.process_user_message(user_text)

    actual_status = enum_value(state.status)
    actual_stage = enum_value(state.stage)
    actual_json = to_plain_json(state.final_json)
    actual_questions = deepcopy(state.questions)

    errors = []

    expected_status = turn.get("expect_status")
    if expected_status is not None and actual_status != expected_status:
        errors.append(
            f"status: expected={expected_status!r}, actual={actual_status!r}"
        )

    expected_stage = turn.get("expect_stage")
    if expected_stage is not None and actual_stage != expected_stage:
        errors.append(
            f"stage: expected={expected_stage!r}, actual={actual_stage!r}"
        )

    errors.extend(
        check_expected_fields(
            actual_json=actual_json,
            expected_fields=turn.get("expect", {}),
        )
    )

    errors.extend(
        check_none_fields(
            actual_json=actual_json,
            none_fields=turn.get("expect_none", []),
        )
    )

    errors.extend(
        check_not_expected_fields(
            actual_json=actual_json,
            not_expected_fields=turn.get("not_expect", {}),
        )
    )

    errors.extend(
        check_question_contains(
            questions=actual_questions,
            expected_parts=turn.get("question_contains", []),
        )
    )

    passed = len(errors) == 0

    checked_fields = (
        len(turn.get("expect", {}))
        + len(turn.get("expect_none", []))
        + len(turn.get("not_expect", {}))
    )

    return {
        "scenario": scenario_name,
        "turn_index": turn_index,
        "passed": passed,
        "user": user_text,
        "errors": errors,
        "expected_status": expected_status,
        "actual_status": actual_status,
        "expected_stage": expected_stage,
        "actual_stage": actual_stage,
        "actual_json": actual_json,
        "actual_questions": actual_questions,
        "checked_fields": checked_fields,
        "error_count": len(errors),
    }


def run_stress_tests() -> None:
    """
    Runs all scenarios from the dataset and writes result files.
    """

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset()

    all_turn_results = []
    failed_cases = []
    scenario_summary_rows = []

    total_scenarios = len(dataset)
    passed_scenarios = 0

    total_turns = 0
    passed_turns = 0

    total_checked_fields = 0
    total_errors = 0

    print("\nLUSTRE CHATBOT - STRESS TESTS")
    print("=" * 80)
    print(f"Dataset : {DATASET_PATH}")
    print(f"Results : {RESULTS_DIR}")
    print("=" * 80)

    for scenario in dataset:
        scenario_name = scenario.get("name", "unnamed_scenario")
        turns = scenario.get("turns", [])

        bot = RequirementChatbot()

        scenario_passed = True
        scenario_turn_results = []

        print(f"\nSCENARIO: {scenario_name}")

        for turn_index, turn in enumerate(turns, start=1):
            total_turns += 1

            result = evaluate_turn(
                scenario_name=scenario_name,
                turn_index=turn_index,
                turn=turn,
                bot=bot,
            )

            all_turn_results.append(result)
            scenario_turn_results.append(result)

            total_checked_fields += result["checked_fields"]
            total_errors += result["error_count"]

            if result["passed"]:
                passed_turns += 1
                print(f"  Turn {turn_index}: PASS ✅")
            else:
                scenario_passed = False
                failed_cases.append(result)

                print(f"  Turn {turn_index}: FAIL ❌")
                print(f"    User: {result['user']}")

                for error in result["errors"]:
                    print(f"    - {error}")

                if ARGS.fail_fast:
                    write_results(
                        all_turn_results=all_turn_results,
                        failed_cases=failed_cases,
                        scenario_summary_rows=scenario_summary_rows,
                        total_scenarios=total_scenarios,
                        passed_scenarios=passed_scenarios,
                        total_turns=total_turns,
                        passed_turns=passed_turns,
                        total_checked_fields=total_checked_fields,
                        total_errors=total_errors,
                    )
                    print("\nStopped because --fail-fast is enabled.")
                    return

        if scenario_passed:
            passed_scenarios += 1

        scenario_summary_rows.append(
            {
                "scenario": scenario_name,
                "passed": scenario_passed,
                "turns": len(turns),
                "failed_turns": sum(
                    1 for item in scenario_turn_results if not item["passed"]
                ),
                "errors": sum(
                    item["error_count"] for item in scenario_turn_results
                ),
            }
        )

    write_results(
        all_turn_results=all_turn_results,
        failed_cases=failed_cases,
        scenario_summary_rows=scenario_summary_rows,
        total_scenarios=total_scenarios,
        passed_scenarios=passed_scenarios,
        total_turns=total_turns,
        passed_turns=passed_turns,
        total_checked_fields=total_checked_fields,
        total_errors=total_errors,
    )


# ============================================================
# RESULTS WRITING
# ============================================================

def write_results(
    all_turn_results: List[Dict[str, Any]],
    failed_cases: List[Dict[str, Any]],
    scenario_summary_rows: List[Dict[str, Any]],
    total_scenarios: int,
    passed_scenarios: int,
    total_turns: int,
    passed_turns: int,
    total_checked_fields: int,
    total_errors: int,
) -> None:
    """
    Writes JSON and CSV result files.
    """

    turn_pass_rate = passed_turns / total_turns if total_turns else 0.0
    scenario_pass_rate = (
        passed_scenarios / total_scenarios
        if total_scenarios
        else 0.0
    )
    field_error_rate = (
        total_errors / total_checked_fields
        if total_checked_fields
        else 0.0
    )

    final_report = {
        "dataset": str(DATASET_PATH),
        "results_dir": str(RESULTS_DIR),
        "summary": {
            "total_scenarios": total_scenarios,
            "passed_scenarios": passed_scenarios,
            "scenario_pass_rate": scenario_pass_rate,
            "total_turns": total_turns,
            "passed_turns": passed_turns,
            "turn_pass_rate": turn_pass_rate,
            "total_checked_fields": total_checked_fields,
            "total_errors": total_errors,
            "field_error_rate": field_error_rate,
        },
        "scenario_summary": scenario_summary_rows,
        "turn_results": all_turn_results,
    }

    results_json_path = RESULTS_DIR / "stress_results.json"
    failed_json_path = RESULTS_DIR / "failed_cases.json"
    summary_csv_path = RESULTS_DIR / "stress_summary.csv"

    with open(results_json_path, "w", encoding="utf-8") as file:
        json.dump(final_report, file, indent=2, ensure_ascii=False)

    with open(failed_json_path, "w", encoding="utf-8") as file:
        json.dump(failed_cases, file, indent=2, ensure_ascii=False)

    with open(summary_csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "scenario",
                "passed",
                "turns",
                "failed_turns",
                "errors",
            ],
        )

        writer.writeheader()
        writer.writerows(scenario_summary_rows)

    print("\n" + "=" * 80)
    print("GLOBAL SUMMARY")
    print("=" * 80)
    print(f"Scenarios passed : {passed_scenarios}/{total_scenarios}")
    print(f"Turns passed     : {passed_turns}/{total_turns}")
    print(f"Turn pass rate   : {turn_pass_rate:.2%}")
    print(f"Field error rate : {field_error_rate:.2%}")

    print("\nFiles generated:")
    print(f"- {results_json_path}")
    print(f"- {failed_json_path}")
    print(f"- {summary_csv_path}")

    if failed_cases:
        print("\nSome tests failed ❌")
        print("Open failed_cases.json to inspect the errors.")
    else:
        print("\nAll stress tests passed ✅")


if __name__ == "__main__":
    run_stress_tests()