from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
RESULTS_DIR = THIS_DIR / "results"


def configure_environment(args) -> None:
    os.environ["ENABLE_LLM_FALLBACK"] = "true" if args.llm else "false"
    os.environ["ENABLE_AI_PLAUSIBILITY_AGENT"] = "true" if args.ai else "false"

    os.environ["OLLAMA_HOST"] = args.ollama_host
    os.environ["OLLAMA_MODEL"] = args.ollama_model

    os.environ["PLAUSIBILITY_AGENT_MODEL"] = args.ai_model
    os.environ["PLAUSIBILITY_AGENT_TEMPERATURE"] = str(args.ai_temperature)
    os.environ["PLAUSIBILITY_AGENT_DEBUG"] = "true" if args.ai_debug else "false"

    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except Exception:
        pass

    sys.path.insert(0, str(PROJECT_ROOT))


def enum_value(value: Any) -> str:
    return getattr(value, "value", str(value))


def plain_final_json(state) -> Dict[str, Any]:
    out = {}

    for key, item in state.final_json.items():
        if item is None:
            out[key] = None
        else:
            out[key] = item.value

    return out


def compare_values(actual: Any, expected: Any, tolerance: float = 1e-6) -> bool:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False

        for key, expected_value in expected.items():
            if key not in actual:
                return False

            if not compare_values(actual[key], expected_value, tolerance):
                return False

        return True

    if isinstance(expected, bool):
        return actual is expected

    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance)

    return actual == expected


def check_expectation(
    state,
    stdout_text: str,
    expect: Dict[str, Any],
) -> Tuple[bool, List[Dict[str, Any]]]:
    checks = []
    final_json = plain_final_json(state)

    def add_check(name: str, passed: bool, expected=None, actual=None) -> None:
        checks.append(
            {
                "name": name,
                "passed": passed,
                "expected": expected,
                "actual": actual,
            }
        )

    if "status" in expect:
        actual_status = enum_value(state.status)
        add_check(
            name="status",
            passed=actual_status == expect["status"],
            expected=expect["status"],
            actual=actual_status,
        )

    if "stage" in expect:
        actual_stage = enum_value(state.stage)
        add_check(
            name="stage",
            passed=actual_stage == expect["stage"],
            expected=expect["stage"],
            actual=actual_stage,
        )

    for field_name, expected_value in expect.get("fields", {}).items():
        actual_value = final_json.get(field_name)
        add_check(
            name=f"field:{field_name}",
            passed=compare_values(actual_value, expected_value),
            expected=expected_value,
            actual=actual_value,
        )

    for field_name in expect.get("null_fields", []):
        actual_value = final_json.get(field_name)
        add_check(
            name=f"null_field:{field_name}",
            passed=actual_value is None,
            expected=None,
            actual=actual_value,
        )

    if "question_contains_any" in expect:
        questions_text = " ".join(state.questions or []).lower()
        expected_terms = [
            term.lower()
            for term in expect["question_contains_any"]
        ]

        passed = any(term in questions_text for term in expected_terms)

        add_check(
            name="question_contains_any",
            passed=passed,
            expected=expect["question_contains_any"],
            actual=state.questions,
        )

    if "stdout_contains_any" in expect:
        stdout_lower = stdout_text.lower()
        expected_terms = [
            term.lower()
            for term in expect["stdout_contains_any"]
        ]

        passed = any(term in stdout_lower for term in expected_terms)

        add_check(
            name="stdout_contains_any",
            passed=passed,
            expected=expect["stdout_contains_any"],
            actual=stdout_text[-1000:],
        )

    if "calculation" in expect:
        calc = state.calculation_result or {}

        for calc_key, expected_value in expect["calculation"].items():
            actual_value = calc.get(calc_key)

            add_check(
                name=f"calculation:{calc_key}",
                passed=compare_values(actual_value, expected_value),
                expected=expected_value,
                actual=actual_value,
            )

    passed_all = all(check["passed"] for check in checks)

    return passed_all, checks


def run_scenario(bot_class, scenario: Dict[str, Any]) -> Dict[str, Any]:
    bot = bot_class()

    scenario_result = {
        "id": scenario["id"],
        "description": scenario.get("description", ""),
        "tags": scenario.get("tags", []),
        "turns": [],
        "passed": True,
    }

    for turn_index, turn in enumerate(scenario["turns"], start=1):
        user_text = turn["user"]
        expect = turn.get("expect", {})

        stdout_buffer = io.StringIO()
        start_time = time.perf_counter()

        with contextlib.redirect_stdout(stdout_buffer):
            state = bot.process_user_message(user_text)

        latency_s = time.perf_counter() - start_time
        stdout_text = stdout_buffer.getvalue()

        passed, checks = check_expectation(
            state=state,
            stdout_text=stdout_text,
            expect=expect,
        )

        final_json = plain_final_json(state)

        turn_result = {
            "turn_index": turn_index,
            "user": user_text,
            "passed": passed,
            "latency_s": latency_s,
            "status": enum_value(state.status),
            "stage": enum_value(state.stage),
            "questions": state.questions,
            "final_json": final_json,
            "calculation_result": state.calculation_result,
            "checks": checks,
            "observed_llm_fallback": "[LLM FALLBACK]" in stdout_text,
            "observed_ai_plausibility": "[AI PLAUSIBILITY]" in stdout_text,
            "stdout_tail": stdout_text[-2000:],
        }

        scenario_result["turns"].append(turn_result)

        if not passed:
            scenario_result["passed"] = False

    return scenario_result


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    scenario_total = len(results)
    scenario_passed = sum(1 for result in results if result["passed"])

    turns = [
        turn
        for result in results
        for turn in result["turns"]
    ]

    turn_total = len(turns)
    turn_passed = sum(1 for turn in turns if turn["passed"])

    checks = [
        check
        for turn in turns
        for check in turn["checks"]
    ]

    check_total = len(checks)
    check_passed = sum(1 for check in checks if check["passed"])

    field_checks = [
        check
        for check in checks
        if check["name"].startswith("field:")
    ]

    field_total = len(field_checks)
    field_passed = sum(1 for check in field_checks if check["passed"])

    status_checks = [
        check
        for check in checks
        if check["name"] == "status"
    ]

    stage_checks = [
        check
        for check in checks
        if check["name"] == "stage"
    ]

    question_checks = [
        check
        for check in checks
        if check["name"] == "question_contains_any"
    ]

    stdout_checks = [
        check
        for check in checks
        if check["name"] == "stdout_contains_any"
    ]

    calculation_checks = [
        check
        for check in checks
        if check["name"].startswith("calculation:")
    ]

    llm_fallback_turns = sum(1 for turn in turns if turn["observed_llm_fallback"])
    ai_plausibility_turns = sum(1 for turn in turns if turn["observed_ai_plausibility"])

    avg_latency = (
        sum(turn["latency_s"] for turn in turns) / turn_total
        if turn_total
        else 0.0
    )

    def rate(passed: int, total: int) -> float:
        if total == 0:
            return 0.0
        return passed / total

    summary = {
        "scenario_total": scenario_total,
        "scenario_passed": scenario_passed,
        "scenario_pass_rate": rate(scenario_passed, scenario_total),

        "turn_total": turn_total,
        "turn_passed": turn_passed,
        "turn_pass_rate": rate(turn_passed, turn_total),

        "check_total": check_total,
        "check_passed": check_passed,
        "check_pass_rate": rate(check_passed, check_total),

        "field_check_total": field_total,
        "field_check_passed": field_passed,
        "field_accuracy": rate(field_passed, field_total),
        "field_error_rate": 1.0 - rate(field_passed, field_total),

        "status_check_total": len(status_checks),
        "status_accuracy": rate(
            sum(1 for check in status_checks if check["passed"]),
            len(status_checks),
        ),

        "stage_check_total": len(stage_checks),
        "stage_accuracy": rate(
            sum(1 for check in stage_checks if check["passed"]),
            len(stage_checks),
        ),

        "question_check_total": len(question_checks),
        "question_accuracy": rate(
            sum(1 for check in question_checks if check["passed"]),
            len(question_checks),
        ),

        "stdout_check_total": len(stdout_checks),
        "stdout_signal_accuracy": rate(
            sum(1 for check in stdout_checks if check["passed"]),
            len(stdout_checks),
        ),

        "calculation_check_total": len(calculation_checks),
        "calculation_accuracy": rate(
            sum(1 for check in calculation_checks if check["passed"]),
            len(calculation_checks),
        ),

        "llm_fallback_turns_observed": llm_fallback_turns,
        "ai_plausibility_turns_observed": ai_plausibility_turns,
        "avg_latency_s_per_turn": avg_latency,
    }

    return summary


def write_outputs(results: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_path = RESULTS_DIR / "test1_results.json"
    summary_path = RESULTS_DIR / "test1_summary.json"
    csv_path = RESULTS_DIR / "test1_summary.csv"

    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])

        for key, value in summary.items():
            writer.writerow([key, value])


def print_summary(summary: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 90)
    print("TEST1 STRESS TEST SUMMARY")
    print("=" * 90)

    print(f"Scenarios passed : {summary['scenario_passed']}/{summary['scenario_total']} "
          f"({summary['scenario_pass_rate']:.2%})")

    print(f"Turns passed     : {summary['turn_passed']}/{summary['turn_total']} "
          f"({summary['turn_pass_rate']:.2%})")

    print(f"Field accuracy   : {summary['field_accuracy']:.2%}")
    print(f"Field error rate : {summary['field_error_rate']:.2%}")
    print(f"Status accuracy  : {summary['status_accuracy']:.2%}")
    print(f"Stage accuracy   : {summary['stage_accuracy']:.2%}")
    print(f"Question accuracy: {summary['question_accuracy']:.2%}")
    print(f"Calc accuracy    : {summary['calculation_accuracy']:.2%}")
    print(f"Avg latency/turn : {summary['avg_latency_s_per_turn']:.2f}s")
    print(f"LLM fallback seen: {summary['llm_fallback_turns_observed']} turns")
    print(f"AI plausibility  : {summary['ai_plausibility_turns_observed']} turns")

    failed = [
        result
        for result in results
        if not result["passed"]
    ]

    if failed:
        print("\nFAILED SCENARIOS")
        print("-" * 90)

        for result in failed:
            print(f"- {result['id']}")

            for turn in result["turns"]:
                if turn["passed"]:
                    continue

                print(f"  Turn {turn['turn_index']} failed")
                print(f"  User: {turn['user']}")

                for check in turn["checks"]:
                    if not check["passed"]:
                        print(
                            f"    ✗ {check['name']} | "
                            f"expected={check['expected']} | actual={check['actual']}"
                        )
    else:
        print("\nAll scenarios passed.")

    print("=" * 90 + "\n")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default=str(THIS_DIR / "stress_dataset_test1.json"),
    )

    parser.add_argument(
        "--llm",
        action="store_true",
        default=True,
        help="Enable LLM fallback.",
    )

    parser.add_argument(
        "--no-llm",
        action="store_false",
        dest="llm",
        help="Disable LLM fallback.",
    )

    parser.add_argument(
        "--ai",
        action="store_true",
        default=True,
        help="Enable AI PlausibilityAgent.",
    )

    parser.add_argument(
        "--no-ai",
        action="store_false",
        dest="ai",
        help="Disable AI PlausibilityAgent.",
    )

    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
    )

    parser.add_argument(
        "--ollama-model",
        default="qwen2.5-coder:7b",
    )

    parser.add_argument(
        "--ai-model",
        default="qwen2.5:3b",
    )

    parser.add_argument(
        "--ai-temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--ai-debug",
        action="store_true",
        default=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_environment(args)

    from requirement_chatbot import RequirementChatbot

    dataset_path = Path(args.dataset)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    results = []

    for scenario in dataset["scenarios"]:
        print(f"Running {scenario['id']}...")
        result = run_scenario(
            bot_class=RequirementChatbot,
            scenario=scenario,
        )
        results.append(result)

    summary = summarize_results(results)

    write_outputs(results, summary)
    print_summary(summary, results)

    print(f"Results written to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()