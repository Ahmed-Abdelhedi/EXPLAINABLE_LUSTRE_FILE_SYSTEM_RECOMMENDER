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
from collections import defaultdict
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
        return math.isclose(
            float(actual),
            float(expected),
            rel_tol=tolerance,
            abs_tol=tolerance,
        )

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

    return all(check["passed"] for check in checks), checks


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

        scenario_result["turns"].append(
            {
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
        )

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

    checks = [
        check
        for turn in turns
        for check in turn["checks"]
    ]

    def rate(passed: int, total: int) -> float:
        return passed / total if total else 0.0

    def check_rate(prefix_or_name: str, exact: bool = False) -> float:
        selected = []

        for check in checks:
            if exact and check["name"] == prefix_or_name:
                selected.append(check)
            elif not exact and check["name"].startswith(prefix_or_name):
                selected.append(check)

        return rate(
            sum(1 for check in selected if check["passed"]),
            len(selected),
        )

    field_checks = [
        check for check in checks
        if check["name"].startswith("field:")
    ]

    scenario_by_tag = defaultdict(lambda: {"total": 0, "passed": 0})

    for result in results:
        for tag in result["tags"]:
            scenario_by_tag[tag]["total"] += 1

            if result["passed"]:
                scenario_by_tag[tag]["passed"] += 1

    tag_metrics = {
        tag: {
            "passed": values["passed"],
            "total": values["total"],
            "pass_rate": rate(values["passed"], values["total"]),
        }
        for tag, values in sorted(scenario_by_tag.items())
    }

    summary = {
        "scenario_total": scenario_total,
        "scenario_passed": scenario_passed,
        "scenario_pass_rate": rate(scenario_passed, scenario_total),

        "turn_total": len(turns),
        "turn_passed": sum(1 for turn in turns if turn["passed"]),
        "turn_pass_rate": rate(
            sum(1 for turn in turns if turn["passed"]),
            len(turns),
        ),

        "check_total": len(checks),
        "check_passed": sum(1 for check in checks if check["passed"]),
        "check_pass_rate": rate(
            sum(1 for check in checks if check["passed"]),
            len(checks),
        ),

        "field_check_total": len(field_checks),
        "field_check_passed": sum(1 for check in field_checks if check["passed"]),
        "field_accuracy": rate(
            sum(1 for check in field_checks if check["passed"]),
            len(field_checks),
        ),
        "field_error_rate": 1.0 - rate(
            sum(1 for check in field_checks if check["passed"]),
            len(field_checks),
        ),

        "status_accuracy": check_rate("status", exact=True),
        "stage_accuracy": check_rate("stage", exact=True),
        "question_accuracy": check_rate("question_contains_any", exact=True),
        "stdout_signal_accuracy": check_rate("stdout_contains_any", exact=True),
        "calculation_accuracy": check_rate("calculation:"),

        "llm_fallback_turns_observed": sum(
            1 for turn in turns if turn["observed_llm_fallback"]
        ),
        "ai_plausibility_turns_observed": sum(
            1 for turn in turns if turn["observed_ai_plausibility"]
        ),
        "avg_latency_s_per_turn": (
            sum(turn["latency_s"] for turn in turns) / len(turns)
            if turns
            else 0.0
        ),
        "tag_metrics": tag_metrics,
    }

    return summary


def write_outputs(results: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    (RESULTS_DIR / "test2_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (RESULTS_DIR / "test2_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (RESULTS_DIR / "test2_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])

        for key, value in summary.items():
            if key == "tag_metrics":
                continue

            writer.writerow([key, value])

    with (RESULTS_DIR / "test2_tag_metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["tag", "passed", "total", "pass_rate"])

        for tag, values in summary["tag_metrics"].items():
            writer.writerow(
                [
                    tag,
                    values["passed"],
                    values["total"],
                    values["pass_rate"],
                ]
            )


def print_summary(summary: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 90)
    print("TEST2 HARD STRESS TEST SUMMARY")
    print("=" * 90)

    print(
        f"Scenarios passed : {summary['scenario_passed']}/"
        f"{summary['scenario_total']} ({summary['scenario_pass_rate']:.2%})"
    )

    print(
        f"Turns passed     : {summary['turn_passed']}/"
        f"{summary['turn_total']} ({summary['turn_pass_rate']:.2%})"
    )

    print(f"Field accuracy   : {summary['field_accuracy']:.2%}")
    print(f"Field error rate : {summary['field_error_rate']:.2%}")
    print(f"Status accuracy  : {summary['status_accuracy']:.2%}")
    print(f"Stage accuracy   : {summary['stage_accuracy']:.2%}")
    print(f"Question accuracy: {summary['question_accuracy']:.2%}")
    print(f"Calc accuracy    : {summary['calculation_accuracy']:.2%}")
    print(f"Avg latency/turn : {summary['avg_latency_s_per_turn']:.2f}s")
    print(f"LLM fallback seen: {summary['llm_fallback_turns_observed']} turns")
    print(f"AI plausibility  : {summary['ai_plausibility_turns_observed']} turns")

    print("\nTAG METRICS")
    print("-" * 90)

    for tag, values in summary["tag_metrics"].items():
        print(
            f"{tag:30s} "
            f"{values['passed']}/{values['total']} "
            f"({values['pass_rate']:.2%})"
        )

    failed = [
        result
        for result in results
        if not result["passed"]
    ]

    if failed:
        print("\nFAILED SCENARIOS")
        print("-" * 90)

        for result in failed:
            print(f"- {result['id']} | tags={result['tags']}")

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
        default=str(THIS_DIR / "stress_dataset_test2.json"),
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