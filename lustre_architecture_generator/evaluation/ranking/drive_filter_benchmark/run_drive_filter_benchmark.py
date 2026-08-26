from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parents[2]
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import mdt_candidate_generator as mdt_generator  # noqa: E402
import ost_candidate_generator as ost_generator  # noqa: E402


DEFAULT_CASES = HERE / "filter_benchmark_cases.json"
DEFAULT_CATALOG = BASE_DIR / "data" / "catalogue_drives_ready_final.json"
DEFAULT_JSON = HERE / "drive_filter_benchmark_results.json"
DEFAULT_CSV = HERE / "drive_filter_benchmark_results.csv"
DEFAULT_MD = HERE / "DRIVE_FILTER_BENCHMARK_REPORT.md"


class FilterBenchmarkError(ValueError):
    """Invalid benchmark definition or execution state."""


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(100.0 * numerator / denominator, 6)


def select_catalog(
    catalog: list[dict[str, Any]],
    scope: dict[str, Any],
) -> list[dict[str, Any]]:
    scope_type = scope.get("type")
    if scope_type == "full":
        return catalog

    if scope_type != "drive_ids":
        raise FilterBenchmarkError(f"Unsupported catalog scope: {scope_type!r}")

    requested = list(scope.get("drive_ids", []))
    by_id = {str(drive["drive_id"]): drive for drive in catalog}
    missing = [drive_id for drive_id in requested if drive_id not in by_id]
    if missing:
        raise FilterBenchmarkError(f"Unknown benchmark drive ids: {missing}")

    return [by_id[drive_id] for drive_id in requested]


def evaluate_case(
    case: dict[str, Any],
    catalog: list[dict[str, Any]],
) -> dict[str, Any]:
    role = str(case["role"])
    selected_catalog = select_catalog(catalog, case["catalog_scope"])
    architecture_case = case["architecture_case"]
    expected = case["expected"]

    if role == "MDT":
        mdt_generator.validate_catalog(selected_catalog)
        output = mdt_generator.generate_case_candidates(
            architecture_case,
            selected_catalog,
            top_k=max(1, len(selected_catalog)),
        )
    elif role == "OST":
        ost_generator.validate_catalog(selected_catalog)
        output = ost_generator.generate_case_candidates(
            architecture_case,
            selected_catalog,
            top_k=max(1, len(selected_catalog)),
        )
    else:
        raise FilterBenchmarkError(f"Unsupported role: {role!r}")

    summary = output["candidate_summary"]
    feasible_count = int(summary["pre_raid_feasible_count"])
    predicted_feasible = feasible_count > 0
    expected_feasible = bool(expected["feasible"])
    decision_correct = predicted_feasible == expected_feasible

    rejection_counts = dict(summary.get("rejection_counts", {}))

    exact_expected = expected.get("exact_rejection_counts")
    exact_reason_correct: bool | None = None
    if exact_expected is not None:
        exact_reason_correct = rejection_counts == exact_expected

    required_reasons = list(expected.get("required_rejection_reasons", []))
    required_reason_correct: bool | None = None
    if required_reasons:
        required_reason_correct = all(
            int(rejection_counts.get(reason, 0)) > 0
            for reason in required_reasons
        )

    reason_checked = exact_reason_correct is not None or required_reason_correct is not None
    reason_correct = True
    if exact_reason_correct is not None:
        reason_correct = reason_correct and exact_reason_correct
    if required_reason_correct is not None:
        reason_correct = reason_correct and required_reason_correct

    return {
        "case_id": case["case_id"],
        "role": role,
        "group": case["group"],
        "catalog_scope": case["catalog_scope"]["type"],
        "catalog_drive_count": len(selected_catalog),
        "expected_feasible": expected_feasible,
        "predicted_feasible": predicted_feasible,
        "decision_correct": decision_correct,
        "feasible_drive_count": feasible_count,
        "rejection_counts": rejection_counts,
        "reason_checked": reason_checked,
        "reason_correct": reason_correct if reason_checked else None,
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    decision_correct = sum(bool(row["decision_correct"]) for row in results)
    expected_feasible = sum(bool(row["expected_feasible"]) for row in results)
    expected_infeasible = total - expected_feasible

    false_feasible = sum(
        (not bool(row["expected_feasible"])) and bool(row["predicted_feasible"])
        for row in results
    )
    false_infeasible = sum(
        bool(row["expected_feasible"]) and (not bool(row["predicted_feasible"]))
        for row in results
    )

    reason_rows = [row for row in results if row["reason_checked"]]
    reason_correct = sum(bool(row["reason_correct"]) for row in reason_rows)

    by_role: dict[str, Any] = {}
    by_group: dict[str, Any] = {}

    for field, target in (("role", by_role), ("group", by_group)):
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in results:
            buckets[str(row[field])].append(row)
        for name, rows in sorted(buckets.items()):
            target[name] = {
                "case_count": len(rows),
                "decision_accuracy_percent": pct(
                    sum(bool(row["decision_correct"]) for row in rows),
                    len(rows),
                ),
            }

    return {
        "case_count": total,
        "expected_feasible_cases": expected_feasible,
        "expected_infeasible_cases": expected_infeasible,
        "correct_decisions": decision_correct,
        "decision_accuracy_percent": pct(decision_correct, total),
        "false_feasible_count": false_feasible,
        "false_feasible_rate_percent": pct(false_feasible, expected_infeasible),
        "false_infeasible_count": false_infeasible,
        "false_infeasible_rate_percent": pct(false_infeasible, expected_feasible),
        "rejection_reason_cases": len(reason_rows),
        "correct_rejection_reason_cases": reason_correct,
        "rejection_reason_accuracy_percent": pct(reason_correct, len(reason_rows)),
        "by_role": by_role,
        "by_group": by_group,
    }


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "case_id",
        "role",
        "group",
        "catalog_scope",
        "catalog_drive_count",
        "expected_feasible",
        "predicted_feasible",
        "decision_correct",
        "feasible_drive_count",
        "reason_checked",
        "reason_correct",
        "rejection_counts",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in results:
            rendered = dict(row)
            rendered["rejection_counts"] = json.dumps(
                row["rejection_counts"], ensure_ascii=False, sort_keys=True
            )
            writer.writerow(rendered)


def write_markdown(
    path: Path,
    benchmark: dict[str, Any],
    results: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    lines = [
        "# Deterministic MDT/OST Drive Filter Benchmark",
        "",
        f"Benchmark version: `{benchmark['benchmark_version']}`",
        f"Scope: `{benchmark['scope']}`",
        f"Throughput contract: {benchmark['throughput_contract']}",
        "",
        "## Purpose",
        "",
        "This benchmark strengthens the negative-case evaluation of the deterministic drive filter.",
        "It contains clearly feasible, clearly infeasible, and boundary requests for both MDT and OST.",
        "The ML ranker is not involved: invalid candidates must be rejected before ranking.",
        "",
        "## Summary",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Cases | {metrics['case_count']} |",
        f"| Decision accuracy | {metrics['decision_accuracy_percent']:.2f}% |",
        f"| False-feasible rate | {metrics['false_feasible_rate_percent']:.2f}% |",
        f"| False-infeasible rate | {metrics['false_infeasible_rate_percent']:.2f}% |",
        f"| Rejection-reason accuracy | {metrics['rejection_reason_accuracy_percent']:.2f}% |",
        "",
        "## Accuracy by role",
        "",
        "| Role | Cases | Accuracy |",
        "|---|---:|---:|",
    ]

    for role, values in metrics["by_role"].items():
        lines.append(
            f"| {role} | {values['case_count']} | {values['decision_accuracy_percent']:.2f}% |"
        )

    lines += [
        "",
        "## Accuracy by benchmark group",
        "",
        "| Group | Cases | Accuracy |",
        "|---|---:|---:|",
    ]
    for group, values in metrics["by_group"].items():
        lines.append(
            f"| {group} | {values['case_count']} | {values['decision_accuracy_percent']:.2f}% |"
        )

    lines += [
        "",
        "## Case-level results",
        "",
        "| Case | Role | Group | Expected | Predicted | Feasible drives | Decision | Reasons |",
        "|---|---|---|---|---|---:|---|---|",
    ]

    for row in results:
        reasons = ", ".join(
            f"{key}:{value}" for key, value in sorted(row["rejection_counts"].items())
        ) or "-"
        lines.append(
            "| {case_id} | {role} | {group} | {expected} | {predicted} | {count} | {decision} | {reasons} |".format(
                case_id=row["case_id"],
                role=row["role"],
                group=row["group"],
                expected="feasible" if row["expected_feasible"] else "infeasible",
                predicted="feasible" if row["predicted_feasible"] else "infeasible",
                count=row["feasible_drive_count"],
                decision="PASS" if row["decision_correct"] else "FAIL",
                reasons=reasons,
            )
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "Budget and power checks remain lower-bound filters at this drive-selection stage.",
        "Final global budget and power validation belongs to the complete architecture layer.",
        "Boundary OST cases explicitly verify that a 550 MB/s catalogue drive is treated as 0.55 GB/s under the frozen historical field contract.",
        "",
        "## Stop condition",
        "",
        "The filter benchmark is considered validated only if decision accuracy and rejection-reason accuracy are 100%, with zero false-feasible and zero false-infeasible cases.",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(
    cases_path: Path = DEFAULT_CASES,
    catalog_path: Path = DEFAULT_CATALOG,
) -> dict[str, Any]:
    benchmark = load_json(cases_path)
    catalog = load_json(catalog_path)
    if not isinstance(catalog, list) or not catalog:
        raise FilterBenchmarkError("Drive catalog must be a non-empty list.")

    results = [evaluate_case(case, catalog) for case in benchmark["cases"]]
    metrics = aggregate(results)

    validated = (
        math.isclose(metrics["decision_accuracy_percent"], 100.0)
        and math.isclose(metrics["false_feasible_rate_percent"], 0.0)
        and math.isclose(metrics["false_infeasible_rate_percent"], 0.0)
        and math.isclose(metrics["rejection_reason_accuracy_percent"], 100.0)
    )

    return {
        "benchmark_version": benchmark["benchmark_version"],
        "scope": benchmark["scope"],
        "throughput_contract": benchmark["throughput_contract"],
        "metrics": metrics,
        "results": results,
        "status": "VALIDATED" if validated else "FAILED",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic MDT/OST drive-filter benchmark.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark_definition = load_json(args.cases)
    result = run_benchmark(args.cases, args.catalog)

    save_json(args.json_output, result)
    write_csv(args.csv_output, result["results"])
    write_markdown(
        args.markdown_output,
        benchmark_definition,
        result["results"],
        result["metrics"],
    )

    metrics = result["metrics"]
    print("Deterministic MDT/OST Drive Filter Benchmark")
    print("--------------------------------------------")
    print(f"Cases                         : {metrics['case_count']}")
    print(f"Decision accuracy             : {metrics['decision_accuracy_percent']:.2f}%")
    print(f"False-feasible rate           : {metrics['false_feasible_rate_percent']:.2f}%")
    print(f"False-infeasible rate         : {metrics['false_infeasible_rate_percent']:.2f}%")
    print(f"Rejection-reason accuracy     : {metrics['rejection_reason_accuracy_percent']:.2f}%")
    print(f"Status                        : {result['status']}")
    print(f"JSON                          : {args.json_output}")
    print(f"CSV                           : {args.csv_output}")
    print(f"Report                        : {args.markdown_output}")

    return 0 if result["status"] == "VALIDATED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
