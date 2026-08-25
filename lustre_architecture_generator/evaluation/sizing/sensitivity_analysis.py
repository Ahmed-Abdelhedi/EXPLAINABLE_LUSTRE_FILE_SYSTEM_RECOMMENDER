"""Run S9 sensitivity analysis for all registered Lustre sizing assumptions.

The analysis replays the 1200 use cases through the existing deterministic
pipeline for controlled perturbations of each sizing assumption. Score weights
are renormalized to preserve the sum-to-one invariant.

Run from repository root:

    python evaluation/sizing/sensitivity_analysis.py

Outputs:
- evaluation/sizing/sensitivity_analysis.json
- evaluation/sizing/sensitivity_analysis.csv
- evaluation/sizing/sensitivity_analysis.md
"""

from __future__ import annotations

import copy
import csv
import json
import math
import sys
from pathlib import Path
from statistics import median
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = BASE_DIR / "src"
SIZING_DIR = BASE_DIR / "evaluation" / "sizing"
CONFIG_FILE = BASE_DIR / "config" / "architecture_rules.json"
REGISTRY_FILE = SIZING_DIR / "sizing_assumptions.json"
CASES_FILE = BASE_DIR / "data" / "use_cases_lustre_1200_v4.json"
OUTPUT_JSON = SIZING_DIR / "sensitivity_analysis.json"
OUTPUT_CSV = SIZING_DIR / "sensitivity_analysis.csv"
OUTPUT_MD = SIZING_DIR / "sensitivity_analysis.md"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from workload_analyzer import analyze_workload, validate_config as validate_workload_config  # noqa: E402
from feature_calculator import calculate_features, validate_config as validate_feature_config  # noqa: E402
from architecture_generator import generate_architecture_case, validate_config as validate_arch_config  # noqa: E402


class SensitivityError(ValueError):
    """Raised when the sensitivity study cannot be reproduced safely."""


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def nested_get(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise SensitivityError(f"Unknown config path: {path}")
        current = current[part]
    return current


def nested_set(data: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current: Any = data
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise SensitivityError(f"Unknown config path: {path}")
        current = current[part]
    if not isinstance(current, dict):
        raise SensitivityError(f"Invalid config parent for: {path}")
    current[parts[-1]] = value


def validate_all_config(config: dict[str, Any]) -> None:
    validate_workload_config(config)
    validate_feature_config(config)
    validate_arch_config(config)


def perturb_weight_group(
    config: dict[str, Any],
    path: str,
    new_target: float,
) -> None:
    parts = path.split(".")
    if len(parts) != 3 or parts[0] != "score_weights":
        raise SensitivityError(f"Not a score-weight path: {path}")
    group = config[parts[0]][parts[1]]
    target_key = parts[2]
    if not 0.0 < new_target < 1.0:
        raise SensitivityError(f"Perturbed weight must be in (0, 1): {path}")

    old_target = float(group[target_key])
    other_keys = [key for key in group if key != target_key]
    old_other_sum = sum(float(group[key]) for key in other_keys)
    if old_other_sum <= 0:
        raise SensitivityError(f"Cannot renormalize score weights for {path}")

    group[target_key] = new_target
    new_other_sum = 1.0 - new_target
    for key in other_keys:
        share = float(group[key]) / old_other_sum
        group[key] = share * new_other_sum

    total = sum(float(value) for value in group.values())
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise SensitivityError(
            f"Renormalization failed for {path}: old_target={old_target}, total={total}"
        )


def build_perturbed_config(
    base_config: dict[str, Any],
    path: str,
    value: float,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    if path.startswith("score_weights."):
        perturb_weight_group(config, path, float(value))
    else:
        nested_set(config, path, float(value))
    validate_all_config(config)
    return config


def prepare_s9_legacy_fixture(
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Materialise explicit 1-year horizons for the historical S9 fixture.

    This adapter exists only to reproduce the already-completed S9 study on
    the legacy 1200-case dataset. The production workload analyzer remains
    strict and never supplies a missing planning horizon.
    """

    prepared: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            raise SensitivityError("Each legacy S9 case must be a JSON object")
        migrated = copy.deepcopy(case)
        migrated.setdefault("planning_horizon_years", 1.0)
        prepared.append(migrated)
    return prepared


def run_case(case: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    analyzed = analyze_workload(case, config)
    features = calculate_features(analyzed, config)
    architecture = generate_architecture_case(features, config)
    return {
        "case_id": architecture["case_id"],
        "workload_type": analyzed["workload_type"],
        "metadata_pressure": features["mdt_features"]["metadata_pressure"],
        "data_pressure": features["ost_features"]["data_pressure"],
        "planned_capacity_tib": float(
            analyzed["capacity_planning"]["planned_usable_capacity_tib"]
        ),
        "mdt_iops": float(
            architecture["MDT_requirement"]["required_total_iops"]
        ),
        "mdt_metadata_capacity_tib": float(
            architecture["MDT_requirement"]["required_metadata_capacity_tib"]
        ),
        "ost_read_bandwidth": float(
            architecture["OST_requirement"]["required_read_bandwidth_gbps"]
        ),
        "ost_write_bandwidth": float(
            architecture["OST_requirement"]["required_write_bandwidth_gbps"]
        ),
        "ost_total_bandwidth": float(
            architecture["OST_requirement"]["required_total_bandwidth_gbps"]
        ),
        "ost_usable_capacity_tib": float(
            architecture["OST_requirement"]["required_usable_capacity_tib"]
        ),
    }


def run_dataset(
    cases: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    return [run_case(case, config) for case in cases]


def relative_change_percent(value: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0 if value == 0 else math.inf
    return (value / baseline - 1.0) * 100.0


def summarize_numeric_changes(
    baseline: list[dict[str, Any]],
    variant: list[dict[str, Any]],
    field: str,
) -> dict[str, float]:
    changes = [
        relative_change_percent(float(v[field]), float(b[field]))
        for b, v in zip(baseline, variant, strict=True)
    ]
    abs_changes = [abs(value) for value in changes]
    return {
        "median_change_percent": round(median(changes), 4),
        "p95_abs_change_percent": round(percentile(abs_changes, 0.95), 4),
        "max_abs_change_percent": round(max(abs_changes), 4),
    }


def summarize_variant(
    baseline: list[dict[str, Any]],
    variant: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(baseline) != len(variant):
        raise SensitivityError("Baseline and variant lengths differ")

    count = len(baseline)
    workload_flips = sum(
        b["workload_type"] != v["workload_type"]
        for b, v in zip(baseline, variant, strict=True)
    )
    metadata_pressure_flips = sum(
        b["metadata_pressure"] != v["metadata_pressure"]
        for b, v in zip(baseline, variant, strict=True)
    )
    data_pressure_flips = sum(
        b["data_pressure"] != v["data_pressure"]
        for b, v in zip(baseline, variant, strict=True)
    )

    numeric_fields = [
        "planned_capacity_tib",
        "mdt_iops",
        "mdt_metadata_capacity_tib",
        "ost_read_bandwidth",
        "ost_write_bandwidth",
        "ost_total_bandwidth",
        "ost_usable_capacity_tib",
    ]
    numeric = {
        field: summarize_numeric_changes(baseline, variant, field)
        for field in numeric_fields
    }

    return {
        "workload_type_flip_count": workload_flips,
        "workload_type_flip_percent": round(100.0 * workload_flips / count, 4),
        "metadata_pressure_flip_count": metadata_pressure_flips,
        "metadata_pressure_flip_percent": round(
            100.0 * metadata_pressure_flips / count,
            4,
        ),
        "data_pressure_flip_count": data_pressure_flips,
        "data_pressure_flip_percent": round(
            100.0 * data_pressure_flips / count,
            4,
        ),
        "numeric": numeric,
    }


def perturbation_values(assumption: dict[str, Any]) -> list[dict[str, Any]]:
    current = float(assumption["current_value"])
    if "sensitivity_values" in assumption:
        values = [float(value) for value in assumption["sensitivity_values"]]
        return [
            {
                "label": f"value={value:g}",
                "value": value,
                "relative_percent": round(relative_change_percent(value, current), 4),
            }
            for value in values
        ]

    percents = assumption.get("sensitivity_percent")
    if not isinstance(percents, list) or not percents:
        raise SensitivityError(
            f"Assumption {assumption['id']} has no sensitivity definition"
        )
    return [
        {
            "label": f"{percent:+g}%",
            "value": current * (1.0 + float(percent) / 100.0),
            "relative_percent": float(percent),
        }
        for percent in percents
    ]


def overall_sensitivity_level(variants: list[dict[str, Any]]) -> str:
    max_flip = max(
        max(
            float(item["summary"]["workload_type_flip_percent"]),
            float(item["summary"]["metadata_pressure_flip_percent"]),
            float(item["summary"]["data_pressure_flip_percent"]),
        )
        for item in variants
    )
    max_p95 = max(
        float(metric["p95_abs_change_percent"])
        for item in variants
        for metric in item["summary"]["numeric"].values()
    )

    if max_flip >= 10.0 or max_p95 >= 20.0:
        return "HIGH"
    if max_flip >= 2.0 or max_p95 >= 10.0:
        return "MEDIUM"
    return "LOW"


def build_report() -> dict[str, Any]:
    config = load_json(CONFIG_FILE)
    registry = load_json(REGISTRY_FILE)
    cases = load_json(CASES_FILE)
    if not isinstance(cases, list) or not cases:
        raise SensitivityError("Use-case dataset must be a non-empty JSON list")

    validate_all_config(config)
    cases = prepare_s9_legacy_fixture(cases)
    baseline = run_dataset(cases, config)

    assumptions_output: list[dict[str, Any]] = []
    for assumption in registry["assumptions"]:
        path = assumption["config_path"]
        current = float(assumption["current_value"])
        actual = float(nested_get(config, path))
        if not math.isclose(current, actual, rel_tol=0.0, abs_tol=1e-12):
            raise SensitivityError(
                f"Registry/config mismatch for {path}: registry={current}, config={actual}"
            )

        variants: list[dict[str, Any]] = []
        for perturbation in perturbation_values(assumption):
            value = float(perturbation["value"])
            variant_config = build_perturbed_config(config, path, value)
            variant_results = run_dataset(cases, variant_config)
            variants.append(
                {
                    **perturbation,
                    "summary": summarize_variant(baseline, variant_results),
                }
            )

        assumptions_output.append(
            {
                "id": assumption["id"],
                "config_path": path,
                "current_value": assumption["current_value"],
                "unit": assumption["unit"],
                "weight_renormalized": path.startswith("score_weights."),
                "sensitivity_level": overall_sensitivity_level(variants),
                "variants": variants,
            }
        )

    return {
        "artifact_version": "1.0",
        "stage": "S9_sensitivity_analysis",
        "config_version": config.get("version"),
        "case_count": len(cases),
        "dataset_note": (
            "The historical 1200-case S9 fixture predates planning_horizon_years. "
            "For reproducibility only, the S9 analysis materialises an explicit "
            "1.0-year horizon in an in-memory copy before calling the strict S10 analyzer. "
            "No production fallback exists."
        ),
        "method": {
            "score_weight_rule": (
                "Perturb the target score weight and proportionally renormalize "
                "the other weights in the same group so the group still sums to 1."
            ),
            "numeric_metrics": [
                "median_change_percent",
                "p95_abs_change_percent",
                "max_abs_change_percent",
            ],
            "classification_metrics": [
                "workload_type_flip_percent",
                "metadata_pressure_flip_percent",
                "data_pressure_flip_percent",
            ],
        },
        "assumptions": assumptions_output,
    }


def write_csv(report: dict[str, Any], path: Path) -> None:
    fields = [
        "id",
        "config_path",
        "sensitivity_level",
        "perturbation",
        "value",
        "relative_percent",
        "workload_type_flip_percent",
        "metadata_pressure_flip_percent",
        "data_pressure_flip_percent",
        "planned_capacity_p95_abs_percent",
        "mdt_iops_p95_abs_percent",
        "mdt_metadata_capacity_p95_abs_percent",
        "ost_read_p95_abs_percent",
        "ost_write_p95_abs_percent",
        "ost_capacity_p95_abs_percent",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for assumption in report["assumptions"]:
            for variant in assumption["variants"]:
                summary = variant["summary"]
                numeric = summary["numeric"]
                writer.writerow(
                    {
                        "id": assumption["id"],
                        "config_path": assumption["config_path"],
                        "sensitivity_level": assumption["sensitivity_level"],
                        "perturbation": variant["label"],
                        "value": round(float(variant["value"]), 10),
                        "relative_percent": variant["relative_percent"],
                        "workload_type_flip_percent": summary["workload_type_flip_percent"],
                        "metadata_pressure_flip_percent": summary["metadata_pressure_flip_percent"],
                        "data_pressure_flip_percent": summary["data_pressure_flip_percent"],
                        "planned_capacity_p95_abs_percent": numeric["planned_capacity_tib"]["p95_abs_change_percent"],
                        "mdt_iops_p95_abs_percent": numeric["mdt_iops"]["p95_abs_change_percent"],
                        "mdt_metadata_capacity_p95_abs_percent": numeric["mdt_metadata_capacity_tib"]["p95_abs_change_percent"],
                        "ost_read_p95_abs_percent": numeric["ost_read_bandwidth"]["p95_abs_change_percent"],
                        "ost_write_p95_abs_percent": numeric["ost_write_bandwidth"]["p95_abs_change_percent"],
                        "ost_capacity_p95_abs_percent": numeric["ost_usable_capacity_tib"]["p95_abs_change_percent"],
                    }
                )


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# S9 - Lustre Sizing Sensitivity Analysis",
        "",
        f"Cases replayed: **{report['case_count']}**.",
        "",
        report["dataset_note"],
        "",
        "Score-weight perturbations preserve the sum-to-one invariant by proportionally renormalizing the other weights in the same score group.",
        "",
        "| Assumption | Path | Sensitivity | Max workload flips | Max metadata-pressure flips | Max p95 numeric impact |",
        "|---|---|---|---:|---:|---:|",
    ]

    for assumption in report["assumptions"]:
        max_workload_flip = max(
            float(v["summary"]["workload_type_flip_percent"])
            for v in assumption["variants"]
        )
        max_metadata_flip = max(
            float(v["summary"]["metadata_pressure_flip_percent"])
            for v in assumption["variants"]
        )
        max_p95 = max(
            float(metric["p95_abs_change_percent"])
            for v in assumption["variants"]
            for metric in v["summary"]["numeric"].values()
        )
        lines.append(
            f"| `{assumption['id']}` | `{assumption['config_path']}` | "
            f"{assumption['sensitivity_level']} | {max_workload_flip:.2f}% | "
            f"{max_metadata_flip:.2f}% | {max_p95:.2f}% |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- HIGH means a tested perturbation causes at least 10% classification flips or at least 20% p95 numeric output change.",
        "- MEDIUM means at least 2% classification flips or at least 10% p95 numeric output change.",
        "- LOW means the tested perturbations remain below those thresholds.",
        "- Sensitivity is not empirical validation. Final statuses must combine this artifact with S8 evidence and design semantics.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    report = build_report()
    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(report, OUTPUT_CSV)
    write_markdown(report, OUTPUT_MD)
    print("S9 sensitivity analysis completed")
    print(f"Cases: {report['case_count']}")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"CSV : {OUTPUT_CSV}")
    print(f"MD  : {OUTPUT_MD}")


if __name__ == "__main__":
    main()
