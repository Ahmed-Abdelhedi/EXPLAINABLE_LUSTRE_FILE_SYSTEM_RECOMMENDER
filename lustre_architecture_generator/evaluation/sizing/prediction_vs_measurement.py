"""Build the S8 prediction-vs-measurement evidence for Lustre sizing.

The sizing equations estimate technical requirements. MDTest and IOR measure
performance delivered by one concrete Lustre installation. Therefore this
module does NOT fit absolute sizing constants directly to benchmark maxima.
Instead it checks directional/scaling behaviour, variability, mixed-workload
contention and the coverage of each assumption in the sizing registry.

Run from the repository root with:

    python evaluation/sizing/prediction_vs_measurement.py

Outputs are written next to this script:

- prediction_vs_measurement.json
- prediction_vs_measurement.csv
- prediction_vs_measurement.md
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent.parent
SIZING_DIR = BASE_DIR / "evaluation" / "sizing"
CONFIG_FILE = BASE_DIR / "config" / "architecture_rules.json"
REGISTRY_FILE = SIZING_DIR / "sizing_assumptions.json"
MEASUREMENTS_FILE = SIZING_DIR / "toubkal_measurements.json"
OUTPUT_JSON = SIZING_DIR / "prediction_vs_measurement.json"
OUTPUT_CSV = SIZING_DIR / "prediction_vs_measurement.csv"
OUTPUT_MD = SIZING_DIR / "prediction_vs_measurement.md"


class ComparisonError(ValueError):
    """Raised when an S8 input artifact is incomplete or inconsistent."""


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ComparisonError(
            f"Invalid JSON in {path}: line {exc.lineno}, column {exc.colno}"
        ) from exc


def scenario_map(measurements: dict[str, Any]) -> dict[str, dict[str, Any]]:
    scenarios = measurements.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ComparisonError("toubkal_measurements.json must contain scenarios")

    result: dict[str, dict[str, Any]] = {}
    for item in scenarios:
        if not isinstance(item, dict):
            raise ComparisonError("Every scenario must be a JSON object")
        scenario_id = item.get("scenario_id")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ComparisonError("Every scenario must have a scenario_id")
        if scenario_id in result:
            raise ComparisonError(f"Duplicate scenario_id: {scenario_id}")
        result[scenario_id] = item
    return result


def pct_change(value: float, reference: float) -> float:
    if reference == 0:
        raise ComparisonError("Cannot compute percentage change from zero")
    return (value / reference - 1.0) * 100.0


def ratio(value: float, reference: float) -> float:
    if reference == 0:
        raise ComparisonError("Cannot compute ratio from zero")
    return value / reference


def rounded(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def build_metadata_scaling(
    scenarios: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    ids = ["M1", "M2-2", "M2-4"]
    operations = [
        ("create", "file_creation_ops_s"),
        ("stat", "file_stat_ops_s"),
        ("remove", "file_removal_ops_s"),
    ]

    baseline = scenarios["M1"]
    baseline_clients = int(baseline["clients"])
    rows: list[dict[str, Any]] = []

    for label, key in operations:
        baseline_rate = float(baseline["metadata"][key]["mean"])
        for scenario_id in ids:
            item = scenarios[scenario_id]
            clients = int(item["clients"])
            measured_rate = float(item["metadata"][key]["mean"])
            model_scale = clients / baseline_clients
            observed_scale = measured_rate / baseline_rate
            scaling_efficiency = observed_scale / model_scale
            rows.append(
                {
                    "family": "MDT",
                    "comparison": "client_scaling",
                    "metric": f"file_{label}_ops_s",
                    "scenario_id": scenario_id,
                    "clients": clients,
                    "model_relative_scale": rounded(model_scale),
                    "observed_relative_scale": rounded(observed_scale),
                    "scaling_efficiency": rounded(scaling_efficiency),
                    "scaling_efficiency_percent": rounded(
                        scaling_efficiency * 100.0,
                        2,
                    ),
                    "measured_mean": rounded(measured_rate, 3),
                    "interpretation": (
                        "baseline"
                        if clients == baseline_clients
                        else "direction_supported_but_sublinear"
                    ),
                }
            )

    return rows


def build_ior_scaling(
    scenarios: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    ids = ["M3", "M4-2", "M4-4"]
    rows: list[dict[str, Any]] = []
    baseline = scenarios["M3"]
    baseline_clients = int(baseline["clients"])

    for access in ("write", "read"):
        baseline_mean = float(baseline[access]["mean_mib_s"])
        baseline_median = float(baseline[access]["median_mib_s"])
        for scenario_id in ids:
            item = scenarios[scenario_id]
            clients = int(item["clients"])
            model_scale = clients / baseline_clients
            measured_mean = float(item[access]["mean_mib_s"])
            measured_median = float(item[access]["median_mib_s"])
            mean_scale = measured_mean / baseline_mean
            median_scale = measured_median / baseline_median
            rows.append(
                {
                    "family": "OST",
                    "comparison": "client_scaling",
                    "metric": f"{access}_mib_s",
                    "scenario_id": scenario_id,
                    "clients": clients,
                    "model_relative_scale": rounded(model_scale),
                    "observed_mean_relative_scale": rounded(mean_scale),
                    "observed_median_relative_scale": rounded(median_scale),
                    "mean_scaling_efficiency_percent": rounded(
                        100.0 * mean_scale / model_scale,
                        2,
                    ),
                    "median_scaling_efficiency_percent": rounded(
                        100.0 * median_scale / model_scale,
                        2,
                    ),
                    "measured_mean_mib_s": rounded(measured_mean, 3),
                    "measured_median_mib_s": rounded(measured_median, 3),
                    "cv_percent": rounded(float(item[access]["cv_percent"]), 2),
                    "interpretation": (
                        "baseline"
                        if clients == baseline_clients
                        else "direction_supported_nonideal_scaling"
                    ),
                }
            )

    return rows


def build_mixed_impact(
    scenarios: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    m1 = scenarios["M1"]
    m5 = scenarios["M5"]
    m6a = scenarios["M6-A"]

    metadata = {}
    for label, key in (
        ("create", "file_creation_ops_s"),
        ("stat", "file_stat_ops_s"),
        ("remove", "file_removal_ops_s"),
    ):
        control = float(m1["metadata"][key]["mean"])
        mixed = float(m5["metadata"][key]["mean"])
        metadata[label] = {
            "control_mean_ops_s": rounded(control, 3),
            "mixed_mean_ops_s": rounded(mixed, 3),
            "change_percent": rounded(pct_change(mixed, control), 2),
            "comparison_note": (
                "M1 uses 20000 files/rank while M5 uses 10000; "
                "interpret as interaction evidence, not isolated calibration."
            ),
        }

    ior = {}
    for access in ("write", "read"):
        sequential = float(m6a[access]["mean_mib_s"])
        mixed = float(m5["ior"][access]["mean_mib_s"])
        degradation = pct_change(mixed, sequential)
        recovery_factor = sequential / mixed
        ior[access] = {
            "sequential_control_mean_mib_s": rounded(sequential, 3),
            "mixed_mean_mib_s": rounded(mixed, 3),
            "change_percent": rounded(degradation, 2),
            "implied_recovery_headroom_factor": rounded(recovery_factor, 4),
        }

    return {"metadata": metadata, "ior": ior}


def build_access_pattern_impact(
    scenarios: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    sequential = scenarios["M6-A"]
    shuffled = scenarios["M6-B"]
    failed_random = scenarios["M6-C"]

    result: dict[str, Any] = {}
    for access in ("write", "read"):
        seq_mean = float(sequential[access]["mean_mib_s"])
        shuffled_mean = float(shuffled[access]["mean_mib_s"])
        result[access] = {
            "sequential_mean_mib_s": rounded(seq_mean, 3),
            "shuffled_mean_mib_s": rounded(shuffled_mean, 3),
            "change_percent": rounded(pct_change(shuffled_mean, seq_mean), 2),
            "relative_factor": rounded(ratio(shuffled_mean, seq_mean), 4),
        }

    result["fully_random_overlapping"] = {
        "status": failed_random.get("status", "FAILED"),
        "failure": failed_random.get("failure", {}),
        "accepted_as_measurement": False,
    }
    return result


def aggregate_metadata_scaling(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for clients in (2, 4):
        efficiencies = [
            float(row["scaling_efficiency_percent"])
            for row in rows
            if row["clients"] == clients
        ]
        summary[str(clients)] = {
            "median_scaling_efficiency_percent": rounded(median(efficiencies), 2),
            "min_scaling_efficiency_percent": rounded(min(efficiencies), 2),
            "max_scaling_efficiency_percent": rounded(max(efficiencies), 2),
        }
    return summary


def assumption_coverage(
    registry: dict[str, Any],
    mixed_impact: dict[str, Any],
    access_pattern: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    coverage = {
        "CAPACITY_FILL_RATIO_DEFAULT": (
            "NOT_TESTED",
            "Capacity fill policy was not exercised by MDTest/IOR performance runs.",
        ),
        "WORKLOAD_METADATA_FILE_COUNT_WEIGHT": (
            "NOT_ISOLATED",
            "File count was not varied independently of client count and scenario shape.",
        ),
        "WORKLOAD_METADATA_SMALL_FILE_WEIGHT": (
            "PARTIAL",
            "MDTest exercised a small/zero-byte file workload, but no contrasting medium/large-file metadata control was run.",
        ),
        "WORKLOAD_METADATA_CLIENT_WEIGHT": (
            "DIRECT_DIRECTIONAL",
            "M1/M2 directly show that metadata operation rates change with 1, 2 and 4 Lustre clients.",
        ),
        "WORKLOAD_DATA_CAPACITY_WEIGHT": (
            "NOT_TESTED",
            "IOR block size is not a validation of the capacity-score weight.",
        ),
        "WORKLOAD_DATA_BANDWIDTH_WEIGHT": (
            "DIRECT_DIRECTIONAL",
            "M3/M4 directly measure read/write throughput scaling across 1, 2 and 4 clients.",
        ),
        "WORKLOAD_DATA_LARGE_FILE_WEIGHT": (
            "PARTIAL",
            "Large sequential IOR workloads were measured, but no controlled small/medium-file IOR comparison was run.",
        ),
        "WORKLOAD_DOMINANCE_MARGIN": (
            "PARTIAL",
            "M5 demonstrates simultaneous MDT/OST pressure, but does not provide labelled score-boundary cases around the 0.15 margin.",
        ),
        "MDT_BASE_IOPS_PER_CLIENT": (
            "DIRECT_DIRECTIONAL_NOT_ABSOLUTE",
            "M1/M2 validate client-count direction. Absolute MDTest capacity must not be fitted directly to a workload-demand proxy.",
        ),
        "MDT_SMALL_FILE_MULTIPLIER": (
            "PARTIAL",
            "Small-file metadata behaviour is measured, but the 3.0 ratio is not isolated against medium/large controls.",
        ),
        "MDT_MEDIUM_FILE_MULTIPLIER": (
            "NOT_TESTED",
            "No medium-file MDTest control was run.",
        ),
        "MDT_LARGE_FILE_MULTIPLIER": (
            "NOT_TESTED",
            "No large-file MDTest control was run.",
        ),
        "MDT_RANDOM_ACCESS_MULTIPLIER": (
            "NOT_TESTED",
            "M6 access-pattern evidence is IOR data I/O, not MDTest metadata access.",
        ),
        "MDT_MIXED_ACCESS_MULTIPLIER": (
            "PARTIAL_NOT_ISOLATED",
            "M5 combines metadata and data activity, but does not isolate the MDT mixed-access multiplier.",
        ),
        "MDT_SEQUENTIAL_ACCESS_MULTIPLIER": (
            "NOT_TESTED",
            "Sequential IOR is an OST control, not a sequential MDTest metadata control.",
        ),
        "MDT_HIGH_METADATA_MULTIPLIER": (
            "NOT_ISOLATED",
            "No controlled high/medium/low metadata-pressure triad was run.",
        ),
        "MDT_MEDIUM_METADATA_MULTIPLIER": (
            "NOT_ISOLATED",
            "M5 supplies interaction evidence but not an isolated medium-pressure multiplier measurement.",
        ),
        "MDT_LOW_METADATA_MULTIPLIER": (
            "PARTIAL",
            "M1 is a low-scale metadata reference, but 1.0 is a normalization choice rather than an absolute benchmark fit.",
        ),
        "MDT_IOPS_SAFETY_FACTOR": (
            "PARTIAL",
            "Observed metadata variability and M5 contention support headroom, but do not uniquely determine 1.25.",
        ),
        "MDT_METADATA_BYTES_PER_FILE": (
            "NOT_TESTED",
            "The benchmark campaign measured operation rates, not bytes of MDT storage consumed per file.",
        ),
        "MDT_METADATA_CAPACITY_SAFETY_FACTOR": (
            "NOT_TESTED",
            "No MDT capacity exhaustion/operational fill experiment was performed.",
        ),
        "OST_BANDWIDTH_SAFETY_FACTOR": (
            "DIRECT_LOCAL",
            "M5 write throughput is lower than the same-size sequential control, providing direct local evidence for bandwidth headroom.",
        ),
        "OST_CAPACITY_SAFETY_FACTOR": (
            "NOT_TESTED",
            "This is a post-planning capacity policy and is not a throughput benchmark parameter.",
        ),
    }

    rows: list[dict[str, Any]] = []
    configured_ost_factor = float(config["ost_estimation"]["bandwidth_safety_factor"])
    mixed_write_factor = float(
        mixed_impact["ior"]["write"]["implied_recovery_headroom_factor"]
    )

    for item in registry["assumptions"]:
        assumption_id = item["id"]
        if assumption_id not in coverage:
            raise ComparisonError(f"Missing coverage rule for {assumption_id}")
        evidence_level, note = coverage[assumption_id]
        row = {
            "id": assumption_id,
            "config_path": item["config_path"],
            "current_value": item["current_value"],
            "toubkal_evidence_level": evidence_level,
            "evidence_note": note,
            "final_status": "PENDING_SENSITIVITY",
        }
        if assumption_id == "OST_BANDWIDTH_SAFETY_FACTOR":
            row["configured_factor"] = configured_ost_factor
            row["m5_write_implied_factor"] = mixed_write_factor
            row["factor_gap_percent"] = rounded(
                pct_change(configured_ost_factor, mixed_write_factor),
                2,
            )
        if assumption_id in {
            "MDT_RANDOM_ACCESS_MULTIPLIER",
            "MDT_MIXED_ACCESS_MULTIPLIER",
            "MDT_SEQUENTIAL_ACCESS_MULTIPLIER",
        }:
            row["ior_pattern_write_change_percent"] = access_pattern["write"][
                "change_percent"
            ]
            row["warning"] = "IOR pattern effect cannot validate an MDT multiplier."
        rows.append(row)

    return rows


def write_csv(
    metadata_scaling: list[dict[str, Any]],
    ior_scaling: list[dict[str, Any]],
    path: Path,
) -> None:
    fields = [
        "family",
        "comparison",
        "metric",
        "scenario_id",
        "clients",
        "model_relative_scale",
        "observed_relative_scale",
        "observed_mean_relative_scale",
        "observed_median_relative_scale",
        "scaling_efficiency_percent",
        "mean_scaling_efficiency_percent",
        "median_scaling_efficiency_percent",
        "measured_mean",
        "measured_mean_mib_s",
        "measured_median_mib_s",
        "cv_percent",
        "interpretation",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in metadata_scaling + ior_scaling:
            writer.writerow(row)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    md = report["metadata_scaling"]
    ior = report["ior_scaling"]
    mixed = report["mixed_workload_impact"]
    access = report["access_pattern_impact"]
    coverage = report["assumption_coverage"]

    lines = [
        "# S8 - Prediction vs Measurement",
        "",
        "## Scope and interpretation rule",
        "",
        "The sizing model estimates **required technical capacity**. Toubkal benchmarks measure **delivered performance on one concrete Lustre system**. Absolute benchmark maxima are therefore not fitted directly to sizing constants. S8 compares direction, relative scaling, variability, contention and evidence coverage.",
        "",
        "## MDT client scaling",
        "",
        "| Operation | Clients | Model relative scale | Observed relative scale | Scaling efficiency |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in md:
        lines.append(
            f"| {row['metric']} | {row['clients']} | {row['model_relative_scale']:.2f}x | "
            f"{row['observed_relative_scale']:.2f}x | {row['scaling_efficiency_percent']:.2f}% |"
        )

    lines += [
        "",
        "The direction predicted by the client-count term is observed, but delivered metadata throughput scales sublinearly. This is evidence for monotonic client influence, not a reason to replace `base_iops_per_client` with MDTest peak rates.",
        "",
        "## OST client scaling",
        "",
        "| Metric | Clients | Ideal relative scale | Mean observed | Median observed | Mean efficiency | Median efficiency |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ior:
        lines.append(
            f"| {row['metric']} | {row['clients']} | {row['model_relative_scale']:.2f}x | "
            f"{row['observed_mean_relative_scale']:.2f}x | {row['observed_median_relative_scale']:.2f}x | "
            f"{row['mean_scaling_efficiency_percent']:.2f}% | {row['median_scaling_efficiency_percent']:.2f}% |"
        )

    write_impact = mixed["ior"]["write"]
    read_impact = mixed["ior"]["read"]
    lines += [
        "",
        "The 2-node and 4-node write means are affected by a very low first iteration, so medians are retained as a robust secondary statistic.",
        "",
        "## Mixed workload impact (M5)",
        "",
        f"- Metadata create change vs M1: {mixed['metadata']['create']['change_percent']:.2f}%.",
        f"- Metadata stat change vs M1: {mixed['metadata']['stat']['change_percent']:.2f}%.",
        f"- Metadata remove change vs M1: {mixed['metadata']['remove']['change_percent']:.2f}%.",
        f"- IOR write change vs same-size sequential control M6-A: {write_impact['change_percent']:.2f}%.",
        f"- IOR read change vs same-size sequential control M6-A: {read_impact['change_percent']:.2f}%.",
        f"- The M5 write result implies a local recovery/headroom factor of about {write_impact['implied_recovery_headroom_factor']:.3f}x; current OST factor is {report['configured_ost_bandwidth_safety_factor']:.2f}x.",
        "",
        f"The configured OST bandwidth factor is now {report['configured_ost_bandwidth_safety_factor']:.2f}x. S8 evidence is local/system-specific and final calibration status is assigned separately in S9-B.",
        "",
        "## Access-pattern evidence (M6)",
        "",
        f"- Shuffled vs sequential write change: {access['write']['change_percent']:.2f}%.",
        f"- Shuffled vs sequential read change: {access['read']['change_percent']:.2f}%.",
        "- M6-C fully random overlapping I/O is rejected as a measurement because IOR 4.1.0+dev failed with SIGFPE / integer divide-by-zero.",
        "- These are IOR/OST observations. They **do not validate MDT access multipliers**.",
        "",
        "## Assumption evidence coverage after Toubkal",
        "",
        "| Assumption | Evidence level | S8 status |",
        "|---|---|---|",
    ]
    for row in coverage:
        lines.append(
            f"| `{row['id']}` | {row['toubkal_evidence_level']} | {row['final_status']} |"
        )

    lines += [
        "",
        "## S8 decision",
        "",
        "1. Do not change the sizing configuration yet.",
        "2. Preserve Toubkal as empirical evidence for trends, variability and mixed-workload contention.",
        "3. Do not claim validation for constants whose required controlled benchmark was not run.",
        "4. Continue to S9 sensitivity analysis; only then assign `SUPPORTED`, `CALIBRATED`, `POLICY_CHOICE` or `NEEDS_REVISION`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report() -> dict[str, Any]:
    config = load_json(CONFIG_FILE)
    registry = load_json(REGISTRY_FILE)
    measurements = load_json(MEASUREMENTS_FILE)
    scenarios = scenario_map(measurements)

    required_ids = {"M1", "M2-2", "M2-4", "M3", "M4-2", "M4-4", "M5", "M6-A", "M6-B", "M6-C"}
    missing = sorted(required_ids - set(scenarios))
    if missing:
        raise ComparisonError("Missing Toubkal scenarios: " + ", ".join(missing))

    metadata_scaling = build_metadata_scaling(scenarios)
    ior_scaling = build_ior_scaling(scenarios)
    mixed_impact = build_mixed_impact(scenarios)
    access_pattern = build_access_pattern_impact(scenarios)
    coverage = assumption_coverage(
        registry,
        mixed_impact,
        access_pattern,
        config,
    )

    report = {
        "artifact_version": "1.0",
        "stage": "S8_prediction_vs_measurement",
        "config_version": config.get("version"),
        "comparison_policy": {
            "absolute_fit_allowed": False,
            "reason": (
                "Sizing equations estimate required demand/headroom while Toubkal "
                "benchmarks measure delivered capacity of a concrete filesystem."
            ),
            "allowed_uses": [
                "directional validation",
                "relative scaling comparison",
                "variability characterization",
                "mixed-workload contention evidence",
                "assumption evidence coverage",
            ],
        },
        "metadata_scaling": metadata_scaling,
        "metadata_scaling_summary": aggregate_metadata_scaling(metadata_scaling),
        "ior_scaling": ior_scaling,
        "mixed_workload_impact": mixed_impact,
        "access_pattern_impact": access_pattern,
        "configured_ost_bandwidth_safety_factor": float(
            config["ost_estimation"]["bandwidth_safety_factor"]
        ),
        "assumption_coverage": coverage,
        "s8_decision": {
            "modify_config_now": False,
            "next_stage": "S9_sensitivity_analysis",
            "final_assumption_statuses_assigned": False,
        },
    }
    return report


def main() -> None:
    report = build_report()
    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(
        report["metadata_scaling"],
        report["ior_scaling"],
        OUTPUT_CSV,
    )
    write_markdown(report, OUTPUT_MD)

    print("S8 prediction-vs-measurement completed")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"CSV : {OUTPUT_CSV}")
    print(f"MD  : {OUTPUT_MD}")


if __name__ == "__main__":
    main()
