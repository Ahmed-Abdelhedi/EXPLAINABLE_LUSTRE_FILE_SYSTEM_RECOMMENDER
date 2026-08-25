"""Finalize Lustre sizing assumptions after S8/S9 evaluation.

This stage combines:
- S8 Toubkal prediction-vs-measurement evidence,
- S9 1200-case sensitivity analysis,
- explicit design semantics for policy/normalization choices.

It deliberately calibrates only constants for which the available evidence
supports a numerical change. Other heuristic constants remain explicit
POLICY_CHOICE values instead of being mislabeled as empirically validated.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent.parent
SIZING_DIR = BASE_DIR / "evaluation" / "sizing"
CONFIG_FILE = BASE_DIR / "config" / "architecture_rules.json"
REGISTRY_FILE = SIZING_DIR / "sizing_assumptions.json"
S8_FILE = SIZING_DIR / "prediction_vs_measurement.json"
S9_FILE = SIZING_DIR / "sensitivity_analysis.json"
OUTPUT_JSON = SIZING_DIR / "calibration_decisions.json"
OUTPUT_CSV = SIZING_DIR / "calibration_decisions.csv"
OUTPUT_MD = SIZING_DIR / "calibration_decisions.md"

ALLOWED_STATUSES = {
    "SUPPORTED",
    "CALIBRATED",
    "POLICY_CHOICE",
    "NEEDS_REVISION",
}

# S9-B policy: only a numerically identifiable quantity is calibrated.
# Toubkal M5 observed an 18.73% sequential-write degradation under a mixed
# metadata+data workload, implying ~1.230x recovery headroom. We round upward
# to 1.25 instead of overfitting the exact Toubkal value.
FINAL_STATUS = {
    "CAPACITY_FILL_RATIO_DEFAULT": "POLICY_CHOICE",
    "WORKLOAD_METADATA_FILE_COUNT_WEIGHT": "POLICY_CHOICE",
    "WORKLOAD_METADATA_SMALL_FILE_WEIGHT": "POLICY_CHOICE",
    "WORKLOAD_METADATA_CLIENT_WEIGHT": "POLICY_CHOICE",
    "WORKLOAD_DATA_CAPACITY_WEIGHT": "POLICY_CHOICE",
    "WORKLOAD_DATA_BANDWIDTH_WEIGHT": "POLICY_CHOICE",
    "WORKLOAD_DATA_LARGE_FILE_WEIGHT": "POLICY_CHOICE",
    "WORKLOAD_DOMINANCE_MARGIN": "POLICY_CHOICE",
    "MDT_BASE_IOPS_PER_CLIENT": "POLICY_CHOICE",
    "MDT_SMALL_FILE_MULTIPLIER": "POLICY_CHOICE",
    "MDT_MEDIUM_FILE_MULTIPLIER": "POLICY_CHOICE",
    "MDT_LARGE_FILE_MULTIPLIER": "POLICY_CHOICE",
    "MDT_RANDOM_ACCESS_MULTIPLIER": "POLICY_CHOICE",
    "MDT_MIXED_ACCESS_MULTIPLIER": "POLICY_CHOICE",
    "MDT_SEQUENTIAL_ACCESS_MULTIPLIER": "POLICY_CHOICE",
    "MDT_HIGH_METADATA_MULTIPLIER": "POLICY_CHOICE",
    "MDT_MEDIUM_METADATA_MULTIPLIER": "POLICY_CHOICE",
    "MDT_LOW_METADATA_MULTIPLIER": "POLICY_CHOICE",
    "MDT_IOPS_SAFETY_FACTOR": "POLICY_CHOICE",
    "MDT_METADATA_BYTES_PER_FILE": "POLICY_CHOICE",
    "MDT_METADATA_CAPACITY_SAFETY_FACTOR": "POLICY_CHOICE",
    "OST_BANDWIDTH_SAFETY_FACTOR": "CALIBRATED",
    "OST_CAPACITY_SAFETY_FACTOR": "SUPPORTED",
}

REASONS = {
    "CAPACITY_FILL_RATIO_DEFAULT": (
        "0.80 is retained as an explicit operational fill policy. MDTest/IOR did "
        "not test capacity-fill behavior; S9 shows medium sensitivity, so the value "
        "must not be presented as a Toubkal-calibrated constant."
    ),
    "WORKLOAD_METADATA_FILE_COUNT_WEIGHT": (
        "The 0.50 classification weight was not isolated by Toubkal. S9 shows medium "
        "classification sensitivity; retain it as a transparent heuristic policy."
    ),
    "WORKLOAD_METADATA_SMALL_FILE_WEIGHT": (
        "Small-file metadata behavior was exercised only partially, without controlled "
        "medium/large metadata contrasts. Retain 0.30 as a policy weight."
    ),
    "WORKLOAD_METADATA_CLIENT_WEIGHT": (
        "M1/M2 directly support the direction of client influence, but not the exact "
        "0.20 score weight. S9 sensitivity is low, so keep the weight as policy."
    ),
    "WORKLOAD_DATA_CAPACITY_WEIGHT": (
        "Capacity contribution was not isolated by the performance campaign. The 0.40 "
        "weight remains a classification policy choice."
    ),
    "WORKLOAD_DATA_BANDWIDTH_WEIGHT": (
        "IOR supports bandwidth relevance, but not the exact 0.40 classification weight. "
        "S9 shows medium sensitivity; retain it as policy."
    ),
    "WORKLOAD_DATA_LARGE_FILE_WEIGHT": (
        "Large sequential I/O was exercised, but no controlled score-weight experiment "
        "identifies the 0.20 coefficient. Retain it as policy."
    ),
    "WORKLOAD_DOMINANCE_MARGIN": (
        "The 0.15 boundary is a decision-policy threshold, not a physical Lustre constant. "
        "S9 shows medium workload-classification sensitivity, so it remains explicit policy."
    ),
    "MDT_BASE_IOPS_PER_CLIENT": (
        "M1/M2 confirm monotonic client influence but Toubkal throughput is delivered "
        "capacity, not user demand. Therefore MDTest peak ops/s cannot be fitted to the "
        "100 IOPS/client demand proxy. High S9 sensitivity requires explicit policy labeling."
    ),
    "MDT_SMALL_FILE_MULTIPLIER": (
        "Small-file pressure is qualitatively supported, but the 3.0 magnitude was not "
        "identified by a controlled file-size sweep. Keep as a high-sensitivity policy prior."
    ),
    "MDT_MEDIUM_FILE_MULTIPLIER": (
        "No controlled medium-file MDTest scenario identifies 1.5. Keep as an explicit "
        "high-sensitivity interpolation policy between small and large reference regimes."
    ),
    "MDT_LARGE_FILE_MULTIPLIER": (
        "1.0 is retained as the large-file reference multiplier. This is a normalization "
        "choice rather than an empirically estimated physical constant."
    ),
    "MDT_RANDOM_ACCESS_MULTIPLIER": (
        "M6 random/shuffled evidence is IOR/OST evidence and cannot validate an MDT access "
        "multiplier. Keep 1.4 as policy and do not claim empirical validation."
    ),
    "MDT_MIXED_ACCESS_MULTIPLIER": (
        "M5 shows mixed-workload contention but does not isolate MDT access-pattern demand. "
        "Keep 1.15 as policy rather than fitting it from IOR."
    ),
    "MDT_SEQUENTIAL_ACCESS_MULTIPLIER": (
        "1.0 is retained as the MDT access-pattern reference baseline. M6 sequential IOR is "
        "OST evidence and does not identify an MDT factor."
    ),
    "MDT_HIGH_METADATA_MULTIPLIER": (
        "High metadata pressure was not independently varied from file/client scenario shape. "
        "Retain 1.5 as an explicit policy prior; S9 shows material numeric impact."
    ),
    "MDT_MEDIUM_METADATA_MULTIPLIER": (
        "Medium metadata pressure was not independently isolated. Retain 1.2 as a policy "
        "multiplier with high sensitivity disclosure."
    ),
    "MDT_LOW_METADATA_MULTIPLIER": (
        "1.0 is retained as the low-pressure reference baseline; this is a normalization/policy "
        "anchor, not a measured Lustre constant."
    ),
    "MDT_IOPS_SAFETY_FACTOR": (
        "M5 shows heterogeneous metadata contention: create/remove change little while stat is "
        "much more variable. The evidence does not identify a single MDT headroom factor, so "
        "1.25 is retained as a conservative engineering policy rather than calibrated."
    ),
    "MDT_METADATA_BYTES_PER_FILE": (
        "4096 bytes/file is an engineering footprint proxy. The Toubkal campaign did not measure "
        "per-file MDT space consumption, so it remains a clearly labeled policy approximation."
    ),
    "MDT_METADATA_CAPACITY_SAFETY_FACTOR": (
        "2.0 is capacity headroom around the metadata-footprint proxy and was not identified by "
        "MDTest. Retain as an explicit conservative policy."
    ),
    "OST_BANDWIDTH_SAFETY_FACTOR": (
        "Calibrated from 1.20 to 1.25. M5 write throughput fell 18.73% relative to the same-size "
        "sequential control M6-A, implying about 1.230x recovery headroom. 1.25 is a conservative "
        "rounded value and avoids fitting the exact system-specific observation."
    ),
    "OST_CAPACITY_SAFETY_FACTOR": (
        "1.0 is supported by the sizing design: growth and free-space headroom are already applied "
        "in planned capacity through planning horizon and target fill ratio. Keeping this factor at "
        "1.0 avoids silently double-counting capacity margin."
    ),
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def value_at_path(config: dict[str, Any], dotted_path: str) -> Any:
    value: Any = config
    for part in dotted_path.split("."):
        value = value[part]
    return value


def build_decisions() -> dict[str, Any]:
    config = load_json(CONFIG_FILE)
    registry = load_json(REGISTRY_FILE)
    s8 = load_json(S8_FILE)
    s9 = load_json(S9_FILE)

    coverage = {row["id"]: row for row in s8["assumption_coverage"]}
    sensitivity = {row["id"]: row for row in s9["assumptions"]}

    registered_ids = {item["id"] for item in registry["assumptions"]}
    if registered_ids != set(FINAL_STATUS):
        missing = sorted(registered_ids - set(FINAL_STATUS))
        extra = sorted(set(FINAL_STATUS) - registered_ids)
        raise ValueError(f"Decision table mismatch. missing={missing}, extra={extra}")

    mixed_write = s8["mixed_workload_impact"]["ior"]["write"]
    implied = float(mixed_write["implied_recovery_headroom_factor"])
    final_ost_factor = float(config["ost_estimation"]["bandwidth_safety_factor"])
    if final_ost_factor != 1.25:
        raise ValueError(
            "S9-B expects ost_estimation.bandwidth_safety_factor=1.25 in the frozen configuration."
        )
    if final_ost_factor < implied:
        raise ValueError(
            "Calibrated OST bandwidth safety factor is below measured recovery headroom."
        )

    rows: list[dict[str, Any]] = []
    for assumption in registry["assumptions"]:
        assumption_id = assumption["id"]
        status = FINAL_STATUS[assumption_id]
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"Invalid final status for {assumption_id}: {status}")

        final_value = value_at_path(config, assumption["config_path"])
        initial_value = assumption.get("initial_value", assumption["current_value"])
        rows.append(
            {
                "id": assumption_id,
                "config_path": assumption["config_path"],
                "initial_value": initial_value,
                "final_value": final_value,
                "status": status,
                "sensitivity_level": sensitivity[assumption_id]["sensitivity_level"],
                "toubkal_evidence_level": coverage[assumption_id]["toubkal_evidence_level"],
                "decision_reason": REASONS[assumption_id],
            }
        )

    return {
        "artifact_version": "1.0",
        "stage": "S9_final_assumption_decisions",
        "config_version": config["version"],
        "decision_policy": {
            "principle": (
                "Calibrate only numerically identifiable quantities; retain non-identifiable "
                "heuristics as explicit POLICY_CHOICE values instead of overclaiming validation."
            ),
            "allowed_statuses": sorted(ALLOWED_STATUSES),
            "toubkal_is_universal_ground_truth": False,
        },
        "calibration": {
            "assumption_id": "OST_BANDWIDTH_SAFETY_FACTOR",
            "old_value": 1.20,
            "measured_mixed_write_change_percent": mixed_write["change_percent"],
            "implied_recovery_headroom_factor": implied,
            "final_value": final_ost_factor,
            "relative_change_from_old_percent": round((final_ost_factor / 1.20 - 1.0) * 100.0, 4),
            "method": "conservative_round_up_from_local_recovery_factor",
        },
        "status_counts": {
            status: sum(1 for row in rows if row["status"] == status)
            for status in sorted(ALLOWED_STATUSES)
        },
        "assumptions": rows,
    }


def finalize_registry(report: dict[str, Any]) -> None:
    registry = load_json(REGISTRY_FILE)
    decisions = {row["id"]: row for row in report["assumptions"]}

    registry["registry_version"] = "2.0"
    registry["finalization"] = {
        "stage": "S9_final_assumption_decisions",
        "config_version": report["config_version"],
        "all_assumptions_finalized": True,
        "decision_artifact": "calibration_decisions.json",
    }

    for assumption in registry["assumptions"]:
        decision = decisions[assumption["id"]]
        assumption.setdefault("initial_value", assumption["current_value"])
        assumption["current_value"] = decision["final_value"]
        assumption["final_value"] = decision["final_value"]
        assumption["status"] = decision["status"]
        assumption["sensitivity_level"] = decision["sensitivity_level"]
        assumption["toubkal_evidence_level"] = decision["toubkal_evidence_level"]
        assumption["decision_reason"] = decision["decision_reason"]

    REGISTRY_FILE.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(report: dict[str, Any]) -> None:
    fields = [
        "id",
        "config_path",
        "initial_value",
        "final_value",
        "status",
        "sensitivity_level",
        "toubkal_evidence_level",
        "decision_reason",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report["assumptions"])


def write_markdown(report: dict[str, Any]) -> None:
    calibration = report["calibration"]
    lines = [
        "# S9-B - Final Sizing Assumption Decisions",
        "",
        f"Configuration version: **{report['config_version']}**.",
        "",
        "## Decision rule",
        "",
        report["decision_policy"]["principle"],
        "",
        "## Numerical calibration",
        "",
        f"- `{calibration['assumption_id']}`: **{calibration['old_value']:.2f} -> {calibration['final_value']:.2f}**.",
        f"- M5 mixed-write change: **{calibration['measured_mixed_write_change_percent']:.2f}%**.",
        f"- Implied recovery headroom: **{calibration['implied_recovery_headroom_factor']:.3f}x**.",
        f"- Config change relative to 1.20: **+{calibration['relative_change_from_old_percent']:.2f}%**.",
        "- The final 1.25 value is deliberately rounded upward rather than fitted to the exact Toubkal observation.",
        "",
        "## Final status table",
        "",
        "| Assumption | Initial | Final | Status | Sensitivity | Toubkal evidence |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in report["assumptions"]:
        lines.append(
            f"| `{row['id']}` | {row['initial_value']} | {row['final_value']} | "
            f"**{row['status']}** | {row['sensitivity_level']} | {row['toubkal_evidence_level']} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- `SUPPORTED`: retained because the current design/evidence supports the value and semantics.",
        "- `CALIBRATED`: numerical value changed using measured evidence plus a conservative engineering rule.",
        "- `POLICY_CHOICE`: retained intentionally, but must not be described as empirically measured or universal.",
        "- `NEEDS_REVISION`: reserved for assumptions contradicted by evidence or structurally unsuitable.",
        "",
        "No assumption remains `TO_VALIDATE` after this stage.",
        "",
        "## Important limitation",
        "",
        "The Toubkal campaign validates trends and local contention behavior on one Lustre installation. It does not turn heuristic workload-demand coefficients into universal hardware-performance constants.",
        "",
    ]
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    report = build_decisions()
    finalize_registry(report)
    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(report)
    write_markdown(report)

    counts = report["status_counts"]
    print("S9-B final assumption decisions completed")
    print(f"Config version: {report['config_version']}")
    print(
        "Statuses: "
        + ", ".join(f"{name}={count}" for name, count in counts.items())
    )
    print(f"JSON: {OUTPUT_JSON}")
    print(f"CSV : {OUTPUT_CSV}")
    print(f"MD  : {OUTPUT_MD}")


if __name__ == "__main__":
    main()
