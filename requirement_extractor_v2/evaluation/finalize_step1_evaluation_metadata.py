from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


ABLATION_FILES = (
    "ablation_A.json",
    "ablation_B.json",
    "ablation_C_semantic_v3_3.json",
    "ablation_D.json",
    "ablation_E_full_system.json",
)

REGRESSION_96_FILES = (
    "quantity_e2e_holdout_after_relation_fix.json",
    "quantity_e2e_holdout_after_relation_safety_fix.json",
    "quantity_e2e_holdout_after_robustness_fix.json",
    "quantity_e2e_holdout_after_llm_prompt_v3.json",
)

FIRST_RUN_96_FILE = "quantity_e2e_holdout_v1_FIRST_RUN.json"

ABLATION_DATASET = (
    "requirement_extractor_v2/evaluation/datasets/"
    "v2_independent_end_to_end_benchmark_v1.jsonl"
)

HOLDOUT_DATASET = (
    "requirement_extractor_v2/evaluation/datasets/"
    "quantity_e2e_holdout_v1.jsonl"
)

# IMPORTANT:
# This note intentionally contains only the actual 300-message benchmark count.
# It does not mention the historical wrong count, because the metadata audit
# correctly interprets every "96-message" occurrence inside a 300-message
# artifact as an inconsistency.
ABLATION_NOTE = (
    "Ablation A-E uses the same fixed 300-message broad evaluation benchmark "
    "(v2_independent_end_to_end_benchmark_v1.jsonl) for all configurations. "
    "ConversationScopeResolver and QuantityScanner are held constant in the "
    "historical ablation design. Configurations A-D expose accepted candidate "
    "links after canonical unit normalization but without final "
    "DeterministicVerifier rejection. Configuration E uses the real "
    "VerifiedRequirementPipeline and exposes only VERIFIED outputs. "
    "This is a metadata-only clarification; stored metrics and per-message "
    "details are unchanged."
)

REGRESSION_NOTE = (
    "Regression execution on the preserved 96-message quantity-only E2E "
    "holdout after implementation changes informed by earlier holdout failures. "
    "This run must be reported as regression_after_holdout_inspection, not as "
    "a new independent holdout. Metrics and predictions are unchanged by this "
    "metadata finalization."
)

FIRST_RUN_NOTE = (
    "Preserved first execution of the 96-message quantity-only end-to-end "
    "holdout. This is the independent_first_run result. Later runs on the same "
    "96 messages after inspecting failures must be reported as regression runs."
)


def load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(
        path.read_text(encoding="utf-8")
    )
    if not isinstance(data, dict):
        raise ValueError(
            f"{path} does not contain a JSON object."
        )
    return data


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()


def stored_count(data: Dict[str, Any]) -> Optional[int]:
    containers: List[Dict[str, Any]] = [data]

    metrics = data.get("metrics")
    if isinstance(metrics, dict):
        containers.append(metrics)

    for container in containers:
        for key in ("n_messages", "n_cases"):
            value = container.get(key)
            if isinstance(value, int):
                return value

    return None


def preserve_payload_hashes(
    data: Dict[str, Any],
) -> Dict[str, Optional[str]]:
    return {
        "metrics_sha256": (
            canonical_hash(data["metrics"])
            if "metrics" in data
            else None
        ),
        "details_sha256": (
            canonical_hash(data["details"])
            if "details" in data
            else None
        ),
    }


def backup_once(path: Path) -> Optional[Path]:
    backup = path.with_suffix(
        path.suffix + ".pre_step1_6_v2.bak"
    )

    if backup.exists():
        return None

    backup.write_bytes(
        path.read_bytes()
    )

    return backup


def verify_payload_unchanged(
    path: Path,
    before_hashes: Dict[str, Optional[str]],
) -> Dict[str, Optional[str]]:
    reloaded = load_json(path)
    after_hashes = preserve_payload_hashes(
        reloaded
    )

    if before_hashes != after_hashes:
        raise RuntimeError(
            f"{path}: metrics/details changed during metadata-only correction."
        )

    return after_hashes


def finalize_ablation(
    path: Path,
) -> Dict[str, Any]:
    data = load_json(path)

    count = stored_count(data)
    if count != 300:
        raise ValueError(
            f"{path}: expected n_messages=300, found {count!r}. "
            "Refusing to modify."
        )

    before_hashes = preserve_payload_hashes(
        data
    )
    backup = backup_once(path)

    # Replace the whole note so no stale "96-message" wording survives.
    data["methodology_note"] = ABLATION_NOTE

    data["evaluation_metadata"] = {
        "dataset_path": ABLATION_DATASET,
        "dataset_name":
            "v2_independent_end_to_end_benchmark_v1.jsonl",
        "n_messages": 300,
        "evaluation_role":
            "broad_ablation_benchmark",
        "run_kind": "evaluation",
        "metadata_revision":
            "step1_6_v2_clean_300_message_ablation_metadata",
        "metrics_recomputed": False,
    }

    write_json(path, data)

    after_hashes = verify_payload_unchanged(
        path,
        before_hashes,
    )

    return {
        "file": str(path),
        "kind": "ablation_300_metadata_fix_v2",
        "count": count,
        "backup_created": (
            None if backup is None else str(backup)
        ),
        "payload_hashes_before": before_hashes,
        "payload_hashes_after": after_hashes,
        "metrics_and_details_unchanged": True,
    }


def finalize_regression_96(
    path: Path,
) -> Dict[str, Any]:
    data = load_json(path)

    count = stored_count(data)
    if count != 96:
        raise ValueError(
            f"{path}: expected n_messages=96, found {count!r}. "
            "Refusing to modify."
        )

    before_hashes = preserve_payload_hashes(
        data
    )
    backup = backup_once(path)

    data["benchmark_note"] = REGRESSION_NOTE

    data["evaluation_metadata"] = {
        "dataset_path": HOLDOUT_DATASET,
        "dataset_name":
            "quantity_e2e_holdout_v1.jsonl",
        "n_messages": 96,
        "evaluation_role":
            "quantity_e2e_holdout",
        "run_kind":
            "regression_after_holdout_inspection",
        "independent_holdout_claim": False,
        "metadata_revision":
            "step1_6_v2_holdout_regression_label",
        "metrics_recomputed": False,
    }

    write_json(path, data)

    after_hashes = verify_payload_unchanged(
        path,
        before_hashes,
    )

    return {
        "file": str(path),
        "kind": "holdout_regression_label_fix_v2",
        "count": count,
        "backup_created": (
            None if backup is None else str(backup)
        ),
        "payload_hashes_before": before_hashes,
        "payload_hashes_after": after_hashes,
        "metrics_and_details_unchanged": True,
    }


def finalize_first_run_96(
    path: Path,
) -> Dict[str, Any]:
    data = load_json(path)

    count = stored_count(data)
    if count != 96:
        raise ValueError(
            f"{path}: expected n_messages=96, found {count!r}. "
            "Refusing to modify."
        )

    before_hashes = preserve_payload_hashes(
        data
    )
    backup = backup_once(path)

    data["benchmark_note"] = FIRST_RUN_NOTE

    data["evaluation_metadata"] = {
        "dataset_path": HOLDOUT_DATASET,
        "dataset_name":
            "quantity_e2e_holdout_v1.jsonl",
        "n_messages": 96,
        "evaluation_role":
            "quantity_e2e_holdout",
        "run_kind":
            "independent_first_run",
        "independent_holdout_claim": True,
        "metadata_revision":
            "step1_6_v2_holdout_first_run_label",
        "metrics_recomputed": False,
    }

    write_json(path, data)

    after_hashes = verify_payload_unchanged(
        path,
        before_hashes,
    )

    return {
        "file": str(path),
        "kind": "preserved_first_holdout_label_v2",
        "count": count,
        "backup_created": (
            None if backup is None else str(backup)
        ),
        "payload_hashes_before": before_hashes,
        "payload_hashes_after": after_hashes,
        "metrics_and_details_unchanged": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize Requirement Extractor V2 evaluation metadata. "
            "Version 2 removes stale 96-message wording from the "
            "300-message ablation artifacts while preserving all metrics "
            "and per-message details."
        )
    )

    parser.add_argument(
        "--evaluation-dir",
        default="requirement_extractor_v2/evaluation",
    )

    parser.add_argument(
        "--report",
        default=(
            "requirement_extractor_v2/evaluation/results/"
            "step1_6_metadata_finalization_report_v2.json"
        ),
    )

    args = parser.parse_args()

    evaluation_dir = Path(
        args.evaluation_dir
    )
    results_dir = (
        evaluation_dir / "results"
    )

    changes: List[
        Dict[str, Any]
    ] = []
    missing: List[str] = []

    for filename in ABLATION_FILES:
        path = results_dir / filename

        if not path.exists():
            missing.append(str(path))
            continue

        changes.append(
            finalize_ablation(path)
        )

    for filename in REGRESSION_96_FILES:
        path = results_dir / filename

        if not path.exists():
            missing.append(str(path))
            continue

        changes.append(
            finalize_regression_96(path)
        )

    first_run_path = (
        evaluation_dir
        / FIRST_RUN_96_FILE
    )

    if first_run_path.exists():
        changes.append(
            finalize_first_run_96(
                first_run_path
            )
        )
    else:
        missing.append(
            str(first_run_path)
        )

    report = {
        "operation":
            "step1_6_metadata_finalization_v2",
        "metadata_only": True,
        "metrics_recomputed": False,
        "changed_files": len(changes),
        "missing_optional_or_expected_files":
            missing,
        "changes": changes,
        "scientific_reporting_rule": {
            "300_message_ablation":
                "broad_ablation_benchmark",
            "96_message_preserved_first_run":
                "independent_first_run",
            "96_message_post_inspection_runs":
                "regression_after_holdout_inspection",
        },
    }

    report_path = Path(
        args.report
    )
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json(
        report_path,
        report,
    )

    print()
    print("=" * 88)
    print(
        "STEP 1.6 METADATA FINALIZATION V2"
    )
    print("=" * 88)

    for item in changes:
        print(
            f"[OK] {item['file']}"
        )
        print(
            f"     kind={item['kind']}"
        )
        print(
            "     metrics/details unchanged=True"
        )

    if missing:
        print()
        print(
            "FILES NOT FOUND (not modified):"
        )
        for path in missing:
            print(
                f"- {path}"
            )

    print()
    print(
        f"Changed files: {len(changes)}"
    )
    print(
        "Metrics recomputed: False"
    )
    print(
        "Metrics/details hashes preserved: True"
    )

    print()
    print(
        "Finalization report:"
    )
    print(
        report_path
    )

    print()
    print(
        "Next command:"
    )
    print(
        "python -m requirement_extractor_v2.evaluation."
        "audit_evaluation_metadata"
    )


if __name__ == "__main__":
    main()