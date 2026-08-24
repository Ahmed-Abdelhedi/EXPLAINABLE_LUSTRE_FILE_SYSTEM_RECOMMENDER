from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path


REQUIRED_FILES = (
    "labels.json",
    "thresholds.json",
    "training_config.json",
    "training_history.json",
    "validation_metrics.json",
    "targeted_validation_metrics.json",
    "targeted_validation_per_family.json",
    "regression_val_v2_metrics.json",
    "regression_holdout_v2_metrics.json",
    "legacy_test_metrics.json",
    "dataset_protocol.json",
    "model/model.safetensors",
    "model/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.txt",
    "tokenizer/special_tokens_map.json",
)


@contextmanager
def artifact_root(path: Path):
    path = path.resolve()

    if path.is_dir():
        yield path
        return

    if path.suffix.lower() != ".zip":
        raise ValueError(
            f"Artifact must be a directory or .zip file: {path}"
        )

    with zipfile.ZipFile(path, "r") as archive:
        bad = archive.testzip()

        if bad is not None:
            raise RuntimeError(
                f"Corrupted ZIP member: {bad}"
            )

        with tempfile.TemporaryDirectory(
            prefix="pref_v21_verify_"
        ) as tmp:
            tmp_path = Path(tmp)
            archive.extractall(tmp_path)

            roots = [
                candidate
                for candidate in tmp_path.iterdir()
                if candidate.is_dir()
            ]

            yield (
                roots[0]
                if len(roots) == 1
                else tmp_path
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def confusion_sum(metrics: dict) -> int:
    return sum(
        int(metrics[key])
        for key in ("tn", "fp", "fn", "tp")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "preference_extractor/evaluation/results/"
            "preference_signal_detector_v2_1_artifact_audit.json"
        ),
    )
    args = parser.parse_args()

    checks = []

    def check(name, condition, observed, expected):
        item = {
            "name": name,
            "pass": bool(condition),
            "observed": observed,
            "expected": expected,
        }
        checks.append(item)

        if not condition:
            raise AssertionError(
                f"{name}: observed={observed!r}, expected={expected!r}"
            )

    with artifact_root(args.artifact) as root:
        missing = [
            rel
            for rel in REQUIRED_FILES
            if not (root / rel).exists()
        ]

        check(
            "required_files_present",
            not missing,
            missing,
            [],
        )

        labels = json.loads(
            (root / "labels.json").read_text(encoding="utf-8")
        )
        model_config = json.loads(
            (root / "model/config.json").read_text(encoding="utf-8")
        )
        thresholds = json.loads(
            (root / "thresholds.json").read_text(encoding="utf-8")
        )
        training = json.loads(
            (root / "training_config.json").read_text(encoding="utf-8")
        )
        validation = json.loads(
            (root / "validation_metrics.json").read_text(encoding="utf-8")
        )
        targeted = json.loads(
            (root / "targeted_validation_metrics.json").read_text(
                encoding="utf-8"
            )
        )
        reg_val = json.loads(
            (root / "regression_val_v2_metrics.json").read_text(
                encoding="utf-8"
            )
        )
        reg_holdout = json.loads(
            (root / "regression_holdout_v2_metrics.json").read_text(
                encoding="utf-8"
            )
        )
        legacy = json.loads(
            (root / "legacy_test_metrics.json").read_text(
                encoding="utf-8"
            )
        )
        protocol = json.loads(
            (root / "dataset_protocol.json").read_text(encoding="utf-8")
        )

        check(
            "labels_match_model_config",
            labels["id2label"] == model_config["id2label"],
            labels["id2label"],
            model_config["id2label"],
        )

        check(
            "version_v2_1",
            training["version"] == "v2.1",
            training["version"],
            "v2.1",
        )

        check(
            "training_max_length_128",
            int(training["max_length"]) == 128,
            training["max_length"],
            128,
        )

        check(
            "dual_gpu_training_recorded",
            bool(training["multi_gpu"])
            and int(training["gpu_count"]) >= 2,
            {
                "multi_gpu": training["multi_gpu"],
                "gpu_count": training["gpu_count"],
            },
            {
                "multi_gpu": True,
                "gpu_count": ">=2",
            },
        )

        check(
            "effective_batch_size_32",
            int(training["effective_batch_size"]) == 32,
            training["effective_batch_size"],
            32,
        )

        check(
            "training_sizes",
            {
                "train": training["train_size"],
                "val": training["val_size"],
                "targeted_train": training["targeted_train_size"],
                "targeted_val": training["targeted_val_size"],
                "regression_val": training["regression_val_v2_size"],
                "regression_holdout": training[
                    "regression_holdout_v2_size"
                ],
                "legacy": training["legacy_test_size"],
            }
            == {
                "train": 28400,
                "val": 5800,
                "targeted_train": 2400,
                "targeted_val": 600,
                "regression_val": 5200,
                "regression_holdout": 1200,
                "legacy": 4000,
            },
            {
                "train": training["train_size"],
                "val": training["val_size"],
                "targeted_train": training["targeted_train_size"],
                "targeted_val": training["targeted_val_size"],
                "regression_val": training["regression_val_v2_size"],
                "regression_holdout": training[
                    "regression_holdout_v2_size"
                ],
                "legacy": training["legacy_test_size"],
            },
            {
                "train": 28400,
                "val": 5800,
                "targeted_train": 2400,
                "targeted_val": 600,
                "regression_val": 5200,
                "regression_holdout": 1200,
                "legacy": 4000,
            },
        )

        check(
            "validation_confusion_sum",
            confusion_sum(validation) == 5800,
            confusion_sum(validation),
            5800,
        )

        check(
            "targeted_validation_confusion_sum",
            confusion_sum(targeted) == 600,
            confusion_sum(targeted),
            600,
        )

        check(
            "regression_val_confusion_sum",
            confusion_sum(reg_val) == 5200,
            confusion_sum(reg_val),
            5200,
        )

        check(
            "regression_holdout_confusion_sum",
            confusion_sum(reg_holdout) == 1200,
            confusion_sum(reg_holdout),
            1200,
        )

        check(
            "legacy_confusion_sum",
            confusion_sum(legacy) == 4000,
            confusion_sum(legacy),
            4000,
        )

        check(
            "threshold_is_raw_candidate_only",
            "RAW_TRANSFORMER_VALIDATION_CANDIDATE_ONLY"
            in thresholds["status"],
            thresholds["status"],
            "RAW_TRANSFORMER_VALIDATION_CANDIDATE_ONLY",
        )

        check(
            "threshold_calibrated_on_v2_1_validation",
            thresholds["calibrated_on"]
            == "preference_signal_val_v2_1",
            thresholds["calibrated_on"],
            "preference_signal_val_v2_1",
        )

        check(
            "fresh_holdout_not_used",
            "NOT GENERATED / NOT LOADED / NOT EVALUATED"
            in protocol["fresh_final_holdout_v3"],
            protocol["fresh_final_holdout_v3"],
            "NOT GENERATED / NOT LOADED / NOT EVALUATED",
        )

        payload = {
            "artifact": str(args.artifact),
            "all_checks_pass": all(
                item["pass"]
                for item in checks
            ),
            "checks_passed": sum(
                int(item["pass"])
                for item in checks
            ),
            "checks_total": len(checks),
            "hashes_sha256": {
                "model/model.safetensors": sha256(
                    root / "model/model.safetensors"
                ),
                "model/config.json": sha256(
                    root / "model/config.json"
                ),
                "thresholds.json": sha256(
                    root / "thresholds.json"
                ),
                "labels.json": sha256(
                    root / "labels.json"
                ),
            },
            "stored_raw_threshold_candidate": thresholds,
            "stored_metrics": {
                "validation_v2_1": validation,
                "targeted_validation_v2_1": targeted,
                "regression_val_v2": reg_val,
                "regression_holdout_v2": reg_holdout,
                "legacy_test_v1_4000": legacy,
            },
            "training_config": training,
            "checks": checks,
        }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
