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
    "training_history.json",
    "independent_holdout_metrics_FIRST_RUN.json",
    "validation_metrics.json",
    "independent_holdout_per_family_FIRST_RUN.json",
    "thresholds.json",
    "dataset_protocol.json",
    "training_config.json",
    "legacy_test_metrics.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.txt",
    "tokenizer/special_tokens_map.json",
    "model/model.safetensors",
    "model/config.json",
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
            prefix="pref_v2_verify_"
        ) as tmp:
            tmp_path = Path(tmp)
            archive.extractall(tmp_path)

            roots = [
                candidate
                for candidate in tmp_path.iterdir()
                if candidate.is_dir()
            ]

            if len(roots) == 1:
                yield roots[0]
            else:
                yield tmp_path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def confusion_sum(metrics: dict) -> int:
    return sum(
        int(metrics[key])
        for key in (
            "tn",
            "fp",
            "fn",
            "tp",
        )
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
            "preference_signal_detector_v2_artifact_audit.json"
        ),
    )
    args = parser.parse_args()

    checks = []

    def check(name, condition, observed, expected):
        checks.append(
            {
                "name": name,
                "pass": bool(condition),
                "observed": observed,
                "expected": expected,
            }
        )

        if not condition:
            raise AssertionError(
                f"{name}: observed={observed!r}, "
                f"expected={expected!r}"
            )

    with artifact_root(
        args.artifact
    ) as root:
        missing = [
            rel
            for rel in REQUIRED_FILES
            if not (
                root / rel
            ).exists()
        ]

        check(
            "required_files_present",
            not missing,
            missing,
            [],
        )

        labels = json.loads(
            (
                root / "labels.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        config = json.loads(
            (
                root / "model/config.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        thresholds = json.loads(
            (
                root / "thresholds.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        training = json.loads(
            (
                root / "training_config.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        validation = json.loads(
            (
                root / "validation_metrics.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        holdout = json.loads(
            (
                root
                / "independent_holdout_metrics_FIRST_RUN.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        legacy = json.loads(
            (
                root
                / "legacy_test_metrics.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        protocol = json.loads(
            (
                root / "dataset_protocol.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        check(
            "labels_match_model_config",
            labels["id2label"]
            == config["id2label"],
            labels["id2label"],
            config["id2label"],
        )

        check(
            "training_max_length_128",
            int(
                training["max_length"]
            )
            == 128,
            training["max_length"],
            128,
        )

        check(
            "training_sizes",
            {
                "train": training[
                    "train_size"
                ],
                "val": training[
                    "val_size"
                ],
                "holdout": training[
                    "independent_holdout_size"
                ],
                "legacy": training[
                    "legacy_test_size"
                ],
            }
            == {
                "train": 26000,
                "val": 5200,
                "holdout": 1200,
                "legacy": 4000,
            },
            {
                "train": training[
                    "train_size"
                ],
                "val": training[
                    "val_size"
                ],
                "holdout": training[
                    "independent_holdout_size"
                ],
                "legacy": training[
                    "legacy_test_size"
                ],
            },
            {
                "train": 26000,
                "val": 5200,
                "holdout": 1200,
                "legacy": 4000,
            },
        )

        check(
            "validation_confusion_sum",
            confusion_sum(
                validation
            )
            == 5200,
            confusion_sum(
                validation
            ),
            5200,
        )

        check(
            "holdout_confusion_sum",
            confusion_sum(
                holdout
            )
            == 1200,
            confusion_sum(
                holdout
            ),
            1200,
        )

        check(
            "legacy_confusion_sum",
            confusion_sum(
                legacy
            )
            == 4000,
            confusion_sum(
                legacy
            ),
            4000,
        )

        check(
            "stored_threshold",
            float(
                thresholds[
                    "preference_signal_probability_threshold"
                ]
            )
            == 0.05,
            thresholds[
                "preference_signal_probability_threshold"
            ],
            0.05,
        )

        check(
            "threshold_calibrated_on_validation",
            thresholds[
                "calibrated_on"
            ]
            == "preference_signal_val_v2",
            thresholds[
                "calibrated_on"
            ],
            "preference_signal_val_v2",
        )

        check(
            "holdout_policy_present",
            "FIRST_RUN"
            in protocol[
                "holdout_policy"
            ]
            or "first"
            in protocol[
                "holdout_policy"
            ].lower(),
            protocol[
                "holdout_policy"
            ],
            "checkpoint and threshold frozen before first holdout evaluation",
        )

        hashes = {
            rel: sha256(
                root / rel
            )
            for rel in (
                "model/model.safetensors",
                "model/config.json",
                "thresholds.json",
                "labels.json",
            )
        }

        payload = {
            "artifact": str(
                args.artifact
            ),
            "all_checks_pass": all(
                item["pass"]
                for item in checks
            ),
            "checks_passed": sum(
                int(
                    item["pass"]
                )
                for item in checks
            ),
            "checks_total": len(
                checks
            ),
            "hashes_sha256": hashes,
            "stored_metrics": {
                "validation": validation,
                "independent_holdout_FIRST_RUN": holdout,
                "legacy_test": legacy,
            },
            "stored_threshold": thresholds,
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

    print(
        f"\nSaved: {args.output}"
    )


if __name__ == "__main__":
    main()
