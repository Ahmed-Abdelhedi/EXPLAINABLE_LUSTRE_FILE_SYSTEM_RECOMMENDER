from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


DEFAULT_CANDIDATE = Path(
    "preference_extractor/evaluation/results/"
    "v2_1_FINAL_guarded_threshold_candidate.json"
)

DEFAULT_OUTPUT = Path(
    "preference_extractor/evaluation/results/"
    "v2_1_FINAL_threshold_FROZEN.json"
)


def sha256_bytes(chunks) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_hash_from_artifact(path: Path) -> str:
    if path.is_dir():
        model_path = path / "model" / "model.safetensors"
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        return sha256_file(model_path)

    if path.suffix.lower() != ".zip":
        raise ValueError("Artifact must be a directory or .zip file.")

    with zipfile.ZipFile(path, "r") as archive:
        candidates = [
            name
            for name in archive.namelist()
            if name.endswith("model/model.safetensors")
        ]

        if len(candidates) != 1:
            raise RuntimeError(
                "Expected exactly one model/model.safetensors "
                f"in artifact, found {len(candidates)}: {candidates}"
            )

        with archive.open(candidates[0], "r") as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the accepted V2.1 final guarded threshold candidate "
            "before fresh-holdout generation."
        )
    )

    parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--candidate",
        type=Path,
        default=DEFAULT_CANDIDATE,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    if args.output.exists():
        raise RuntimeError(
            f"Frozen threshold already exists: {args.output}. "
            "Do not overwrite it."
        )

    candidate = json.loads(
        args.candidate.read_text(encoding="utf-8")
    )

    if candidate.get("step") != "3.1G":
        raise AssertionError(
            f"Expected Step 3.1G candidate, got {candidate.get('step')!r}"
        )

    if candidate.get("model_version") != "v2.1":
        raise AssertionError(
            f"Expected model_version v2.1, got {candidate.get('model_version')!r}"
        )

    if int(candidate.get("calibration_dataset_size", -1)) != 5800:
        raise AssertionError(
            "Final candidate must be calibrated on the 5800-case V2.1 validation."
        )

    if not candidate.get("fresh_final_holdout_v3_is_not_read", False):
        raise AssertionError(
            "Candidate does not prove fresh holdout isolation."
        )

    final = candidate["final_candidate"]
    threshold = float(final["threshold"])
    metrics = final["guarded_pipeline_metrics_on_validation"]

    if not (0.0 < threshold < 1.0):
        raise AssertionError(f"Invalid threshold: {threshold}")

    if float(metrics["precision"]) < 0.99:
        raise AssertionError(
            f"Validation precision too low to freeze: {metrics['precision']}"
        )

    if float(metrics["recall"]) < float(candidate["target_recall"]):
        raise AssertionError(
            f"Validation recall {metrics['recall']} is below target "
            f"{candidate['target_recall']}"
        )

    model_hash = model_hash_from_artifact(args.artifact)
    candidate_hash = sha256_file(args.candidate)

    payload = {
        "step": "3.1H",
        "status": "FROZEN_BEFORE_FRESH_HOLDOUT",
        "model_version": "v2.1",
        "model_sha256": model_hash,
        "artifact": str(args.artifact),
        "threshold": threshold,
        "policy": final["policy"],
        "calibrated_on": candidate["calibration_dataset"],
        "calibration_dataset_size": candidate["calibration_dataset_size"],
        "target_recall": candidate["target_recall"],
        "guarded_validation_metrics_at_frozen_threshold": metrics,
        "candidate_json_sha256": candidate_hash,
        "fresh_holdout_seen_before_freeze": False,
        "protocol": (
            "This threshold is frozen before generating or evaluating the "
            "fresh final holdout V3. Any later tuning from V3 is prohibited."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
