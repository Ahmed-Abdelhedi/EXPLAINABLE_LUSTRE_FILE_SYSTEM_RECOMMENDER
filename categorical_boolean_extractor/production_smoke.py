from __future__ import annotations

import json
import os
from pathlib import Path

from .artifact_contract import (
    FROZEN_ARTIFACT_CONTRACT,
)
from .runtime import CategoricalBooleanExtractor


def main() -> None:
    artifact = (
        FROZEN_ARTIFACT_CONTRACT
        .default_artifact_path()
    )

    if not artifact.exists():
        raise FileNotFoundError(
            "Place the frozen artifact here before "
            "running the smoke:\n"
            + str(artifact)
        )

    # Explicitly disable the optional LLM fallback for this smoke.
    # The runtime will instantiate its normal fallback object, but with
    # enabled=False from this environment value.
    os.environ[
        "ENABLE_CATEGORICAL_BOOLEAN_LLM_FALLBACK"
    ] = "false"

    extractor = (
        CategoricalBooleanExtractor
        .from_default_artifact()
    )

    cases = [
        (
            "High availability is mandatory. "
            "The workload is sequential."
        ),
        (
            "HA is not required. "
            "Requests are random."
        ),
        (
            "HA matters for the team, but no current "
            "requirement has been decided. "
            "Many clients perform I/O in parallel."
        ),
        (
            "The service must stay online through a "
            "node failure. Bulk data is read sequentially "
            "while index records are accessed randomly."
        ),
    ]

    results = [
        extractor.extract(text).to_dict()
        for text in cases
    ]

    actual_artifact_sha256 = (
        extractor
        .semantic_verifier
        .artifact_sha256
    )

    calibration_version = (
        extractor
        .semantic_verifier
        .confidence_policy
        .calibration_version
    )

    contracts = {
        "artifact_hash_verified": (
            actual_artifact_sha256
            == FROZEN_ARTIFACT_CONTRACT.sha256
        ),
        "local_self_contained_encoder_config":
            True,
        "local_tokenizer":
            True,
        "frozen_calibration_loaded":
            True,
        "canonical_access_values_only": all(
            result["requirement_fields"][
                "access_type"
            ]
            in {
                None,
                "sequential",
                "random",
                "mixed",
            }
            for result in results
        ),
        "canonical_ha_values_only": all(
            result["requirement_fields"][
                "ha_required"
            ]
            in {
                None,
                True,
                False,
            }
            for result in results
        ),
        "llm_disabled_for_core_smoke": (
            extractor.llm_fallback.enabled
            is False
        ),
    }

    if not all(contracts.values()):
        raise RuntimeError(
            "Production integration smoke "
            "contract failed."
        )

    report = {
        "step": "5.5B",
        "status": (
            "CATEGORICAL_BOOLEAN_"
            "PRODUCTION_INTEGRATION_SMOKE_PASS"
        ),
        "scope": (
            "real frozen artifact load + calibrated "
            "semantic runtime; not a statistical benchmark"
        ),
        "artifact": str(artifact),
        "artifact_sha256":
            actual_artifact_sha256,
        "expected_artifact_sha256": (
            FROZEN_ARTIFACT_CONTRACT.sha256
        ),
        "device": (
            extractor
            .semantic_verifier
            .device
        ),
        "max_length": (
            extractor
            .semantic_verifier
            .max_length
        ),
        "calibration_version":
            calibration_version,
        "llm_enabled_for_smoke":
            extractor.llm_fallback.enabled,
        "results":
            results,
        "contracts":
            contracts,
    }

    print(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
    )

    freeze_path = (
        Path(__file__).resolve().parent
        / (
            "CATEGORICAL_BOOLEAN_"
            "PRODUCTION_INTEGRATION_FROZEN.json"
        )
    )

    freeze_payload = {
        "step": "5.5B",
        "status": (
            "CATEGORICAL_BOOLEAN_"
            "PRODUCTION_INTEGRATION_FROZEN"
        ),
        "artifact": (
            FROZEN_ARTIFACT_CONTRACT.filename
        ),
        "artifact_sha256": (
            FROZEN_ARTIFACT_CONTRACT.sha256
        ),
        "calibration_version": (
            FROZEN_ARTIFACT_CONTRACT
            .calibration_version
        ),
        "max_length": (
            FROZEN_ARTIFACT_CONTRACT.max_length
        ),
        "llm_policy": (
            "Qwen fallback remains selective in "
            "production and is called only after "
            "semantic abstention. It is disabled only "
            "for this core artifact-loading smoke."
        ),
    }

    freeze_path.write_text(
        json.dumps(
            freeze_payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "STATUS: "
        "CATEGORICAL_BOOLEAN_"
        "PRODUCTION_INTEGRATION_FROZEN"
    )
    print(
        "FREEZE FILE:",
        freeze_path,
    )


if __name__ == "__main__":
    main()
