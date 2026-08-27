from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

EXPECTED_ARTIFACT_FILENAME = "categorical_boolean_xlmr_v1_FROZEN.zip"
EXPECTED_ARTIFACT_SHA256 = "fcda293810e1ca735ea1744b8278a2f41dc1be8b2cdb1c4c10c3ebc66da11ff3"

EXPECTED_ARCHITECTURE = "CategoricalBooleanMultiTaskXLMR"
EXPECTED_BASE_MODEL = "FacebookAI/xlm-roberta-base"
EXPECTED_MAX_LENGTH = 128
EXPECTED_HA_NUM_LABELS = 4
EXPECTED_ACCESS_NUM_LABELS = 4
EXPECTED_LOSS = "CE_HA + CE_ACCESS"
EXPECTED_CALIBRATION_STATUS = "CALIBRATED_ON_VALIDATION_BEFORE_TEST"
EXPECTED_CALIBRATION_VERSION = "exact_2d_suffix_precision_constrained_v1_20260826"
EXPECTED_TARGET_PRECISION = 0.99


@dataclass(frozen=True)
class FrozenArtifactContract:
    filename: str = EXPECTED_ARTIFACT_FILENAME
    sha256: str = EXPECTED_ARTIFACT_SHA256
    architecture: str = EXPECTED_ARCHITECTURE
    base_model_name: str = EXPECTED_BASE_MODEL
    max_length: int = EXPECTED_MAX_LENGTH
    ha_num_labels: int = EXPECTED_HA_NUM_LABELS
    access_num_labels: int = EXPECTED_ACCESS_NUM_LABELS
    loss: str = EXPECTED_LOSS
    calibration_status: str = EXPECTED_CALIBRATION_STATUS
    calibration_version: str = EXPECTED_CALIBRATION_VERSION
    target_precision: float = EXPECTED_TARGET_PRECISION

    @staticmethod
    def default_artifact_path() -> Path:
        return (
            Path(__file__).resolve().parent
            / "artifacts"
            / EXPECTED_ARTIFACT_FILENAME
        )

    def validate_model_config(self, payload: Mapping[str, Any]) -> None:
        expected = {
            "architecture": self.architecture,
            "base_model_name": self.base_model_name,
            "max_length": self.max_length,
            "ha_num_labels": self.ha_num_labels,
            "access_num_labels": self.access_num_labels,
            "loss": self.loss,
        }
        mismatches = {
            key: {"expected": value, "actual": payload.get(key)}
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "Frozen model_config contract mismatch: "
                + json.dumps(mismatches, sort_keys=True)
            )

    def validate_calibration(self, payload: Mapping[str, Any]) -> None:
        errors = {}

        if payload.get("status") != self.calibration_status:
            errors["status"] = {
                "expected": self.calibration_status,
                "actual": payload.get("status"),
            }

        if payload.get("calibration_version") != self.calibration_version:
            errors["calibration_version"] = {
                "expected": self.calibration_version,
                "actual": payload.get("calibration_version"),
            }

        if float(payload.get("target_precision", -1.0)) != self.target_precision:
            errors["target_precision"] = {
                "expected": self.target_precision,
                "actual": payload.get("target_precision"),
            }

        if payload.get("test_used") is not False:
            errors["test_used"] = {
                "expected": False,
                "actual": payload.get("test_used"),
            }

        if payload.get("final_holdout_used") is not False:
            errors["final_holdout_used"] = {
                "expected": False,
                "actual": payload.get("final_holdout_used"),
            }

        thresholds = payload.get("thresholds")
        if not isinstance(thresholds, dict):
            errors["thresholds"] = "missing_or_invalid"
        else:
            if set(thresholds.get("ha", {})) != {
                "HA_REQUIRED",
                "HA_NOT_REQUIRED",
                "HA_MENTION_NO_COMMITMENT",
                "HA_NO_EVIDENCE",
            }:
                errors["ha_threshold_labels"] = sorted(
                    thresholds.get("ha", {}).keys()
                )
            if set(thresholds.get("access", {})) != {
                "SEQUENTIAL",
                "RANDOM",
                "MIXED",
                "NO_SUPPORTED_ACCESS_CLASS",
            }:
                errors["access_threshold_labels"] = sorted(
                    thresholds.get("access", {}).keys()
                )

        if errors:
            raise ValueError(
                "Frozen calibration contract mismatch: "
                + json.dumps(errors, sort_keys=True)
            )


FROZEN_ARTIFACT_CONTRACT = FrozenArtifactContract()
