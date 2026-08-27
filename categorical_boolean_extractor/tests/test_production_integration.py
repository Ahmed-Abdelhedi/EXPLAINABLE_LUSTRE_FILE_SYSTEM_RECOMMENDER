from __future__ import annotations

import json
from pathlib import Path

import pytest

from categorical_boolean_extractor.artifact_contract import (
    EXPECTED_ARTIFACT_FILENAME,
    EXPECTED_ARTIFACT_SHA256,
    FROZEN_ARTIFACT_CONTRACT,
)
from categorical_boolean_extractor.semantic.confidence import (
    CalibratedConfidencePolicy,
)


REFERENCE = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "reference_metadata"
)


def test_expected_frozen_artifact_identity():
    assert (
        EXPECTED_ARTIFACT_FILENAME
        == "categorical_boolean_xlmr_v1_FROZEN.zip"
    )
    assert (
        EXPECTED_ARTIFACT_SHA256
        == "fcda293810e1ca735ea1744b8278a2f41dc1be8b2cdb1c4c10c3ebc66da11ff3"
    )


def test_default_artifact_path_is_inside_artifacts_directory():
    path = FROZEN_ARTIFACT_CONTRACT.default_artifact_path()

    assert path.name == EXPECTED_ARTIFACT_FILENAME
    assert path.parent.name == "artifacts"


def test_uploaded_model_config_satisfies_frozen_contract():
    payload = json.loads(
        (REFERENCE / "model_config.json").read_text(
            encoding="utf-8"
        )
    )
    FROZEN_ARTIFACT_CONTRACT.validate_model_config(
        payload
    )


def test_uploaded_calibration_satisfies_frozen_contract():
    payload = json.loads(
        (REFERENCE / "calibration.json").read_text(
            encoding="utf-8"
        )
    )
    FROZEN_ARTIFACT_CONTRACT.validate_calibration(
        payload
    )


def test_calibration_loader_ignores_audit_only_keys():
    policy = CalibratedConfidencePolicy.from_json(
        REFERENCE / "calibration.json"
    )

    ha_required = policy.thresholds["ha"]["HA_REQUIRED"]

    assert (
        ha_required.min_top_probability
        == pytest.approx(0.9974586367607117)
    )
    assert (
        ha_required.min_margin
        == pytest.approx(0.9960382422432303)
    )
    assert ha_required.validation_precision == 1.0
    assert ha_required.validation_accepted == 1500


def test_training_metadata_records_single_dual_gpu_run():
    payload = json.loads(
        (REFERENCE / "training_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["epochs_run"] == 4
    assert payload["best_epoch"] == 1
    assert payload["world_size"] == 2
    assert payload["visible_gpu_count"] == 2
    assert payload["test_used"] is False
    assert payload["final_holdout_used"] is False
    assert payload["retraining_planned"] is False


def test_best_validation_snapshot_matches_frozen_training():
    payload = json.loads(
        (
            REFERENCE
            / "best_validation_metrics.json"
        ).read_text(encoding="utf-8")
    )

    assert payload["samples"] == 6000
    assert payload["ha_macro_f1"] == 1.0
    assert payload["access_macro_f1"] == 1.0
    assert payload["composite_macro_f1"] == 1.0
    assert payload["epoch"] == 1
    assert payload["world_size"] == 2


def test_encoder_config_is_xlm_roberta_base_shape():
    payload = json.loads(
        (REFERENCE / "encoder_config.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["model_type"] == "xlm-roberta"
    assert payload["hidden_size"] == 768
    assert payload["num_hidden_layers"] == 12
    assert payload["num_attention_heads"] == 12
    assert payload["vocab_size"] == 250002


def test_tokenizer_config_is_xlm_roberta():
    payload = json.loads(
        (
            REFERENCE
            / "tokenizer_config.json"
        ).read_text(encoding="utf-8")
    )

    assert (
        payload["tokenizer_class"]
        == "XLMRobertaTokenizer"
    )
    assert payload["model_max_length"] == 512
    assert payload["pad_token"] == "<pad>"
    assert payload["unk_token"] == "<unk>"
