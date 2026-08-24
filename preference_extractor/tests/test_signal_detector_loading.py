import pytest

from preference_extractor.signal_detector.runtime import (
    PreferenceSignalDetector,
)


FINAL_THRESHOLD = 0.00039663209463469684


def test_final_v2_2_model_loading():
    detector = PreferenceSignalDetector()

    assert detector.model is not None
    assert detector.tokenizer is not None

    assert detector.model_version == "v2.2"
    assert detector.artifact_source.name == (
        "preference_signal_detector_v2_2.zip"
    )

    assert detector.threshold == pytest.approx(
        FINAL_THRESHOLD,
        rel=0.0,
        abs=1e-18,
    )

    assert 0.0 < detector.raw_artifact_threshold < 1.0
    assert detector.use_context_guard is True


def test_layer1_info_is_final():
    detector = PreferenceSignalDetector()

    info = detector.info()

    assert info["layer"] == 1
    assert info["status"] == "CLOSED_FINAL"
    assert info["model_version"] == "v2.2"
    assert info["final_threshold"] == pytest.approx(
        FINAL_THRESHOLD,
        rel=0.0,
        abs=1e-18,
    )
