
from preference_extractor.signal_detector.runtime import (
    PreferenceSignalDetector
)


def test_model_loading():

    detector = PreferenceSignalDetector()

    assert detector is not None

    assert detector.model is not None

    assert detector.tokenizer is not None

    assert detector.threshold > 0

    assert detector.threshold < 1