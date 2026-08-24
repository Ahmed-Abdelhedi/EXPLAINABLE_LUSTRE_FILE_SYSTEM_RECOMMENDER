from preference_extractor import (
    PreferenceSignalDetector,
    PreferenceSignalResult,
)


def test_public_imports():
    assert PreferenceSignalDetector is not None
    assert PreferenceSignalResult is not None
