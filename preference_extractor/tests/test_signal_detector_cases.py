import pytest

from preference_extractor.signal_detector.runtime import (
    PreferenceSignalDetector,
)


@pytest.fixture(scope="module")
def detector():
    return PreferenceSignalDetector()


@pytest.mark.parametrize(
    "text",
    [
        "Reliability is our main priority.",
        "Performance is more important than cost.",
        "We prefer lower power consumption.",
        "Cost is not a concern for this project.",
        "We are willing to pay more for better performance.",
        "La fiabilité est notre priorité absolue.",
        "La performance est très importante.",
        "Should the final design prefer lower lifecycle cost even if this slightly reduces throughput?",
        "Ordre de classement souple : la fiabilité d'abord, puis le coût.",
    ],
)
def test_preference_signal_detected(detector, text):
    result = detector.predict(text)
    assert result.has_preference_signal is True


@pytest.mark.parametrize(
    "text",
    [
        "Maximum power is 15 kW.",
        "The system requires 500 TiB usable storage.",
        "Around 300 clients will mount Lustre.",
        "Read throughput target is 80 GB/s.",
        "The system contains 10 million files.",
        "The budget limit is 100000 USD.",
    ],
)
def test_requirement_is_not_preference(detector, text):
    result = detector.predict(text)
    assert result.has_preference_signal is False


@pytest.mark.parametrize(
    "text",
    [
        "Performance is not a concern.",
        "We do not care about power consumption.",
        "Cost should be minimized.",
        "Reliability is mandatory.",
    ],
)
def test_negative_or_priority_preference(detector, text):
    result = detector.predict(text)
    assert result.has_preference_signal is True
