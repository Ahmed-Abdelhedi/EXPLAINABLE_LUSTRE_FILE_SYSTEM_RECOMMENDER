from preference_extractor.layer2.confidence import (
    ConfidencePolicy,
    selective_decision,
)
from preference_extractor.layer2.labels import (
    PreferenceLevel,
    ResolutionStatus,
)


POLICY = ConfidencePolicy(
    presence_negative_max=0.05,
    presence_positive_min=0.95,
    intensity_min_confidence=0.70,
)


def test_high_confidence_no_signal():
    status, level, _, _ = selective_decision(
        0.01,
        [0.9, 0.8, 0.7, 0.6],
        POLICY,
    )

    assert status == ResolutionStatus.NO_SIGNAL
    assert level is None


def test_presence_abstention_routes_to_fallback():
    status, level, _, _ = selective_decision(
        0.50,
        [0.9, 0.8, 0.7, 0.6],
        POLICY,
    )

    assert status == ResolutionStatus.NEEDS_FALLBACK
    assert level is None


def test_very_high_decode():
    status, level, _, _ = selective_decision(
        0.99,
        [0.99, 0.98, 0.97, 0.96],
        POLICY,
    )

    assert status == ResolutionStatus.RESOLVED
    assert level == PreferenceLevel.VERY_HIGH


def test_relative_only_never_guesses_absolute_level():
    status, level, _, _ = selective_decision(
        0.99,
        [0.99, 0.90, 0.40, 0.10],
        POLICY,
        relative_only=True,
    )

    assert status == ResolutionStatus.RELATIVE_ONLY
    assert level is None
