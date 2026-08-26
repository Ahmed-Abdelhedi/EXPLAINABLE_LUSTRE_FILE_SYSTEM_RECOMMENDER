from preference_extractor.layer2.labels import (
    PreferenceDimension,
    PreferenceLevel,
    ResolutionSource,
    ResolutionStatus,
)
from preference_extractor.layer2.schemas import (
    DimensionPreferenceResult,
    PreferenceExtractionResult,
)


def test_no_signal_is_not_very_low():
    result = PreferenceExtractionResult(
        text="Performance is essential.",
        dimensions={
            PreferenceDimension.COST:
                DimensionPreferenceResult(
                    dimension=PreferenceDimension.COST,
                    status=ResolutionStatus.NO_SIGNAL,
                    source=ResolutionSource.TRANSFORMER,
                ),
            PreferenceDimension.POWER:
                DimensionPreferenceResult(
                    dimension=PreferenceDimension.POWER,
                    status=ResolutionStatus.NO_SIGNAL,
                    source=ResolutionSource.TRANSFORMER,
                ),
            PreferenceDimension.PERFORMANCE:
                DimensionPreferenceResult(
                    dimension=PreferenceDimension.PERFORMANCE,
                    status=ResolutionStatus.RESOLVED,
                    source=ResolutionSource.TRANSFORMER,
                    level=PreferenceLevel.VERY_HIGH,
                ),
            PreferenceDimension.RELIABILITY:
                DimensionPreferenceResult(
                    dimension=PreferenceDimension.RELIABILITY,
                    status=ResolutionStatus.NO_SIGNAL,
                    source=ResolutionSource.TRANSFORMER,
                ),
        },
    )

    fields = result.to_requirement_fields()

    assert fields["performance_priority"] == "VERY_HIGH"
    assert fields["power_priority"] == "NO_SIGNAL"
    assert fields["power_priority"] != "VERY_LOW"
