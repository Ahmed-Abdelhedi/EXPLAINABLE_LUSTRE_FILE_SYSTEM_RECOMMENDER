from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple, Union

from .labels import (
    ID_TO_LEVEL,
    PreferenceDimension,
    PreferenceLevel,
    ResolutionStatus,
)


EXPECTED_LAYER2_ARTIFACT_SHA256 = (
    "cdbb6f6544d4b4d96578e0901a00f46c68916ad66547800ffc4d232095298890"
)


@dataclass(frozen=True)
class ConfidencePolicy:
    """
    Backward-compatible single policy.

    Existing unit tests and callers may still use this class directly.
    Production should prefer FrozenCalibratedConfidencePolicy loaded from the
    frozen Layer-2 artifact because calibration is dimension-specific.
    """
    presence_negative_max: float = 0.05
    presence_positive_min: float = 0.95
    intensity_min_confidence: float = 0.80

    def validate(self) -> None:
        if not (
            0.0
            <= self.presence_negative_max
            < self.presence_positive_min
            <= 1.0
        ):
            raise ValueError(
                "Invalid presence confidence thresholds."
            )

        if not (
            0.0
            <= self.intensity_min_confidence
            <= 1.0
        ):
            raise ValueError(
                "Invalid intensity confidence threshold."
            )


@dataclass(frozen=True)
class FrozenCalibratedConfidencePolicy:
    """
    Dimension-specific confidence policy frozen in the trained artifact.
    """
    by_dimension: Mapping[
        PreferenceDimension,
        ConfidencePolicy,
    ]
    max_length: int = 128
    calibration_status: str = "FROZEN_BEFORE_TEST"

    def validate(self) -> None:
        if set(self.by_dimension) != set(PreferenceDimension):
            raise ValueError(
                "Frozen calibration must contain exactly four dimensions."
            )

        for policy in self.by_dimension.values():
            policy.validate()

        if int(self.max_length) <= 0:
            raise ValueError("max_length must be positive.")

    def for_dimension(
        self,
        dimension: PreferenceDimension,
    ) -> ConfidencePolicy:
        return self.by_dimension[dimension]

    @classmethod
    def from_calibration_dict(
        cls,
        calibration: Mapping[str, Any],
    ) -> "FrozenCalibratedConfidencePolicy":
        dimensions = calibration.get("dimensions")

        if not isinstance(dimensions, Mapping):
            raise ValueError(
                "calibration.json has no dimensions object."
            )

        parsed = {}

        for dimension in PreferenceDimension:
            raw = dimensions.get(dimension.value)

            if not isinstance(raw, Mapping):
                raise ValueError(
                    f"Missing calibration for {dimension.value}."
                )

            presence = raw.get("presence")
            intensity = raw.get("intensity")

            if not isinstance(presence, Mapping):
                raise ValueError(
                    f"Missing presence calibration for {dimension.value}."
                )

            if not isinstance(intensity, Mapping):
                raise ValueError(
                    f"Missing intensity calibration for {dimension.value}."
                )

            parsed[dimension] = ConfidencePolicy(
                presence_negative_max=float(
                    presence["negative_max"]
                ),
                presence_positive_min=float(
                    presence["positive_min"]
                ),
                intensity_min_confidence=float(
                    intensity["min_confidence"]
                ),
            )

        instance = cls(
            by_dimension=parsed,
            max_length=int(
                calibration.get("max_length", 128)
            ),
            calibration_status=str(
                calibration.get("status", "")
            ),
        )
        instance.validate()
        return instance

    @classmethod
    def from_artifact_zip(
        cls,
        artifact_zip: str | Path,
        *,
        verify_sha256: bool = True,
    ) -> "FrozenCalibratedConfidencePolicy":
        path = Path(artifact_zip)

        if not path.exists():
            raise FileNotFoundError(path)

        if verify_sha256:
            digest = hashlib.sha256()

            with path.open("rb") as handle:
                for chunk in iter(
                    lambda: handle.read(1024 * 1024),
                    b"",
                ):
                    digest.update(chunk)

            actual = digest.hexdigest()

            if actual != EXPECTED_LAYER2_ARTIFACT_SHA256:
                raise RuntimeError(
                    "Wrong Layer-2 artifact ZIP.\n"
                    f"Expected: {EXPECTED_LAYER2_ARTIFACT_SHA256}\n"
                    f"Actual:   {actual}"
                )

        with zipfile.ZipFile(path, "r") as archive:
            calibration = json.loads(
                archive.read("calibration.json").decode("utf-8")
            )

        return cls.from_calibration_dict(calibration)


RuntimeConfidencePolicy = Union[
    ConfidencePolicy,
    FrozenCalibratedConfidencePolicy,
]


def policy_for_dimension(
    policy: RuntimeConfidencePolicy,
    dimension: PreferenceDimension,
) -> ConfidencePolicy:
    if isinstance(
        policy,
        FrozenCalibratedConfidencePolicy,
    ):
        return policy.for_dimension(dimension)

    return policy


def model_max_length(
    policy: RuntimeConfidencePolicy,
) -> int:
    if isinstance(
        policy,
        FrozenCalibratedConfidencePolicy,
    ):
        return int(policy.max_length)

    # Historical runtime used 192. Production frozen artifact uses 128.
    return 192


def monotonic_cumulative_probabilities(
    probabilities: Sequence[float],
) -> Tuple[float, float, float, float]:
    if len(probabilities) != 4:
        raise ValueError(
            "Expected four cumulative ordinal probabilities."
        )

    values = [
        min(
            1.0,
            max(
                0.0,
                float(value),
            ),
        )
        for value in probabilities
    ]

    for index in range(
        1,
        len(values),
    ):
        values[index] = min(
            values[index],
            values[index - 1],
        )

    return tuple(values)  # type: ignore[return-value]


def ordinal_class_probabilities(
    cumulative_probabilities: Sequence[float],
) -> Tuple[float, float, float, float, float]:
    q0, q1, q2, q3 = (
        monotonic_cumulative_probabilities(
            cumulative_probabilities
        )
    )

    raw = (
        1.0 - q0,
        q0 - q1,
        q1 - q2,
        q2 - q3,
        q3,
    )

    clipped = tuple(
        max(
            0.0,
            float(value),
        )
        for value in raw
    )

    total = sum(
        clipped
    )

    if total <= 0.0:
        return (
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
        )

    return tuple(
        value / total
        for value in clipped
    )  # type: ignore[return-value]


def decode_ordinal_intensity(
    cumulative_probabilities: Sequence[float],
) -> Tuple[PreferenceLevel, float]:
    class_probabilities = (
        ordinal_class_probabilities(
            cumulative_probabilities
        )
    )

    best_id = max(
        range(5),
        key=lambda index:
            class_probabilities[index],
    )

    return (
        ID_TO_LEVEL[best_id],
        float(
            class_probabilities[
                best_id
            ]
        ),
    )


def selective_decision(
    presence_probability: float,
    cumulative_intensity_probabilities: Sequence[float],
    policy: ConfidencePolicy,
    *,
    relative_only: bool = False,
) -> Tuple[
    ResolutionStatus,
    PreferenceLevel | None,
    float | None,
    str,
]:
    policy.validate()

    presence_probability = float(
        presence_probability
    )

    if relative_only:
        return (
            ResolutionStatus.RELATIVE_ONLY,
            None,
            None,
            (
                "Pure comparative relation: preserve ordering "
                "without guessing an absolute intensity."
            ),
        )

    if (
        presence_probability
        <= policy.presence_negative_max
    ):
        return (
            ResolutionStatus.NO_SIGNAL,
            None,
            None,
            "High-confidence absence of preference signal.",
        )

    if (
        presence_probability
        < policy.presence_positive_min
    ):
        return (
            ResolutionStatus.NEEDS_FALLBACK,
            None,
            None,
            "Presence probability is inside abstention band.",
        )

    level, confidence = (
        decode_ordinal_intensity(
            cumulative_intensity_probabilities
        )
    )

    if (
        confidence
        < policy.intensity_min_confidence
    ):
        return (
            ResolutionStatus.NEEDS_FALLBACK,
            None,
            confidence,
            "Ordinal intensity confidence is too low.",
        )

    return (
        ResolutionStatus.RESOLVED,
        level,
        confidence,
        "High-confidence Transformer extraction.",
    )
