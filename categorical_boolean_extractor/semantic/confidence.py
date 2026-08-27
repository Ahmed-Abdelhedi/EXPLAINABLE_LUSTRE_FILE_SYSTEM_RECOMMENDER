from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping

from .schemas import (
    SemanticDecision,
    SemanticDecisionStatus,
    SemanticHeadOutput,
)


@dataclass(frozen=True)
class ClassThreshold:
    min_top_probability: float
    min_margin: float
    validation_precision: float
    validation_accepted: int


class CalibratedConfidencePolicy:
    """
    Frozen validation-calibrated confidence policy.

    Missing threshold => abstain.
    There are no arbitrary production defaults.

    Calibration JSON may also contain audit-only fields such as
    `predicted_validation_cases` and `search`.  They are deliberately ignored
    by the runtime and do not alter the frozen decision thresholds.
    """

    def __init__(
        self,
        *,
        thresholds: Mapping[str, Mapping[str, ClassThreshold]],
        target_precision: float,
        calibration_version: str,
    ) -> None:
        self.thresholds = {
            head: dict(values)
            for head, values in thresholds.items()
        }
        self.target_precision = float(target_precision)
        self.calibration_version = str(calibration_version)

    @classmethod
    def from_json(cls, path):
        payload = json.loads(
            Path(path).read_text(encoding="utf-8")
        )

        thresholds: Dict[str, Dict[str, ClassThreshold]] = {}

        for head, values in payload["thresholds"].items():
            thresholds[head] = {
                label: ClassThreshold(
                    min_top_probability=float(
                        threshold["min_top_probability"]
                    ),
                    min_margin=float(
                        threshold["min_margin"]
                    ),
                    validation_precision=float(
                        threshold["validation_precision"]
                    ),
                    validation_accepted=int(
                        threshold["validation_accepted"]
                    ),
                )
                for label, threshold in values.items()
            }

        return cls(
            thresholds=thresholds,
            target_precision=float(payload["target_precision"]),
            calibration_version=str(payload["calibration_version"]),
        )

    def decide(
        self,
        *,
        head: str,
        output: SemanticHeadOutput,
    ) -> SemanticDecision:
        threshold = self.thresholds.get(
            head, {}
        ).get(output.top_label)

        if threshold is None:
            return SemanticDecision(
                status=SemanticDecisionStatus.ABSTAIN,
                label=None,
                head_output=output,
                probability_threshold=None,
                margin_threshold=None,
                reason="NO_CALIBRATED_THRESHOLD_FOR_PREDICTED_CLASS",
            )

        accepted = (
            output.top_probability
            >= threshold.min_top_probability
            and output.margin
            >= threshold.min_margin
        )

        return SemanticDecision(
            status=(
                SemanticDecisionStatus.ACCEPTED
                if accepted
                else SemanticDecisionStatus.ABSTAIN
            ),
            label=output.top_label if accepted else None,
            head_output=output,
            probability_threshold=threshold.min_top_probability,
            margin_threshold=threshold.min_margin,
            reason=(
                "CALIBRATED_CONFIDENCE_ACCEPT"
                if accepted
                else "CALIBRATED_CONFIDENCE_ABSTAIN"
            ),
        )
