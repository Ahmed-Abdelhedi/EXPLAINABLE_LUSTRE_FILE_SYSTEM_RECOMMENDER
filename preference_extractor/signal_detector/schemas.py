
from dataclasses import dataclass


@dataclass
class PreferenceSignalResult:
    """
    Output contract of Layer 1.

    Layer 1 only answers:
    "Does this text contain a user preference signal?"
    """

    has_preference_signal: bool
    label: str
    probability: float
    threshold: float

    def to_dict(self) -> dict:
        return {
            "has_preference_signal": self.has_preference_signal,
            "label": self.label,
            "probability": self.probability,
            "threshold": self.threshold,
        }