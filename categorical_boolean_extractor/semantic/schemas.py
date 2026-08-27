from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

class SemanticDecisionStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    ABSTAIN = "ABSTAIN"

@dataclass(frozen=True)
class SemanticHeadOutput:
    probabilities: Dict[str, float]
    top_label: str
    top_probability: float
    second_probability: float
    margin: float

    def to_dict(self):
        return {
            "probabilities": dict(self.probabilities),
            "top_label": self.top_label,
            "top_probability": self.top_probability,
            "second_probability": self.second_probability,
            "margin": self.margin,
        }

@dataclass(frozen=True)
class SemanticDecision:
    status: SemanticDecisionStatus
    label: Optional[str]
    head_output: SemanticHeadOutput
    probability_threshold: Optional[float]
    margin_threshold: Optional[float]
    reason: str

    @property
    def accepted(self) -> bool:
        return self.status == SemanticDecisionStatus.ACCEPTED

    def to_dict(self):
        return {
            "status": self.status.value,
            "label": self.label,
            "probability_threshold": self.probability_threshold,
            "margin_threshold": self.margin_threshold,
            "reason": self.reason,
            "head_output": self.head_output.to_dict(),
        }
