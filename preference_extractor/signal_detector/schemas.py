from dataclasses import dataclass
from typing import Optional


@dataclass
class PreferenceSignalResult:
    """
    Output contract of Layer 1.

    Layer 1 only answers:
    "Does this text contain a user preference signal?"

    ``probability`` always stores the RAW Transformer probability for the
    PREFERENCE_SIGNAL class. If the high-precision context guard overrides the
    binary decision, the model score is intentionally preserved for audit,
    calibration analysis, and scientific ablation.

    ``decision_source``:
        - "transformer"
        - "context_guard_positive"
        - "context_guard_negative"
    """

    has_preference_signal: bool
    label: str
    probability: float
    threshold: float

    # Optional metadata keeps existing callers backward-compatible while
    # making deterministic overrides fully explainable.
    decision_source: str = "transformer"
    guard_decision: Optional[str] = None
    guard_reason: Optional[str] = None
    guard_evidence: Optional[str] = None
    transformer_has_preference_signal: Optional[bool] = None

    def to_dict(self) -> dict:
        return {
            "has_preference_signal": self.has_preference_signal,
            "label": self.label,
            "probability": self.probability,
            "threshold": self.threshold,
            "decision_source": self.decision_source,
            "guard_decision": self.guard_decision,
            "guard_reason": self.guard_reason,
            "guard_evidence": self.guard_evidence,
            "transformer_has_preference_signal":
                self.transformer_has_preference_signal,
        }
