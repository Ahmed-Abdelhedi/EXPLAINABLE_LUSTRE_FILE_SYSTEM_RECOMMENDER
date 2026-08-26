from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional

from preference_extractor.layer2.labels import PreferenceDimension


METHOD_LINEAR_BWM = "LINEAR_BWM"
METHOD_SINGLE_ACTIVE = "SINGLE_ACTIVE"


class WeightingStatus(str, Enum):
    NO_ACTIVE_PREFERENCE = "NO_ACTIVE_PREFERENCE"
    BLOCKED_UNRESOLVED = "BLOCKED_UNRESOLVED"
    NEEDS_SINGLE_CRITERION_CONFIRMATION = (
        "NEEDS_SINGLE_CRITERION_CONFIRMATION"
    )
    NEEDS_BEST_WORST = "NEEDS_BEST_WORST"
    NEEDS_BWM_COMPARISONS = "NEEDS_BWM_COMPARISONS"
    INVALID_BWM_JUDGMENTS = "INVALID_BWM_JUDGMENTS"
    INCONSISTENT_PREFERENCES = "INCONSISTENT_PREFERENCES"
    WEIGHTS_READY = "WEIGHTS_READY"


class ConsistencyStatus(str, Enum):
    NOT_CHECKED = "NOT_CHECKED"
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True)
class BWMQuestion:
    comparison_id: str
    kind: str
    left: PreferenceDimension
    right: PreferenceDimension
    prompt: str
    scale_min: int = 1
    scale_max: int = 9

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "kind": self.kind,
            "left": self.left.value,
            "right": self.right.value,
            "prompt": self.prompt,
            "scale": {
                "min": self.scale_min,
                "max": self.scale_max,
                "meaning": (
                    "1=equal, 3=moderate, 5=strong, "
                    "7=very strong, 9=extreme; "
                    "2/4/6/8 are intermediate judgments"
                ),
            },
        }


@dataclass(frozen=True)
class BWMSolution:
    weights: Mapping[PreferenceDimension, float]
    xi_star: float
    solver: str
    solver_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": {
                dimension.value: float(weight)
                for dimension, weight in self.weights.items()
            },
            "xi_star": float(self.xi_star),
            "solver": self.solver,
            "solver_status": self.solver_status,
        }


@dataclass(frozen=True)
class ConsistencyReport:
    status: ConsistencyStatus
    violations: List[str] = field(default_factory=list)
    deviation_threshold: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "violations": list(self.violations),
            "deviation_threshold": self.deviation_threshold,
        }


@dataclass
class WeightingResult:
    status: WeightingStatus
    active_dimensions: List[PreferenceDimension]
    method: Optional[str] = None
    best: Optional[PreferenceDimension] = None
    worst: Optional[PreferenceDimension] = None
    weights: Dict[PreferenceDimension, float] = field(
        default_factory=dict
    )
    xi_star: Optional[float] = None
    consistency: ConsistencyReport = field(
        default_factory=lambda: ConsistencyReport(
            status=ConsistencyStatus.NOT_CHECKED
        )
    )
    missing_questions: List[BWMQuestion] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    source: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def all_four_weights(self) -> Dict[str, float]:
        return {
            dimension.value: float(self.weights.get(dimension, 0.0))
            for dimension in PreferenceDimension
        }

    def to_dict(self) -> Dict[str, Any]:
        all_weights = self.all_four_weights()

        return {
            "status": self.status.value,
            "method": self.method,
            "active_dimensions": [
                dimension.value
                for dimension in self.active_dimensions
            ],
            "best": self.best.value if self.best else None,
            "worst": self.worst.value if self.worst else None,
            "weights": all_weights,
            "sum": (
                float(sum(all_weights.values()))
                if self.weights
                else None
            ),
            "xi_star": self.xi_star,
            "ordinal_consistency": self.consistency.status.value,
            "consistency": self.consistency.to_dict(),
            "missing_questions": [
                question.to_dict()
                for question in self.missing_questions
            ],
            "violations": list(self.violations),
            "source": self.source,
            "notes": list(self.notes),
        }
