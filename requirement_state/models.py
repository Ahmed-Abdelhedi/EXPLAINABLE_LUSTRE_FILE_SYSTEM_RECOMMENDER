from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .contract import (
    PREFERENCE_WEIGHT_DIMENSIONS,
    REQUIREMENT_FIELDS,
)


class RequirementFieldStatus(str, Enum):
    VERIFIED = "VERIFIED"
    DECLINED = "DECLINED"
    MISSING = "MISSING"
    PARTIAL = "PARTIAL"
    UNRESOLVED = "UNRESOLVED"
    CONFLICT = "CONFLICT"


@dataclass(frozen=True)
class RequirementFieldTrace:
    field: str
    status: RequirementFieldStatus
    value: Any = None
    source: Optional[str] = None
    evidence: Optional[str] = None
    confidence: Optional[float] = None
    message_id: Optional[str] = None
    revision: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "status": self.status.value,
            "value": self.value,
            "source": self.source,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "message_id": self.message_id,
            "revision": self.revision,
        }


@dataclass(frozen=True)
class PreferenceWeights:
    cost: float
    power: float
    performance: float
    reliability: float
    method: Optional[str] = None
    xi_star: Optional[float] = None
    consistency_status: Optional[str] = None
    source: Optional[str] = None

    def values_dict(self) -> Dict[str, float]:
        return {
            "cost": float(self.cost),
            "power": float(self.power),
            "performance": float(self.performance),
            "reliability": float(self.reliability),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": self.values_dict(),
            "sum": float(sum(self.values_dict().values())),
            "method": self.method,
            "xi_star": self.xi_star,
            "consistency_status": self.consistency_status,
            "source": self.source,
        }

    @classmethod
    def from_mapping(
        cls,
        weights: Dict[str, float],
        *,
        method: Optional[str] = None,
        xi_star: Optional[float] = None,
        consistency_status: Optional[str] = None,
        source: Optional[str] = None,
    ) -> "PreferenceWeights":
        missing = [
            name
            for name in PREFERENCE_WEIGHT_DIMENSIONS
            if name not in weights
        ]

        if missing:
            raise ValueError(
                "Missing preference-weight dimensions: "
                + ", ".join(missing)
            )

        return cls(
            cost=float(weights["cost"]),
            power=float(weights["power"]),
            performance=float(weights["performance"]),
            reliability=float(weights["reliability"]),
            method=method,
            xi_star=xi_star,
            consistency_status=consistency_status,
            source=source,
        )


@dataclass
class FinalRequirementState:
    # Quantitative
    requested_usable_capacity_tib: Optional[float] = None
    client_count: Optional[int] = None
    average_file_size_gb: Optional[float] = None
    max_file_size_gb: Optional[float] = None
    total_file_count: Optional[int] = None
    read_write_ratio: Optional[Any] = None
    target_read_gbps: Optional[float] = None
    target_write_gbps: Optional[float] = None
    max_budget_usd: Optional[float] = None
    max_power_w: Optional[float] = None
    annual_growth_percent: Optional[float] = None
    planning_horizon_years: Optional[int] = None

    # Categorical / Boolean
    access_type: Optional[str] = None
    ha_required: Optional[bool] = None

    # Qualitative preference labels
    cost_priority: Optional[str] = None
    power_priority: Optional[str] = None
    reliability_priority: Optional[str] = None
    performance_priority: Optional[str] = None

    # Derived preference weights. Kept separate from qualitative labels.
    preference_weights: Optional[PreferenceWeights] = None

    # Traceability / validation envelope
    field_traces: Dict[str, RequirementFieldTrace] = field(
        default_factory=dict
    )
    missing_fields: List[str] = field(default_factory=list)
    unresolved_fields: List[str] = field(default_factory=list)
    validation_issues: List[Dict[str, Any]] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)
    ready_for_sizing: bool = False

    def canonical_fields(self) -> Dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in REQUIREMENT_FIELDS
        }

    def to_dict(
        self,
        *,
        include_traceability: bool = True,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            **self.canonical_fields(),
            "preference_weights": (
                None
                if self.preference_weights is None
                else self.preference_weights.to_dict()
            ),
            "ready_for_sizing": bool(self.ready_for_sizing),
        }

        if include_traceability:
            payload.update(
                {
                    "field_traces": {
                        name: trace.to_dict()
                        for name, trace in self.field_traces.items()
                    },
                    "missing_fields": list(self.missing_fields),
                    "unresolved_fields": list(self.unresolved_fields),
                    "validation_issues": list(self.validation_issues),
                    "follow_up_questions": list(self.follow_up_questions),
                }
            )

        return payload

    def to_canonical_json_dict(self) -> Dict[str, Any]:
        """
        Compact payload intended for the future sizing/recommendation stages.

        Traceability stays available in the full state but is deliberately not
        mixed into the canonical Requirement JSON.
        """
        return {
            **self.canonical_fields(),
            "preference_weights": (
                None
                if self.preference_weights is None
                else self.preference_weights.values_dict()
            ),
        }
