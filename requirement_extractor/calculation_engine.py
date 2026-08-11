from __future__ import annotations

from typing import Any, Dict, Optional

from .models import FinalFieldValue


class CalculationEngine:
    """
    Premier moteur de calcul déterministe.

    Pour le MVP :
    - growth_factor
    - planned_usable_capacity_tib
    """

    def __init__(self, target_fill_ratio: float = 0.80):
        self.target_fill_ratio = target_fill_ratio

    def _value(
        self,
        final_json: Dict[str, Optional[FinalFieldValue]],
        key: str,
    ):
        item = final_json.get(key)

        if item is None:
            return None

        return item.value

    def calculate(
        self,
        final_json: Dict[str, Optional[FinalFieldValue]],
    ) -> Dict[str, Any]:
        required_capacity = self._value(
            final_json,
            "requested_usable_capacity_tib",
        )

        growth_percent = self._value(
            final_json,
            "annual_growth_percent",
        )

        if required_capacity is None or growth_percent is None:
            return {
                "ready": False,
                "reason": "capacity_or_growth_missing",
            }

        growth_factor = 1 + (float(growth_percent) / 100.0)

        planned_usable_capacity_tib = (
            float(required_capacity)
            * growth_factor
            / self.target_fill_ratio
        )

        return {
            "ready": True,
            "growth_factor": round(growth_factor, 4),
            "target_fill_ratio": self.target_fill_ratio,
            "planned_usable_capacity_tib": round(planned_usable_capacity_tib, 4),
        }