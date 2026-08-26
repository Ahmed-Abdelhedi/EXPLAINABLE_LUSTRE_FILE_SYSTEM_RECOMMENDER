from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from .semantic_guard import (
    COMPARATIVE_PATTERNS,
    INTENSITY_PATTERNS,
    _normalize,
)


RESIDUAL_VALIDATOR_VERSION = (
    "layer2_residual_validator_v1_20260825"
)

# Natural French elision used by the dataset and by real user text.
# The original guard recognizes "rien n'est plus important que ..." but the
# common grammatical form before a vowel is "qu'un / qu'une".
_EXTRA_INTENSITY_PATTERNS: Tuple[
    Tuple[str, re.Pattern[str]], ...
] = (
    (
        "VERY_HIGH",
        re.compile(
            r"rien n.est plus important qu(?:e|['’])"
        ),
    ),
)

# These are relation markers accepted only when the LLM has already returned
# RELATIVE_ONLY and supplied exact evidence. They do not create an absolute
# level and therefore cannot upgrade a preference.
_EXTRA_COMPARATIVE_PATTERNS: Tuple[
    re.Pattern[str], ...
] = (
    re.compile(r"\bavant\b"),
    re.compile(r"\bpuis\b"),
    re.compile(r"\bthen\b"),
)


@dataclass(frozen=True)
class ResidualValidationResult:
    prediction: Dict[str, Any]
    action: str
    raw_status: Optional[str]
    raw_level: Optional[str]
    canonical_level: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "raw_status": self.raw_status,
            "raw_level": self.raw_level,
            "canonical_level": self.canonical_level,
        }


def _canonical_intensity_levels(
    evidence: str,
) -> Tuple[str, ...]:
    normalized = _normalize(evidence)
    found = set()

    for level, patterns in INTENSITY_PATTERNS:
        if any(
            pattern.search(normalized)
            for pattern in patterns
        ):
            found.add(level)

    for level, pattern in _EXTRA_INTENSITY_PATTERNS:
        if pattern.search(normalized):
            found.add(level)

    return tuple(sorted(found))


def _has_comparative_support(
    evidence: str,
) -> bool:
    normalized = _normalize(evidence)

    if any(
        pattern.search(normalized)
        for pattern in COMPARATIVE_PATTERNS
    ):
        return True

    return any(
        pattern.search(normalized)
        for pattern in _EXTRA_COMPARATIVE_PATTERNS
    )


def _reject(
    prediction: Mapping[str, Any],
    *,
    reason: str,
) -> ResidualValidationResult:
    output = dict(prediction)
    output.update(
        {
            "status": "UNRESOLVED",
            "level": None,
            "accepted": False,
            "validation_error": reason,
        }
    )

    return ResidualValidationResult(
        prediction=output,
        action=f"REJECTED:{reason}",
        raw_status=(
            str(prediction.get("status"))
            if prediction.get("status") is not None
            else None
        ),
        raw_level=(
            str(prediction.get("level"))
            if prediction.get("level") is not None
            else None
        ),
        canonical_level=None,
    )


def validate_residual_prediction(
    *,
    text: str,
    dimension: str,
    prediction: Mapping[str, Any],
) -> ResidualValidationResult:
    """
    Safety validator for the small residual Qwen branch.

    The LLM is allowed to *propose* semantics. Automatic acceptance requires
    deterministic support in the exact evidence:

    - RESOLVED requires a recognized explicit ordinal cue; the deterministic
      mapping is authoritative and canonicalizes the LLM level.
    - RELATIVE_ONLY requires a recognized comparison/order cue.
    - NO_SIGNAL is never accepted from the residual LLM alone because absence
      of a preference is not positively evidenced by an intensity phrase.
      It remains UNRESOLVED unless the earlier deterministic guard already
      resolved it.
    - UNRESOLVED stays safely abstained.

    Gold labels and dataset metadata are never used.
    """
    del dimension  # kept in the signature for production traceability.

    raw = dict(prediction)
    raw_status = raw.get("status")
    raw_level = raw.get("level")

    if not raw.get("accepted", False):
        return ResidualValidationResult(
            prediction=raw,
            action="KEEP_ABSTENTION",
            raw_status=(
                str(raw_status)
                if raw_status is not None
                else None
            ),
            raw_level=(
                str(raw_level)
                if raw_level is not None
                else None
            ),
            canonical_level=None,
        )

    evidence = raw.get("evidence")

    if evidence is not None:
        evidence = str(evidence)

        if evidence not in text:
            return _reject(
                raw,
                reason="EVIDENCE_NOT_EXACT_SUBSTRING",
            )

    status = str(raw_status or "").upper()

    if status == "RESOLVED":
        if not evidence:
            return _reject(
                raw,
                reason="RESOLVED_WITHOUT_EVIDENCE",
            )

        levels = _canonical_intensity_levels(evidence)

        if len(levels) == 0:
            return _reject(
                raw,
                reason=(
                    "RESOLVED_WITHOUT_DETERMINISTIC_"
                    "INTENSITY_SUPPORT"
                ),
            )

        if len(levels) > 1:
            return _reject(
                raw,
                reason="MULTIPLE_INTENSITY_CUES_IN_EVIDENCE",
            )

        canonical = levels[0]
        output = dict(raw)
        output["level"] = canonical
        output["accepted"] = True
        output["validation_error"] = None

        action = "SUPPORTED_RESOLVED"

        if str(raw_level) != canonical:
            action = "CANONICALIZED_RESOLVED_LEVEL"

        return ResidualValidationResult(
            prediction=output,
            action=action,
            raw_status=status,
            raw_level=(
                str(raw_level)
                if raw_level is not None
                else None
            ),
            canonical_level=canonical,
        )

    if status == "RELATIVE_ONLY":
        if not evidence:
            return _reject(
                raw,
                reason="RELATIVE_ONLY_WITHOUT_EVIDENCE",
            )

        if not _has_comparative_support(evidence):
            return _reject(
                raw,
                reason=(
                    "RELATIVE_ONLY_WITHOUT_DETERMINISTIC_"
                    "COMPARISON_SUPPORT"
                ),
            )

        output = dict(raw)
        output["level"] = None
        output["accepted"] = True
        output["validation_error"] = None

        return ResidualValidationResult(
            prediction=output,
            action="SUPPORTED_RELATIVE_ONLY",
            raw_status=status,
            raw_level=None,
            canonical_level=None,
        )

    if status == "NO_SIGNAL":
        return _reject(
            raw,
            reason="RESIDUAL_NO_SIGNAL_NOT_AUTO_ACCEPTED",
        )

    if status == "UNRESOLVED":
        output = dict(raw)
        output["level"] = None
        output["accepted"] = False

        return ResidualValidationResult(
            prediction=output,
            action="KEEP_ABSTENTION",
            raw_status=status,
            raw_level=None,
            canonical_level=None,
        )

    return _reject(
        raw,
        reason="UNSUPPORTED_RESIDUAL_STATUS",
    )
