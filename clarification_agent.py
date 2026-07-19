from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Set

from models import (
    CandidateSource,
    ExtractedCandidate,
    ParamName,
    ValidationIssue,
)
from unit_normalizer import normalize_unit_value


@dataclass
class ClarificationDecision:
    resolved_candidates: List[ExtractedCandidate]
    should_ask_user: bool
    questions: List[str]


NUMERIC_FIELDS = {
    ParamName.requested_usable_capacity_tib,
    ParamName.client_count,
    ParamName.average_file_size_gb,
    ParamName.max_file_size_gb,
    ParamName.total_file_count,
    ParamName.target_read_gbps,
    ParamName.target_write_gbps,
    ParamName.max_budget_usd,
    ParamName.max_power_w,
    ParamName.annual_growth_percent,
}


def extract_first_number(text: str) -> Optional[float]:
    text = text.lower().replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", text)

    if not match:
        return None

    value = float(match.group(0))

    if value.is_integer():
        return int(value)

    return value


def infer_unit_for_numeric_field(
    field: ParamName,
    text: str,
) -> str | None:
    text = text.lower()

    if field in {
        ParamName.average_file_size_gb,
        ParamName.max_file_size_gb,
    }:
        if re.search(r"\b(mb|mib)\b", text):
            return "MB"

        return "GB"

    if field == ParamName.max_power_w:
        if re.search(r"\b(kw|kilowatt|kilowatts)\b", text):
            return "kW"

        return "W"

    if field == ParamName.requested_usable_capacity_tib:
        return "TiB"

    if field in {
        ParamName.target_read_gbps,
        ParamName.target_write_gbps,
    }:
        return "GB/s"

    if field == ParamName.max_budget_usd:
        return "USD"

    if field == ParamName.annual_growth_percent:
        return "%"

    return None


def parse_ratio(text: str) -> Optional[dict]:
    text = text.lower().replace(",", ".")

    slash_match = re.search(
        r"(?:ratio\s*)?(\d{1,3})\s*/\s*(\d{1,3})",
        text,
    )

    if slash_match:
        return {
            "read_percent": int(slash_match.group(1)),
            "write_percent": int(slash_match.group(2)),
        }

    named_match = re.search(
        r"(\d{1,3})\s*%?\s*(lecture|read)"
        r"\D+(\d{1,3})\s*%?\s*(écriture|ecriture|write)",
        text,
    )

    if named_match:
        return {
            "read_percent": int(named_match.group(1)),
            "write_percent": int(named_match.group(3)),
        }

    return None


def parse_ha(text: str) -> Optional[bool]:
    text = text.lower().strip()

    false_markers = [
        "pas de ha",
        "sans ha",
        "ha non obligatoire",
        "no ha",
        "without ha",
        "non",
        "false",
    ]

    true_markers = [
        "oui",
        "yes",
        "true",
        "avec ha",
        "ha yes",
        "ha obligatoire",
        "ha required",
        "haute disponibilité",
        "haute disponibilite",
        "high availability",
        "fiable",
        "robuste",
        "redondance",
    ]

    if any(marker in text for marker in false_markers):
        return False

    if any(marker in text for marker in true_markers):
        return True

    return None


def parse_access_type(text: str) -> Optional[str]:
    """
    Résout les réponses courtes à la question access_type.

    Exemples acceptés :
    - mixed
    - mixte
    - random
    - aléatoire
    - sequential
    - séquentiel
    - parallel
    - parallèle
    - streaming
    """

    text = text.lower().strip()

    mapping = {
        "mixed": "mixed",
        "mixte": "mixed",
        "hybride": "mixed",

        "random": "random",
        "aléatoire": "random",
        "aleatoire": "random",

        "sequential": "sequential",
        "séquentiel": "sequential",
        "sequentiel": "sequential",

        "parallel": "parallel",
        "parallèle": "parallel",
        "parallele": "parallel",

        "streaming": "streaming",
    }

    # Réponse exacte courte : "mixed", "random", etc.
    if text in mapping:
        return mapping[text]

    # Réponse plus longue : "accès random", "type mixte", etc.
    for marker, value in mapping.items():
        if marker in text:
            return value

    return None


def make_candidate(
    field: ParamName,
    value,
    unit,
    evidence: str,
    turn_id: int,
) -> ExtractedCandidate:
    return ExtractedCandidate(
        field=field,
        value=value,
        unit=unit,
        evidence=evidence,
        confidence=1.0,
        source=CandidateSource.USER_CLARIFICATION,
        source_text=evidence,
        turn_id=turn_id,
    )


class ClarificationAgent:
    """
    Clarification déterministe.

    Il résout les réponses courtes :
    - 200
    - 90/10
    - pas de HA
    - 8 kW
    - mixed
    - random
    """

    def resolve(
        self,
        user_text: str,
        issues: List[ValidationIssue],
        turn_id: int,
    ) -> ClarificationDecision:
        pending_fields: Set[ParamName] = {
            issue.field
            for issue in issues
        }

        resolved: List[ExtractedCandidate] = []

        if ParamName.read_write_ratio in pending_fields:
            ratio = parse_ratio(user_text)

            if ratio is not None:
                resolved.append(
                    make_candidate(
                        field=ParamName.read_write_ratio,
                        value=ratio,
                        unit="%",
                        evidence=user_text,
                        turn_id=turn_id,
                    )
                )

        if ParamName.ha_required in pending_fields:
            ha = parse_ha(user_text)

            if ha is not None:
                resolved.append(
                    make_candidate(
                        field=ParamName.ha_required,
                        value=ha,
                        unit=None,
                        evidence=user_text,
                        turn_id=turn_id,
                    )
                )

        if ParamName.access_type in pending_fields:
            access_type = parse_access_type(user_text)

            if access_type is not None:
                resolved.append(
                    make_candidate(
                        field=ParamName.access_type,
                        value=access_type,
                        unit=None,
                        evidence=user_text,
                        turn_id=turn_id,
                    )
                )

        numeric_pending = [
            field
            for field in pending_fields
            if field in NUMERIC_FIELDS
        ]

        if len(numeric_pending) == 1:
            number = extract_first_number(user_text)

            if number is not None:
                field = numeric_pending[0]
                raw_unit = infer_unit_for_numeric_field(field, user_text)

                value, unit = normalize_unit_value(
                    field=field,
                    value=number,
                    unit=raw_unit,
                )

                resolved.append(
                    make_candidate(
                        field=field,
                        value=value,
                        unit=unit,
                        evidence=user_text,
                        turn_id=turn_id,
                    )
                )

        if resolved:
            return ClarificationDecision(
                resolved_candidates=resolved,
                should_ask_user=False,
                questions=[],
            )

        return ClarificationDecision(
            resolved_candidates=[],
            should_ask_user=True,
            questions=[
                issue.question
                for issue in issues
            ],
        )