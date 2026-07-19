from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional

from field_defs import (
    ALLOWED_ACCESS_TYPES,
    FIELD_QUESTIONS,
    LOW_CONFIDENCE_THRESHOLD,
    REQUIRED_FIELDS,
)
from models import (
    CandidateSource,
    ChatbotStatus,
    ExtractedCandidate,
    FinalFieldValue,
    IssueType,
    ParamName,
    PipelineStage,
    ValidationIssue,
    ValidationReport,
)


MISSING_IF_UNSUPPORTED_FIELDS = {
    ParamName.access_type,
    ParamName.client_count,
}


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def canonical_value(value: Any) -> str:
    if is_number(value):
        return str(round(float(value), 6))

    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, ensure_ascii=False).lower()

    return str(value).strip().lower()


def select_constraint_limit_candidate(
    field: ParamName,
    candidates: List[ExtractedCandidate],
) -> ExtractedCandidate | None:
    """
    Résout automatiquement les cas :
    - budget idéal 50000 USD mais maximum 75000 USD
    - puissance idéale 6 kW mais maximum 9 kW

    Pour max_budget_usd et max_power_w, la valeur métier correcte
    est la limite maximale.
    """

    if field not in {
        ParamName.max_budget_usd,
        ParamName.max_power_w,
    }:
        return None

    maximum_markers = [
        "maximum",
        "max",
        "maximal",
        "maximale",
        "limite",
        "limit",
        "power limit",
        "budget max",
        "max budget",
        "max power",
        "ne pas dépasser",
        "ne pas depasser",
    ]

    ideal_markers = [
        "idéal",
        "ideal",
        "idéale",
        "ideale",
        "préféré",
        "prefere",
        "préférée",
        "preferred",
        "souhaité",
        "souhaite",
        "souhaitée",
        "souhaitee",
    ]

    maximum_candidates = []
    ideal_candidates = []

    for candidate in candidates:
        evidence = (candidate.evidence or "").lower()

        if any(marker in evidence for marker in maximum_markers):
            maximum_candidates.append(candidate)

        if any(marker in evidence for marker in ideal_markers):
            ideal_candidates.append(candidate)

    if maximum_candidates:
        return max(
            maximum_candidates,
            key=lambda candidate: float(candidate.value),
        )

    if ideal_candidates and len(candidates) >= 2:
        return max(
            candidates,
            key=lambda candidate: float(candidate.value),
        )

    return None


def validate_candidate(candidate: ExtractedCandidate) -> Optional[str]:
    field = candidate.field
    value = candidate.value

    if value is None or value == "":
        return "Valeur vide."

    numeric_positive_fields = {
        ParamName.requested_usable_capacity_tib,
        ParamName.client_count,
        ParamName.average_file_size_gb,
        ParamName.max_file_size_gb,
        ParamName.total_file_count,
        ParamName.target_read_gbps,
        ParamName.target_write_gbps,
        ParamName.max_budget_usd,
        ParamName.max_power_w,
    }

    if field in numeric_positive_fields:
        if not is_number(value):
            return "La valeur doit être numérique."

        if value <= 0:
            return "La valeur doit être strictement positive."

    if field == ParamName.annual_growth_percent:
        if not is_number(value):
            return "La croissance annuelle doit être numérique."

        if value < 0:
            return "La croissance annuelle ne peut pas être négative."

    if field in {
        ParamName.client_count,
        ParamName.total_file_count,
    }:
        if not float(value).is_integer():
            return "La valeur doit être un entier."

    if field == ParamName.ha_required:
        if not isinstance(value, bool):
            return "ha_required doit être true ou false."

    if field == ParamName.access_type:
        if not isinstance(value, str):
            return "access_type doit être une chaîne."

        if value not in ALLOWED_ACCESS_TYPES:
            return f"access_type doit être dans {sorted(ALLOWED_ACCESS_TYPES)}."

    if field == ParamName.read_write_ratio:
        if not isinstance(value, dict):
            return "read_write_ratio doit être un objet."

        if "read_percent" not in value or "write_percent" not in value:
            return "read_write_ratio doit contenir read_percent et write_percent."

        read_percent = value["read_percent"]
        write_percent = value["write_percent"]

        if not is_number(read_percent) or not is_number(write_percent):
            return "read_percent et write_percent doivent être numériques."

        total = read_percent + write_percent

        if abs(total - 100) > 0.001:
            return (
                f"Ratio invalide : {read_percent}% lecture + "
                f"{write_percent}% écriture = {total}%, "
                f"alors que la somme doit être exactement 100%."
            )

    return None


def candidate_supported_by_evidence(candidate: ExtractedCandidate) -> bool:
    """
    Validation evidence-based.

    USER_CLARIFICATION est acceptée directement parce qu'elle répond
    à une question active.

    LLM_FALLBACK n'est pas accepté automatiquement.
    Il doit passer les mêmes règles evidence-based que les règles normales.
    """

    if candidate.source == CandidateSource.USER_CLARIFICATION:
        return True

    evidence = (candidate.evidence or "").lower()

    if candidate.field == ParamName.client_count:
        invalid_markers = [
            "salle",
            "salles",
            "salle machine",
            "salles machines",
            "datacenter",
            "data center",
            "rack",
            "racks",
            "année",
            "annee",
            "year",
        ]

        valid_markers = [
            "client",
            "clients",
            "nœud",
            "nœuds",
            "noeud",
            "noeuds",
            "node",
            "nodes",
            "compute",
        ]

        if any(marker in evidence for marker in invalid_markers):
            return False

        return any(marker in evidence for marker in valid_markers)

    if candidate.field == ParamName.average_file_size_gb:
        valid_markers = [
            "moyen",
            "moyenne",
            "moyens",
            "en moyenne",
            "average",
            "avg",
            "taille moyenne",
            "fichiers moyens",
            "fichiers font",
            "file size",
        ]

        invalid_markers = [
            "capacité",
            "capacite",
            "capacity",
            "stockage",
            "storage",
            "utile",
            "utiles",
            "budget",
            "puissance",
            "power",
            "fichiers environ",
        ]

        if any(marker in evidence for marker in invalid_markers):
            return False

        return any(marker in evidence for marker in valid_markers)

    if candidate.field == ParamName.max_file_size_gb:
        valid_markers = [
            "max",
            "maximum",
            "maximale",
            "taille max",
            "taille maximale",
            "fichier max",
            "fichiers max",
            "max file",
            "maximum file",
            "largest file",
            "largest files",
            "biggest file",
            "biggest files",
            "plus gros",
            "gros fichiers",
        ]

        invalid_markers = [
            "capacité",
            "capacite",
            "capacity",
            "stockage",
            "storage",
            "utile",
            "utiles",
            "budget",
            "puissance",
            "power",
            "clients",
            "client",
        ]

        if any(marker in evidence for marker in invalid_markers):
            return False

        return any(marker in evidence for marker in valid_markers)

    if candidate.field == ParamName.requested_usable_capacity_tib:
        valid_markers = [
            "tib",
            "tb",
            "capacité",
            "capacite",
            "capacity",
            "stockage",
            "storage",
            "utile",
            "utiles",
            "usable",
        ]

        return any(marker in evidence for marker in valid_markers)

    return True


def make_conflict_question(
    field: ParamName,
    candidates: List[ExtractedCandidate],
) -> str:
    values = []

    for candidate in candidates:
        values.append(
            f"{candidate.value} {candidate.unit or ''} "
            f"(evidence: '{candidate.evidence}')"
        )

    joined_values = " / ".join(values)

    return (
        f"J'ai trouvé plusieurs valeurs pour {field.value} : "
        f"{joined_values}. Quelle valeur dois-je garder ?"
    )


def make_invalid_question(
    field: ParamName,
    candidates: List[ExtractedCandidate],
) -> str:
    """
    Génère une question plus explicative pour les valeurs invalides.
    """

    if field == ParamName.read_write_ratio and candidates:
        value = candidates[0].value

        if isinstance(value, dict):
            read_percent = value.get("read_percent")
            write_percent = value.get("write_percent")

            if is_number(read_percent) and is_number(write_percent):
                total = read_percent + write_percent

                return (
                    f"Le ratio lecture/écriture est invalide : "
                    f"{read_percent}% lecture + {write_percent}% écriture = "
                    f"{total}%. La somme doit être exactement 100%. "
                    f"Donnez un ratio valide, par exemple 70/30 ou 60/40."
                )

    return FIELD_QUESTIONS[field]


class StateGuard:
    """
    StateGuard déterministe.

    Rôle :
    - accepter les valeurs valides ;
    - bloquer les valeurs invalides ;
    - transformer les hallucinations évidentes en missing ;
    - détecter les vrais conflits ;
    - résoudre les conflits métier évidents ;
    - produire les questions nécessaires.
    """

    def analyze(
        self,
        candidates: List[ExtractedCandidate],
    ) -> ValidationReport:
        grouped: Dict[ParamName, List[ExtractedCandidate]] = defaultdict(list)

        for candidate in candidates:
            grouped[candidate.field].append(candidate)

        final_json: Dict[str, Optional[FinalFieldValue]] = {}
        issues: List[ValidationIssue] = []

        for field in REQUIRED_FIELDS:
            field_candidates = grouped.get(field, [])

            if not field_candidates:
                final_json[field.value] = None

                issues.append(
                    ValidationIssue(
                        type=IssueType.MISSING_FIELD,
                        field=field,
                        message=f"Le champ {field.value} est manquant.",
                        question=FIELD_QUESTIONS[field],
                        candidates=[],
                    )
                )
                continue

            valid_candidates: List[ExtractedCandidate] = []
            invalid_candidates: List[ExtractedCandidate] = []
            unsupported_candidates: List[ExtractedCandidate] = []

            for candidate in field_candidates:
                error = validate_candidate(candidate)

                if error:
                    invalid_candidates.append(candidate)
                    continue

                if not candidate_supported_by_evidence(candidate):
                    unsupported_candidates.append(candidate)
                    continue

                valid_candidates.append(candidate)

            if invalid_candidates and not valid_candidates and not unsupported_candidates:
                final_json[field.value] = None

                issues.append(
                    ValidationIssue(
                        type=IssueType.INVALID_VALUE,
                        field=field,
                        message=(
                            validate_candidate(invalid_candidates[0])
                            or f"Valeur invalide pour {field.value}."
                        ),
                        question=make_invalid_question(field, invalid_candidates),
                        candidates=invalid_candidates,
                    )
                )
                continue

            if unsupported_candidates and not valid_candidates and not invalid_candidates:
                final_json[field.value] = None

                issue_type = (
                    IssueType.MISSING_FIELD
                    if field in MISSING_IF_UNSUPPORTED_FIELDS
                    else IssueType.UNSUPPORTED_BY_EVIDENCE
                )

                issues.append(
                    ValidationIssue(
                        type=issue_type,
                        field=field,
                        message=(
                            f"La valeur extraite pour {field.value} "
                            f"n'est pas suffisamment supportée par le texte."
                        ),
                        question=FIELD_QUESTIONS[field],
                        candidates=(
                            unsupported_candidates
                            if issue_type != IssueType.MISSING_FIELD
                            else []
                        ),
                    )
                )
                continue

            if not valid_candidates:
                final_json[field.value] = None

                issues.append(
                    ValidationIssue(
                        type=IssueType.INVALID_VALUE,
                        field=field,
                        message=f"Aucune valeur utilisable pour {field.value}.",
                        question=FIELD_QUESTIONS[field],
                        candidates=invalid_candidates + unsupported_candidates,
                    )
                )
                continue

            distinct_values = {
                canonical_value(candidate.value)
                for candidate in valid_candidates
            }

            if len(distinct_values) > 1:
                resolved_candidate = select_constraint_limit_candidate(
                    field=field,
                    candidates=valid_candidates,
                )

                if resolved_candidate is not None:
                    final_json[field.value] = FinalFieldValue(
                        value=resolved_candidate.value,
                        unit=resolved_candidate.unit,
                        confidence=resolved_candidate.confidence,
                        evidence=resolved_candidate.evidence,
                        source=resolved_candidate.source.value,
                    )
                    continue

                final_json[field.value] = None

                issues.append(
                    ValidationIssue(
                        type=IssueType.CONFLICTING_VALUES,
                        field=field,
                        message=(
                            f"Plusieurs valeurs différentes ont été détectées "
                            f"pour {field.value}."
                        ),
                        question=make_conflict_question(field, valid_candidates),
                        candidates=valid_candidates,
                    )
                )
                continue

            best_candidate = max(
                valid_candidates,
                key=lambda candidate: candidate.confidence,
            )

            if best_candidate.confidence < LOW_CONFIDENCE_THRESHOLD:
                final_json[field.value] = None

                issues.append(
                    ValidationIssue(
                        type=IssueType.LOW_CONFIDENCE,
                        field=field,
                        message=(
                            f"La valeur détectée pour {field.value} est incertaine."
                        ),
                        question=FIELD_QUESTIONS[field],
                        candidates=[best_candidate],
                    )
                )
                continue

            final_json[field.value] = FinalFieldValue(
                value=best_candidate.value,
                unit=best_candidate.unit,
                confidence=best_candidate.confidence,
                evidence=best_candidate.evidence,
                source=best_candidate.source.value,
            )

        missing_fields = [
            issue.field.value
            for issue in issues
            if issue.type == IssueType.MISSING_FIELD
        ]

        conflicting_fields = [
            issue.field.value
            for issue in issues
            if issue.type == IssueType.CONFLICTING_VALUES
        ]

        invalid_fields = [
            issue.field.value
            for issue in issues
            if issue.type == IssueType.INVALID_VALUE
        ]

        unsupported_fields = [
            issue.field.value
            for issue in issues
            if issue.type == IssueType.UNSUPPORTED_BY_EVIDENCE
        ]

        questions = [issues[0].question] if issues else []

        if issues:
            return ValidationReport(
                status=ChatbotStatus.NEEDS_CLARIFICATION,
                stage=PipelineStage.CLARIFICATION,
                final_json=final_json,
                issues=issues,
                missing_fields=missing_fields,
                conflicting_fields=conflicting_fields,
                invalid_fields=invalid_fields,
                unsupported_fields=unsupported_fields,
                questions=questions,
            )

        return ValidationReport(
            status=ChatbotStatus.VALID,
            stage=PipelineStage.READY_FOR_CALCULATION,
            final_json=final_json,
            issues=[],
            missing_fields=[],
            conflicting_fields=[],
            invalid_fields=[],
            unsupported_fields=[],
            questions=[],
        )