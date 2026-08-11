from __future__ import annotations

import json
from collections import defaultdict
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional

from .field_defs import (
    ALLOWED_ACCESS_TYPES,
    FIELD_QUESTIONS,
    LOW_CONFIDENCE_THRESHOLD,
    REQUIRED_FIELDS,
)
from .models import (
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
    """
    Accepte les types numériques sans considérer bool comme un nombre.
    """

    return isinstance(value, (int, float, Decimal)) and not isinstance(
        value,
        bool,
    )


def canonical_value(value: Any) -> str:
    """
    Forme canonique utilisée uniquement pour comparer des candidats.

    Elle ne modifie pas les valeurs stockées dans le résultat final.
    """

    if is_number(value):
        return str(round(float(value), 6))

    if isinstance(value, dict):
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
        ).lower()

    if isinstance(value, bool):
        return "true" if value else "false"

    return str(value).strip().lower()


def _distinct_candidates(
    candidates: Iterable[ExtractedCandidate],
) -> List[ExtractedCandidate]:
    """
    Déduplique uniquement les candidats qui portent exactement la même valeur.

    Les preuves différentes d'une même valeur ne créent pas de conflit.
    """

    output: List[ExtractedCandidate] = []
    seen = set()

    for candidate in candidates:
        key = canonical_value(candidate.value)

        if key in seen:
            continue

        seen.add(key)
        output.append(candidate)

    return output


def select_constraint_limit_candidate(
    field: ParamName,
    candidates: List[ExtractedCandidate],
) -> ExtractedCandidate | None:
    """
    Résout uniquement les préférences/limites explicitement structurées.

    Exemples :
    - budget idéal 50 000 USD, maximum 75 000 USD ;
    - puissance préférée 6 kW, limite 9 kW.

    Il ne résout pas un vrai conflit de deux limites maximales différentes.
    """

    if field not in {
        ParamName.max_budget_usd,
        ParamName.max_power_w,
    }:
        return None

    maximum_markers = (
        "maximum",
        "maximal",
        "maximale",
        "max ",
        "limite",
        "limit",
        "cap",
        "ne pas dépasser",
        "ne pas depasser",
    )

    preference_markers = (
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
        "target",
    )

    maximum_candidates: List[ExtractedCandidate] = []
    preference_candidates: List[ExtractedCandidate] = []

    for candidate in candidates:
        evidence = (candidate.evidence or "").lower()

        if any(marker in evidence for marker in maximum_markers):
            maximum_candidates.append(candidate)

        if any(marker in evidence for marker in preference_markers):
            preference_candidates.append(candidate)

    distinct_maximums = _distinct_candidates(maximum_candidates)

    if len(distinct_maximums) == 1:
        return distinct_maximums[0]

    if len(distinct_maximums) > 1:
        return None

    # Cas où une valeur est explicitement préférée et l'autre représente
    # implicitement la contrainte supérieure.
    if preference_candidates and len(candidates) == 2:
        non_preferred = [
            candidate
            for candidate in candidates
            if candidate not in preference_candidates
        ]

        if len(non_preferred) == 1:
            return non_preferred[0]

    return None


def validate_candidate(candidate: ExtractedCandidate) -> Optional[str]:
    """
    Validation de type et de domaine.

    Aucune valeur n'est corrigée ici. Une valeur invalide reste invalide et
    doit être clarifiée par l'utilisateur.
    """

    field = candidate.field
    value = candidate.value

    if value is None or value == "":
        return "Valeur vide."

    strictly_positive_fields = {
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

    if field in strictly_positive_fields:
        if not is_number(value):
            return "La valeur doit être numérique."

        if float(value) <= 0:
            return "La valeur doit être strictement positive."

    if field in {
        ParamName.client_count,
        ParamName.total_file_count,
    }:
        if not is_number(value):
            return "La valeur doit être numérique."

        if not float(value).is_integer():
            return "La valeur doit être un entier."

    if field == ParamName.annual_growth_percent:
        if not is_number(value):
            return "La croissance annuelle doit être numérique."

        if float(value) < 0:
            return "La croissance annuelle ne peut pas être négative."

    if field == ParamName.ha_required:
        if not isinstance(value, bool):
            return "ha_required doit être true ou false."

    if field == ParamName.access_type:
        if not isinstance(value, str):
            return "access_type doit être une chaîne."

        if value not in ALLOWED_ACCESS_TYPES:
            return (
                "access_type doit appartenir à "
                f"{sorted(ALLOWED_ACCESS_TYPES)}."
            )

    if field == ParamName.read_write_ratio:
        if not isinstance(value, dict):
            return "read_write_ratio doit être un objet."

        if (
            "read_percent" not in value
            or "write_percent" not in value
        ):
            return (
                "read_write_ratio doit contenir "
                "read_percent et write_percent."
            )

        read_percent = value["read_percent"]
        write_percent = value["write_percent"]

        if not is_number(read_percent) or not is_number(write_percent):
            return (
                "read_percent et write_percent doivent être numériques."
            )

        read_value = float(read_percent)
        write_value = float(write_percent)

        if not 0 <= read_value <= 100:
            return "read_percent doit être compris entre 0 et 100."

        if not 0 <= write_value <= 100:
            return "write_percent doit être compris entre 0 et 100."

        total = read_value + write_value

        if abs(total - 100.0) > 0.001:
            return (
                f"Ratio invalide : {read_percent}% lecture + "
                f"{write_percent}% écriture = {total:g}%, "
                "alors que la somme doit être exactement 100%."
            )

    return None


_LLM_EVIDENCE_MARKERS: Dict[ParamName, tuple[str, ...]] = {
    ParamName.requested_usable_capacity_tib: (
        "tib",
        "tb",
        "capacité",
        "capacite",
        "capacity",
        "stockage",
        "storage",
        "usable",
    ),
    ParamName.client_count: (
        "client",
        "clients",
        "cliente",
        "clientes",
        "nœud",
        "nœuds",
        "noeud",
        "noeuds",
        "node",
        "nodes",
        "compute",
    ),
    ParamName.average_file_size_gb: (
        "moyen",
        "moyenne",
        "average",
        "avg",
        "taille moyenne",
        "fichiers moyens",
        "average file",
    ),
    ParamName.max_file_size_gb: (
        "max",
        "maximum",
        "taille maximale",
        "largest",
        "biggest",
    ),
    ParamName.total_file_count: (
        "fichier",
        "fichiers",
        "file",
        "files",
        "archivo",
        "archivos",
        "million",
        "millions",
    ),
    ParamName.read_write_ratio: (
        "ratio",
        "read/write",
        "lecture/écriture",
        "lecture/ecriture",
        "r/w",
        "r:w",
    ),
    ParamName.access_type: (
        "accès",
        "acces",
        "access",
        "sequential",
        "séquentiel",
        "random",
        "parallel",
        "parallèle",
        "streaming",
        "mixed",
        "mixte",
    ),
    ParamName.target_read_gbps: (
        "lecture",
        "read",
        "lectura",
    ),
    ParamName.target_write_gbps: (
        "écriture",
        "ecriture",
        "write",
        "escritura",
    ),
    ParamName.ha_required: (
        "ha",
        "haute disponibilité",
        "haute disponibilite",
        "high availability",
        "fault tolerance",
        "redondance",
    ),
    ParamName.max_budget_usd: (
        "budget",
        "usd",
        "dollar",
        "$",
        "presupuesto",
    ),
    ParamName.max_power_w: (
        "puissance",
        "power",
        "kw",
        "watt",
        "potencia",
        "pwr",
    ),
    ParamName.annual_growth_percent: (
        "croissance",
        "growth",
        "crecimiento",
        "pourcent",
        "percent",
        "%",
    ),
}


def candidate_supported_by_evidence(
    candidate: ExtractedCandidate,
) -> bool:
    """
    Vérifie que la valeur est soutenue par une preuve textuelle.

    - RULE : accepté car l'extracteur déterministe a déjà utilisé une règle
      bornée sur le texte ;
    - USER_CLARIFICATION : accepté car il répond à une question active ;
    - LLM_FALLBACK : doit contenir un marqueur du champ dans la preuve ;
    - autres sources : preuve non vide obligatoire.
    """

    if candidate.source in {
        CandidateSource.RULE,
        CandidateSource.USER_CLARIFICATION,
        CandidateSource.NORMALIZER,
    }:
        return bool(candidate.evidence or candidate.source_text)

    evidence = (
        candidate.evidence
        or candidate.source_text
        or ""
    ).lower()

    if not evidence:
        return False

    markers = _LLM_EVIDENCE_MARKERS.get(candidate.field, ())

    if not markers:
        return True

    return any(marker in evidence for marker in markers)


def make_conflict_question(
    field: ParamName,
    candidates: List[ExtractedCandidate],
) -> str:
    values: List[str] = []

    for candidate in _distinct_candidates(candidates):
        rendered = (
            f"{candidate.value}"
            f"{' ' + candidate.unit if candidate.unit else ''}"
        )
        values.append(rendered)

    return (
        f"J'ai trouvé plusieurs valeurs pour {field.value} : "
        f"{' / '.join(values)}. Quelle valeur dois-je garder ?"
    )


def make_invalid_question(
    field: ParamName,
    candidates: List[ExtractedCandidate],
) -> str:
    """
    Génère une question ciblée pour les valeurs invalides.
    """

    if field == ParamName.read_write_ratio and candidates:
        value = candidates[0].value

        if isinstance(value, dict):
            read_percent = value.get("read_percent")
            write_percent = value.get("write_percent")

            if is_number(read_percent) and is_number(write_percent):
                total = float(read_percent) + float(write_percent)

                return (
                    "Le ratio lecture/écriture est invalide : "
                    f"{read_percent}% + {write_percent}% = {total:g}%. "
                    "La somme doit être exactement 100%. "
                    "Donnez un ratio comme 70/30 ou 60/40."
                )

    if field in {
        ParamName.client_count,
        ParamName.total_file_count,
    }:
        return (
            f"La valeur donnée pour {field.value} doit être un entier "
            "strictement positif. Quelle valeur faut-il utiliser ?"
        )

    return FIELD_QUESTIONS[field]


def _final_field_value(
    candidate: ExtractedCandidate,
) -> FinalFieldValue:
    return FinalFieldValue(
        value=candidate.value,
        unit=candidate.unit,
        confidence=candidate.confidence,
        evidence=candidate.evidence,
        source=candidate.source.value,
    )


class StateGuard:
    """
    Validateur déterministe de l'état des exigences.

    Responsabilités :
    - accepter une valeur unique, valide et supportée ;
    - rejeter les valeurs invalides sans les corriger ;
    - préserver les conflits réels ;
    - résoudre uniquement la distinction préférence/limite ;
    - produire une clarification unique et ciblée.
    """

    def analyze(
        self,
        candidates: List[ExtractedCandidate],
    ) -> ValidationReport:
        grouped: Dict[
            ParamName,
            List[ExtractedCandidate],
        ] = defaultdict(list)

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
                validation_error = validate_candidate(candidate)

                if validation_error is not None:
                    invalid_candidates.append(candidate)
                    continue

                if not candidate_supported_by_evidence(candidate):
                    unsupported_candidates.append(candidate)
                    continue

                valid_candidates.append(candidate)

            valid_candidates = _distinct_candidates(valid_candidates)
            invalid_candidates = _distinct_candidates(invalid_candidates)
            unsupported_candidates = _distinct_candidates(
                unsupported_candidates
            )

            if not valid_candidates:
                final_json[field.value] = None

                if invalid_candidates:
                    message = (
                        validate_candidate(invalid_candidates[0])
                        or f"Valeur invalide pour {field.value}."
                    )

                    issues.append(
                        ValidationIssue(
                            type=IssueType.INVALID_VALUE,
                            field=field,
                            message=message,
                            question=make_invalid_question(
                                field,
                                invalid_candidates,
                            ),
                            candidates=invalid_candidates,
                        )
                    )
                    continue

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
                            "n'est pas suffisamment supportée par le texte."
                        ),
                        question=FIELD_QUESTIONS[field],
                        candidates=(
                            []
                            if issue_type == IssueType.MISSING_FIELD
                            else unsupported_candidates
                        ),
                    )
                )
                continue

            if len(valid_candidates) > 1:
                resolved_candidate = select_constraint_limit_candidate(
                    field=field,
                    candidates=valid_candidates,
                )

                if resolved_candidate is not None:
                    final_json[field.value] = _final_field_value(
                        resolved_candidate
                    )
                    continue

                final_json[field.value] = None
                issues.append(
                    ValidationIssue(
                        type=IssueType.CONFLICTING_VALUES,
                        field=field,
                        message=(
                            "Plusieurs valeurs différentes ont été détectées "
                            f"pour {field.value}."
                        ),
                        question=make_conflict_question(
                            field,
                            valid_candidates,
                        ),
                        candidates=valid_candidates,
                    )
                )
                continue

            best_candidate = valid_candidates[0]

            if best_candidate.confidence < LOW_CONFIDENCE_THRESHOLD:
                final_json[field.value] = None
                issues.append(
                    ValidationIssue(
                        type=IssueType.LOW_CONFIDENCE,
                        field=field,
                        message=(
                            f"La valeur détectée pour {field.value} "
                            "est incertaine."
                        ),
                        question=FIELD_QUESTIONS[field],
                        candidates=[best_candidate],
                    )
                )
                continue

            final_json[field.value] = _final_field_value(best_candidate)

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
