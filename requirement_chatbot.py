from __future__ import annotations

import json
import re
from typing import List, Set

from calculation_engine import CalculationEngine
from clarification_agent import ClarificationAgent
from hybrid_extractor import HybridExtractor
from models import (
    CandidateSource,
    ChatbotStatus,
    ExtractedCandidate,
    IssueType,
    ParamName,
    PipelineStage,
    RequirementState,
    ValidationIssue,
)
from state_guard import StateGuard
from unit_normalizer import normalize_unit_value
from ai_plausibility_agent import (
    AIPlausibilityAgent,
    AIPlausibilityIssue,
    AIPlausibilityReport,
    AIPlausibilityStatus,
)


def candidate_key(candidate: ExtractedCandidate) -> str:
    return (
        f"{candidate.field.value}|"
        f"{str(candidate.value).lower()}|"
        f"{str(candidate.unit).lower()}|"
        f"{str(candidate.evidence).lower()}"
    )


class RequirementChatbot:
    """
    Chatbot principal.

    Rôle :
    - recevoir chaque message utilisateur ;
    - lancer l'extraction hybride rule-first ;
    - gérer les clarifications courtes ;
    - valider l'état avec StateGuard ;
    - vérifier la plausibilité globale avec AI PlausibilityAgent ;
    - lancer les calculs si tous les champs sont valides et plausibles.

    Important :
    - Les règles déterministes restent prioritaires.
    - Le LLM fallback ne doit jamais remplacer une valeur déjà extraite par règle.
    - Le PlausibilityAgent ne corrige rien automatiquement.
    - Une correction proposée par l'agent doit être confirmée par l'utilisateur.
    """

    def __init__(self):
        self.state = RequirementState()

        self.extractor = HybridExtractor()
        self.guard = StateGuard()
        self.clarification_agent = ClarificationAgent()
        self.calculation_engine = CalculationEngine()
        self.plausibility_agent = AIPlausibilityAgent()

        self.confirmed_plausibility_values = set()
        self.turn_id = 0

    def reset(self) -> None:
        """
        Réinitialise toute la conversation.
        """

        self.state = RequirementState()
        self.turn_id = 0
        self.confirmed_plausibility_values.clear()

    # ============================================================
    # STATE HELPERS
    # ============================================================

    def _pending_fields(self) -> List[str]:
        fields = []

        fields.extend(self.state.missing_fields)
        fields.extend(self.state.conflicting_fields)
        fields.extend(self.state.invalid_fields)
        fields.extend(self.state.unsupported_fields)

        return sorted(set(fields))

    def _remove_candidates_for_fields(self, fields: Set[str]) -> None:
        """
        Supprime les anciennes valeurs pour les champs donnés.

        Exemple :
        - ancien budget = 40000
        - message = "modifier budget à 55000"
        => on supprime l'ancien budget avant d'ajouter le nouveau.
        """

        self.state.extracted_candidates = [
            candidate
            for candidate in self.state.extracted_candidates
            if candidate.field.value not in fields
        ]

    def _extend_candidates(
        self,
        candidates: List[ExtractedCandidate],
    ) -> None:
        """
        Ajoute des candidats sans dupliquer exactement les mêmes valeurs.
        """

        existing = {
            candidate_key(candidate)
            for candidate in self.state.extracted_candidates
        }

        for candidate in candidates:
            key = candidate_key(candidate)

            if key not in existing:
                self.state.extracted_candidates.append(candidate)
                existing.add(key)

    def _value_key(self, value) -> str:
        try:
            return json.dumps(value, sort_keys=True, ensure_ascii=False)
        except Exception:
            return str(value)

    def _mark_plausibility_confirmed(
        self,
        field: ParamName,
        value,
    ) -> None:
        """
        Marque une valeur comme confirmée par l'utilisateur après une alerte IA.

        Exemple :
        - AI Agent signale max_power_w=50 W suspect
        - il propose 50000 W
        - l'utilisateur répond oui
        - on mémorise max_power_w=50000 comme confirmé
        """

        self.confirmed_plausibility_values.add(
            (
                field.value,
                self._value_key(value),
            )
        )

    def _is_plausibility_confirmed(
        self,
        field: ParamName,
        value,
    ) -> bool:
        return (
            field.value,
            self._value_key(value),
        ) in self.confirmed_plausibility_values

    def _is_yes_answer(self, text: str) -> bool:
        normalized = text.strip().lower()
        normalized = re.sub(r"[.!?]+$", "", normalized)

        yes_values = {
            "oui",
            "tu as raison",
            "tu a raison",
            "oui tu as raison",
            "oui tu a raison",
            "c'est correct",
            "cest correct",
            "yes",
            "y",
            "ok",
            "okay",
            "d'accord",
            "daccord",
            "je confirme",
            "confirm",
            "confirmed",
            "correct",
            "c'est ça",
            "c'est ca",
            "cest ca",
            "exact",
        }

        return normalized in yes_values

    def _is_unit_confirmation_answer(self, text: str) -> bool:
        """
        Permet d'accepter des réponses courtes comme :
        - kw
        - kilowatt
        - kilowatts

        Utile après une question du type :
        "Voulez-vous dire 50 kW ?"
        """

        normalized = text.strip().lower()

        unit_confirmations = {
            "kw",
            "k w",
            "kilowatt",
            "kilowatts",
            "en kw",
            "en kilowatt",
            "en kilowatts",
        }

        return normalized in unit_confirmations

    # ============================================================
    # TURN TYPE DETECTION
    # ============================================================

    def _is_explicit_update(self, text: str) -> bool:
        """
        Détecte si l'utilisateur veut modifier une valeur déjà donnée.

        Exemples :
        - modifier la capacité à 450 TiB
        - finalement sans HA
        - changer le ratio à 90/10
        """

        text = text.lower()

        markers = [
            "finalement",
            "plutôt",
            "plutot",
            "enfin plutôt",
            "enfin plutot",
            "au lieu de",
            "à la place",
            "a la place",
            "je préfère",
            "je prefere",
            "modifier",
            "modifie",
            "modifié",
            "changer",
            "change",
            "changé",
            "corriger",
            "corrige",
            "corrigé",
            "remplacer",
            "remplace",
            "remplacé",
            "mettre à jour",
            "mettre a jour",
            "augmenter",
            "augmente",
            "augmenté",
            "réduire",
            "reduire",
            "baisser",
            "increase",
            "decrease",
            "update",
            "set",
        ]

        return any(marker in text for marker in markers)

    def _is_rich_message(self, text: str) -> bool:
        """
        Détecte si le message contient plusieurs informations.

        Réponse courte :
        - 200
        - mixed
        - oui
        - 60/40

        Message riche :
        - fichiers moyens 2 GB, taille max 50 GB, ratio 70/30...
        """

        text = text.lower()

        number_count = len(
            re.findall(r"-?\d+(?:[\.,]\d+)?", text)
        )

        separator_count = sum(
            1
            for separator in [",", ";", ".", " et "]
            if separator in text
        )

        keywords = [
            "tib",
            "tb",
            "clients",
            "client",
            "noeud",
            "noeuds",
            "nœud",
            "nœuds",
            "fichier",
            "fichiers",
            "million",
            "millions",
            "ratio",
            "lecture",
            "read",
            "écriture",
            "ecriture",
            "write",
            "budget",
            "usd",
            "$",
            "puissance",
            "power",
            "kw",
            "w",
            "croissance",
            "growth",
            "ha",
            "accès",
            "acces",
        ]

        keyword_count = sum(
            1
            for keyword in keywords
            if keyword in text
        )

        return (
            number_count >= 2
            or separator_count >= 2
            or keyword_count >= 2
            or len(text.split()) >= 8
        )

    # ============================================================
    # CLARIFICATION PARSERS
    # ============================================================

    def _make_user_clarification_candidate(
        self,
        field: ParamName,
        value,
        unit,
        evidence: str,
    ) -> ExtractedCandidate:
        return ExtractedCandidate(
            field=field,
            value=value,
            unit=unit,
            evidence=evidence,
            confidence=1.0,
            source=CandidateSource.USER_CLARIFICATION,
            source_text=evidence,
            turn_id=self.turn_id,
        )

    def _parse_access_type_answer(self, text: str) -> str | None:
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

        if text in mapping:
            return mapping[text]

        for marker, value in mapping.items():
            if marker in text:
                return value

        return None

    def _parse_ha_answer(self, text: str) -> bool | None:
        text = text.lower().strip()

        false_markers = [
            "non",
            "no",
            "false",
            "pas de ha",
            "sans ha",
            "no ha",
            "without ha",
            "not required",
            "not mandatory",
        ]

        true_markers = [
            "oui",
            "yes",
            "true",
            "avec ha",
            "ha obligatoire",
            "ha required",
            "ha yes",
            "haute disponibilité",
            "haute disponibilite",
            "high availability",
            "required",
            "mandatory",
        ]

        if any(marker == text or marker in text for marker in false_markers):
            return False

        if any(marker == text or marker in text for marker in true_markers):
            return True

        return None

    def _parse_ratio_answer(self, text: str) -> dict | None:
        text = text.lower()

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

    def _extract_first_number(self, text: str) -> int | float | None:
        """
        Extrait le premier nombre.

        Supporte :
        - 200
        - 1.5
        - 1,5
        """

        match = re.search(
            r"-?\d+(?:[\.,]\d+)?",
            text.lower(),
        )

        if not match:
            return None

        value = float(match.group(0).replace(",", "."))

        if value.is_integer():
            return int(value)

        return value

    def _apply_count_multiplier(
        self,
        field: ParamName,
        value: int | float,
        text: str,
    ) -> int | float:
        """
        Applique le multiplicateur million/milliard pour total_file_count.

        Corrige :
        - "3 millions" -> 3_000_000
        - "5 millions" -> 5_000_000
        """

        if field != ParamName.total_file_count:
            return value

        text = text.lower()

        if "milliard" in text or "milliards" in text or "billion" in text:
            return int(value * 1_000_000_000)

        if "million" in text or "millions" in text:
            return int(value * 1_000_000)

        return value
    def _apply_money_multiplier(
        self,
        field: ParamName,
        value: int | float,
        text: str,
    ) -> int | float:
        """
        Applique les multiplicateurs million/milliard pour le budget.

        Exemples :
        - 100 millions dollars -> 100_000_000
        - 100000 millions dollars -> 100_000_000_000
        - 2 milliards dollars -> 2_000_000_000
        """

        if field != ParamName.max_budget_usd:
            return value

        text = text.lower()

        if (
            "milliard" in text
            or "milliards" in text
            or "billion" in text
            or "billions" in text
        ):
            return int(value * 1_000_000_000)

        if (
            "million" in text
            or "millions" in text
        ):
            return int(value * 1_000_000)

        return value

    def _extract_explicit_unit_after_number(self, text: str) -> str | None:
        """
        Détecte une unité explicitement écrite après un nombre.

        Exemples :
        - 20g -> g
        - 2000000megabit -> megabit
        - 2 GB -> gb
        - 9 kW -> kw
        - 50 GB/s -> gb/s
        """

        text = text.lower().strip()

        match = re.search(
            r"-?\d+(?:[\.,]\d+)?\s*([a-zA-Z/%$]+)",
            text,
        )

        if not match:
            return None

        return match.group(1).lower()

    def _unit_allowed_for_active_field(
        self,
        field: ParamName,
        unit: str | None,
    ) -> bool:
        """
        Vérifie si l'unité explicite est compatible avec la question active.

        Si aucune unité explicite n'est donnée, on accepte :
        - car la question active donne déjà l'unité attendue.

        Exemple :
        Question : taille moyenne en GB ?
        Réponse : 2
        => accepté comme 2 GB.
        """

        if unit is None:
            return True

        unit = unit.lower()

        allowed_units = {
            ParamName.requested_usable_capacity_tib: {
                "tib",
                "tb",
            },
            ParamName.average_file_size_gb: {
                "gb",
                "gib",
                "mb",
                "mib",
            },
            ParamName.max_file_size_gb: {
                "gb",
                "gib",
                "mb",
                "mib",
            },
            ParamName.target_read_gbps: {
                "gb/s",
                "gbs",
                "gbps",
            },
            ParamName.target_write_gbps: {
                "gb/s",
                "gbs",
                "gbps",
            },
            ParamName.max_power_w: {
                "w",
                "kw",
                "watt",
                "watts",
                "kilowatt",
                "kilowatts",
            },
            ParamName.max_budget_usd: {
                "usd",
                "dollar",
                "dollars",
                "$",
                "million",
                "millions",
                "milliard",
                "milliards",
                "billion",
                "billions",
            },
            ParamName.annual_growth_percent: {
                "%",
                "percent",
                "pourcent",
            },
            ParamName.client_count: {
                "client",
                "clients",
                "machine",
                "machines",
                "node",
                "nodes",
                "noeud",
                "noeuds",
                "nœud",
                "nœuds",
            },
            ParamName.total_file_count: {
                "fichier",
                "fichiers",
                "file",
                "files",
                "million",
                "millions",
                "milliard",
                "milliards",
            },
        }

        if field not in allowed_units:
            return True

        return unit in allowed_units[field]

    def _make_invalid_clarification_candidate(
        self,
        field: ParamName,
        evidence: str,
    ) -> ExtractedCandidate:
        """
        Crée un candidat volontairement invalide.

        Objectif :
        - ne pas accepter une réponse dangereuse ;
        - garder le système sur la même question ;
        - laisser StateGuard produire NEEDS_CLARIFICATION.
        """

        return ExtractedCandidate(
            field=field,
            value=None,
            unit=None,
            evidence=evidence,
            confidence=1.0,
            source=CandidateSource.USER_CLARIFICATION,
            source_text=evidence,
            turn_id=self.turn_id,
        )

    def _infer_unit_for_active_field(
        self,
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

            if re.search(r"\b(gb|gib)\b", text):
                return "GB"

            return "GB"

        if field == ParamName.max_power_w:
            if re.search(r"\b(kw|kilowatt|kilowatts)\b", text):
                return "kW"

            return "W"

        if field == ParamName.requested_usable_capacity_tib:
            if re.search(r"\b(tb|tib)\b", text):
                return "TiB"

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

    def _resolve_active_issue_directly(
        self,
        user_text: str,
        active_issue: ValidationIssue,
    ) -> ExtractedCandidate | None:
        """
        Résout une réponse courte en utilisant la question active.

        Exemples :
        - Question active : Combien de clients ?
          Réponse : 200
          => client_count = 200

        - Question active : Combien de fichiers ?
          Réponse : 3 millions
          => total_file_count = 3_000_000

        - Question active : type d'accès ?
          Réponse : mixed
          => access_type = mixed
        """

        field = active_issue.field

        if field == ParamName.access_type:
            access_type = self._parse_access_type_answer(user_text)

            if access_type is None:
                return None

            return self._make_user_clarification_candidate(
                field=ParamName.access_type,
                value=access_type,
                unit=None,
                evidence=user_text,
            )

        if field == ParamName.ha_required:
            ha_value = self._parse_ha_answer(user_text)

            if ha_value is None:
                return None

            return self._make_user_clarification_candidate(
                field=ParamName.ha_required,
                value=ha_value,
                unit=None,
                evidence=user_text,
            )

        if field == ParamName.read_write_ratio:
            ratio = self._parse_ratio_answer(user_text)

            if ratio is None:
                return None

            return self._make_user_clarification_candidate(
                field=ParamName.read_write_ratio,
                value=ratio,
                unit="%",
                evidence=user_text,
            )

        numeric_fields = {
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

        if field in numeric_fields:
            explicit_unit = self._extract_explicit_unit_after_number(user_text)

            if not self._unit_allowed_for_active_field(
                field=field,
                unit=explicit_unit,
            ):
                return self._make_invalid_clarification_candidate(
                    field=field,
                    evidence=user_text,
                )

            number = self._extract_first_number(user_text)

            if number is None:
                return None

            number = self._apply_count_multiplier(
                field=field,
                value=number,
                text=user_text,
            )
            number = self._apply_money_multiplier(
                field=field,
                value=number,
                text=user_text,
            )

            raw_unit = self._infer_unit_for_active_field(
                field=field,
                text=user_text,
            )

            value, unit = normalize_unit_value(
                field=field,
                value=number,
                unit=raw_unit,
            )

            return self._make_user_clarification_candidate(
                field=field,
                value=value,
                unit=unit,
                evidence=user_text,
            )

        return None

    # ============================================================
    # AI PLAUSIBILITY CONFIRMATION
    # ============================================================

    def _apply_ai_plausibility_report_to_state(
        self,
        report: AIPlausibilityReport,
    ) -> None:
        """
        Convertit les alertes de l'AI PlausibilityAgent en clarification.

        Important :
        - l'agent IA ne corrige rien automatiquement ;
        - le système prépare une candidate de correction ;
        - si l'utilisateur répond 'oui', cette candidate sera acceptée ;
        - si l'utilisateur donne une autre valeur, la clarification normale la parse.
        """

        issues = []

        for ai_issue in report.issues[:1]:
            field = ai_issue.field

            current_item = self.state.final_json.get(field.value)
            current_value = None
            current_unit = None

            if current_item is not None:
                current_value = current_item.value
                current_unit = current_item.unit

            suggested_candidate = None
            suggested = ai_issue.suggested_correction

            if isinstance(suggested, dict):
                suggested_value = suggested.get("value")
                suggested_unit = suggested.get("unit")

                if suggested_value is not None:
                    suggested_candidate = ExtractedCandidate(
                        field=field,
                        value=suggested_value,
                        unit=suggested_unit,
                        evidence=(
                            "AI plausibility suggested correction; "
                            "waiting for user confirmation"
                        ),
                        confidence=1.0,
                        source=CandidateSource.USER_CLARIFICATION,
                        source_text="AI plausibility suggested correction",
                        turn_id=self.turn_id,
                    )

            if suggested_candidate is None and current_value is not None:
                suggested_candidate = ExtractedCandidate(
                    field=field,
                    value=current_value,
                    unit=current_unit,
                    evidence=(
                        "AI plausibility current value; "
                        "waiting for user confirmation"
                    ),
                    confidence=1.0,
                    source=CandidateSource.USER_CLARIFICATION,
                    source_text="AI plausibility current value confirmation",
                    turn_id=self.turn_id,
                )

            question = self._build_ai_plausibility_question(
                ai_issue=ai_issue,
                current_value=current_value,
            )

            should_clear_field = False

            suggested = ai_issue.suggested_correction

            # On efface le champ seulement si l'agent propose une correction directe.
            # Exemple :
            # - max_power_w = 50 W
            # - correction proposée = 50000 W
            #
            # Pour une incohérence relationnelle sans correction directe,
            # on garde les valeurs déjà extraites.
            if isinstance(suggested, dict) and suggested.get("value") is not None:
                should_clear_field = True

            if should_clear_field and field.value in self.state.final_json:
                self.state.final_json[field.value] = None

            candidates = []

            if suggested_candidate is not None:
                candidates.append(suggested_candidate)

            issues.append(
                ValidationIssue(
                    type=IssueType.INVALID_VALUE,
                    field=field,
                    message=ai_issue.message,
                    question=question,
                    candidates=candidates,
                )
            )

        self.state.stage = PipelineStage.CLARIFICATION
        self.state.status = ChatbotStatus.NEEDS_CLARIFICATION

        self.state.issues = issues
        self.state.missing_fields = []
        self.state.conflicting_fields = []
        self.state.invalid_fields = [
            issue.field.value
            for issue in issues
        ]
        self.state.unsupported_fields = []
        self.state.questions = [
            issues[0].question
        ] if issues else []

        self.state.calculation_result = {}

    def _build_ai_plausibility_question(
        self,
        ai_issue: AIPlausibilityIssue,
        current_value,
    ) -> str:
        suggested = ai_issue.suggested_correction
        issue_type = ai_issue.issue_type.lower()

        if (
            "average_file_size" in issue_type
            or "total_file_count" in issue_type
            or "file_size" in issue_type
        ):
            return (
                f"{ai_issue.message} "
                "Cette incohérence peut venir de la capacité demandée, "
                "de la taille moyenne des fichiers ou du nombre total de fichiers. "
                "Veuillez corriger la valeur concernée, par exemple : "
                "'capacité 6000 TiB', 'taille moyenne 200 MB', "
                "ou 'nombre de fichiers 300000'."
            )
        if isinstance(suggested, dict):
            suggested_value = suggested.get("value")
            suggested_unit = suggested.get("unit")

            if suggested_value is not None:
                unit_text = f" {suggested_unit}" if suggested_unit else ""

                return (
                    f"{ai_issue.message} "
                    f"Proposition : {suggested_value}{unit_text}. "
                    "Répondez 'oui' pour confirmer cette valeur, "
                    "ou donnez une autre valeur correcte."
                )

        return (
            f"{ai_issue.message} "
            "Veuillez confirmer cette valeur ou donner une autre valeur correcte."
        )

    def _try_ai_plausibility_confirmation(
        self,
        user_text: str,
    ) -> bool:
        """
        Gère une confirmation courte après une question du PlausibilityAgent.

        Exemple :
        Question : Proposition : 50000 W. Répondez 'oui'...
        User    : oui
        Résultat: max_power_w = 50000

        Si l'utilisateur donne une valeur explicite comme '50 kw',
        on laisse _try_clarification parser la valeur normalement.
        """

        if self.state.status != ChatbotStatus.NEEDS_CLARIFICATION:
            return False

        if not self.state.issues:
            return False

        active_issue = self.state.issues[0]

        if not active_issue.candidates:
            return False

        has_number = self._extract_first_number(user_text) is not None

        if has_number:
            return False

        if not (
            self._is_yes_answer(user_text)
            or self._is_unit_confirmation_answer(user_text)
        ):
            return False

        candidate = active_issue.candidates[0]
        field = candidate.field

        self._remove_candidates_for_fields({field.value})

        confirmed_candidate = ExtractedCandidate(
            field=field,
            value=candidate.value,
            unit=candidate.unit,
            evidence=user_text,
            confidence=1.0,
            source=CandidateSource.USER_CLARIFICATION,
            source_text=user_text,
            turn_id=self.turn_id,
        )

        self._extend_candidates([confirmed_candidate])

        self._mark_plausibility_confirmed(
            field=field,
            value=confirmed_candidate.value,
        )

        self.state.issues = []
        self.state.questions = []

        self._run_validation_and_maybe_calculate()

        return True

    def _drop_confirmed_ai_plausibility_issues(
        self,
        report: AIPlausibilityReport,
    ) -> AIPlausibilityReport:
        """
        Évite que l'agent IA redemande la même clarification
        après confirmation utilisateur.
        """

        kept_issues = []

        for issue in report.issues:
            item = self.state.final_json.get(issue.field.value)

            if item is None:
                kept_issues.append(issue)
                continue

            if self._is_plausibility_confirmed(
                field=issue.field,
                value=item.value,
            ):
                continue

            kept_issues.append(issue)

        if not kept_issues:
            return AIPlausibilityReport(
                status=AIPlausibilityStatus.OK,
                issues=[],
                raw_response=report.raw_response,
            )

        return AIPlausibilityReport(
            status=report.status,
            issues=kept_issues,
            raw_response=report.raw_response,
        )

    # ============================================================
    # VALIDATION AND CALCULATION
    # ============================================================

    def _apply_report_to_state(self, report) -> None:
        self.state.stage = report.stage
        self.state.status = report.status
        self.state.final_json = report.final_json

        self.state.issues = report.issues

        self.state.missing_fields = report.missing_fields
        self.state.conflicting_fields = report.conflicting_fields
        self.state.invalid_fields = report.invalid_fields
        self.state.unsupported_fields = report.unsupported_fields
        self.state.questions = report.questions

        if self.state.status == ChatbotStatus.NEEDS_CLARIFICATION:
            self.state.calculation_result = {}

    def _run_validation_and_maybe_calculate(self) -> RequirementState:
        self.state.stage = PipelineStage.VALIDATION

        # 1. StateGuard : validation locale des champs
        report = self.guard.analyze(
            self.state.extracted_candidates
        )

        self._apply_report_to_state(report)

        if self.state.status == ChatbotStatus.NEEDS_CLARIFICATION:
            return self.state

        # 2. AI PlausibilityAgent : validation globale de cohérence
        plausibility_report = self.plausibility_agent.analyze(
            self.state.final_json
        )

        plausibility_report = self._drop_confirmed_ai_plausibility_issues(
            plausibility_report
        )

        if plausibility_report.needs_clarification:
            self._apply_ai_plausibility_report_to_state(
                plausibility_report
            )
            return self.state

        # 3. CalculationEngine uniquement si l'état est valide et plausible
        self.state.stage = PipelineStage.CALCULATION

        self.state.calculation_result = self.calculation_engine.calculate(
            self.state.final_json
        )

        self.state.stage = PipelineStage.RECOMMENDATION

        return self.state

    # ============================================================
    # CLARIFICATION FLOW
    # ============================================================

    def _try_clarification(self, user_text: str) -> bool:
        """
        Essaie d'abord de résoudre la question active.

        Message court :
        - on résout uniquement la première question active.

        Message riche :
        - on résout la question active si possible ;
        - puis on laisse aussi l'extracteur rule-first lire les autres infos.
        """

        if self.state.status != ChatbotStatus.NEEDS_CLARIFICATION:
            return False

        if not self.state.issues:
            return False

        active_issue = self.state.issues[0]
        # Si l'alerte vient de l'AI PlausibilityAgent et que l'utilisateur
        # donne une mise à jour explicite, on laisse l'extracteur normal lire
        # le message au lieu de forcer la réponse sur le champ actif.
        if active_issue.candidates and self._is_explicit_update(user_text):
            return False
        direct_candidate = self._resolve_active_issue_directly(
            user_text=user_text,
            active_issue=active_issue,
        )

        if direct_candidate is not None:
            fields = {direct_candidate.field.value}

            self._remove_candidates_for_fields(fields)
            self._extend_candidates([direct_candidate])

            if active_issue.candidates:
                self._mark_plausibility_confirmed(
                    field=direct_candidate.field,
                    value=direct_candidate.value,
                )

            self._run_validation_and_maybe_calculate()
            return True

        if self._is_rich_message(user_text):
            issues_for_resolution = self.state.issues
        else:
            issues_for_resolution = [active_issue]

        decision = self.clarification_agent.resolve(
            user_text=user_text,
            issues=issues_for_resolution,
            turn_id=self.turn_id,
        )

        if decision.resolved_candidates:
            fields = {
                candidate.field.value
                for candidate in decision.resolved_candidates
            }

            self._remove_candidates_for_fields(fields)
            self._extend_candidates(decision.resolved_candidates)

            if active_issue.candidates:
                for candidate in decision.resolved_candidates:
                    self._mark_plausibility_confirmed(
                        field=candidate.field,
                        value=candidate.value,
                    )

            if self._is_rich_message(user_text):
                return False

            self._run_validation_and_maybe_calculate()
            return True

        if decision.should_ask_user and not self._is_rich_message(user_text):
            self.state.stage = PipelineStage.CLARIFICATION
            self.state.status = ChatbotStatus.NEEDS_CLARIFICATION
            self.state.questions = [active_issue.question]
            return True

        return False

    # ============================================================
    # MAIN ENTRY POINT
    # ============================================================

    def process_user_message(self, user_text: str) -> RequirementState:
        """
        Traite un nouveau message utilisateur.
        """

        self.turn_id += 1
        self.state.raw_user_inputs.append(user_text)

        if self._try_ai_plausibility_confirmation(user_text):
            return self.state

        if self._try_clarification(user_text):
            return self.state

        self.state.stage = PipelineStage.EXTRACTION

        extraction_result = self.extractor.extract(
            user_text=user_text,
            turn_id=self.turn_id,
        )

        new_fields = {
            candidate.field.value
            for candidate in extraction_result.candidates
        }

        # Si l'utilisateur fait une mise à jour explicite,
        # les nouvelles valeurs remplacent les anciennes.
        if self._is_explicit_update(user_text) and new_fields:
            self._remove_candidates_for_fields(new_fields)

        # Si on était en clarification, les champs en attente
        # sont remplacés par les nouvelles réponses.
        if self.state.status == ChatbotStatus.NEEDS_CLARIFICATION:
            pending = set(self._pending_fields())

            fields_to_replace = {
                field
                for field in new_fields
                if field in pending
            }

            if fields_to_replace:
                self._remove_candidates_for_fields(fields_to_replace)

        self._extend_candidates(extraction_result.candidates)

        return self._run_validation_and_maybe_calculate()