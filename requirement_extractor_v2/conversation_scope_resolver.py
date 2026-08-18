from __future__ import annotations

import re
import unicodedata
from typing import Optional

from .field_defs import TARGET_UNITS
from .models import ParamName, ScopeIntent, ScopeResolution


class ConversationScopeResolver:
    """
    Conservative deterministic resolver for the first stage of the
    selective requirement-extraction cascade.

    Responsibilities:
    - distinguish a new requirement from a short answer to an active
      clarification question;
    - detect explicit corrections;
    - reject clearly unrelated messages;
    - inherit the unit requested by the previous question only when the
      user gives a unitless contextual answer.

    Non-responsibilities:
    - it does not extract requirement values;
    - it does not assign semantic roles;
    - it does not validate business constraints;
    - it does not call an LLM.

    The resolver is deliberately conservative. Unclear in-domain messages
    remain NEW_REQUIREMENT and are handled by later cascade stages.
    """

    _CORRECTION_PATTERNS = (
        r"\bfinalement\b",
        r"\bplutot\b",
        r"\bau\s+lieu\s+de\b",
        r"\ba\s+la\s+place\b",
        r"\bje\s+prefere\b",
        r"\b(?:mettre|mets?|mettons|mettez|mis|mise|mises|mettent)\s+a\s+jour\b",
        r"\bmise\s+a\s+jour\b",
        r"\bmodifi(?:er|e|es|ez|ons|ent|ee?s?)\b",
        r"\bchang(?:er|e|es|ez|ons|ent|ee?s?)\b",
        r"\bcorrig(?:er|e|es|ez|ons|ent|ee?s?)\b",
        r"\bremplac(?:er|e|es|ez|ons|ent|ee?s?)\b",
        r"\bajust(?:er|e|es|ez|ons|ent|ee?s?)\b",
        r"\brevis(?:er|e|es|ez|ons|ent|ee?s?)\b",
        r"\bupdate(?:d|s|ing)?\b",
        r"\bchange(?:d|s|ing)?\b",
        r"\breplace(?:d|s|ing)?\b",
        r"\bcorrect(?:ed|s|ing)?\b",
        r"\badjust(?:ed|s|ing)?\b",
        r"\bset\b",
    )

    _OUT_OF_SCOPE_PATTERNS = (
        r"\b(?:weather|meteo)\b",
        r"\b(?:temperature\s+(?:today|outside)|temps\s+qu[' ]?il\s+fait)\b",
        r"\b(?:football|soccer|basketball|tennis)\s+(?:score|match|game)\b",
        r"\b(?:recipe|recette|cook|cooking|cuisiner)\b",
        r"\b(?:tell\s+me\s+a\s+joke|raconte[- ]moi\s+une\s+blague)\b",
        r"\b(?:movie|film|music|musique)\s+(?:recommendation|recommend|recommande)\b",
    )

    _YES_NO_VALUES = {
        "oui", "non", "yes", "no", "true", "false",
        "required", "mandatory", "not required",
        "obligatoire", "pas obligatoire", "sans ha", "avec ha",
        "ha required", "no ha", "yes ha",
    }

    _ACCESS_VALUES = {
        "mixed", "mixte", "random", "aleatoire", "sequential",
        "sequentiel", "parallel", "parallele", "streaming",
    }

    _EXPLICIT_UNIT_RE = re.compile(
        r"(?:"
        r"TiB|TB|GiB|GB|MiB|MB|"
        r"GB/s|Gbps|MB/s|Mbps|"
        r"kW|MW|W|watts?|kilowatts?|"
        r"USD|dollars?|\$|"
        r"%|percent|pourcent"
        r")",
        flags=re.IGNORECASE,
    )

    _NUMBER_RE = re.compile(
        r"[+-]?\d+(?:[\s_.,]\d+)*",
        flags=re.IGNORECASE,
    )

    _RATIO_RE = re.compile(
        r"\b\d{1,3}\s*/\s*\d{1,3}\b",
        flags=re.IGNORECASE,
    )

    def resolve(
        self,
        user_text: str,
        previous_question_field: Optional[ParamName] = None,
        requested_unit: Optional[str] = None,
        previous_question: Optional[str] = None,
    ) -> ScopeResolution:
        """
        Classify the current turn using the message plus conversation state.

        `previous_question_field` represents the field currently requested by
        the chatbot. `requested_unit` is the unit explicitly requested by the
        question when known. If omitted, the canonical target unit from
        `field_defs.TARGET_UNITS` is used for a unitless contextual answer.
        """

        raw = user_text or ""
        normalized = self._normalize(raw)

        if not normalized:
            return ScopeResolution(
                intent=ScopeIntent.OUT_OF_SCOPE,
                reason="empty_message",
            )

        if self._is_explicit_correction(normalized):
            return ScopeResolution(
                intent=ScopeIntent.CORRECTION,
                reason="explicit_correction_marker",
            )

        # A rich message must not be force-bound to the active clarification.
        # This preserves the useful behavior already present in the current
        # chatbot: users are allowed to provide several requirements at once.
        if previous_question_field is not None:
            if self._looks_like_contextual_answer(normalized):
                inherited_unit = None

                if not self._contains_explicit_unit(raw):
                    inherited_unit = (
                        requested_unit
                        if requested_unit is not None
                        else TARGET_UNITS.get(previous_question_field)
                    )

                return ScopeResolution(
                    intent=ScopeIntent.ANSWER_TO_PREVIOUS_QUESTION,
                    target_field=previous_question_field,
                    inherited_unit=inherited_unit,
                    reason="short_answer_bound_to_active_question",
                )

        if self._is_explicit_out_of_scope(normalized):
            return ScopeResolution(
                intent=ScopeIntent.OUT_OF_SCOPE,
                reason="clearly_unrelated_topic",
            )

        return ScopeResolution(
            intent=ScopeIntent.NEW_REQUIREMENT,
            reason="continue_with_requirement_extraction",
        )

    def _normalize(self, text: str) -> str:
        value = (
            text.strip()
            .lower()
            .replace("’", "'")
            .replace("`", "'")
        )

        value = unicodedata.normalize("NFKD", value)
        value = "".join(
            char
            for char in value
            if not unicodedata.combining(char)
        )

        value = re.sub(r"\s+", " ", value)
        return value.strip()

    def _is_explicit_correction(self, normalized: str) -> bool:
        return any(
            re.search(pattern, normalized)
            for pattern in self._CORRECTION_PATTERNS
        )

    def _is_explicit_out_of_scope(self, normalized: str) -> bool:
        return any(
            re.search(pattern, normalized)
            for pattern in self._OUT_OF_SCOPE_PATTERNS
        )

    def _contains_explicit_unit(self, raw_text: str) -> bool:
        return bool(self._EXPLICIT_UNIT_RE.search(raw_text))

    def _looks_like_contextual_answer(self, normalized: str) -> bool:
        """
        Detect short replies that can safely inherit meaning from the active
        clarification question.

        Examples:
        - "200"
        - "15 kW"
        - "70/30"
        - "oui"
        - "mixed"

        Rich messages intentionally return False so that the full extraction
        cascade handles them instead of binding every quantity to one field.
        """

        # Ratios such as "70/30" contain two numbers but are still a
        # canonical short answer to a read/write clarification.
        if self._RATIO_RE.fullmatch(normalized):
            return True

        if self._looks_like_rich_requirement(normalized):
            return False

        if normalized in self._YES_NO_VALUES:
            return True

        if normalized in self._ACCESS_VALUES:
            return True

        if self._NUMBER_RE.search(normalized):
            return True

        # Unit-only clarification such as "kW" after a power question.
        if self._EXPLICIT_UNIT_RE.fullmatch(normalized):
            return True

        return False

    def _looks_like_rich_requirement(self, normalized: str) -> bool:
        number_count = len(self._NUMBER_RE.findall(normalized))

        separator_count = (
            normalized.count(",")
            + normalized.count(";")
            + normalized.count(" and ")
            + normalized.count(" et ")
        )

        token_count = len(normalized.split())

        # Multiple quantities are enough to treat the turn as a fresh/rich
        # requirement rather than a short answer to one active question.
        if number_count >= 2:
            return True

        if separator_count >= 2:
            return True

        # A longer sentence with one number can still be a meaningful new
        # requirement statement. Short contextual replies stay below this.
        if token_count >= 8:
            return True

        return False