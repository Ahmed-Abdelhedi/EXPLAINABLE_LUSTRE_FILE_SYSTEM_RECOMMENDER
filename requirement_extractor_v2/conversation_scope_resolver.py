from __future__ import annotations

import re
import unicodedata
from typing import Optional

from .field_defs import TARGET_UNITS
from .models import (
    ParamName,
    ScopeIntent,
    ScopeResolution,
)


class ConversationScopeResolver:
    """
    Conservative deterministic resolver for the first stage of the
    selective requirement-extraction cascade.

    Responsibilities
    ----------------
    - distinguish a new requirement from a short answer to an active
      clarification question;
    - detect explicit corrections;
    - reject clearly unrelated messages;
    - inherit the unit requested by the previous question only when the
      user gives a unitless contextual answer.

    Non-responsibilities
    --------------------
    - it does not extract requirement values;
    - it does not assign semantic roles;
    - it does not validate business constraints;
    - it does not call an LLM.

    The resolver is deliberately conservative.

    Unclear but potentially in-domain messages remain NEW_REQUIREMENT and
    are handled by later cascade stages.
    """

    # =================================================================
    # CORRECTION SIGNALS
    # =================================================================

    _CORRECTION_PATTERNS = (
        # -------------------------------------------------------------
        # French explicit correction markers
        # -------------------------------------------------------------
        r"\bfinalement\b",
        r"\bau\s+lieu\s+de\b",
        r"\ba\s+la\s+place\b",
        r"\bje\s+prefere\b",

        r"\b(?:mettre|mets?|mettons|mettez|mis|mise|mises|mettent)"
        r"\s+a\s+jour\b",

        r"\bmise\s+a\s+jour\b",

        r"\bmodifi(?:er|e|es|ez|ons|ent|ee?s?)\b",
        r"\bchang(?:er|e|es|ez|ons|ent|ee?s?)\b",
        r"\bcorrig(?:er|e|es|ez|ons|ent|ee?s?)\b",
        r"\bremplac(?:er|e|es|ez|ons|ent|ee?s?)\b",
        r"\bajust(?:er|e|es|ez|ons|ent|ee?s?)\b",
        r"\brevis(?:er|e|es|ez|ons|ent|ee?s?)\b",

        # -------------------------------------------------------------
        # English explicit correction markers
        # -------------------------------------------------------------
        r"\bactually\b",
        r"\binstead\b",

        r"\bupdate(?:d|s|ing)?\b",
        r"\bchange(?:d|s|ing)?\b",
        r"\breplace(?:d|s|ing)?\b",
        r"\bcorrect(?:ed|s|ing)?\b",
        r"\badjust(?:ed|s|ing)?\b",

        # IMPORTANT:
        #
        # The previous implementation included:
        #
        #     r"\bset\b"
        #
        # This was too broad.
        #
        # Example:
        #
        #     "Set it to 50."
        #
        # Without previous conversational state this is not sufficient
        # evidence for CORRECTION, so "set" alone is intentionally no
        # longer treated as a correction marker.
    )

    # =================================================================
    # OUT-OF-SCOPE SIGNALS
    # =================================================================

    _OUT_OF_SCOPE_PATTERNS = (
        # -------------------------------------------------------------
        # Weather
        # -------------------------------------------------------------
        r"\b(?:weather|meteo)\b",

        r"\b(?:"
        r"temperature\s+(?:today|outside)"
        r"|"
        r"temps\s+qu[' ]?il\s+fait"
        r")\b",

        # -------------------------------------------------------------
        # Cooking
        # -------------------------------------------------------------
        r"\b(?:recipe|recette|cook|cooking|cuisiner)\b",

        # -------------------------------------------------------------
        # Jokes
        # -------------------------------------------------------------
        r"\b(?:"
        r"tell\s+me\s+a\s+joke"
        r"|"
        r"raconte[- ]moi\s+une\s+blague"
        r")\b",

        # -------------------------------------------------------------
        # Movie / music recommendation
        #
        # Support BOTH directions:
        #
        #   recommend a movie
        #   movie recommendation
        #
        # and French equivalents.
        # -------------------------------------------------------------
        r"\b(?:recommend|suggest)\w*\b"
        r".{0,30}"
        r"\b(?:movie|film|music|song)\w*\b",

        r"\b(?:movie|film|music|song)\w*\b"
        r".{0,30}"
        r"\b(?:recommend|suggest)\w*\b",

        r"\b(?:recommand|sugger)\w*\b"
        r".{0,30}"
        r"\b(?:film|musique|chanson)\w*\b",

        r"\b(?:film|musique|chanson)\w*\b"
        r".{0,30}"
        r"\b(?:recommand|sugger)\w*\b",

        # -------------------------------------------------------------
        # Football / sport scores
        #
        # Support BOTH:
        #
        #   football score
        #   score of the football match
        #
        # French normalization turns:
        #
        #   "Quel est le score du match de football ?"
        #
        # into:
        #
        #   "quel est le score du match de football ?"
        # -------------------------------------------------------------
        r"\b(?:football|soccer|basketball|tennis)\b"
        r".{0,40}"
        r"\b(?:score|match|game)\b",

        r"\b(?:score|match|game)\b"
        r".{0,40}"
        r"\b(?:football|soccer|basketball|tennis)\b",
    )

    # =================================================================
    # SHORT CONTEXTUAL VALUES
    # =================================================================

    _YES_NO_VALUES = {
        "oui",
        "non",
        "yes",
        "no",
        "true",
        "false",

        "required",
        "mandatory",
        "not required",

        "obligatoire",
        "pas obligatoire",

        "sans ha",
        "avec ha",

        "ha required",
        "no ha",
        "yes ha",
    }

    _ACCESS_VALUES = {
        "mixed",
        "mixte",
        "random",
        "aleatoire",
        "sequential",
        "sequentiel",
        "parallel",
        "parallele",
        "streaming",
    }

    # =================================================================
    # UNITS
    # =================================================================

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

    # =================================================================
    # NUMBERS
    # =================================================================

    _NUMBER_RE = re.compile(
        r"[+-]?\d+(?:[\s_.,]\d+)*",
        flags=re.IGNORECASE,
    )

    _RATIO_RE = re.compile(
        r"\b\d{1,3}\s*/\s*\d{1,3}\b",
        flags=re.IGNORECASE,
    )

    # =================================================================
    # PUBLIC API
    # =================================================================

    def resolve(
        self,
        user_text: str,
        previous_question_field: Optional[
            ParamName
        ] = None,
        requested_unit: Optional[str] = None,
        previous_question: Optional[str] = None,
    ) -> ScopeResolution:
        """
        Classify the current turn using the message plus conversation state.

        Parameters
        ----------
        user_text:
            Current user message.

        previous_question_field:
            Field currently requested by the chatbot.

        requested_unit:
            Explicit unit requested by the active clarification question.

        previous_question:
            Previous clarification question text.

        Notes
        -----
        If the user provides a short unitless answer to an active
        clarification, the requested/canonical unit can be inherited.

        Example:

            previous question:
                "What is the maximum power in watts?"

            user:
                "200"

            result:
                ANSWER_TO_PREVIOUS_QUESTION
                target_field=max_power_w
                inherited_unit=W
        """

        raw = (
            user_text
            or ""
        )

        normalized = self._normalize(
            raw
        )

        # -------------------------------------------------------------
        # 1. Empty message
        # -------------------------------------------------------------

        if not normalized:

            return ScopeResolution(
                intent=ScopeIntent.OUT_OF_SCOPE,
                reason="empty_message",
            )

        # -------------------------------------------------------------
        # 2. Explicit correction
        # -------------------------------------------------------------

        if self._is_explicit_correction(
            normalized
        ):

            return ScopeResolution(
                intent=ScopeIntent.CORRECTION,
                reason=(
                    "explicit_correction_marker"
                ),
            )

        # -------------------------------------------------------------
        # 3. Contextual answer
        # -------------------------------------------------------------
        #
        # A rich message must NOT automatically be attached to the active
        # clarification question.
        #
        # Users remain free to provide multiple fresh requirements.
        # -------------------------------------------------------------

        if (
            previous_question_field
            is not None
        ):

            if self._looks_like_contextual_answer(
                normalized
            ):

                inherited_unit = None

                if not self._contains_explicit_unit(
                    raw
                ):

                    inherited_unit = (
                        requested_unit
                        if requested_unit
                        is not None
                        else TARGET_UNITS.get(
                            previous_question_field
                        )
                    )

                return ScopeResolution(
                    intent=(
                        ScopeIntent
                        .ANSWER_TO_PREVIOUS_QUESTION
                    ),
                    target_field=
                        previous_question_field,
                    inherited_unit=
                        inherited_unit,
                    reason=(
                        "short_answer_bound_"
                        "to_active_question"
                    ),
                )

        # -------------------------------------------------------------
        # 4. Clearly unrelated topic
        # -------------------------------------------------------------

        if self._is_explicit_out_of_scope(
            normalized
        ):

            return ScopeResolution(
                intent=ScopeIntent.OUT_OF_SCOPE,
                reason=(
                    "clearly_unrelated_topic"
                ),
            )

        # -------------------------------------------------------------
        # 5. Conservative default
        # -------------------------------------------------------------
        #
        # An unclear message is NOT automatically rejected.
        #
        # Later extraction stages can still determine whether it contains
        # useful requirement information.
        # -------------------------------------------------------------

        return ScopeResolution(
            intent=ScopeIntent.NEW_REQUIREMENT,
            reason=(
                "continue_with_requirement_extraction"
            ),
        )

    # =================================================================
    # NORMALIZATION
    # =================================================================

    def _normalize(
        self,
        text: str,
    ) -> str:

        value = (
            text
            .strip()
            .lower()
            .replace("’", "'")
            .replace("`", "'")
        )

        value = unicodedata.normalize(
            "NFKD",
            value,
        )

        value = "".join(
            char
            for char in value
            if not unicodedata.combining(
                char
            )
        )

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value.strip()

    # =================================================================
    # CORRECTION
    # =================================================================

    def _is_explicit_correction(
        self,
        normalized: str,
    ) -> bool:

        return any(
            re.search(
                pattern,
                normalized,
            )
            for pattern
            in self._CORRECTION_PATTERNS
        )

    # =================================================================
    # OUT OF SCOPE
    # =================================================================

    def _is_explicit_out_of_scope(
        self,
        normalized: str,
    ) -> bool:

        return any(
            re.search(
                pattern,
                normalized,
            )
            for pattern
            in self._OUT_OF_SCOPE_PATTERNS
        )

    # =================================================================
    # UNITS
    # =================================================================

    def _contains_explicit_unit(
        self,
        raw_text: str,
    ) -> bool:

        return bool(
            self._EXPLICIT_UNIT_RE.search(
                raw_text
            )
        )

    # =================================================================
    # CONTEXTUAL ANSWER
    # =================================================================

    def _looks_like_contextual_answer(
        self,
        normalized: str,
    ) -> bool:
        """
        Detect short replies that can safely inherit meaning from an
        active clarification question.

        Examples
        --------
        200
        15 kW
        70/30
        oui
        mixed

        Rich messages intentionally return False so the complete extraction
        cascade handles them rather than attaching every quantity to one
        active field.
        """

        # -------------------------------------------------------------
        # Ratios
        # -------------------------------------------------------------

        if self._RATIO_RE.fullmatch(
            normalized
        ):
            return True

        # -------------------------------------------------------------
        # Rich requirement protection
        # -------------------------------------------------------------

        if self._looks_like_rich_requirement(
            normalized
        ):
            return False

        # -------------------------------------------------------------
        # Closed values
        # -------------------------------------------------------------

        if normalized in self._YES_NO_VALUES:
            return True

        if normalized in self._ACCESS_VALUES:
            return True

        # -------------------------------------------------------------
        # Numeric short reply
        # -------------------------------------------------------------

        if self._NUMBER_RE.search(
            normalized
        ):
            return True

        # -------------------------------------------------------------
        # Unit-only clarification
        # -------------------------------------------------------------

        if self._EXPLICIT_UNIT_RE.fullmatch(
            normalized
        ):
            return True

        return False

    # =================================================================
    # RICH REQUIREMENT DETECTION
    # =================================================================

    def _looks_like_rich_requirement(
        self,
        normalized: str,
    ) -> bool:
        """
        Protect full requirement statements from accidental contextual
        binding.

        Examples
        --------
        "500 TiB for 200 clients"
            -> rich/new requirement

        "200"
            -> possible short contextual answer
        """

        number_count = len(
            self._NUMBER_RE.findall(
                normalized
            )
        )

        separator_count = (
            normalized.count(",")
            + normalized.count(";")
            + normalized.count(" and ")
            + normalized.count(" et ")
        )

        token_count = len(
            normalized.split()
        )

        # Multiple numerical quantities strongly suggest a fresh/rich
        # requirement statement.
        if number_count >= 2:
            return True

        # Multiple semantic clauses also suggest a rich statement.
        if separator_count >= 2:
            return True

        # A longer sentence containing one number can still be a meaningful
        # new requirement rather than a short clarification response.
        if token_count >= 8:
            return True

        return False