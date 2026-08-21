from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from .models import (
    ParamName,
    Quantity,
    SemanticLink,
    SemanticRole,
)

from .semantic_linker.compatibility import (
    allowed_fields_for_dimension,
    allowed_roles_for_field,
    is_valid_field_role_pair,
)

from .semantic_linker.labels import SemanticField


class LLMFallbackExtractor:
    """
    Fallback LLM sélectif de Requirement Extractor V2.

    V2.1 strategy
    -------------
    This version keeps the successful Prompt V2 semantic guide, while adding
    only targeted deterministic protections/corrections discovered during the
    regression benchmark:

    - explicit ambiguity guard before the LLM call;
    - deterministic role canonicalization for fields that have one concrete
      business role;
    - conservative power-role repair from explicit lexical cues;
    - previous-question support remains semantic context only;
    - evidence must always come from the current user message.

    Responsibility
    --------------
    QuantityScanner has already detected the quantity and its value.

    The LLM MUST NOT:
    - extract another value;
    - change the detected value;
    - convert the unit;
    - invent a Requirement field;
    - bypass compatibility rules;
    - make the final business decision.

    The LLM ONLY proposes:

        Quantity
            ↓
        FIELD + ROLE

    The DeterministicVerifier remains the final authority.
    """

    def __init__(
        self,
        enabled: Optional[bool] = None,
        host: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:

        load_dotenv()

        env_enabled = (
            os.getenv(
                "ENABLE_LLM_FALLBACK",
                "false",
            )
            .strip()
            .lower()
            == "true"
        )

        self.enabled = (
            env_enabled
            if enabled is None
            else enabled
        )

        self.host = (
            host
            or os.getenv(
                "OLLAMA_HOST",
                "http://localhost:11434",
            )
        )

        self.model = (
            model
            or os.getenv(
                "OLLAMA_MODEL",
                "qwen2.5-coder:7b",
            )
        )

        self.call_count = 0
        self.call_log = []

    # ================================================================
    # TEXT / EVIDENCE SAFETY
    # ================================================================

    @staticmethod
    def _normalize_spaces(
        text: str,
    ) -> str:
        return " ".join(
            (text or "").strip().split()
        )

    @staticmethod
    def _clean_evidence(
        evidence: str,
    ) -> str:
        """
        Remove internal [Q] markers before validating evidence.
        """

        return (
            (evidence or "")
            .replace("[Q]", "")
            .replace("[/Q]", "")
            .strip()
        )

    def _evidence_is_supported(
        self,
        evidence: str,
        user_text: str,
    ) -> bool:
        """
        Evidence must really come from the current user text.
        """

        if not evidence:
            return False

        if evidence in user_text:
            return True

        evidence_norm = (
            self._normalize_spaces(
                evidence
            )
        )

        text_norm = (
            self._normalize_spaces(
                user_text
            )
        )

        return (
            evidence_norm
            in text_norm
        )

    def _evidence_targets_quantity(
        self,
        evidence: str,
        quantity: Quantity,
    ) -> bool:
        """
        Evidence must contain the target Quantity raw span.
        """

        if not evidence:
            return False

        raw = quantity.raw.strip()

        if not raw:
            return False

        if raw in evidence:
            return True

        return (
            self._normalize_spaces(raw)
            in self._normalize_spaces(
                evidence
            )
        )

    # ================================================================
    # TARGET MARKING
    # ================================================================

    @staticmethod
    def _mark_target_quantity(
        user_text: str,
        quantity: Quantity,
    ) -> str:
        """
        Mark only the target quantity with [Q] ... [/Q].
        """

        start = quantity.start
        end = quantity.end

        if (
            0 <= start < end <= len(user_text)
            and
            user_text[start:end]
            == quantity.raw
        ):
            return (
                user_text[:start]
                + "[Q]"
                + user_text[start:end]
                + "[/Q]"
                + user_text[end:]
            )

        index = user_text.find(
            quantity.raw
        )

        if index >= 0:
            end_index = (
                index
                + len(quantity.raw)
            )

            return (
                user_text[:index]
                + "[Q]"
                + user_text[
                    index:end_index
                ]
                + "[/Q]"
                + user_text[end_index:]
            )

        return user_text

    # ================================================================
    # COMPATIBILITY
    # ================================================================

    @staticmethod
    def _build_allowed_pairs(
        quantity: Quantity,
    ) -> list[Dict[str, str]]:
        """
        Build exactly the FIELD/ROLE pairs allowed by the deterministic
        compatibility rules used by the Semantic Linker.
        """

        pairs: list[
            Dict[str, str]
        ] = []

        allowed_fields = (
            allowed_fields_for_dimension(
                quantity.dimension
            )
        )

        for field in allowed_fields:

            roles = (
                allowed_roles_for_field(
                    field
                )
            )

            for role in roles:
                pairs.append(
                    {
                        "field": field.value,
                        "role": role.value,
                    }
                )

        return pairs

    # ================================================================
    # V2.1 TARGETED SAFETY / ROLE REPAIR
    # ================================================================

    _EXPLICIT_AMBIGUITY_PATTERNS = (
        r"\bnot clear whether\b",
        r"\bunclear whether\b",
        r"\bwithout saying whether\b",
        r"\bwithout specifying whether\b",
        r"\bwithout indicating whether\b",
        r"\bdoes not say whether\b",
        r"\bdoes not specify whether\b",
        r"\bdoes not indicate whether\b",
        r"\bwithout saying if\b",
        r"\bsans préciser\b",
        r"\bsans indiquer\b",
        r"\bsans dire\b",
        r"\bne précise pas si\b",
        r"\bne précise pas s'il\b",
        r"\bne précise pas s’elle\b",
        r"\bne précise pas s'elle\b",
        r"\bpas clair si\b",
    )

    # Fields for which the concrete business role is deterministic once
    # the FIELD is correct.
    _CANONICAL_ROLE_BY_FIELD = {
        SemanticField.CLIENT_COUNT:
            SemanticRole.TOTAL_COUNT,

        SemanticField.TOTAL_FILE_COUNT:
            SemanticRole.TOTAL_COUNT,

        SemanticField.AVERAGE_FILE_SIZE_GB:
            SemanticRole.AVERAGE_VALUE,

        SemanticField.MAX_FILE_SIZE_GB:
            SemanticRole.MAXIMUM_LIMIT,

        SemanticField.READ_WRITE_RATIO:
            SemanticRole.RATIO_COMPONENT,

        SemanticField.ANNUAL_GROWTH_PERCENT:
            SemanticRole.GROWTH_RATE,
    }

    @classmethod
    def _has_explicit_ambiguity(
        cls,
        text: str,
    ) -> bool:
        """
        Explicit user-stated ambiguity must always win over LLM guessing.
        """

        normalized = (
            cls._normalize_spaces(
                text
            )
            .casefold()
        )

        return any(
            re.search(
                pattern,
                normalized,
                flags=re.IGNORECASE,
            )
            is not None
            for pattern
            in cls._EXPLICIT_AMBIGUITY_PATTERNS
        )

    @classmethod
    def _canonical_role_for_field(
        cls,
        field: SemanticField,
    ) -> Optional[SemanticRole]:
        """
        Return a deterministic role only when the selected FIELD has one
        concrete business role in the current contract.
        """

        return (
            cls
            ._CANONICAL_ROLE_BY_FIELD
            .get(field)
        )

    @staticmethod
    def _repair_power_role_from_text(
        user_text: str,
        field: SemanticField,
    ) -> Optional[SemanticRole]:
        """
        Conservative lexical repair for max_power_w only.

        We repair only when the current user text contains an explicit cue.
        """

        if (
            field
            is not SemanticField.MAX_POWER_W
        ):
            return None

        text = (
            " "
            + LLMFallbackExtractor
            ._normalize_spaces(
                user_text
            )
            .casefold()
            + " "
        )

        maximum_patterns = (
            r"\bcapped\b",
            r"\bcap\b",
            r"\bmaximum\b",
            r"\bmax\b",
            r"\bceiling\b",
            r"\blimit\b",
            r"\bmust not exceed\b",
            r"\bno more than\b",
            r"\bne doit pas dépasser\b",
            r"\bplafond\b",
            r"\bmaximale\b",
        )

        current_patterns = (
            r"\bcurrent\b",
            r"\bcurrently\b",
            r"\bobserved\b",
            r"\bmeasured\b",
            r"\bactuel\b",
            r"\bactuellement\b",
            r"\bmesuré\b",
            r"\bmesurée\b",
        )

        expected_patterns = (
            r"\bexpected\b",
            r"\bprojected\b",
            r"\bnominal\b",
            r"\bshould consume\b",
            r"\bexpected consumption\b",
            r"\bdevrait consommer\b",
            r"\bconsommation prévue\b",
            r"\bconsommation attendue\b",
            r"\brégime nominal\b",
        )

        if any(
            re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
            for pattern
            in maximum_patterns
        ):
            return (
                SemanticRole.MAXIMUM_LIMIT
            )

        if any(
            re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
            for pattern
            in current_patterns
        ):
            return (
                SemanticRole.CURRENT_VALUE
            )

        if any(
            re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
            for pattern
            in expected_patterns
        ):
            return (
                SemanticRole.EXPECTED_VALUE
            )

        return None

    # ================================================================
    # LOGGING
    # ================================================================

    def _record(
        self,
        **data: Any,
    ) -> None:
        self.call_log.append(
            data
        )

    # ================================================================
    # MAIN V2 API
    # ================================================================

    def resolve_quantity(
        self,
        user_text: str,
        quantity: Quantity,
        previous_question: Optional[str] = None,
    ) -> Optional[SemanticLink]:
        """
        Resolve one already-detected Quantity into FIELD + ROLE.

        Returns:
            SemanticLink if the LLM proposes a safe compatible mapping.

            None if disabled, uncertain, invalid, hallucinated,
            incompatible, explicitly ambiguous or if the LLM fails.
        """

        # ------------------------------------------------------------
        # 1. Fallback disabled
        # ------------------------------------------------------------

        if not self.enabled:

            self._record(
                quantity_id=quantity.id,
                status="disabled",
            )

            return None

        # ------------------------------------------------------------
        # 2. Explicit ambiguity deterministic guard
        # ------------------------------------------------------------

        if self._has_explicit_ambiguity(
            user_text
        ):

            self._record(
                quantity_id=quantity.id,
                status=(
                    "explicit_ambiguity_abstention"
                ),
                reason=(
                    "The current user text explicitly states that "
                    "the semantic interpretation is ambiguous."
                ),
            )

            return None

        # ------------------------------------------------------------
        # 3. Ollama dependency
        # ------------------------------------------------------------

        try:
            from ollama import Client

        except Exception as exc:

            self._record(
                quantity_id=quantity.id,
                status="ollama_import_error",
                error=str(exc),
            )

            return None

        # ------------------------------------------------------------
        # 4. Compatibility space
        # ------------------------------------------------------------

        allowed_fields = (
            allowed_fields_for_dimension(
                quantity.dimension
            )
        )

        allowed_pairs = (
            self._build_allowed_pairs(
                quantity
            )
        )

        marked_text = (
            self._mark_target_quantity(
                user_text=user_text,
                quantity=quantity,
            )
        )

        # ------------------------------------------------------------
        # 5. Prompt metadata/context
        # ------------------------------------------------------------

        quantity_info = {
            "id": quantity.id,
            "raw": quantity.raw,
            "normalized":
                quantity.normalized,
            "value": quantity.value,
            "unit": quantity.unit,
            "dimension":
                quantity.dimension.value,
            "corrected":
                quantity.corrected,
        }

        previous_context = ""

        if previous_question:
            previous_context = (
                "\nPrevious clarification question "
                "(semantic context only; it MAY determine FIELD/ROLE "
                "for a short answer, but NEVER use it as evidence):\n"
                f"{previous_question}\n"
            )

        # ------------------------------------------------------------
        # 6. Successful Prompt V2 semantic guide — preserved
        # ------------------------------------------------------------

        semantic_guide = """
Semantic mapping guide
----------------------
Map natural language to the internal labels. The user does NOT need to
literally say the FIELD or ROLE name.

FIELD cues:
- client_count:
  clients, hosts, endpoints, compute nodes, machines mounting/accessing Lustre.
- total_file_count:
  files, objects, inodes, namespace object count.
- average_file_size_gb:
  average, typical, usual, representative, normal file/object size.
- max_file_size_gb:
  maximum, largest, capped, ceiling, must not exceed, no file above.
- target_read_gbps:
  reads, restore/restart, re-reading, storage delivering data to clients,
  data flowing FROM storage TO clients.
- target_write_gbps:
  writes, checkpoints written to Lustre, ingestion into Lustre,
  data flowing FROM clients/jobs TO storage.
- requested_usable_capacity_tib:
  usable/visible/requested/target storage capacity or namespace capacity.
- max_budget_usd:
  budget, spend, cost ceiling, must remain under.
- max_power_w:
  power envelope, power consumption, watts/kW/MW.
- annual_growth_percent:
  annual/year-over-year growth or expansion.
- read_write_ratio:
  read/write mix or a read/write percentage component.

ROLE cues:
- total_count:
  number/count of clients, hosts, files, objects or inodes.
- average_value:
  typical, usual, representative, average.
- maximum_limit:
  maximum, cap, ceiling, under, no more than, must not exceed.
- minimum_limit:
  minimum, at least, not below, must provide at least.
- target:
  target, aim, should sustain/reach/deliver, planned requirement.
- current_value:
  current/currently observed or available value.
- expected_value:
  expected/projected/nominal value, especially expected power consumption.
- growth_rate:
  annual/year-over-year growth.
- ratio_component:
  component of the read/write mix.
- unspecified:
  use only when FIELD is clear but ROLE truly cannot be determined.

Direction rules for throughput:
- FROM storage TO clients / restore / restart / read/re-read
  => target_read_gbps.
- INTO Lustre / checkpoint writes / ingestion / writes
  => target_write_gbps.

Power role rules:
- capped / maximum / ceiling / must not exceed
  => maximum_limit.
- current / observed
  => current_value.
- expected / projected / nominal / should consume approximately
  => expected_value.
- Do NOT use role "target" for max_power_w.

Context rule:
A previous clarification question MAY determine FIELD/ROLE for a short answer
such as "320", "18 kW", or "42 GB/s". It is semantic context only.
Evidence must still be copied ONLY from the current user text.

Abstention rule:
Choose __UNRESOLVED__ only when the text plus optional previous question
still leaves two or more allowed interpretations genuinely plausible.
Do NOT abstain merely because the user did not literally use an internal
FIELD or ROLE label.

Exact-label rule:
FIELD and ROLE strings must be copied exactly from the allowed FIELD/ROLE
pairs. Never invent labels such as "representative_value".
""".strip()

        # ------------------------------------------------------------
        # 7. User prompt
        # ------------------------------------------------------------

        user_prompt = (
            "You are the LAST semantic fallback of a guarded "
            "requirement extraction pipeline.\n\n"

            "A deterministic QuantityScanner has ALREADY extracted "
            "the target quantity.\n"
            "DO NOT extract another number.\n"
            "DO NOT modify its value.\n"
            "DO NOT convert its unit.\n"
            "DO NOT invent information.\n\n"

            "Your ONLY task is to decide which FIELD and semantic ROLE "
            "the marked target quantity refers to.\n\n"

            "The target quantity is marked with [Q] and [/Q].\n\n"

            f"Target quantity metadata:\n"
            f"{json.dumps(quantity_info, ensure_ascii=False)}\n\n"

            f"User text:\n"
            f"{marked_text}\n"

            f"{previous_context}\n"

            f"{semantic_guide}\n\n"

            "You MUST choose exactly one of the following allowed "
            "FIELD/ROLE pairs:\n"
            f"{json.dumps(allowed_pairs, ensure_ascii=False)}\n\n"

            "Safety rules:\n"
            "1. Choose ONLY one allowed FIELD/ROLE pair.\n"
            "2. FIELD and ROLE strings MUST be copied exactly from "
            "the allowed pairs.\n"
            "3. First apply the semantic mapping guide to natural-language "
            "cues.\n"
            "4. Choose __UNRESOLVED__ only when multiple allowed "
            "interpretations remain genuinely plausible.\n"
            "5. If the user explicitly says the meaning is unclear or "
            "ambiguous, choose __UNRESOLVED__.\n"
            "6. Evidence must be copied from the ORIGINAL USER TEXT only.\n"
            "7. Evidence must contain the target quantity itself.\n"
            "8. NEVER include [Q] or [/Q] markers in evidence.\n"
            "9. The previous clarification question may be used only as "
            "semantic context and never as evidence.\n"
            "10. Do not return a value or unit; they already exist.\n"
            "11. Return JSON only.\n\n"

            "Required JSON format:\n"
            "{\n"
            '  "field": "client_count",\n'
            '  "role": "total_count",\n'
            '  "evidence": "200 hosts",\n'
            '  "reason": "short explanation"\n'
            "}\n"
        )

        # ------------------------------------------------------------
        # 8. Actual LLM call
        # ------------------------------------------------------------

        self.call_count += 1

        print(
            "[LLM FALLBACK] "
            f"quantity={quantity.id} "
            f"dimension={quantity.dimension.value}"
        )

        client = Client(
            host=self.host
        )

        try:
            response = client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict semantic classification "
                            "fallback. You never extract or invent values. "
                            "You classify only the marked quantity using "
                            "the provided allowed FIELD/ROLE pairs. "
                            "Map ordinary natural-language cues to the "
                            "internal FIELD/ROLE labels using the supplied "
                            "semantic guide. "
                            "A previous clarification question may "
                            "disambiguate a short answer, but it may never "
                            "be used as evidence. "
                            "Abstain only when multiple allowed "
                            "interpretations remain genuinely plausible. "
                            "Return valid JSON only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                format="json",
                options={
                    "temperature": 0,
                    "num_predict": 250,
                    "num_ctx": 4096,
                },
                stream=False,
            )

            raw = (
                response["message"][
                    "content"
                ]
            )

            data = json.loads(
                raw
            )

        except Exception as exc:

            self._record(
                quantity_id=quantity.id,
                status="llm_error",
                error=str(exc),
            )

            return None

        # ------------------------------------------------------------
        # 9. Read FIELD / ROLE / evidence
        # ------------------------------------------------------------

        field_raw = str(
            data.get(
                "field",
                "",
            )
        ).strip()

        role_raw = str(
            data.get(
                "role",
                "",
            )
        ).strip()

        evidence = (
            self._clean_evidence(
                str(
                    data.get(
                        "evidence",
                        "",
                    )
                )
            )
        )

        reason = str(
            data.get(
                "reason",
                "",
            )
        ).strip()

        try:
            field = SemanticField(
                field_raw
            )

        except ValueError:

            self._record(
                quantity_id=quantity.id,
                status="invalid_field",
                raw_response=data,
            )

            return None

        # ------------------------------------------------------------
        # 10. Explicit LLM abstention
        # ------------------------------------------------------------

        if (
            field
            is SemanticField.UNRESOLVED
        ):

            self._record(
                quantity_id=quantity.id,
                status="llm_abstained",
                reason=reason,
                raw_response=data,
            )

            return None

        # ------------------------------------------------------------
        # 11. FIELD must be compatible with QuantityDimension
        # ------------------------------------------------------------

        if field not in allowed_fields:

            self._record(
                quantity_id=quantity.id,
                status="field_dimension_rejected",
                field=field.value,
                dimension=quantity.dimension.value,
                raw_response=data,
            )

            return None

        # ------------------------------------------------------------
        # 12. ROLE validation / targeted deterministic repair
        # ------------------------------------------------------------

        role_repaired = False
        repair_reason = None

        canonical_role = (
            self._canonical_role_for_field(
                field
            )
        )

        if canonical_role is not None:
            # Safe because these FIELDs imply one concrete role.
            role = canonical_role

            if (
                role_raw
                != canonical_role.value
            ):
                role_repaired = True
                repair_reason = (
                    "canonical_role_for_field"
                )

        else:
            # Multi-role fields remain LLM-controlled unless there is an
            # explicit, conservative lexical repair.
            try:
                role = SemanticRole(
                    role_raw
                )

            except ValueError:

                repaired_role = (
                    self
                    ._repair_power_role_from_text(
                        user_text=user_text,
                        field=field,
                    )
                )

                if repaired_role is None:
                    self._record(
                        quantity_id=quantity.id,
                        status="invalid_role",
                        raw_response=data,
                    )

                    return None

                role = repaired_role
                role_repaired = True
                repair_reason = (
                    "explicit_power_role_cue"
                )

            if not is_valid_field_role_pair(
                field,
                role,
            ):

                repaired_role = (
                    self
                    ._repair_power_role_from_text(
                        user_text=user_text,
                        field=field,
                    )
                )

                if repaired_role is None:
                    self._record(
                        quantity_id=quantity.id,
                        status=(
                            "invalid_field_role_pair"
                        ),
                        field=field.value,
                        role=role.value,
                        raw_response=data,
                    )

                    return None

                role = repaired_role
                role_repaired = True
                repair_reason = (
                    "explicit_power_role_cue"
                )

        # Defensive final check.
        if not is_valid_field_role_pair(
            field,
            role,
        ):

            self._record(
                quantity_id=quantity.id,
                status=(
                    "invalid_field_role_pair_after_repair"
                ),
                field=field.value,
                role=role.value,
                raw_response=data,
            )

            return None

        # ------------------------------------------------------------
        # 13. Evidence validation
        # ------------------------------------------------------------

        if not self._evidence_is_supported(
            evidence=evidence,
            user_text=user_text,
        ):

            self._record(
                quantity_id=quantity.id,
                status="unsupported_evidence",
                evidence=evidence,
                raw_response=data,
            )

            return None

        if not self._evidence_targets_quantity(
            evidence=evidence,
            quantity=quantity,
        ):

            self._record(
                quantity_id=quantity.id,
                status="wrong_quantity_evidence",
                evidence=evidence,
                raw_response=data,
            )

            return None

        # ------------------------------------------------------------
        # 14. Convert SemanticField -> official ParamName
        # ------------------------------------------------------------

        try:
            param_name = ParamName(
                field.value
            )

        except ValueError:

            self._record(
                quantity_id=quantity.id,
                status="unknown_param_name",
                field=field.value,
                raw_response=data,
            )

            return None

        # ------------------------------------------------------------
        # 15. Safe result
        # ------------------------------------------------------------

        link = SemanticLink(
            quantity_id=quantity.id,
            field=param_name,
            role=role,
            evidence=evidence,
            resolver="llm_fallback",
        )

        self._record(
            quantity_id=quantity.id,
            status="resolved",
            field=param_name.value,
            role=role.value,
            evidence=evidence,
            reason=reason,
            original_role=role_raw,
            role_repaired=role_repaired,
            repair_reason=repair_reason,
        )

        return link

    # ================================================================
    # DEBUG / INFORMATION
    # ================================================================

    def info(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "host": self.host,
            "model": self.model,
            "call_count": self.call_count,
        }