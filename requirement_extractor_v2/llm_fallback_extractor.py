from __future__ import annotations

import json
import os
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

    Responsabilité unique
    ----------------------
    Le QuantityScanner a déjà détecté la quantité et sa valeur.

    Le LLM NE DOIT PAS :
    - extraire une nouvelle valeur ;
    - modifier la valeur détectée ;
    - convertir une unité ;
    - inventer un Requirement field ;
    - contourner les compatibility rules ;
    - prendre la décision métier finale.

    Le LLM DOIT uniquement proposer :

        Quantity
            ↓
        FIELD + ROLE

    lorsque :
        ExplicitPatternResolver -> unresolved
        SemanticLinker          -> unresolved / abstention

    Le résultat reste ensuite soumis au DeterministicVerifier.
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

        # Useful for tests / evaluation.
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
            text.strip().split()
        )

    @staticmethod
    def _clean_evidence(
        evidence: str,
    ) -> str:
        """
        Remove internal [Q] markers before validating evidence.

        The markers are only used to guide the semantic model/LLM.
        They do not belong to the original user text and must never
        appear in the final SemanticLink evidence.
        """

        return (
            evidence
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
        Evidence must really come from the user text.

        Prefer exact substring matching. A normalized-space fallback is
        kept only to tolerate harmless whitespace differences.
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

        return evidence_norm in text_norm

    def _evidence_targets_quantity(
        self,
        evidence: str,
        quantity: Quantity,
    ) -> bool:
        """
        Prevent the LLM from explaining another quantity appearing in
        the same message.

        The evidence must contain the target Quantity raw span.
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
        Mark only the target quantity.

        Example:

            Around 200 hosts will connect.

        becomes:

            Around [Q]200[/Q] hosts will connect.
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

        # Defensive fallback if offsets were altered upstream.
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
        Build exactly the FIELD/ROLE pairs allowed by the same
        deterministic compatibility rules used by Semantic Linker.
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
    # LOGGING
    # ================================================================

    def _record(
        self,
        **data: Any,
    ) -> None:
        self.call_log.append(data)

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
            SemanticLink
                if the LLM proposes a valid, supported and compatible
                mapping.

            None
                if disabled, uncertain, invalid, hallucinated,
                incompatible or if the LLM fails.

        Important:
            None means "still unresolved".

            It is NOT an error and must never force a business decision.
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
        # 2. Ollama dependency
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
        # 3. Compatibility space
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
        # 4. Prompt
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
                "(context only, never use it as evidence):\n"
                f"{previous_question}\n"
            )

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

            "You MUST choose exactly one of the following allowed "
            "FIELD/ROLE pairs:\n"
            f"{json.dumps(allowed_pairs, ensure_ascii=False)}\n\n"

            "Safety rules:\n"
            "1. Choose ONLY one allowed FIELD/ROLE pair.\n"
            "2. If the meaning is not reliably supported, choose "
            "__UNRESOLVED__ + unspecified.\n"
            "3. Evidence must be copied from the ORIGINAL USER TEXT only.\n"
            "4. Evidence must contain the target quantity itself.\n"
            "5. NEVER include [Q] or [/Q] markers in evidence.\n"
            "6. The [Q] markers are internal hints only and are not part "
            "of the original user text.\n"
            "7. Do not use domain knowledge to infer a missing fact.\n"
            "8. Do not return a value or unit; they already exist.\n"
            "9. Return JSON only.\n\n"

            "Required JSON format:\n"
            "{\n"
            '  "field": "client_count",\n'
            '  "role": "total_count",\n'
            '  "evidence": "200 hosts",\n'
            '  "reason": "short explanation"\n'
            "}\n"
        )

        # ------------------------------------------------------------
        # 5. Actual LLM call
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
                            "When uncertain, return "
                            "__UNRESOLVED__ + unspecified. "
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
                    "num_ctx": 2048,
                },
                stream=False,
            )

            raw = (
                response["message"][
                    "content"
                ]
            )

            data = json.loads(raw)

        except Exception as exc:

            self._record(
                quantity_id=quantity.id,
                status="llm_error",
                error=str(exc),
            )

            return None

        # ------------------------------------------------------------
        # 6. Read FIELD
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


        evidence = self._clean_evidence(
            str(
                data.get(
                    "evidence",
                    "",
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
        # 7. Explicit LLM abstention
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
        # 8. FIELD must be compatible with QuantityDimension
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
        # 9. ROLE validation
        # ------------------------------------------------------------

        try:
            role = SemanticRole(
                role_raw
            )

        except ValueError:

            self._record(
                quantity_id=quantity.id,
                status="invalid_role",
                raw_response=data,
            )

            return None

        if not is_valid_field_role_pair(
            field,
            role,
        ):

            self._record(
                quantity_id=quantity.id,
                status="invalid_field_role_pair",
                field=field.value,
                role=role.value,
                raw_response=data,
            )

            return None

        # ------------------------------------------------------------
        # 10. Evidence validation
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
        # 11. Convert SemanticField -> official ParamName
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
        # 12. Safe result
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
