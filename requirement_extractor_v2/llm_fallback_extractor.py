from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from .models import (
    ParamName,
    Quantity,
    QuantityDimension,
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
    Guarded selective LLM fallback for Requirement Extractor V2.

    Step 1.5 design
    ----------------
    The QuantityScanner has already detected the quantity. The LLM is NEVER
    responsible for extracting, correcting, replacing, converting or inventing
    the numerical value.

    The LLM has only two possible actions:

        RESOLVE -> classify the already-detected quantity as FIELD + ROLE
        ABSTAIN -> return no semantic link

    The deterministic pipeline remains authoritative:
    - the LLM cannot change Quantity.value;
    - the LLM cannot change Quantity.unit;
    - FIELD must be compatible with Quantity.dimension;
    - ROLE must be compatible with FIELD;
    - evidence must be an exact substring of the current user message;
    - evidence must contain the already-detected quantity;
    - explicit ambiguity forces abstention;
    - malformed or unsupported model output forces abstention.

    The prompt is intentionally robust to noisy language. A spelling error in
    the semantic noun (for example "endponts" for "endpoints") may still be
    interpreted when the surrounding sentence strongly identifies one field.
    In contrast, vague unitless statements such as "The limit is 20" must
    remain unresolved.
    """

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

    # Once these fields are correct, the concrete business role is fixed by
    # the current Requirement Contract. The LLM therefore does not need to
    # invent or debate a role for them.
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
        self.call_log: List[Dict[str, Any]] = []

    # ==================================================================
    # Generic helpers
    # ==================================================================

    @staticmethod
    def _normalize_spaces(
        text: str,
    ) -> str:
        return " ".join(
            (text or "")
            .strip()
            .split()
        )

    @staticmethod
    def _clean_evidence(
        evidence: str,
    ) -> str:
        return (
            (evidence or "")
            .replace(
                "[Q]",
                "",
            )
            .replace(
                "[/Q]",
                "",
            )
            .strip()
        )

    @staticmethod
    def _record_payload_copy(
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Keep logs JSON-serializable and detached from mutable model output.
        """
        try:
            return json.loads(
                json.dumps(
                    data,
                    ensure_ascii=False,
                )
            )
        except Exception:
            return {
                "raw": str(data),
            }

    def _record(
        self,
        **data: Any,
    ) -> None:
        self.call_log.append(
            data
        )

    # ==================================================================
    # Evidence safety
    # ==================================================================

    def _evidence_is_supported(
        self,
        evidence: str,
        user_text: str,
    ) -> bool:
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
            bool(evidence_norm)
            and
            evidence_norm
            in text_norm
        )

    def _evidence_targets_quantity(
        self,
        evidence: str,
        quantity: Quantity,
    ) -> bool:
        if not evidence:
            return False

        raw = (
            quantity.raw
            or ""
        ).strip()

        if not raw:
            return False

        if raw in evidence:
            return True

        return (
            self._normalize_spaces(
                raw
            )
            in
            self._normalize_spaces(
                evidence
            )
        )

    # ==================================================================
    # Quantity marking
    # ==================================================================

    @staticmethod
    def _mark_target_quantity(
        user_text: str,
        quantity: Quantity,
    ) -> str:
        start = int(
            quantity.start
        )
        end = int(
            quantity.end
        )

        if (
            0 <= start < end <= len(
                user_text
            )
            and
            user_text[
                start:end
            ]
            == quantity.raw
        ):
            return (
                user_text[:start]
                + "[Q]"
                + user_text[
                    start:end
                ]
                + "[/Q]"
                + user_text[end:]
            )

        index = user_text.find(
            quantity.raw
        )

        if index >= 0:
            end_index = (
                index
                + len(
                    quantity.raw
                )
            )

            return (
                user_text[:index]
                + "[Q]"
                + user_text[
                    index:end_index
                ]
                + "[/Q]"
                + user_text[
                    end_index:
                ]
            )

        return user_text

    # ==================================================================
    # Deterministic ambiguity guard
    # ==================================================================

    @classmethod
    def _has_explicit_ambiguity(
        cls,
        text: str,
    ) -> bool:
        normalized = (
            cls
            ._normalize_spaces(
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

    # ==================================================================
    # Compatibility helpers
    # ==================================================================

    @classmethod
    def _canonical_role_for_field(
        cls,
        field: SemanticField,
    ) -> Optional[
        SemanticRole
    ]:
        return (
            cls
            ._CANONICAL_ROLE_BY_FIELD
            .get(
                field
            )
        )

    @classmethod
    def _build_prompt_pairs(
        cls,
        quantity: Quantity,
    ) -> List[
        Dict[str, str]
    ]:
        """
        Build a compact candidate space for the LLM.

        Important differences from the old prompt:
        - __UNRESOLVED__ is NOT mixed with normal fields. Abstention is a
          separate decision.
        - UNSPECIFIED is omitted when a concrete business role is available.
          If the role is truly unclear, the model should ABSTAIN instead of
          returning a vague role.
        - canonical single-role fields expose only their canonical role.

        This reduces prompt noise, especially for QuantityDimension.UNKNOWN,
        where every quantitative field is otherwise technically available.
        """
        pairs: List[
            Dict[str, str]
        ] = []

        for field in (
            allowed_fields_for_dimension(
                quantity.dimension
            )
        ):
            if (
                field
                is SemanticField.UNRESOLVED
            ):
                continue

            canonical = (
                cls
                ._canonical_role_for_field(
                    field
                )
            )

            if canonical is not None:
                pairs.append(
                    {
                        "field":
                            field.value,
                        "role":
                            canonical.value,
                    }
                )
                continue

            concrete_roles = [
                role
                for role
                in allowed_roles_for_field(
                    field
                )
                if (
                    role
                    is not
                    SemanticRole.UNSPECIFIED
                )
            ]

            for role in concrete_roles:
                pairs.append(
                    {
                        "field":
                            field.value,
                        "role":
                            role.value,
                    }
                )

        return pairs

    @staticmethod
    def _repair_power_role_from_text(
        user_text: str,
        field: SemanticField,
    ) -> Optional[
        SemanticRole
    ]:
        if (
            field
            is not
            SemanticField.MAX_POWER_W
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
            r"\bmaximum\b",
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
                SemanticRole
                .MAXIMUM_LIMIT
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
                SemanticRole
                .CURRENT_VALUE
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
                SemanticRole
                .EXPECTED_VALUE
            )

        return None

    # ==================================================================
    # Prompt construction
    # ==================================================================

    @staticmethod
    def _system_prompt() -> str:
        return """
You are the final SEMANTIC CLASSIFIER in a guarded Lustre requirement
extraction pipeline.

A deterministic scanner has ALREADY detected the target quantity.
You are NOT a quantity extractor.

You have exactly two actions:
1. RESOLVE: identify the FIELD and ROLE of the marked quantity.
2. ABSTAIN: use when the message does not provide enough semantic evidence.

STRICT SAFETY CONTRACT
----------------------
- NEVER invent a number.
- NEVER extract a different number.
- NEVER change the detected number.
- NEVER infer, repair, add, convert or replace a unit.
- NEVER output a numerical value or unit.
- NEVER use world knowledge to create a missing requirement.
- Classify ONLY the quantity marked [Q]...[/Q].
- Evidence must be copied exactly from the CURRENT user text.
- Evidence must contain the target quantity.
- A previous question may help understand a short answer, but it is never
  evidence.

ROBUST LANGUAGE RULE
--------------------
The user may make spelling mistakes in ordinary semantic words.
A small, obvious typo in a semantic noun may still identify a field when the
whole sentence strongly supports ONE interpretation.

Examples:
- "endponts" can clearly be a misspelling of "endpoints".
- "clents" can clearly be a misspelling of "clients".
- "objcts" can clearly be a misspelling of "objects".

Do NOT correct the user's text in evidence. Quote the original typo exactly.

CRITICAL DISTINCTION
--------------------
UNKNOWN quantity dimension does NOT automatically mean ABSTAIN.
Many legitimate counts have no unit.

Example:
"About [Q]120[/Q] endponts will connect."
The noun "endponts" strongly identifies a client/end-point count, so resolve it.

But a bare quantity with generic wording has no field evidence:
"The limit is [Q]20[/Q]."
"Set it to [Q]50[/Q]."
"Around [Q]200[/Q] should be enough."
These MUST be ABSTAIN.

If two or more fields remain genuinely plausible, ABSTAIN.

Return valid JSON only.
""".strip()

    @staticmethod
    def _semantic_guide() -> str:
        return """
FIELD GUIDE
-----------
requested_usable_capacity_tib
  usable/requested/target storage capacity or storage footprint.

client_count
  clients, hosts, endpoints, compute nodes, machines or nodes connecting to,
  mounting or accessing Lustre.

average_file_size_gb
  average, typical, representative, usual file/object size.

max_file_size_gb
  maximum/largest file size, file-size cap or ceiling.

total_file_count
  number of files, objects, inodes or namespace objects.

read_write_ratio
  read/write mix or one explicit component of that mix.

target_read_gbps
  read/re-read/restore/restart bandwidth; data delivered FROM storage TO
  clients/jobs.

target_write_gbps
  write/checkpoint/ingest bandwidth; data sent FROM clients/jobs INTO storage.

max_budget_usd
  budget, spending/cost ceiling or maximum allowed cost.

max_power_w
  power envelope/consumption/limit.

annual_growth_percent
  annual/year-over-year storage growth.

ROLE GUIDE
----------
total_count
  number of clients/hosts/endpoints/files/objects.

average_value
  average/typical/representative value.

maximum_limit
  maximum/cap/ceiling/must-not-exceed.

minimum_limit
  minimum/at-least/must-provide-at-least.

target
  target/planned/sustain/should-deliver requirement.

current_value
  current/observed/measured value.

expected_value
  expected/projected/nominal value.

growth_rate
  annual/year-over-year growth.

ratio_component
  one component of the read/write mix.
""".strip()

    @staticmethod
    def _few_shot_messages() -> List[
        Dict[str, str]
    ]:
        """
        Deliberately tiny set of contrasted examples.

        The first example is the exact failure mode observed in Step 1.4:
        a misspelled semantic noun with a safe unitless count.

        The second and third examples teach abstention on vague unitless
        numbers, preventing the model from inventing a field.
        """
        return [
            {
                "role": "user",
                "content": (
                    "EXAMPLE INPUT\n"
                    "Target metadata: "
                    '{"raw":"120","value":120,"unit":null,'
                    '"dimension":"unknown"}\n'
                    "User text: Around [Q]120[/Q] endponts will connect.\n"
                    "Allowed candidates include "
                    '{"field":"client_count","role":"total_count"}.\n'
                    "Classify the marked quantity."
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "decision":
                            "RESOLVE",
                        "field":
                            "client_count",
                        "role":
                            "total_count",
                        "evidence":
                            "120 endponts",
                        "reason":
                            (
                                "The misspelled noun "
                                "'endponts' clearly denotes "
                                "endpoints/clients in context."
                            ),
                    },
                    ensure_ascii=False,
                ),
            },
            {
                "role": "user",
                "content": (
                    "EXAMPLE INPUT\n"
                    "Target metadata: "
                    '{"raw":"20","value":20,"unit":null,'
                    '"dimension":"unknown"}\n'
                    "User text: The limit is [Q]20[/Q].\n"
                    "Classify the marked quantity."
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "decision":
                            "ABSTAIN",
                        "field":
                            None,
                        "role":
                            None,
                        "evidence":
                            "",
                        "reason":
                            (
                                "The message contains no semantic "
                                "noun identifying what 20 represents."
                            ),
                    },
                    ensure_ascii=False,
                ),
            },
            {
                "role": "user",
                "content": (
                    "EXAMPLE INPUT\n"
                    "Target metadata: "
                    '{"raw":"200","value":200,"unit":null,'
                    '"dimension":"unknown"}\n'
                    "User text: Around [Q]200[/Q] should be enough.\n"
                    "Classify the marked quantity."
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "decision":
                            "ABSTAIN",
                        "field":
                            None,
                        "role":
                            None,
                        "evidence":
                            "",
                        "reason":
                            (
                                "The quantity is present but its "
                                "business meaning is unspecified."
                            ),
                    },
                    ensure_ascii=False,
                ),
            },
        ]

    def _build_user_prompt(
        self,
        user_text: str,
        quantity: Quantity,
        previous_question: Optional[
            str
        ],
        prompt_pairs: List[
            Dict[str, str]
        ],
    ) -> str:
        marked_text = (
            self
            ._mark_target_quantity(
                user_text=user_text,
                quantity=quantity,
            )
        )

        quantity_info = {
            "id":
                quantity.id,
            "raw":
                quantity.raw,
            "value":
                quantity.value,
            "unit":
                quantity.unit,
            "dimension":
                quantity.dimension.value,
            "corrected":
                bool(
                    quantity.corrected
                ),
        }

        previous_context = (
            previous_question
            if previous_question
            else None
        )

        return (
            "REAL INPUT\n"
            "==========\n"
            "Already-detected target metadata:\n"
            f"{json.dumps(quantity_info, ensure_ascii=False)}\n\n"

            "Current user text:\n"
            f"{marked_text}\n\n"

            "Previous clarification question "
            "(semantic context only, never evidence):\n"
            f"{json.dumps(previous_context, ensure_ascii=False)}\n\n"

            f"{self._semantic_guide()}\n\n"

            "ALLOWED RESOLUTION CANDIDATES\n"
            "-----------------------------\n"
            f"{json.dumps(prompt_pairs, ensure_ascii=False)}\n\n"

            "DECISION PROCEDURE\n"
            "------------------\n"
            "1. Ignore the numerical magnitude when deciding FIELD. "
            "Use the surrounding words and sentence relation.\n"
            "2. Check whether one candidate is clearly supported by a "
            "semantic noun/phrase, including an obvious spelling error.\n"
            "3. If exactly one candidate is strongly supported, RESOLVE.\n"
            "4. If no field-specific semantic cue exists, ABSTAIN.\n"
            "5. If more than one field remains plausible, ABSTAIN.\n"
            "6. Never create or infer a missing value or unit.\n"
            "7. For RESOLVE, evidence must be the shortest exact substring "
            "of CURRENT user text that contains both the target quantity "
            "and the semantic cue.\n"
            "8. Do not copy [Q] or [/Q] into evidence.\n\n"

            "OUTPUT JSON\n"
            "-----------\n"
            "For RESOLVE:\n"
            "{\n"
            '  "decision": "RESOLVE",\n'
            '  "field": "<exact allowed field>",\n'
            '  "role": "<exact allowed role>",\n'
            '  "evidence": "<exact substring of current user text>",\n'
            '  "reason": "<brief semantic reason>"\n'
            "}\n\n"

            "For ABSTAIN:\n"
            "{\n"
            '  "decision": "ABSTAIN",\n'
            '  "field": null,\n'
            '  "role": null,\n'
            '  "evidence": "",\n'
            '  "reason": "<brief reason why the field is not unique>"\n'
            "}\n\n"

            "IMPORTANT: do not output keys named value, normalized_value, "
            "unit, inferred_unit, corrected_value or confidence.\n"
        )

    # ==================================================================
    # Ollama response helpers
    # ==================================================================

    @staticmethod
    def _response_content(
        response: Any,
    ) -> str:
        """
        Support both dict-like and object-like Ollama Python clients.
        """
        if isinstance(
            response,
            dict,
        ):
            message = response.get(
                "message",
                {},
            )

            if isinstance(
                message,
                dict,
            ):
                return str(
                    message.get(
                        "content",
                        "",
                    )
                )

        message = getattr(
            response,
            "message",
            None,
        )

        if message is not None:
            content = getattr(
                message,
                "content",
                None,
            )

            if content is not None:
                return str(
                    content
                )

        return ""

    @staticmethod
    def _parse_json_object(
        raw: str,
    ) -> Optional[
        Dict[str, Any]
    ]:
        """
        Ollama is requested to return JSON. This helper remains conservative:
        it first parses the full response and, only if necessary, tries one
        outer {...} object. It never executes model text.
        """
        if not raw:
            return None

        try:
            data = json.loads(
                raw
            )

            if isinstance(
                data,
                dict,
            ):
                return data

        except json.JSONDecodeError:
            pass

        match = re.search(
            r"\{.*\}",
            raw,
            flags=re.DOTALL,
        )

        if match is None:
            return None

        try:
            data = json.loads(
                match.group(0)
            )

        except json.JSONDecodeError:
            return None

        return (
            data
            if isinstance(
                data,
                dict,
            )
            else None
        )

    @staticmethod
    def _normalize_decision(
        data: Dict[str, Any],
    ) -> str:
        decision = str(
            data.get(
                "decision",
                "",
            )
        ).strip().upper()

        if decision in {
            "RESOLVE",
            "ABSTAIN",
        }:
            return decision

        # Backward-compatible interpretation for older model behavior.
        field_raw = data.get(
            "field"
        )

        if (
            field_raw is None
            or str(
                field_raw
            ).strip()
            in {
                "",
                "__UNRESOLVED__",
                "UNRESOLVED",
                "null",
                "None",
            }
        ):
            return "ABSTAIN"

        return "RESOLVE"

    # ==================================================================
    # Main API
    # ==================================================================

    def resolve_quantity(
        self,
        user_text: str,
        quantity: Quantity,
        previous_question: Optional[
            str
        ] = None,
    ) -> Optional[
        SemanticLink
    ]:
        """
        Resolve one already-detected Quantity into FIELD + ROLE.

        Returns:
            SemanticLink:
                only when the LLM proposes one safe compatible mapping.

            None:
                when disabled, ambiguous, unresolved, malformed,
                hallucinated, incompatible or unsupported.
        """

        # --------------------------------------------------------------
        # 1. Disabled
        # --------------------------------------------------------------
        if not self.enabled:
            self._record(
                quantity_id=
                    quantity.id,
                status=
                    "disabled",
            )
            return None

        # --------------------------------------------------------------
        # 2. Explicit ambiguity guard
        # --------------------------------------------------------------
        if self._has_explicit_ambiguity(
            user_text
        ):
            self._record(
                quantity_id=
                    quantity.id,
                status=
                    "explicit_ambiguity_abstention",
                reason=(
                    "Current user text explicitly states semantic ambiguity."
                ),
            )
            return None

        # --------------------------------------------------------------
        # 3. Ollama dependency
        # --------------------------------------------------------------
        try:
            from ollama import Client

        except Exception as exc:
            self._record(
                quantity_id=
                    quantity.id,
                status=
                    "ollama_import_error",
                error=
                    str(exc),
            )
            return None

        # --------------------------------------------------------------
        # 4. Candidate space
        # --------------------------------------------------------------
        allowed_fields = tuple(
            field
            for field
            in allowed_fields_for_dimension(
                quantity.dimension
            )
            if (
                field
                is not
                SemanticField.UNRESOLVED
            )
        )

        prompt_pairs = (
            self
            ._build_prompt_pairs(
                quantity
            )
        )

        if not prompt_pairs:
            self._record(
                quantity_id=
                    quantity.id,
                status=
                    "no_allowed_pairs",
                dimension=
                    quantity.dimension.value,
            )
            return None

        # --------------------------------------------------------------
        # 5. Prompt
        # --------------------------------------------------------------
        user_prompt = (
            self
            ._build_user_prompt(
                user_text=
                    user_text,
                quantity=
                    quantity,
                previous_question=
                    previous_question,
                prompt_pairs=
                    prompt_pairs,
            )
        )

        messages: List[
            Dict[str, str]
        ] = [
            {
                "role":
                    "system",
                "content":
                    self
                    ._system_prompt(),
            },
            *self
            ._few_shot_messages(),
            {
                "role":
                    "user",
                "content":
                    user_prompt,
            },
        ]

        # --------------------------------------------------------------
        # 6. LLM call
        # --------------------------------------------------------------
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
                messages=messages,
                format="json",
                options={
                    "temperature":
                        0,
                    "num_predict":
                        180,
                    "num_ctx":
                        4096,
                },
                stream=False,
            )

            raw = (
                self
                ._response_content(
                    response
                )
            )

            data = (
                self
                ._parse_json_object(
                    raw
                )
            )

            if data is None:
                self._record(
                    quantity_id=
                        quantity.id,
                    status=
                        "invalid_json",
                    raw_response=
                        raw,
                )
                return None

        except Exception as exc:
            self._record(
                quantity_id=
                    quantity.id,
                status=
                    "llm_error",
                error=
                    str(exc),
            )
            return None

        raw_response = (
            self
            ._record_payload_copy(
                data
            )
        )

        # --------------------------------------------------------------
        # 7. Decision
        # --------------------------------------------------------------
        decision = (
            self
            ._normalize_decision(
                data
            )
        )

        reason = str(
            data.get(
                "reason",
                "",
            )
        ).strip()

        if decision == "ABSTAIN":
            self._record(
                quantity_id=
                    quantity.id,
                status=
                    "llm_abstained",
                reason=
                    reason,
                raw_response=
                    raw_response,
            )
            return None

        # --------------------------------------------------------------
        # 8. FIELD
        # --------------------------------------------------------------
        field_raw = str(
            data.get(
                "field",
                "",
            )
        ).strip()

        try:
            field = SemanticField(
                field_raw
            )

        except ValueError:
            self._record(
                quantity_id=
                    quantity.id,
                status=
                    "invalid_field",
                raw_response=
                    raw_response,
            )
            return None

        if (
            field
            is SemanticField.UNRESOLVED
        ):
            self._record(
                quantity_id=
                    quantity.id,
                status=
                    "llm_abstained",
                reason=
                    reason,
                raw_response=
                    raw_response,
            )
            return None

        if field not in allowed_fields:
            self._record(
                quantity_id=
                    quantity.id,
                status=
                    "field_dimension_rejected",
                field=
                    field.value,
                dimension=
                    quantity.dimension.value,
                raw_response=
                    raw_response,
            )
            return None

        # --------------------------------------------------------------
        # 9. ROLE
        # --------------------------------------------------------------
        role_raw = str(
            data.get(
                "role",
                "",
            )
        ).strip()

        role_repaired = False
        repair_reason = None

        canonical_role = (
            self
            ._canonical_role_for_field(
                field
            )
        )

        if canonical_role is not None:
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
            try:
                role = SemanticRole(
                    role_raw
                )

            except ValueError:
                repaired = (
                    self
                    ._repair_power_role_from_text(
                        user_text=
                            user_text,
                        field=
                            field,
                    )
                )

                if repaired is None:
                    self._record(
                        quantity_id=
                            quantity.id,
                        status=
                            "invalid_role",
                        raw_response=
                            raw_response,
                    )
                    return None

                role = repaired
                role_repaired = True
                repair_reason = (
                    "explicit_power_role_cue"
                )

            if not is_valid_field_role_pair(
                field,
                role,
            ):
                repaired = (
                    self
                    ._repair_power_role_from_text(
                        user_text=
                            user_text,
                        field=
                            field,
                    )
                )

                if repaired is None:
                    self._record(
                        quantity_id=
                            quantity.id,
                        status=
                            "invalid_field_role_pair",
                        field=
                            field.value,
                        role=
                            role.value,
                        raw_response=
                            raw_response,
                    )
                    return None

                role = repaired
                role_repaired = True
                repair_reason = (
                    "explicit_power_role_cue"
                )

        if not is_valid_field_role_pair(
            field,
            role,
        ):
            self._record(
                quantity_id=
                    quantity.id,
                status=
                    "invalid_field_role_pair_after_repair",
                field=
                    field.value,
                role=
                    role.value,
                raw_response=
                    raw_response,
            )
            return None

        # --------------------------------------------------------------
        # 10. Evidence
        # --------------------------------------------------------------
        evidence = (
            self
            ._clean_evidence(
                str(
                    data.get(
                        "evidence",
                        "",
                    )
                )
            )
        )

        if not self._evidence_is_supported(
            evidence=
                evidence,
            user_text=
                user_text,
        ):
            self._record(
                quantity_id=
                    quantity.id,
                status=
                    "unsupported_evidence",
                evidence=
                    evidence,
                raw_response=
                    raw_response,
            )
            return None

        if not self._evidence_targets_quantity(
            evidence=
                evidence,
            quantity=
                quantity,
        ):
            self._record(
                quantity_id=
                    quantity.id,
                status=
                    "wrong_quantity_evidence",
                evidence=
                    evidence,
                raw_response=
                    raw_response,
            )
            return None

        # --------------------------------------------------------------
        # 11. Convert to official ParamName
        # --------------------------------------------------------------
        try:
            param_name = ParamName(
                field.value
            )

        except ValueError:
            self._record(
                quantity_id=
                    quantity.id,
                status=
                    "unknown_param_name",
                field=
                    field.value,
                raw_response=
                    raw_response,
            )
            return None

        # --------------------------------------------------------------
        # 12. Safe semantic link
        # --------------------------------------------------------------
        link = SemanticLink(
            quantity_id=
                quantity.id,
            field=
                param_name,
            role=
                role,
            evidence=
                evidence,
            resolver=
                "llm_fallback",
        )

        ignored_generated_keys = [
            key
            for key
            in (
                "value",
                "normalized_value",
                "unit",
                "inferred_unit",
                "corrected_value",
                "confidence",
            )
            if key in data
        ]

        self._record(
            quantity_id=
                quantity.id,
            status=
                "resolved",
            field=
                param_name.value,
            role=
                role.value,
            evidence=
                evidence,
            reason=
                reason,
            original_role=
                role_raw,
            role_repaired=
                role_repaired,
            repair_reason=
                repair_reason,
            ignored_generated_keys=
                ignored_generated_keys,
            raw_response=
                raw_response,
        )

        return link

    # ==================================================================
    # Debug / information
    # ==================================================================

    def info(
        self,
    ) -> Dict[str, Any]:
        return {
            "enabled":
                self.enabled,
            "host":
                self.host,
            "model":
                self.model,
            "call_count":
                self.call_count,
            "strategy":
                "guarded_semantic_resolve_or_abstain_v3",
        }