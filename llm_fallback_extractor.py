from __future__ import annotations

import json
import os
import re
from typing import Any, List, Optional, Set

from models import CandidateSource, ExtractedCandidate, ExtractionResult, ParamName
from unit_normalizer import normalize_unit_value


class LLMFallbackExtractor:
    """
    Fallback LLM contrôlé.

    Règles de sécurité :
    - Le LLM n'est jamais l'extracteur principal.
    - Il est appelé seulement si ENABLE_LLM_FALLBACK=true.
    - Il reçoit uniquement les champs autorisés par HybridExtractor.
    - Il ne peut pas retourner un champ déjà extrait par les règles.
    - Chaque candidat LLM doit avoir une evidence présente dans le texte utilisateur.
    - Le StateGuard reste responsable de la validation finale.
    """

    def __init__(self):
        self.enabled = os.getenv("ENABLE_LLM_FALLBACK", "false").lower() == "true"
        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")

        self.call_count = 0
        self.call_log = []

    def _normalize_spaces(self, text: str) -> str:
        return " ".join(text.lower().split())

    def _evidence_is_supported(
        self,
        evidence: str,
        user_text: str,
    ) -> bool:
        """
        Vérifie que l'evidence existe réellement dans le texte utilisateur.

        Le LLM n'a pas le droit de fournir une evidence inventée.
        """

        if not evidence:
            return False

        evidence_norm = self._normalize_spaces(evidence)
        text_norm = self._normalize_spaces(user_text)

        return evidence_norm in text_norm

    def _safe_number(self, value: Any) -> Optional[float | int]:
        """
        Convertit une valeur LLM en nombre si possible.
        Supporte :
        - 2
        - 2.5
        - "2,5"
        - "2.5 millions"
        """

        if isinstance(value, bool):
            return None

        if isinstance(value, (int, float)):
            if float(value).is_integer():
                return int(value)
            return float(value)

        if isinstance(value, str):
            text = value.lower().replace(",", ".")

            match = re.search(r"-?\d+(?:\.\d+)?", text)

            if not match:
                return None

            number = float(match.group(0))

            if number.is_integer():
                return int(number)

            return number

        return None

    def _scale_multiplier(
        self,
        value: Any,
        unit: Any,
        evidence: str,
    ) -> int:
        """
        Détecte million / milliard pour total_file_count.
        """

        joined = f"{value} {unit} {evidence}".lower()

        if "milliard" in joined or "billion" in joined:
            return 1_000_000_000

        if "million" in joined:
            return 1_000_000

        return 1

    def _parse_ratio(self, value: Any, evidence: str) -> Optional[dict]:
        if isinstance(value, dict):
            read_percent = value.get("read_percent")
            write_percent = value.get("write_percent")

            if read_percent is not None and write_percent is not None:
                return {
                    "read_percent": int(read_percent),
                    "write_percent": int(write_percent),
                }

        text = f"{value} {evidence}".lower()

        match = re.search(
            r"(\d{1,3})\s*/\s*(\d{1,3})",
            text,
        )

        if match:
            return {
                "read_percent": int(match.group(1)),
                "write_percent": int(match.group(2)),
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

    def _parse_bool(self, value: Any, evidence: str) -> Optional[bool]:
        if isinstance(value, bool):
            return value

        text = f"{value} {evidence}".lower()

        false_markers = [
            "false",
            "no",
            "non",
            "sans ha",
            "pas de ha",
            "no ha",
            "without ha",
        ]

        true_markers = [
            "true",
            "yes",
            "oui",
            "ha required",
            "ha obligatoire",
            "high availability",
            "haute disponibilité",
            "haute disponibilite",
            "required",
            "obligatoire",
        ]

        if any(marker in text for marker in false_markers):
            return False

        if any(marker in text for marker in true_markers):
            return True

        return None

    def _parse_access_type(self, value: Any, evidence: str) -> Optional[str]:
        text = f"{value} {evidence}".lower()

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

        for marker, normalized_value in mapping.items():
            if marker in text:
                return normalized_value

        return None

    def _normalize_candidate_value(
        self,
        field: ParamName,
        value: Any,
        unit: Any,
        evidence: str,
    ):
        """
        Normalise la valeur retournée par le LLM selon le champ.
        """

        if field == ParamName.read_write_ratio:
            ratio = self._parse_ratio(value, evidence)

            if ratio is None:
                return None, None

            return ratio, "%"

        if field == ParamName.ha_required:
            ha = self._parse_bool(value, evidence)

            if ha is None:
                return None, None

            return ha, None

        if field == ParamName.access_type:
            access_type = self._parse_access_type(value, evidence)

            if access_type is None:
                return None, None

            return access_type, None

        number = self._safe_number(value)

        if number is None:
            return None, None

        if field == ParamName.total_file_count:
            multiplier = self._scale_multiplier(
                value=value,
                unit=unit,
                evidence=evidence,
            )

            return int(number * multiplier), None

        normalized_value, normalized_unit = normalize_unit_value(
            field=field,
            value=number,
            unit=unit,
        )

        return normalized_value, normalized_unit

    def extract(
        self,
        user_text: str,
        unresolved_fields: List[ParamName],
        turn_id: int,
        protected_fields: Optional[Set[ParamName]] = None,
    ) -> ExtractionResult:
        protected_fields = protected_fields or set()

        requested_fields = [
            field
            for field in unresolved_fields
            if field not in protected_fields
        ]

        if not self.enabled or not requested_fields:
            return ExtractionResult(
                candidates=[],
                warnings=[],
                unresolved_fields=[
                    field.value
                    for field in unresolved_fields
                ],
            )

        try:
            from ollama import Client
        except Exception:
            return ExtractionResult(
                candidates=[],
                warnings=["LLM fallback disabled: ollama package not installed."],
                unresolved_fields=[
                    field.value
                    for field in unresolved_fields
                ],
            )

        self.call_count += 1

        call_info = {
            "turn_id": turn_id,
            "requested_fields": [
                field.value
                for field in requested_fields
            ],
            "user_text": user_text,
        }

        self.call_log.append(call_info)

        print(
            f"[LLM FALLBACK] turn={turn_id} "
            f"fields={call_info['requested_fields']}"
        )

        client = Client(host=self.host)

        field_names = [
            field.value
            for field in requested_fields
        ]

        prompt = {
            "role": "user",
            "content": (
                "Extract ONLY the requested fields from the user text.\n"
                "Do not invent values.\n"
                "Do not infer values from general domain knowledge.\n"
                "Every candidate must include an evidence string copied exactly "
                "from the user text.\n"
                "If a requested field is not explicitly supported by the text, "
                "omit it.\n"
                "Return JSON only.\n\n"
                f"Requested fields: {field_names}\n"
                f"User text: {user_text}\n\n"
                "Return format:\n"
                "{"
                "\"candidates\":["
                "{\"field\":\"...\",\"value\":...,\"unit\":\"...\","
                "\"evidence\":\"...\",\"confidence\":0.0}"
                "],"
                "\"warnings\":[]"
                "}"
            ),
        }

        try:
            response = client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict information extraction fallback. "
                            "You only extract fields explicitly present in the text. "
                            "You return only valid JSON."
                        ),
                    },
                    prompt,
                ],
                format="json",
                options={
                    "temperature": 0,
                    "num_predict": 500,
                    "num_ctx": 2048,
                },
                stream=False,
            )

            raw = response["message"]["content"]
            data = json.loads(raw)

            candidates = []
            warnings = list(data.get("warnings", []))

            for item in data.get("candidates", []):
                try:
                    field = ParamName(item.get("field"))

                    if field not in requested_fields:
                        continue

                    if field in protected_fields:
                        continue

                    evidence = str(item.get("evidence", "")).strip()

                    if not self._evidence_is_supported(
                        evidence=evidence,
                        user_text=user_text,
                    ):
                        warnings.append(
                            f"Dropped LLM candidate for {field.value}: "
                            "evidence not supported by user text."
                        )
                        continue

                    value, unit = self._normalize_candidate_value(
                        field=field,
                        value=item.get("value"),
                        unit=item.get("unit"),
                        evidence=evidence,
                    )

                    if value is None:
                        warnings.append(
                            f"Dropped LLM candidate for {field.value}: "
                            "value could not be normalized."
                        )
                        continue

                    candidates.append(
                        ExtractedCandidate(
                            field=field,
                            value=value,
                            unit=unit,
                            evidence=evidence,
                            confidence=float(item.get("confidence", 0.65)),
                            source=CandidateSource.LLM_FALLBACK,
                            source_text=user_text,
                            turn_id=turn_id,
                        )
                    )

                except Exception as error:
                    warnings.append(
                        f"Dropped invalid LLM item: {error}"
                    )
                    continue

            return ExtractionResult(
                candidates=candidates,
                warnings=warnings,
                unresolved_fields=[
                    field.value
                    for field in unresolved_fields
                    if field not in {
                        candidate.field
                        for candidate in candidates
                    }
                ],
            )

        except Exception as error:
            return ExtractionResult(
                candidates=[],
                warnings=[f"LLM fallback failed: {error}"],
                unresolved_fields=[
                    field.value
                    for field in unresolved_fields
                ],
            )