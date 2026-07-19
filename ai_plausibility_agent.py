from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Dict, List, Optional

from ollama import Client

from models import FinalFieldValue, ParamName


class AIPlausibilityStatus:
    OK = "OK"
    WARNING = "WARNING"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    BLOCKING = "BLOCKING"


@dataclass(frozen=True)
class AIPlausibilityIssue:
    issue_type: str
    field: ParamName
    severity: str
    message: str
    question: str
    confidence: float = 0.0
    suggested_correction: Optional[Dict[str, Any]] = None
    evidence_fields: Dict[str, Any] = dataclass_field(default_factory=dict)


@dataclass(frozen=True)
class AIPlausibilityReport:
    status: str
    issues: List[AIPlausibilityIssue] = dataclass_field(default_factory=list)
    raw_response: str = ""

    @property
    def needs_clarification(self) -> bool:
        return self.status in {
            AIPlausibilityStatus.NEEDS_CLARIFICATION,
            AIPlausibilityStatus.BLOCKING,
        }

    @property
    def ok(self) -> bool:
        return self.status == AIPlausibilityStatus.OK


class AIPlausibilityAgent:
    """
    AI plausibility reviewer.

    Rôle :
    - lire uniquement le final_json déjà validé par StateGuard ;
    - détecter des combinaisons globalement suspectes ;
    - retourner des alertes structurées ;
    - ne jamais modifier directement les valeurs ;
    - ne jamais recommander une configuration hardware.
    """

    def __init__(self) -> None:
        self.enabled = (
            os.getenv("ENABLE_AI_PLAUSIBILITY_AGENT", "true").lower()
            == "true"
        )

        self.host = os.getenv(
            "OLLAMA_HOST",
            os.getenv("PLAUSIBILITY_AGENT_HOST", "http://localhost:11434"),
        )

        self.model = os.getenv(
            "PLAUSIBILITY_AGENT_MODEL",
            os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
        )

        try:
            self.temperature = float(
                os.getenv("PLAUSIBILITY_AGENT_TEMPERATURE", "0.0")
            )
        except Exception:
            self.temperature = 0.0

        self.debug = (
            os.getenv("PLAUSIBILITY_AGENT_DEBUG", "false").lower()
            == "true"
        )

        self.client = Client(host=self.host)

    def analyze(
        self,
        final_json: Dict[str, Optional[FinalFieldValue]],
    ) -> AIPlausibilityReport:
        if not self.enabled:
            if self.debug:
                print("[AI PLAUSIBILITY] disabled")

            return AIPlausibilityReport(
                status=AIPlausibilityStatus.OK,
                issues=[],
            )

        payload = self._final_json_to_plain_dict(final_json)
        derived_facts = self._build_derived_facts(payload)

        prompt = self._build_prompt(
            payload=payload,
            derived_facts=derived_facts,
        )

        try:
            if self.debug:
                print(
                    f"[AI PLAUSIBILITY] enabled={self.enabled} "
                    f"model={self.model} host={self.host}"
                )
                print("[AI PLAUSIBILITY PAYLOAD]")
                print(json.dumps(payload, indent=2, ensure_ascii=False))
                print("[AI PLAUSIBILITY DERIVED FACTS]")
                print(json.dumps(derived_facts, indent=2, ensure_ascii=False))

            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                format="json",
                options={
                    "temperature": self.temperature,
                },
            )

            raw_text = response["message"]["content"]

            if self.debug:
                print("[AI PLAUSIBILITY RAW]")
                print(raw_text)

        except Exception as exc:
            if self.debug:
                print(f"[AI PLAUSIBILITY ERROR] {exc}")

            return AIPlausibilityReport(
                status=AIPlausibilityStatus.WARNING,
                issues=[],
                raw_response=f"AI plausibility agent unavailable: {exc}",
            )

        parsed = self._parse_json_response(raw_text)

        if parsed is None:
            if self.debug:
                print("[AI PLAUSIBILITY PARSE ERROR]")
                print(raw_text)

            return AIPlausibilityReport(
                status=AIPlausibilityStatus.WARNING,
                issues=[],
                raw_response=raw_text,
            )

        report = self._sanitize_report(
        parsed=parsed,
        raw_text=raw_text,
        payload=payload,
        )
        if self.debug:
            print(
                f"[AI PLAUSIBILITY] status={report.status} "
                f"issues={len(report.issues)}"
            )

        return report

    # ============================================================
    # Prompt
    # ============================================================

    def _system_prompt(self) -> str:
        return """
You are an HPC Lustre plausibility reviewer agent.

You review already validated structured requirements.
Your job is NOT extraction.
Your job is NOT recommendation.
Your job is only to detect globally suspicious or inconsistent combinations.

Important domain interpretation:
- max_power_w is the maximum power budget for the whole Lustre storage system, not for one drive.
- Do not flag a power value only because the system is large.
- For large Lustre systems, tens of kilowatts can be plausible.
- For example, 50000 W = 50 kW should not be flagged only as a unit error.
- Only flag power if it is still extremely low or inconsistent with other values.
- max_budget_usd is the maximum budget for the whole Lustre storage system.
- requested_usable_capacity_tib is usable storage capacity.
- client_count is the number of Lustre clients / compute nodes.
- target_read_gbps and target_write_gbps are global throughput targets.

You must flag suspicious combinations such as:
- hundreds of TiB with only tens of watts;
- thousands of clients with only tens of watts;
- high throughput with extremely low power;
- very large storage with extremely low budget;
- max file size greater than total capacity;
- average file size multiplied by total file count much larger than requested capacity;
- very high annual growth that probably requires confirmation.

Allowed statuses:
- OK
- WARNING
- NEEDS_CLARIFICATION
- BLOCKING

Use NEEDS_CLARIFICATION when the user likely made a unit mistake or gave an unrealistic constraint.
Use BLOCKING when the values are logically inconsistent.
Use WARNING only when the system can continue but should warn the user.
Use OK only when no issue is found.

Allowed fields:
- requested_usable_capacity_tib
- client_count
- average_file_size_gb
- max_file_size_gb
- total_file_count
- read_write_ratio
- access_type
- target_read_gbps
- target_write_gbps
- ha_required
- max_budget_usd
- max_power_w
- annual_growth_percent

Each issue must contain:
- issue_type
- field
- severity
- message
- question
- confidence
- suggested_correction
- evidence_fields

Return valid JSON only.
Do not include markdown.
Do not include explanations outside JSON.
Use French for message and question.
""".strip()

    def _build_prompt(
        self,
        payload: Dict[str, Any],
        derived_facts: Dict[str, Any],
    ) -> str:
        return f"""
Review the following validated Lustre requirement state.

The values already passed local field validation.
Your task is to detect whether the combination is globally plausible.

You are given:
1. The original validated JSON.
2. Derived facts computed from the JSON.

Use the derived facts to help your reasoning.

Important:
- A value can be locally valid but globally implausible.
- If max_power_w is extremely small compared with capacity, clients, or throughput, ask clarification.
- If estimated dataset volume from average file size and file count is much larger than requested capacity, ask clarification.
- If the budget is extremely small compared with the requested scale, ask clarification.
- Do not correct values automatically.
- Do not recommend hardware.
- Return JSON only.

Validated requirement JSON:
{json.dumps(payload, indent=2, ensure_ascii=False)}

Derived facts:
{json.dumps(derived_facts, indent=2, ensure_ascii=False)}

Expected JSON structure:
{{
  "status": "OK | WARNING | NEEDS_CLARIFICATION | BLOCKING",
  "issues": [
    {{
      "issue_type": "string",
      "field": "one allowed field name",
      "severity": "warning | needs_clarification | blocking",
      "message": "short explanation in French",
      "question": "clarification question in French",
      "confidence": 0.0,
      "suggested_correction": {{
        "field": "field name or null",
        "value": "suggested value or null",
        "unit": "unit or null",
        "requires_user_confirmation": true
      }},
      "evidence_fields": {{
        "field_name": "value"
      }}
    }}
  ]
}}

Rules:
- If no issue is found, return exactly:
  {{
    "status": "OK",
    "issues": []
  }}
- If a value is suspicious, ask the user to confirm or correct it.
- If you suggest a correction, it must require user confirmation.
- Never silently correct values.
- Never add fields that are missing.
- Do not recommend hardware.
- Use French for message and question.

Carefully inspect:
- max_power_w versus requested_usable_capacity_tib;
- max_power_w versus client_count;
- max_power_w versus total_throughput_gbps;
- average_file_size_gb × total_file_count versus requested_usable_capacity_tib;
- max_budget_usd versus requested scale.
""".strip()

    # ============================================================
    # Derived facts
    # ============================================================

    def _as_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None

        if isinstance(value, bool):
            return None

        try:
            return float(value)
        except Exception:
            return None

    def _build_derived_facts(
        self,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        capacity_tib = self._as_float(
            payload.get("requested_usable_capacity_tib")
        )
        client_count = self._as_float(
            payload.get("client_count")
        )
        avg_file_gb = self._as_float(
            payload.get("average_file_size_gb")
        )
        max_file_gb = self._as_float(
            payload.get("max_file_size_gb")
        )
        total_files = self._as_float(
            payload.get("total_file_count")
        )
        read_gbps = self._as_float(
            payload.get("target_read_gbps")
        )
        write_gbps = self._as_float(
            payload.get("target_write_gbps")
        )
        budget_usd = self._as_float(
            payload.get("max_budget_usd")
        )
        power_w = self._as_float(
            payload.get("max_power_w")
        )

        total_throughput_gbps = None
        if read_gbps is not None or write_gbps is not None:
            total_throughput_gbps = float(read_gbps or 0.0) + float(write_gbps or 0.0)

        estimated_dataset_volume_tib = None
        if avg_file_gb is not None and total_files is not None:
            estimated_dataset_volume_tib = (avg_file_gb * total_files) / 1024.0

        capacity_gb = None
        if capacity_tib is not None:
            capacity_gb = capacity_tib * 1024.0

        power_per_tib = None
        if power_w is not None and capacity_tib not in (None, 0):
            power_per_tib = power_w / capacity_tib

        power_per_client = None
        if power_w is not None and client_count not in (None, 0):
            power_per_client = power_w / client_count

        budget_per_tib = None
        if budget_usd is not None and capacity_tib not in (None, 0):
            budget_per_tib = budget_usd / capacity_tib

        budget_per_client = None
        if budget_usd is not None and client_count not in (None, 0):
            budget_per_client = budget_usd / client_count

        max_file_to_capacity_ratio = None
        if max_file_gb is not None and capacity_gb not in (None, 0):
            max_file_to_capacity_ratio = max_file_gb / capacity_gb

        estimated_volume_to_capacity_ratio = None
        if (
            estimated_dataset_volume_tib is not None
            and capacity_tib not in (None, 0)
        ):
            estimated_volume_to_capacity_ratio = (
                estimated_dataset_volume_tib / capacity_tib
            )

        return {
            "total_throughput_gbps": total_throughput_gbps,
            "capacity_gb": capacity_gb,
            "estimated_dataset_volume_tib_from_average_size": estimated_dataset_volume_tib,
            "power_per_tib_w": power_per_tib,
            "power_per_client_w": power_per_client,
            "budget_per_tib_usd": budget_per_tib,
            "budget_per_client_usd": budget_per_client,
            "max_file_to_capacity_ratio": max_file_to_capacity_ratio,
            "estimated_volume_to_capacity_ratio": estimated_volume_to_capacity_ratio,
            "interpretation_notes": [
                "max_power_w is the power limit for the whole Lustre storage system.",
                "requested_usable_capacity_tib is usable capacity.",
                "estimated_dataset_volume_tib_from_average_size is computed as average_file_size_gb * total_file_count / 1024.",
                "power_per_tib_w and power_per_client_w are sanity indicators for plausibility review.",
            ],
        }

    # ============================================================
    # Parsing and sanitization
    # ============================================================

    def _final_json_to_plain_dict(
        self,
        final_json: Dict[str, Optional[FinalFieldValue]],
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        for key, item in final_json.items():
            if item is None:
                out[key] = None
            else:
                out[key] = item.value

        return out

    def _parse_json_response(self, raw_text: str) -> Optional[Dict[str, Any]]:
        text = (raw_text or "").strip()

        try:
            return json.loads(text)
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)

        if not match:
            return None

        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    def _sanitize_report(
        self,
        parsed: Dict[str, Any],
        raw_text: str,
        payload: Dict[str, Any],
    ) -> AIPlausibilityReport:
        """
        Nettoie la sortie du LLM.

        Objectif :
        - accepter seulement les champs connus ;
        - corriger les mauvais champs retournés par le LLM ;
        - supprimer les fausses alertes ;
        - garder une seule alerte à la fois.
        """

        allowed_statuses = {
            AIPlausibilityStatus.OK,
            AIPlausibilityStatus.WARNING,
            AIPlausibilityStatus.NEEDS_CLARIFICATION,
            AIPlausibilityStatus.BLOCKING,
        }

        status = str(
            parsed.get("status", AIPlausibilityStatus.OK)
        ).strip().upper()

        if status not in allowed_statuses:
            status = AIPlausibilityStatus.WARNING

        raw_issues = parsed.get("issues", [])

        if not isinstance(raw_issues, list):
            raw_issues = []

        allowed_fields = {field.value: field for field in ParamName}
        issues: List[AIPlausibilityIssue] = []

        derived_facts = self._build_derived_facts(payload)

        power_w = self._as_float(payload.get("max_power_w"))
        capacity_tib = self._as_float(
            payload.get("requested_usable_capacity_tib")
        )
        estimated_volume_ratio = self._as_float(
            derived_facts.get("estimated_volume_to_capacity_ratio")
        )
        estimated_volume_tib = self._as_float(
            derived_facts.get("estimated_dataset_volume_tib_from_average_size")
        )
        budget_per_tib = self._as_float(
            derived_facts.get("budget_per_tib_usd")
        )

        for raw_issue in raw_issues:
            if not isinstance(raw_issue, dict):
                continue

            issue_type = str(
                raw_issue.get("issue_type", "AI_PLAUSIBILITY_WARNING")
            ).strip()

            raw_field_value = raw_issue.get("field")
            raw_field_name = ""

            if raw_field_value is not None:
                raw_field_name = str(raw_field_value).strip()

            message = str(raw_issue.get("message", "")).strip()
            question = str(raw_issue.get("question", "")).strip()

            combined_text = (
                f"{issue_type} {raw_field_name} {message} {question}"
            ).lower()

            is_power_issue = (
                "max_power_w" in combined_text
                or "power" in combined_text
                or "puissance" in combined_text
            )

            is_budget_issue = (
                "max_budget_usd" in combined_text
                or "budget" in combined_text
                or raw_field_name == "requested_scale"
            )

            is_file_volume_issue = (
                "average_file_size" in combined_text
                or "total_file_count" in combined_text
                or "file_size" in combined_text
                or "taille moyenne" in combined_text
                or "nombre de fichiers" in combined_text
                or "fichier" in combined_text
            )

            field_name = raw_field_name

            if is_power_issue:
                field_name = ParamName.max_power_w.value

            elif is_budget_issue:
                field_name = ParamName.max_budget_usd.value

            elif is_file_volume_issue:
                if raw_field_name in allowed_fields:
                    field_name = raw_field_name
                else:
                    field_name = ParamName.requested_usable_capacity_tib.value

            if field_name not in allowed_fields:
                continue

            field = allowed_fields[field_name]

            severity = str(
                raw_issue.get("severity", "warning")
            ).strip().lower()

            try:
                confidence = float(raw_issue.get("confidence", 0.0))
            except Exception:
                confidence = 0.0

            confidence = max(0.0, min(1.0, confidence))

            suggested_correction = raw_issue.get("suggested_correction")

            if not isinstance(suggested_correction, dict):
                suggested_correction = None
            else:
                suggested_correction["requires_user_confirmation"] = True

            evidence_fields = raw_issue.get("evidence_fields")

            if not isinstance(evidence_fields, dict):
                evidence_fields = {}

            # ------------------------------------------------------------
            # 1. Power sanity filter
            #
            # 50 W is suspicious.
            # 50 kW / 50000 W or more should not be blocked repeatedly.
            # ------------------------------------------------------------
            if field == ParamName.max_power_w:
                if power_w is not None and power_w >= 10_000:
                    continue

                if (
                    power_w is not None
                    and power_w < 1000
                    and suggested_correction is None
                ):
                    suggested_correction = {
                        "field": ParamName.max_power_w.value,
                        "value": power_w * 1000,
                        "unit": "W",
                        "requires_user_confirmation": True,
                    }

                    message = (
                        f"La puissance maximale de {power_w:g} W semble "
                        "probablement être une erreur d'unité pour cette configuration."
                    )

                    question = (
                        f"Voulez-vous dire {power_w:g} kW, c'est-à-dire "
                        f"{power_w * 1000:g} W ? Répondez 'oui' pour confirmer, "
                        "ou donnez une autre valeur correcte."
                    )

                    severity = "needs_clarification"

            # ------------------------------------------------------------
            # 2. File volume sanity filter
            #
            # If average size × file count is not significantly greater
            # than the requested capacity, the LLM alert is ignored.
            # ------------------------------------------------------------
            if is_file_volume_issue:
                if (
                    estimated_volume_ratio is not None
                    and estimated_volume_ratio <= 1.25
                ):
                    continue

                if (
                    estimated_volume_ratio is not None
                    and estimated_volume_tib is not None
                    and capacity_tib is not None
                ):
                    message = (
                        f"Le volume estimé à partir de la taille moyenne "
                        f"et du nombre de fichiers est d'environ "
                        f"{estimated_volume_tib:.2f} TiB, alors que la "
                        f"capacité demandée est {capacity_tib:g} TiB."
                    )

                    question = (
                        "Cette incohérence peut venir de la capacité demandée, "
                        "de la taille moyenne des fichiers ou du nombre total "
                        "de fichiers. Veuillez corriger la valeur concernée, "
                        "par exemple : 'capacité 6000 TiB', "
                        "'taille moyenne 200 MB', ou "
                        "'nombre de fichiers 300000'."
                    )

                    severity = "needs_clarification"

            # ------------------------------------------------------------
            # 3. Budget sanity filter
            #
            # If the budget is already reasonable per TiB, ignore
            # weak LLM budget alerts.
            # ------------------------------------------------------------
            if field == ParamName.max_budget_usd:
                if budget_per_tib is not None and budget_per_tib >= 100:
                    continue

                if budget_per_tib is not None and budget_per_tib < 100:
                    message = (
                        f"Le budget semble très bas par rapport à la capacité "
                        f"demandée : environ {budget_per_tib:.2f} USD/TiB."
                    )

                    question = (
                        "Veuillez confirmer le budget maximal ou donner une "
                        "nouvelle valeur, par exemple : 'budget 500000 USD'."
                    )

                    severity = "needs_clarification"

            if not message:
                message = (
                    "L'agent de plausibilité a détecté une valeur suspecte."
                )

            if not question:
                question = (
                    "Une valeur semble incohérente avec les autres paramètres. "
                    "Pouvez-vous confirmer ou corriger cette valeur ?"
                )

            issues.append(
                AIPlausibilityIssue(
                    issue_type=issue_type,
                    field=field,
                    severity=severity,
                    message=message,
                    question=question,
                    confidence=confidence,
                    suggested_correction=suggested_correction,
                    evidence_fields=evidence_fields,
                )
            )

        def issue_priority(issue: AIPlausibilityIssue) -> int:
            if issue.field == ParamName.max_power_w:
                return 0

            if issue.field in {
                ParamName.average_file_size_gb,
                ParamName.total_file_count,
                ParamName.requested_usable_capacity_tib,
            } and (
                "average" in issue.issue_type.lower()
                or "file" in issue.issue_type.lower()
                or "fichier" in issue.issue_type.lower()
            ):
                return 1

            if issue.field == ParamName.max_budget_usd:
                return 2

            return 3

        issues = sorted(issues, key=issue_priority)

        if issues:
            issues = [issues[0]]

        if not issues:
            return AIPlausibilityReport(
                status=AIPlausibilityStatus.OK,
                issues=[],
                raw_response=raw_text,
            )

        return AIPlausibilityReport(
            status=AIPlausibilityStatus.NEEDS_CLARIFICATION,
            issues=issues,
            raw_response=raw_text,
        )
