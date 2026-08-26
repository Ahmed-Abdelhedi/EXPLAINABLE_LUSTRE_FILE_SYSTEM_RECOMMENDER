from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

from dotenv import load_dotenv

from .labels import (
    PreferenceDimension,
    PreferenceLevel,
    ResolutionSource,
    ResolutionStatus,
)
from .residual_validator import (
    RESIDUAL_VALIDATOR_VERSION,
    validate_residual_prediction,
)
from .schemas import (
    DimensionPreferenceResult,
    PreferenceRelation,
)


HYBRID_PROMPT_VERSION = (
    "layer2_hybrid_residual_qwen_v1_20260825"
)

HYBRID_PROMPT = r"""
You are the residual fallback for Layer 2 of a multilingual preference
extractor.

A deterministic semantic guard has already resolved clear absolute intensity,
clear pure comparisons, and clear hard-negative/no-preference cases.

Resolve ONLY the dimensions listed in REQUESTED_DIMENSIONS.

Allowed dimensions:
cost, power, performance, reliability

Allowed statuses:
RESOLVED, NO_SIGNAL, RELATIVE_ONLY, UNRESOLVED

Allowed RESOLVED levels:
VERY_LOW, LOW, MEDIUM, HIGH, VERY_HIGH

Rules:
1. NO_SIGNAL is not VERY_LOW.
2. Pure comparison/order without an independent absolute cue is RELATIVE_ONLY.
3. Do not infer priority intensity from numeric limits, throughput values,
   system capabilities, API/log fields, or technical adjectives.
4. Current/final/latest user choice overrides superseded/history.
5. Vendor/third-party opinion is not the user's preference unless adopted.
6. If uncertain, return UNRESOLVED.
7. Evidence for RESOLVED or RELATIVE_ONLY must be an exact substring copied
   from CURRENT_USER_MESSAGE.
8. Return every requested dimension exactly once and no others.
9. JSON only.

Output:
{
  "dimensions": {
    "<requested dimension>": {
      "status": "RESOLVED|NO_SIGNAL|RELATIVE_ONLY|UNRESOLVED",
      "level": "VERY_LOW|LOW|MEDIUM|HIGH|VERY_HIGH|null",
      "evidence": "exact substring|null"
    }
  }
}
""".strip()


def _clean_json_candidate(
    raw: str,
) -> str:
    value = (
        raw
        or ""
    ).strip()

    if value.startswith("```"):
        value = re.sub(
            r"^```(?:json)?\s*",
            "",
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(
            r"\s*```$",
            "",
            value,
        )

    first = value.find("{")
    last = value.rfind("}")

    if (
        first >= 0
        and last >= first
    ):
        value = value[
            first:
            last + 1
        ]

    return value


def parse_llm_response(
    *,
    raw_text: str,
    requested_dimensions: Sequence[str],
    user_text: str,
) -> Dict[str, Any]:
    requested = tuple(
        requested_dimensions
    )
    requested_set = set(requested)
    violations: List[str] = []

    try:
        payload = json.loads(
            _clean_json_candidate(
                raw_text
            )
        )
    except Exception:
        return {
            "valid": False,
            "violations": [
                "INVALID_JSON",
            ],
            "dimensions": {
                dimension: {
                    "status": "UNRESOLVED",
                    "level": None,
                    "evidence": None,
                    "accepted": False,
                    "validation_error": "INVALID_JSON",
                }
                for dimension
                in requested
            },
        }

    raw_dimensions = payload.get(
        "dimensions"
    )

    if not isinstance(
        raw_dimensions,
        dict,
    ):
        raw_dimensions = {}
        violations.append(
            "MISSING_DIMENSIONS_OBJECT"
        )

    extras = sorted(
        set(raw_dimensions)
        - requested_set
    )

    if extras:
        violations.append(
            "UNREQUESTED_DIMENSIONS:"
            + ",".join(extras)
        )

    output: Dict[
        str,
        Dict[str, Any]
    ] = {}

    allowed_status = {
        "RESOLVED",
        "NO_SIGNAL",
        "RELATIVE_ONLY",
        "UNRESOLVED",
    }

    levels = {
        "VERY_LOW",
        "LOW",
        "MEDIUM",
        "HIGH",
        "VERY_HIGH",
    }

    for dimension in requested:
        item = raw_dimensions.get(
            dimension
        )
        errors: List[str] = []

        if not isinstance(
            item,
            dict,
        ):
            output[dimension] = {
                "status": "UNRESOLVED",
                "level": None,
                "evidence": None,
                "accepted": False,
                "validation_error":
                    "MISSING_REQUESTED_DIMENSION",
            }
            violations.append(
                f"MISSING:{dimension}"
            )
            continue

        status = str(
            item.get(
                "status",
                "UNRESOLVED",
            )
        ).strip().upper()

        level_raw = item.get("level")
        level = (
            str(level_raw)
            .strip()
            .upper()
            if level_raw is not None
            else None
        )

        evidence_raw = item.get(
            "evidence"
        )
        evidence = (
            str(evidence_raw)
            if evidence_raw is not None
            else None
        )

        if status not in allowed_status:
            errors.append(
                "INVALID_STATUS"
            )
            status = "UNRESOLVED"
            level = None

        if status == "RESOLVED":
            if level not in levels:
                errors.append(
                    "INVALID_RESOLVED_LEVEL"
                )
                status = "UNRESOLVED"
                level = None

            if (
                not evidence
                or evidence not in user_text
            ):
                errors.append(
                    "UNSUPPORTED_EVIDENCE"
                )
                status = "UNRESOLVED"
                level = None
        else:
            if level is not None:
                errors.append(
                    "LEVEL_MUST_BE_NULL"
                )
                status = "UNRESOLVED"
                level = None

            if (
                status
                == "RELATIVE_ONLY"
                and (
                    not evidence
                    or evidence
                    not in user_text
                )
            ):
                errors.append(
                    "UNSUPPORTED_EVIDENCE"
                )
                status = "UNRESOLVED"

            if (
                status
                == "NO_SIGNAL"
                and evidence is not None
                and evidence not in user_text
            ):
                errors.append(
                    "UNSUPPORTED_EVIDENCE"
                )
                status = "UNRESOLVED"

        accepted = (
            not errors
            and status
            in {
                "RESOLVED",
                "NO_SIGNAL",
                "RELATIVE_ONLY",
            }
        )

        output[dimension] = {
            "status": status,
            "level": level,
            "evidence": evidence,
            "accepted": accepted,
            "validation_error": (
                ";".join(errors)
                if errors
                else None
            ),
        }

        if errors:
            violations.extend(
                f"{dimension}:{error}"
                for error
                in errors
            )

    return {
        "valid": not violations,
        "violations": violations,
        "dimensions": output,
    }


class PreferenceLLMFallback:
    """
    Production residual fallback.

    Production flow:
        Transformer abstention
        -> deterministic semantic guard
        -> this residual Qwen fallback
        -> deterministic residual evidence validator

    Default backend:
        Ollama + qwen2.5-coder:7b

    The LLM can propose a result, but the deterministic residual validator is
    the final automatic-acceptance boundary.
    """

    prompt_version = HYBRID_PROMPT_VERSION
    residual_validator_version = (
        RESIDUAL_VALIDATOR_VERSION
    )

    def __init__(
        self,
        enabled: Optional[bool] = None,
        host: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: int = 120,
    ) -> None:
        load_dotenv()

        env_enabled = (
            os.getenv(
                "ENABLE_PREFERENCE_LLM_FALLBACK",
                os.getenv(
                    "ENABLE_LLM_FALLBACK",
                    "false",
                ),
            )
            .strip()
            .lower()
            == "true"
        )

        self.enabled = (
            env_enabled
            if enabled is None
            else bool(enabled)
        )

        self.host = (
            host
            or os.getenv(
                "OLLAMA_HOST",
                "http://localhost:11434",
            )
        ).rstrip("/")

        self.model = (
            model
            or os.getenv(
                "OLLAMA_MODEL",
                "qwen2.5-coder:7b",
            )
        )

        self.timeout_seconds = int(
            timeout_seconds
        )

        self.call_count = 0

        # Traceability for production smoke/debugging.
        self.last_requested_dimensions: List[str] = []
        self.last_raw_response: Optional[str] = None
        self.last_response_valid: Optional[bool] = None
        self.last_response_violations: List[str] = []
        self.last_latency_seconds: Optional[float] = None
        self.last_accepted_dimensions: List[str] = []

    @staticmethod
    def _prompt(
        text: str,
        unresolved: Sequence[
            PreferenceDimension
        ],
        relations: List[
            PreferenceRelation
        ],
    ) -> str:
        del relations  # frozen hybrid prompt does not consume relation metadata.

        dimensions = [
            dimension.value
            for dimension in unresolved
        ]

        return (
            HYBRID_PROMPT
            + "\n\nREQUESTED_DIMENSIONS\n"
            + json.dumps(
                dimensions,
                ensure_ascii=False,
            )
            + "\n\nCURRENT_USER_MESSAGE\n"
            + text
        )

    def _call_ollama(
        self,
        prompt: str,
    ) -> tuple[str, float]:
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "keep_alive": "30m",
                "options": {
                    "temperature": 0.0,
                    "num_predict": 320,
                },
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={
                "Content-Type":
                    "application/json"
            },
            method="POST",
        )

        started = time.perf_counter()

        with urllib.request.urlopen(
            request,
            timeout=self.timeout_seconds,
        ) as response:
            outer = json.loads(
                response.read().decode(
                    "utf-8"
                )
            )

        latency = (
            time.perf_counter()
            - started
        )

        return (
            str(
                outer.get(
                    "response",
                    "",
                )
            ),
            latency,
        )

    def resolve(
        self,
        *,
        text: str,
        unresolved_dimensions: List[
            PreferenceDimension
        ],
        relations: List[
            PreferenceRelation
        ],
    ) -> Dict[
        PreferenceDimension,
        DimensionPreferenceResult,
    ]:
        self.last_requested_dimensions = [
            dimension.value
            for dimension in unresolved_dimensions
        ]
        self.last_raw_response = None
        self.last_response_valid = None
        self.last_response_violations = []
        self.last_latency_seconds = None
        self.last_accepted_dimensions = []

        if (
            not self.enabled
            or not unresolved_dimensions
        ):
            return {}

        prompt = self._prompt(
            text,
            unresolved_dimensions,
            relations,
        )

        self.call_count += 1

        try:
            raw_text, latency = (
                self._call_ollama(
                    prompt
                )
            )
        except (
            OSError,
            ValueError,
            json.JSONDecodeError,
            urllib.error.URLError,
        ):
            return {}

        self.last_raw_response = raw_text
        self.last_latency_seconds = latency

        parsed = parse_llm_response(
            raw_text=raw_text,
            requested_dimensions=[
                dimension.value
                for dimension
                in unresolved_dimensions
            ],
            user_text=text,
        )

        self.last_response_valid = bool(
            parsed["valid"]
        )
        self.last_response_violations = list(
            parsed["violations"]
        )

        output: Dict[
            PreferenceDimension,
            DimensionPreferenceResult,
        ] = {}

        for dimension in unresolved_dimensions:
            raw_prediction = (
                parsed[
                    "dimensions"
                ][
                    dimension.value
                ]
            )

            validated = (
                validate_residual_prediction(
                    text=text,
                    dimension=dimension.value,
                    prediction=raw_prediction,
                )
            )

            prediction = validated.prediction

            if not prediction.get(
                "accepted",
                False,
            ):
                continue

            try:
                status = ResolutionStatus(
                    str(
                        prediction["status"]
                    )
                )
            except ValueError:
                continue

            level = None

            if (
                status
                == ResolutionStatus.RESOLVED
            ):
                try:
                    level = PreferenceLevel(
                        str(
                            prediction["level"]
                        )
                    )
                except ValueError:
                    continue

            evidence = prediction.get(
                "evidence"
            )

            output[dimension] = (
                DimensionPreferenceResult(
                    dimension=dimension,
                    status=status,
                    source=(
                        ResolutionSource.LLM_FALLBACK
                    ),
                    level=level,
                    evidence=(
                        str(evidence)
                        if evidence is not None
                        else None
                    ),
                    reason=(
                        "Residual LLM accepted by deterministic "
                        f"validator: {validated.action}"
                    ),
                )
            )

        self.last_accepted_dimensions = [
            dimension.value
            for dimension in output
        ]

        return output
