from __future__ import annotations
import json, os, re, time, urllib.request
from typing import Dict, List, Optional

PROMPT_VERSION = "categorical_boolean_semantic_residual_qwen_v2_20260826"
PROMPT = r"""
You are the residual semantic fallback for two storage requirement fields.
Resolve ONLY REQUESTED_FIELDS.

HA labels:
HA_REQUIRED | HA_NOT_REQUIRED | HA_MENTION_NO_COMMITMENT | HA_NO_EVIDENCE

Access labels:
SEQUENTIAL | RANDOM | MIXED | NO_SUPPORTED_ACCESS_CLASS

Rules:
1. "HA matters", "HA is important", "HA is expensive" do NOT entail HA_REQUIRED.
2. "HA is required", mandatory HA, "HA is not optional", inability to tolerate
   downtime, or a mandatory no-single-point-of-failure requirement may support HA_REQUIRED.
3. Explicitly optional/not-required HA or tolerance for downtime may support HA_NOT_REQUIRED.
4. Historical HA mention is not a current requirement.
5. access_type is I/O ordering/pattern, not concurrency.
6. Parallel/concurrent I/O alone is NO_SUPPORTED_ACCESS_CLASS.
7. Streaming-like access may be SEQUENTIAL.
8. Random + sequential without explicit dominance is MIXED.
9. Explicit dominance such as "mostly sequential" uses the dominant class.
10. If uncertain, return UNRESOLVED.
11. VERIFIED evidence must be an exact substring copied from CURRENT_USER_MESSAGE.
12. Never invent evidence. JSON only.

Output:
{
  "fields": {
    "<requested field>": {
      "status": "VERIFIED|NO_EVIDENCE|UNRESOLVED",
      "label": "<one allowed label or null>",
      "evidence": "exact substring or null"
    }
  }
}
""".strip()

def _clean_json(raw: str) -> str:
    value = (raw or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
        value = re.sub(r"\s*```$", "", value)
    a, b = value.find("{"), value.rfind("}")
    return value[a:b+1] if a >= 0 and b >= a else value

class CategoricalBooleanLLMFallback:
    def __init__(
        self,
        *,
        enabled: Optional[bool]=None,
        host: Optional[str]=None,
        model: Optional[str]=None,
        timeout_seconds: int=90,
    ) -> None:
        env = os.getenv("ENABLE_CATEGORICAL_BOOLEAN_LLM_FALLBACK", "false").strip().lower() == "true"
        self.enabled = env if enabled is None else bool(enabled)
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
        self.timeout_seconds = int(timeout_seconds)
        self.call_count = 0
        self.last_latency_seconds = None
        self.last_raw_response = None
        self.last_requested_fields: List[str] = []

    def _call(self, prompt: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "keep_alive": "30m",
            "options": {"temperature": 0.0, "num_predict": 320},
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        self.call_count += 1
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            outer = json.loads(response.read().decode("utf-8"))
        self.last_latency_seconds = time.perf_counter() - started
        self.last_raw_response = str(outer.get("response", ""))
        return self.last_raw_response

    def propose(self, *, text: str, requested_fields: List[str]) -> Dict[str, dict]:
        if not self.enabled or not requested_fields:
            return {}
        self.last_requested_fields = list(requested_fields)
        prompt = PROMPT + "\n\nREQUESTED_FIELDS\n" + json.dumps(requested_fields) + "\n\nCURRENT_USER_MESSAGE\n" + text
        try:
            payload = json.loads(_clean_json(self._call(prompt)))
        except Exception:
            return {}
        fields = payload.get("fields")
        if not isinstance(fields, dict) or set(fields) - set(requested_fields):
            return {}
        return {
            field: proposal
            for field, proposal in fields.items()
            if field in requested_fields and isinstance(proposal, dict)
        }
