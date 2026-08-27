from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
from ..semantic.labels import ACCESS_LABELS, HA_LABELS

ALLOWED_LANGUAGES = {"en", "fr", "mixed"}

@dataclass(frozen=True)
class TrainingRecord:
    sample_id: str
    text: str
    language: str
    ha_label: str
    access_label: str
    semantic_family: str
    template_id: str
    structure_fingerprint: str

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]):
        required = {
            "sample_id", "text", "language", "ha_label", "access_label",
            "semantic_family", "template_id", "structure_fingerprint",
        }
        missing = required - set(payload)
        if missing:
            raise ValueError("Missing fields: " + ", ".join(sorted(missing)))
        record = cls(**{key: str(payload[key]) for key in required})
        record.validate()
        return record

    def validate(self):
        if not self.text.strip():
            raise ValueError("text must be non-empty")
        if self.language not in ALLOWED_LANGUAGES:
            raise ValueError(f"Invalid language: {self.language}")
        if self.ha_label not in HA_LABELS:
            raise ValueError(f"Invalid HA label: {self.ha_label}")
        if self.access_label not in ACCESS_LABELS:
            raise ValueError(f"Invalid access label: {self.access_label}")
        for value, name in [
            (self.sample_id, "sample_id"),
            (self.semantic_family, "semantic_family"),
            (self.template_id, "template_id"),
            (self.structure_fingerprint, "structure_fingerprint"),
        ]:
            if not value.strip():
                raise ValueError(f"{name} must be non-empty")
