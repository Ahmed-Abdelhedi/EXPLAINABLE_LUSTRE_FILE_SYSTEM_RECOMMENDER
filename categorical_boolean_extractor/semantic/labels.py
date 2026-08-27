from __future__ import annotations
from enum import Enum

class HASemanticLabel(str, Enum):
    HA_REQUIRED = "HA_REQUIRED"
    HA_NOT_REQUIRED = "HA_NOT_REQUIRED"
    HA_MENTION_NO_COMMITMENT = "HA_MENTION_NO_COMMITMENT"
    HA_NO_EVIDENCE = "HA_NO_EVIDENCE"

class AccessSemanticLabel(str, Enum):
    SEQUENTIAL = "SEQUENTIAL"
    RANDOM = "RANDOM"
    MIXED = "MIXED"
    NO_SUPPORTED_ACCESS_CLASS = "NO_SUPPORTED_ACCESS_CLASS"

HA_LABELS = [label.value for label in HASemanticLabel]
ACCESS_LABELS = [label.value for label in AccessSemanticLabel]
HA_TO_ID = {label: i for i, label in enumerate(HA_LABELS)}
ACCESS_TO_ID = {label: i for i, label in enumerate(ACCESS_LABELS)}
ID_TO_HA = {i: label for label, i in HA_TO_ID.items()}
ID_TO_ACCESS = {i: label for label, i in ACCESS_TO_ID.items()}
