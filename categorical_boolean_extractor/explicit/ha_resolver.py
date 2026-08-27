from __future__ import annotations
import re
from typing import Optional
from ..models import FieldResult, FieldStatus, ResolutionSource
from ..text_utils import fold, normalize_text, is_short_affirmation, is_short_negation

def _c(p: str): return re.compile(p, re.I)

FALSE_PATTERNS = [
    _c(r"\b(high availability|ha)\b.{0,18}\b(is\s+)?not\s+(required|mandatory|necessary|essential)\b"),
    _c(r"\bdo not require\b.{0,28}\b(high availability|ha)\b"),
    _c(r"\bdon't require\b.{0,28}\b(high availability|ha)\b"),
    _c(r"\bcan tolerate\b.{0,18}\b(downtime|outage)\b"),
    _c(r"\b(high availability|ha)\b.{0,18}\b(optional|nice to have)\b"),
    _c(r"\b(haute disponibilite|ha)\b.{0,20}\bn[' ]?est pas (obligatoire|requise|necessaire|indispensable|essentielle)\b"),
    _c(r"\b(nous )?n[' ]?(avons )?pas besoin\b.{0,28}\b(haute disponibilite|ha)\b"),
    _c(r"\b(haute disponibilite|ha)\b.{0,18}\b(facultative|optionnelle)\b"),
    _c(r"\bpeut tolerer\b.{0,20}\b(indisponibilite|panne|interruption)\b"),
]
TRUE_PATTERNS = [
    _c(r"\b(ha|high availability)\b.{0,12}\b(is\s+)?not\s+optional\b"),
    _c(r"\b(haute disponibilite|ha)\b.{0,16}\bn[' ]?est pas facultative\b"),
    _c(r"\b(we )?(need|require|must have)\b.{0,28}\b(high availability|ha)\b"),
    _c(r"\b(high availability|ha)\b.{0,18}\b(is\s+)?(mandatory|required|essential|critical)\b"),
    _c(r"\b(mandatory|required|essential|critical)\b.{0,18}\b(high availability|ha)\b"),
    _c(r"\bmust remain available\b"),
    _c(r"\bcannot tolerate\b.{0,18}\b(downtime|outage)\b"),
    _c(r"\bno single point of failure\b"),
    _c(r"\b(nous )?(avons besoin|exigeons|requerons)\b.{0,28}\b(haute disponibilite|ha)\b"),
    _c(r"\b(haute disponibilite|ha)\b.{0,18}\b(est\s+)?(obligatoire|requise|indispensable|essentielle|critique)\b"),
    _c(r"\b(obligatoire|requise|indispensable|essentielle|critique)\b.{0,18}\b(haute disponibilite|ha)\b"),
    _c(r"\bne (pouvons|peut) pas tolerer\b.{0,22}\b(indisponibilite|panne|interruption)\b"),
    _c(r"\baucun point unique de defaillance\b"),
]
HA_MENTION = _c(r"\b(high availability|haute disponibilite|ha)\b")
HISTORICAL = _c(r"\b(previous|old|former|historical|last|ancien|ancienne|precedent|precedente)\b")
CURRENT_ADOPTION = _c(r"\b(now|current|currently|final|this design|this system|maintenant|actuel|actuelle|ce systeme|cette architecture)\b")

class HAExplicitResolver:
    def resolve(self, text: str, *, question_context: Optional[str]=None) -> FieldResult:
        surface = normalize_text(text)
        value = fold(surface)

        if question_context:
            q = fold(question_context)
            asks = bool(re.search(r"\b(high availability|haute disponibilite|ha)\b", q))
            if asks and is_short_affirmation(surface):
                return FieldResult("ha_required", FieldStatus.VERIFIED, True, ResolutionSource.CONTEXT_RESOLVER, surface, "CONTEXTUAL_AFFIRMATION_OF_HA_REQUIREMENT")
            if asks and is_short_negation(surface):
                return FieldResult("ha_required", FieldStatus.VERIFIED, False, ResolutionSource.CONTEXT_RESOLVER, surface, "CONTEXTUAL_NEGATION_OF_HA_REQUIREMENT")

        special = [m for p in TRUE_PATTERNS[:2] if (m := p.search(value))]
        if special:
            m = min(special, key=lambda x: x.start())
            return FieldResult("ha_required", FieldStatus.VERIFIED, True, ResolutionSource.EXPLICIT_RESOLVER, surface[m.start():m.end()], "EXPLICIT_HA_NOT_OPTIONAL")

        neg = [m for p in FALSE_PATTERNS if (m := p.search(value))]
        if neg:
            m = min(neg, key=lambda x: x.start())
            return FieldResult("ha_required", FieldStatus.VERIFIED, False, ResolutionSource.EXPLICIT_RESOLVER, surface[m.start():m.end()], "EXPLICIT_HA_NOT_REQUIRED")

        pos = [m for p in TRUE_PATTERNS[2:] if (m := p.search(value))]
        if pos:
            m = min(pos, key=lambda x: x.start())
            return FieldResult("ha_required", FieldStatus.VERIFIED, True, ResolutionSource.EXPLICIT_RESOLVER, surface[m.start():m.end()], "EXPLICIT_HA_REQUIRED")

        if HA_MENTION.search(value):
            if HISTORICAL.search(value) and not CURRENT_ADOPTION.search(value):
                return FieldResult("ha_required", FieldStatus.NO_EVIDENCE, None, ResolutionSource.NONE, surface, "HISTORICAL_HA_MENTION_WITHOUT_CURRENT_COMMITMENT")
            return FieldResult("ha_required", FieldStatus.UNRESOLVED, None, ResolutionSource.NONE, surface, "HA_MENTION_WITHOUT_EXPLICIT_COMMITMENT")

        return FieldResult("ha_required", FieldStatus.NO_EVIDENCE, None, ResolutionSource.NONE, None, "NO_EXPLICIT_HA_EVIDENCE")
