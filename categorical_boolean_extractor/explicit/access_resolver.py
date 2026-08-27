from __future__ import annotations
import re
from typing import Optional
from ..models import AccessType, FieldResult, FieldStatus, ResolutionSource
from ..text_utils import fold, normalize_text

def _c(p: str): return re.compile(p, re.I)
SEQ = [
    _c(r"\bsequential(?:ly)?\b"),
    _c(r"\bsequentiel(?:le|s|les)?\b"),
    _c(r"\bcontiguous (?:reads?|writes?|access)\b"),
    _c(r"\bfull (?:file )?scans?\b"),
    _c(r"\bstreaming\b"),
    _c(r"\bstream(?:ed|ing)? (?:reads?|writes?|io|i/o)\b"),
    _c(r"\bflux continu\b"),
]
RND = [
    _c(r"\brandom(?:ly)?\b"),
    _c(r"\bnon[- ]?contiguous (?:reads?|writes?|access)\b"),
    _c(r"\bscattered (?:reads?|writes?|access)\b"),
    _c(r"\baleatoire(?:s)?\b"),
]
MIX = [
    _c(r"\bmixed (?:access|workload|io|i/o)\b"),
    _c(r"\bworkload is mixed\b"),
    _c(r"\bcharge mixte\b"),
    _c(r"\bacces mixtes?\b"),
    _c(r"\bboth\b.{0,32}\b(random|aleatoire)\b.{0,32}\b(sequential|sequentiel|streaming)\b"),
    _c(r"\bboth\b.{0,32}\b(sequential|sequentiel|streaming)\b.{0,32}\b(random|aleatoire)\b"),
]
DOM_SEQ = [
    _c(r"\b(mostly|mainly|primarily|predominantly)\s+(sequential|streaming)\b"),
    _c(r"\b(principalement|majoritairement|surtout)\s+(sequentiel|sequentielle|streaming)\b"),
]
DOM_RND = [
    _c(r"\b(mostly|mainly|primarily|predominantly)\s+random\b"),
    _c(r"\b(principalement|majoritairement|surtout)\s+(aleatoire|aleatoires)\b"),
]
PARALLEL = _c(r"\b(parallel|parallelism|concurrent|concurrently|parallele|paralleles|simultane|simultanes)\b")
CORR_MIX = _c(r"\b(mixed|mixte)\b.{0,30}\b(rather than|instead of|plutot que|au lieu de)\b")

class AccessTypeExplicitResolver:
    def resolve(self, text: str, *, question_context: Optional[str]=None) -> FieldResult:
        surface = normalize_text(text)
        value = fold(surface)

        if question_context:
            q = fold(question_context)
            if re.search(r"\b(access type|io pattern|i/o pattern|pattern of access|type d[' ]?acces|mode d[' ]?acces)\b", q):
                mapping = {
                    "sequential":"sequential","sequentiel":"sequential","streaming":"sequential",
                    "random":"random","aleatoire":"random","mixed":"mixed","mixte":"mixed",
                }
                compact = value.strip(" .!?")
                if compact in mapping:
                    return FieldResult("access_type", FieldStatus.VERIFIED, mapping[compact], ResolutionSource.CONTEXT_RESOLVER, surface, "CONTEXTUAL_ACCESS_TYPE_ANSWER")

        if CORR_MIX.search(value):
            return FieldResult("access_type", FieldStatus.VERIFIED, AccessType.MIXED.value, ResolutionSource.EXPLICIT_RESOLVER, surface, "EXPLICIT_ACCESS_TYPE_CORRECTION_TO_MIXED")

        ds = [m for p in DOM_SEQ if (m := p.search(value))]
        dr = [m for p in DOM_RND if (m := p.search(value))]
        if ds and dr:
            return FieldResult("access_type", FieldStatus.CONFLICT, None, ResolutionSource.EXPLICIT_RESOLVER, surface, "CONFLICTING_DOMINANT_ACCESS_CUES")
        if ds:
            m = min(ds, key=lambda x: x.start())
            return FieldResult("access_type", FieldStatus.VERIFIED, "sequential", ResolutionSource.EXPLICIT_RESOLVER, surface[m.start():m.end()], "EXPLICIT_SEQUENTIAL_DOMINANCE")
        if dr:
            m = min(dr, key=lambda x: x.start())
            return FieldResult("access_type", FieldStatus.VERIFIED, "random", ResolutionSource.EXPLICIT_RESOLVER, surface[m.start():m.end()], "EXPLICIT_RANDOM_DOMINANCE")

        mm = [m for p in MIX if (m := p.search(value))]
        if mm:
            m = min(mm, key=lambda x: x.start())
            return FieldResult("access_type", FieldStatus.VERIFIED, "mixed", ResolutionSource.EXPLICIT_RESOLVER, surface[m.start():m.end()], "EXPLICIT_MIXED_ACCESS")

        ss = [m for p in SEQ if (m := p.search(value))]
        rr = [m for p in RND if (m := p.search(value))]
        if ss and rr:
            return FieldResult("access_type", FieldStatus.VERIFIED, "mixed", ResolutionSource.EXPLICIT_RESOLVER, surface, "COEXISTING_RANDOM_AND_SEQUENTIAL_WITHOUT_DOMINANCE")
        if ss:
            m = min(ss, key=lambda x: x.start())
            return FieldResult("access_type", FieldStatus.VERIFIED, "sequential", ResolutionSource.EXPLICIT_RESOLVER, surface[m.start():m.end()], "EXPLICIT_SEQUENTIAL_ACCESS")
        if rr:
            m = min(rr, key=lambda x: x.start())
            return FieldResult("access_type", FieldStatus.VERIFIED, "random", ResolutionSource.EXPLICIT_RESOLVER, surface[m.start():m.end()], "EXPLICIT_RANDOM_ACCESS")
        if PARALLEL.search(value):
            return FieldResult("access_type", FieldStatus.UNRESOLVED, None, ResolutionSource.NONE, surface, "PARALLELISM_DOES_NOT_DEFINE_IO_ORDER")

        return FieldResult("access_type", FieldStatus.NO_EVIDENCE, None, ResolutionSource.NONE, None, "NO_EXPLICIT_ACCESS_TYPE_EVIDENCE")
