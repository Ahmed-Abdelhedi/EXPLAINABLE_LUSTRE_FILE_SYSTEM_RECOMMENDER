from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .models import ParamName, Quantity, SemanticLink, SemanticRole


class CandidateRelationType(str, Enum):
    """Controlled relation between candidates extracted from one message."""

    SINGLE_VALUE = "SINGLE_VALUE"
    ALTERNATIVE = "ALTERNATIVE"
    RANGE = "RANGE"
    CORRECTION = "CORRECTION"
    COMPARISON = "COMPARISON"
    MULTIPLE_FIELDS = "MULTIPLE_FIELDS"
    CONFLICT = "CONFLICT"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class CandidateRelation:
    """Relation assigned to a group of quantity candidates."""

    relation: CandidateRelationType
    quantity_ids: List[str]
    fields: List[ParamName]
    selected_quantity_ids: List[str] = field(default_factory=list)
    blocked_quantity_ids: List[str] = field(default_factory=list)
    evidence: str = ""
    reason: str = ""
    resolver: str = "deterministic_structure"

    @property
    def blocks_automatic_acceptance(self) -> bool:
        return bool(self.blocked_quantity_ids)

    def to_dict(self) -> dict:
        return {
            "relation": self.relation.value,
            "quantity_ids": list(self.quantity_ids),
            "fields": [item.value for item in self.fields],
            "selected_quantity_ids": list(self.selected_quantity_ids),
            "blocked_quantity_ids": list(self.blocked_quantity_ids),
            "evidence": self.evidence,
            "reason": self.reason,
            "resolver": self.resolver,
            "blocks_automatic_acceptance": self.blocks_automatic_acceptance,
        }


@dataclass
class CandidateRelationResolution:
    """Complete relation analysis for one user message."""

    text: str
    relations: List[CandidateRelation] = field(default_factory=list)

    @property
    def blocked_quantity_ids(self) -> List[str]:
        blocked: List[str] = []
        seen = set()
        for relation in self.relations:
            for quantity_id in relation.blocked_quantity_ids:
                if quantity_id not in seen:
                    blocked.append(quantity_id)
                    seen.add(quantity_id)
        return blocked

    @property
    def has_blocking_relation(self) -> bool:
        return any(relation.blocks_automatic_acceptance for relation in self.relations)

    def relations_for_quantity(self, quantity_id: str) -> List[CandidateRelation]:
        return [
            relation
            for relation in self.relations
            if quantity_id in relation.quantity_ids
        ]

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "relations": [relation.to_dict() for relation in self.relations],
            "blocked_quantity_ids": self.blocked_quantity_ids,
            "has_blocking_relation": self.has_blocking_relation,
        }


# Optional small semantic relation classifier. It is deliberately not required.
# The deterministic resolver remains conservative when no classifier is supplied.
SemanticRelationClassifier = Callable[
    [str, Sequence[Quantity], Sequence[SemanticLink]],
    Optional[CandidateRelationType],
]


@dataclass(frozen=True)
class _CandidateEntry:
    quantity: Quantity
    link: SemanticLink


class CandidateRelationResolver:
    """
    Resolve relations between all candidates produced from the same message.

    This component runs after field/role linking and before final deterministic
    acceptance. It does not infer fields and it never repairs a semantic link.

    The deterministic part only uses local structure around quantity spans.
    A semantic relation classifier may optionally be injected for cases where
    deterministic structure is insufficient. Without one, uncertain cases are
    kept conservative and cannot be automatically accepted.
    """

    _ALTERNATIVE_CONNECTOR = re.compile(r"\b(?:or|ou)\b", re.IGNORECASE)
    _ALTERNATIVE_PREFIX = re.compile(r"\b(?:either|soit)\b", re.IGNORECASE)

    _BETWEEN_PREFIX = re.compile(r"\b(?:between|entre)\b", re.IGNORECASE)
    _FROM_PREFIX = re.compile(r"\b(?:from|de)\s*$", re.IGNORECASE)
    _RANGE_AND_CONNECTOR = re.compile(r"\b(?:and|et)\b", re.IGNORECASE)
    _RANGE_TO_CONNECTOR = re.compile(r"\b(?:to|a|au)\b", re.IGNORECASE)

    _CORRECTION_PREFIX = re.compile(
        r"\b(?:change|changed|replace|replaced|update|updated|correct|corrected|"
        r"modify|modified|set|switch|changer|change|remplace|remplacer|remplacee|"
        r"remplacez|mettre|mets|mettez|modifier|modifie|modifiez|corriger|corrige|"
        r"corrigez|actualiser|actualise|actualisez)\b",
        re.IGNORECASE,
    )
    _CORRECTION_CONNECTOR = re.compile(
        r"\b(?:to|with|by|into|a|en|par|vers)\b",
        re.IGNORECASE,
    )

    _COMPARISON_CONNECTOR = re.compile(
        r"\b(?:vs|versus|than|against|contre|plutot\s+que|rather\s+than|"
        r"instead\s+of|au\s+lieu\s+de|compared\s+to|compared\s+with|"
        r"compare\s+to|compare\s+avec)\b",
        re.IGNORECASE,
    )

    _READ_MARKER = re.compile(r"\b(?:read|reads|reading|lecture|lire)\b", re.IGNORECASE)
    _WRITE_MARKER = re.compile(r"\b(?:write|writes|writing|ecriture|ecrire)\b", re.IGNORECASE)

    def __init__(
        self,
        semantic_classifier: Optional[SemanticRelationClassifier] = None,
    ) -> None:
        self.semantic_classifier = semantic_classifier

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def resolve(
        self,
        text: str,
        quantities: Iterable[Quantity],
        links: Iterable[SemanticLink],
    ) -> CandidateRelationResolution:
        quantity_list = list(quantities)
        link_list = list(links)
        quantity_by_id = {quantity.id: quantity for quantity in quantity_list}

        entries: List[_CandidateEntry] = []
        for link in link_list:
            quantity = quantity_by_id.get(link.quantity_id)
            if quantity is None or link.field is None:
                continue
            entries.append(_CandidateEntry(quantity=quantity, link=link))

        entries.sort(key=lambda item: (item.quantity.start, item.quantity.end, item.quantity.id))
        relations: List[CandidateRelation] = []

        unique_fields = self._ordered_unique_fields(entries)
        if len(unique_fields) > 1:
            quantity_ids = self._ordered_unique_ids(
                [entry.quantity.id for entry in entries]
            )
            relations.append(
                CandidateRelation(
                    relation=CandidateRelationType.MULTIPLE_FIELDS,
                    quantity_ids=quantity_ids,
                    fields=unique_fields,
                    selected_quantity_ids=list(quantity_ids),
                    blocked_quantity_ids=[],
                    evidence=self._group_evidence(text, [entry.quantity for entry in entries]),
                    reason="candidates_map_to_multiple_distinct_fields",
                )
            )

        entries_by_field: Dict[ParamName, List[_CandidateEntry]] = {}
        for entry in entries:
            entries_by_field.setdefault(entry.link.field, []).append(entry)

        for field_name, field_entries in entries_by_field.items():
            relations.append(
                self._resolve_field_group(
                    text=text,
                    field_name=field_name,
                    entries=field_entries,
                )
            )

        # A quantity with no semantic link remains a semantic-linking problem,
        # not a relation problem. The DeterministicVerifier already reports it
        # as UNRESOLVED, so we intentionally do not manufacture a relation here.
        return CandidateRelationResolution(text=text, relations=relations)

    def info(self) -> dict:
        return {
            "name": "CandidateRelationResolver",
            "deterministic_first": True,
            "semantic_fallback_configured": self.semantic_classifier is not None,
            "relations": [item.value for item in CandidateRelationType],
        }

    # -----------------------------------------------------------------
    # Field-group resolution
    # -----------------------------------------------------------------
    def _resolve_field_group(
        self,
        text: str,
        field_name: ParamName,
        entries: Sequence[_CandidateEntry],
    ) -> CandidateRelation:
        ordered = sorted(
            entries,
            key=lambda item: (item.quantity.start, item.quantity.end, item.quantity.id),
        )
        quantity_ids = self._ordered_unique_ids(
            [entry.quantity.id for entry in ordered]
        )
        quantities = [entry.quantity for entry in ordered]
        links = [entry.link for entry in ordered]
        evidence = self._group_evidence(text, quantities)

        if len(quantity_ids) <= 1:
            return CandidateRelation(
                relation=CandidateRelationType.SINGLE_VALUE,
                quantity_ids=quantity_ids,
                fields=[field_name],
                selected_quantity_ids=list(quantity_ids),
                evidence=evidence,
                reason="one_candidate_for_field",
            )

        prefix, connectors, suffix = self._local_structure(text, quantities)
        normalized_prefix = self._fold(prefix)
        normalized_connectors = [self._fold(item) for item in connectors]
        normalized_evidence = self._fold(evidence)

        relation = self._detect_structural_relation(
            field_name=field_name,
            entries=ordered,
            prefix=normalized_prefix,
            connectors=normalized_connectors,
            evidence=normalized_evidence,
        )

        if relation is None and self.semantic_classifier is not None:
            relation = self.semantic_classifier(text, quantities, links)
            if relation is not None:
                resolver_name = "semantic_relation_classifier"
            else:
                resolver_name = "deterministic_structure"
        else:
            resolver_name = "deterministic_structure"

        if relation is None:
            relation = CandidateRelationType.CONFLICT

        if relation == CandidateRelationType.CORRECTION:
            selected = [quantity_ids[-1]]
            blocked = quantity_ids[:-1]
            reason = "later_candidate_replaces_earlier_candidate"
        elif relation in {
            CandidateRelationType.ALTERNATIVE,
            CandidateRelationType.RANGE,
            CandidateRelationType.COMPARISON,
            CandidateRelationType.CONFLICT,
            CandidateRelationType.UNRESOLVED,
        }:
            selected = []
            blocked = list(quantity_ids)
            reason = self._blocking_reason(relation)
        else:
            selected = list(quantity_ids)
            blocked = []
            reason = self._non_blocking_reason(relation, field_name)

        return CandidateRelation(
            relation=relation,
            quantity_ids=quantity_ids,
            fields=[field_name],
            selected_quantity_ids=selected,
            blocked_quantity_ids=blocked,
            evidence=evidence,
            reason=reason,
            resolver=resolver_name,
        )

    def _detect_structural_relation(
        self,
        field_name: ParamName,
        entries: Sequence[_CandidateEntry],
        prefix: str,
        connectors: Sequence[str],
        evidence: str,
    ) -> Optional[CandidateRelationType]:
        joined_connectors = " | ".join(connectors)

        # CORRECTION must be checked before RANGE because formulations such
        # as "change ... from 20 to 30" contain a from/to structure but are
        # semantically replacements, not ranges.
        if self._looks_like_correction(prefix, connectors, evidence):
            return CandidateRelationType.CORRECTION

        # RANGE is then checked before generic conjunctions.
        if self._looks_like_range(prefix, connectors):
            return CandidateRelationType.RANGE

        # ALTERNATIVE is evaluated on the local text between candidate spans,
        # not on a global "if the sentence contains or" rule.
        if any(self._ALTERNATIVE_CONNECTOR.search(item) for item in connectors):
            return CandidateRelationType.ALTERNATIVE
        if self._ALTERNATIVE_PREFIX.search(prefix) and joined_connectors:
            return CandidateRelationType.ALTERNATIVE

        if any(self._COMPARISON_CONNECTOR.search(item) for item in connectors):
            return CandidateRelationType.COMPARISON

        # read_write_ratio is intentionally represented by two scalar
        # components in the current V2 contract. Treat a structurally valid
        # pair as one composite field interpretation, not as a conflict.
        if field_name == ParamName.read_write_ratio and self._is_ratio_composite(
            entries=entries,
            connectors=connectors,
            evidence=evidence,
        ):
            return CandidateRelationType.SINGLE_VALUE

        # Repeated identical values for the same field are redundant but safe.
        values = [entry.quantity.value for entry in entries]
        if self._all_values_equivalent(values):
            return CandidateRelationType.SINGLE_VALUE

        return None

    # -----------------------------------------------------------------
    # Structural recognizers
    # -----------------------------------------------------------------
    def _looks_like_range(self, prefix: str, connectors: Sequence[str]) -> bool:
        if self._BETWEEN_PREFIX.search(prefix):
            return any(self._RANGE_AND_CONNECTOR.search(item) for item in connectors)

        if self._FROM_PREFIX.search(prefix):
            return any(self._RANGE_TO_CONNECTOR.search(item) for item in connectors)

        for connector in connectors:
            stripped = connector.strip()
            if stripped in {"-", "–", "—"}:
                return True

        return False

    def _looks_like_correction(
        self,
        prefix: str,
        connectors: Sequence[str],
        evidence: str,
    ) -> bool:
        correction_signal = bool(
            self._CORRECTION_PREFIX.search(prefix)
            or self._CORRECTION_PREFIX.search(evidence)
        )
        if not correction_signal:
            return False

        return any(
            self._CORRECTION_CONNECTOR.search(connector)
            for connector in connectors
        )

    def _is_ratio_composite(
        self,
        entries: Sequence[_CandidateEntry],
        connectors: Sequence[str],
        evidence: str,
    ) -> bool:
        if not all(entry.link.role == SemanticRole.RATIO_COMPONENT for entry in entries):
            return False

        # Short contextual answer such as "70/30".
        if any(connector.strip() in {"/", ":"} for connector in connectors):
            return True

        return bool(
            self._READ_MARKER.search(evidence)
            and self._WRITE_MARKER.search(evidence)
        )

    # -----------------------------------------------------------------
    # Text/span helpers
    # -----------------------------------------------------------------
    @classmethod
    def _fold(cls, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text or "")
        without_marks = "".join(
            ch for ch in normalized if not unicodedata.combining(ch)
        )
        return " ".join(without_marks.casefold().split())

    @staticmethod
    def _ordered_unique_fields(entries: Sequence[_CandidateEntry]) -> List[ParamName]:
        out: List[ParamName] = []
        seen = set()
        for entry in entries:
            field_name = entry.link.field
            if field_name is None or field_name in seen:
                continue
            out.append(field_name)
            seen.add(field_name)
        return out

    @staticmethod
    def _ordered_unique_ids(quantity_ids: Sequence[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for quantity_id in quantity_ids:
            if quantity_id in seen:
                continue
            out.append(quantity_id)
            seen.add(quantity_id)
        return out

    @staticmethod
    def _all_values_equivalent(values: Sequence[object]) -> bool:
        if not values:
            return True
        first = values[0]
        for value in values[1:]:
            try:
                if float(value) != float(first):
                    return False
            except (TypeError, ValueError):
                if value != first:
                    return False
        return True

    @staticmethod
    def _local_structure(
        text: str,
        quantities: Sequence[Quantity],
    ) -> Tuple[str, List[str], str]:
        ordered = sorted(quantities, key=lambda item: (item.start, item.end))
        if not ordered:
            return "", [], ""

        first = ordered[0]
        last = ordered[-1]
        prefix = text[max(0, first.start - 100): first.start]
        suffix = text[last.end: min(len(text), last.end + 100)]

        connectors: List[str] = []
        for left, right in zip(ordered, ordered[1:]):
            connectors.append(text[left.end: right.start])

        return prefix, connectors, suffix

    @staticmethod
    def _group_evidence(text: str, quantities: Sequence[Quantity]) -> str:
        if not quantities:
            return ""
        ordered = sorted(quantities, key=lambda item: (item.start, item.end))
        start = max(0, ordered[0].start - 80)
        end = min(len(text), ordered[-1].end + 80)
        return text[start:end].strip()

    @staticmethod
    def _blocking_reason(relation: CandidateRelationType) -> str:
        return {
            CandidateRelationType.ALTERNATIVE:
                "alternative_values_require_user_choice",
            CandidateRelationType.RANGE:
                "range_cannot_be_written_into_current_scalar_contract",
            CandidateRelationType.COMPARISON:
                "comparison_does_not_define_one_authoritative_requirement_value",
            CandidateRelationType.CONFLICT:
                "multiple_distinct_values_for_same_field_without_resolution",
            CandidateRelationType.UNRESOLVED:
                "candidate_relation_could_not_be_resolved_safely",
        }.get(relation, "relation_blocks_automatic_acceptance")

    @staticmethod
    def _non_blocking_reason(
        relation: CandidateRelationType,
        field_name: ParamName,
    ) -> str:
        if (
            relation == CandidateRelationType.SINGLE_VALUE
            and field_name == ParamName.read_write_ratio
        ):
            return "ratio_components_form_one_structured_requirement"
        if relation == CandidateRelationType.SINGLE_VALUE:
            return "field_has_one_authoritative_interpretation"
        if relation == CandidateRelationType.MULTIPLE_FIELDS:
            return "candidates_belong_to_distinct_fields"
        return "relation_is_non_blocking"
