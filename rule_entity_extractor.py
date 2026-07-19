from __future__ import annotations

import re
from typing import List

from closed_vocabulary_mapper import map_access_type, map_ha_required
from models import CandidateSource, ExtractedCandidate, ExtractionResult, ParamName
from text_preprocessor import normalize_text, normalized_for_matching
from unit_normalizer import normalize_unit_value


def _number_pattern() -> str:
    return r"-?\d+(?:\.\d+)?"


def _to_number(value: str):
    number = float(value)

    if number.is_integer():
        return int(number)

    return number


def _make_candidate(
    field: ParamName,
    value,
    unit,
    evidence: str,
    turn_id: int,
    confidence: float = 1.0,
) -> ExtractedCandidate:
    normalized_value, normalized_unit = normalize_unit_value(
        field=field,
        value=value,
        unit=unit,
    )

    return ExtractedCandidate(
        field=field,
        value=normalized_value,
        unit=normalized_unit,
        evidence=evidence.strip(),
        confidence=confidence,
        source=CandidateSource.RULE,
        source_text=evidence.strip(),
        turn_id=turn_id,
    )


class RuleBasedEntityExtractor:
    """
    Extracteur principal rule-first.

    Il extrait les paramètres explicites sans appeler un LLM.
    """

    def extract(
        self,
        user_text: str,
        turn_id: int,
    ) -> ExtractionResult:
        original = normalize_text(user_text)
        text = normalized_for_matching(original)

        candidates: List[ExtractedCandidate] = []
        warnings: List[str] = []

        candidates.extend(self._extract_capacity(original, text, turn_id))
        candidates.extend(self._extract_clients(original, text, turn_id))
        candidates.extend(self._extract_file_sizes(original, text, turn_id))
        candidates.extend(self._extract_file_count(original, text, turn_id))
        candidates.extend(self._extract_ratio(original, text, turn_id))
        candidates.extend(self._extract_access_type(original, turn_id))
        candidates.extend(self._extract_throughput(original, text, turn_id))
        candidates.extend(self._extract_ha(original, turn_id))
        candidates.extend(self._extract_budget(original, text, turn_id))
        candidates.extend(self._extract_power(original, text, turn_id))
        candidates.extend(self._extract_growth(original, text, turn_id))

        return ExtractionResult(
            candidates=candidates,
            warnings=warnings,
        )

    def _extract_capacity(
        self,
        original: str,
        text: str,
        turn_id: int,
    ) -> List[ExtractedCandidate]:
        out: List[ExtractedCandidate] = []
        number = _number_pattern()

        patterns = [
            rf"(?P<value>{number})\s*(?P<unit>tib|tb)\s*(?:utiles?|usable|capacite|capacity|stockage|storage)?",
            rf"(?:capacite|capacity|stockage|storage|besoin|need|we need|nous voulons|je veux)\D{{0,30}}(?P<value>{number})\s*(?P<unit>tib|tb)",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                evidence = original[match.start():match.end()]

                out.append(
                    _make_candidate(
                        field=ParamName.requested_usable_capacity_tib,
                        value=_to_number(match.group("value")),
                        unit=match.group("unit"),
                        evidence=evidence,
                        turn_id=turn_id,
                    )
                )

        return out

    def _extract_clients(
        self,
        original: str,
        text: str,
        turn_id: int,
    ) -> List[ExtractedCandidate]:
        out: List[ExtractedCandidate] = []
        number = _number_pattern()

        patterns = [
            rf"(?P<value>{number})\s*(?:clients?|noeuds?|nodes?|compute nodes?)\b",

            rf"(?:nombre\s+de\s+)?(?:clients?|noeuds?|nodes?|compute nodes?)\s*(?:=|:|est|sont|a|à)?\s*(?P<value>{number})\b",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                evidence = original[match.start():match.end()]
                value = _to_number(match.group("value"))

                out.append(
                    _make_candidate(
                        field=ParamName.client_count,
                        value=value,
                        unit=None,
                        evidence=evidence,
                        turn_id=turn_id,
                    )
                )

        return out

    def _extract_file_sizes(
        self,
        original: str,
        text: str,
        turn_id: int,
    ) -> List[ExtractedCandidate]:
        out: List[ExtractedCandidate] = []

        number = _number_pattern()
        unit = r"mb|mib|gb|gib"

        avg_patterns = [
            # fichiers moyens 2 GB
            # taille moyenne 2 GB
            # average file size 2 GB
            rf"(?:fichiers?\s*)?(?:moyens?|moyenne|taille moyenne|average file size|avg file size|average size|avg size|average|avg)\D{{0,25}}(?P<value>{number})\s*(?P<unit>{unit})\b",

            # Les fichiers font en moyenne 1.5 GB
            rf"(?:fichiers?|files?)\D{{0,25}}(?:moyenne|average|avg|en moyenne)\D{{0,25}}(?P<value>{number})\s*(?P<unit>{unit})\b",

            # 2 GB moyens
            rf"(?P<value>{number})\s*(?P<unit>{unit})\s*(?:moyens?|moyenne|average|avg)\b",
        ]

        max_patterns = [
            # taille max 100 GB
            # taille maximale 100 GB
            # max file size 100 GB
            # maximum file size 100 GB
            rf"(?:taille\s*)?(?:max|maximum|maximale|taille max|taille maximale|max file size|maximum file size|largest files?|biggest files?)\D{{0,25}}(?P<value>{number})\s*(?P<unit>{unit})\b",

            # les plus gros fichiers environ 80 GB
            # plus gros fichier 80 GB
            rf"(?:plus gros fichiers?|gros fichiers?|largest files?|biggest files?)\D{{0,35}}(?P<value>{number})\s*(?P<unit>{unit})\b",

            # 100 GB max
            rf"(?P<value>{number})\s*(?P<unit>{unit})\s*(?:max|maximum|maximale)\b",
        ]

        for pattern in avg_patterns:
            for match in re.finditer(pattern, text):
                evidence = original[match.start():match.end()]

                out.append(
                    _make_candidate(
                        field=ParamName.average_file_size_gb,
                        value=_to_number(match.group("value")),
                        unit=match.group("unit"),
                        evidence=evidence,
                        turn_id=turn_id,
                    )
                )

        for pattern in max_patterns:
            for match in re.finditer(pattern, text):
                evidence = original[match.start():match.end()]

                out.append(
                    _make_candidate(
                        field=ParamName.max_file_size_gb,
                        value=_to_number(match.group("value")),
                        unit=match.group("unit"),
                        evidence=evidence,
                        turn_id=turn_id,
                    )
                )

        return out

    def _extract_file_count(
        self,
        original: str,
        text: str,
        turn_id: int,
    ) -> List[ExtractedCandidate]:
        out: List[ExtractedCandidate] = []
        number = _number_pattern()

        patterns = [
            # 10 millions de fichiers
            # 2.5 millions de fichiers
            rf"(?P<value>{number})\s*(?P<scale>millions?|milliards?|billion|billions)\s*(?:de\s*)?(?:fichiers?|files?)\b",

            # 500000 fichiers
            rf"(?P<value>{number})\s*(?:fichiers?|files?)\b",

            # nombre de fichiers : 500000
            # total fichiers = 2 millions
            rf"(?:total|nombre|count|number)\s*(?:de\s*)?(?:fichiers?|files?)\D{{0,25}}(?P<value>{number})\s*(?P<scale>millions?|milliards?|billion|billions)?",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                evidence = original[match.start():match.end()]
                value = _to_number(match.group("value"))

                scale = match.groupdict().get("scale")
                multiplier = 1

                if scale:
                    if "milliard" in scale or "billion" in scale:
                        multiplier = 1_000_000_000
                    elif "million" in scale:
                        multiplier = 1_000_000

                out.append(
                    _make_candidate(
                        field=ParamName.total_file_count,
                        value=value * multiplier,
                        unit=None,
                        evidence=evidence,
                        turn_id=turn_id,
                    )
                )

        return out

    def _extract_ratio(
        self,
        original: str,
        text: str,
        turn_id: int,
    ) -> List[ExtractedCandidate]:
        out: List[ExtractedCandidate] = []

        for match in re.finditer(
            r"(?:ratio\s*)?(\d{1,3})\s*/\s*(\d{1,3})",
            text,
        ):
            evidence = original[match.start():match.end()]

            out.append(
                ExtractedCandidate(
                    field=ParamName.read_write_ratio,
                    value={
                        "read_percent": int(match.group(1)),
                        "write_percent": int(match.group(2)),
                    },
                    unit="%",
                    evidence=evidence.strip(),
                    confidence=1.0,
                    source=CandidateSource.RULE,
                    source_text=evidence.strip(),
                    turn_id=turn_id,
                )
            )

        pattern = (
            r"(\d{1,3})\s*%?\s*(?:lecture|read)"
            r"\D+(\d{1,3})\s*%?\s*(?:ecriture|write)"
        )

        for match in re.finditer(pattern, text):
            evidence = original[match.start():match.end()]

            out.append(
                ExtractedCandidate(
                    field=ParamName.read_write_ratio,
                    value={
                        "read_percent": int(match.group(1)),
                        "write_percent": int(match.group(2)),
                    },
                    unit="%",
                    evidence=evidence.strip(),
                    confidence=1.0,
                    source=CandidateSource.RULE,
                    source_text=evidence.strip(),
                    turn_id=turn_id,
                )
            )

        return out

    def _extract_access_type(
        self,
        original: str,
        turn_id: int,
    ) -> List[ExtractedCandidate]:
        access_type = map_access_type(original)

        if access_type is None:
            return []

        return [
            ExtractedCandidate(
                field=ParamName.access_type,
                value=access_type,
                unit=None,
                evidence=original,
                confidence=0.95,
                source=CandidateSource.RULE,
                source_text=original,
                turn_id=turn_id,
            )
        ]

    def _extract_throughput(
        self,
        original: str,
        text: str,
        turn_id: int,
    ) -> List[ExtractedCandidate]:
        out: List[ExtractedCandidate] = []
        number = _number_pattern()

        seen = set()

        def add_candidate(
            field: ParamName,
            value,
            unit: str,
            evidence: str,
        ) -> None:
            key = (
                field.value,
                str(value),
                unit.lower(),
                evidence.strip().lower(),
            )

            if key in seen:
                return

            seen.add(key)

            out.append(
                _make_candidate(
                    field=field,
                    value=value,
                    unit=unit,
                    evidence=evidence,
                    turn_id=turn_id,
                )
            )

        # ============================================================
        # Forme 1 : nombre AVANT le type de débit
        # Exemples :
        # - 55 GB/s en lecture
        # - 25 GB/s en écriture
        # - 50 GB/s read
        # - 20 GB/s write
        # ============================================================

        read_value_before_patterns = [
            rf"(?P<value>{number})\s*(?P<unit>gb/s)\s*(?:en|de|pour)?\s*(?:lecture|read)\b",
        ]

        write_value_before_patterns = [
            rf"(?P<value>{number})\s*(?P<unit>gb/s)\s*(?:en|de|pour)?\s*(?:ecriture|write)\b",
        ]

        for pattern in read_value_before_patterns:
            for match in re.finditer(pattern, text):
                evidence = original[match.start():match.end()]

                add_candidate(
                    field=ParamName.target_read_gbps,
                    value=_to_number(match.group("value")),
                    unit="GB/s",
                    evidence=evidence,
                )

        for pattern in write_value_before_patterns:
            for match in re.finditer(pattern, text):
                evidence = original[match.start():match.end()]

                add_candidate(
                    field=ParamName.target_write_gbps,
                    value=_to_number(match.group("value")),
                    unit="GB/s",
                    evidence=evidence,
                )

        # ============================================================
        # Forme 2 : type de débit AVANT le nombre
        # Exemples :
        # - lecture 55 GB/s
        # - écriture 25 GB/s
        # - target read 50 GB/s
        # - target write 20 GB/s
        # - read throughput 50 GB/s
        # - write throughput 20 GB/s
        #
        # Important :
        # On évite les regex trop larges comme :
        # lecture\D{0,25}25 GB/s
        # car elles capturent à tort :
        # "55 GB/s en lecture et 25 GB/s en écriture"
        # ============================================================

        read_label_before_patterns = [
            rf"(?:lecture|read)\s*(?:=|:)?\s*(?P<value>{number})\s*(?P<unit>gb/s)\b",
            rf"(?:target read|read target|read throughput|debit lecture|debit de lecture|lecture cible)\s*(?:=|:)?\s*(?P<value>{number})\s*(?P<unit>gb/s)\b",
        ]

        write_label_before_patterns = [
            rf"(?:ecriture|write)\s*(?:=|:)?\s*(?P<value>{number})\s*(?P<unit>gb/s)\b",
            rf"(?:target write|write target|write throughput|debit ecriture|debit d ecriture|ecriture cible)\s*(?:=|:)?\s*(?P<value>{number})\s*(?P<unit>gb/s)\b",
        ]

        for pattern in read_label_before_patterns:
            for match in re.finditer(pattern, text):
                evidence = original[match.start():match.end()]

                add_candidate(
                    field=ParamName.target_read_gbps,
                    value=_to_number(match.group("value")),
                    unit="GB/s",
                    evidence=evidence,
                )

        for pattern in write_label_before_patterns:
            for match in re.finditer(pattern, text):
                evidence = original[match.start():match.end()]

                add_candidate(
                    field=ParamName.target_write_gbps,
                    value=_to_number(match.group("value")),
                    unit="GB/s",
                    evidence=evidence,
                )

        return out

    def _extract_ha(
        self,
        original: str,
        turn_id: int,
    ) -> List[ExtractedCandidate]:
        value = map_ha_required(original)

        if value is None:
            return []

        return [
            ExtractedCandidate(
                field=ParamName.ha_required,
                value=value,
                unit=None,
                evidence=original,
                confidence=0.90,
                source=CandidateSource.RULE,
                source_text=original,
                turn_id=turn_id,
            )
        ]

    def _extract_budget(
        self,
        original: str,
        text: str,
        turn_id: int,
    ) -> List[ExtractedCandidate]:
        out: List[ExtractedCandidate] = []
        number = _number_pattern()

        patterns = [
            rf"(?:budget|budget ideal|budget ideal|budget max|budget maximum|max budget|maximum budget|ne pas depasser)\D{{0,30}}(\$?\s*{number})\s*(usd|dollars?)?",
            rf"(?:maximum|max|limite)\D{{0,20}}({number})\s*(usd|dollars?)",
            rf"(\$)\s*({number})",
            rf"({number})\s*(usd|dollars?)",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                evidence = original[match.start():match.end()]
                numeric = re.search(number, match.group(0))

                if numeric:
                    out.append(
                        _make_candidate(
                            field=ParamName.max_budget_usd,
                            value=_to_number(numeric.group(0)),
                            unit="USD",
                            evidence=evidence,
                            turn_id=turn_id,
                        )
                    )

        return out

    def _extract_power(
        self,
        original: str,
        text: str,
        turn_id: int,
    ) -> List[ExtractedCandidate]:
        out: List[ExtractedCandidate] = []
        number = _number_pattern()

        patterns = [
            rf"(?:puissance|power|power limit|max power|puissance max|puissance maximum|puissance ideale|maximum|limite)\D{{0,30}}({number})\s*(kw|w|watts?|kilowatts?)",
            rf"({number})\s*(kw|w|watts?|kilowatts?)",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                evidence = original[match.start():match.end()]

                out.append(
                    _make_candidate(
                        field=ParamName.max_power_w,
                        value=_to_number(match.group(1)),
                        unit=match.group(2),
                        evidence=evidence,
                        turn_id=turn_id,
                    )
                )

        return out

    def _extract_growth(
        self,
        original: str,
        text: str,
        turn_id: int,
    ) -> List[ExtractedCandidate]:
        out: List[ExtractedCandidate] = []
        number = _number_pattern()

        patterns = [
            rf"(?:croissance|growth|annual growth|croissance annuelle)\D{{0,25}}({number})\s*%",
            rf"({number})\s*%\D{{0,25}}(?:croissance|growth|annual)",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                evidence = original[match.start():match.end()]

                out.append(
                    _make_candidate(
                        field=ParamName.annual_growth_percent,
                        value=_to_number(match.group(1)),
                        unit="%",
                        evidence=evidence,
                        turn_id=turn_id,
                    )
                )

        return out