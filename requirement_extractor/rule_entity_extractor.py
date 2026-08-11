from __future__ import annotations

import re
from typing import Any, Iterable, List, Optional

from .closed_vocabulary_mapper import (
    detect_access_types,
    detect_ha_values,
)
from .models import (
    CandidateSource,
    ExtractedCandidate,
    ExtractionResult,
    ParamName,
)
from .text_preprocessor import normalize_text, normalized_for_matching
from .unit_normalizer import normalize_unit_value


def _number_pattern() -> str:
    """
    Nombre signé compatible avec :
    - 500
    - -500
    - 12.5
    - 12,5
    - 10 000 000
    - 100_000
    """

    return (
        r"[+-]?"
        r"(?:"
        r"\d{1,3}(?:[ _]\d{3})+"
        r"|"
        r"\d+"
        r")"
        r"(?:[\.,]\d+)?"
    )


def _to_number(value: str) -> int | float:
    cleaned = (
        value.strip()
        .replace(" ", "")
        .replace("_", "")
        .replace(",", ".")
    )

    number = float(cleaned)

    if number.is_integer():
        return int(number)

    return number


def _scale_multiplier(scale: Optional[str]) -> int:
    if not scale:
        return 1

    normalized = scale.lower().strip().rstrip(".")

    if normalized in {
        "k",
        "thousand",
        "thousands",
        "mille",
        "milles",
    }:
        return 1_000

    if normalized in {
        "m",
        "million",
        "millions",
    }:
        return 1_000_000

    if normalized in {
        "b",
        "billion",
        "billions",
        "milliard",
        "milliards",
    }:
        return 1_000_000_000

    return 1


def _scaled_number(
    value: str,
    scale: Optional[str] = None,
) -> int | float:
    parsed = _to_number(value)
    result = float(parsed) * _scale_multiplier(scale)

    if result.is_integer():
        return int(result)

    return result


def _canonical_capacity_unit(unit: str) -> str:
    normalized = unit.lower()

    if normalized in {
        "tib",
        "tebioctet",
        "tebioctets",
    }:
        return "TiB"

    return "TB"


def _canonical_size_unit(unit: str) -> str:
    normalized = unit.lower()

    if normalized in {"mb", "mib"}:
        return "MB" if normalized == "mb" else "MiB"

    return "GB" if normalized in {"gb", "giga", "gigas", "gigabyte", "gigabytes"} else "GiB"


def _candidate_value_key(value: Any) -> str:
    if isinstance(value, dict):
        return repr(sorted(value.items()))

    return f"{type(value).__name__}:{value!r}"


def _deduplicate_candidates(
    candidates: Iterable[ExtractedCandidate],
) -> List[ExtractedCandidate]:
    """
    Supprime uniquement les duplications exactes de valeur.

    Deux valeurs différentes du même champ sont conservées afin que
    StateGuard puisse déclarer un conflit.
    """

    output: List[ExtractedCandidate] = []
    seen = set()

    for candidate in candidates:
        key = (
            candidate.field.value,
            _candidate_value_key(candidate.value),
            str(candidate.unit).lower(),
        )

        if key in seen:
            continue

        seen.add(key)
        output.append(candidate)

    return output


def _make_candidate(
    field: ParamName,
    value: Any,
    unit: Optional[str],
    evidence: str,
    turn_id: int,
    confidence: float = 1.0,
) -> ExtractedCandidate:
    """
    Crée un candidat canonique.

    Les champs entiers restent non arrondis avant StateGuard. Ainsi,
    ``12.5 clients`` reste 12.5 et peut être rejeté comme valeur non entière,
    au lieu d'être silencieusement transformé en 12.
    """

    if field in {
        ParamName.client_count,
        ParamName.total_file_count,
    }:
        normalized_value = value
        normalized_unit = None
    else:
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
    Extracteur déterministe rule-first.

    Il couvre les structures fiables :
    - libellé + valeur ;
    - valeur + unité + libellé ;
    - unités et multiplicateurs ;
    - listes de valeurs contradictoires ;
    - négations explicites.

    Les formulations ouvertes ou fortement bruitées restent destinées au
    fallback LLM.
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
        candidates.extend(self._extract_ha(original, text, turn_id))
        candidates.extend(self._extract_budget(original, text, turn_id))
        candidates.extend(self._extract_power(original, text, turn_id))
        candidates.extend(self._extract_growth(original, text, turn_id))

        return ExtractionResult(
            candidates=_deduplicate_candidates(candidates),
            warnings=warnings,
        )

    # =================================================================
    # CAPACITY
    # =================================================================

    def _extract_capacity(
        self,
        original: str,
        text: str,
        turn_id: int,
    ) -> List[ExtractedCandidate]:
        output: List[ExtractedCandidate] = []
        number = _number_pattern()
        unit = r"tib|tb|tebioctets?|terabytes?"

        patterns = [
            rf"(?P<value>{number})\s*(?P<unit>{unit})\b"
            rf"(?:\s*(?:utiles?|usable|capacity|capacite|storage|stockage))?",

            rf"(?:capacite(?:\s+utile)?|capacity|usable\s+storage|"
            rf"stockage|storage|besoin|need|we\s+need|je\s+veux|"
            rf"nous\s+voulons)[^0-9+\-]{{0,35}}"
            rf"(?P<value>{number})\s*(?P<unit>{unit})\b",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                evidence = original[match.start():match.end()]

                output.append(
                    _make_candidate(
                        field=ParamName.requested_usable_capacity_tib,
                        value=_to_number(match.group("value")),
                        unit=_canonical_capacity_unit(match.group("unit")),
                        evidence=evidence,
                        turn_id=turn_id,
                    )
                )

        return output

    # =================================================================
    # CLIENTS
    # =================================================================

    def _extract_clients(
        self,
        original: str,
        text: str,
        turn_id: int,
    ) -> List[ExtractedCandidate]:
        output: List[ExtractedCandidate] = []
        number = _number_pattern()

        subject = (
            r"clients?|clientes?|"
            r"n(?:oe|œ)uds?|nodes?|"
            r"compute(?:\s+(?:nodes?|clients?))?"
        )

        patterns = [
            rf"(?P<value>{number})\s*(?:{subject})\b",

            rf"(?:nombre\s+de\s+|number\s+of\s+)?(?:{subject})\s*"
            rf"(?:=|:|est|sont|a|à|is|are)?\s*"
            rf"(?P<value>{number})\b",
        ]

        # Deux valeurs partageant le même libellé :
        # "200 et 400 clients".
        shared_label_pattern = (
            rf"(?P<first>{number})\s*(?:et|and|ou|or|,)\s*"
            rf"(?P<second>{number})\s*(?:{subject})\b"
        )

        for match in re.finditer(shared_label_pattern, text):
            evidence = original[match.start():match.end()]

            for group_name in ("first", "second"):
                output.append(
                    _make_candidate(
                        field=ParamName.client_count,
                        value=_to_number(match.group(group_name)),
                        unit=None,
                        evidence=evidence,
                        turn_id=turn_id,
                    )
                )

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                evidence = original[match.start():match.end()]

                output.append(
                    _make_candidate(
                        field=ParamName.client_count,
                        value=_to_number(match.group("value")),
                        unit=None,
                        evidence=evidence,
                        turn_id=turn_id,
                    )
                )

        return output

    # =================================================================
    # FILE SIZES
    # =================================================================

    def _extract_file_sizes(
        self,
        original: str,
        text: str,
        turn_id: int,
    ) -> List[ExtractedCandidate]:
        output: List[ExtractedCandidate] = []
        number = _number_pattern()
        unit = (
            r"mb|mib|gb|gib|"
            r"gigas?|gigabytes?"
        )

        average_labels = (
            r"fichiers?\s+moyens?|fichier\s+moyen|"
            r"taille\s+moyenne|moyenne(?:\s+fichier)?|"
            r"average[-\s]+file(?:[-\s]+size)?|"
            r"avg[-\s]+file(?:[-\s]+size)?|"
            r"average[-\s]+files?|avg[-\s]+files?|"
            r"average|avg|"
            r"archivos?\s+medios?"
        )

        maximum_labels = (
            r"taille\s+max(?:imale)?|"
            r"max(?:imum)?[-\s]+file(?:[-\s]+size)?|"
            r"maximum|"
            r"plus\s+gros\s+fichiers?|"
            r"largest\s+files?|biggest\s+files?|"
            r"max(?:imo)?"
        )

        average_patterns = [
            rf"(?:{average_labels})[^0-9+\-]{{0,25}}"
            rf"(?P<value>{number})\s*(?P<unit>{unit})\b",

            rf"(?P<value>{number})\s*(?P<unit>{unit})\s*"
            rf"(?:moyens?|moyenne|average|avg)\b",
        ]

        maximum_patterns = [
            rf"(?:{maximum_labels})[^0-9+\-]{{0,25}}"
            rf"(?P<value>{number})\s*(?P<unit>{unit})\b",

            # Reverse-order notation is valid in "20 GB max", but must not
            # steal an average value when "max" introduces the NEXT value:
            #
            #   "avg 0.5 GB max 20 GB"
            #
            # In that phrase, 0.5 GB is the average and 20 GB is the max.
            rf"(?P<value>{number})\s*(?P<unit>{unit})\s*"
            rf"(?:max|maximum|maximale)\b"
            rf"(?!\s*[+\-]?\d)",
        ]

        for pattern in average_patterns:
            for match in re.finditer(pattern, text):
                evidence = original[match.start():match.end()]

                output.append(
                    _make_candidate(
                        field=ParamName.average_file_size_gb,
                        value=_to_number(match.group("value")),
                        unit=_canonical_size_unit(match.group("unit")),
                        evidence=evidence,
                        turn_id=turn_id,
                    )
                )

        for pattern in maximum_patterns:
            for match in re.finditer(pattern, text):
                evidence = original[match.start():match.end()]

                output.append(
                    _make_candidate(
                        field=ParamName.max_file_size_gb,
                        value=_to_number(match.group("value")),
                        unit=_canonical_size_unit(match.group("unit")),
                        evidence=evidence,
                        turn_id=turn_id,
                    )
                )

        # Borne haute d'un intervalle :
        # "entre 1 et 100 GB" / "between 1 and 100 GB".
        range_pattern = (
            rf"(?:entre|between)\s*"
            rf"(?P<low>{number})\s*(?P<low_unit>{unit})?\s*"
            rf"(?:et|and|to|-)\s*"
            rf"(?P<high>{number})\s*(?P<high_unit>{unit})\b"
        )

        for match in re.finditer(range_pattern, text):
            evidence = original[match.start():match.end()]
            selected_unit = (
                match.group("high_unit")
                or match.group("low_unit")
                or "GB"
            )

            output.append(
                _make_candidate(
                    field=ParamName.max_file_size_gb,
                    value=_to_number(match.group("high")),
                    unit=_canonical_size_unit(selected_unit),
                    evidence=evidence,
                    turn_id=turn_id,
                )
            )

        return output

    # =================================================================
    # FILE COUNT
    # =================================================================

    def _extract_file_count(
        self,
        original: str,
        text: str,
        turn_id: int,
    ) -> List[ExtractedCandidate]:
        output: List[ExtractedCandidate] = []
        number = _number_pattern()
        scale = (
            r"k|m|b|"
            r"millions?|milliards?|"
            r"billion|billions|"
            r"thousand|thousands"
        )
        file_label = r"fichiers?|files?|archivos?"

        patterns = [
            # Valeur suivie explicitement du nom de l'entité :
            # "250000 fichiers", "5 million files".
            rf"(?P<value>{number})\s*(?P<scale>{scale})?\s*"
            rf"(?:de\s+)?(?:{file_label})\b",

            # Libellé de comptage explicite avant la valeur.
            # Le mot isolé "fichier" n'est pas utilisé ici afin de ne pas
            # confondre une taille de fichier avec un nombre de fichiers.
            rf"(?:total(?:\s+(?:de\s+)?(?:{file_label}))?|"
            rf"nombre\s+(?:de\s+)?(?:{file_label})|"
            rf"(?:file|files)\s+count|"
            rf"number\s+of\s+(?:{file_label})|"
            rf"count\s+of\s+(?:{file_label}))"
            rf"\s*(?:=|:|est|is|de)?\s*"
            rf"(?P<value>{number})\s*(?P<scale>{scale})?"
            rf"(?:\s*(?:de\s+)?(?:{file_label}))?",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                evidence = original[match.start():match.end()]

                output.append(
                    _make_candidate(
                        field=ParamName.total_file_count,
                        value=_scaled_number(
                            match.group("value"),
                            match.groupdict().get("scale"),
                        ),
                        unit=None,
                        evidence=evidence,
                        turn_id=turn_id,
                    )
                )

        # Inférence contextuelle limitée :
        # "moyenne 2 GB, max 100 GB, 10 millions"
        # Le multiplicateur seul est interprété comme un nombre de fichiers
        # uniquement si le message contient déjà un contexte de taille de
        # fichiers et si la valeur n'est pas suivie d'une devise.
        has_file_context = bool(
            re.search(
                r"\b(?:fichiers?|files?|moyenne|average|avg|"
                r"taille\s+max|max\s+file|largest)\b",
                text,
            )
        )

        if has_file_context:
            standalone_scale_pattern = (
                rf"(?P<value>{number})\s*"
                rf"(?P<scale>millions?|milliards?|billion|billions)\b"
                rf"(?!\s*(?:usd|dollars?|\$))"
            )

            for match in re.finditer(standalone_scale_pattern, text):
                evidence = original[match.start():match.end()]

                output.append(
                    _make_candidate(
                        field=ParamName.total_file_count,
                        value=_scaled_number(
                            match.group("value"),
                            match.group("scale"),
                        ),
                        unit=None,
                        evidence=evidence,
                        turn_id=turn_id,
                    )
                )

        return output

    # =================================================================
    # READ / WRITE RATIO
    # =================================================================

    def _extract_ratio(
        self,
        original: str,
        text: str,
        turn_id: int,
    ) -> List[ExtractedCandidate]:
        output: List[ExtractedCandidate] = []

        def add_ratio(match: re.Match[str]) -> None:
            read_value = int(match.group("read"))
            write_value = int(match.group("write"))
            evidence = original[match.start():match.end()]

            output.append(
                ExtractedCandidate(
                    field=ParamName.read_write_ratio,
                    value={
                        "read_percent": read_value,
                        "write_percent": write_value,
                    },
                    unit="%",
                    evidence=evidence.strip(),
                    confidence=1.0,
                    source=CandidateSource.RULE,
                    source_text=evidence.strip(),
                    turn_id=turn_id,
                )
            )

        patterns = [
            # 70/30 ou 70:30, avec ou sans préfixe.
            (
                r"(?:(?:ratio|read\s*/\s*write|lecture\s*/\s*ecriture|"
                r"r\s*[/\:]\s*w)\s*)?"
                r"(?P<read>\d{1,3})\s*[/\:]\s*"
                r"(?P<write>\d{1,3})"
            ),

            # 70% lecture et 30% écriture.
            (
                r"(?P<read>\d{1,3})\s*%?\s*"
                r"(?:lecture|read|r)\b"
                r"\D{0,20}"
                r"(?P<write>\d{1,3})\s*%?\s*"
                r"(?:ecriture|write|w)\b"
            ),

            # read 70 / write 30.
            (
                r"(?:lecture|read|r)\s*"
                r"(?P<read>\d{1,3})\s*%?"
                r"\s*(?:/|:|et|and)\s*"
                r"(?:ecriture|write|w)\s*"
                r"(?P<write>\d{1,3})\s*%?"
            ),

            # 70%r 30%w.
            (
                r"(?P<read>\d{1,3})\s*%\s*r\b"
                r"\D{0,10}"
                r"(?P<write>\d{1,3})\s*%\s*w\b"
            ),

            # ratio 70 30.
            (
                r"\bratio\s*"
                r"(?P<read>\d{1,3})\s+"
                r"(?P<write>\d{1,3})\b"
            ),
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                add_ratio(match)

        return output

    # =================================================================
    # CLOSED VOCABULARY
    # =================================================================

    def _extract_access_type(
        self,
        original: str,
        turn_id: int,
    ) -> List[ExtractedCandidate]:
        output: List[ExtractedCandidate] = []

        for access_type in detect_access_types(original):
            output.append(
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
            )

        return output

    def _extract_ha(
        self,
        original: str,
        text: str,
        turn_id: int,
    ) -> List[ExtractedCandidate]:
        values = list(detect_ha_values(original))

        # Contradiction booléenne explicite dans une même proposition :
        # "HA oui et non" / "HA yes and no".
        ha_window = re.search(
            r"\bha\b.{0,40}",
            text,
        )

        if ha_window:
            fragment = ha_window.group(0)

            has_positive = bool(
                re.search(r"\b(?:oui|yes|true|on)\b", fragment)
            )
            has_negative = bool(
                re.search(r"\b(?:non|no|false|off)\b", fragment)
            )

            if has_positive and True not in values:
                values.append(True)

            if has_negative and False not in values:
                values.append(False)

        output: List[ExtractedCandidate] = []

        for value in values:
            output.append(
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
            )

        return output

    # =================================================================
    # THROUGHPUT
    # =================================================================

    def _extract_throughput(
        self,
        original: str,
        text: str,
        turn_id: int,
    ) -> List[ExtractedCandidate]:
        output: List[ExtractedCandidate] = []
        number = _number_pattern()
        unit = (
            r"gb/s|gbps|gbs|"
            r"gigas?/s|gigabytes?/s"
        )

        read_labels = (
            r"lecture|read|"
            r"target\s+read|read\s+target|read\s+throughput|"
            r"debit\s+(?:de\s+)?lecture|lecture\s+cible|"
            r"lectura"
        )

        write_labels = (
            r"ecriture|write|"
            r"target\s+write|write\s+target|write\s+throughput|"
            r"debit\s+(?:d\s+)?ecriture|ecriture\s+cible|"
            r"escritura"
        )

        def extract_for_field(
            field: ParamName,
            labels: str,
        ) -> None:
            # Libellé avant la valeur. L'unité peut être omise car le libellé
            # indique explicitement un objectif de débit.
            label_before = (
                rf"(?:{labels})\s*"
                rf"(?:=|:|cible|target|around|about|environ|maybe|~)?\s*"
                rf"(?P<value>{number})\s*"
                rf"(?P<unit>{unit})?"
            )

            for match in re.finditer(label_before, text):
                evidence = original[match.start():match.end()]
                raw_unit = match.group("unit")

                # Une valeur sans unité immédiatement suivie de / ou %
                # appartient probablement à un ratio.
                tail = text[match.end():match.end() + 3]

                if raw_unit is None and re.match(r"\s*[/:%]", tail):
                    continue

                output.append(
                    _make_candidate(
                        field=field,
                        value=_to_number(match.group("value")),
                        unit="GB/s",
                        evidence=evidence,
                        turn_id=turn_id,
                    )
                )

            # Valeur avant le libellé : unité obligatoire pour éviter de
            # confondre "70 read / 30 write" avec des débits.
            value_before = (
                rf"(?P<value>{number})\s*"
                rf"(?P<unit>{unit})\s*"
                rf"(?:en|de|pour)?\s*(?:{labels})\b"
            )

            for match in re.finditer(value_before, text):
                evidence = original[match.start():match.end()]

                output.append(
                    _make_candidate(
                        field=field,
                        value=_to_number(match.group("value")),
                        unit="GB/s",
                        evidence=evidence,
                        turn_id=turn_id,
                    )
                )

        extract_for_field(
            ParamName.target_read_gbps,
            read_labels,
        )
        extract_for_field(
            ParamName.target_write_gbps,
            write_labels,
        )

        return output

    # =================================================================
    # BUDGET
    # =================================================================

    def _extract_budget(
        self,
        original: str,
        text: str,
        turn_id: int,
    ) -> List[ExtractedCandidate]:
        output: List[ExtractedCandidate] = []
        number = _number_pattern()
        scale = r"k|m|million|millions"
        currency = r"usd|dollars?"

        patterns = [
            (
                rf"(?:budget(?:\s+(?:ideal|prefere|max|maximum))?|"
                rf"max\s+budget|maximum\s+budget|budget\s+limit|"
                rf"presupuesto|ne\s+pas\s+depasser)"
                rf"[^0-9+\-]{{0,30}}"
                rf"\$?\s*(?P<value>{number})\s*"
                rf"(?P<scale>{scale})?\s*(?P<currency>{currency})?"
            ),

            (
                rf"\$\s*(?P<value>{number})\s*"
                rf"(?P<scale>{scale})?"
            ),

            (
                rf"(?P<value>{number})\s*"
                rf"(?P<scale>{scale})?\s*"
                rf"(?P<currency>{currency})\b"
            ),
        ]

        for pattern_index, pattern in enumerate(patterns):
            for match in re.finditer(pattern, text):
                # Guard against cross-field capture when the label appears
                # after a valid currency amount, e.g.:
                #
                #   "500000 USD budget, 30000 W max power"
                #
                # The broad label-driven pattern can otherwise capture
                # "budget, 30000" and invent a 30000 USD candidate.
                if pattern_index == 0:
                    following = text[match.end():match.end() + 24]

                    if re.match(
                        r"\s*(?:"
                        r"%|"
                        r"kw\b|w\b|watts?\b|kilowatts?\b|"
                        r"percent\b|pourcent\b|"
                        r"gb/s\b|gib/s\b|mb/s\b|mib/s\b|"
                        r"tb\b|tib\b|gb\b|gib\b"
                        r")",
                        following,
                        flags=re.IGNORECASE,
                    ):
                        continue

                evidence = original[match.start():match.end()]

                output.append(
                    _make_candidate(
                        field=ParamName.max_budget_usd,
                        value=_scaled_number(
                            match.group("value"),
                            match.groupdict().get("scale"),
                        ),
                        unit="USD",
                        evidence=evidence,
                        turn_id=turn_id,
                    )
                )

        return output

    # =================================================================
    # POWER
    # =================================================================

    def _extract_power(
        self,
        original: str,
        text: str,
        turn_id: int,
    ) -> List[ExtractedCandidate]:
        output: List[ExtractedCandidate] = []
        number = _number_pattern()
        unit = r"kw|w|watts?|kilowatts?"

        labels = (
            r"puissance(?:\s+(?:ideale|maximale|max))?|"
            r"power(?:\s+(?:limit|cap|max|maximum|preferred))?|"
            r"max\s+power|power\s+limit|power\s+cap|"
            r"potencia|pwr"
        )

        label_pattern = (
            rf"(?:{labels})[^0-9+\-]{{0,25}}"
            rf"(?P<value>{number})\s*(?P<unit>{unit})?"
        )

        for match in re.finditer(label_pattern, text):
            # A unitless contextual power match must not consume the value
            # of the next semantic field. Example:
            #
            #   "30000 W max power, 15% annual growth"
            #
            # The explicit-unit rule below already extracts 30000 W.
            raw_unit = match.group("unit")

            if raw_unit is None:
                following = text[match.end():match.end() + 24]

                if re.match(
                    r"\s*(?:"
                    r"%|\$|"
                    r"percent\b|pourcent\b|"
                    r"usd\b|dollars?\b|"
                    r"gb/s\b|gib/s\b|mb/s\b|mib/s\b|"
                    r"tb\b|tib\b|gb\b|gib\b"
                    r")",
                    following,
                    flags=re.IGNORECASE,
                ):
                    continue

            evidence = original[match.start():match.end()]
            raw_unit = raw_unit or "W"

            output.append(
                _make_candidate(
                    field=ParamName.max_power_w,
                    value=_to_number(match.group("value")),
                    unit=raw_unit,
                    evidence=evidence,
                    turn_id=turn_id,
                )
            )

        # Forme autonome avec unité explicite.
        #
        # Cette règle couvre notamment :
        # - "limite 15 kW" ;
        # - "15000 W" ;
        # - "power max: 5 kW".
        #
        # Un libellé générique comme "limite" n'est jamais suffisant sans
        # unité électrique. Cela empêche "Limites: 200000 USD, 15000 W"
        # de produire le faux candidat 200000 W.
        generic_pattern = (
            rf"(?P<value>{number})\s*(?P<unit>{unit})\b"
        )

        for match in re.finditer(generic_pattern, text):
            evidence = original[match.start():match.end()]

            output.append(
                _make_candidate(
                    field=ParamName.max_power_w,
                    value=_to_number(match.group("value")),
                    unit=match.group("unit"),
                    evidence=evidence,
                    turn_id=turn_id,
                )
            )

        return output

    # =================================================================
    # GROWTH
    # =================================================================

    def _extract_growth(
        self,
        original: str,
        text: str,
        turn_id: int,
    ) -> List[ExtractedCandidate]:
        output: List[ExtractedCandidate] = []
        number = _number_pattern()

        growth_label = (
            r"croissance(?:\s+annuelle)?|"
            r"annual\s+growth|yearly\s+growth|growth|"
            r"crecimiento"
        )

        label_before = (
            rf"(?:{growth_label})[^0-9+\-]{{0,20}}"
            rf"(?P<value>{number})\s*"
            rf"(?:%|pourcent|percent)?"
        )

        number_before = (
            rf"(?P<value>{number})\s*"
            rf"(?:%|pourcent|percent)[^0-9+\-]{{0,20}}"
            rf"(?:{growth_label})\b"
        )

        annual_suffix = (
            rf"(?P<value>{number})\s*%"
            rf"\s*(?:/|par\s+)?(?:an|annee|year)\b"
        )

        for pattern in (
            label_before,
            number_before,
            annual_suffix,
        ):
            for match in re.finditer(pattern, text):
                evidence = original[match.start():match.end()]

                output.append(
                    _make_candidate(
                        field=ParamName.annual_growth_percent,
                        value=_to_number(match.group("value")),
                        unit="%",
                        evidence=evidence,
                        turn_id=turn_id,
                    )
                )

        # Plusieurs pourcentages dans la proposition de croissance :
        # "croissance 20% puis 30%".
        for label_match in re.finditer(growth_label, text):
            clause_start = label_match.start()
            clause_end = len(text)

            punctuation = re.search(
                r"[.;]",
                text[label_match.end():],
            )

            if punctuation:
                clause_end = label_match.end() + punctuation.start()

            clause = text[clause_start:clause_end]

            for value_match in re.finditer(
                rf"(?P<value>{number})\s*"
                rf"(?:%|pourcent|percent)",
                clause,
            ):
                absolute_start = clause_start + value_match.start()
                absolute_end = clause_start + value_match.end()
                evidence = original[absolute_start:absolute_end]

                output.append(
                    _make_candidate(
                        field=ParamName.annual_growth_percent,
                        value=_to_number(value_match.group("value")),
                        unit="%",
                        evidence=evidence,
                        turn_id=turn_id,
                    )
                )

        return output