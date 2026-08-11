from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Optional, Tuple

from .models import ParamName


def is_number(value: Any) -> bool:
    """
    Vérifie qu'une valeur est numérique sans accepter les booléens.
    """

    return isinstance(value, (int, float, Decimal)) and not isinstance(
        value,
        bool,
    )


def _to_decimal(value: Any) -> Decimal:
    """
    Convertit une valeur numérique en Decimal sans perdre son signe.
    """

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"Valeur numérique invalide : {value!r}") from exc


def normalize_number(value: Any) -> int | float:
    """
    Produit un entier uniquement lorsque la valeur est réellement entière.

    Exemples :
    - 15.0  -> 15
    - -5.0  -> -5
    - 12.5  -> 12.5
    - -10.5 -> -10.5

    Aucune valeur décimale n'est tronquée ou arrondie vers un entier.
    """

    decimal_value = _to_decimal(value)

    if decimal_value == decimal_value.to_integral_value():
        return int(decimal_value)

    return round(float(decimal_value), 6)


def _normalize_unit(unit: Optional[str]) -> str:
    normalized = (unit or "").strip().lower()

    aliases = {
        "kilowatt": "kw",
        "kilowatts": "kw",
        "watt": "w",
        "watts": "w",
        "megawatt": "mw",
        "megawatts": "mw",
        "gigabyte": "gb",
        "gigabytes": "gb",
        "giga": "gb",
        "gigas": "gb",
        "megabyte": "mb",
        "megabytes": "mb",
        "tebioctet": "tib",
        "tebioctets": "tib",
        "terabyte": "tb",
        "terabytes": "tb",
        "dollar": "usd",
        "dollars": "usd",
        "percent": "%",
        "pourcent": "%",
        "gbps": "gb/s",
        "gbs": "gb/s",
        "mbps": "mb/s",
        "mbs": "mb/s",
        "tbps": "tb/s",
        "tbs": "tb/s",
    }

    return aliases.get(normalized, normalized)


def normalize_unit_value(
    field: ParamName,
    value: Any,
    unit: Optional[str],
) -> Tuple[Any, Optional[str]]:
    """
    Convertit une valeur vers l'unité canonique du système.

    Unités canoniques :
    - capacité : TiB ;
    - taille de fichier : GB ;
    - débit : GB/s ;
    - puissance : W ;
    - budget : USD ;
    - croissance : % ;
    - comptes : sans unité.

    Les signes négatifs et les décimales sont conservés. La validation métier
    décide ensuite si la valeur est acceptable.
    """

    if not is_number(value):
        return value, unit

    unit_norm = _normalize_unit(unit)
    numeric_value = _to_decimal(value)

    # ================================================================
    # FILE SIZE
    # ================================================================

    if field in {
        ParamName.average_file_size_gb,
        ParamName.max_file_size_gb,
    }:
        if unit_norm in {"kb", "kib"}:
            converted = numeric_value / Decimal("1000000")

        elif unit_norm in {"mb", "mib"}:
            # Convention du dataset MVP : MB et MiB sont normalisés
            # avec le facteur décimal 1000.
            converted = numeric_value / Decimal("1000")

        elif unit_norm in {"tb", "tib"}:
            converted = numeric_value * Decimal("1000")

        else:
            converted = numeric_value

        return normalize_number(converted), "GB"

    # ================================================================
    # POWER
    # ================================================================

    if field == ParamName.max_power_w:
        if unit_norm == "mw":
            converted = numeric_value * Decimal("1000000")

        elif unit_norm == "kw":
            converted = numeric_value * Decimal("1000")

        else:
            converted = numeric_value

        return normalize_number(converted), "W"

    # ================================================================
    # CAPACITY
    # ================================================================

    if field == ParamName.requested_usable_capacity_tib:
        # Convention actuelle du projet et du dataset :
        # TB et TiB sont normalisés vers la même valeur numérique MVP.
        return normalize_number(numeric_value), "TiB"

    # ================================================================
    # THROUGHPUT
    # ================================================================

    if field in {
        ParamName.target_read_gbps,
        ParamName.target_write_gbps,
    }:
        if unit_norm == "mb/s":
            converted = numeric_value / Decimal("1000")

        elif unit_norm == "tb/s":
            converted = numeric_value * Decimal("1000")

        else:
            converted = numeric_value

        return normalize_number(converted), "GB/s"

    # ================================================================
    # BUDGET
    # ================================================================

    if field == ParamName.max_budget_usd:
        if unit_norm in {"kusd", "k$"}:
            converted = numeric_value * Decimal("1000")

        elif unit_norm in {"musd", "m$"}:
            converted = numeric_value * Decimal("1000000")

        else:
            converted = numeric_value

        return normalize_number(converted), "USD"

    # ================================================================
    # GROWTH
    # ================================================================

    if field == ParamName.annual_growth_percent:
        return normalize_number(numeric_value), "%"

    # ================================================================
    # INTEGER-LIKE FIELDS
    # ================================================================

    if field in {
        ParamName.client_count,
        ParamName.total_file_count,
    }:
        # Une valeur entière est rendue comme int.
        # Une valeur décimale reste décimale afin que StateGuard puisse
        # la rejeter comme non entière.
        return normalize_number(numeric_value), None

    return normalize_number(numeric_value), unit
