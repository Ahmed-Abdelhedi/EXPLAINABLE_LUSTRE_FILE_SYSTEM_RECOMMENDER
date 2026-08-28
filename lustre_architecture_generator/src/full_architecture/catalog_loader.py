from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .hardware_schema import (
    PROTOCOLS,
    validate_hardware_catalog_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HARDWARE_CATALOG = (
    PROJECT_ROOT / "data" / "hardware_catalog_reference_v1.json"
)
DEFAULT_HARDWARE_MANIFEST = (
    PROJECT_ROOT / "data" / "hardware_catalog_reference_v1_manifest.json"
)


class HardwareCatalogError(RuntimeError):
    """Erreur de chargement ou validation du catalogue hardware H4."""


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _all_component_ids(catalog: dict[str, Any]) -> list[str]:
    ids: list[str] = []

    for section in (
        "servers",
        "controllers",
        "enclosures",
        "networks",
        "protection_profiles",
        "ha_profiles",
    ):
        values = catalog.get(section, [])
        if not isinstance(values, list):
            continue

        for item in values:
            if isinstance(item, dict):
                component_id = item.get("id")
                if isinstance(component_id, str):
                    ids.append(component_id)

    return ids


def validate_reference_catalog(catalog: dict[str, Any]) -> None:
    validate_hardware_catalog_bundle(catalog)

    if catalog.get("catalog_kind") != "REFERENCE_POLICY_CATALOG":
        raise HardwareCatalogError(
            "catalog_kind doit être REFERENCE_POLICY_CATALOG."
        )

    provenance = catalog.get("provenance")
    if not isinstance(provenance, dict):
        raise HardwareCatalogError("provenance absente.")

    if (
        provenance.get("status")
        != "POLICY_REFERENCE_NOT_VENDOR_BENCHMARKED"
    ):
        raise HardwareCatalogError(
            "Le statut de provenance H4 doit rester explicite."
        )

    all_ids = _all_component_ids(catalog)
    if len(all_ids) != len(set(all_ids)):
        raise HardwareCatalogError(
            "Les IDs hardware doivent être uniques globalement."
        )

    server_roles = {
        str(item["role"]).upper()
        for item in catalog["servers"]
    }

    if "MDS" not in server_roles:
        raise HardwareCatalogError("Aucun profil serveur MDS.")

    if "OSS" not in server_roles:
        raise HardwareCatalogError("Aucun profil serveur OSS.")

    supported_protocols: set[str] = set()

    for item in catalog["controllers"]:
        supported_protocols.update(
            str(value).upper()
            for value in item["supported_protocols"]
        )

    missing_protocols = PROTOCOLS - supported_protocols

    if missing_protocols:
        raise HardwareCatalogError(
            "Protocoles drive sans contrôleur de référence : "
            + ", ".join(sorted(missing_protocols))
        )

    fabrics = {
        str(item["fabric"]).upper()
        for item in catalog["networks"]
    }

    for required_fabric in ("ETHERNET", "INFINIBAND"):
        if required_fabric not in fabrics:
            raise HardwareCatalogError(
                f"Fabric réseau absent : {required_fabric}"
            )

    raid_levels = {
        str(item["raid_level"]).upper()
        for item in catalog["protection_profiles"]
    }

    for required_level in ("RAID1", "RAID10", "RAID6"):
        if required_level not in raid_levels:
            raise HardwareCatalogError(
                f"Protection absente : {required_level}"
            )

    ha_modes = {
        str(item["mode"]).upper()
        for item in catalog["ha_profiles"]
    }

    for required_mode in ("NONE", "ACTIVE_PASSIVE"):
        if required_mode not in ha_modes:
            raise HardwareCatalogError(
                f"Mode HA absent : {required_mode}"
            )


def validate_manifest(
    *,
    catalog_path: Path,
    manifest: dict[str, Any],
) -> None:
    expected_hash = manifest.get("catalog_sha256")

    if not isinstance(expected_hash, str):
        raise HardwareCatalogError(
            "catalog_sha256 absent du manifest."
        )

    actual_hash = sha256_file(catalog_path)

    if actual_hash != expected_hash:
        raise HardwareCatalogError(
            "SHA256 du catalogue hardware incorrect."
        )


def load_reference_catalog(
    catalog_path: Path = DEFAULT_HARDWARE_CATALOG,
    manifest_path: Path = DEFAULT_HARDWARE_MANIFEST,
) -> dict[str, Any]:
    catalog = load_json(catalog_path)
    manifest = load_json(manifest_path)

    if not isinstance(catalog, dict):
        raise HardwareCatalogError(
            "Le catalogue hardware doit être un objet."
        )

    if not isinstance(manifest, dict):
        raise HardwareCatalogError(
            "Le manifest hardware doit être un objet."
        )

    validate_reference_catalog(catalog)
    validate_manifest(
        catalog_path=catalog_path,
        manifest=manifest,
    )

    return catalog
