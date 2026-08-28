from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import Any

from .architecture_state import (
    build_complete_state_from_choices,
    validate_full_architecture_state,
)
from .compatibility_rules import find_compatible_hardware_paths
from .protection_arithmetic import enumerate_candidate_protections


GENERATOR_SCHEMA_VERSION = "1.0"


class FullArchitectureGeneratorError(RuntimeError):
    """Erreur de génération déterministe d'architectures complètes H8."""


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise FullArchitectureGeneratorError(f"{field}: entier > 0 requis.")

    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise FullArchitectureGeneratorError(
            f"{field}: entier > 0 requis."
        ) from error

    if number <= 0:
        raise FullArchitectureGeneratorError(f"{field}: entier > 0 requis.")

    return number


def _optional_positive_int(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field=field)


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FullArchitectureGeneratorError(f"{field}: objet JSON requis.")
    return value


def _list(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise FullArchitectureGeneratorError(f"{field}: liste JSON requise.")
    return value


def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FullArchitectureGeneratorError(f"{field}: chaîne non vide requise.")
    return value.strip()


def _candidate_order(candidate: dict[str, Any]) -> tuple[int, int, str]:
    selection_rank = candidate.get("selection_rank")
    ranking = candidate.get("ranking")

    if not isinstance(ranking, dict):
        ranking = {}

    ml_rank = ranking.get("ml_rank")

    try:
        selection_rank_int = int(selection_rank)
    except (TypeError, ValueError):
        selection_rank_int = 10**9

    try:
        ml_rank_int = int(ml_rank)
    except (TypeError, ValueError):
        ml_rank_int = 10**9

    identity = candidate.get("identity")
    drive_id = ""

    if isinstance(identity, dict):
        drive_id = str(identity.get("drive_id", ""))

    return (
        selection_rank_int,
        ml_rank_int,
        drive_id,
    )


def _path_signature(path: dict[str, Any]) -> tuple[Any, ...]:
    resources = path.get("minimum_resources")
    if not isinstance(resources, dict):
        resources = {}

    return (
        path.get("attachment_mode"),
        path.get("server_id"),
        path.get("controller_id"),
        path.get("enclosure_id"),
        path.get("network_id"),
        path.get("ha_profile_id"),
        resources.get("physical_drive_count"),
        resources.get("server_count"),
        resources.get("controller_count"),
        resources.get("enclosure_count"),
        resources.get("network_adapter_count"),
    )


def _role_option_signature(option: dict[str, Any]) -> tuple[Any, ...]:
    candidate = _mapping(option.get("candidate"), field="option.candidate")
    protection = _mapping(option.get("protection"), field="option.protection")
    path = _mapping(option.get("hardware_path"), field="option.hardware_path")
    identity = _mapping(candidate.get("identity"), field="candidate.identity")

    return (
        option.get("role"),
        identity.get("drive_id"),
        protection.get("protection_profile_id"),
        *_path_signature(path),
    )


def architecture_signature(state: dict[str, Any]) -> tuple[Any, ...]:
    """Signature stable d'une architecture H7 complète."""

    validate_full_architecture_state(state)

    if state.get("stage") != "COMPLETE":
        raise FullArchitectureGeneratorError(
            "architecture_signature exige un state COMPLETE."
        )

    selected = _mapping(state.get("selected"), field="state.selected")
    counts = _mapping(state.get("counts"), field="state.counts")

    mdt_drive = _mapping(selected.get("mdt_drive"), field="selected.mdt_drive")
    ost_drive = _mapping(selected.get("ost_drive"), field="selected.ost_drive")
    mdt_protection = _mapping(
        selected.get("mdt_protection"),
        field="selected.mdt_protection",
    )
    ost_protection = _mapping(
        selected.get("ost_protection"),
        field="selected.ost_protection",
    )
    mdt_path = _mapping(
        selected.get("mdt_hardware_path"),
        field="selected.mdt_hardware_path",
    )
    ost_path = _mapping(
        selected.get("ost_hardware_path"),
        field="selected.ost_hardware_path",
    )

    return (
        state.get("case_id"),
        mdt_drive.get("drive_id"),
        ost_drive.get("drive_id"),
        mdt_protection.get("protection_profile_id"),
        ost_protection.get("protection_profile_id"),
        *_path_signature(mdt_path),
        *_path_signature(ost_path),
        counts.get("mdt_physical_drives"),
        counts.get("ost_physical_drives"),
        counts.get("mdt_count"),
        counts.get("ost_count"),
        counts.get("mds_count"),
        counts.get("oss_count"),
    )


def architecture_id(state: dict[str, Any]) -> str:
    signature = architecture_signature(state)
    encoded = json.dumps(
        signature,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    case_id = _string(state.get("case_id"), field="state.case_id")
    return f"ARCH_{case_id}_{digest}"


def enumerate_role_options(
    *,
    handoff: dict[str, Any],
    hardware_catalog: dict[str, Any],
    role: str,
    max_paths_per_variant: int,
    max_role_options: int | None = None,
) -> list[dict[str, Any]]:
    """
    Énumère des options complètes pour un seul côté MDT ou OST.

    Ordre déterministe :
    candidat ranking -> ordre catalogue des protections -> ordre H6 des paths.

    Aucun score architecture et aucun Beam Search ne sont appliqués.
    """

    expected_role = str(role).strip().upper()
    if expected_role not in {"MDT", "OST"}:
        raise FullArchitectureGeneratorError(f"role non supporté={role!r}.")

    path_limit = _positive_int(
        max_paths_per_variant,
        field="max_paths_per_variant",
    )
    role_limit = _optional_positive_int(
        max_role_options,
        field="max_role_options",
    )

    requirements = _mapping(handoff.get("requirements"), field="handoff.requirements")
    constraints = _mapping(requirements.get("constraints"), field="requirements.constraints")

    candidate_key = "mdt_candidates" if expected_role == "MDT" else "ost_candidates"
    requirement_key = "MDT_requirement" if expected_role == "MDT" else "OST_requirement"

    candidates = _list(handoff.get(candidate_key), field=f"handoff.{candidate_key}")
    requirement = _mapping(
        requirements.get(requirement_key),
        field=f"requirements.{requirement_key}",
    )
    protection_profiles = _list(
        hardware_catalog.get("protection_profiles"),
        field="hardware_catalog.protection_profiles",
    )

    if not candidates:
        raise FullArchitectureGeneratorError(f"Aucun candidat {expected_role}.")
    if not protection_profiles:
        raise FullArchitectureGeneratorError("Aucun profil de protection.")

    ha_required = bool(constraints.get("ha_required", False))
    options: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for candidate in sorted(candidates, key=_candidate_order):
        if candidate.get("role") != expected_role:
            raise FullArchitectureGeneratorError(
                f"Candidat {expected_role} avec role incohérent."
            )

        protections = enumerate_candidate_protections(
            candidate=candidate,
            protection_profiles=protection_profiles,
            requirement=requirement,
        )

        for protection_index, protection in enumerate(protections, start=1):
            paths = find_compatible_hardware_paths(
                candidate=candidate,
                protection_result=protection,
                role=expected_role,
                hardware_catalog=hardware_catalog,
                ha_required=ha_required,
                max_paths=path_limit,
            )

            for path_index, path in enumerate(paths, start=1):
                option = {
                    "role": expected_role,
                    "candidate": candidate,
                    "protection": protection,
                    "hardware_path": path,
                    "provenance": {
                        "selection_rank": candidate.get("selection_rank"),
                        "ml_rank": (
                            candidate.get("ranking", {}).get("ml_rank")
                            if isinstance(candidate.get("ranking"), dict)
                            else None
                        ),
                        "protection_catalog_index": protection_index,
                        "hardware_path_index": path_index,
                    },
                }

                signature = _role_option_signature(option)
                if signature in seen:
                    continue

                seen.add(signature)
                options.append(option)

                if role_limit is not None and len(options) >= role_limit:
                    return options

    return options


def iter_full_architectures(
    *,
    handoff: dict[str, Any],
    hardware_catalog: dict[str, Any],
    max_paths_per_variant: int,
    max_role_options_per_role: int | None = None,
) -> Iterator[dict[str, Any]]:
    """
    Produit le produit cartésien MDT × OST des options H8.

    Cette fonction est paresseuse : aucun `max_architectures` n'est appliqué.
    Un consommateur peut donc parcourir tout le domaine configuré ou s'arrêter
    explicitement. Beam Search n'est pas utilisé ici.
    """

    case_id = _string(handoff.get("case_id"), field="handoff.case_id")

    mdt_options = enumerate_role_options(
        handoff=handoff,
        hardware_catalog=hardware_catalog,
        role="MDT",
        max_paths_per_variant=max_paths_per_variant,
        max_role_options=max_role_options_per_role,
    )
    ost_options = enumerate_role_options(
        handoff=handoff,
        hardware_catalog=hardware_catalog,
        role="OST",
        max_paths_per_variant=max_paths_per_variant,
        max_role_options=max_role_options_per_role,
    )

    if not mdt_options:
        raise FullArchitectureGeneratorError(
            f"{case_id}: aucune option MDT compatible."
        )
    if not ost_options:
        raise FullArchitectureGeneratorError(
            f"{case_id}: aucune option OST compatible."
        )

    seen_architectures: set[str] = set()
    generation_index = 0

    for mdt_index, mdt_option in enumerate(mdt_options, start=1):
        for ost_index, ost_option in enumerate(ost_options, start=1):
            state = build_complete_state_from_choices(
                handoff=handoff,
                mdt_candidate=mdt_option["candidate"],
                ost_candidate=ost_option["candidate"],
                mdt_protection=mdt_option["protection"],
                ost_protection=ost_option["protection"],
                mdt_path=mdt_option["hardware_path"],
                ost_path=ost_option["hardware_path"],
            )
            validate_full_architecture_state(state)

            arch_id = architecture_id(state)
            if arch_id in seen_architectures:
                continue

            seen_architectures.add(arch_id)
            generation_index += 1

            yield {
                "schema_version": GENERATOR_SCHEMA_VERSION,
                "architecture_id": arch_id,
                "case_id": case_id,
                "generation_index": generation_index,
                "mdt_option_index": mdt_index,
                "ost_option_index": ost_index,
                "generation_semantics": {
                    "beam_search_applied": False,
                    "architecture_score_applied": False,
                    "hard_compatibility_precedes_generation": True,
                    "cartesian_pairing": True,
                    "role_instances_are_isolated": True,
                },
                "state": state,
            }


def generate_full_architectures(
    *,
    handoff: dict[str, Any],
    hardware_catalog: dict[str, Any],
    max_paths_per_variant: int = 2,
    max_role_options_per_role: int | None = None,
    max_architectures: int | None = None,
) -> dict[str, Any]:
    """
    Collecte les architectures H8 dans un artifact structuré.

    Les limites sont des bornes d'évaluation/génération explicites, jamais un
    score ou une heuristique d'optimalité.
    """

    path_limit = _positive_int(
        max_paths_per_variant,
        field="max_paths_per_variant",
    )
    role_limit = _optional_positive_int(
        max_role_options_per_role,
        field="max_role_options_per_role",
    )
    architecture_limit = _optional_positive_int(
        max_architectures,
        field="max_architectures",
    )

    case_id = _string(handoff.get("case_id"), field="handoff.case_id")

    mdt_options = enumerate_role_options(
        handoff=handoff,
        hardware_catalog=hardware_catalog,
        role="MDT",
        max_paths_per_variant=path_limit,
        max_role_options=role_limit,
    )
    ost_options = enumerate_role_options(
        handoff=handoff,
        hardware_catalog=hardware_catalog,
        role="OST",
        max_paths_per_variant=path_limit,
        max_role_options=role_limit,
    )

    if not mdt_options or not ost_options:
        raise FullArchitectureGeneratorError(
            f"{case_id}: options MDT/OST insuffisantes pour H8."
        )

    potential_pairs = len(mdt_options) * len(ost_options)
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for mdt_index, mdt_option in enumerate(mdt_options, start=1):
        for ost_index, ost_option in enumerate(ost_options, start=1):
            state = build_complete_state_from_choices(
                handoff=handoff,
                mdt_candidate=mdt_option["candidate"],
                ost_candidate=ost_option["candidate"],
                mdt_protection=mdt_option["protection"],
                ost_protection=ost_option["protection"],
                mdt_path=mdt_option["hardware_path"],
                ost_path=ost_option["hardware_path"],
            )
            arch_id = architecture_id(state)

            if arch_id in seen_ids:
                continue

            seen_ids.add(arch_id)
            records.append(
                {
                    "schema_version": GENERATOR_SCHEMA_VERSION,
                    "architecture_id": arch_id,
                    "case_id": case_id,
                    "generation_index": len(records) + 1,
                    "mdt_option_index": mdt_index,
                    "ost_option_index": ost_index,
                    "generation_semantics": {
                        "beam_search_applied": False,
                        "architecture_score_applied": False,
                        "hard_compatibility_precedes_generation": True,
                        "cartesian_pairing": True,
                        "role_instances_are_isolated": True,
                    },
                    "state": state,
                }
            )

            if (
                architecture_limit is not None
                and len(records) >= architecture_limit
            ):
                break

        if architecture_limit is not None and len(records) >= architecture_limit:
            break

    if not records:
        raise FullArchitectureGeneratorError(
            f"{case_id}: aucune architecture H8 générée."
        )

    return {
        "schema_version": GENERATOR_SCHEMA_VERSION,
        "stage": "full_architecture_generation",
        "case_id": case_id,
        "generation_contract": {
            "beam_search_applied": False,
            "architecture_score_applied": False,
            "full_validator_applied": False,
            "states_remain_pending_full_validator": True,
            "role_instances_are_isolated": True,
            "ordering": (
                "candidate_selection_rank_then_protection_catalog_order_"
                "then_hardware_path_order_then_mdt_x_ost_cartesian_product"
            ),
        },
        "limits": {
            "max_paths_per_variant": path_limit,
            "max_role_options_per_role": role_limit,
            "max_architectures": architecture_limit,
        },
        "summary": {
            "mdt_role_options": len(mdt_options),
            "ost_role_options": len(ost_options),
            "potential_pair_count": potential_pairs,
            "generated_architecture_count": len(records),
            "truncated_by_max_architectures": (
                architecture_limit is not None
                and len(records) < potential_pairs
            ),
        },
        "architectures": records,
    }
