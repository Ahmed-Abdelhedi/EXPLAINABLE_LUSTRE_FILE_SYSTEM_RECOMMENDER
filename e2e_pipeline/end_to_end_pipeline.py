"""Online Requirement -> S10 -> Ranking -> H8/H9/H10 orchestration.

This module only wires already-frozen business components together.  It does
not duplicate sizing formulas, LightGBM logic, protection arithmetic,
hardware compatibility rules, scoring rules, or H10 validation rules.
"""

from __future__ import annotations

import copy
import importlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .requirement_to_sizing_adapter import (
    RequirementToSizingAdapterError,
    adapt_requirement_to_sizing_case,
)


PIPELINE_VERSION = "1.0"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_E2E_OUTPUT = PROJECT_ROOT / "output" / "final_e2e_result.json"


class E2EPipelineError(RuntimeError):
    """Raised for unexpected integration failures."""


@dataclass(frozen=True)
class PipelineLimits:
    """Validated H8 evaluation limits used before Beam Search exists."""

    top_k: int = 10
    max_paths_per_variant: int = 2
    max_role_options_per_role: int = 4
    max_architectures: int = 16

    def __post_init__(self) -> None:
        for field_name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} doit être un entier > 0.")


class RuntimeBackend(Protocol):
    def load_and_validate_config(self, project_root: Path) -> dict[str, Any]: ...
    def analyze_workload(self, case: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]: ...
    def calculate_features(self, workload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]: ...
    def generate_technical_architecture(self, features: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]: ...
    def load_drive_catalog(self, project_root: Path) -> list[dict[str, Any]]: ...
    def build_handoff(self, architecture: dict[str, Any], catalog: list[dict[str, Any]], top_k: int) -> dict[str, Any]: ...
    def load_hardware_catalog(self) -> dict[str, Any]: ...
    def generate_architectures(self, handoff: dict[str, Any], hardware_catalog: dict[str, Any], limits: PipelineLimits) -> dict[str, Any]: ...
    def score_architectures(self, generated: dict[str, Any], handoff: dict[str, Any]) -> dict[str, Any]: ...
    def validate_architectures(self, generated: dict[str, Any], handoff: dict[str, Any], hardware_catalog: dict[str, Any]) -> dict[str, Any]: ...


class FrozenRuntimeBackend:
    """Thin runtime bridge to the existing frozen project modules."""

    def __init__(self, project_root: Path = PROJECT_ROOT) -> None:
        self.project_root = Path(project_root).resolve()
        self.generator_root = self.project_root / "lustre_architecture_generator"
        self.src_dir = self.generator_root / "src"

        if not self.src_dir.exists():
            raise E2EPipelineError(
                "Dossier lustre_architecture_generator/src introuvable: "
                f"{self.src_dir}"
            )

        src_text = str(self.src_dir)
        if src_text not in sys.path:
            sys.path.insert(0, src_text)

        self.workload_module = importlib.import_module("workload_analyzer")
        self.feature_module = importlib.import_module("feature_calculator")
        self.technical_module = importlib.import_module("architecture_generator")
        self.runtime_adapter_module = importlib.import_module(
            "full_architecture.runtime_adapter"
        )
        self.catalog_loader_module = importlib.import_module(
            "full_architecture.catalog_loader"
        )
        self.h8_module = importlib.import_module(
            "full_architecture.full_architecture_generator"
        )
        self.h9_module = importlib.import_module(
            "full_architecture.architecture_scoring"
        )
        self.h10_module = importlib.import_module(
            "full_architecture.full_architecture_validator"
        )

    @staticmethod
    def _load_json(path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable: {path}")
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_and_validate_config(self, project_root: Path) -> dict[str, Any]:
        path = (
            Path(project_root)
            / "lustre_architecture_generator"
            / "config"
            / "architecture_rules.json"
        )
        config = self._load_json(path)
        if not isinstance(config, dict):
            raise E2EPipelineError(
                "architecture_rules.json doit contenir un objet JSON."
            )

        # Each frozen stage validates the portions of the same configuration
        # contract that it owns.
        self.workload_module.validate_config(config)
        self.feature_module.validate_config(config)
        self.technical_module.validate_config(config)
        return config

    def analyze_workload(
        self,
        case: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        return self.workload_module.analyze_workload(case, config)

    def calculate_features(
        self,
        workload: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        return self.feature_module.calculate_features(workload, config)

    def generate_technical_architecture(
        self,
        features: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        return self.technical_module.generate_architecture_case(features, config)

    def load_drive_catalog(self, project_root: Path) -> list[dict[str, Any]]:
        path = (
            Path(project_root)
            / "lustre_architecture_generator"
            / "data"
            / "catalogue_drives_ready_final.json"
        )
        catalog = self._load_json(path)
        if not isinstance(catalog, list) or not catalog:
            raise E2EPipelineError(
                "Le catalogue de drives doit être une liste JSON non vide."
            )
        return catalog

    def build_handoff(
        self,
        architecture: dict[str, Any],
        catalog: list[dict[str, Any]],
        top_k: int,
    ) -> dict[str, Any]:
        return self.runtime_adapter_module.build_runtime_handoff(
            architecture=architecture,
            catalog=catalog,
            top_k=top_k,
        )

    def load_hardware_catalog(self) -> dict[str, Any]:
        return self.catalog_loader_module.load_reference_catalog()

    def generate_architectures(
        self,
        handoff: dict[str, Any],
        hardware_catalog: dict[str, Any],
        limits: PipelineLimits,
    ) -> dict[str, Any]:
        return self.h8_module.generate_full_architectures(
            handoff=handoff,
            hardware_catalog=hardware_catalog,
            max_paths_per_variant=limits.max_paths_per_variant,
            max_role_options_per_role=limits.max_role_options_per_role,
            max_architectures=limits.max_architectures,
        )

    def score_architectures(
        self,
        generated: dict[str, Any],
        handoff: dict[str, Any],
    ) -> dict[str, Any]:
        result = self.h9_module.score_generated_architectures(
            generation_result=generated,
            handoff=handoff,
        )
        self.h9_module.assert_scoring_result_valid(result)
        return result

    def validate_architectures(
        self,
        generated: dict[str, Any],
        handoff: dict[str, Any],
        hardware_catalog: dict[str, Any],
    ) -> dict[str, Any]:
        result = self.h10_module.validate_generated_architectures(
            generation_result=generated,
            handoff=handoff,
            hardware_catalog=hardware_catalog,
        )
        self.h10_module.assert_full_validation_result_valid(result)
        return result


def _save_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
    return path.resolve()


def _status_from_handoff_error(error: Exception) -> tuple[str, str]:
    text = str(error)
    lowered = text.casefold()

    if "mdt ranker" in lowered or "aucun candidat mdt" in lowered:
        return "NO_FEASIBLE_MDT", "RANKING_MDT"

    if "ost ranker" in lowered or "aucun candidat ost" in lowered:
        return "NO_FEASIBLE_OST", "RANKING_OST"

    return "PIPELINE_ERROR", "RANKING_HANDOFF"


def _base_result(
    *,
    requirement: dict[str, Any],
    limits: PipelineLimits,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "pipeline_version": PIPELINE_VERSION,
        "status": "RUNNING",
        "failure_stage": None,
        "message": "",
        "requirement": copy.deepcopy(requirement),
        "sizing_input": None,
        "sizing": None,
        "candidate_space": None,
        "architecture_search": None,
        "best_architecture": None,
        "trace": {
            "beam_search_applied": False,
            "h10_is_final_validity_authority": True,
            "frozen_business_logic_reimplemented_here": False,
            "limits": asdict(limits),
        },
    }


def _finish_failure(
    result: dict[str, Any],
    *,
    status: str,
    stage: str,
    message: str,
    output_path: Path,
    started: float,
) -> dict[str, Any]:
    result["status"] = status
    result["failure_stage"] = stage
    result["message"] = message
    result["trace"]["elapsed_seconds"] = round(
        time.perf_counter() - started,
        6,
    )
    result["trace"]["output_path"] = str(_save_json(output_path, result))
    return result


def _candidate_space_from_handoff(handoff: dict[str, Any]) -> dict[str, Any]:
    return {
        "requested_top_k": handoff.get("requested_top_k"),
        "actual_top_k": copy.deepcopy(handoff.get("actual_top_k")),
        "ranking_provenance": copy.deepcopy(
            handoff.get("ranking_provenance")
        ),
        "mdt_candidates": copy.deepcopy(handoff.get("mdt_candidates", [])),
        "ost_candidates": copy.deepcopy(handoff.get("ost_candidates", [])),
        "contract_invariants": copy.deepcopy(
            handoff.get("contract_invariants")
        ),
    }


def _select_best_valid(
    *,
    generated: dict[str, Any],
    scored: dict[str, Any],
    validated: dict[str, Any],
) -> dict[str, Any] | None:
    generation_by_id = {
        str(item["architecture_id"]): item
        for item in generated.get("architectures", [])
        if isinstance(item, dict) and "architecture_id" in item
    }
    validation_by_id = {
        str(item["architecture_id"]): item
        for item in validated.get("architectures", [])
        if isinstance(item, dict) and "architecture_id" in item
    }

    for score in scored.get("architectures", []):
        if not isinstance(score, dict):
            continue
        architecture_id = str(score.get("architecture_id", ""))
        decision = validation_by_id.get(architecture_id)
        architecture = generation_by_id.get(architecture_id)

        if (
            architecture
            and decision
            and decision.get("valid") is True
        ):
            return {
                "architecture_id": architecture_id,
                "score": copy.deepcopy(score),
                "validation": copy.deepcopy(decision),
                "architecture": copy.deepcopy(architecture),
            }

    return None


def run_e2e(
    requirement: dict[str, Any],
    *,
    output_path: Path = DEFAULT_E2E_OUTPUT,
    project_root: Path = PROJECT_ROOT,
    backend: RuntimeBackend | None = None,
    limits: PipelineLimits | None = None,
) -> dict[str, Any]:
    """Run one canonical Requirement through the frozen online downstream."""

    started = time.perf_counter()
    resolved_limits = limits or PipelineLimits()
    root = Path(project_root).resolve()
    result = _base_result(
        requirement=requirement,
        limits=resolved_limits,
    )

    try:
        sizing_case = adapt_requirement_to_sizing_case(requirement)
    except RequirementToSizingAdapterError as error:
        return _finish_failure(
            result,
            status="INVALID_REQUIREMENT_FOR_SIZING",
            stage="REQUIREMENT_TO_SIZING_ADAPTER",
            message=str(error),
            output_path=Path(output_path),
            started=started,
        )

    result["sizing_input"] = copy.deepcopy(sizing_case)

    runtime: RuntimeBackend
    try:
        runtime = backend or FrozenRuntimeBackend(root)
        config = runtime.load_and_validate_config(root)
        workload = runtime.analyze_workload(sizing_case, config)
        features = runtime.calculate_features(workload, config)
        technical = runtime.generate_technical_architecture(features, config)
    except Exception as error:
        return _finish_failure(
            result,
            status="PIPELINE_ERROR",
            stage="S10_SIZING",
            message=f"{type(error).__name__}: {error}",
            output_path=Path(output_path),
            started=started,
        )

    result["sizing"] = {
        "workload_analysis": copy.deepcopy(workload),
        "technical_requirements": copy.deepcopy(technical),
    }

    try:
        drive_catalog = runtime.load_drive_catalog(root)
        handoff = runtime.build_handoff(
            technical,
            drive_catalog,
            resolved_limits.top_k,
        )
    except Exception as error:
        status, stage = _status_from_handoff_error(error)
        return _finish_failure(
            result,
            status=status,
            stage=stage,
            message=f"{type(error).__name__}: {error}",
            output_path=Path(output_path),
            started=started,
        )

    result["candidate_space"] = _candidate_space_from_handoff(handoff)

    try:
        hardware_catalog = runtime.load_hardware_catalog()
        generated = runtime.generate_architectures(
            handoff,
            hardware_catalog,
            resolved_limits,
        )
    except Exception as error:
        return _finish_failure(
            result,
            status="NO_VALID_ARCHITECTURE",
            stage="H8_FULL_ARCHITECTURE_GENERATION",
            message=f"{type(error).__name__}: {error}",
            output_path=Path(output_path),
            started=started,
        )

    try:
        scored = runtime.score_architectures(generated, handoff)
        validated = runtime.validate_architectures(
            generated,
            handoff,
            hardware_catalog,
        )
    except Exception as error:
        return _finish_failure(
            result,
            status="PIPELINE_ERROR",
            stage="H9_H10_EVALUATION",
            message=f"{type(error).__name__}: {error}",
            output_path=Path(output_path),
            started=started,
        )

    result["architecture_search"] = {
        "h8_summary": copy.deepcopy(generated.get("summary", {})),
        "h9_summary": copy.deepcopy(scored.get("summary", {})),
        "h10_summary": copy.deepcopy(validated.get("summary", {})),
        "selection_rule": (
            "highest H9 score among architectures independently declared "
            "valid by H10"
        ),
        "beam_search_applied": False,
    }

    best = _select_best_valid(
        generated=generated,
        scored=scored,
        validated=validated,
    )

    if best is None:
        return _finish_failure(
            result,
            status="NO_VALID_ARCHITECTURE",
            stage="H10_FULL_VALIDATION",
            message=(
                "H10 n'a déclaré valide aucune architecture du pool H8 "
                "évalué. Ce résultat ne doit pas être transformé en succès "
                "par le score H9."
            ),
            output_path=Path(output_path),
            started=started,
        )

    result["status"] = "SUCCESS"
    result["message"] = (
        "Architecture complète sélectionnée parmi les architectures H10 valides."
    )
    result["best_architecture"] = best
    result["trace"]["elapsed_seconds"] = round(
        time.perf_counter() - started,
        6,
    )
    result["trace"]["output_path"] = str(
        _save_json(Path(output_path), result)
    )
    return result


def run_e2e_from_file(
    requirement_path: Path,
    *,
    output_path: Path = DEFAULT_E2E_OUTPUT,
    project_root: Path = PROJECT_ROOT,
    limits: PipelineLimits | None = None,
) -> dict[str, Any]:
    path = Path(requirement_path)
    if not path.exists():
        raise FileNotFoundError(f"Requirement JSON introuvable: {path}")

    with path.open("r", encoding="utf-8") as handle:
        requirement = json.load(handle)

    if not isinstance(requirement, dict):
        raise E2EPipelineError(
            "Le fichier final Requirement doit contenir un objet JSON."
        )

    return run_e2e(
        requirement,
        output_path=Path(output_path),
        project_root=Path(project_root),
        limits=limits,
    )
