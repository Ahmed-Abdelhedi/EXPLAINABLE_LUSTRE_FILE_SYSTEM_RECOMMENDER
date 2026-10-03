"""Deterministic bounded search around the unchanged H5--H10 primitives."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, replace
from statistics import fmean
from time import perf_counter
from typing import Any

from .architecture_scoring import normalize_preference_weights, score_generated_architectures
from .architecture_state import (
    apply_drive_selection,
    apply_protection_selection,
    build_complete_state_from_choices,
    new_full_architecture_state,
)
from .compatibility_rules import find_compatible_hardware_paths
from .feasibility_coverage import role_option_cost_power
from .full_architecture_generator import architecture_id
from .full_architecture_validator import validate_complete_architecture
from .handoff_contract import assert_valid_architecture_handoff
from .hardware_schema import validate_hardware_catalog_bundle
from .protection_arithmetic import ProtectionArithmeticError, enumerate_candidate_protections


BEAM_SCHEMA_VERSION = "2.1"
BEAM_HEURISTIC_POLICY_ID = "BEAM_OPTIMISTIC_COMPLETION_V2_JOINT_RESOURCES"
NO_VALID_ARCHITECTURE = "NO_VALID_ARCHITECTURE_FOUND_WITHIN_SEARCH_DOMAIN"
SEARCH_STAGES = (
    "MDT_DRIVE", "MDT_PROTECTION", "MDT_HARDWARE",
    "OST_DRIVE", "OST_PROTECTION", "OST_HARDWARE",
)


class BeamSearchError(ValueError):
    """Invalid search parameters or non-finite search input."""


@dataclass(frozen=True)
class BeamState:
    """Search choices and provenance; H7 remains the physical state only.

    H7 requires both roles for a transition. Until OST_DRIVE, its state stays
    EMPTY; after that it advances to DRIVES_SELECTED / PROTECTION_SELECTED.
    """

    architecture_state: dict[str, Any]
    mdt: dict[str, Any] | None = None
    ost: dict[str, Any] | None = None
    branch_id: int = 0
    beam_score: float = 0.0
    trace: tuple[dict[str, Any], ...] = ()


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise BeamSearchError(f"{field}: integer > 0 required.")
    return value


def _nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise BeamSearchError(f"{field}: finite nonnegative number required.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise BeamSearchError(f"{field}: finite nonnegative number required.") from error
    if not math.isfinite(number) or number < 0:
        raise BeamSearchError(f"{field}: finite nonnegative number required.")
    return number


def _lower_bounds(node: BeamState) -> dict[str, float]:
    contributions = []
    for choice in (node.mdt, node.ost):
        if choice is None:
            continue  # Future resources are unknown, not infeasible.
        protection = choice.get("protection")
        if choice.get("hardware_path") is not None:
            contributions.append(role_option_cost_power(choice))
        elif protection is not None:
            contributions.append({
                "cost_usd": float(protection["protected_drive_cost_usd"]),
                "power_w": float(protection["protected_drive_power_w"]),
            })
        else:
            pre_raid = choice["candidate"]["pre_raid"]
            contributions.append({
                "cost_usd": float(pre_raid["drive_level_cost_usd"]),
                "power_w": float(pre_raid["drive_level_power_w"]),
            })
    return {
        key: sum(item[key] for item in contributions)
        for key in ("cost_usd", "power_w")
    }


def _bound_exceeds(value: float, maximum: float) -> bool:
    # H10 sums four terms in a different order. Conservatively allow rounding
    # at a boundary; H10 still applies its own exact 1e-12 tolerance at the end.
    tolerance = max(1e-12, 4 * math.ulp(value), 4 * math.ulp(maximum))
    return value > maximum + tolerance


def _hard_reasons(bounds: dict[str, float], constraints: dict[str, Any]) -> list[str]:
    reasons = []
    if _bound_exceeds(bounds["cost_usd"], float(constraints["max_budget_usd"])):
        reasons.append("budget_lower_bound_exceeded")
    if _bound_exceeds(bounds["power_w"], float(constraints["max_power_w"])):
        reasons.append("power_lower_bound_exceeded")
    return reasons


def _resource_credit(value: float, maximum: float) -> float:
    if maximum == 0:
        return 1.0 if value == 0 else 0.0
    return 1.0 / (1.0 + value / maximum)


def beam_heuristic(
    node: BeamState,
    *,
    preference_weights: dict[str, float],
    constraints: dict[str, Any],
    ha_modes: dict[str, str],
    completion_envelopes: tuple[dict[str, Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Pool-independent heuristic, higher is better; never a validity decision.

    With completion envelopes: h2 = w_perf P+ + w_cost C+ + w_power E+ + w_rel L+.
    Without envelopes, preserve the original public local heuristic below.

    h = .25 R + .75 (w_perf P + w_cost C + w_power E + w_rel L).
    R: mean reciprocal selection rank of selected drives.
    P: mean max(0, 1-required/provided) of known active H5 metrics.
    C/E: 1/(1+known_lower_bound/case_limit), with explicit zero-limit handling.
    L: mean of known protection f/(1+f) and known HA-presence credits.
    Missing performance/reliability evidence contributes zero if none is known.
    """
    if completion_envelopes is not None:
        # Componentwise optima may describe different completions. This is an
        # optimistic ordering signal, never a physical architecture or H9 score.
        bounds = {key: sum(e[key] for e in completion_envelopes)
                  for key in ("cost_usd", "power_w")}
        metric_count = sum(e["metric_count"] for e in completion_envelopes)
        components = {
            "ranking": fmean(e["ranking"] for e in completion_envelopes),
            "performance": sum(e["performance_sum"] for e in completion_envelopes) / metric_count if metric_count else 0.0,
            "cost": _resource_credit(bounds["cost_usd"], float(constraints["max_budget_usd"])),
            "power": _resource_credit(bounds["power_w"], float(constraints["max_power_w"])),
            "reliability": sum(e["reliability_sum"] for e in completion_envelopes) / 4,
        }
        score = sum(preference_weights[f"{key}_priority"] * components[key]
                    for key in ("performance", "cost", "power", "reliability"))
        return {"score": score, "components": components, "lower_bounds": bounds,
                "completion_option_counts": [e["option_count"] for e in completion_envelopes],
                "semantics": "optimistic componentwise completion; not H9 or validity"}
    ranks, headroom, reliability = [], [], []
    for choice in (node.mdt, node.ost):
        if choice is None:
            continue
        ranks.append(1.0 / choice["candidate"]["selection_rank"])
        protection = choice.get("protection")
        if protection is not None:
            for metric, required in protection["requirements"].items():
                if required > 0:
                    provided = protection["provided"][metric]
                    headroom.append(max(0.0, 1.0 - required / provided))
            fault_tolerance = protection["fault_tolerance_drives_per_group"]
            reliability.append(fault_tolerance / (1.0 + fault_tolerance))
        path = choice.get("hardware_path")
        if path is not None:
            reliability.append(float(ha_modes[path["ha_profile_id"]] != "NONE"))
    bounds = _lower_bounds(node)
    components = {
        "ranking": fmean(ranks) if ranks else 0.0,
        "performance": fmean(headroom) if headroom else 0.0,
        "cost": _resource_credit(bounds["cost_usd"], float(constraints["max_budget_usd"])),
        "power": _resource_credit(bounds["power_w"], float(constraints["max_power_w"])),
        "reliability": fmean(reliability) if reliability else 0.0,
    }
    score = 0.25 * components["ranking"] + 0.75 * sum(
        preference_weights[f"{key}_priority"] * components[key]
        for key in ("performance", "cost", "power", "reliability")
    )
    return {"score": score, "components": components, "lower_bounds": bounds}


def _choice_key(choice: dict[str, Any] | None) -> tuple[Any, ...]:
    if choice is None:
        return ()
    candidate = choice["candidate"]
    protection = choice.get("protection") or {}
    path = choice.get("hardware_path") or {}
    return (
        candidate["selection_rank"],
        str(candidate["identity"]["drive_id"]),
        choice.get("protection_catalog_index", 0),
        str(protection.get("protection_profile_id", "")),
        choice.get("hardware_path_index", 0),
        *(str(path.get(key) or "") for key in (
            "attachment_mode", "server_id", "controller_id", "enclosure_id",
            "network_id", "ha_profile_id",
        )),
    )


def _node_key(node: BeamState) -> tuple[Any, ...]:
    ranking = node.trace[-1]["heuristic"]["components"]["ranking"] if node.trace else 0.0
    return (-node.beam_score, -ranking, _choice_key(node.mdt), _choice_key(node.ost))


def _prefix_key(choice: dict[str, Any] | None) -> tuple[Any, ...]:
    if choice is None:
        return ()
    result = (choice["candidate"]["identity"]["drive_id"],)
    if choice.get("protection") is not None:
        result += (choice["protection"]["protection_profile_id"],)
    if choice.get("hardware_path") is not None:
        result += (choice["hardware_path_index"],)
    return result


def _add_completion(index: dict, choice: dict, ha_modes: dict) -> None:
    bounds = role_option_cost_power(choice)
    protection = choice["protection"]
    metrics = [max(0.0, 1.0-required/protection["provided"][metric])
               for metric, required in protection["requirements"].items() if required > 0]
    fault = protection["fault_tolerance_drives_per_group"]
    reliability = fault/(1+fault) + float(ha_modes[choice["hardware_path"]["ha_profile_id"]] != "NONE")
    full_key = _prefix_key(choice)
    for length in range(4):
        key = full_key[:length]
        values = {**bounds, "performance_sum": sum(metrics), "metric_count": len(metrics),
                  "reliability_sum": reliability, "ranking": 1.0/choice["candidate"]["selection_rank"],
                  "option_count": 1, "resource_points": [(bounds["cost_usd"], bounds["power_w"])]}
        if key not in index:
            index[key] = values
        else:
            old = index[key]
            for field in ("cost_usd", "power_w"):
                old[field] = min(old[field], values[field])
            for field in ("performance_sum", "reliability_sum", "ranking"):
                old[field] = max(old[field], values[field])
            old["option_count"] += 1
            old["resource_points"].extend(values["resource_points"])


def _pareto_resources(points) -> tuple[tuple[float, float], ...]:
    """Keep real cost/power pairs, sorted by cost with strictly decreasing power."""
    frontier = []
    lowest_power = math.inf
    for cost, power in sorted(set(points)):
        if power < lowest_power:
            frontier.append((cost, power))
            lowest_power = power
    return tuple(frontier)


def _joint_resources_possible(envelopes: tuple[dict, dict], constraints: dict) -> bool:
    """Test two role frontiers in linear time, without Cartesian materialization.

    Independent minima may refer to different completions. This necessary
    resource test removes only prefixes with no real joint completion. Fixed
    conservative rounding margins preserve the monotonic two-pointer scan;
    H10 remains the authority at the exact boundary.
    """
    mdt, ost = (e["resource_points"] for e in envelopes)
    budget = float(constraints["max_budget_usd"])
    power = float(constraints["max_power_w"])
    budget_limit = budget + max(1e-12, 8*math.ulp(budget))
    power_limit = power + max(1e-12, 8*math.ulp(power))
    ost_index = len(ost)-1
    for mdt_cost, mdt_power in mdt:
        while ost_index >= 0 and mdt_cost+ost[ost_index][0] > budget_limit:
            ost_index -= 1
        if ost_index < 0:
            break
        # The last affordable OST point has the lowest power among all
        # affordable points. Later MDT points cost more but need less power.
        if mdt_power+ost[ost_index][1] <= power_limit:
            return True
    return False


def _frontier_diversity(frontier: list[BeamState], constraints: dict) -> dict:
    result = {}
    for role in ("mdt", "ost"):
        choices = [getattr(n, role) for n in frontier if getattr(n, role) is not None]
        result[f"{role}_drives"] = len({c["candidate"]["identity"]["drive_id"] for c in choices})
        result[f"{role}_protections"] = len({c["protection"]["protection_profile_id"] for c in choices if c.get("protection")})
        paths = [c["hardware_path"] for c in choices if c.get("hardware_path")]
        result[f"{role}_hardware_paths"] = len({
            (*(p.get(k) for k in ("attachment_mode", "server_id", "controller_id", "enclosure_id", "network_id", "ha_profile_id")),
             tuple(sorted(p["minimum_resources"].items()))) for p in paths
        })
    bounds = [_lower_bounds(n) for n in frontier]
    for field, limit in (("cost_usd", "max_budget_usd"), ("power_w", "max_power_w")):
        result[f"distinct_{field}"] = len({b[field] for b in bounds})
        maximum = float(constraints[limit])
        result[f"{field}_deciles_of_limit"] = sorted({int(10*b[field]/maximum) for b in bounds}) if maximum else []
    return result


def _validate_input(handoff: dict[str, Any], hardware_catalog: dict[str, Any]) -> None:
    assert_valid_architecture_handoff(handoff)
    validate_hardware_catalog_bundle(hardware_catalog)
    _positive_int(handoff["requested_top_k"], "requested_top_k")
    # H1 evidence is authoritative about filtering, but validate numeric input
    # before search so malformed data cannot masquerade as an exhausted domain.
    requirements = handoff["requirements"]
    for role, fields in (
        ("MDT", ("required_metadata_capacity_tib", "required_read_iops", "required_write_iops")),
        ("OST", ("required_usable_capacity_tib", "required_read_bandwidth_gbps",
                 "required_write_bandwidth_gbps", "required_total_bandwidth_gbps")),
    ):
        for field in fields:
            _nonnegative(requirements[f"{role}_requirement"].get(field), f"{role}.{field}")
        for candidate in handoff[f"{role.lower()}_candidates"]:
            _positive_int(candidate["selection_rank"], "selection_rank")
            fields = ("minimum_drive_count", "provided_capacity_tib", "drive_level_cost_usd", "drive_level_power_w")
            fields += (("provided_read_iops", "provided_write_iops") if role == "MDT" else
                       ("provided_read_bandwidth_gb_s", "provided_write_bandwidth_gb_s", "provided_total_bandwidth_gb_s"))
            for field in fields:
                _nonnegative(candidate["pre_raid"].get(field), f"{role}.pre_raid.{field}")
            for field, value in candidate["pre_raid"].items():
                if field != "is_lower_bound_only":
                    _nonnegative(value, f"{role}.pre_raid.{field}")
    constraints = requirements["constraints"]
    for field in ("max_budget_usd", "max_power_w"):
        _nonnegative(constraints.get(field), field)
    if not isinstance(constraints.get("ha_required"), bool):
        raise BeamSearchError("ha_required: boolean required.")


def beam_search_architectures(
    *,
    handoff: dict[str, Any],
    hardware_catalog: dict[str, Any],
    beam_width: int,
    max_paths_per_variant: int = 2,
) -> dict[str, Any]:
    """Search before Cartesian materialization, then score with H9 and validate H10.

    Top-K comes exclusively from the supplied H1/H2 handoff. Per-role H5/H6
    completion envelopes inform all six bounded stages, without pairing roles.
    No exhaustive H8 pool is generated.
    H9/H10 artifacts keep their original beam_search_applied=False contracts.
    """
    started = perf_counter()
    width = _positive_int(beam_width, "beam_width")
    path_limit = _positive_int(max_paths_per_variant, "max_paths_per_variant")
    _validate_input(handoff, hardware_catalog)
    requirements = handoff["requirements"]
    constraints = requirements["constraints"]
    weight_info = normalize_preference_weights(requirements["preferences"])
    weights = weight_info["normalized"]
    ha_modes = {p["id"]: str(p["mode"]).upper() for p in hardware_catalog["ha_profiles"]}

    frontier = [BeamState(new_full_architecture_state(handoff=handoff))]
    branch_count = hard_count = width_count = max_active = max_buffer = 0
    reason_counts: Counter[str] = Counter()
    stages = []
    # Role preparation fills these caches before width selection. The main
    # search reuses them; neither phase constructs an exhaustive H8 pool.
    protection_cache: dict[tuple[str, str, int], tuple[list[dict[str, Any]], str | None]] = {}
    path_cache: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    h5_calls = h6_calls = 0

    # Linear role preparation, not a Cartesian search. Store only aggregate
    # envelopes per drive/protection/path prefix; caches are reused below.
    completion_indexes = {"MDT": {}, "OST": {}}
    lookahead_trace = []
    for role in ("MDT", "OST"):
        prepared = rejected = 0
        prep_reasons: Counter[str] = Counter()
        index = completion_indexes[role]
        for candidate in handoff[f"{role.lower()}_candidates"]:
            prepared += 1
            choice = {"role": role, "candidate": candidate}
            node = BeamState({}, **{role.lower(): choice})
            reasons = _hard_reasons(_lower_bounds(node), constraints)
            if reasons:
                rejected += 1
                prep_reasons.update(reasons)
                continue
            for profile_index, profile in enumerate(hardware_catalog["protection_profiles"], 1):
                prepared += 1
                key = (role, candidate["identity"]["drive_id"], profile_index)
                h5_calls += 1
                try:
                    protections = enumerate_candidate_protections(
                        candidate=candidate, protection_profiles=[profile],
                        requirement=requirements[f"{role}_requirement"],
                    )
                    protection_cache[key] = (protections, None)
                except ProtectionArithmeticError:
                    protections = []
                    protection_cache[key] = ([], "h5_protection_impossible")
                if not protections:
                    rejected += 1
                    prep_reasons["h5_protection_impossible"] += 1
                for protection in protections:
                    protected = {**choice, "protection": protection, "protection_catalog_index": profile_index}
                    reasons = _hard_reasons(_lower_bounds(BeamState({}, **{role.lower(): protected})), constraints)
                    if reasons:
                        rejected += 1
                        prep_reasons.update(reasons)
                        continue
                    path_key = (role, candidate["identity"]["drive_id"], protection["protection_profile_id"])
                    h6_calls += 1
                    paths = find_compatible_hardware_paths(
                        candidate=candidate, protection_result=protection, role=role,
                        hardware_catalog=hardware_catalog, ha_required=constraints["ha_required"],
                        max_paths=path_limit,
                    )
                    path_cache[path_key] = paths
                    if not paths:
                        prepared += 1
                        rejected += 1
                        prep_reasons["h6_no_compatible_path"] += 1
                    for path_index, path in enumerate(paths, 1):
                        prepared += 1
                        complete_choice = {**protected, "hardware_path": path, "hardware_path_index": path_index}
                        reasons = _hard_reasons(role_option_cost_power(complete_choice), constraints)
                        if reasons:
                            rejected += 1
                            prep_reasons.update(reasons)
                            continue
                        _add_completion(index, complete_choice, ha_modes)
        for envelope in index.values():
            envelope["resource_points"] = _pareto_resources(envelope["resource_points"])
        lookahead_trace.append({"search_stage": f"LOOKAHEAD_{role}", "branches_created": prepared,
                                "hard_pruned": rejected, "width_pruned": 0,
                                "pruning_reason_counts": dict(sorted(prep_reasons.items())),
                                "role_options": index.get((), {}).get("option_count", 0),
                                "prefix_envelopes": len(index),
                                "resource_frontier_points": sum(len(e["resource_points"]) for e in index.values())})

    for depth, stage in enumerate(SEARCH_STAGES, start=1):
        role, dimension = stage.split("_", 1)
        role_field = role.lower()
        retained: list[BeamState] = []
        stage_created = stage_hard = stage_width = 0
        stage_reasons: Counter[str] = Counter()
        seen_choices: set[tuple[Any, ...]] = set()

        def consider(parent: BeamState, choice: dict[str, Any] | None, error: str | None = None) -> None:
            nonlocal branch_count, hard_count, width_count, stage_created, stage_hard, stage_width
            nonlocal max_buffer
            branch_count += 1
            stage_created += 1
            if error is not None:
                hard_count += 1
                stage_hard += 1
                reason_counts[error] += 1
                stage_reasons[error] += 1
                return
            node = replace(parent, **{role_field: choice}, branch_id=branch_count)
            bounds = _lower_bounds(node)
            reasons = _hard_reasons(bounds, constraints)
            envelopes = tuple(completion_indexes[r].get(_prefix_key(getattr(node, r.lower())))
                              for r in ("MDT", "OST"))
            if not reasons:
                if any(e is None for e in envelopes):
                    reasons = ["no_compatible_completion"]
                else:
                    completed_bounds = {key: sum(e[key] for e in envelopes)
                                        for key in ("cost_usd", "power_w")}
                    reasons = [f"completion_{reason}" for reason in _hard_reasons(completed_bounds, constraints)]
                    if not reasons and not _joint_resources_possible(envelopes, constraints):
                        reasons = ["completion_joint_budget_power_conflict"]
            if reasons:
                hard_count += 1
                stage_hard += 1
                reason_counts.update(reasons)
                stage_reasons.update(reasons)
                return
            # Only perform H7 joint transitions when both choices exist.
            physical = parent.architecture_state
            if stage == "OST_DRIVE":
                physical = apply_drive_selection(
                    state=physical,
                    mdt_candidate=node.mdt["candidate"],
                    ost_candidate=node.ost["candidate"],
                )
            elif stage == "OST_PROTECTION":
                physical = apply_protection_selection(
                    state=physical,
                    mdt_protection=node.mdt["protection"],
                    ost_protection=node.ost["protection"],
                )
            info = beam_heuristic(
                node, preference_weights=weights, constraints=constraints, ha_modes=ha_modes,
                completion_envelopes=envelopes,
            )
            trace_entry = {
                "branch_id": node.branch_id, "parent_branch_id": parent.branch_id,
                "depth": depth, "search_stage": stage,
                "drive_id": choice["candidate"]["identity"]["drive_id"],
                "selection_rank": choice["candidate"]["selection_rank"],
                "ml_rank": choice["candidate"].get("ranking", {}).get("ml_rank"),
                "selection_reasons": choice["candidate"].get("ranking", {}).get("selection_reasons", []),
                "protection_profile_id": (choice.get("protection") or {}).get("protection_profile_id"),
                "protection_catalog_index": choice.get("protection_catalog_index"),
                "hardware_path_index": choice.get("hardware_path_index"),
                "physical_stage": physical["stage"],
                "physical_validation_status": physical["validation"]["status"],
                "heuristic": info,
            }
            node = replace(node, architecture_state=physical, beam_score=info["score"],
                           trace=(*parent.trace, trace_entry))
            signature = (_choice_key(node.mdt), _choice_key(node.ost))
            if signature in seen_choices:
                # H1/H4 reject duplicate IDs; this is only a defensive guard.
                raise BeamSearchError("Duplicate search choice in validated inputs.")
            seen_choices.add(signature)
            retained.append(node)
            max_buffer = max(max_buffer, len(retained))
            retained.sort(key=_node_key)
            if len(retained) > width:
                retained.pop()
                width_count += 1
                stage_width += 1
                reason_counts["beam_width_limit"] += 1
                stage_reasons["beam_width_limit"] += 1

        for parent in frontier:
            previous = getattr(parent, role_field)
            if dimension == "DRIVE":
                for candidate in sorted(handoff[f"{role_field}_candidates"], key=lambda c: (
                    c["selection_rank"], str(c["identity"]["drive_id"]),
                )):
                    consider(parent, {"role": role, "candidate": candidate})
            elif dimension == "PROTECTION":
                candidate = previous["candidate"]
                for index, profile in enumerate(hardware_catalog["protection_profiles"], start=1):
                    key = (role, candidate["identity"]["drive_id"], index)
                    if key not in protection_cache:
                        h5_calls += 1
                        try:
                            protections = enumerate_candidate_protections(
                                candidate=candidate, protection_profiles=[profile],
                                requirement=requirements[f"{role}_requirement"],
                            )
                            protection_cache[key] = (protections, None)
                        except ProtectionArithmeticError:
                            protection_cache[key] = ([], "h5_protection_impossible")
                    protections, failure = protection_cache[key]
                    if failure or not protections:
                        consider(parent, None, "h5_protection_impossible")
                    for protection in protections:
                        consider(parent, {**previous, "protection": protection,
                                          "protection_catalog_index": index})
            else:
                candidate, protection = previous["candidate"], previous["protection"]
                key = (role, candidate["identity"]["drive_id"], protection["protection_profile_id"])
                if key not in path_cache:
                    h6_calls += 1
                    path_cache[key] = find_compatible_hardware_paths(
                        candidate=candidate, protection_result=protection, role=role,
                        hardware_catalog=hardware_catalog,
                        ha_required=constraints["ha_required"], max_paths=path_limit,
                    )
                paths = path_cache[key]
                if not paths:
                    consider(parent, None, "h6_no_compatible_path")
                for index, path in enumerate(paths, start=1):
                    consider(parent, {**previous, "hardware_path": path, "hardware_path_index": index})

        frontier = [replace(node, trace=(*node.trace[:-1], {**node.trace[-1], "beam_rank": rank}))
                    for rank, node in enumerate(retained, start=1)]
        max_active = max(max_active, len(frontier))
        stages.append({
            "search_stage": stage, "depth": depth, "branches_created": stage_created,
            "hard_pruned": stage_hard, "width_pruned": stage_width,
            "pruning_reason_counts": dict(sorted(stage_reasons.items())),
            "retained": len(frontier),
            "best_retained_score": frontier[0].beam_score if frontier else None,
            "worst_retained_score": frontier[-1].beam_score if frontier else None,
            "diversity": _frontier_diversity(frontier, constraints),
        })
        if not frontier:
            break

    records = []
    seen_ids: set[str] = set()
    for node in frontier:
        state = build_complete_state_from_choices(
            handoff=handoff, mdt_candidate=node.mdt["candidate"], ost_candidate=node.ost["candidate"],
            mdt_protection=node.mdt["protection"], ost_protection=node.ost["protection"],
            mdt_path=node.mdt["hardware_path"], ost_path=node.ost["hardware_path"],
        )
        arch_id = architecture_id(state)
        if arch_id in seen_ids:
            continue
        seen_ids.add(arch_id)
        records.append({
            "architecture_id": arch_id, "case_id": handoff["case_id"],
            "generation_index": len(records) + 1, "state": state,
            "search_provenance": {"beam_heuristic": node.beam_score, "trace": list(node.trace)},
        })

    generation = {"case_id": handoff["case_id"], "architectures": records}
    scored = score_generated_architectures(generation_result=generation, handoff=handoff) if records else None
    by_id = {record["architecture_id"]: record for record in records}
    evaluated = []
    if scored is not None:
        for score in scored["architectures"]:
            record = by_id[score["architecture_id"]]
            decision = validate_complete_architecture(
                architecture=record, handoff=handoff, hardware_catalog=hardware_catalog,
            )
            evaluated.append({
                "architecture_id": record["architecture_id"],
                "case_id": record["case_id"],
                "state": decision["validated_state"],
                "h9": score, "h10": decision,
                "search_provenance": record["search_provenance"],
            })
    valid = [r for r in evaluated if r["h10"]["decision"] == "VALID" and r["h10"]["valid"] is True]
    best = valid[0] if valid else None  # Already in H9 score / architecture_id order.
    elapsed = perf_counter() - started
    lookahead_created = sum(s["branches_created"] for s in lookahead_trace)
    lookahead_hard = sum(s["hard_pruned"] for s in lookahead_trace)
    for preparation in lookahead_trace:
        reason_counts.update(preparation["pruning_reason_counts"])
    return {
        "schema_version": BEAM_SCHEMA_VERSION, "stage": "beam_search",
        "case_id": handoff["case_id"],
        "status": "VALID_ARCHITECTURE_FOUND" if best else NO_VALID_ARCHITECTURE,
        "beam_search_applied": True, "global_infeasibility_claimed": False,
        "beam_width": width, "requested_top_k": handoff["requested_top_k"],
        "actual_top_k": dict(handoff["actual_top_k"]),
        "max_paths_per_variant": path_limit,
        "heuristic_policy": {"id": BEAM_HEURISTIC_POLICY_ID, "preference_weights": weight_info,
                             "tie_breaks": "optimistic score descending, reciprocal ranking descending, MDT choice key, OST choice key"},
        "summary": {
            "branches_created": branch_count + lookahead_created, "branches_explored": branch_count + lookahead_created,
            "main_search_branches_created": branch_count, "lookahead_branches_created": lookahead_created,
            "lookahead_role_options": {r.lower(): completion_indexes[r].get((), {}).get("option_count", 0) for r in ("MDT", "OST")},
            "branches_pruned": hard_count + lookahead_hard + width_count, "hard_pruned": hard_count + lookahead_hard,
            "width_pruned": width_count, "pruning_reason_counts": dict(sorted(reason_counts.items())),
            "max_active_states": max_active, "max_selection_buffer": max_buffer,
            "h5_calls": h5_calls, "h6_calls": h6_calls,
            "complete_architectures_produced": len(records), "h10_calls": len(evaluated),
            "valid_architecture_count": len(valid), "search_seconds": elapsed,
            "best_architecture_id": best["architecture_id"] if best else None,
            "best_final_h9_score": best["h9"]["score"] if best else None,
        },
        "search_trace": stages,
        "lookahead_trace": lookahead_trace,
        "scoring_result": scored,
        "architectures": evaluated,
        "best_validated_architecture": best,
    }
