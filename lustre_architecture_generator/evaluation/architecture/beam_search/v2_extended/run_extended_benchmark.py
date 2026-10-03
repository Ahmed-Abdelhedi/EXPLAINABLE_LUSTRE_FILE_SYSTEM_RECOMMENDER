"""Deterministic, offline V2 evaluation; production search and ranking stay frozen."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, median
from time import perf_counter

HERE = Path(__file__).resolve().parent
BENCHMARK_DIR = HERE.parent
PROJECT_ROOT = HERE.parents[3]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(BENCHMARK_DIR))
import benchmark_beam_search as existing
from full_architecture.beam_search import BEAM_SCHEMA_VERSION, BEAM_HEURISTIC_POLICY_ID

KS = (5, 10, 20, 50)
WIDTHS = (1, 4, 8, 16, 32)
DEVELOPMENT_IDS = {"REQ_000001", "REQ_000002", "REQ_000003"}
EPS = 1e-12


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def features(case):
    return {
        "budget_usd": case["constraints"]["max_budget_usd"],
        "power_w": case["constraints"]["max_power_w"],
        "capacity_tib": case["OST_requirement"]["required_usable_capacity_tib"],
        "throughput_gbps": case["OST_requirement"]["required_total_bandwidth_gbps"],
        "mdt_iops": case["MDT_requirement"]["required_total_iops"],
        "ha_required": case["constraints"]["ha_required"],
        "ost_load": case["OST_requirement"]["priority"],
    }


def band(value, bounds):
    return "low" if value <= bounds[0] else "medium" if value <= bounds[1] else "high"


def select_cases(dataset, sample_size):
    """Include numeric extremes, then round-robin input strata, never feasibility."""
    ordered = sorted(dataset, key=lambda c: c["case_id"])
    ids = [c["case_id"] for c in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate dataset case IDs")
    vectors = {c["case_id"]: features(c) for c in ordered}
    numeric = ("budget_usd", "power_w", "capacity_tib", "throughput_gbps", "mdt_iops")
    thresholds = {name: [percentile([v[name] for v in vectors.values()], q) for q in (1/3, 2/3)]
                  for name in numeric}
    candidates = [c for c in ordered if c["case_id"] not in DEVELOPMENT_IDS]
    chosen = {}
    for name in numeric:
        for sign in (1, -1):
            case = min(candidates, key=lambda c: (sign * vectors[c["case_id"]][name], c["case_id"]))
            chosen[case["case_id"]] = case
    if sample_size < len(chosen):
        raise ValueError("Sample too small to include the documented extremes")
    strata = defaultdict(list)
    for case in candidates:
        v = vectors[case["case_id"]]
        key = (v["ha_required"], band(v["budget_usd"], thresholds["budget_usd"]),
               band(v["power_w"], thresholds["power_w"]), v["ost_load"])
        if case["case_id"] not in chosen:
            strata[key].append(case)
    for group in strata.values():
        group.sort(key=lambda c: (hashlib.sha256(c["case_id"].encode()).hexdigest(), c["case_id"]))
    while len(chosen) < min(sample_size, len(candidates)):
        for key in sorted(strata):
            if strata[key] and len(chosen) < sample_size:
                case = strata[key].pop(0)
                chosen[case["case_id"]] = case
    sample = sorted(chosen.values(), key=lambda c: c["case_id"])
    distributions = {}
    for label, cases in (("population", ordered), ("sample", sample)):
        vals = [vectors[c["case_id"]] for c in cases]
        distributions[label] = {
            "numeric": {name: {"min": min(v[name] for v in vals), "median": median(v[name] for v in vals),
                                "max": max(v[name] for v in vals),
                                "bands": dict(Counter(band(v[name], thresholds[name]) for v in vals))}
                        for name in numeric},
            "ha_required": dict(Counter(str(v["ha_required"]).lower() for v in vals)),
            "ost_load": dict(Counter(v["ost_load"] for v in vals)),
        }
    return sample, {
        "source": str(PROJECT_ROOT / "output/lustre_architecture_dataset.json"),
        "total_cases": len(ordered), "eligible_cases": len(candidates), "evaluated_cases": len(sample),
        "excluded_development_ids": sorted(DEVELOPMENT_IDS),
        "method": "Numeric minima/maxima, then sorted round-robin over HA x budget tertile x power tertile x OST priority; within each stratum SHA256(case_id) order. No feasibility filtering.",
        "nonempty_strata": len(strata), "tertile_thresholds": thresholds,
        "distributions": distributions,
        "selected_cases": [{"case_id": c["case_id"], **vectors[c["case_id"]]} for c in sample],
    }


def fingerprint(result):
    return {
        "status": result["status"], "selected": result["summary"]["best_architecture_id"],
        "counts": {k: v for k, v in result["summary"].items() if k != "search_seconds"},
        "complete_pool": [(r["architecture_id"], r["h9"]["score"], r["h10"]["decision"])
                          for r in result["architectures"]],
    }


def observe_caches(handoff, hardware_catalog, paths):
    """A separate untimed B=32 call reads return-frame locals, without patching code.

    These dictionaries only grow, so final entry counts are their maxima. Line
    events are disabled. The instrumented pass never enters runtime statistics.
    """
    observed = {}
    code = existing.beam_search_architectures.__code__

    def local_trace(frame, event, arg):
        if event == "return":
            values = frame.f_locals
            observed.update({
                "max_cached_h5_entries": len(values["protection_cache"]),
                "max_cached_h6_entries": len(values["path_cache"]),
                "role_completion_envelope_count": sum(map(len, values["completion_indexes"].values())),
                "resource_frontier_entries": sum(len(e.get("resource_points", ()))
                                                 for index in values["completion_indexes"].values() for e in index.values()),
            })
        return local_trace

    def trace(frame, event, arg):
        if event == "call" and frame.f_code is code:
            frame.f_trace_lines = False
            frame.f_trace_opcodes = False
            return local_trace
        return None

    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        result = existing.beam_search_architectures(
            handoff=handoff, hardware_catalog=hardware_catalog, beam_width=32,
            max_paths_per_variant=paths,
        )
    finally:
        sys.settrace(previous)
    summary = result["summary"]
    assert observed["max_cached_h5_entries"] == summary["h5_calls"]
    assert observed["max_cached_h6_entries"] == summary["h6_calls"]
    assert observed["role_completion_envelope_count"] == sum(s["prefix_envelopes"] for s in result["lookahead_trace"])
    return observed, result


def quality(result, reference):
    value = existing.compare_quality(result, reference)
    if value["status"] == "COMPARED_IN_COMMON_POOL":
        ranks = {r["architecture_id"]: i for i, r in enumerate(reference["valid_scores"], 1)}
        value.update({
            "best_surviving_architecture_common_rank": ranks[value["retained_best_in_common_pool"]],
            "selected_architecture_common_rank": value["exhaustive_valid_rank"],
            "total_common_score_regret": value["common_score_regret"],
        })
        assert abs(value["total_common_score_regret"] - value["search_common_score_regret"]
                   - value["pool_normalization_common_score_regret"]) <= EPS
    return value


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["requested_top_k"], row["beam_width"])].append(row)
    output = []
    for (k, width), group in sorted(groups.items()):
        runtimes = [r["runtime_total"] for r in group]
        branches = [r["branches_generated"] for r in group]
        total_branches = sum(branches)
        output.append({
            "k": k, "b": width, "valid_cases": sum(r["valid_found"] for r in group),
            "total_cases": len(group), "feasible_solution_rate": sum(r["valid_found"] for r in group)/len(group),
            "runtime_mean": fmean(runtimes), "runtime_median": median(runtimes),
            "runtime_p90": percentile(runtimes, .9), "runtime_p95": percentile(runtimes, .95),
            "branches_mean": fmean(branches), "branches_median": median(branches),
            "hard_pruned_total": sum(r["hard_pruned"] for r in group),
            "width_pruned_total": sum(r["beam_width_pruned"] for r in group),
            "hard_pruning_pct": 100*sum(r["hard_pruned"] for r in group)/total_branches if total_branches else 0,
            "width_pruning_pct": 100*sum(r["beam_width_pruned"] for r in group)/total_branches if total_branches else 0,
            "complete_mean": fmean(r["complete_architectures"] for r in group),
            "h10_calls_mean": fmean(r["h10_calls"] for r in group),
            "h10_valid_mean": fmean(r["h10_valid_count"] for r in group),
        })
    return output


def analyze(rows, baselines):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["case_id"], r["requested_top_k"])].append(r)
    sensitivity = []
    for (case_id, k), group in sorted(groups.items()):
        group.sort(key=lambda r: r["beam_width"])
        validity_violations = [{"lower_b": x["beam_width"], "higher_b": y["beam_width"]}
                               for i, x in enumerate(group) for y in group[i+1:]
                               if x["valid_found"] and not y["valid_found"]]
        comparable = [r for r in group if r["quality"]["status"] == "COMPARED_IN_COMMON_POOL"]
        decreases = [{"lower_b": x["beam_width"], "higher_b": y["beam_width"],
                      "lower_quality": x["quality"]["relative_quality"],
                      "higher_quality": y["quality"]["relative_quality"]}
                     for i, x in enumerate(comparable) for y in comparable[i+1:]
                     if y["quality"]["common_h9_score"] + EPS < x["quality"]["common_h9_score"]]
        qs = [r["quality"]["relative_quality"] for r in comparable]
        sensitivity.append({
            "case_id": case_id, "k": k, "validity_monotonicity": not validity_violations,
            "validity_counterexamples": validity_violations,
            "selected_quality_monotonicity": not decreases if len(comparable) >= 2 else None,
            "quality_counterexamples": decreases,
            "architecture_changes": sum(x["selected_architecture_id"] != y["selected_architecture_id"]
                                        for x, y in zip(group, group[1:])),
            "distinct_selected_architectures": len({r["selected_architecture_id"] for r in group if r["valid_found"]}),
            "relative_quality_range": max(qs)-min(qs) if qs else None,
            "widths": [{"b": r["beam_width"], "valid": r["valid_found"],
                        "architecture_id": r["selected_architecture_id"],
                        "relative_quality": r["quality"].get("relative_quality"),
                        "common_rank": r["quality"].get("selected_architecture_common_rank")} for r in group],
        })
    compared = [r for r in rows if r["quality"]["status"] == "COMPARED_IN_COMMON_POOL"]
    losses = Counter()
    for r in compared:
        search = r["quality"]["search_common_score_regret"] > EPS
        normalization = r["quality"]["pool_normalization_common_score_regret"] > EPS
        losses["both" if search and normalization else "search_only" if search else
               "normalization_only" if normalization else "no_measurable_loss"] += 1
    loss_distribution = {name: {"count": losses[name], "fraction": losses[name]/len(compared) if compared else None}
                         for name in ("search_only", "normalization_only", "both", "no_measurable_loss")}

    def compact(row):
        return {key: row[key] for key in ("case_id", "requested_top_k", "beam_width", "status", "runtime_total",
                "branches_generated", "complete_architectures", "h10_calls", "h10_valid_count",
                "selected_architecture_id", "exhaustive_status", "exhaustive_valid_count", "quality",
                "pruning_reason_counts", "lookahead_role_options") if key in row}

    def top_unique(values, score):
        seen, result = set(), []
        for row in sorted(values, key=score, reverse=True):
            if row["case_id"] not in seen:
                seen.add(row["case_id"])
                result.append(compact(row))
                if len(result) == 10:
                    break
        return result

    problems = {
        "no_solution": top_unique([r for r in rows if not r["valid_found"]],
                                  lambda r: (r.get("exhaustive_valid_count", 0) or 0, r["requested_top_k"], r["beam_width"])),
        "width_sensitive": sorted(sensitivity, key=lambda r: (not r["validity_monotonicity"],
                                  r["relative_quality_range"] or 0, r["architecture_changes"]), reverse=True)[:10],
        "high_search_regret": top_unique([r for r in compared if r["quality"]["search_common_score_regret"] > EPS],
                                        lambda r: r["quality"]["search_common_score_regret"]),
        "high_normalization_regret": top_unique([r for r in compared if r["quality"]["pool_normalization_common_score_regret"] > EPS],
                                               lambda r: r["quality"]["pool_normalization_common_score_regret"]),
        "slow": top_unique(rows, lambda r: r["runtime_total"]),
        "high_branches": top_unique(rows, lambda r: r["branches_generated"]),
    }
    cache_fields = ("max_cached_h5_entries", "max_cached_h6_entries", "max_beam_states",
                    "role_completion_envelope_count", "max_selection_buffer", "resource_frontier_entries")
    memory = [{"k": k, **{f"max_{field}": max(r["memory"].get(field, 0) for r in rows if r["requested_top_k"] == k)
                           for field in cache_fields}} for k in KS]
    baseline_counts = Counter(r["status"] for r in baselines)
    feasible_misses = [compact(r) for r in rows if not r["valid_found"] and (r.get("exhaustive_valid_count") or 0) > 0]
    summary = {
        "configurations": len(rows), "valid_configurations": sum(r["valid_found"] for r in rows),
        "feasible_solution_rate": sum(r["valid_found"] for r in rows)/len(rows),
        "unique_cases_valid_at_any_configuration": len({r["case_id"] for r in rows if r["valid_found"]}),
        "execution_errors": sum(r["status"] == "EXECUTION_ERROR" for r in rows),
        "baseline_status_counts": dict(baseline_counts),
        "exhaustive_runs_completed": baseline_counts["COMPLETE"],
        "exhaustive_no_role_options": baseline_counts["NO_ROLE_OPTIONS"],
        "quality_comparisons": len(compared),
        "unique_cases_quality_compared": len({r["case_id"] for r in compared}),
        "best_survived": sum(r["quality"]["exhaustive_best_survived_search"] for r in compared),
        "best_chosen": sum(r["quality"]["did_beam_find_exhaustive_best"] for r in compared),
        "mean_relative_quality": fmean(r["quality"]["relative_quality"] for r in compared) if compared else None,
        "min_relative_quality": min((r["quality"]["relative_quality"] for r in compared), default=None),
        "mean_search_regret": fmean(r["quality"]["search_common_score_regret"] for r in compared) if compared else None,
        "mean_normalization_regret": fmean(r["quality"]["pool_normalization_common_score_regret"] for r in compared) if compared else None,
        "max_selected_common_rank": max((r["quality"]["selected_architecture_common_rank"] for r in compared), default=None),
        "loss_distribution": loss_distribution,
        "validity_groups": len(sensitivity),
        "validity_monotone_groups": sum(r["validity_monotonicity"] for r in sensitivity),
        "quality_comparable_groups": sum(r["selected_quality_monotonicity"] is not None for r in sensitivity),
        "quality_monotone_groups": sum(r["selected_quality_monotonicity"] is True for r in sensitivity),
        "architecture_change_groups": sum(r["architecture_changes"] > 0 for r in sensitivity),
        "feasible_exhaustive_but_beam_missing": feasible_misses,
        "memory": memory,
    }
    return summary, sensitivity, problems


def table(headers, values):
    return ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"] + [
        "| " + " | ".join(str(v) for v in row) + " |" for row in values]


def write_report(payload):
    rows, aggs, summary = payload["cases"], payload["aggregates"], payload["summary"]
    sample = payload["dataset"]["distributions"]["sample"]
    initial_version = payload.get("before_phase", {}).get("beam_schema_version", payload["beam_schema_version"])
    lines = [f"# Beam V{payload['beam_schema_version']} — validation élargie", "",
             "La [revue finale V2.1](release_review.md) distingue le verdict courant du verdict historique V2 et donne les preuves, tests et fichiers recommandés pour le futur commit.", "",
             "## A. État initial", "",
             f"Version initiale {initial_version}; version finale mesurée {payload['beam_schema_version']}; politique finale `{payload['heuristic_policy']}`. Tests initiaux : 57 dédiés et 490 globaux PASS.", "",
             "État Git initial conservé intégralement dans `initial_git_status.txt`. H5–H10, ranking et historique vérifiés par SHA256 dans `frozen_sources_initial.json`. L'unique changement de production autorisé est le patch Beam documenté après la campagne BEFORE ; son diff est dans `engine_change.diff`.", "",
             "```text", (HERE / "initial_git_status.txt").read_text(encoding="utf-8-sig").rstrip(), "```", "",
             "## B. Dataset utilisé", "", f"Source : `{payload['dataset']['source']}`. {payload['dataset']['total_cases']} cas disponibles, {payload['dataset']['evaluated_cases']} évalués.", "",
             payload["dataset"]["method"], "", "Les trois cas de développement sont exclus. Les classes numériques utilisent les tertiles du dataset complet ; elles servent uniquement à l'analyse. Échantillon de stress équilibré par strates, sans pondération représentative de la fréquence de production. Données synthétiques contrôlées, sans validation de performances Lustre réelles.", ""]
    lines += table(["Contrainte", "Faible", "Moyenne", "Élevée", "Min", "Médiane", "Max"],
                   [[key, val["bands"].get("low", 0), val["bands"].get("medium", 0), val["bands"].get("high", 0),
                     val["min"], val["median"], val["max"]] for key, val in sample["numeric"].items()])
    lines += ["", "Unités : budget USD, puissance W, capacité TiB, IOPS MDT opérations/s. Les champs historiques `_gbps` désignent ici des GB/s, conformément au contrat du projet.", "",
              "Datasets inspectés : architectures, candidats MDT/OST, workload analysis et workload features (1 200 cas chacun). La source architecture porte les contraintes et exigences par rôle ; les candidats sont recalculés avec le runtime LightGBM officiel au lieu de réutiliser les exports candidats.", ""]
    pilot_reuse = payload.get("pilot_reuse") or payload.get("before_phase", {}).get("pilot_reuse")
    if pilot_reuse:
        lines += [pilot_reuse["reason"], "",
                  f"Cas complets réutilisés du pilote pour BEFORE : {pilot_reuse['case_ids']}. Aucun cas réutilisé pour AFTER. Sources exactes archivées dans les sous-dossiers BEFORE, AFTER et `pilot_100/runner_source.py`.", ""]
    lines += ["", f"HA : {sample['ha_required']}. Charge OST native : {sample['ost_load']}.", "",
              f"Top-K réel par rôle : `{payload['actual_top_k_distribution']}`.", "",
              "## C. Configuration benchmark", "", f"K={list(KS)}; B={list(WIDTHS)}; répétitions={payload['config']['repeats']}; chemins/variante={payload['config']['max_paths_per_variant']}; seuil exhaustif={payload['config']['exhaustive_max_pairs']} paires.", "",
              f"Environnement : `{payload['environment']}`.", "",
              "Runtime_total mesure l'appel Beam entier (validation d'entrée, préparation H5/H6, recherche, H9, H10). Ranking et préflight/exhaustif sont exclus et enregistrés séparément. Chaque valeur est la médiane des répétitions non instrumentées ; les statistiques décrivent les distributions entre cas. Quantiles interpolés linéairement. Le passage caches séparé est exclu des timings.", "",
              "## D. Couverture", ""]
    if payload.get("before_after_summary"):
        comparison = payload["before_after_summary"]
        lines += ["Les tableaux D–K décrivent la campagne finale V2.1. La matrice complète V2 originale est conservée dans `before_joint_fix/extended_benchmark.json/.csv/.md` ; la campagne finale brute dans `after_joint_fix/`. Les 1 000 comparaisons sont dans `before_after.csv`.", "",
                  f"BEFORE : {comparison['before_valid']}/1000 ; AFTER : {comparison['after_valid']}/1000. Gains : {comparison['validity_gains']} ; pertes : {comparison['validity_losses']} ; changements de recommandation : {comparison['recommendation_changes']}. Qualités améliorées/dégradées : {comparison['quality_improvements']}/{comparison['quality_decreases']}.", ""]
    lines += table(["K", "B", "Valid cases", "Total", "Feasible rate"],
                   [[r["k"], r["b"], r["valid_cases"], r["total_cases"], f"{r['feasible_solution_rate']:.1%}"] for r in aggs])
    lines += ["", "Le taux mesure une solution dans le domaine fourni, pas une preuve d'infaisabilité globale. Les cas sans solution restent au dénominateur.", "", "## E. Performance", ""]
    if payload.get("no_solution_audit"):
        lines[-2:-2] = [
            f"Interprétation finale : {sum(r['valid_found'] for r in rows)} configurations VALID ; "
            f"{payload['no_solution_audit']['missing_configurations']} absences localement infaisables selon l'audit scalaire ; "
            f"{payload['no_solution_audit']['feasible_missed_groups']} domaine faisable connu manqué. "
            "Aucune preuve d'infaisabilité globale. Les absences ne constituent pas un taux d'échec Beam.", ""]
    lines += table(["K", "B", "mean s", "median s", "p90 s", "p95 s"],
                   [[r["k"], r["b"], *[f"{r[field]:.6f}" for field in ("runtime_mean", "runtime_median", "runtime_p90", "runtime_p95")]] for r in aggs])
    if payload.get("interleaved_runtime_check"):
        lines += ["", "Les temps bruts BEFORE/AFTER ont été mesurés sous des conditions d'exécution différentes ; les temps de l'exhaustif frozen changent également. Aucune attribution causale de leur différence au patch. Contrôle alterné dans le même processus, B=32, dix cas, deux répétitions :", ""]
        lines += table(["K", "Paires", "Temps BEFORE mean s", "Temps AFTER mean s", "Ratio AFTER/BEFORE médian"],
                       [[r["k"], r["pairs"], f"{r['mean_before_seconds']:.6f}", f"{r['mean_after_seconds']:.6f}",
                         f"{r['median_ratio_after_over_before']:.4f}"] for r in payload["interleaved_runtime_check"]["aggregates"]])
    lines += ["", "## F. Espace de recherche", "", "Branches, COMPLETE et H10 sont des moyennes par cas. Les pourcentages utilisent les branches générées totales comme dénominateur, préparation linéaire incluse. Plusieurs raisons peuvent expliquer une même branche hard-pruned ; les raisons ne sont pas additionnées pour calculer ce taux.", ""]
    lines += table(["K", "B", "branches mean", "branches median", "hard prune %", "width prune %", "COMPLETE mean", "H10 mean"],
                   [[r["k"], r["b"], f"{r['branches_mean']:.1f}", f"{r['branches_median']:.1f}", f"{r['hard_pruning_pct']:.2f}", f"{r['width_pruning_pct']:.2f}", f"{r['complete_mean']:.2f}", f"{r['h10_calls_mean']:.2f}"] for r in aggs])
    lines += ["", "## G. K=50", "", f"K=50 exécuté sur les {payload['dataset']['evaluated_cases']} cas et toutes les largeurs, soit {sum(r['requested_top_k'] == 50 for r in rows)} configurations. Aucune approximation exhaustive pour les domaines trop grands.", ""]
    lines += table(["B", "VALID/total", "FSR", "Runtime mean s", "p95 s", "Branches mean", "COMPLETE mean", "H10 mean"],
                   [[r["b"], f"{r['valid_cases']}/{r['total_cases']}", f"{r['feasible_solution_rate']:.1%}",
                     f"{r['runtime_mean']:.6f}", f"{r['runtime_p95']:.6f}", f"{r['branches_mean']:.1f}",
                     f"{r['complete_mean']:.2f}", f"{r['h10_calls_mean']:.2f}"] for r in aggs if r["k"] == 50])
    lines += ["", "## H. Sensibilité à B", "", f"Validité monotone : {summary['validity_monotone_groups']}/{summary['validity_groups']} groupes case×K. Changements d'architecture : {summary['architecture_change_groups']} groupes.", "",
              f"Groupes avec au moins une solution : {sum(any(w['valid'] for w in r['widths']) for r in payload['width_sensitivity'])}. Les autres groupes sont toujours sans solution : leur monotonie de validité est vraie par vacuité.", "",
              f"Qualité monotone : {summary['quality_monotone_groups']}/{summary['quality_comparable_groups']} groupes ayant au moins deux qualités comparables. Les autres groupes n'ont aucune conclusion de qualité.", "",
              "Tous les contre-exemples (paires de largeurs, architectures et qualités) figurent dans `width_sensitivity.json`.", ""]
    sensitive = [r for r in payload["width_sensitivity"] if not r["validity_monotonicity"] or r["selected_quality_monotonicity"] is False][:10]
    lines += table(["Case", "K", "Validité monotone", "Qualité monotone", "Changements", "Contre-exemples qualité"],
                   [[r["case_id"], r["k"], r["validity_monotonicity"], r["selected_quality_monotonicity"], r["architecture_changes"], str(r["quality_counterexamples"][:2])] for r in sensitive])
    lines += ["", "## I. Comparaison exhaustive", "", f"Statuts des {len(payload['baselines'])} préflights : {summary['baseline_status_counts']}. Comparaisons effectives : {summary['quality_comparisons']} configurations, {summary['unique_cases_quality_compared']} cas distincts.", "",
              "Un domaine est un cas × K × cap de chemins, indépendant de B ; une configuration ajoute B. Les 51 domaines entièrement parcourus comprennent les 50 K=5 et un K=10 : 36 domaines ont un VALID et donnent 36×5=180 configurations comparables, 15 n'ont aucun VALID. Les 149 autres domaines n'ont pas de référence exhaustive de qualité.", "",
              "Les analyses de search regret et de H9 normalization regret sont terminées sur toutes les comparaisons disponibles : analysis available on exhaustively tractable domains only. PARTIAL qualifie uniquement la couverture de validation (cas A), sans travail analytique inachevé sur les domaines traitables.", "",
              "La référence utilise le même Top-K et les mêmes chemins, H8/H9/H10 frozen, sans troncature de paires. Le pool H9 exhaustif inclut les architectures invalides ; le meilleur candidat et les rangs VALID sont pris après H10. Les scores de pools différents ne sont jamais comparés directement.", "",
              f"Qualité relative moyenne : {summary['mean_relative_quality']}; minimale : {summary['min_relative_quality']}; rang VALID commun maximal sélectionné : {summary['max_selected_common_rank']}. Meilleur exhaustif survivant : {summary['best_survived']}/{summary['quality_comparisons']}; sélectionné : {summary['best_chosen']}/{summary['quality_comparisons']}.", "",
              f"Regret moyen search : {summary['mean_search_regret']}; normalisation : {summary['mean_normalization_regret']}. Tous les rangs, indicateurs et regrets sont dans JSON et CSV, champs `quality.*`.", "",
              "`total_common_score_regret = search_common_score_regret + pool_normalization_common_score_regret`, vérifié pour chaque comparaison. Les domaines sautés n'ont ni qualité ni regret inventés. Les cas sans référence VALID sont classés séparément.", ""]
    quality_rows = []
    for aggregate_row in aggs:
        compared = [r["quality"] for r in rows if r["requested_top_k"] == aggregate_row["k"]
                    and r["beam_width"] == aggregate_row["b"] and r["quality"]["status"] == "COMPARED_IN_COMMON_POOL"]
        if compared:
            quality_rows.append([aggregate_row["k"], aggregate_row["b"], len(compared),
                                 f"{fmean(q['relative_quality'] for q in compared):.6f}",
                                 f"{min(q['relative_quality'] for q in compared):.6f}",
                                 median(q["selected_architecture_common_rank"] for q in compared),
                                 sum(q["exhaustive_best_survived_search"] for q in compared),
                                 sum(q["did_beam_find_exhaustive_best"] for q in compared),
                                 f"{fmean(q['search_common_score_regret'] for q in compared):.9f}",
                                 f"{fmean(q['pool_normalization_common_score_regret'] for q in compared):.9f}"])
    lines += table(["K", "B", "Comparaisons", "Qualité moyenne", "Qualité min", "Rang médian", "Best survived", "Best chosen", "Search regret mean", "H9 regret mean"], quality_rows)
    lines += ["", "## J. Origine des pertes", "", f"Dénominateur : {summary['quality_comparisons']} configurations comparables ; tolérance score {EPS}. Les taux portent sur case×K×B, pas sur des requêtes distinctes.", ""]
    lines += table(["Origine", "Configurations", "Part"], [[name, value["count"], f"{value['fraction']:.1%}" if value["fraction"] is not None else "n/a"] for name, value in summary["loss_distribution"].items()])
    lines += ["", "## K. Mémoire / caches", "", f"Les dictionnaires H5/H6 et les enveloppes ne font que croître : leur taille finale est leur maximum. Les compteurs publics sont utilisés pour chaque configuration et vérifiés par lecture des variables locales au retour d'un passage B=32 séparé pour chacun des {payload['memory_direct_checks']} couples case×K. Le résultat instrumenté doit égaler le résultat non instrumenté. Aucun changement du moteur. Mesure en entrées, sans estimation RSS ni borne en octets ; B borne la frontière, pas les caches ni la mémoire totale du processus.", ""]
    lines += table(["K", "Max H5 cache", "Max H6 cache", "Max beam states", "Max envelope count", "Max selection buffer", "Frontier entries after compaction"],
                   [[r["k"], r["max_max_cached_h5_entries"], r["max_max_cached_h6_entries"], r["max_max_beam_states"], r["max_role_completion_envelope_count"], r["max_max_selection_buffer"], r.get("max_resource_frontier_entries", 0)] for r in summary["memory"]])
    resource_bound = max(4*sum(r["lookahead_role_options"].values()) for r in rows)
    lines += ["", f"Les entrées de frontières sont mesurées après compaction. Une borne conservatrice avant compaction vaut quatre fois les options de rôle préparées, soit au maximum {resource_bound} entrées dans cette campagne ; ce nombre n'est pas une mesure RSS.", ""]
    lines += ["", "Max beam states désigne la frontière retenue (`max_active_states`), et non tous les objets vivants. L'ancienne frontière coexiste avec le buffer de sélection (au plus B+1). Les entrées de caches incluent aussi les échecs et les listes vides.", ""]
    lines += ["", "## L. Cas problématiques", "", "Les 10 exemples par catégorie sont dans `problem_cases.json` et `problem_cases.md` : absence de solution, sensibilité à B, regret search, regret H9, lenteur, branches. Les listes de cas sans solution, lents et avec regret ne répètent pas un même case_id. Les compteurs d'options permettent d'identifier les domaines sans option de complétion.", "",
              f"Référence exhaustive faisable mais Beam sans solution : {len(summary['feasible_exhaustive_but_beam_missing'])} configurations.", ""]
    if payload.get("no_solution_audit"):
        audit = payload["no_solution_audit"]
        lines += [f"Audit indépendant : {audit['audited_case_k_groups']} groupes, {audit['missing_configurations']} configurations sans solution, {audit['classification_counts']}. Témoins faisables manqués : {audit['feasible_missed_groups']}.", "",
                  "Audit détaillé dans `no_solution_audit.json/.md`. Il vérifie la faisabilité coût/puissance avec les fonctions publiques frozen, sans pool H9 et sans comparaison de qualité supplémentaire.", ""]
    examples = []
    for category, values in payload["problem_cases"].items():
        labels = []
        for r in values[:5]:
            if category == "width_sensitive":
                labels.append(f"{r['case_id']} K={r['k']} amplitude={r['relative_quality_range']} changements={r['architecture_changes']}")
            else:
                metric = r["quality"].get("search_common_score_regret") if category == "high_search_regret" else r["quality"].get("pool_normalization_common_score_regret") if category == "high_normalization_regret" else r["runtime_total"] if category == "slow" else r["branches_generated"] if category == "high_branches" else r.get("no_solution_classification", r["status"])
                labels.append(f"{r['case_id']} K={r['requested_top_k']} B={r['beam_width']} valeur={metric}")
        examples.append([category, "; ".join(labels)])
    lines += table(["Catégorie", "Cinq exemples"], examples)
    lines += ["", "## M. Conclusion algorithmique", "", payload.get("recommendation", "PENDING — évaluation à analyser avant décision"), "",
              payload.get("decision_rationale", ""), ""]
    if payload.get("original_v2_verdict"):
        lines += [f"Verdict historique, V2 originale uniquement : `{payload['original_v2_verdict']}`.", ""]
    lines += ["### Known limitations — not correctness bugs", "",
              "Optimalité/complétude non garanties avec B limité ; qualité non monotone avec B ; H9 normalisé par pool ; références exhaustives indisponibles pour les grands domaines ; 50 cas synthétiques seulement ; caches/enveloppes non bornés par B ; résultats dépendants des catalogues/dataset actuels. Aucun nouveau défaut de correction V2.1 n'est démontré pendant la revue finale.", ""]
    verification_path = HERE / "verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8")) if verification_path.exists() else {}
    for title, key in (("N. Tests finaux", "tests"), ("O. Git status final", "git_status"),
                       ("P. Git diff stat", "git_diff_stat"), ("Q. Git diff check", "git_diff_check")):
        lines += [f"## {title}", "", "```text", verification.get(key, "PENDING"), "```", ""]
    lines += ["## R. État final", "", "```text", verification.get("status_block", "PENDING"), "```", ""]
    (HERE / "extended_benchmark.md").write_text("\n".join(lines), encoding="utf-8")
    with (HERE / "extended_benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        flat_rows = []
        for row in rows:
            flat = {k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v
                    for k, v in row.items() if k not in ("quality", "memory", "search_trace", "lookahead_trace")}
            for section in ("quality", "memory"):
                flat.update({f"{section}.{k}": json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                             for k, v in row[section].items()})
            flat_rows.append(flat)
        writer = csv.DictWriter(handle, fieldnames=sorted(set().union(*(r.keys() for r in flat_rows))))
        writer.writeheader()
        writer.writerows(flat_rows)
    with (HERE / "aggregates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggs[0]))
        writer.writeheader()
        writer.writerows(aggs)
    problem_lines = [f"# Cas problématiques — Beam V{payload['beam_schema_version']}", "", "Références limitées au domaine configuré. Absence de solution ne prouve pas l'infaisabilité globale. Les regrets ne sont rapportés que dans un pool exhaustif commun.", ""]
    for name, values in payload["problem_cases"].items():
        problem_lines += [f"## {name}", ""]
        if name == "width_sensitive":
            problem_lines += table(["Case", "K", "Changements", "Amplitude qualité", "Validité monotone"],
                                   [[v["case_id"], v["k"], v["architecture_changes"], v["relative_quality_range"], v["validity_monotonicity"]] for v in values])
        else:
            problem_lines += table(["Case", "K", "B", "Statut", "Runtime s", "Branches", "Exhaustif VALID", "Search regret", "H9 regret"],
                                   [[v["case_id"], v["requested_top_k"], v["beam_width"], v["status"], f"{v['runtime_total']:.6f}", v["branches_generated"], v.get("exhaustive_valid_count"), v["quality"].get("search_common_score_regret"), v["quality"].get("pool_normalization_common_score_regret")] for v in values])
        problem_lines += [""]
    (HERE / "problem_cases.md").write_text("\n".join(problem_lines), encoding="utf-8")


def main():
    global HERE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-paths-per-variant", type=int, default=1)
    parser.add_argument("--exhaustive-max-pairs", type=int, default=2000)
    parser.add_argument("--resume-pilot", action="store_true", help="Reuse completed cases from the saved 100-case timing pilot")
    parser.add_argument("--output-directory", type=Path, default=HERE)
    parser.add_argument("--expected-beam-sha256", help="Permit only the documented Beam change with this exact digest")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    HERE = args.output_directory.resolve()
    HERE.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        write_report(json.loads((HERE / "extended_benchmark.json").read_text(encoding="utf-8")))
        return
    if min(args.sample_size, args.repeats, args.max_paths_per_variant, args.exhaustive_max_pairs) <= 0:
        parser.error("Numeric arguments must be positive")
    dataset_path = PROJECT_ROOT / "output/lustre_architecture_dataset.json"
    drives_path = PROJECT_ROOT / "data/catalogue_drives_ready_final.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    sample, dataset_info = select_cases(dataset, args.sample_size)
    dump(HERE / "sample_manifest.json", dataset_info)
    drive_catalog = json.loads(drives_path.read_text(encoding="utf-8"))
    hardware_catalog = existing.load_reference_catalog()
    input_paths = [dataset_path, drives_path, existing.DEFAULT_HARDWARE_CATALOG, Path(existing.__file__), Path(__file__)]
    payload = {
        "schema_version": "1.0", "beam_schema_version": BEAM_SCHEMA_VERSION,
        "heuristic_policy": BEAM_HEURISTIC_POLICY_ID, "dataset": dataset_info,
        "config": {"ks": list(KS), "widths": list(WIDTHS), "repeats": args.repeats,
                   "max_paths_per_variant": args.max_paths_per_variant, "exhaustive_max_pairs": args.exhaustive_max_pairs},
        "environment": {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
                        "logical_cpus": os.cpu_count(), "executable": sys.executable},
        "input_hashes": {str(p.relative_to(REPO_ROOT)): existing.sha256_file(p) for p in input_paths},
        "cases": [], "baselines": [], "handoffs": [], "completed": False,
    }
    started = perf_counter()
    memory_checks = 0
    reused_ids = set()
    if args.resume_pilot:
        pilot = json.loads((HERE / "pilot_100/checkpoint.json").read_text(encoding="utf-8"))
        for name in ("repeats", "max_paths_per_variant", "exhaustive_max_pairs"):
            if pilot["config"][name] != payload["config"][name]:
                raise ValueError(f"Incompatible pilot configuration: {name}")
        selected_ids = {case["case_id"] for case in sample}
        counts = Counter(row["case_id"] for row in pilot["cases"])
        reused_ids = {case_id for case_id in selected_ids if counts[case_id] == len(KS)*len(WIDTHS)}
        for name in ("cases", "baselines", "handoffs"):
            payload[name] = [row for row in pilot[name] if row["case_id"] in reused_ids]
        memory_checks = len(reused_ids)*len(KS)
        payload["pilot_reuse"] = {"case_ids": sorted(reused_ids), "pilot_wall_seconds": pilot["campaign_wall_seconds"],
                                 "original_input_hashes": pilot["input_hashes"],
                                 "reason": "Timing pilot took 20-43 seconds/case. Final deterministic stratified sample reduced from 100 to 50 before completing evaluation; selection uses no feasibility outcomes."}
    for case_index, case in enumerate(sample, 1):
        if case["case_id"] in reused_ids:
            print(f"{case_index}/{len(sample)} {case['case_id']} reused completed pilot case", flush=True)
            continue
        for k in KS:
            ranking_started = perf_counter()
            handoff = existing.build_runtime_handoff(architecture=case, catalog=drive_catalog, top_k=k)
            ranking_seconds = perf_counter() - ranking_started
            payload["handoffs"].append({"case_id": case["case_id"], "k": k,
                                        "actual_top_k": handoff["actual_top_k"], "ranking_seconds": ranking_seconds})
            reference = existing.exhaustive_reference(
                handoff=handoff, hardware_catalog=hardware_catalog,
                max_paths_per_variant=args.max_paths_per_variant, max_pairs=args.exhaustive_max_pairs,
            )
            if reference["status"] == "SKIPPED_PAIR_LIMIT":
                reference["status"] = "SKIPPED_DOMAIN_TOO_LARGE"
            payload["baselines"].append({"case_id": case["case_id"], "requested_top_k": k,
                                         **{key: val for key, val in reference.items() if key not in ("scores", "valid_scores")}})
            last_result = None
            for width in WIDTHS:
                wall_samples, engine_samples, first_result = [], [], None
                for repetition in range(args.repeats):
                    call_started = perf_counter()
                    result = existing.beam_search_architectures(
                        handoff=handoff, hardware_catalog=hardware_catalog, beam_width=width,
                        max_paths_per_variant=args.max_paths_per_variant,
                    )
                    wall_samples.append(perf_counter()-call_started)
                    engine_samples.append(result["summary"]["search_seconds"])
                    if first_result is None:
                        first_result = result
                    elif fingerprint(result) != fingerprint(first_result):
                        raise RuntimeError(f"Nondeterministic result: {case['case_id']}, K={k}, B={width}")
                result = first_result
                summary = result["summary"]
                row = {
                    "case_id": case["case_id"], "requested_top_k": k, "beam_width": width,
                    "actual_top_k": handoff["actual_top_k"], "status": result["status"],
                    "valid_found": result["best_validated_architecture"] is not None,
                    "runtime_total": median(wall_samples), "runtime_samples_seconds": wall_samples,
                    "engine_runtime_samples_seconds": engine_samples, "ranking_seconds": ranking_seconds,
                    "branches_generated": summary["branches_created"], "hard_pruned": summary["hard_pruned"],
                    "beam_width_pruned": summary["width_pruned"],
                    "complete_architectures": summary["complete_architectures_produced"],
                    "h10_calls": summary["h10_calls"], "h10_valid_count": summary["valid_architecture_count"],
                    "selected_architecture_id": summary["best_architecture_id"],
                    "selected_local_h9_score": summary["best_final_h9_score"],
                    "beam_search_applied": result["beam_search_applied"],
                    "peak_beam_states": summary["max_active_states"],
                    "pruning_reason_counts": summary["pruning_reason_counts"],
                    "main_search_branches_created": summary["main_search_branches_created"],
                    "lookahead_branches_created": summary["lookahead_branches_created"],
                    "lookahead_role_options": summary["lookahead_role_options"],
                    "search_trace": result["search_trace"], "lookahead_trace": result["lookahead_trace"],
                    "exhaustive_status": reference["status"],
                    "exhaustive_candidate_pairs": reference["potential_pairs"],
                    "exhaustive_valid_count": reference.get("valid_count"),
                    "quality": quality(result, reference),
                    "memory": {"max_cached_h5_entries": summary["h5_calls"],
                               "max_cached_h6_entries": summary["h6_calls"],
                               "max_beam_states": summary["max_active_states"],
                               "max_selection_buffer": summary["max_selection_buffer"],
                               "role_completion_envelope_count": sum(s["prefix_envelopes"] for s in result["lookahead_trace"])},
                    "deterministic_repeats": True,
                }
                row["memory"]["resource_frontier_entries"] = sum(s.get("resource_frontier_points", 0) for s in result["lookahead_trace"])
                assert row["complete_architectures"] <= width
                assert row["h10_calls"] == row["complete_architectures"]
                assert row["peak_beam_states"] <= width
                payload["cases"].append(row)
                last_result = result
            observed, observed_result = observe_caches(handoff, hardware_catalog, args.max_paths_per_variant)
            assert fingerprint(observed_result) == fingerprint(last_result)
            assert all(observed[key] == payload["cases"][-1]["memory"][key] for key in observed)
            memory_checks += 1
        payload["campaign_wall_seconds"] = perf_counter()-started
        dump(HERE / "checkpoint.json", payload)
        print(f"{case_index}/{len(sample)} {case['case_id']} rows={len(payload['cases'])} elapsed={payload['campaign_wall_seconds']:.1f}s", flush=True)
    payload["completed"] = True
    payload["memory_direct_checks"] = memory_checks
    payload["aggregates"] = aggregate(payload["cases"])
    payload["summary"], payload["width_sensitivity"], payload["problem_cases"] = analyze(payload["cases"], payload["baselines"])
    payload["actual_top_k_distribution"] = {
        str(k): {role: dict(Counter(str(r["actual_top_k"][role]) for r in payload["handoffs"] if r["k"] == k))
                 for role in ("mdt", "ost")} for k in KS}
    manifest = json.loads((HERE / "frozen_sources_initial.json").read_text(encoding="utf-8"))
    payload["frozen_source_changes"] = [p for p, digest in manifest.items() if existing.sha256_file(REPO_ROOT/p) != digest]
    engine_path = "lustre_architecture_generator/src/full_architecture/beam_search.py"
    documented_change = (args.expected_beam_sha256 is not None
                         and payload["frozen_source_changes"] == [engine_path]
                         and existing.sha256_file(REPO_ROOT/engine_path) == args.expected_beam_sha256)
    if payload["frozen_source_changes"] and not documented_change:
        raise RuntimeError(f"Frozen files changed: {payload['frozen_source_changes']}")
    if documented_change:
        payload["documented_beam_change"] = {"path": engine_path, "before_sha256": manifest[engine_path],
                                             "after_sha256": args.expected_beam_sha256}
    dump(HERE / "extended_benchmark.json", payload)
    dump(HERE / "width_sensitivity.json", payload["width_sensitivity"])
    dump(HERE / "problem_cases.json", payload["problem_cases"])
    write_report(payload)
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
