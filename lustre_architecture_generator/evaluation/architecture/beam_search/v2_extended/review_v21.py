"""Audit archived V2/V2.1 measurements and replay gains without a large H9 pool.

Run from the repository with its venv. This does not change production code,
delete files, stage files or commit. Large campaign JSON inputs stay local;
the compact release_review.json records independently checked evidence.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median

import run_extended_benchmark as extended
from full_architecture.full_architecture_validator import validate_complete_architecture


HERE = Path(__file__).resolve().parent
ROOT = extended.REPO_ROOT
ENGINE = extended.PROJECT_ROOT / "src/full_architecture/beam_search.py"
VERDICT = "BEAM V2.1 ACCEPTED WITH VALIDATION LIMITATIONS"


def read(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def keyed(rows):
    return {(r["case_id"], r["requested_top_k"], r["beam_width"]): r for r in rows}


def load_before():
    name = "full_architecture._beam_v2_release_review"
    spec = importlib.util.spec_from_file_location(name, HERE / "before_joint_fix/beam_search_v2_reference.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def audit():
    current = read("extended_benchmark.json")
    before = read("before_joint_fix/extended_benchmark.json")
    after = read("after_joint_fix/extended_benchmark.json")
    initial = read("release_review_initial.json")
    no_solution = read("no_solution_audit.json")
    old_rows, new_rows = keyed(before["cases"]), keyed(current["cases"])
    assert current["config"] == before["config"] == after["config"]
    assert current["dataset"] == before["dataset"] == after["dataset"]
    assert len(old_rows) == len(new_rows) == 1000 and old_rows.keys() == new_rows.keys()
    assert current["beam_schema_version"] == after["beam_schema_version"] == "2.1"
    assert before["beam_schema_version"] == "2.0"
    assert current["heuristic_policy"] == "BEAM_OPTIMISTIC_COMPLETION_V2_JOINT_RESOURCES"
    # The merged report adds a classification field only; all recorded metrics
    # and traces still equal the archived AFTER campaign, including timings.
    assert [{k: v for k, v in r.items() if k != "no_solution_classification"}
            for r in current["cases"]] == after["cases"]
    assert current["baselines"] == after["baselines"]
    assert current["summary"] == after["summary"]
    assert digest(ENGINE) == current["documented_beam_change"]["after_sha256"]
    assert digest(ENGINE) == initial["files"][ENGINE.relative_to(ROOT).as_posix()]["sha256"]
    before_path = HERE / "before_joint_fix/beam_search_v2_reference.py"
    assert digest(before_path) == current["documented_beam_change"]["before_sha256"]
    regenerated = subprocess.run(["git", "diff", "--no-index", "--unified=0", "--", str(before_path), str(ENGINE)],
                                 capture_output=True, text=True, encoding="utf-8")
    assert regenerated.returncode == 1
    assert regenerated.stdout.splitlines() == (HERE / "engine_change.diff").read_text(encoding="utf-8").splitlines()
    frozen = read("frozen_sources_initial.json")
    for name, sha in frozen.items():
        target = before_path if name == current["documented_beam_change"]["path"] else ROOT / name
        assert digest(target) == sha, name
    for phase_name, phase in (("before_joint_fix", before), ("after_joint_fix", after), ("", current)):
        for name, sha in phase["input_hashes"].items():
            target = HERE / phase_name / "campaign_runner_snapshot.py" if Path(name).name == "run_extended_benchmark.py" else ROOT / name
            assert digest(target) == sha, str(target)
    pilot = read("pilot_100/checkpoint.json")
    for name, sha in before["pilot_reuse"]["original_input_hashes"].items():
        target = HERE / "pilot_100/runner_source.py" if Path(name).name == "run_extended_benchmark.py" else ROOT / name
        assert digest(target) == sha, str(target)
    pilot_rows = keyed(pilot["cases"])
    reused_rows = [r for r in before["cases"] if r["case_id"] in before["pilot_reuse"]["case_ids"]]
    assert len(reused_rows) == 60
    assert all(r == pilot_rows[(r["case_id"], r["requested_top_k"], r["beam_width"])] for r in reused_rows)
    # Function bodies governing scores, tie breaks and validation are identical.
    old_ast, new_ast = ast.parse(before_path.read_text(encoding="utf-8")), ast.parse(ENGINE.read_text(encoding="utf-8"))
    for name in ("beam_heuristic", "_node_key", "_choice_key", "_validate_input", "_hard_reasons", "_bound_exceeds"):
        functions = [next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
                     for tree in (old_ast, new_ast)]
        assert ast.dump(functions[0], include_attributes=False) == ast.dump(functions[1], include_attributes=False), name
    engine_text = ENGINE.read_text(encoding="utf-8")
    assert "REQ_" not in engine_text and "evaluation" not in engine_text and "benchmark" not in engine_text
    with (HERE / "before_after.csv").open(encoding="utf-8", newline="") as handle:
        pairs = list(csv.DictReader(handle))
    assert len(pairs) == 1000
    for pair in pairs:
        key = (pair["case_id"], int(pair["k"]), int(pair["b"]))
        for label, rows in (("before", old_rows), ("after", new_rows)):
            for field in ("status", "selected_architecture_id"):
                assert pair[f"{label}_{field}"] == (rows[key][field] or "")
            assert float(pair[f"{label}_runtime_total"]) == rows[key]["runtime_total"]

    baselines = {(b["case_id"], b["requested_top_k"]): b for b in current["baselines"]}
    complete = {k: b for k, b in baselines.items() if b["status"] == "COMPLETE"}
    feasible = {k for k, b in complete.items() if b["valid_count"] > 0}
    scalar = {(r["case_id"], r["requested_top_k"]): r for r in no_solution["cases"]}
    missing = [r for r in current["cases"] if not r["valid_found"]]
    assert {(r["case_id"], r["requested_top_k"], r["beam_width"]) for r in missing} == {
        (r["case_id"], r["requested_top_k"], b) for r in scalar.values() for b in r["missing_widths"]}
    assert all(r["domain_proven_infeasible"] and not r["recovered_valid_architecture"] for r in scalar.values())
    exhaustive_misses = [key for key, row in new_rows.items() if key[:2] in feasible and not row["valid_found"]]
    quality_rows = [r for r in current["cases"] if r["quality"]["status"] == "COMPARED_IN_COMMON_POOL"]
    assert len(quality_rows) == len(feasible) * len(current["config"]["widths"]) == 180
    assert all(r["quality"] == old_rows[(r["case_id"], r["requested_top_k"], r["beam_width"])]["quality"] for r in quality_rows)
    comparable = []
    for r in quality_rows:
        reference = baselines[(r["case_id"], r["requested_top_k"])]
        comparable.append({"reduction": 1-r["complete_architectures"]/reference["potential_pairs"],
                           "speedup": reference["runtime_seconds"]/r["runtime_total"]})

    # Rebuild the official runtime handoff once. Replay both engines at all five
    # gained widths and call H10 independently on the actual selected records.
    gains = [key for key in sorted(new_rows) if not old_rows[key]["valid_found"] and new_rows[key]["valid_found"]]
    assert len(gains) == 5 and not any(old_rows[k]["valid_found"] and not new_rows[k]["valid_found"] for k in new_rows)
    dataset = json.loads((extended.PROJECT_ROOT / "output/lustre_architecture_dataset.json").read_text(encoding="utf-8"))
    drives = json.loads((extended.PROJECT_ROOT / "data/catalogue_drives_ready_final.json").read_text(encoding="utf-8"))
    hardware = extended.existing.load_reference_catalog()
    old_engine = load_before()
    handoffs = {}
    recovered = []
    for case_id, k, b in gains:
        if (case_id, k) not in handoffs:
            source = next(r for r in dataset if r["case_id"] == case_id)
            handoffs[(case_id, k)] = extended.existing.build_runtime_handoff(architecture=source, catalog=drives, top_k=k)
        h = handoffs[(case_id, k)]
        old = old_engine.beam_search_architectures(handoff=h, hardware_catalog=hardware, beam_width=b, max_paths_per_variant=1)
        new = extended.existing.beam_search_architectures(handoff=h, hardware_catalog=hardware, beam_width=b, max_paths_per_variant=1)
        assert old["status"] == old_rows[(case_id, k, b)]["status"] and old["best_validated_architecture"] is None
        selected = new["best_validated_architecture"]
        assert selected["architecture_id"] == new_rows[(case_id, k, b)]["selected_architecture_id"]
        decision = validate_complete_architecture(architecture=selected, handoff=h, hardware_catalog=hardware)
        assert decision["decision"] == selected["h10"]["decision"] == "VALID" and decision["valid"]
        assert new["summary"]["pruning_reason_counts"]["completion_joint_budget_power_conflict"] > 0
        renamed = copy.deepcopy(h)
        renamed["case_id"] = "RELEASE_REVIEW_RENAMED"
        renamed_result = extended.existing.beam_search_architectures(handoff=renamed, hardware_catalog=hardware, beam_width=b, max_paths_per_variant=1)
        renamed_selected = renamed_result["best_validated_architecture"]
        assert renamed_selected["h10"]["decision"] == "VALID"
        assert renamed_selected["state"]["selected"] == selected["state"]["selected"]
        recovered.append({"case_id": case_id, "k": k, "b": b, "v2_status": old["status"], "v21_status": new["status"],
                          "architecture_id": selected["architecture_id"], "independent_h10_decision": decision["decision"],
                          "joint_prunes": new["summary"]["pruning_reason_counts"]["completion_joint_budget_power_conflict"],
                          "renamed_case_same_physical_selection": True})
        print(f"Recovered: {case_id} K={k} B={b}: {selected['architecture_id']} H10={decision['decision']}", flush=True)

    review = {
        "review_date": "2026-10-03", "evidence_captured_utc": datetime.now(timezone.utc).isoformat(),
        "recommendation": VERDICT, "engine_unchanged_during_review": True,
        "provenance": {"engine_sha256": digest(ENGINE), "v2_sha256": digest(before_path),
                       "diff_matches_current_engine": True, "frozen_sources_verified": len(frozen),
                       "phase_input_hashes_verified": True, "root_after_metrics_identical": True,
                       "pilot_reused_rows_verified": len(reused_rows),
                       "pilot_runner_sha256": digest(HERE / "pilot_100/runner_source.py"),
                       "before_after_csv_verified": True, "config": current["config"],
                       "phase_versions": {"before": "2.0", "after": "2.1"},
                       "campaign_date": current["evaluation_date"],
                       "before_wall_seconds": before["campaign_wall_seconds"], "after_wall_seconds": after["campaign_wall_seconds"],
                       "review_script_sha256": digest(Path(__file__))},
        "recovered_configurations": recovered,
        "feasibility": {
            "configurations": len(new_rows), "valid": sum(r["valid_found"] for r in new_rows.values()),
            "known_feasible_domains_all_evidence": len({k[:2] for k, r in new_rows.items() if r["valid_found"]}),
            "known_globally_infeasible": 0, "known_locally_infeasible": len(missing), "known_feasible_missed": len(exhaustive_misses),
            "unknown_feasibility_after_scalar_audit": 0,
            "unknown_feasibility_from_full_quality_exhaustive_alone": sum(r["exhaustive_status"] != "COMPLETE" for r in missing),
            "full_quality_exhaustive_skipped_configurations": sum(r["exhaustive_status"] != "COMPLETE" for r in new_rows.values()),
            "skipped_quality_but_valid": sum(r["exhaustive_status"] != "COMPLETE" and r["valid_found"] for r in new_rows.values()),
            "local_infeasible_with_full_exhaustive": sum(r["exhaustive_status"] == "COMPLETE" for r in missing),
            "scalar_audited_domains": len(scalar), "scalar_classifications": dict(Counter(r["classification"] for r in scalar.values())),
        },
        "exhaustive": {"domains": len(baselines), "fully_evaluated": len(complete), "quality_skipped": len(baselines)-len(complete),
                       "feasible_domains_exhaustively_known": len(feasible),
                       "beam_recovered_feasible_domains": sum(all(new_rows[(*k, b)]["valid_found"] for b in current["config"]["widths"]) for k in feasible),
                       "beam_missed_feasible_domains": len({k[:2] for k in exhaustive_misses}),
                       "feasible_configurations_recovered": len(quality_rows),
                       "no_valid_reference_domains": len(complete)-len(feasible),
                       "fully_evaluated_by_k": dict(Counter(k[1] for k in complete)),
                       "architectures_fully_evaluated": sum(b["architectures_explored"] for b in complete.values())},
        "quality": {k: current["summary"][k] for k in ("quality_comparisons", "mean_relative_quality", "min_relative_quality", "mean_search_regret", "mean_normalization_regret", "best_survived", "best_chosen", "quality_monotone_groups", "quality_comparable_groups", "loss_distribution")},
        "performance_comparable": {"configurations": len(comparable), "materialization_reduction_min": min(r["reduction"] for r in comparable),
                                   "materialization_reduction_mean": fmean(r["reduction"] for r in comparable),
                                   "speedup_min": min(r["speedup"] for r in comparable), "speedup_median": median(r["speedup"] for r in comparable),
                                   "speedup_max": max(r["speedup"] for r in comparable)},
    }
    extended.dump(HERE / "release_review.json", review)
    print(json.dumps({k: review[k] for k in ("recommendation", "feasibility", "exhaustive", "performance_comparable")}, indent=2))


def recommendation(path):
    """Return an exact per-file commit category and rationale, without staging."""
    name = path.relative_to(ROOT).as_posix()
    if name == "README.md" or "/docs/beam_search" in name:
        return "Documentation", "Documentation actuelle ou historique identifié"
    if "/tests/" in name:
        return "Tests", "Régression Beam ou fixture immuable"
    if "/src/full_architecture/" in name:
        return "Production code", "Moteur V2.1 et exports publics"
    relative = path.relative_to(HERE.parent).as_posix()
    if relative == "v2_extended/pilot_100/runner_source.py":
        return "Evaluation scripts", "Runner exact du pilote : 60 configurations réutilisées dans BEFORE"
    if "/pilot_100/" in name:
        return "Local only", "Pilote partiel reproductible, hors campagne finale ; conserver sur disque"
    if path.name == "checkpoint.json":
        return "Local only", "Checkpoint reproductible, volumineux et/ou non final"
    if path.name in ("extended_benchmark.json",):
        return "Local only", "JSON détaillé > 7 MiB reproductible ; garder en archive externe avec SHA256"
    if "/after_joint_fix/" in name:
        return "Local only", "Copie brute AFTER conservée ; doublon des exports finaux utiles"
    if "/before_joint_fix/" in name:
        if path.suffix == ".py":
            return "Evaluation scripts", "Source V2 exacte ou runner historique pour reproduction"
        if path.name in ("extended_benchmark.csv", "extended_benchmark.md", "aggregates.csv",
                          "diagnostic_joint.json", "diagnostic_joint.md", "joint_feasible_drive_ranks.json",
                          "no_solution_audit.json", "no_solution_audit.md", "reproduction_tests.txt"):
            return "Benchmark artifacts worth tracking", "Preuve historique BEFORE, notamment témoin et tests rouges"
        return "Local only", "Export intermédiaire BEFORE reproductible ou log local"
    local_names = {"initial_git_status_v2.txt", "initial_git_status.txt", "initial_verification.json",
                   "verification.json", "git_status.txt", "git_diff_stat.txt", "git_diff_check.txt",
                   "final_unit_tests.txt", "final_global_tests.txt", "joint_fix_unit_tests.txt",
                   "release_review_initial.json", "release_tests.json", "release_dedicated_tests.txt",
                   "release_architecture_tests.txt", "release_global_tests.txt", "release_git_status.txt",
                   "release_git_diff_stat.txt", "release_git_diff_check.txt", "release_untracked.txt"}
    if path.name in local_names:
        return "Local only", "État de travail, chemins locaux ou log duplicable ; résumé final versionnable"
    if path.suffix == ".py":
        return "Evaluation scripts", "Script d'évaluation/revue ou snapshot de provenance"
    assert path.suffix in (".json", ".csv", ".md", ".diff"), relative
    return "Benchmark artifacts worth tracking", "Mesure compacte, diagnostic utile ou preuve de provenance"


def beam_files():
    paths = [ROOT / "README.md", ENGINE, extended.PROJECT_ROOT / "src/full_architecture/__init__.py"]
    paths += list((extended.PROJECT_ROOT / "docs").glob("beam_search*.md"))
    paths += list((extended.PROJECT_ROOT / "tests").glob("test_beam_search*.py"))
    paths += list((extended.PROJECT_ROOT / "tests/fixtures").glob("beam_*.json"))
    paths += [p for p in HERE.parent.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    return sorted(set(paths))


def git_result(args):
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    return {"command": "git " + " ".join(args), "exit_code": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def status_block(tests):
    return f"""BEAM SEARCH V2.1 FINAL STATUS

Algorithm status:
ACCEPTED WITH VALIDATION LIMITATIONS

Generic V2 defect fixed: YES

50-case extended validation: PASS
1000 configurations evaluated: YES
K=50 evaluated: YES

Known feasible exhaustive domains missed: 0

Dedicated tests: {tests['dedicated']['result']}
Global tests: {tests['global']['result']}

H5-H10 modified: NO
Ranking modified: NO
Case-specific hacks: NO

Documentation current with V2.1: YES

Ready for a focused Beam Search commit:
YES

Remaining limitations:
- No global optimality/completeness guarantee with bounded B.
- Quality can be nonmonotone with B; H9 normalization depends on its pool.
- Quality/regret unavailable for 149 large domains; analysis completed on all tractable domains.
- Validation covers 50 synthetic cases and the current dataset/catalogues.
- Caches/envelopes and total memory are not bounded by B."""


def table(headers, rows):
    return extended.table(headers, rows)


def report():
    review = read("release_review.json")
    tests = read("release_tests.json")
    assert all(test["exit_code"] == 0 for test in tests.values())
    paths = beam_files()
    initial = read("release_review_initial.json")
    # Preserve all historical measurements byte for byte, including large local
    # files, and every production/test source already present at review start.
    unchanged = []
    for name, entry in initial["files"].items():
        if any(part in name for part in ("/before_joint_fix/", "/after_joint_fix/", "/pilot_100/", "/tests/", "/src/full_architecture/")):
            assert digest(ROOT / name) == entry["sha256"], name
            unchanged.append(name)
    assert digest(ENGINE) == review["provenance"]["engine_sha256"]
    review["tests"] = tests
    review["preserved_historical_and_source_files"] = unchanged
    review["commit_files"] = {category: [p.relative_to(ROOT).as_posix() for p in paths if recommendation(p)[0] == category]
                              for category in ("Production code", "Tests", "Documentation", "Evaluation scripts", "Benchmark artifacts worth tracking")}
    review["local_only_files"] = [{"path": p.relative_to(ROOT).as_posix(), "reason": recommendation(p)[1]}
                                  for p in paths if recommendation(p)[0] == "Local only"]
    git = {"status": git_result(["status", "--short"]), "stat": git_result(["diff", "--stat"]),
           "check": git_result(["diff", "--check"]), "untracked": git_result(["ls-files", "--others", "--exclude-standard"])}
    assert all(r["exit_code"] == 0 for r in git.values())
    deletions = [line for line in git["status"]["stdout"].splitlines() if line.startswith(" D ")]
    assert deletions == [line for line in initial["git_status"].splitlines() if line.startswith(" D ")]
    assert len(deletions) == 59
    review["git"] = git
    review["preexisting_unrelated_deletions_preserved"] = len(deletions)
    review["benchmark_artifact_sha256"] = {
        p.relative_to(HERE.parent).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p)}
        for p in HERE.parent.rglob("*")
        if p.is_file() and p.suffix in (".json", ".csv", ".diff")
        and not p.name.startswith(("release_", "initial_")) and p.name != "verification.json"
    }
    extended.dump(HERE / "release_review.json", review)
    for name, key in (("release_git_status.txt", "status"), ("release_git_diff_stat.txt", "stat"),
                      ("release_git_diff_check.txt", "check"), ("release_untracked.txt", "untracked")):
        (HERE / name).write_text(git[key]["stdout"]+git[key]["stderr"], encoding="utf-8")
    verification = read("verification.json")
    verification["tests"] = "\n\n".join(t["command"]+"\n"+t["stdout"]+t["stderr"] for t in tests.values())
    verification.update({"git_status": git["status"]["stdout"], "git_diff_stat": git["stat"]["stdout"]+git["stat"]["stderr"],
                         "git_diff_check": git["check"]["stdout"]+git["check"]["stderr"],
                         "status_block": status_block(tests), "review_date": review["review_date"]})
    extended.dump(HERE / "verification.json", verification)
    extended.write_report(read("extended_benchmark.json"))

    criteria = [
        ("Deterministic search", "PASS", "1 000/1 000 fingerprints de répétitions identiques ; test déterminisme et cinq gains rejoués"),
        ("No H5 violation", "PASS", "Module frozen par SHA256 ; variantes uniquement publiques H5 ; tests géométries et H10"),
        ("No H6 violation", "PASS", "Module frozen ; chemins publics H6 ; tests compatibilité et recomputation H10"),
        ("H7 contract preserved", "PASS", "EMPTY puis transitions jointes publiques ; tests states partiels et COMPLETE"),
        ("H8 contract preserved", "PASS", "Constructeur public et architecture_id inchangés ; aucun produit cartésien anticipé"),
        ("H9 unchanged", "PASS", "Hash H9 frozen ; formule Beam et tie-breaks comparés par AST"),
        ("H10 mandatory", "PASS", "COMPLETE == H10 calls ≤ B sur 1 000 lignes ; chaque sélection rejouée passe H10 indépendant"),
        ("No partial state marked VALID", "PASS", "Tests monkeypatch sur scoring/validation ; seul validated_state H10 devient terminal"),
        ("Hard pruning physically admissible", "PASS", "Preuve Pareto/additivité, 300 comparaisons aléatoires avec produit scalaire brut et frontière flottante"),
        ("No case-specific hacks", "PASS", "Aucun ID REQ/benchmark dans le moteur ; renommage aux cinq largeurs conserve les choix physiques"),
        ("Global regression suite passes", "PASS", tests["global"]["result"]),
        ("Extended dataset evaluated", "PASS", "50 nouveaux cas stratifiés, 1 000 configurations, deux répétitions chacune"),
        ("K=50 evaluated", "PASS", "250 configurations K=50 ; 42/50 cas VALID à chacune des cinq largeurs"),
        ("Known feasible exhaustive domains recovered", "PASS", "36/36 domaines complets faisables, 180/180 configurations ; audit des absences : zéro témoin manqué"),
        ("No quality regression introduced", "PASS", "180/180 objets quality identiques V2/V2.1 ; hors référence exhaustive aucune assertion de qualité"),
        ("Search-space reduction remains significant", "PASS", "COMPLETE évités : minimum 94,0741 %, moyenne 98,4671 % sur 180 comparaisons"),
        ("Runtime substantially below exhaustive where comparable", "PASS", "Rapport exhaustive/Beam : minimum 6,32×, médiane 41,14×, maximum 224,29× ; ranking/préflight exclus"),
        ("Global quality/regret validation coverage", "PARTIAL", "Analyse terminée ; références de qualité absentes pour 149 grands domaines"),
        ("Known global infeasibility proof", "N/A", "Aucune preuve globale produite ; les 215 absences sont localement infaisables")]
    lines = ["# Beam Search V2.1 — revue finale A–R", "", "Revue du 3 octobre 2026. Aucun commit, staging ou suppression effectué. Le moteur candidat est conservé byte pour byte.", "",
             "## A. Verdict V2.1", "", f"**{VERDICT}**.", "",
             "Aucun nouveau bug algorithmique, prune non admissible, nondéterminisme ou violation de contrat n'est démontré. Les limites de qualité/portée ne justifient pas une nouvelle optimisation pendant cette clôture. Le verdict historique REQUIRES FURTHER ALGORITHMIC WORK concernait exclusivement la V2 originale.", "",
             "## B. Correction V2.1", "",
             "**Cause → correction → preuve.** En V2, la condition nécessaire `min(cost_MDT)+min(cost_OST) ≤ budget` et son analogue puissance étaient admissibles séparément, mais leurs minima pouvaient venir d'options différentes. Leur réussite n'établissait aucune complétion respectant simultanément les deux contraintes. Des préfixes irréalisables pouvaient occuper toutes les places B ; le beam-width pruning éliminait alors les véritables branches faisables. Il ne s'agissait pas d'une ancienne borne dure rejetant directement des solutions valides.", "",
             "V2.1 ajoute, après ces bornes et avant l'heuristique/sélection de largeur, la raison dure `completion_joint_budget_power_conflict`. Pour chaque préfixe H5/H6, elle conserve des couples réels (coût, puissance) non dominés. Un point dominé a un remplaçant avec coût et puissance inférieurs ou égaux : l'existence d'une complétion satisfaisant ces deux contraintes est préservée. Une frontière ordonnée par coût croissant a une puissance strictement décroissante ; pour chaque MDT, le dernier OST abordable est le plus sobre des OST abordables. Le balayage à deux indices est O(f_MDT+f_OST), sans produit cartésien d'architectures COMPLETE.", "",
             "Il s'agit d'un **hard pruning supplémentaire physiquement admissible des complétions impossibles**, qui protège la largeur des préfixes fantômes. Les lower bounds indépendantes, l'heuristique, ses poids et les tie-breaks ne changent pas. Le test reste une condition nécessaire de ressources, pas une décision VALID. Les marges fixes `max(1e-12, 8*ulp(limite))` autorisent conservativement les frontières arrondies ; H10 recompute H5/H6 et décide exactement selon son contrat inchangé.", "",
             "Le code n'utilise ni case_id, ni ID de drive spécifique, ni imports d'évaluation pour cette règle. Les enveloppes proviennent de chaque entrée actuelle. Les cinq replays avec un case_id renommé conservent les choix physiques. Deux reproductions historiques échouaient en V2 ; les régressions Pareto, jointes, flottantes et les 300 confrontations avec une recherche scalaire brute passent en V2.1.", "",
             "Provenance logique : BEFORE schéma 2.0 → patch exact → AFTER schéma 2.1 → combinaison des deux phases → revue finale du même hash. La date de campagne et les temps historiques sont conservés ; la date UTC de capture dans release_review.json identifie l'audit, pas une nouvelle campagne. Les dates de fichiers ne servent pas à établir les versions.", "",
             f"SHA256 V2 : `{review['provenance']['v2_sha256']}`.", "",
             f"SHA256 V2.1 : `{review['provenance']['engine_sha256']}`.", "",
             "Le diff régénéré entre la source V2 archivée et le moteur courant correspond ligne pour ligne à `engine_change.diff`. Les 42 sources/artefacts frozen, les entrées des trois rapports et les snapshots exacts des runners sont vérifiés par SHA256. Tous les nombres/traces/temps du rapport combiné sont identiques à AFTER ; seules les métadonnées éditoriales du verdict sont actualisées. `before_after.csv` est confronté aux deux campagnes, 1 000 lignes. Les manifests de rankers officiels sont contrôlés par check_artifacts.py. Les 60 configurations de trois cas réutilisées du pilote dans BEFORE sont identiques au checkpoint pilote ; son runner exact et ses hashes sont également vérifiés. AFTER ne réutilise aucun cas du pilote.", "",
             "## C. Five recovered configurations", "",
             "Statuts complets : NO VALID = `NO_VALID_ARCHITECTURE_FOUND_WITHIN_SEARCH_DOMAIN` ; VALID = `VALID_ARCHITECTURE_FOUND`. H10 est recalculé par la fonction publique indépendamment de la décision stockée par le Beam.", ""]
    lines += table(["case_id", "K", "B", "V2 status", "V2.1 status", "Selected architecture_id", "H10 decision"],
                   [[r["case_id"], r["k"], r["b"], "NO VALID", "VALID", r["architecture_id"], r["independent_h10_decision"]] for r in review["recovered_configurations"]])
    lines += ["", "Le témoin initial ARCH_REQ_000267_e62f7c71ba739177 avait un drive MDT de rang 44, hors B≤32. Seuls les rangs MDT 44/47/48/50 possédaient une complétion jointe faisable ; les 32 préfixes V2 retenus étaient irréalisables. La nouvelle règle élimine ces préfixes avant la largeur, sans priorité de cas ni modification de contraintes. Les sélections finales peuvent différer du témoin et entre B, car H9 classe le pool effectivement retenu.", "",
              "## D. Feasibility interpretation", "",
              "785/1 000 = 78,5 % mesure la présence d'une solution H10 VALID dans le domaine configuré. Les 21,5 % d'absences ne sont pas un taux d'échec de recherche.", ""]
    lines += table(["Catégorie", "Configurations", "Preuve/portée"], [
        ["Beam VALID", 785, "180 avec référence exhaustive commune ; 605 sur domaines sans référence de qualité"],
        ["Known globally infeasible", 0, "Aucune preuve globale ; Top-K/cap limitent le domaine"],
        ["Known locally infeasible", 215, "75 selon H8/H9/H10 complets ; 140 supplémentaires selon audit scalaire"],
        ["Known feasible but Beam missed", 0, "Références faisables et audit de toutes les absences"],
        ["Unknown feasibility after scalar audit", 0, "Toutes les absences classées, dans le domaine configuré"],
        ["Unknown feasibility without scalar audit", 140, "Vue limitée aux seuls pools H8/H9/H10 complets ; levée par l'audit scalaire"]])
    lines += ["", "Les quatre catégories VALID/globalement infaisable/faisable manqué/inconnu omettraient ici l'infaisabilité **locale**, qui doit être ajoutée pour obtenir une partition complète : 785+215+0+0+0=1 000. Les 745 configurations dont la qualité exhaustive est sautée se décomposent en 605 VALID et 140 localement infaisables. Qualité inconnue ne signifie pas faisabilité inconnue. Les 43 domaines sans solution sont tous prouvés infaisables localement : 11 budget+puissance, 16 puissance, 10 conflit joint, 6 budget. Les 157 autres domaines ont une preuve positive H10 (36 avec qualité exhaustive, 121 sans cette référence de qualité).", "",
              "## E. Exhaustive audit", ""]
    lines += table(["Terme", "Définition", "Nombre par phase"], [
        ["case", "Requête avec exigences/contraintes/préférences", 50],
        ["configuration", "case × K × B × cap de chemins", 1000],
        ["domain", "case × K × cap de chemins, indépendant de B", 200],
        ["exhaustively evaluated domain", "Toutes les paires H8, tout le pool H9, H10 sur chaque architecture", 51],
        ["configuration comparable", "Référence exhaustive VALID et choix Beam VALID dans ce même pool H9", 180]])
    lines += ["", "Par phase : 50 domaines K=5 et un K=10 (REQ_001042), **42 840 architectures** entièrement construites/scorées/validées, 149 domaines de qualité sautés au-delà de 2 000 paires. Le K=10 complet n'a aucun VALID. Les 36 domaines faisables sont tous à K=5 ; les 15 autres références complètes n'ont aucun VALID. Ce sont des domaines Top-K, sans preuve globale.", "",
              "```text", "feasible_domains_exhaustively_known = 36", "beam_recovered_feasible_domains = 36 (all five widths)", "beam_missed_feasible_domains = 0", "comparable_configurations = 36 * 5 = 180", "```", "",
              "L'audit scalaire complémentaire porte sur 43 domaines/215 configurations sans solution après correction, sans pool H9. BEFORE comportait un domaine faisable manqué supplémentaire (cinq configurations K=50). AFTER n'en comporte aucun. Ne pas ajouter les domaines scalaires aux 51 références exhaustives de qualité : les ensembles se recoupent et leurs objectifs diffèrent.", "",
              "## F. Search regret / normalization regret", "",
              "**Cas A : analyse terminée sur tous les domaines traitables ; couverture de validation PARTIAL.** Aucun travail analytique attendu n'est laissé inachevé sur les références disponibles. Formulation : **analysis available on exhaustively tractable domains only**. Les 15 domaines complets sans VALID n'ont pas de regret défini ; les 149 domaines trop grands n'ont pas de pool commun et aucun regret n'est inventé.", "",
              "Les 180 comparaisons séparent `search = best exhaustive VALID − best retained VALID in common pool` et `normalization = best retained VALID in common pool − actual local H9 winner in common pool`. Leur somme égale le regret total pour chaque ligne. Moyennes : recherche 0,010925760807154445 ; normalisation 0,003824268361532696. Qualité moyenne 0,982148422072597, minimum 0,8142958377766908 ; 97/180 meilleurs exhaustifs survivent, 36/180 sont choisis. Pertes : recherche seule 63, normalisation seule 61, les deux 20, aucune 36. Tous ces objets quality sont inchangés entre V2 et V2.1.", "",
              "## G. Acceptance criteria", ""]
    lines += table(["Criterion", "Result", "Evidence"], criteria)
    lines += ["", "PASS s'applique à la preuve et au périmètre vérifiés ; il ne promet pas une propriété générale de complétude/optimalité. La comparaison de temps se fait dans la campagne AFTER avec sa baseline exhaustive, sur 180 configurations comparables. La variation brute de temps entre campagnes ne prouve aucun gain causal du patch ; le contrôle alterné existant sur 40 couples est conservé.", "",
              "## H. Tests finaux", "",
              "Une seule campagne finale propre : les trois commandes ci-dessous exécutées une fois chacune, séquentiellement, avec le venv du dépôt. Les répertoires temporaires sont distincts ; le cache pytest est désactivé. Aucun test n'est ajouté pendant cette revue documentaire.", ""]
    for key in ("dedicated", "architecture", "global"):
        t = tests[key]
        lines += [f"**{key} : {t['result']}**, exit code {t['exit_code']}.", "", "```powershell", t["command"], "```", "", "```text", t["stdout"].rstrip(), t["stderr"].rstrip(), "```", ""]
    lines += ["## I. Documentation changes", "",
              "README.md : résumé courant V2.1, anciennes tables déplacées vers leurs liens historiques, tests finaux et limites sans taux d'échec inventé. docs/beam_search.md : référence actuelle, politique 2.1, prune joint, domaines/audits et limitations. docs/beam_search_v2.md : bannière historique V1→V2 conservant ses mesures. docs/beam_search_v2_extended.md : verdict courant, défaut historique et regrets terminés sur la portée disponible.", "",
              "v2_extended/extended_benchmark.md : titre courant, verdict V2.1, verdict original séparé, définitions et limites, tests/Git final actualisés. v2_extended/problem_cases.md : titre identifié V2.1 pour les exports finaux. release_review.md : ce rapport unique de clôture A–R. Le JSON principal ne change que recommendation/original_v2_verdict/decision_rationale. Les scripts combine_before_after.py et run_extended_benchmark.py gardent ces formulations lors d'une régénération. Les raw phase reports restent des archives de collecte, éventuellement avec placeholders éditoriaux historiques ; ils ne sont pas le verdict courant.", "",
              "Structure recommandée : README bref → docs/beam_search.md actuel → evaluation/ pour preuves historiques et clôture. Conserver beam_search_v2.md comme historique ; fusion éventuelle lors d'un futur cleanup. Aucun artifact supprimé.", "",
              "## J. Files for future commit", "",
              "Liste exacte recommandée, relative à la racine du dépôt. Aucun git add/commit exécuté. Cette allowlist inclut les fichiers non suivis absents du diff stat. Le script de revue utilise les grands JSON locaux pour recalculer ses preuves ; leur archivage externe doit préserver les SHA256 avant un futur cleanup.", ""]
    for category, names in review["commit_files"].items():
        lines += [f"### {category}", "", "```text", *names, "```", ""]
    lines += ["`diagnostic_v1.json` et `diagnostic_v2.json` sont retenus : petits diagnostics historiques utiles (préfixes, identités et regrets). `engine_change.diff` est retenu : preuve compacte du correctif, régénérable avec la source V2 conservée. Les snapshots des runners conservent la provenance des hash de campagnes malgré l'évolution de leur rendu documentaire.", "",
              "## K. Files not recommended for commit", "",
              "Recommandations seulement : tous ces fichiers restent présents. Les JSON > 1 MiB sont signalés ; les trois gros JSON de campagne font chacun > 7 MiB. Préférer archive externe avec hashes et regeneration plutôt qu'un commit de dumps/duplications/checkpoints. Les tailles et SHA256 des artefacts de données, y compris les grands fichiers proposés hors Git, sont enregistrés dans release_review.json, champ benchmark_artifact_sha256.", ""]
    lines += table(["Fichier exact", "Raison"], [[r["path"], r["reason"]] for r in review["local_only_files"]])
    lines += ["", "Les 59 suppressions non liées au Beam sont également exclues du futur commit ; leur liste exacte est visible en N. initial_git_status_v2.txt et les snapshots/logs Git locaux ne sont pas des résultats scientifiques à versionner. Le rapport et release_review.json conservent les résultats nécessaires à la revue.", "",
              "## L. Artifact sizes", "",
              "Inventaire récursif de **chaque** .json/.csv/.md/.diff sous evaluation/architecture/beam_search/, en octets et KiB (1 KiB=1 024 octets). Racine commune : lustre_architecture_generator/evaluation/architecture/beam_search/. Les documents Beam et le fixture sont inclus séparément par leur chemin complet. Les gros fichiers et les duplications reproductibles sont exclus de l'allowlist, sans suppression.", ""]
    artifacts = [p for p in beam_files() if p.suffix in (".json", ".csv", ".md", ".diff") and p.name != "README.md"]
    size_insert = len(lines)
    lines += ["", "## M. Known limitations", "",
              "### Known limitations — not correctness bugs", "",
              "- Optimalité globale/complétude non garanties avec une largeur B limitée et sans backtracking.",
              "- Qualité non monotone avec B ; 9/36 groupes comparables monotones seulement.",
              "- H9 dépend du pool, sans modification de ses normalisateurs.",
              "- Références exhaustives de qualité sautées pour 149 grands domaines ; regrets terminés seulement sur les traitables.",
              "- 50 cas synthétiques stratifiés seulement ; catalogue/dataset/rankers actuels, sans validation en production.",
              "- Caches/enveloppes et mémoire totale non bornés par B ; mesures en entrées, sans RSS.",
              "- Top-K et cap de chemins peuvent exclure des solutions hors domaine.", "",
              "**Defects :** aucun défaut actuel V2.1 démontré. Le défaut historique V2 était la perte par largeur de branches réellement faisables lorsque des minima indépendants laissaient survivre des préfixes fantômes ; il est corrigé et couvert par les régressions. Les limitations ci-dessus ne sont pas des bugs de correction.", "",
              "## N. Git status", "",
              "Beam-related : README, exports publics, moteur, tests/fixture, docs et evaluation Beam. Pre-existing unrelated : **59 suppressions**, exactement identiques à l'état initial ; elles n'ont pas été touchées. Pas de staging. Sortie exacte :", "", "```text", git["status"]["stdout"].rstrip(), "```", "",
              "Fichiers untracked, sortie exacte de `git ls-files --others --exclude-standard` (ils appartiennent tous au périmètre Beam dans cet état de travail, y compris les locaux déconseillés) :", "", "```text", git["untracked"]["stdout"].rstrip(), "```", "",
              "## O. Git diff stat", "", "Commande `git diff --stat`, exit code 0. stdout puis stderr exacts. Le stat comprend les suppressions préexistantes et omet les fichiers non suivis, listés en N et classés en J/K.", "", "```text", (git["stat"]["stdout"]+git["stat"]["stderr"]).rstrip(), "```", "",
              "## P. Git diff check", "", "Commande `git diff --check`, exit code 0 : **PASS** (aucune erreur whitespace). stdout vide ; stderr exact ci-dessous contient uniquement les avertissements LF→CRLF.", "", "```text", (git["check"]["stdout"]+git["check"]["stderr"]).rstrip(), "```", "",
              "## Q. Suggested commit message", "", "Suggested commit title:", "", "```text", "feat(beam): finalize V2.1 joint-resource pruning and validation", "```", "", "Suggested commit body:", "", "```text",
              "Add deterministic bounded Beam Search with generic Pareto cost/power completion pruning.",
              "Keep H5-H10, ranking, heuristic weights and validity contracts unchanged.",
              "Validate V2.1 on 50 synthetic cases / 1,000 configurations, including K=50:",
              "785 H10 VALID selections, five recoveries, zero losses and no known feasible domain missed.",
              "Preserve quality on all 180 comparable configurations and document validation limitations.",
              "Verify 76 dedicated, 303 architecture and 509 global tests.",
              "Track focused code, fixtures, documentation and compact reproducibility evidence.", "```", "",
              "Proposition uniquement. Aucun commit effectué ; les suppressions préexistantes et artifacts locaux sont exclus.", "",
              "## R. Final status", "", "```text", status_block(tests), "```", ""]
    output = HERE / "release_review.md"
    # Include this report's own size without an inaccurate provisional count.
    for _ in range(8):
        size_rows = []
        for path in artifacts:
            size = path.stat().st_size
            label = path.relative_to(HERE.parent).as_posix() if path.is_relative_to(HERE.parent) else path.relative_to(ROOT).as_posix()
            size_rows.append([label, size, f"{size/1024:.2f}", "LARGE ≥1 MiB" if size >= 1024**2 else "", recommendation(path)[0]])
        rendered = lines[:size_insert]+table(["Fichier", "Octets", "KiB", "Volume", "Recommandation"], size_rows)+lines[size_insert:]
        text = "\n".join(rendered)
        previous_size = output.stat().st_size
        output.write_text(text, encoding="utf-8")
        if output.stat().st_size == previous_size:
            break
    else:
        raise AssertionError("Artifact size table did not stabilize")
    print(f"Final review: {output.relative_to(ROOT)}; {len(paths)} files classified; {len(artifacts)} sizes; 59 deletions preserved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true", help="Render A-R using already captured audit/tests; do not replay gains")
    args = parser.parse_args()
    if args.report_only:
        report()
    else:
        audit()
