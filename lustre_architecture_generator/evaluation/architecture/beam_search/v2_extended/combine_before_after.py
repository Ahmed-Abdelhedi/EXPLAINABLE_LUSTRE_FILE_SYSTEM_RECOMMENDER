"""Publish final measured results while preserving both complete campaigns."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess

import run_extended_benchmark as extended


def main():
    root = extended.HERE
    before = json.loads((root/"before_joint_fix/extended_benchmark.json").read_text(encoding="utf-8"))
    after = json.loads((root/"after_joint_fix/extended_benchmark.json").read_text(encoding="utf-8"))
    audit = json.loads((root/"after_joint_fix/no_solution_audit.json").read_text(encoding="utf-8"))
    assert before["completed"] and after["completed"]
    assert before["dataset"]["selected_cases"] == after["dataset"]["selected_cases"]
    assert before["config"] == after["config"]
    assert audit["feasible_missed_groups"] == 0
    before_rows = {(r["case_id"], r["requested_top_k"], r["beam_width"]): r for r in before["cases"]}
    after_rows = {(r["case_id"], r["requested_top_k"], r["beam_width"]): r for r in after["cases"]}
    assert before_rows.keys() == after_rows.keys()
    timing_keys = {"runtime_seconds", "preflight_seconds"}
    strip = lambda r: {k: v for k, v in r.items() if k not in timing_keys}
    assert [strip(r) for r in before["baselines"]] == [strip(r) for r in after["baselines"]]
    paired = []
    for key in sorted(before_rows):
        old, new = before_rows[key], after_rows[key]
        row = {"case_id": key[0], "k": key[1], "b": key[2]}
        for phase, value in (("before", old), ("after", new)):
            for field in ("valid_found", "status", "selected_architecture_id", "runtime_total", "branches_generated",
                          "hard_pruned", "beam_width_pruned", "complete_architectures", "h10_calls"):
                row[f"{phase}_{field}"] = value[field]
            for field in ("relative_quality", "search_common_score_regret", "pool_normalization_common_score_regret"):
                row[f"{phase}_{field}"] = value["quality"].get(field)
        paired.append(row)
    comparison = {
        "configurations": len(paired),
        "before_valid": before["summary"]["valid_configurations"],
        "after_valid": after["summary"]["valid_configurations"],
        "validity_gains": sum(not r["before_valid_found"] and r["after_valid_found"] for r in paired),
        "validity_losses": sum(r["before_valid_found"] and not r["after_valid_found"] for r in paired),
        "recommendation_changes": sum(r["before_selected_architecture_id"] != r["after_selected_architecture_id"] for r in paired),
        "quality_improvements": sum(r["before_relative_quality"] is not None and r["after_relative_quality"] > r["before_relative_quality"]+extended.EPS for r in paired),
        "quality_decreases": sum(r["before_relative_quality"] is not None and r["after_relative_quality"] < r["before_relative_quality"]-extended.EPS for r in paired),
        "before_feasible_missed_groups": 1, "after_feasible_missed_groups": audit["feasible_missed_groups"],
        "runtime_note": "Separate campaigns experienced different execution conditions; frozen exhaustive runtimes also changed. Raw timing differences are not causal algorithm speedups. See same-process alternating interleaved_runtime.json.",
    }
    with (root/"before_after.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    after["before_phase"] = {"path": "before_joint_fix/extended_benchmark.json",
                             "beam_schema_version": before["beam_schema_version"],
                             "heuristic_policy": before["heuristic_policy"], "summary": before["summary"],
                             "pilot_reuse": before.get("pilot_reuse")}
    after["before_after_summary"] = comparison
    after["before_after"] = paired
    after["no_solution_audit"] = {k: v for k, v in audit.items() if k != "cases"}
    classification = {(r["case_id"], r["requested_top_k"]): r["classification"] for r in audit["cases"]}
    for row in after["cases"]:
        if not row["valid_found"]:
            row["no_solution_classification"] = classification[(row["case_id"], row["requested_top_k"])]
    for row in after["problem_cases"]["no_solution"]:
        row["no_solution_classification"] = classification[(row["case_id"], row["requested_top_k"])]
    after["original_v2_verdict"] = "BEAM V2 REQUIRES FURTHER ALGORITHMIC WORK"
    after["recommendation"] = "BEAM V2.1 ACCEPTED WITH VALIDATION LIMITATIONS"
    after["decision_rationale"] = (
        "La V2 originale ne peut pas être acceptée sans changement : elle élimine tous les préfixes faisables d'un domaine pourtant H10 VALID. "
        "Les résultats BEFORE complets ont été sauvegardés, les deux reproductions ont échoué avant correction, puis une correction générique minimale a été appliquée en V2.1. "
        "Des frontières de couples coût/puissance non dominés remplacent l'hypothèse implicite que deux minima séparés peuvent être réalisés ensemble. "
        "Le nouveau contrôle joint intervient avant la sélection de largeur, sans modifier la formule ni les poids de l'heuristique, H5–H10, le ranking ou les contraintes. "
        f"La même campagne complète est relancée : {comparison['before_valid']}/1000 → {comparison['after_valid']}/1000, "
        f"{comparison['validity_gains']} gains, {comparison['validity_losses']} pertes, aucun témoin faisable manqué après correction. "
        "Le verdict historique REQUIRES FURTHER ALGORITHMIC WORK porte uniquement sur la V2 originale. "
        "La V2.1 est acceptée avec limitations de validation : aucun défaut actuel n'est démontré dans la revue finale. "
        "Les analyses de regrets sont terminées sur toutes les références traitables : analysis available on exhaustively tractable domains only. "
        "Les pertes de qualité mesurées restent documentées, sans garantie d'optimalité. La qualité des domaines trop grands demeure non mesurée."
    )
    after["evaluation_date"] = "2026-10-03"
    after["environment"]["cpu_model"] = "11th Gen Intel(R) Core(TM) i5-11320H @ 3.20GHz"
    runtime_path = root/"interleaved_runtime.json"
    if runtime_path.exists():
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        after["interleaved_runtime_check"] = {k: v for k, v in runtime.items() if k != "rows"}
    extended.dump(root/"extended_benchmark.json", after)
    extended.dump(root/"problem_cases.json", after["problem_cases"])
    extended.dump(root/"width_sensitivity.json", after["width_sensitivity"])
    for name in ("no_solution_audit.json", "no_solution_audit.md", "campaign_runner_snapshot.py"):
        shutil.copy2(root/"after_joint_fix"/name, root/name)
    old_source = root/"before_joint_fix/beam_search_v2_reference.py"
    new_source = extended.PROJECT_ROOT/"src/full_architecture/beam_search.py"
    diff = subprocess.run(["git", "diff", "--no-index", "--unified=0", "--", str(old_source), str(new_source)],
                          capture_output=True, text=True, encoding="utf-8")
    assert diff.returncode == 1
    (root/"engine_change.diff").write_text(diff.stdout, encoding="utf-8")
    extended.write_report(after)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
