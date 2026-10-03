# Beam V2.1 — validation élargie

La [revue finale V2.1](release_review.md) distingue le verdict courant du verdict historique V2 et donne les preuves, tests et fichiers recommandés pour le futur commit.

## A. État initial

Version initiale 2.0; version finale mesurée 2.1; politique finale `BEAM_OPTIMISTIC_COMPLETION_V2_JOINT_RESOURCES`. Tests initiaux : 57 dédiés et 490 globaux PASS.

État Git initial conservé intégralement dans `initial_git_status.txt`. H5–H10, ranking et historique vérifiés par SHA256 dans `frozen_sources_initial.json`. L'unique changement de production autorisé est le patch Beam documenté après la campagne BEFORE ; son diff est dans `engine_change.diff`.

```text
 M README.md
 D lustre_architecture_generator/artifacts/rankers/mdt/mdt_ranker.cbm
 D lustre_architecture_generator/artifacts/rankers/mdt/mdt_ranker_metadata.json
 D lustre_architecture_generator/artifacts/rankers/ost/ost_ranker.cbm
 D lustre_architecture_generator/artifacts/rankers/ost/ost_ranker_metadata.json
 M lustre_architecture_generator/src/full_architecture/__init__.py
 D requirement_extractor/__init__.py
 D requirement_extractor/ai_plausibility_agent.py
 D requirement_extractor/calculation_engine.py
 D requirement_extractor/clarification_agent.py
 D requirement_extractor/closed_vocabulary_mapper.py
 D requirement_extractor/field_defs.py
 D requirement_extractor/hybrid_extractor.py
 D requirement_extractor/llm_fallback_extractor.py
 D requirement_extractor/main.py
 D requirement_extractor/models.py
 D requirement_extractor/quick_test.py
 D requirement_extractor/requirement_chatbot.py
 D requirement_extractor/rule_entity_extractor.py
 D requirement_extractor/state_guard.py
 D requirement_extractor/text_preprocessor.py
 D requirement_extractor/unit_normalizer.py
 D requirement_extractor/validation/__init__.py
 D requirement_extractor/validation/datasets/ai_plausibility_stress_dataset_v1.json
 D requirement_extractor/validation/datasets/end_to_end_stress_dataset_v1.json
 D requirement_extractor/validation/datasets/stress_requests_v1.json
 D requirement_extractor/validation/end_to_end_error_analyzer.py
 D requirement_extractor/validation/end_to_end_metrics.py
 D requirement_extractor/validation/error_analyzer.py
 D requirement_extractor/validation/metrics.py
 D requirement_extractor/validation/plausibility_enrichment_metrics.py
 D requirement_extractor/validation/plausibility_error_analyzer.py
 D requirement_extractor/validation/plausibility_full_agent_error_analyzer.py
 D requirement_extractor/validation/plausibility_full_agent_metrics.py
 D requirement_extractor/validation/plausibility_metrics.py
 D requirement_extractor/validation/reports/ai_plausibility/enrichment/metrics_ai_plausibility_enrichment.json
 D requirement_extractor/validation/reports/ai_plausibility/enrichment/results_ai_plausibility_enrichment.json
 D requirement_extractor/validation/reports/ai_plausibility/errors_ai_plausibility.json
 D requirement_extractor/validation/reports/ai_plausibility/full_agent/errors_ai_plausibility_full_agent.json
 D requirement_extractor/validation/reports/ai_plausibility/full_agent/metrics_ai_plausibility_full_agent.json
 D requirement_extractor/validation/reports/ai_plausibility/full_agent/results_ai_plausibility_full_agent.json
 D requirement_extractor/validation/reports/ai_plausibility/metrics_ai_plausibility.json
 D requirement_extractor/validation/reports/ai_plausibility/results_ai_plausibility.json
 D requirement_extractor/validation/reports/deterministic/errors_deterministic.json
 D requirement_extractor/validation/reports/deterministic/metrics_deterministic.json
 D requirement_extractor/validation/reports/deterministic/results_deterministic.json
 D requirement_extractor/validation/reports/end_to_end/errors_end_to_end_v1.json
 D requirement_extractor/validation/reports/end_to_end/metrics_end_to_end_v1.json
 D requirement_extractor/validation/reports/end_to_end/results_end_to_end_v1.json
 D requirement_extractor/validation/reports/hybrid/errors_hybrid.json
 D requirement_extractor/validation/reports/hybrid/metrics_hybrid.json
 D requirement_extractor/validation/reports/hybrid/results_hybrid.json
 D requirement_extractor/validation/reports/llm_fallback/errors_llm_fallback.json
 D requirement_extractor/validation/reports/llm_fallback/metrics_llm_fallback.json
 D requirement_extractor/validation/reports/llm_fallback/results_llm_fallback.json
 D requirement_extractor/validation/run_ai_plausibility_enrichment_validation.py
 D requirement_extractor/validation/run_ai_plausibility_full_agent_validation.py
 D requirement_extractor/validation/run_ai_plausibility_validation.py
 D requirement_extractor/validation/run_end_to_end_validation.py
 D requirement_extractor/validation/run_validation.py
 D requirement_extractor/validation/validate_end_to_end_dataset.py
?? lustre_architecture_generator/docs/beam_search.md
?? lustre_architecture_generator/docs/beam_search_v2.md
?? lustre_architecture_generator/evaluation/architecture/beam_search/
?? lustre_architecture_generator/src/full_architecture/beam_search.py
?? lustre_architecture_generator/tests/test_beam_search.py
?? lustre_architecture_generator/tests/test_beam_search_benchmark.py
?? lustre_architecture_generator/tests/test_beam_search_v2.py
```

## B. Dataset utilisé

Source : `C:\Users\LENOVO\Desktop\internship\version2\lustre_architecture_generator\output\lustre_architecture_dataset.json`. 1200 cas disponibles, 50 évalués.

Numeric minima/maxima, then sorted round-robin over HA x budget tertile x power tertile x OST priority; within each stratum SHA256(case_id) order. No feasibility filtering.

Les trois cas de développement sont exclus. Les classes numériques utilisent les tertiles du dataset complet ; elles servent uniquement à l'analyse. Échantillon de stress équilibré par strates, sans pondération représentative de la fréquence de production. Données synthétiques contrôlées, sans validation de performances Lustre réelles.

| Contrainte | Faible | Moyenne | Élevée | Min | Médiane | Max |
|---|---|---|---|---|---|---|
| budget_usd | 12 | 16 | 22 | 50165.0 | 5873499.5 | 24996056.0 |
| power_w | 13 | 17 | 20 | 5062.0 | 97919.0 | 299758.0 |
| capacity_tib | 11 | 20 | 19 | 140.25 | 28100.4375 | 186628.125 |
| throughput_gbps | 10 | 23 | 17 | 31.25 | 231.25 | 3307.5 |
| mdt_iops | 16 | 18 | 16 | 2750 | 438159.5 | 7715138 |

Unités : budget USD, puissance W, capacité TiB, IOPS MDT opérations/s. Les champs historiques `_gbps` désignent ici des GB/s, conformément au contrat du projet.

Datasets inspectés : architectures, candidats MDT/OST, workload analysis et workload features (1 200 cas chacun). La source architecture porte les contraintes et exigences par rôle ; les candidats sont recalculés avec le runtime LightGBM officiel au lieu de réutiliser les exports candidats.

Timing pilot took 20-43 seconds/case. Final deterministic stratified sample reduced from 100 to 50 before completing evaluation; selection uses no feasibility outcomes.

Cas complets réutilisés du pilote pour BEFORE : ['REQ_000007', 'REQ_000015', 'REQ_000034']. Aucun cas réutilisé pour AFTER. Sources exactes archivées dans les sous-dossiers BEFORE, AFTER et `pilot_100/runner_source.py`.


HA : {'true': 26, 'false': 24}. Charge OST native : {'low': 19, 'medium': 14, 'high': 11, 'critical': 6}.

Top-K réel par rôle : `{'5': {'mdt': {'5': 50}, 'ost': {'5': 50}}, '10': {'mdt': {'10': 50}, 'ost': {'10': 49, '5': 1}}, '20': {'mdt': {'20': 50}, 'ost': {'20': 49, '5': 1}}, '50': {'mdt': {'50': 48, '48': 2}, 'ost': {'50': 44, '46': 1, '41': 1, '48': 1, '5': 1, '49': 1, '45': 1}}}`.

## C. Configuration benchmark

K=[5, 10, 20, 50]; B=[1, 4, 8, 16, 32]; répétitions=2; chemins/variante=1; seuil exhaustif=2000 paires.

Environnement : `{'python': '3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)]', 'platform': 'Windows-11-10.0.26200-SP0', 'processor': 'Intel64 Family 6 Model 140 Stepping 2, GenuineIntel', 'logical_cpus': 8, 'executable': 'C:\\Users\\LENOVO\\Desktop\\internship\\version2\\venv\\Scripts\\python.exe', 'cpu_model': '11th Gen Intel(R) Core(TM) i5-11320H @ 3.20GHz'}`.

Runtime_total mesure l'appel Beam entier (validation d'entrée, préparation H5/H6, recherche, H9, H10). Ranking et préflight/exhaustif sont exclus et enregistrés séparément. Chaque valeur est la médiane des répétitions non instrumentées ; les statistiques décrivent les distributions entre cas. Quantiles interpolés linéairement. Le passage caches séparé est exclu des timings.

## D. Couverture

Les tableaux D–K décrivent la campagne finale V2.1. La matrice complète V2 originale est conservée dans `before_joint_fix/extended_benchmark.json/.csv/.md` ; la campagne finale brute dans `after_joint_fix/`. Les 1 000 comparaisons sont dans `before_after.csv`.

BEFORE : 780/1000 ; AFTER : 785/1000. Gains : 5 ; pertes : 0 ; changements de recommandation : 5. Qualités améliorées/dégradées : 0/0.

| K | B | Valid cases | Total | Feasible rate |
|---|---|---|---|---|
| 5 | 1 | 36 | 50 | 72.0% |
| 5 | 4 | 36 | 50 | 72.0% |
| 5 | 8 | 36 | 50 | 72.0% |
| 5 | 16 | 36 | 50 | 72.0% |
| 5 | 32 | 36 | 50 | 72.0% |
| 10 | 1 | 38 | 50 | 76.0% |
| 10 | 4 | 38 | 50 | 76.0% |
| 10 | 8 | 38 | 50 | 76.0% |
| 10 | 16 | 38 | 50 | 76.0% |
| 10 | 32 | 38 | 50 | 76.0% |
| 20 | 1 | 41 | 50 | 82.0% |
| 20 | 4 | 41 | 50 | 82.0% |
| 20 | 8 | 41 | 50 | 82.0% |
| 20 | 16 | 41 | 50 | 82.0% |
| 20 | 32 | 41 | 50 | 82.0% |
| 50 | 1 | 42 | 50 | 84.0% |
| 50 | 4 | 42 | 50 | 84.0% |
| 50 | 8 | 42 | 50 | 84.0% |
| 50 | 16 | 42 | 50 | 84.0% |
| 50 | 32 | 42 | 50 | 84.0% |

Le taux mesure une solution dans le domaine fourni, pas une preuve d'infaisabilité globale. Les cas sans solution restent au dénominateur.

Interprétation finale : 785 configurations VALID ; 215 absences localement infaisables selon l'audit scalaire ; 0 domaine faisable connu manqué. Aucune preuve d'infaisabilité globale. Les absences ne constituent pas un taux d'échec Beam.

## E. Performance

| K | B | mean s | median s | p90 s | p95 s |
|---|---|---|---|---|---|
| 5 | 1 | 0.009769 | 0.008254 | 0.016364 | 0.021779 |
| 5 | 4 | 0.015423 | 0.015980 | 0.027053 | 0.029631 |
| 5 | 8 | 0.024314 | 0.027058 | 0.039138 | 0.041433 |
| 5 | 16 | 0.042196 | 0.049067 | 0.068124 | 0.073437 |
| 5 | 32 | 0.079125 | 0.092399 | 0.125848 | 0.140294 |
| 10 | 1 | 0.021904 | 0.021425 | 0.034087 | 0.037595 |
| 10 | 4 | 0.027251 | 0.029563 | 0.034573 | 0.034932 |
| 10 | 8 | 0.033748 | 0.034042 | 0.047870 | 0.052240 |
| 10 | 16 | 0.053469 | 0.058732 | 0.079100 | 0.082819 |
| 10 | 32 | 0.103052 | 0.112738 | 0.171313 | 0.181879 |
| 20 | 1 | 0.036489 | 0.036133 | 0.049955 | 0.055297 |
| 20 | 4 | 0.036575 | 0.035119 | 0.053248 | 0.056080 |
| 20 | 8 | 0.048219 | 0.049592 | 0.070921 | 0.074914 |
| 20 | 16 | 0.077425 | 0.078183 | 0.116126 | 0.127838 |
| 20 | 32 | 0.141943 | 0.130335 | 0.232759 | 0.241942 |
| 50 | 1 | 0.065359 | 0.064586 | 0.086586 | 0.094826 |
| 50 | 4 | 0.075991 | 0.073179 | 0.109990 | 0.120135 |
| 50 | 8 | 0.092550 | 0.088163 | 0.141626 | 0.143267 |
| 50 | 16 | 0.137058 | 0.122048 | 0.223665 | 0.225118 |
| 50 | 32 | 0.244238 | 0.207449 | 0.438239 | 0.442822 |

Les temps bruts BEFORE/AFTER ont été mesurés sous des conditions d'exécution différentes ; les temps de l'exhaustif frozen changent également. Aucune attribution causale de leur différence au patch. Contrôle alterné dans le même processus, B=32, dix cas, deux répétitions :

| K | Paires | Temps BEFORE mean s | Temps AFTER mean s | Ratio AFTER/BEFORE médian |
|---|---|---|---|---|
| 5 | 10 | 0.066664 | 0.058131 | 0.9836 |
| 10 | 10 | 0.087832 | 0.072707 | 0.8798 |
| 20 | 10 | 0.135570 | 0.125987 | 0.9246 |
| 50 | 10 | 0.249823 | 0.241068 | 0.9606 |

## F. Espace de recherche

Branches, COMPLETE et H10 sont des moyennes par cas. Les pourcentages utilisent les branches générées totales comme dénominateur, préparation linéaire incluse. Plusieurs raisons peuvent expliquer une même branche hard-pruned ; les raisons ne sont pas additionnées pour calculer ce taux.

| K | B | branches mean | branches median | hard prune % | width prune % | COMPLETE mean | H10 mean |
|---|---|---|---|---|---|---|---|
| 5 | 1 | 144.1 | 151.0 | 16.58 | 7.07 | 0.72 | 0.72 |
| 5 | 4 | 185.1 | 208.0 | 17.41 | 16.16 | 2.88 | 2.88 |
| 5 | 8 | 226.9 | 266.0 | 19.08 | 20.06 | 5.76 | 5.76 |
| 5 | 16 | 301.7 | 370.0 | 21.72 | 22.97 | 11.52 | 11.52 |
| 5 | 32 | 439.7 | 557.0 | 23.87 | 25.87 | 23.00 | 23.00 |
| 10 | 1 | 278.8 | 291.0 | 16.62 | 5.87 | 0.76 | 0.76 |
| 10 | 4 | 333.5 | 363.0 | 18.32 | 12.79 | 3.04 | 3.04 |
| 10 | 8 | 406.4 | 459.0 | 19.88 | 19.10 | 6.08 | 6.08 |
| 10 | 16 | 525.0 | 615.0 | 22.92 | 23.76 | 12.16 | 12.16 |
| 10 | 32 | 743.9 | 903.0 | 26.81 | 27.39 | 24.32 | 24.32 |
| 20 | 1 | 547.3 | 568.0 | 17.20 | 5.25 | 0.82 | 0.82 |
| 20 | 4 | 630.7 | 670.0 | 19.82 | 10.55 | 3.28 | 3.28 |
| 20 | 8 | 741.7 | 806.0 | 22.37 | 15.78 | 6.56 | 6.56 |
| 20 | 16 | 961.6 | 1078.0 | 25.60 | 22.65 | 13.12 | 13.12 |
| 20 | 32 | 1339.5 | 1550.0 | 30.11 | 27.67 | 26.24 | 26.24 |
| 50 | 1 | 1330.2 | 1384.0 | 17.87 | 4.72 | 0.84 | 0.84 |
| 50 | 4 | 1491.0 | 1576.0 | 21.15 | 8.78 | 3.36 | 3.36 |
| 50 | 8 | 1704.8 | 1832.0 | 24.57 | 12.97 | 6.72 | 6.72 |
| 50 | 16 | 2131.8 | 2344.0 | 29.37 | 18.81 | 13.44 | 13.44 |
| 50 | 32 | 2971.8 | 3368.0 | 34.55 | 25.61 | 26.76 | 26.76 |

## G. K=50

K=50 exécuté sur les 50 cas et toutes les largeurs, soit 250 configurations. Aucune approximation exhaustive pour les domaines trop grands.

| B | VALID/total | FSR | Runtime mean s | p95 s | Branches mean | COMPLETE mean | H10 mean |
|---|---|---|---|---|---|---|---|
| 1 | 42/50 | 84.0% | 0.065359 | 0.094826 | 1330.2 | 0.84 | 0.84 |
| 4 | 42/50 | 84.0% | 0.075991 | 0.120135 | 1491.0 | 3.36 | 3.36 |
| 8 | 42/50 | 84.0% | 0.092550 | 0.143267 | 1704.8 | 6.72 | 6.72 |
| 16 | 42/50 | 84.0% | 0.137058 | 0.225118 | 2131.8 | 13.44 | 13.44 |
| 32 | 42/50 | 84.0% | 0.244238 | 0.442822 | 2971.8 | 26.76 | 26.76 |

## H. Sensibilité à B

Validité monotone : 200/200 groupes case×K. Changements d'architecture : 152 groupes.

Groupes avec au moins une solution : 157. Les autres groupes sont toujours sans solution : leur monotonie de validité est vraie par vacuité.

Qualité monotone : 9/36 groupes ayant au moins deux qualités comparables. Les autres groupes n'ont aucune conclusion de qualité.

Tous les contre-exemples (paires de largeurs, architectures et qualités) figurent dans `width_sensitivity.json`.

| Case | K | Validité monotone | Qualité monotone | Changements | Contre-exemples qualité |
|---|---|---|---|---|---|
| REQ_000015 | 5 | True | False | 1 | [{'lower_b': 1, 'higher_b': 4, 'lower_quality': 1.0, 'higher_quality': 0.9995429092492749}, {'lower_b': 1, 'higher_b': 8, 'lower_quality': 1.0, 'higher_quality': 0.9995429092492749}] |
| REQ_000034 | 5 | True | False | 2 | [{'lower_b': 1, 'higher_b': 16, 'lower_quality': 0.8960006486500196, 'higher_quality': 0.8864218781373774}, {'lower_b': 4, 'higher_b': 16, 'lower_quality': 0.8960006486500196, 'higher_quality': 0.8864218781373774}] |
| REQ_000270 | 5 | True | False | 4 | [{'lower_b': 1, 'higher_b': 8, 'lower_quality': 0.9999908854222244, 'higher_quality': 0.9991948778593102}, {'lower_b': 1, 'higher_b': 16, 'lower_quality': 0.9999908854222244, 'higher_quality': 0.98787106515314}] |
| REQ_000289 | 5 | True | False | 3 | [{'lower_b': 4, 'higher_b': 16, 'lower_quality': 1.0, 'higher_quality': 0.9986759693245613}, {'lower_b': 8, 'higher_b': 16, 'lower_quality': 1.0, 'higher_quality': 0.9986759693245613}] |
| REQ_000338 | 5 | True | False | 1 | [{'lower_b': 1, 'higher_b': 4, 'lower_quality': 1.0, 'higher_quality': 0.9864516514189383}, {'lower_b': 1, 'higher_b': 8, 'lower_quality': 1.0, 'higher_quality': 0.9864516514189383}] |
| REQ_000373 | 5 | True | False | 2 | [{'lower_b': 1, 'higher_b': 4, 'lower_quality': 1.0, 'higher_quality': 0.9998923363474784}, {'lower_b': 1, 'higher_b': 8, 'lower_quality': 1.0, 'higher_quality': 0.9998923363474784}] |
| REQ_000386 | 5 | True | False | 4 | [{'lower_b': 1, 'higher_b': 8, 'lower_quality': 0.9773884149542338, 'higher_quality': 0.9771483259093531}, {'lower_b': 1, 'higher_b': 16, 'lower_quality': 0.9773884149542338, 'higher_quality': 0.9753992910116274}] |
| REQ_000392 | 5 | True | False | 3 | [{'lower_b': 1, 'higher_b': 4, 'lower_quality': 0.9768946216150831, 'higher_quality': 0.9766787919510895}, {'lower_b': 1, 'higher_b': 8, 'lower_quality': 0.9768946216150831, 'higher_quality': 0.9766787919510895}] |
| REQ_000407 | 5 | True | False | 1 | [{'lower_b': 1, 'higher_b': 32, 'lower_quality': 1.0, 'higher_quality': 0.9518635861692486}, {'lower_b': 4, 'higher_b': 32, 'lower_quality': 1.0, 'higher_quality': 0.9518635861692486}] |
| REQ_000432 | 5 | True | False | 3 | [{'lower_b': 1, 'higher_b': 16, 'lower_quality': 0.9749628420467567, 'higher_quality': 0.9689716503186121}, {'lower_b': 1, 'higher_b': 32, 'lower_quality': 0.9749628420467567, 'higher_quality': 0.9666299434211711}] |

## I. Comparaison exhaustive

Statuts des 200 préflights : {'COMPLETE': 51, 'SKIPPED_DOMAIN_TOO_LARGE': 149}. Comparaisons effectives : 180 configurations, 36 cas distincts.

Un domaine est un cas × K × cap de chemins, indépendant de B ; une configuration ajoute B. Les 51 domaines entièrement parcourus comprennent les 50 K=5 et un K=10 : 36 domaines ont un VALID et donnent 36×5=180 configurations comparables, 15 n'ont aucun VALID. Les 149 autres domaines n'ont pas de référence exhaustive de qualité.

Les analyses de search regret et de H9 normalization regret sont terminées sur toutes les comparaisons disponibles : analysis available on exhaustively tractable domains only. PARTIAL qualifie uniquement la couverture de validation (cas A), sans travail analytique inachevé sur les domaines traitables.

La référence utilise le même Top-K et les mêmes chemins, H8/H9/H10 frozen, sans troncature de paires. Le pool H9 exhaustif inclut les architectures invalides ; le meilleur candidat et les rangs VALID sont pris après H10. Les scores de pools différents ne sont jamais comparés directement.

Qualité relative moyenne : 0.982148422072597; minimale : 0.8142958377766908; rang VALID commun maximal sélectionné : 500. Meilleur exhaustif survivant : 97/180; sélectionné : 36/180.

Regret moyen search : 0.010925760807154445; normalisation : 0.003824268361532696. Tous les rangs, indicateurs et regrets sont dans JSON et CSV, champs `quality.*`.

`total_common_score_regret = search_common_score_regret + pool_normalization_common_score_regret`, vérifié pour chaque comparaison. Les domaines sautés n'ont ni qualité ni regret inventés. Les cas sans référence VALID sont classés séparément.

| K | B | Comparaisons | Qualité moyenne | Qualité min | Rang médian | Best survived | Best chosen | Search regret mean | H9 regret mean |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 1 | 36 | 0.978475 | 0.814296 | 5.0 | 7 | 7 | 0.017569573 | 0.000000000 |
| 5 | 4 | 36 | 0.979414 | 0.828268 | 4.0 | 19 | 8 | 0.012406473 | 0.004458022 |
| 5 | 8 | 36 | 0.983163 | 0.876227 | 5.0 | 22 | 6 | 0.009014916 | 0.005011655 |
| 5 | 16 | 36 | 0.985554 | 0.876227 | 4.5 | 24 | 7 | 0.008683843 | 0.003225359 |
| 5 | 32 | 36 | 0.984136 | 0.876227 | 5.5 | 25 | 8 | 0.006953999 | 0.006426306 |

## J. Origine des pertes

Dénominateur : 180 configurations comparables ; tolérance score 1e-12. Les taux portent sur case×K×B, pas sur des requêtes distinctes.

| Origine | Configurations | Part |
|---|---|---|
| search_only | 63 | 35.0% |
| normalization_only | 61 | 33.9% |
| both | 20 | 11.1% |
| no_measurable_loss | 36 | 20.0% |

## K. Mémoire / caches

Les dictionnaires H5/H6 et les enveloppes ne font que croître : leur taille finale est leur maximum. Les compteurs publics sont utilisés pour chaque configuration et vérifiés par lecture des variables locales au retour d'un passage B=32 séparé pour chacun des 200 couples case×K. Le résultat instrumenté doit égaler le résultat non instrumenté. Aucun changement du moteur. Mesure en entrées, sans estimation RSS ni borne en octets ; B borne la frontière, pas les caches ni la mémoire totale du processus.

| K | Max H5 cache | Max H6 cache | Max beam states | Max envelope count | Max selection buffer | Frontier entries after compaction |
|---|---|---|---|---|---|---|
| 5 | 60 | 60 | 32 | 132 | 33 | 135 |
| 10 | 120 | 120 | 32 | 262 | 33 | 266 |
| 20 | 240 | 240 | 32 | 522 | 33 | 525 |
| 50 | 600 | 600 | 32 | 1276 | 33 | 1282 |

Les entrées de frontières sont mesurées après compaction. Une borne conservatrice avant compaction vaut quatre fois les options de rôle préparées, soit au maximum 2352 entrées dans cette campagne ; ce nombre n'est pas une mesure RSS.


Max beam states désigne la frontière retenue (`max_active_states`), et non tous les objets vivants. L'ancienne frontière coexiste avec le buffer de sélection (au plus B+1). Les entrées de caches incluent aussi les échecs et les listes vides.


## L. Cas problématiques

Les 10 exemples par catégorie sont dans `problem_cases.json` et `problem_cases.md` : absence de solution, sensibilité à B, regret search, regret H9, lenteur, branches. Les listes de cas sans solution, lents et avec regret ne répètent pas un même case_id. Les compteurs d'options permettent d'identifier les domaines sans option de complétion.

Référence exhaustive faisable mais Beam sans solution : 0 configurations.

Audit indépendant : 43 groupes, 215 configurations sans solution, {'BUDGET_AND_POWER_LOWER_BOUNDS_EXCEED': 11, 'POWER_LOWER_BOUND_EXCEEDS': 16, 'JOINT_BUDGET_POWER_CONFLICT': 10, 'BUDGET_LOWER_BOUND_EXCEEDS': 6}. Témoins faisables manqués : 0.

Audit détaillé dans `no_solution_audit.json/.md`. Il vérifie la faisabilité coût/puissance avec les fonctions publiques frozen, sans pool H9 et sans comparaison de qualité supplémentaire.

| Catégorie | Cinq exemples |
|---|---|
| no_solution | REQ_000220 K=50 B=32 valeur=BUDGET_AND_POWER_LOWER_BOUNDS_EXCEED; REQ_000221 K=50 B=32 valeur=JOINT_BUDGET_POWER_CONFLICT; REQ_000323 K=50 B=32 valeur=POWER_LOWER_BOUND_EXCEEDS; REQ_000391 K=50 B=32 valeur=BUDGET_AND_POWER_LOWER_BOUNDS_EXCEED; REQ_000637 K=50 B=32 valeur=BUDGET_LOWER_BOUND_EXCEEDS |
| width_sensitive | REQ_000034 K=5 amplitude=0.08286569165955593 changements=2; REQ_000593 K=5 amplitude=0.0706857176232436 changements=2; REQ_000957 K=5 amplitude=0.06251179981073618 changements=2; REQ_000794 K=5 amplitude=0.06193071202637268 changements=2; REQ_000407 K=5 amplitude=0.0481364138307514 changements=1 |
| high_search_regret | REQ_000794 K=5 B=1 valeur=0.155010136207392; REQ_000034 K=5 B=1 valeur=0.0740427488472476; REQ_000593 K=5 B=1 valeur=0.0645433042851209; REQ_000007 K=5 B=1 valeur=0.04487966962730028; REQ_000957 K=5 B=1 valeur=0.04453338572867105 |
| high_normalization_regret | REQ_000957 K=5 B=4 valeur=0.04454252330202457; REQ_000407 K=5 B=32 valeur=0.04005525042558278; REQ_000392 K=5 B=32 valeur=0.032441261062785; REQ_000034 K=5 B=16 valeur=0.028685479444077733; REQ_000593 K=5 B=4 valeur=0.027091951080639887 |
| slow | REQ_000015 K=50 B=32 valeur=0.47049754997715354; REQ_000751 K=50 B=32 valeur=0.44579110003542155; REQ_000338 K=50 B=32 valeur=0.4443754499661736; REQ_001079 K=50 B=32 valeur=0.4409241499961354; REQ_000034 K=50 B=32 valeur=0.4407398500479758 |
| high_branches | REQ_000914 K=50 B=32 valeur=3398; REQ_000665 K=50 B=32 valeur=3395; REQ_000738 K=50 B=32 valeur=3395; REQ_001104 K=50 B=32 valeur=3395; REQ_000007 K=50 B=32 valeur=3394 |

## M. Conclusion algorithmique

BEAM V2.1 ACCEPTED WITH VALIDATION LIMITATIONS

La V2 originale ne peut pas être acceptée sans changement : elle élimine tous les préfixes faisables d'un domaine pourtant H10 VALID. Les résultats BEFORE complets ont été sauvegardés, les deux reproductions ont échoué avant correction, puis une correction générique minimale a été appliquée en V2.1. Des frontières de couples coût/puissance non dominés remplacent l'hypothèse implicite que deux minima séparés peuvent être réalisés ensemble. Le nouveau contrôle joint intervient avant la sélection de largeur, sans modifier la formule ni les poids de l'heuristique, H5–H10, le ranking ou les contraintes. La même campagne complète est relancée : 780/1000 → 785/1000, 5 gains, 0 pertes, aucun témoin faisable manqué après correction. Le verdict historique REQUIRES FURTHER ALGORITHMIC WORK porte uniquement sur la V2 originale. La V2.1 est acceptée avec limitations de validation : aucun défaut actuel n'est démontré dans la revue finale. Les analyses de regrets sont terminées sur toutes les références traitables : analysis available on exhaustively tractable domains only. Les pertes de qualité mesurées restent documentées, sans garantie d'optimalité. La qualité des domaines trop grands demeure non mesurée.

Verdict historique, V2 originale uniquement : `BEAM V2 REQUIRES FURTHER ALGORITHMIC WORK`.

### Known limitations — not correctness bugs

Optimalité/complétude non garanties avec B limité ; qualité non monotone avec B ; H9 normalisé par pool ; références exhaustives indisponibles pour les grands domaines ; 50 cas synthétiques seulement ; caches/enveloppes non bornés par B ; résultats dépendants des catalogues/dataset actuels. Aucun nouveau défaut de correction V2.1 n'est démontré pendant la revue finale.

## N. Tests finaux

```text
.\venv\Scripts\python.exe -m pytest lustre_architecture_generator/tests/test_beam_search.py lustre_architecture_generator/tests/test_beam_search_benchmark.py lustre_architecture_generator/tests/test_beam_search_v2.py lustre_architecture_generator/tests/test_beam_search_extended_benchmark.py lustre_architecture_generator/tests/test_beam_search_joint_resources.py -q -p no:cacheprovider --basetemp C:/Users/LENOVO/AppData/Local/Temp/beam_v21_release_20261003_dedicated_r1
........................................................................ [ 94%]
....                                                                     [100%]
76 passed in 2.75s


.\venv\Scripts\python.exe -m pytest lustre_architecture_generator/tests -q -p no:cacheprovider --basetemp C:/Users/LENOVO/AppData/Local/Temp/beam_v21_release_20261003_architecture_r1
........................................................................ [ 23%]
........................................................................ [ 47%]
........................................................................ [ 71%]
........................................................................ [ 95%]
...............                                                          [100%]
303 passed in 13.81s


.\venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp C:/Users/LENOVO/AppData/Local/Temp/beam_v21_release_20261003_global_r1
........................................................................ [ 14%]
........................................................................ [ 28%]
........................................................................ [ 42%]
........................................................................ [ 56%]
........................................................................ [ 70%]
........................................................................ [ 84%]
........................................................................ [ 99%]
.....                                                                    [100%]
509 passed in 48.46s

```

## O. Git status final

```text
 M README.md
 D lustre_architecture_generator/artifacts/rankers/mdt/mdt_ranker.cbm
 D lustre_architecture_generator/artifacts/rankers/mdt/mdt_ranker_metadata.json
 D lustre_architecture_generator/artifacts/rankers/ost/ost_ranker.cbm
 D lustre_architecture_generator/artifacts/rankers/ost/ost_ranker_metadata.json
 M lustre_architecture_generator/src/full_architecture/__init__.py
 D requirement_extractor/__init__.py
 D requirement_extractor/ai_plausibility_agent.py
 D requirement_extractor/calculation_engine.py
 D requirement_extractor/clarification_agent.py
 D requirement_extractor/closed_vocabulary_mapper.py
 D requirement_extractor/field_defs.py
 D requirement_extractor/hybrid_extractor.py
 D requirement_extractor/llm_fallback_extractor.py
 D requirement_extractor/main.py
 D requirement_extractor/models.py
 D requirement_extractor/quick_test.py
 D requirement_extractor/requirement_chatbot.py
 D requirement_extractor/rule_entity_extractor.py
 D requirement_extractor/state_guard.py
 D requirement_extractor/text_preprocessor.py
 D requirement_extractor/unit_normalizer.py
 D requirement_extractor/validation/__init__.py
 D requirement_extractor/validation/datasets/ai_plausibility_stress_dataset_v1.json
 D requirement_extractor/validation/datasets/end_to_end_stress_dataset_v1.json
 D requirement_extractor/validation/datasets/stress_requests_v1.json
 D requirement_extractor/validation/end_to_end_error_analyzer.py
 D requirement_extractor/validation/end_to_end_metrics.py
 D requirement_extractor/validation/error_analyzer.py
 D requirement_extractor/validation/metrics.py
 D requirement_extractor/validation/plausibility_enrichment_metrics.py
 D requirement_extractor/validation/plausibility_error_analyzer.py
 D requirement_extractor/validation/plausibility_full_agent_error_analyzer.py
 D requirement_extractor/validation/plausibility_full_agent_metrics.py
 D requirement_extractor/validation/plausibility_metrics.py
 D requirement_extractor/validation/reports/ai_plausibility/enrichment/metrics_ai_plausibility_enrichment.json
 D requirement_extractor/validation/reports/ai_plausibility/enrichment/results_ai_plausibility_enrichment.json
 D requirement_extractor/validation/reports/ai_plausibility/errors_ai_plausibility.json
 D requirement_extractor/validation/reports/ai_plausibility/full_agent/errors_ai_plausibility_full_agent.json
 D requirement_extractor/validation/reports/ai_plausibility/full_agent/metrics_ai_plausibility_full_agent.json
 D requirement_extractor/validation/reports/ai_plausibility/full_agent/results_ai_plausibility_full_agent.json
 D requirement_extractor/validation/reports/ai_plausibility/metrics_ai_plausibility.json
 D requirement_extractor/validation/reports/ai_plausibility/results_ai_plausibility.json
 D requirement_extractor/validation/reports/deterministic/errors_deterministic.json
 D requirement_extractor/validation/reports/deterministic/metrics_deterministic.json
 D requirement_extractor/validation/reports/deterministic/results_deterministic.json
 D requirement_extractor/validation/reports/end_to_end/errors_end_to_end_v1.json
 D requirement_extractor/validation/reports/end_to_end/metrics_end_to_end_v1.json
 D requirement_extractor/validation/reports/end_to_end/results_end_to_end_v1.json
 D requirement_extractor/validation/reports/hybrid/errors_hybrid.json
 D requirement_extractor/validation/reports/hybrid/metrics_hybrid.json
 D requirement_extractor/validation/reports/hybrid/results_hybrid.json
 D requirement_extractor/validation/reports/llm_fallback/errors_llm_fallback.json
 D requirement_extractor/validation/reports/llm_fallback/metrics_llm_fallback.json
 D requirement_extractor/validation/reports/llm_fallback/results_llm_fallback.json
 D requirement_extractor/validation/run_ai_plausibility_enrichment_validation.py
 D requirement_extractor/validation/run_ai_plausibility_full_agent_validation.py
 D requirement_extractor/validation/run_ai_plausibility_validation.py
 D requirement_extractor/validation/run_end_to_end_validation.py
 D requirement_extractor/validation/run_validation.py
 D requirement_extractor/validation/validate_end_to_end_dataset.py
?? lustre_architecture_generator/docs/beam_search.md
?? lustre_architecture_generator/docs/beam_search_v2.md
?? lustre_architecture_generator/docs/beam_search_v2_extended.md
?? lustre_architecture_generator/evaluation/architecture/beam_search/
?? lustre_architecture_generator/src/full_architecture/beam_search.py
?? lustre_architecture_generator/tests/fixtures/
?? lustre_architecture_generator/tests/test_beam_search.py
?? lustre_architecture_generator/tests/test_beam_search_benchmark.py
?? lustre_architecture_generator/tests/test_beam_search_extended_benchmark.py
?? lustre_architecture_generator/tests/test_beam_search_joint_resources.py
?? lustre_architecture_generator/tests/test_beam_search_v2.py

```

## P. Git diff stat

```text
 README.md                                          |   147 +-
 .../artifacts/rankers/mdt/mdt_ranker.cbm           |   Bin 7610744 -> 0 bytes
 .../artifacts/rankers/mdt/mdt_ranker_metadata.json |   136 -
 .../artifacts/rankers/ost/ost_ranker.cbm           |   Bin 7627712 -> 0 bytes
 .../artifacts/rankers/ost/ost_ranker_metadata.json |   145 -
 .../src/full_architecture/__init__.py              |    16 +
 requirement_extractor/__init__.py                  |     1 -
 requirement_extractor/ai_plausibility_agent.py     |  1742 -
 requirement_extractor/calculation_engine.py        |    65 -
 requirement_extractor/clarification_agent.py       |   338 -
 requirement_extractor/closed_vocabulary_mapper.py  |   337 -
 requirement_extractor/field_defs.py                |    93 -
 requirement_extractor/hybrid_extractor.py          |   305 -
 requirement_extractor/llm_fallback_extractor.py    |   542 -
 requirement_extractor/main.py                      |   119 -
 requirement_extractor/models.py                    |   197 -
 requirement_extractor/quick_test.py                |    47 -
 requirement_extractor/requirement_chatbot.py       |  1347 -
 requirement_extractor/rule_entity_extractor.py     |  1076 -
 requirement_extractor/state_guard.py               |   709 -
 requirement_extractor/text_preprocessor.py         |   232 -
 requirement_extractor/unit_normalizer.py           |   218 -
 requirement_extractor/validation/__init__.py       |     1 -
 .../ai_plausibility_stress_dataset_v1.json         |  7699 ----
 .../datasets/end_to_end_stress_dataset_v1.json     |  1216 -
 .../validation/datasets/stress_requests_v1.json    |  8331 ----
 .../validation/end_to_end_error_analyzer.py        |   573 -
 .../validation/end_to_end_metrics.py               |   935 -
 requirement_extractor/validation/error_analyzer.py |   594 -
 requirement_extractor/validation/metrics.py        |   829 -
 .../validation/plausibility_enrichment_metrics.py  |  1767 -
 .../validation/plausibility_error_analyzer.py      |   590 -
 .../plausibility_full_agent_error_analyzer.py      |   791 -
 .../validation/plausibility_full_agent_metrics.py  |  1756 -
 .../validation/plausibility_metrics.py             |  1500 -
 .../metrics_ai_plausibility_enrichment.json        |  1954 -
 .../results_ai_plausibility_enrichment.json        |  8542 ----
 .../ai_plausibility/errors_ai_plausibility.json    |    17 -
 .../errors_ai_plausibility_full_agent.json         |    17 -
 .../metrics_ai_plausibility_full_agent.json        |   670 -
 .../results_ai_plausibility_full_agent.json        | 14361 -------
 .../ai_plausibility/metrics_ai_plausibility.json   |   616 -
 .../ai_plausibility/results_ai_plausibility.json   | 14434 -------
 .../deterministic/errors_deterministic.json        |  6080 ---
 .../deterministic/metrics_deterministic.json       |   393 -
 .../deterministic/results_deterministic.json       | 38567 ------------------
 .../reports/end_to_end/errors_end_to_end_v1.json   |   263 -
 .../reports/end_to_end/metrics_end_to_end_v1.json  |  1076 -
 .../reports/end_to_end/results_end_to_end_v1.json  |  6505 ---
 .../validation/reports/hybrid/errors_hybrid.json   | 10254 -----
 .../validation/reports/hybrid/metrics_hybrid.json  |   393 -
 .../validation/reports/hybrid/results_hybrid.json  | 40336 -------------------
 .../reports/llm_fallback/errors_llm_fallback.json  |  3848 --
 .../reports/llm_fallback/metrics_llm_fallback.json |   393 -
 .../reports/llm_fallback/results_llm_fallback.json | 38776 ------------------
 .../run_ai_plausibility_enrichment_validation.py   |  1514 -
 .../run_ai_plausibility_full_agent_validation.py   |  1257 -
 .../validation/run_ai_plausibility_validation.py   |   839 -
 .../validation/run_end_to_end_validation.py        |  2310 --
 requirement_extractor/validation/run_validation.py |   459 -
 .../validation/validate_end_to_end_dataset.py      |    47 -
 61 files changed, 114 insertions(+), 228201 deletions(-)
warning: in the working copy of 'lustre_architecture_generator/src/full_architecture/__init__.py', LF will be replaced by CRLF the next time Git touches it

```

## Q. Git diff check

```text
warning: in the working copy of 'lustre_architecture_generator/src/full_architecture/__init__.py', LF will be replaced by CRLF the next time Git touches it

```

## R. État final

```text
BEAM SEARCH V2.1 FINAL STATUS

Algorithm status:
ACCEPTED WITH VALIDATION LIMITATIONS

Generic V2 defect fixed: YES

50-case extended validation: PASS
1000 configurations evaluated: YES
K=50 evaluated: YES

Known feasible exhaustive domains missed: 0

Dedicated tests: 76 passed in 2.75s
Global tests: 509 passed in 48.46s

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
- Caches/envelopes and total memory are not bounded by B.
```
