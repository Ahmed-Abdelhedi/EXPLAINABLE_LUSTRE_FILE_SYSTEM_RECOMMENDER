# Beam V2 — validation élargie

## A. État initial

Version 2.0; politique `BEAM_OPTIMISTIC_COMPLETION_V2`. Tests initiaux : 57 dédiés et 490 globaux PASS.

État Git initial conservé intégralement dans `initial_git_status.txt`. Sources frozen et historique vérifiés par SHA256 dans `frozen_sources_initial.json`.

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

HA : {'true': 26, 'false': 24}. Charge OST native : {'low': 19, 'medium': 14, 'high': 11, 'critical': 6}.

Top-K réel par rôle : `{'5': {'mdt': {'5': 50}, 'ost': {'5': 50}}, '10': {'mdt': {'10': 50}, 'ost': {'10': 49, '5': 1}}, '20': {'mdt': {'20': 50}, 'ost': {'20': 49, '5': 1}}, '50': {'mdt': {'50': 48, '48': 2}, 'ost': {'50': 44, '46': 1, '41': 1, '48': 1, '5': 1, '49': 1, '45': 1}}}`.

## C. Configuration benchmark

K=[5, 10, 20, 50]; B=[1, 4, 8, 16, 32]; répétitions=2; chemins/variante=1; seuil exhaustif=2000 paires.

Environnement : `{'python': '3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)]', 'platform': 'Windows-11-10.0.26200-SP0', 'processor': 'Intel64 Family 6 Model 140 Stepping 2, GenuineIntel', 'logical_cpus': 8, 'executable': 'C:\\Users\\LENOVO\\Desktop\\internship\\version2\\venv\\Scripts\\python.exe'}`.

Runtime_total mesure l'appel Beam entier (validation d'entrée, préparation H5/H6, recherche, H9, H10). Ranking et préflight/exhaustif sont exclus et enregistrés séparément. Chaque valeur est la médiane des répétitions non instrumentées ; les statistiques décrivent les distributions entre cas. Quantiles interpolés linéairement. Le passage caches séparé est exclu des timings.

## D. Couverture

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
| 50 | 1 | 41 | 50 | 82.0% |
| 50 | 4 | 41 | 50 | 82.0% |
| 50 | 8 | 41 | 50 | 82.0% |
| 50 | 16 | 41 | 50 | 82.0% |
| 50 | 32 | 41 | 50 | 82.0% |

Le taux mesure une solution dans le domaine fourni, pas une preuve d'infaisabilité globale. Les cas sans solution restent au dénominateur.

## E. Performance

| K | B | mean s | median s | p90 s | p95 s |
|---|---|---|---|---|---|
| 5 | 1 | 0.045222 | 0.040062 | 0.085930 | 0.101292 |
| 5 | 4 | 0.074485 | 0.079258 | 0.121181 | 0.128587 |
| 5 | 8 | 0.112434 | 0.133208 | 0.169994 | 0.180152 |
| 5 | 16 | 0.200242 | 0.229424 | 0.341652 | 0.352259 |
| 5 | 32 | 0.388478 | 0.461509 | 0.639237 | 0.677841 |
| 10 | 1 | 0.075509 | 0.069424 | 0.106613 | 0.134833 |
| 10 | 4 | 0.108049 | 0.112845 | 0.160324 | 0.189128 |
| 10 | 8 | 0.154104 | 0.170241 | 0.221716 | 0.235044 |
| 10 | 16 | 0.259321 | 0.277002 | 0.399231 | 0.425383 |
| 10 | 32 | 0.505265 | 0.539587 | 0.833596 | 0.870489 |
| 20 | 1 | 0.128166 | 0.121277 | 0.186211 | 0.200359 |
| 20 | 4 | 0.173982 | 0.176725 | 0.263651 | 0.281731 |
| 20 | 8 | 0.237696 | 0.246321 | 0.359705 | 0.378420 |
| 20 | 16 | 0.383840 | 0.391763 | 0.592928 | 0.619589 |
| 20 | 32 | 0.703109 | 0.641936 | 1.131267 | 1.238106 |
| 50 | 1 | 0.293829 | 0.293949 | 0.402460 | 0.419922 |
| 50 | 4 | 0.360282 | 0.352285 | 0.527801 | 0.538191 |
| 50 | 8 | 0.461278 | 0.434637 | 0.708052 | 0.769744 |
| 50 | 16 | 0.668915 | 0.630518 | 1.083943 | 1.116976 |
| 50 | 32 | 1.205588 | 1.015425 | 2.179852 | 2.231511 |

## F. Espace de recherche

Branches, COMPLETE et H10 sont des moyennes par cas. Les pourcentages utilisent les branches générées totales comme dénominateur, préparation linéaire incluse. Plusieurs raisons peuvent expliquer une même branche hard-pruned ; les raisons ne sont pas additionnées pour calculer ce taux.

| K | B | branches mean | branches median | hard prune % | width prune % | COMPLETE mean | H10 mean |
|---|---|---|---|---|---|---|---|
| 5 | 1 | 144.1 | 151.0 | 16.58 | 7.07 | 0.72 | 0.72 |
| 5 | 4 | 185.1 | 208.0 | 17.41 | 16.16 | 2.88 | 2.88 |
| 5 | 8 | 226.9 | 266.0 | 19.08 | 20.06 | 5.76 | 5.76 |
| 5 | 16 | 301.7 | 370.0 | 21.72 | 22.97 | 11.52 | 11.52 |
| 5 | 32 | 439.7 | 557.0 | 23.87 | 25.87 | 23.00 | 23.00 |
| 10 | 1 | 279.1 | 291.0 | 16.61 | 5.95 | 0.76 | 0.76 |
| 10 | 4 | 334.8 | 363.0 | 18.44 | 12.87 | 3.04 | 3.04 |
| 10 | 8 | 409.2 | 459.0 | 20.11 | 19.16 | 6.08 | 6.08 |
| 10 | 16 | 529.7 | 615.0 | 23.30 | 23.70 | 12.16 | 12.16 |
| 10 | 32 | 752.1 | 903.0 | 27.35 | 27.15 | 24.32 | 24.32 |
| 20 | 1 | 547.8 | 568.0 | 17.19 | 5.32 | 0.82 | 0.82 |
| 20 | 4 | 632.8 | 670.0 | 19.97 | 10.60 | 3.28 | 3.28 |
| 20 | 8 | 746.0 | 806.0 | 22.65 | 15.80 | 6.56 | 6.56 |
| 20 | 16 | 970.3 | 1078.0 | 26.02 | 22.59 | 13.12 | 13.12 |
| 20 | 32 | 1355.4 | 1550.0 | 30.70 | 27.45 | 26.24 | 26.24 |
| 50 | 1 | 1330.0 | 1384.0 | 17.80 | 4.79 | 0.82 | 0.82 |
| 50 | 4 | 1490.5 | 1576.0 | 21.08 | 8.84 | 3.28 | 3.28 |
| 50 | 8 | 1704.1 | 1832.0 | 24.50 | 13.04 | 6.56 | 6.56 |
| 50 | 16 | 2131.0 | 2344.0 | 29.30 | 18.89 | 13.12 | 13.12 |
| 50 | 32 | 2984.8 | 3368.0 | 34.76 | 25.59 | 26.24 | 26.24 |

## G. K=50

K=50 exécuté sur les 50 cas et toutes les largeurs, soit 250 configurations. Les tableaux D–F présentent ses mesures propres. Aucune approximation exhaustive pour les domaines trop grands.

## H. Sensibilité à B

Validité monotone : 200/200 groupes case×K. Changements d'architecture : 151 groupes.

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

La référence utilise le même Top-K et les mêmes chemins, H8/H9/H10 frozen, sans troncature de paires. Le pool H9 exhaustif inclut les architectures invalides ; le meilleur candidat et les rangs VALID sont pris après H10. Les scores de pools différents ne sont jamais comparés directement.

Qualité relative moyenne : 0.982148422072597; minimale : 0.8142958377766908; rang VALID commun maximal sélectionné : 500. Meilleur exhaustif survivant : 97/180; sélectionné : 36/180.

Regret moyen search : 0.010925760807154445; normalisation : 0.003824268361532696. Tous les rangs, indicateurs et regrets sont dans JSON et CSV, champs `quality.*`.

`total_common_score_regret = search_common_score_regret + pool_normalization_common_score_regret`, vérifié pour chaque comparaison. Les domaines sautés n'ont ni qualité ni regret inventés. Les cas sans référence VALID sont classés séparément.

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

| K | Max H5 cache | Max H6 cache | Max beam states | Max envelope count | Max selection buffer |
|---|---|---|---|---|---|
| 5 | 60 | 60 | 32 | 132 | 33 |
| 10 | 120 | 120 | 32 | 262 | 33 |
| 20 | 240 | 240 | 32 | 522 | 33 |
| 50 | 600 | 600 | 32 | 1276 | 33 |

## L. Cas problématiques

Les 10 exemples par catégorie sont dans `problem_cases.json` et `problem_cases.md` : absence de solution, sensibilité à B, regret search, regret H9, lenteur, branches. Les listes de cas sans solution, lents et avec regret ne répètent pas un même case_id. Les compteurs d'options permettent d'identifier les domaines sans option de complétion.

Référence exhaustive faisable mais Beam sans solution : 0 configurations.

## M. Conclusion algorithmique

PENDING — évaluation à analyser avant décision



## N. Tests finaux

```text
PENDING
```

## O. Git status final

```text
PENDING
```

## P. Git diff stat

```text
PENDING
```

## Q. Git diff check

```text
PENDING
```

## R. État final

```text
PENDING
```
