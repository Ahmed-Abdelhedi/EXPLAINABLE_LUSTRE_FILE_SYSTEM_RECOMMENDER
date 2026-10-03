# Beam Search V2.1 — revue finale A–R

Revue du 3 octobre 2026. Aucun commit, staging ou suppression effectué. Le moteur candidat est conservé byte pour byte.

## A. Verdict V2.1

**BEAM V2.1 ACCEPTED WITH VALIDATION LIMITATIONS**.

Aucun nouveau bug algorithmique, prune non admissible, nondéterminisme ou violation de contrat n'est démontré. Les limites de qualité/portée ne justifient pas une nouvelle optimisation pendant cette clôture. Le verdict historique REQUIRES FURTHER ALGORITHMIC WORK concernait exclusivement la V2 originale.

## B. Correction V2.1

**Cause → correction → preuve.** En V2, la condition nécessaire `min(cost_MDT)+min(cost_OST) ≤ budget` et son analogue puissance étaient admissibles séparément, mais leurs minima pouvaient venir d'options différentes. Leur réussite n'établissait aucune complétion respectant simultanément les deux contraintes. Des préfixes irréalisables pouvaient occuper toutes les places B ; le beam-width pruning éliminait alors les véritables branches faisables. Il ne s'agissait pas d'une ancienne borne dure rejetant directement des solutions valides.

V2.1 ajoute, après ces bornes et avant l'heuristique/sélection de largeur, la raison dure `completion_joint_budget_power_conflict`. Pour chaque préfixe H5/H6, elle conserve des couples réels (coût, puissance) non dominés. Un point dominé a un remplaçant avec coût et puissance inférieurs ou égaux : l'existence d'une complétion satisfaisant ces deux contraintes est préservée. Une frontière ordonnée par coût croissant a une puissance strictement décroissante ; pour chaque MDT, le dernier OST abordable est le plus sobre des OST abordables. Le balayage à deux indices est O(f_MDT+f_OST), sans produit cartésien d'architectures COMPLETE.

Il s'agit d'un **hard pruning supplémentaire physiquement admissible des complétions impossibles**, qui protège la largeur des préfixes fantômes. Les lower bounds indépendantes, l'heuristique, ses poids et les tie-breaks ne changent pas. Le test reste une condition nécessaire de ressources, pas une décision VALID. Les marges fixes `max(1e-12, 8*ulp(limite))` autorisent conservativement les frontières arrondies ; H10 recompute H5/H6 et décide exactement selon son contrat inchangé.

Le code n'utilise ni case_id, ni ID de drive spécifique, ni imports d'évaluation pour cette règle. Les enveloppes proviennent de chaque entrée actuelle. Les cinq replays avec un case_id renommé conservent les choix physiques. Deux reproductions historiques échouaient en V2 ; les régressions Pareto, jointes, flottantes et les 300 confrontations avec une recherche scalaire brute passent en V2.1.

Provenance logique : BEFORE schéma 2.0 → patch exact → AFTER schéma 2.1 → combinaison des deux phases → revue finale du même hash. La date de campagne et les temps historiques sont conservés ; la date UTC de capture dans release_review.json identifie l'audit, pas une nouvelle campagne. Les dates de fichiers ne servent pas à établir les versions.

SHA256 V2 : `0e583802bb9ac9c265c07a95ef8e2daf31ef90ef8fe43e7077459659bc09cd28`.

SHA256 V2.1 : `fd832a3eb473c724297c36e25139adaffcc168e5c994c0afc6fd398627c598e4`.

Le diff régénéré entre la source V2 archivée et le moteur courant correspond ligne pour ligne à `engine_change.diff`. Préparation du commit : le diff utilise `--unified=0` pour éviter les espaces structurels de lignes de contexte vides. Ses 47 ajouts/4 retraits sont inchangés et son applicabilité est vérifiée avec `git apply --check --unidiff-zero` sur une copie externe ; les outils de publication/vérification emploient le même format. Les 42 sources/artefacts frozen, les entrées des trois rapports et les snapshots exacts des runners sont vérifiés par SHA256. Tous les nombres/traces/temps du rapport combiné sont identiques à AFTER ; seules les métadonnées éditoriales du verdict sont actualisées. `before_after.csv` est confronté aux deux campagnes, 1 000 lignes. Les manifests de rankers officiels sont contrôlés par check_artifacts.py. Les 60 configurations de trois cas réutilisées du pilote dans BEFORE sont identiques au checkpoint pilote ; son runner exact et ses hashes sont également vérifiés. AFTER ne réutilise aucun cas du pilote.

## C. Five recovered configurations

Statuts complets : NO VALID = `NO_VALID_ARCHITECTURE_FOUND_WITHIN_SEARCH_DOMAIN` ; VALID = `VALID_ARCHITECTURE_FOUND`. H10 est recalculé par la fonction publique indépendamment de la décision stockée par le Beam.

| case_id | K | B | V2 status | V2.1 status | Selected architecture_id | H10 decision |
|---|---|---|---|---|---|---|
| REQ_000267 | 50 | 1 | NO VALID | VALID | ARCH_REQ_000267_58bcdfb030a4cdc7 | VALID |
| REQ_000267 | 50 | 4 | NO VALID | VALID | ARCH_REQ_000267_a6112e674e5ff569 | VALID |
| REQ_000267 | 50 | 8 | NO VALID | VALID | ARCH_REQ_000267_d3f13aa7530a6815 | VALID |
| REQ_000267 | 50 | 16 | NO VALID | VALID | ARCH_REQ_000267_88c4cb8b7cead7c8 | VALID |
| REQ_000267 | 50 | 32 | NO VALID | VALID | ARCH_REQ_000267_88c4cb8b7cead7c8 | VALID |

Le témoin initial ARCH_REQ_000267_e62f7c71ba739177 avait un drive MDT de rang 44, hors B≤32. Seuls les rangs MDT 44/47/48/50 possédaient une complétion jointe faisable ; les 32 préfixes V2 retenus étaient irréalisables. La nouvelle règle élimine ces préfixes avant la largeur, sans priorité de cas ni modification de contraintes. Les sélections finales peuvent différer du témoin et entre B, car H9 classe le pool effectivement retenu.

## D. Feasibility interpretation

785/1 000 = 78,5 % mesure la présence d'une solution H10 VALID dans le domaine configuré. Les 21,5 % d'absences ne sont pas un taux d'échec de recherche.

| Catégorie | Configurations | Preuve/portée |
|---|---|---|
| Beam VALID | 785 | 180 avec référence exhaustive commune ; 605 sur domaines sans référence de qualité |
| Known globally infeasible | 0 | Aucune preuve globale ; Top-K/cap limitent le domaine |
| Known locally infeasible | 215 | 75 selon H8/H9/H10 complets ; 140 supplémentaires selon audit scalaire |
| Known feasible but Beam missed | 0 | Références faisables et audit de toutes les absences |
| Unknown feasibility after scalar audit | 0 | Toutes les absences classées, dans le domaine configuré |
| Unknown feasibility without scalar audit | 140 | Vue limitée aux seuls pools H8/H9/H10 complets ; levée par l'audit scalaire |

Les quatre catégories VALID/globalement infaisable/faisable manqué/inconnu omettraient ici l'infaisabilité **locale**, qui doit être ajoutée pour obtenir une partition complète : 785+215+0+0+0=1 000. Les 745 configurations dont la qualité exhaustive est sautée se décomposent en 605 VALID et 140 localement infaisables. Qualité inconnue ne signifie pas faisabilité inconnue. Les 43 domaines sans solution sont tous prouvés infaisables localement : 11 budget+puissance, 16 puissance, 10 conflit joint, 6 budget. Les 157 autres domaines ont une preuve positive H10 (36 avec qualité exhaustive, 121 sans cette référence de qualité).

## E. Exhaustive audit

| Terme | Définition | Nombre par phase |
|---|---|---|
| case | Requête avec exigences/contraintes/préférences | 50 |
| configuration | case × K × B × cap de chemins | 1000 |
| domain | case × K × cap de chemins, indépendant de B | 200 |
| exhaustively evaluated domain | Toutes les paires H8, tout le pool H9, H10 sur chaque architecture | 51 |
| configuration comparable | Référence exhaustive VALID et choix Beam VALID dans ce même pool H9 | 180 |

Par phase : 50 domaines K=5 et un K=10 (REQ_001042), **42 840 architectures** entièrement construites/scorées/validées, 149 domaines de qualité sautés au-delà de 2 000 paires. Le K=10 complet n'a aucun VALID. Les 36 domaines faisables sont tous à K=5 ; les 15 autres références complètes n'ont aucun VALID. Ce sont des domaines Top-K, sans preuve globale.

```text
feasible_domains_exhaustively_known = 36
beam_recovered_feasible_domains = 36 (all five widths)
beam_missed_feasible_domains = 0
comparable_configurations = 36 * 5 = 180
```

L'audit scalaire complémentaire porte sur 43 domaines/215 configurations sans solution après correction, sans pool H9. BEFORE comportait un domaine faisable manqué supplémentaire (cinq configurations K=50). AFTER n'en comporte aucun. Ne pas ajouter les domaines scalaires aux 51 références exhaustives de qualité : les ensembles se recoupent et leurs objectifs diffèrent.

## F. Search regret / normalization regret

**Cas A : analyse terminée sur tous les domaines traitables ; couverture de validation PARTIAL.** Aucun travail analytique attendu n'est laissé inachevé sur les références disponibles. Formulation : **analysis available on exhaustively tractable domains only**. Les 15 domaines complets sans VALID n'ont pas de regret défini ; les 149 domaines trop grands n'ont pas de pool commun et aucun regret n'est inventé.

Les 180 comparaisons séparent `search = best exhaustive VALID − best retained VALID in common pool` et `normalization = best retained VALID in common pool − actual local H9 winner in common pool`. Leur somme égale le regret total pour chaque ligne. Moyennes : recherche 0,010925760807154445 ; normalisation 0,003824268361532696. Qualité moyenne 0,982148422072597, minimum 0,8142958377766908 ; 97/180 meilleurs exhaustifs survivent, 36/180 sont choisis. Pertes : recherche seule 63, normalisation seule 61, les deux 20, aucune 36. Tous ces objets quality sont inchangés entre V2 et V2.1.

## G. Acceptance criteria

| Criterion | Result | Evidence |
|---|---|---|
| Deterministic search | PASS | 1 000/1 000 fingerprints de répétitions identiques ; test déterminisme et cinq gains rejoués |
| No H5 violation | PASS | Module frozen par SHA256 ; variantes uniquement publiques H5 ; tests géométries et H10 |
| No H6 violation | PASS | Module frozen ; chemins publics H6 ; tests compatibilité et recomputation H10 |
| H7 contract preserved | PASS | EMPTY puis transitions jointes publiques ; tests states partiels et COMPLETE |
| H8 contract preserved | PASS | Constructeur public et architecture_id inchangés ; aucun produit cartésien anticipé |
| H9 unchanged | PASS | Hash H9 frozen ; formule Beam et tie-breaks comparés par AST |
| H10 mandatory | PASS | COMPLETE == H10 calls ≤ B sur 1 000 lignes ; chaque sélection rejouée passe H10 indépendant |
| No partial state marked VALID | PASS | Tests monkeypatch sur scoring/validation ; seul validated_state H10 devient terminal |
| Hard pruning physically admissible | PASS | Preuve Pareto/additivité, 300 comparaisons aléatoires avec produit scalaire brut et frontière flottante |
| No case-specific hacks | PASS | Aucun ID REQ/benchmark dans le moteur ; renommage aux cinq largeurs conserve les choix physiques |
| Global regression suite passes | PASS | 509 passed in 48.46s |
| Extended dataset evaluated | PASS | 50 nouveaux cas stratifiés, 1 000 configurations, deux répétitions chacune |
| K=50 evaluated | PASS | 250 configurations K=50 ; 42/50 cas VALID à chacune des cinq largeurs |
| Known feasible exhaustive domains recovered | PASS | 36/36 domaines complets faisables, 180/180 configurations ; audit des absences : zéro témoin manqué |
| No quality regression introduced | PASS | 180/180 objets quality identiques V2/V2.1 ; hors référence exhaustive aucune assertion de qualité |
| Search-space reduction remains significant | PASS | COMPLETE évités : minimum 94,0741 %, moyenne 98,4671 % sur 180 comparaisons |
| Runtime substantially below exhaustive where comparable | PASS | Rapport exhaustive/Beam : minimum 6,32×, médiane 41,14×, maximum 224,29× ; ranking/préflight exclus |
| Global quality/regret validation coverage | PARTIAL | Analyse terminée ; références de qualité absentes pour 149 grands domaines |
| Known global infeasibility proof | N/A | Aucune preuve globale produite ; les 215 absences sont localement infaisables |

PASS s'applique à la preuve et au périmètre vérifiés ; il ne promet pas une propriété générale de complétude/optimalité. La comparaison de temps se fait dans la campagne AFTER avec sa baseline exhaustive, sur 180 configurations comparables. La variation brute de temps entre campagnes ne prouve aucun gain causal du patch ; le contrôle alterné existant sur 40 couples est conservé.

## H. Tests finaux

Une seule campagne finale propre : les trois commandes ci-dessous exécutées une fois chacune, séquentiellement, avec le venv du dépôt. Les répertoires temporaires sont distincts ; le cache pytest est désactivé. Aucun test n'est ajouté pendant cette revue documentaire.

**dedicated : 76 passed in 2.75s**, exit code 0.

```powershell
.\venv\Scripts\python.exe -m pytest lustre_architecture_generator/tests/test_beam_search.py lustre_architecture_generator/tests/test_beam_search_benchmark.py lustre_architecture_generator/tests/test_beam_search_v2.py lustre_architecture_generator/tests/test_beam_search_extended_benchmark.py lustre_architecture_generator/tests/test_beam_search_joint_resources.py -q -p no:cacheprovider --basetemp C:/Users/LENOVO/AppData/Local/Temp/beam_v21_release_20261003_dedicated_r1
```

```text
........................................................................ [ 94%]
....                                                                     [100%]
76 passed in 2.75s

```

**architecture : 303 passed in 13.81s**, exit code 0.

```powershell
.\venv\Scripts\python.exe -m pytest lustre_architecture_generator/tests -q -p no:cacheprovider --basetemp C:/Users/LENOVO/AppData/Local/Temp/beam_v21_release_20261003_architecture_r1
```

```text
........................................................................ [ 23%]
........................................................................ [ 47%]
........................................................................ [ 71%]
........................................................................ [ 95%]
...............                                                          [100%]
303 passed in 13.81s

```

**global : 509 passed in 48.46s**, exit code 0.

```powershell
.\venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp C:/Users/LENOVO/AppData/Local/Temp/beam_v21_release_20261003_global_r1
```

```text
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

## I. Documentation changes

README.md : résumé courant V2.1, anciennes tables déplacées vers leurs liens historiques, tests finaux et limites sans taux d'échec inventé. docs/beam_search.md : référence actuelle, politique 2.1, prune joint, domaines/audits et limitations. docs/beam_search_v2.md : bannière historique V1→V2 conservant ses mesures. docs/beam_search_v2_extended.md : verdict courant, défaut historique et regrets terminés sur la portée disponible.

v2_extended/extended_benchmark.md : titre courant, verdict V2.1, verdict original séparé, définitions et limites, tests/Git final actualisés. v2_extended/problem_cases.md : titre identifié V2.1 pour les exports finaux. release_review.md : ce rapport unique de clôture A–R. Le JSON principal ne change que recommendation/original_v2_verdict/decision_rationale. Les scripts combine_before_after.py et run_extended_benchmark.py gardent ces formulations lors d'une régénération. Les raw phase reports restent des archives de collecte, éventuellement avec placeholders éditoriaux historiques ; ils ne sont pas le verdict courant.

Structure recommandée : README bref → docs/beam_search.md actuel → evaluation/ pour preuves historiques et clôture. Conserver beam_search_v2.md comme historique ; fusion éventuelle lors d'un futur cleanup. Aucun artifact supprimé.

## J. Files for future commit

Liste exacte recommandée, relative à la racine du dépôt. Aucun git add/commit exécuté. Cette allowlist inclut les fichiers non suivis absents du diff stat. Le script de revue utilise les grands JSON locaux pour recalculer ses preuves ; leur archivage externe doit préserver les SHA256 avant un futur cleanup.

### Production code

```text
lustre_architecture_generator/src/full_architecture/__init__.py
lustre_architecture_generator/src/full_architecture/beam_search.py
```

### Tests

```text
lustre_architecture_generator/tests/fixtures/beam_joint_resource_conflict.json
lustre_architecture_generator/tests/test_beam_search.py
lustre_architecture_generator/tests/test_beam_search_benchmark.py
lustre_architecture_generator/tests/test_beam_search_extended_benchmark.py
lustre_architecture_generator/tests/test_beam_search_joint_resources.py
lustre_architecture_generator/tests/test_beam_search_v2.py
```

### Documentation

```text
lustre_architecture_generator/docs/beam_search.md
lustre_architecture_generator/docs/beam_search_v2.md
lustre_architecture_generator/docs/beam_search_v2_extended.md
README.md
```

### Evaluation scripts

```text
lustre_architecture_generator/evaluation/architecture/beam_search/beam_search_v1_reference.py
lustre_architecture_generator/evaluation/architecture/beam_search/benchmark_beam_search.py
lustre_architecture_generator/evaluation/architecture/beam_search/diagnose_beam_search.py
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/audit_no_solution.py
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_joint_fix/beam_search_v2_reference.py
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_joint_fix/campaign_runner_snapshot.py
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/campaign_runner_snapshot.py
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/check_artifacts.py
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/check_interleaved_runtime.py
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/combine_before_after.py
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/pilot_100/runner_source.py
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/review_v21.py
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/run_extended_benchmark.py
```

### Benchmark artifacts worth tracking

```text
lustre_architecture_generator/evaluation/architecture/beam_search/beam_search_benchmark.csv
lustre_architecture_generator/evaluation/architecture/beam_search/beam_search_benchmark.json
lustre_architecture_generator/evaluation/architecture/beam_search/beam_search_benchmark.md
lustre_architecture_generator/evaluation/architecture/beam_search/beam_search_v2_benchmark.csv
lustre_architecture_generator/evaluation/architecture/beam_search/beam_search_v2_benchmark.json
lustre_architecture_generator/evaluation/architecture/beam_search/beam_search_v2_benchmark.md
lustre_architecture_generator/evaluation/architecture/beam_search/beam_search_v2_benchmark_before_after.csv
lustre_architecture_generator/evaluation/architecture/beam_search/diagnostic_v1.json
lustre_architecture_generator/evaluation/architecture/beam_search/diagnostic_v1.md
lustre_architecture_generator/evaluation/architecture/beam_search/diagnostic_v2.json
lustre_architecture_generator/evaluation/architecture/beam_search/diagnostic_v2.md
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/aggregates.csv
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/artifact_validation.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_after.csv
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_joint_fix/aggregates.csv
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_joint_fix/diagnostic_joint.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_joint_fix/diagnostic_joint.md
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_joint_fix/extended_benchmark.csv
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_joint_fix/extended_benchmark.md
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_joint_fix/joint_feasible_drive_ranks.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_joint_fix/no_solution_audit.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_joint_fix/no_solution_audit.md
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_joint_fix/reproduction_tests.txt
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/engine_change.diff
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/extended_benchmark.csv
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/extended_benchmark.md
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/frozen_sources_initial.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/interleaved_runtime.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/no_solution_audit.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/no_solution_audit.md
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/official_ranker_manifest.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/problem_cases.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/problem_cases.md
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/release_review.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/release_review.md
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/sample_manifest.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/width_sensitivity.json
lustre_architecture_generator/evaluation/architecture/beam_search/width_sensitivity.csv
lustre_architecture_generator/evaluation/architecture/beam_search/width_sensitivity.json
lustre_architecture_generator/evaluation/architecture/beam_search/width_sensitivity.md
lustre_architecture_generator/evaluation/architecture/beam_search/width_sensitivity_v2.csv
lustre_architecture_generator/evaluation/architecture/beam_search/width_sensitivity_v2.json
lustre_architecture_generator/evaluation/architecture/beam_search/width_sensitivity_v2.md
```

`diagnostic_v1.json` et `diagnostic_v2.json` sont retenus : petits diagnostics historiques utiles (préfixes, identités et regrets). `engine_change.diff` est retenu : preuve compacte du correctif, régénérable avec la source V2 conservée. Les snapshots des runners conservent la provenance des hash de campagnes malgré l'évolution de leur rendu documentaire.

## K. Files not recommended for commit

Recommandations seulement : tous ces fichiers restent présents. Les JSON > 1 MiB sont signalés ; les trois gros JSON de campagne font chacun > 7 MiB. Préférer archive externe avec hashes et regeneration plutôt qu'un commit de dumps/duplications/checkpoints. Les tailles et SHA256 des artefacts de données, y compris les grands fichiers proposés hors Git, sont enregistrés dans release_review.json, champ benchmark_artifact_sha256.

| Fichier exact | Raison |
|---|---|
| lustre_architecture_generator/evaluation/architecture/beam_search/initial_git_status_v2.txt | État de travail, chemins locaux ou log duplicable ; résumé final versionnable |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/after_joint_fix/aggregates.csv | Copie brute AFTER conservée ; doublon des exports finaux utiles |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/after_joint_fix/campaign_runner_snapshot.py | Copie brute AFTER conservée ; doublon des exports finaux utiles |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/after_joint_fix/checkpoint.json | Checkpoint reproductible, volumineux et/ou non final |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/after_joint_fix/extended_benchmark.csv | Copie brute AFTER conservée ; doublon des exports finaux utiles |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/after_joint_fix/extended_benchmark.json | JSON détaillé > 7 MiB reproductible ; garder en archive externe avec SHA256 |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/after_joint_fix/extended_benchmark.md | Copie brute AFTER conservée ; doublon des exports finaux utiles |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/after_joint_fix/frozen_sources_initial.json | Copie brute AFTER conservée ; doublon des exports finaux utiles |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/after_joint_fix/initial_git_status.txt | Copie brute AFTER conservée ; doublon des exports finaux utiles |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/after_joint_fix/initial_verification.json | Copie brute AFTER conservée ; doublon des exports finaux utiles |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/after_joint_fix/no_solution_audit.json | Copie brute AFTER conservée ; doublon des exports finaux utiles |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/after_joint_fix/no_solution_audit.md | Copie brute AFTER conservée ; doublon des exports finaux utiles |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/after_joint_fix/official_ranker_manifest.json | Copie brute AFTER conservée ; doublon des exports finaux utiles |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/after_joint_fix/problem_cases.json | Copie brute AFTER conservée ; doublon des exports finaux utiles |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/after_joint_fix/problem_cases.md | Copie brute AFTER conservée ; doublon des exports finaux utiles |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/after_joint_fix/sample_manifest.json | Copie brute AFTER conservée ; doublon des exports finaux utiles |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/after_joint_fix/width_sensitivity.json | Copie brute AFTER conservée ; doublon des exports finaux utiles |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_joint_fix/extended_benchmark.json | JSON détaillé > 7 MiB reproductible ; garder en archive externe avec SHA256 |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_joint_fix/problem_cases.json | Export intermédiaire BEFORE reproductible ou log local |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_joint_fix/problem_cases.md | Export intermédiaire BEFORE reproductible ou log local |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_joint_fix/unit_tests.txt | Export intermédiaire BEFORE reproductible ou log local |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_joint_fix/width_sensitivity.json | Export intermédiaire BEFORE reproductible ou log local |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/checkpoint.json | Checkpoint reproductible, volumineux et/ou non final |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/extended_benchmark.json | JSON détaillé > 7 MiB reproductible ; garder en archive externe avec SHA256 |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/final_global_tests.txt | État de travail, chemins locaux ou log duplicable ; résumé final versionnable |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/final_unit_tests.txt | État de travail, chemins locaux ou log duplicable ; résumé final versionnable |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/git_diff_check.txt | État de travail, chemins locaux ou log duplicable ; résumé final versionnable |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/git_diff_stat.txt | État de travail, chemins locaux ou log duplicable ; résumé final versionnable |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/git_status.txt | État de travail, chemins locaux ou log duplicable ; résumé final versionnable |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/initial_git_status.txt | État de travail, chemins locaux ou log duplicable ; résumé final versionnable |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/initial_verification.json | État de travail, chemins locaux ou log duplicable ; résumé final versionnable |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/joint_fix_unit_tests.txt | État de travail, chemins locaux ou log duplicable ; résumé final versionnable |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/pilot_100/checkpoint.json | Pilote partiel reproductible, hors campagne finale ; conserver sur disque |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/pilot_100/sample_manifest.json | Pilote partiel reproductible, hors campagne finale ; conserver sur disque |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/release_architecture_tests.txt | État de travail, chemins locaux ou log duplicable ; résumé final versionnable |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/release_dedicated_tests.txt | État de travail, chemins locaux ou log duplicable ; résumé final versionnable |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/release_git_diff_check.txt | État de travail, chemins locaux ou log duplicable ; résumé final versionnable |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/release_git_diff_stat.txt | État de travail, chemins locaux ou log duplicable ; résumé final versionnable |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/release_git_status.txt | État de travail, chemins locaux ou log duplicable ; résumé final versionnable |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/release_global_tests.txt | État de travail, chemins locaux ou log duplicable ; résumé final versionnable |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/release_review_initial.json | État de travail, chemins locaux ou log duplicable ; résumé final versionnable |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/release_tests.json | État de travail, chemins locaux ou log duplicable ; résumé final versionnable |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/release_untracked.txt | État de travail, chemins locaux ou log duplicable ; résumé final versionnable |
| lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/verification.json | État de travail, chemins locaux ou log duplicable ; résumé final versionnable |

Les 59 suppressions non liées au Beam sont également exclues du futur commit ; leur liste exacte est visible en N. initial_git_status_v2.txt et les snapshots/logs Git locaux ne sont pas des résultats scientifiques à versionner. Le rapport et release_review.json conservent les résultats nécessaires à la revue.

## L. Artifact sizes

Inventaire récursif de **chaque** .json/.csv/.md/.diff sous evaluation/architecture/beam_search/, en octets et KiB (1 KiB=1 024 octets). Racine commune : lustre_architecture_generator/evaluation/architecture/beam_search/. Les documents Beam et le fixture sont inclus séparément par leur chemin complet. Les gros fichiers et les duplications reproductibles sont exclus de l'allowlist, sans suppression.

| Fichier | Octets | KiB | Volume | Recommandation |
|---|---|---|---|---|
| lustre_architecture_generator/docs/beam_search.md | 21687 | 21.18 |  | Documentation |
| lustre_architecture_generator/docs/beam_search_v2.md | 12003 | 11.72 |  | Documentation |
| lustre_architecture_generator/docs/beam_search_v2_extended.md | 5717 | 5.58 |  | Documentation |
| beam_search_benchmark.csv | 3286 | 3.21 |  | Benchmark artifacts worth tracking |
| beam_search_benchmark.json | 174303 | 170.22 |  | Benchmark artifacts worth tracking |
| beam_search_benchmark.md | 4542 | 4.44 |  | Benchmark artifacts worth tracking |
| beam_search_v2_benchmark.csv | 1494 | 1.46 |  | Benchmark artifacts worth tracking |
| beam_search_v2_benchmark.json | 336844 | 328.95 |  | Benchmark artifacts worth tracking |
| beam_search_v2_benchmark.md | 8448 | 8.25 |  | Benchmark artifacts worth tracking |
| beam_search_v2_benchmark_before_after.csv | 5398 | 5.27 |  | Benchmark artifacts worth tracking |
| diagnostic_v1.json | 42550 | 41.55 |  | Benchmark artifacts worth tracking |
| diagnostic_v1.md | 2564 | 2.50 |  | Benchmark artifacts worth tracking |
| diagnostic_v2.json | 59936 | 58.53 |  | Benchmark artifacts worth tracking |
| diagnostic_v2.md | 3351 | 3.27 |  | Benchmark artifacts worth tracking |
| v2_extended/after_joint_fix/aggregates.csv | 3741 | 3.65 |  | Local only |
| v2_extended/after_joint_fix/checkpoint.json | 7362884 | 7190.32 | LARGE ≥1 MiB | Local only |
| v2_extended/after_joint_fix/extended_benchmark.csv | 683647 | 667.62 |  | Local only |
| v2_extended/after_joint_fix/extended_benchmark.json | 7777123 | 7594.85 | LARGE ≥1 MiB | Local only |
| v2_extended/after_joint_fix/extended_benchmark.md | 20922 | 20.43 |  | Local only |
| v2_extended/after_joint_fix/frozen_sources_initial.json | 6493 | 6.34 |  | Local only |
| v2_extended/after_joint_fix/initial_verification.json | 628 | 0.61 |  | Local only |
| v2_extended/after_joint_fix/no_solution_audit.json | 80852 | 78.96 |  | Local only |
| v2_extended/after_joint_fix/no_solution_audit.md | 4447 | 4.34 |  | Local only |
| v2_extended/after_joint_fix/official_ranker_manifest.json | 810 | 0.79 |  | Local only |
| v2_extended/after_joint_fix/problem_cases.json | 98079 | 95.78 |  | Local only |
| v2_extended/after_joint_fix/problem_cases.md | 6742 | 6.58 |  | Local only |
| v2_extended/after_joint_fix/sample_manifest.json | 16575 | 16.19 |  | Local only |
| v2_extended/after_joint_fix/width_sensitivity.json | 273059 | 266.66 |  | Local only |
| v2_extended/aggregates.csv | 3741 | 3.65 |  | Benchmark artifacts worth tracking |
| v2_extended/artifact_validation.json | 731 | 0.71 |  | Benchmark artifacts worth tracking |
| v2_extended/before_after.csv | 230917 | 225.50 |  | Benchmark artifacts worth tracking |
| v2_extended/before_joint_fix/aggregates.csv | 3688 | 3.60 |  | Benchmark artifacts worth tracking |
| v2_extended/before_joint_fix/diagnostic_joint.json | 13304 | 12.99 |  | Benchmark artifacts worth tracking |
| v2_extended/before_joint_fix/diagnostic_joint.md | 2313 | 2.26 |  | Benchmark artifacts worth tracking |
| v2_extended/before_joint_fix/extended_benchmark.csv | 676798 | 660.94 |  | Benchmark artifacts worth tracking |
| v2_extended/before_joint_fix/extended_benchmark.json | 7663340 | 7483.73 | LARGE ≥1 MiB | Local only |
| v2_extended/before_joint_fix/extended_benchmark.md | 17275 | 16.87 |  | Benchmark artifacts worth tracking |
| v2_extended/before_joint_fix/joint_feasible_drive_ranks.json | 493 | 0.48 |  | Benchmark artifacts worth tracking |
| v2_extended/before_joint_fix/no_solution_audit.json | 82761 | 80.82 |  | Benchmark artifacts worth tracking |
| v2_extended/before_joint_fix/no_solution_audit.md | 4531 | 4.42 |  | Benchmark artifacts worth tracking |
| v2_extended/before_joint_fix/problem_cases.json | 98175 | 95.87 |  | Local only |
| v2_extended/before_joint_fix/problem_cases.md | 6743 | 6.58 |  | Local only |
| v2_extended/before_joint_fix/width_sensitivity.json | 272914 | 266.52 |  | Local only |
| v2_extended/checkpoint.json | 7249763 | 7079.85 | LARGE ≥1 MiB | Local only |
| v2_extended/engine_change.diff | 3708 | 3.62 |  | Benchmark artifacts worth tracking |
| v2_extended/extended_benchmark.csv | 690784 | 674.59 |  | Benchmark artifacts worth tracking |
| v2_extended/extended_benchmark.json | 8986012 | 8775.40 | LARGE ≥1 MiB | Local only |
| v2_extended/extended_benchmark.md | 38280 | 37.38 |  | Benchmark artifacts worth tracking |
| v2_extended/frozen_sources_initial.json | 6493 | 6.34 |  | Benchmark artifacts worth tracking |
| v2_extended/initial_verification.json | 628 | 0.61 |  | Local only |
| v2_extended/interleaved_runtime.json | 24883 | 24.30 |  | Benchmark artifacts worth tracking |
| v2_extended/no_solution_audit.json | 80852 | 78.96 |  | Benchmark artifacts worth tracking |
| v2_extended/no_solution_audit.md | 4447 | 4.34 |  | Benchmark artifacts worth tracking |
| v2_extended/official_ranker_manifest.json | 810 | 0.79 |  | Benchmark artifacts worth tracking |
| v2_extended/pilot_100/checkpoint.json | 1091503 | 1065.92 | LARGE ≥1 MiB | Local only |
| v2_extended/pilot_100/sample_manifest.json | 29317 | 28.63 |  | Local only |
| v2_extended/problem_cases.json | 98769 | 96.45 |  | Benchmark artifacts worth tracking |
| v2_extended/problem_cases.md | 6744 | 6.59 |  | Benchmark artifacts worth tracking |
| v2_extended/release_review.json | 59635 | 58.24 |  | Benchmark artifacts worth tracking |
| v2_extended/release_review.md | 64210 | 62.71 |  | Benchmark artifacts worth tracking |
| v2_extended/release_review_initial.json | 27348 | 26.71 |  | Local only |
| v2_extended/release_tests.json | 2505 | 2.45 |  | Local only |
| v2_extended/sample_manifest.json | 16575 | 16.19 |  | Benchmark artifacts worth tracking |
| v2_extended/verification.json | 12420 | 12.13 |  | Local only |
| v2_extended/width_sensitivity.json | 273059 | 266.66 |  | Benchmark artifacts worth tracking |
| width_sensitivity.csv | 783 | 0.76 |  | Benchmark artifacts worth tracking |
| width_sensitivity.json | 61531 | 60.09 |  | Benchmark artifacts worth tracking |
| width_sensitivity.md | 2050 | 2.00 |  | Benchmark artifacts worth tracking |
| width_sensitivity_v2.csv | 1302 | 1.27 |  | Benchmark artifacts worth tracking |
| width_sensitivity_v2.json | 132642 | 129.53 |  | Benchmark artifacts worth tracking |
| width_sensitivity_v2.md | 3403 | 3.32 |  | Benchmark artifacts worth tracking |
| lustre_architecture_generator/tests/fixtures/beam_joint_resource_conflict.json | 260620 | 254.51 |  | Tests |

## M. Known limitations

### Known limitations — not correctness bugs

- Optimalité globale/complétude non garanties avec une largeur B limitée et sans backtracking.
- Qualité non monotone avec B ; 9/36 groupes comparables monotones seulement.
- H9 dépend du pool, sans modification de ses normalisateurs.
- Références exhaustives de qualité sautées pour 149 grands domaines ; regrets terminés seulement sur les traitables.
- 50 cas synthétiques stratifiés seulement ; catalogue/dataset/rankers actuels, sans validation en production.
- Caches/enveloppes et mémoire totale non bornés par B ; mesures en entrées, sans RSS.
- Top-K et cap de chemins peuvent exclure des solutions hors domaine.

**Defects :** aucun défaut actuel V2.1 démontré. Le défaut historique V2 était la perte par largeur de branches réellement faisables lorsque des minima indépendants laissaient survivre des préfixes fantômes ; il est corrigé et couvert par les régressions. Les limitations ci-dessus ne sont pas des bugs de correction.

## N. Git status

Beam-related : README, exports publics, moteur, tests/fixture, docs et evaluation Beam. Pre-existing unrelated : **59 suppressions**, exactement identiques à l'état initial ; elles n'ont pas été touchées. Pas de staging. Sortie exacte :

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

Fichiers untracked, sortie exacte de `git ls-files --others --exclude-standard` (ils appartiennent tous au périmètre Beam dans cet état de travail, y compris les locaux déconseillés) :

```text
lustre_architecture_generator/docs/beam_search.md
lustre_architecture_generator/docs/beam_search_v2.md
lustre_architecture_generator/docs/beam_search_v2_extended.md
lustre_architecture_generator/evaluation/architecture/beam_search/beam_search_benchmark.csv
lustre_architecture_generator/evaluation/architecture/beam_search/beam_search_benchmark.json
lustre_architecture_generator/evaluation/architecture/beam_search/beam_search_benchmark.md
lustre_architecture_generator/evaluation/architecture/beam_search/beam_search_v1_reference.py
lustre_architecture_generator/evaluation/architecture/beam_search/beam_search_v2_benchmark.csv
lustre_architecture_generator/evaluation/architecture/beam_search/beam_search_v2_benchmark.json
lustre_architecture_generator/evaluation/architecture/beam_search/beam_search_v2_benchmark.md
lustre_architecture_generator/evaluation/architecture/beam_search/beam_search_v2_benchmark_before_after.csv
lustre_architecture_generator/evaluation/architecture/beam_search/benchmark_beam_search.py
lustre_architecture_generator/evaluation/architecture/beam_search/diagnose_beam_search.py
lustre_architecture_generator/evaluation/architecture/beam_search/diagnostic_v1.json
lustre_architecture_generator/evaluation/architecture/beam_search/diagnostic_v1.md
lustre_architecture_generator/evaluation/architecture/beam_search/diagnostic_v2.json
lustre_architecture_generator/evaluation/architecture/beam_search/diagnostic_v2.md
lustre_architecture_generator/evaluation/architecture/beam_search/initial_git_status_v2.txt
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/after_joint_fix/aggregates.csv
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/after_joint_fix/campaign_runner_snapshot.py
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/after_joint_fix/checkpoint.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/after_joint_fix/extended_benchmark.csv
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/after_joint_fix/extended_benchmark.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/after_joint_fix/extended_benchmark.md
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/after_joint_fix/frozen_sources_initial.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/after_joint_fix/initial_git_status.txt
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/after_joint_fix/initial_verification.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/after_joint_fix/no_solution_audit.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/after_joint_fix/no_solution_audit.md
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/after_joint_fix/official_ranker_manifest.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/after_joint_fix/problem_cases.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/after_joint_fix/problem_cases.md
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/after_joint_fix/sample_manifest.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/after_joint_fix/width_sensitivity.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/aggregates.csv
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/artifact_validation.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/audit_no_solution.py
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_after.csv
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_joint_fix/aggregates.csv
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_joint_fix/beam_search_v2_reference.py
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_joint_fix/campaign_runner_snapshot.py
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_joint_fix/diagnostic_joint.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_joint_fix/diagnostic_joint.md
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_joint_fix/extended_benchmark.csv
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_joint_fix/extended_benchmark.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_joint_fix/extended_benchmark.md
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_joint_fix/joint_feasible_drive_ranks.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_joint_fix/no_solution_audit.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_joint_fix/no_solution_audit.md
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_joint_fix/problem_cases.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_joint_fix/problem_cases.md
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_joint_fix/reproduction_tests.txt
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_joint_fix/unit_tests.txt
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/before_joint_fix/width_sensitivity.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/campaign_runner_snapshot.py
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/check_artifacts.py
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/check_interleaved_runtime.py
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/checkpoint.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/combine_before_after.py
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/engine_change.diff
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/extended_benchmark.csv
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/extended_benchmark.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/extended_benchmark.md
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/final_global_tests.txt
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/final_unit_tests.txt
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/frozen_sources_initial.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/git_diff_check.txt
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/git_diff_stat.txt
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/git_status.txt
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/initial_git_status.txt
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/initial_verification.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/interleaved_runtime.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/joint_fix_unit_tests.txt
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/no_solution_audit.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/no_solution_audit.md
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/official_ranker_manifest.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/pilot_100/checkpoint.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/pilot_100/runner_source.py
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/pilot_100/sample_manifest.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/problem_cases.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/problem_cases.md
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/release_architecture_tests.txt
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/release_dedicated_tests.txt
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/release_git_diff_check.txt
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/release_git_diff_stat.txt
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/release_git_status.txt
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/release_global_tests.txt
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/release_review.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/release_review.md
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/release_review_initial.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/release_tests.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/release_untracked.txt
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/review_v21.py
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/run_extended_benchmark.py
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/sample_manifest.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/verification.json
lustre_architecture_generator/evaluation/architecture/beam_search/v2_extended/width_sensitivity.json
lustre_architecture_generator/evaluation/architecture/beam_search/width_sensitivity.csv
lustre_architecture_generator/evaluation/architecture/beam_search/width_sensitivity.json
lustre_architecture_generator/evaluation/architecture/beam_search/width_sensitivity.md
lustre_architecture_generator/evaluation/architecture/beam_search/width_sensitivity_v2.csv
lustre_architecture_generator/evaluation/architecture/beam_search/width_sensitivity_v2.json
lustre_architecture_generator/evaluation/architecture/beam_search/width_sensitivity_v2.md
lustre_architecture_generator/src/full_architecture/beam_search.py
lustre_architecture_generator/tests/fixtures/beam_joint_resource_conflict.json
lustre_architecture_generator/tests/test_beam_search.py
lustre_architecture_generator/tests/test_beam_search_benchmark.py
lustre_architecture_generator/tests/test_beam_search_extended_benchmark.py
lustre_architecture_generator/tests/test_beam_search_joint_resources.py
lustre_architecture_generator/tests/test_beam_search_v2.py
```

## O. Git diff stat

Commande `git diff --stat`, exit code 0. stdout puis stderr exacts. Le stat comprend les suppressions préexistantes et omet les fichiers non suivis, listés en N et classés en J/K.

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

## P. Git diff check

Commande `git diff --check`, exit code 0 : **PASS** (aucune erreur whitespace). stdout vide ; stderr exact ci-dessous contient uniquement les avertissements LF→CRLF.

```text
warning: in the working copy of 'lustre_architecture_generator/src/full_architecture/__init__.py', LF will be replaced by CRLF the next time Git touches it
```

## Q. Suggested commit message

Suggested commit title:

```text
feat(beam): finalize V2.1 joint-resource pruning and validation
```

Suggested commit body:

```text
Add deterministic bounded Beam Search with generic Pareto cost/power completion pruning.
Keep H5-H10, ranking, heuristic weights and validity contracts unchanged.
Validate V2.1 on 50 synthetic cases / 1,000 configurations, including K=50:
785 H10 VALID selections, five recoveries, zero losses and no known feasible domain missed.
Preserve quality on all 180 comparable configurations and document validation limitations.
Verify 76 dedicated, 303 architecture and 509 global tests.
Track focused code, fixtures, documentation and compact reproducibility evidence.
```

Proposition uniquement. Aucun commit effectué ; les suppressions préexistantes et artifacts locaux sont exclus.

## R. Final status

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
