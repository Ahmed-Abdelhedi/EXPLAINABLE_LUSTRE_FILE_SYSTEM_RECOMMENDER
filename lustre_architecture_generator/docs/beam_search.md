# Beam Search V2.1 au-dessus de H5–H10

`beam_search_architectures(handoff=..., hardware_catalog=..., beam_width=B,
max_paths_per_variant=P)` est une couche de recherche dédiée. Elle consomme le
handoff H1/H2 actuel, sans refaire le ranking, puis limite les branches **avant**
la construction du produit cartésien H8. Elle ne trie pas un pool exhaustif déjà
généré. Les modules physiques H5–H10 restent inchangés. La V2 ajoute une
préparation indépendante des deux rôles pour rendre visibles les ressources
matérielles futures avant la sélection par largeur. La V1 et ses mesures sont
conservées dans les artifacts de référence.

La version courante est **V2.1** : elle ajoute un contrôle de faisabilité
conjointe coût/puissance sur les couples non dominés par préfixe. Les minima
séparés et la formule de l'heuristique restent identiques. Le
[diagnostic élargi](beam_search_v2_extended.md) présente le défaut V2,
les tests de reproduction et les deux campagnes complètes conservées.

## Fonctions existantes réutilisées

| Contrat | Fonction publique | Usage |
|---|---|---|
| H1/H2 | `assert_valid_architecture_handoff` | Contrat Top-K et évidence du filtre |
| H4 | `validate_hardware_catalog_bundle` | Schémas, géométries et IDs catalogue |
| H5 | `enumerate_candidate_protections` | Calcul et vérification de chaque profil |
| H6 | `find_compatible_hardware_paths` | Chemins compatibles et ressources minimales |
| Couverture H10-B | `role_option_cost_power` | Contribution additive H5 + H6 |
| H7 | `new_full_architecture_state`, `apply_drive_selection`, `apply_protection_selection` | États physiques partiels légaux |
| H7/H8 | `build_complete_state_from_choices`, `architecture_id` | Construction complète et identité physique H8 |
| H9 | `normalize_preference_weights`, `score_generated_architectures` | Poids amont et scoring final |
| H10 | `validate_complete_architecture` | Décision finale indépendante |

## Étapes et largeur

```text
Top-K MDT/OST reçu dans le handoff
  ↓ LOOKAHEAD_MDT / LOOKAHEAD_OST : H5 puis H6, sans coupler les rôles
  ↓ enveloppes par préfixe : ressources minimales et crédits optimistes
  ↓ MDT_DRIVE : drives MDT → borne pré-RAID → heuristique → garder B
  ↓ MDT_PROTECTION : profils du catalogue → H5 → borne protégée → garder B
  ↓ MDT_HARDWARE : chemins H6 compatibles → borne avec hardware → garder B
  ↓ OST_DRIVE : chaque survivant × drives OST → borne conjointe → garder B
  ↓ OST_PROTECTION : chaque survivant × profils H5 OST → garder B
  ↓ OST_HARDWARE : chaque survivant × chemins H6 OST → garder B
  ↓ au plus B ensembles de choix
Construction publique H7 identique à H8 → COMPLETE, PENDING_FULL_VALIDATOR
  ↓ H9 sur tous les COMPLETE survivants
  ↓ H10 sur chaque COMPLETE, dans l’ordre H9
Premier VALID de cet ordre → recommandation
```

À chaque étape, seules les branches survivantes de l’étape précédente sont
développées. Les rejets durs précèdent l’heuristique. Une sélection incrémentale
conserve au plus B branches et utilise au plus B+1 places dans son buffer de tri.
Toutes les expansions du niveau sont examinées : aucun arrêt au premier B
enfants. Les caches H5/H6 préparés par rôle évitent les appels identiques dans
la recherche principale. Cette préparation visite les options individuelles
du Top-K avant les suppressions par largeur, sans construire de state COMPLETE
ni visiter le produit MDT×OST. Son travail est comptabilisé et chronométré.

`BeamState` est un wrapper séparé, avec choix par rôle, `branch_id`, score Beam,
trace et `architecture_state`. H7 exige les deux rôles pour ses transitions :
le state reste EMPTY pendant les choix MDT, devient DRIVES_SELECTED à
OST_DRIVE, puis PROTECTION_SELECTED à OST_PROTECTION. Les informations MDT
connues auparavant restent dans le wrapper. Aucun state H7 artificiel avec un
seul rôle ni validation partielle n’est introduit. La construction finale
utilise les six transitions publiques H7 appelées par H8.

## Heuristique exacte et politique V2.1

Schéma courant : `2.1` ; politique :
`BEAM_OPTIMISTIC_COMPLETION_V2_JOINT_RESOURCES`. La formule h2 introduite
en V2 est inchangée ; le suffixe désigne le contrôle joint de ressources.

Les poids numériques amont sont normalisés par la fonction publique H9 :
`w_i = preference_i / sum(preference_i)`. Les labels qualitatifs ne sont pas
convertis en poids.

```text
h2 = w_performance P+ + w_cost C+ + w_power E+ + w_reliability L+
```

Pour chaque rôle r, E_r contient les complétions H5/H6 compatibles avec les
choix déjà fixés : tous les choix si le rôle est inconnu ; ceux du drive à
l’étape DRIVE ; ceux du drive/profil à PROTECTION ; le chemin exact à HARDWARE.
Les options impossibles même sans l’autre rôle sont éliminées en préparation.

- **C+** : `1 / (1 + (min_E_MDT cost + min_E_OST cost)/budget)`.
- **E+** : même formule pour la puissance et sa limite.
- **P+** : somme des maxima par rôle des sommes de headroom
  `max(0, 1-required/provided)`, divisée par le nombre total de métriques actives
  des deux rôles. Les exigences sont fixes, donc ce dénominateur ne dépend pas
  de la protection. Sans métrique active, le crédit vaut zéro.
- **L+** : somme des maxima par rôle de
  `f/(1+f) + HA_enabled`, divisée par quatre : une protection et un chemin HA
  pour chacun des deux rôles.
- **R+**, utilisé seulement pour départager : moyenne des maxima par rôle de
  `1/selection_rank` sur les complétions compatibles.

Pour une limite nulle, le crédit ressource vaut 1 si la borne est zéro, sinon
zéro. Les minima et maxima de composantes peuvent provenir d’options différentes.
Ils constituent un signal optimiste de classement, sans prétendre représenter
une architecture réalisable ni une borne supérieure du score H9.

La V1 utilisait `h1 = .25 R + .75 (wP P + wC C + wE E + wL L)` sur les seules
informations déjà sélectionnées. La V2 fait intervenir les deux rôles et leurs
ressources matérielles futures, et laisse les préférences utilisateur définir
le score principal. Le ranking définit toujours le Top-K et départage les
égalités. Aucun poids n’a été ajusté selon un case_id ou un drive du benchmark.
L’appel public `beam_heuristic` sans enveloppes conserve le calcul local V1
pour compatibilité ; le moteur utilise systématiquement les enveloppes V2.

Ordre de sélection : h2 décroissant, R+ décroissant, clés MDT puis OST croissantes.
Une clé de choix contient : rang de sélection, ID drive, indice catalogue et
ID protection, indice de chemin H6, puis IDs/mode hardware. Une valeur absente
est représentée par 0 ou une chaîne vide. L’ordre des profils du catalogue et
l’ordre des chemins H6 sont préservés. Le classement final H9 utilise son
tie-break existant `architecture_id`. Aucun aléatoire ni temps n’entre dans
les clés. Seul `summary.search_seconds` varie entre deux exécutions identiques.

## Pruning dur et suppression par largeur

| Raison | Condition et étape | Justification |
|---|---|---|
| `budget_lower_bound_exceeded` | Toute expansion : borne connue > budget | Les coûts H5/H6 sont non négatifs et les rôles sont isolés ; une continuation ne réduit pas la contribution choisie |
| `power_lower_bound_exceeded` | Toute expansion : borne connue > puissance | Même monotonie sur les contributions de puissance |
| `h5_protection_impossible` | Étape protection : H5 ne peut produire une variante satisfaite | La branche fixe ce drive et ce profil ; aucune géométrie de remplacement n’est inventée |
| `h6_no_compatible_path` | Étape hardware : aucun chemin retourné par H6 | H6 est l’autorité de compatibilité du catalogue fourni ; une continuation OST ne répare pas un chemin MDT impossible |
| `no_compatible_completion` | Toute expansion : un préfixe n’a aucune complétion H5/H6 encore possible | Les choices fixés ne peuvent pas être complétés dans le domaine fourni |
| `completion_budget_lower_bound_exceeded` | Toute expansion : somme des minima de coût des deux rôles > budget | Chaque continuation coûte au moins le minimum de chacun des rôles isolés |
| `completion_power_lower_bound_exceeded` | Même condition pour la puissance | Chaque continuation consomme au moins ces minima indépendants |
| `completion_joint_budget_power_conflict` | Après les bornes indépendantes, avant la largeur : aucun couple réel de complétions MDT/OST ne satisfait simultanément budget et puissance | Les points Pareto proviennent uniquement de complétions H5/H6 ; supprimer un point dominé conserve toute possibilité de respecter ces deux contraintes |

La sûreté ne demande pas que les minima coût et puissance correspondent à la
même option : chacun est inférieur ou égal à la contribution de toute option.
Leur admissibilité est testée sur toutes les complétions d’un fixture. Les
bornes pré-RAID et les rejets H5/H6 existants restent présents. Aucun crédit
optimiste de performance ou fiabilité n’est employé pour le pruning dur.

Pour éviter un rejet dû au regroupement différent des sommes en virgule
flottante, le test de borne utilise
`tolerance = max(1e-12, 4*ulp(borne), 4*ulp(limite))`. Une branche à la frontière
peut donc survivre ; H10 applique ensuite sa propre tolérance inchangée.
Le contrôle joint utilise des limites fixes `limite + max(1e-12, 8*ulp(limite))`
pour garder le balayage monotone et éviter les faux rejets dus au regroupement
des sommes. Ces marges peuvent retenir un candidat que H10 rejettera ensuite.

`beam_width_limit` est compté séparément : ce rejet est **heuristique**, pas une
preuve d’impossibilité. Une branche physiquement réalisable peut être perdue.
Une branche violant budget et puissance compte une seule fois dans
`hard_pruned`, avec deux occurrences dans les compteurs de raisons. Ces
occurrences ne doivent donc pas être additionnées pour compter les branches.

Les entrées mal formées, IDs dupliqués et géométries invalides du catalogue
déclenchent une erreur de contrat. Elles ne deviennent pas des échecs de
recherche silencieux. Aucun déficit provisoire de capacité/IOPS/débit, ni HA
encore inconnue, n’est ajouté comme règle de rejet indépendante de H5/H6.

## API et résultat

```python
from lustre_architecture_generator.src.full_architecture import beam_search_architectures

result = beam_search_architectures(
    handoff=handoff,
    hardware_catalog=hardware_catalog,
    beam_width=8,
    max_paths_per_variant=2,
)
selected = result["best_validated_architecture"]
if selected is not None:
    assert selected["h10"]["decision"] == "VALID"
```

K est uniquement `handoff.requested_top_k` ; `actual_top_k` indique le nombre
effectivement disponible par rôle. B et le nombre maximum de chemins par
variante sont des entiers strictement positifs, sans coercition des booléens,
chaînes ou nombres fractionnaires.

Le résultat JSON contient les paramètres, `case_id`, les compteurs de branches
créées/explorées, rejets durs/par largeur et raisons par étape, nombre maximum
d’états retenus et taille du buffer, appels H5/H6, COMPLETE produits, appels H10,
nombre VALID, durée, score H9 final et meilleur ID. `search_trace` résume chaque
niveau. Chaque architecture évaluée contient son état terminal H10, les
artifacts `h9` et `h10`, ainsi qu’une trace de choix avec parent, profondeur,
rang Beam, métriques et bornes de l’heuristique.

`lookahead_trace` comptabilise la préparation par rôle. `branches_created` et
`branches_explored` incluent préparation et recherche principale ;
`main_search_branches_created` et `lookahead_branches_created` les séparent.
Les rejets durs des deux phases entrent dans le total ; les rejets par largeur
concernent seulement les six niveaux principaux. `max_active_states` désigne
les states du beam principal, sans cacher le stockage des caches/enveloppes.

`beam_search_applied=True` appartient au résultat Beam. Les artifacts H9 et
H10 gardent **`beam_search_applied=False`** : ces couches n’appliquent pas
elles-mêmes de Beam Search. Les states sont PENDING avant H10 ; seul son état
`validated_state` devient VALIDATED ou INVALID.

Sans décision VALID, le statut est
`NO_VALID_ARCHITECTURE_FOUND_WITHIN_SEARCH_DOMAIN`, et la recommandation vaut
`None`. `global_infeasibility_claimed=False` est toujours explicite.

## Benchmark K × B et comparaison équitable

Le script `evaluation/architecture/beam_search/benchmark_beam_search.py` utilise
le dataset architecture existant, le catalogue existant et les rankers
LightGBM officiels via `build_runtime_handoff`. Il accepte les chemins d’entrée,
IDs de cas, K, B, nombre de chemins, répétitions, limite exhaustive et sortie.
Les valeurs par défaut sont K = 5/10/20/50 et B = 1/2/4/8/16/32.

La baseline couvre intégralement **le même** handoff, tous les profils, et le
même cap de chemins H6, sans cap de role options ni préfixe d’architectures.
Un précontrôle énumère les options pour compter le produit. Au-delà de
`--exhaustive-max-pairs`, la baseline est `SKIPPED_PAIR_LIMIT` : ses métriques
de qualité/temps ne sont pas estimées. Beam continue sur le domaine configuré.

H9 est appliqué à **tout le pool exhaustif**, y compris les INVALID, selon son
contrat actuel. H10 valide toutes les architectures de ce pool. Le meilleur
VALID exhaustif est choisi dans l’ordre H9. L’architecture réellement choisie
par Beam est retrouvée par `architecture_id` dans ce **même pool H9** :

- qualité relative = score commun du choix Beam / score commun du meilleur VALID ;
- regret = meilleur score VALID commun − score commun du choix Beam ;
- rang parmi les VALID exhaustifs et rang dans tout le pool ;
- égalité d’ID et égalité de score avec la baseline.

La V2 distingue aussi deux pertes dans la référence commune :

```text
search_common_score_regret = score(best exhaustive VALID)
                          - score(best retained VALID in common pool)
pool_normalization_common_score_regret = score(best retained VALID in common pool)
                                      - score(actual local H9 winner in common pool)
total regret = search regret + pool normalization regret
```

`exhaustive_best_survived_search` indique si le meilleur a survécu aux choix
Beam ; `did_beam_find_exhaustive_best` indique s’il a effectivement été choisi.
`exhaustive_valid_recall` mesure la fraction du pool VALID retenue. Les
normalisateurs locaux et exhaustifs coût/puissance/tolérance sont enregistrés.
Ces données restent exclusivement dans le benchmark et ne guident pas le moteur.

Le score H9 local du pool Beam reste enregistré, mais n’est jamais directement
comparé au score d’un autre pool. Une sélection absente du domaine exhaustif
ou rejetée par son H10 déclenche une erreur de benchmark.

Le temps de ranking et le précontrôle sont enregistrés séparément. Le temps
Beam inclut contrat d’entrée, recherche, H9 et tous ses appels H10 ; le temps
exhaustif inclut génération H8, H9 et H10 pour toutes les paires. Les runs Beam
répétés donnent une médiane par cas ; la baseline s’exécute une fois par cas/K.
Les agrégats utilisent la moyenne des médianes et la moyenne des ratios de
temps par cas. Ils ne sont pas des percentiles de latence de production.

Métriques exportées : taux de cas ayant au moins un VALID, temps, branches
créées/explorées, rejets et raisons, appels H5/H6/H10, COMPLETE et VALID,
réduction de matérialisation `1 − COMPLETE_Beam / paires_exhaustives`, qualité
commune, regret et gain de temps. Un nombre de branches Beam de plusieurs
profondeurs n’est pas directement comparable à un nombre de paires COMPLETE
H8 : les deux sont nommés séparément.

Les artifacts `.json`, `.csv`, `.md` partagent le même préfixe. Le JSON conserve
les cas détaillés, paramètres, temps individuels, hashes des sources et un
exemple complet avec décision H10 recalculée. Le Markdown contient les mesures
réellement exécutées. Les résultats du run contrôlé sont liés depuis le README.

### Lecture de la campagne V1 conservée

La campagne initiale sur les trois premiers cas du dataset mesure 72
configurations K × B, avec deux répétitions par configuration. Neuf domaines
exhaustifs K=5/10/20 ont été parcourus (51 120 architectures). Les trois
domaines K=50 sont explicitement sautés pour la baseline (79 200 à 86 400
paires/cas, seuil de 20 000).

L’exhaustif trouve au moins un VALID sur les trois cas, tandis que B≤32 n’en
trouve que sur les deux premiers. Les ratios moyens de qualité sont donc
**conditionnels aux solutions effectivement trouvées** : ils ne doivent pas
masquer le troisième cas perdu. La campagne complémentaire sur ce cas, K=5,
montre qu’il reste perdu à B=64/128, revient à B=256 (qualité 97,556 %), et que
B=1024 conserve les 90 VALID (qualité 99,985 %, rang exhaustif VALID 2).

Ce dernier run n’a aucun rejet par largeur ; le regret non nul provient du
changement de normalisation H9 après retrait des 810 branches impossibles.
Le fixture sans pruning dur vérifie séparément l’équivalence exacte du pool
et du meilleur score avec l’exhaustif. Ces deux observations sont compatibles
avec les contrats conservés.

## Validation courante et terminologie

**BEAM V2.1 ACCEPTED WITH VALIDATION LIMITATIONS**. La
[revue finale A–R](../evaluation/architecture/beam_search/v2_extended/release_review.md)
contient les cinq sélections rejouées et leur H10 indépendant, les tests finaux,
la provenance et les recommandations de versionnement. Aucun défaut actuel
de correction n'est démontré pendant cette revue ; le défaut V2 décrit dans
[le diagnostic étendu](beam_search_v2_extended.md) est corrigé.

| Terme | Définition | Campagne par phase |
|---|---|---:|
| Cas | Une requête et ses contraintes/préférences | 50 |
| Domaine | Cas × K × cap de chemins H6 (indépendant de B) | 200 |
| Configuration | Recherche dans un domaine avec une largeur B | 1 000 |
| Domaine exhaustivement évalué | Toutes ses paires sont construites par H8, scorées par H9 et validées par H10 | 51 |
| Configuration comparable | Domaine exhaustif avec une référence VALID et choix Beam VALID, retrouvé dans le même pool H9 | 180 |

Les 51 domaines complets comprennent les 50 domaines K=5 et un domaine K=10 :
36 ont une référence VALID, retrouvée par Beam à toutes les cinq largeurs,
et 15 n'ont aucune référence VALID. Les 149 autres domaines dépassent le seuil
fixé de 2 000 paires et leur qualité exhaustive est indisponible.

V2.1 retourne 785 recommandations VALID. Les 215 absences sont localement
infaisables : 75 configurations selon H8/H9/H10 complets et 140 supplémentaires
selon l'audit scalaire complet des couples coût/puissance. Cet audit utilise
les fonctions publiques de faisabilité, sans pool H9 ; il ne fournit aucun
regret ou score comparatif pour les grands domaines. Aucune infaisabilité
globale n'est démontrée. Aucun domaine faisable connu n'est manqué.

L'analyse des regrets est **terminée** sur les 180 comparaisons disponibles
(36 domaines K=5). `PARTIAL` décrit uniquement la couverture de validation :
**analysis available on exhaustively tractable domains only**. Il s'agit du
cas A, une limitation normale de portée. Qualité moyenne 98,2148 %, minimum
81,4296 % ; V2.1 conserve exactement les 180 qualités et regrets V2.

## Known limitations — not correctness bugs

- Recherche approximative : aucune garantie d'optimalité globale ou de
  complétude avec B limité ; pas de backtracking.
- La qualité n'est pas nécessairement monotone avec B (9/36 groupes comparables
  monotones dans cette campagne). La validité est monotone sur 200/200 groupes
  observés, sans garantie générale ; 43 groupes n'ont jamais de solution.
- H9 normalise par pool. Le pruning de seuls INVALID peut modifier le gagnant
  local, même quand le meilleur exhaustif survit. H9 est inchangé.
- Les grandes références exhaustives sont sautées selon le seuil configuré ;
  leur qualité et leurs regrets restent indisponibles, y compris à K=50.
- L'évaluation étendue couvre 50 cas synthétiques stratifiés sur les 1 200 cas,
  sans estimation pondérée de la fréquence de production ni validation réelle.
- B borne la frontière retenue, pas la mémoire totale : caches H5/H6 et
  enveloppes dépendent de K, des profils et du cap de chemins. Les mesures
  comptent des entrées ; elles ne mesurent pas la RSS en octets.
- K et le cap de chemins bornent le domaine et peuvent masquer une solution.
  Les résultats dépendent du dataset, des catalogues et des rankers actuels.
- Les crédits optimistes sont distincts de H9 et de la validité ; ils ne
  modélisent pas toute l'évidence constructeur de fiabilité.

## Organisation documentaire

`README.md` résume la livraison ; ce document décrit V2.1. Les scripts,
benchmarks et diagnostics historiques restent dans `evaluation/architecture/beam_search/`.
[beam_search_v2.md](beam_search_v2.md) conserve le diagnostic V1 → V2 ; sa
fusion éventuelle relève d'un futur cleanup. Aucun artifact historique n'est
supprimé pendant la revue V2.1.
