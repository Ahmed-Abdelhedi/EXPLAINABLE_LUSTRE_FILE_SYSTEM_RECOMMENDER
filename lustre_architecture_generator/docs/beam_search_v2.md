# Beam Search V2 : diagnostic et recherche avec complétion optimiste

> **Historique V1 → V2 (schéma 2.0).** Les résultats et comptes de tests ci-dessous
> décrivent cette livraison antérieure. La version actuelle est
> [V2.1](beam_search.md), acceptée avec limitations de validation après
> [revue finale](../evaluation/architecture/beam_search/v2_extended/release_review.md).

La V2 prolonge le moteur existant. Elle conserve les six niveaux de recherche,
le wrapper BeamState, les transitions H7, la construction publique H7/H8, le
scoring final H9 et la validation obligatoire H10. Aucun module H5–H10 ou
ranking n’est modifié. L’API principale garde sa signature.

## Diagnostic effectué avant le changement d’algorithme

La V1 est conservée dans
`evaluation/architecture/beam_search/beam_search_v1_reference.py` exclusivement
pour l’évaluation. Elle n’est jamais importée par le moteur de production.
Le script `diagnose_beam_search.py` observe les choix construits, scorés et
retenus via un patch temporaire de `replace`, puis les compare aux préfixes
des architectures VALID exhaustives. Il n’injecte aucune vérité exhaustive
dans le classement Beam. Ses durées instrumentées ne sont pas des benchmarks.

Sur REQ_000003, K=5, un chemin H6 par variante, l’exhaustif contient 900
architectures dont 90 VALID. Le meilleur ID est
`ARCH_REQ_000003_d6d34b86f177b1cd`.

| B | Étape | Générés | Hard | Largeur | Retenus | Retenus pouvant mener à VALID |
|---:|---|---:|---:|---:|---:|---:|
| 32 | MDT_DRIVE | 5 | 0 | 0 | 5 | 5 |
| 32 | MDT_PROTECTION | 30 | 0 | 0 | 30 | 30 |
| 32 | MDT_HARDWARE | 30 | 0 | 0 | 30 | 30 |
| 32 | OST_DRIVE | 150 | 0 | 118 | 32 | 0 |
| 32 | OST_PROTECTION | 192 | 0 | 160 | 32 | 0 |
| 32 | OST_HARDWARE | 32 | 32 | 0 | 0 | 0 |
| 32 | COMPLETE / H10 | 0 | 0 | 0 | 0 | 0 |
| 128 | MDT_DRIVE | 5 | 0 | 0 | 5 | 5 |
| 128 | MDT_PROTECTION | 30 | 0 | 0 | 30 | 30 |
| 128 | MDT_HARDWARE | 30 | 0 | 0 | 30 | 30 |
| 128 | OST_DRIVE | 150 | 0 | 22 | 128 | 15 |
| 128 | OST_PROTECTION | 768 | 0 | 640 | 128 | 0 |
| 128 | OST_HARDWARE | 128 | 128 | 0 | 0 | 0 |
| 128 | COMPLETE / H10 | 0 | 0 | 0 | 0 | 0 |

La connaissance VALID de la dernière colonne appartient uniquement au diagnostic.
À B=32, les 30 préfixes viables générés à OST_DRIVE sont perdus par largeur.
À B=128, 15 survivent à OST_DRIVE, puis leurs 45 descendants viables sont
tous perdus par largeur à OST_PROTECTION. Aucun préfixe viable n’est rejeté
par hard pruning dans ces reproductions.

## Chemin du meilleur exhaustif

| Choix | Valeur |
|---|---|
| MDT drive | DRV_000085 — Kioxia FL6 KFL61HUL800G |
| MDT protection | PROT_RAID6_12, 12 drives |
| MDT hardware | DIRECT, SRV_DUALROLE_01, CTRL_NVME_G4X16_01, NET_ETH100_01, HA_NONE |
| OST drive | DRV_000137 — Solidigm D5-P5336 SBFPF2BV0P12001 |
| OST protection | PROT_RAID6_12, 51 groupes / 612 drives |
| OST hardware | DIRECT, mêmes IDs de profils, 20 serveurs, 77 controllers, 20 adaptateurs réseau |

Les ressources des deux rôles restent isolées, même quand leurs IDs de profils
sont identiques. Aucun enclosure n’est choisi pour ces chemins DIRECT.

| Étape | Rang avant largeur | Score V1 | B=32 | B=128 |
|---|---:|---:|---|---|
| MDT_DRIVE | 2 | 0.5209596548830706 | conservé | conservé |
| MDT_PROTECTION | 7 | 0.7786409634469199 | conservé | conservé |
| MDT_HARDWARE | 7 | 0.6831501492856288 | conservé | conservé |
| OST_DRIVE | 87 | 0.5605307930504818 | perdu par largeur | conservé |
| OST_PROTECTION | 416 à B=128 | 0.5699833575099614 | parent déjà perdu | perdu par largeur |

À B=256, 14 VALID arrivent jusqu’à H10, mais le meilleur chemin reste perdu à
OST_PROTECTION : son rang 416 dépasse encore la largeur.

## Cause racine et diversité

MDT_HARDWARE conserve les 30 options disponibles, couvrant 5 drives, 6
protections, 10 chemins matériels avec counts, 25 coûts et 20 puissances
distincts. Les coûts et puissances MDT sont dans le premier décile de leurs
limites. La largeur 32 n’est donc pas saturée par 32 variantes du même MDT.

Le biais dominant est un classement OST prématuré : son ranking et son coût
pré-RAID/protégé entrent dans le score, tandis que les ressources serveur,
controller et réseau OST sont absentes avant OST_HARDWARE. Les choix retenus
se révèlent ensuite tous impossibles en puissance : 32/32 à B=32 et 128/128
à B=128. Certaines branches dépassent aussi le budget. Le hard pruning final
est correct ; le classement par largeur a déjà supprimé les alternatives viables.

L’ordre MDT→OST amplifie l’absence de vision future, mais la perte observée
n’est pas située dans les trois niveaux MDT. Changer uniquement l’ordre ou
augmenter B ne traite pas la ressource encore invisible.

## Algorithme et preuve des bornes

Avant les six niveaux, H5/H6 calculent les complétions individuelles des deux
rôles. Aucun couple MDT×OST ni état COMPLETE n’est construit. Des enveloppes
sont agrégées pour les préfixes rôle entier, drive, drive/protection et chemin
exact. Les caches H5/H6 sont ensuite réutilisés.

Pour toute continuation compatible x du préfixe p :

```text
min_cost(r,p) ≤ cost(x)
min_power(r,p) ≤ power(x)
min_cost(MDT,pM) + min_cost(OST,pO) ≤ cost(architecture)
min_power(MDT,pM) + min_power(OST,pO) ≤ power(architecture)
```

La séparation physique des rôles et les contributions publiques additives
`role_option_cost_power` justifient ces sommes. Chaque minimum est indépendant :
ils peuvent provenir d’options différentes sans invalider leur admissibilité.
Une borne dépassant la limite permet le rejet dur ; une borne sous la limite
ne prouve pas qu’un couple satisfaisant toutes les contraintes existe.

## Formule avant/après

```text
V1 : h1 = .25 R + .75 (wP P + wC C + wE E + wL L)
V2 : h2 = wP P+ + wC C+ + wE E+ + wL L+
```

Les crédits V2 utilisent les complétions compatibles des deux rôles : minima
de ressources pour C+/E+, maxima de sommes de headroom pour P+, maxima des
crédits protection/HA pour L+. Le détail exact et les cas de limite nulle sont
dans [beam_search.md](beam_search.md). R+ devient un départage décroissant,
avant les clés MDT/OST existantes. Aucun poids ou identifiant n’est ajusté
selon le cas de benchmark. Les crédits optimistes ne sont pas utilisés comme
preuves physiques et ne constituent pas une borne du score H9.

## Comptage et comparaison

Les branches totales incluent la préparation et les six niveaux. La largeur B
s’applique aux niveaux principaux ; elle ne borne pas les caches H5/H6 et les
enveloppes. Le benchmark expose les deux phases séparément et inclut toutes
leurs durées dans le temps Beam. H10 reçoit au plus B COMPLETE.

Les artifacts V1 `beam_search_benchmark.*` et `width_sensitivity.*` sont
conservés. Les mesures V2 utilisent des noms explicites. Le BEFORE/AFTER reprend
les temps V1 archivés, donc ce n’est pas une expérimentation temporelle simultanée.
Les hashes des trois entrées et le cap de chemins doivent correspondre.

## H9 : deux sources de différences

Le benchmark repère le meilleur VALID exhaustif, le meilleur VALID retenu dans
ce même pool commun et le choix effectivement fait par le H9 local :

```text
search regret = score(exhaustive best) - score(retained common-pool best)
normalization regret = score(retained common-pool best) - score(local winner)
total regret = search regret + normalization regret
```

`exhaustive_best_survived_search` et `did_beam_find_exhaustive_best` sont donc
distincts. La recall du pool VALID et les normalisateurs coût/puissance/tolérance
sont aussi enregistrés. Ces métriques ne sont calculées qu’après la recherche.

## Mesures V2 réellement exécutées

La campagne principale exécute 27 configurations, deux fois chacune, sur
REQ_000001/2/3, K=5/10/20 et B=8/16/32, avec un chemin H6 par variante. La
baseline exhaustive couvre neuf domaines et 51 120 architectures.

| K | B | Couverture V1 → V2 | Branches V1 → V2 | COMPLETE / H10 V2 | Temps V2 moyen/cas | Qualité V2 moyenne | Speedup moyen |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 8 | 2/3 → 3/3 | 417 → 805 | 24 | 0.029008 s | 98.83 % | 46.8x |
| 5 | 16 | 2/3 → 3/3 | 729 → 1117 | 48 | 0.050722 s | 99.79 % | 26.0x |
| 5 | 32 | 2/3 → 3/3 | 1317 → 1693 | 96 | 0.096340 s | 98.47 % | 13.7x |
| 10 | 8 | 2/3 → 3/3 | 606 → 1377 | 24 | 0.041707 s | 99.78 % | 151.4x |
| 10 | 16 | 2/3 → 3/3 | 1074 → 1845 | 48 | 0.064520 s | 98.97 % | 93.1x |
| 10 | 32 | 2/3 → 3/3 | 1938 → 2709 | 96 | 0.125614 s | 99.67 % | 48.1x |
| 20 | 8 | 2/3 → 3/3 | 876 → 2408 | 24 | 0.067587 s | 98.75 % | 536.0x |
| 20 | 16 | 2/3 → 3/3 | 1692 → 3224 | 48 | 0.094405 s | 99.46 % | 386.6x |
| 20 | 32 | 2/3 → 3/3 | 3108 → 4640 | 96 | 0.168664 s | 99.45 % | 218.4x |

Les branches sont les totaux des trois cas, et incluent la préparation V2.
Les temps sont les moyennes des médianes par cas ; le speedup est la moyenne
des ratios individuels. Les qualités V1 étaient conditionnelles à deux cas,
contre trois pour V2. Le tableau BEFORE/AFTER individuel est donc la comparaison
de qualité appropriée : 17/18 configurations auparavant faisables améliorent
leur score commun ; REQ_000001, K=5, B=8 passe de 99.744 % à 99.280 %.

Les neuf configurations V2 trouvent les trois cas. REQ_000003 trouve un VALID
dès B=1 à K=5. À B=4, le meilleur exhaustif est choisi ; à B=8/16/32/64, il
survit dans la recherche, mais la normalisation H9 locale peut changer le choix.

À B=1024, les 90 VALID survivent, aucun rejet par largeur, search regret=0,
normalization regret=0.00012249798183283644. Le rang final commun est 2.
La plage de coût exhaustif est [4695280, 9410540] USD, contre
[5156000, 5664860] USD dans le pool VALID local. La plage de puissance est
[33032, 500548] W contre [33032, 35788] W. Le normalisateur de tolérance reste 2.
La perte restante est donc attribuée au pool H9, sans modifier H9.

Les rapports sont dans `evaluation/architecture/beam_search/` :

- `diagnostic_v1.json/.md` et `diagnostic_v2.json/.md` : niveaux, scores, raisons,
  diversité, préfixes viables et chemin du meilleur exhaustif.
- `beam_search_v2_benchmark.json/.csv/.md` : campagne complète, hashes et métriques.
- `beam_search_v2_benchmark_before_after.csv` : comparaison individuelle V1/V2.
- `width_sensitivity_v2.json/.csv/.md` : B=1/2/4/8/16/32/64/1024 sur REQ_000003.
- `initial_git_status_v2.txt` : état initial dirty conservé.

Vérifications : 57 tests Beam/benchmark réussis ; suite architecture 284
réussis en 61.73 s ; suite globale 490 réussis en 91.54 s. Aucun échec ou skip
dans ces vérifications finales. Le répertoire temporaire pytest a été fixé
explicitement pour éviter un dossier temporaire Windows préexistant inaccessible.
Les deux fixtures de perte V1 couvrent le coût matériel OST encore invisible
et l’allocation MDT-first du budget laissant trop peu au rôle futur. La V2
retrouve une architecture H10 VALID à B=1 dans les deux cas.

## Limites restantes

- Une largeur limitée peut encore perdre le meilleur VALID ; l’optimalité et la
  monotonie avec B ne sont pas garanties.
- Les extrema optimistes peuvent provenir de choix différents ; aucun oracle
  conjoint ni backtracking n’est ajouté.
- La préparation dépend de K, du catalogue et du cap de chemins, même à B=1.
- Les proxies Beam ne reproduisent pas toute l’évidence de fiabilité H9.
- H9 reste frozen et dépendant de son pool. Une amélioration de couverture ne
  garantit pas que chaque configuration améliore son choix final.
- Les trois cas contrôlés ne prouvent pas la couverture du dataset complet.
- La V2 n’est pas benchmarkée à K=50 dans cette campagne.
