# H8 — FullArchitectureGenerator

H8 est la première étape qui génère **plusieurs architectures physiques
complètes** à partir des Top-K MDT/OST.

Il s'appuie exclusivement sur les sémantiques déjà validées :

1. H1/H2 : handoff ranking -> architecture ;
2. H5 : variantes de protection et nombres physiques de drives ;
3. H6 : chemins hardware compatibles ;
4. H7 : transitions de `ArchitectureState`.

## Ce que H8 fait

Pour chaque rôle MDT et OST :

`drive candidat -> profil de protection -> chemin hardware compatible`

Une option de rôle contient donc déjà : drive, protection, serveur,
contrôleur, enclosure éventuelle, réseau, HA et minimums de composants.

H8 forme ensuite le produit cartésien :

`options MDT × options OST -> ArchitectureState COMPLETE`

Chaque résultat reçoit un `architecture_id` stable construit à partir de sa
signature physique.

## Ce que H8 ne fait pas

H8 n'applique :

- aucun score d'architecture ;
- aucun Beam Search ;
- aucune déclaration d'optimalité ;
- aucune validation finale H10.

Tous les states restent :

`validation.status = PENDING_FULL_VALIDATOR`

## Génération exhaustive vs validation contrôlée

`iter_full_architectures()` est paresseux et permet de parcourir tout le
produit cartésien du domaine d'expansion configuré.

Pour éviter une explosion combinatoire pendant les tests runtime, le
validateur utilise des limites explicites :

- `max_paths_per_variant` ;
- `max_role_options` ;
- `max_architectures`.

Ces limites sont des bornes d'évaluation, pas une heuristique d'optimalité.
Le futur Beam Search aura sa propre politique d'exploration après le freeze de
la couche architecture.

## Isolation des rôles

À H8, MDT et OST sont comptabilisés comme des **instances physiques isolées**.
Même si un profil serveur `BOTH` existe dans le catalogue, H8 ne déduplique pas
les instances MDS et OSS entre rôles. Cela évite d'introduire implicitement une
politique de co-location avant le validateur/scoring global.
