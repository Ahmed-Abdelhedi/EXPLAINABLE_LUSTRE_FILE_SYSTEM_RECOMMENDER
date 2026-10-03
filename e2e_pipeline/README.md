# Pipeline production Requirement → Lustre

`run_e2e` et `run_e2e_from_file` relient le Requirement canonique à S10, au
filtrage/ranking MDT/OST et à la recherche d’architecture. La version du
pipeline est **1.1** ; le schéma JSON reste **1.0**, avec des champs additifs.
Beam V2.1, H5–H10, S10 et les rankers sont réutilisés sans modification.

## Stratégies et configuration

**`beam` est le choix par défaut.** `FrozenRuntimeBackend.run_beam_search`
délègue une seule fois à `beam_search_architectures`. Le moteur applique déjà
H9 et H10 : l’E2E ne les rappelle pas séparément et ne génère pas de pool H8.
**`exhaustive`** conserve le chemin historique H8 → H9 → H10 et la sélection
du premier candidat H10 VALID dans l’ordre H9.

`SearchOptions(strategy="beam", beam_width=8)` est une configuration immuable.
La stratégie doit être `beam` ou `exhaustive` ; la largeur doit être un entier
strictement positif, sans accepter les booléens, chaînes ou nombres flottants.

`PipelineLimits` conserve ses quatre champs et leurs valeurs par défaut :

| Champ | Défaut | Utilisation |
| --- | ---: | --- |
| `top_k` | 10 | Handoff Top-K MDT/OST, partagé |
| `max_paths_per_variant` | 2 | Cap de chemins H6, partagé |
| `max_role_options_per_role` | 4 | Cap d’options du mode exhaustif |
| `max_architectures` | 16 | Cap d’architectures du mode exhaustif |

Le mode exhaustif conserve ces caps historiques : il ne couvre tout le
domaine que si les limites choisies permettent de le parcourir entièrement.
Ces deux caps H8 ne sont pas transmis au moteur Beam.

## CLI

```powershell
python main.py --device cpu
python main.py --device cpu --search-strategy beam --beam-width 8
python main.py --device cpu --search-strategy exhaustive
```

`--top-k`, `--max-paths-per-variant`, `--max-role-options` et
`--max-architectures` restent disponibles. Le bandeau affiche la stratégie et,
en mode Beam, sa largeur. `--requirement-only` et la protection contre la
réutilisation d’un ancien Requirement après `/quit` sont conservés.

Pour exécuter le downstream sans extraction NLP interactive :

```python
from pathlib import Path
from e2e_pipeline import PipelineLimits, SearchOptions, run_e2e_from_file

result = run_e2e_from_file(
    Path("output/final_requirement.json"),
    output_path=Path("output/final_e2e_result.json"),
    limits=PipelineLimits(top_k=5, max_paths_per_variant=1),
    search_options=SearchOptions(strategy="beam", beam_width=8),
)
```

## Contrat de sortie

`best_architecture` garde les champs `architecture_id`, `score`, `validation`
et `architecture`. En mode Beam, `score` conserve l’objet H9, `validation`
conserve la décision H10 et `architecture` conserve l’état physique et
`search_provenance`. L’adaptateur exige `valid is True`, `decision == "VALID"`
et des identités cohérentes entre Beam, H9 et H10.

Le résultat normal de recherche indique sa stratégie dans `architecture_search`.
En Beam, il expose `beam_search_applied`, `beam_width`, `beam_status`,
`requested_top_k`, `actual_top_k`, `max_paths_per_variant`, `summary`,
`heuristic_policy`, `search_trace`, `lookahead_trace`, `selection_rule` et
`global_infeasibility_claimed=False`, sans recopier tout le pool Beam.
En exhaustif, il conserve `h8_summary`, `h9_summary`, `h10_summary` et
`selection_rule`, avec `beam_search_applied=False`.

`trace.beam_search_applied` indique l’appel Beam effectué ;
`trace.h10_is_final_validity_authority` reste `True`. Les objets H9/H10 gardent
leurs propres marqueurs frozen `beam_search_applied=False` : ils ne réalisent
pas eux-mêmes la recherche Beam.

## Absence de solution et erreurs

- `SUCCESS` : une architecture est confirmée H10 VALID.
- `NO_VALID_ARCHITECTURE_WITHIN_SEARCH_DOMAIN`, étape `BEAM_SEARCH` : Beam
  termine sans recommandation. Le message précise Top-K, largeur et cap de
  chemins ; aucune infaisabilité globale n’est déclarée.
- `PIPELINE_ERROR`, étape `BEAM_SEARCH` : une exception du moteur ou de
  l’adaptateur conserve son type et son message.

Les statuts historiques d’adaptation/ranking et le statut exhaustif
`NO_VALID_ARCHITECTURE` sont conservés. Une absence normale de solution ne
fait pas échouer le process ; `PIPELINE_ERROR` retourne le code 4.

Beam reste borné, sans garantie d’optimalité ou de complétude globale. Sa
qualité peut être non monotone avec B ; H9 dépend du pool et les caches et
enveloppes ne sont pas strictement bornés par B. Les
[limites du moteur](../lustre_architecture_generator/docs/beam_search.md#known-limitations--not-correctness-bugs)
restent applicables. H8 sert de référence sur les domaines tractables.
