VERSION2 E2E FINAL - OVERLAY A COPIER DIRECTEMENT
=================================================

OBJECTIF
-------
Ce dossier relie le pipeline Requirement déjà validé au downstream déjà figé :

Requirement final JSON
  -> adapter de contrat
  -> S10 sizing/workload
  -> technical MDT/OST requirements
  -> filtres déterministes + LightGBM officiel + Top-K
  -> handoff Ranking -> Full Architecture
  -> H5/H6/H7 via les modules existants
  -> H8 génération d'architectures complètes
  -> H9 scoring soft
  -> H10 validation déterministe
  -> meilleure architecture H10-valide selon H9

AUCUNE formule métier frozen n'est réécrite dans ce dossier.

INSTALLATION
------------
1. Fermer le programme s'il tourne.
2. Extraire le ZIP.
3. Ouvrir le dossier `version2_E2E_FINAL` extrait.
4. Copier TOUT son contenu dans :

   C:\Users\LENOVO\Desktop\internship\version2

5. Accepter le remplacement de `main.py`.

Le dossier ajoute seulement :

   version2/
   |-- main.py                     (remplacement du point d'entrée racine)
   `-- e2e_pipeline/
       |-- __init__.py
       |-- requirement_to_sizing_adapter.py
       |-- end_to_end_pipeline.py
       `-- tests/

Les fichiers frozen déjà présents dans :
- requirement_state/
- lustre_architecture_generator/src/
- rankers LightGBM
- H5/H6/H7/H8/H9/H10

ne sont PAS remplacés par cet overlay.

VERIFICATION APRES COPIE
------------------------
Depuis PowerShell :

cd C:\Users\LENOVO\Desktop\internship\version2

python -m pytest e2e_pipeline/tests -q
python -m pytest -q

Puis lancement online complet :

python main.py --device cpu

Le Requirement final reste écrit dans :

output\final_requirement.json

Le résultat E2E est écrit dans :

output\final_e2e_result.json

COMPORTEMENT IMPORTANT
----------------------
- H9 ne peut jamais rendre valide une architecture invalide.
- La recommandation finale est le meilleur score H9 PARMI les architectures
  que H10 a déclarées valides.
- Si aucun drive MDT n'est faisable : NO_FEASIBLE_MDT.
- Si aucun drive OST n'est faisable : NO_FEASIBLE_OST.
- Si H10 ne valide aucune architecture du pool évalué :
  NO_VALID_ARCHITECTURE.
- Les limites H8 par défaut sont les limites d'évaluation déjà utilisées dans
  le projet avant Beam Search : Top-K=10, paths=2, role-options=4,
  architectures=16.
- Beam Search n'est PAS encore appliqué.

ANCIEN COMPORTEMENT MAIN UNIQUEMENT REQUIREMENT
-----------------------------------------------
Pour arrêter après final_requirement.json :

python main.py --device cpu --requirement-only

LIMITES CONFIGURABLES
---------------------
Exemple :

python main.py --device cpu --top-k 20 --max-paths-per-variant 2 --max-role-options 8 --max-architectures 64

Ne pas interpréter un NO_VALID_ARCHITECTURE sous un pool tronqué comme une
preuve mathématique qu'aucune architecture physique n'existe dans tout le
domaine. H10 décide la validité de chaque architecture examinée ; Beam Search
et l'évaluation K x B viendront ensuite optimiser/couvrir la recherche.
