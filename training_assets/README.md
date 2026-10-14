# Données et notebooks d'entraînement récupérés — 14 octobre 2026

Ce dossier rassemble les données et notebooks retrouvés sur cet ordinateur. Les copies d'origine sont distinguées des données régénérées. Aucun modèle n'a été réentraîné.

## État des quatre composants

| Composant | Données disponibles | Entraînement | Provenance et vérification |
| --- | --- | --- | --- |
| Semantic Linker v3 | Train **98 100**, validation **15 360**, test **15 360** | Notebook final v3.2 et notebook exécuté retrouvés | Copies des originaux dans Téléchargements ; SHA-256 identiques aux métadonnées d'origine |
| Preference Signal v2.1 | Train **28 400**, validation **5 800** | Notebook exécuté v2.1 et générateurs V2/V2.1 d'origine retrouvés | JSONL finaux régénérés ; QA V2 et V2.1 : **39/39** chacun ; identité avec les anciens fichiers V2.1 non certifiable sans leurs anciennes empreintes |
| Preference Layer 2 R3 | Train **48 000**, validation **6 000**, test **6 000**, final holdout **4 000** | Notebook final et notebook exécuté retrouvés | Archive déjà présente à la racine du projet ; quatre SHA-256 identiques aux empreintes d'origine |
| Categorical/Boolean | Train **40 000**, validation **6 000**, test **6 000**, final holdout **4 000**, soit **56 000** | Notebook exécuté et notebook DDP final retrouvés ; scripts d'entraînement du dépôt copiés | Archive d'origine dans Téléchargements ; quatre SHA-256 identiques aux métadonnées d'origine ; **générateur du dataset non retrouvé** |

## 1. Semantic Linker

Les fichiers demandés sont dans `01_semantic_linker_v3/dataset/` :

- `semantic_linker_train_v3_combined.jsonl`
- `semantic_linker_val_v3_global_stress.jsonl`
- `semantic_linker_test_v3_global_stress.jsonl`

Le dossier comprend aussi le générateur d'origine `generate_semantic_linker_v3.py`, les métadonnées, le rapport QA et les fichiers complémentaires d'origine. L'ancien train v1_1 est conservé séparément dans `v1_1_source/`.

Dans `01_semantic_linker_v3/notebooks/` :

- `semantic_linker_v3_2_FINAL_dualT4_resilient.ipynb` : version finale v3.2, sans sorties d'exécution.
- `semantinlinkerv3.1.ipynb` : notebook exécuté ; malgré son nom, son code exporte `semantic_linker_version = 3.2` et utilise les fichiers v3 demandés.
- `semantic-linker-transformer-v3-kaggle.ipynb` : version v3 antérieure conservée pour référence.

La configuration du modèle actuellement présent dans le dépôt est copiée dans `artifact_training_config.json`. Le notebook v1_1 du dépôt n'a pas été remplacé.

Source locale : `C:/Users/LENOVO/Downloads/semantic_linker_v3_global_stress_dataset/` et notebooks du dossier Téléchargements.

## 2. Preference Signal Detector v2.1

Les deux fichiers demandés sont dans :

`02_preference_signal_v2_1/preference_extractor/training/data/data_layer1_v2_1/`

Les JSONL finaux V2.1 d'origine n'ont pas été retrouvés. Leur reconstruction utilise :

1. Les données V1 d'origine de `data_layer1.zip`.
2. Le générateur V2 d'origine de `step3_1B_preference_retraining_V2_FINAL.zip`.
3. Le générateur V2.1 d'origine de `step3_1E_preference_v2_1_targeted_data_FINAL.zip`.
4. Le benchmark de régression de 500 cas récupéré dans le commit Git `19a4865ad4f78207f4e3f493bc0ecbc0609e8b9a`. Ce benchmark avait été supprimé dans un commit ultérieur ; il est distinct du test V1 de 4 000 cas portant le même nom.

Les générateurs sont copiés sans modification et utilisent leurs seeds d'origine. Les JSONL régénérés utilisent des fins de ligne LF, comme les fichiers archivés d'origine. Les fichiers hard V2 reconstruits sont comparés aux copies présentes dans l'archive d'origine. Les contrôles QA vérifient notamment les tailles, l'équilibre des labels et les séparations entre jeux de données.

Les anciennes empreintes des fichiers finaux V2.1 sont absentes des éléments retrouvés. **Cette reconstruction ne permet donc pas de certifier une identité octet par octet avec les fichiers historiques V2.1.**

Le notebook exécuté `layer1-v2-1.ipynb` confirme les tailles historiques de 28 400 et 5 800 exemples. Il est conservé dans `02_preference_signal_v2_1/notebooks/`, avec les versions de préparation retrouvées.

Pour reproduire la reconstruction avec Python, depuis ce dossier racine :

```powershell
python .\02_preference_signal_v2_1\regenerer_preference_v2_1.py
```

Cette commande réécrit uniquement les sorties V2/V2.1 de ce dossier de récupération. Elle exécute les scripts QA d'origine et conserve les journaux dans `recovery_logs/`.

Le notebook exécuté indique aussi un dataset Kaggle `ahmedabdelhedii/data-layer1-v2-1`. Une tentative de téléchargement public a renvoyé HTTP 403 ; aucune connexion au compte Kaggle n'a été effectuée. Les originaux peuvent éventuellement être récupérés depuis ce compte pour comparer leurs empreintes à la reconstruction.

## 3. Preference Layer 2 R3

Les quatre JSONL demandés sont dans `03_preference_layer2_r3/data/`.

Les métadonnées, le rapport QA et `KAGGLE_DATASET_FINGERPRINTS.json` sont conservés. Les notebooks sont dans `03_preference_layer2_r3/notebooks/` :

- `layer2_xlmr_KAGGLE_FRESH_SESSION_FINAL.ipynb` : notebook final ciblant explicitement les noms R3.
- `layer-2_runned.ipynb` : notebook exécuté utilisant ces fichiers R3.

Source : `preference_layer2_dataset_KAGGLE_FRESH_R3.zip`, déjà présent à la racine du projet, avec une copie identique en taille dans Téléchargements.

## 4. Categorical/Boolean

Les quatre JSONL sont dans `04_categorical_boolean_56k/data/`. Les métadonnées et le rapport QA de l'archive sont conservés.

Dans `04_categorical_boolean_56k/notebooks/` :

- `categorical-semantic-extractor (2).ipynb` : notebook exécuté, avec les empreintes des données d'entraînement et de validation.
- `Categorical_Semantic_Extractor_FINAL_DUAL_GPU_V3_DDP_FIXED.ipynb` : notebook final DDP.

Les scripts actuellement présents dans `categorical_boolean_extractor/training/` sont copiés dans `training_from_repository/`. `train_multitask.py` entraîne le modèle à partir de données existantes ; ce fichier n'est pas le générateur des 56 000 exemples. Le notebook DDP est conservé séparément des scripts actuels du dépôt.

**Le script original qui a généré le dataset de 56 000 cas reste non retrouvé** dans les dossiers et archives inspectés. Les données d'origine sont néanmoins disponibles et vérifiées ; leur récupération ne nécessite aucune régénération.

Source : `C:/Users/LENOVO/Downloads/categorical_boolean_FINAL_DATASET_56K.zip`.

## Preuves et partage

- `manifest_sources.json` : provenance et empreinte de chaque copie d'origine.
- `verification_original_datasets.json` : tailles et empreintes comparées aux métadonnées d'origine.
- `verification_preference_signal_regenere.json` : reconstruction V2.1, comparaison avec les données hard V2 archivées et résultats QA.
- `inventaire_final.json` : empreintes des fichiers du dossier livré.

Ce dossier `training_assets/` rassemble les fichiers destinés à être versionnés dans le dépôt. Les données JSONL sont suivies avec Git LFS. Les attributs Git préservent les octets des fichiers pour permettre la vérification des empreintes. Les copies locales sous `output/` et les originaux de Téléchargements sont conservés.

Après un clonage, récupérer les données Git LFS depuis la racine du dépôt :

```powershell
git lfs pull
```

Les notebooks se trouvent dans le sous-dossier `notebooks/` de chaque composant. Les limites de provenance indiquées ci-dessus restent applicables : les données Preference Signal v2.1 sont régénérées et le générateur du dataset Categorical/Boolean n'a pas été retrouvé.
