Explainable Lustre Recommender

Projet de stage — système hybride et explicable de recommandation d’architectures Lustre pour environnements HPC

Ce dépôt contient un système qui transforme une demande utilisateur exprimée en langage naturel en exigences structurées pour une architecture Lustre, vérifie leur validité et leur plausibilité, prépare les besoins techniques MDT/OST, filtre les disques compatibles, puis utilise des modèles de Learning-to-Rank pour préparer la future recherche d’architecture complète.

Le principe architectural central est volontairement conservateur :

Les modèles IA améliorent la compréhension linguistique et le classement des candidats, mais les décisions critiques de validité et de faisabilité restent contrôlées par des règles déterministes.

1. État actuel du projet

Composant

Statut

Extraction des exigences utilisateur

✅ Terminé et validé

Normalisation des unités et vocabulaires

✅ Terminé et validé

Gestion des conflits / clarifications

✅ Terminé et validé

LLM fallback contrôlé

✅ Terminé et validé

AI Plausibility Agent

✅ Terminé et validé

Validation end-to-end du Requirement Pipeline

✅ 30/30 scénarios

Workload Analyzer

✅ Terminé

Feature Calculator

✅ Terminé

Lustre Architecture Generator

✅ Terminé

MDT Candidate Generator

✅ Terminé

OST Candidate Generator

✅ Terminé

Datasets de ranking MDT / OST

✅ Terminé

Entraînement MDT Ranker

✅ Terminé

Entraînement OST Ranker

✅ Terminé

Comparaison CatBoost / LightGBM

✅ Terminée

LightGBM retenu pour MDT et OST

✅ Décision prise

Intégration des modèles LightGBM dans le runtime VS Code

🚧 Prochaine étape

Top-K MDT / OST en inférence réelle

🚧 Prochaine étape

ArchitectureState et scoring global

⏳ À faire

Beam Search

⏳ À faire

Validation finale d’architecture complète

⏳ À faire

Recommandation Lustre finale de bout en bout

⏳ À faire

Important : le pipeline de compréhension et de validation des exigences est actuellement exécutable et validé.La partie de recommandation physique complète n’est pas encore terminée : l’intégration des deux modèles LightGBM et le Beam Search constituent les prochaines étapes.

2. Objectif du système

L’utilisateur ne doit pas connaître les détails internes de Lustre.

Il décrit simplement son besoin, par exemple :

Je souhaite environ 500 TiB utilisables pour 200 clients.
Les fichiers font environ 2 GB en moyenne et peuvent atteindre 100 GB.
Le workload est plutôt 70 % lecture / 30 % écriture.
Je souhaite environ 80 GB/s en lecture et 40 GB/s en écriture.
La haute disponibilité est obligatoire.
Budget maximum : 100000 USD.
Puissance maximum : 15 kW.
Croissance annuelle estimée : 30 %.

Le système transforme progressivement cette demande en un contrat structuré, puis en besoins techniques MDT/OST.

À terme, le pipeline complet sera :

User text
   ↓
Requirement Extractor
   ↓
StateGuard
   ↓
AI Plausibility Agent
   ↓
Requirement Contract
   ↓
Workload Analyzer
   ↓
Feature Calculator
   ↓
Architecture Generator
   ↓
Deterministic MDT / OST filters
   ↓
LightGBM MDT Ranker + LightGBM OST Ranker
   ↓
Top-K MDT / OST
   ↓
Beam Search
   ↓
Deterministic final validator
   ↓
Explainable Lustre recommendation

3. Architecture générale

flowchart TD
    A[Texte utilisateur] --> B[Text Preprocessor]
    B --> C[Rule / Entity Extractor]
    C --> D[Unit Normalizer + Vocabulary Mapper]

    D --> E{Informations correctement extraites ?}
    E -- Oui --> F[StateGuard]
    E -- Cas difficile --> G[LLM Fallback]
    G --> F

    F --> H{Besoin de clarification ?}
    H -- Oui --> I[Question de clarification]
    I --> A

    H -- Non --> J[AI Plausibility Agent]

    J --> K{Plausibilité}
    K -- COHERENT --> L[Requirement Contract]
    K -- AMBIGUOUS --> M[Warning + éventuel LLM enrichment]
    M --> L
    K -- INCOHERENT --> N[Blocage / correction utilisateur]

    L --> O[Workload Analyzer]
    O --> P[Feature Calculator]
    P --> Q[Architecture Generator]
    Q --> R[MDT Candidate Generator]
    Q --> S[OST Candidate Generator]

    R --> T[LightGBM MDT Ranker]
    S --> U[LightGBM OST Ranker]

    T --> V[Top-K MDT]
    U --> W[Top-K OST]

    V --> X[Beam Search - à intégrer]
    W --> X
    X --> Y[Validation déterministe finale]
    Y --> Z[Recommandation explicable]

4. Séparation des responsabilités

4.1 Requirement Extractor

Le module requirement_extractor traite le texte utilisateur.

Il couvre notamment :

extraction des valeurs ;

normalisation des unités ;

vocabulaire contrôlé ;

gestion des conflits ;

gestion multi-tour ;

clarification ;

LLM fallback ;

validation locale ;

plausibilité inter-champs.

Le contrat utilisateur contient notamment les champs suivants :

requested_usable_capacity_tib
client_count
average_file_size_gb
max_file_size_gb
total_file_count
read_write_ratio
access_type
target_read_gbps
target_write_gbps
ha_required
max_budget_usd
max_power_w
annual_growth_percent

Note importante sur les débits : dans la version actuelle du projet, les noms de champs utilisent encore le suffixe gbps, mais le benchmark et le contrat métier courant les interprètent comme GB/s. Cette convention doit être conservée tant qu’un renommage contrôlé n’a pas été effectué.

4.2 StateGuard

StateGuard contrôle les exigences extraites.

Il vérifie notamment :

les valeurs positives attendues ;

les compteurs entiers ;

les ratios lecture/écriture ;

les conflits entre valeurs ;

les informations manquantes ;

les clarifications nécessaires ;

la cohérence de l’état conversationnel.

Exemple :

Tour 1 : budget maximum = 100000 USD
Tour 2 : mon budget est 150000 USD

Sans expression explicite de correction, le système détecte deux valeurs différentes et demande laquelle conserver.

En revanche :

Correction : remplace le budget par 150000 USD

est traité comme une mise à jour explicite.

4.3 LLM Fallback

Le système est rule-first.

Le LLM fallback n’est appelé que lorsqu’une formulation est difficile à traiter avec les règles déterministes.

Modèle utilisé pour les évaluations finales :

qwen2.5-coder:7b

Le LLM ne devient jamais la source autoritaire de vérité.

Les valeurs candidates qu’il propose restent soumises aux contrôles du pipeline.

4.4 AI Plausibility Agent

L’AI Plausibility Agent vérifie les relations entre plusieurs champs déjà acceptés.

Exemples de contradictions fortes :

average_file_size > max_file_size
estimated_dataset_volume >> requested_capacity
read_percent + write_percent != 100

Les décisions possibles sont :

COHERENT
AMBIGUOUS
INCOHERENT

COHERENT

Aucune contradiction détectée.

AMBIGUOUS

Le besoin est techniquement valide, mais sa faisabilité dépend encore de l’architecture.

Exemples : budget très contraignant, power très contraignant, throughput très élevé, nombre de clients très élevé.

Ces situations génèrent des warnings, pas un blocage automatique.

INCOHERENT

Une contradiction calculable est détectée.

La recommandation est alors bloquée jusqu’à correction.

4.5 Garde de l’enrichissement LLM

Pour les warnings AMBIGUOUS, le système peut demander au LLM de rendre l’explication plus naturelle.

La décision métier existe avant l’appel LLM.

Le LLM est uniquement autorisé à reformuler.

Le garde intégré à ai_plausibility_agent.py vérifie notamment :

que la réponse est exploitable ;

que le JSON peut être parsé ;

que le problème détecté n’est pas changé ;

que les nombres ne sont pas modifiés ;

que les unités protégées ne sont pas modifiées ;

que le LLM n’introduit pas de faits non supportés.

Si la reformulation est invalide :

LLM output rejected
        ↓
deterministic warning preserved

Le benchmark end-to-end a réellement rencontré un JSON LLM brut invalide ; le pipeline a tout de même conservé une sortie correcte.

5. Architecture Generator

Le générateur ne choisit pas encore directement un disque, un RAID ou un serveur.

Il transforme le besoin utilisateur en exigences techniques indépendantes du hardware.

La chaîne hors ligne utilisée pour construire et vérifier les données est :

requirements
    ↓
workload analysis
    ↓
feature calculation
    ↓
architecture requirements
    ↓
MDT candidate generation
    ↓
OST candidate generation
    ↓
training datasets

Les composants principaux sont :

workload_analyzer.py
feature_calculator.py
architecture_generator.py
mdt_candidate_generator.py
ost_candidate_generator.py
training_dataset_builder.py

6. MDT et OST : deux problèmes différents

Le système sépare la sélection des drives en deux tâches.

MDT

Le ranking MDT favorise principalement : IOPS, faible latence, endurance et fiabilité.

OST

Le ranking OST favorise principalement : capacité, bande passante, coût par capacité, puissance et fiabilité.

Cette séparation est volontaire : un bon disque MDT n’est pas nécessairement un bon disque OST.

7. Modèles de ranking

Deux familles ont été comparées :

CatBoostRanker — YetiRankPairwise
LightGBM — LambdaRank

La comparaison multi-seeds a conduit à retenir LightGBM LambdaRank comme ranker principal pour MDT et OST.

Les modèles ne construisent pas l’architecture complète.

Leur rôle est uniquement :

Drives techniquement faisables
        ↓
LightGBM Ranker
        ↓
liste ordonnée
        ↓
Top-K

La faisabilité reste déterministe.

Résultats principaux du benchmark de ranking

MDT

LightGBM NDCG@10   ≈ 0.9784
LightGBM Recall@10 ≈ 0.9461
Top-1 agreement    ≈ 96.44 %

OST

LightGBM NDCG@10   ≈ 0.9455
LightGBM Recall@10 ≈ 0.9018
Top-1 agreement    ≈ 83.00 %

Ces scores justifient l’utilisation du Top-K LightGBM comme espace d’entrée du futur Beam Search.

8. Beam Search — prochaine grande phase

Le Beam Search n’est pas encore intégré dans le runtime final.

Son rôle sera de construire des architectures complètes en combinant :

Top-K MDT drives
+
Top-K OST drives
+
RAID
+
nombre de drives
+
nombre de MDT / OST
+
groupes de protection
+
serveurs
+
striping
+
coût
+
puissance
+
performance

Chaque branche devra être contrôlée par les contraintes dures.

Exemple :

capacity < required capacity
→ reject

cost > max_budget
→ reject

power > max_power
→ reject

required throughput not satisfied
→ reject

hardware incompatible
→ reject

Le Beam Search gardera uniquement les meilleurs états à chaque étape afin d’éviter une explosion combinatoire.

9. Structure principale du dépôt

La structure exacte peut évoluer pendant l’intégration du ranker et du Beam Search, mais l’organisation actuelle est centrée sur :

version2/
│
├── requirement_extractor/
│   ├── __init__.py
│   ├── ai_plausibility_agent.py
│   ├── calculation_engine.py
│   ├── clarification_agent.py
│   ├── closed_vocabulary_mapper.py
│   ├── field_defs.py
│   ├── hybrid_extractor.py
│   ├── llm_fallback_extractor.py
│   ├── main.py
│   ├── models.py
│   ├── requirement_chatbot.py
│   ├── rule_entity_extractor.py
│   ├── state_guard.py
│   ├── text_preprocessor.py
│   ├── unit_normalizer.py
│   └── validation/
│
├── lustre_architecture_generator/
│   ├── config/
│   ├── input/
│   ├── output/
│   └── src/
│
├── lustre_recommender/
│
├── drive_selector_dataset_v3/
│
├── docs/
├── .env
├── requirements.txt
└── README.md

10. Installation

10.1 Ouvrir le projet

Depuis VS Code :

cd <CHEMIN_VERS_LE_PROJET>\version2

10.2 Créer un environnement virtuel

Sous PowerShell :

python -m venv .venv
.\.venv\Scripts\Activate.ps1

Sous Linux / macOS :

python -m venv .venv
source .venv/bin/activate

10.3 Installer les dépendances Python

python -m pip install --upgrade pip
pip install -r requirements.txt

11. Installation d’Ollama

Ollama est nécessaire uniquement pour les chemins utilisant les LLM.

Le chemin déterministe peut être testé sans LLM.

Les modèles utilisés dans la campagne finale sont :

qwen2.5-coder:7b
qwen2.5:3b

Après installation d’Ollama :

ollama pull qwen2.5-coder:7b
ollama pull qwen2.5:3b

Vérifier les modèles disponibles :

ollama list

Vérifier les modèles actuellement chargés :

ollama ps

Selon l’installation, Ollama peut déjà fonctionner comme service local. Sinon :

ollama serve

12. Configuration .env

Le projet utilise un fichier .env à la racine.

Pour les tests avec LLM fallback :

ENABLE_LLM_FALLBACK=true

Les paramètres de sécurité/latence utilisés pour l’AI Plausibility peuvent inclure :

PLAUSIBILITY_AGENT_TIMEOUT_SECONDS=60
PLAUSIBILITY_AGENT_NUM_PREDICT=192
PLAUSIBILITY_AGENT_KEEP_ALIVE=30s

Ne jamais placer de secret ou token externe dans le dépôt Git.

13. Exécuter le chatbot

Depuis la racine version2/ :

python -m requirement_extractor.main

Le chatbot peut ensuite recevoir une demande en langage naturel.

Exemple :

Je veux 500 TiB utilisables pour 200 clients,
avec des fichiers moyens de 2 GB et maximum 100 GB.
Le ratio est 70/30, accès mixed.
Je vise 80 GB/s en lecture et 40 GB/s en écriture.
HA obligatoire, budget 100000 USD,
puissance maximum 15 kW et croissance 30 %.

Le système peut extraire les informations, demander une clarification, détecter un conflit, signaler une incohérence et produire un RequirementContract prêt pour la suite du pipeline.

14. Tester la baseline déterministe

Aucun LLM n’est nécessaire.

python -m requirement_extractor.validation.run_validation --mode deterministic
python -m requirement_extractor.validation.metrics --mode deterministic
python -m requirement_extractor.validation.error_analyzer --mode deterministic

Résultats de référence :

150 scénarios
175 tours
0 erreur technique
Scenario success          ≈ 87.33 %
Full-turn exact           ≈ 89.14 %
Field-slot accuracy       ≈ 97.67 %
Field precision           = 100 %
Field recall/value        ≈ 97.02 %
Field F1                  ≈ 98.49 %
Hallucination             = 0 %
Normalization             = 100 %
Multi-turn consistency    = 100 %
Average latency           ≈ 1.15 ms

15. Tester le LLM fallback

Ollama et qwen2.5-coder:7b doivent être disponibles.

python -m requirement_extractor.validation.run_validation --mode llm_fallback
python -m requirement_extractor.validation.metrics --mode llm_fallback
python -m requirement_extractor.validation.error_analyzer --mode llm_fallback

Résultats de référence :

Scenario success          ≈ 91.33 %
Full-turn exact           ≈ 92.57 %
Field exact               ≈ 93.71 %
Field precision           = 100 %
Field recall/value        ≈ 98.87 %
Field F1                  ≈ 99.43 %
Hallucination             = 0 %

Le LLM améliore principalement les formulations difficiles, notamment certaines requêtes multilingues et contenant des fautes de frappe.

16. Tester l’AI Plausibility Agent

Ollama et qwen2.5:3b doivent être disponibles.

python -m requirement_extractor.validation.run_ai_plausibility_full_agent_validation `
  --dataset requirement_extractor/validation/datasets/ai_plausibility_stress_dataset_v1.json `
  --ollama-model qwen2.5:3b `
  --temperature 0

Sous Linux/macOS :

python -m requirement_extractor.validation.run_ai_plausibility_full_agent_validation \
  --dataset requirement_extractor/validation/datasets/ai_plausibility_stress_dataset_v1.json \
  --ollama-model qwen2.5:3b \
  --temperature 0

Le benchmark dédié contient :

50 COHERENT
50 AMBIGUOUS
50 INCOHERENT
----------------
150 scénarios

La couche de décision déterministe gardée atteint la référence attendue sur ce benchmark dédié.

17. Test end-to-end du Requirement Pipeline

Ce test exécute la chaîne réelle :

raw text
→ HybridExtractor
→ optional LLM fallback
→ StateGuard
→ AI Plausibility
→ optional LLM enrichment
→ final outcome

Commande PowerShell :

python -m requirement_extractor.validation.run_end_to_end_validation `
  --dataset requirement_extractor/validation/datasets/end_to_end_stress_dataset_v1.json `
  --fallback-model qwen2.5-coder:7b `
  --plausibility-model qwen2.5:3b `
  --plausibility-temperature 0

Puis :

python -m requirement_extractor.validation.end_to_end_metrics
python -m requirement_extractor.validation.end_to_end_error_analyzer

Sous Linux/macOS :

python -m requirement_extractor.validation.run_end_to_end_validation \
  --dataset requirement_extractor/validation/datasets/end_to_end_stress_dataset_v1.json \
  --fallback-model qwen2.5-coder:7b \
  --plausibility-model qwen2.5:3b \
  --plausibility-temperature 0

python -m requirement_extractor.validation.end_to_end_metrics
python -m requirement_extractor.validation.end_to_end_error_analyzer

Résultat de référence :

30 scénarios
37 tours
30 / 30 scénarios réussis
0 erreur fonctionnelle
0 erreur technique

Distribution des sorties :

READY_COHERENT          7
READY_AMBIGUOUS         8
BLOCKED_PLAUSIBILITY    5
CLARIFICATION_REQUIRED 10

18. Interpréter les quatre sorties end-to-end

READY_COHERENT

Les exigences sont suffisamment complètes et cohérentes pour continuer.

READY_AMBIGUOUS

Les exigences sont valides, mais une contrainte dépend de l’architecture finale, par exemple budget, puissance ou débit à confirmer. La recommandation peut continuer avec un warning.

BLOCKED_PLAUSIBILITY

Les champs sont individuellement acceptables mais une contradiction inter-champs a été détectée. La génération doit s’arrêter jusqu’à correction.

CLARIFICATION_REQUIRED

Une information est manquante, invalide, conflictuelle ou doit être confirmée. Le chatbot interroge alors l’utilisateur.

19. Rejouer la génération des données d’architecture

Depuis :

cd lustre_architecture_generator

La chaîne de référence est :

python src/workload_analyzer.py
python src/feature_calculator.py
python src/architecture_generator.py
python src/mdt_candidate_generator.py
python src/validate_mdt_candidates.py
python src/ost_candidate_generator.py
python src/validate_ost_candidates.py
python src/training_dataset_builder.py

Ces scripts construisent successivement : workload analysis, workload features, hardware-independent Lustre requirements, MDT feasible candidates, OST feasible candidates et training datasets.

La génération des datasets et l’entraînement des modèles appartiennent au pipeline hors ligne.

Ils ne doivent pas être rejoués pour chaque nouvelle requête utilisateur.

20. Hors ligne vs en ligne

Hors ligne — développement

Créer / mettre à jour les datasets
↓
Générer les candidats
↓
Construire les labels
↓
Split train / validation / test
↓
Entraîner MDT / OST
↓
Évaluer
↓
Sauvegarder les modèles

En ligne — future recommandation réelle

Nouvelle requête utilisateur
↓
Extraction
↓
Validation
↓
Workload / features
↓
Filtrage déterministe
↓
Modèles déjà entraînés
↓
Top-K
↓
Beam Search
↓
Validation finale
↓
Réponse utilisateur

Une nouvelle demande utilisateur ne déclenche jamais un réentraînement des rankers.

21. Artefacts principaux de validation

Dans requirement_extractor/validation/, les artefacts importants incluent :

datasets/stress_requests_v1.json
datasets/ai_plausibility_stress_dataset_v1.json
datasets/end_to_end_stress_dataset_v1.json

metrics_deterministic.json
metrics_llm_fallback.json
metrics_ai_plausibility_full_agent.json

results_end_to_end_v1.json
metrics_end_to_end_v1.json
errors_end_to_end_v1.json

Ces fichiers constituent la baseline de non-régression du Requirement Pipeline.

Une modification future ne doit pas casser les comportements déjà validés.

22. Résultats end-to-end : latence

La latence dépend fortement du chemin réellement exécuté.

Référence du benchmark final :

deterministic fast path     ≈ quelques ms
plausibility LLM only       ≈ 10 s
fallback only               ≈ 25.5 s
both LLM paths              ≈ 41.2 s

Statistiques globales :

mean     ≈ 11.55 s
median   ≈ 7.93 s
P95      ≈ 42.43 s
P99      ≈ 49.84 s
maximum  ≈ 50.60 s

La principale source de latence est donc l’appel aux modèles locaux Ollama, pas le moteur déterministe.

23. Observations connues du benchmark end-to-end

Le run final comporte :

0 erreur fonctionnelle
11 observations non bloquantes

Fallback appelé sans candidat final

Plusieurs appels fallback n’ont produit aucun candidat finalement retenu.

Ce n’est pas une erreur de correction. Cela montre surtout qu’il est possible d’améliorer le routing afin d’éviter certains appels LLM coûteux lorsqu’aucune preuve textuelle exploitable n’existe.

JSON brut LLM invalide

Un enrichissement LLM a produit un JSON brut invalide.

Le garde de sécurité l’a rejeté et a conservé le warning déterministe. Le scénario final est donc resté correct.

24. Procédure rapide de démonstration pour l’encadrement

Pour une démonstration courte, utiliser cet ordre :

1. Activer l’environnement Python
2. Vérifier Ollama
3. Lancer le chatbot
4. Tester une requête complète
5. Tester une requête incomplète
6. Tester une contradiction
7. Exécuter la validation déterministe
8. Exécuter le benchmark end-to-end
9. Montrer les métriques
10. Montrer le pipeline de génération MDT / OST

Commandes minimales :

.\.venv\Scripts\Activate.ps1
ollama list
python -m requirement_extractor.main

Puis pour la validation rapide :

python -m requirement_extractor.validation.run_validation --mode deterministic
python -m requirement_extractor.validation.metrics --mode deterministic

Pour reproduire le run complet avec LLM, utiliser les commandes de la section 17.

25. Exemples de tests manuels

Cas complet

Je souhaite 500 TiB utilisables pour 200 clients.
Les fichiers font 2 GB en moyenne et 100 GB maximum.
Il y aura environ 10 millions de fichiers.
Le ratio est 70 % lecture et 30 % écriture.
Accès mixed.
Je vise 80 GB/s en lecture et 40 GB/s en écriture.
HA obligatoire.
Budget maximum 100000 USD.
Puissance maximale 15000 W.
Croissance annuelle 30 %.

Attendu : pas de valeur inventée, normalisation correcte, contrat complet, poursuite vers la plausibilité.

Cas incomplet

Je souhaite environ 500 TiB pour 200 clients.

Attendu :

CLARIFICATION_REQUIRED

Le système doit demander une information pertinente au lieu d’inventer les champs manquants.

Cas de conflit

Tour 1 : Mon budget maximum est 100000 USD.
Tour 2 : Mon budget est 150000 USD.

Attendu : conflit détecté puis clarification.

Cas de correction explicite

Correction : remplace mon budget par 150000 USD.

Attendu : mise à jour directe, pas de conflit artificiel.

Cas incohérent

Taille moyenne des fichiers : 100 GB
Taille maximale : 10 GB

Attendu :

INCOHERENT
AVERAGE_FILE_EXCEEDS_MAXIMUM

26. Dépannage

ollama n’est pas reconnu

Vérifier qu’Ollama est installé et présent dans le PATH, puis redémarrer le terminal VS Code.

Modèle Ollama absent

ollama list
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5:3b

Le LLM est trop lent

ollama ps

Le moteur déterministe peut toujours être validé indépendamment. Les benchmarks LLM locaux dépendent fortement du CPU/GPU disponible.

Une validation LLM semble bloquée

Vérifier le .env :

PLAUSIBILITY_AGENT_TIMEOUT_SECONDS=60
PLAUSIBILITY_AGENT_NUM_PREDICT=192
PLAUSIBILITY_AGENT_KEEP_ALIVE=30s

Erreur d’import Python

Toujours exécuter les commandes python -m requirement_extractor... depuis la racine version2/ et vérifier que l’environnement virtuel est activé.

27. Limites actuelles

Les résultats actuels doivent être interprétés dans le cadre de leurs benchmarks.

Points importants :

les datasets de ranking reposent encore sur un teacher déterministe synthétique ;

un score élevé sur le benchmark interne ne prouve pas une généralisation parfaite à toutes les requêtes réelles ;

le catalogue de drives doit être régulièrement revalidé contre les données constructeurs ;

la nomenclature gbps / GB/s doit être nettoyée dans une future version ;

les performances LLM dépendent fortement du hardware local ;

le Beam Search n’est pas encore intégré ;

la recommandation d’une architecture Lustre physique complète n’est donc pas encore la sortie finale du runtime actuel.

28. Prochaines étapes de développement

Ordre recommandé :

1. Exporter / figer les deux modèles LightGBM
2. Sauvegarder le feature schema et les mappings catégoriels
3. Créer le module d’inférence MDT / OST
4. Vérifier Kaggle vs VS Code sur les mêmes candidats
5. Produire un Top-K MDT et OST explicable
6. Définir ArchitectureState
7. Implémenter calculs capacité / coût / puissance / performance
8. Implémenter contraintes dures d’architecture
9. Définir le score global
10. Implémenter Beam Search
11. Ajouter le validateur final
12. Produire Top-N architectures
13. Générer les explications
14. Construire le benchmark end-to-end de recommandation complète

29. Philosophie scientifique du projet

Le système suit trois principes :

Deterministic rules
→ correctness and safety

Machine Learning rankers
→ efficient prioritization

LLM
→ linguistic understanding and explanation

Le modèle génératif ne doit jamais remplacer une contrainte métier vérifiable.

Le ranker ne doit jamais rendre faisable un candidat rejeté par le filtre déterministe.

Le futur Beam Search ne devra jamais retourner une architecture qui viole les contraintes du validateur final.

30. Résumé

Requirement understanding       ✅
Requirement validation          ✅
AI plausibility                 ✅
End-to-end requirement tests    ✅

Workload / feature pipeline     ✅
Architecture requirements       ✅
MDT / OST candidate datasets    ✅
MDT / OST training              ✅
LightGBM selection              ✅

LightGBM runtime integration    🚧
ArchitectureState               ⏳
Beam Search                     ⏳
Final architecture validator    ⏳
Full Lustre recommendation      ⏳

La prochaine étape logicielle est donc l’intégration des deux modèles LightGBM LambdaRank dans le code d’inférence avant le développement du Beam Search.