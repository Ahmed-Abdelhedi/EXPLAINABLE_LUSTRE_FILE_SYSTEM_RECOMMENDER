# Hybrid Lustre Requirement Chatbot v2

Assistant conversationnel en ligne de commande destiné à recueillir, structurer et valider les exigences d'une infrastructure de stockage Lustre. Le projet privilégie une extraction déterministe par règles, peut solliciter un modèle local via Ollama pour les formulations ambiguës, puis demande à l'utilisateur de clarifier toute information manquante, invalide ou contradictoire.

> Le projet est un MVP de qualification du besoin. Il calcule une capacité utile planifiée, mais ne dimensionne pas encore le matériel et ne produit pas de configuration Lustre complète.

## Fonctionnalités

- extraction en français et en anglais avec priorité aux règles déterministes ;
- fallback LLM facultatif, limité aux champs encore inconnus et appuyé par une preuve textuelle ;
- conversation multi-tour pour compléter ou corriger les valeurs ;
- détection des champs manquants, conflits, valeurs invalides et extractions non justifiées ;
- normalisation des unités (`MB` vers `GB`, `kW` vers `W`, etc.) ;
- vérification globale facultative de la plausibilité par un second agent Ollama ;
- conservation de la valeur, de l'unité, de la confiance, de la source et du passage justificatif ;
- calcul de la capacité planifiée après validation complète ;
- suites de tests déterministes, de robustesse et de généralisation.

## Fonctionnement

```text
Message utilisateur
        │
        ▼
Extraction par règles ──► fallback LLM facultatif
        │
        ▼
Normalisation et validation (StateGuard)
        │
        ├── problème détecté ──► question de clarification ──► nouveau tour
        │
        ▼
Contrôle de plausibilité facultatif
        │
        ▼
Calcul de capacité et résultat final
```

Les règles restent toujours prioritaires : le fallback LLM ne peut pas remplacer un champ déjà extrait par une règle. Le contrôle de plausibilité ne modifie jamais une valeur automatiquement ; une éventuelle correction doit être confirmée par l'utilisateur.

## Données collectées

Les 13 champs suivants sont obligatoires avant le calcul :

| Champ JSON | Description | Unité ou format final |
| --- | --- | --- |
| `requested_usable_capacity_tib` | Capacité utile demandée | `TiB` |
| `client_count` | Nombre de clients ou nœuds de calcul | entier |
| `average_file_size_gb` | Taille moyenne des fichiers | `GB` |
| `max_file_size_gb` | Taille maximale typique des fichiers | `GB` |
| `total_file_count` | Nombre total de fichiers | entier |
| `read_write_ratio` | Répartition lecture/écriture | objet dont la somme vaut 100 % |
| `access_type` | Profil d'accès | `sequential`, `random`, `parallel`, `streaming` ou `mixed` |
| `target_read_gbps` | Débit de lecture cible | `GB/s` |
| `target_write_gbps` | Débit d'écriture cible | `GB/s` |
| `ha_required` | Haute disponibilité obligatoire | booléen |
| `max_budget_usd` | Budget maximal | `USD` |
| `max_power_w` | Puissance maximale | `W` |
| `annual_growth_percent` | Croissance annuelle prévue | `%` |

## Prérequis

- Python 3.10 ou version ultérieure ;
- `pip` ;
- Ollama uniquement si le fallback LLM ou le contrôle de plausibilité est activé.

## Installation

Depuis la racine du projet :

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Sous Linux ou macOS, l'activation de l'environnement se fait avec :

```bash
source venv/bin/activate
```

## Configuration

L'application console charge automatiquement le fichier `.env` situé à la racine. Pour exécuter le pipeline sans dépendance à un modèle local :

```dotenv
ENABLE_LLM_FALLBACK=false
ENABLE_AI_PLAUSIBILITY_AGENT=false
```

Pour activer les deux composants Ollama :

```dotenv
ENABLE_LLM_FALLBACK=true
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5-coder:7b

ENABLE_AI_PLAUSIBILITY_AGENT=true
PLAUSIBILITY_AGENT_MODEL=qwen2.5:3b
PLAUSIBILITY_AGENT_TEMPERATURE=0.0
PLAUSIBILITY_AGENT_DEBUG=false
```

Il faut alors démarrer Ollama et télécharger les modèles configurés, par exemple :

```bash
ollama serve
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5:3b
```

| Variable | Valeur par défaut | Rôle |
| --- | --- | --- |
| `ENABLE_LLM_FALLBACK` | `false` | Active l'extraction de secours par LLM. |
| `OLLAMA_HOST` | `http://localhost:11434` | Adresse du serveur Ollama. |
| `OLLAMA_MODEL` | `qwen2.5-coder:7b` | Modèle utilisé pour l'extraction de secours. |
| `ENABLE_AI_PLAUSIBILITY_AGENT` | `true` | Active la vérification globale de cohérence. |
| `PLAUSIBILITY_AGENT_MODEL` | valeur de `OLLAMA_MODEL`, sinon `qwen2.5:3b` | Modèle chargé de la plausibilité. |
| `PLAUSIBILITY_AGENT_TEMPERATURE` | `0.0` | Température du contrôle de plausibilité. |
| `PLAUSIBILITY_AGENT_DEBUG` | `false` | Affiche les échanges de diagnostic avec le modèle. |

Le fichier `.env` est ignoré par Git. Pour un fonctionnement entièrement déterministe et hors ligne, désactivez explicitement les deux options d'IA.

## Utilisation

Lancer l'interface interactive :

```bash
python main.py
```

Exemple de besoin complet :

```text
On veut 500 TiB utiles, 200 clients, des fichiers moyens de 2 GB et de
100 GB au maximum, 10 millions de fichiers, un ratio 70/30, un accès mixte,
80 GB/s en lecture, 40 GB/s en écriture, HA obligatoire, un budget maximal
de 100000 USD, une puissance de 15 kW et une croissance annuelle de 30 %.
```

À chaque tour, le programme affiche l'étape du pipeline, son statut, le JSON courant, puis la prochaine question si une clarification est nécessaire.

Commandes disponibles dans la console :

- `reset`, `restart` ou `new` : recommencer une conversation ;
- `exit` ou `quit` : quitter l'application.

## Utilisation depuis Python

```python
from requirement_chatbot import RequirementChatbot

bot = RequirementChatbot()
state = bot.process_user_message(
    "Je cherche 500 TiB utiles pour 200 clients."
)

print(state.status.value)
print(state.to_dict())

# Répondre à la première demande de clarification
if state.questions:
    state = bot.process_user_message("2 GB")
```

Lors d'une intégration directe, chargez vous-même `.env` avec `load_dotenv()` avant de créer `RequirementChatbot` si vous souhaitez utiliser cette configuration.

## Calcul réalisé

Quand tous les champs sont valides et plausibles, `CalculationEngine` applique une réserve de croissance sur un an et un taux de remplissage cible de 80 % :

```text
facteur_de_croissance = 1 + croissance_annuelle / 100
capacité_planifiée = capacité_demandée × facteur_de_croissance / 0,80
```

Le résultat contient notamment `growth_factor`, `target_fill_ratio` et `planned_usable_capacity_tib`.

## Tests

Test rapide de trois cas représentatifs :

```bash
python quick_test.py
```

Suite de stress principale, sans fallback LLM par défaut :

```bash
python tests/run_stress_tests.py
```

Quelques variantes utiles :

```bash
# Activer le fallback Ollama
python tests/run_stress_tests.py --llm

# Utiliser les jeux de robustesse ou de généralisation
python tests/run_stress_tests.py --dataset tests/hard_stress_dataset.json --results tests/results_hard
python tests/run_stress_tests.py --dataset tests/generalization_dataset.json --results tests/results_generalization

# Arrêter au premier échec
python tests/run_stress_tests.py --fail-fast
```

Deux campagnes supplémentaires testent notamment le multi-tour, les corrections explicites, les unités incorrectes et la plausibilité relationnelle. Elles activent Ollama par défaut :

```bash
python test1/run_test1.py
python test2/run_test2.py

# Exécution entièrement déterministe
python test1/run_test1.py --no-llm --no-ai
python test2/run_test2.py --no-llm --no-ai
```

Les rapports générés sont enregistrés dans les dossiers `results` correspondants, aux formats JSON et CSV.

## Structure du projet

```text
.
├── main.py                       # Interface interactive
├── requirement_chatbot.py        # Orchestration de la conversation
├── hybrid_extractor.py           # Priorité aux règles et appel du fallback
├── rule_entity_extractor.py      # Extraction déterministe
├── llm_fallback_extractor.py     # Extraction Ollama contrôlée
├── clarification_agent.py        # Interprétation des réponses courtes
├── state_guard.py                # Validation et résolution des conflits
├── ai_plausibility_agent.py      # Contrôle global facultatif
├── calculation_engine.py         # Calcul de capacité planifiée
├── text_preprocessor.py          # Prétraitement du texte
├── unit_normalizer.py            # Conversion des unités
├── closed_vocabulary_mapper.py   # Normalisation des valeurs fermées
├── field_defs.py                 # Champs, unités et questions
├── models.py                     # Modèles de données et états
├── quick_test.py                 # Vérification rapide
├── tests/                        # Stress tests et généralisation
├── test1/                        # Première campagne avancée
├── test2/                        # Deuxième campagne avancée
└── requirements.txt              # Dépendances Python
```

## Limites actuelles

- le stockage cible est exclusivement Lustre ;
- tous les champs définis sont obligatoires ;
- la conversion `TB` vers `TiB` est volontairement simplifiée dans ce MVP ;
- le contrôle par IA dépend de la disponibilité et du comportement du modèle Ollama choisi ;
- la sortie de recommandation se limite actuellement au calcul de capacité, sans sélection de serveurs, disques ou réseau.
