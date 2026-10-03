# Validation élargie V2 et correction V2.1

Le [rapport A–R](../evaluation/architecture/beam_search/v2_extended/extended_benchmark.md)
présente deux campagnes complètes sur les mêmes 50 nouveaux cas, les mêmes
K=5/10/20/50 et B=1/4/8/16/32, deux répétitions, un chemin H6/variante.
Les contraintes et préférences des cas n'ont pas été modifiées.

L'échantillonnage inclut les extrema numériques puis parcourt des strates
HA × tertile budget × tertile puissance × charge OST. L'ordre interne utilise
SHA256(case_id). Il exclut les trois cas de développement et ne filtre pas sur
la faisabilité. Les 50 cas constituent un échantillon de stress synthétique,
pas une estimation pondérée des charges de production.

## Défaut avant correction

La V2 produit 780/1 000 solutions, sans erreur ni nondéterminisme. Les seules
références exhaustives sous 2 000 paires ne révèlent aucun cas faisable manqué.
Un audit complémentaire, sans pool H9, trouve pourtant un témoin H10 VALID
dans un domaine de 90 000 paires. Le Beam le manque pour les cinq largeurs.

Le [diagnostic BEFORE](../evaluation/architecture/beam_search/v2_extended/before_joint_fix/diagnostic_joint.md)
montre que le drive MDT du témoin est classé 44e et éliminé dès MDT_DRIVE à
B=32. Les 32 drives retenus ne possèdent aucune complétion jointe faisable.
Les rangs possédant une vraie complétion sont 44, 47, 48 et 50.

Des minima séparés peuvent provenir de deux options OST différentes : une
option bon marché consomme trop, tandis qu'une option sobre coûte trop cher.
La borne de chaque contrainte reste admissible, mais leur combinaison ne
prouve pas l'existence d'une option satisfaisant les deux à la fois. Les
préfixes impossibles occupent la largeur et éliminent les préfixes faisables.

Les deux tests de reproduction échouent avant correction, avec un case_id
renommé. L'entrée immuable et le moteur V2 exact sont conservés ; aucun nom
de cas ni élément du benchmark n'est importé dans le moteur.

## Correction minimale

V2.1 conserve les couples (coût, puissance) non dominés de chaque préfixe.
Un point est supprimé seulement si un autre coûte et consomme autant ou moins.
Après tri par coût croissant, la puissance décroît strictement. Un balayage
à deux indices teste l'existence d'une somme MDT+OST respectant les deux
limites. Il évite la construction du produit cartésien et ne crée aucun état
COMPLETE anticipé.

Le contrôle intervient avant la sélection de largeur et ajoute la raison
`completion_joint_budget_power_conflict`. Les marges de comparaison sont
conservatrices vis-à-vis des arrondis ; H10 conserve sa décision exacte finale.
La formule, les poids et les tie-breaks de l'heuristique restent identiques.
H5–H10, le ranking et le contrat d'entrée restent frozen. Le schéma devient
2.1 et la politique est `BEAM_OPTIMISTIC_COMPLETION_V2_JOINT_RESOURCES`.

Chaque requête sur deux frontières prend O(f_MDT + f_OST), sans stocker leurs
paires. Avant compaction, il y a au plus quatre entrées de ressources par
option complète de rôle. La mémoire de ces index dépend du Top-K, des profils
et du cap de chemins ; elle n'est pas bornée par B. Les mesures portent sur
les nombres d'entrées, pas sur une borne RSS en octets.

## Résultats et portée

La même matrice relancée produit 785/1 000 solutions : cinq gains, aucune
perte, aucune modification des 180 qualités comparables. Le cas récupéré
devient VALID dès B=1. L'audit des 215 absences restantes (43 domaines case×K)
ne trouve aucun domaine faisable manqué sous les limites et caps configurés.
Cette infaisabilité locale ne prouve pas l'infaisabilité globale.

La validité est monotone dans 200/200 groupes. La qualité n'est monotone que
dans 9/36 groupes comparables. Qualité moyenne 98,2148 %, minimum 81,4296 % ;
les pertes de recherche et de normalisation sont conservées séparément.
Les grands domaines, notamment K=50, n'ont pas de qualité exhaustive inventée.

Les temps bruts des campagnes ne constituent pas un gain causal de la
correction : les temps de l'exhaustif frozen varient eux aussi. Le contrôle
alterné dans un même processus mesure 40 couples BEFORE/AFTER à B=32 sur
dix cas, deux répétitions. Ses résultats sont conservés dans
`interleaved_runtime.json`.

Le verdict courant est **BEAM V2.1 ACCEPTED WITH VALIDATION LIMITATIONS**,
établi dans la [revue finale A–R](../evaluation/architecture/beam_search/v2_extended/release_review.md).
Le verdict historique **BEAM V2 REQUIRES FURTHER ALGORITHMIC WORK** concernait
uniquement la V2 originale et son défaut aujourd'hui corrigé. Aucun nouveau
bug V2.1 n'est démontré. Le moteur est gelé à son hash de campagne.

Les 51 domaines H8/H9/H10 entièrement parcourus donnent 36 domaines faisables,
retrouvés aux cinq largeurs (180 configurations comparables), et 15 domaines
sans VALID. Les deux regrets sont analysés sur toutes les comparaisons
accessibles : **analysis available on exhaustively tractable domains only**.
`PARTIAL` décrit cette portée (cas A), sans travail analytique restant sur les
domaines traitables. L'audit scalaire des 43 domaines sans solution est une
preuve de faisabilité locale, pas une nouvelle référence exhaustive de qualité.

Les limitations connues, distinctes des bugs de correction, sont listées dans
[Known limitations — not correctness bugs](beam_search.md#known-limitations--not-correctness-bugs).
La couverture synthétique et les références limitées ne garantissent ni
optimalité globale ni généralisation à des données réelles.
