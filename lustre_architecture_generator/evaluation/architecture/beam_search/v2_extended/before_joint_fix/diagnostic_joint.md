# Défaut V2 mesuré avant correction

La campagne entière (50 nouveaux cas, 1 000 configurations, deux répétitions)
est sauvegardée dans ce dossier, avec le moteur V2 exact et ses empreintes.
L'audit indépendant trouve un témoin H10 VALID dans le domaine K=50,
un chemin/variante, de REQ_000267. Aucune des cinq largeurs ne trouve de solution.

Budget : 3 060 580 USD. Puissance : 94 897 W. Le témoin coûte 3 030 960 USD
et consomme 94 430 W ; son identifiant est
`ARCH_REQ_000267_e62f7c71ba739177`. Le domaine contient 90 000 paires.
Il n'a pas été utilisé comme référence de qualité exhaustive.

À B=32, le préfixe du témoin est WIDTH_PRUNED dès MDT_DRIVE, au rang 44.
Les drives MDT ayant au moins une vraie complétion jointe sont aux rangs
44, 47, 48 et 50. Aucun des 32 drives retenus n'en possède une.
La frontière devient vide à OST_DRIVE ; aucun COMPLETE/H10 n'est produit.

Les minima séparés coût et puissance sont des bornes admissibles, mais peuvent
provenir de deux options OST différentes. Leur combinaison laisse croire
qu'une complétion respecte les deux contraintes simultanément, alors qu'aucune
option réelle ne le peut. Ces préfixes impossibles occupent toute la largeur
et éliminent les préfixes faisables. Il ne s'agit ni d'un faux hard pruning,
ni d'un effet H9, ni d'un mauvais résultat H10.

Reproduction générique : une contribution MDT (3, 3), des options OST
(1, 10) et (10, 1), et des limites (12, 12) passent les minima séparés (4, 4),
mais ne possèdent aucune paire faisable. Une contribution MDT (1, 1) permet
une vraie complétion. Le mécanisme ne dépend d'aucun case_id.

Le test `test_beam_search_joint_resources.py` reprend le handoff immuable,
renomme le cas, démontre indépendamment la faisabilité avec les fonctions
publiques, puis exige une solution à B=1 et B=32. Il est exécuté avant
toute modification du moteur ; sa sortie rouge est conservée.

Correction envisagée : conserver par préfixe les couples coût/puissance
non dominés des options de chaque rôle, et tester l'existence d'une somme
respectant conjointement les deux limites avant la sélection de largeur.
Un balayage des deux frontières triées évite le produit cartésien et
ne construit aucune architecture COMPLETE. Aucun poids ni score H9 ne change.
