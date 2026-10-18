# Comparaison PO / Codebase-memory-MCP en conteneurs

Préparé le 13 octobre 2026. **Aucun conteneur ni benchmark n'a été exécuté dans
l'environnement local d'origine** : moteur Docker absent, namespaces interdits et
aucun hôte distant configuré. Le workflow `Code index comparison` exécute ce kit sur
un runner Ubuntu avec Docker ; ses logs et son artefact sont les preuves d'exécution.

## Lancer

Sur la branche dédiée, un push des fichiers de ce kit déclenche le workflow isolé.
La configuration `compose.ci.yaml` utilise les miroirs CI existants et leurs digests :
Neo4j 5.26.31 et Meilisearch 1.34.3. Le rapport conserve les fichiers Compose utilisés.

Prérequis : Docker avec Compose, git, Python 3.10+, connexion Internet et environ
8 Go de RAM disponibles pour Docker. Sur un Mac Apple Silicon ou un DGX ARM64,
le Dockerfile sélectionne les binaires Linux ARM64 ; sur x86_64, les binaires AMD64.
Il n'utilise ni mode privileged ni montage du socket Docker dans les conteneurs.

```bash
bash run.sh
```

Cette commande :

1. Récupère le dépôt PO au commit `cfb58ee2f0927c975f3fe7e19dd60e1cbf646fdf`.
2. Génère deux mots de passe locaux dans `.env` sans les afficher.
3. Construit une image avec PO v0.0.17 et CBM v0.11.0, vérifiés par leurs SHA-256.
4. Lance PO, Neo4j 5.26.20, Meilisearch 1.34.2, CBM et notre lecteur Tree-sitter.
5. Déclare un projet PO avec son root_path, puis appelle son API de synchronisation.
6. Indexe exactement le même checkout avec CBM et le lecteur ; conserve les réponses
   brutes et les durées dans `results/`.

Les sources sont montées en lecture seule. Les trois index sont indépendants.
Les services HTTP publiés sont limités à la boucle locale : PO 18080, Neo4j 17474.
NATS est optionnel dans PO et absent de cette expérience.

## Ce qui est comparable

Le contenu source est commun. Le parser et le résolveur de PO v0.0.17 sont identiques
à ceux du commit source, vérifié par git diff sur src/parser et src/resolver.
Le runtime PO complet est plus ancien : commit `8a8f0b770beb10bf58075f8f89ad2e99c200ee4f`.
Le rapport ne doit donc pas être présenté comme un benchmark de main en entier.

Les embeddings PO sont désactivés : ce parcours compare l'index structurel et la
recherche textuelle. CBM en mode full suit son propre parcours ; ses coûts de
traitement ne sont pas équivalents à ceux de PO. Les résultats ne prouvent pas une
équivalence de recherche sémantique. Les durées incluent les transports et le
démarrage des commandes. Une seule exécution n'est pas un benchmark statistique.

Les requêtes portent sur with_cognitive_routing, decision_confidence,
build_enrichment_pipeline et close_decision_with. Inspecter les fichiers de preuve :
un résultat vide ne démontre pas l'absence d'un appel, notamment dans des macros.
Le nombre d'arêtes résolues n'est pas comparable directement au nombre de sites
d'appel syntaxiques. Les tailles sont en octets UTF-8, pas en tokens mesurés.

La vérité terrain des call sites (grep sur cfb58ee2, 13/10/2026) est dans
`ground-truth.json` : 12, 7, 3 et 3 sites pour ces quatre symboles. C'est la
référence du LECTEUR (syntaxe) ; PO (`CALLS`) et CBM (graphe) résolvent des
arêtes et peuvent rester en dessous sans être faux.

Une seconde exécution conserve les volumes et devient une expérience avec des
états préexistants ; le rapport enregistre l'état PO avant la synchronisation et
la liste CBM avant l'indexation. Aucun volume n'est effacé automatiquement.

## Arrêter et lire les logs

```bash
docker compose logs --tail=100 po neo4j meilisearch
docker compose down
```

`down` conserve les volumes. Ne supprimer les volumes que si tu veux effacer ces
index dédiés. Ne pas exécuter la configuration contre une instance PO de production.

Pour une exécution distante, copier le kit sur l'hôte Docker et y lancer run.sh.
Un contexte Docker distant seul ne transfère pas les sources des bind mounts.
L'archive ne contient ni credentials ni données de benchmark inventées.
