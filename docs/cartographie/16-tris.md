# 16 — Tris `ORDER BY` : totalité, départage et règle obrain

## Le problème

`src/neo4j/` contient **274 clauses `ORDER BY`** (le comptage « 282 » inclut 8
commentaires et une assertion de test). Un tri **partiel** — par exemple
`coalesce(priority) DESC, coalesce(last_seen, created_at) DESC` — laisse des
égalités que le moteur tranche arbitrairement. Neo4j et obrain peuvent choisir
un ordre différent **sans lever d'erreur** : les mêmes lignes sortent dans un
autre ordre et une UI casse en silence. Aucun test ne créait d'égalité
volontaire.

## Classification

- **total** — aucune égalité possible : un identifiant unique (`id`, `path`,
  `hash`, `server_id`, `run_id`, `key`, `label`, `relationshipType`, `cid`,
  `tid`, `entity_id`) est une clé de tri.
- **partiel** — départage nécessaire (timestamp, score, poids, énergie,
  priorité, nom, compteur, pagerank, profondeur… sans identifiant en fin de tri).

## La règle (spécification pour obrain)

> **À égalité sur toutes les colonnes de tri, l'ordre est celui de
> l'identifiant unique du nœud retourné, dans le même sens que la dernière
> colonne de tri (ASC si le tri est croissant, DESC s'il est décroissant ;
> croissant quand la direction est implicite).**
>
> Concrètement : un tri partiel se termine toujours par `<nœud>.id <dir>`
> (ou `path`/`hash` pour les nœuds sans `id` — `File`, `Commit` —, ou
> l'identifiant des extrémités pour une relation). La garantie d'ordre est
> dans le contrat, pas dans le hasard d'un moteur.

## État

- **Corrigés** : les ~80 clauses partielles restantes recensées dans la
  précédente passe ont reçu leur clé de départage totale en fin de tri, réparties
  par catégorie :
  - **relations/événements journalisés** (`ExternalEvent`, `TriggerFiring`) →
    `e.id DESC` / `f.id DESC` ;
  - **scores/poids de recherche sémantique** (flottants) → `id` du nœud retourné
    (`d.id`, `n.id`, `c.id`, `fn.id`, `t.id`), ou `f.path` quand le nœud est un
    `File` ;
  - **poids SYNAPSE** → extrémités de l'arête (`source_id, target_id` ou
    `neighbor_id`) ;
  - **agrégats analytics** → identifiant du groupe (`cid`, `wl_hash`, ou les
    colonnes du `GROUP BY` implicite) ;
  - **entités hétérogènes** → `coalesce(e.id, e.path)` ;
  - **chemin critique** → liste ordonnée des identifiants du chemin
    (`[n IN nodes(path) | n.id]`).
- Rien d'autre n'a changé dans les requêtes.

## Restants (aucun départage possible — ou inutile)

- `client.rs:1123` — `ORDER BY t.created_at DESC` : c'est une **chaîne de test**
  de `is_read_only_cypher` (garde-fou lecture seule), pas une requête exécutée.
  Aucun départage nécessaire.
- `neighborhood.rs:139` — déjà total avant cette passe : le tri se terminait
  déjà par `coalesce(m.id, m.path, m.hash) ASC`. L'entrée « Restants » était
  périmée.
- `analytics.rs:2580` (`ORDER BY rel_type`) — `rel_type` est `type(r)`, la clé
  unique d'une relation : déjà total.
- `persona.rs:532` (`length(f.path) LIMIT 1`) — `LIMIT 1`, une seule ligne :
  l'ordre n'a pas d'objet.

## Fichiers corrigés (37)

`agent_execution`, `alert`, `analytics`, `beliefs`, `chat`, `code`, `commit`,
`data_migrations`, `decision`, `document`, `environment`, `event_trigger`,
`feature_graph`, `graph_sampler`, `graph_schema`, `ingress`, `lifecycle_hook`,
`mcp_federation`, `milestone`, `note`, `persona`, `plan`, `plan_run`, `profile`,
`project`, `protocol`, `reasoning`, `registry`, `release`, `skill`, `task`,
`topology`, `trajectory_migration`, `trajectory_store`, `trigger`, `user`,
`workspace`.

## Test

`tests/order_tiebreak.rs` — garde-fou statique (tourne **sans** Neo4j) qui
relit les sources, extrait les `ORDER BY` des fichiers corrigés et vérifie que
chaque clause se termine par une clé totale. La liste `CASES` couvre désormais,
en plus des tris nom/priorité/énergie, les nouvelles formes : score de
recherche sémantique, poids SYNAPSE, agrégats analytics, entités hétérogènes et
chemin critique.
