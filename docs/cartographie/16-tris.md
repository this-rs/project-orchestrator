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

- **Corrigés** : ~111 clauses réparties sur 33 fichiers (chaque tri partiel a
  reçu sa clé de départage totale en fin de tri). Rien d'autre n'a changé dans
  les requêtes.
- **Restants (documentés, à départager)** : les tris « score » et « poids » de
  recherche sémantique, les agrégats d'analytics, les relations sans `id`
  (`TriggerFiring`, `ExternalEvent`) et quelques tris internes — voir la liste
  ci-dessous.

## Restants (partiels non encore départagés)

- `decision.rs:527,536` — `ORDER BY score DESC` (recherche sémantique, nœud à confirmer)
- `feature_graph.rs:522,1223` — `pagerank`/`connections` (nœud `peer`)
- `persona.rs` (17) — `energy`, `weight`, `name`, `depth`, `file_count`,
  `overlap`, `avg_weight`, `co_count`
- `ingress.rs:75` — `e.received_at DESC` (`ExternalEvent` sans `id`)
- `trigger.rs:368` — `f.fired_at DESC` (`TriggerFiring` sans `id`)
- `graph_schema.rs:119` — `depth LIMIT 200`
- `plan.rs:625` — `pathLength DESC`
- `note.rs` (10) — `score DESC`, `weight DESC` (recherche sémantique)
- `data_migrations.rs:183,376` — `members/refs/created_at`, `notes DESC`
- `analytics.rs` (12) — `file_count`, `in_deg`, `size(cycle)`, `betweenness`,
  `commit_count`, `raw_density`, `churn_score`, `knowledge_density`,
  `SIZE(members)`, `rel_type`, `WITH … created_at/updated_at`
- `document.rs:717,726` — `score DESC`
- `topology.rs:92,260,318,382` — `rule_type/source_pattern`, `fan_out`,
  `shortest_cycle`, `dist`
- `neighborhood.rs:139` — `w DESC, coalesce(m.energy, m.pagerank, 0) DESC`
- `chat.rs` (17) — `updated_at`, `score`, `collected_at`, `created_at`, `depth`,
  `length(path)`, `mention_count`, `count`
- `trajectory_store.rs:487,542` — `created_at`, `score DESC`
- `graph_sampler.rs:277,285` — `pr DESC`
- `client.rs:1123` — `t.created_at DESC` (doublon de `trigger.rs`)

## Fichiers corrigés (33)

`agent_execution`, `alert`, `beliefs`, `code`, `commit`, `decision`, `document`,
`environment`, `event_trigger`, `feature_graph`, `lifecycle_hook`,
`mcp_federation`, `milestone`, `note`, `plan`, `plan_run`, `profile`, `project`,
`protocol`, `reasoning`, `registry`, `release`, `skill`, `task`,
`trajectory_migration`, `trigger`, `user`, `workspace`.

## Test

`src/neo4j/order_tiebreak.rs` — garde-fou statique (tourne **sans** Neo4j) qui
relit les sources, extrait les `ORDER BY` des fichiers corrigés et vérifie que
chaque clause se termine par une clé totale.
