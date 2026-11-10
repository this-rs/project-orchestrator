# 14 — Périmètre des traits de stockage : ce que PO exige réellement des 3 moteurs

Audit **lecture seule** (aucun build, aucune modif de source, aucun git). Dépôt : `worktrees/obrain-default` (branche consolidation).
Méthode : lecture croisée des traits vs adaptateurs, plus le portillon `cargo xtask storage-boundary --strict` (résultat : **0 référence de pilote hors adaptateur non exceptée** — le mur de types est propre ; les fuites sont *sémantiques*, pas de types).

## Verdict court
L'abstraction **de types** est bonne : les signatures des traits ne nomment jamais `neo4rs`/`meilisearch_sdk`/`async_nats` (0 occurrence dans `traits.rs`). Mais trois familles de **comportements sans nom** casseraient en silence à la migration. Détail ci-dessous.

---

## 1. Neo4j — 21 sous-traits, 732 `async fn`

| Capacité | Utilisée où (chemin:ligne) | Dans un trait ? | Verdict |
|---|---|---|---|
| Arithmétique `datetime()`/`duration.*` (décroissance, fraîcheur, énergie) | `src/neo4j/note.rs:2116,2138,2971,3381`, `mcp_federation.rs:480`, `synapse_shadow.rs:169` (265 sites) | `StoreCapabilities.datetime_arithmetic` (`traits.rs:60`) | **nommé, jamais consommé** |
| `DETACH DELETE` (cascade) | 100 sites dans `src/neo4j/*.rs` | non | trou de sémantique |
| `UNWIND` en lot | 208 sites | non (détail d'impl acceptable) | neutre |
| Sous-requêtes `CALL {}` | 19 sites | `StoreCapabilities.subqueries` | nommé, jamais consommé |
| Tri départagé (`ORDER BY coalesce(a.priority,0) DESC, coalesce(a.last_seen,...) DESC`) | `src/neo4j/alert.rs:98,106,211` (282 `ORDER BY` au total) | non | **trou** : l'ordre des ex æquo change les tops |
| Index de relation `FOR ()-[r:TOUCHES]-() ON (r.file_path)` | `src/neo4j/client.rs:665-679` | `StoreCapabilities.relationship_property_index` | nommé, jamais consommé |
| Contraintes d'unicité composites / simples | `src/neo4j/client.rs:448-477` (191 `CREATE CONSTRAINT/INDEX`) | `StoreCapabilities.composite_unique` / `unique_constraints` | nommé, jamais consommé |
| Retry transient (`Neo.TransientError.Transaction.Outdated`) read-only vs write | `src/neo4j/client.rs:956,1010` (`read_retrying`, `retry_transient`) | non | trou de comportement |
| Décodage `BoltType`/`BoltMap`/`neo4rs::Node` | `src/neo4j/client.rs:1135-1144` et ~124 sites `Bolt*` | non (confiné à l'adaptateur — OK) | obrain n'a pas à le faire ; PO oui |

**Points structurants :**
- `StoreCapabilities` (`traits.rs:42`) est un **manifeste déclaratif** : il nomme bien `vector_search`, `vector_filter`, `composite_unique`, `subqueries`, `datetime_arithmetic`, `relationship_property_index`, `shortest_path`, `variable_length_paths`, `explicit_transactions`. Mais **aucun code de production ne branche dessus** — seule lecture : `src/api/handlers.rs:227,290` qui l'expose à l'UI. Un obrain qui annonce `datetime_arithmetic: false` reçoit quand même le même Cypher. C'est la bombe n°1 de ce périmètre.
- **Trajectory store hors fabrique** : `neural_routing_core::traits::TrajectoryStore` (`crates/neural-routing-core/src/traits.rs:11`) est un trait séparé, mais l'adaptateur est ouvert **en direct** dans la fabrique (`src/storage.rs` → `crate::neo4j::trajectory::open()`), pas via le sélecteur de backend. obrain ne peut pas le remplacer sans modifier `storage.rs`.
- Delta brut 888 méthodes publiques d'adaptateur vs 737 de trait : en grande partie les helpers exclus *documentés* (`execute`, `execute_with_params`, `init_schema`, décodeurs). À vérifier par un diff mécanique 1:1, pas un trou prouvé.

## 2. Meilisearch — `SearchStore`, 27 méthodes

| Capacité | Utilisée où | Dans un trait ? | Verdict |
|---|---|---|---|
| Filtres (`=`, `IN […]`, `STARTS WITH`, `IS EMPTY`) | `src/meilisearch/client.rs:234,296,383` | oui, en **paramètres typés** (`project_slug`, `note_type`, `status`, `importance`, `path_prefix`, `project_slugs`) — pas de chaîne de filtre brute | couvert |
| Schéma filterable/sortable posé à la création d'index | `client.rs:97-151` (`with_filterable_attributes`, `with_sortable_attributes`) | non | trou (obrain doit reproduire l'indexation) |
| Scores de classement | `client.rs:247,259` (`with_show_ranking_score`) + variantes `_with_scores` | oui | couvert |
| **Tolérance aux fautes + règles de classement par défaut** | implicite (jamais configuré) | non | **trou** : PO s'appuie sur les défauts Meili non exprimés |
| Facettes | non utilisé | — | neutre |
| Pagination par curseur / offset | **aucun** (`grep offset/page/cursor` = 0) | — | `limit` seul ; obrain sans curseur = OK |
| Stats d'index | `get_code_stats`, `get_notes_stats` → `IndexStats` | oui | couvert |
| Mapping d'erreur | `client.rs:44,66` (`.context("Failed to …")`) | non | forme d'erreur non contractuelle |

## 3. NATS — `EventEmitter` (fire-and-forget) vs transport de chat

| Capacité | Utilisée où | Dans un trait ? | Verdict |
|---|---|---|---|
| `emit` CRUD + `transport_state` + `emit_graph` | `src/events/types.rs:192` (`EventEmitter`) | oui | couvert |
| **Request/reply** (`request`, `publish_with_reply`, inbox, `StatusCode::NO_RESPONDERS`) | `src/events/nats.rs:426,442,452` | non (types `Subject`/`Message`/`Subscriber` dans `transport.rs`, mais **aucun trait** qui les porte) | **trou n°1** |
| **Pub/sub chat par session** (`publish_chat_event`, `subscribe_chat_events`) | `manager.rs:684,1413…`, `ws_chat_handler.rs:391,829`, `mirror.rs:1037` | non | trou |
| **Signaux** (`publish_interrupt`, `publish_cancel_tools`, `subscribe_*`) | `manager.rs:5187,5277,15070,15550` | non | trou |
| **Snapshot streaming** (`request_streaming_snapshot`, `subscribe_snapshot_requests`) | `recovery.rs:647`, `ws_chat_handler.rs:418` | non | trou |
| **Cancel RPC** (`request_cancel`, `RelayFailure::{NoResponders,Timeout,SendFailed}`) | `cancel_relay.rs:280,322`, `nats.rs:55-59` | non | trou |
| Timeouts RPC | `nats.rs:316` (`tokio::time::timeout`), `with_cancel_rpc_timeout:115` | non | trou (durées + forme d'erreur) |
| JetStream / en-têtes / groupes de file / ack / redelivery | **absent** (le commentaire `mod.rs` « JetStream » est périmé : tout passe par core NATS) | — | neutre, mais le doc ment |

Le transport de chat est consommé **concrètement** : `Option<Arc<NatsEmitter>>` est tissé dans `manager.rs` (~40 sites), `post_stream.rs`, `drain.rs`, `oob_listener.rs`, `mirror.rs`, `recovery.rs`, `ws_chat_handler.rs`, `vault_handlers.rs`. Seul le CRUD passe par le trait.

---

## 4. Comportements sans nom (le précieux)

| Comportement | Chemin:ligne | Ce qui casse si obrain ne le fait pas comme Neo4j/NATS |
|---|---|---|
| Ordre des ex æquo (`coalesce(priority) DESC, coalesce(last_seen, created_at) DESC`) | `alert.rs:98` | tops d'alertes/analytics différents, ordres UI instables |
| Arithmétique de dates *dans* les requêtes (décroissance, fraîcheur) | `note.rs:2138,2971` | scores neuronaux/fraîcheur faux sans erreur |
| Cascade `DETACH DELETE` | `src/neo4j/*.rs` (100) | suppression laissant des orphelins ou en échouant |
| Retry des erreurs transientes (lecture oui, écriture non) | `client.rs:956,1010` | migrations interrompues sur contention |
| Forme d'erreur (`RelayFailure::NoResponders` vs timeout vs échec d'envoi) | `nats.rs:55` | la reprise/cancel ne distingue plus « absent » de « lent » |
| Délais RPC (`CANCEL_TASK_RPC_TIMEOUT`) | `nats.rs:91` | cancel/snapshot pendent ou échouent |
| Défauts de ranking/typo Meili non configurés | `meilisearch/client.rs` (implicite) | classement de recherche différent |

---

## 5. Verdict final

### obrain doit fournir (X)
1. Un **transport pub/sub + request/reply** équivalent à core NATS : `publish` → sujet, `subscribe` → flux, `request` avec timeout + signal « no responders », inbox de réponse. Sémar­tique de livraison : fire-and-forget pour le CRUD, au-moins-une-réponse pour le RPC cancel/snapshot.
2. Un **graphe** supportant : contraintes d'unicité (simples **et** composites), index de **propriétés de relation**, `DETACH DELETE` cascade, sous-requêtes, chemins à longueur variable + `shortestPath`, arithmétique `datetime`/`duration` dans les requêtes.
3. Une **recherche** : filtres `=`, `IN`, `STARTS WITH`, `IS EMPTY`, attributs filterable/sortable, scores de classement, stats d'index — avec des **règles de classement et une tolérance aux fautes explicites** (PO ne peut plus se reposer sur un défaut non écrit).
4. La même **sémantique de tri départagé** et la même **forme d'erreur** (transient vs permanent, absent vs timeout).

### PO doit changer (Y)
1. **Introduire un trait de transport de chat** (ex. `ChatTransport`) portant `publish_chat_event`/`subscribe_*`/`request_*`, et remplacer `Arc<NatsEmitter>` par `Arc<dyn ChatTransport>` dans `chat/` (~40 sites). Sans cela, la bascule du chat est impossible.
2. **Consommer `StoreCapabilities`** (ou le supprimer) : aujourd'hui il est affiché mais jamais utilisé pour dégrader. Chaque capacité doit avoir un chemin de repli nommé.
3. **Router le trajectory store par la fabrique** (`storage.rs`) au lieu d'ouvrir `neo4j::trajectory::open()` en dur.
4. **Expliciter** les défauts Meili (typo/ranking) et la politique de retry transient dans le trait (au moins en doc).

### Ne peut pas être abstrait (et pourquoi)
- **L'ordre exact des ex æquo** dans les `ORDER BY` à 2+ colonnes : c'est une décision de résultat, pas de mécanisme — elle doit être *spécifiée* (une clé de tri totale), pas « abstraite ».
- **Le décodage `neo4rs::BoltType`** : il est déjà confiné à l'adaptateur et ne doit pas migrer — mais il représente ~2 300 sites à réécrire côté PO si obrain a un autre format filaire.
- **La sémantique « no responders » de NATS** : c'est un signal de protocole que peu de bus ont ; il faut soit que obrain l'émule, soit que PO le remplace par une primitive plus pauvre (timeout seul) en acceptant la perte de diagnostic.

### Compte de trous
- **Neo4j** : 3 trous durs (capacités jamais consommées, tri départagé, trajectory hors fabrique) + 2 comportements non nommés (retry transient, cascade).
- **Meilisearch** : 2 trous (défauts typo/ranking implicites, schéma filterable/sortable hors trait) — le périmètre méthode est sinon complet.
- **NATS** : 1 trou majeur (tout le transport de chat request/reply + pub/sub hors trait) = 7 méthodes concrètes non abstraites.
