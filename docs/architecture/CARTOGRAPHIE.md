# Cartographie avant refonte — PO, this-rs, obrain et leurs coutures

Relevé du 10/11/2026, en lecture seule. Aucun build lancé. Chaque ligne porte sa preuve.
But : décider **qui possède quoi** avant d'écrire la première ligne de refactor.

## 1. Les trois systèmes

| Système | Taille | Ce qu'il est | Ce qu'il n'est pas |
|---|---|---|---|
| **PO backend** | 655 000 lignes `.rs` | serveur d'orchestration : REST + MCP + chat + graphe de connaissance | un framework |
| **this-rs** | 112 565 (`src/`) + 57 800 (crates) | framework d'API par entités : routage, auth, événements, 9 backends | un moteur de graphe, ni un orchestrateur |
| **obrain** | 548 000 lignes | base de graphes embarquée + cognitif + substrat durable | un serveur, ni un framework |

## 2. PO — les couches

Transports : `src/api/routes.rs` (3188 l., **483 `.route(`**), `src/mcp/handlers.rs` (**13 483 l.**).
Domaine : `src/chat` **205 451 l.**, `src/runner` 32 155, `src/skills` 26 841, `src/graph` 21 098, `src/orchestrator` 20 440, `src/notes` 14 732, `src/protocol` 12 367, `src/neurons` 6 592, `src/plan` 3 065.
Stockage : `src/neo4j` (57 fichiers), `src/meilisearch`, `src/storage.rs`.
Événements : `src/events` (9 792 l., 17 fichiers). Auth : `src/auth` (6 765 l., 11 fichiers).

**Le chat est le vrai poids** : `src/chat/manager.rs` fait 40 957 lignes, dont **la moitié est du code de test inline** (lignes ~20 410 → 40 957). Production = 20 410 lignes, 1 086 `async fn`, un seul `impl ChatManager`. Sous-ensembles réels : cycle de vie de session, ouverture/spawn d'agent, permissions/annulation, mémoire tenue au resume, livraison d'événements. Sous-dossiers : `provider/` 41 293, `stages/` 10 322, `provider/cognitive/` 10 328, `provider/limits/` 6 804.

**Couture de stockage — le fait qui compte.** `src/neo4j/traits.rs` porte 21 traits et 732 `async fn`. Mais `GraphStore` est un **supertrait** qui combine les 20 autres (`pub trait GraphStore: ProjectStore + … + Send + Sync`), et **`dyn GraphStore` apparaît dans 140 fichiers**. Aucun sous-trait n'est consommé isolément : les 20 autres n'existent que pour deux implémentations (`impl_graph_store.rs` 5 089 l., `mock.rs`). **La découpe en traits de domaine ne sert donc à rien tant qu'aucun consommateur ne migre vers un sous-trait.**

**Coutures techniques.** `cargo xtask storage-boundary` : 57 références hors adaptateurs, **0 non exceptée** (55 dans `tests/`, 2 dans `nats_broker_test.rs`). Émetteurs d'événements : `src/api` (31 fichiers), `src/chat` (10), `src/runner` (4), `src/orchestrator` (2). Auth : vérification concentrée dans `middleware.rs` (`require_auth:31`, `enforce_token_policy:429`) ; JWT, OAuth server, OIDC, jetons d'agent et Google coexistent.

**Surface.** REST groupée : chat 61, code 49, projects 38, admin 30, plans 28, notes 24, protocols 20, personas 20, workspaces 13, tasks 12… MCP : **427 actions sur 32 méga-outils** (`MEGA_TOOL_ACTIONS`, `handlers.rs:220`), dont code 54, admin 43, persona 30, plan 27.

**Noyau cognitif — pur vs I/O.** `src/graph/algorithms.rs` (8 718 l.) : **0 `async fn`, 0 référence au stockage** — pagerank, betweenness, Louvain, similarité, WL hash, tout est pur, donc portable tel quel. `src/homeostasis.rs` : 2 fonctions pures, 1 seule fonction d'écriture (`execute_actions:268`).

## 3. this-rs — ce qu'il impose, ce qu'il donne

**Ce qu'il impose.** `Entity` (`src/core/entity.rs:18`) : `resource_name`, `service_from_host`, `id`, `entity_type`, `created_at/updated_at/deleted_at`, `status`. Les macros (`impl_data_entity!:193`, `impl_link_entity!:416`, `impl_data_entity_validated!:638`) **ne génèrent aucune route** : elles génèrent le modèle et le descripteur. `impl_entity!` et `define_module!` **n'existent pas** dans le dépôt.
Les routes viennent de `DescriptorEntity::build_routes_with` (`server/descriptor_entity.rs:74`) : `GET/POST /{plural}`, `GET/PUT/PATCH/DELETE /{plural}/{id}`, statuts 201/200/204/404, plus les routes de liens (`server/router.rs:71`).
Stockage : `DataService` = `create/get/list/update/delete/search(field,value)` — `search` est un **LIKE textuel**, pas un index. `LinkService` = CRUD de liens + `find_by_source/target`.

**Ce qu'il donne, et son état réel.** REST, GraphQL, gRPC, WebSocket, SSE, EventBus, Event Flows YAML (9 opérateurs), notifications, webhooks, WAMI (STS/RBAC/politiques), multi-tenant, pagination/validation : matures. Backends : obrain 2 487 l., scylladb 1 282, postgres 1 166, mysql 1 033, dynamodb 1 004, in_memory 922, neo4j 843, mongodb 679, lmdb 635.
**Branché et sérieux** : `storage/obrain.rs` (write fencing, `with_obrain_store`), `storage/writer_lease.rs` (1 434 l., **fail-closed**, armé par défaut `builder.rs:1210`), `storage/iam.rs` (2 637 l.), `storage/obrain_identity.rs`, `storage/harness_graph.rs`.
**Dormant / à ne pas surestimer** : les « signaux cognitifs » de `core/events.rs:96` sont un **vocabulaire seul**, non branchés au moteur obrain (`Cargo.toml` : `obrain-cognitive` = alias de `obrain`). `this-platform` est hors membres du workspace côté serveur.

**Recouvrement avec PO, non trivial** : `this-agent` (10 691 l. : Goal/Step/Run/Policy/Decision/Stop), `this-harness` (5 351 l. : boucle d'agent, interrupt, enqueue, ports Transport/Toolbox/Recorder, protocole filaire, liaison MCP), `this-llm-providers` (3 140 l. : streaming, budget). **C'est un second harnais.** PO a le sien (`src/runner`, `src/chat`). Ce doublon n'est pas tranché par la carte.

**Dépendances par chemin hors dépôt** : 6 crates obrain (`Cargo.toml:124-142`), `../../vite-rs` (`:147`), WAMI par git rev `5663db0`.

## 4. obrain — surface, durabilité, manques

Crates : engine 171 334, substrate 132 693, core 78 593, cognitive 63 286, adapters 62 417 (38 plugins d'algos), indexer 16 549, common 15 132, reactive 6 698, façade 1 509.

**Surface.** `execute`/`execute_with_params`, `session()`, transactions (`begin_transaction`, `commit`, `rollback`, savepoints), `open`/`open_read_only`/`open_shared_write`/`close`/`checkpoint`, `create_vector_index`/`vector_search`, `create_text_index`/`text_search`, `hybrid_search`, `create_property_index`/`find_nodes_by_property`.

**Cognitif — ce qui survit à une réouverture** (le point le plus important de cette section) :

| Store | Survit ? | Détail |
|---|---|---|
| EnergyStore | **oui** | colonne u16 Q1.15 du substrat |
| SynapseStore | **partiel** | le poids oui ; `reinforcement_count` repart à 1 et `last_reinforced` à `Instant::now()` |
| ScarStore | **partiel** | seule l'intensité cumulée est écrite (sous-champ 5 bits) ; raison et id des cicatrices **perdus** |
| Consolidation / search / activation | sans objet | purs, calculés à la demande |

Le cognitif n'est **pas câblé par défaut** : `CognitiveEngineBuilder::with_substrate` est derrière `cfg(feature="cognitive")`, absente du défaut.

**Réactif — la découverte.** `MutationBus` (`obrain-reactive/src/bus.rs:36`) est un `tokio::broadcast` de capacité 16 384, en mémoire. **Aucun éditeur de production** : `bus.publish` n'apparaît que dans des benches, `set_post_commit_hook` (`session/mod.rs:576`) n'a aucun appelant. Le bus est créé mais **rien ne le nourrit**.
En revanche le **CDC** existe et est persisté : `CdcLog` dans la zone `substrate.cdc_journal`, API `history`, `history_since`, `changes_between` (`database/admin.rs:959-980`, feature `cdc`). **C'est ça, le journal durable rejouable.**

**Algorithmes** : 38 procédures `CALL` (pagerank, personalized_pagerank, betweenness, closeness, hits, louvain, leiden, label_propagation, dijkstra, astar, bellman_ford, sssp, floyd_warshall, bfs, dfs, composantes, topological_sort, bridges, articulation_points, kcore, subgraphs, clustering_coefficient, triangle_count, similarity, max_flow, kruskal, prim, spectral_embedding, heat_kernel, hilbert_features…). Toutes **à la demande** : `GdsRefreshScheduler` existe (`gds_refresh.rs:69`) mais `database/mod.rs` ne l'instancie pas, et aucun warden ni thinker n'est démarré.

**Manques pour un orchestrateur** : unicité **composite non appliquée** (le contrôle teste chaque propriété séparément : `catalog/mod.rs:1838` `props.iter().any(|p| p == key)`), unicité simple partielle, `text_search` **sans paramètre de filtre** (`search.rs:581`), pas d'API de sauvegarde trouvée au niveau `ObrainDB`, pas de migration de schéma applicatif, écriture multi-processus garantie pour le journal seul (propriétés MP-3, transactions MP-4 en v0.0.2).

## 5. Les coutures — qui possède quoi

| Brique | Aujourd'hui | Cible | Pourquoi (preuve) |
|---|---|---|---|
| **Interface de domaine (732 fn)** | PO, `GraphStore` supertrait | **PO** | les `DataService`/`LinkService` de this-rs ne font que du CRUD + traversée ; `list()` balaie tout l'index uuid→node, `search` est un LIKE, **aucune agrégation** |
| Routage REST (483 routes) | PO, axum | **this-rs** (descripteurs) | routes générées par `DescriptorEntity::build_routes_with` |
| MCP JSON-RPC (427 actions) | PO | **PO, hors framework** | aucun équivalent dans this-rs |
| Métier du chat (205 k l.) | PO | **PO** | c'est le produit |
| Transport du chat (SSE/WS) | PO | **this-rs** | 15 fichiers de streaming |
| Boucle d'agent | PO **et** this-harness | **à trancher** | 5 351 + 10 691 l. qui recoupent `src/runner` |
| Persistance graphe + texte | Neo4j + Meili | **obrain** | |
| Événements infra | NATS | **CDC obrain** + bus this-rs intra-processus | `history_since` est persisté ; MutationBus n'a pas d'éditeur |
| Cognitif (énergie/synapse/scar) | PO en Cypher batch | **obrain-cognitive** | à condition d'accepter les pertes de persistance ci-dessus |
| Algorithmes de graphe | PO pur (8 718 l., 0 I/O) | **PO** (portable) ou obrain (38 plugins) | recouvrement partiel, pas total |
| Auth | PO JWT/OIDC/OAuth | **this-rs WAMI** | 6 765 l. à remplacer, jetons à réémettre |
| Recherche textuelle filtrée | Meilisearch | **manque chez obrain** | `text_search` ne prend aucun filtre |

## 6. Corrections que cette carte impose

1. **MCP = 427 actions**, pas 79. Le chiffre de 79 comptait des occurrences de dispatch, pas la table `MEGA_TOOL_ACTIONS`.
2. **La découpe en traits ne rapporte rien en l'état** : 140 fichiers dépendent du supertrait unique. Découper sans migrer les consommateurs est cosmétique.
3. **MutationBus ne peut pas remplacer le bus local** : aucun éditeur en production. Le journal durable, c'est le **CDC**.
4. **Les stores cognitifs ne persistent pas tout** : énergie oui, poids de synapse oui (compteur perdu), cicatrices en intensité seulement.
5. **Rien n'est planifié dans obrain** : ni GDS, ni warden, ni thinkers. Le « rafraîchissement par seuil de mutations » n'existe pas tant que personne ne le démarre.
6. **L'unicité composite n'est pas appliquée**, alors que PO s'appuie dessus (`anchor_session_target`, `day_plan_key`).
7. **this-rs n'a pas de `impl_entity!` ni de `define_module!`** : les routes viennent des descripteurs, pas des macros.
8. **this-rs porte un second harnais d'agent** (`this-agent` + `this-harness`), à comparer avec `src/runner` avant de bâtir dessus.

## 7. Ce qui reste à vérifier à l'exécution

Durabilité sous `kill -9`, persistance des contraintes à la réouverture, comportement réel de `hybrid_search` sous substrat, coût de la reconstruction paresseuse des index, procédures `CALL` sous substrat, `ObrainDB` partageable entre threads, et l'absence d'éditeur du bus réactif (établie par `grep`, pas par exécution).
