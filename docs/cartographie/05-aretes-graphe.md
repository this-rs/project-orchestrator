# 05 — Arêtes du graphe : du 0 Ko au graphe vrai

Analyse en lecture seule (aucun build, aucun git). Chemins absolus cités ; étiquettes **[lu]** = lu dans le code, **[doc]** = affirmé par un commentaire/doc, **[déduit]** = conclusion de ma lecture, non rejouée.

## 1. Diagnostic du 0 Ko

`substrate.edges` est la zone mmap des **slots d'arêtes** : 32 o par `EdgeRecord`, slot 0 réservé (« never a real edge »). Constante : `obrain-substrate/src/file.rs:538` (`NodeFile::EDGES = "substrate.edges"`), allocation `store.rs:15558`, écriture par `splice_edge_at_head` dans `create_edge` (`store.rs:22742-22790`). `substrate.props.v2` est, lui, la **chaîne de propriétés** (nœuds ET arêtes) : 164 Ko parce que chaque nœud porte son blob `_data` JSON.

La cause du 0 Ko n'est pas un défaut du substrat : **aucun chemin de code n'appelait `create_edge`**. Le 06/11, `src-tauri/src/nodes.rs` (l'ancêtre de `store.rs`) n'écrivait que des nœuds (`put` → `create_node` + `set_node_property`), et `this-rs/this/src/storage/harness_graph.rs` rangeait runs et étapes comme **nœuds**, sans arête `HAS_STEP`. 0 arête créée ⇒ zone `substrate.edges` jamais allouée ⇒ 0 Ko. C'est exactement ce que dit `GRAPHE.md:5-8`.

## 2. API d'arête native typée (obrain-db)

Trait `GraphStoreMut` — `obrain-core/src/graph/traits.rs` :

- `fn create_edge(&self, src: NodeId, dst: NodeId, edge_type: &str) -> EdgeId` — `traits.rs:1134`
- `fn set_edge_property(&self, id: EdgeId, key: &str, value: Value)` — `traits.rs:1208`
- `fn create_edge_with_props(&self, src: NodeId, dst: NodeId, edge_type: &str, properties: &[(PropertyKey, Value)]) -> EdgeId` — `traits.rs:1326` (défaut : `create_edge` puis `set_edge_property` en boucle)

Niveau `ObrainDB` — `obrain-engine/src/database/crud.rs` :

- `pub fn create_edge(&self, src: NodeId, dst: NodeId, edge_type: &str) -> EdgeId` — `crud.rs:750` (WAL/CDC)
- `pub fn create_edge_with_props(&self, src, dst, edge_type, properties: impl IntoIterator<Item = (impl Into<PropertyKey>, impl Into<Value>)>) -> EdgeId` — `crud.rs:851`
- `pub fn set_edge_property(&self, id: EdgeId, key: &str, value: Value)` — `crud.rs:970`

**Unicité `(from, type, to)` : NON garantie par le substrat.** `SubstrateStore::create_edge` (`store.rs:22742`) alloue un slot neuf sans déduplication (`allocate_edge_id` + `splice_edge_at_head`), et autorise même des extrémités absentes (« dangling », commentaire `store.rs:22743-22745`). L'unicité est donc appliquée **à la couche applicative**, par un check-then-create :

- atelier `Store::link` : `arete(a, kind, b)` cherche l'arête existante avant `create_edge` — `atelier/src-tauri/src/store.rs:572` (`arete`) et `:581-602` (`link`) ; idem `batch_links` `:610-651`.
- `HarnessGraph::link_step` : même patron sur `HAS_STEP` — `this-rs/this/src/storage/harness_graph.rs:251-258`.

Sans verrou autour de `arete`→`create_edge`, deux sessions concurrentes peuvent insérer le même triplet **[déduit]** (même famille que la course d'unicité notée en 02 §1c).

## 3. L'arête « opportuniste » du framework (`this-rs/this/src/storage/obrain.rs`)

`ObrainLinkService` stocke les liens d'entités **comme nœuds** de label `_link:{type}` (propriétés `_source_id`, `_target_id`, `_link_type`, `_data` JSON) — `obrain.rs:901-915` (docstring) et `:975-995` (`create`). L'arête native n'est qu'un **index secondaire** : `try_create_edge` (`obrain.rs:947-960`) n'appelle `session.create_edge(src, tgt, &link.link_type)` (3 arguments, **sans propriétés, sans dédoublonnage**) que si **les deux extrémités existent déjà comme nœuds d'entité**. Ce que ça rate : (a) l'arête n'est pas la vérité (le nœud `_link:*` l'est) ; (b) pas de propriétés d'arête (`_name`, `_target`, `_index`) ; (c) pas de garantie `(from,type,to)` unique.

## 4. Le plus petit pas qui rend le graphe vrai

Le côté **écriture** existe désormais (ajouté les 07-08/11, après la mesure du 06/11) ; il faut le **brancher sur des chemins réellement appelés** et le **vérifier sur l'appareil** :

- **Indexeur de modules** : `ingest.rs:345` (`store.batch_links(&liens)`) pose `HAS_DOCUMENT`/`HAS_CHUNK` ; `vfs_index.rs:194` (`store.link(de, type_arete, vers, &[])`) pose la hiérarchie de fichiers. Ces deux-là écrivent des arêtes dès l'installation/indexage d'un module.
- **Journal** : `harness_graph.rs:251` pose `run -HAS_STEP{_index}-> step` via `record_run`/`append_step` (`:358-412`), appelé par `graph.rs:128-139`.
- **Coquille** : les commandes Tauri `graph_link`/`graph_status`/… sont enregistrées `lib.rs:987-1013` et lisent/écrivent via `store.rs:1977` (`graph_link`), `:1929` (`graph_status`, rend `counts()` → `g.edge_count()`, `store.rs:1620-1622`).

Le pas minimal restant = **build + déploiement + re-mesure sur l'appareil** : `graph_status().edges > 0` et `substrate.edges > 0` (item 1 de `GRAPHE.md:99`). Avant cela, les arêtes n'existent qu'en test. La reliaison des lignes (`REFERENCES` via `row_create`/`row_patch`, `store.rs:1702-1731`) et des notes (`ABOUT` via `remember`) passe par les mêmes `link`/`batch_links`.

## 5. Sert-il le Project Orchestrator ?

**Oui, exactement.** Les relations de PO sont des arêtes typées `(from, type, to)` : `graph/models.rs:2040-2052` (`CodeEdgeType::Imports→"IMPORTS"`, `CoChanged→"CO_CHANGED"`, `Affects→"AFFECTS"`, `Synapse→"SYNAPSE"`) et `neo4j/beliefs.rs:59` (`LINK_EDGES = ["CALLS","IMPLEMENTS_TRAIT","DEPENDS_ON","IMPORTS"]`). `create_edge_with_props` + unicité applicative `(from,type,to)` est précisément le substrat d'une synapse (une arête par paire, poids en propriété) et des relations du tissu de connaissances. Réserve : PO est aujourd'hui sur **Neo4j** (`backend/src/neo4j/…`) ; cette cartographie évalue la migration vers obrain, pas l'état courant.

## 6. Ce que dit `docs/LIMITES-v0.0.1.md` des arêtes

Rien de spécifique aux arêtes natives — elles ne figurent dans **aucune** ligne de limite (donc : chemin de premier ordre). Les limites adjacentes qui les touchent par transitivité : **« propriétés et tas » non garantis en écriture multi-processus** (`LIMITES-v0.0.1.md:17`) — les propriétés d'arête partagent la même page de propriétés — et la couche simpliciale « hors du chemin nœuds/arêtes/propriétés » (`:21`).
