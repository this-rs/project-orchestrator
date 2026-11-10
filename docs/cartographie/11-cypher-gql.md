# 11 — Les requêtes Cypher de PO face à GQL / Cypher d'obrain

Analyse en lecture seule (aucun build, aucun git), 10/11/2026. Sources : `backend/src/neo4j/` (57 fichiers dont `mock.rs`, exclu des comptes) ; `obrain-db/docs/gql/*` ; `obrain-db/crates/obrain-adapters/src/query/{gql,cypher}/` ; `obrain-engine`, `obrain-core`.

## 0. Trois constats qui changent la question

1. **Obrain n'a pas seulement GQL : il a aussi un front-end Cypher (openCypher 9.0)**, derrière la feature `cypher` (`obrain-engine/Cargo.toml:93`, `obrain-adapters/Cargo.toml:69`). Il parse `MERGE … ON CREATE/ON MATCH`, `UNWIND`, `CALL { }`, `FOREACH`, `EXISTS { }`, `COUNT { }`, compréhension de liste et de motif, `reduce`, projection de map, `shortestPath` (`cypher/ast.rs:55-91, 521-575`), le DDL `CREATE INDEX/CONSTRAINT` et `SHOW` (`cypher/parser.rs:83-111`). Entrée : `Session::execute_cypher` (`obrain-engine/src/session/mod.rs:2854`) et `execute_language("cypher", q, Some(params))` (`session/mod.rs:3407`, paramètres `HashMap<String, Value>`). Le parseur GQL accepte aussi `CREATE`, `MERGE … ON CREATE`, `UNWIND`, `DETACH` (`gql/parser.rs:366-378, 1367-1379`). « Traduire en GQL » n'est donc pas obligatoire : la question est sémantique, pas syntaxique.
2. **Réserve forte** : ce front-end n'est « pas couvert par la CI par défaut » (`obrain-db/docs/LIMITES-v0.0.1.md:51`, job `features` = compilation seule). Je n'ai rien exécuté : tout ce qui suit sur le comportement est **lu**, pas mesuré.
3. **Le point d'étranglement n'est pas le texte Cypher mais le décodage** des résultats : dans `neo4j/` on compte 243 `neo4rs::Node`, 862 `.get::<`, 1 123 `row.get`, 99 `BoltType`, et 763 appels `self.graph.` (483 `execute`, 272 `run`). Un second moteur doit rendre une forme `Row`/`Node` compatible, ou ces ~2 300 sites changent.

## 1. Méthode et décompte

Extraction par script (hors dépôt) de tous les littéraux de chaîne (`"…"`, `r#"…"#`) hors modules `#[cfg(test)]` commençant par une clause Cypher : **1 300 littéraux** = 196 DDL (`CREATE CONSTRAINT/INDEX`) + 1 104 littéraux de données (1 093 distincts ; 1 077 contiennent une clause terminale RETURN/SET/DELETE/MERGE/CREATE/UNWIND/CALL). 152 littéraux contiennent des trous `{}`/`{{…}}` (`format!`, 151 appels `format!` Cypher) : un même « requête logique » peut y apparaître en plusieurs fragments, et `batch.rs:130` est un faux positif (commentaire de doc). **L'écart avec « ~545 » est réel et non résolu** : 545 ne correspond à aucun de mes comptes (appels `query(` ≈ 1 000, littéraux 1 104). Le classement est une heuristique de motifs lexicaux (marge estimée ±5 % entre T et A), pas un parsing.

Classes (littéraux, hors DDL) — T = triviale, A = adaptation / à tester, R = réécriture, I = impossible tel quel :

| Classe | Littéraux | Fichiers | Définition utilisée |
|---|---:|---:|---|
| **T** triviale | **661** | 48 | MATCH/WHERE/RETURN/OPTIONAL/WITH/ORDER/LIMIT, `count/collect/coalesce/CASE`, `DETACH DELETE`, `REMOVE`, `type()/labels()`, `=~`, var-length, UNION, projections |
| **A** adaptation | **389** | 47 | `MERGE` (± `ON CREATE/ON MATCH`) 106+, `datetime(` seul 53, `MERGE`+`datetime` 36, `EXISTS { }` 22+, `UNWIND` lot 16+, `toString(datetime)` 10, `FOREACH` 12, pattern-comprehension/`shortestPath` 10, `reduce`, compteurs |
| **R** réécriture | **51** | 17 | `db.index.vector.queryNodes` 15, `CALL { }` 14, `duration.between/inSeconds/({…})` 11, `db.create.setNodeVectorProperty`/`db.labels`/`db.relationshipTypes`/`SHOW` 8, `apoc.coll.*` 3 |
| **I** impossible tel quel | **3** | 3 | `apoc.meta.schema` (`graph_schema.rs:54`), `apoc.merge.relationship` (`path_migration.rs:187`), `apoc.do.when` (`data_migrations.rs:392`) |
| DDL simple | 167 | 2 (`client.rs` 164, `trajectory_store.rs` 3) | `CREATE INDEX … FOR (n:L) ON (n.p)`, contrainte `IS UNIQUE` mono-propriété |
| DDL dur | 29 | 2 (`client.rs` 25, `trajectory_store.rs` 4) | 7 contraintes composites, 14 index de relation, 8 index vectoriels |

Total traité : 1 300. Fichiers par classe (nombre de littéraux) :

- **T** : chat 59, code 56, analytics 52, note 42, persona 42, workspace 39, protocol 37, project 29, task 24, plan 23, feature_graph 17, decision 15, user 15, skill 14, trigger 14, path_migration 13, commit 11, document 10, environment 10, milestone 9, release 9, data_migrations 7, lifecycle_hook 7, mcp_federation 7, plan_run 7, profile 7, trajectory_store 7, alert 6, constraint 6, event_trigger 6, routing 6, topology 6, step 5, + 15 fichiers de 1 à 4.
- **A** : code 54, chat 43, note 27, analytics 25, persona 22, workspace 18, feature_graph 15, data_migrations 14, protocol 12, task 12, trajectory_store 11, commit 10, skill 10, mcp_federation 9, decision 8, path_migration 8, plan 8, project 8, document 7, milestone 6, routing 5, synapse_shadow 5, trigger 5, + 24 fichiers de 1 à 4.
- **R** : note 18, code 10, decision 3, document 3, data_migrations 2, graph_schema 2, persona 2, trigger 2, + analytics, chat, client, graph_sampler, mcp_federation, neighborhood, path_migration, synapse_shadow, trajectory_store (1 chacun).
- **I** : data_migrations, graph_schema, path_migration.
- **DDL** : tout dans `client.rs:448-790` et `trajectory_store.rs:44-64`.

Échantillon lu réellement : 72 requêtes tirées au hasard par classe (22 T, 28 A, 14 R, 8 « I » avant reclassement) + ~45 lues ciblées (voir §5) = **> 115 requêtes lues**. Résultat de la lecture : j'ai **rétrogradé** `db.create.setNodeVectorProperty` et `db.labels()` de « impossible » à « réécriture » (un `SET`/`SHOW` suffit), et **promu** pattern-comprehension et `shortestPath` de R à A (obrain les supporte : `gql/ast.rs:1271`, `conformance.md` « shortestPath / allShortestPaths — Supported »).

## 2. Ce qui est supporté côté obrain (lu)

`docs/gql/conformance.md` : MATCH/OPTIONAL MATCH, ORDER BY (+ `NULLS FIRST/LAST`, `basic-queries.md:87-91`), SKIP/LIMIT, `MERGE … ON CREATE/ON MATCH`, `DETACH DELETE`, `REMOVE`, `SET n += {map}` (`mutations.md`), `UNWIND`, `CALL { }`, `UNION`, `EXISTS { }`, `COUNT { }`, `collect`, `count(DISTINCT)`, `avg/sum/min/max`, `percentile/stdev`, `START TRANSACTION/COMMIT/ROLLBACK/SAVEPOINT`. Fonctions (`obrain-core/src/execution/operators/filter.rs:2802-4300`) : `id, elementId, labels, type, size, coalesce, toString/Integer/Float/Boolean, head, tail, last, reverse, keys, properties, trim…, replace, substring, split, toUpper/toLower, range, left/right, abs/ceil/floor/round/sqrt/log/exp, date/time/datetime/duration, now/timestamp, year…second, date_trunc, nodes, relationships, startNode, endNode, nullIf, cosine_similarity, euclidean_distance, dot_product`. `datetime()` sans argument rend l'instant courant (`filter.rs:4027-4029`) ; `datetime(str)` parse ISO ; `Timestamp − Timestamp = Duration` et `Timestamp ± Duration` existent (`filter.rs:2455-2498`).

**Absent (grep sur les 5 crates)** : tout `apoc.*`, `duration.between/inSeconds/inMilliseconds` (espace de noms), `duration({hours: n})` en forme map (le constructeur lit une chaîne, `filter.rs:4073-4100`), `db.index.vector.queryNodes`, `db.create.setNodeVectorProperty`, `db.labels`, `db.relationshipTypes`.

## 3. Dépendances cachées au-delà du langage

| Dépendance | Combien / où | Ce qui casse | Que faire |
|---|---|---|---|
| **Contraintes composites** `REQUIRE (a,b[,c]) IS UNIQUE` | 7 contraintes : `anchor_session_target` (`client.rs:492`), `day_plan_key` (`:530`), `routing_arm_key` (`:720`), `trigger_signal_key` (`:728`), `external_delivery_key` (`:733`), `conversation_segment_key` (`:742`), `conversation_boundary_key` (`:744`) | Le contrôle d'obrain teste chaque propriété séparément : `TypeConstraint::Unique(props) => props.iter().any(\|p\| p == key)` (`obrain-engine/src/catalog/mod.rs:1838`, idem `PrimaryKey`). Une unicité `(conversation, id)` devient « `conversation` seul unique » : **le 2ᵉ segment d'une même conversation est refusé à tort**. Et un composite jamais appliqué laisse passer de vrais doublons (le `MERGE` sur clé composite — `LlmSetting {scope,key}` `chat.rs:629`, `RoutingArm` `routing.rs:108`, `DayPlan` `day_plan.rs:63`, `TouchedNode` `trajectory_store.rs:117` — ne protège pas d'une course entre deux écrivains) | Ne **pas** poser ces 7 contraintes telles quelles. Soit clé synthétique concaténée (`key = a\|b\|c`, contrainte mono-propriété), soit corriger le moteur (contrainte n-aire). Le `MERGE` reste, il porte l'unicité logique dans un seul processus. |
| **Index de relation** `FOR ()-[r:T]-() ON (r.p)` | 14 : `client.rs:665-705` (12), `trajectory_store.rs:62,64` | Le DDL Cypher d'obrain n'accepte que `FOR (n:Label)` (`cypher/parser.rs:2031-2058` : exige un identifiant puis `:Label` entre parenthèses) → erreur de syntaxe | Supprimer (obrain n'indexe pas les arêtes par propriété selon la doc lue) ; vérifier par `EXPLAIN` que les requêtes concernées (`co_changed_project`, `touches_file_path`…) restent acceptables. |
| **Index vectoriels** `CREATE VECTOR INDEX name … OPTIONS {indexConfig: …}` | 8 DDL (`client.rs:750-787`, `trajectory_store.rs:44,47`), 15 requêtes `queryNodes`, 8 `setNodeVectorProperty` | Le front-end Cypher ne parse que `INDEX`/`CONSTRAINT` après `CREATE` (`parser.rs:2017-2027`) ; GQL : `CREATE VECTOR INDEX FOR (d:Doc) ON (d.embedding)` sans nom ni options (`schema.md:228`). Pas de procédure `queryNodes`. Sémantique PO : le filtre projet/statut est appliqué **après** l'index (`StoreCapabilities.vector_filter = false`, `client.rs` ~l.330) | Réécrire en 2 temps derrière un petit adaptateur `vector_search(index, k, emb)` ; remplacer `setNodeVectorProperty` par `SET n.embedding = $v` ; les 15 requêtes `queryNodes` sont toutes de la forme `CALL … YIELD node, score RETURN … ORDER BY score DESC` → un seul gabarit. |
| **`datetime`** | 162 littéraux `datetime(`, 19 `toString(`, 22 stockages `created_at: datetime($created_at)` dans `CREATE`, 8 `datetime() - duration(…)` | (a) `duration.inSeconds/between/inMilliseconds` n'existent pas (11 requêtes, ex. `note.rs:2964`, `mcp_federation.rs:473`) ; (b) `duration({hours: $n})` (map) non supporté (`trigger.rs:311`, `persona.rs:1227`) ; (c) `toString(datetime)` : Neo4j rend de l'ISO-8601 jusqu'à 9 décimales (`sharing.rs:376`), obrain rend autre chose — jamais comparé ; (d) **lecture** : `note.rs:4111` lit `created_at` en `String`, alors que `note.rs:184` l'écrit via `datetime()` : un des deux chemins convertit, **non vérifié** | Remplacer `duration.inSeconds(a,b).seconds` par `(datetime(b) - datetime(a))` + extraction (à valider : accesseur de secondes sur `Duration` inconnu) ; passer les durées en chaîne ISO (`duration('PT48H')` fonctionne). Stocker les horodatages en chaîne RFC 3339 côté obrain (plus de typage temporel dans le décodage). |
| **`DETACH DELETE`** | 72 littéraux | Supporté (`conformance.md`). Risque : `MATCH … DELETE r SET x = null` (`plan.rs:515`, DELETE puis SET), `DELETE … RETURN count(r)` (30 requêtes), suppression par lots `WITH n LIMIT k DELETE n RETURN count(n)` (29, `code.rs:2349`) | Ordre des clauses et valeur du `count` après `DELETE` : **non lus**, à couvrir par la suite dorée. |
| **Transactions explicites** | `chat.rs:3424`, `trajectory_store.rs:153,197,336` (`start_txn`/`commit`/`rollback`, 24 sites) ; `StoreCapabilities.explicit_transactions` | `START TRANSACTION`/`COMMIT`/`ROLLBACK` existent côté GQL. Mais : « Pas d'isolation ni de conflit d'écriture entre processus » (`LIMITES-v0.0.1.md:18-19`) ; écritures API directe hors transaction (`:26`) | Garder les transactions mono-processus ; interdire un second processus écrivain. |
| **Verrou par écriture** | `trigger.rs:311-323` : `SET t.reservation_lock = true … REMOVE t.reservation_lock` sert de verrou d'exclusion (Neo4j verrouille le nœud jusqu'à la fin de la transaction) | Obrain n'offre pas ce verrou cross-processus ; dans un seul processus, la détection write-write est à vérifier | Remplacer par un verrou applicatif (mutex Rust) ou une condition de garde `WHERE t.reserved_at IS NULL`. |
| **Types de résultat** | §0, point 3 | `neo4rs::Node/Row/BoltType` omniprésents | Une couche `Row`/`Node` de compatibilité, ou une impl `GraphStore` qui décode sans neo4rs (voir §4). |
| **Fragments dynamiques** | 152 littéraux avec `{}` / `{{…}}` / `{edge}` / `{label}` (ex. `chat.rs:1172,1217`, `note.rs:1566`, `feature_graph.rs:355`) | Un type de relation ou un libellé interpolé : le texte final n'existe qu'à l'exécution ; impossible de tout valider statiquement | Lister les ensembles de valeurs possibles et les ajouter à la suite de conformité. |
| **`SHOW CONSTRAINTS YIELD name`** | `client.rs:903` (1) | Le front-end Cypher parse `SHOW` (`parser.rs:108-111`) ; la forme `YIELD name RETURN collect(name)` n'est pas lue | Test dédié. |

## 4. Stratégie : traduire sur place ou assainir d'abord ?

Ce qui existe déjà : **la couture est faite**. `GraphStore` (`traits.rs:93`, 732 `async fn`) ; `impl GraphStore for Neo4jClient` (`impl_graph_store.rs:22`, 725) ; `MockGraphStore` en mémoire (`mock.rs:980`, 874 `async fn`, ~20 000 lignes) ; `StoreCapabilities` (`traits.rs:42`) qui déclare déjà `composite_unique`, `relationship_property_index`, `subqueries`, `datetime_arithmetic`, `vector_filter`, `explicit_transactions` (`client.rs` ~l.320-350). Autrement dit : **« rassembler les requêtes dans un adaptateur unique » est déjà vrai au niveau du trait** ; ce qui manque est un second implémenteur et un moyen de comparer.

| Option | Éléments touchés | Reste comparable à Neo4j ? | Verdict |
|---|---:|---|---|
| **1. Traduire sur place** (GQL ou Cypher d'obrain, dans les 49 fichiers) | 389 A + 51 R + 3 I = **443** requêtes + **196** DDL = **639** énoncés, + les ~2 300 sites de décodage si le format change | **Non** : on remplace le texte, plus d'oracle Neo4j | Le plus risqué : la moitié des énoncés change, sans témoin. |
| **2. Assainir d'abord** (extraire les 1 104 littéraux + 151 `format!` vers un catalogue paramétré) | **1 104** littéraux + 763 appels `self.graph.` | Oui | Le plus coûteux (100 % des énoncés) pour un gain nul sur les 7 % réellement divergents. La couture de trait rend ce chantier inutile. |
| **3. Recommandée : passer le texte tel quel par le front-end Cypher d'obrain + variantes ciblées + suite dorée** | **54** requêtes (51 R + 3 I) + **29** DDL durs = **83 énoncés (≈ 13 % de 639)** à écrire en double ; les 661 T et 389 A passent d'abord inchangés ; DDL simple (167) généré par une boucle ; 1 couche `Row`/`Node` | Oui, **jusqu'à la suppression de Neo4j** : la suite dorée tourne sur les deux moteurs | Moins cher et réversible. |

Condition de l'option 3 (à lever **d'abord**) : compiler obrain avec `--features cypher` et rejouer la suite dorée (§5). Si le front-end Cypher échoue sur > 15 % des 30 requêtes, basculer en GQL pour la classe A, mais seulement pour les énoncés qui échouent, jamais en bloc. Chiffres : 83 énoncés au lieu de 639 (÷ 7,7) ou 1 104 (÷ 13) ; aucune mesure de temps, je n'en ai pas.

Ordre : (1) feature `cypher` + les 30 requêtes ; (2) corriger les contraintes composites (clé synthétique) avant toute donnée ; (3) vecteur et durées ; (4) couche `Row`/`Node` ; (5) `apoc.*` réécrits en Rust (boucle sur `type(r)` pour `apoc.merge.relationship`, 1 requête par type de relation parmi les 5 de `KNOWLEDGE_EDGES`, `data_migrations.rs:392` `path_migration.rs:187`) ; (6) `graph_schema.rs:29-54` remplacé par `SHOW`/`labels()` ou une table statique.

## 5. Jeu doré : 30 requêtes (+5 DDL/transaction)

Choisies pour la fréquence d'usage (chemins chauds : synchro, chat, notes, tâches) puis pour la dureté. Chacune tourne sur Neo4j (référence) et obrain, mêmes paramètres, comparaison des lignes (ordre inclus quand `ORDER BY`), des compteurs, et de l'état du graphe après écriture.

| # | Requête | Pourquoi |
|--:|---|---|
| 1 | `note.rs:184` création `Note` (`datetime($created_at)`) | écriture la plus fréquente ; type temporel |
| 2 | `project.rs:22` création `Project` | ~20 propriétés dont `datetime()` |
| 3 | `step.rs:140` `SET s.status…, datetime($now)` | mise à jour la plus courante |
| 4 | `task.rs:343` prochaines tâches + `collect(blocker)` | `get_next` |
| 5 | `task.rs:253` contexte de tâche : 5 `OPTIONAL MATCH` | produit cartésien |
| 6 | `task.rs:428` `SET` conditionnel `coalesce(...)` | compteurs/valeurs par défaut |
| 7 | `code.rs:307` `UNWIND` d'adoption de fichier par `uid` | synchro, MERGE conditionnel |
| 8 | `code.rs:1335` `UNWIND $items … MERGE (f:Function {id})` | synchro, lot 10 000 (`batch.rs:42`) |
| 9 | `code.rs:2054` `UNWIND … CALL { … LIMIT 1 MERGE CALLS }` | sous-requête corrélée + MERGE |
| 10 | `code.rs:1881` `UNWIND … CALL { … IMPORTS_SYMBOL }` | idem, sans `RETURN` |
| 11 | `code.rs:2349` suppression par lots `NOT (n)-[]-() … LIMIT … DELETE … RETURN count` | prédicat de motif + compteur après DELETE |
| 12 | `chat.rs:2162` ajout d'événement `CREATE (s)-[:HAS_EVENT]->(e:ChatEvent…)` | écriture la plus chaude du chat |
| 13 | `chat.rs:2193` relecture `WHERE e.seq > $after_seq ORDER BY e.seq LIMIT` | tri + pagination |
| 14 | `chat.rs:629` `MERGE (s:LlmSetting {scope,key})` | clé composite |
| 15 | `chat.rs:2794` `toString(cs.created_at)` | format ISO du temps |
| 16 | `chat.rs:912` `queryNodes('session_episode_embeddings') … (e)-[:EPISODE_OF]->()` | vecteur + filtre de motif |
| 17 | `note.rs:2583` `queryNodes('file_embedding_index')` | vecteur, gabarit des 15 |
| 18 | `note.rs:2511` `CALL db.create.setNodeVectorProperty` | écriture vecteur |
| 19 | `note.rs:2964` `duration.inSeconds(datetime(...), datetime())` | arithmétique datetime |
| 20 | `note.rs:2681` `UNWIND range(0,size($ids)-1) … $ids[i] … MERGE ON CREATE/ON MATCH` ×2 | synapses |
| 21 | `trigger.rs:311` verrou `reservation_lock` + `datetime() - duration({hours})` | verrou + durée map |
| 22 | `day_plan.rs:63` `MERGE (dp:DayPlan {user_id,date})` + `DELETE` + `UNWIND` + `CREATE` | composite + 4 clauses |
| 23 | `routing.rs:108` `MERGE (a:RoutingArm {class,provider_id,model}) ON CREATE SET …` | bandit : compteurs, clé triple |
| 24 | `environment.rs:43` `WHERE NOT EXISTS { … } CREATE …` | unicité par requête |
| 25 | `neighborhood.rs:130` `UNWIND … CALL { … ORDER BY … LIMIT }` + `elementId` | le plus dur des `CALL` |
| 26 | `graph_sampler.rs:111` `CALL { … UNION … }` | UNION dans sous-requête |
| 27 | `graph_schema.rs:252` `shortestPath` + compréhensions de liste | chemin + `nodes(p)` |
| 28 | `protocol.rs:1181` `OPTIONAL MATCH path=…[:CHILD_OF*1..5]` + `UNWIND ([root]+descendants)` | longueur variable |
| 29 | `path_migration.rs:187` `apoc.merge.relationship(k, type(r), {}, properties(r), c)` (avec `data_migrations.rs:392` et `graph_schema.rs:54` pour les deux autres `apoc`) | le cas « impossible » |
| 30 | `project.rs:429,439,449,529` cascade `DETACH DELETE` (4 requêtes en séquence) | suppression en cascade |

Réserve (à ajouter si place) : `analytics.rs:3165` (`reduce` pour l'écart-type), `persona.rs:1227` (`datetime() - duration('P…')`), `topology.rs:187` (`=~`), `trajectory_store.rs:117` (`MERGE` sur `TouchedNode` composite), `commit.rs:19` (`MERGE (c:Commit {hash})`).

**+5 énoncés de schéma et de transaction** (hors des 30) : `client.rs:492` (contrainte composite → doit échouer proprement ou être convertie), `client.rs:665` (index de relation), `client.rs:750` (index vectoriel), `client.rs:903` (`SHOW CONSTRAINTS`), `trajectory_store.rs:153` (transaction explicite avec rollback).

Règles de la suite : (a) mêmes paramètres Rust → les deux moteurs ; (b) ordre déterministe imposé par un `ORDER BY` complet, sinon comparaison en ensemble ; (c) chaque écriture est suivie d'une lecture de l'état complet des nœuds touchés ; (d) un test par valeur d'interpolation pour les requêtes à trous (`chat.rs:1172,1217`, `feature_graph.rs:355`) ; (e) rouge sans, vert avec : chaque réécriture est d'abord prouvée **rouge** sur l'ancien texte.

## 6. Limites de cette analyse

- Comptes issus d'une extraction lexicale ; 152 littéraux à trous et ≈ 27 littéraux sans clause terminale faussent le total d'environ 3 %.
- Aucune requête exécutée : tous les « supporté » viennent de la documentation et du code lus, le front-end Cypher d'obrain n'est pas couvert par la CI par défaut (`LIMITES-v0.0.1.md:51`).
- Non vérifiés : l'accesseur de secondes d'une `Duration` obrain ; le rendu de `toString(Timestamp)` ; l'ordre des clauses `DELETE … SET` ; le tri des valeurs nulles par défaut (`ORDER BY … DESC` 131 requêtes, dont beaucoup sur colonnes nullables) ; la lecture de `created_at` en `String` (`note.rs:4111`) face à l'écriture `datetime()` (`note.rs:184`).
- « ~545 requêtes » : non reproduit (§1).
