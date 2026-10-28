# Contrats implicites de la couche de stockage

Ce document inventorie ce que le code d'aujourd'hui **exige** de la couche de
stockage, point par point, avec la preuve `fichier:ligne` pour chaque affirmation.
Il ne décrit pas une abstraction idéale : il relève ce qu'un **adaptateur**
(l'implémentation du trait `GraphStore`, `src/neo4j/traits.rs:93`) doit reproduire
pour que le comportement observable ne change pas.

**Méthode.** Corpus = `src/neo4j/*.rs` (l'implémentation de référence
`Neo4jClient` dans `client.rs` + les modules de domaine). Les comptages sont des
`rg` sur ce corpus, reproduits tels quels ; une affirmation non vérifiée est
marquée « non vérifié ».

**Rappel du cadre.** L'abstraction est décrite dans `docs/storage/README.md` : le
manifeste de capacités (`StoreCapabilities`, `src/neo4j/traits.rs:42`) déclare ce
qu'un adaptateur sait faire, et chaque drapeau est prouvé par un commentaire qui
cite le fichier et la ligne (`src/neo4j/client.rs:320-341`). Ce document
complète : il recense les **contrats de comportement** que le manifeste ne couvre
pas.

---

## 1. Tri et départage

**Mesure :** 253 clauses `ORDER BY` dans `src/neo4j/*.rs`. Deux familles
coexistent, et la différence est déterminante pour un paginateur.

### 1.1 Clauses avec critère de départage explicite

Quand l'ordre importe, le code ajoute une clé secondaire (souvent `id` ou
`created_at`) pour départager les ex æquo :

| Requête | Clause | Preuve |
|---|---|---|
| Priorité de tâche | `ORDER BY COALESCE(t.priority, 0) DESC, t.created_at` | `src/neo4j/task.rs:76` |
| Liste des plans | `ORDER BY p.priority DESC, p.created_at DESC` | `src/neo4j/plan.rs:294` |
| Traces synapse (rétention) | `ORDER BY t.created_at DESC, t.id DESC` | `src/neo4j/synapse_shadow.rs:228` |
| Déploiements | `ORDER BY d.started_at DESC, d.id DESC` | `src/neo4j/environment.rs:318` |
| Événements d'exécution | `ORDER BY e.at ASC, e.id ASC` | `src/neo4j/execution_place.rs:79` |
| Jalons | `ORDER BY m.target_date ASC, m.created_at ASC` | `src/neo4j/milestone.rs:135` |

### 1.2 Clauses **sans** critère de départage

Beaucoup de tris reposent sur une clé unique **non garantie unique**, donc l'ordre
entre ex æquo est indéterminé. C'est sans gravité tant que la requête ne pagine
pas ; ça l'est dès qu'elle pagine :

| Requête paginée | Clause | Preuve |
|---|---|---|
| Liste des décisions | `ORDER BY d.decided_at DESC SKIP $offset LIMIT $limit` | `src/neo4j/decision.rs:876` |
| Liste des exécutions de protocole | `ORDER BY r.started_at DESC SKIP $offset LIMIT $limit` | `src/neo4j/protocol.rs:1015` |
| Liste des exécutions de plan | `ORDER BY r.started_at DESC SKIP $offset LIMIT $limit` | `src/neo4j/plan_run.rs:24` |
| Liste des personas | `ORDER BY p.energy DESC SKIP $offset LIMIT $limit` | `src/neo4j/persona.rs:321` |
| Backfill des décisions sans synapse | `ORDER BY d.decided_at SKIP $offset LIMIT $limit` | `src/neo4j/note.rs:2804` |
| Backfill des notes sans synapse | `ORDER BY n.created_at SKIP $offset LIMIT $limit` | `src/neo4j/note.rs:3666` |
| Journal de partage | `ORDER BY se.timestamp DESC SKIP $offset LIMIT $limit` | `src/neo4j/sharing.rs:201` |
| Décisions de routage | `ORDER BY d.at DESC SKIP $offset LIMIT $limit` | `src/neo4j/routing.rs:220` |

`p.energy` (float) et `d.decided_at` (horodatage non garanti unique) sont des
clés où les ex æquo sont fréquents.

**Conséquence pour un paginateur.** Avec `SKIP`/`LIMIT` sur un tri non
déterminé, la même entité peut apparaître sur deux pages, ou être sautée, si
l'ordre change entre deux appels. Le code d'aujourd'hui n'a pas de règle
uniforme : certaines listes paginées ont un départage (`registry.rs:145`
`ORDER BY ps.trust_score DESC, ps.published_at DESC`), d'autres non. Un
adaptateur ne peut pas « choisir » un ordre : il doit reproduire l'ordre **et**
son instabilité pour être conforme, ou bien le tri doit être rendu déterministe
partout (voir « Ce qui reste indécis »).

### 1.3 Le champ de tri est interpolé dans le texte Cypher

Le champ et la direction sont **splicés** dans la requête par `format!` :

- `ORDER BY n.{} {}` / `SKIP {}` / `LIMIT {}` — `src/neo4j/note.rs:403-405`.

Pour rester sûr, le champ vient d'une **liste blanche** et d'une fonction de
repli, jamais de la requête brute :

- `NOTE_SORT_FIELDS` (liste blanche) — `src/neo4j/note.rs:32-43` ;
- `note_order_field` (repli sur `created_at`) — `src/neo4j/note.rs:45-50` ;
- `note_order_dir` (repli sur `DESC`) — `src/neo4j/note.rs:52-57`.

Les autres modules suivent le même motif avec leur propre `match` inline (ex.
`src/neo4j/milestone.rs:518-522`). Il n'y a **pas** de liste blanche partagée :
chaque entité a son propre ensemble de champs triables.

---

## 2. Pagination

**Mesure :** 37 `SKIP`, 199 `LIMIT`, 62 retours nommés `AS total`.

### 2.1 Le motif dominant : `COUNT` séparé, puis `SKIP`/`LIMIT`

La liste et le total sont **deux requêtes distinctes**. Le total n'a pas de
`SKIP`/`LIMIT` :

```
// Count total
MATCH (n:Note) {} RETURN count(n) AS total          ← src/neo4j/note.rs:383-388
// Get notes
MATCH (n:Note) {} RETURN n ORDER BY n.{} {} SKIP {} LIMIT {}   ← src/neo4j/note.rs:398-405
```

Même motif ailleurs :

- Jalons — `src/neo4j/milestone.rs:529-530` (count) et `:549-553` (données) ;
- Releases — `src/neo4j/release.rs:367` (count) et `:382-390` (données) ;
- Sessions de chat — `src/neo4j/chat.rs:271`/`:300`/`:318` (count) et
  `:263`/`:286`/`:311` (données) ;
- Événements de chat — `get_chat_events_paginated` (`SKIP $offset LIMIT $limit`,
  `src/neo4j/chat.rs:1375-1377`) + `count_chat_events` (count séparé,
  `src/neo4j/chat.rs:1396-1400`).

**Pourquoi le total ne doit pas dépendre de `limit`/`offset`.** La requête de
count n'a ni l'un ni l'autre : un adaptateur qui calculerait le total **après**
avoir appliqué la pagination rendrait un total faux (borné par `limit`), et
casserait l'interface `(Vec<T>, usize)` que ces méthodes renvoient.

### 2.2 `SKIP`/`LIMIT` ne sert pas qu'à paginer

Le couple est aussi utilisé comme **garde-fou de volume** pour des suppressions
par lots, où le sens est « traiter au plus N », pas « page N » :

- `WITH r LIMIT {} DELETE r` — `src/neo4j/code.rs:2080` ;
- `SKIP $max_relations` (recalcul des relations transitives) — `src/neo4j/commit.rs:333`.

Un adaptateur doit distinguer ces deux usages : dans le second, `SKIP` signifie
« ignorer les N premières déjà traitées », et l'ordre sous-jacent doit être stable
pour que le découpage en lots ne perde ni ne duplique de lignes.

---

## 3. Entité absente

Il n'y a **pas de politique unique** : la valeur de retour dépend de la méthode.
Sept comportements contrastés, tous constatés dans le code :

| Comportement | Exemple | Preuve |
|---|---|---|
| Suppression → `Ok(false)` si rien n'existait | `delete_document` (`RETURN count(*) AS deleted`, `Ok(deleted > 0)` sinon `Ok(false)`) | `src/neo4j/document.rs:529` |
| Suppression → `Ok(false)` | `delete_feature_graph`, `delete_skill`, `delete_chat_session` | `src/neo4j/feature_graph.rs:229`, `src/neo4j/skill.rs:365`, `src/neo4j/chat.rs:773` |
| Score → `Ok(0.0)` | frustration, densité, pagerank d'un fichier absent | `src/neo4j/task.rs:981`, `src/neo4j/code.rs:4121`, `src/neo4j/code.rs:4142` |
| Comptage → `Ok(0)` | purge de jetons, `max_count` absent | `src/neo4j/user.rs:473`, `src/neo4j/commit.rs:403` |
| Prédicat → `Ok(false)` | `is_mcp_token_active` (`None => Ok(false)`) | `src/neo4j/user.rs:392` |
| Erreur (`bail!`) | précondition manquante : environnement, skill, fichier | `src/neo4j/environment.rs:71`, `src/neo4j/skill.rs:334`, `src/neo4j/code.rs:3531` |
| No-op explicite → `Ok(false)` | `interrupt_task` sur une tâche déjà finie | `src/neo4j/task.rs:423` |

Le contraste le plus net : **supprimer** un objet absent est un succès
(`Ok(false)`), tandis que **créer** un environnement dont le projet n'existe pas
est une erreur (`bail!`, `src/neo4j/environment.rs:71`), et **lire** un score
absent est `Ok(0.0)`. Un adaptateur doit reproduire ce choix **méthode par
méthode** ; il n'existe pas de règle « absent ⇒ erreur » ni « absent ⇒ défaut »
applicable globalement.

---

## 4. `NULL` vs chaîne vide

Deux conventions **coexistent et se contredisent partiellement** dans le code.

### 4.1 Convention « chaîne vide = absent »

Pour les champs optionnels, la convention du dépôt est explicite :

> `//! follow the repo convention of "empty string means absent".`
> — `src/neo4j/environment.rs:10`

En lecture, `""` est reconverti en `None` par `filter(|s| !s.is_empty())` :

- 78 sites de lecture le font — ex. `opt_str` (`src/neo4j/environment.rs:25`),
  et des dizaines de champs inline (`src/neo4j/task.rs:140`,
  `src/neo4j/release.rs:101`, `src/neo4j/protocol.rs:43`…).

### 4.2 Convention « `None` doit rester `NULL`, jamais `''` »

Pour la **portée** d'une note, la règle inverse est écrite noir sur blanc :

> `` `None` must reach the database as a NULL, never as an empty string: an empty
> string is a third state the read path decodes back to `None`, yet every
> predicate that tests `IS NULL` misses it. 3 319 notes were stranded that way.``
> — `src/neo4j/note.rs:16-18`

La fonction `note_project_param` (`src/neo4j/note.rs:19-22`) garantit
`None → None` (donc `NULL` en Cypher), et un test la verrouille
(`a_global_note_writes_a_null_and_not_an_empty_string`,
`src/neo4j/note.rs:4046-4050`).

### 4.3 Les filtres qui confondent les deux

Certains prédicats traitent explicitement `NULL` et `''` comme équivalents :

- `(n.project_id IS NULL OR n.project_id = '')` — filtre `global_only`,
  `src/neo4j/note.rs:83` ;
- `coalesce(n.project_id, '') IN ['', $scope_project]` — `src/neo4j/note.rs:1546` ;
- `... n.project_id IS NULL OR n.project_id = '' OR m.project_id IS NULL OR ...`
  — `src/neo4j/note.rs:3074`.

**Ce qu'un adaptateur doit rendre.** Pour ces champs de portée, `NULL` et `''`
doivent rester **deux états distincts à l'écriture** (la 2e convention), mais
être **indiscernables dans ces prédicats** à la lecture. Rendre systématiquement
l'un en l'autre changerait les filtres `IS NULL` (qui rateraient les `''`) ou les
filtres `= ''` (qui rateraient les `NULL`).

---

## 5. Horodatage

Deux représentations coexistent, avec deux horloges.

**Mesure :** 121 écritures `to_rfc3339()` / `to_rfc3339_opts(...)` (côté Rust,
horloge `chrono`), 138 occurrences de `datetime()` (côté moteur, horloge du
serveur).

### 5.1 Chaîne RFC 3339 écrite côté Rust

La plupart des `created_at` / `updated_at` / `started_at` sont des **chaînes**
produites côté Rust et passées en paramètre :

- `.param("created_at", ...to_rfc3339())` — ex. `src/neo4j/workspace.rs:114`,
  `src/neo4j/release.rs:65`.

La largeur est censée être fixe pour que l'ordre lexicographique soit
chronologique :

> `//! Timestamps are persisted as fixed-width RFC 3339 strings (microsecond
> precision, `Z` suffix) so that lexicographic `ORDER BY` is chronological.`
> — `src/neo4j/environment.rs:8-10`

avec le helper `ts()` (`to_rfc3339_opts(SecondsFormat::Micros, true)`,
`src/neo4j/environment.rs:20-22`).

**Mais la précision n'est pas uniforme** dans le dépôt :

| Précision | Où | Preuve |
|---|---|---|
| Micros | environnements, lieux d'exécution | `src/neo4j/environment.rs:21`, `src/neo4j/execution_place.rs:61` |
| Millis | décisions de routage | `src/neo4j/routing.rs:31` |
| Nanos | ingress, partage | `src/neo4j/ingress.rs:58`, `src/neo4j/sharing.rs:375` |
| Auto (par défaut) | le reste, via `to_rfc3339()` | ex. `src/neo4j/release.rs:65` |

La lecture reconvertit la chaîne : `opt_ts` (`src/neo4j/environment.rs:28`),
ou lit une `String` puis parse (`created_at_str` / `updated_at_str`,
`src/neo4j/mcp_federation.rs:361-363`).

### 5.2 `datetime()` évalué par le moteur

Pour « maintenant » **au moment de l'écriture côté moteur**, le code écrit
`SET x.updated_at = datetime()` :

- `src/neo4j/commit.rs:311`, `src/neo4j/project.rs:262`, `src/neo4j/persona.rs:485`.

### 5.3 Le pont entre les deux

Certaines requêtes enveloppent la chaîne écrite côté Rust dans `datetime(...)`
pour la réinterpréter comme valeur temporelle :

- `SET t.updated_at = datetime($now)` — `src/neo4j/task.rs:399` (où `$now`
  est une chaîne RFC 3339 passée en paramètre).

Cela exige que la chaîne soit **parseable par le moteur** : un adaptateur doit
accepter la chaîne RFC 3339 et la convertir en valeur temporelle équivalente.

### 5.4 Arithmétique de durée

Le moteur calcule des seuils et des délais **dans la requête** :

- `datetime(c.created_at) > datetime() - duration('P7D')` — `src/neo4j/persona.rs:1228` ;
- `t.last_fired + duration({seconds: $cooldown_secs}) <= datetime()` — `src/neo4j/trigger.rs:317` ;
- `datetime(active_at) < datetime() - duration({days: $days})` — `src/neo4j/note.rs:3115` ;
- `duration.inSeconds(datetime(...), datetime()).seconds) / 86400.0 AS idle_days` — `src/neo4j/data_migrations.rs:181`.

Le drapeau de capacité correspondant est `datetime_arithmetic`
(`src/neo4j/traits.rs:61`), prouvé par ces mêmes lignes (`src/neo4j/client.rs:338-340`).

**Deux horloges.** La comparaison entre une chaîne écrite par l'horloge Rust et
un `datetime()` évalué par l'horloge du serveur suppose que les deux horloges
sont proches. Le code ne documente pas de tolérance sur cet écart (voir « Ce qui
reste indécis »).

---

## 6. Entier vs flottant

**Mesure :** lectures typées `get::<i64>` **209**, `get::<f64>` **72** ; casts
`as i64` **205**, `as f64` **97** ; conversions Cypher `toFloat(...)` **41**,
`toInteger(...)` **8**.

### 6.1 Ce qui est entier, ce qui est flottant

- **Entier** : les `count(*)` / `count(...)` / `size(...)`, lus en `i64` — ex.
  `row.get("total")?` (`src/neo4j/note.rs:392`), `row.get::<i64>("cnt")`
  (`src/neo4j/code.rs:395`).
- **Flottant** : scores, énergies, pagerank, densité, cohésion — ex.
  `row.get::<f64>("score")` (`src/neo4j/task.rs:981`), `row.get::<f64>("density")`
  (`src/neo4j/code.rs:4121`).

### 6.2 Pourquoi « tout en double » casse silencieusement

Les lectures sont **typées**, et la plupart sont enveloppées dans un défaut qui
avale l'erreur de désérialisation :

- **145** `get::<i64>(...).unwrap_or(...)` et **39** `get::<f64>(...).unwrap_or(...)`.

Un moteur qui renverrait un `count` en flottant ferait échouer le
`get::<i64>` ; or `row.get::<i64>("total").unwrap_or(0)` (`src/neo4j/decision.rs:866`)
retournerait **0 sans erreur** — un total de page faux, muet. À l'inverse, le
pont entier→flottant est parfois explicite :

- `row.get::<i64>("max_count").unwrap_or(1).max(1) as f64` — `src/neo4j/commit.rs:401`.

Le comportement exact du pilote face à un type inattendu n'est **pas vérifié**
ici ; ce qui est constaté, c'est que le code **compte** sur la distinction et
que l'erreur éventuelle est le plus souvent convertie en défaut (0, 0.0, 1), pas
en panne.

---

## 7. Unicité et idempotence

**Mesure :** 56 `CREATE CONSTRAINT ... IS UNIQUE`, dont **7 composites** ;
282 `MERGE`, 50 `ON CREATE SET`, 23 `ON MATCH SET`.

### 7.1 Contraintes d'unicité

Les 56 contraintes sont déclarées dans `init_schema`
(`src/neo4j/client.rs:439-511` puis `:674-704`). Les **7 composites** :

| Contrainte | Champs | Preuve |
|---|---|---|
| `anchor_session_target` | `(session_id, target_type, target_id)` | `src/neo4j/client.rs:478` |
| `day_plan_key` | `(user_id, date)` | `src/neo4j/client.rs:511` |
| `routing_arm_key` | `(class, provider_id, model)` | `src/neo4j/client.rs:680` |
| `trigger_signal_key` | `(trigger_id, key)` | `src/neo4j/client.rs:688` |
| `external_delivery_key` | `(trigger_id, delivery_key)` | `src/neo4j/client.rs:693` |
| `conversation_segment_key` | `(conversation, id)` | `src/neo4j/client.rs:702` |
| `conversation_boundary_key` | `(conversation, dimension, at_turn)` | `src/neo4j/client.rs:704` |

Le drapeau de capacité est `composite_unique` (`src/neo4j/traits.rs`), prouvé par
le commentaire `src/neo4j/client.rs:327-328` ; les contraintes sont à
`client.rs:478` et `:511`.

### 7.2 `MERGE` avec `ON CREATE SET` / `ON MATCH SET`

Trois formes d'idempotence se distinguent par la **branche** qu'elles remplissent :

- **Compteur incrémental** (le `MERGE` agrège) :
  `ON CREATE SET r.mention_count = 1, ON MATCH SET r.mention_count = r.mention_count + 1`
  — `src/neo4j/chat.rs:1522-1525`.
- **Max-merge** (ne jamais faire décroître un poids) :
  `ON CREATE SET k.weight = $weight, ON MATCH SET k.weight = CASE WHEN k.weight < $weight THEN $weight ELSE k.weight END`
  — `src/neo4j/persona.rs:1724-1725`.
- **Écriture identique sur les deux branches** :
  `ON CREATE SET dp.updated_at = datetime($updated_at) / ON MATCH SET dp.updated_at = datetime($updated_at)`
  — `src/neo4j/day_plan.rs:63-64`.

### 7.3 Le cas de concurrence que l'absence de contrainte a provoqué

Le commentaire le raconte explicitement :

> `// External events (ingress): the constraint is what makes the MERGE of
> claim_external_delivery atomic (without it, 8 concurrent claims of one
> delivery all won: seen on the conformance suite).`
> — `src/neo4j/client.rs:691-693`

La réservation atomique repose sur le `MERGE` + la comparaison de
`claimed_at` avec la valeur qu'on vient d'écrire :

```
MERGE (d:ExternalDelivery {trigger_id: $trigger_id, delivery_key: $delivery_key})
ON CREATE SET d.claimed_at = $now
RETURN d.claimed_at = $now AS created
```
— `src/neo4j/ingress.rs:116-118` ; seul le premier appelant voit `created = true`
(`src/neo4j/ingress.rs:128-129`).

**Contrat fort.** Sans la contrainte composite `(trigger_id, delivery_key)`, le
`MERGE` n'est pas atomique : deux `ExternalDelivery` peuvent être créés et tous
les appelants gagnent. La contrainte **change le comportement**, pas la vitesse.

### 7.4 L'exception délibérée

Tous les champs d'identité ne sont pas uniques : `document.sha256` est
**volontairement non unique** (les mêmes octets peuvent être attachés à deux
projets) — `src/neo4j/client.rs:547-548`.

### 7.5 Auto-vérification

Chaque `CREATE CONSTRAINT` de `client.rs` doit figurer dans une liste de
contrôle relue par le code (pas seulement déclarée) :
`src/neo4j/client.rs:1316-1324`.

---

## 8. Index

**Mesure :** 118 déclarations `CREATE INDEX` / `CREATE VECTOR INDEX` dans
`init_schema` : 100 index de propriété de nœud, 13 index de propriété de
relation, 5 index vectoriels.

### 8.1 Index de propriété de nœud (100)

Couvrent les champs filtrés et triés : ex. `function_name_file_path` (composite,
`src/neo4j/client.rs:522`), `chat_event_session_seq` (`src/neo4j/client.rs:572`),
`note_project` / `note_status` (`src/neo4j/client.rs:539-540`).

### 8.2 Index de propriété de relation (13)

`FOR ()-[r:REL]-() ON (r.prop)` — ex. `touches_file_path`
(`src/neo4j/client.rs:625`), `co_changed_count` (`src/neo4j/client.rs:627`),
`affects_rel_idx` (`src/neo4j/client.rs:633`). Drapeau de capacité :
`relationship_property_index` (`src/neo4j/traits.rs`), prouvé par
`src/neo4j/client.rs:625-633`.

### 8.3 Index vectoriels (5)

`CREATE VECTOR INDEX ... OPTIONS { indexConfig: { `vector.dimensions`: 768,
`vector.similarity_function`: 'cosine' } }` sur `Note`, `File`, `Function`,
`DocumentChunk`, `Decision` — `src/neo4j/client.rs:717-744`.

### 8.4 Ce qui change la vitesse, ce qui change le comportement

| Catégorie | Effet | Preuve |
|---|---|---|
| Index de plage (`chat_session_updated`) | **vitesse seulement** : sans lui, chaque liste trie en mémoire tous les nœuds | `src/neo4j/client.rs:594-597` |
| Composite `(session_id, seq)` | **vitesse** (ORDER BY adossé à l'index), résultat équivalent | `src/neo4j/client.rs:568-573`, `src/neo4j/chat.rs:1365-1373` |
| Index vectoriel | **comportement** : la recherche par similarité (cosinus, 768d) n'existe pas sans lui ; s'il est absent, l'index est sauté « gracieusement » et `vector_search` passe à `false` | `src/neo4j/client.rs:707`, `src/neo4j/client.rs:320` |
| Contraintes d'unicité | **comportement** : c'est elles qui rendent les `MERGE` idempotents/atomiques (§7.3) | `src/neo4j/client.rs:691-693` |

---

## Ce qui reste indécis

Points où le code ne tranche pas, constatés (pas des généralités) :

1. **Largeur d'horodatage non uniforme.** `ts()` promet du micros + `Z` pour
   l'ordre lexicographique (`src/neo4j/environment.rs:8-22`), mais `routing.rs:31`
   écrit du millis, `ingress.rs:58`/`sharing.rs:375` du nanos, et le reste du
   `to_rfc3339()` auto. La promesse « largeur fixe ⇒ tri lexicographique
   chronologique » n'est tenue que par **certains** écrivains ; aucun test ne
   verrouille la largeur commune.

2. **Tris paginés sans départage.** Huit listes paginées trient sur une clé non
   unique sans clé secondaire (`decision.rs:876`, `protocol.rs:1015`,
   `plan_run.rs:24`, `persona.rs:321`, `note.rs:2804`, `note.rs:3666`,
   `sharing.rs:201`, `routing.rs:220`). Aucune décision n'est prise : faut-il
   ajouter un `, id` partout, ou assumer l'ordre instable ?

3. **Deux conventions `NULL`/`''` opposées.** « chaîne vide = absent »
   (`environment.rs:10`, 78 sites de lecture) **vs** « `None` doit rester `NULL` »
   (`note.rs:16-18`, avec 3 319 notes échouées). Aucune règle n'énonce quel
   champ suit quelle convention ; un adaptateur doit le déduire champ par champ.

4. **Lectures typées à défaut silencieux.** 145 `get::<i64>(...).unwrap_or(...)`
   et 39 `get::<f64>(...).unwrap_or(...)` transforment un écart de type en
   `0`/`0.0`/`1` muet. Le comportement du pilote sur type inattendu n'est pas
   documenté, donc on ne sait pas si un adaptateur « tout-double » échoue ou se
   tait.

5. **Deux horloges.** `chrono::Utc::now()` (côté Rust, ~121 écritures de chaîne)
   et `datetime()` (côté moteur, 138 occurrences) coexistent, et des requêtes les
   comparent (`trigger.rs:317`, `persona.rs:1228`). Aucune tolérance d'écart
   d'horloge n'est écrite.

6. **Listes blanches de tri non partagées.** `note.rs:32-43` a une liste blanche
   centralisée ; `milestone.rs:518-522`, `release.rs`, `project.rs`, `task.rs`,
   `plan.rs`, `workspace.rs` ont chacune leur `match` inline. L'ensemble des
   champs triables par entité n'est pas un contrat unique.

---

*Document généré à partir du code de la branche `storage-trait` (commit de base
`955b7228`). Les comptages `rg` sont reproductibles sur `src/neo4j/*.rs`.*
