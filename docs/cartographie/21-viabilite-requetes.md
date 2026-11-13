# Viabilité des requêtes : ce qu'obrain sait faire, mesuré sur un graphe réel

Mesuré le 11/11/2026 sur l'archive `~/.obrain/db/po` (**1 227 051 nœuds, 1 889 164 arêtes**,
schéma réel de PO : `ChatEvent`, `Function`, `Import`, `Note`, `Task`, `Decision`…), en
**lecture seule** (`obrain query --read-only`). Aucune écriture, aucune base vivante touchée.

Cette archive date de juillet (**non vérifié** : plusieurs de ses fichiers portent une date de
modification du 11/11/2026 17:24, postérieure à la mesure ; 1,23 M nœuds contre 3,85 M aujourd'hui) : elle ne sert pas de
cible de migration, mais elle a **la forme** du graphe de PO, ce qui suffit à trancher la
question « telle requête passe-t-elle ? ».

## Ce qui passe

| Forme | Résultat |
|---|---|
| `MATCH (n) RETURN count(n)` | 1 227 051 |
| label + propriété (`MATCH (n:Note)`) | 3 357 |
| `labels(n)`, `size(labels(n))` | ok |
| `DISTINCT` + `ORDER BY` + `LIMIT` | ok |
| `ORDER BY coalesce(a, 0) DESC, b DESC` | ok |
| `RETURN <expr> AS a, count(*) AS c` | ok |
| `WITH … ORDER BY … LIMIT` | ok |
| `WITH collect(x) AS ids` **sans clé de groupe** | ok |
| `OPTIONAL MATCH`, `WHERE … IS NOT NULL` | ok |
| chemin de longueur variable `-[:DEPENDS_ON*1..3]->` | ok |
| `shortestPath` | ok (rend `null` s'il n'y a pas de chemin, sans erreur) |
| `COUNT { (n)--() }`, `EXISTS { (n)--() }` | ok |
| `CASE WHEN`, `STARTS WITH`, `CONTAINS`, `IN [...]` | ok |
| `toLower`, `toUpper`, `split` | ok |
| `datetime()`, `datetime('…')`, `datetime() + duration({days: 3})` | ok |
| `duration('P1D')` | ok |
| `UNWIND` + `collect` | ok |

## Ce qui échoue, et ce que ça coûte

### 1. Agrégat dans `WITH` : l'alias de groupement devient invisible — **104 requêtes**

```cypher
MATCH (n:Note) WITH n.project_id AS p, count(*) AS c RETURN p, c
-- semantic error: Undefined variable 'p'
```

Reproduit avec **tous** les agrégats : `count(*)`, `collect(...)`, `sum(...)`, `avg(...)`.
Deux formes voisines, elles, marchent : `RETURN <expr> AS p, count(*) AS c`, et
`WITH collect(x) AS ids` quand il n'y a **pas** de clé de groupe.

C'est donc une règle précise, pas un manque général : **quand `WITH` mêle une expression de
groupement et un agrégat, les alias de groupement ne sont pas exposés.**

Dépendance mesurée dans `backend/src`, **en lignes** et non en requêtes (recompté le 13/11/2026,
identique sur `origin/main` f41bf545 et `origin/consolidation/obrain-default`) :

| Forme | Commande | Lignes |
|---|---|---|
| `WITH … collect(` | `git grep -E 'WITH .*collect\(' -- src` | **85** |
| `WITH … count(*)` | `git grep -E 'WITH .*count\(\*\)' -- src` | **3** |
| `WITH … sum/avg/min/max` | `git grep -E 'WITH .*(sum\|avg\|min\|max)\(' -- src` | **16** |
| **Total** | | **104** |

**Ce total est une borne grossière, pas un compte de requêtes touchées (non vérifié requête par
requête).** Le grep ligne à ligne compte aussi des `WITH collect(x)` **sans** clé de groupe (qui
passent, voir plus haut), manque les `WITH` dont l'agrégat est sur une autre ligne, et ignore
`count(n)` / `count(DISTINCT …)` (`git grep -E 'WITH .*count\(' -- src` : 62 lignes).

Conséquence : ce n'est **pas** 104 requêtes à réécrire. Réécrire `WITH` en `RETURN` n'est pas
toujours possible (le `WITH` sert justement à poursuivre le pipeline). C'est un **défaut de
sémantique à corriger dans obrain**, et il est étroit : la liaison des alias après agrégation.

### 2. `duration.between(a, b)` — **3 requêtes** (4 occurrences dont 1 commentaire)

```
syntax error: Expected end of query   (à la colonne du point)
```

La forme `duration('P1D')` et l'arithmétique `datetime() + duration({days:3})` passent. C'est
l'appel qualifié `duration.between` qui n'est pas reconnu.

Recompté le 13/11/2026 par `git grep -n 'duration\.between' -- src` sur `origin/main` (f41bf545)
et `origin/consolidation/obrain-default` : `src/neo4j/mcp_federation.rs:482`,
`src/neo4j/note.rs:2116`, `src/neo4j/note.rs:2138` (requêtes) et `src/neo4j/client.rs:348`
(commentaire). La première version de ce document annonçait 10 occurrences ; ce chiffre n'est
reproductible sur aucune des deux branches, ni sur `main` du 11/11 (e978eb3c).

### 3. `apoc.*` — **utilisé par l'application, pas seulement par la migration**

La première version de ce document affirmait « 30 occurrences, toutes dans `path_migration.rs`,
pas dans l'application ». **C'est faux.** Recompté le 13/11/2026 par
`git grep -n 'apoc\.' -- src` (identique sur `origin/main` f41bf545,
`origin/consolidation/obrain-default` et `main` du 11/11 e978eb3c) : 31 occurrences dans
6 fichiers, dont des requêtes du **chemin applicatif** :

| Fichier:ligne | Appel | Contexte |
|---|---|---|
| `src/neo4j/analytics.rs:822-823` | `apoc.coll.sort` | `get_project_percentiles` |
| `src/neo4j/note.rs:1421,1425` | `apoc.coll.toSet` | `get_pairwise_coupling` |
| `src/neo4j/graph_schema.rs:54` | `apoc.meta.schema` | `schema_catalog` |
| `src/neo4j/data_migrations.rs:510-545` | `apoc.coll.indexOf`, `apoc.text.join`, `apoc.do.when`, `apoc.merge.relationship` | migrations de données du registre (exécutées au démarrage) |
| `src/neo4j/path_migration.rs:196,488,505` | `apoc.merge.relationship`, `apoc.refactor.mergeNodes` | `migrate file-paths` (outil ponctuel) |

Le reste (`src/graph/schema.rs`, `path_migration.rs:883-891,1013,1248`, `data_migrations.rs:389,1088`)
est du commentaire, un test ou la vérification de présence d'APOC.

Conséquence : APOC **est** un blocage de la bascule tant que ces requêtes applicatives ne sont pas
réécrites ou qu'obrain n'offre pas d'équivalent (`apoc.coll.sort`, `apoc.coll.toSet` et
`apoc.meta.schema` au minimum). Le support de ces fonctions par obrain n'a **pas** été mesuré :
**non vérifié**.

### 4. `db.index.fulltext.*` — **0 occurrence**

Absent d'obrain (il a ses propres procédures `obrain.*`), mais PO n'utilise pas le plein-texte
de Neo4j : il passe par Meilisearch. Sans effet.

## Une prudence : ce que le CLI affiche n'est pas le contrat

`obrain query --format json` rend les dates en `"Timestamp(Timestamp(1794396228509982000ns))"`
et les listes en `"[a,b]"`. C'est l'**affichage du CLI**, pas nécessairement le type que
l'API Rust rendra. Un consommateur qui passerait par la bibliothèque recevrait un type, pas
cette chaîne. **À vérifier avant d'en faire un défaut** — c'est exactement l'erreur que j'ai
faite plus tôt en lisant un log après coup.

## Ce que ça change dans l'ordre des travaux

1. Corriger la liaison d'alias après agrégat dans obrain (débloque l'essentiel des ~104 lignes relevées ci-dessus ; nombre exact de requêtes **non vérifié**).
2. `duration.between` (3 requêtes, correctif de parseur).
3. APOC : les requêtes applicatives (`analytics.rs`, `note.rs`, `graph_schema.rs`, `data_migrations.rs`)
   réécrites ou couvertes par obrain, et les deux gestes de `path_migration.rs` réécrits côté outil de
   migration.
4. Le reste est déjà couvert.

Rien de tout cela ne demande la base vivante : tout se rejoue sur une copie.
