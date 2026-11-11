# Viabilité des requêtes : ce qu'obrain sait faire, mesuré sur un graphe réel

Mesuré le 11/11/2026 sur l'archive `~/.obrain/db/po` (**1 227 051 nœuds, 1 889 164 arêtes**,
schéma réel de PO : `ChatEvent`, `Function`, `Import`, `Note`, `Task`, `Decision`…), en
**lecture seule** (`obrain query --read-only`). Aucune écriture, aucune base vivante touchée.

Cette archive date de juillet (1,35 M nœuds contre 3,85 M aujourd'hui) : elle ne sert pas de
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

Dépendance mesurée dans `backend/src` :

| Forme | Requêtes |
|---|---|
| `WITH … collect(` | **85** |
| `WITH … count(*)` | **3** |
| `WITH … sum/avg/min/max` | **16** |
| **Total** | **104** |

Conséquence : ce n'est **pas** 104 requêtes à réécrire. Réécrire `WITH` en `RETURN` n'est pas
toujours possible (le `WITH` sert justement à poursuivre le pipeline). C'est un **défaut de
sémantique à corriger dans obrain**, et il est étroit : la liaison des alias après agrégation.

### 2. `duration.between(a, b)` — **10 occurrences**

```
syntax error: Expected end of query   (à la colonne du point)
```

La forme `duration('P1D')` et l'arithmétique `datetime() + duration({days:3})` passent. C'est
l'appel qualifié `duration.between` qui n'est pas reconnu.

### 3. `apoc.*` — **30 occurrences, toutes dans `path_migration.rs`**

`apoc.merge.relationship`, `apoc.refactor.mergeNodes`. Vérifié : ces appels ne vivent que dans
un **programme de migration ponctuel**, pas dans l'application. Ce n'est donc pas un blocage de
la bascule — c'est un blocage de **l'outil de migration**, qui devra faire ces deux gestes
autrement.

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

1. Corriger la liaison d'alias après agrégat dans obrain (**débloque 104 requêtes d'un coup**).
2. `duration.between` (10 occurrences, correctif de parseur).
3. Les deux gestes APOC de `path_migration.rs`, réécrits côté outil de migration.
4. Le reste est déjà couvert.

Rien de tout cela ne demande la base vivante : tout se rejoue sur une copie.
