# Charte de documentation : les diagrammes du depot font foi

Les diagrammes d'architecture et de conception de Project Orchestrator vivent dans le depot, en Mermaid standard, rendus nativement par GitHub : un fichier `docs/diagrams/<nom>.mmd` par diagramme. Ils sont relus et modifies comme du code, dans la meme PR que le code qu'ils decrivent. Aucun service externe n'est necessaire pour les lire, les verifier ou les mettre a jour.

- Index : `docs/diagrams/INDEX.yml`
- Verification (hors reseau) : `scripts/diagrams/check-index.mjs`

## 1. Nomenclature

| Sorte | Forme | Exemple |
|---|---|---|
| Cartographie d'architecture (code existant) | `po-<domaine>` | `po-api`, `po-graph-store` |
| Conception d'une fonctionnalite | `<feature>-<aspect>` | `today-data`, `today-actions` |

kebab-case, ASCII, un domaine ou un aspect par diagramme. Jamais de suffixe numerique (`-2`) : si un domaine ne tient pas dans un diagramme, on le scinde par aspect. Le nom du fichier est le nom du diagramme : `docs/diagrams/po-api.mmd`.

Branches : `docs/<domaine>`, `fix/<domaine>-<sujet>`, `test/<domaine>-<sujet>`.

## 2. En-tete obligatoire

Les trois premieres lignes du fichier :

```
%% name: po-api
%% covers: backend:src/api/{mod,routes,handlers}.rs backend:src/auth/**
%% verified: 94c04bce
```

- `name` : identique au nom du fichier et a l'entree de l'index.
- `covers` : globs de fichiers dont le diagramme est proprietaire, separes par des espaces, prefixes par depot (`backend:`, `frontend:`, `nexus:`, `website:`). Doit etre identique a la liste `covers` de `INDEX.yml` (le script le verifie).
- `verified` : sha court du commit du depot principal sur lequel le contenu a ete relu contre le code. Mis a jour a chaque modification du diagramme.

Viennent ensuite des commentaires `%%` : ce que le diagramme PROUVE, les SOURCES LUES (fichiers et fonctions) et ce qui est NON ETABLI. Une ligne de commentaire vide s'ecrit `%% ` (avec une espace).

## 3. Legende des statuts (par noeud)

| Marque | Sens | A preciser dans le noeud |
|---|---|---|
| ✅ | verifie | COMMENT : mock, Neo4j reel, ou test rejoue sans le correctif |
| 🟠 | decide, non ecrit | la tache qui l'ecrira |
| 🔴 | refuse ou casse | la raison (et la tache `bug` si c'est un defaut) |
| ⚪ | amont : existe ailleurs, hors perimetre | le depot ou le systeme |

Un noeud sans marque est une erreur de relecture. « Aucun appelant » est une affirmation qui se prouve par `grep` ; la commande figure dans les SOURCES LUES.

## 4. Noms de fonctions, jamais de numeros de ligne

On cite `create_router`, `require_auth`, `Config::from_yaml_and_env`, pas `routes.rs:58`. Un numero de ligne est faux des le premier commit voisin (mesure : `public_routes` a bouge de deux lignes en quelques jours) ; un nom de fonction ne l'est que si la fonction change reellement.

## 5. Statuts d'un diagramme dans l'index

`INDEX.yml` : une entree par diagramme `{name, file, covers, owner, status}`.

- `status: planned` : le diagramme est prevu, sans fichier. C'est le cas de toutes les cartes d'architecture existantes tant que leur contenu n'a pas ete releve contre le code : on n'exporte pas un contenu non verifie.
- `status: verified` : `docs/diagrams/<name>.mmd` existe, son en-tete est valide et son contenu a ete relu contre le code au sha `verified`. Le champ `file` n'apparait que dans ce cas.

Passer de `planned` a `verified` se fait dans une PR qui ajoute le `.mmd`, ajoute `file:` et passe le statut.

## 6. Reference de conception pour les taches

Une note PO de type `context`, taguee `design-ref`, par diagramme : nom, chemin `docs/diagrams/<name>.mmd`, sha `verified` cite, ce que le diagramme TRANCHE (resume actionnable, refus compris) et la commande de lecture exacte :

```
git show <sha>:docs/diagrams/<name>.mmd
```

La note est liee aux TACHES qu'elle specifie, jamais au plan (lie au plan, le diagramme arriverait sur toutes les taches). Le diagramme du depot fait foi sur la note : si le fichier a evolue depuis le sha cite, on le lit avant de coder et on signale l'ecart sur la tache.

## 7. Un bug = une tache + une note + un noeud

1. une tache taguee `bug`, avec le test de regression qui echoue sans le correctif ;
2. une note `gotcha` liee a la tache et aux fichiers concernes ;
3. un noeud 🔴 dans le diagramme du domaine, avec la raison. Au correctif, le noeud passe en ✅ avec « rejoue sans le correctif ».

## 8. Cycle de vie

1. **Creer** : ajouter `docs/diagrams/<name>.mmd` (en-tete complete) et passer l'entree de l'index a `verified` avec `file:`.
2. **Citer** : creer la note `design-ref` (section 6) et la lier aux taches concernees.
3. **Mettre a jour** : modifier le `.mmd` DANS LA MEME PR que le code, mettre a jour `verified` (et `covers` si le perimetre change, dans le fichier ET dans l'index).
4. **Archiver** : quand le code couvert disparait, supprimer le `.mmd` et l'entree de l'index dans la PR qui supprime le code ; invalider la note `design-ref`.

## 9. Lien avec les PR et le gate de derive (tache 1.1)

Une PR = une tranche verticale : correctif + test de regression (rejoue sans le correctif) + diagramme mis a jour + doc. Jamais de PR « diagrammes seuls » qui decrit du code non verifie.

Le gate de derive de la tache 1.1 fonctionnera sur ces seuls fichiers, sans reseau : pour chaque fichier modifie par la PR, il retrouve via `INDEX.yml` les diagrammes dont un `covers` matche, et signale la PR si un diagramme `verified` concerne n'est pas modifie (ni son `verified` mis a jour), ou si un fichier source n'a aucun proprietaire. `check-index.mjs` en est la brique de base (meme resolution des globs). Il n'est pas encore cable en CI.

## 10. Verification

```
node scripts/diagrams/check-index.mjs              # exit 1 si une regle est violee
node scripts/diagrams/check-index.mjs --list       # noms des fichiers orphelins
node scripts/diagrams/check-index.mjs --json out.json
node scripts/diagrams/check-index.mjs --strict     # echoue aussi si un depot voisin est absent
node --test scripts/diagrams/check-index.test.mjs
```

Regles verifiees : chaque glob `covers` matche au moins un fichier existant ; chaque entree `verified` a son `.mmd` avec les en-tetes `%% name`, `%% covers` (identique a l'index) et `%% verified` (sha court) ; aucune entree `planned` n'a de fichier ; tout `.mmd` du dossier est dans l'index. Il liste enfin les **orphelins** : fichiers source (Rust et TS/TSX sous `src/`, hors tests) sans diagramme proprietaire.

Les depots voisins sont resolus par `DIAGRAM_ROOT_BACKEND|FRONTEND|NEXUS|WEBSITE` (defaut : `../frontend`, `../nexus`, `../website`) ; un depot absent est ignore avec un avertissement.

## 11. Etat

- 27 diagrammes prevus dans l'index, tous `planned` : aucun n'est encore exporte.
- Les `covers` ne couvrent que ce que la cartographie a reellement lu ; le reste est orphelin, c'est voulu : on ne gonfle pas la couverture.
