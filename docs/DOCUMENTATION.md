# Charte de documentation : les diagrammes du depot font foi

Les diagrammes d'architecture et de conception de Project Orchestrator sont en Mermaid standard. **Regle par defaut (2026-10) : un nouveau diagramme vit dans le service Mermaid de l'equipe** (workspace `project-orchestrator`). Ce depot est public : l'hote du service n'y est jamais ecrit (gate `scripts/forbidden/check-forbidden-tokens.mjs`), il est nomme hors depot, dans le skill `mermaid-design`, et le depot ne garde que son **entree d'index** (`INDEX.yml`, section 5) : identifiant, emplacement sans hote, date et sha du releve. Il n'y a pas de `.mmd` dans le depot pour un tel diagramme : une copie locale serait une deuxieme source qui derive. Les `docs/diagrams/<nom>.mmd` deja presents restent valides (format historique) jusqu'a leur migration. Dans les deux cas le diagramme est relu contre le code dans la meme PR que le code qu'il decrit, et la verification du depot reste hors reseau : elle controle l'index, jamais le service.

- Index : `docs/diagrams/INDEX.yml`
- Fichiers sans proprietaire : `docs/diagrams/ORPHANS.md` (genere)
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

`INDEX.yml` : une entree par diagramme `{name, covers, owner, status}`, plus `file` quand il existe,
`role: index` sur la carte d'index et `supersedes` sur une entree qui reprend une carte provisoire.

- `status: planned` : le diagramme est prevu, sans fichier. C'est le cas de toutes les cartes d'architecture existantes tant que leur contenu n'a pas ete releve contre le code : on n'exporte pas un contenu non verifie.
- `status: verified` : le diagramme existe et a ete relu contre le code. Deux formes :
  - **externe (regle par defaut)** : le diagramme est publie dans le service Mermaid ; l'entree porte `mermaid_id` (l'identifiant rendu a la creation) et/ou `external` (emplacement SANS hote : `<workspace>/<session>/<name>@<version>`, le nom devant etre celui de l'entree ; un segment qui est un nom d'hote, `localhost` compris, est refuse), plus `verified_at` (AAAA-MM-JJ) et `verified_sha` (sha du backend contre lequel le contenu a ete relu, qui doit etre un ancetre de HEAD : `git merge-base --is-ancestor`, hors reseau ; un sha de branche non fusionnee ou ecrasee par un squash est refuse, on releve contre un sha dont la branche verifiee descend. Les branches `integ/*` sont promues vers `main` par un commit de fusion, qui garde leurs commits dans l'historique ; une promotion par squash devra relever les sha). Aucun `.mmd` local : le script refuse une entree externe qui en a un. Une cle `mermaid_id:` ou `external:` vide est une erreur, de meme qu'une entree externe `planned`. Les en-tetes `%% name`, `%% covers`, ce que le diagramme PROUVE, les SOURCES LUES et le NON ETABLI (section 2) restent obligatoires, dans le diagramme publie.
  - **locale (historique)** : `docs/diagrams/<name>.mmd` existe, son en-tete est valide et son contenu a ete relu contre le code au sha `verified`. Le champ `file` n'apparait que dans ce cas.

Une entree `verified`, externe ou locale, possede ses `covers` (proprietaire unique, cliquet d'orphelins) ; le controle reste hors reseau. Passer de `planned` a `verified` se fait dans une PR qui publie le diagramme (externe) ou ajoute le `.mmd` et `file:` (locale), et passe le statut. Corriger un diagramme externe = une nouvelle version (`/update`), jamais une suppression ; la PR met a jour `verified_at` et `verified_sha`.

### Un fichier, un proprietaire

Deux `covers` ne doivent jamais matcher le meme fichier : un fichier couvert par deux diagrammes
n'a pas de proprietaire, il a deux moities de proprietaire, et personne ne repond de lui quand il
change. Le script echoue sur tout recouvrement. Onze recouvrements existaient a l'amorcage ; chaque
arbitrage est ecrit en commentaire au-dessus de l'entree concernee dans `INDEX.yml`. Exemples lus
dans le code, pas devines : `src/api/ws_handlers.rs` multiplexe les flux CRUD et graphe
(`ws_events`, `passes_filters`) et n'est donc pas du chat, il reste a `po-api` ; `src/orchestrator/runner.rs`
scanne et synchronise le code (`scan_files`, `init_embedding_provider`) et appartient a `po-sync-parser`,
pas au runner de plans ; `src/events/reactions.rs` est le jeu de reactions de `EventReactor` et va
a `po-autonomic`, pas au cycle des plans.

### Seule une entree `verified` possede un fichier

Une entree `planned` **ne possede rien**. Elle annonce un perimetre ; aucun diagramme n'existe,
personne n'a relu son contenu contre le code. Compter ses `covers` comme de la propriete ferait
baisser le plafond d'orphelins et taire le gate de derive **sans qu'une seule ligne de diagramme
soit ecrite** : l'index acheterait du credit sur des intentions. Les `covers` d'une entree
`planned` servent uniquement a reserver un perimetre, ce qui suffit a faire jouer la regle du
proprietaire unique.

Cette regle vient du verificateur de nexus (`claude-code-api/tests/diagram_index.rs`,
`owning_globs`), qui l'appliquait avant cet index. Elle est reprise ici parce qu'elle est juste,
et elle a change le chiffre affiche dans le mauvais sens — c'est-a-dire le bon.

### Les orphelins sont publies, pas caches

Un fichier source qu'aucun diagramme **verifie** ne couvre est **orphelin**. La liste complete
est generee dans `docs/diagrams/ORPHANS.md` et versionnee. Deux chiffres, toujours les deux :

| | |
|---|---|
| Fichiers source | 1204 |
| **Orphelins** (aucun diagramme verifie) | **1201 (99,8 %)** |
| dont deja **reserves** par une entree `planned` | 470 |
| Possedes par un diagramme verifie | 3 (via l'index de nexus) |

Les 470 sont un sous-ensemble strict des orphelins : un proprietaire est designe, son diagramme
reste a ecrire. On ne retire pas des fichiers du denominateur pour embellir le chiffre ; on reduit
le chiffre en ecrivant des diagrammes. Le fichier est deterministe (ni sha ni date) pour servir de
gate hors reseau, et il ne s'edite pas a la main.

### Un depot peut tenir son propre index (et la regle vaut ENTRE les index)

Cet index est l'index de la cartographie : il porte les cartes `po-*`, dont les `.mmd` vivent
ici. Un depot voisin qui veut tenir ses propres diagrammes de conception a un probleme que cet
index ne resout pas : il n'a aucun moyen de pointer un `.mmd` situe la-bas, et son gate tourne
dans sa chaine d'outils, pas dans la notre (nexus est un depot Rust : son verificateur d'index
est un test `cargo`, pas un script Node, et c'est le bon choix pour lui).

Un depot voisin peut donc tenir son propre `docs/diagrams/INDEX.yml`, au meme format, pour les
diagrammes dont le `.mmd` vit chez lui. Un index **derive** des en-tetes (nexus :
`scripts/derive_diagram_index.py`) est lu aussi : ses globs sans prefixe designent son propre
depot, une entree qui a un `file` sans `status` possede (le .mmd existe), et sa section
`orphans:` est ignoree (liste informative, propriete de personne). Deux regles rendent la coexistence verifiable plutot que
polie :

1. **Un index local ne possede que des chemins de SON depot.** Un glob `frontend:` dans l'index
   de nexus serait une prise de pouvoir sur un depot voisin : le script l'ignore.
2. **La regle du proprietaire unique vaut entre les index.** Si cet index et un index local
   revendiquent le meme fichier, c'est une **erreur** : deux gates se contrediraient sur le meme
   fichier, et chacun se croirait couvert par l'autre. Le script lit les index locaux des depots
   voisins, signale toute collision en nommant les deux revendications, et **ne compte pas comme
   orphelins** les fichiers qu'un index local possede — ils ont un proprietaire, ailleurs.

En pratique, nexus tient un index de 3 diagrammes verifies (`release-readiness`, `po-bugs`,
`nexus-model-catalogue`) qui possede 24 fichiers. Aucune collision avec les cartes `po-*` : c'est
verifie a chaque execution, pas suppose. Le decompte des orphelins en tient compte.

### Le nombre d'orphelins ne peut que descendre (cliquet)

Viser « zero orphelin » echouerait des le premier jour, et un gate que personne ne peut
satisfaire finit desactive. `ORPHANS.md` porte donc un marqueur
`<!-- orphan-ceiling: N -->`, et le verificateur echoue si le compte reel le depasse :
ajouter un fichier source sans proprietaire casse la build. La sortie est un glob `covers`
sur une entree `verified`, **pas un plafond plus haut**.

Trois proprietes, chacune verifiee par un test de bout en bout, parce qu'un cliquet qui se
desserre tout seul ne tient rien :

- `--write-orphans` **ne releve jamais** le plafond : au-dessus, il echoue au lieu d'absoudre.
  Cela vaut pour le total ET pour chaque plafond par depot : au-dessus de l'un d'eux, le
  registre n'est pas reecrit (mesure du 02/10/2026 : il etait regenere depuis le compte reel,
  et relevait le plafond du depot par la commande meme censee l'en empecher).
- le relever exige `--raise-ceiling "<raison>"`, qui laisse une trace lisible en revue.
- supprimer le marqueur **echoue** aussi : on ne desarme pas le cliquet en effacant son compteur.

Il descend seul des qu'un fichier gagne un proprietaire. Le marqueur est le meme que celui du
gate de nexus (`claude-code-api/tests/diagram_index.rs`), pour que les deux depots se tiennent a
la meme regle et qu'un lecteur n'ait qu'une forme a connaitre.

**Un plafond par depot, en plus du total.** `ORPHANS.md` porte
`<!-- orphan-ceiling: 1201 -->` et un marqueur par depot :

```
<!-- orphan-ceiling-backend: 456 -->
<!-- orphan-ceiling-frontend: 545 -->
<!-- orphan-ceiling-nexus: 73 -->
<!-- orphan-ceiling-website: 127 -->
```

C'est ce qui rend le cliquet reel la ou il tourne. Un checkout d'un seul depot — la CI — ne voit
pas les quatre : comparer son compte partiel au total (456 contre 1201) serait un vert permanent
qui ne verifie rien. Un plafond par depot porte sur le meme perimetre que le compte, donc il est
comparable, et **chaque depot present est tenu a son chiffre meme en checkout partiel**. Ajouter
un fichier source non revendique au backend fait echouer la CI du backend.

Ce qui reste reserve a la portee complete : le **total**, qui n'est ni applique ni mis a jour
quand un depot manque, et `--write-orphans`, refuse — il abaisserait le total a la valeur
partielle et effacerait de la liste les orphelins des depots absents. Un depot absent n'est pas
tenu a son plafond. Les autres regles s'appliquent toujours : une portee partielle n'est pas une
amnistie, et un index casse echoue.

### Cartes provisoires reprises (`supersedes`)

Deux cartes de la cartographie d'origine portaient un suffixe numerique et n'etaient rattachees a
aucun index : leur perimetre est repris, sous un nom conforme, par une entree qui le declare.

| Carte provisoire | Reprise par | Perimetre |
|---|---|---|
| `po-chat-transport-2` | `po-p2p-transport` | `src/transport/**`, `src/reception/**` |
| `po-code-intelligence-2` | `po-architecture-derive` | `src/architecture/**` |

Le script verifie qu'une carte declaree dans `supersedes` n'est plus une entree de l'index (sinon
le perimetre serait compte deux fois) et que l'index porte exactement une carte `role: index`
(`po-carte`), qui couvre `docs/diagrams/INDEX.yml` : c'est par la que toute entree est rattachee.

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

1. **Creer** : publier le diagramme dans le service Mermaid (en-tete complete) et passer l'entree de l'index a `verified` avec `mermaid_id`/`external`, `verified_at`, `verified_sha` (forme historique : ajouter `docs/diagrams/<name>.mmd` et `file:`).
2. **Citer** : creer la note `design-ref` (section 6) et la lier aux taches concernees.
3. **Mettre a jour** : modifier le `.mmd` DANS LA MEME PR que le code, mettre a jour `verified` (et `covers` si le perimetre change, dans le fichier ET dans l'index).
4. **Archiver** : quand le code couvert disparait, supprimer le `.mmd` et l'entree de l'index dans la PR qui supprime le code ; invalider la note `design-ref`.

## 9. Lien avec les PR et le gate de derive (tache 1.1)

Une PR = une tranche verticale : correctif + test de regression (rejoue sans le correctif) + diagramme mis a jour + doc. Jamais de PR « diagrammes seuls » qui decrit du code non verifie.

Le gate de derive de la tache 1.1 fonctionnera sur ces seuls fichiers, sans reseau : pour chaque fichier modifie par la PR, il retrouve via `INDEX.yml` les diagrammes dont un `covers` matche, et signale la PR si un diagramme `verified` concerne n'est pas modifie (ni son `verified` mis a jour), ou si un fichier source n'a aucun proprietaire. `check-index.mjs` en est la brique de base (meme resolution des globs). Il n'est pas encore cable en CI.

## 10. Verification

```
node scripts/diagrams/check-index.mjs                  # exit 1 si une regle est violee
node scripts/diagrams/check-index.mjs --list           # noms des fichiers orphelins
node scripts/diagrams/check-index.mjs --write-orphans  # regenere docs/diagrams/ORPHANS.md
node scripts/diagrams/check-index.mjs --check-orphans   # echoue si ORPHANS.md n'est pas a jour
node scripts/diagrams/check-index.mjs --fail-on-orphans # cible finale : plus aucun orphelin
node scripts/diagrams/check-index.mjs --write-orphans --raise-ceiling "<raison>"  # a eviter
node scripts/diagrams/check-index.mjs --json out.json
node scripts/diagrams/check-index.mjs --strict         # echoue aussi si un depot voisin est absent
node --test scripts/diagrams/check-index.test.mjs
```

Regles verifiees : chaque glob `covers` matche au moins un fichier existant ; **aucun fichier n'est
couvert par deux diagrammes**, ni ici ni entre cet index et l'index local d'un depot voisin ;
seule une entree `verified` compte comme proprietaire pour le calcul des orphelins ; le nombre
d'orphelins ne depasse pas le plafond publie dans `ORPHANS.md`, qui ne peut que descendre ;
l'index porte une seule carte `role: index` et elle couvre
`INDEX.yml` ; toute carte declaree dans `supersedes` a bien disparu de l'index ; chaque entree
`verified` a son `.mmd` avec les en-tetes `%% name`, `%% covers` (identique a l'index) et
`%% verified` (sha court) ; aucune entree `planned` n'a de fichier ; tout `.mmd` du dossier est dans
l'index ; `docs/diagrams/ORPHANS.md` est a jour (avertissement par defaut, erreur avec
`--check-orphans`). Les **orphelins** sont les fichiers source (Rust et TS/TSX sous `src/`, hors
tests) sans diagramme proprietaire.

Les depots voisins sont resolus par `DIAGRAM_ROOT_BACKEND|FRONTEND|NEXUS|WEBSITE` (defaut : `../frontend`, `../nexus`, `../website`) ; un depot absent est ignore avec un avertissement.

## 11. Ce que la CI verifie, et ce qu'elle ne verifie pas

Le job `lint` de `.github/workflows/ci.yml` lance `check-index.mjs` et ses tests. Node est deja
sur les runners `ubuntu-latest` : pas de `setup-node`, pas de `package.json`, aucune nouvelle
chaine d'outils dans un depot Rust.

Ce vert couvre : la structure de l'index, un glob `backend:` qui ne matche plus rien, un fichier
revendique par deux diagrammes, une entree `verified` sans son `.mmd`, un en-tete invalide.

Ce vert couvre aussi le **cliquet d'orphelins du backend** : son plafond par depot porte sur le
meme perimetre que le checkout, donc ajouter un fichier source non revendique fait echouer la CI.

Ce vert NE couvre PAS : les depots voisins, absents de ce checkout — leurs globs ne sont pas
resolus, leurs plafonds ne sont pas appliques, le plafond TOTAL et la fraicheur de `ORPHANS.md`
non plus. **Ce vert ne prouve donc pas l'absence de derive** et ne doit pas etre cite comme tel ;
c'est la tache 1.1 qui branchera le gate complet. Le script annonce lui-meme ses limites au lieu
de passer vert en silence.

## 12. Etat

- 27 diagrammes prevus dans l'index, tous `planned` : **aucun `.mmd` n'existe encore**. Un index de
  27 entrees n'est pas 27 diagrammes ; c'est la liste des proprietaires, et c'est deja ce qui manquait
  au gate de derive.
- 1204 fichiers source, **1201 orphelins (99,8 %)**, dont 470 deja reserves par une entree
  `planned`. Le seul code reellement couvert par un diagramme verifie l'est par l'index de nexus
  (3 diagrammes, 24 fichiers possedes la-bas). C'est l'etat honnete : la cartographie n'a pas
  encore produit un seul diagramme dans ce depot.
- Zero recouvrement : chaque fichier reserve ou possede l'est par exactement un diagramme, ici
  comme entre cet index et celui de nexus.
- Les `covers` ne couvrent que ce que la cartographie a reellement lu ; le reste est orphelin, c'est
  voulu : on ne gonfle pas la couverture.
