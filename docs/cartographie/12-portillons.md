# 12 — Les portillons qui protègent la refonte (`xtask pins --path-deps`, `containment`, `parity`)

Conception en lecture seule (aucun build, aucun git exécuté d'écriture). Elle est écrite pour être codée telle quelle, dans le style des sous-commandes existantes de `backend/xtask/`.

## 0. Ce que les sous-commandes existantes imposent (le style à reproduire)

| Fait | Source |
|---|---|
| xtask est HORS workspace, **zéro dépendance** (std seule, `tempfile` interdit : un `TempRepo` maison) | `xtask/Cargo.toml:8-24` |
| Dispatcher : une sous-commande = un module, `run(args) -> i32` est le SEUL endroit qui touche disque/réseau/`git` ; la logique est en fonctions pures testées | `xtask/src/main.rs:1-9,27-31` |
| Codes de sortie communs : **0** rien à signaler, **1** divergence CONSTATÉE, **2** non vérifiable (clone absent, arguments faux) | `xtask/src/main.rs:7-9` |
| Seul `impl Repo` lance `git` ; les parseurs de sortie sont des fonctions libres | `xtask/src/git.rs:1-6,89-102` |
| `fetch` est « signalé, pas fatal » (on continue hors ligne) | `xtask/src/git.rs:104-108` |
| Parseur TOML **par lignes** (`toml_str`, `dep_name`, `parse_pins`), pas de crate `toml` | `xtask/src/pins.rs:89-135` |
| Clones voisins : `--clone <nom>=<chemin>`, sinon `../<nom>` ; `--no-fetch` pour rejouer hors ligne | `xtask/src/pins.rs:16-24,373-380` |
| Liste d'exceptions **gelée**, `--strict` exit 1, une exception inutilisée est elle-même une erreur | `xtask/src/storage_boundary.rs:23-38`, `xtask/storage-boundary-exceptions.txt` |
| Fixtures de test : `TempRepo` (dépôt jetable, `commit`, `side_branch`) | `xtask/src/testrepo.rs:12-72` |
| CI : xtask est formaté, clippy `-D warnings`, testé et « smoke » dans le job **Lint & Format** | `.github/workflows/ci.yml:339-357` |

Conséquence : les trois portillons ajoutent `mod containment; mod parity;` et étendent `pins`, rien d'autre dans `main.rs`. Ce qui exige `serde_json` (rejeu HTTP) NE va PAS dans xtask : voir §3.

Note de terrain : à ce jour **aucun** `Cargo.toml` de la branche ne contient de dépendance `path = "../…"` hors du dépôt (seuls `crates/…` et `.`, `Cargo.toml:120-121,235-238,301`) ; `this-rs` et `atelier` y entreront avec O-THISRS-1. Le portillon doit donc être livré AVANT la première, pour que la première soit déjà protégée (et un test garantit qu'il est silencieux tant qu'il n'y en a aucune).

---

## 1. `cargo xtask pins --path-deps`

### 1.1 Problème
`pins` couvre les dépendances `git = …, rev = …` (`pins.rs:1-8`). Une dépendance `path = "../this"` n'a **pas de rev** : le build dépend de ce que le dépôt voisin a checkouté ce jour-là. Deux développeurs, ou le CI et un poste, compilent deux codes différents sous le même sha de PO. Ce portillon rend ce HEAD explicite et versionné.

### 1.2 Forme du fichier `pins.toml` (racine du dépôt, versionné)

```toml
# pins.toml — HEAD attendu de chaque dépôt voisin référencé par `path = "…"`.
# Généré/rafraîchi par `cargo xtask pins --path-deps --update` ; relu en revue.
version = 1

[repo.this]                       # clé = nom du dossier du toplevel git du voisin
path = "../this"                  # indice relatif à la racine de PO (non normatif : la résolution réelle vient de Cargo.toml)
url  = "https://github.com/this-rs/this.git"   # informatif ; sert au message et au checkout CI
rev  = "9f3c2a1d4e5b6a7c8d9e0f1a2b3c4d5e6f708192"   # 40 hex, obligatoire
deps = ["this", "this-descriptor"]  # noms de crates par `path` qui mènent ici (détecte l'ajout/retrait)

[repo.atelier]
path = "../atelier"
url  = "https://github.com/this-rs/atelier.git"
rev  = "0a1b2c3d4e5f60718293a4b5c6d7e8f901234567"
deps = ["atelier-core"]
```

Règles de format (parseur par lignes, comme `parse_pins`) : `version = 1` obligatoire (autre valeur → exit 2) ; sections `[repo.<nom>]` ; `rev` 40 caractères hex minuscules (`git::looks_like_sha` accepte 7-40 : ici on exige 40, car un sha court devient ambigu) ; clé inconnue → exit 2 (pas de silence) ; clé répétée → exit 2.

### 1.3 Logique de résolution

1. **Manifestes lus** : `Cargo.toml` racine + chaque `crates/*/Cargo.toml` (et `desktop/` s'il est hors workspace) — liste obtenue par `read_dir`, pas codée en dur. Sections : `[dependencies]`, `[dev-dependencies]`, `[build-dependencies]`, `[workspace.dependencies]`, `[target.*.dependencies]`, `[patch.*]` (un `patch` par `path` compte autant), inline `x = { path = "…" }` ET table `[dependencies.x]` avec `path = "…"` seule sur sa ligne (mêmes deux formes que `parse_pins`, `pins.rs:125-135`).
2. **Extraction** `parse_path_deps(manifest: &str) -> Vec<PathDep { dep, path, line }>` : réutilise `toml_str(line, "path")` (garde-fou `before_ok` déjà écrit : `path` ne doit pas être la fin de `rpath`/`my_path`, `pins.rs:93-97`). Les lignes de commentaire (`#`) sont retirées AVANT, sinon `# path = "../old"` produit un faux voisin.
3. **Résolution** : `abs = normalize(manifest_dir.join(path))` (normalisation lexicale des `..` ; `canonicalize` seulement si le dossier existe, sinon exit 2 « voisin absent » nommant la dépendance et le fichier:ligne).
4. **Filtre « voisin »** : `top = git -C abs rev-parse --show-toplevel`. Si `top` == toplevel de PO → dépendance interne (`crates/…`) → ignorée. Sinon `name = basename(top)`. Plusieurs deps → un même dépôt : regroupées (`group_by_repo`-like, `pins.rs:205`).
5. **Transitivité** (borne 3, ensemble des toplevels déjà vus pour les cycles) : on relit les manifestes du voisin et on suit ses propres `path` qui sortent de LUI. C'est ce qui rattrape « this-rs lui-même dépend de `../atelier` ».
6. **HEAD** : `git -C top rev-parse HEAD`. **État sale** : `git -C top status --porcelain --untracked-files=no` non vide → verdict `Dirty` (le HEAD ne décrit pas ce qui est compilé).
7. **Comparaison** à `pins.toml[repo.<name>].rev`, puis, si différent, `git rev-list --count <pin>..HEAD` et `HEAD..<pin>` pour dire « N commits d'avance » ou « voisin en retard de N » (si le pin est inconnu du clone → « pin absent du clone (clone peu profond ou pin jamais poussé) »).
8. **Pin joignable** : `git merge-base --is-ancestor <pin> origin/main` dans le voisin (réutilise `check_rev`, `pins.rs:268`). Un pin sur un commit local jamais poussé est un échec : personne d'autre ne pourrait le reconstruire.

### 1.4 Verdicts, cas d'erreur, codes

| Cas | Sortie | Code |
|---|---|---|
| Aucune `path` hors dépôt ET `pins.toml` absent ou vide | `aucune dépendance path hors dépôt` | 0 |
| Tous les HEAD égaux aux pins, propres | `this : 9f3c2a1d ✔` | 0 |
| **Voisin a bougé** | `✘ this (../this) : pin 9f3c2a1d, HEAD a4b8c0e2 — 1 commit d'avance. Soit `git -C ../this checkout 9f3c2a1d`, soit `cargo xtask pins --path-deps --update` et commiter pins.toml` | 1 |
| Voisin en retard | `HEAD 3d4e5f60 est 2 commits derrière le pin` | 1 |
| Dépendance `path` vers un dépôt absent de `pins.toml` | `✘ atelier référencé par atelier-core (crates/x/Cargo.toml:14) mais non épinglé` | 1 |
| Pin orphelin (aucune dépendance ne mène plus au dépôt) | `✘ pin orphelin « atelier »` | 1 |
| `deps` du pin ≠ deps réelles | `✘ this : deps épinglées [this] ≠ réelles [this, this-descriptor]` | 1 |
| Pin non joignable depuis `origin/main` du voisin | `✘ pin 9f3c2a1d absent de origin/main de this` | 1 |
| HEAD égal mais arbre sale | `✘ this : arbre modifié (3 fichiers) — le HEAD ne décrit pas ce qui est compilé`; `--allow-dirty` le ramène à un avertissement local, **refusé si `CI=true`** | 1 |
| Dossier voisin absent / pas un dépôt git | `non vérifiable — ../this : …` | 2 |
| `pins.toml` illisible / version ≠ 1 / sha ≠ 40 hex | message précis | 2 |
| `--update` avec `CI=true` | `--update interdit en CI` | 2 |

Options : `--path-deps` (active ce mode, `pins` sans lui reste inchangé), `--pins-file <f>` (défaut `pins.toml`), `--clone <nom>=<chemin>` (existant), `--no-fetch` (existant : par défaut `git fetch` du voisin, pour que `origin/main` soit frais), `--update`, `--allow-dirty`.

`--update` réécrit `pins.toml` depuis les HEAD courants (jamais implicite, jamais en CI). C'est le seul moyen « normal » de bouger un pin : la montée devient un commit lisible, relu, et non un état de machine.

### 1.5 Tests (fonctions pures + `TempRepo`) — au moins 8

Parseurs purs, sans disque :
1. `parse_path_deps` : inline `this = { path = "../this" }`, table `[dependencies.atelier-core]` + `path = "../atelier/crates/core"`, `[workspace.dependencies]`, `[patch.crates-io]`.
2. `parse_path_deps` ignore `# path = "../vieux"`, `rpath = "x"`, `description = "path = \"../x\""`.
3. `parse_pins_toml` : refuse sha de 7 caractères, clé inconnue, clé répétée, `version = 2`.
4. `filter_neighbors` : `crates/neural-routing-core` (même toplevel) est ignoré ; `../this` est gardé.

Avec dépôts jetables (`TempRepo::new()` ×2, `path = "../voisin"` écrit dans un Cargo.toml de fixture) :
5. **En phase** : pin = HEAD du voisin → exit 0.
6. **Le voisin a bougé d'un commit** : `voisin.commit("x")` après le pin → exit 1, la sortie contient le NOM du dépôt, les 8 premiers caractères des DEUX shas (`git::short`) et « 1 commit d'avance » ; *rouge sans* : sans le test sha, le code 0 ferait échouer ce test.
7. **Voisin absent** → exit 2 (pas 0) : un voisin introuvable ne peut jamais valoir « vert ».
8. **Dépôt non épinglé** : une `path` vers un 2ᵉ voisin absent de `pins.toml` → exit 1 nommant le dépôt.
9. **Pin non poussé** : pin = commit hors `origin/main` (avec `side_branch`) → exit 1.
10. **Arbre sale** : fichier suivi modifié, HEAD égal → exit 1 ; avec `CI=true` et `--allow-dirty` → toujours 1.
11. **Silence initial** : manifeste sans `path` externe et sans `pins.toml` → exit 0 (garantit qu'on peut fusionner le portillon avant la première dépendance).

### 1.6 Branchement
Dans `lint` (comme `storage-boundary`, `ci.yml:356-357`) : `cargo xtask pins --path-deps --no-fetch`. Le job qui compile (Build/Unit Tests) extrait ses voisins **depuis `pins.toml`** (`actions/checkout` avec `repository`/`ref` lus du fichier par une étape `cargo xtask pins --path-deps --print-checkout`). Voir §5 : sans cette règle, le contrôle en CI serait tautologique.

---

## 2. `cargo xtask containment`

### 2.1 Sémantique exacte
« **échouer si `origin/main` contient un commit absent de la branche testée** » = `git rev-list <head>..<base>` non vide, où `base = <remote>/main` et `head = HEAD` (ou `--head <sha>`). Équivalent à `! git merge-base --is-ancestor <base> <head>`. On utilise `rev-list` (et non `merge-base`) parce qu'il donne la LISTE à afficher.

Étapes de `run` :
1. **Résoudre** `head` (`--head`, défaut `HEAD`) et `base` (`--base`, défaut `origin/main`) ; l'un des deux inconnu → exit 2.
2. **Fetch** : par défaut `git fetch --quiet <remote> main` (remote = partie avant `/` de `base`, `git::remote_of`, `git.rs:63-65`). Échec de fetch : **avertissement** et on continue sur le ref local (comportement `git.rs:104-108`) — sauf `--require-fetch`, imposé en CI, où il devient exit 2 : un ref périmé ferait un vert sur des données mortes. `--no-fetch` pour le rejeu hors ligne.
3. **Clone peu profond** : `git rev-parse --is-shallow-repository` = `true` → exit 2 « historique tronqué, impossible de conclure » (sinon `rev-list` ne voit qu'une tranche et conclut « contenu » à tort). En CI : `fetch-depth: 0`.
4. **Histoires disjointes** : `git merge-base head base` vide → exit 1 « aucune base commune » (≠ 2 : c'est un fait constaté).
5. **Calcul** : `missing = rev-list --format=%H%x09%s head..base` (parseur existant `git::parse_log`, `git.rs:19-35`). Vide → exit 0. Sinon exit 1.
6. **Exemption de chemins (optionnelle)** : `--paths <glob,glob>` — n'échoue que si un commit manquant touche ces chemins (`git diff --name-only <merge-base> <base> -- <paths>`). Réduit les faux positifs documentaires (un README modifié sur `main` ne doit pas bloquer la branche) ; **désactivé par défaut**, le défaut est strict.

### 2.2 Message (dit quoi faire)

```
containment ✘ origin/main contient 7 commits absents de consolidation/obrain-default
  base commune : 61b4e229    origin/main : 8c1d9e04    branche : bd782f8d
  absents (les 5 plus récents) :
    8c1d9e04 fix(chat): …
    …
  À faire (règle 2 de CONSOLIDATION.md : main redescend ICI, jamais l'inverse) :
    git fetch origin && git merge origin/main
  puis relancer. Ne pas rebaser une branche partagée.
  Fichiers déjà touchés des deux côtés (conflits probables) : src/api/routes.rs, Cargo.toml
```
La dernière ligne = intersection de `diff --name-only base-commune..head` et `base-commune..base` : prévient avant le merge.

### 2.3 Cas d'un fork
- Sur un dépôt dont `origin` est le fork d'un développeur, `origin/main` n'est pas le `main` de référence : option `--remote <nom>` (défaut : `upstream` s'il existe, sinon `origin`) ; le message nomme le remote utilisé.
- PR depuis un fork en CI : le jeton est en lecture seule, `fetch` du dépôt de base reste possible par `git fetch https://github.com/${{ github.repository }} main` (`--base-url`) ; si impossible → exit 2 (jamais 0).

### 2.4 Faux positifs et faux négatifs (nommés)
- **Faux positif** : un commit fusionné dans `main` ENTRE le dernier merge de la branche et le run CI. C'est voulu (la branche est « en retard »), mais cela bloque une PR sans faute. Remède : le message donne la commande ; ne pas ajouter de tolérance numérique par défaut (`--allow-behind N` existe pour le job *schedulé*, pas pour le check requis).
- **Faux positif** : `main` a déjà reçu un revert de ce qui est sur la branche — sans importance, le commit est quand même « absent ».
- **Faux négatif (le grave)** : en CI `pull_request`, `actions/checkout` checkoute `refs/pull/N/merge`, un commit de merge qui **contient déjà `origin/main`** → `HEAD` contient toujours `main` → vert garanti. Garde-fou : le workflow passe `--head ${{ github.event.pull_request.head.sha }}` (le sha de la branche réelle), et le test 6 ci-dessous reproduit exactement ce montage.
- **Sans objet** : squash-merge. La règle regarde les commits de `main` présents dans la branche, pas l'inverse (le piège « ahead>0 après squash » concerne l'autre sens, voir la directive des branches obsolètes).
- `HEAD == base` ou HEAD sur `main` (push sur main) → 0 sans calcul.

### 2.5 Tests (≥ 6, `TempRepo` ×2 : « origin » en dépôt nu et « clone »)
1. Branche créée depuis `main`, `main` n'a pas bougé → 0.
2. `main` avance d'un commit → exit 1, la sortie cite son sha court et son sujet.
3. Après `git merge origin/main` dans la branche → 0.
4. Branche en avance, `main` immobile → 0.
5. `--head <sha>` différent de HEAD (simulation du checkout `pull/N/merge`) : HEAD contient main, mais `--head` ne le contient pas → 1.
6. Clone peu profond (`git clone --depth 1`) → 2.
7. `origin/main` introuvable → 2 ; `--require-fetch` avec remote injoignable → 2.
8. Histoires disjointes → 1.

---

## 3. `cargo xtask parity`

### 3.1 Problème, et la frontière de responsabilité

On veut que « this-rs sert PO comme axum sert PO aujourd'hui », **opposablement** : un inventaire qu'on ne peut pas écrire à la main, des goldens qu'on ne peut pas enregistrer après coup, un critère automatique.

Frontière imposée par le style d'xtask (zéro dépendance, `main.rs:1-9`) :

| Où | Quoi | Pourquoi |
|---|---|---|
| `xtask/src/parity.rs` (std seule) | `inventory`, `verify` : génération d'inventaire, portillons **statiques** (dérive, couverture, immuabilité, exceptions) | texte et git seulement |
| `tests/parity.rs` du backend (a `serde_json`, `tower::ServiceExt::oneshot` — déjà utilisé : `src/auth/middleware.rs`, `src/api/note_handlers.rs`) | **enregistrement et rejeu** HTTP/MCP en mémoire, normalisation | besoin de JSON et du routeur réel |
| `xtask parity record|replay` | lance `cargo test --test parity` avec `PARITY_MODE` / `PARITY_IMPL`, relaie le code | un seul point d'entrée pour la CI |

### 3.2 L'inventaire opposable : `docs/parity/inventory.tsv`

**Généré, jamais édité.** Format ligne à ligne (lisible en diff, parsable en std) :

```
# GENERATED by `cargo xtask parity inventory --write` — DO NOT EDIT
# source routes.rs blob=<git hash-object> handlers.rs blob=<…> tools.rs blob=<…>
# counts routes=483 operations=629 mcp_tools=32 mcp_actions=<N>
R	GET	/api/projects/{slug}/environments	tier=protected	h=environment_handlers::list_environments	src/api/routes.rs:2210
R	POST	/api/projects/{slug}/environments	tier=protected	h=environment_handlers::create_environment	src/api/routes.rs:2210
R	GET	/health	tier=public	h=handlers::health	src/api/routes.rs:350
M	project	list	src/mcp/handlers.rs:221
```
Colonnes `R` : méthode, chemin normalisé (`{x}` conservé), niveau d'auth (`public` / `bounded_public` / `protected` / `websocket`, déduit de la fonction `*_routes()` qui contient l'appel, `routes.rs:68-130`), handler, `fichier:ligne`. `M` : outil, action.

**Génération** (extraction du code, pas saisie) :
- **Routes** : balayer `src/api/routes.rs` (+ `.route(` des deux `Router::new().route(` hors `protected_routes`, `routes.rs:330,341`) : `.route(` → lire le littéral de chemin → lire l'expression jusqu'à la parenthèse équilibrée de `.route(` → relever `get(`/`post(`/`put(`/`patch(`/`delete(` de premier niveau (un `.route(` porte souvent plusieurs verbes : 483 appels → 629 opérations, `08-routes-descripteurs.md:5-8`). Aucun `.nest(` dans `routes.rs` (mesuré : 0) ; si un apparaît, la génération échoue (exit 2) plutôt que d'oublier un préfixe. Les `#[cfg(test)]` (dès `routes.rs:2455`) sont exclus par masquage, via `mask_non_code` déjà écrit (`storage_boundary.rs:141`).
- **Actions MCP** : lire le tableau `MEGA_TOOL_ACTIONS: &[(&str,&str,&str)]` (`src/mcp/handlers.rs:220`) **et** les `enum` d'actions de `src/mcp/tools.rs` (ex. `tools.rs:376-382`), les comparer : toute différence → exit 2. (Un test du backend le fait déjà côté Rust, `handlers.rs:13168`: ne pas le dupliquer, s'y référer.)

**Trois garde-fous de non-écriture-à-la-main** :
1. `inventory --check` régénère et compare octet à octet au fichier commité ; écart → exit 1 (« inventaire périmé : relancer --write »). Le CI ne fait QUE `--check`.
2. **Recomptage indépendant** : un second comptage volontairement naïf (`grep -c '\.route('` hors tests) doit égaler `routes=` de l'en-tête ; un parseur cassé qui perd des routes ne passe pas sans que le naïf bouge aussi.
3. **Preuve d'atteinte au rejeu** (voir 3.5, critère 4) : une route inventoriée dont le rejeu atteint le 404 du routeur (pas celui d'un handler) échoue.

### 3.3 Scénarios, goldens, fichiers

```
docs/parity/
  BASE                         # sha du dernier commit AVANT la première ligne this-rs côté routes (une ligne)
  inventory.tsv                # §3.2, généré
  scenarios/<domaine>.scn      # entrées : écrites à la main (ce sont des ENTRÉES, pas des résultats)
  golden/<id>.json             # sorties : enregistrées, jamais écrites
  divergences.toml             # écarts ACCEPTÉS, par règle (§3.6)
  exemptions.txt               # routes/actions non rejouées, avec raison, liste gelée (§3.7)
```

`scenarios/environments.scn` (exemple, une ligne par appel, JSON compact en fin de ligne) :
```
id=environments.create  route=POST /api/projects/{slug}/environments  seed=base  path.slug=demo  body={"name":"dev","kind":"dev"}
id=environments.list.p2 route=GET  /api/projects/{slug}/environments  seed=many  path.slug=demo  query=limit=2&offset=2  unordered=false
id=mcp.project.list     route=MCP  project/list                       seed=base  args={"limit":2}
```
`seed=` nomme un jeu de départ déterministe (liste d'appels de création, ids générés normalisés §3.4). **Au moins un scénario par route de liste avec `seed=many` (≥ limit+2 lignes) et `offset>0`** (voir §5, piège 3).

`golden/<id>.json` :
```json
{
  "meta": {"id":"environments.create","impl":"axum","recorded_sha":"61b4e229…","tree_clean":true,
           "route":"POST /api/projects/{slug}/environments","masked":3,"leaves":11},
  "request": {"method":"POST","path":"/api/projects/{slug}/environments","body":{"name":"dev","kind":"dev"}},
  "response": {"status":200,"headers":{"content-type":"application/json"},
               "body":{"id":"<uuid:1>","name":"dev","kind":"dev","created_at":"<ts>"}}
}
```

### 3.4 Normalisation des champs volatils (fonction pure, partagée par `record` et `replay`)

`normalize(value, rules) -> value` remplace des **valeurs**, jamais des **clés** (invariant testé) :

| Volatil | Règle |
|---|---|
| UUID v4/v7 | `<uuid:N>`, N = ordre de première apparition dans la réponse **et dans la séquence du scénario** (un id créé puis relu garde le même N, ce qui vérifie la cohérence) |
| Horodatages | toute chaîne RFC 3339 / epoch ms dans une clé terminant par `_at`, `_ts`, `timestamp`, `date` → `<ts>` ; une chaîne RFC 3339 ailleurs → `<ts>` |
| Durées / compteurs de temps | clés `duration_ms`, `elapsed*`, `latency*` → `<num>` (type conservé, pas la valeur) |
| Jetons / secrets | JWT (`eyJ…`), `Bearer …` → `<token>` |
| Chemins temporaires, ports | `/tmp/…`, `/var/folders/…`, `127.0.0.1:<port>` → `<tmp>`, `<addr>` |
| Build | `/api/version` : `version`, `git_sha` → `<build>` |
| En-têtes | **liste blanche** (`content-type`, `location` normalisé, `www-authenticate`, `allow`) ; `date`, `server`, `x-request-id`, `content-length` écartés |
| Ordre | tableaux comparés **dans l'ordre** par défaut ; un scénario déclare `unordered=true` (trié canoniquement) seulement avec une justification en commentaire |

Garde-fous de la normalisation (anti-tautologie) : (a) `normalize` conserve exactement le même ensemble de chemins de clés (test de propriété) ; (b) le golden stocke `masked`/`leaves` ; le portillon échoue si `masked/leaves > 0,4` pour un scénario non marqué `volatile=high` (une normalisation qui efface tout compare le vide) ; (c) un type ne change pas de nature : `"<num>"` n'est écrit que sur un nombre, `<ts>` que sur une chaîne.

### 3.5 Enregistrement et rejeu

**Enregistrement** — `cargo xtask parity record --impl axum` :
- refus (exit 2) si `routing_impl` rapporté par le binaire de test ≠ `axum`, si l'arbre est sale, ou si `HEAD` n'est pas ancêtre-ou-égal de `BASE` (`git merge-base --is-ancestor HEAD $(cat docs/parity/BASE)`) : **on n'enregistre que depuis l'implémentation d'origine, avant la première ligne this-rs** ;
- construit le routeur par le constructeur de production (`create_router(state)`, `src/api/routes.rs:68`) avec le magasin simulé (`features = ["mock-store"]`, `Cargo.toml:301`) — jamais un routeur « de test » à part ;
- joue chaque scénario dans l'ordre, normalise, écrit `golden/<id>.json` avec `meta.impl`, `recorded_sha`.

**Rejeu** — `cargo xtask parity replay --impl this` : construit le routeur this-rs **par le constructeur de production aussi**, rejoue les mêmes scénarios, mêmes seeds, mêmes normalisations ; compare statut, en-têtes retenus, corps normalisé, **avec l'adaptateur de divergence de production appliqué** (§3.6). `--impl axum` est aussi rejouable : il doit rendre 0 écart contre ses propres goldens (preuve que le rejeu est déterministe ; sinon le bruit est pris pour de la dérive).

**Critère d'échec CI (exit 1)** — l'un quelconque :
1. `inventory --check` en écart ;
2. une opération `R` ou `M` de l'inventaire **sans scénario et sans ligne d'exemption** ;
3. un scénario dont le rejeu diffère d'un golden (statut, en-têtes retenus ou corps) et dont l'écart n'est couvert par **aucune** règle de `divergences.toml` ;
4. une opération qui atteint le 404 **du routeur** (corps vide, pas de `content-type`) alors que son golden est un statut de handler (route non montée) ;
5. un golden modifié ou ajouté depuis `BASE` : `git log $(cat docs/parity/BASE)..HEAD --format=%H -- docs/parity/golden/` **non vide** ;
6. une règle de `divergences.toml` qui n'a servi **aucune fois** (périmée), ou qui s'est appliquée à plus d'opérations que son `max=` ;
7. une exemption nouvelle par rapport à `origin/main` (liste gelée, comme `storage-boundary-exceptions.txt`) ou inutilisée ;
8. `masked/leaves` > 0,4 (3.4).

Exit 2 : arbre introuvable, seed non reproductible, `cargo test` n'a pas produit de rapport.

### 3.6 Les ruptures de contrat déjà mesurées : un registre de règles, pas un joker

`08-routes-descripteurs.md:36-37` : (a) PO rend `{items,total,limit,offset}` (`backend/src/api/query.rs:163-171`) quand this-rs rend `{data,pagination:{page,limit,total,total_pages}}` (`descriptor_entity.rs:160-170`) ; (b) création : PO 200 (`environment_handlers.rs:147,184`), this-rs 201 (`descriptor_entity.rs:178`). Et côté requête : PO pagine par `limit`/`offset`, this-rs par `page`/`limit`.

Ces écarts ne sont **pas** « tolérés » globalement ; ils sont déclarés une fois, avec leur forme exacte :

```toml
[[divergence]]
id        = "list-envelope"
decision  = "adapter"            # "adapter" (la couche de sortie de prod le rend à l'identique) | "frontend-follows" (le contrat change, goldens versionnés par la décision, voir ci-dessous)
match     = "GET list: this={data,pagination:{page,limit,total,total_pages}} vs golden={items,total,limit,offset}"
map       = "items=data; total=pagination.total; limit=pagination.limit; offset=(pagination.page-1)*pagination.limit"
request_map = "offset -> page=(offset/limit)+1"
max       = 140                  # nombre d'opérations autorisé à l'utiliser ; plafond mesuré au gel
decided_in = "decision <uuid>"   # décision PO obligatoire (sinon exit 2 au chargement)

[[divergence]]
id       = "create-status"
decision = "adapter"
match    = "POST create: this.status=201 vs golden.status=200, body identique"
map      = "status 201 -> 200"
max      = 60
decided_in = "decision <uuid>"
```

Règles d'application :
- Le rejeu applique `map` **par la même fonction que celle montée en production** (une couche de sortie `ResponseAdapter` branchée au serveur this-rs par `with_custom_routes`/couche, `08:20`) — jamais une transformation vivant seulement dans le test. Sinon on compare un faux.
- Un écart couvert doit correspondre **exactement** : 201 vs 200 ne couvre ni 202 ni un corps différent.
- `decision = "frontend-follows"` : le golden n'est PAS réécrit (règle 5 du critère) ; la divergence porte `new_shape` et le rejeu compare à la **forme déclarée** ; frontend et fixtures de contrat (`tests/chat_event_golden.rs`, `tests/fixtures`) doivent être mis à jour dans la même PR, ce que le critère 3 prouve (le frontend lit la forme).
- Le plafond `max=` est la barre d'une mesure : il ne peut qu'baisser (comparé à `origin/main`).

### 3.7 Exemptions (liste gelée)
`docs/parity/exemptions.txt`, une ligne `GET /health  # raison`. Autorisées : WebSocket (pas de rejeu HTTP en mémoire), routes non déterministes (`/api/version`), routes de fichiers statiques. Une ligne ajoutée après le gel (comparée à `origin/main` via `git show origin/main:docs/parity/exemptions.txt`) fait échouer ; une ligne dont l'opération est couverte aussi (exception inutilisée).

### 3.8 Tests de `parity` (xtask : pures ; backend : intégration)
1. `parse_routes` : `.route("/a", get(h1).post(h2))` multi-lignes → 2 opérations ; `#[cfg(test)]` exclu ; `.nest(` → erreur.
2. `inventory --check` détecte l'ajout d'une route dans la source de fixture.
3. `normalize` : même ensemble de clés avant/après ; UUID `<uuid:1>` stable à l'intérieur d'un scénario, distinct entre deux ids.
4. **Mutation** (`parity_canary`) : une réponse mutée (clé renommée ; statut changé ; tableau permuté) doit être refusée par le comparateur — preuve que le rejeu *peut* échouer.
5. **Rouge sans, vert avec** : supprimer une route de la fixture de routeur → critère 4 échoue ; la remettre → passe.
6. Règle `list-envelope` : `{data,pagination}` accepté face à `{items,…}` ; `{data}` sans `pagination` refusé.
7. Règle `create-status` : 201 accepté pour POST create ; 201 pour un GET refusé.
8. Golden touché après `BASE` (dépôt jetable) → critère 5.

---

## 4. Branchement CI et articulation avec les 6 checks requis

Les 6 checks requis sur `main` : **Build, Lint & Format, Unit Tests, Integration Tests, Code Coverage, API Tests** (noms des jobs `ci.yml:175,297,452,544,639,750`). Les paliers : un PR vers `integ/**` ne passe que le **palier rapide** (lint, sécurité, unitaires, intégration) ; `Build`, `API Tests`, `Code Coverage` sont **palier complet** (PR prêt vers `main`, push sur `main`, nuit) — `ci.yml:3-8,180,414,643,755`.

| Portillon | Job | Quand | Pourquoi là |
|---|---|---|---|
| `pins --path-deps --no-fetch` | **étape de `Lint & Format`** (comme `storage-boundary`, `ci.yml:356-357`) | tout PR, palier rapide | statique, 1 s ; rapide = on voit la dérive avant d'attendre le build |
| `containment --require-fetch --head ${{ github.event.pull_request.head.sha }}` | **nouvelle étape de `Lint & Format`**, `if: github.event_name == 'pull_request' && startsWith(github.head_ref,'consolidation/')` | PR vers la branche de consolidation ; PAS sur `main` (où HEAD = main) | une branche longue qui ne voit jamais main devient infusible (`CONSOLIDATION.md`, règle 2) ; `fetch-depth: 0` pour l'étape |
| `parity inventory --check` + `parity verify` (statique) | **étape de `Lint & Format`** | tout PR touchant `src/api/**`, `src/mcp/**`, `docs/parity/**`, `xtask/**` (filtre `paths`) | pas de service externe ; échoue vite |
| `parity replay --impl axum` (déterminisme) puis `--impl this` | **nouveau job `Parity`**, `needs: [lint]`, palier rapide si mock-store, complet sinon | PR vers la branche de consolidation et vers `main` dès que le premier code this-rs y est | magasin simulé → pas de Neo4j/Meilisearch ; ne pas gonfler `API Tests` (qui exige services et build, `ci.yml:640-748`) |

**Articulation avec les six** :
- **Pas de 7ᵉ check requis tout de suite** : `Lint & Format` **contient** pins, containment et la partie statique de parity ⇒ ils sont requis *par héritage*, sans toucher à la protection de branche (qui énumère les noms de jobs). C'est volontaire : ajouter un check requis qui n'existe pas encore sur `main` bloque tous les PR.
- `Parity` (rejeu) est ajouté aux checks requis de **la branche de consolidation** dès sa création, et de `main` **seulement** au moment où la porte de bascule (CONSOLIDATION.md, lignes a-e) l'exige ; avant, il reste informatif sur `main`.
- `API Tests` (suite d'intégration réelle contre le serveur compilé, `ci.yml:743-748`) et `Parity` ne se remplacent pas : l'un valide un serveur contre un vrai Neo4j, l'autre valide l'**équivalence de contrat** entre deux routeurs. Un `Parity` vert ne dispense pas `API Tests`.
- `Code Coverage` : les nouveaux tests `tests/parity.rs` entrent dans le total ; la couverture des fonctions pures d'xtask n'est pas mesurée (`xtask` hors workspace) — d'où les tests unitaires obligatoires ci-dessus.
- Fusion en série (règle 7 du portillon des PR) : après chaque fusion, `containment` est rejoué sur la suivante, ce qui est précisément sa fonction.
- Le `schedule` nocturne existant (`ci.yml:14`) rejoue `parity` complet avec `--no-cache` : détecte le bruit non déterministe qu'aucun PR n'a heurté.

---

## 5. Le piège : comment ces portillons peuvent être verts sans rien garantir

1. **Contrôle vert parce que le CI checkoute le pin.** Si le job de build fait `actions/checkout ref: <rev codé dans ci.yml>` puis lance `pins --path-deps`, le contrôle compare le pin à lui-même. **Garde-fou** : le CI lit le `rev` DANS `pins.toml` (`--print-checkout`) — une seule source ; la valeur du contrôle est locale et à la montée d'un pin (test 6).
2. **`containment` toujours vert en PR** parce que `refs/pull/N/merge` contient déjà `main`. **Garde-fou** : `--head <sha réel de la PR>` obligatoire dans le workflow ; test 5 de §2.5 le reproduit ; l'option `--head` n'a pas de défaut CI.
3. **Rejeu sur jeu trop petit** : la liste de 3 éléments renvoie la même chose avec ou sans `offset` : `{items:[…3]}` ↔ `{data:[…3]}` passe alors que this-rs ignore `offset`. **Garde-fou** : `seed=many` ≥ `limit+2` et `offset>0` exigés pour toute route de liste (critère de couverture) ; le comparateur échoue si deux scénarios d'une même route donnent le même corps normalisé.
4. **Golden enregistré après la migration** : on enregistre depuis this-rs, le rejeu compare this-rs à lui-même. **Garde-fou** : `record` refuse hors `impl=axum` et hors ancêtre de `BASE` ; le critère 5 interdit toute modification de `golden/` après `BASE` (`git log BASE..HEAD -- docs/parity/golden/` doit être vide) ; `BASE` lui-même sous CODEOWNERS.
5. **Inventaire écrit à la main** qui oublie justement les routes difficiles. **Garde-fou** : généré, `--check` octet à octet, recomptage `grep -c` indépendant, critère 4 (route inventoriée non montée).
6. **Normalisation qui vide la comparaison** (tout est `<…>`). **Garde-fou** : invariants de clés, ratio `masked/leaves`, test de mutation (`parity_canary`).
7. **Divergence-joker** (« 201 accepté ») qui avale aussi les vraies régressions. **Garde-fou** : règles par forme exacte, `max=` décroissant, règle inutilisée = échec, `decided_in` obligatoire.
8. **Adaptateur existant seulement dans le test** : le rejeu applique une conversion qui n'existe pas en production. **Garde-fou** : `ResponseAdapter` unique, monté par le constructeur de production, importé par le test.

Un portillon qu'on n'a jamais vu rouge n'est pas un portillon : chaque garde-fou ci-dessus a son test « rouge sans, vert avec » dans §1.5, §2.5, §3.8.

## 6. Ordre de réalisation proposé
1. `containment` (le plus petit, 1 module ; protège déjà la branche).
2. `pins --path-deps` + `pins.toml` + test de silence initial (à fusionner avant le premier `path = "../this"`).
3. `parity inventory` + `verify` (statique), `BASE` posé sur le dernier commit avant tout code this-rs.
4. `tests/parity.rs` + `parity record --impl axum` **avant** O-THISRS-1 ; commit des goldens ; `parity replay --impl axum` vert.
5. Job `Parity` ; ajout du registre de divergences quand les premiers écarts de §3.6 sont mesurés sur this-rs.
