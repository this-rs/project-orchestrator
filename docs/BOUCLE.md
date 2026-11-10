# La boucle — comment ce chantier avance

Ce fichier décrit la mécanique, pas les intentions. Il a été écrit après une
journée où six défauts de méthode se sont produits ; chacun est devenu une
règle avec sa preuve obligatoire.

## Forme de la boucle

1. **Choisir** le lot suivant dans le backlog ci-dessous (ordre = dépendances, pas envie).
2. **Déléguer** avec un brief qui contient : la question, les sources exactes, le
   livrable, les contraintes, et la preuve attendue.
3. **Vérifier moi-même** — jamais sur le message de l'agent.
4. **Intégrer** : fmt (deux workspaces), clippy, tests ciblés, commit lisible.
5. **Pousser** et lire la CI.
6. **Mettre à jour** l'état ici et dans PO.

## Limite de parallélisme

- **3 agents au maximum**, et **un seul par crate**. Deux agents dans le même
  crate se marchent dessus : mesuré le 10/11, deux agents ont écrit dans
  `src/events` et `src/neo4j` en même temps, l'un a vu l'arbre de l'autre en
  pleine écriture et a conclu que « la branche ne compile pas ».
- **Un seul build lourd à la fois** (disque à 94 %). `cargo check` avant tout
  `cargo test` ; tests ciblés, jamais la suite entière en boucle.

## Les preuves obligatoires, par lot

| Preuve | Pourquoi elle existe |
|---|---|
| `git status --porcelain` collé à la fin | Un agent a rendu un rapport complet — mécanisme, 10 gardes, 4 tests, sorties rouge/vert — et **rien n'était sur le disque**. Il avait travaillé dans une copie scratch. |
| `cargo fmt --all` **et** `cargo fmt --manifest-path xtask/Cargo.toml` | `xtask` a son propre workspace : la racine ne le voit pas, la CI le vérifie. Un rouge CI entièrement dû à ça. |
| `clippy -D warnings` sur les deux workspaces | Même raison. |
| Rouge sans / vert avec, rejoué | Un test qui reste vert sans le correctif est tautologique. On le prouve en neutralisant la production. |
| `git diff -w` pour un déplacement | Un « pur déplacement » peut reflow et re-indenter ; on écrit le chiffre exact. |

## Chemins interdits

- `/Users/triviere/projects/project-orchestrator/backend` est un **lien** vers le
  service vivant. L'écrire, c'est modifier le service en cours d'exécution.
  Tout travail passe par un worktree sous `worktrees/`.
- Ne jamais travailler dans une copie scratch (`/tmp`), ou alors la détruire et
  dire pourquoi.

## Backlog ordonné

### Bloquants du target
1. **Harnais, famille A** — `gate` sur le hook de permission du moteur, `Recorder`
   hors-bande. C'est le cœur de la parité et le plus gros morceau.
2. **ChatTransport, les 16 sites restants** (le trait existe, 10 sites migrés).
3. **Bench d'insertion obrain** — condition de fusion de la PR #99.
4. **Journal durable** (CDC obrain) puis **retrait de NATS**.

### Gros morceaux suivants
5. Routes par lots de domaine (38 % descriptibles, 62 % verbes).
6. Auth : double vérification WAMI / HS256, réémission des jetons.
7. Migration 4,5 M nœuds, aller-retour vérifié.
8. Bascule du défaut + `O-FLIP`.

### Ce qui n'est dans aucun plan — à ne pas perdre
9. **~2 300 sites de décodage `BoltType`** : le coût caché le plus lourd de la migration.
10. **282 `ORDER BY` avec tri départagé** : non spécifié, casse une UI en silence.
11. **Trajectory store ouvert en dur** dans la fabrique, hors sélecteur de backend.
12. **Chien de garde inactif en wasm** (`Clock::sleep` rend `None`).
13. **Lint de liste noire wasm** : sans lui, une dépendance de plateforme rentre en trois semaines.
14. **`DROP CONSTRAINT` no-op** dans obrain.
15. **Fenêtre de rétention du CDC** jamais mesurée (~50 k événements, ~1 h 23 à 10 mutations/s).
16. **6 capacités encore advisory** sur 9.

## Ce que cette boucle ne fait pas

Elle ne promet pas de date. Elle garantit qu'aucun lot ne se déclare fini sans
la preuve qui va avec — et que ce qui est oublié est écrit quelque part plutôt
que perdu dans une conversation.

## Recette : le conflit `manager.rs` à chaque fois que `main` touche le chat

Arrivé deux fois le 10/11 (#1152, puis #1154). `main` modifie le code de
production de `manager.rs` ; nous avons retiré ses 20 198 lignes de tests.
Git ne peut pas aligner : il produit **un** hunk de conflit qui couvre tout.

Procédure qui marche, dans cet ordre :

1. `git checkout --ours src/chat/manager.rs` — garder notre version (tests
   extraits, 13 déclarations `#[path]`).
2. `git diff origin/main~1 origin/main -- src/chat/manager.rs > /tmp/mgr.patch`
   puis `patch -p1 --forward < /tmp/mgr.patch` — applique les hunks de
   production, laisse un `.rej` pour le bloc de test.
3. Dans le `.rej` : ne garder que les lignes `+`, retirer le `+`, dé-indenter
   de 4 espaces, et insérer ce bloc **à la fin du test nommé** dans
   `src/chat/manager/tests/tests.rs`. Le nom du test est celui qu'on trouve
   en remontant depuis la ligne du conflit dans la version de `main`.
4. `rm` le `.rej`, `cargo fmt --all`, puis vérifier qu'il ne reste **aucun**
   marqueur `<<<<<<<`.
5. Le critère de réussite n'est pas « ça compile » : c'est que le test importé
   de `main` s'exécute et passe.

Coût mesuré : ~3 minutes. Ce qui le rend supportable, c'est que le portillon
`containment` le réclame immédiatement au lieu de laisser la dette s'accumuler.

## Objectifs suivis (mis à jour à chaque fin de vague)

Cinq objectifs, chacun avec son état mesuré — pas une impression. Un objectif
ne passe à ✅ que sur une preuve reproductible.

### O1 — Contrat verrouillé sur toute la surface
- ✅ Inventaire généré (483 `route()`, 629 opérations, 427 actions MCP) + portillon `xtask parity`
- ✅ Contrat filaire du chat : 41 variantes, 5 scénarios, comparateur positionnel
- 🟡 Goldens **enregistrés** contre un vrai serveur — le comparateur existe, l'enregistrement reste `#[ignore]`
- 🟡 DTO TypeScript engendrés — portillon `xtask ts-dto` (premier lot : Note, Task, TaskStatus, Project, Plan + fermeture = 25 types dans `frontend/src/types/generated/`) ; le basculement d'usage et les 502 DTO restants = lots suivants
- 🔴 Angle mort : le contrat REST n'est gardé nulle part hors chat

### O2 — Harnais à parité
- ✅ Famille A : `gate`, `Recorder`, `specs` retrouvés (PR this-rs #230)
- ✅ Cœur wasm32 compile (12,19 s) ; `?Send` tranché par thread + LocalSet
- ✅ `ChatTransport` : plus aucun consommateur ne connaît NATS
- ✅ Sélecteur `events.transport` (`nats`|`journal`|`memory`) + câblage au boot : le journal est instanciable par configuration (chemin + rétention annoncés au démarrage), NATS reste le défaut
- ⬜ Journal durable (CDC obrain) puis retrait de NATS
- ⬜ Transports serveur (nexus, HTTP) en features d'hôte

### O3 — obrain comme stockage par défaut
- ✅ Vectoriel tranché : balayage exact, recall 1,0000, p95 1,5-3,1 ms
- ✅ Unicité composite corrigée + bench (PR obrain #99)
- ✅ `StoreCapabilities` refuse au lieu de se taire
- ⬜ Migration 4,5 M nœuds, aller-retour vérifié
- ⬜ Fenêtre de rétention du CDC mesurée
- ⬜ Bascule du défaut + retour arrière par configuration

### O4 — PO démontre this-rs + obrain
- ✅ Module `plan` : descripteur + CRUD sans stockage + vues ; **le critère passe** (ajouter un champ ne touche que le descripteur)
- ⬜ L'adaptateur de stockage `Request/Effect` ↔ neo4j — **le vrai coût, non payé**
- ⬜ Un écran rendu via this-ui
- ⬜ La décision du front, une fois la preuve faite (engendrer vs réécrire)

### O5 — Hygiène de branche
- ✅ `containment` + `path-pins` + `strict: true` sur main
- ✅ CI de la branche déclenchée au push (`consolidation/**`)
- 🟡 CI verte : trois rouges successifs, tous sur des règles du dépôt (xtask fmt, clippy de test, plafond de diagrammes)
- ✅ `main` reversé dans la branche à chaque fois que le portillon l'a exigé (3 fois)

**Règle de suivi** : à chaque fin de vague, un objectif ne peut changer d'état
que si la preuve est dans le dépôt. Un ⬜ qui devient 🟡 sans commit n'est pas
un progrès.

## Surveiller le disque, avant chaque vague

Mesuré une fois à **9,7 Go libres sur 3,6 To** — un agent obrain s'est arrêté
en plein travail, correctement. La cause n'était pas les `target/` visibles
mais **106 Go de caches `.cible-*` cachés** dans les worktrees, dont un de
59 Go.

**Avant de lancer une vague** :
```sh
df -h /Users/triviere | tail -1
du -sh <worktree>/.cible-* <worktree>/target 2>/dev/null | sort -h | tail -5
```
- **sous 40 Go** : ne rien lancer, nettoyer d'abord.
- **sous 20 Go** : les agents doivent s'arrêter, et le dire — pas continuer en
  espérant.

**Ce qui se nettoie sans risque** : les `target/` et `.cible-*` des worktrees
dont le travail est **fusionné ou poussé**. Ce qui ne se nettoie pas : le cache
d'un worktree sur lequel quelqu'un travaille — le supprimer coûte 30 à 40 min
de reconstruction à cet agent.

**Le piège** : `du -sh dossier/*` ne voit pas les `.quelquechose`. Utiliser
`du -sh dossier/.[!.]*` en plus, sinon 106 Go restent invisibles.
