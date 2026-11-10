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
