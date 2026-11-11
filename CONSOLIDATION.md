# Branche de consolidation — Obrain par défaut + this-rs comme framework

Cette branche (`consolidation/obrain-default`) est le lieu unique du chantier.
Plan PO : `bf3ab074-467e-4b34-b92e-4b19040d8ece` (projet PO Backend).
Plan jumeau pour la migration des données : `250c63fb-0298-4980-bb16-7e6d3773b5d7`.

**État au 11/11/2026, 14 h UTC.** Ce fichier dit ce qui est fait, ce qui est en cours et ce qui ne l'est pas. Une ligne « en cours » nomme la PR ou la branche qui la porte ; une ligne sans preuve reste « à faire ».

## Règles de la branche

1. **Rien ne remonte dans `main` avant que la porte de bascule soit verte.** Aucune PR vers `main` n'est ouverte tant que ce fichier dit autre chose.
2. **`main` redescend ICI** (`git merge origin/main` dans cette branche), jamais l'inverse. Une branche qui ne voit jamais `main` pendant des mois devient infusible : c'est exactement ce qui est arrivé à `consolidation/obrain` (2 commits, abandonnée). Dernière descente : #1156 incluse.
3. **Chaque agent travaille dans son propre worktree**, sur une branche fille de celle-ci, et ouvre sa PR **vers cette branche**. Jamais dans un checkout principal, jamais directement sur cette branche.
4. **Un lot = un domaine.** Un commit qui touche stockage ET routes n'est pas relisible : on le coupe.
5. **Aucun défaut connu ne reste « hors périmètre ».** Ce qu'une revue relève est corrigé, dans la PR ou dans une PR empilée ; il n'est pas listé pour plus tard.
6. **Une CI verte n'est pas une revue.** Le 11/11, trois PR obrain à CI verte portaient chacune un défaut de justesse que seule une relecture indépendante a trouvé. Toute PR passe une relecture indépendante AVANT fusion.

## Barre de performance (contrainte du plan)

L'attente est la **nanoseconde sur des millions de nœuds**, pas la centaine de millisecondes sur 200 000.

- Une recherche indexée, un décompte, un contrôle d'unicité ne croissent pas avec la taille de la base.
- Aucune reconstruction d'index, migration de format ni balayage proportionnel à N sur le chemin d'une requête, d'une écriture ou d'une ouverture.
- Toute PR qui touche un chemin chaud livre ses mesures à 1 M et 5 M de nœuds (médiane et p99, charge notée), avant et après. Un coût qui croît avec N est un défaut.

## Transferts assumés (décisions consignées)

- **Les traits `GraphStore` de PO restent la couche de domaine.** Les services de stockage de this-rs n'ont ni agrégation, ni tri serveur, ni recherche vectorielle : le backend obrain s'écrit DANS PO, domaine par domaine. De this-rs on reprend le bail d'écriture et l'index UUID → NodeId, pas l'interface.
- **Obrain porte un index vectoriel à dimension native.** Mesure du 10/11 : `create_vector_index(dim=768)` refusé (L2_DIM = 384) ; cascade 384-d recall@10 0,63 à N=8 131 contre 0,94 pour Neo4j ; latence obrain 0,30 ms contre 1,27-2,05 ms.
- **PO est l'écrivain unique de la base obrain.** La v0.0.1 ne garantit pas les propriétés, les transactions ni la visibilité inter-processus.
- **NATS disparaît.** EventBus this-rs + journal d'événements durable dans obrain + WS/SSE. Conséquence assumée : topologie mono-instance pour le chat, plus d'interrupt cross-instance.
- **this-rs devient le framework**, intégration complète, par lots. MCP JSON-RPC reste hors framework.
- **La migration des données se fait au démarrage**, avant toute écriture : PO démarre en lecture seule, migre, vérifie par type, relance sur le nouveau backend.

## Porte de bascule (« ça fonctionne complètement »)

| # | Condition | État |
|---|---|---|
| a | Conformité générique identique sur Neo4j et Obrain (mêmes valeurs, même ordre) | **en cours** — premier domaine sur la branche `consolidation/o-double` (pas encore de PR) |
| b | recall@10 ≥ 0,95 à 25 000 vecteurs 768-d, après réouverture | en cours côté obrain (plan O-VEC), non vérifié ici |
| c | Graphe migré, aller-retour vérifié | **en cours** — source figée et référence par type faites (voir plus bas) ; migrateur en spike |
| d | Double-run sans divergence de justesse | à faire |
| e | Retour arrière = une variable de configuration | à faire — la fabrique `src/storage.rs` est le point unique, la variante obrain arrive avec (a) |

## État des lots

### Fait sur cette branche

| Lot | Tâche PO | Preuve |
|---|---|---|
| Portillons `containment` et `path-pins` | O-BRANCHE | `xtask/`, job « Lint & Format » |
| Inventaire opposable de la surface (483 routes → 629 opérations, 427 actions MCP) | O-PARITE (partiel) | `docs/parity/inventory.tsv`, généré |
| Contrat filaire du chat (41 variantes, comparateur positionnel) | O-GOLDEN-CHAT (partiel : goldens pas encore enregistrés contre un vrai serveur) | `docs/parity/chat/`, `tests/parity_chat.rs` |
| Tests de `manager.rs` sortis du fichier (532 avant = 532 après) | O-TESTS-EXTRACT | `src/chat/manager/tests/` |
| Transport de chat extrait de NATS : un trait, trois implémentations | O-CHAT-TRANSPORT | `src/events/chat_transport.rs` ; plus aucun `Arc<NatsEmitter>` hors `src/events/` |
| Reconnexion par le journal, plus par Neo4j | O-REPLAY | commit `4794762e`, 4 tests par la route WS |
| Périmètre des traits : ce que Neo4j / Meilisearch / NATS font vraiment | O-PERIMETRE | `docs/cartographie/14-perimetre-traits.md` |
| Garde-fou des tris non totaux | O-TRI (partiel) | `tests/order_tiebreak.rs` |
| Cartographie (19 documents) | — | `docs/cartographie/` |

### En cours

| Lot | Tâche PO | Où |
|---|---|---|
| Backend obrain derrière la fabrique : `BackendKind::Obrain`, un premier domaine, refus nommé pour le reste, conformité croisée | O-DOUBLE | branche `consolidation/o-double` |
| Mode Auto du harnais porté dans `this-harness` (cœur de décision pur, compatible wasm) | O-HARNESS-PARITE | this-rs, branche `feat/harness-mode-auto` |
| Module `plan` : descripteur, CRUD sans base, adaptateur, écran dérivé | O-MODULE-PLAN | this-rs `main` (#234, #235, #236) ; PO ne le consomme pas encore |
| Référence de migration et copie figée | plan 250c63fb | PR #1163 (`docs/cartographie/20`, `21`, `22`) |
| Spike du migrateur : débits de lecture et d'import, fidélité par type et par propriété | plan 250c63fb | hors dépôt (`worktrees/spike-migrateur`) |

### À faire, dans l'ordre du chemin critique

1. Fusionner les correctifs obrain dont PO dépend (tableau suivant).
2. **O-BASE** — remonter l'épingle obrain de `pins.toml`, dans le commit qui introduit la première dépendance.
3. **O-TRAITS** (reliquat) — `GraphStore` exige ses 20 sous-traits en bloc : un backend partiel doit fournir ~730 méthodes d'un coup.
4. **Magasin de trajectoires** — ouvert en dur sur Neo4j dans `src/storage.rs`, hors de la fabrique ; aucune tâche ne le portait.
5. **O-GRAPH** — portage par vagues (voir plus bas), puis **O-SEARCH** (27 méthodes de `SearchStore`).
6. **O-CLONE → O-MIGRATE → O-SOAK → O-FLIP.**

Hors chemin critique : **O-THISRS-1** (socle this-rs monté sous `/_this` à côté d'axum ; plan écrit, aucun conflit de version d'axum/tower/tokio ; la dépendance par chemin de this-rs vers obrain et vite-rs est le vrai risque en CI), O-THISRS-2/3, O-NATS-OUT, O-COG, O-REACTIVE.

## Ce dont PO dépend dans obrain (dépôt `obrain-db/obrain`)

Aucune de ces PR n'est fusionnée au moment où ce fichier est écrit.

| PR | Sujet | État |
|---|---|---|
| #99 | `UNIQUE (a, b)` porte sur le tuple ; contraintes persistées | CI verte ; revue : cœur juste, trois défauts introduits (contraintes qui s'accumulent à chaque redéclaration, `DROP CONSTRAINT` sans effet, régression sur `MERGE … ON CREATE SET`) — en correction |
| #101 | Recherche par la propriété la plus sélective ; clé d'index cohérente avec l'égalité (`5` = `5.0`) | deux revues ; justesse corrigée ; ne tient pas encore la barre de performance (reconstruction d'index à la migration, décompte non constant) — en correction |
| #108 | Un agrégat dans `WITH` ne rend plus la clé de groupement invisible (104 requêtes de PO) | bloquée : étend à `WITH` un résultat faux sur deux clés calculées — en correction, avec quatre autres résultats faux Cypher/GQL et `duration.between` |
| #110 | Les tests n'ouvrent plus les bases réelles sous `$HOME` | objectif tenu ; revue : le mode « lecture seule » du moteur écrit — les lecteurs passent par une copie, en correction |
| (à venir) | Mode lecture seule réel ; unicité vérifiée sur `SET`, `MERGE … ON MATCH`, ajout de label, insertion concurrente, import en masse | en cours |

Mesuré sur `main` d'obrain (`c4b6ada9`) : `MATCH (n) WITH n.p AS p, count(*) AS c RETURN p, c` échoue en Cypher (`Undefined variable 'p'`) et passe en GQL ; un second `MATCH` dont la cible est déjà liée rend 36 lignes au lieu de 14 en Cypher. Tant que #108 et ses suites ne sont pas fusionnées, le backend obrain de PO passe par l'API Rust ou par des requêtes simples.

## Migration des données : ce qui est établi

- **Source figée** : copie du volume Neo4j vivant monté en lecture seule, prise le 11/11 à 11:36 UTC, servie par un Neo4j jetable (ports 27474 / 27687). **3 901 702 nœuds, 6 013 360 relations**, 64 labels peuplés sur 70, 80 types de relation sur 82, un seul label par nœud. Deux méthodes de comptage indépendantes concordent. Détail : `docs/cartographie/22-copie-figee.md` (PR #1163).
- **Convertisseur existant** (`obrain/neo4j2obrain`) : ne satisfait pas « aucune perte » (nulls retirés des listes, dates aplaties en texte).
- **Volume** : la cible obrain du seul sous-ensemble sans `ChatEvent` (~1,2 M nœuds) pèse 16 Go, contre 14,6 Go pour la source Neo4j entière. À expliquer avant la migration.

## Portage de `GraphStore` : vagues

Compté dans le code : 724 méthodes à porter, dont 198 dures (agrégats dans `WITH`, `UNWIND`, sous-requêtes, chemins variables) ; ~84 sans appelant de production détecté.

| Vague | Domaines | Méthodes (dont dures) |
|---|---|---|
| V0 — socle, un seul agent | backend dans la fabrique, refus nommés, schéma, aides de décodage et de dates, banc de conformité | — |
| V1 | Project, Chat-noyau, Auth, Protocol, Trigger, Federation | 108 (14) |
| V2 | Workspace, Plan, Release, ProtocolRun, PlanRun, Analytics-précoce | 208 (27) |
| V3 | Chat-reste, Decision, Commit, Note, Sharing, Code | 209 (79) |
| V4 | FeatureGraph, Skill, Persona | 72 (20) |
| V5 | Analytics en 13 sous-lots, cascade de suppression de projet | 127 (58) |

Les domaines d'une vague se portent en parallèle à condition d'écrire un fichier par sous-trait côté obrain. **Premier jalon visible : un chat de bout en bout sur obrain = 44 méthodes** (Project 4, Chat 40), toutes en V1 ; +17 si l'authentification est active (estimation par graphe d'appels, pas par exécution).

Trois risques chiffrés : `AnalyticsStore` n'est pas un domaine (145 méthodes sur 21 fichiers, lit 14 domaines, écrit chez 9) ; 78 fonctions enchaînent au moins deux écritures sans transaction, et `init_schema` pose 189 instructions dont 7 contraintes composites ; ~1 960 sites de décodage `neo4rs`, avec des dates stockées tantôt en type temporel, tantôt en chaîne.

## Sécurité des données (incidents du 11/11)

- **Les tests d'obrain ouvraient les bases réelles** sous `~/.obrain/db` en écriture (`cargo test -p obrain-engine` sur `main`). Tant que #110 n'est pas fusionnée : toute commande `cargo test` / `cargo bench` d'obrain se lance avec un `HOME` vide.
- **Le mode « lecture seule » d'obrain écrit** (rejeu du journal dans les zones, reconstruction des degrés, création de fichiers). Tant que ce n'est pas corrigé : une base réelle ne se sonde que sur une copie, y compris avec `--read-only`.
- **PO** : un binaire de test a ouvert la base vivante parce que les clés de `config.yaml` sont imbriquées. Les tests du backend obrain n'ouvrent que des répertoires temporaires.

## Les portillons de cette branche (en place)

| Portillon | Ce qu'il empêche | Commande |
|---|---|---|
| `containment` | la branche qui dérive de `main` pendant des mois sans que personne ne le voie | `cargo xtask containment` |
| `path-pins` | un voisin (this, obrain, atelier) qui bouge sous les pieds sans qu'aucun diff ne le montre | `cargo xtask path-pins` |

Les deux tournent dans le job « Lint & Format », donc ils sont bloquants par héritage — sans toucher à la protection de branche de `main`.

`pins.toml` dit contre QUELS commits on construit. Mettre un `rev` à jour est un acte délibéré, dans le même commit que le besoin qui le motive. **État au 11/11** : `path-pins` est rouge EN LOCAL, parce que les checkouts voisins ont avancé (this-rs de 9 commits, obrain de 20) alors qu'aucune dépendance par chemin n'existe encore dans PO ; il sort 0 en CI, où les voisins sont absents. Les épingles se remontent avec la première dépendance réelle.

`path-pins` ne fait pas de `git fetch` par défaut : un voisin dont le remote est en ssh bloquerait le portillon. `--fetch` l'active.

## À savoir avant d'écrire une ligne ici : `backend/` est un LIEN

`/Users/triviere/projects/project-orchestrator/backend` est un lien symbolique vers `/Users/triviere/.openclaw/workspace/skills/project-orchestrator`, qui est le **répertoire du service vivant**.

Conséquences, mesurées le 10/11/2026 :
- La « copie principale » du dépôt n'est pas `backend/` : c'est ce répertoire-là. `git rev-parse --show-toplevel` depuis `backend/` le dit.
- Éditer `backend/` **c'est** éditer le service vivant. La consigne « ne jamais toucher `~/.openclaw/workspace/skills/project-orchestrator` » est donc impossible à tenir en même temps que « éditer `backend/` » : ce sont les mêmes fichiers (même inode).
- Le `pre-push` partage `CARGO_TARGET_DIR` avec cette copie principale : un build lancé depuis un worktree écrit dans le `target/` du service vivant.
- D'où la règle : **tout travail passe par un worktree**, jamais par `backend/`. C'est déjà le cas pour cette branche.
