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
| Premier lot du backend obrain : `BackendKind::Obrain`, `ProjectStore` (9 méthodes), 748 refus nommés générés (`build/refus_obrain.rs`), job CI « Obrain backend » | O-DOUBLE | **#1165 fusionnée** (11/11 15:28Z, avant ses corrections → suite #1171) |
| Serveur réel démarré sur obrain sans Neo4j ni Meilisearch : `POST`/`GET /api/projects/<slug>` par les vrais handlers, donnée survivant à SIGTERM + redémarrage | O-DOUBLE | mesuré le 11/11 (#1171) |
| Descente de `main` (#1161 #1167 #1168 #1169 #1170) avec le conflit `manager.rs` résolu, relecture de fidélité (530 tests → 530, zéro perte/doublon) | O-BRANCHE | **#1174 fusionnée** (fast-forward, `origin/main` contenu) |
| Test instable des ancres : les balayages `backfill_*`/`revert_*` étaient globaux ; bornés par slug/session/id, binaire `anchor_sweeps_unscoped` pour le chemin non borné | — | **#1172 fusionnée dans main** (`f0dbe001`) |

### En cours

| Lot | Tâche PO | Où |
|---|---|---|
| Suite de #1165 : création atomique au crash (marqueur `_brouillon`), un fsync WAL par écriture ≤ 256 o, compteur O(1) pour la suppression, garde-fou d'ouverture (`storage.obrain_create`) | O-DOUBLE | **#1171** (draft) ; relecture : mise à jour déchirée non réparée, test kill -9 non probant, update/delete visibles avant fsync — en correction |
| Mode Auto du harnais porté dans `this-harness` (cœur de décision pur, compatible wasm) | O-HARNESS-PARITE | this-rs **#249, #251, #252 fusionnées** (avant corrections) ; **#254** = corrections (consultatif ne bouge rien, réparation survit à un `settle` qui pend, paire jamais atteinte oubliée, JSON seulement) — relecture finale : fusionnable après retouche de doc |
| Suite de #1161 (fusionnée avant corrections) : `idle_verdict` unique, plafond d'attente humaine 24 h, `AskedRequest.turn`, Claude Code via `can_use_tool` | — | **#1173** (draft) ; relecture : demande orpheline à la fermeture, demande hors tour non protégée — en correction |
| neo4rs 0.8.0 : entiers négatifs lus de travers (65 `state_timeout_secs = 255` fantômes) : vendor patché, migration au boot, garde d'hôte | — | **#1166** (draft) ; relecture : `tests/data_migrations.rs` visait encore la base VIVANTE (7687) — bloquant en correction |
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

État au 11/11 au soir. Ordre de fusion prévu : #99 → #101 → #117 ; #119 → #121 ; #113 ; #115 → #118. Toute PR reste en **brouillon** jusqu'au verdict du portillon (sept fusions prématurées ce jour : this #249/#251/#252, obrain #108/#110/#114, PO #1165/#1161).

| PR | Sujet | État |
|---|---|---|
| #110 | Les tests n'ouvrent plus les bases réelles sous `$HOME` | **fusionnée** (`e871411c`) ; suite #113 (garde contournable, copie sans détection d'écrivain vivant, cibles jamais exécutées) en draft, CI 9/11 verts |
| #114 | Un `WITH` groupé ne pose plus de projection inutile | **fusionnée** |
| #108 | Un agrégat dans `WITH` ne rend plus la clé de groupement invisible (104 requêtes de PO) | **fusionnée** avant corrections ; suites **#119** (MATCH à cible liée, `WITH` GQL, FOREACH) et **#121** (`duration.between`, `push_value`) en draft — relecture EXÉCUTÉE : #119 rend encore un produit cartésien silencieux en GQL sur clé de groupe aliasée (`WITH f AS g … MATCH (…)->(g)` : 18 lignes au lieu de 6) et une erreur sur ancre `UNWIND`/alias en Cypher ; #121 calcule le calendrier en UTC et rend NULL sur `ns[0].sha` — en correction |
| #99 | `UNIQUE (a, b)` porte sur le tuple ; DDL ; types implicites ; persistance | corrigée (`c9e20441` : SET / MERGE ON MATCH SET contrôlés, verrou DDL) ; re-relecture avec sondes : la réparation du catalogue à l'ouverture RÉÉCRIT le fichier en lecture seule, `SET n:T` écrit un doublon sans erreur, `CREATE CONSTRAINT` sur données en doublon répond « Created », contrôle asymétrique 5/5.0, coût O(n/cardinalité de la PREMIÈRE colonne) (744 µs vs 25 µs selon l'ordre) — en correction |
| #101 | Index d'égalité canonique (`5` = `5.0`, instants i128 ns), migration v1→v2 | quatre tours ; barre 100 ns NON atteinte (0,8–1,3 µs à 1 M : revérification par relecture, sondes mmap, hachage — refonte d'index en suite) ; dernière relecture : la récolte des instants longs est annulable (index partiel marqué complet) et une écriture blob n'a aucune trace durable (régression après kill) → règle « une clé à colonne blob n'est jamais crue sur la zone persistée, reconstruite au premier usage (0,83 s à 1 M) » — en correction |
| #117 | Unicité sur tous les chemins (SET, MERGE, label, concurrence, import), EMPILÉE sur #99 + #101 | rebasée (`7bef83b2`) ; `UNIQUE` sur flottant non entier = O(label) : **237 ms par écriture à 1 M** → clé d'index canonique puis index unique avec réservation jusqu'au commit |
| #115 | Lecture seule réelle (`open_read_only` n'écrit plus) | régression d'ouverture corrigée ; écart résiduel +5 %/+9 % réfuté comme « défauts de page » ; lint Linux corrigé ; CI en file |
| #118 | Ouverture O(1) en lecture seule, empilée sur #115 : vérification des compteurs au premier lecteur, sonde de 512 emplacements au-delà du filigrane | **8,3 → 0,7 ms à 1 M, 39 → 3,7 ms à 5 M**, RSS 107 → 4 Mo ; CI en file |
| **issue #120** | **Durabilité : une valeur > 256 octets est hors WAL, peut contaminer un autre nœud après SIGKILL, et le flush complet est O(base)** (34 ms vide → 2,6 s à 5 M hérité) | ouverte le 11/11 ; **bloque le portage de Chat (75), Code (85), Analytics (145)** |

Mesuré sur le premier lot (#1171, bases 0 / 1 M / 5 M, opt-level 3) : `get_project` 128–185 ns indépendant de N ; `create_project` ≤ 256 o = 4–5 ms p50 (fsync WAL) ; `create_project` avec valeur > 256 o = 128 ms → 2,8 s (paie le flush complet) ; ouverture 5 ms (vide), 37 ms (1 M neuf), 1,0 s (1 M hérité), 91 ms (5 M neuf), 5,6 s (5 M hérité) — le format hérité coûte ×10 à ×60. Unicité (#99/#117) : `UNIQUE` entier 22–47 µs/écriture de 100 k à 5 M ; composite 23–42 µs ; flottant non entier 26 ms (100 k) → 237 ms (1 M).

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

**Ordre retenu après le premier lot (mesuré le 11/11)** : **Workspace (45) puis Release (41)** — sans Release, `GET /api/projects` rend 500, car il passe par `ReleaseStore::list_projects_filtered` et non par `ProjectStore`. Puis Plan (54) et Note (41), qui lèvent le refus de `delete_project`. Chat (75), Code (85) et Analytics (145) seulement après obrain #120 (défauts 1–3) : leurs propriétés dépassent couramment 256 octets et chaque écriture paierait le flush complet proportionnel à la base.

Défaut de justesse trouvé en passant, côté PO et indépendant du backend : `update_staleness_scores` lit `duration.between(...).days` comme une ancienneté, or `.days` est le reste après les mois (sémantique Neo4j reprise par obrain) — faux au-delà d'un mois (tâche O-STALENESS-DAYS).

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
