# Branche de consolidation — Obrain par défaut + this-rs comme framework

Cette branche (`consolidation/obrain-default`) est le lieu unique du chantier.
Plan PO : `bf3ab074-467e-4b34-b92e-4b19040d8ece` (projet PO Backend).
Plan jumeau pour la migration des données : `250c63fb-0298-4980-bb16-7e6d3773b5d7`.

**État au 23/11/2026, relu contre `git` et `gh` (voir « Relecture du 23/11 » en fin de fichier pour ce qui a été vérifié et ce qui ne l'a pas été).** Ce fichier dit ce qui est fait, ce qui est en cours et ce qui ne l'est pas. Une ligne « en cours » nomme la PR ou la branche qui la porte ; une ligne sans preuve reste « à faire ». **La définition et l'état détaillé des portes de la migration vivent dans `docs/migration/GATES.md`** : ce fichier ne les recopie pas, il renvoie à elles.

## Règles de la branche

1. **Rien ne remonte dans `main` avant que la porte de bascule soit verte.** Aucune PR vers `main` n'est ouverte tant que ce fichier dit autre chose.
2. **`main` redescend ICI** (`git merge origin/main` dans cette branche), jamais l'inverse. Une branche qui ne voit jamais `main` pendant des mois devient infusible : c'est exactement ce qui est arrivé à `consolidation/obrain` (2 commits, abandonnée). Dernière descente : #1278 (22/11/2026, `main` descendu en portant #1268 dans la structure découpée de `manager.rs`), suivie de #1279 qui restaure l'ascendance de `main` ; après #1300 (rustix 1.1.6, `main` redescendu), la branche a au 23/11 **144 commits d'avance et 0 de retard** sur `main` (`git rev-list --left-right --count origin/main...origin/consolidation/obrain-default` → `0	144`). Avant : #1253 (20/11, #1223 → #1249), #1235 (`d79bcf30`).
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

État au 23/11/2026, d'après `docs/migration/GATES.md` (qui fait foi pour le détail et les limites de chaque porte). **Aucune condition n'est verte** : la bascule n'a pas lieu, et la directive du 20/11 la tient fermée tant que la migration n'est pas complète.

| # | Condition | Porte (GATES.md) | État |
|---|---|---|---|
| a | Conformité générique identique sur Neo4j et Obrain (mêmes valeurs, même ordre) | G4 | **en cours** — la jambe Neo4j tourne en CI ; sur obrain, 13 des 30 contrôles transverses tournent, **17 restent à couvrir** |
| b | recall@10 ≥ 0,95 à 25 000 vecteurs 768-d, après réouverture | — | non vérifié ici (côté obrain, plan O-VEC) |
| c | Graphe migré, aller-retour vérifié | G7 | **en cours** — le migrateur réel est dans le dépôt (#1256) et démontré sur une base jetable de 13 nœuds ; **nanosecondes tronquées : non résolu, bascule bloquée** ; **non rejoué sur la copie complète** ; composants fail-closed de publication de la copie écrits mais non branchés (#1270) |
| d | Double-run sans divergence de justesse | G9 | **non conçu** |
| e | Retour arrière = une variable de configuration | G10 | **rouge** — la sélection de la source est faite et testée, mais les écritures faites sur obrain après la bascule sont perdues pour la source (ni recopiées ni réconciliées) |

Autres portes qui bloquent la bascule : **G6** parité de lecture sur copie (à rejouer, dernière mesure le 16/11 sur une branche antérieure), **G8** performance (barre non tenue sur `SearchStore` : ouverture, filtre par projet, suppression), **G11** stabilité des tests (`kill9_during_creations…` instable ; `obrain::auth::tests::la_consommation_prend_l_heure_sous_le_verrou` exclu de la CI).

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
| Suites de relecture de #1165, #1161, #1166, #1163 et du mode Auto de this-rs | O-DOUBLE, O-HARNESS-PARITE | **#1171** (12/11, branche), **#1173** (11/11, `main`), **#1166** (12/11, `main`), **#1163** (13/11, `main` : référence datée de la migration), this **#254** (11/11) : toutes fusionnées (état lu par `gh pr view`) |
| Domaines portés sur obrain (vagues V1 à V4) : Workspace, Release, Note, Auth, Fédération, Plan (noyau, contraintes, liens, tâches et étapes), Chat (noyau), Decision (noyau), Commit, Protocole, Code (fichiers et `CO_CHANGED`), Persona, FeatureGraph, Skill, Sharing, Trigger, PlanRun | O-GRAPH | #1182 #1183 #1184 #1185 #1186 #1191 #1210 #1214 #1193 #1195 #1196 #1197 #1194 #1209 #1213 #1208, plus les corrections de relecture #1205 #1206 #1207 #1211 : fusionnées dans la branche |
| `SearchStore` porté (27 méthodes sur l'index texte d'obrain) | O-SEARCH | #1251 ; **barre de performance non tenue** (porte G8) |
| `AnalyticsStore` en partie : A1 (25 méthodes), A2 (23), mémoire-maintenance (15), agents et vues (15) | O-GRAPH (V5) | #1250 #1258 #1260 #1261 ; 79 des 145 méthodes servent, 66 refusées par nom (voir le tableau V5 plus bas, GATES.md G2) |
| Dates migrées lues dans leurs trois formes, plus jamais remplacées par le temps de la lecture ; un refus de store n'est plus affiché comme 0 | PORT-DATES, API-NONPORTE-MASQUE | commits du 16/11 (« une date migrée est LUE… », « un comptage refusé n'est plus affiché comme 0 », « les onze derniers replis silencieux ») ; résidus #1257 |
| Garde de bascule au démarrage (lecture seule nommée, vérification par type, retour arrière par configuration) | tâche 806bde21 | #1248 — **code inactif** : la production n'appelle pas la bascule |
| Portes de qualité (cliquet des traits, défaut neo4j, catalogue `docs/migration/GATES.md`) et protection de la branche | O-BRANCHE | #1255 #1263 #1264 ; protection posée le 21/11 (checks requis `Lint & Format`, `Unit Tests`, `Integration Tests`, `Obrain backend`, zéro approbation exigée, `enforce_admins`) |
| Migrateur réel Neo4j → obrain dans le dépôt, sous `obrain-backend` ; composants fail-closed de publication de la copie (non branchés) ; conformité transversale étendue ; tests unitaires obrain en CI | MIGRATEUR-REEL | #1256 (22/11) ; #1270 #1265 #1266 #1267 (23/11) |

### En cours

**Aucune PR n'est ouverte vers cette branche** (`gh pr list --state open --base consolidation/obrain-default` : vide, relevé le 23/11). La seule PR ouverte du chantier est **#1153**, la branche elle-même vers `main`, en draft (`BEHIND`, 3 commits de retard). Deux PR de la file de lots ont été **closes sans fusion** : #1190 (retrait du flush complet par écriture) et #1219 (quarantaine du domaine Fichier) ; les sujets voisins ont été fusionnés par #1220 et #1194.

| Ce qui reste ouvert | Où | Porte |
|---|---|---|
| Disque : la base pèse 35 Go sur obrain contre 12 Go sur Neo4j (mesuré le 15/11, `props.v2` ×5,3) | obrain **#181** (ouverte), tâche PO DISQUE-x3 | G7, bascule |
| Cypher d'obrain : traversées multi-sauts et agrégats de voisinage 1 000 à 24 000× plus lents que Neo4j | obrain **#189** (ouverte) | G8 |
| Contrats de livraison de la couche harnais ↔ frontend ↔ providers (reprise durable, garanties WASM/natif, événements, autorisations, matrice providers, livraison reproductible, contexte cognitif) | this-rs **#284 à #290** (toutes ouvertes), suivies par l'issue PO **#1246** (qualification transverse, propriétaire unique par mécanisme, porte de livraison) | hors des portes G1 à G11 |
| Module `plan` de this-rs : descripteur, CRUD, adaptateur, écran dérivé | this-rs `main` (#234, #235, #236, fusionnées) ; **PO ne le consomme pas encore** | O-MODULE-PLAN |
| `AnalyticsStore` : 66 méthodes refusées par nom | voir le tableau V5 plus bas | G2 |
| Instabilité de `kill9_during_creations…` | tâche TEST-KILL9-INSTABLE | G11 |
| Copie Neo4j de référence (27688) à recréer ou verrouiller en lecture seule | tâche COPIE-REFERENCE | G6, G7 |

### À faire, dans l'ordre du chemin critique

Fait depuis la dernière version de cette liste : fusion des correctifs obrain dont PO dépend (tous fusionnés, voir le tableau plus bas), `SearchStore` (**O-SEARCH**, #1251), migrateur dans le dépôt (#1256), **magasin de trajectoires** (désormais dans la fabrique : `ObrainTrajectoryStore`, qui **refuse par nom**, voir `src/storage/mod.rs`), cliquet des traits portés (G2).

1. **Rejouer la parité de lecture sur copie (G6)** sur la branche actuelle : la dernière mesure date du 16/11, sur une branche antérieure.
2. **Résoudre les nanosecondes tronquées du migrateur et le rejouer sur la copie complète** (G7), avec la publication de la copie (#1270, composants non branchés).
3. **Tenir la barre de performance de `SearchStore`** (G8) : ouverture, filtre par projet, suppression.
4. **Concevoir le double-run** (G9, non conçu) et **la réconciliation du retour arrière** (G10, rouge : les écritures post-bascule sont perdues).
5. **Étendre la conformité transversale** : 17 contrôles sur 30 sans jambe obrain (G4) ; stabiliser G11.
6. **Épingles** : `pins.toml` pointe `this` à `61040a2c` et `obrain` à `82467ff1` ; au 23/11 le `main` de this est à `eec7a96b` et celui d'obrain à `bd9b3b4f`. Une remontée d'épingle est un acte délibéré, dans le commit qui en a besoin.
7. **Descendre `main`** dans la branche (3 commits de retard) avant toute PR vers `main`.
8. **O-CLONE → O-SOAK → O-FLIP** : pas commencés ; aucune bascule avant que G6 à G11 soient verts.

Hors chemin critique : **O-THISRS-1** (socle this-rs monté sous `/_this` à côté d'axum ; plan écrit, aucun conflit de version d'axum/tower/tokio ; la dépendance par chemin de this-rs vers obrain et vite-rs est le vrai risque en CI), O-THISRS-2/3, O-NATS-OUT, O-COG, O-REACTIVE.

## Ce dont PO dépend dans obrain (dépôt `obrain-db/obrain`)

**État au 23/11/2026 : toutes les PR de ce tableau sont fusionnées** (relevé par `gh pr view -R obrain-db/obrain`, voir la colonne « Fusionnée »). La colonne de droite garde ce que les relectures du 11/11 avaient relevé avant correction : elle dit pourquoi la PR a eu des suites, pas qu'elle est encore ouverte ou « en correction ». Restent ouvertes côté obrain : **#181** (disque ×5,3) et **#189** (traversées multi-sauts très lentes). Rappel de méthode, toujours vrai : une PR reste en **brouillon** jusqu'au verdict du portillon (sept fusions prématurées le 11/11 : this #249/#251/#252, obrain #108/#110/#114, PO #1165/#1161).

Fusionnées (date de fusion lue sur GitHub) : #110 (11/11), #114 (11/11), #108 (11/11), #115 (11/11), #99 (12/11), #113 (12/11), #118 (12/11), #119 (12/11), #121 (12/11), #101 (13/11), #117 (13/11), #190 (16/11, unicité des relations dans un même `MATCH`, issue #182). Durabilité (issue #120, **close**) : #123, #137, #138 (12/11) et #153 (13/11).

| PR | Sujet | État relevé avant correction (11/11) |
|---|---|---|
| #110 | Les tests n'ouvrent plus les bases réelles sous `$HOME` | **fusionnée** (`e871411c`) ; suite #113 (garde contournable, copie sans détection d'écrivain vivant, cibles jamais exécutées) en draft, CI 9/11 verts |
| #114 | Un `WITH` groupé ne pose plus de projection inutile | **fusionnée** |
| #108 | Un agrégat dans `WITH` ne rend plus la clé de groupement invisible (104 requêtes de PO) | **fusionnée** avant corrections ; suites **#119** (MATCH à cible liée, `WITH` GQL, FOREACH) et **#121** (`duration.between`, `push_value`) en draft — relecture EXÉCUTÉE : #119 rend encore un produit cartésien silencieux en GQL sur clé de groupe aliasée (`WITH f AS g … MATCH (…)->(g)` : 18 lignes au lieu de 6) et une erreur sur ancre `UNWIND`/alias en Cypher ; #121 calcule le calendrier en UTC et rend NULL sur `ns[0].sha` — en correction |
| #99 | `UNIQUE (a, b)` porte sur le tuple ; DDL ; types implicites ; persistance | corrigée (`c9e20441` : SET / MERGE ON MATCH SET contrôlés, verrou DDL) ; re-relecture avec sondes : la réparation du catalogue à l'ouverture RÉÉCRIT le fichier en lecture seule, `SET n:T` écrit un doublon sans erreur, `CREATE CONSTRAINT` sur données en doublon répond « Created », contrôle asymétrique 5/5.0, coût O(n/cardinalité de la PREMIÈRE colonne) (744 µs vs 25 µs selon l'ordre) — en correction |
| #101 | Index d'égalité canonique (`5` = `5.0`, instants i128 ns), migration v1→v2 | quatre tours ; barre 100 ns NON atteinte (0,8–1,3 µs à 1 M : revérification par relecture, sondes mmap, hachage — refonte d'index en suite) ; dernière relecture : la récolte des instants longs est annulable (index partiel marqué complet) et une écriture blob n'a aucune trace durable (régression après kill) → règle « une clé à colonne blob n'est jamais crue sur la zone persistée, reconstruite au premier usage (0,83 s à 1 M) » — en correction |
| #117 | Unicité sur tous les chemins (SET, MERGE, label, concurrence, import), EMPILÉE sur #99 + #101 | rebasée (`7bef83b2`) ; `UNIQUE` sur flottant non entier = O(label) : **237 ms par écriture à 1 M** → clé d'index canonique puis index unique avec réservation jusqu'au commit |
| #115 | Lecture seule réelle (`open_read_only` n'écrit plus) | régression d'ouverture corrigée ; écart résiduel +5 %/+9 % réfuté comme « défauts de page » ; lint Linux corrigé ; CI en file |
| #118 | Ouverture O(1) en lecture seule, empilée sur #115 : vérification des compteurs au premier lecteur, sonde de 512 emplacements au-delà du filigrane | **8,3 → 0,7 ms à 1 M, 39 → 3,7 ms à 5 M**, RSS 107 → 4 Mo ; CI en file |
| **issue #120** | **Durabilité : une valeur > 256 octets est hors WAL, peut contaminer un autre nœud après SIGKILL, et le flush complet est O(base)** (34 ms vide → 2,6 s à 5 M hérité) | corrigée côté obrain (#123 valeur > 256 o journalisée dans le WAL avec sa valeur, #137 oracle de durabilité, #138 création atomique, #153 ordre en transaction) ; côté PO, le flush complet par écriture de valeur > 256 o est RETIRÉ, épingle obrain `c4b6ada9` → `e5648dbf`. **Débloque Chat (75), Code (85), Analytics (145)** |

Mesuré sur le premier lot (#1171, bases 0 / 1 M / 5 M, opt-level 3) : `get_project` 128–185 ns indépendant de N ; `create_project` ≤ 256 o = 4–5 ms p50 (fsync WAL) ; `create_project` avec valeur > 256 o = 128 ms → 2,8 s (paie le flush complet) ; ouverture 5 ms (vide), 37 ms (1 M neuf), 1,0 s (1 M hérité), 91 ms (5 M neuf), 5,6 s (5 M hérité) — le format hérité coûte ×10 à ×60. Unicité (#99/#117) : `UNIQUE` entier 22–47 µs/écriture de 100 k à 5 M ; composite 23–42 µs ; flottant non entier 26 ms (100 k) → 237 ms (1 M).

Après le retrait du flush complet par écriture (PR de l'époque, 11/11, obrain `e5648dbf`) : une valeur > 256 o paie le MÊME fsync WAL qu'une valeur ≤ 256 o — mesuré en debug sur 10 k nœuds / 20 k arêtes, `create_blob_p50` = 4,3 ms contre `create_wal_p50` = 5,2 ms, là où le pas « flush complet après chaque écriture » (`create_flush_complet_p50`) reste à 40,7 ms. Le test `a_write_past_the_blob_threshold_pays_no_full_flush` verrouille l'égalité des fsyncs (rouge sans le correctif, vert avec), et `kill9_during_creations…` rejoue les descriptions longues depuis le WAL.

Mesuré sur `main` d'obrain (`c4b6ada9`) : `MATCH (n) WITH n.p AS p, count(*) AS c RETURN p, c` échoue en Cypher (`Undefined variable 'p'`) et passe en GQL ; un second `MATCH` dont la cible est déjà liée rend 36 lignes au lieu de 14 en Cypher. #108 et ses suites #119 et #121 sont fusionnées sur `main` d'obrain et incluses dans l'épingle `e5648dbf` (de même que #99, #115 et #118 ; #117 n'y est pas) ; ces deux mesures n'ont PAS été refaites à `e5648dbf`, et les réserves de relecture notées plus haut sur #119/#121 n'ont pas été re-vérifiées. Le backend obrain de PO passe donc toujours par l'API Rust, jamais par une requête, jusqu'à un lot qui mesure les requêtes dont il a besoin.

## Migration des données : ce qui est établi

- **Source figée** : copie du volume Neo4j vivant monté en lecture seule, prise le 11/11 à 11:36 UTC, servie par un Neo4j jetable (ports 27474 / 27687). **3 901 702 nœuds, 6 013 360 relations**, 64 labels peuplés sur 70, 80 types de relation sur 82, un seul label par nœud. Deux méthodes de comptage indépendantes concordent. Détail : `docs/cartographie/22-copie-figee.md` (PR #1163).
- **Convertisseur existant** (`obrain/neo4j2obrain`) : ne satisfait pas « aucune perte » (nulls retirés des listes, dates aplaties en texte). **Remplacé par le migrateur réel du dépôt** (`src/storage/migrator/`, feature `obrain-backend`, #1256) ; son état est la porte **G7** de `docs/migration/GATES.md` : comptes par type, empreinte par label et zéro quarantaine démontrés sur une base jetable de 13 nœuds, **nanosecondes tronquées non résolu**, **non rejoué sur la copie complète**.
- **Volume** : mesuré le 15/11 sur la copie complète (run4), **35 Go alloués sur obrain contre 12 Go sur Neo4j**, surcoût concentré dans `substrate.props.v2` (×5,3 ; une chaîne de pages par nœud). Ticket moteur obrain **#181**, toujours ouvert au 23/11 ; tâche PO DISQUE-x3. L'ancienne mesure (16 Go pour un sous-ensemble sans `ChatEvent`) est dépassée.

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

**État au 23/11** : les domaines des vagues V1 à V4 sont portés (voir « Fait sur cette branche »), **Decision et Plan par leur noyau seulement** (#1195, #1191 puis #1210, #1214 : le reste refuse par nom), V5 en partie (79 des 145 méthodes d'`AnalyticsStore`). Le cliquet des traits portés (G2 de `GATES.md`) est le compte opposable ; ce tableau est le plan de découpe, pas un état.

Les domaines d'une vague se portent en parallèle à condition d'écrire un fichier par sous-trait côté obrain. **Premier jalon visible : un chat de bout en bout sur obrain = 44 méthodes** (Project 4, Chat 40), toutes en V1 ; +17 si l'authentification est active (estimation par graphe d'appels, pas par exécution).

Trois risques chiffrés : `AnalyticsStore` n'est pas un domaine (145 méthodes sur 21 fichiers, lit 14 domaines, écrit chez 9) ; 78 fonctions enchaînent au moins deux écritures sans transaction, et `init_schema` pose 189 instructions dont 7 contraintes composites ; ~1 960 sites de décodage `neo4rs`, avec des dates stockées tantôt en type temporel, tantôt en chaîne.

**Ordre retenu après le premier lot (mesuré le 11/11)** : **Workspace (45) puis Release (41)** — sans Release, `GET /api/projects` rend 500, car il passe par `ReleaseStore::list_projects_filtered` et non par `ProjectStore`. Puis Plan (54) et Note (41). **Attention : ni l'un ni l'autre ne lève le refus de `delete_project`.** Sa garde compte les nœuds de la base et refuse dès qu'il en existe plus que de projets ; une note crée un nœud, donc le refus reste tant que la cascade (archivage des notes, plans, tâches, fichiers…) n'est pas portée (preuve : `delete_project` dans `src/obrain/mod.rs` et `src/obrain/project.rs`, test `a_project_written_outside_is_read_and_its_cascade_delete_is_refused`). Chat (75), Code (85) et Analytics (145) sont débloqués par le retrait du flush complet (obrain #120 corrigé via #123/#138, épingle `e5648dbf`) : leurs propriétés dépassent couramment 256 octets et ne paient plus que le fsync WAL.

Défaut de justesse trouvé en passant, côté PO et indépendant du backend : `update_staleness_scores` lit `duration.between(...).days` comme une ancienneté, or `.days` est le reste après les mois (sémantique Neo4j reprise par obrain) — faux au-delà d'un mois (tâche O-STALENESS-DAYS).

## Portage d'`AnalyticsStore` : sous-lots (V5)

Classement des 145 méthodes de `AnalyticsStore` (`src/neo4j/traits.rs`), fait le 20/11/2026 en lisant le Cypher de chaque méthode dans `src/neo4j/` (script de comptage, puis relecture) : lecture / écriture, labels touchés, et présence d'un agrégat dans un `WITH`, d'un `UNWIND`, d'un chemin de longueur variable, d'un `CALL`/`apoc`/`gds`. Le compte « dures » est un compte de présence dans le Cypher, pas un jugement de coût : il surestime (un `count()` simple y entre), jamais l'inverse.

| Sous-lot | Contenu | Méthodes | Écr. | Lect. | Dures | Dépend de (domaines) | État |
|---|---|---|---|---|---|---|---|
| **A1** | Profils d'analyse, règles de topologie, profils utilisateur + `WORKS_ON`, bras et décisions de routage, frustration, parent d'étape, `health_check`, réglages LLM par clé, `run_data_migrations` | 26 | 13 | 13 | 0 | projet, tâches (index), chat (réglages) | **porté** (#1250) |
| A2 | Synapses de notes, énergie, cicatrices, consolidation, `init_note_energy`, synapses inter-entités, graphe de synapses (skills) | 24 | 13 | 11 | 8 | notes, décisions, projet | **porté en partie** (23 méthodes, voir ci-dessous) |
| A3 | Synapses fantômes (`shadow_*`, traces, récompense, clôtures de tâche) | 8 | 4 | 4 | 4 | notes, plan, exécutions d'agent | **porté en partie** (7 méthodes, mémoire-maintenance ci-dessous) ; `shadow_trace_closures` refusée (lit `AgentExecution`, non créable sans `Task`, `planrun.rs`) |
| A4 | Embeddings de fichiers/fonctions, recherche vectorielle, embeddings de notes | 5 | 3 | 2 | 4 | code (`File`/`Function` non portés), index vectoriel obrain 384-d | refusé, nommé |
| A5 | Lecture du graphe de code : arêtes IMPORTS/CALLS/EXTENDS/IMPLEMENTS, ADN structurel, empreintes, signaux, symboles, hotspots, pagerank, proximité | 16 | 0 | 16 | 4 | code | refusé, nommé |
| A6 | Écritures en lot sur le code : analytics de fichiers/fonctions/fabric, ADN, empreintes, liens prédits, churn, densité, risque, processus | 12 | 12 | 0 | 11 | code (`UNWIND` partout) | refusé, nommé |
| A7 | Scores et rapports : churn, densité de connaissance, risque, homéostasie, dérive structurelle, métriques neurales, santé d'apprentissage, audit des lacunes, groupes isomorphes | 12 | 0 | 12 | 10 | code, commits, notes, décisions, skills, personas | refusé, nommé |
| A8 | Cartes de contexte | 5 | 2 | 3 | 3 | code | refusé, nommé |
| A9 | Données de graphes de visualisation (PM, chat, décisions) + contraintes d'un projet | 4 | 0 | 4 | 1 | plan, release, commits, chat, décisions | refusé, nommé |
| A10 | Chemins : sous-graphe de pont, règles de topologie vérifiées, voisinage d'entité, catalogue de schéma, marche causale, chemin typé | 8 | 0 | 8 | 5 | code, tout le graphe (longueur variable) | refusé, nommé |
| A11 | Agents et « soi » : `ensure_agent`, liens agent→note/épisode, `agent_knows`, fondation, croyances, assertions, personas (`auto_grow_file_knows`, seuils adaptatifs) | 19 | 10 | 9 | 4 | notes, personas, graphes de fonctionnalités, épisodes, code | refusé, nommé (sauf `coactivate_session` et `rate_limited_energy_boost`, portés avec A2) |
| A12 | Maintenance : rapport et archivage des notes auto-observées, `rescope_protocol`, racines canoniques, passes profondes | 6 | 3 | 3 | 0 | projet, protocoles, notes | **porté** (sauf passes déjà portées avec A2 : `get_deep_maintenance_times`, `mark_deep_maintenance`) |

Total 145 = 26 + 119. A1 sert 25 méthodes ; `run_data_migrations` répond sa liste vide (les migrations ponctuelles sont celles de Neo4j, les données obrain arrivent par le migrateur). Les 119 autres sont écrites dans l'`impl AnalyticsStore for ObrainGraphStore` de `src/obrain/analytics.rs`, chacune rendant `NonPorte("AnalyticsStore::<méthode>")` ; le registre `refusees_analytics()` les liste et un test le compare au source.

**A2 porté en partie : la couche neuronale des notes (23 méthodes, `src/obrain/synapse.rs`).** Critère de découpe : un seul mécanisme de stockage (une arête `SYNAPSE` entre deux nœuds `Note`/`Decision`, propriétés sur l'arête ; des propriétés numériques ou datées sur le nœud de la note) et des dépendances toutes portées (index des notes, décisions, personas, projets, chat). Pas d'index propre : les arêtes d'un nœud sont lues du moteur, et `delete_note` (un `DETACH DELETE`) emporte déjà les siennes. Servies : `create_synapses`, `get_synapses`, `delete_synapses`, `get_all_synapse_weights`, `get_project_note_synapses`, `get_synapse_graph`, `reinforce_synapses`, `coactivate_session`, `weaken_node_synapses`, `create_cross_entity_synapses`, `get_cross_entity_synapses`, `list_notes_needing_synapses`, `list_decisions_needing_synapses`, `get_project_note_entity_links`, `update_energy_scores`, `boost_energy`, `track_reactivation`, `init_note_energy`, `apply_scars`, `heal_scars`, `rate_limited_energy_boost`, `get_deep_maintenance_times`, `mark_deep_maintenance`. Refusées, nommées : `decay_synapses` et `decay_project_synapses` (passes d'oubli : archivage `FORGOTTEN` sous plafond et plancher de degré, ordre non spécifié par le Cypher, et le mock ne les modélise pas : aucune référence comparable), `consolidate_memory` (balayage du cycle de vie des notes, à porter avec ses règles), `get_project_synapse_edges` (jointure avec les personas et les notes, A5/A7). Écarts nommés en tête de `src/obrain/synapse.rs` : un id en quarantaine refuse (lecture et écriture), un parcours de toutes les notes refuse si une note est en quarantaine, un poids absent ou non numérique refuse (Neo4j saute la ligne ou écrit null), deux arêtes parallèles refusent une écriture qui devrait choisir, les nombres non finis refusent, les résultats que le Cypher ne trie pas sont triés. `reinforce_synapses` et `coactivate_session` ressuscitent une arête `FORGOTTEN` d'une base migrée à son poids (plancher 0,02).

**Lot « mémoire-maintenance » porté (15 méthodes ; A3 sauf `shadow_trace_closures`, A12, les passes laissées par A2 ; `src/obrain/shadow.rs`, `src/obrain/memoire.rs`, `project.rs`, `protocol.rs`).** En Rust sur les index et les arêtes, sans GQL. Servies : `shadow_synapse_edges`, `reward_shadow_synapses`, `decay_shadow_synapses`, `record_synapse_shadow_trace`, `list_synapse_shadow_traces`, `get_synapse_shadow_trace`, `claim_shadow_trace_reward` (propriétés `shadow_*` des arêtes `SYNAPSE` et nœuds `SynapseShadowTrace` indexés en mémoire, reconstruits à l'ouverture) ; `decay_synapses`, `decay_project_synapses`, `consolidate_memory`, `auto_observed_report`, `archive_unconfirmed_auto_observed`, `audit_knowledge_gaps` ; `ensure_project_canonical_paths`, `rescope_protocol`. Restent refusées, nommées : `shadow_trace_closures` et `get_note_embeddings_for_project` (un plongement ne s'écrit ni ne se relit sur ce backend : `get_note_embedding` refuse déjà, obrain ne journalise pas un vecteur, et une liste vide ne se distinguerait pas de plongements perdus ; A4). **Écarts nommés** (en tête de `shadow.rs` et `memoire.rs`) : l'archivage sous plafond de `decay_synapses` / `decay_project_synapses` n'a pas d'ordre dans le Cypher (`LIMIT $cap` sans `ORDER BY`) et son plancher de degré est évalué sur l'état d'AVANT la passe ; ce backend définit un ordre TOTAL (poids croissant, puis id source, puis id cible) et un plancher sur les degrés COURANTS (aucune note ne passe sous `min_degree` en une passe, ce que la politique promet) ; une arête `FORGOTTEN` déjà présente sur la paire est remplacée (Neo4j en écrirait une seconde, que `reinforce_synapses` refuse ensuite de ressusciter), plus d'une refuse la passe ; la guérison des cicatrices (étape 3) touche toutes les notes et décisions quel que soit le périmètre, comme la requête Neo4j ; un poids absent refuse ; une horloge `shadow_decayed_at` illisible refuse (Neo4j la lisait `None`) ; `consolidate_memory` refuse sur une note en quarantaine et sur un `changes_json` illisible ; `audit_knowledge_gaps` lit les nœuds et arêtes du moteur (quarantaine comprise) et trie ses échantillons par id ; `ensure_project_canonical_paths` refuse sur un projet non servi et propage ses erreurs d'écriture ; `rescope_protocol` d'un protocole ou d'un projet absent REFUSE (Neo4j répond `Ok` sans rien écrire, ou écrit un `project_id` sans arête) et n'écrit que la propriété `project_id` (pas d'arête `BELONGS_TO`). Atomicité : l'arête `FORGOTTEN` est créée entière AVANT la suppression de la `SYNAPSE` (une coupure laisse les deux), `decayed_at` / `shadow_decayed_at` sont écrits AVANT le poids (une coupure ne facture jamais deux fois un intervalle), l'historique d'une note est écrit AVANT son effet.

Ordre conseillé pour la suite : A12 (dépend peu), A2 puis A3 (domaine note déjà porté, l'agrégat se calcule en Rust sur l'index ; `src/graph/algorithms.rs` est déjà portable), puis A5/A6/A8 quand le domaine code (`File`, symboles) sera porté (aujourd'hui ses lectures refusent : `src/obrain/code.rs`), A7/A9/A10 en dernier (jointures de plusieurs domaines, chemins variables : aucune requête GQL sans l'avoir exécutée contre obrain, voir « Mesuré sur `main` d'obrain »).

**Ce que A1 fait autrement que Neo4j (écarts nommés dans `src/obrain/analytics.rs`)** : le projet absent de `create_analysis_profile` et l'utilisateur absent de `update_user_profile` REFUSENT au lieu d'écrire rien et de répondre `Ok` ; `upsert_works_on` garde en revanche le no-op de Neo4j (journalisé), car l'agrégateur de profils (`src/profile/aggregator.rs`) enchaîne les projets d'un flush avec `?` et un refus perdrait les suivants. Un seuil absent reste absent (Neo4j écrit 0 et relit `Some(0)`). Pas d'arête `HAS_PROFILE` ni `ROUTED_FOR` (la propriété `project_id` / `session_id` porte le lien ; ces arêtes feraient refuser la suppression des tâches et sessions voisines). Décodage strict : un nœud illisible est mis en quarantaine nommée à l'ouverture et sa clé réservée ; une liste qui pourrait le manquer refuse (profils visibles d'un projet, règles d'un projet, bras d'une classe, `WORKS_ON` d'un utilisateur), sauf le journal des décisions, où une décision au document illisible est sautée comme Neo4j le fait (`routing.rs`), et nommée à l'ouverture. `list_routing_decisions` est en O(décisions) sur une entrée légère en mémoire. (Pas un écart : un id de règle déjà pris refuse, comme la contrainte Neo4j `topology_rule_id`.)

**Lot « agents et vues » (15 méthodes de plus : 104 refusées, 41 qui servent)** — `src/obrain/agent.rs` (nœud `Agent`, arêtes `AUTHORED` / `LIVED` / `KNOWS` / `USES`, croyances) et `src/obrain/vues.rs` (lectures de vues). Servies : `ensure_agent`, `backfill_agents`, `link_agent_authored`, `link_agent_lived`, `link_agent_knows`, `link_agent_uses`, `release_agent_knows`, `agent_knows`, `project_synced`, `list_assertion_notes`, `record_assertion_check` (sous-lot A11), `get_project_decisions_for_graph`, `get_pm_graph_data`, `get_chat_graph_data`, `get_project_constraints` (A9 entier). Restent refusées, nommées, avec leur raison dans `analytics.rs` : `entity_exists`, `entities_linked`, `ranked_foundation` (lisent `File` / `Function` / `Struct` / `Trait` / `Enum`, `LINKED_TO` vers `File` et `SYNAPSE` : répondre `false` RÉFUTERAIT une croyance), `get_entity_neighborhood`, `get_scoped_entity_neighborhood`, `schema_catalog`, `causal_walk`, `typed_path` (A10 : tout le graphe, `apoc`/`CALL`, chemins de longueur variable, aucune requête GQL exécutée contre obrain).

Décisions nommées (la tâche demandait de ne pas les prendre en silence) : (1) `save_session_episode` ne crée toujours NI l'agent NI l'arête `LIVED` (Neo4j le fait au mieux) ; l'arête est écrite par `link_agent_lived` et reconstruite par `backfill_agents`. (2) `create_note` refuse toujours un auteur `agent:<uuid>` (note.rs) ; `link_agent_authored` / `backfill_agents` tracent l'arête pour les notes que la base porte. (3) `backfill_agents` NE TOUCHE PAS `DataMigration` : seul `migrate_agent_edges_to_self` (hors contrat) l'écrit, et il n'est pas porté ; `backfill_agents` est idempotent (une seconde passe trace 0 arête). (4) `delete_note`, `delete_chat_session` (et son épisode) sont des `DETACH DELETE` : les arêtes de l'agent partent avec (prouvé dans `tests/obrain_agents.rs`) ; un nœud `Agent` fait refuser `delete_project`, comme tout nœud qu'aucune cascade n'atteint. (5) Les trois vues « graphe » sont exactes pour une base écrite par ce backend et REFUSENT, nommées, sur la forme de base migrée où elles ne le seraient pas (arête `AFFECTS` ou `INFORMED_BY` sur une décision, `DISCUSSED` sur une session, `LINKED_TO_PLAN` / `LINKED_TO_TASK` depuis un `Commit`). (6) `get_project_constraints` reproduit la référence, qui suit `(:Plan)-[:BELONGS_TO_PROJECT]->(:Project)` : aucune écriture du dépôt ne trace cette arête (un plan atteint son projet par `HAS_PLAN`), la réponse est donc vide sur toute base écrite ici, comme sur Neo4j aujourd'hui ; savoir si la référence voulait `HAS_PLAN` est une question pour son propriétaire, pas tranchée ici. (7) Les tris sont totaux là où Neo4j n'en dit aucun (décisions : `(decided_at, id)` ; contraintes : `(id, plan)` ; PM : plans `priority DESC, id DESC`, le reste par id).

**Migrations de données Neo4j que le migrateur devra jouer AVANT la copie** : `run_data_migrations` ne fait rien sur obrain (elles sont écrites en Cypher pour une base Neo4j, aucune n'est portée), et le migrateur est hors dépôt. Sur la source figée, il faut donc avoir joué, ou vérifier comme déjà jouées (nœud `DataMigration` par id), celles de `src/neo4j/data_migrations.rs` : `2026-09-fold-legacy-alerts`, `2026-09-archive-duplicate-skills`, `2026-09-restore-energy-decay-victims`, `2026-09-rebase-active-note-energy`, `2026-09-purge-empty-archived-skills`, `2026-10-rehome-worktree-knowledge`, `2026-10-purge-worktree-nodes`, `2026-10-backfill-stable-key`, `-types`, `-impls`, `-imports`, `2026-10-backfill-file-uid`, `2026-10-backfill-file-uid-orphans`, `2026-11-retype-auto-observed` ; la réparation rejouable `2026-11-reset-phantom-state-timeouts` (états de protocole à `state_timeout_secs = 255`, entiers négatifs mal lus par neo4rs 0.8.0) ; et la migration d'identité d'agent `agent_identity_self`. Tant que le migrateur ne le prouve pas par type, ces réparations ne sont PAS garanties sur une base obrain.

## Sécurité des données (incidents du 11/11)

- **Les tests d'obrain ouvraient les bases réelles** sous `~/.obrain/db` en écriture (`cargo test -p obrain-engine` sur `main`). **#110 et sa suite #113 sont fusionnées** (11/11 et 12/11) : une base réelle ne s'ouvre plus que si on la désigne. Par prudence, une commande `cargo test` / `cargo bench` d'obrain sur une machine qui porte une base réelle se lance quand même avec un `HOME` vide tant qu'on n'a pas relu la garde sur l'épingle utilisée.
- **Le mode « lecture seule » d'obrain écrivait** (rejeu du journal dans les zones, reconstruction des degrés, création de fichiers). **#115 et #118 sont fusionnées** (11/11 et 12/11). La règle pratique reste : une base réelle ne se sonde que sur une copie, y compris avec `--read-only`.
- **PO** : un binaire de test a ouvert la base vivante parce que les clés de `config.yaml` sont imbriquées. Les tests du backend obrain n'ouvrent que des répertoires temporaires.

## Les portillons de cette branche (en place)

| Portillon | Ce qu'il empêche | Commande |
|---|---|---|
| `containment` | la branche qui dérive de `main` pendant des mois sans que personne ne le voie | `cargo xtask containment` |
| `path-pins` | un voisin (this, obrain, atelier) qui bouge sous les pieds sans qu'aucun diff ne le montre | `cargo xtask path-pins` |

Les deux tournent dans le job « Lint & Format », donc ils sont bloquants par héritage — sans toucher à la protection de branche de `main`.

`pins.toml` dit contre QUELS commits on construit. Mettre un `rev` à jour est un acte délibéré, dans le même commit que le besoin qui le motive. **État au 23/11** : `pins.toml` pointe `this` à `61040a2c`, `obrain` à `82467ff1` et `atelier` à `261237bc` ; le `main` de this est à `eec7a96b` et celui d'obrain à `bd9b3b4f` (relevé par l'API GitHub). Le portillon n'a **pas été rejoué** lors de cette relecture ; l'état du 11/11 (rouge en local parce que les voisins avaient avancé, 0 en CI où ils sont absents) n'est donc pas confirmé. Les épingles se remontent dans le commit qui a besoin du nouveau comportement.

`path-pins` ne fait pas de `git fetch` par défaut : un voisin dont le remote est en ssh bloquerait le portillon. `--fetch` l'active.

## À savoir avant d'écrire une ligne ici : `backend/` est un LIEN

`/Users/triviere/projects/project-orchestrator/backend` est un lien symbolique vers `/Users/triviere/.openclaw/workspace/skills/project-orchestrator`, qui est le **répertoire du service vivant**.

Conséquences, mesurées le 10/11/2026 :
- La « copie principale » du dépôt n'est pas `backend/` : c'est ce répertoire-là. `git rev-parse --show-toplevel` depuis `backend/` le dit.
- Éditer `backend/` **c'est** éditer le service vivant. La consigne « ne jamais toucher `~/.openclaw/workspace/skills/project-orchestrator` » est donc impossible à tenir en même temps que « éditer `backend/` » : ce sont les mêmes fichiers (même inode).
- Le `pre-push` partage `CARGO_TARGET_DIR` avec cette copie principale : un build lancé depuis un worktree écrit dans le `target/` du service vivant.
- D'où la règle : **tout travail passe par un worktree**, jamais par `backend/`. C'est déjà le cas pour cette branche.

## Relecture du 23/11/2026 : ce qui a été vérifié, et ce qui ne l'a pas été

Ce fichier avait dérivé : son en-tête et ses tableaux « En cours », « À faire » et « Ce dont PO dépend dans obrain » disaient l'état du 11/11, alors que le corps avait été enrichi jusqu'au 21/11 (sous-lots d'`AnalyticsStore`). Une relecture contre le réel a corrigé les lignes périmées.

**Relevé directement (commande rejouable)**
- Avance et retard de la branche : `git rev-list --left-right --count origin/main...origin/consolidation/obrain-default` → 3 / 141.
- Ce qui est entré depuis le 11/11 : `git log --first-parent origin/consolidation/obrain-default --since=2026-11-11` (82 entrées).
- État de chaque PR et issue citée : `gh pr view` / `gh issue view` sur `this-rs/project-orchestrator`, `obrain-db/obrain`, `this-rs/this`. Un premier listage par `gh pr list --search` mélangeait des PR fusionnées à des PR ouvertes ; seul `--state open --base consolidation/obrain-default` (vide) et la lecture PR par PR font foi.
- Fabrique de stockage : `ObrainTrajectoryStore` dans `src/storage/mod.rs`. Épingles : `pins.toml`. Têtes de `main` de this et d'obrain : API GitHub.
- Définition et état des portes : `docs/migration/GATES.md`, lu en entier jusqu'à G4 ; G5 à G11 repris de son tableau d'ensemble.

**Non vérifié (à ne pas lire comme confirmé)**
- Le plan PO `bf3ab074` (trop volumineux pour une lecture complète) : les statuts de ses tâches n'ont pas été relus. Le plan jumeau `250c63fb` l'a été : MIGRATEUR-REEL y reste « pending » alors que #1256 est fusionné, à corriger côté plan.
- Aucune porte n'a été rejouée : ni `cargo xtask path-pins`, ni le test du cliquet, ni G6 (parité de lecture sur copie).
- Les chiffres de performance (barre de la nanoseconde, 8,3 → 0,7 ms, etc.) et le recall@10 viennent des mesures datées de ce fichier, non refaites.
- L'état des PR de la file de lots #1214, #1250, #1258 est déduit du journal de la branche (elles y figurent), pas d'un `gh pr view`.
