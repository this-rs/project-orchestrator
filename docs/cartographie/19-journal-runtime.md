# 19 — Journal en vrai : PO tourne-t-il sans Neo4j ni NATS ?

Vérifié le 10/11/2026, binaire `orchestrator` (debug), ports décalés (HTTP 18080, Neo4j 17687, Meili 17700), `HOME` et données sous `/tmp/po-journal-verif/`. Service vivant non touché.

## Verdict : NON. Le journal n'a pas pu être exercé.

Lancement : `EVENTS_TRANSPORT=journal EVENTS_JOURNAL_PATH=… NEO4J_URI=bolt://127.0.0.1:17687 MEILISEARCH_URL=http://127.0.0.1:17700 orchestrator --config config.yaml serve --port 18080 --no-frontend` (sans NATS_URL).

Constat :
1. Sans `setup_completed: true` dans un YAML (le flag n'existe pas en env, `src/lib.rs:1041`), PO ne monte qu'un serveur « setup-only » (`lib.rs:1434`) : /health et /api/setup-status. Aucun transport n'est construit.
2. Avec ce flag, `start_server` appelle `AppState::new` → `StorageBackend::from_config` (`lib.rs:1303`, `storage.rs:57`). Le seul backend accepté est `default` (`storage.rs:46`, `bail!("unknown storage backend…")` sinon), qui construit `Neo4jClient::new` (`storage.rs:66`), `MeiliClient::new` (`storage.rs:73`) et `trajectory::open` sur Neo4j (`storage.rs:77`), tous avant la ligne 1498 où `select_transport` est lu et avant `build_chat_transport` (`lib.rs:1583`).
3. Neo4j absent : pas d'erreur nette. `Neo4jClient::new` (`neo4j/client.rs:394`) passe par neo4rs qui retente avec backoff (0,6 s → 22 s…), et `init_schema` (`client.rs:405`, `:803`) avale chaque échec en avertissement « Constraint may already exist: an IO error occurred: Connection refused (os error 61) ». Après 45 s le processus n'avait toujours pas écouté sur 18080 ni créé le journal (`journal.log` absent). Pas de message d'abandon : un blocage, pas un refus.

Rien n'a donc été publié, rejoué ni testé en `CursorTooOld` en instance réelle : seuls les tests unitaires (`chat_transport.rs:898`, `:934`) prouvent le journal.

## Ce qui empêche PO de tourner sans Neo4j/NATS
- **Neo4j** : obligatoire et bloquant (backend `default` unique, `storage.rs`), schéma initialisé au démarrage, magasin de trajectoires aussi.
- **Meilisearch** : obligatoire (`MeiliClient::new`, `storage.rs:73`).
- **NATS** : NON bloquant à lui seul (repli « local-only », `lib.rs:1535-1555`) — son retrait est donc possible côté démarrage ; ce qui manque est la preuve du journal derrière un vrai serveur.
- **Config** : `setup_completed` n'est lisible que par YAML.
- **Défaut** `events.transport = nats` inchangé.

## Pour débloquer
Un backend de stockage sans Neo4j/Meili (mémoire ou embarqué) sélectionnable par `STORAGE_BACKEND`, et un échec rapide et net quand Neo4j est injoignable. Ensuite seulement, rejouer : publier, vérifier la croissance du journal, redémarrer, `replay`, curseur antérieur → `CursorTooOld`.

---

## Réessai avec pile jetable — VERDICT : OUI, les 5 points vérifiés (10/11/2026)

Même machine, mêmes ports décalés, mais cette fois **Neo4j + Meili fournis par conteneurs jetables** (la seule raison de l'échec précédent : `StorageBackend::from_config` n'accepte que `default`, qui exige les deux). Binaire debug `target/debug/orchestrator`, 1 build (3 min). `git status --porcelain` vide avant/après.

### Recette reproductible (commandes exactes)
```bash
# 1. Pile jetable (ports interdits en prod : 8080/7687/7474/7700/4222 jamais touchés)
docker run -d --name po-test-neo4j \
  -e NEO4J_AUTH=neo4j/orchestrator123 \
  -e NEO4J_server_memory_heap_initial__size=256m \
  -e NEO4J_server_memory_heap_max__size=512m \
  -e NEO4J_server_memory_pagecache_size=256m \
  -p 17687:7687 -p 17474:7474 neo4j:5.26
docker run -d --name po-test-meili \
  -e MEILI_MASTER_KEY=orchestrator-meili-key-change-me \
  -e MEILI_NO_ANALYTICS=true -p 17700:7700 getmeili/meilisearch:v1.34

# 2. Config minimale (setup_completed lisible UNIQUEMENT en YAML, src/lib.rs:1041)
mkdir -p /tmp/po-test/workspace
cat > /tmp/po-test/config.yaml <<'YAML'
setup_completed: true
server: { port: 18080, workspace_path: /tmp/po-test/workspace, serve_frontend: false }
neo4j: { uri: "bolt://localhost:17687", user: neo4j, password: orchestrator123 }
meilisearch: { url: "http://localhost:17700", key: orchestrator-meili-key-change-me }
events: { transport: journal, journal_path: /tmp/po-test/chat-journal.log }
YAML

# 3. Build unique puis lancement SANS NATS_URL
cargo build --bin orchestrator
EVENTS_TRANSPORT=journal EVENTS_JOURNAL_PATH=/tmp/po-test/chat-journal.log \
NEO4J_URI=bolt://localhost:17687 NEO4J_USER=neo4j NEO4J_PASSWORD=orchestrator123 \
MEILISEARCH_URL=http://localhost:17700 MEILISEARCH_KEY=orchestrator-meili-key-change-me \
WORKSPACE_PATH=/tmp/po-test/workspace \
target/debug/orchestrator --config /tmp/po-test/config.yaml serve --port 18080 --no-frontend
```

### Les 5 verdicts

1. **PO démarre** ✔ — `/health` → `{"status":"ok","services":{"neo4j":"connected","meilisearch":"connected","nats":"disabled"}}` ; log : `Chat transport: journal — chat events are persisted and replayable path=/tmp/po-test/chat-journal.log max_events=100000`.

2. **Événements réels** ✔ — `POST /api/chat/sessions {"message":"hello journal"}` crée le journal (0 → 244 octets) et écrit 6 événements typés dans l'ordre : `user_message`, `streaming_status{true}`, `system_init`, `assistant_text`, `result`, `streaming_status{false}`. C'est le flux complet d'un tour, journalé ligne à ligne (JSON, un `JournalEvent` par ligne).

3. **Redémarrage** ✔ — kill, relance même `EVENTS_JOURNAL_PATH`. Réouverture : `next_cursor=13 oldest_cursor=Some(1)` (aucune collision, les curseurs sont relus du disque). Nouveau message → curseurs 7→12 (continue). `replay(since=0)` → les 12 événements des deux sessions, dans l'ordre. (Sonde hors-dépôt `/tmp/po-test/replay-probe` liant la lib : `JournalTransport::open(…)` puis `.replay(0)`.)

4. **Perte explicite** ✔ — sur copie du journal, `open(path, max_events=5)` tronque à `oldest_cursor=8` ; `replay(since=0)` → `CursorTooOld { oldest: 8, requested: 0 }` (erreur typée, pas de flux amputé) ; `replay(since=7)` → les 5 événements restants (8→12). Le journal principal (17 lignes) reste intact.

5. **Chat sans NATS** ✔ — aucun `NATS_URL` : envoi, stream (statuts + `system_init` + `assistant_text` + `result`) et interruption fonctionnent. `POST /api/chat/sessions/{id}/interrupt` → `{"delivered":true,"routed":"local"}` : l'interruption est délivrée **en process**, pas par le bus. Le seul échec observé est `result.error_code="auth_required"` (CLI Claude non connecté), indépendant de NATS.

### Ce qui dépend encore de NATS (aucun blocage mono-instance)
- Le contrôle **inter-instance** : `request_cancel`, `request_streaming_snapshot`, `subscribe_snapshot_requests`, relais `cancel_task`. En journal/mono-instance ce sont des no-op documentés (`RelayFailure::NoResponders`, `None`) — le journal les déclare sans propriétaire distant, rien n'est perdu localement.
- Le bus CRUD (`start_nats_bridge`, `start_graph_nats_bridge`) est toujours appelé au boot mais sans NATS c'est le repli « local-only » (`lib.rs:1535-1555`) : aucun effet.

### Points restés ouverts (non requis par la mission)
- `max_events` (rétention) est **codé en dur** à `DEFAULT_MAX_EVENTS=100000` au boot (`lib.rs:1587`) : pas de config `events.max_events` ; la rétention minuscule du point 4 ne se pilote donc pas en instance, seulement via `JournalTransport::open` (tests `chat_transport.rs:898/934` + la sonde).
- Le rejeu du WS (`ws_chat_handler.rs`) lit l'historique depuis **Neo4j** (`get_events_since`), pas depuis `JournalTransport::replay` — le `replay()` du journal n'est consommé par aucune route aujourd'hui.
- Nettoyage fait : conteneurs `po-test-neo4j`/`po-test-meili` stoppés+supprimés, `/tmp/po-test` effacé.

---

## PO sans aucun service externe : `storage.backend = memory` (12/11/2026)

`StorageBackend::from_config` n'acceptait que `default`, qui construit Neo4j + Meili + le magasin de trajectoires avant tout choix de transport (le blocage mesuré plus haut). Le sélecteur accepte maintenant aussi `memory` : le graphe et la recherche sont les stores en mémoire du dépôt (`src/neo4j/mock.rs`, `src/meilisearch/mock.rs`), aucun client Neo4j/Meili, aucun magasin de trajectoires (un no-op). Le mode `memory` est éphémère — une ligne WARN le dit au démarrage, il n'est jamais présenté comme un mode de production.

Le mode `memory` exige le build `--features mock-store` : les stores en mémoire ne sont compilés que pour les tests, jamais dans un binaire de prod sans la feature.

### Recette (sans Docker, sans Neo4j/Meili/NATS)
```bash
# 1. Config minimale (setup_completed lisible UNIQUEMENT en YAML, src/lib.rs:1041)
mkdir -p /tmp/po-test/workspace
cat > /tmp/po-test/config.yaml <<'YAML'
setup_completed: true
server: { port: 18080, workspace_path: /tmp/po-test/workspace, serve_frontend: false }
YAML

# 2. Build avec les stores en mémoire, puis lancement SANS aucune variable externe
cargo build --bin orchestrator --features mock-store
STORAGE_BACKEND=memory EVENTS_TRANSPORT=memory WORKSPACE_PATH=/tmp/po-test/workspace \
target/debug/orchestrator --config /tmp/po-test/config.yaml serve --port 18080 --no-frontend
```

### Verdict
- `/health` → `{"status":"ok","storage_backend":"memory", ...}` : les stores en mémoire répondent `health_check=true`, donc le statut est `ok` sans qu'aucun service externe existe.
- Log de démarrage : `storage backend: memory — ephemeral, all data is lost at shutdown (not a production mode)`.
- Aucun `NEO4J_URI`/`MEILISEARCH_URL`/`NATS_URL` requis, aucun conteneur.
