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
