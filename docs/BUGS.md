# Registre de bugs

Chaque bug de Project Orchestrator suit un cycle uniforme, tracé dans ce document.
Le diagramme `docs/diagrams/po-bugs.mmd` est le registre visuel : un nœud par bug,
coloré selon son statut.

## Convention

Un bug = quatre artefacts synchronisés :

| Artefact | Outil | Contenu |
|---|---|---|
| Tâche PO | MCP `task` | titre, description, critères d'acceptation, tags `bug` + `<domaine>` + sévérité |
| Note `gotcha` | MCP `note` | cause racine, solution, noms de fonctions (jamais de n° de ligne) |
| Test de régression | code | échoue sans le correctif, passe avec |
| Nœud dans po-bugs | `docs/diagrams/po-bugs.mmd` | 🔴 ouvert / 🟠 PR ouverte / ✅ corrigé + preuve |

### Tags de tâche

```
bug  <domaine>  <sévérité>
```

Domaines : `neo4j`, `api`, `chat`, `auth`, `runner`, `frontend`, `protocol`, `coverage`, `docs`, `ci`.  
Sévérité : `critical`, `high`, `medium`, `low`.

### Niveau de preuve (nœud ✅)

Le nœud précise COMMENT le correctif est prouvé :

- `mock` : testé via le double de test en mémoire
- `neo4j-reel` : testé contre une vraie instance Neo4j
- `rejoue-sans` : le test de régression a été exécuté sans le correctif et échoue
- `ci-verte` : le correctif est dans un commit CI vert sur `main`

Un nœud sans niveau de preuve est incomplet.

## Cycle de vie

```
découverte
    │
    ▼
tâche PO (status: pending)
    │  + note gotcha (cause racine)
    │  + nœud 🔴 dans po-bugs
    ▼
en cours (status: in_progress)
    │  + test de régression qui échoue sans le correctif
    ▼
PR ouverte (status: in_progress)
    │  + nœud 🟠 dans po-bugs
    ▼
merge + CI verte (status: completed)
       + nœud ✅ dans po-bugs (niveau de preuve)
```

## Inventaire (état au commit `ff5bb7b0`, 2026-10-02)

| Bug | Domaine | Sévérité | Statut | PR / commit |
|---|---|---|---|---|
| `TaskStatus` et `ConstraintType` sérialisés en `{:?}` dans le payload WS et le prompt de compaction | api | medium | ✅ ci-verte | PR #501 (`d42312a2`) |
| `create_release`, `create_decision`, `create_constraint` renvoient `Ok(())` quand le parent est absent | neo4j | high | ✅ ci-verte | PR #500 (`64e508e6`) |
| `protocolApi.triggerEvent` envoyait `{event}` au lieu de `{trigger}` | frontend | high | ✅ ci-verte | PR #466 |
| Routes frontend appelant le backend absent (`retry`, `/api/progress`, `neighborhood`, `runs/{id}/history`) | api | high | ✅ ci-verte | PR #466 + routes ajoutées |
| Tombstone signé par 128 zéros (signature non vérifiée) | auth | critical | ✅ ci-verte | PR #490 |
| Injection Cypher dans `WhereBuilder` (paramètres de type non échappés) | neo4j | critical | ✅ ci-verte | PR #489 |
| `require_auth` en mode anonyme sans configuration explicite | auth | high | ✅ ci-verte | PR #480 |
| `ChatEvent::InputRequest` défini mais jamais émis (type mort, non supprimé) | chat | low | ✅ documenté | commentaire `attention.rs:31` |
| `update_workspace_milestone` paniquait sur id inconnu | api | medium | ✅ ci-verte | PR #483 |
| `check-index.mjs` ne lit pas l'index dérivé de nexus (77 fichiers comptés orphelins) et `--write-orphans` relève les plafonds par dépôt | docs | medium | ✅ rejoue-sans | backend#506 (5 tests rouges sans le correctif) |
| nexus : `%% verified:` non rejouable (`yesterday`, `TODO`, date impossible) accepté par le gate | docs | low | ✅ rejoue-sans | nexus#60 (4 tests rouges sans le correctif) |
| nexus : MSRV 1.88 cassé par `uuid` 1.27 (`rustc 1.89`) parce que `Cargo.lock` n'est pas versionné et que le MSRV n'était pas déclaré | ci | high | ✅ rejoue-sans | nexus#61 (`cargo +1.88 check --all-features`) |
| `session_error` (mort du CLI) non rendu par le frontend, et type mort `ClientMessage` | chat | medium | ✅ rejoue-sans | frontend#198 · `chatAssembly.sessionError.test` |
| `useConversationWs` : trois implémentations sous le même nom ; le parseur privé du panneau de discussion lisait `tool_use.name` (le backend envoie `tool`) | chat | medium | ✅ rejoue-sans | frontend#199 (`InlineConversationPanel.events.test` rouge sur l'ancien panneau) |
| `ChatConfig::mcp_server_config` : aucun appelant de production (`build_options` construit `McpServerConfig::Stdio`) | chat | low | ✅ preuve par absence | backend#508 (`grep` → 0 ; `NATS_URL` : `mcp_server` est un proxy HTTP) |
| `EventReactor` : un `EventTrigger` persistant qui correspond incrémente `triggers_fired` et émet `Trigger::Created`, mais ne démarre aucun run de protocole | protocol | high | 🟠 PR | `fix/bugs-reactor-wake-federation` (#518) · `start_trigger_run` · `test_event_trigger_starts_protocol_run` (rejoue-sans) · tâche PO `adc10369` |
| `/hooks/wake` (termine une tâche) et `/internal/events` (injecte un événement dans le bus) montés sans `require_auth` | auth | high | 🟠 PR | `fix/bugs-reactor-wake-federation` (#518) · `test_state_changing_hooks_require_auth` (rejoue-sans) · tâche PO `21dda16f` |
| Fédération MCP : `SecurityEnforcer` (mutations, liste de serveurs, débit) référencé par un test seulement ; `handle_external` appelait l'outil externe sans contrôle | api | high | 🟠 PR | `fix/bugs-reactor-wake-federation` (#518) · `McpServerRegistry::enforce_call` · `test_external_mutating_tool_is_refused_by_default_policy` (rejoue-sans) · tâche PO `01376efd` |
| `/oauth/authorize` émettait un code dès qu'un cookie de session était présent, alors que l'enregistrement de client (DCR) est ouvert ; `redirect_uri_allowed` acceptait un userinfo (`http://127.0.0.1:80@hôte`) | auth | high | 🟠 PR | `fix/bugs-reactor-wake-federation` (#518) · page de consentement + `authorize_consent` (POST) · `test_oauth_authorize_requires_explicit_consent` (rejoue-sans) · tâche PO `d0d09cae` |
| Une commande d'arrière-plan silencieuse (sortie redirigée) était déclarée morte après 30 min sans vérifier son processus ; la session, vue sans travail de fond, était fermée pour inactivité et le CLI tué avec la commande (« The CLI subprocess for this session has exited ») | chat | high | 🟠 PR | `fix/chat-idle-expiry-kills-background-work` · `background_task_is_idle_dead`, `process_alive`, `cli_background_task_count` · `test_tick_purge_keeps_a_silent_task_*` (rejoue-sans) |

### Bugs non reproduits / invalidés après vérification du code

| Bug signalé | Résultat de vérification |
|---|---|
| `triggerEvent` envoie `{event}` | Invalidé : le code courant envoie `{trigger}`, testé par `apiContract.test.ts` |

## Vérification des covers

Le script `scripts/diagrams/check-index.mjs` vérifie que chaque glob `covers` d'une entrée
`verified` matche au moins un fichier existant. `po-bugs` couvre `docs/BUGS.md` (ce fichier) :
ajouter ce fichier sans le déclarer dans l'index déclencherait le plafond d'orphelins.
