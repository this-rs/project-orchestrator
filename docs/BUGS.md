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

Domaines : `neo4j`, `api`, `chat`, `auth`, `runner`, `frontend`, `protocol`, `coverage`, `docs`.  
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

## Inventaire (état au commit `d42312a2`, 2026-10-02)

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
| `check-index.mjs` ne lit pas l'index dérivé de nexus (77 fichiers comptés orphelins) et `--write-orphans` relève les plafonds par dépôt | docs | medium | 🟠 PR | [fix/diagrams-gate-neighbour-index](https://github.com/this-rs/project-orchestrator/tree/fix/diagrams-gate-neighbour-index) · tâche PO `efa9fd3c` |
| `session_error` (mort du CLI) non rendu par le frontend, et type mort `ClientMessage` | chat | medium | ✅ rejoue-sans | frontend#198 · `chatAssembly.sessionError.test` |
| `useConversationWs` : trois implémentations sous le même nom, dont deux anciens parseurs privés qui contournent `historyEventsToMessages` | chat | low | 🔴 ouvert | tâche PO `c2139265` |
| `ChatConfig::mcp_server_config` : aucun appelant de production (`build_options` construit `McpServerConfig::Stdio`) | chat | low | 🔴 ouvert | tâche PO `e6919f99` · nœud MCPJSON de po-chat-manager |

### Bugs non reproduits / invalidés après vérification du code

| Bug signalé | Résultat de vérification |
|---|---|
| `triggerEvent` envoie `{event}` | Invalidé : le code courant envoie `{trigger}`, testé par `apiContract.test.ts` |

## Vérification des covers

Le script `scripts/diagrams/check-index.mjs` vérifie que chaque glob `covers` d'une entrée
`verified` matche au moins un fichier existant. `po-bugs` couvre `docs/BUGS.md` (ce fichier) :
ajouter ce fichier sans le déclarer dans l'index déclencherait le plafond d'orphelins.
