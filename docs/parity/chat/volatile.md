# Champs à normaliser avant comparaison (golden)

Un golden qui compare des UUID ou des horodatages échoue à chaque rejeu. Liste exacte, par ordre de
fréquence d'apparition sur le fil. Règle : **normaliser vers un marqueur déterministe** (`<uuid>`, `<ts>`,
`<dur>`, `<session>`, `<cli-session>`, `<tool-call-id>`) AVANT toute diff, et ne jamais normaliser un champ
que deux scénarios distincts doivent pouvoir distinguer (ex. le `content` d'un `user_message`).

## 1. `seq` — le plus piégeux
- **Live** : toujours `0` (`live_frame`, `ws_chat_handler.rs:1230`). **Rejoué** : le compteur Neo4j
  (`replayed_frame:1205`, `neo4j/chat.rs:2193-2197`). Deux rejeux de la même conversation produisent le même
  `seq` **seulement si** la même session est rejouée ; une session neuve repart de 1. → normaliser `seq` en
  `<seq>` quand on compare des formes, ou comparer la suite relative quand on teste l'ordre.
- Attention : `seq` est **par session**, pas global (`neo4j/chat.rs:2215-2216`). Deux sessions ont chacune
  leur `seq=1`.

## 2. `created_at` — deux significations, deux normalisations
- Unité : secondes epoch, millisecondes en fraction (`wire_seconds`, `ws_chat_handler.rs:1190-1192`).
- Live : l'heure de *forward* côté serveur (non déterministe) ; rejoué : l'heure de *stockage*.
  → `<ts>`, sauf si le test vérifie précisément l'unité (alors ne comparer que l'ordre, pas la valeur).

## 3. `session_id` — généré serveur à l'ouverture
- `CreateSessionResponse.session_id` (`types.rs:1904-1906`), `Result.session_id`, `SessionClosed.session_id`,
  `AutoContinue.session_id`, `auto_continue_state_changed.session_id`. UUID v4, nouveau à chaque ouverture.
  → `<session>` (et réinjecter la même valeur partout où le même tour la porte).

## 4. `cli_session_id` — généré par le moteur (Claude CLI / natif)
- `SystemInit.cli_session_id` (`types.rs:1020-1022`), repris dans `ChatSession.cli_session_id` pour `--resume`
  (`types.rs:1727-1729`). → `<cli-session>`.

## 5. `id` / `tool_use_id` / `tool_call_id` / `request_id` / `parent_tool_use_id` / `correlation_id`
- `ToolUse.id`, `ToolResult.id`, `ToolCancelled.id`, `ToolTiming.id`, `PermissionRequest.id`,
  `PermissionDecision.id`, `AskUserQuestion.id` : ce sont des `tool_call_id` du moteur (aléatoires par rejeu).
  → `<tool-call-id>`, mais **conserver la correspondance** : le `ToolResult.id` DOIT égaler le `ToolUse.id` du
  même appel — c'est une assertion à faire APRÈS normalisation (remplacer par le même jeton indexé, ex.
  `<tool-call-1>`).
- `PermissionRequest.tool_use_id` (`types.rs:846-847`) pointe le `tool_use` associé : même règle de couplage.
- `Error.request_id` (`types.rs:956-957`) = l'`id` de la `permission_request`/`permission_response` refusée.
- `AskUserQuestion.tool_call_id` (`types.rs:869-870`) = le `tool_use` qui répondra.

## 6. Horodatages à l'intérieur des corps
- `BackgroundOutput.received_at`, `SessionError.received_at`, `ProviderOutage.started_at`
  (`types.rs:1115-1116,1187-1188,1313-1314`) : `DateTime<Utc>` sérialisé ISO-8601. → `<ts>`.
- `ToolTiming.called_at/started_at/permission_requested_at/permission_resolved_at/run_started_at/ended_at`
  (`types.rs:798-821`) : `f64` epoch. → `<ts>`.

## 7. Durées et coûts mesurés
- `Result.duration_ms` (`types.rs:883`), `Result.cost_usd`, `Result.usage` (jetons), `ToolTiming` (déjà couvert).
  `AutoContinue.delay_ms`, `Retrying.delay_ms`, `CompactionRecovery.build_latency_ms` : non déterministes.
  → `<dur>`, `<cost>`, `<usage>`, sauf test ciblé de la **forme** (présence, type numérique).

## 8. `Result.model` / `SystemInit.model` / `ModelChanged.model` / `ProviderSwitched.model`
- Nom de modèle résolu par le serveur à l'ouverture (peut changer selon la config, le routage cognitif, le
  provider). → `<model>` si le test n'est pas celui du routage lui-même.

## 9. `Workflow.data.uuid` et tout payload CLI passé tel quel
- `Workflow { subtype, data }` passe `data` **verbatim** (`types.rs:1066-1072`) ; le `uuid` interne du CLI est
  aléatoire (le dédup s'en sert, `types.rs:1426-1433`). → normaliser `data.uuid` → `<uuid>`, garder le reste.

## 10. `id` injecté en REST (`list_messages`)
- L'UUID Neo4j de l'événement devient `id` quand l'événement n'en a pas (`chat_handlers.rs:305-309`).
  → `<event-id>` (et surtout : il est ABSENT des frames WS rejouées/live, qui n'ont que le `id` métier).

## 11. UUID de session/entité dans les corps
- `TeamMessage.from_session_id` (`types.rs:1141`), `PendingQueueEntry.id` (`types.rs:2231+`),
  `ConversationRelayed.conversation_id`, `SecretRequest.id`, `RefResolution.id/label/subtitle`
  (`refs/wire.rs:79-93`), `BackgroundTaskInfo.id` (parfois synthétique `recovered-{pid}`, `types.rs:587-593`).
  → `<uuid>` / `<recovered-pid>`.

## 12. `user.id` / `user.email` / `user.name` de `auth_ok`
- Identité du JWT (`ws_auth.rs:277-290`). → `<user-id>`, `<email>`, `<name>`.

## 13. PID et compteurs
- `ToolsCancelled.cli_pid`, `killed_count` (`types.rs:1159-1161`), `ProviderOutage.failures/duration_ms`
  (`types.rs:1311-1317`), `Retrying.attempt/max_attempts` : dépendent de l'environnement. → `<pid>`, `<n>`.

## Pourquoi chacun (le contrat sous-jacent)
- Tout ce qui est généré **côté serveur** (uuid de session, `created_at` de forward, `seq`) est par définition
  volatil : c'est l'horodatage et l'identité du *moment de l'exécution*, pas du *contenu du tour*.
- Tout ce qui est généré **côté moteur** (`tool_call_id`, `cli_session_id`, `Workflow.data.uuid`) est aléatoire
  par rejeu, même sur la même conversation.
- Les **durées/coûts/usage** sont des mesures, pas des données de contrat : leur valeur ne doit jamais faire
  échouer un golden, seulement leur présence et leur type.
- Les **correspondances** (`ToolResult.id == ToolUse.id`, `Error.request_id == permission_request.id`) sont,
  elles, du contrat : à vérifier après normalisation, jamais à écraser.
