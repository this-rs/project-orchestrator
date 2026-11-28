# Contrat filaire du chat — inventaire opposable

Source : `backend/src/chat/types.rs`, `backend/src/api/ws_chat_handler.rs`, `backend/src/api/chat_handlers.rs`,
`backend/src/api/ws_auth.rs`, `backend/src/api/routes.rs`, `backend/src/chat/manager.rs`, `backend/src/neo4j/chat.rs`.
Chaque affirmation porte `chemin:ligne`. Rédigé en lecture seule, le 10/11/2026.

## 1. Deux transports, un seul vocabulaire d'événements

- **WebSocket** `/ws/chat/{session_id}` : flux bidirectionnel temps réel. `ws_chat_handler.rs:429`.
- **REST** `/api/chat/sessions/...` : ouverture, historique, envoi, interruption, permission. `routes.rs:2072-2205`.

Les deux transports partagent le même type d'événement serveur→client, `ChatEvent` (`types.rs:710`), sérialisé
avec un champ discriminant `type` (`#[serde(tag = "type", rename_all = "snake_case")]`, `types.rs:709`). C'est le
seul vocabulaire opposable côté serveur. Le REST renvoie les mêmes événements persistés (avec `seq`), le WS les
renvoie en direct **et** en rejeu.

## 2. Serveur → client : `ChatEvent`, 41 variantes

Le nom filaire de chaque variante est donné par `event_type()` (`types.rs:1354-1398`). Champs et ordre d'apparition
dans un tour normal en §5.

| # | Variante | Nom filaire | Champs (types) |
|---|---|---|---|
| 1 | UserMessage | `user_message` | `content: String` |
| 2 | RefsResolved | `refs_resolved` | `refs: Vec<RefResolution>` |
| 3 | CitesResolved | `cites_resolved` | `refs: Vec<RefResolution>` |
| 4 | SystemHint | `system_hint` | `content: String` |
| 5 | AssistantText | `assistant_text` | `content: String`, `parent_tool_use_id?: String` |
| 6 | Thinking | `thinking` | `content: String`, `parent_tool_use_id?: String` |
| 7 | ToolUse | `tool_use` | `id: String`, `tool: String`, `input: Value`, `parent_tool_use_id?`, `category?`, `canonical?` |
| 8 | ToolResult | `tool_result` | `id: String`, `result: Value`, `is_error: bool`, `parent_tool_use_id?` |
| 9 | ToolUseInputResolved | `tool_use_input_resolved` | `id`, `input: Value`, `parent_tool_use_id?` |
| 10 | ToolCancelled | `tool_cancelled` | `id`, `parent_tool_use_id?` |
| 11 | ToolTiming | `tool_timing` | `id`, `called_at?/started_at?/permission_requested_at?/permission_resolved_at?/run_started_at?: f64`, `ended_at: f64`, `cancelled/incomplete: bool`, `permission_outcome?: String`, `parent_tool_use_id?` |
| 12 | PermissionRequest | `permission_request` | `id`, `tool`, `input`, `parent_tool_use_id?`, `category?`, `canonical?`, `tool_use_id?`, `always?: bool`, `consent?: ConsentSubject` |
| 13 | AskUserQuestion | `ask_user_question` | `id`, `tool_call_id`, `questions: Value`, `input: Value`, `parent_tool_use_id?`, `synthetic?: bool` |
| 14 | Result | `result` | `session_id`, `duration_ms: u64`, `cost_usd?: f64`, `subtype: String`, `is_error: bool`, `num_turns?: i32`, `result_text?`, `cost?`, `usage?`, `model?`, `stop_reason?`, `error_code?`, `error_message?` |
| 15 | StreamDelta | `stream_delta` | `text: String`, `parent_tool_use_id?` |
| 16 | StreamingStatus | `streaming_status` | `is_streaming: bool` |
| 17 | PendingQueue | `pending_queue` | `messages: Vec<PendingQueueEntry>` |
| 18 | Error | `error` | `message`, `parent_tool_use_id?`, `code?`, `reason?`, `index?: usize`, `request_id?` |
| 19 | PermissionDecision | `permission_decision` | `id`, `allow: bool`, `scope?`, `rule?` |
| 20 | PermissionModeChanged | `permission_mode_changed` | `mode: String`, `policy_mode?` |
| 21 | ModelChanged | `model_changed` | `model`, `reason?` |
| 22 | CompactionStarted | `compaction_started` | `trigger: String` |
| 23 | CompactionRecovery | `compaction_recovery` | `hint_tokens: u32`, `build_latency_ms: u64`, `recovery_success: bool` |
| 24 | CompactBoundary | `compact_boundary` | `trigger: String`, `pre_tokens?: u64` |
| 25 | SystemInit | `system_init` | `cli_session_id`, `model?`, `tools: Vec<String>`, `mcp_servers: Vec<Value>`, `permission_mode?`, `provider?`, `capabilities?`, `tool_policy?`, `policy_mode?`, `engine?`, `degraded_features?` |
| 26 | Workflow | `workflow` | `subtype: String`, `data: Value` (passé tel quel) |
| 27 | AutoContinue | `auto_continue` | `session_id`, `delay_ms: u64` |
| 28 | AutoContinueStateChanged | `auto_continue_state_changed` | `session_id`, `enabled: bool` |
| 29 | Retrying | `retrying` | `attempt: u32`, `max_attempts: u32`, `delay_ms: u64`, `error_message: String` |
| 30 | BackgroundOutput | `background_output` | `source`, `content`, `received_at: DateTime<Utc>`, `correlation_id?`, `tool?: BackgroundTool` |
| 31 | TeamMessage | `team_message` | `from_session_id: Uuid`, `from_name`, `content` |
| 32 | SessionError | `session_error` | `reason`, `message`, `received_at: DateTime<Utc>` |
| 33 | ToolsCancelled | `tools_cancelled` | `cli_pid?: u32`, `killed_count: usize`, `requested_by: String` |
| 34 | ActiveTasksUpdate | `active_tasks_update` | `tasks: Vec<BackgroundTaskInfo>` |
| 35 | SecretRequest | `secret_request` | `id`, `name`, `reason`, `exists: bool` |
| 36 | SecretRequestResolved | `secret_request_resolved` | `id`, `outcome: String` |
| 37 | SessionClosed | `session_closed` | `session_id`, `reason?` (`closed`/`idle`/`error`) |
| 38 | ConversationRelayed | `conversation_relayed` | `from_session_id`, `to_session_id`, `from_provider`, `to_provider`, `relayed_entries: usize`, `omitted_entries: usize`, `moved_by`, `conversation_id?` |
| 39 | ProviderSwitched | `provider_switched` | `session_id`, `from_provider`, `to_provider`, `model`, `relayed_entries?`, `omitted_entries?` |
| 40 | ProviderOutage | `provider_outage` | `provider_id`, `phase` (`down`/`up`), `cause?`, `failures: u32`, `started_at`, `duration_ms?` |
| 41 | ProviderNotice | `provider_notice` | `kind`, `code?`, `message`, `severity` (`action_required`/`latency`/`diagnostic`), `data?` |

Tous les champs `?` ci-dessus sont `Option` avec `skip_serializing_if = "Option::is_none"` (donc absents du fil
quand `None`). Les champs `bool` à défaut (`is_error`, `cancelled`, `incomplete`, `allow`) sont toujours présents.

### Champs serveur ajoutés autour de chaque événement (hors enum)

- `seq` : sur les événements **live** du WS, `0` (`live_frame`, `ws_chat_handler.rs:1230`). Sur les événements
  **rejoués**, le `seq` Neo4j réel (`replayed_frame`, `ws_chat_handler.rs:1205`).
- `created_at` : secondes epoch, millisecondes en fraction (`wire_seconds`, `ws_chat_handler.rs:1190-1192`).
  Live = heure de *forward* (`live_frame:1232`) ; rejoué = heure de *stockage* (`replayed_frame:1202`).
- `replaying` : `true` sur rejeu et snapshot, absent sur le live (`replayed_frame:1206`, `ws_chat_handler.rs:449,479`).
- `id` : en REST (`list_messages`), l'UUID Neo4j de l'événement n'est injecté comme `id` **que si l'événement n'a
  pas déjà son propre `id`** (un `tool_use`/`tool_result` porte déjà un `tool_call_id`), `chat_handlers.rs:305-309`.

## 3. Client → serveur : le vrai ensemble, et un enum mort

### 3.1 Sur le WebSocket — `WsChatClientMessage`, 9 variantes (`ws_chat_handler.rs:47-107`)

| Variante | Nom filaire | Champs |
|---|---|---|
| UserMessage | `user_message` | `content: String`, `attachments: Vec<Uuid>`, `queue: bool`, `refs: Vec<Value>` |
| QueueOp | `queue_op` | `op: QueueOp` (aplati) — `edit`/`remove`/`prioritize`/`send_now`/`snapshot` |
| Interrupt | `interrupt` | — (unitaire) |
| PermissionResponse | `permission_response` | `id?: String`, `allow: bool`, `scope: PermissionAnswerScope` |
| InputResponse | `input_response` | `id?: String`, `content: String` |
| SetPermissionMode | `set_permission_mode` | `mode: String` |
| SetModel | `set_model` | `model: String` |
| SetAutoContinue | `set_auto_continue` | `enabled: bool` (accepté et **ignoré**, `ws_chat_handler.rs:1141-1148`) |
| CancelTools | `cancel_tools` | — (unitaire) |

### 3.2 Sur REST — corps dédiés

`POST /messages` → `SendMessageRequest { content, attachments, refs }` (`chat_handlers.rs:1042-1051`) ·
`POST /permissions/{request_id}` → `PermissionAnswerRequest { allow, scope }` (`chat_handlers.rs:1032-1037`) ·
`POST /interrupt` → `InterruptRequest { scope?, cascade? }` (`chat_handlers.rs:909-919`) ·
`POST /api/chat/sessions` → `ChatRequest` (43 champs, `types.rs:384-527`, la majorité `#[serde(skip)]` interne) ·
`POST /switch-provider` → `SwitchProviderBody { provider: String, model?: String, message: String }` (`chat_handlers.rs:6138-6149`) ·
`POST /cancel-tools`, `POST /close`, `POST /cancel-task/{task_id}` sans corps.

### 3.3 La frontière contrat/produit à découper en premier

`ClientMessage` (`types.rs:1639-1650`) — 3 variantes (`UserMessage { content }`, `PermissionResponse { allow,
reason }`, `InputResponse { content }`) — **n'est jamais utilisé sur le fil** : seuls les tests et le ré-export
`chat/mod.rs:96` le référencent (`grep` du dépôt). C'est le cas type de contrat mélangé au produit : ce type décrit
une version morte du protocole. Le vrai vocabulaire client→serveur est `WsChatClientMessage` + les corps REST. Toute
matrice de parité bâtie sur `ClientMessage` est fausse.

## 4. Points de reprise (reconnexion avec rattrapage)

1. Curseur : paramètre de requête `last_event: i64` (`WsChatQuery.last_event`, `ws_chat_handler.rs:36-42`).
2. Seuil de court-circuit : `last_event >= 1_000_000_000_000` (le frontend envoie `Number.MAX_SAFE_INTEGER`)
   → aucun rejeu Neo4j, l'historique passe par REST (`ws_chat_handler.rs:256,340-346`).
3. Rejeu : `get_events_since(session_id, after_seq)` → `WHERE e.seq > after_seq ORDER BY e.seq ASC LIMIT 10000`
   (`neo4j/chat.rs:2193-2197`, `manager.rs:14641-14650`). Le `seq` est un compteur Neo4j par session, index composé
   `(session_id, seq)` (`neo4j/chat.rs:2215-2216`).
4. Séquence de connexion **exacte** (l'ordre est le contrat) :
   a. `ready` (client) → `auth_ok` (serveur). Le serveur accepte `ready` nu ou `{"type":"ready"}`, timeout 5 s sinon
      il envoie `auth_ok` quand même (`ws_auth.rs:313-375`). `auth_ok = { type, user:{id,email,name}, features? }`
      (`ws_auth.rs:277-290`).
   b. **Phase 1** : rejeu des événements persistés (frames avec `seq` réel + `replaying:true`), `ws_chat_handler.rs:258-339`.
   c. **Phase 1.5a** : abonnement broadcast/NATS **avant** le snapshot (anti-course), `ws_chat_handler.rs:354-403`.
   d. **Phase 1.5b** : si un flux tourne, `streaming_events` (seq 0, `replaying:true`), puis `partial_text`
      (frame synthétique `{type:"partial_text", content, seq:0, replaying:true}`, hors enum, `ws_chat_handler.rs:474-479`),
      puis `streaming_status { is_streaming:true }` (`ws_chat_handler.rs:491-494`).
   e. `pending_queue` (toujours, même vide), `ws_chat_handler.rs:510-531`.
   f. `replay_complete` (`ws_chat_handler.rs:558`).
   g. `auto_continue_state_changed` (`ws_chat_handler.rs:572-580`).
5. Déduplication de la fenêtre micro-course entre 1.5a et 1.5b : par `fingerprint()` (`types.rs:1406-1618`) —
   `type:id` pour les événements à id, `type:hash(contenu)` sinon, `None` pour `stream_delta`/`streaming_status`/
   `pending_queue`/`active_tasks_update`/`secret_*`/`session_closed`/`provider_switched`/`provider_outage`/
   `team_message` (toujours transmis). `Result` vide l'ensemble des empreintes (fin de tour),
   `ws_chat_handler.rs:674-677,737-739`.
6. Événement rattrapage : si le client **décroche** (`RecvError::Lagged(n)`), le serveur envoie `events_lagged
   { skipped: n }` — au client de refaire un rejeu complet, `ws_chat_handler.rs:684-691`.

## 5. Ordre d'émission dans un tour normal

- **À l'ouverture d'une session** (une seule fois) : `system_init` (du message système `init` du moteur,
  `manager.rs:7504-7533`), puis la séquence de connexion §4.
- **À chaque tour** (l'ordre intra-tour est dicté par le flux du provider, pas re-déclaré côté PO) :
  `user_message` (écho multi-onglets, `manager.rs:5719,5729`) → `refs_resolved` **avant** le premier événement
  assistant si la question porte des références (`types.rs:713-717`) → `thinking` / `stream_delta` / `assistant_text`
  → `tool_use` → `permission_request` (si la permission est demandée) → [`permission_decision` après réponse] →
  `tool_result` → `tool_timing` → ... → `cites_resolved` **après** la réponse si l'agent a cité (`types.rs:721-729`)
  → `result` (fin de tour).
- `stream_delta` est le jeton temps réel ; `assistant_text` est le bloc texte finalisé (`manager.rs:7451-7460`).
  `ToolUse` naît de `ContentBlockStart` (entrée vide), `ToolUseInputResolved` complète l'entrée si
  `AssistantMessage` fournit le plein input après (`types.rs:774-781`, `manager.rs:7464-7491`).

## 6. États terminaux (trames de fin / erreur / interruption / annulation)

- **Fin de tour** : `result` (`subtype ∈ success|error_max_turns|error_during_execution`, `is_error`, `stop_reason`
  neutre ∈ `completed|max_turns|max_tokens|interrupted|refusal|budget_exceeded|error`, `error_code`/`error_message`
  stables), `types.rs:881-921`.
- **Erreur récupérable mi-tour** : `error` (avec `code`/`reason`/`index`/`request_id` typés), `types.rs:938-958`.
- **Erreur fatale de session** : `session_error` (`reason`, `message`, `received_at`) — les messages suivants sur
  ce `session_id` deviennent des no-op jusqu'à un `create_session`/`resume_session`, `types.rs:1166-1189`.
- **Interruption utilisateur** : `tool_cancelled` par `ToolUse` sans `ToolResult` (`types.rs:782-790`), puis `result`
  avec `stop_reason == "interrupted"` (`manager.rs:3435`).
- **Annulation d'outil sans fin de tour** : `tools_cancelled` (`cli_pid`, `killed_count`, `requested_by`), suivi du
  `tool_result` annulé (`types.rs:1145-1165`). Distinction `interrupt` (fin de tour + abort) vs `cancel_tools`
  (SIGINT aux descendants, le tour continue), `chat_handlers.rs:837-899`.
- **Fermeture** : `session_closed` (`reason ∈ closed|idle|error`), `types.rs:1237-1243`.
- **Dormance** : frame `session_dormant` quand le broadcast se ferme (nettoyage idle), `ws_chat_handler.rs:693-707`.
