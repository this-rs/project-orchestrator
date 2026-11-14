# Scénarios à enregistrer — replaçables sans l'auteur

Chaque scénario nomme la séquence attendue de trames. Source des noms de trames : `contract.md`.
Un enregistrement = le flux WS `ready → auth_ok → … → fermeture` complet, ou l'échange REST, capturé
avec un enregistreur qui note `type`, `seq`, `replaying`, `created_at` et les champs du corps.
Références : `ws_chat_handler.rs`, `chat_handlers.rs`, `manager.rs`.

## S1 — Ouverture de session (create + connexion WS)
1. `POST /api/chat/sessions` `{ message, cwd?, project_slug?, model?, permission_mode? }`.
2. Réponse `CreateSessionResponse { session_id, stream_url: "/ws/chat/{id}", execution_place?, access?, notices? }`
   (`types.rs:1904-1917`, `chat_handlers.rs:59-241`).
3. `GET /ws/chat/{session_id}` avec `last_event=0` ; envoyer `ready`.
4. Attendu, **dans cet ordre** : `auth_ok { user, features? }` → rejeu (vide, session neuve) → `pending_queue
   { messages: [] }` → `replay_complete` → `auto_continue_state_changed { session_id, enabled }`
   (`ws_chat_handler.rs:248-580`).
5. Au premier message : `system_init { cli_session_id, model, tools, … }` puis le tour (voir S2).

## S2 — Tour simple (texte, pas d'outil)
1. `user_message` (echo, `content` = texte composé) — `manager.rs:5719,5729`.
2. `stream_delta*` (jetons) et/ou `assistant_text` (bloc final), `manager.rs:7451-7460`.
3. `result { session_id, duration_ms, subtype:"success", is_error:false, cost_usd, num_turns, usage, model,
   stop_reason:"completed" }` — `types.rs:881-921`, `manager.rs:7425-7449`.
- Variante à enregistrer aussi : question avec référence `#plan:<uuid>` → `refs_resolved` AVANT le premier
  événement assistant ; réponse citant une entité → `cites_resolved` APRÈS la réponse (`types.rs:713-729`).

## S3 — Appel d'outil (Bash/Read/Edit)
1. `user_message` → `assistant_text?`/`stream_delta` → `tool_use { id, tool, input, category?, canonical? }`.
2. `tool_result { id, result, is_error }` (même `id` que le `tool_use`).
3. `tool_timing { id, called_at?, started_at?, ended_at, cancelled:false }` une fois le résultat/cancel en place
   (`types.rs:791-830`).
4. `assistant_text?` (la suite) → `result`.
- Enregistrer aussi `tool_use_input_resolved` : `ContentBlockStart` produit un `ToolUse` à `input` vide, puis
  `AssistantMessage` plein → `tool_use_input_resolved` (`types.rs:774-781`).

## S4 — Demande de permission et réponse
1. `permission_request { id, tool, input, tool_use_id?, always?, consent?, category?, canonical? }`.
2. Client répond `permission_response { id, allow: true, scope: "once" }` (ou `session`/`always`).
3. Attendu : `permission_decision { id, allow, scope? }` (persisté à côté de la requête, `types.rs:959-975`),
   puis `tool_result`, `tool_timing` (`permission_outcome:"allowed"`), `result`.
- Variantes : `allow:false` → `tool_result { is_error:true }` + `permission_decision { allow:false }`.
  Scope `always` non offert → frame `error { code:"permission_scope_unsupported", reason, request_id }`
  (`ws_chat_handler.rs:1381-1397`). Réponse d'un non-propriétaire → `error { code:"permission_forbidden",
  reason:"not_owner" }` (`ws_chat_handler.rs:1586-1610`). Double-clic → seconde réponse ignorée
  (`ws_chat_handler.rs:988-996`).

## S5 — Interruption en cours de tour
1. `user_message` → `stream_delta`/`tool_use` (un outil long, ex. `sleep 60`).
2. Client envoie `interrupt` (WS) ou `POST /interrupt { scope:"turn_and_tools" }`.
3. Attendu : `tool_cancelled { id }` pour chaque `ToolUse` sans `ToolResult` (`types.rs:782-790`), puis
   `result { stop_reason:"interrupted" }` (`manager.rs:3435`).
- Enregistrer les deux `scope` : `turn` (laisse les descendants vivants) vs `turn_and_tools` (SIGINT aux
  descendants), et `cascade:true` (rapport `{ stopped, total }`), `chat_handlers.rs:948-1023`.

## S6 — Reconnexion avec rattrapage (mid-stream join)
1. Ouvrir une session, lancer un tour long (`tool_use` en cours).
2. **Déconnecter**, puis reconnecter `GET /ws/chat/{id}?last_event=<dernier seq vu>`.
3. Attendu, dans l'ordre : `auth_ok` → rejeu des événements persistés depuis `last_event` (frames `seq` réel,
   `replaying:true`) → `streaming_events` snapshot (seq 0, `replaying:true`) → `partial_text` (texte delta
   accumulé) → `streaming_status { is_streaming:true }` → `pending_queue` → `replay_complete` →
   `auto_continue_state_changed` (`ws_chat_handler.rs:248-580`).
4. Vérifier qu'aucun événement n'est dupliqué (dédup par `fingerprint`) et aucun perdu (abonnement avant
   snapshot, `ws_chat_handler.rs:354-403,533-550`).
- Variante : décrochage réel → `events_lagged { skipped: n }` puis rejeu complet (`ws_chat_handler.rs:684-691`).

## S7 — Erreur de provider
1. `user_message` sur un provider qui refuse (ex. vault verrouillé, `invalid_request`, `rate_limited`).
2. Attendu : `retrying { attempt, max_attempts, delay_ms, error_message }` avant chaque nouvelle tentative
   (`types.rs:1086-1097`) ; puis `result { is_error:true, subtype:"error_during_execution", error_code,
   error_message }` (`types.rs:912-920`).
3. Si l'envoi échoue avant tout flux : frame `error { type, message, code?, retryable?, provider_id? }` — le
   `code` (ex. `credentials_locked`) provient de `add_typed_failure` (`ws_chat_handler.rs:1564-1581`).
4. Si le transport tombe : `session_error { reason, message }` (`types.rs:1180-1189`).

## Matrice de rejeu minimale
| Scénario | Transport | Trames critiques attendues |
|---|---|---|
| S1 | REST + WS | `auth_ok`→`pending_queue`→`replay_complete`→`auto_continue_state_changed`→`system_init` |
| S2 | WS | `user_message`→`stream_delta`/`assistant_text`→`result` |
| S3 | WS | `tool_use`→`tool_result`→`tool_timing`→`result` |
| S4 | WS | `permission_request`→`permission_decision`→`tool_result` |
| S5 | WS/REST | `tool_cancelled`→`result{stop_reason:interrupted}` |
| S6 | WS | rejeu `seq>N`→`partial_text`→`replay_complete`, zéro doublon |
| S7 | WS | `retrying`→`result{is_error,error_code}` ou `error{code}` |
