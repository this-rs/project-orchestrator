# Contrat de fil du chat (WebSocket `/ws/chat/{session_id}`)

Trames d'exemple et contrat au niveau des champs, générés par le backend (décision A44).
Ne modifiez aucun fichier de ce dossier à la main, sauf ce README.

## Contenu

- `server-events.json` : chaque variante de `ChatEvent` (`src/chat/types.rs`), sous `events.<tag>`.
- `client-messages.json` : chaque variante de `WsChatClientMessage` (`src/api/ws_chat_handler.rs`), sous `messages.<tag>`.
- `control-frames.json` : les trames construites à la main par le handler WS (`frames.<tag>`), et
  `event_envelope`, les champs `seq` / `replaying` / `created_at` que le handler ajoute aux événements
  (`created_at` : secondes depuis l'epoch, millisecondes en fraction, comme l'historique REST).
- `SHA256SUMS` : une ligne `<sha256>  <fichier>` par fichier JSON (format `shasum -a 256`).

Chaque entrée porte `fields` (`type` JSON, `required`, `nullable: true` si le champ peut valoir `null`) et
`examples.full` / `examples.minimal`. Un champ est `required` s'il est présent dans la trame `minimal`.
Le `type` vient de l'exemple `full` : pour un champ libre (`input`, `result`, `data`), il est indicatif.
Seuls les champs de premier niveau sont décrits ; la forme des objets imbriqués se lit dans les exemples.
Hors JSON : le client envoie le texte `ready` après l'ouverture, avant de recevoir `auth_ok`.

## Génération

Source : `src/chat/wire_contract.rs` (module de test). Le test `wire_contract_matches_committed_fixtures`
régénère les fichiers en mémoire et les compare octet pour octet à ceux du dépôt. Pour les réécrire :
`UPDATE_CHAT_CONTRACT=1 cargo test --lib chat::wire_contract`.
Toute évolution du fil (variante, champ, attribut serde, trame de contrôle) passe par cette régénération
**dans le même commit**, sinon le test échoue. Les trames de contrôle sont des copies statiques : si le
handler change, mettez à jour `control_frames()` dans le module.

## Trames gardées pour compatibilité

- `set_auto_continue` (client) : accepté et **ignoré** depuis la décision 7b95342a (12/10/2026) —
  l'auto-continue est toujours actif, plafonné à 5 continuations d'affilée, sur les deux moteurs.
  Le handler journalise en debug et répond `auto_continue_state_changed` avec l'état réel. Un
  frontend à jour ne l'envoie plus ; il reste au contrat pour ne pas refuser les anciens.
- `auto_continue_state_changed` (serveur) : toujours envoyé une fois à la connexion (après
  `replay_complete`, `seq: 0`) et en réponse à `set_auto_continue`, pour les anciens frontends
  qui affichent encore un interrupteur. `enabled` vaut `true` sauf si l'opérateur a coupé
  l'auto-continue (`CHAT_AUTO_CONTINUE=false`). Un frontend à jour peut l'ignorer.
- `auto_continue` (serveur) : inchangé, annonce une continuation automatique (`delay_ms`).

Le format de ces trames ne change pas : les fichiers JSON et `SHA256SUMS` sont inchangés.

## Côté frontend

Copiez le dossier tel quel, puis vérifiez la copie depuis le dossier copié : `shasum -a 256 -c SHA256SUMS`.
Les types TypeScript se comparent ensuite à `fields`, et les exemples servent de trames de test.

## `provider-additions.json` (écrit à la main, statut `planned`)

Les champs que le harness multi-provider AJOUTERA au fil et à l'API (`category`, `canonical`,
`cost{usd,basis}`, `usage`, `synthetic`, `provider`, `capabilities`, `tool_policy`, `policy_mode`,
`session_closed`, DTO `ChatSession`, `GET /api/chat/providers`, corps et codes d'erreur). Les noms y
sont FIXÉS pour que le frontend code contre eux ; rien n'y est encore émis sauf mention `since`.
Ce fichier n'est pas dans `SHA256SUMS` : il n'est pas généré. Un champ livré passe dans
`server-events.json` et sort d'ici, dans le même commit.

## Fin de tour par une limite configurée (`error.code`, ISO5)

Une session NATIVE n'a, par défaut, ni limite de durée par tour ni budget de jetons (comme Claude Code).
Quand l'opérateur en pose une (`chat.native_turn_timeout_secs`, `chat.native_session_max_tokens`, ou les
variables `CHAT_NATIVE_TURN_TIMEOUT_SECS`, `CHAT_NATIVE_SESSION_MAX_TOKENS`) et qu'elle arrête un tour, le fil
porte, AVANT le `result`, une trame `error` dont `code` dit laquelle :

- `turn_time_limit` : « Turn stopped after N by the configured turn time limit (L, …) ». Le `result` qui suit a
  `subtype: "error_timeout"`. Le travail du tour est gardé, et le modèle lit au tour suivant une ligne
  `[system]` qui lui dit pourquoi son tour s'est arrêté. Le tour n'est pas relancé automatiquement.
- `session_budget_reached` : « Session token budget reached: S of L tokens (cache reads not counted; …) », ou
  le budget USD. Le `result` qui suit a `subtype: "error_max_budget"` et `stop_reason: "budget_exceeded"`. Tant
  que le budget n'est pas relevé, chaque nouveau tour s'arrête avant d'appeler le modèle, et redit la trame.

Le `message` est bâti des seuls nombres, jamais d'un texte du provider. Un `result` `budget_exceeded` sans
`subtype` du provider prend `error_max_budget` (plus `error_max_budget_usd` : le budget peut être en jetons).
