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
