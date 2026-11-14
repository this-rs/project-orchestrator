# Goldens du chat — fixtures à rejouer

Format : un fichier = un tableau JSON, une trame par élément, dans l'ordre du flux.
Trame serveur : objet filaire exact (`type` + champs + `seq`/`created_at`/`replaying`).
Trame client : `{"$client": "<nom filaire>", ...champs}` — nom = `WsChatClientMessage`
(contract.md §3.1), jamais confondu avec une trame serveur.

Comparaison : normaliser les volatils (volatile.md), puis diff **cadre à cadre, dans
l'ordre**, sans trier par `seq` (le live est tout à 0, gaps.md G6). Marqueurs : `created_at`
→ `<ts>`, `session_id` → `<session>`, `tool_call_id` → `<tool-call-N>` (couplage conservé :
`tool_result.id == tool_use.id`). Ne jamais normaliser `content` d'un `user_message` ni `type`.
Ordres non garantis (gaps.md G1) : `interrupt`/`permission_response` peuvent précéder un
`tool_result` déjà prêt — le comparateur accepte ces deux ordres.

À vérifier à l'enregistrement (non nommés/non figés par le contrat) :
- `result` d'interruption : `subtype`/`is_error` exacts (seul `stop_reason:"interrupted"`
  est contractuel) — session-interruption.json.
- `system_init` : position (contract §5 « à l'ouverture » vs scenarios S1 « au premier
  message ») et champs moteur `tools`/`mcp_servers`/`permission_mode`/`capabilities`.
- `tool_result.result` / `tool_use.input` : forme pass-through du moteur, non contractuelle.

Non prouvé tant que non rejoué contre un vrai serveur : l'ordre intra-tour réel (flux
provider), la forme exacte des `result`/`error` par provider, et les variantes
(`refs_resolved`/`cites_resolved`, `always` → `permission_scope_unsupported`, `events_lagged`,
`session_dormant`, `tool_use_input_resolved`, `streaming_events` non vide avant `partial_text`).
