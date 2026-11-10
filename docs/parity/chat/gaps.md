# Ce que le contrat ne dit pas

Comportements observables, écrits nulle part, qui cassent une migration **en silence**. Chacun est localisé
à l'endroit exact où le code le produit, pour qu'une relecture future vérifie s'il est devenu explicite.

## G1 — L'ordre entre un message client et un événement serveur est non déterministe
La boucle WS est un `tokio::select!` sur **cinq** branches : `ctl_rx` (frames de contrôle), `event_rx`
(broadcast local), `nats_msg`, `ping_interval`, `ws_receiver` (messages client) (`ws_chat_handler.rs:637-1181`).
Un `interrupt`/`permission_response` reçu pendant qu'un `tool_result` est prêt est traité **selon celle des
branches qui se réveille en premier**. Aucun contrat n'ordonne une réponse client avant la fin d'un tour.
Un golden qui suppose `interrupt` avant `tool_result` est faux.

## G2 — Un message envoyé pendant un tour l'interrompt, SAUF `queue: true`
`WsChatClientMessage::UserMessage.queue` (`ws_chat_handler.rs:58-59`) : absent/`false` → le message
**interrompt** le flux en cours (comportement historique) ; `true` → le message est **retenu** et délivré à la
fin (voir `pending_queue`). Le `queue` est un ajout récent ; les anciens clients qui l'ignorent gardent
l'interruption. La frontière est un booléen de sérialisation par défaut (`#[serde(default)]`), pas un contrat
écrit.

## G3 — La forme exacte d'une permission refusée est en `code`/`reason`, pas dans le type
Il existe **quatre** refus distincts, tous des frames `error` plates, différenciées seulement par leur champ
`code` (jamais par un type dédié) :
- `permission_forbidden` + `reason ∈ not_owner|owner_unreadable|not_a_person` (`ws_chat_handler.rs:1401-1415,1586-1610`).
- `permission_scope_unsupported` + `reason` = le scope refusé (`ws_chat_handler.rs:1381-1397`).
- `mirror_read_only` (session miroir d'un sous-agent) (`ws_chat_handler.rs:1503-1513`).
- `cancel_failed` + `reason` (annulation d'outil relayée échouée) (`ws_chat_handler.rs:1800-1810`).
Un client qui lit `message` (texte) sans `code` affiche le bon texte mais ne peut pas router le bon écran. La
migration qui change un `code` sans changer le `message` casse silencieusement toute UI qui branche sur `code`.

## G4 — `replay_complete`, `partial_text`, `events_lagged`, `session_dormant`, `auth_ok` sont HORS enum
Ces cinq trames sont des `serde_json::json!` ad hoc, pas des variantes de `ChatEvent` (`ws_chat_handler.rs:558,
474-479, 687-691, 703-706` ; `ws_auth.rs:277-290`). Elles n'apparaissent donc ni dans `event_type()`
(`types.rs:1354-1398`) ni dans `fingerprint()`, et ne sont **jamais persistées**. Toute parité qui rejoue
l'historique REST et le compare au flux WS échouera : l'historique n'a pas de `replay_complete`. C'est la
frontière la plus nette entre « événements persistés » et « trames de transport ».

## G5 — `streaming_status` est émis deux fois, avec deux statuts possibles, mais jamais persisté
`StreamingStatus { is_streaming }` existe comme variante (`types.rs:928-929`) ET comme frame synthétique en
snapshot (`ws_chat_handler.rs:491-494`, toujours `true`). `is_streaming:false` n'est pas explicitement émis à la
fin d'un tour dans le snapshot ; la fin est portée par `result`. Un golden qui attend un
`streaming_status { is_streaming:false }` symétrique est faux.

## G6 — `seq` live est toujours 0 : on ne peut pas ordonner le live par `seq`
Les événements live portent `seq:0` (`live_frame:1230`) ; seuls les rejoués portent un `seq` croissant. Deux
`tool_result` live successifs ont le même `seq:0` : l'ordre n'est porté que par l'ordre du flux WS. Une
normalisation qui trie par `seq` mélange le live (tout à 0) et casse l'ordre réel. C'est l'opposé exact de
l'historique REST, où `seq` est l'ordre (`neo4j/chat.rs:2196`).

## G7 — `id` de `list_messages` est polyphonique : UUID Neo4j OU `tool_call_id`
En REST, `id` est l'UUID Neo4j de l'événement **sauf** si l'événement a déjà un `id` (un `tool_use`/`tool_result`
porte son `tool_call_id`), auquel cas l'UUID n'est PAS injecté (`chat_handlers.rs:305-309`). Sur le WS rejoué,
`id` est toujours le champ métier. La même clé `id` signifie donc deux choses différentes selon le transport et
l'événement — un golden naïf sur `id` compare des UUID à des `tool_call_id`.

## G8 — Le `ClientMessage` de `types.rs` est un contrat mort qui ressemble au vrai
`ClientMessage { UserMessage, PermissionResponse, InputResponse }` (`types.rs:1639-1650`) est exporté
(`chat/mod.rs:96`) et testé, mais **jamais** lu sur le fil. Le vrai ensemble est `WsChatClientMessage` (9
variantes) + corps REST. Un outil qui génère la parité depuis l'AST de `types.rs` produira un contrat faux.
La découpe contrat/produit doit commencer par supprimer ou marquer `ClientMessage` comme historique.

## G9 — Un `error` n'est PAS la fin d'un tour, `session_error` si
`Error` est récupérable mi-tour (un outil, une référence invalide) : le tour continue. `SessionError` signale que
le subprocess/moteur est parti et que **les messages suivants sont des no-op jusqu'à réouverture**
(`types.rs:1166-1189`). Aucun champ ne relie explicitement `session_error` à « il faut un `create_session` » :
c'est écrit dans un doc-comment, pas dans un état machine observable. Une migration qui traite `session_error`
comme un simple `error` affiché laisse le client envoyer des messages qui seront silencieusement ignorés.

## G10 — `set_auto_continue` est accepté mais ignoré
Le serveur répond toujours `auto_continue_state_changed { session_id, enabled: <état réel> }` et ignore la
valeur envoyée (auto-continue est toujours on, décision 7b95342a) (`ws_chat_handler.rs:1141-1148`). Un golden qui
vérifie « j'ai envoyé `enabled:false` donc l'état passe à `false` » échouera. La réponse n'est PAS un écho, c'est
un correctif.

## G11 — Le rejeu est borné à 10 000 événements, sans signal de troncature
`get_events_since` charge `LIMIT 10000` (`manager.rs:14641-14650`, `neo4j/chat.rs:2193-2197`). Si plus de 10 000
événements manquent, le client reçoit les 10 000 plus anciens **sans savoir** qu'il y a un trou (aucun `has_more`,
aucun marqueur). La seule voie de rattrapage complet est de recharger l'historique REST et de passer
`last_event = Number.MAX_SAFE_INTEGER` pour sauter le rejeu (`ws_chat_handler.rs:256,340-346`). Un test de
reconnexion sur une très longue session doit le savoir.

## G12 — La déduplication d'une reconnexion mid-stream est par empreinte, pas par `seq`
Le chevauchement entre l'abonnement (1.5a) et le snapshot (1.5b) est dédupliqué par `fingerprint()` (type+id ou
type+hash de contenu), jamais par `seq` (`ws_chat_handler.rs:533-550`). Deux `assistant_text` de contenu identique
produisent la même empreinte et le second est **supprimé** comme doublon — un golden qui rejoue deux phrases
identiques d'affilée verra la seconde disparaître sur un mid-stream join. C'est un effet de bord de la dédup,
écrit nulle part.

## G13 — `user_message` n'a pas d'`id` : on ne peut pas le corréler par clé
`UserMessage { content }` est le seul événement de tour sans identifiant (`types.rs:711-712`) ; `refs_resolved`
lui est rattaché « par position » (dernier `user_message` avant lui), pas par clé (`types.rs:713-717`). Toute
parité qui veut relier une `refs_resolved` à son `user_message` doit compter les `user_message` dans l'ordre,
pas chercher un `id`.
