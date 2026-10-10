# Parité des moteurs de chat — matrice MESURÉE

<!-- Généré par src/chat/engine_parity_tests.rs : ne pas éditer à la main.
     UPDATE_PARITY_MATRIX=1 cargo test --lib chat::engine_parity_tests::the_same_scenario -->

Un seul scénario, joué sur les deux moteurs, même assertion par fonction :

- **Claude Code** : moteur historique (chemin de production de Claude Code) sur `fake_claude`, le faux CLI du dépôt nexus ;
- **natif** : moteur agent, instance OpenAI-compatible sur `fake_openai`, outils project-orchestrator sur `fake_mcp`, outils de base sur le vrai `nexus-tools`.

`ok` : mesuré et conforme. `gap (cause, tâche)` : mesuré absent, cause `harnais` (le backend ou le harnais nexus ne le fait pas) ou `modèle` (le modèle ne le peut pas), fermé par la tâche PO nommée (plan 5ad54c48). `not_measured` : les faux binaires ne permettent pas de l'exercer sur ce moteur (raison ci-dessous). Un écart non déclaré, ou un écart déclaré qui disparaît, fait échouer le test.

| Fonction | Ce qui est vérifié | Claude Code | natif |
|---|---|---|---|
| `hooks.before_tool` | le hook PreToolUse du graphe est consulté avant un outil | ok | ok |
| `hooks.after_tool` | le hook PostToolUse du graphe répond après un Grep bruyant (find_references) | ok | ok |
| `hooks.before_compaction` | le hook PreCompact du graphe nomme le projet avant une compaction | ok | ok |
| `compaction` | compaction_started et compact_boundary sur le fil | ok | ok |
| `message_queue` | un message envoyé pendant un tour l'interrompt et passe ensuite | ok | ok |
| `auto_continue` | un tour arrêté sur sa limite est continué (auto_continue) | ok | ok |
| `nats.send` | un message d'une autre instance (NATS) est joué ici | ok | ok |
| `nats.interrupt` | un Stop d'une autre instance (NATS) arrête le tour | ok | ok |
| `nats.permission_response` | une réponse de permission d'une autre instance (NATS) débloque l'outil | gap (harnais, P13) | ok |
| `nats.cancel_tools` | un cancel_tools d'une autre instance (NATS) arrête l'outil, le tour continue | ok | ok |
| `resume` | après redémarrage du backend, la session reprend sur le jeton relu du graphe | ok | ok |
| `set_model` | set_model en cours de conversation atteint le provider | ok | ok |
| `cancel_tools` | cancel_tools arrête l'outil en cours, le tour continue | ok | ok |
| `permissions.once` | une permission accordée une fois débloque l'outil | ok | ok |
| `permissions.session` | une permission accordée pour la session n'est pas redemandée | gap (harnais, P11) | gap (harnais, P11) |
| `permissions.always` | une permission accordée pour toujours est retenue au-delà de la session | gap (harnais, P11) | gap (harnais, P11) |
| `po_tools` | les outils project-orchestrator (MCP) sont donnés et appelables | ok | ok |
| `nexus_tools` | Read / Edit / Bash s'exécutent sur le projet | not_measured | ok |
| `enrichment` | le contexte du graphe précède le message du tour | ok | ok |
| `refs` | une référence #kind:id atteint le modèle en pointeur po-context | ok | ok |
| `provider_switch.relay` | la bascule vers ce moteur relaie la conversation sur le fil (conversation_relayed) | ok | ok |
| `images` | une image jointe atteint le provider en bloc image | ok | ok |
| `session_record` | le dossier de session porte message_count, total_cost_usd et un titre | ok | gap (harnais, P8) |
| `background_tasks` | une tâche d'arrière-plan est suivie (active_tasks_update) | ok | gap (harnais, P4) |
| `cancel_task` | cancel_task arrête une tâche d'arrière-plan | ok | gap (harnais, P12) |
| `system_init.degraded` | system_init n'annonce comme manquant qu'une limite du modèle (liste fermée) | ok | ok |

`ok` : Claude Code 22/26, natif 21/26.

## Écarts déclarés

| Moteur | Fonction | Cause | Tâche | Ce qui manque |
|---|---|---|---|---|
| Claude Code | `nats.permission_response` | harnais | P13 | l'écouteur RPC NATS du moteur historique écrit au CLI une control_response sans request_id ni behavior (et sous le verrou du client) : l'outil reste bloqué alors que la RPC répond success |
| Claude Code | `permissions.session` | harnais | P11 | la réponse de permission ne porte aucune portée : le backend n'écrit au CLI qu'un allow/deny ponctuel (pas d'updatedPermissions de session) |
| Claude Code | `permissions.always` | harnais | P11 | aucune portée persistante (updatedPermissions vers les réglages) n'est écrite au CLI |
| natif | `permissions.session` | harnais | P11 | le harnais natif sait retenir une portée session, mais le backend répond toujours allow_once : la permission est redemandée |
| natif | `permissions.always` | harnais | P11 | le harnais natif ne déclare pas la portée always (permission_scopes = once, session) et le backend ne la transmet pas |
| natif | `session_record` | harnais | P8 | le moteur agent ne met pas à jour le dossier de session à la fin d'un tour (message_count / total_cost_usd) |
| natif | `background_tasks` | harnais | P4 | le natif ne rapporte pas de tâche d'arrière-plan (capacité background_tasks = false) : rien n'est suivi |
| natif | `cancel_task` | harnais | P12 | cancel_task n'a pas de branche moteur agent (no-op idempotent) et le natif n'a pas de tâche à annuler |

## Non mesuré

| Moteur | Fonction | Pourquoi |
|---|---|---|
| Claude Code | `nexus_tools` | Read / Edit / Bash sont les outils internes du CLI Claude : fake_claude rejoue un transcript et n'exécute aucun outil |
