# Parité des moteurs de chat — matrice MESURÉE

<!-- Généré par src/chat/engine_parity_tests.rs : ne pas éditer à la main.
     UPDATE_PARITY_MATRIX=1 cargo test --lib chat::engine_parity_tests::the_same_scenario -->

Un seul scénario, joué sur les moteurs mesurables, même assertion par fonction :

- **Claude Code** : moteur historique (`CHAT_PROVIDER_PATH=legacy`) sur `fake_claude`, le faux CLI du dépôt nexus ;
- **Claude Code (moteur agent)** : le même CLI et le même transcript, piloté par le moteur agent à travers la façade Claude Code de nexus (`CHAT_PROVIDER_PATH=agent`) ;
- **natif** : moteur agent, instance OpenAI-compatible sur `fake_openai`, outils project-orchestrator sur `fake_mcp`, outils de base sur le vrai `nexus-tools` ;
- **Codex** : harnais Codex — `fake_codex` absent du dépôt nexus, toutes les fonctions `not_measured` (YARDSTICK, tâche 87eb345e) ;
- **opencode** : harnais opencode — `fake_opencode` absent, toutes les fonctions `not_measured` (YARDSTICK, tâche 87eb345e) ;
- **ACP** : harnais ACP — `fake_acp` présent mais le scénario complet n'a pas encore été adapté, toutes les fonctions `not_measured` (YARDSTICK, tâche 87eb345e).

`ok` : mesuré et conforme. `gap (cause, tâche)` : mesuré absent, cause `harnais` (le backend ou le harnais nexus ne le fait pas) ou `modèle` (le modèle ne le peut pas), fermé par la tâche PO nommée (plan 5ad54c48 « Parité totale », ou e87abcc2 « Claude Code devient optionnel » pour une tâche M…). `not_measured` : les faux binaires ne permettent pas de l'exercer sur ce moteur (raison ci-dessous). Un écart non déclaré, ou un écart déclaré qui disparaît, fait échouer le test.

| Fonction | Ce qui est vérifié | Claude Code | Claude Code (moteur agent) | natif | Codex | opencode | ACP |
|---|---|---|---|---|---|---|---|
| `hooks.before_tool` | le hook PreToolUse du graphe est consulté avant un outil | ok | ok | ok | not_measured | not_measured | not_measured |
| `hooks.after_tool` | le hook PostToolUse du graphe répond après un Grep bruyant (find_references) | ok | ok | ok | not_measured | not_measured | not_measured |
| `hooks.before_compaction` | le hook PreCompact du graphe nomme le projet dans les instructions du résumé (Claude Code : `systemMessage`, le champ que lit le CLI) | ok | ok | ok | not_measured | not_measured | not_measured |
| `compaction` | compaction_started et compact_boundary sur le fil | ok | ok | ok | not_measured | not_measured | not_measured |
| `message_queue` | un message envoyé pendant un tour l'interrompt et passe ensuite | ok | ok | ok | not_measured | not_measured | not_measured |
| `auto_continue` | un tour arrêté sur sa limite est continué (auto_continue) | ok | ok | ok | not_measured | not_measured | not_measured |
| `nats.send` | un message d'une autre instance (NATS) est joué ici | ok | ok | ok | not_measured | not_measured | not_measured |
| `nats.interrupt` | un Stop d'une autre instance (NATS) arrête le tour | ok | ok | ok | not_measured | not_measured | not_measured |
| `nats.permission_response` | une réponse de permission d'une autre instance (NATS) débloque l'outil | ok | ok | ok | not_measured | not_measured | not_measured |
| `nats.cancel_tools` | un cancel_tools d'une autre instance (NATS) arrête l'outil, le tour continue | ok | ok | ok | not_measured | not_measured | not_measured |
| `resume` | après redémarrage du backend, la session reprend sur le jeton relu du graphe | ok | ok | ok | not_measured | not_measured | not_measured |
| `set_model` | set_model en cours de conversation atteint le provider | ok | ok | ok | not_measured | not_measured | not_measured |
| `cancel_tools` | cancel_tools arrête l'outil en cours, le tour continue | ok | ok | ok | not_measured | not_measured | not_measured |
| `permissions.once` | une permission accordée une fois débloque l'outil | ok | ok | ok | not_measured | not_measured | not_measured |
| `permissions.session` | une permission accordée pour la session n'est pas redemandée | ok | ok | ok | not_measured | not_measured | not_measured |
| `permissions.always` | une permission accordée pour toujours est retenue au-delà de la session | ok | ok | ok | not_measured | not_measured | not_measured |
| `tool_timing` | un outil qui a attendu une permission porte tool_timing persisté : prise en charge, attente, exécution, fin | ok | ok | ok | not_measured | not_measured | not_measured |
| `po_tools` | les outils project-orchestrator (MCP) sont donnés et appelables | ok | ok | ok | not_measured | not_measured | not_measured |
| `nexus_tools` | Read / Edit / Bash s'exécutent sur le projet | not_measured | not_measured | ok | not_measured | not_measured | not_measured |
| `enrichment` | le contexte du graphe précède le message du tour | ok | ok | ok | not_measured | not_measured | not_measured |
| `refs` | une référence #kind:id atteint le modèle en pointeur po-context | ok | ok | ok | not_measured | not_measured | not_measured |
| `provider_switch.relay` | la bascule vers ce moteur relaie la conversation sur le même fil (provider_switched) | ok | ok | ok | not_measured | not_measured | not_measured |
| `images` | une image jointe atteint le provider en bloc image | ok | ok | ok | not_measured | not_measured | not_measured |
| `session_record` | le dossier de session porte message_count, total_cost_usd et un titre | ok | ok | ok | not_measured | not_measured | not_measured |
| `background_tasks` | une tâche d'arrière-plan est suivie (active_tasks_update) | ok | ok | ok | not_measured | not_measured | not_measured |
| `cancel_task` | cancel_task arrête une tâche d'arrière-plan | ok | ok | ok | not_measured | not_measured | not_measured |
| `ask_user_question` | une question du modèle s'affiche (ask_user_question) et la réponse de l'utilisateur lui revient au tour suivant | ok | ok | ok | not_measured | not_measured | not_measured |
| `subagents` | un sous-agent (Agent / Task) travaille et sa réponse revient en résultat ; sur le natif, une session enfant qui ne peut pas re-déléguer | ok | ok | ok | not_measured | not_measured | not_measured |
| `system_init.degraded` | system_init n'annonce comme manquant qu'une limite du modèle (liste fermée) | ok | ok | ok | not_measured | not_measured | not_measured |
| `session_limits` | sans réglage, aucune limite de durée par tour ni budget de session (ISO5) : le CLI ne reçoit pas --max-budget-usd, une session du moteur agent s'ouvre et reprend sans turn_timeout_ms ni max_tokens | ok | ok | ok | not_measured | not_measured | not_measured |

`ok` : Claude Code 29/30, Claude Code (moteur agent) 29/30, natif 30/30, Codex 0/30, opencode 0/30, ACP 0/30.

## Écarts déclarés

| Moteur | Fonction | Cause | Tâche | Ce qui manque |
|---|---|---|---|---|

## Non mesuré

| Moteur | Fonction | Pourquoi |
|---|---|---|
| Claude Code | `nexus_tools` | Read / Edit / Bash sont les outils internes du CLI Claude : fake_claude rejoue un transcript et n'exécute aucun outil |
| Claude Code (moteur agent) | `nexus_tools` | Read / Edit / Bash sont les outils internes du CLI Claude : fake_claude rejoue un transcript et n'exécute aucun outil |
| Codex | `hooks.before_tool` | fake_codex n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| Codex | `hooks.after_tool` | fake_codex n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| Codex | `hooks.before_compaction` | fake_codex n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| Codex | `compaction` | fake_codex n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| Codex | `message_queue` | fake_codex n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| Codex | `auto_continue` | fake_codex n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| Codex | `nats.send` | fake_codex n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| Codex | `nats.interrupt` | fake_codex n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| Codex | `nats.permission_response` | fake_codex n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| Codex | `nats.cancel_tools` | fake_codex n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| Codex | `resume` | fake_codex n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| Codex | `set_model` | fake_codex n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| Codex | `cancel_tools` | fake_codex n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| Codex | `permissions.once` | fake_codex n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| Codex | `permissions.session` | fake_codex n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| Codex | `permissions.always` | fake_codex n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| Codex | `tool_timing` | fake_codex n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| Codex | `po_tools` | fake_codex n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| Codex | `nexus_tools` | fake_codex n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| Codex | `enrichment` | fake_codex n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| Codex | `refs` | fake_codex n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| Codex | `provider_switch.relay` | fake_codex n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| Codex | `images` | fake_codex n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| Codex | `session_record` | fake_codex n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| Codex | `background_tasks` | fake_codex n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| Codex | `cancel_task` | fake_codex n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| Codex | `ask_user_question` | fake_codex n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| Codex | `subagents` | fake_codex n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| Codex | `system_init.degraded` | fake_codex n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| Codex | `session_limits` | fake_codex n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `hooks.before_tool` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `hooks.after_tool` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `hooks.before_compaction` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `compaction` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `message_queue` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `auto_continue` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `nats.send` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `nats.interrupt` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `nats.permission_response` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `nats.cancel_tools` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `resume` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `set_model` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `cancel_tools` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `permissions.once` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `permissions.session` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `permissions.always` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `tool_timing` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `po_tools` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `nexus_tools` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `enrichment` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `refs` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `provider_switch.relay` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `images` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `session_record` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `background_tasks` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `cancel_task` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `ask_user_question` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `subagents` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `system_init.degraded` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `session_limits` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| ACP | `hooks.before_tool` | fake_acp existe mais le scénario complet n'a pas encore été adapté à l'harnais ACP ; chaque fonction sera mesurée au fur et à mesure (YARDSTICK, tâche 87eb345e) |
| ACP | `hooks.after_tool` | fake_acp existe mais le scénario complet n'a pas encore été adapté à l'harnais ACP ; chaque fonction sera mesurée au fur et à mesure (YARDSTICK, tâche 87eb345e) |
| ACP | `hooks.before_compaction` | fake_acp existe mais le scénario complet n'a pas encore été adapté à l'harnais ACP ; chaque fonction sera mesurée au fur et à mesure (YARDSTICK, tâche 87eb345e) |
| ACP | `compaction` | fake_acp existe mais le scénario complet n'a pas encore été adapté à l'harnais ACP ; chaque fonction sera mesurée au fur et à mesure (YARDSTICK, tâche 87eb345e) |
| ACP | `message_queue` | fake_acp existe mais le scénario complet n'a pas encore été adapté à l'harnais ACP ; chaque fonction sera mesurée au fur et à mesure (YARDSTICK, tâche 87eb345e) |
| ACP | `auto_continue` | fake_acp existe mais le scénario complet n'a pas encore été adapté à l'harnais ACP ; chaque fonction sera mesurée au fur et à mesure (YARDSTICK, tâche 87eb345e) |
| ACP | `nats.send` | fake_acp existe mais le scénario complet n'a pas encore été adapté à l'harnais ACP ; chaque fonction sera mesurée au fur et à mesure (YARDSTICK, tâche 87eb345e) |
| ACP | `nats.interrupt` | fake_acp existe mais le scénario complet n'a pas encore été adapté à l'harnais ACP ; chaque fonction sera mesurée au fur et à mesure (YARDSTICK, tâche 87eb345e) |
| ACP | `nats.permission_response` | fake_acp existe mais le scénario complet n'a pas encore été adapté à l'harnais ACP ; chaque fonction sera mesurée au fur et à mesure (YARDSTICK, tâche 87eb345e) |
| ACP | `nats.cancel_tools` | fake_acp existe mais le scénario complet n'a pas encore été adapté à l'harnais ACP ; chaque fonction sera mesurée au fur et à mesure (YARDSTICK, tâche 87eb345e) |
| ACP | `resume` | fake_acp existe mais le scénario complet n'a pas encore été adapté à l'harnais ACP ; chaque fonction sera mesurée au fur et à mesure (YARDSTICK, tâche 87eb345e) |
| ACP | `set_model` | fake_acp existe mais le scénario complet n'a pas encore été adapté à l'harnais ACP ; chaque fonction sera mesurée au fur et à mesure (YARDSTICK, tâche 87eb345e) |
| ACP | `cancel_tools` | fake_acp existe mais le scénario complet n'a pas encore été adapté à l'harnais ACP ; chaque fonction sera mesurée au fur et à mesure (YARDSTICK, tâche 87eb345e) |
| ACP | `permissions.once` | fake_acp existe mais le scénario complet n'a pas encore été adapté à l'harnais ACP ; chaque fonction sera mesurée au fur et à mesure (YARDSTICK, tâche 87eb345e) |
| ACP | `permissions.session` | fake_acp existe mais le scénario complet n'a pas encore été adapté à l'harnais ACP ; chaque fonction sera mesurée au fur et à mesure (YARDSTICK, tâche 87eb345e) |
| ACP | `permissions.always` | fake_acp existe mais le scénario complet n'a pas encore été adapté à l'harnais ACP ; chaque fonction sera mesurée au fur et à mesure (YARDSTICK, tâche 87eb345e) |
| ACP | `tool_timing` | fake_acp existe mais le scénario complet n'a pas encore été adapté à l'harnais ACP ; chaque fonction sera mesurée au fur et à mesure (YARDSTICK, tâche 87eb345e) |
| ACP | `po_tools` | fake_acp existe mais le scénario complet n'a pas encore été adapté à l'harnais ACP ; chaque fonction sera mesurée au fur et à mesure (YARDSTICK, tâche 87eb345e) |
| ACP | `nexus_tools` | fake_acp existe mais le scénario complet n'a pas encore été adapté à l'harnais ACP ; chaque fonction sera mesurée au fur et à mesure (YARDSTICK, tâche 87eb345e) |
| ACP | `enrichment` | fake_acp existe mais le scénario complet n'a pas encore été adapté à l'harnais ACP ; chaque fonction sera mesurée au fur et à mesure (YARDSTICK, tâche 87eb345e) |
| ACP | `refs` | fake_acp existe mais le scénario complet n'a pas encore été adapté à l'harnais ACP ; chaque fonction sera mesurée au fur et à mesure (YARDSTICK, tâche 87eb345e) |
| ACP | `provider_switch.relay` | fake_acp existe mais le scénario complet n'a pas encore été adapté à l'harnais ACP ; chaque fonction sera mesurée au fur et à mesure (YARDSTICK, tâche 87eb345e) |
| ACP | `images` | fake_acp existe mais le scénario complet n'a pas encore été adapté à l'harnais ACP ; chaque fonction sera mesurée au fur et à mesure (YARDSTICK, tâche 87eb345e) |
| ACP | `session_record` | fake_acp existe mais le scénario complet n'a pas encore été adapté à l'harnais ACP ; chaque fonction sera mesurée au fur et à mesure (YARDSTICK, tâche 87eb345e) |
| ACP | `background_tasks` | fake_acp existe mais le scénario complet n'a pas encore été adapté à l'harnais ACP ; chaque fonction sera mesurée au fur et à mesure (YARDSTICK, tâche 87eb345e) |
| ACP | `cancel_task` | fake_acp existe mais le scénario complet n'a pas encore été adapté à l'harnais ACP ; chaque fonction sera mesurée au fur et à mesure (YARDSTICK, tâche 87eb345e) |
| ACP | `ask_user_question` | fake_acp existe mais le scénario complet n'a pas encore été adapté à l'harnais ACP ; chaque fonction sera mesurée au fur et à mesure (YARDSTICK, tâche 87eb345e) |
| ACP | `subagents` | fake_acp existe mais le scénario complet n'a pas encore été adapté à l'harnais ACP ; chaque fonction sera mesurée au fur et à mesure (YARDSTICK, tâche 87eb345e) |
| ACP | `system_init.degraded` | fake_acp existe mais le scénario complet n'a pas encore été adapté à l'harnais ACP ; chaque fonction sera mesurée au fur et à mesure (YARDSTICK, tâche 87eb345e) |
| ACP | `session_limits` | fake_acp existe mais le scénario complet n'a pas encore été adapté à l'harnais ACP ; chaque fonction sera mesurée au fur et à mesure (YARDSTICK, tâche 87eb345e) |

## À re-mesurer à chaque montée du CLI Claude Code

`fake_claude` rejoue ce que le vrai CLI a fait à une version donnée : rien ici ne verrait un CLI qui changerait d'avis sans le dire. À chaque montée de version du CLI Claude Code, re-mesurer sur le vrai CLI :

| Comportement | Mesuré sur | Comment | Note PO |
|---|---|---|---|
| re-check PreCompact systemMessage : le CLI ajoute le `systemMessage` d'une réponse au hook PreCompact aux instructions du résumé (et l'affiche à l'utilisateur dans la ligne « Compacted… »), ignore `reason`, rejette `hookSpecificOutput.additionalContext` | Claude Code 2.1.295 | `docs/parity/m2c-precompact/` (README) | 172c164e (tâche M2c 79cc48df) |
