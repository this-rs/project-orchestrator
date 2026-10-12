# Parité des moteurs de chat — matrice MESURÉE

<!-- Généré par src/chat/engine_parity_tests.rs : ne pas éditer à la main.
     UPDATE_PARITY_MATRIX=1 cargo test --lib chat::engine_parity_tests::the_same_scenario -->

Un seul scénario, joué sur les deux moteurs, même assertion par fonction :

- **Claude Code** : moteur historique (`CHAT_PROVIDER_PATH=legacy`) sur `fake_claude`, le faux CLI du dépôt nexus ;
- **Claude Code (moteur agent)** : le même CLI et le même transcript, piloté par le moteur agent à travers la façade Claude Code de nexus (`CHAT_PROVIDER_PATH=agent`) ;
- **natif** : moteur agent, instance OpenAI-compatible sur `fake_openai`, outils project-orchestrator sur `fake_mcp`, outils de base sur le vrai `nexus-tools`.

`ok` : mesuré et conforme. `gap (cause, tâche)` : mesuré absent, cause `harnais` (le backend ou le harnais nexus ne le fait pas) ou `modèle` (le modèle ne le peut pas), fermé par la tâche PO nommée (plan 5ad54c48 « Parité totale », ou e87abcc2 « Claude Code devient optionnel » pour une tâche M…). `not_measured` : les faux binaires ne permettent pas de l'exercer sur ce moteur (raison ci-dessous). Un écart non déclaré, ou un écart déclaré qui disparaît, fait échouer le test.

| Fonction | Ce qui est vérifié | Claude Code | Claude Code (moteur agent) | natif |
|---|---|---|---|---|
| `hooks.before_tool` | le hook PreToolUse du graphe est consulté avant un outil | ok | ok | ok |
| `hooks.after_tool` | le hook PostToolUse du graphe répond après un Grep bruyant (find_references) | ok | ok | ok |
| `hooks.before_compaction` | le hook PreCompact du graphe nomme le projet dans les instructions du résumé (Claude Code : `systemMessage`, le champ que lit le CLI) | ok | ok | ok |
| `compaction` | compaction_started et compact_boundary sur le fil | ok | ok | ok |
| `message_queue` | un message envoyé pendant un tour l'interrompt et passe ensuite | ok | ok | ok |
| `auto_continue` | un tour arrêté sur sa limite est continué (auto_continue) | ok | ok | ok |
| `nats.send` | un message d'une autre instance (NATS) est joué ici | ok | ok | ok |
| `nats.interrupt` | un Stop d'une autre instance (NATS) arrête le tour | ok | ok | ok |
| `nats.permission_response` | une réponse de permission d'une autre instance (NATS) débloque l'outil | ok | ok | ok |
| `nats.cancel_tools` | un cancel_tools d'une autre instance (NATS) arrête l'outil, le tour continue | ok | ok | ok |
| `resume` | après redémarrage du backend, la session reprend sur le jeton relu du graphe | ok | ok | ok |
| `set_model` | set_model en cours de conversation atteint le provider | ok | ok | ok |
| `cancel_tools` | cancel_tools arrête l'outil en cours, le tour continue | ok | ok | ok |
| `permissions.once` | une permission accordée une fois débloque l'outil | ok | ok | ok |
| `permissions.session` | une permission accordée pour la session n'est pas redemandée | ok | ok | ok |
| `permissions.always` | une permission accordée pour toujours est retenue au-delà de la session | ok | ok | ok |
| `tool_timing` | un outil qui a attendu une permission porte tool_timing persisté : prise en charge, attente, exécution, fin | ok | ok | ok |
| `po_tools` | les outils project-orchestrator (MCP) sont donnés et appelables | ok | ok | ok |
| `nexus_tools` | Read / Edit / Bash s'exécutent sur le projet | not_measured | not_measured | ok |
| `enrichment` | le contexte du graphe précède le message du tour | ok | ok | ok |
| `refs` | une référence #kind:id atteint le modèle en pointeur po-context | ok | ok | ok |
| `provider_switch.relay` | la bascule vers ce moteur relaie la conversation sur le fil (conversation_relayed) | ok | ok | ok |
| `images` | une image jointe atteint le provider en bloc image | ok | ok | ok |
| `session_record` | le dossier de session porte message_count, total_cost_usd et un titre | ok | ok | ok |
| `background_tasks` | une tâche d'arrière-plan est suivie (active_tasks_update) | ok | ok | ok |
| `cancel_task` | cancel_task arrête une tâche d'arrière-plan | ok | ok | ok |
| `ask_user_question` | une question du modèle s'affiche (ask_user_question) et la réponse de l'utilisateur lui revient au tour suivant | ok | ok | ok |
| `subagents` | un sous-agent (Agent / Task) travaille et sa réponse revient en résultat ; sur le natif, une session enfant qui ne peut pas re-déléguer | ok | ok | ok |
| `system_init.degraded` | system_init n'annonce comme manquant qu'une limite du modèle (liste fermée) | ok | ok | ok |

`ok` : Claude Code 28/29, Claude Code (moteur agent) 28/29, natif 29/29.

## Écarts déclarés

| Moteur | Fonction | Cause | Tâche | Ce qui manque |
|---|---|---|---|---|

## Non mesuré

| Moteur | Fonction | Pourquoi |
|---|---|---|
| Claude Code | `nexus_tools` | Read / Edit / Bash sont les outils internes du CLI Claude : fake_claude rejoue un transcript et n'exécute aucun outil |
| Claude Code (moteur agent) | `nexus_tools` | Read / Edit / Bash sont les outils internes du CLI Claude : fake_claude rejoue un transcript et n'exécute aucun outil |

## À re-mesurer à chaque montée du CLI Claude Code

`fake_claude` rejoue ce que le vrai CLI a fait à une version donnée : rien ici ne verrait un CLI qui changerait d'avis sans le dire. À chaque montée de version du CLI Claude Code, re-mesurer sur le vrai CLI :

| Comportement | Mesuré sur | Comment | Note PO |
|---|---|---|---|
| re-check PreCompact systemMessage : le CLI ajoute le `systemMessage` d'une réponse au hook PreCompact aux instructions du résumé (et l'affiche à l'utilisateur dans la ligne « Compacted… »), ignore `reason`, rejette `hookSpecificOutput.additionalContext` | Claude Code 2.1.295 | `docs/parity/m2c-precompact/` (README) | 172c164e (tâche M2c 79cc48df) |
