# Parité des moteurs de chat — matrice MESURÉE

<!-- Généré par src/chat/engine_parity_tests.rs : ne pas éditer à la main.
     UPDATE_PARITY_MATRIX=1 cargo test --lib chat::engine_parity_tests::the_same_scenario -->

Un seul scénario, joué sur les moteurs mesurables, même assertion par fonction :

- **Claude Code** : moteur historique (`CHAT_PROVIDER_PATH=legacy`) sur `fake_claude`, le faux CLI du dépôt nexus ;
- **Claude Code (moteur agent)** : le même CLI et le même transcript, piloté par le moteur agent à travers la façade Claude Code de nexus (`CHAT_PROVIDER_PATH=agent`) ;
- **natif** : moteur agent, instance OpenAI-compatible sur `fake_openai`, outils project-orchestrator sur `fake_mcp`, outils de base sur le vrai `nexus-tools` ;
- **Codex** : harnais Codex — `fake_codex` présent dans `nexus/src/bin/fake_codex.rs`, replay de transcripts JSONL ; verdicts par fonction déclarés dans `EXPECTED` (ISO4) ;
- **opencode** : harnais opencode — `fake_opencode` absent, toutes les fonctions `not_measured` (YARDSTICK, tâche 87eb345e) ;
- **ACP** : harnais ACP — `fake_acp` présent dans `nexus/src/bin/fake_acp.rs`, replay de transcripts JSONL ; verdicts par fonction déclarés dans `EXPECTED` (ISO4).

`ok` : mesuré et conforme. `gap (cause, tâche)` : mesuré absent, cause `harnais` (le backend ou le harnais nexus ne le fait pas) ou `modèle` (le modèle ne le peut pas), fermé par la tâche PO nommée (plan 5ad54c48 « Parité totale », ou e87abcc2 « Claude Code devient optionnel » pour une tâche M…). `declared (cause, tâche)` : écart déclaré dans EXPECTED mais le scénario n'a PAS tourné sur ce moteur (pas de transcript dédié) ; distinct de `gap` (mesuré) et de `not_measured` (impossible à exercer). `not_measured` : les faux binaires ne permettent pas de l'exercer sur ce moteur (raison ci-dessous). Un écart non déclaré, ou un écart déclaré qui disparaît, fait échouer le test.

| Fonction | Ce qui est vérifié | Claude Code | Claude Code (moteur agent) | natif | Codex | opencode | ACP |
|---|---|---|---|---|---|---|---|
| `hooks.before_tool` | le hook PreToolUse du graphe est consulté avant un outil | ok | ok | ok | declared (harnais, ISO4) | not_measured | declared (harnais, ISO4) |
| `hooks.after_tool` | le hook PostToolUse du graphe répond après un Grep bruyant (find_references) | ok | ok | ok | declared (harnais, ISO4) | not_measured | declared (harnais, ISO4) |
| `hooks.before_compaction` | le hook PreCompact du graphe nomme le projet dans les instructions du résumé (Claude Code : `systemMessage`, le champ que lit le CLI) | ok | ok | ok | declared (harnais, ISO4) | not_measured | declared (harnais, ISO4) |
| `hooks.session_start` | le hook SessionStart du graphe est consulté à l'ouverture de la session (contract v7) | ok | ok | ok | declared (harnais, ISO4) | not_measured | declared (harnais, ISO4) |
| `hooks.user_prompt_submit` | le hook UserPromptSubmit du graphe est consulté avant chaque tour (contract v7) | ok | ok | ok | declared (harnais, ISO4) | not_measured | declared (harnais, ISO4) |
| `hooks.stop` | le hook Stop du graphe est consulté avant la fin du tour ; un block fait continuer le tour avec la raison (contract v7) | ok | ok | ok | declared (harnais, ISO4) | not_measured | declared (harnais, ISO4) |
| `hooks.subagent_stop` | le hook SubagentStop du graphe est consulté quand un sous-agent s'arrête (contract v7) | ok | ok | ok | declared (harnais, ISO4) | not_measured | declared (harnais, ISO4) |
| `hooks.notification` | le hook Notification du graphe est appelé quand une permission est attendue (contract v7) | ok | ok | ok | declared (harnais, ISO4) | not_measured | declared (harnais, ISO4) |
| `compaction` | compaction_started et compact_boundary sur le fil | ok | ok | ok | not_measured | not_measured | not_measured |
| `compaction.manual` | `/compact <instructions>` envoyé depuis le chat : Claude Code reçoit la commande telle quelle (le CLI compacte, compact_boundary trigger manual) ; le natif compacte hors tour (AgentSession::compact), les instructions et le hook PreCompact dans la requête de résumé, aucun tour envoyé au modèle, compaction_started / compact_boundary trigger manual sur le fil (ISO3) | ok | ok | ok | not_measured | not_measured | declared (harnais, ISO4) |
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
| `tool_search` | un outil différé n'est pas envoyé ; ToolSearch le charge, il est ajouté EN FIN de liste et appelable dans le même tour (CTX2) | not_measured | not_measured | ok | not_measured | not_measured | not_measured |
| `nexus_tools` | Read / Edit / Bash s'exécutent sur le projet | not_measured | not_measured | ok | not_measured | not_measured | declared (harnais, ISO4) |
| `tool_result.huge` | un résultat d'outil MCP d'environ 1 Mo ne tue pas le tour : seul un avis borné (moins de 8 Ko) atteint le fil et le modèle, le tour se clôt et le suivant démarre ; natif : le mégaoctet est gardé sur disque, au chemin que l'avis nomme (Claude Code : ce que fait le CLI est relevé, puis rejoué par fake_claude) | ok | ok | ok | not_measured | not_measured | not_measured |
| `enrichment` | le contexte du graphe précède le message du tour | ok | ok | ok | not_measured | not_measured | not_measured |
| `refs` | une référence #kind:id atteint le modèle en pointeur po-context | ok | ok | ok | not_measured | not_measured | not_measured |
| `provider_switch.relay` | la bascule vers ce moteur relaie la conversation sur le même fil (provider_switched) | ok | ok | ok | ok | not_measured | ok |
| `images` | une image jointe atteint le provider en bloc image | ok | ok | ok | declared (harnais, ISO4) | not_measured | not_measured |
| `session_record` | le dossier de session porte message_count, total_cost_usd et un titre | ok | ok | ok | not_measured | not_measured | not_measured |
| `background_tasks` | une tâche d'arrière-plan est suivie (active_tasks_update) | ok | ok | ok | declared (harnais, ISO4) | not_measured | not_measured |
| `cancel_task` | cancel_task arrête une tâche d'arrière-plan | ok | ok | ok | declared (harnais, ISO4) | not_measured | not_measured |
| `ask_user_question` | une question du modèle s'affiche (ask_user_question) et la réponse de l'utilisateur lui revient au tour suivant | ok | ok | ok | declared (harnais, ISO4) | not_measured | not_measured |
| `subagents` | un sous-agent (Agent / Task) travaille et sa réponse revient en résultat ; sur le natif, une session enfant qui ne peut pas re-déléguer | ok | ok | ok | not_measured | not_measured | not_measured |
| `subagents.background` | un sous-agent lancé en arrière-plan (run_in_background: true) rend la main au parent immédiatement ; sa fin est injectée comme task_notification au tour suivant | not_measured | not_measured | not_measured | not_measured | not_measured | not_measured |
| `subagents.types` | subagent_type valide parmi les types déclarés ; un type inconnu est refusé avec la liste des types connus | not_measured | not_measured | ok | not_measured | not_measured | not_measured |
| `system_init.degraded` | system_init n'annonce comme manquant qu'une limite du modèle (liste fermée) | ok | ok | ok | not_measured | not_measured | not_measured |
| `session_limits` | sans réglage, aucune limite de durée par tour ni budget de session (ISO5) : le CLI ne reçoit pas --max-budget-usd, une session du moteur agent s'ouvre et reprend sans turn_timeout_ms ni max_tokens | ok | ok | ok | not_measured | not_measured | not_measured |
| `anthropic_messages` | une instance `anthropic` (clé au coffre) : le harnais natif parle lui-même l'API Messages, et la requête reçue par le modèle porte les trois points cache_control — dernier outil, système, dernier tour (CTX5) | not_measured | not_measured | ok | not_measured | not_measured | not_measured |

`ok` : Claude Code 36/41, Claude Code (moteur agent) 36/41, natif 40/41, Codex 1/41, opencode 0/41, ACP 1/41.

## Bilan d'attestation

Un **vert attesté** est une cellule que ce scénario a jouée sur ce moteur et a vue conforme (`ok`). Un **écart déclaré** est un écart connu, inscrit dans `EXPECTED`, que le scénario n'a PAS joué sur ce moteur : il n'est pas un vert et n'est jamais compté comme tel. La ligne de totaux ci-dessus ne publie que des attestés. `total` est la somme des cinq colonnes et vaut le nombre de fonctions de la matrice.

| Moteur | attestés | écarts déclarés | écarts mesurés | non mesurés | manquants | total |
|---|---|---|---|---|---|---|
| Claude Code | 36 | 0 | 0 | 5 | 0 | 41 |
| Claude Code (moteur agent) | 36 | 0 | 0 | 5 | 0 | 41 |
| natif | 40 | 0 | 0 | 1 | 0 | 41 |
| Codex | 1 | 12 | 0 | 28 | 0 | 41 |
| opencode | 0 | 0 | 0 | 41 | 0 | 41 |
| ACP | 1 | 10 | 0 | 30 | 0 | 41 |

## Écarts déclarés

| Moteur | Fonction | Cause | Tâche | Ce qui manque |
|---|---|---|---|---|
| Codex | `hooks.before_tool` | harnais | ISO4 | Codex v1 n'a pas d'exécutable relais pour les hooks (A40) ; `SessionSpec::hooks` est ignoré avec `provider_notice { hooks_not_supported }` |
| Codex | `hooks.after_tool` | harnais | ISO4 | Codex v1 n'a pas d'exécutable relais pour les hooks (A40) ; `SessionSpec::hooks` est ignoré avec `provider_notice { hooks_not_supported }` |
| Codex | `hooks.before_compaction` | harnais | ISO4 | Codex v1 n'a pas d'exécutable relais pour les hooks (A40) ; `SessionSpec::hooks` est ignoré avec `provider_notice { hooks_not_supported }` |
| Codex | `hooks.session_start` | harnais | ISO4 | Codex v1 n'a pas d'exécutable relais pour les hooks (A40) ; `SessionSpec::hooks` est ignoré avec `provider_notice { hooks_not_supported }` |
| Codex | `hooks.user_prompt_submit` | harnais | ISO4 | Codex v1 n'a pas d'exécutable relais pour les hooks (A40) ; `SessionSpec::hooks` est ignoré avec `provider_notice { hooks_not_supported }` |
| Codex | `hooks.stop` | harnais | ISO4 | Codex v1 n'a pas d'exécutable relais pour les hooks (A40) ; `SessionSpec::hooks` est ignoré avec `provider_notice { hooks_not_supported }` |
| Codex | `hooks.subagent_stop` | harnais | ISO4 | Codex v1 n'a pas d'exécutable relais pour les hooks (A40) ; `SessionSpec::hooks` est ignoré avec `provider_notice { hooks_not_supported }` |
| Codex | `hooks.notification` | harnais | ISO4 | Codex v1 n'a pas d'exécutable relais pour les hooks (A40) ; `SessionSpec::hooks` est ignoré avec `provider_notice { hooks_not_supported }` |
| Codex | `images` | harnais | ISO4 | Codex v1 n'accepte pas les blocs image (A12) |
| Codex | `background_tasks` | harnais | ISO4 | `tool_cancel` et `background_tasks` ne sont pas supportés par Codex ; `interrupt` est le seul mécanisme de coupe-circuit |
| Codex | `cancel_task` | harnais | ISO4 | `tool_cancel` et `background_tasks` ne sont pas supportés par Codex ; `interrupt` est le seul mécanisme de coupe-circuit |
| Codex | `ask_user_question` | harnais | ISO4 | requestUserInput est expérimental en v1 ; le backend le synthétise mais ne peut pas l'interroger via Codex |
| ACP | `hooks.before_tool` | harnais | ISO4 | L'harnais ACP envoie `provider_notice { hooks_not_supported }` à l'ouverture ; `SessionSpec::hooks` est ignoré |
| ACP | `hooks.after_tool` | harnais | ISO4 | L'harnais ACP envoie `provider_notice { hooks_not_supported }` à l'ouverture ; `SessionSpec::hooks` est ignoré |
| ACP | `hooks.before_compaction` | harnais | ISO4 | L'harnais ACP envoie `provider_notice { hooks_not_supported }` à l'ouverture ; `SessionSpec::hooks` est ignoré |
| ACP | `hooks.session_start` | harnais | ISO4 | L'harnais ACP envoie `provider_notice { hooks_not_supported }` à l'ouverture ; `SessionSpec::hooks` est ignoré |
| ACP | `hooks.user_prompt_submit` | harnais | ISO4 | L'harnais ACP envoie `provider_notice { hooks_not_supported }` à l'ouverture ; `SessionSpec::hooks` est ignoré |
| ACP | `hooks.stop` | harnais | ISO4 | L'harnais ACP envoie `provider_notice { hooks_not_supported }` à l'ouverture ; `SessionSpec::hooks` est ignoré |
| ACP | `hooks.subagent_stop` | harnais | ISO4 | L'harnais ACP envoie `provider_notice { hooks_not_supported }` à l'ouverture ; `SessionSpec::hooks` est ignoré |
| ACP | `hooks.notification` | harnais | ISO4 | L'harnais ACP envoie `provider_notice { hooks_not_supported }` à l'ouverture ; `SessionSpec::hooks` est ignoré |
| ACP | `compaction.manual` | harnais | ISO4 | /compact n'est pas transmis à un agent ACP : la compaction manuelle n'a pas de canal côté harnais |
| ACP | `nexus_tools` | harnais | ISO4 | ACP utilise son propre environnement d'outils ; nexus-tools n'est pas joint à une session ACP |

## Non mesuré

| Moteur | Fonction | Pourquoi |
|---|---|---|
| Claude Code | `tool_search` | les outils différés et ToolSearch sont internes au CLI Claude : la liste d'outils qu'il envoie au modèle n'est pas visible du harnais, et fake_claude rejoue un transcript |
| Claude Code | `nexus_tools` | Read / Edit / Bash sont les outils internes du CLI Claude : fake_claude rejoue un transcript et n'exécute aucun outil |
| Claude Code | `subagents.background` | fake_claude rejoue un transcript sans Agent(run_in_background) ; le miroir P18c le mesure séparément |
| Claude Code | `subagents.types` | la validation des types est propre au natif (ISO2) ; sur Claude Code le CLI gère ses propres types d'agents |
| Claude Code | `anthropic_messages` | le CLI Claude parle lui-même l'API Messages et pose ses propres points de cache : fake_claude rejoue un transcript, la requête au modèle n'est pas visible du harnais |
| Claude Code (moteur agent) | `tool_search` | les outils différés et ToolSearch sont internes au CLI Claude : la liste d'outils qu'il envoie au modèle n'est pas visible du harnais, et fake_claude rejoue un transcript |
| Claude Code (moteur agent) | `nexus_tools` | Read / Edit / Bash sont les outils internes du CLI Claude : fake_claude rejoue un transcript et n'exécute aucun outil |
| Claude Code (moteur agent) | `subagents.background` | fake_claude rejoue un transcript sans Agent(run_in_background) ; le miroir P18c le mesure séparément |
| Claude Code (moteur agent) | `subagents.types` | la validation des types est propre au natif (ISO2) ; sur Claude Code le CLI gère ses propres types d'agents |
| Claude Code (moteur agent) | `anthropic_messages` | le CLI Claude parle lui-même l'API Messages et pose ses propres points de cache : fake_claude rejoue un transcript, la requête au modèle n'est pas visible du harnais |
| natif | `subagents.background` | le scénario turn-based ne peut pas exercer inject_hint sans entrée script ; chemin testé par les tests unitaires ISO2 |
| Codex | `compaction` | fake_codex (nexus src/bin/fake_codex.rs) rejoue des transcripts JSONL déterministes ; cette fonction exige un transcript dédié non encore écrit (ISO4) |
| Codex | `compaction.manual` | fake_codex (nexus src/bin/fake_codex.rs) rejoue des transcripts JSONL déterministes ; cette fonction exige un transcript dédié non encore écrit (ISO4) |
| Codex | `message_queue` | fake_codex (nexus src/bin/fake_codex.rs) rejoue des transcripts JSONL déterministes ; cette fonction exige un transcript dédié non encore écrit (ISO4) |
| Codex | `auto_continue` | fake_codex (nexus src/bin/fake_codex.rs) rejoue des transcripts JSONL déterministes ; cette fonction exige un transcript dédié non encore écrit (ISO4) |
| Codex | `nats.send` | fake_codex (nexus src/bin/fake_codex.rs) rejoue des transcripts JSONL déterministes ; cette fonction exige un transcript dédié non encore écrit (ISO4) |
| Codex | `nats.interrupt` | fake_codex (nexus src/bin/fake_codex.rs) rejoue des transcripts JSONL déterministes ; cette fonction exige un transcript dédié non encore écrit (ISO4) |
| Codex | `nats.permission_response` | fake_codex (nexus src/bin/fake_codex.rs) rejoue des transcripts JSONL déterministes ; cette fonction exige un transcript dédié non encore écrit (ISO4) |
| Codex | `nats.cancel_tools` | fake_codex (nexus src/bin/fake_codex.rs) rejoue des transcripts JSONL déterministes ; cette fonction exige un transcript dédié non encore écrit (ISO4) |
| Codex | `resume` | fake_codex (nexus src/bin/fake_codex.rs) rejoue des transcripts JSONL déterministes ; cette fonction exige un transcript dédié non encore écrit (ISO4) |
| Codex | `set_model` | fake_codex (nexus src/bin/fake_codex.rs) rejoue des transcripts JSONL déterministes ; cette fonction exige un transcript dédié non encore écrit (ISO4) |
| Codex | `cancel_tools` | fake_codex (nexus src/bin/fake_codex.rs) rejoue des transcripts JSONL déterministes ; cette fonction exige un transcript dédié non encore écrit (ISO4) |
| Codex | `permissions.once` | fake_codex (nexus src/bin/fake_codex.rs) rejoue des transcripts JSONL déterministes ; cette fonction exige un transcript dédié non encore écrit (ISO4) |
| Codex | `permissions.session` | fake_codex (nexus src/bin/fake_codex.rs) rejoue des transcripts JSONL déterministes ; cette fonction exige un transcript dédié non encore écrit (ISO4) |
| Codex | `permissions.always` | fake_codex (nexus src/bin/fake_codex.rs) rejoue des transcripts JSONL déterministes ; cette fonction exige un transcript dédié non encore écrit (ISO4) |
| Codex | `tool_timing` | fake_codex (nexus src/bin/fake_codex.rs) rejoue des transcripts JSONL déterministes ; cette fonction exige un transcript dédié non encore écrit (ISO4) |
| Codex | `po_tools` | fake_codex (nexus src/bin/fake_codex.rs) rejoue des transcripts JSONL déterministes ; cette fonction exige un transcript dédié non encore écrit (ISO4) |
| Codex | `tool_search` | fake_codex (nexus src/bin/fake_codex.rs) rejoue des transcripts JSONL déterministes ; cette fonction exige un transcript dédié non encore écrit (ISO4) |
| Codex | `nexus_tools` | fake_codex (nexus src/bin/fake_codex.rs) rejoue des transcripts JSONL déterministes ; cette fonction exige un transcript dédié non encore écrit (ISO4) |
| Codex | `tool_result.huge` | fake_codex (nexus src/bin/fake_codex.rs) rejoue des transcripts JSONL déterministes ; cette fonction exige un transcript dédié non encore écrit (ISO4) |
| Codex | `enrichment` | fake_codex (nexus src/bin/fake_codex.rs) rejoue des transcripts JSONL déterministes ; cette fonction exige un transcript dédié non encore écrit (ISO4) |
| Codex | `refs` | fake_codex (nexus src/bin/fake_codex.rs) rejoue des transcripts JSONL déterministes ; cette fonction exige un transcript dédié non encore écrit (ISO4) |
| Codex | `session_record` | fake_codex (nexus src/bin/fake_codex.rs) rejoue des transcripts JSONL déterministes ; cette fonction exige un transcript dédié non encore écrit (ISO4) |
| Codex | `subagents` | collabToolCall (sous-agents Codex) NOT VERIFIED dans la suite nexus ; transcript nécessaire |
| Codex | `subagents.background` | collabToolCall (sous-agents Codex) NOT VERIFIED dans la suite nexus ; transcript nécessaire |
| Codex | `subagents.types` | collabToolCall (sous-agents Codex) NOT VERIFIED dans la suite nexus ; transcript nécessaire |
| Codex | `system_init.degraded` | fake_codex (nexus src/bin/fake_codex.rs) rejoue des transcripts JSONL déterministes ; cette fonction exige un transcript dédié non encore écrit (ISO4) |
| Codex | `session_limits` | fake_codex (nexus src/bin/fake_codex.rs) rejoue des transcripts JSONL déterministes ; cette fonction exige un transcript dédié non encore écrit (ISO4) |
| Codex | `anthropic_messages` | Codex utilise son propre moteur LLM (API Codex v1), pas l'API Anthropic Messages ; concept sans objet sur ce moteur |
| opencode | `hooks.before_tool` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `hooks.after_tool` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `hooks.before_compaction` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `hooks.session_start` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `hooks.user_prompt_submit` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `hooks.stop` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `hooks.subagent_stop` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `hooks.notification` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `compaction` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `compaction.manual` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
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
| opencode | `tool_search` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `nexus_tools` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `tool_result.huge` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `enrichment` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `refs` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `provider_switch.relay` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `images` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `session_record` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `background_tasks` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `cancel_task` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `ask_user_question` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `subagents` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `subagents.background` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `subagents.types` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `system_init.degraded` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `session_limits` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| opencode | `anthropic_messages` | fake_opencode n'existe pas dans le dépôt nexus ; le scénario ne peut pas être joué sans binaire faux déterministe (YARDSTICK, tâche 87eb345e) |
| ACP | `compaction` | fake_acp (nexus src/bin/fake_acp.rs) rejoue des transcripts JSONL ; le scénario de parité complet n'est pas encore adapté pour cette fonction (ISO4) |
| ACP | `message_queue` | fake_acp (nexus src/bin/fake_acp.rs) rejoue des transcripts JSONL ; le scénario de parité complet n'est pas encore adapté pour cette fonction (ISO4) |
| ACP | `auto_continue` | fake_acp (nexus src/bin/fake_acp.rs) rejoue des transcripts JSONL ; le scénario de parité complet n'est pas encore adapté pour cette fonction (ISO4) |
| ACP | `nats.send` | fake_acp (nexus src/bin/fake_acp.rs) rejoue des transcripts JSONL ; le scénario de parité complet n'est pas encore adapté pour cette fonction (ISO4) |
| ACP | `nats.interrupt` | fake_acp (nexus src/bin/fake_acp.rs) rejoue des transcripts JSONL ; le scénario de parité complet n'est pas encore adapté pour cette fonction (ISO4) |
| ACP | `nats.permission_response` | fake_acp (nexus src/bin/fake_acp.rs) rejoue des transcripts JSONL ; le scénario de parité complet n'est pas encore adapté pour cette fonction (ISO4) |
| ACP | `nats.cancel_tools` | fake_acp (nexus src/bin/fake_acp.rs) rejoue des transcripts JSONL ; le scénario de parité complet n'est pas encore adapté pour cette fonction (ISO4) |
| ACP | `resume` | fake_acp (nexus src/bin/fake_acp.rs) rejoue des transcripts JSONL ; le scénario de parité complet n'est pas encore adapté pour cette fonction (ISO4) |
| ACP | `set_model` | fake_acp (nexus src/bin/fake_acp.rs) rejoue des transcripts JSONL ; le scénario de parité complet n'est pas encore adapté pour cette fonction (ISO4) |
| ACP | `cancel_tools` | fake_acp (nexus src/bin/fake_acp.rs) rejoue des transcripts JSONL ; le scénario de parité complet n'est pas encore adapté pour cette fonction (ISO4) |
| ACP | `permissions.once` | fake_acp (nexus src/bin/fake_acp.rs) rejoue des transcripts JSONL ; le scénario de parité complet n'est pas encore adapté pour cette fonction (ISO4) |
| ACP | `permissions.session` | fake_acp (nexus src/bin/fake_acp.rs) rejoue des transcripts JSONL ; le scénario de parité complet n'est pas encore adapté pour cette fonction (ISO4) |
| ACP | `permissions.always` | fake_acp (nexus src/bin/fake_acp.rs) rejoue des transcripts JSONL ; le scénario de parité complet n'est pas encore adapté pour cette fonction (ISO4) |
| ACP | `tool_timing` | fake_acp (nexus src/bin/fake_acp.rs) rejoue des transcripts JSONL ; le scénario de parité complet n'est pas encore adapté pour cette fonction (ISO4) |
| ACP | `po_tools` | fake_acp (nexus src/bin/fake_acp.rs) rejoue des transcripts JSONL ; le scénario de parité complet n'est pas encore adapté pour cette fonction (ISO4) |
| ACP | `tool_search` | fake_acp (nexus src/bin/fake_acp.rs) rejoue des transcripts JSONL ; le scénario de parité complet n'est pas encore adapté pour cette fonction (ISO4) |
| ACP | `tool_result.huge` | fake_acp (nexus src/bin/fake_acp.rs) rejoue des transcripts JSONL ; le scénario de parité complet n'est pas encore adapté pour cette fonction (ISO4) |
| ACP | `enrichment` | ACP injecte le contexte PO via le bloc system du premier session/prompt ; la vérification parity cherche PLAN_TITLE dans les requêtes OpenAI qui n'existent pas pour ACP ; non exercé dans ce scénario (ISO4) |
| ACP | `refs` | fake_acp (nexus src/bin/fake_acp.rs) rejoue des transcripts JSONL ; le scénario de parité complet n'est pas encore adapté pour cette fonction (ISO4) |
| ACP | `images` | fake_acp (nexus src/bin/fake_acp.rs) rejoue des transcripts JSONL ; le scénario de parité complet n'est pas encore adapté pour cette fonction (ISO4) |
| ACP | `session_record` | fake_acp (nexus src/bin/fake_acp.rs) rejoue des transcripts JSONL ; le scénario de parité complet n'est pas encore adapté pour cette fonction (ISO4) |
| ACP | `background_tasks` | fake_acp (nexus src/bin/fake_acp.rs) rejoue des transcripts JSONL ; le scénario de parité complet n'est pas encore adapté pour cette fonction (ISO4) |
| ACP | `cancel_task` | fake_acp (nexus src/bin/fake_acp.rs) rejoue des transcripts JSONL ; le scénario de parité complet n'est pas encore adapté pour cette fonction (ISO4) |
| ACP | `ask_user_question` | le backend synthétise la question via le harnais natif ; non exercé dans le scénario de parité (transcripts ACP non adaptés, ISO4) |
| ACP | `subagents` | fake_acp (nexus src/bin/fake_acp.rs) rejoue des transcripts JSONL ; le scénario de parité complet n'est pas encore adapté pour cette fonction (ISO4) |
| ACP | `subagents.background` | fake_acp (nexus src/bin/fake_acp.rs) rejoue des transcripts JSONL ; le scénario de parité complet n'est pas encore adapté pour cette fonction (ISO4) |
| ACP | `subagents.types` | fake_acp (nexus src/bin/fake_acp.rs) rejoue des transcripts JSONL ; le scénario de parité complet n'est pas encore adapté pour cette fonction (ISO4) |
| ACP | `system_init.degraded` | fake_acp (nexus src/bin/fake_acp.rs) rejoue des transcripts JSONL ; le scénario de parité complet n'est pas encore adapté pour cette fonction (ISO4) |
| ACP | `session_limits` | fake_acp (nexus src/bin/fake_acp.rs) rejoue des transcripts JSONL ; le scénario de parité complet n'est pas encore adapté pour cette fonction (ISO4) |
| ACP | `anthropic_messages` | ACP utilise son propre moteur LLM (protocole ACP), pas l'API Anthropic Messages ; concept sans objet sur ce moteur |

## À re-mesurer à chaque montée du CLI Claude Code

`fake_claude` rejoue ce que le vrai CLI a fait à une version donnée : rien ici ne verrait un CLI qui changerait d'avis sans le dire. À chaque montée de version du CLI Claude Code, re-mesurer sur le vrai CLI :

| Comportement | Mesuré sur | Comment | Note PO |
|---|---|---|---|
| re-check PreCompact systemMessage : le CLI ajoute le `systemMessage` d'une réponse au hook PreCompact aux instructions du résumé (et l'affiche à l'utilisateur dans la ligne « Compacted… »), ignore `reason`, rejette `hookSpecificOutput.additionalContext` | Claude Code 2.1.295 | `docs/parity/m2c-precompact/` (README) | 172c164e (tâche M2c 79cc48df) |
| re-check résultat d'outil de 1 Mo : le CLI écrit le résultat entier dans un fichier de la session (`tool-results/`) et ne rend au modèle qu'un avis d'environ 1,6 Ko qui nomme ce fichier, sans aperçu du contenu ni `is_error` ; le tour se clôt normalement (`tool_result.huge`, rejoué par fake_claude) | Claude Code 2.1.296 (18/10/2026) | `docs/parity/gros1b-big-tool-result/` (README) | tâche GROS-1b 4b687177 |
| re-check reprise en lieu neutre : une session ouverte dans le répertoire neutre de l'hôte (`chat-neutral/<id>`), ce répertoire supprimé puis recréé au même chemin, se reprend avec `--resume` sans perte (en 2.1.296 la reprise réussit aussi depuis un AUTRE chemin) ; aucun `CLAUDE.md` ni `.claude/` dans les ancêtres hors `~/.claude` | Claude Code 2.1.296 (18/10/2026) | `docs/parity/t1bis-neutral-place/` (README) | tâche T1-bis 6dae6085 |
