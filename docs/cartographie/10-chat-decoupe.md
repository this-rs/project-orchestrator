# 10 — Découpe de `backend/src/chat`

Analyse en lecture seule (aucun build, aucune modification). Mesure : `find src/chat -name '*.rs' | xargs wc -l` = **205 451 lignes, 128 fichiers**.
Méthode de la colonne « tests » : lignes comprises dans les blocs `#[cfg(test)] mod … { }` de premier niveau (comptées avec awk, tous les blocs, pas seulement le premier) ; les 8 fichiers déclarés `#[cfg(test)] mod` dans `src/chat/mod.rs:21-88` sont comptés en entier. La colonne « code » est le reste.
Les catégories sont une lecture des en-têtes `//!` et des signatures publiques, pas d'un graphe d'appels : les zones grises sont dites.

## 1. Totaux

| Catégorie | Code livré | Tests inline |
|---|---:|---:|
| PRODUIT | 41 412 | (37 285) |
| TRANSPORT | 5 921 | (5 443) |
| SESSION (zone grise) | 42 821 | (41 569) |
| MORT / TEST | 115 297 | — |
| **Total** | **205 451** | |

MORT / TEST = 31 000 lignes de fichiers entièrement de test (`agent_e2e_tests.rs` 17 922, `engine_parity_tests.rs` 6 306, `routing_modes_e2e_tests.rs` 1 610, `eval_transverse.rs` 1 694, `wire_contract.rs` 1 456, `cancel_e2e_tests.rs` 641, `refs_wiring_tests.rs` 433, `stages/pipeline_e2e_tests.rs` 938) + 84 297 lignes de tests inline. **56 % du répertoire est du test** ; le code livré fait **90 154 lignes**.
Les tests inline suivent leur module (colonne « tests »), ils sont totalisés dans MORT / TEST et non dans leur catégorie d'origine.

Code mort : 13 `#[allow(dead_code)]` seulement (composer.rs, subagents.rs, manager.rs, skill_hook.rs, stages/knowledge_injection.rs). Je n'ai pas fait d'analyse d'appels : aucun module entier n'est établi mort ici. À mesurer par `cargo udeps`/`--cfg` hors de cette analyse.

## 2. Table module par module

Colonnes : lignes totales / code livré / tests inline.

| Module | Lignes | Code | Tests | Catégorie | Rôle (en-tête `//!`) |
|---|---:|---:|---:|---|---|
| `prompt.rs` |     3895 | 2667 | 1228 | PRODUIT | constructeur de prompt système |
| `stages/knowledge_injection.rs` |     4973 | 2196 | 2777 | PRODUIT | injection de connaissance |
| `prompt_sections.rs` |     4145 | 1987 | 2158 | PRODUIT | sections de prompt |
| `agent_hooks.rs` |     4928 | 1981 | 2947 | PRODUIT | hooks graphe de connaissance servis à tout provider |
| `anchor_resolver.rs` |     3075 | 1678 | 1397 | PRODUIT | résolution d'ancres |
| `post_stream.rs` |     2731 | 1402 | 1329 | PRODUIT | traitement post-stream (feedback, notes) |
| `viz.rs` |     1872 | 1255 | 617 | PRODUIT | VizBlock |
| `model_catalog.rs` |     1979 | 1185 | 794 | PRODUIT | catalogue de modèles |
| `provider/limits/ledger.rs` |     1985 | 1046 | 939 | PRODUIT | limites : compteurs |
| `feedback.rs` |     1773 | 942 | 831 | PRODUIT | feedback neuronal |
| `composer.rs` |     1799 | 904 | 895 | PRODUIT | FsmPromptComposer |
| `provider/cognitive/decider.rs` |     2180 | 897 | 1283 | PRODUIT | routage cognitif |
| `provider/limits/breaker.rs` |     1502 | 895 | 607 | PRODUIT | limites : breaker |
| `compaction_context.rs` |     1678 | 814 | 864 | PRODUIT | réinjection post-compaction |
| `skill_hook.rs` |     1690 | 793 | 897 | PRODUIT | hook skills |
| `anchor.rs` |     1335 | 788 | 547 | PRODUIT | ancres de contexte |
| `provider/cognitive/segment.rs` |     1429 | 719 | 710 | PRODUIT | routage cognitif |
| `enrichment.rs` |     1390 | 717 | 673 | PRODUIT | pipeline de pré-enrichissement |
| `posture.rs` |     1109 | 712 | 397 | PRODUIT | posture de conversation |
| `viz_builder.rs` |     1326 | 686 | 640 | PRODUIT | VizBlock builder |
| `provider/quota.rs` |     1422 | 673 | 749 | PRODUIT | quota de compte |
| `entity_extractor.rs` |     1096 | 667 | 429 | PRODUIT | extraction d'entités |
| `provider/limits/classify.rs` |     1333 | 659 | 674 | PRODUIT | limites : classification |
| `stages/status_injection.rs` |     1559 | 638 | 921 | PRODUIT | injection statut/raisonnement |
| `provider/cognitive/replay.rs` |     1060 | 632 | 428 | PRODUIT | routage cognitif |
| `provider/limits/admission.rs` |     1370 | 612 | 758 | PRODUIT | limites : admission |
| `routing.rs` |     1197 | 603 | 594 | PRODUIT | sélection de sections de prompt |
| `lasting_rules.rs` |     1053 | 579 | 474 | PRODUIT | approbations durables (coffre de règles) |
| `config.rs` |     1037 | 570 | 467 | PRODUIT | configuration chat |
| `segment_state.rs` |     1306 | 555 | 751 | PRODUIT | état de segment |
| `observation_detector.rs` |      877 | 538 | 339 | PRODUIT | détection d'observations |
| `provider/cognitive/conversation_graph.rs` |      814 | 514 | 300 | PRODUIT | routage cognitif |
| `continuity.rs` |      775 | 486 | 289 | PRODUIT | reprise de contexte entre sessions |
| `provider/cognitive/decision.rs` |      768 | 478 | 290 | PRODUIT | routage cognitif |
| `stages/skill_activation.rs` |      684 | 475 | 209 | PRODUIT | skills |
| `provider/cognitive/candidates.rs` |      993 | 465 | 528 | PRODUIT | routage cognitif |
| `provider/cognitive/mode.rs` |      726 | 437 | 289 | PRODUIT | routage cognitif |
| `provider/cognitive/signature.rs` |      693 | 420 | 273 | PRODUIT | routage cognitif |
| `provider/cognitive/report.rs` |      744 | 417 | 327 | PRODUIT | routage cognitif |
| `provider/cognitive/scorer.rs` |      777 | 401 | 376 | PRODUIT | routage cognitif |
| `untrusted.rs` |      582 | 386 | 196 | PRODUIT | conteneur de texte non fiable |
| `place_confirmation.rs` |      668 | 372 | 296 | PRODUIT | confirmation d'exécution |
| `attention.rs` |      704 | 367 | 337 | PRODUIT | attente utilisateur |
| `provider/cognitive/feedback.rs` |      780 | 351 | 429 | PRODUIT | routage cognitif |
| `post_tool_hook.rs` |      453 | 324 | 129 | PRODUIT | hook PostToolUse |
| `provider/cognitive/topic.rs` |      451 | 317 | 134 | PRODUIT | routage cognitif |
| `relay.rs` |      628 | 306 | 322 | PRODUIT | relais de contexte au changement de moteur |
| `attachment.rs` |      636 | 305 | 331 | PRODUIT | rattachement session→thread |
| `provider/cognitive/store.rs` |      527 | 305 | 222 | PRODUIT | routage cognitif |
| `stages/file_context.rs` |      428 | 258 | 170 | PRODUIT | contexte fichier |
| `message_attachments.rs` |      489 | 252 | 237 | PRODUIT | documents joints |
| `provider/cognitive/demotion.rs` |      481 | 234 | 247 | PRODUIT | routage cognitif |
| `provider/limits/profiles.rs` |      321 | 204 | 117 | PRODUIT | limites : profils |
| `cost.rs` |      337 | 197 | 140 | PRODUIT | comptabilité de coût |
| `stages/biomimicry.rs` |      173 | 173 | 0 | PRODUIT | biomimétisme |
| `stages/persona.rs` |      447 | 173 | 274 | PRODUIT | personas |
| `activation_registry.rs` |      253 | 172 | 81 | PRODUIT | registre skills/personas injectés |
| `segment_tap.rs` |      249 | 150 | 99 | PRODUIT | tap de segment |
| `retrieval_trigger.rs` |      260 | 145 | 115 | PRODUIT | invitation à interroger l'index |
| `stages/mcp_federation_stage.rs` |      616 | 144 | 472 | PRODUIT | fédération MCP |
| `stages/intent_weights.rs` |      256 | 133 | 123 | PRODUIT | poids d'intention |
| `stages/user_profile.rs` |      219 | 118 | 101 | PRODUIT | profil utilisateur |
| `provider/limits/store.rs` |      169 | 116 | 53 | PRODUIT | limites : persistance |
| `provider/cognitive/auto_models.rs` |      205 | 114 | 91 | PRODUIT | routage cognitif |
| `manual_compaction.rs` |      188 | 109 | 79 | PRODUIT | /compact |
| `content_policy.rs` |      166 | 100 | 66 | PRODUIT | posture de contenu |
| `provider/cognitive/observe.rs` |      197 | 86 | 111 | PRODUIT | routage cognitif |
| `tree.rs` |      190 | 86 | 104 | PRODUIT | annotation d'arbre |
| `provider/cognitive/segment_store.rs` |      176 | 85 | 91 | PRODUIT | routage cognitif |
| `provider/cognitive/mod.rs` |      143 | 75 | 68 | PRODUIT | routage cognitif |
| `provider/cognitive/wiring.rs` |       74 | 74 | 0 | PRODUIT | routage cognitif |
| `provider/limits/mod.rs` |      124 | 69 | 55 | PRODUIT | test seul |
| `stages/mod.rs` |       29 | 29 | 0 | PRODUIT | test seul |
| `manager.rs` |    40957 | 20726 | 20231 | SESSION | Pôle de session (ChatManager) – à éclater |
| `agent_runtime.rs` |     8767 | 3809 | 4958 | SESSION | moteur de session neutre (boucle d'un tour) – candidat harnais |
| `provider/settings.rs` |     3388 | 1862 | 1526 | SESSION | réglages des instances provider |
| `provider/nexus_tools.rs` |     3823 | 1717 | 2106 | SESSION | nexus-tools attaché à une session native |
| `subagents.rs` |     2566 | 1554 | 1012 | SESSION | sous-agents natifs |
| `session_grants.rs` |     2725 | 1262 | 1463 | SESSION | portée des permissions de session |
| `mirror.rs` |     1991 | 1131 | 860 | SESSION | miroirs de sous-agents Claude Code |
| `manager/recovery.rs` |     1292 | 917 | 375 | SESSION | reprise des tours coupés par un redémarrage |
| `envelope.rs` |     1748 | 777 | 971 | SESSION | enveloppe de spawn |
| `provider/native_factory.rs` |     1207 | 662 | 545 | SESSION | fabrique du harnais natif |
| `provider/errors.rs` |     1460 | 656 | 804 | SESSION | erreurs d'ouverture typées |
| `provider/endpoint_guard.rs` |     1295 | 648 | 647 | SESSION | SSRF/TLS |
| `provider/outage.rs` |      918 | 598 | 320 | SESSION | pannes provider |
| `provider/resolver.rs` |     1201 | 553 | 648 | SESSION | résolution provider/modèle |
| `cli_version.rs` |      855 | 497 | 358 | SESSION | version CLI Claude |
| `team.rs` |      957 | 480 | 477 | SESSION | TeamHooks |
| `session_pool.rs` |      719 | 421 | 298 | SESSION | pool de sessions chaudes |
| `tool_clock.rs` |      895 | 396 | 499 | SESSION | horloge des outils |
| `provider/hooks.rs` |      483 | 335 | 148 | SESSION | hooks neutres |
| `provider/policy.rs` |      739 | 329 | 410 | SESSION | modes de politique |
| `provider/circuit_breaker.rs` |      599 | 322 | 277 | SESSION | disjoncteur par instance |
| `provider/listing.rs` |      621 | 295 | 326 | SESSION | GET providers |
| `spill.rs` |      497 | 294 | 203 | SESSION | déversement de résultats d'outil |
| `provider/transcripts.rs` |      610 | 271 | 339 | SESSION | transcripts disque |
| `turn_admission.rs` |      503 | 251 | 252 | SESSION | admission d'un tour |
| `deferred_tools.rs` |      378 | 234 | 144 | SESSION | outils différés (CTX2) |
| `hook_ledger.rs` |      302 | 204 | 98 | SESSION | mémoire du PreToolUse |
| `provider/catalog.rs` |      526 | 202 | 324 | SESSION | catalogue d'instances |
| `provider/oneshot.rs` |      444 | 201 | 243 | SESSION | sessions one-shot |
| `cancelled_parents.rs` |      298 | 200 | 98 | SESSION | tombes parent annulé |
| `delegation_limits.rs` |      193 | 143 | 50 | SESSION | plafonds de délégation |
| `neutral_place.rs` |      207 | 136 | 71 | SESSION | cwd neutre |
| `cli_auth.rs` |      308 | 130 | 178 | SESSION | auth CLI Claude |
| `session_record.rs` |      176 | 121 | 55 | SESSION | nœud ChatSession |
| `path_detect.rs` |      188 | 117 | 71 | SESSION | PATH login shell |
| `provider/credentials.rs` |      229 | 107 | 122 | SESSION | credentials (coffre) |
| `mod.rs` |       96 | 96 | 0 | SESSION | test seul |
| `provider/store.rs` |       74 | 74 | 0 | SESSION | lecture réglages |
| `resume_guard.rs` |      125 | 63 | 62 | SESSION | garde de reprise |
| `provider/mod.rs` |       30 | 30 | 0 | SESSION | test seul |
| `agent_e2e_tests.rs` |    17922 | 0 |    17922 | TEST | test seul |
| `cancel_e2e_tests.rs` |      641 | 0 |      641 | TEST | test seul |
| `engine_parity_tests.rs` |     6306 | 0 |     6306 | TEST | test seul |
| `eval_transverse.rs` |     1694 | 0 |     1694 | TEST | test seul |
| `refs_wiring_tests.rs` |      433 | 0 |      433 | TEST | test seul |
| `routing_modes_e2e_tests.rs` |     1610 | 0 |     1610 | TEST | test seul |
| `stages/pipeline_e2e_tests.rs` |      938 | 0 |      938 | TEST | test seul |
| `wire_contract.rs` |     1456 | 0 |     1456 | TEST | test seul |
| `types.rs` |     4464 | 2518 | 1946 | TRANSPORT | ChatEvent/ClientMessage (contrat filaire) + types mélangés |
| `oob_listener.rs` |     2374 | 1220 | 1154 | TRANSPORT | écouteur de messages hors-bande SDK |
| `provider/event_map.rs` |     2202 | 935 | 1267 | TRANSPORT | AgentEvent→ChatEvent |
| `drain.rs` |      764 | 459 | 305 | TRANSPORT | drain de la file après stream |
| `cancel_relay.rs` |      681 | 405 | 276 | TRANSPORT | relais d'annulation inter-instances (NATS) |
| `pending_queue.rs` |      498 | 195 | 303 | TRANSPORT | file des messages tenus |
| `control_pump.rs` |      381 | 189 | 192 | TRANSPORT | pompe du canal de contrôle SDK |

## 3. Destinations proposées

| Catégorie | Destination | Condition |
|---|---|---|
| PRODUIT (41 412) | Reste dans PO. Sous-ensembles : prompt (`prompt.rs`, `prompt_sections.rs`, `composer.rs`, `routing.rs`) ; injection (`stages/*`, `enrichment.rs`, `agent_hooks.rs`, `skill_hook.rs`, `post_*`) ; mémoire/reprise (`continuity.rs`, `compaction_context.rs`, `relay.rs`, `anchor*.rs`, `posture.rs`, `segment_*.rs`) ; routage cognitif (`provider/cognitive/*`, 7 021 lignes de code) ; quota et limites (`provider/quota.rs`, `provider/limits/*`, 3 601) ; règles durables (`lasting_rules.rs`) | `provider/limits/*` et `provider/circuit_breaker.rs`/`outage.rs` sont de la résilience de provider : ils pourraient aller au harnais (zone grise, classés PRODUIT/SESSION faute d'avoir lu leurs appelants) |
| TRANSPORT (5 921) | this-harness : `drain.rs`, `pending_queue.rs`, `control_pump.rs`, `oob_listener.rs`, `cancel_relay.rs`, `provider/event_map.rs`, et la part « contrat filaire » de `types.rs` (`ChatEvent` `types.rs:710`, `ClientMessage` `types.rs:1639`) | `types.rs` mélange le contrat filaire (ChatEvent, ClientMessage, ~1 000 lignes) et des types produit (`ChatSession` `types.rs:1724`, `ChatLinkedPlan`, `SessionWorkLog`) : à scinder avant tout déplacement. Le WebSocket lui-même n'est PAS dans `chat/` : `api/ws_chat_handler.rs` (2 284 lignes) + `api/chat_handlers.rs` (7 745) — le vrai transport est hors de ce répertoire |
| SESSION (42 821) | À trancher module par module : `agent_runtime.rs` (boucle neutre d'un tour, `TurnServices`/`ProviderSource` `agent_runtime.rs:684,754`) est le meilleur candidat au harnais ; `manager.rs` reste PO tant que `ChatManager` porte 85 fichiers appelants hors de `chat/` ; `provider/*` (instances, résolution, fabrique native) est de la configuration produit en pratique | |
| MORT / TEST (115 297) | Reste avec son code ; les 8 fichiers de test entiers (31 000) bougent avec le sous-système qu'ils testent. `agent_e2e_tests.rs` (17 922) et `engine_parity_tests.rs` (6 306) testent la parité des moteurs : ils appartiennent au harnais si la boucle y va | |

## 4. `manager.rs` (40 957 lignes) par sous-ensembles

Moitié test (20 231, blocs `idle_expiry_tests` ~`manager.rs:20411` à `granted_answer_clock_tests` ~`:40764`, `test_support` `:38871`). Les 20 726 lignes de code, par plus gros items :

| Sous-ensemble | Plage / signature | Lignes | Catégorie |
|---|---|---:|---|
| Boucle de stream Claude Code (legacy) | `stream_response` `manager.rs:9633` | 1 525 | SESSION (miroir de `agent_runtime` pour le CLI) |
| Création de session | `create_session_resolved` `:7957` | 933 | SESSION |
| Réouverture legacy / agent | `reopen_legacy_session_for` `:13583` (581), `reopen_agent_session_with` `:18951` (369), `open_agent_session` `:18034` (278) | 1 228 | SESSION |
| Listeners NATS (snapshot, interrupt, cancel) | `spawn_nats_rpc_listener` `:5372` (455), `spawn_agent_nats_listeners` `:18675` (162), `spawn_nats_cancel_tools_listener` `:15531` (126), … | ~900 | TRANSPORT (nommé SESSION ici car dans ChatManager) |
| Annulation / interruption | `interrupt_scoped` `:14938`, `cancel_running_tools` `:15118`, `cancel_legacy_task` `:15329` | ~480 | TRANSPORT/SESSION |
| Permissions | `deliver_permission_answer` `:11987`, `set_session_permission_mode` `:12499` | ~410 | SESSION |
| Envoi / file / équipe | `send_message` `:11209`, `inject_team_message` `:11491`, `route_queued_user_message` `:11832` | ~530 | SESSION + TRANSPORT |
| Routage cognitif de tour | `cognitive_decision` `:16856`, `routing_pool` `:16630`, `close_routing_decision` `:16423` | ~600 | PRODUIT, coincé dans le manager |
| Après-tour | `after_turn` `:2696` (370), `message_to_events` `:7371` (248) | ~620 | PRODUIT/TRANSPORT |
| Posture shadow | `spawn_posture_shadow` `:1736` | 200 | PRODUIT |
| Structure | `ChatManager` `:1065` (180 lignes de champs), `ActiveSession` `:453` (213) | ~390 | SESSION |

Les 3 funcs de plus de 450 lignes sont les points durs : `stream_response` (1 525), `create_session_resolved` (933), `reopen_legacy_session_for` (581).

## 5. Les cinq découpes les plus risquées

1. **`manager.rs` : `stream_response` (1 525 lignes, `:9633`) et la boucle legacy Claude Code.** C'est la jumelle de `agent_runtime.rs` pour le CLI ; elle emmêle stream, file de messages tenus (`hold_legacy_turn` `:9622`), pannes (`record_legacy_outage_up` `:9603`) et post-stream. Les deux boucles doivent rester équivalentes (`engine_parity_tests.rs`, 6 306 lignes) : couper l'une sans l'autre casse la parité Claude Code ↔ natif, objet d'un plan en cours.
2. **`types.rs` (4 464) : séparer le contrat filaire du modèle produit.** `ChatEvent` (`types.rs:710`, ~900 lignes) est le contrat avec le frontend, verrouillé par `wire_contract.rs` (1 456) et des fixtures ; tout champ déplacé change le JSON. Le même fichier porte `ChatSession`, `SpawnedBy`, `RunnerContext`, importés par 85 fichiers hors de `chat/` : un déplacement casse des centaines d'imports.
3. **`agent_runtime.rs` (8 767) vers le harnais.** Il dépend de services PO via `TurnServices` (`:684`) et `ProviderSource` (`:754`), mais aussi de `agent_hooks`, `session_grants`, `subagents`, `envelope`. Sortir la boucle sans ses hooks de graphe revient à refaire le harnais ; la sortir avec en importe tout le produit. La frontière réelle est ces deux traits : il faut les faire tenir sur un contrat avant de bouger du code.
4. **Permissions et attentes : `session_grants.rs` (2 725), `lasting_rules.rs`, `deliver_permission_answer` `manager.rs:11987`, `oob_listener.rs` (2 374).** Règles de sécurité (comparateur strict, jamais transmis au provider, réponse humaine seulement). Un déplacement qui change où se décide « ce que couvre un `always` » peut ouvrir un consentement fantôme ; le test d'échec fermé doit suivre.
5. **Les listeners NATS et l'annulation inter-instances (`manager.rs:5372`, `:15531`, `cancel_relay.rs`, `cancelled_parents.rs`).** Transport au sens propre, mais couplés à `ActiveSession` (`:453`) et à des courses documentées (« B36-RACE », `cancelled_parents.rs`). Les sortir sans le registre de sessions vivantes crée des annulations perdues ou des tours fantômes en multi-instance.

## 6. Première découpe recommandée

**Extraire les 20 231 lignes de tests inline de `manager.rs` vers des fichiers séparés** (`manager/tests/*.rs`, un par module déjà nommé : `idle_expiry_tests`, `runner_spawn_tests`, `held_context_tests`, `turn_reward_play_tests`…). Gain : `manager.rs` passe à ~20 700 lignes de code sans toucher la production ; risque nul, mécanique. Ensuite : sortir `cognitive_decision`/`routing_pool`/`close_routing_decision` (~600 lignes PRODUIT) vers `provider/cognitive`, puis scinder `types.rs`.

## 7. Limites de cette analyse

Pas de graphe d'appels : les catégories viennent des en-têtes et des noms ; plusieurs modules `provider/*` sont classés SESSION par défaut. Les « tests » excluent les `#[cfg(test)] fn`/`impl` isolés hors `mod`. « Code mort » : seul le compte de `allow(dead_code)` a été relevé.
