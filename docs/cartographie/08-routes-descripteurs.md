# 08 — Les routes REST de PO exprimées comme descripteurs this-rs

Analyse en lecture seule (aucun build, aucun git). Comptes produits par un script d'extraction des `.route(` de `backend/src/api/routes.rs` (script et sortie dans le scratchpad de session, non versionnés), puis classés par règles + mesure de chaque handler (longueur, émissions d'événements). Les règles sont données plus bas : la classification est **reproductible, pas intuitive**, mais c'est une heuristique — voir « Limites ».

## 0. Ce qui est compté (et l'écart avec « 514 »)

- `routes.rs` contient **483** appels `.route(` (tous en code de production : le bloc `#[cfg(test)]` commence à `backend/src/api/routes.rs:2455`, le dernier `.route(` est à `:2448`). Les `.route(` de `auth_handlers.rs`, `document_handlers.rs`, `hook_handlers.rs`, `handlers.rs` sont tous dans des modules de test (`auth_handlers.rs:1138`, `document_handlers.rs:1089`, `hook_handlers.rs` ≈ `:682`).
- Un `.route(` porte souvent plusieurs verbes (`get(..).post(..)`, ex. `routes.rs:470`) : **629 opérations** (méthode × chemin), dont 4 dans les deux `Router::new().route(` hors `protected_routes` (`/mcp` `routes.rs:335` ; `/hooks/events/{source}/{trigger_id}` `:344`).
- Le chiffre « 514 » n'est reproduit ni par l'un ni par l'autre (483 / 629). Je n'ai trouvé aucune source qui donne 514 ; ce document compte les deux unités et ne s'appuie pas sur 514. Les routes SPA/fallback statiques (`routes.rs:144-210`) ne sont pas des `.route(`.
- Plusieurs routes sont des doublons de compat : `/api/chat/providers/*` et `/api/agent/providers/*` (`routes.rs:2222-2298`), `/api/projects/{slug}/llm-consent` et `llm-consents` (`:2329-2341`), `/api/wake` et `/hooks/wake` (`:1050,1053`).

## 1. Ce que this-rs sait générer (les faits qui commandent la classification)

| Fait | Source |
|---|---|
| Par entité : `GET/POST /{plural}` et `GET/PUT/PATCH/DELETE /{plural}/{id}`, **plat**, `PUT` et `PATCH` branchés sur le MÊME handler | `src/server/descriptor_entity.rs:80-91` |
| `{id}` est un `Uuid` (pas de slug, pas de clé naturelle) | `descriptor_entity.rs:183,190,197` |
| Création → **201** ; liste → enveloppe `{data, pagination:{page,limit,total,total_pages}}`, filtres `?filter=<json>&sort=&page=&limit=` (défaut 20) | `descriptor_entity.rs:176-178`, `:143-170`, `:123-141` |
| Imbrication et relations = **liens** : `GET/POST /{type}/{id}/{route_name}`, `POST/PUT/DELETE …/{route_name}/{target_id}`, 2 niveaux auto, 3+ via handler générique | `src/server/router.rs:55-71` |
| Une entité = champs typés, index, permissions, i18n, `custom` ; **aucun** hook, aucune route nommée | `crates/this-descriptor/src/entity.rs:22-45` |
| Un module déclare aussi des `operations` nommées (« terminer », « recalculer ») — non CRUD, atteintes par `handle(op:"call")`, **pas** par une route REST générée | `crates/this-descriptor/src/module.rs:23,73` |
| Échappatoire officielle : `with_custom_routes(Router)` → les routes hors descripteur cohabitent dans le même serveur | `src/server/builder.rs:480` |
| Un transport SSE existe côté exposition ; pas de WebSocket généré | `src/server/exposure/sse.rs:47` |
| Chaque opération passe par `SurfaceGuard` (politique par opération) | `descriptor_entity.rs:70-72` |

Conséquence : **« descripteur » = entité + liens + permissions. Ce n'est pas un routeur.** Tout ce qui n'est pas `{plural}` / `{plural}/{id}` / lien est hors descripteur, qu'on le veuille ou non.

## 2. Règles de classification

- **A — CRUD d'entité pur** : chemin plat `/api/<plural>` (GET/POST) ou `/api/<plural>/{id}` (GET/PUT/PATCH/DELETE), clé non-slug, ET handler « mince » (≤ 30 lignes, aucune émission `event_bus`/`emit_*`). Descripteur direct.
- **B — CRUD + logique** : même forme, mais (a) le handler fait plus que le store (> 30 lignes ou émet un événement WS — mesuré par `ev`), ou (b) la route est **imbriquée** (`/projects/{id}/environments`, `/tasks/{id}/steps`, `/personas/{id}/skills/{sid}`…) donc exprimable seulement comme **lien**, ou (c) l'identifiant est un slug (`/api/projects/{slug}`, `/api/workspaces/{slug}`). Descripteur + hook (inexistant aujourd'hui) ou handler conservé.
- **C — verbe applicatif** : tout le reste. Sous-types mesurés : *verbe* (`run`, `cancel`, `sync`, `activate`, `transition`, `enable/disable`…), *lecture calculée* (`progress`, `stats`, `overview`, `waves`, `topology`…), *recherche/calcul* (`search`, `similar`, `simulate`, `compose`…), *analyse de code* (`/api/code/*`), *graphe calculé*, *maintenance* (`/api/admin/*`, `/api/meilisearch/*`), *singleton/config* (route avec `PUT`/`PATCH` sur un chemin sans `{id}` : un réglage, pas une collection), *coffre*, *connexion vivante* (MCP fédéré), *protocole/auth/système* (OAuth, `/auth/*`, `/health`, hooks, `/mcp`), *stream* (4 WebSocket, `routes.rs:427-430`).

Deux ruptures de contrat s'appliquent à **toute** la classe A/B, indépendamment de la classe :

1. **Enveloppe de liste** : PO rend `PaginatedResponse {items,total,limit,offset}` (`backend/src/api/query.rs:163-171`, ex. `environment_handlers.rs:269-289`) ; this-rs rend `{data,pagination:{page,limit,total,total_pages}}` (`descriptor_entity.rs:160-170`). Le frontend et les fixtures de contrat chat doivent suivre, ou un adaptateur de sortie est nécessaire.
2. **Statut de création** : PO rend 200 + corps (`environment_handlers.rs:147,184` : `Json<EnvironmentNode>`) ; this-rs rend 201 (`descriptor_entity.rs:178`).
Un troisième point touche chaque `PATCH` de PO qui n'est pas une mise à jour de champs : `PATCH /api/plans/{plan_id}` est `update_plan_status` (`routes.rs:687`), c'est-à-dire une transition d'état, alors que this-rs envoie `PATCH` et `PUT` au même `update_from_json` (`descriptor_entity.rs:89`).

## 3. Les TROIS comptes

| Unité | **A** CRUD pur | **B** CRUD + logique | **C** verbe applicatif | Total |
|---|---|---|---|---|
| Appels `.route(` (classe = la plus lourde de ses opérations) | **19** (3,9 %) | **116** (24,0 %) | **348** (72,0 %) | **483** |
| Opérations (méthode × chemin) | **61** (9,7 %) | **179** (28,5 %) | **389** (61,8 %) | **629** |

Détail des opérations par sous-type :

| Classe | Sous-type | Ops |
|---|---|---|
| A | CRUD pur | 61 |
| B | imbriqué (lien) — avec ou sans logique | 100 |
| B | CRUD à plat + logique (hook/événement) | 75 |
| B | clé slug (Uuid imposé) | 4 |
| C | lecture calculée | 71 |
| C | verbe applicatif | 59 |
| C | singleton/config | 57 |
| C | analyse de code | 50 |
| C | protocole/auth/système | 43 |
| C | maintenance | 34 |
| C | recherche/calcul | 29 |
| C | autre | 14 |
| C | coffre (secret, pas une entité) | 12 |
| C | graphe calculé | 9 |
| C | connexion vivante (MCP fédéré) | 7 |
| C | stream (WebSocket) | 4 |

Lecture honnête : **seulement ~10 % des opérations sont du CRUD qu'un descripteur absorbe tel quel**, ~28 % le sont avec un crochet (ou en lien), et **~62 % n'ont aucune forme de ressource**. 72 des 179 opérations B ont un handler qui émet un événement `event_bus` (ex. `environment_handlers.rs:206-236`, `create_environment` `:147-184`) : sans crochet post-écriture dans this-rs, la mise à jour live du frontend casse.

## 4. Exemples nommés par classe

- **A** : `GET /api/environments/{id}` (`environment_handlers.rs:186`, 6 l.), `GET /api/notes/{note_id}` (`routes.rs:1365`), `GET /api/trajectories/{id}` (`:602`), `GET /api/skills/{skill_id}`, `GET/DELETE /api/documents/{document_id}` (`:1332`), `GET /api/analysis-profiles` (`:1495`).
- **B** : `POST /api/projects/{project_id}/environments` (nom unique + événement, `environment_handlers.rs:147-184`) ; `PATCH /api/deployments/{id}` (55 l., 4 émissions, `:348`) ; `PATCH /api/workspaces/{slug}` (clé slug) ; `POST /api/personas/{id}/skills/{skill_id}` (lien pur) ; `POST /api/lifecycle-hooks`.
- **C** : `POST /api/plans/{plan_id}/run` (`handlers.rs:7094`), `POST /api/chat/sessions/{id}/interrupt` (`chat_handlers.rs:948`), `GET /api/code/impact` (`code_handlers.rs:593`, 343 l.), `POST /api/admin/deep-maintenance/{project_id}` (`handlers.rs:5885`), `GET /ws/chat/{session_id}` (`ws_chat_handler.rs:114`), `PUT /api/chat/roles` (singleton).

## 5. Les cinq familles les plus dures

| Famille | Ops C | Pourquoi ça ne rentre pas | Ce qu'il faudrait dans this-rs |
|---|---|---|---|
| **chat** (74 routes / 113 ops, dont 75 C) | 41 singletons de configuration (`/api/chat/roles`, `model-aliases`, `model-policy`, `auto-models`, `delegation-limits`, `deferred-tools`, `routing`, `llm-consents`… `routes.rs:2210-2416`), ~28 verbes de session (`interrupt` `:2174`, `switch-provider` `:2186`, `cancel-tools`, `associate`, `send_session_message` `chat_handlers.rs:1332`), providers avec références de secret (`vault:`), 2 WebSocket | (1) `EntityKind::Singleton` : `GET/PUT /{name}` sans id ; (2) routes de **commande** déclarées (`POST /{plural}/{id}/{verb}`) liées à une `OperationDesc` ; (3) un transport WebSocket bidirectionnel (seul SSE existe, `exposure/sse.rs:47`) ; (4) champ de type « référence de secret » avec politique de lecture |
| **code** (56 routes / 59 ops, 53 C) | 53 | Lectures/POST-lecture sur le graphe de code, paramétrées par `project_slug`/`file_path`, jamais par un id d'entité (`GET /api/code/symbols/{*file_path}` `routes.rs:1077`) ; `analyze_impact` fait 343 lignes ; `POST` utilisé pour des *lectures* (`structural-twins`, `stress-test-*` `:1236-1275`) | Une catégorie **« vue calculée »** (lecture nommée, entrée typée, sortie typée) à côté de `ViewDesc`/`OperationDesc` (`module.rs:73`) ; paramètres de chemin joker (`{*path}`) ; clé naturelle (slug/chemin) comme identifiant |
| **admin** (32 routes, 34 ops, 100 % C) | 34 | Tâches de maintenance idempotentes, souvent asynchrones avec état (`backfill-embeddings` POST + DELETE + `/status`, `routes.rs:1792-1808`) : ce sont des **jobs**, pas des ressources | Une notion de **job** (démarrer / annuler / statut / progression) déclarable ; sinon `with_custom_routes` (`builder.rs:480`) — aucune valeur à les forcer en descripteur |
| **runs** (31 routes, 35 ops, 23 C) | 23 | Cycle de vie d'exécution : `run`, `cancel`, `force-cancel`, `retry`, `auto-pr`, budget (`routes.rs:707-736`), déclencheurs, webhooks, placements d'exécution, `/ws/run/{run_id}` (`:430`). L'état change par **verbes gardés** (FSM de run, `protocol_handlers.rs`), pas par `PATCH` | Un **automate d'état déclaré** (états, transitions, gardes) avec routes `POST …/{transition}` générées, plus WebSocket de suivi ; sinon handler conservé |
| **graph** (9 routes, 9 C) | 9 | Sorties qui ne sont **pas des entités** : voisinage, chemin typé, schéma, graphe projet/workspace (`graph_handlers.rs:113,149`, `routes.rs:1480-1487`, `:492`) — résultats de requêtes sur le graphe, formes variables | Rien à générer : c'est le rôle de l'exposition GraphQL/gRPC de this-rs ; côté REST, une **route de requête** déclarée (entrée = requête, sortie = sous-graphe) |

À noter : `ws` (4), `oauth/auth` (21 ops, dont 15 `/auth/*` et le serveur OAuth `routes.rs:307-321`), `vault` (12) et `sys` (17) sont aussi C ; ils restent de toute façon hors descripteur (protocole, secret, santé).

## 6. Le geste concret — premier lot

**Lot 1 : « environnements & déploiements » + trois CRUD voisins, plats ou à un seul niveau** — 11 appels `.route(`, 23 opérations, tous dans le dépôt `backend`, aucun WebSocket, aucun slug.

| Route (`routes.rs`) | Ops | Entité/lien |
|---|---|---|
| `:645` `/api/projects/{project_id}/environments` | GET, POST | `environment` ← lien `project→environments` |
| `:649` `/api/projects/{project_id}/deployment-matrix` | GET | **reste hors descripteur** (vue calculée, `environment_handlers.rs:410`) |
| `:655` `/api/environments/{id}` | GET, PATCH, DELETE | `environment` |
| `:660` `/api/environments/{id}/deployments` | GET, POST | `deployment` ← lien `environment→deployments` |
| `:664` `/api/deployments/{id}` | PATCH | `deployment` |
| `:789` `/api/plans/{plan_id}/constraints` | GET, POST | `constraint` ← lien `plan→constraints` |
| `:795` `/api/constraints/{constraint_id}` | GET, PATCH, DELETE | `constraint` |
| `:800` `/api/lifecycle-hooks` | GET, POST | `lifecycle_hook` |
| `:806` `/api/lifecycle-hooks/{hook_id}` | GET, PATCH, DELETE | `lifecycle_hook` |
| `:1495` `/api/analysis-profiles` | GET, POST | `analysis_profile` |
| `:1500` `/api/analysis-profiles/{profile_id}` | GET, DELETE | `analysis_profile` |

Pourquoi ce lot : (i) tous les handlers ont une longueur de 6 à 61 lignes et un seul type de logique — validation de nom/enum et émission d'un événement (`environment_handlers.rs:84-132`, `:147-184`) ; (ii) il force à résoudre, sur un petit périmètre, les **trois** difficultés génériques : lien imbriqué (`project→environments`), crochet post-écriture (événement WS), et adaptateur de contrat (201 vs 200, enveloppe de liste) ; (iii) tous les ids sont des UUID.

**Ce qui prouve que le lot est fini** (critères vérifiables, à rejouer) :
1. Un `ModuleDescriptor` (`crates/this-descriptor/src/module.rs:23`) déclare 5 entités (`environment`, `deployment`, `constraint`, `lifecycle_hook`, `analysis_profile`) et 3 liens ; il est chargé par l'hôte et les routes ne sont **plus** déclarées à la main dans `routes.rs` (les 10 `.route(` correspondants supprimés ; `deployment-matrix` reste à la main, via `with_custom_routes`, `builder.rs:480`).
2. Test de contrat différentiel : pour chacune des 22 opérations (23 − matrice), le même scénario HTTP est rejoué contre l'ancien et le nouveau routeur ; les corps JSON et les statuts sont identiques **ou** l'écart est listé et accepté (201/200, enveloppe de liste) avec mise à jour du frontend et des fixtures de contrat.
3. Les événements `Environment`/`Deployment`/`Constraint` sont toujours émis (test sur le bus : création, mise à jour, suppression), preuve que le crochet remplace bien `event_bus.emit_*`.
4. Règles métier conservées : refus du nom d'environnement dupliqué (`ensure_name_available`, `environment_handlers.rs:95`), validation d'enum (`kind`, `status`), existence du projet parent (`require_project`, `:84`).
5. `cargo test` sur les modules touchés, portillon diagrammes vert, compte de routes dans `routes.rs` : 483 → 473 `.route(`.

Ce que le lot **ne** prouve pas : ni les WebSocket, ni les jobs, ni les verbes d'exécution. Il dit seulement si le plafond (~10 % A, ~28 % B) est atteignable sans perte de comportement.

## 7. Verdict de faisabilité

- Atteignable par descripteur, avec crochets + liens + adaptateur de contrat : **A + B = 240 ops / 629 (38 %)**.
- Hors descripteur par construction (garder des handlers ou `with_custom_routes`) : **C = 389 ops (62 %)** — dont 57 singletons et 71 lectures calculées qui pourraient entrer avec deux notions nouvelles (`Singleton`, `vue calculée`), soit un plafond théorique de ~**75 %** si ces deux ajouts existent ; le reste (maintenance, analyse de code, coffre, auth/OAuth, MCP fédéré, WebSocket) demande des notions (job, automate, transport WS) qui ne sont pas des descripteurs d'entité.

## 8. Limites de cette mesure

- La classe A/B repose sur deux mesures de handler (longueur ≤ 30 l., présence d'`event_bus`) ; un handler court peut cacher de la logique dans une fonction appelée. Les handlers à longueur 0 dans le script (`feedback_handlers::*`, `routes.rs:1309-1313`) n'ont pas été retrouvés ; classés avec la règle de forme seule.
- Plusieurs noms de handlers existent dans plusieurs fichiers (`list_triggers`, `get_trigger`…) : l'index est par `module::fn`, donc non ambigu pour les routes préfixées ; pour les rares `super::`/`anchors::`, la résolution se fait par le dernier module.
- Je n'ai pas relu les 629 handlers un par un ; seuls ceux cités ci-dessus l'ont été.
- Les routes sont comptées telles que déclarées, sans tenir compte des `cfg(feature)` : `remote_mcp` (`routes.rs:82,119`) et l'ingress (`:85`) sont conditionnels.

## Tableau complet par domaine

Une ligne = un appel `.route(` (n° de ligne dans `backend/src/api/routes.rs`). La classe de la ligne est la PLUS lourde de ses opérations ; entre parenthèses, la classe de chaque opération. `loc` = lignes du handler, `ev` = émissions `event_bus`/`emit_*` dans son corps.

### Comptes par domaine

| Domaine | Routes A | B | C | Total | Opérations A | B | C | Total |
|---|---|---|---|---|---|---|---|---|
| admin | 0 | 0 | 32 | 32 | 0 | 0 | 34 | 34 |
| chat | 8 | 13 | 53 | 74 | 16 | 22 | 75 | 113 |
| code | 0 | 4 | 52 | 56 | 1 | 5 | 53 | 59 |
| environments | 0 | 4 | 1 | 5 | 1 | 7 | 1 | 9 |
| graph | 0 | 0 | 9 | 9 | 0 | 0 | 9 | 9 |
| hooks | 0 | 0 | 5 | 5 | 0 | 0 | 5 | 5 |
| mcp-federation | 0 | 0 | 5 | 5 | 0 | 0 | 7 | 7 |
| neural/trajectory/feedback | 3 | 1 | 11 | 15 | 6 | 1 | 12 | 19 |
| notes/knowledge | 1 | 5 | 23 | 29 | 4 | 5 | 24 | 33 |
| oauth/auth | 0 | 0 | 18 | 18 | 0 | 0 | 21 | 21 |
| plans/tasks | 3 | 32 | 34 | 69 | 12 | 51 | 36 | 99 |
| projects | 0 | 4 | 3 | 7 | 0 | 7 | 4 | 11 |
| protocols/rfc/reason/episode | 0 | 14 | 15 | 29 | 3 | 21 | 15 | 39 |
| runs/plans-exec | 2 | 6 | 23 | 31 | 4 | 8 | 23 | 35 |
| sharing | 0 | 6 | 5 | 11 | 0 | 6 | 6 | 12 |
| skills/personas/registry | 1 | 11 | 25 | 37 | 7 | 22 | 25 | 54 |
| sys | 0 | 0 | 13 | 13 | 0 | 0 | 17 | 17 |
| vault | 0 | 0 | 11 | 11 | 0 | 0 | 12 | 12 |
| workspaces | 1 | 16 | 6 | 23 | 7 | 24 | 6 | 37 |
| ws | 0 | 0 | 4 | 4 | 0 | 0 | 4 | 4 |
| **Total** | **19** | **116** | **348** | **483** | **61** | **179** | **389** | **629** |

### admin

| l. | Chemin | Opérations → handler (classe, loc/ev) | Classe | Motif |
|---|---|---|---|---|
| 1404 | `/api/admin/analyze-runner-feedback` | `POST` note_handlers::analyze_runner_feedback (C, 12/0) | **C** | maintenance |
| 1408 | `/api/admin/generate-feedback-rfcs` | `POST` note_handlers::generate_feedback_rfcs (C, 21/0) | **C** | maintenance |
| 1412 | `/api/admin/apply-feedback-rfc` | `POST` note_handlers::apply_feedback_rfc (C, 17/0) | **C** | maintenance |
| 1792 | `/api/admin/backfill-embeddings` | `POST` note_handlers::start_backfill_embeddings (C, 70/0)<br>`DELETE` note_handlers::cancel_backfill_embeddings (C, 20/0) | **C** | maintenance |
| 1796 | `/api/admin/backfill-embeddings/status` | `GET` note_handlers::get_backfill_embeddings_status (C, 6/0) | **C** | maintenance |
| 1804 | `/api/admin/backfill-synapses` | `POST` note_handlers::start_backfill_synapses (C, 83/0)<br>`DELETE` note_handlers::cancel_backfill_synapses (C, 20/0) | **C** | maintenance |
| 1808 | `/api/admin/backfill-synapses/status` | `GET` note_handlers::get_backfill_synapses_status (C, 6/0) | **C** | maintenance |
| 1815 | `/api/admin/reindex-decisions` | `POST` handlers::reindex_decisions (C, 14/0) | **C** | maintenance |
| 1819 | `/api/admin/backfill-decision-embeddings` | `POST` handlers::backfill_decision_embeddings (C, 14/0) | **C** | maintenance |
| 1823 | `/api/admin/backfill-decision-project-slugs` | `POST` handlers::backfill_decision_project_slugs (C, 14/0) | **C** | maintenance |
| 1827 | `/api/admin/backfill-discussed` | `POST` handlers::backfill_discussed (C, 16/0) | **C** | maintenance |
| 1834 | `/api/meilisearch/stats` | `GET` handlers::get_meilisearch_stats (C, 10/0) | **C** | maintenance |
| 1838 | `/api/meilisearch/orphans` | `DELETE` handlers::delete_meilisearch_orphans (C, 14/0) | **C** | maintenance |
| 1845 | `/api/admin/cleanup-cross-project-calls` | `POST` handlers::cleanup_cross_project_calls (C, 10/0) | **C** | maintenance |
| 1849 | `/api/admin/cleanup-builtin-calls` | `POST` handlers::cleanup_builtin_calls (C, 6/0) | **C** | maintenance |
| 1853 | `/api/admin/migrate-calls-confidence` | `POST` handlers::migrate_calls_confidence (C, 10/0) | **C** | maintenance |
| 1857 | `/api/admin/cleanup-sync-data` | `POST` handlers::cleanup_sync_data (C, 13/0) | **C** | maintenance |
| 1864 | `/api/admin/update-fabric-scores` | `POST` handlers::update_fabric_scores (C, 109/4) | **C** | maintenance |
| 1868 | `/api/admin/bootstrap-knowledge-fabric` | `POST` handlers::bootstrap_knowledge_fabric (C, 178/2) | **C** | maintenance |
| 1869 | `/api/admin/audit-gaps` | `POST` handlers::audit_gaps (C, 24/0) | **C** | maintenance |
| 1873 | `/api/admin/persist-health-report` | `POST` handlers::persist_health_report (C, 276/0) | **C** | maintenance |
| 1877 | `/api/admin/reinforce-isomorphic` | `POST` handlers::reinforce_isomorphic_synapses (C, 85/2) | **C** | maintenance |
| 1878 | `/api/admin/detect-skills` | `POST` handlers::detect_skills (C, 66/0) | **C** | maintenance |
| 1882 | `/api/admin/detect-skill-fission` | `POST` handlers::detect_skill_fission (C, 30/0) | **C** | maintenance |
| 1886 | `/api/admin/detect-skill-fusion` | `POST` handlers::detect_skill_fusion (C, 30/0) | **C** | maintenance |
| 1890 | `/api/admin/auto-anchor-notes` | `POST` handlers::auto_anchor_notes (C, 33/0) | **C** | maintenance |
| 1894 | `/api/admin/reconstruct-knowledge` | `POST` handlers::reconstruct_knowledge (C, 88/0) | **C** | maintenance |
| 1898 | `/api/admin/skill-maintenance` | `POST` handlers::skill_maintenance (C, 61/0) | **C** | maintenance |
| 1902 | `/api/admin/detect-stagnation/{project_id}` | `GET` handlers::detect_stagnation (C, 21/0) | **C** | maintenance |
| 1906 | `/api/admin/deep-maintenance/{project_id}` | `POST` handlers::run_deep_maintenance (C, 30/0) | **C** | maintenance |
| 1910 | `/api/admin/learning-metrics/{project_id}` | `GET` handlers::get_learning_metrics (C, 19/0) | **C** | maintenance |
| 1914 | `/api/admin/seed-prompt-fragments/{project_id}` | `POST` handlers::seed_prompt_fragments (C, 19/0) | **C** | maintenance |

### chat

| l. | Chemin | Opérations → handler (classe, loc/ev) | Classe | Motif |
|---|---|---|---|---|
| 360 | `/api/chat/models` | `GET` chat_handlers::get_model_catalog (C, 5/0) | **C** | lecture calculée |
| 933 | `/api/agents/live` | `GET` attention_aggregate::get_live_agents (C, 11/0) | **C** | lecture calculée |
| 2075 | `/api/chat/sessions` | `GET` chat_handlers::list_sessions (C, 102/0)<br>`POST` chat_handlers::create_session (B, 183/1) | **C** | lecture calculée |
| 2076 | `/api/chat/search` | `GET` chat_handlers::search_messages (C, 22/0) | **C** | recherche/calcul |
| 2084 | `/api/chat/live-activity` | `GET` chat_handlers::get_live_activity (C, 18/0) | **C** | lecture calculée |
| 2088 | `/api/chat/sessions/backfill-previews` | `POST` chat_handlers::backfill_previews (C, 50/0) | **C** | verbe applicatif |
| 2094 | `/api/chat/sessions/{id}` | `GET` chat_handlers::get_session (B, 40/0)<br>`PATCH` chat_handlers::update_session (B, 31/1)<br>`DELETE` chat_handlers::delete_session (B, 34/1) | **B** | CRUD+logique, CRUD+logique (événement) |
| 2100 | `/api/chat/sessions/{id}/close` | `POST` chat_handlers::close_session (C, 23/1) | **C** | verbe applicatif |
| 2104 | `/api/chat/sessions/{id}/messages` | `GET` chat_handlers::list_messages (B, 63/0)<br>`POST` chat_handlers::send_session_message (B, 57/0) | **B** | imbriqué (lien)+logique |
| 2110 | `/api/chat/sessions/{id}/permissions/{request_id}` | `POST` chat_handlers::respond_permission (C, 111/0) | **C** | autre |
| 2115 | `/api/chat/permission-rules` | `GET` chat_handlers::list_permission_rules (B, 46/0) | **B** | CRUD+logique |
| 2119 | `/api/chat/permission-rules/{id}` | `DELETE` chat_handlers::revoke_permission_rule (A, 24/0) | **A** | CRUD pur |
| 2123 | `/api/chat/sessions/{id}/children` | `GET` chat_handlers::get_session_children (C, 16/0) | **C** | lecture calculée |
| 2127 | `/api/chat/sessions/{id}/tree` | `GET` chat_handlers::get_session_tree (C, 45/0) | **C** | lecture calculée |
| 2132 | `/api/chat/sessions/{id}/usage` | `GET` chat_handlers::get_session_usage (C, 40/0) | **C** | lecture calculée |
| 2136 | `/api/chat/runs/{run_id}/costs` | `GET` chat_handlers::get_run_costs (C, 12/0) | **C** | lecture calculée |
| 2140 | `/api/chat/runs/{run_id}/sessions` | `GET` chat_handlers::get_run_sessions (C, 12/0) | **C** | lecture calculée |
| 2145 | `/api/chat/sessions/{id}/discussed` | `POST` chat_handlers::add_discussed (B, 24/0)<br>`GET` chat_handlers::get_session_entities (B, 19/0) | **B** | imbriqué (lien) |
| 2151 | `/api/chat/sessions/{id}/anchors` | `GET` chat_handlers::anchors::list_anchors (B, 921/0)<br>`POST` chat_handlers::anchors::add_anchor (B, 903/0) | **B** | imbriqué (lien)+logique |
| 2156 | `/api/chat/sessions/{id}/anchors/{anchor_id}` | `PATCH` chat_handlers::anchors::update_anchor (B, 859/0)<br>`DELETE` chat_handlers::anchors::remove_anchor (B, 818/0) | **B** | imbriqué (lien)+logique |
| 2161 | `/api/chat/anchored-sessions` | `GET` chat_handlers::anchors::list_anchored_sessions (C, 793/0) | **C** | recherche/calcul |
| 2166 | `/api/chat/sessions/{id}/associate` | `POST` chat_handlers::associate_session (C, 31/0) | **C** | verbe applicatif |
| 2174 | `/api/chat/sessions/{id}/interrupt` | `POST` chat_handlers::interrupt_session (C, 76/0) | **C** | verbe applicatif |
| 2180 | `/api/chat/sessions/{id}/routing` | `PUT` chat_handlers::set_session_routing (C, 70/1) | **C** | singleton/config |
| 2186 | `/api/chat/sessions/{id}/switch-provider` | `POST` chat_handlers::switch_provider (C, 37/0) | **C** | verbe applicatif |
| 2193 | `/api/chat/sessions/{id}/cancel-tools` | `POST` chat_handlers::cancel_tools (C, 18/0) | **C** | verbe applicatif |
| 2201 | `/api/chat/sessions/{id}/background-tasks` | `GET` chat_handlers::get_background_tasks (B, 14/0) | **B** | imbriqué (lien) |
| 2205 | `/api/chat/sessions/{id}/cancel-task/{task_id}` | `POST` chat_handlers::cancel_task (C, 17/0) | **C** | autre |
| 2210 | `/api/chat/config/permissions` | `GET` chat_handlers::get_chat_permissions (C, 18/0)<br>`PUT` chat_handlers::update_chat_permissions (C, 16/0) | **C** | singleton/config |
| 2215 | `/api/chat/config` | `GET` chat_handlers::get_chat_config (C, 22/0)<br>`PATCH` chat_handlers::update_chat_config (C, 82/0) | **C** | singleton/config |
| 2217 | `/api/chat/detect-path` | `GET` chat_handlers::detect_path (C, 9/0) | **C** | lecture calculée |
| 2222 | `/api/chat/providers` | `GET` chat_handlers::list_providers (B, 101/0)<br>`POST` provider_handlers::create_provider (A, 9/0) | **B** | CRUD+logique |
| 2226 | `/api/chat/providers/test` | `POST` provider_handlers::test_provider (C, 8/0) | **C** | verbe applicatif |
| 2230 | `/api/chat/providers/ssh-host-key` | `POST` provider_handlers::scan_ssh_host_key (C, 8/0) | **C** | recherche/calcul |
| 2237 | `/api/chat/providers/{id}` | `GET` provider_handlers::get_provider (A, 8/0)<br>`PUT` provider_handlers::update_provider (A, 9/0)<br>`PATCH` provider_handlers::update_provider (A, 9/0)<br>`DELETE` provider_handlers::delete_provider (A, 13/0) | **A** | CRUD pur |
| 2241 | `/api/chat/providers/{id}/status` | `GET` provider_handlers::provider_status (C, 6/0) | **C** | lecture calculée |
| 2248 | `/api/chat/providers/{id}/limits` | `GET` provider_handlers::provider_limits (C, 27/0) | **C** | lecture calculée |
| 2255 | `/api/chat/provider-quotas` | `GET` quota_handlers::list_quotas (A, 6/0) | **A** | CRUD pur |
| 2261 | `/api/chat/providers/{id}/quota` | `GET` quota_handlers::get_quota (C, 12/0)<br>`PUT` quota_handlers::put_quota (C, 18/0)<br>`DELETE` quota_handlers::delete_quota (C, 11/0) | **C** | singleton/config |
| 2265 | `/api/chat/providers/{id}/models` | `GET` provider_handlers::provider_models (C, 6/0) | **C** | lecture calculée |
| 2276 | `/api/agent/providers` | `GET` provider_handlers::agent_list_providers (A, 7/0)<br>`POST` provider_handlers::agent_create_provider (A, 9/0) | **A** | CRUD pur |
| 2280 | `/api/agent/providers/test` | `POST` provider_handlers::agent_test_provider (C, 8/0) | **C** | verbe applicatif |
| 2284 | `/api/agent/providers/ssh-host-key` | `POST` provider_handlers::agent_scan_host_key (C, 8/0) | **C** | recherche/calcul |
| 2290 | `/api/agent/providers/{id}` | `GET` provider_handlers::agent_get_provider (A, 8/0)<br>`PATCH` provider_handlers::agent_update_provider (A, 9/0)<br>`DELETE` provider_handlers::agent_delete_provider (A, 13/0) | **A** | CRUD pur |
| 2294 | `/api/agent/providers/{id}/status` | `GET` provider_handlers::agent_provider_status (C, 8/0) | **C** | lecture calculée |
| 2298 | `/api/agent/providers/{id}/models` | `GET` provider_handlers::agent_provider_models (C, 8/0) | **C** | lecture calculée |
| 2302 | `/api/chat/send-journal` | `GET` provider_handlers::send_journal (A, 24/0) | **A** | CRUD pur |
| 2306 | `/api/chat/roles` | `GET` provider_handlers::get_roles (C, 5/0)<br>`PUT` provider_handlers::put_roles (C, 9/0) | **C** | singleton/config |
| 2310 | `/api/chat/model-aliases` | `GET` provider_handlers::get_aliases (C, 5/0)<br>`PUT` provider_handlers::put_aliases (C, 23/0) | **C** | singleton/config |
| 2314 | `/api/chat/model-policy` | `GET` provider_handlers::get_policy (C, 12/0)<br>`PUT` provider_handlers::put_policy (C, 17/0) | **C** | singleton/config |
| 2320 | `/api/chat/llm-consents` | `GET` provider_handlers::list_no_project_consents (C, 5/0)<br>`PUT` provider_handlers::allow_no_project_consent (C, 15/0) | **C** | singleton/config |
| 2325 | `/api/chat/llm-consents/{provider_id}` | `PUT` provider_handlers::allow_no_project_consent_for (A, 15/0)<br>`DELETE` provider_handlers::revoke_no_project_consent (A, 7/0) | **A** | CRUD pur |
| 2329 | `/api/projects/{slug}/llm-consent` | `GET` provider_handlers::list_consents (C, 6/0)<br>`PUT` provider_handlers::allow_consent (C, 16/0) | **C** | singleton/config |
| 2333 | `/api/projects/{slug}/llm-consents` | `GET` provider_handlers::list_consents (C, 6/0)<br>`PUT` provider_handlers::allow_consent (C, 16/0) | **C** | singleton/config |
| 2337 | `/api/projects/{slug}/llm-consent/{provider_id}` | `PUT` provider_handlers::allow_consent_for (B, 15/0)<br>`DELETE` provider_handlers::revoke_consent (B, 7/0) | **B** | imbriqué (lien) |
| 2341 | `/api/projects/{slug}/llm-consents/{provider_id}` | `PUT` provider_handlers::allow_consent_for (B, 15/0)<br>`DELETE` provider_handlers::revoke_consent (B, 7/0) | **B** | imbriqué (lien) |
| 2345 | `/api/projects/{slug}/llm-roles` | `GET` provider_handlers::get_project_roles (C, 8/0)<br>`PUT` provider_handlers::put_project_roles (C, 8/0) | **C** | singleton/config |
| 2351 | `/api/projects/{slug}/network-tools` | `GET` network_tools_handlers::get_network_tools (B, 22/0) | **B** | imbriqué (lien) |
| 2355 | `/api/projects/{slug}/network-tools/origins` | `PUT` network_tools_handlers::put_origin (C, 14/0)<br>`DELETE` network_tools_handlers::delete_origin (C, 27/0) | **C** | singleton/config |
| 2359 | `/api/projects/{slug}/network-tools/browser` | `PUT` network_tools_handlers::put_browser (C, 39/0) | **C** | singleton/config |
| 2365 | `/api/chat/search-engines` | `GET` network_tools_handlers::list_search_engines (C, 16/0)<br>`POST` network_tools_handlers::create_search_engine (C, 28/0)<br>`PUT` network_tools_handlers::put_search_engine_order (C, 48/0) | **C** | singleton/config |
| 2369 | `/api/chat/search-engines/{id}` | `DELETE` network_tools_handlers::delete_search_engine (A, 28/0) | **A** | CRUD pur |
| 2374 | `/api/chat/auto-models` | `GET` routing_handlers::get_auto_models (C, 12/0)<br>`PUT` routing_handlers::put_auto_models (C, 21/0) | **C** | singleton/config |
| 2382 | `/api/chat/delegation-limits` | `GET` routing_handlers::get_delegation_limits (C, 6/0)<br>`PUT` routing_handlers::put_delegation_limits (C, 20/0)<br>`DELETE` routing_handlers::delete_delegation_limits (C, 12/0) | **C** | singleton/config |
| 2390 | `/api/chat/deferred-tools` | `GET` routing_handlers::get_deferred_tools (C, 8/0)<br>`PUT` routing_handlers::put_deferred_tools (C, 14/0)<br>`DELETE` routing_handlers::delete_deferred_tools (C, 12/0) | **C** | singleton/config |
| 2393 | `/api/chat/team` | `GET` team_handlers::get_team (B, 32/0) | **B** | CRUD+logique |
| 2397 | `/api/chat/team/message` | `POST` team_handlers::post_team_message (B, 39/0) | **B** | CRUD+logique |
| 2402 | `/api/chat/routing` | `GET` routing_handlers::get_routing (C, 5/0)<br>`PUT` routing_handlers::put_routing (C, 8/0) | **C** | singleton/config |
| 2406 | `/api/chat/routing/decisions` | `GET` routing_handlers::list_decisions (C, 39/0) | **C** | lecture calculée |
| 2410 | `/api/chat/routing/report` | `GET` routing_handlers::get_report (C, 10/0) | **C** | lecture calculée |
| 2416 | `/api/projects/{slug}/routing` | `GET` routing_handlers::get_project_routing (C, 6/0)<br>`PUT` routing_handlers::put_project_routing (C, 9/0)<br>`DELETE` routing_handlers::delete_project_routing (C, 20/0) | **C** | singleton/config |
| 2417 | `/api/chat/cli/status` | `GET` chat_handlers::get_cli_status (C, 3/0) | **C** | lecture calculée |
| 2418 | `/api/chat/cli/install` | `POST` chat_handlers::install_cli (C, 32/0) | **C** | verbe applicatif |
| 2423 | `/api/chat/cli/auth-status` | `GET` chat_handlers::get_cli_auth_status (C, 3/0) | **C** | lecture calculée |

### code

| l. | Chemin | Opérations → handler (classe, loc/ev) | Classe | Motif |
|---|---|---|---|---|
| 1072 | `/api/code/search` | `GET` code_handlers::search_code (C, 74/0) | **C** | analyse code |
| 1077 | `/api/code/symbols/{*file_path}` | `GET` code_handlers::get_file_symbols (C, 64/0) | **C** | analyse code |
| 1079 | `/api/code/references` | `GET` code_handlers::find_references (C, 54/0) | **C** | analyse code |
| 1084 | `/api/code/dependencies/{*file_path}` | `GET` code_handlers::get_file_dependencies (C, 42/0) | **C** | analyse code |
| 1086 | `/api/code/callgraph` | `GET` code_handlers::get_call_graph (C, 85/0) | **C** | analyse code |
| 1088 | `/api/code/impact` | `GET` code_handlers::analyze_impact (C, 343/0) | **C** | analyse code |
| 1093 | `/api/code/impact/multi` | `GET` code_handlers::analyze_impact_v2 (C, 170/0) | **C** | analyse code |
| 1098 | `/api/code/architecture` | `GET` code_handlers::get_architecture (C, 141/0) | **C** | analyse code |
| 1100 | `/api/code/similar` | `POST` code_handlers::find_similar_code (C, 23/0) | **C** | analyse code |
| 1105 | `/api/code/trait-impls` | `GET` code_handlers::find_trait_implementations (C, 36/0) | **C** | analyse code |
| 1109 | `/api/code/type-traits` | `GET` code_handlers::find_type_traits (C, 26/0) | **C** | analyse code |
| 1110 | `/api/code/impl-blocks` | `GET` code_handlers::get_impl_blocks (C, 26/0) | **C** | analyse code |
| 1117 | `/api/feature-graphs` | `GET` code_handlers::list_feature_graphs (A, 28/0)<br>`POST` code_handlers::create_feature_graph (B, 35/2) | **B** | CRUD+logique (événement) |
| 1121 | `/api/feature-graphs/auto-build` | `POST` code_handlers::auto_build_feature_graph (C, 53/0) | **C** | verbe applicatif |
| 1125 | `/api/feature-graphs/{id}` | `GET` code_handlers::get_feature_graph (B, 45/0)<br>`DELETE` code_handlers::delete_feature_graph (B, 16/2) | **B** | CRUD+logique, CRUD+logique (événement) |
| 1129 | `/api/feature-graphs/{id}/entities` | `POST` code_handlers::add_entity_to_feature_graph (B, 32/0) | **B** | imbriqué (lien)+logique |
| 1133 | `/api/feature-graphs/{id}/statistics` | `GET` code_handlers::get_feature_graph_statistics (C, 19/0) | **C** | lecture calculée |
| 1137 | `/api/feature-graphs/{id}/overlapping` | `GET` code_handlers::find_overlapping_feature_graphs (B, 14/0) | **B** | imbriqué (lien) |
| 1141 | `/api/feature-graphs/compare` | `GET` code_handlers::compare_feature_graphs (C, 19/0) | **C** | recherche/calcul |
| 1148 | `/api/code/communities` | `GET` code_handlers::get_code_communities (C, 55/0) | **C** | analyse code |
| 1149 | `/api/code/health` | `GET` code_handlers::get_code_health (C, 205/0) | **C** | analyse code |
| 1153 | `/api/code/node-importance` | `GET` code_handlers::get_node_importance (C, 100/0) | **C** | analyse code |
| 1158 | `/api/code/hotspots` | `GET` code_handlers::get_change_hotspots (C, 28/0) | **C** | analyse code |
| 1162 | `/api/code/knowledge-gaps` | `GET` code_handlers::get_knowledge_gaps (C, 36/0) | **C** | analyse code |
| 1166 | `/api/code/risk-assessment` | `GET` code_handlers::get_risk_assessment (C, 47/0) | **C** | analyse code |
| 1167 | `/api/code/homeostasis` | `GET` code_handlers::get_homeostasis (C, 21/0) | **C** | analyse code |
| 1171 | `/api/code/structural-drift` | `GET` code_handlers::get_structural_drift (C, 25/0) | **C** | analyse code |
| 1178 | `/api/code/class-hierarchy` | `GET` code_handlers::get_class_hierarchy (C, 13/0) | **C** | analyse code |
| 1179 | `/api/code/subclasses` | `GET` code_handlers::find_subclasses (C, 16/0) | **C** | analyse code |
| 1183 | `/api/code/interface-implementors` | `GET` code_handlers::find_interface_implementors (C, 16/0) | **C** | analyse code |
| 1190 | `/api/code/processes/detect` | `POST` code_handlers::detect_processes (C, 38/0) | **C** | analyse code |
| 1191 | `/api/code/processes` | `GET` code_handlers::list_processes (C, 24/0) | **C** | analyse code |
| 1195 | `/api/code/processes/detail` | `GET` code_handlers::get_process_detail (C, 13/0) | **C** | analyse code |
| 1199 | `/api/code/entry-points` | `GET` code_handlers::get_entry_points (C, 25/0) | **C** | analyse code |
| 1206 | `/api/code/communities/enrich` | `POST` code_handlers::enrich_communities (C, 38/0) | **C** | analyse code |
| 1210 | `/api/code/bridge` | `GET` code_handlers::get_bridge (C, 152/0) | **C** | analyse code |
| 1217 | `/api/code/topology/check` | `GET` code_handlers::check_topology (C, 41/0) | **C** | analyse code |
| 1221 | `/api/code/topology/rules` | `GET` code_handlers::list_topology_rules (C, 18/0)<br>`POST` code_handlers::create_topology_rule (C, 55/2) | **C** | analyse code |
| 1225 | `/api/code/topology/rules/{rule_id}` | `DELETE` code_handlers::delete_topology_rule (C, 13/2) | **C** | analyse code |
| 1229 | `/api/code/topology/check-file` | `POST` code_handlers::check_file_topology (C, 24/0) | **C** | analyse code |
| 1236 | `/api/code/structural-profile` | `POST` code_handlers::get_structural_profile (C, 43/0) | **C** | analyse code |
| 1240 | `/api/code/structural-twins` | `POST` code_handlers::find_structural_twins (C, 167/0) | **C** | analyse code |
| 1244 | `/api/code/structural-clusters` | `POST` code_handlers::cluster_dna (C, 40/0) | **C** | analyse code |
| 1248 | `/api/code/structural-twins/cross-project` | `POST` code_handlers::find_cross_project_twins (C, 164/0) | **C** | analyse code |
| 1255 | `/api/code/predict-links` | `POST` code_handlers::predict_missing_links (C, 82/0) | **C** | analyse code |
| 1259 | `/api/code/link-plausibility` | `POST` code_handlers::check_link_plausibility (C, 49/0) | **C** | analyse code |
| 1266 | `/api/code/stress-test-node` | `POST` code_handlers::stress_test_node (C, 26/0) | **C** | analyse code |
| 1270 | `/api/code/stress-test-edge` | `POST` code_handlers::stress_test_edge (C, 26/0) | **C** | analyse code |
| 1274 | `/api/code/stress-test-cascade` | `POST` code_handlers::stress_test_cascade (C, 28/0) | **C** | analyse code |
| 1275 | `/api/code/find-bridges` | `POST` code_handlers::find_bridges (C, 24/0) | **C** | analyse code |
| 1282 | `/api/code/context-card` | `GET` code_handlers::get_context_card (C, 34/0) | **C** | analyse code |
| 1286 | `/api/code/context-cards/refresh` | `POST` code_handlers::refresh_context_cards (C, 20/0) | **C** | analyse code |
| 1290 | `/api/code/fingerprint` | `GET` code_handlers::get_fingerprint (C, 61/0) | **C** | analyse code |
| 1291 | `/api/code/isomorphic` | `GET` code_handlers::find_isomorphic (C, 25/0) | **C** | analyse code |
| 1295 | `/api/code/structural-templates` | `GET` code_handlers::suggest_structural_templates (C, 77/0) | **C** | analyse code |
| 1302 | `/api/code/plan-implementation` | `POST` code_handlers::plan_implementation (C, 39/0) | **C** | analyse code |

### environments

| l. | Chemin | Opérations → handler (classe, loc/ev) | Classe | Motif |
|---|---|---|---|---|
| 645 | `/api/projects/{project_id}/environments` | `GET` environment_handlers::list_environments (B, 11/0)<br>`POST` environment_handlers::create_environment (B, 37/2) | **B** | imbriqué (lien), imbriqué (lien)+logique |
| 649 | `/api/projects/{project_id}/deployment-matrix` | `GET` environment_handlers::get_deployment_matrix (C, 12/0) | **C** | lecture calculée |
| 655 | `/api/environments/{id}` | `GET` environment_handlers::get_environment (A, 6/0)<br>`PATCH` environment_handlers::update_environment (B, 30/2)<br>`DELETE` environment_handlers::delete_environment (B, 13/2) | **B** | CRUD+logique (événement) |
| 660 | `/api/environments/{id}/deployments` | `GET` environment_handlers::list_deployments (B, 20/0)<br>`POST` environment_handlers::create_deployment (B, 44/2) | **B** | imbriqué (lien), imbriqué (lien)+logique |
| 664 | `/api/deployments/{id}` | `PATCH` environment_handlers::update_deployment (B, 55/4) | **B** | CRUD+logique (événement) |

### graph

| l. | Chemin | Opérations → handler (classe, loc/ev) | Classe | Motif |
|---|---|---|---|---|
| 492 | `/api/projects/{slug}/graph` | `GET` project_handlers::get_project_graph (C, 42/0) | **C** | graphe calculé |
| 496 | `/api/projects/{slug}/intelligence/summary` | `GET` project_handlers::get_intelligence_summary (C, 15/0) | **C** | graphe calculé |
| 500 | `/api/projects/{slug}/embeddings/projection` | `GET` project_handlers::get_embeddings_projection (C, 207/0) | **C** | graphe calculé |
| 1480 | `/api/graph/transverse` | `GET` project_handlers::get_transverse_graph (C, 8/0) | **C** | graphe calculé |
| 1484 | `/api/graph/neighborhood` | `GET` graph_handlers::get_neighborhood (C, 15/0) | **C** | graphe calculé |
| 1486 | `/api/graph/schema` | `GET` graph_handlers::get_schema_catalog (C, 5/0) | **C** | graphe calculé |
| 1487 | `/api/graph/path` | `GET` graph_handlers::get_typed_path (C, 31/0) | **C** | graphe calculé |
| 2064 | `/api/workspaces/{slug}/graph` | `GET` workspace_handlers::get_workspace_graph (C, 176/0) | **C** | graphe calculé |
| 2068 | `/api/workspaces/{slug}/intelligence/summary` | `GET` workspace_handlers::get_workspace_intelligence_summary (C, 246/0) | **C** | graphe calculé |

### hooks

| l. | Chemin | Opérations → handler (classe, loc/ev) | Classe | Motif |
|---|---|---|---|---|
| 408 | `/api/hooks/activate` | `POST` hook_handlers::activate_hook (C, 158/0) | **C** | proto/auth/sys |
| 409 | `/api/hooks/health` | `GET` hook_handlers::hooks_health (C, 28/0) | **C** | proto/auth/sys |
| 413 | `/api/hooks/session-context` | `GET` hook_handlers::session_context (C, 128/0) | **C** | proto/auth/sys |
| 417 | `/api/hooks/resolve-project` | `GET` hook_handlers::resolve_project (C, 37/0) | **C** | proto/auth/sys |
| 344 | `/hooks/events/{source}/{trigger_id}` | `POST` ingress_handlers::receive_external_event (C, 31/0) | **C** | proto/auth/sys |

### mcp-federation

| l. | Chemin | Opérations → handler (classe, loc/ev) | Classe | Motif |
|---|---|---|---|---|
| 2431 | `/api/mcp-federation/servers` | `GET` mcp_federation_handlers::list_servers (C, 10/0)<br>`POST` mcp_federation_handlers::connect_server (C, 50/0) | **C** | connexion vivante |
| 2436 | `/api/mcp-federation/servers/{id}` | `GET` mcp_federation_handlers::get_server_status (C, 18/0)<br>`DELETE` mcp_federation_handlers::disconnect_server (C, 29/0) | **C** | connexion vivante |
| 2440 | `/api/mcp-federation/servers/{id}/tools` | `GET` mcp_federation_handlers::list_server_tools (C, 22/0) | **C** | connexion vivante |
| 2444 | `/api/mcp-federation/servers/{id}/probe` | `POST` mcp_federation_handlers::probe_server (C, 31/0) | **C** | connexion vivante |
| 2448 | `/api/mcp-federation/servers/{id}/reconnect` | `POST` mcp_federation_handlers::reconnect_server (C, 41/0) | **C** | connexion vivante |

### neural/trajectory/feedback

| l. | Chemin | Opérations → handler (classe, loc/ev) | Classe | Motif |
|---|---|---|---|---|
| 567 | `/api/neural-routing/status` | `GET` neural_routing_handlers::get_status (C, 19/0) | **C** | lecture calculée |
| 571 | `/api/neural-routing/config` | `GET` neural_routing_handlers::get_config (C, 8/0)<br>`PUT` neural_routing_handlers::update_config (C, 98/0) | **C** | singleton/config |
| 575 | `/api/neural-routing/enable` | `POST` neural_routing_handlers::enable (C, 15/0) | **C** | verbe applicatif |
| 579 | `/api/neural-routing/disable` | `POST` neural_routing_handlers::disable (C, 20/0) | **C** | verbe applicatif |
| 583 | `/api/neural-routing/mode` | `PUT` neural_routing_handlers::set_mode (C, 31/0) | **C** | singleton/config |
| 590 | `/api/trajectories` | `GET` trajectory_handlers::list_trajectories (A, 29/0) | **A** | CRUD pur |
| 594 | `/api/trajectories/stats` | `GET` trajectory_handlers::get_stats (C, 12/0) | **C** | lecture calculée |
| 598 | `/api/trajectories/similar` | `POST` trajectory_handlers::search_similar (C, 38/0) | **C** | recherche/calcul |
| 602 | `/api/trajectories/{id}` | `GET` trajectory_handlers::get_trajectory (A, 14/0) | **A** | CRUD pur |
| 1309 | `/api/feedback` | `GET` feedback_handlers::list_feedback (A, 0/0)<br>`POST` feedback_handlers::create_feedback (C, 0/0) | **C** | verbe applicatif |
| 1313 | `/api/feedback/stats` | `GET` feedback_handlers::get_feedback_stats (C, 0/0) | **C** | lecture calculée |
| 1328 | `/api/documents` | `GET` document_handlers::list_documents (B, 41/0)<br>`POST` document_handlers::upload_document (A, 22/0) | **B** | CRUD+logique |
| 1332 | `/api/documents/{document_id}` | `GET` document_handlers::get_document (A, 7/0)<br>`DELETE` document_handlers::delete_document (A, 17/0) | **A** | CRUD pur |
| 1336 | `/api/documents/{document_id}/chunks` | `GET` document_handlers::get_document_chunks (C, 18/0) | **C** | lecture calculée |
| 1340 | `/api/documents/{document_id}/raw` | `GET` document_handlers::get_document_raw (C, 32/0) | **C** | lecture calculée |

### notes/knowledge

| l. | Chemin | Opérations → handler (classe, loc/ev) | Classe | Motif |
|---|---|---|---|---|
| 1348 | `/api/notes` | `GET` note_handlers::list_notes (A, 20/0)<br>`POST` note_handlers::create_note (B, 183/0) | **B** | CRUD+logique |
| 1353 | `/api/knowledge/pinned-notes` | `GET` note_handlers::get_pinned_notes (C, 16/0)<br>`PUT` note_handlers::put_pinned_notes (C, 74/0) | **C** | singleton/config |
| 1359 | `/api/knowledge/self-core` | `GET` note_handlers::get_self_core (C, 111/0) | **C** | lecture calculée |
| 1365 | `/api/notes/{note_id}` | `GET` note_handlers::get_note (A, 13/0)<br>`PATCH` note_handlers::update_note (A, 21/0)<br>`DELETE` note_handlers::delete_note (A, 16/0) | **A** | CRUD pur |
| 1369 | `/api/notes/{note_id}/scope` | `POST` note_handlers::rescope_note (C, 17/0) | **C** | verbe applicatif |
| 1371 | `/api/notes/search` | `GET` note_handlers::search_notes (C, 33/0) | **C** | recherche/calcul |
| 1375 | `/api/notes/search-semantic` | `GET` note_handlers::search_notes_semantic (C, 59/0) | **C** | recherche/calcul |
| 1379 | `/api/notes/neurons/search` | `GET` note_handlers::search_neurons (C, 180/8) | **C** | recherche/calcul |
| 1383 | `/api/notes/neurons/reinforce` | `POST` note_handlers::reinforce_neurons (C, 63/2) | **C** | recherche/calcul |
| 1387 | `/api/notes/neurons/decay` | `POST` note_handlers::decay_synapses (C, 43/2) | **C** | recherche/calcul |
| 1391 | `/api/notes/neurons/heal-scars` | `POST` note_handlers::heal_scars (C, 16/0) | **C** | recherche/calcul |
| 1395 | `/api/notes/consolidate-memory` | `POST` note_handlers::consolidate_memory (C, 16/0) | **C** | verbe applicatif |
| 1399 | `/api/notes/purge-auto-observed` | `POST` note_handlers::purge_auto_observed (C, 13/0) | **C** | recherche/calcul |
| 1417 | `/api/notes/needs-review` | `GET` note_handlers::get_notes_needing_review (C, 15/0) | **C** | lecture calculée |
| 1422 | `/api/notes/update-staleness` | `POST` note_handlers::update_staleness_scores (C, 13/0) | **C** | recherche/calcul |
| 1427 | `/api/notes/update-energy` | `POST` note_handlers::update_energy_scores (C, 17/0) | **C** | recherche/calcul |
| 1432 | `/api/projects/{project_id}/notes` | `GET` note_handlers::list_project_notes (B, 21/0) | **B** | imbriqué (lien) |
| 1437 | `/api/notes/{note_id}/confirm` | `POST` note_handlers::confirm_note (C, 25/0) | **C** | verbe applicatif |
| 1441 | `/api/notes/{note_id}/invalidate` | `POST` note_handlers::invalidate_note (C, 20/0) | **C** | verbe applicatif |
| 1445 | `/api/notes/{note_id}/supersede` | `POST` note_handlers::supersede_note (C, 32/0) | **C** | verbe applicatif |
| 1450 | `/api/notes/{note_id}/links` | `POST` note_handlers::link_note_to_entity (B, 40/0) | **B** | imbriqué (lien)+logique |
| 1454 | `/api/notes/{note_id}/links/{entity_type}/{entity_id}` | `DELETE` note_handlers::unlink_note_from_entity (B, 16/0) | **B** | imbriqué (lien) |
| 1456 | `/api/notes/context` | `GET` note_handlers::get_context_notes (C, 22/0) | **C** | lecture calculée |
| 1460 | `/api/notes/propagated` | `GET` note_handlers::get_propagated_notes (C, 35/0) | **C** | lecture calculée |
| 1465 | `/api/notes/context-knowledge` | `GET` note_handlers::get_context_knowledge (C, 22/0) | **C** | lecture calculée |
| 1470 | `/api/notes/propagated-knowledge` | `GET` note_handlers::get_propagated_knowledge (C, 34/0) | **C** | lecture calculée |
| 1475 | `/api/entities/{entity_type}/{entity_id}/notes` | `GET` note_handlers::get_entity_notes (B, 16/0) | **B** | imbriqué (lien) |
| 1659 | `/api/refs/search` | `GET` refs_handlers::search_refs (C, 19/0) | **C** | recherche/calcul |
| 1660 | `/api/refs/kinds` | `GET` refs_handlers::ref_kinds (C, 3/0) | **C** | recherche/calcul |

### oauth/auth

| l. | Chemin | Opérations → handler (classe, loc/ev) | Classe | Motif |
|---|---|---|---|---|
| 307 | `/.well-known/oauth-authorization-server` | `GET` auth::oauth_server::authorization_server_metadata (C, 15/0) | **C** | proto/auth/sys |
| 311 | `/.well-known/oauth-protected-resource` | `GET` auth::oauth_server::protected_resource_metadata (C, 10/0) | **C** | proto/auth/sys |
| 315 | `/oauth/register` | `POST` auth::oauth_server::register_client (C, 50/0) | **C** | proto/auth/sys |
| 320 | `/oauth/authorize` | `GET` auth::oauth_server::authorize (C, 118/0)<br>`POST` auth::oauth_server::authorize_consent (C, 82/0) | **C** | proto/auth/sys |
| 321 | `/oauth/token` | `POST` auth::oauth_server::token (C, 93/0) | **C** | proto/auth/sys |
| 379 | `/auth/providers` | `GET` auth_handlers::get_auth_providers (C, 40/0) | **C** | proto/auth/sys |
| 380 | `/auth/login` | `POST` auth_handlers::password_login (C, 98/0) | **C** | proto/auth/sys |
| 381 | `/auth/register` | `POST` auth_handlers::register (C, 61/0) | **C** | proto/auth/sys |
| 386 | `/auth/oidc` | `GET` auth_handlers::oidc_login (C, 19/0) | **C** | proto/auth/sys |
| 387 | `/auth/oidc/callback` | `POST` auth_handlers::oidc_callback (C, 73/0) | **C** | proto/auth/sys |
| 390 | `/auth/google` | `GET` auth_handlers::google_login (C, 21/0) | **C** | proto/auth/sys |
| 394 | `/auth/google/callback` | `POST` auth_handlers::google_callback (C, 69/0) | **C** | proto/auth/sys |
| 396 | `/auth/refresh` | `POST` auth_handlers::refresh_token (C, 74/0) | **C** | proto/auth/sys |
| 398 | `/auth/logout` | `POST` auth_handlers::logout (C, 39/0) | **C** | proto/auth/sys |
| 401 | `/auth/ws-ticket` | `POST` auth_handlers::ws_ticket (C, 55/0) | **C** | proto/auth/sys |
| 445 | `/auth/me` | `GET` auth_handlers::get_me (C, 29/0) | **C** | proto/auth/sys |
| 456 | `/auth/mcp-tokens` | `GET` auth_handlers::list_mcp_tokens (C, 11/0)<br>`POST` auth_handlers::create_mcp_token (C, 60/0)<br>`DELETE` auth_handlers::purge_mcp_tokens (C, 13/0) | **C** | proto/auth/sys |
| 460 | `/auth/mcp-tokens/{jti}` | `DELETE` auth_handlers::revoke_mcp_token (C, 17/0) | **C** | proto/auth/sys |

### plans/tasks

| l. | Chemin | Opérations → handler (classe, loc/ev) | Classe | Motif |
|---|---|---|---|---|
| 484 | `/api/projects/{slug}/plans` | `GET` project_handlers::list_project_plans (B, 26/0) | **B** | imbriqué (lien) |
| 607 | `/api/projects/{project_id}/releases` | `GET` handlers::list_releases (B, 27/0)<br>`POST` handlers::create_release (B, 36/2) | **B** | imbriqué (lien), imbriqué (lien)+logique |
| 669 | `/api/projects/{project_id}/milestones` | `GET` handlers::list_milestones (B, 27/0)<br>`POST` handlers::create_milestone (B, 25/2) | **B** | imbriqué (lien), imbriqué (lien)+logique |
| 674 | `/api/projects/{project_id}/roadmap` | `GET` handlers::get_project_roadmap (C, 106/0) | **C** | lecture calculée |
| 681 | `/api/plans` | `GET` handlers::list_plans (B, 40/0)<br>`POST` handlers::create_plan (B, 17/2) | **B** | CRUD+logique, CRUD+logique (événement) |
| 687 | `/api/plans/{plan_id}` | `GET` handlers::get_plan (A, 12/0)<br>`PATCH` handlers::update_plan_status (B, 56/4)<br>`DELETE` handlers::delete_plan (B, 14/2) | **B** | CRUD+logique (événement) |
| 692 | `/api/plans/{plan_id}/project` | `PUT` handlers::link_plan_to_project (C, 11/0)<br>`DELETE` handlers::unlink_plan_from_project (C, 7/0) | **C** | singleton/config |
| 696 | `/api/plans/{plan_id}/next-task` | `GET` handlers::get_next_task (C, 11/0) | **C** | lecture calculée |
| 700 | `/api/plans/{plan_id}/dependency-graph` | `GET` handlers::get_plan_dependency_graph (C, 116/0) | **C** | lecture calculée |
| 704 | `/api/plans/{plan_id}/critical-path` | `GET` handlers::get_plan_critical_path (C, 13/0) | **C** | lecture calculée |
| 705 | `/api/plans/{plan_id}/waves` | `GET` handlers::get_plan_waves (C, 8/0) | **C** | lecture calculée |
| 739 | `/api/plans/{plan_id}/runs` | `GET` handlers::list_plan_runs (B, 16/0) | **B** | imbriqué (lien) |
| 789 | `/api/plans/{plan_id}/constraints` | `GET` handlers::get_plan_constraints (B, 11/0)<br>`POST` handlers::add_constraint (B, 28/2) | **B** | imbriqué (lien), imbriqué (lien)+logique |
| 795 | `/api/constraints/{constraint_id}` | `GET` handlers::get_constraint (A, 12/0)<br>`PATCH` handlers::update_constraint (B, 24/2)<br>`DELETE` handlers::delete_constraint (B, 14/2) | **B** | CRUD+logique (événement) |
| 800 | `/api/lifecycle-hooks` | `GET` handlers::list_lifecycle_hooks (A, 15/0)<br>`POST` handlers::create_lifecycle_hook (B, 35/2) | **B** | CRUD+logique (événement) |
| 806 | `/api/lifecycle-hooks/{hook_id}` | `GET` handlers::get_lifecycle_hook (A, 13/0)<br>`PATCH` handlers::update_lifecycle_hook (B, 21/2)<br>`DELETE` handlers::delete_lifecycle_hook (B, 25/2) | **B** | CRUD+logique (événement) |
| 811 | `/api/plans/{plan_id}/sessions` | `GET` chat_handlers::get_plan_sessions (C, 12/0) | **C** | lecture calculée |
| 813 | `/api/tasks` | `GET` handlers::list_all_tasks (B, 32/0) | **B** | CRUD+logique |
| 815 | `/api/plans/{plan_id}/tasks` | `POST` handlers::add_task (B, 18/2) | **B** | imbriqué (lien)+logique |
| 821 | `/api/tasks/{task_id}` | `GET` handlers::get_task (A, 24/0)<br>`PATCH` handlers::update_task (B, 143/4)<br>`DELETE` handlers::delete_task (B, 14/2) | **B** | CRUD+logique (événement) |
| 826 | `/api/day-plan` | `GET` handlers::get_day_plan (C, 17/0)<br>`PUT` handlers::set_day_plan (C, 19/2) | **C** | singleton/config |
| 830 | `/api/day-plan/tasks/{task_id}` | `POST` handlers::add_to_day (C, 21/2)<br>`DELETE` handlers::remove_from_day (B, 21/2) | **C** | autre |
| 835 | `/api/tasks/{task_id}/sessions` | `GET` chat_handlers::get_task_sessions (C, 12/0) | **C** | lecture calculée |
| 840 | `/api/tasks/{task_id}/dependencies` | `POST` handlers::add_task_dependencies (B, 13/0) | **B** | imbriqué (lien) |
| 844 | `/api/tasks/{task_id}/dependencies/{dep_id}` | `DELETE` handlers::remove_task_dependency (B, 10/0) | **B** | imbriqué (lien) |
| 848 | `/api/tasks/{task_id}/blockers` | `GET` handlers::get_task_blockers (C, 11/0) | **C** | lecture calculée |
| 852 | `/api/tasks/{task_id}/blocking` | `GET` handlers::get_tasks_blocked_by (C, 11/0) | **C** | lecture calculée |
| 857 | `/api/tasks/{task_id}/steps` | `GET` handlers::get_task_steps (B, 7/0)<br>`POST` handlers::add_step (B, 18/0) | **B** | imbriqué (lien) |
| 861 | `/api/tasks/{task_id}/steps/progress` | `GET` handlers::get_step_progress (C, 22/0) | **C** | lecture calculée |
| 867 | `/api/steps/{step_id}` | `GET` handlers::get_step (A, 12/0)<br>`PATCH` handlers::update_step (B, 42/0)<br>`DELETE` handlers::delete_step (B, 10/2) | **B** | CRUD+logique, CRUD+logique (événement) |
| 872 | `/api/plans/{plan_id}/tasks/{task_id}/context` | `GET` handlers::get_task_context (C, 11/0) | **C** | lecture calculée |
| 876 | `/api/plans/{plan_id}/tasks/{task_id}/prompt` | `GET` handlers::get_task_prompt (C, 15/0) | **C** | lecture calculée |
| 880 | `/api/plans/{plan_id}/tasks/{task_id}/build_prompt` | `POST` handlers::build_task_prompt (C, 19/0) | **C** | verbe applicatif |
| 906 | `/api/tasks/{task_id}/execution-places` | `GET` execution_place_handlers::list_execution_place_events (B, 12/0) | **B** | imbriqué (lien) |
| 908 | `/api/plans/{plan_id}/enrich` | `POST` handlers::enrich_plan (C, 11/0) | **C** | verbe applicatif |
| 912 | `/api/plans/{plan_id}/tasks/{task_id}/enrich` | `POST` handlers::enrich_task (C, 11/0) | **C** | verbe applicatif |
| 917 | `/api/tasks/{task_id}/decisions` | `POST` handlers::add_decision (B, 89/2) | **B** | imbriqué (lien)+logique |
| 922 | `/api/attention` | `GET` attention_aggregate::get_attention (A, 21/0) | **A** | CRUD pur |
| 928 | `/api/activity/snapshot` | `GET` activity_handlers::get_activity_snapshot (C, 259/1) | **C** | lecture calculée |
| 934 | `/api/retrieval-ledger` | `GET` handlers::get_retrieval_ledger (B, 38/0) | **B** | CRUD+logique |
| 938 | `/api/decisions` | `GET` handlers::list_decisions (C, 39/0)<br>`POST` handlers::add_decision_global (B, 66/2) | **C** | lecture calculée |
| 942 | `/api/decisions/affecting` | `GET` handlers::get_decisions_affecting (C, 15/0) | **C** | lecture calculée |
| 946 | `/api/decisions/timeline` | `GET` handlers::get_decision_timeline (C, 11/0) | **C** | lecture calculée |
| 952 | `/api/decisions/{decision_id}` | `GET` handlers::get_decision (A, 12/0)<br>`PATCH` handlers::update_decision (A, 17/0)<br>`DELETE` handlers::delete_decision (A, 7/0) | **A** | CRUD pur |
| 953 | `/api/decisions/search` | `GET` handlers::search_decisions (C, 49/0) | **C** | recherche/calcul |
| 957 | `/api/decisions/search-semantic` | `GET` handlers::search_decisions_semantic (C, 37/0) | **C** | recherche/calcul |
| 964 | `/api/decisions/{decision_id}/affects` | `POST` handlers::add_decision_affects (B, 29/2)<br>`GET` handlers::list_decision_affects (B, 11/0)<br>`DELETE` handlers::remove_decision_affects_query (C, 24/2) | **C** | autre |
| 968 | `/api/decisions/{decision_id}/affects/{entity_type}/{entity_id}` | `DELETE` handlers::remove_decision_affects (B, 23/2) | **B** | imbriqué (lien)+logique |
| 973 | `/api/decisions/{new_id}/supersedes/{old_id}` | `POST` handlers::supersede_decision (C, 11/0) | **C** | autre |
| 982 | `/api/releases/{release_id}` | `GET` handlers::get_release (A, 17/0)<br>`PATCH` handlers::update_release (B, 53/4)<br>`DELETE` handlers::delete_release (B, 12/2) | **B** | CRUD+logique (événement) |
| 986 | `/api/releases/{release_id}/tasks` | `POST` handlers::add_task_to_release (B, 18/2) | **B** | imbriqué (lien)+logique |
| 990 | `/api/releases/{release_id}/commits` | `POST` handlers::add_commit_to_release (B, 11/0) | **B** | imbriqué (lien) |
| 994 | `/api/releases/{release_id}/commits/{commit_sha}` | `DELETE` handlers::remove_commit_from_release (B, 10/0) | **B** | imbriqué (lien) |
| 1001 | `/api/milestones/{milestone_id}` | `GET` handlers::get_milestone (B, 110/0)<br>`PATCH` handlers::update_milestone (B, 53/4)<br>`DELETE` handlers::delete_milestone (B, 12/2) | **B** | CRUD+logique, CRUD+logique (événement) |
| 1005 | `/api/milestones/{milestone_id}/tasks` | `POST` handlers::add_task_to_milestone (B, 18/2) | **B** | imbriqué (lien)+logique |
| 1009 | `/api/milestones/{milestone_id}/plans` | `POST` handlers::link_plan_to_milestone (B, 18/2) | **B** | imbriqué (lien)+logique |
| 1013 | `/api/milestones/{milestone_id}/plans/{plan_id}` | `DELETE` handlers::unlink_plan_from_milestone (B, 10/0) | **B** | imbriqué (lien) |
| 1014 | `/api/progress` | `GET` handlers::get_progress_batch (C, 35/0) | **C** | lecture calculée |
| 1018 | `/api/milestones/{milestone_id}/progress` | `GET` handlers::get_milestone_progress (C, 24/0) | **C** | lecture calculée |
| 1020 | `/api/commits` | `POST` handlers::create_commit (B, 208/2) | **B** | CRUD+logique (événement) |
| 1024 | `/api/tasks/{task_id}/commits` | `GET` handlers::get_task_commits (B, 7/0)<br>`POST` handlers::link_commit_to_task (B, 18/2) | **B** | imbriqué (lien), imbriqué (lien)+logique |
| 1028 | `/api/plans/{plan_id}/commits` | `GET` handlers::get_plan_commits (B, 7/0)<br>`POST` handlers::link_commit_to_plan (B, 18/2) | **B** | imbriqué (lien), imbriqué (lien)+logique |
| 1033 | `/api/commits/{commit_sha}/files` | `GET` handlers::get_commit_files (C, 11/0) | **C** | lecture calculée |
| 1034 | `/api/files/history` | `GET` handlers::get_file_history (C, 11/0) | **C** | lecture calculée |
| 1039 | `/api/projects/{project_id}/co-changes` | `GET` handlers::get_co_change_graph (C, 16/0) | **C** | lecture calculée |
| 1043 | `/api/files/co-changers` | `GET` handlers::get_file_co_changers (C, 15/0) | **C** | lecture calculée |
| 1048 | `/api/projects/{project_slug}/backfill-touches` | `POST` handlers::backfill_commit_touches (C, 26/0) | **C** | recherche/calcul |
| 1946 | `/api/alerts` | `GET` handlers::list_alerts (A, 28/0) | **A** | CRUD pur |
| 1950 | `/api/alerts/{alert_id}/acknowledge` | `POST` handlers::acknowledge_alert (C, 27/0) | **C** | verbe applicatif |

### projects

| l. | Chemin | Opérations → handler (classe, loc/ev) | Classe | Motif |
|---|---|---|---|---|
| 470 | `/api/projects` | `GET` project_handlers::list_projects (B, 47/0)<br>`POST` project_handlers::create_project (B, 66/0) | **B** | CRUD+logique |
| 476 | `/api/projects/{slug}` | `GET` project_handlers::get_project (B, 30/0)<br>`PATCH` project_handlers::update_project (B, 35/0)<br>`DELETE` project_handlers::delete_project (B, 17/0) | **B** | CRUD+logique, clé slug (id Uuid imposé) |
| 480 | `/api/projects/{slug}/sync` | `POST` project_handlers::sync_project (C, 91/3) | **C** | verbe applicatif |
| 488 | `/api/projects/{slug}/code/search` | `GET` project_handlers::search_project_code (C, 26/0) | **C** | recherche/calcul |
| 504 | `/api/projects/{slug}/health-dashboard` | `GET` project_handlers::get_health_dashboard (B, 111/0) | **B** | imbriqué (lien)+logique |
| 508 | `/api/projects/{slug}/auto-roadmap` | `GET` project_handlers::get_auto_roadmap (B, 294/0) | **B** | imbriqué (lien)+logique |
| 513 | `/api/projects/{slug}/scaffolding` | `GET` project_handlers::get_scaffolding_level (C, 17/0)<br>`PUT` project_handlers::set_scaffolding_level (C, 32/0) | **C** | singleton/config |

### protocols/rfc/reason/episode

| l. | Chemin | Opérations → handler (classe, loc/ev) | Classe | Motif |
|---|---|---|---|---|
| 1667 | `/api/rfcs` | `GET` rfc_handlers::list_rfcs (B, 63/0)<br>`POST` rfc_handlers::create_rfc (B, 47/0) | **B** | CRUD+logique |
| 1673 | `/api/rfcs/{rfc_id}` | `GET` rfc_handlers::get_rfc (B, 89/0)<br>`PATCH` rfc_handlers::update_rfc (B, 67/0)<br>`DELETE` rfc_handlers::delete_rfc (A, 28/0) | **B** | CRUD+logique |
| 1677 | `/api/rfcs/{rfc_id}/transition` | `POST` rfc_handlers::transition_rfc (C, 157/0) | **C** | verbe applicatif |
| 1684 | `/api/protocols` | `GET` protocol_handlers::list_protocols (A, 28/0)<br>`POST` protocol_handlers::create_protocol (B, 196/2) | **B** | CRUD+logique (événement) |
| 1688 | `/api/protocols/route` | `GET` protocol_handlers::route_protocols (C, 133/0) | **C** | recherche/calcul |
| 1692 | `/api/protocols/validate` | `GET` protocol_handlers::validate_all_protocols (C, 105/0) | **C** | recherche/calcul |
| 1696 | `/api/protocols/compose` | `POST` protocol_handlers::compose_protocol (C, 284/2) | **C** | recherche/calcul |
| 1700 | `/api/protocols/simulate` | `POST` protocol_handlers::simulate_activation (C, 80/0) | **C** | recherche/calcul |
| 1706 | `/api/protocols/{protocol_id}` | `GET` protocol_handlers::get_protocol (B, 32/0)<br>`PUT` protocol_handlers::update_protocol (B, 87/2)<br>`DELETE` protocol_handlers::delete_protocol (B, 25/2) | **B** | CRUD+logique, CRUD+logique (événement) |
| 1710 | `/api/protocols/{protocol_id}/states` | `GET` protocol_handlers::list_states (B, 13/0)<br>`POST` protocol_handlers::add_state (B, 88/2) | **B** | imbriqué (lien), imbriqué (lien)+logique |
| 1714 | `/api/protocols/{protocol_id}/states/{state_id}` | `DELETE` protocol_handlers::delete_state (B, 28/2) | **B** | imbriqué (lien)+logique |
| 1718 | `/api/protocols/{protocol_id}/transitions` | `GET` protocol_handlers::list_transitions (B, 13/0)<br>`POST` protocol_handlers::add_transition (B, 79/2) | **B** | imbriqué (lien), imbriqué (lien)+logique |
| 1722 | `/api/protocols/{protocol_id}/transitions/{transition_id}` | `DELETE` protocol_handlers::delete_transition (B, 28/2) | **B** | imbriqué (lien)+logique |
| 1726 | `/api/protocols/{protocol_id}/link-skill` | `POST` protocol_handlers::link_to_skill (B, 34/2) | **B** | imbriqué (lien)+logique |
| 1731 | `/api/protocols/{protocol_id}/runs` | `GET` protocol_handlers::list_runs (B, 25/0)<br>`POST` protocol_handlers::start_run (B, 38/2) | **B** | imbriqué (lien), imbriqué (lien)+logique |
| 1732 | `/api/protocols/runs` | `GET` protocol_handlers::list_all_runs (B, 44/0) | **B** | CRUD+logique |
| 1736 | `/api/protocols/runs/{run_id}` | `GET` protocol_handlers::get_run (A, 21/0)<br>`DELETE` protocol_handlers::delete_run (B, 25/2) | **B** | CRUD+logique (événement) |
| 1740 | `/api/protocols/runs/{run_id}/transition` | `POST` protocol_handlers::fire_transition (C, 86/4) | **C** | verbe applicatif |
| 1744 | `/api/protocols/runs/{run_id}/cancel` | `POST` protocol_handlers::cancel_run (C, 18/0) | **C** | verbe applicatif |
| 1748 | `/api/protocols/runs/{run_id}/fail` | `POST` protocol_handlers::fail_run (C, 48/4) | **C** | verbe applicatif |
| 1752 | `/api/protocols/runs/{run_id}/progress` | `POST` protocol_handlers::report_progress (C, 40/2) | **C** | verbe applicatif |
| 1756 | `/api/protocols/runs/{run_id}/children` | `GET` protocol_handlers::get_run_children (C, 19/0) | **C** | lecture calculée |
| 1760 | `/api/protocols/runs/{run_id}/tree` | `GET` protocol_handlers::get_run_tree (C, 37/0) | **C** | lecture calculée |
| 1764 | `/api/reason` | `POST` reason_handlers::reason (C, 48/0) | **C** | verbe applicatif |
| 1768 | `/api/reason/{tree_id}/feedback` | `POST` reason_handlers::reason_feedback (C, 182/0) | **C** | verbe applicatif |
| 1772 | `/api/episodes` | `GET` episode_handlers::list_episodes (B, 31/0) | **B** | CRUD+logique |
| 1776 | `/api/episodes/collect` | `POST` episode_handlers::collect_episode (C, 49/2) | **C** | verbe applicatif |
| 1780 | `/api/episodes/anonymize` | `POST` episode_handlers::anonymize_episode (C, 13/0) | **C** | verbe applicatif |
| 1784 | `/api/episodes/export-artifact` | `POST` episode_handlers::export_artifact (B, 72/0) | **B** | CRUD+logique |

### runs/plans-exec

| l. | Chemin | Opérations → handler (classe, loc/ev) | Classe | Motif |
|---|---|---|---|---|
| 707 | `/api/plans/{plan_id}/run` | `POST` handlers::run_plan (C, 32/0) | **C** | verbe applicatif |
| 711 | `/api/plans/{plan_id}/run/status` | `GET` handlers::get_run_status (C, 11/0) | **C** | lecture calculée |
| 715 | `/api/plans/{plan_id}/run/cancel` | `POST` handlers::cancel_run (C, 18/0) | **C** | verbe applicatif |
| 719 | `/api/plans/{plan_id}/run/force-cancel` | `POST` handlers::force_cancel_run (C, 21/0) | **C** | verbe applicatif |
| 723 | `/api/plans/{plan_id}/run/budget` | `PATCH` handlers::update_run_budget (C, 35/0) | **C** | singleton/config |
| 728 | `/api/runner/cost-breakdown` | `GET` handlers::get_runner_cost_breakdown (C, 28/0) | **C** | lecture calculée |
| 732 | `/api/plans/{plan_id}/run/auto-pr` | `POST` handlers::create_auto_pr (C, 96/0) | **C** | verbe applicatif |
| 736 | `/api/plans/{plan_id}/run/tasks/{task_id}/retry` | `POST` handlers::retry_plan_task (C, 26/0) | **C** | verbe applicatif |
| 738 | `/api/runs` | `GET` handlers::list_all_plan_runs (A, 24/0) | **A** | CRUD pur |
| 740 | `/api/runs/{run_id}` | `GET` handlers::get_plan_run (B, 45/0) | **B** | CRUD+logique |
| 744 | `/api/runs/{run_id}/agent-executions` | `GET` handlers::get_run_agent_executions (C, 11/0) | **C** | lecture calculée |
| 748 | `/api/plans/{plan_id}/runs/compare` | `POST` handlers::compare_plan_runs (C, 46/0) | **C** | recherche/calcul |
| 752 | `/api/plans/{plan_id}/runs/predict` | `POST` handlers::predict_plan_run (C, 22/0) | **C** | verbe applicatif |
| 754 | `/api/runs/{run_id}/gates` | `GET` handlers::get_run_gates (C, 12/0) | **C** | lecture calculée |
| 758 | `/api/runs/{run_id}/progress` | `GET` handlers::get_run_progress (C, 23/0) | **C** | lecture calculée |
| 763 | `/api/plans/{plan_id}/triggers` | `GET` handlers::list_triggers (B, 13/0)<br>`POST` handlers::create_trigger (B, 74/2) | **B** | imbriqué (lien), imbriqué (lien)+logique |
| 767 | `/api/triggers/{trigger_id}` | `DELETE` handlers::delete_trigger (B, 20/2) | **B** | CRUD+logique (événement) |
| 771 | `/api/triggers/{trigger_id}/enable` | `POST` handlers::enable_trigger (C, 26/2) | **C** | verbe applicatif |
| 775 | `/api/triggers/{trigger_id}/disable` | `POST` handlers::disable_trigger (C, 22/2) | **C** | verbe applicatif |
| 779 | `/api/triggers/{trigger_id}/firings` | `GET` handlers::list_trigger_firings (C, 17/0) | **C** | lecture calculée |
| 784 | `/api/webhooks/{trigger_id}` | `POST` handlers::receive_webhook (C, 114/0) | **C** | autre |
| 885 | `/api/plans/{plan_id}/tasks/{task_id}/delegate` | `POST` handlers::delegate_task (C, 106/0) | **C** | verbe applicatif |
| 890 | `/api/execution-places` | `GET` execution_place_handlers::list_pending_execution_places (B, 32/0) | **B** | CRUD+logique |
| 894 | `/api/execution-places/{placement_id}` | `GET` execution_place_handlers::get_execution_place (A, 11/0) | **A** | CRUD pur |
| 898 | `/api/execution-places/{placement_id}/confirm` | `POST` execution_place_handlers::confirm_execution_place (C, 20/0) | **C** | verbe applicatif |
| 902 | `/api/execution-places/{placement_id}/reject` | `POST` execution_place_handlers::reject_execution_place (C, 31/0) | **C** | verbe applicatif |
| 1923 | `/api/event-triggers` | `GET` trigger_handlers::list_triggers (A, 13/0)<br>`POST` trigger_handlers::create_trigger (B, 74/2) | **B** | CRUD+logique (événement) |
| 1928 | `/api/event-triggers/stats` | `GET` trigger_handlers::trigger_stats (C, 36/0) | **C** | lecture calculée |
| 1934 | `/api/event-triggers/{id}` | `GET` trigger_handlers::get_trigger (A, 12/0)<br>`PUT` trigger_handlers::update_trigger (B, 33/1)<br>`DELETE` trigger_handlers::delete_trigger (B, 20/2) | **B** | CRUD+logique (événement) |
| 1938 | `/api/event-triggers/{id}/enable` | `POST` trigger_handlers::enable_trigger (C, 26/2) | **C** | verbe applicatif |
| 1942 | `/api/event-triggers/{id}/disable` | `POST` trigger_handlers::disable_trigger (C, 22/2) | **C** | verbe applicatif |

### sharing

| l. | Chemin | Opérations → handler (classe, loc/ev) | Classe | Motif |
|---|---|---|---|---|
| 520 | `/api/projects/{slug}/sharing` | `GET` sharing_handlers::get_sharing_status (B, 12/0) | **B** | imbriqué (lien) |
| 524 | `/api/projects/{slug}/sharing/enable` | `POST` sharing_handlers::enable_sharing (C, 20/0) | **C** | verbe applicatif |
| 528 | `/api/projects/{slug}/sharing/disable` | `POST` sharing_handlers::disable_sharing (C, 20/0) | **C** | verbe applicatif |
| 532 | `/api/projects/{slug}/sharing/policy` | `GET` sharing_handlers::get_sharing_policy (C, 8/0)<br>`PUT` sharing_handlers::set_sharing_policy (C, 33/0) | **C** | singleton/config |
| 536 | `/api/notes/{note_id}/sharing/consent` | `PUT` sharing_handlers::set_sharing_consent (C, 18/0) | **C** | singleton/config |
| 540 | `/api/projects/{slug}/sharing/history` | `GET` sharing_handlers::get_sharing_history (C, 18/0) | **C** | lecture calculée |
| 544 | `/api/projects/{slug}/sharing/preview` | `GET` sharing_handlers::preview_sharing (B, 30/0) | **B** | imbriqué (lien) |
| 548 | `/api/projects/{slug}/sharing/suggest` | `GET` sharing_handlers::suggest_sharing (B, 37/0) | **B** | imbriqué (lien)+logique |
| 552 | `/api/projects/{slug}/sharing/retract` | `POST` sharing_handlers::retract_sharing (B, 77/0) | **B** | imbriqué (lien)+logique |
| 556 | `/api/projects/{slug}/sharing/tombstones` | `GET` sharing_handlers::list_tombstones (B, 22/0) | **B** | imbriqué (lien) |
| 560 | `/api/projects/{slug}/sharing/last-report` | `GET` sharing_handlers::get_last_privacy_report (B, 16/0) | **B** | imbriqué (lien) |

### skills/personas/registry

| l. | Chemin | Opérations → handler (classe, loc/ev) | Classe | Motif |
|---|---|---|---|---|
| 1495 | `/api/analysis-profiles` | `GET` profile_handlers::list_analysis_profiles (A, 13/0)<br>`POST` profile_handlers::create_analysis_profile (B, 61/2) | **B** | CRUD+logique (événement) |
| 1500 | `/api/analysis-profiles/{profile_id}` | `GET` profile_handlers::get_analysis_profile (A, 16/0)<br>`DELETE` profile_handlers::delete_analysis_profile (B, 26/2) | **B** | CRUD+logique (événement) |
| 1507 | `/api/skills` | `GET` skill_handlers::list_skills (A, 23/0)<br>`POST` skill_handlers::create_skill (B, 56/2) | **B** | CRUD+logique (événement) |
| 1511 | `/api/skills/search` | `POST` skill_handlers::search_skills (C, 39/0) | **C** | recherche/calcul |
| 1517 | `/api/skills/{skill_id}` | `GET` skill_handlers::get_skill (A, 14/0)<br>`PUT` skill_handlers::update_skill (B, 113/4)<br>`DELETE` skill_handlers::delete_skill (B, 43/2) | **B** | CRUD+logique (événement) |
| 1521 | `/api/skills/{skill_id}/members` | `GET` skill_handlers::get_skill_members (B, 13/0)<br>`POST` skill_handlers::add_skill_member (B, 48/2) | **B** | imbriqué (lien), imbriqué (lien)+logique |
| 1525 | `/api/skills/{skill_id}/members/{entity_type}/{entity_id}` | `DELETE` skill_handlers::remove_skill_member (B, 59/2) | **B** | imbriqué (lien)+logique |
| 1529 | `/api/skills/{skill_id}/activate` | `POST` skill_handlers::activate_skill (C, 27/0) | **C** | verbe applicatif |
| 1533 | `/api/skills/{skill_id}/export` | `GET` skill_handlers::export_skill (B, 24/0) | **B** | imbriqué (lien) |
| 1534 | `/api/skills/import` | `POST` skill_handlers::import_skill (C, 43/1) | **C** | verbe applicatif |
| 1538 | `/api/skills/{skill_id}/health` | `GET` skill_handlers::get_skill_health (C, 27/0) | **C** | lecture calculée |
| 1542 | `/api/skills/{skill_id}/split` | `POST` skill_handlers::split_skill (C, 148/2) | **C** | verbe applicatif |
| 1543 | `/api/skills/merge` | `POST` skill_handlers::merge_skills (C, 107/2) | **C** | verbe applicatif |
| 1550 | `/api/personas` | `GET` persona_handlers::list_personas (A, 23/0)<br>`POST` persona_handlers::create_persona (B, 80/2) | **B** | CRUD+logique (événement) |
| 1555 | `/api/personas/global` | `GET` persona_handlers::list_global_personas (C, 13/0) | **C** | recherche/calcul |
| 1559 | `/api/personas/find-for-file` | `GET` persona_handlers::find_for_file (B, 66/0) | **B** | CRUD+logique |
| 1563 | `/api/personas/import` | `POST` persona_handlers::import_persona (C, 155/0) | **C** | verbe applicatif |
| 1567 | `/api/personas/auto-build` | `POST` persona_handlers::auto_build_persona (C, 108/0) | **C** | verbe applicatif |
| 1571 | `/api/personas/maintain` | `POST` persona_handlers::maintain_personas (C, 17/0) | **C** | verbe applicatif |
| 1575 | `/api/personas/detect` | `POST` persona_handlers::detect_personas (C, 29/0) | **C** | verbe applicatif |
| 1579 | `/api/personas/learning-health` | `GET` persona_handlers::get_learning_health (C, 32/0) | **C** | lecture calculée |
| 1585 | `/api/personas/{persona_id}` | `GET` persona_handlers::get_persona (A, 14/0)<br>`PUT` persona_handlers::update_persona (B, 104/4)<br>`DELETE` persona_handlers::delete_persona (B, 25/2) | **B** | CRUD+logique (événement) |
| 1589 | `/api/personas/{persona_id}/subgraph` | `GET` persona_handlers::get_subgraph (C, 13/0) | **C** | lecture calculée |
| 1593 | `/api/personas/{persona_id}/export` | `GET` persona_handlers::export_persona (B, 92/0) | **B** | imbriqué (lien)+logique |
| 1597 | `/api/personas/{persona_id}/activate` | `POST` persona_handlers::activate_persona (C, 122/0) | **C** | verbe applicatif |
| 1602 | `/api/personas/{persona_id}/skills/{skill_id}` | `POST` persona_handlers::add_skill (C, 21/2)<br>`DELETE` persona_handlers::remove_skill (B, 21/2) | **C** | autre |
| 1607 | `/api/personas/{persona_id}/protocols/{protocol_id}` | `POST` persona_handlers::add_protocol (C, 21/2)<br>`DELETE` persona_handlers::remove_protocol (B, 21/2) | **C** | autre |
| 1613 | `/api/personas/{persona_id}/feature-graphs/{feature_graph_id}` | `POST` persona_handlers::scope_to_feature_graph (C, 12/0)<br>`DELETE` persona_handlers::unscope_feature_graph (B, 12/0) | **C** | autre |
| 1618 | `/api/personas/{persona_id}/files` | `POST` persona_handlers::add_file (B, 28/2)<br>`DELETE` persona_handlers::remove_file (C, 22/2) | **C** | autre |
| 1623 | `/api/personas/{persona_id}/functions` | `POST` persona_handlers::add_function (B, 28/2)<br>`DELETE` persona_handlers::remove_function (C, 22/2) | **C** | autre |
| 1628 | `/api/personas/{persona_id}/notes/{note_id}` | `POST` persona_handlers::add_note (C, 23/2)<br>`DELETE` persona_handlers::remove_note (B, 21/2) | **C** | autre |
| 1633 | `/api/personas/{persona_id}/decisions/{decision_id}` | `POST` persona_handlers::add_decision (C, 89/2)<br>`DELETE` persona_handlers::remove_decision (B, 21/2) | **C** | autre |
| 1638 | `/api/personas/{persona_id}/extends/{parent_persona_id}` | `POST` persona_handlers::add_extends (C, 26/2)<br>`DELETE` persona_handlers::remove_extends (B, 21/2) | **C** | autre |
| 1645 | `/api/registry/publish` | `POST` registry_handlers::publish_skill (C, 62/0) | **C** | verbe applicatif |
| 1649 | `/api/registry/search` | `GET` registry_handlers::search_registry (C, 73/0) | **C** | recherche/calcul |
| 1653 | `/api/registry/{id}` | `GET` registry_handlers::get_published_skill (A, 14/0) | **A** | CRUD pur |
| 1657 | `/api/registry/{id}/import` | `POST` registry_handlers::import_from_registry (C, 72/1) | **C** | verbe applicatif |

### sys

| l. | Chemin | Opérations → handler (classe, loc/ev) | Classe | Motif |
|---|---|---|---|---|
| 350 | `/health` | `GET` handlers::health (C, 58/0) | **C** | proto/auth/sys |
| 351 | `/api/version` | `GET` handlers::get_version (C, 23/0) | **C** | proto/auth/sys |
| 352 | `/api/setup-status` | `GET` handlers::setup_status (C, 5/0) | **C** | proto/auth/sys |
| 610 | `/api/update/check` | `POST` update_handlers::check_update (C, 8/0) | **C** | proto/auth/sys |
| 611 | `/api/update/install` | `POST` update_handlers::install_update (C, 9/0) | **C** | proto/auth/sys |
| 612 | `/api/update/restart` | `POST` update_handlers::restart_update (C, 17/0) | **C** | proto/auth/sys |
| 975 | `/api/sync` | `POST` handlers::sync_directory (C, 110/1) | **C** | proto/auth/sys |
| 1050 | `/api/wake` | `POST` handlers::wake (C, 16/0) | **C** | proto/auth/sys |
| 1053 | `/hooks/wake` | `POST` handlers::wake (C, 16/0) | **C** | proto/auth/sys |
| 1056 | `/internal/events` | `POST` handlers::receive_event (C, 13/1) | **C** | proto/auth/sys |
| 1065 | `/api/watch` | `GET` handlers::watch_status (C, 15/0)<br>`POST` handlers::start_watch (C, 44/0)<br>`DELETE` handlers::stop_watch (C, 49/0) | **C** | proto/auth/sys |
| 1067 | `/api/reactor/status` | `GET` handlers::reactor_status (C, 18/0) | **C** | proto/auth/sys |
| 335 | `/mcp` | `POST` mcp::http_transport::mcp_post (C, 99/0)<br>`GET` mcp::http_transport::mcp_get (C, 7/0)<br>`DELETE` mcp::http_transport::mcp_delete (C, 28/0) | **C** | proto/auth/sys |

### vault

| l. | Chemin | Opérations → handler (classe, loc/ev) | Classe | Motif |
|---|---|---|---|---|
| 614 | `/api/vault` | `GET` vault_handlers::get_vault (C, 15/0) | **C** | coffre (secret, pas une entité) |
| 615 | `/api/vault/init` | `POST` vault_handlers::init_vault (C, 20/0) | **C** | coffre (secret, pas une entité) |
| 616 | `/api/vault/unlock` | `POST` vault_handlers::unlock_vault (C, 17/0) | **C** | coffre (secret, pas une entité) |
| 617 | `/api/vault/lock` | `POST` vault_handlers::lock_vault (C, 8/0) | **C** | coffre (secret, pas une entité) |
| 621 | `/api/vault/secrets/{name}` | `PUT` vault_handlers::put_secret (C, 14/0)<br>`DELETE` vault_handlers::delete_secret (C, 11/0) | **C** | coffre (secret, pas une entité) |
| 622 | `/api/vault/grants` | `POST` vault_handlers::create_grant (C, 23/0) | **C** | coffre (secret, pas une entité) |
| 626 | `/api/vault/grants/{id}` | `DELETE` vault_handlers::revoke_grant (C, 12/0) | **C** | coffre (secret, pas une entité) |
| 630 | `/api/vault/requests/{id}/answer` | `POST` vault_handlers::answer_request (C, 88/1) | **C** | coffre (secret, pas une entité) |
| 631 | `/api/vault/agent/read` | `POST` vault_handlers::agent_read (C, 19/0) | **C** | coffre (secret, pas une entité) |
| 635 | `/api/vault/agent/requests` | `POST` vault_handlers::agent_request (C, 42/1) | **C** | coffre (secret, pas une entité) |
| 639 | `/api/vault/agent/available` | `GET` vault_handlers::agent_available (C, 11/0) | **C** | coffre (secret, pas une entité) |

### workspaces

| l. | Chemin | Opérations → handler (classe, loc/ev) | Classe | Motif |
|---|---|---|---|---|
| 1957 | `/api/workspaces` | `GET` workspace_handlers::list_workspaces (A, 25/0)<br>`POST` workspace_handlers::create_workspace (B, 47/0) | **B** | CRUD+logique |
| 1963 | `/api/workspaces/{slug}` | `GET` workspace_handlers::get_workspace (B, 13/0)<br>`PATCH` workspace_handlers::update_workspace (B, 52/0)<br>`DELETE` workspace_handlers::delete_workspace (B, 15/0) | **B** | CRUD+logique, clé slug (id Uuid imposé) |
| 1967 | `/api/workspaces/{slug}/overview` | `GET` workspace_handlers::get_workspace_overview (C, 58/0) | **C** | lecture calculée |
| 1972 | `/api/workspaces/{slug}/projects` | `GET` workspace_handlers::list_workspace_projects (B, 19/0)<br>`POST` workspace_handlers::add_project_to_workspace (B, 24/0) | **B** | imbriqué (lien) |
| 1976 | `/api/workspaces/{slug}/projects/{project_id}` | `DELETE` workspace_handlers::remove_project_from_workspace (B, 22/0) | **B** | imbriqué (lien) |
| 1982 | `/api/workspaces/{slug}/milestones` | `GET` workspace_handlers::list_workspace_milestones (B, 39/0)<br>`POST` workspace_handlers::create_workspace_milestone (B, 39/0) | **B** | imbriqué (lien)+logique |
| 1986 | `/api/workspace-milestones` | `GET` workspace_handlers::list_all_workspace_milestones (B, 52/0) | **B** | CRUD+logique |
| 1992 | `/api/workspace-milestones/{id}` | `GET` workspace_handlers::get_workspace_milestone (B, 105/0)<br>`PATCH` workspace_handlers::update_workspace_milestone (B, 50/0)<br>`DELETE` workspace_handlers::delete_workspace_milestone (A, 12/0) | **B** | CRUD+logique |
| 1997 | `/api/workspace-milestones/{id}/tasks` | `GET` workspace_handlers::list_workspace_milestone_tasks (B, 16/0)<br>`POST` workspace_handlers::add_task_to_workspace_milestone (B, 21/0) | **B** | imbriqué (lien) |
| 2001 | `/api/workspace-milestones/{id}/plans` | `POST` workspace_handlers::link_plan_to_workspace_milestone (B, 21/0) | **B** | imbriqué (lien) |
| 2005 | `/api/workspace-milestones/{id}/plans/{plan_id}` | `DELETE` workspace_handlers::unlink_plan_from_workspace_milestone (B, 19/0) | **B** | imbriqué (lien) |
| 2009 | `/api/workspace-milestones/{id}/progress` | `GET` workspace_handlers::get_workspace_milestone_progress (C, 28/0) | **C** | lecture calculée |
| 2014 | `/api/workspaces/{slug}/resources` | `GET` workspace_handlers::list_resources (B, 21/0)<br>`POST` workspace_handlers::create_resource (B, 44/0) | **B** | imbriqué (lien), imbriqué (lien)+logique |
| 2020 | `/api/resources/{id}` | `GET` workspace_handlers::get_resource (A, 17/0)<br>`PATCH` workspace_handlers::update_resource (A, 26/0)<br>`DELETE` workspace_handlers::delete_resource (A, 12/0) | **A** | CRUD pur |
| 2024 | `/api/resources/{id}/projects` | `POST` workspace_handlers::link_resource_to_project (B, 36/0) | **B** | imbriqué (lien)+logique |
| 2029 | `/api/workspaces/{slug}/components` | `GET` workspace_handlers::list_components (B, 24/0)<br>`POST` workspace_handlers::create_component (B, 43/0) | **B** | imbriqué (lien), imbriqué (lien)+logique |
| 2035 | `/api/components/{id}` | `GET` workspace_handlers::get_component (A, 17/0)<br>`PATCH` workspace_handlers::update_component (B, 43/0)<br>`DELETE` workspace_handlers::delete_component (A, 12/0) | **B** | CRUD+logique |
| 2039 | `/api/components/{id}/dependencies` | `POST` workspace_handlers::add_component_dependency (B, 21/0) | **B** | imbriqué (lien) |
| 2043 | `/api/components/{id}/dependencies/{dep_id}` | `DELETE` workspace_handlers::remove_component_dependency (B, 19/0) | **B** | imbriqué (lien) |
| 2047 | `/api/components/{id}/project` | `PUT` workspace_handlers::map_component_to_project (C, 21/0) | **C** | singleton/config |
| 2051 | `/api/workspaces/{slug}/topology` | `GET` workspace_handlers::get_workspace_topology (C, 35/0) | **C** | lecture calculée |
| 2055 | `/api/workspaces/{slug}/topology/derive` | `POST` workspace_handlers::derive_workspace_topology (C, 19/0) | **C** | verbe applicatif |
| 2059 | `/api/workspaces/{slug}/coupling-matrix` | `GET` workspace_handlers::get_coupling_matrix (C, 27/0) | **C** | lecture calculée |

### ws

| l. | Chemin | Opérations → handler (classe, loc/ev) | Classe | Motif |
|---|---|---|---|---|
| 427 | `/ws/events` | `GET` ws_handlers::ws_events (C, 116/0) | **C** | stream |
| 428 | `/ws/activity` | `GET` ws_activity_handler::ws_activity (C, 58/0) | **C** | stream |
| 429 | `/ws/chat/{session_id}` | `GET` ws_chat_handler::ws_chat (C, 70/0) | **C** | stream |
| 430 | `/ws/run/{run_id}` | `GET` ws_run_handler::ws_run (C, 28/0) | **C** | stream |

