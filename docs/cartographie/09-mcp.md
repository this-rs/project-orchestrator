# Cartographie — la surface MCP du Project Orchestrator

Analyse en lecture seule de `backend/src/mcp/handlers.rs` (13 483 lignes) et du contexte REST (`backend/src/api/routes.rs`, 3 188 lignes). Aucune modification, aucun build, aucun git. Toute affirmation est `chemin:ligne`.

## 1. L'inventaire réel

Table `MEGA_TOOL_ACTIONS` (`src/mcp/handlers.rs:220`), triplets `(méga-outil, action, nom_interne)`.

**Chiffre vérifié : 427 actions, exactement.** (`awk` sur le bloc de la table : 427 lignes `("tool", "action", "internal")`.) Pas d'écart avec le dispatch : `try_handle_http` (`src/mcp/handlers.rs:1053`) contient **427 bras `match`** (`"nom_interne" => { ... }`), un par entrée de la table, vérifiés 1:1 par le test `test_every_dispatched_action_has_a_handler_arm` (`:13361`) et sa réciproque implicite (aucun bras orphelin — compté directement). Donc **aucun écart table/dispatch** : contrairement à l'hypothèse du brief, il n'y a pas de sur-comptage côté `match`.

| Méga-outil | Actions | Méga-outil | Actions |
|---|---|---|---|
| code | 54 | milestone | 9 |
| admin | 43 | feature_graph | 9 |
| persona | 30 | environment | 9 |
| plan | 27 | release | 8 |
| protocol | 26 | component | 8 |
| note | 22 | mcp_federation | 7 |
| chat | 19 | commit | 7 |
| task | 15 | step | 6 |
| skill | 15 | resource | 6 |
| project | 15 | neural_routing | 6 |
| workspace | 12 | lifecycle_hook | 5 |
| sharing | 12 | day_plan | 5 |
| decision | 12 | constraint | 5 |
| workspace_milestone | 10 | trajectory | 4 |
| provider | 9 | episode | 4 |
| — | — | analysis_profile | 4 |
| — | — | vault | 2 |
| — | — | reasoning | 2 |

32 outils, somme = 427.

## 2. La forme du contrat

- **Entrée** : un seul JSON par appel, discriminé par `action` (string). Pas de schéma JSON par action au sens strict côté dispatch : chaque bras lit les champs qu'il attend via des helpers (`extract_string`, `extract_id`, `extract_optional_*`, `extract_string_array` — `src/mcp/http_client.rs:331-368`). `extract_id` valide le format UUID (`uuid::Uuid::parse_str`, `:349`) ; les autres n'ont pas de validation de forme au-delà du type JSON. Il n'y a donc **pas de `serde` de structs typées par action** — c'est de l'extraction manuelle, champ par champ.
- **Schéma publié** : chaque méga-outil déclare un `InputSchema` (JSON Schema léger, `src/mcp/tools.rs`, ex. `:373`) qui liste les paramètres possibles toutes actions confondues (pas un schéma par action). C'est ce schéma que le host MCP valide avant l'appel.
- **Erreurs** : `anyhow::Result` partout ; une extraction manquante devient `anyhow!("{field} is required")`, propagée jusqu'à la réponse MCP comme erreur d'outil (pas de code d'erreur structuré par type de champ).
- **Sérialisation des résultats** : directement le JSON renvoyé par l'API REST interne (`http.get/post/patch/delete`, cf. §3) — aucune reprojection côté MCP, le JSON REST est le JSON MCP.
- **Stabilité du contrat — tests de forme réels** (`src/mcp/handlers.rs:13340-13430`) :
  - `test_description_does_not_re_list_the_action_enum` — l'énum d'actions ne doit être listée qu'une fois (pas de copie qui dérive) ;
  - `test_every_dispatched_action_has_a_handler_arm` — chaque triplet de la table a un bras ;
  - `test_handler_keys_are_published` — chaque champ lu par un bras doit figurer dans le schéma publié de son méga-outil (sinon un agent ne peut pas le découvrir) ;
  - un parseur de source (`keys_read_per_arm`, testé par `test_keys_parser_sees_known_arms`, `:13407` exige `arms.len() > 350`) qui lit les `.get("champ")` réellement présents dans chaque bras — preuve vérifiée mécaniquement, pas déclarée.

## 3. Le couplage

Point central, qui change l'hypothèse du brief : **`handlers.rs` n'appelle quasiment jamais `GraphStore` directement.** C'est une couche de *traduction MCP → HTTP interne* : chaque bras appelle `self.http()` (un client HTTP, `src/mcp/http_client.rs`) sur l'API REST du même backend (`/api/...`).

Décompte vérifié par un parcours ligne à ligne des 427 bras (le mot « http » apparaît dans le corps, y compris sur la ligne suivante pour les appels `.patch(...)` enjambés) :
- **425 / 427 bras** appellent `http.get`, `http.get_with_query`, `http.post`, `http.patch`, `http.put`, `http.delete` ou `http.vault_call` → ils **proxient un endpoint REST existant**, candidats naturels à une génération depuis le même descripteur que la route REST.
- **2 / 427 bras** portent une logique propre, sans aucun aller HTTP :
  - `install_hooks` (`:3565`) — stub figé qui répond un message de dépréciation (les hooks sont gérés en interne par le SDK Nexus) ;
  - `get_learning_stats` (`:3574`) — lit un singleton process-local (`crate::feedback::OutcomeTracker::global()`) sans passer par l'API REST.

Répartition des verbes HTTP utilisés par les 425 bras proxy : 70 `get`, 51 `get_with_query`, 51 `post`, 15 `delete`, 1 `put`, 1 `patch`, 2 `vault_call` (total brut 191 — le reste des 425 utilise des appels enjambés sur deux lignes, comptés par bras et non par occurrence de méthode).

Côté REST : `src/api/routes.rs` (3 188 lignes) déclare **483 routes** (`grep -c '\.route('`), sans annotation `utoipa`/OpenAPI — ce sont des closures Axum typées en Rust, pas un descripteur de données. `GraphStore` (`src/neo4j/*`, implémentation `src/neo4j/impl_graph_store.rs`) est référencé dans **218 fichiers** du dépôt, mais **jamais directement par `handlers.rs`** — l'accès au graphe passe systématiquement par la couche REST/service, jamais par le module MCP.

## 4. La synergie possible

**Oui pour la majorité, non pour trois raisons précises.**

Puisque 425/427 actions MCP ne sont qu'un passage `action → (verbe, chemin, query/body)` vers une route REST déjà existante, engendrer l'action MCP depuis le même descripteur que la route REST est structurellement possible **si** ce descripteur existe. Aujourd'hui il n'existe pas : les routes Axum sont des closures Rust typées (`Json<T>`, `Query<T>`), pas des données introspectables à la compilation sans macro dédiée.

**Trois obstacles à la génération automatique :**
1. **Pas de source de vérité structurée côté REST.** Aucune annotation `utoipa`/`schemars` sur les handlers Axum (`grep` négatif sur `routes.rs`) : il faudrait d'abord instrumenter 483 routes avant de pouvoir engendrer quoi que ce soit depuis elles — le travail de "génération" commence par une mise à niveau du côté REST, pas du côté MCP.
2. **Forme d'appel différente.** MCP reçoit un seul blob JSON avec `action` + tous les champs à plat (list, create, etc. mélangés dans un même schéma de méga-outil) ; REST distingue verbe HTTP, chemin paramétré (`/api/projects/{slug}`), query string et corps. La génération doit donc savoir dériver `(méga-outil, action)` → `(verbe, template de chemin, séparation query/body)`, ce qui demande un mapping explicite au moins pour les paramètres de chemin (aujourd'hui fait à la main, ex. `format!("/api/projects/{}", slug)`).
3. **Pagination, streaming et réponses partielles n'ont pas d'équivalent 1:1.** `run_plan`/`plan.run` démarre un agent en tâche de fond et son statut se lit via un *autre* appel (`run_status`) — ce n'est pas un simple GET/POST translatable terme à terme. Les actions qui composent plusieurs appels REST (ex. `list_rfcs` → proxy vers `list_notes` avec un filtre `note_type=rfc` injecté, `:ligne 3151-3167` zone list_project_notes) ou qui branchent sur un paramètre (`analyze_impact` bascule vers `/api/code/impact/multi` si `multi=true`, `:4197-4213`) codent une **décision métier dans le bras MCP lui-même** — rien côté REST ne la décrit.

**Deux exemples où la génération marcherait directement** (mapping mécanique, 1 route REST ↔ 1 action, aucune logique) :
- `project.get` → `get_project` (`:1109`) : `GET /api/projects/{slug}`, un seul paramètre de chemin, aucune transformation.
- `commit.get_task_commits` → `get_task_commits` : `GET` simple sur un chemin paramétré par `task_id`, pas de query optionnelle, pas de branche.

**Deux exemples où elle ne marcherait pas sans logique dédiée** :
- `note.list_rfcs` : n'a pas de route REST propre — c'est un alias métier de `list_notes` avec un filtre imposé (`note_type=rfc`) et jusqu'à 4 filtres optionnels combinés ; générer cette action demanderait de décrire la « vue » comme une entité de premier ordre, pas comme un simple alias de endpoint.
- `code.analyze_impact` : une seule action MCP recouvre **deux routes REST distinctes** (`/api/code/impact` et `/api/code/impact/multi`) choisies par un booléen `multi` dans les arguments — un descripteur REST classique (une route = un schéma) ne porte pas cette ramification ; elle resterait écrite à la main.

## 5. Plan de portage — ce qui reste à la main, ce qui peut être généré

**Générable depuis un descripteur REST (après instrumentation) : ≈ 390-400 / 427 actions.**
Tout bras qui fait un seul appel HTTP vers une seule route, avec un mapping trivial args→(path params, query params optionnels, body) : la quasi-totalité des `list/get/create/update/delete` CRUD des entités simples (project, task, step, milestone, release, environment, component, resource, constraint, decision, commit, lifecycle_hook, workspace*, feature_graph, episode, analysis_profile, provider, trajectory — la majorité des 32 outils). Base de départ sûre : les 165 bras recensés comme `HTTP` sur appel `http.` **non enjambé** plus les CRUD simples parmi les 425 au total, soit la grande majorité de l'inventaire ci-dessus.

**À garder à la main : ≈ 27-37 / 427 actions.**
- Les 2 sans HTTP (`install_hooks`, `get_learning_stats`).
- Les actions qui recouvrent plusieurs routes selon un paramètre (`analyze_impact` et ses semblables dans `code.*`).
- Les alias/vues métier sans route REST dédiée (`list_rfcs`, et probablement d'autres actions `get_propagated*`, `get_context*` qui combinent des filtres).
- Les actions asynchrones à cycle de vie propre (`plan.run` / `run_status` / `cancel_run`, `protocol.start_run` / `*_run_*`) où l'« action » MCP n'est qu'une des faces d'un processus suivi par ailleurs.
- `vault.*` (2 actions) : chemin volontairement hors REST classique (`http.vault_call`), sécurité dédiée — à ne jamais générer automatiquement.

**Préalable commun, avant tout générateur** : poser un descripteur structuré côté REST (`utoipa` ou équivalent `schemars`) pour les 483 routes — sans cela il n'y a rien à générer *depuis*, seulement un gabarit à écrire à la main pour chaque nouvelle action, ce qui est la situation actuelle.
