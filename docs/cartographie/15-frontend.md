# Cartographie frontend (lecture seule) — project-orchestrator-frontend

Dépôt cartographié : `/Users/triviere/projects/project-orchestrator/frontend` (React 19, TypeScript, three.js, Tauri).
Méthode : grep/find uniquement, aucun build. Chaque chiffre est un compte obtenu par commande, pas une impression.

## 0. Volume

- Fichiers `src` : **2 423** (`find src -type f | wc -l`).
- Fichiers TS/TSX : **1 559 `.ts` + 824 `.tsx`**.
- LOC non-test (`*.ts`+`*.tsx`, hors `*.test.*`) : **311 652**.
- Composants non-test : **424** `src/components/*.tsx` (dont **64** dans `src/components/ui/`).
- Hooks : **70**, atoms (jotai) : **21**, workers : **1** (`communityWorker.ts`).

## 1. Pages et routes

- **57 pages** (`find src/pages -name '*.tsx' | grep -v .test.`) — dont 9 pages du wizard `src/pages/setup/` et 2 pages documents.
- **70 éléments `<Route`** dans `src/App.tsx` (routeur : `react-router-dom` v7, `BrowserRouter`).
- Entrée : `src/App.tsx` → `/` redirige vers `/today` ; tout le domaine applicatif est **sous `/workspace/:slug`** (garde `WorkspaceRouteGuard` + layout `MainLayout`).

### Vues de domaine (sous `/workspace/:slug/…`)

| Domaine | Liste | Détail |
|---|---|---|
| Projets | `projects` → `ProjectsPage.tsx` | `projects/:projectSlug` → `ProjectDetailPage.tsx` |
| Plans | `plans` → `PlansPage.tsx` | `plans/:planId` → `PlanDetailPage.tsx` + `plans/:planId/runner` → `RunnerDashboard.tsx` |
| Tâches | `tasks` → `TasksPage.tsx` | `tasks/:taskId` → `TaskDetailPage.tsx` |
| Notes | `notes` → `NotesPage.tsx` | `notes/:noteId` → `NoteDetailPage.tsx` |
| Décisions | `decisions` → `DecisionsPage.tsx` | `decisions/:decisionId` → `DecisionDetailPage.tsx` |
| Code | `code` → `CodePage.tsx` | (pas de détail page ; vue exploreur) |
| Chat | `chat/:conversationId` → `ChatSessionPage.tsx` | `chat/:sessionId/timeline` → `ChatTimelinePage.tsx` |
| Activité | `activity` → `ActivityHubPage.tsx` | — |
| Jalons | `milestones` → `MilestonesPage.tsx` | `milestones/:milestoneId` + `projects/:p/milestones/:m` |
| Skills | `skills` → `SkillsPage.tsx` | `skills/:skillId` → `SkillDetailPage.tsx` |
| Personas | `personas` → `PersonasPage.tsx` | `personas/:personaId` → `PersonaDetailPage.tsx` |
| Protocoles/RFC | `protocols`, `rfcs` | `ProtocolDetailPage.tsx`, `RfcDetailPage.tsx` |
| Feature graphs | `feature-graphs` | `feature-graphs/:id` |
| Intelligence (graphe) | `projects/:slug/intelligence` (+ `/graph`, `/vector-space`) | `IntelligencePage.tsx`, `IntelligenceGraphPage.tsx`, `VectorSpaceExplorer.tsx` |
| Workspace | `workspace` → `WorkspaceDetailPage.tsx` | — |

### Transverses (hors domaine)

`/login`, `/auth/callback`, `/setup` (wizard), `/workspace-selector`, `/settings`, `/vault`, `/providers`, `/today` (global), `/pipelines`, `/triggers`, `/sharing`, `/neural-routing`, `/mcp-federation`, `/admin`, `/trajectory`, `/architecture`, `/documents`, `/deployments` — plus redirects legacy (`/plans`, `/tasks`, `/notes`, … vers workspace).

## 2. Couche réseau

Un seul client HTTP : `src/services/api.ts` → `api.get/post/put/patch/delete/blob` (+ `apiAuth` sur base `/auth`, + `apiRequest` bas niveau). Token Bearer + refresh-on-401, retry au démarrage Tauri.

- **427 appels REST** (`api.<verbe>` hors tests, tous fichiers) : 195 `get<`, 103 `post<`, 30 `post(`, 30 `delete(`, 19 `delete<`, 23 `put<`, 17 `patch<`, 5 `get(`, 2 `put(`, 3 `blob(` + 4 `apiAuth.*`.
- **51 fichiers de service** dans `src/services/` (39+ consomment `api`).
- Endpoints construits par template literal (`/tasks/${id}`, `/personas/${id}/skills`, …) ; ~29 littéraux `/api/…` statiques retrouvés.

### Les 3 canaux temps réel (WebSocket) — 3 endpoints distincts

| Endpoint | Fichier client | Transporte |
|---|---|---|
| `/ws/chat/:sessionId` | `src/services/chatWebSocket.ts` (canonique) + `src/hooks/runner/useConversationWs.ts` (re-lecture runner) | le flux de conversation : replay par `last_seq`, deltas, tool use/result, permission, statut, erreurs. **Remplace EventSource/SSE + REST sendMessage/interrupt** (commentaire en tête de `chatWebSocket.ts`). |
| `/ws/activity` | `src/services/activityWs.ts` | Live Activity Hub : événements d'activité filtrables (projet, catégories, statuts), curseur `last_seq`, `lag_dropped` sur trou. |
| `/ws/events` | `src/services/eventBus.ts` | bus partagé : événements CRUD, événements graphe (`GraphEvent`), et relais de trames chat pour les sessions surveillées (`?sessions=`, max 64). Consommé aussi par `useActivationWebSocket.ts` et `useGraphWebSocket.ts` (intelligence). |

SSE : **abandonné** au profit du WS (mention unique dans `chatWebSocket.ts:4` « Replaces EventSource (SSE)… »).
Authentification : cookie `refresh_token` (navigateur) ou ticket WS passé en query (Tauri), validée **avant** l'upgrade (`auth_ok`/`auth_error` comme premier message). Adaptateur `src/services/wsAdapter.ts` unifie navigateur/Tauri (`@tauri-apps/plugin-websocket`).

## 3. Le chat — consommation du flux et sensibilité aux noms de trames

- Client : `src/services/chatWebSocket.ts` (reconnexion backoff, watchdog stall, `lastEventSeq`, replay).
- Types : `src/types/chat.ts` — union `ChatEvent` (source de vérité côté UI).
- **63 `type:` distincts** dans `chat.ts` (grep), dont :
  - **52 événements serveur** (variants `ChatEvent`) : `assistant_text`, `stream_delta`, `tool_use`, `tool_result`, `permission_request`, `system_init`, `session_error`, `team_message`, `secret_request`, `workflow`, `provider_notice`, `background_output`, `compaction_*`, `session_closed`, `session_dormant`, `replay_complete`, `auth_ok`, `auth_error`, `events_lagged`, `partial_text`, …
  - **9 messages client** (`WS_CLIENT_MESSAGE_TYPES`, `chat.ts:1348`) : `user_message`, `queue_op`, `interrupt`, `cancel_tools`, `permission_response`, `input_response`, `set_permission_mode`, `set_model`, `set_auto_continue`.
  - **6 trames de contrôle transport** (`CHAT_CONTROL_FRAME_TYPES`, `chat.ts:1361`) : `partial_text`, `replay_complete`, `events_lagged`, `session_dormant`, `auth_ok`, `auth_error`.

**Ce qui casserait si les noms changeaient** : le frontend vérifie les noms de trames **au niveau champ**, pas seulement au niveau variante. `src/services/chatContract.ts` contient `CHAT_EVENT_FIELDS` (table exhaustive `type → {champ: required|optional}`, liée au compilateur par `satisfies`) et compare chaque trame vendored du backend (`src/services/__fixtures__/chat-contract/`, recopiée par `scripts/sync-chat-contract.mjs` avec SHA256) contre cette table. Le test `chatContract.test.ts` **refuse** : un champ émis par le backend non typé côté front, une variante front sans frame d'exemple, un champ `required` côté front absent sur le fil. → un renommage de trame ou un champ ajouté au backend **casse la CI frontend** (portillon de contrat), et côté runtime toute trame inconnue n'est pas rendue (aucun fallback silencieux).

## 4. Composants réutilisables vs spécifiques

- **64 composants UI** (`src/components/ui/`, incl. `Button, Input, Textarea, Select, Dialog, Badge, Surface, Toast, Spinner, Switch, EntityRow, EmptyState, Pagination, TabLayout, Status, FormDialog, Dropdown, Tooltip, …`).
- **360 composants de domaine** (hors `ui/`). Ratio réutilisable/spécifique ≈ **64 / 360 (~15 %)**.

### Top 10 composants UI les plus importés (compte d'imports depuis le barrel `@/components/ui` + chemin direct, hors tests)

| Rang | Composant | Importé par |
|---|---|---|
| 1 | `Button` | 82 fichiers |
| 2 | `Input` | 39 |
| 3 | `Textarea` | 24 |
| 4 | `Select` | 23 |
| 5 | `EmptyState` | 18 |
| 6 | `Switch` | 12 |
| 7 | `ErrorState` | 12 |
| 8 | `MetaLine` | 11 |
| 9 | `EntityRow` | 10 |
| 10 | `CostDisplay` | 10 |

Ce sont les candidats naturels à une bibliothèque partagée (déjà regroupés dans `ui/`, exportés via `ui/index.ts`).

## 5. Formulaires et tableaux — manuels vs schématisés (point central)

- **26 fichiers de formulaire** dans `src/components/forms/` (hors tests) : `Create*Form` (15), `Edit*Form` (7), plus `DecisionForms.tsx`, `NoteForms.tsx`, `ImportSkillForm.tsx`, `AutoBuildFeatureGraphForm.tsx`, `ProjectProfileField.tsx`.
- **Tous écrits à la main** : chaque formulaire est un composant React avec ses `useState`, sa validation et ses champs (`Input`/`Textarea`/`Select`) montés un par un (ex. `CreateTaskForm.tsx`, `EditPersonaForm.tsx`).
- **Aucune génération par schéma** : `package.json` ne contient **ni zod, ni react-hook-form, ni formik, ni yup, ni ajv, ni react-jsonschema** (`grep -iE "zod|formik|hook-form|yup|react-jsonschema|ajv" package.json` → vide). Pas de `FormSchema`/`FieldDescriptor`/`EntityDescriptor` côté front.
- **Ratio formulaires manuels / schématisés = 26 / 0** (100 % manuels).
- Les formulaires sont néanmoins **typés** contre des types de requête partagés (`AutoBuildFeatureGraphRequest`, `CreatePersonaRequest`, …) importés de `@/types` — mais ces types sont des miroirs manuels, pas des descripteurs exécutables.
- `FormDialog.tsx` (`ui/`) n'est qu'une coquille de dialogue, pas un moteur de schéma (utilisé par 4 formulaires).

### Tableaux

- `<table>` natif : **6 occurrences** seulement. Les « tables » de données sont majoritairement des listes de composants (`EntityRow`, `EntityGrid`, `WindowedList`, `Kanban`, `FilterBar` + `Pagination`). **36 fichiers** utilisent `<table>`/`Table`/`WindowedList`/`EntityGrid`/`Kanban`. → pas de grille de données déclarative partagée (type `DataGrid` générique) : chaque liste de domaine assemble ses colonnes/rows à la main.

**Conséquence pour une UI engendrée depuis les descripteurs d'entités** : les formulaires **existent déjà tous à la main** et devraient être soit remplacés, soit — plus vraisemblablement — **cohabiter** : le générateur fournirait le cas nominal (create/edit d'une entité dont le descripteur est déclaré), pendant que les formulaires à logique spécifique (wizard setup, import skill, auto-build de graphe, persona builder) restent écrits à la main.

## 6. Le graphe 3D — `src/components/intelligence/graph3d/`

- **54 fichiers non-test** (`*.ts`/`*.tsx` hors `*.test.*`), **19 654 LOC** non-test.
- **~150 fichiers au total** (la moitié sont des `*.test.ts`). Dépendances internes denses : `linkVascularRenderer` (importé 19×), `linkMagnitude` (12×), `nodePoints` (11×), `linkSegments`/`linkRouting`/`linkFamily`/`linkCurve` (8×) — un sous-système « vascular links » très ramifié.
- Dépendances externes : `three` / `react-force-graph-3d` dans **15 fichiers** ; imports vers le reste de l'app limités (`@/types/intelligence` ×6, `@/i18n` ×5, `@/constants/intelligence` ×2, `@/atoms/intelligence` ×2, `@/services/eventBus` ×1, `@/hooks/useLiveAgents` ×1, `@/constants/graphPalette` ×1).
- **Isolable en module autonome ?** Fortement couplé en interne mais **faiblement couplé à l'extérieur** (types + i18n + atoms + eventBus). Importé par seulement 5 fichiers hors graph3d : `atoms/intelligence.ts`, `IntelligenceGraphPage.tsx`, `WorkspaceGraphPage.tsx`, `UnifiedGraphSection.tsx`, `workspaceBackdrop.ts`. → extraction faisable en paquet à part, moyennant l'emport de `src/three/` (10 fichiers : `math3d`, `sceneV2`, `sdfScene`, …) et des constantes de palette. `vendor/three-iso/` (rendu iso/chorégraphie) est un second îlot, distinct du graphe.

## 7. Dépendances au contrat backend

- **Copie manuelle, pas de génération.** Aucun `openapi`/`ts-rs`/`specta`/`codegen` actif côté front (`grep` sur `scripts/` et `src/` → seuls `sync-chat-contract.mjs` et `sync-iso-lib.mjs` copient des artefacts, pas des types TS).
- Les types sont des **miroirs manuels** des DTO Rust, explicitement documentés :
  - `src/types/activity.ts:2` « hand mirror of the Rust DTOs … kept aligned by hand because the Rust side is the contract ».
  - `src/types/attention.ts:4` « Mirror of the backend DTOs (`src/api/attention.rs`) … checked with the shared `enums.json` fixture ».
  - `src/types/chat.ts` : `// matches backend PermissionConfig struct`, `// field for field the backend's chat::types::SpawnedBy`, etc.
- Seul garde-fou automatisé : **fixtures vendored + test de contrat** (`chatContract.ts` + `sync-chat-contract.mjs` + `chatContract.test.ts`) pour le canal chat. Les autres domaines (projets, plans, tâches, notes, personas…) n'ont **pas** de test de contrat champ-à-champ : le drift est détecté au runtime seulement.

## 8. Ce qui s'engendrerait depuis un descripteur vs ce qui doit rester écrit à la main

**S'engendrerait bien** (formulaires CRUD monotones, déjà purement additifs et typés contre un DTO) : les 15 `Create*Form` + 7 `Edit*Form` simples (`CreateProjectForm`, `EditProjectForm`, `CreateMilestoneForm`, `EditMilestoneForm`, `CreateNoteForm`, `CreateDecisionForm`, `CreateTaskForm`, `EditTaskForm`, `CreatePlanForm`, `EditPlanForm`, `CreateStepForm`, `EditStepForm`, `CreateWorkspaceForm`, `EditWorkspaceForm`, `CreateReleaseForm`, `CreateResourceForm`, `CreateComponentForm`, `CreateConstraintForm`, `CreateSkillForm`, …). Un descripteur d'entité (champs + types + labels i18n) les remplacerait un à un.

**Doit rester écrit à la main** : le wizard setup (`src/pages/setup/`, 9 pages à étapes non-CRUD), `ImportSkillForm`, `AutoBuildFeatureGraphForm` (slider de profondeur + relations), `EditPersonaForm` (graphe de relations skills/protocoles/fichiers), `DecisionForms`/`NoteForms` (multi-formes par entité), le chat (`src/components/chat/`, 107 composants — flux temps réel), le graphe 3D (19,6k LOC, géométrie/vascular), le composeur de références `@`/`#` (`src/refs/`), et toute liste de domaine à colonnes non génériques.

## 9. Angles morts (obligatoire)

1. **Contrat REST non verrouillé** : seuls les événements chat sont checkés champ-à-champ (`chatContract`). Les DTO REST (projets/plans/tâches/notes/personas/…) n'ont pas de test de contrat → drift silencieux possible entre `src/types/*` et le Rust.
2. **Le compte de 63 `type:`** mélange événements serveur, messages client et trames de contrôle dans le même fichier `chat.ts` ; la ventilation exacte (52/9/6) repose sur les deux tables `WS_CLIENT_MESSAGE_TYPES` et `CHAT_CONTROL_FRAME_TYPES`, pas sur une lecture exhaustive de chaque variant.
3. **427 appels REST** est un compte de *call-sites* (`api.<verbe>`), pas d'*endpoints distincts* : plusieurs call-sites partagent le même endpoint templaté.
4. **Top 10 composants** : le compte d'imports ne mesure que l'usage interne au frontend ; il ne mesure ni l'usage par d'autres dépôts ni l'adéquation sémantique à une bibliothèque.
5. **`src/vendor/three-iso/`** (rendu iso, `sync-iso-lib.mjs`) n'a pas été cartographié en profondeur : îlot séparé du graphe 3D, non inclus dans les 19,6k LOC.
6. **La CI frontend** n'a pas été inventoriée (`.github/`), seuls les scripts `scripts/` l'ont été.
7. Le **routage de modèle** et les **atoms** (jotai, 21 fichiers) n'ont pas été cartographiés individuellement — seule leur existence est comptée.
