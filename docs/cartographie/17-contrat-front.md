# Contrat frontend (lecture seule) — les DTO REST copiés à la main

Suite de `15-frontend.md`. Frontend cartographié : `/Users/triviere/projects/project-orchestrator/frontend` ; backend : `/Users/triviere/projects/project-orchestrator/backend` (symlink). Méthode : grep/find uniquement, aucun build, aucun git. Chaque chiffre est un compte obtenu par commande, pas une impression.

## 0. Le problème, mesuré

- **473** sites d'appel `api(?:Auth)?.<méthode>` dans `frontend/src` (`grep -rEo "\bapi(?:Auth)?\.(get|post|patch|put|delete|blob)" src --include="*.ts" --include="*.tsx" | wc -l`), dont **361** génériquement typés (`api.get<T>`, même grep avec `<`).
- Ces 473 appels sont répartis sur **40** fichiers de `frontend/src/services/` (`grep -rl … | wc -l`).
- **Correction du postulat du brief** : ils ne sont PAS dans « un seul client `services/api.ts` ». `frontend/src/services/api.ts` fait **330 lignes** et ne contient **aucun DTO** : c'est le transport (`fetch`, bearer, refresh, `NonJsonResponseError`) + **7 méthodes** (`get/post/patch/put/delete/blob` sur `/api`) et **3** sur `/auth` (`apiAuth`). Les appels métier vivent dans les 40 fichiers de services.
- **DTO TypeScript** : **507** déclarations `export interface|type` dans `frontend/src/types/` (20 fichiers) ; **1 917** déclarations `interface`/`type` hors tests sur tout `src` (borne haute : inclut les props de composants).
- **Types Rust de fil** : **1 202** dérivations `#[derive(…Serialize…)]` dans `backend/src` (`grep -rE "derive\([^)]*Serialize" src --include="*.rs" | wc -l`), dont **246** dans `src/api/`, **133** `pub struct` + **25** `pub enum` dans `src/neo4j/models.rs`.
- **Aucun outil de génération aujourd'hui** : ni `ts-rs`, ni `specta`, ni `utoipa`/`paperclip`/`openapi` dans `backend/Cargo.toml` ni `crates/*/Cargo.toml` ; ni `openapi-typescript` dans `frontend/package.json`.
- **Un seul garde-fou de contrat** : le chat — `frontend/src/services/chatContract.ts` + `scripts/sync-chat-contract.mjs` + `__fixtures__/chat-contract/` + `__tests__/chatContract.test.ts`. Tout le REST hors chat dérive en silence.

## 1. Correspondance TS ↔ Rust, et divergences réelles

16 paires comparées champ par champ. « OK » = mêmes champs ; « DIVERGENT » = un champ diffère (manquant, nullabilité, ou valeur d'enum).

| TS (frontend/src/types/index.ts) | Rust (backend/src) | Verdict |
|---|---|---|
| `TaskStatus` `index.ts:12` = 5 valeurs, sans `interrupted` | `neo4j/models.rs:962` enum `TaskStatus` à **6** variantes, `Interrupted` à `:972` | **DIVERGENT (cassant)** : un statut `"interrupted"` émis n'est pas dans l'union TS |
| `Task` `index.ts:109` | `neo4j/models.rs:908` `TaskNode` | **DIVERGENT** : TS n'a ni `interrupted_at`/`interrupted_reason` (`:918, :921`), ni `due_date`/`estimated_minutes` (`:937, :939`), ni `frustration_score` (`:947`), ni `execution_context`/`persona`/`prompt_cache` (`:950, :953, :956`) |
| `Note` `index.ts:367` | `notes/models.rs:798` `Note` | **DIVERGENT (cassant sur nullabilité)** : TS `project_id: string` non-null (`index.ts:369`) vs Rust `Option<Uuid>` (`:802`) — une note globale sort `null` ; TS `scope?` optionnel vs Rust `scope: NoteScope` obligatoire (`:810`) ; TS ignore ~21 champs (`author_kind`, `confidence`, `human_confirmed_at`/`by`, `last_confirmed_by`, `energy`, `last_activated`, `reactivation_count`, `last_reactivated`, `freshness_pinged_at`, `activation_count`, `scar_intensity`, `memory_horizon`, `changes`, `assertion_rule`, `last_assertion_result`, `revision_reason`, `valid_from`/`until`, `revision`, `sharing_consent`) |
| `Project` `index.ts:85` | `api/project_handlers.rs:41` `ProjectResponse` | **DIVERGENT** : TS n'a ni `file_count` ni `plan_count` (`:51-52`, toujours envoyés) ; `profile` TS optionnel vs Rust obligatoire (`:47`) |
| `Plan` `index.ts:98` | `neo4j/models.rs:865` `PlanNode` | **DIVERGENT** : TS n'a ni `execution_context` ni `persona` (`:877, :880`) |
| `Workspace` `index.ts:67` | `api/workspace_handlers.rs:76` `WorkspaceResponse` | **DIVERGENT** : `metadata` TS optionnel vs Rust `serde_json::Value` toujours présent (`:83`) |
| `Decision` `index.ts:190` | `neo4j/models.rs:1130` `DecisionNode` | **DIVERGENT** : TS n'a ni `embedding`/`embedding_model` (`:1144, :1147`) ni `scar_intensity` (`:1151`) ni `sharing_consent` (`:1156`) |
| `Step` `index.ts:179` | `neo4j/models.rs:1052` `StepNode` | **DIVERGENT** : TS n'a ni `execution_context` ni `persona` (`:1063, :1066`) |
| `Milestone` `index.ts:317` | `neo4j/models.rs:1609` `MilestoneNode` | OK |
| `Release` `index.ts:305` | `neo4j/models.rs:1424` `ReleaseNode` | OK |
| `StepStatus` `index.ts:14` | `neo4j/models.rs:1072` | OK (4 valeurs) |
| `MilestoneStatus` `index.ts:16` | `neo4j/models.rs:1623` | OK (5 valeurs) |
| `DecisionStatus` `index.ts:33` | `neo4j/models.rs:1093` | OK (4 valeurs) |
| `ReleaseStatus` `index.ts:18` | `neo4j/models.rs:1441` | OK (4 valeurs) |
| `Persona` `index.ts:988` | `neo4j/models.rs:2938` `PersonaNode` | comparé partiellement (champs d'exécution présents des deux côtés ; agrégats `energy`/`cohesion`/`success_rate` non rapprochés ligne à ligne) |

**Bilan : 10 paires divergentes sur 16 comparées.** Deux sont cassantes — `TaskStatus` sans `interrupted` (`index.ts:12`) et `Note.project_id` non-nullable (`index.ts:367`). Les huit autres sont du **sous-typage additif** : le backend envoie des champs que le frontend ignore (bénin tant qu'on ne les lit pas, mais un futur `read` de `task.due_date` ferait `undefined` en silence).

## 2. Conception de la génération — trois options, comptées

### A. `ts-rs` (`#[derive(TS)]` dans le backend)
- **Surface** : annoter les **1 202** types `Serialize` de `backend/src` (au minimum les ~**404** de fil : 246 de `src/api/` + 133 structs + 25 enums de `models.rs`).
- **Coût** : chaque type → `#[derive(TS)]` + `#[ts(rename_all = "camelCase")]` (serde est `snake_case`, le TS est `camelCase`) ; surcharges pour `DateTime<Utc>` (feature `chrono` + `#[ts(type = "string")]`), `Uuid` (feature `uuid`), `NaiveDate` (feature), `serde_json::Value` (`#[ts(type = "unknown")]`), enums à payload (`#[ts(export)]`). + 1 dépendance proc-macro + build plus lent.
- **Atout** : aucune source de vérité ajoutée ; une passe régénère tout.

### B. OpenAPI (handlers → `openapi-typescript`)
- **Surface** : aucun inventaire de schémas n'existe (vérifié : zéro `utoipa`/`paperclip`). Il faudrait déclarer **~650 schémas** (517 `pub struct` de `src/api/` + 133 de `models.rs` + 25 enums) **et** annoter de routes les **~50** fichiers handlers.
- **Coût** : le plus élevé. `openapi-typescript` remplacerait en plus le client typé → churn sur les 40 fichiers de services et 361 appels typés.

### C. xtask maison (lit la source de vérité serde, écrit les `.ts`)
- **Surface** : une sous-commande `ts-dto` dans `backend/xtask/src/` (à côté de `pins.rs`, `in_service.rs`, `storage_boundary.rs`), même contrat que `main.rs:1-46` et `pins.rs:1-13` : **std uniquement, `run(args) -> i32`, exit 0 = identique / 1 = divergence constatée / 2 = invérifiable, n'écrit que par sa sous-commande**.
- Elle lit un **manifest** de DTO (comme `pins.rs` lit `Cargo.toml`), extrait textuellement `pub struct X { pub field: Type }` et `pub enum E` depuis `backend/src/`, applique une table Rust→TS (`String`→`string`, `Option<T>`→`T?`, `Vec<T>`→`T[]`, `DateTime<Utc>`/`Uuid`→`string`, `f64`/`i32`/`u32`/`i64`/`usize`→`number`, `serde_json::Value`→`unknown`, enum → union de littéraux `snake_case`), écrit `frontend/src/types/generated/*.ts`.
- **Coût** : **0 annotation, 0 schéma à écrire** (les types serde existent déjà) ; ~1 sous-commande + 1 manifest + 1 table de mapping. Le moins cher.
- **Limite honnête** : un extracteur std ne couvre que les DTO plats (structs de primitives/`Option`/`Vec`/structs imbriqués/enums sans payload) ; `serde(flatten)`, enums taggés, génériques restent hors périmètre — d'où un premier lot borné.

### Verdict
**L'option C (xtask maison) coûte le moins pour ce dépôt.** Comptes : **0** annotation côté backend (contre ~1 202 pour ts-rs) et **0** schéma à écrire (contre ~650 pour OpenAPI), en réutilisant exactement deux mécanismes déjà en place : le style `run(args)->i32` de `pins.rs`, et le modèle `--check` de `scripts/sync-chat-contract.mjs`. ts-rs n'est pertinent que si l'on veut couvrir *tous* les types d'un coup ; OpenAPI n'est pertinent que si le contrat doit devenir un artefact versionné consommé par des tiers — ce qui n'est pas le besoin mesuré ici (un frontend unique, un backend unique).

## 3. Premier lot (générer d'abord ceux qui divergent ET sont les plus lus)

Critère : divergence réelle × fréquence d'usage. Fréquences mesurées (`grep -rE ":\s*T\b|<$T>|\b$T\b" src …` hors tests) : `Project` **364**, `Plan` **342**, `Note` **169**, `Task` **151**.

| DTO | Pourquoi en premier |
|---|---|
| `Note` | diverge le plus (nullabilité + ~21 champs ignorés) ; 169 usages |
| `Task` + `TaskStatus` | l'enum cassant `interrupted` + 8 champs ; 151 usages |
| `Project` | le plus lu (364) ; manque `file_count`/`plan_count` |
| `Plan` | 2ᵉ le plus lu (342) ; manque `execution_context`/`persona` |

**Commandes exactes** (le manifest liste ces 4 DTO + leurs enums) :
- régénérer : `cargo xtask ts-dto --write`  → écrit `frontend/src/types/generated/{note,task,project,plan}.ts`
- vérifier : `cargo xtask ts-dto --check`  → exit **1** si le généré diffère du commité

## 4. Garde-fou CI — une dérive devient un échec de CI

Sur le modèle exact de `chatContract.ts` (backend génère → front vendorise + checksum → test rejoue), transposé au contrat d'exit de `pins.rs` :

1. `ts-dto --check` régénère **en mémoire** et diff **octet à octet** contre `frontend/src/types/generated/*.ts` commités ; **exit 1** sur toute dérive, **2** si le manifest/backend est illisible, **0** si identique (`main.rs:1-46`).
2. Le job CI backend qui exécute déjà `cargo xtask pins` / `in-service` / `storage-boundary` ajoute `cargo xtask ts-dto --check`. Une dérive devient un **échec de CI**, pas une découverte à l'exécution — symétrique du `--check` de `sync-chat-contract.mjs`.
3. Comme `chatContract.test.ts`, le diff porte sur le **produit généré** (bytes), jamais sur une comparaison de noms à la main : un champ renommé côté Rust fait échouer le `--check` tant que personne n'a relancé `--write` **et** revu le diff.

## 5. Angles morts

1. **Divergence statique, pas oracle runtime.** Mes 10 divergences sont des comparaisons de *jeux de champs* (sous-typage, nullabilité, enum). Un renommage simultané Rust *et* TS (même forme, champ différent) n'est pas détectable par comparaison de champs — seul un replay de fixtures live (comme les frames vendorisés de `chatContract`) le voit. Le générateur de l'option C ferme ce trou en rendant le TS *dérivé*, pas en le vérifiant à l'exécution.
2. **Deux divergences cassantes, huit bénignes.** Seuls `TaskStatus` sans `interrupted` (`index.ts:12`) et `Note.project_id` non-nullable (`index.ts:367`) cassent réellement ; les huit autres sont des champs additifs ignorés (cassants seulement au premier `read`).
3. **Le brief a dérivé.** « 427 sites dans un seul `api.ts` » ne décrit plus le dépôt : `api.ts` fait 330 lignes de transport pur (0 DTO), et les **473** appels mesurés vivent dans **40** fichiers. 427 mesure un état antérieur ; mon chiffre est 473 (avec 361 typés).
4. **Bornes de comptage des DTO.** 507 = déclarations exportées de `types/` (dont des unions/alias non-DTO) ; 1 917 = toute déclaration hors tests (dont les props de composants). Le vrai nombre de « DTO de fil » est entre les deux et n'est pas mesurable exactement sans schéma généré — c'est précisément ce que l'option C fournirait.
5. **Le parser std de l'option C est borné.** `serde(flatten)`, enums à payload, génériques et alias de type ne sont pas couverts par un extracteur texte std ; les 1 202 types `Serialize` ne sont donc **pas** tous générables. Le premier lot (4 DTO plats) est choisi pour cette raison.
6. **Deux checkouts, deux branches.** Ce document vit dans `worktrees/obrain-default` (branche `consolidation/obrain-default`) ; le code Rust comparé est lu via le symlink `backend` → `.openclaw/workspace/skills/project-orchestrator`, qui est **en retard** sur obrain-default (le `xtask` y a déjà `containment.rs`/`parity.rs`/`path_pins.rs` ; `PersonaNode` à `:2938` vs `:2895`). Les numéros de ligne des structs comparées ont été **re-vérifiés un à un dans obrain-default** (ce sont ceux cités) ; la re-vérification sur la branche cible est faite.
7. **Lecture seule, aucune exécution.** La table Rust→TS (section 2.C) est un principe, pas un générateur testé ; aucun build, aucun `--check` réel n'a tourné (consigne de la tâche).
