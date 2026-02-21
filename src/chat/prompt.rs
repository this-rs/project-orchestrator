//! System prompt builder for the chat agent
//!
//! Two-layer architecture:
//! 1. Hardcoded base prompt (~2500 words) — protocols, data model, git workflow, statuses, best practices
//! 2. Dynamic context — oneshot Opus via send_and_receive() analyzes user request + Neo4j data
//!    to build a tailored contextual section (<500 words). Programmatic fallback if oneshot fails.

/// Base system prompt — hardcoded protocols, data model, git workflow, statuses, best practices.
/// MCP-first directive: the agent uses EXCLUSIVELY the Project Orchestrator MCP tools.
pub const BASE_SYSTEM_PROMPT: &str = r#"# Agent de développement — Project Orchestrator

## 1. Identité & rôle

Tu es un agent de développement autonome intégré au **Project Orchestrator**.
Tu disposes de **130+ outils MCP** couvrant le cycle de vie complet d'un projet : planification, exécution, suivi, exploration de code, gestion des connaissances.

**IMPORTANT — Directive MCP-first :**
Tu utilises **EXCLUSIVEMENT les outils MCP du Project Orchestrator** pour organiser ton travail.
Tu ne dois **PAS** utiliser les features internes de Claude Code pour la gestion de projet :
- ❌ Plan mode (EnterPlanMode / ExitPlanMode) — utilise `create_plan`, `create_task`, `create_step`
- ❌ TodoWrite — utilise `update_task`, `update_step` pour suivre la progression
- ❌ Tout autre outil interne de planification

Quand on te demande de "planifier", tu crées un **Plan MCP** avec des Tasks et des Steps.
Quand on te demande de "suivre la progression", tu mets à jour les **statuts via les outils MCP**.

## 2. Modèle de données

### Hiérarchie des entités

```
Workspace
  └─ Project (codebase trackée)
       ├─ Plan (objectif de développement)
       │    ├─ Task (unité de travail)
       │    │    ├─ Step (sous-étape atomique)
       │    │    └─ Decision (choix architectural)
       │    └─ Constraint (règle à respecter)
       ├─ Milestone (jalon de progression)
       ├─ Release (version livrable)
       ├─ Note (connaissance capturée)
       └─ Commit (enregistrement git)
```

### Relations clés (avec outils MCP)

- Plan → Project : `link_plan_to_project(plan_id, project_id)`
- Task → Task : `add_task_dependencies(task_id, [dep_ids])`
- Task → Milestone : `add_task_to_milestone(milestone_id, task_id)`
- Task → Release : `add_task_to_release(release_id, task_id)`
- Commit → Task : `link_commit_to_task(task_id, commit_sha)`
- Commit → Plan : `link_commit_to_plan(plan_id, commit_sha)`
- Note → Entité : `link_note_to_entity(note_id, entity_type, entity_id)`

### Notes (base de connaissances)

- **Types** : guideline, gotcha, pattern, context, tip, observation, assertion
- **Importance** : critical, high, medium, low
- **Statuts** : active, needs_review, stale, obsolete, archived
- Attachables à : project, file, function, struct, trait, task, plan, workspace...
- Consulter avant de travailler : `get_context_notes(entity_type, entity_id)`

### Tree-sitter & synchronisation du code

- `sync_project(slug)` / `sync_directory(path)` parse le code source avec Tree-sitter
- Construit le **graphe de connaissances** : fichiers, fonctions, structs, traits, enums, imports, appels entre fonctions
- `start_watch(path)` active la synchronisation automatique sur changements fichiers
- **Sync incrémentale au commit** : `create_commit` avec `files_changed` + `project_id` déclenche automatiquement la re-sync des fichiers modifiés en arrière-plan. Pas besoin de `sync_project` après chaque commit.
- **Requis avant toute exploration de code** : si `last_synced` est absent, lancer `sync_project` en premier ; sinon, le sync au commit maintient la fraîcheur automatiquement
- Outils d'exploration disponibles après sync :
  - `search_code(query)` / `search_project_code(slug, query)` — recherche sémantique
  - `get_file_symbols(file_path)` — fonctions, structs, traits d'un fichier
  - `find_references(symbol)` — tous les usages d'un symbole
  - `get_file_dependencies(file_path)` — imports et dépendants
  - `get_call_graph(function)` — graphe d'appels
  - `analyze_impact(target)` — impact d'une modification
  - `get_architecture()` — vue d'ensemble (fichiers les plus connectés)
  - `find_trait_implementations(trait_name)` — implémentations d'un trait
  - `find_type_traits(type_name)` — traits implémentés par un type
  - `get_impl_blocks(type_name)` — blocs impl d'un type
  - `find_similar_code(code_snippet)` — code similaire

## 3. Workflow Git

### Avant de commencer une tâche

1. `git status` + `git log --oneline -5` — vérifier que l'état est propre
2. S'assurer que le working tree est propre (pas de changements non commités)
3. Se positionner sur la branche principale et pull si possible
4. Créer une branche dédiée : `git checkout -b <type>/<description-courte>`
   - Types : `feat/`, `fix/`, `refactor/`, `docs/`, `test/`

### Pendant le travail

- **Commits atomiques** : un commit = un changement logique cohérent
- Format : `<type>(<scope>): <description courte>`
  - Exemples : `feat(chat): add smart system prompt`, `fix(neo4j): handle null workspace`
- Ne jamais commiter de fichiers sensibles (.env, credentials, secrets)

### Après chaque commit

1. `create_commit(sha, message, author, files_changed)` — enregistrer dans le graphe
2. `link_commit_to_task(task_id, sha)` — lier au task en cours
3. `link_commit_to_plan(plan_id, sha)` — lier au plan (au moins le dernier commit)

## 4. Protocole d'exécution de tâche

### Phase 0 — Warm-up (OBLIGATOIRE au début de chaque conversation)

Avant tout travail, charger les connaissances pertinentes :
1. `search_notes_semantic(query)` — recherche vectorielle de notes (cosine similarity, trouve les notes sémantiquement proches même sans correspondance de mots-clés)
2. `get_context_notes(entity_type, entity_id)` — notes contextuelles pour les fichiers/fonctions concernés
3. `search_decisions(query)` — décisions architecturales passées sur le sujet
4. `search_notes(query)` — recherche BM25 complémentaire si besoin de correspondance exacte de mots-clés

Cela évite de refaire un travail déjà documenté ou de violer une convention déjà établie.

### Phase 1 — Préparation

1. `get_next_task(plan_id)` — récupérer la prochaine tâche non bloquée (priorité la plus haute)
2. `get_task_context(plan_id, task_id)` — charger le contexte complet (steps, constraints, decisions, notes, code)
3. `get_task_blockers(task_id)` — vérifier qu'il n'y a pas de bloqueurs non résolus
4. `search_decisions(<sujet>)` — consulter les décisions architecturales passées
5. `analyze_impact(<fichier>)` — évaluer l'impact avant modification
6. `update_task(task_id, status: "in_progress")` — passer la tâche en cours
7. Préparer git (branche dédiée si pas encore fait)

### Phase 2 — Exécution (pour chaque step)

1. `update_step(step_id, status: "in_progress")`
2. Effectuer le travail (coder, modifier, tester)
3. Vérifier selon le critère du step (champ `verification`)
4. `update_step(step_id, status: "completed")`
5. Si le step est devenu irrelevant : `update_step(step_id, status: "skipped")`

Si une décision architecturale est prise :
`add_decision(task_id, description, rationale, alternatives, chosen_option)`

### Phase 3 — Clôture

1. Commit final + `link_commit_to_task(task_id, sha)`
2. Vérifier les `acceptance_criteria` du task
3. `update_task(task_id, status: "completed")`
4. Vérifier la progression du plan → si toutes les tâches sont complétées : `update_plan_status(plan_id, "completed")`
5. Mettre à jour milestones/releases si applicable

## 5. Protocole de planification

Quand l'utilisateur demande de planifier un travail :

### Étape 1 — Analyser et créer le plan

1. Explorer le code existant : `search_code`, `get_architecture`, `analyze_impact`
2. `create_plan(title, description, priority, project_id)`
3. `link_plan_to_project(plan_id, project_id)`

### Étape 2 — Ajouter les contraintes

- `add_constraint(plan_id, type, description, severity)`
- Types : performance, security, style, compatibility, other

### Étape 3 — Décomposer en tâches avec steps

Pour chaque tâche :
1. `create_task(plan_id, title, description, priority, tags, acceptance_criteria, affected_files)`
2. **TOUJOURS ajouter des steps** : `create_step(task_id, description, verification)`
   - Minimum 2-3 steps par tâche
   - Chaque step doit être **actionnable** et **vérifiable**
3. `add_task_dependencies(task_id, [dep_ids])` — définir l'ordre d'exécution

### Étape 4 — Organiser le suivi

- `create_milestone(project_id, title, target_date)` + `add_task_to_milestone`
- `create_release(project_id, version, title, target_date)` + `add_task_to_release`

### Granularité attendue

**TOUJOURS descendre au niveau step.** Un plan sans steps est incomplet.

Exemple de décomposition correcte :
- Task: "Ajouter l'endpoint GET /api/releases/:id"
  - Step 1: "Ajouter la méthode get_release dans neo4j/client.rs" → vérif: "cargo check"
  - Step 2: "Ajouter le handler dans api/handlers.rs" → vérif: "cargo check"
  - Step 3: "Enregistrer la route dans api/routes.rs" → vérif: "curl test"
  - Step 4: "Ajouter le tool MCP dans mcp/tools.rs" → vérif: "test_all_tools_count passe"
  - Step 5: "Ajouter le handler MCP dans mcp/handlers.rs" → vérif: "cargo test"

## 6. Gestion des statuts

### Règles fondamentales

- Mettre à jour **EN TEMPS RÉEL**, pas en batch à la fin
- Un seul task `in_progress` à la fois par plan
- Ne **JAMAIS** marquer `completed` sans vérification
- En cas de blocage → `update_task(task_id, status: "blocked")` + note expliquant pourquoi

### Transitions valides

| Entité    | De          | Vers        | Quand                          |
|-----------|-------------|-------------|--------------------------------|
| Plan      | draft       | approved    | Plan validé et prêt            |
| Plan      | approved    | in_progress | Première tâche démarrée        |
| Plan      | in_progress | completed   | Toutes tâches complétées       |
| Task      | pending     | in_progress | Début du travail               |
| Task      | in_progress | completed   | Critères d'acceptation remplis |
| Task      | in_progress | blocked     | Dépendance non résolue         |
| Task      | blocked     | in_progress | Bloqueur résolu                |
| Task      | pending     | failed      | Impossible à réaliser          |
| Step      | pending     | in_progress | Début de l'étape               |
| Step      | in_progress | completed   | Vérification passée            |
| Step      | pending     | skipped     | Step devenu irrelevant         |
| Milestone | planned     | in_progress | Première tâche démarre         |
| Milestone | in_progress | completed   | Toutes tâches complétées       |

## 7. Bonnes pratiques

### Liaison systématique

- **TOUJOURS** lier plans aux projets, commits aux tasks, tasks aux milestones/releases
- Vérifier `get_dependency_graph(plan_id)` et `get_critical_path(plan_id)` avant de démarrer l'exécution

### Analyse d'impact avant modification

- `analyze_impact(cible)` → fichiers et symboles affectés
- `get_file_dependencies(file_path)` → imports et dépendants
- `get_context_notes(entity_type, entity_id)` → notes pertinentes (guidelines, gotchas...)

### Analyse structurelle (GDS)

Quand les données GDS (Graph Data Science) sont disponibles sur le projet :

1. **Comprendre la structure modulaire** → `get_code_communities(project_slug)`
   - Clusters Louvain de fichiers/fonctions fortement couplés
   - Chaque communauté a ses fichiers clés et métriques de cohésion
   - **Utilise ceci** : avant un refactoring, pour comprendre les frontières modulaires

2. **Évaluer la santé du codebase** → `get_code_health(project_slug)`
   - God functions (trop de connexions), fichiers orphelins (0 connexions), couplage moyen, dépendances circulaires
   - **Utilise ceci** : en début de projet, revue de code, ou priorisation de dette technique

3. **Évaluer l'importance d'un nœud** → `get_node_importance(project_slug, node_path, node_type)`
   - PageRank, betweenness centrality, bridge detection, risk level
   - Retourne un summary interprétatif (critical/high/medium/low)
   - **Utilise ceci** : avant de modifier un fichier/fonction, pour évaluer le risque de régression

**Note** : ces outils nécessitent que les métriques GDS aient été calculées. Si les résultats sont vides, le codebase n'a pas encore été analysé par GDS.

### Stratégie de recherche — MCP-first (OBLIGATOIRE)

**Règle absolue** : TOUJOURS utiliser les outils MCP d'exploration de code EN PREMIER.
N'utiliser Grep/Read/Glob qu'en dernier recours pour des chaînes littérales exactes.

Hiérarchie de recherche (du plus recommandé au moins recommandé) :

1. **Recherche exploratoire** → `search_code(query)` / `search_project_code(slug, query)`
   - Recherche sémantique MeiliSearch, cross-fichier, ranking par pertinence
   - Supporte `path_prefix` pour filtrer un sous-répertoire
   - **Utilise ceci au lieu de** : Grep pour chercher un concept, Task(Explore) pour explorer

2. **Usages d'un symbole** → `find_references(symbol)`
   - Résolution via le graphe Neo4j (imports, exports, appels)
   - Plus fiable que grep car comprend la structure du code
   - **Utilise ceci au lieu de** : Grep pour "où est utilisé X"

3. **Comprendre un flux** → `get_call_graph(function)`
   - Qui appelle cette fonction ? Qui elle appelle ?
   - **Utilise ceci au lieu de** : lire manuellement chaque fichier

4. **Avant de modifier** → `analyze_impact(target)`
   - Fichiers et symboles affectés par un changement
   - **Utilise ceci au lieu de** : deviner quels fichiers sont impactés

5. **Vue d'ensemble** → `get_architecture(project_slug?)`
   - Fichiers les plus connectés, stats langages, structure du projet
   - **Utilise ceci au lieu de** : parcourir manuellement l'arborescence

6. **Symboles d'un fichier** → `get_file_symbols(file_path)`
   - Toutes les fonctions, structs, traits, enums d'un fichier
   - **Utilise ceci au lieu de** : lire tout le fichier pour trouver les définitions

7. **Types et traits** → `find_trait_implementations(trait)` / `find_type_traits(type)` / `get_impl_blocks(type)`
   - Naviguer le système de types via le graphe

8. **Recherche de notes** → `search_notes_semantic(query)` (vectorielle) / `search_notes(query)` (BM25)
   - `search_notes_semantic` : recherche par similarité cosine via embeddings — trouve les notes conceptuellement proches même sans correspondance de mots-clés. Préférer pour les questions en langage naturel.
   - `search_notes` : recherche BM25 classique — meilleure pour les mots-clés exacts, noms de fonctions, identifiants.
   - **Utilise ceci au lieu de** : parcourir manuellement les notes ou deviner leur existence

9. **Dernier recours** → Grep/Read de Claude Code
   - UNIQUEMENT pour des chaînes littérales exactes (messages d'erreur, constantes, URLs)
   - UNIQUEMENT si les outils MCP ci-dessus ne retournent pas de résultat pertinent

### Capture des connaissances (OBLIGATOIRE)

**Règle absolue** : L'agent DOIT créer des notes pour capitaliser les connaissances découvertes.
Ne JAMAIS terminer une session sans avoir capturé les apprentissages importants.

**Quand créer une note :**
- Après avoir résolu un bug → `create_note(type: "gotcha", importance: "high")` avec la cause racine et la solution
- Après avoir découvert un pattern architectural → `create_note(type: "pattern")` avec l'explication
- Après avoir identifié une convention → `create_note(type: "guideline")` avec la règle
- Après avoir trouvé un piège/subtilité → `create_note(type: "gotcha")` avec l'avertissement
- Après avoir trouvé une astuce utile → `create_note(type: "tip")` avec l'explication

**TOUJOURS** lier la note à l'entité concernée :
```
create_note(project_id, type, content, importance, tags)
→ note_id
link_note_to_entity(note_id, "file", "src/chat/manager.rs")
link_note_to_entity(note_id, "function", "build_system_prompt")
```

**Décisions architecturales** : à chaque choix non trivial
- Documenter les alternatives considérées + la raison du choix
- `add_decision(task_id, description, rationale, alternatives, chosen_option)`

### Workspace (multi-projets)

- `get_workspace_overview(slug)` — vue d'ensemble workspace
- `create_workspace_milestone(slug, title)` — milestones cross-projets
- `get_workspace_topology(slug)` — composants et dépendances entre services
- `list_resources(slug)` — contrats API, schémas partagés
"#;

use anyhow::Result;
use chrono::{DateTime, Utc};
use std::sync::Arc;
use uuid::Uuid;

// ============================================================================
// Tool catalog — static grouping of MCP tools for meta-prompting
// ============================================================================

/// A single MCP tool reference with a concise description.
#[derive(Debug, Clone)]
pub struct ToolRef {
    pub name: &'static str,
    pub description: &'static str,
}

/// A semantic group of related MCP tools.
#[derive(Debug, Clone)]
pub struct ToolGroup {
    pub name: &'static str,
    pub description: &'static str,
    pub keywords: &'static [&'static str],
    pub tools: &'static [ToolRef],
}

/// Static catalog of all 137 MCP tools organized into 12 semantic groups.
/// Used by the oneshot Opus refinement to select relevant tools per request,
/// and by the keyword fallback when the oneshot fails.
pub static TOOL_GROUPS: &[ToolGroup] = &[
    // ── Planning (21 tools) ──────────────────────────────────────────
    ToolGroup {
        name: "planning",
        description: "Créer et gérer plans, tâches, étapes",
        keywords: &[
            "plan",
            "tâche",
            "task",
            "step",
            "étape",
            "planifier",
            "organiser",
            "dépendance",
            "priorité",
            "chemin critique",
            "bloquer",
        ],
        tools: &[
            ToolRef {
                name: "create_plan",
                description: "Créer un plan de développement",
            },
            ToolRef {
                name: "get_plan",
                description: "Détails d'un plan avec tâches",
            },
            ToolRef {
                name: "list_plans",
                description: "Lister les plans avec filtres",
            },
            ToolRef {
                name: "update_plan_status",
                description: "Changer le statut d'un plan",
            },
            ToolRef {
                name: "delete_plan",
                description: "Supprimer un plan",
            },
            ToolRef {
                name: "link_plan_to_project",
                description: "Lier un plan à un projet",
            },
            ToolRef {
                name: "unlink_plan_from_project",
                description: "Délier un plan d'un projet",
            },
            ToolRef {
                name: "create_task",
                description: "Ajouter une tâche à un plan",
            },
            ToolRef {
                name: "get_task",
                description: "Détails d'une tâche avec steps",
            },
            ToolRef {
                name: "list_tasks",
                description: "Lister les tâches avec filtres",
            },
            ToolRef {
                name: "update_task",
                description: "Mettre à jour statut/assignee",
            },
            ToolRef {
                name: "delete_task",
                description: "Supprimer une tâche",
            },
            ToolRef {
                name: "get_next_task",
                description: "Prochaine tâche non bloquée",
            },
            ToolRef {
                name: "add_task_dependencies",
                description: "Ajouter des dépendances",
            },
            ToolRef {
                name: "remove_task_dependency",
                description: "Retirer une dépendance",
            },
            ToolRef {
                name: "get_task_blockers",
                description: "Tâches bloquant celle-ci",
            },
            ToolRef {
                name: "get_tasks_blocked_by",
                description: "Tâches bloquées par celle-ci",
            },
            ToolRef {
                name: "get_dependency_graph",
                description: "Graphe de dépendances du plan",
            },
            ToolRef {
                name: "get_critical_path",
                description: "Chemin critique du plan",
            },
            ToolRef {
                name: "get_task_context",
                description: "Contexte complet pour exécution",
            },
            ToolRef {
                name: "get_task_prompt",
                description: "Prompt généré pour une tâche",
            },
        ],
    },
    // ── Steps (6 tools) ──────────────────────────────────────────────
    ToolGroup {
        name: "steps",
        description: "Sous-étapes atomiques des tâches",
        keywords: &["step", "étape", "sous-étape", "vérification", "progression"],
        tools: &[
            ToolRef {
                name: "create_step",
                description: "Ajouter un step à une tâche",
            },
            ToolRef {
                name: "list_steps",
                description: "Lister les steps d'une tâche",
            },
            ToolRef {
                name: "get_step",
                description: "Détails d'un step",
            },
            ToolRef {
                name: "update_step",
                description: "Changer le statut d'un step",
            },
            ToolRef {
                name: "delete_step",
                description: "Supprimer un step",
            },
            ToolRef {
                name: "get_step_progress",
                description: "Progression des steps",
            },
        ],
    },
    // ── Code exploration (12 tools) ──────────────────────────────────
    ToolGroup {
        name: "code_exploration",
        description: "Recherche sémantique, graphe d'appels, impact",
        keywords: &[
            "code",
            "fonction",
            "struct",
            "fichier",
            "import",
            "appel",
            "architecture",
            "symbole",
            "trait",
            "impl",
            "référence",
            "dépendance",
            "impact",
            "chercher",
            "explorer",
        ],
        tools: &[
            ToolRef {
                name: "search_code",
                description: "Recherche sémantique cross-projet",
            },
            ToolRef {
                name: "search_project_code",
                description: "Recherche scopée à un projet",
            },
            ToolRef {
                name: "get_file_symbols",
                description: "Fonctions/structs d'un fichier",
            },
            ToolRef {
                name: "find_references",
                description: "Tous les usages d'un symbole",
            },
            ToolRef {
                name: "get_file_dependencies",
                description: "Imports et dépendants",
            },
            ToolRef {
                name: "get_call_graph",
                description: "Graphe d'appels d'une fonction",
            },
            ToolRef {
                name: "analyze_impact",
                description: "Impact d'une modification",
            },
            ToolRef {
                name: "get_architecture",
                description: "Vue d'ensemble du codebase",
            },
            ToolRef {
                name: "find_similar_code",
                description: "Code similaire à un snippet",
            },
            ToolRef {
                name: "find_trait_implementations",
                description: "Types implémentant un trait",
            },
            ToolRef {
                name: "find_type_traits",
                description: "Traits implémentés par un type",
            },
            ToolRef {
                name: "get_impl_blocks",
                description: "Blocs impl d'un type",
            },
        ],
    },
    // ── Structural Analytics (3 tools) ────────────────────────────────
    ToolGroup {
        name: "structural_analytics",
        description: "Analyse structurelle GDS : communautés, santé du code, importance des nœuds",
        keywords: &[
            "communauté",
            "community",
            "louvain",
            "cluster",
            "santé",
            "health",
            "god function",
            "orphelin",
            "couplage",
            "circulaire",
            "importance",
            "pagerank",
            "betweenness",
            "centralité",
            "bridge",
            "risque",
            "analytique",
            "GDS",
        ],
        tools: &[
            ToolRef {
                name: "get_code_communities",
                description: "Communautés de code (clusters Louvain) avec fichiers clés et métriques de cohésion",
            },
            ToolRef {
                name: "get_code_health",
                description: "Rapport santé du codebase : god functions, fichiers orphelins, couplage, dépendances circulaires",
            },
            ToolRef {
                name: "get_node_importance",
                description: "Importance structurelle d'un fichier/fonction : PageRank, betweenness, bridge detection, risk level, summary interprétatif",
            },
        ],
    },
    // ── Knowledge / Notes (17 tools) ─────────────────────────────────
    ToolGroup {
        name: "knowledge",
        description: "Notes, guidelines, gotchas, patterns",
        keywords: &[
            "note",
            "guideline",
            "gotcha",
            "pattern",
            "connaissance",
            "tip",
            "observation",
            "assertion",
            "savoir",
            "contexte",
            "mémoire",
        ],
        tools: &[
            ToolRef {
                name: "create_note",
                description: "Créer une note de connaissance",
            },
            ToolRef {
                name: "get_note",
                description: "Détails d'une note",
            },
            ToolRef {
                name: "update_note",
                description: "Modifier contenu/importance/tags",
            },
            ToolRef {
                name: "delete_note",
                description: "Supprimer une note",
            },
            ToolRef {
                name: "search_notes_semantic",
                description: "Recherche vectorielle de notes (cosine similarity)",
            },
            ToolRef {
                name: "search_notes",
                description: "Recherche BM25 de notes (mots-clés)",
            },
            ToolRef {
                name: "list_notes",
                description: "Lister avec filtres type/importance",
            },
            ToolRef {
                name: "list_project_notes",
                description: "Notes d'un projet",
            },
            ToolRef {
                name: "get_context_notes",
                description: "Notes contextuelles d'une entité",
            },
            ToolRef {
                name: "get_entity_notes",
                description: "Notes directement attachées",
            },
            ToolRef {
                name: "get_propagated_notes",
                description: "Notes propagées via le graphe",
            },
            ToolRef {
                name: "link_note_to_entity",
                description: "Lier note à fichier/fonction/…",
            },
            ToolRef {
                name: "unlink_note_from_entity",
                description: "Délier une note",
            },
            ToolRef {
                name: "confirm_note",
                description: "Confirmer validité (reset staleness)",
            },
            ToolRef {
                name: "invalidate_note",
                description: "Marquer comme obsolète",
            },
            ToolRef {
                name: "supersede_note",
                description: "Remplacer par une nouvelle note",
            },
            ToolRef {
                name: "get_notes_needing_review",
                description: "Notes à vérifier",
            },
            ToolRef {
                name: "update_staleness_scores",
                description: "Recalculer la fraîcheur",
            },
            ToolRef {
                name: "update_energy_scores",
                description: "Recalculer l'énergie neuronale (decay)",
            },
            ToolRef {
                name: "search_neurons",
                description: "Recherche neuronale (spreading activation via synapses)",
            },
            ToolRef {
                name: "reinforce_neurons",
                description: "Renforcement hebbien (boost énergie + synapses)",
            },
            ToolRef {
                name: "decay_synapses",
                description: "Decay et pruning des synapses faibles",
            },
        ],
    },
    // ── Git tracking (5 tools) ───────────────────────────────────────
    ToolGroup {
        name: "git_tracking",
        description: "Enregistrer et lier les commits",
        keywords: &["commit", "git", "branche", "sha", "push"],
        tools: &[
            ToolRef {
                name: "create_commit",
                description: "Enregistrer un commit",
            },
            ToolRef {
                name: "link_commit_to_task",
                description: "Lier commit → tâche",
            },
            ToolRef {
                name: "link_commit_to_plan",
                description: "Lier commit → plan",
            },
            ToolRef {
                name: "get_task_commits",
                description: "Commits d'une tâche",
            },
            ToolRef {
                name: "get_plan_commits",
                description: "Commits d'un plan",
            },
        ],
    },
    // ── Decisions (5 tools) ──────────────────────────────────────────
    ToolGroup {
        name: "decisions",
        description: "Choix architecturaux et alternatives",
        keywords: &["décision", "choix", "alternative", "rationale", "arbitrage"],
        tools: &[
            ToolRef {
                name: "add_decision",
                description: "Enregistrer une décision",
            },
            ToolRef {
                name: "get_decision",
                description: "Détails d'une décision",
            },
            ToolRef {
                name: "update_decision",
                description: "Modifier une décision",
            },
            ToolRef {
                name: "delete_decision",
                description: "Supprimer une décision",
            },
            ToolRef {
                name: "search_decisions",
                description: "Rechercher les décisions passées",
            },
        ],
    },
    // ── Constraints (5 tools) ────────────────────────────────────────
    ToolGroup {
        name: "constraints",
        description: "Règles et contraintes des plans",
        keywords: &[
            "contrainte",
            "règle",
            "sécurité",
            "performance",
            "compatibilité",
        ],
        tools: &[
            ToolRef {
                name: "add_constraint",
                description: "Ajouter une contrainte",
            },
            ToolRef {
                name: "get_constraint",
                description: "Détails d'une contrainte",
            },
            ToolRef {
                name: "update_constraint",
                description: "Modifier une contrainte",
            },
            ToolRef {
                name: "delete_constraint",
                description: "Supprimer une contrainte",
            },
            ToolRef {
                name: "list_constraints",
                description: "Contraintes d'un plan",
            },
        ],
    },
    // ── Project management (8 tools) ─────────────────────────────────
    ToolGroup {
        name: "project_management",
        description: "CRUD projets, sync, roadmap",
        keywords: &["projet", "codebase", "sync", "roadmap", "créer projet"],
        tools: &[
            ToolRef {
                name: "list_projects",
                description: "Lister tous les projets",
            },
            ToolRef {
                name: "create_project",
                description: "Créer un nouveau projet",
            },
            ToolRef {
                name: "get_project",
                description: "Détails d'un projet par slug",
            },
            ToolRef {
                name: "update_project",
                description: "Modifier nom/description/path",
            },
            ToolRef {
                name: "delete_project",
                description: "Supprimer un projet",
            },
            ToolRef {
                name: "sync_project",
                description: "Synchroniser le codebase",
            },
            ToolRef {
                name: "get_project_roadmap",
                description: "Roadmap avec milestones",
            },
            ToolRef {
                name: "list_project_plans",
                description: "Plans d'un projet",
            },
        ],
    },
    // ── Releases & Milestones (14 tools) ─────────────────────────────
    ToolGroup {
        name: "releases_milestones",
        description: "Versions livrables et jalons",
        keywords: &[
            "release",
            "milestone",
            "version",
            "livrable",
            "jalon",
            "livraison",
            "déploiement",
            "cible",
        ],
        tools: &[
            ToolRef {
                name: "create_release",
                description: "Créer une release",
            },
            ToolRef {
                name: "get_release",
                description: "Détails d'une release",
            },
            ToolRef {
                name: "update_release",
                description: "Modifier une release",
            },
            ToolRef {
                name: "delete_release",
                description: "Supprimer une release",
            },
            ToolRef {
                name: "list_releases",
                description: "Releases d'un projet",
            },
            ToolRef {
                name: "add_task_to_release",
                description: "Lier tâche → release",
            },
            ToolRef {
                name: "add_commit_to_release",
                description: "Lier commit → release",
            },
            ToolRef {
                name: "remove_commit_from_release",
                description: "Délier commit d'une release",
            },
            ToolRef {
                name: "create_milestone",
                description: "Créer un milestone",
            },
            ToolRef {
                name: "get_milestone",
                description: "Détails d'un milestone",
            },
            ToolRef {
                name: "update_milestone",
                description: "Modifier un milestone",
            },
            ToolRef {
                name: "delete_milestone",
                description: "Supprimer un milestone",
            },
            ToolRef {
                name: "list_milestones",
                description: "Milestones d'un projet",
            },
            ToolRef {
                name: "get_milestone_progress",
                description: "Progression d'un milestone",
            },
            ToolRef {
                name: "add_task_to_milestone",
                description: "Lier tâche → milestone",
            },
            ToolRef {
                name: "link_plan_to_milestone",
                description: "Lier plan → milestone (relation directe TARGETS_MILESTONE)",
            },
            ToolRef {
                name: "unlink_plan_from_milestone",
                description: "Délier plan d'un milestone",
            },
        ],
    },
    // ── Workspace (35 tools) ─────────────────────────────────────────
    ToolGroup {
        name: "workspace",
        description: "Multi-projets, topologie, ressources partagées",
        keywords: &[
            "workspace",
            "composant",
            "ressource",
            "topologie",
            "service",
            "multi-projet",
            "cross-projet",
            "contrat",
            "API contract",
        ],
        tools: &[
            ToolRef {
                name: "list_workspaces",
                description: "Lister les workspaces",
            },
            ToolRef {
                name: "create_workspace",
                description: "Créer un workspace",
            },
            ToolRef {
                name: "get_workspace",
                description: "Détails d'un workspace",
            },
            ToolRef {
                name: "update_workspace",
                description: "Modifier un workspace",
            },
            ToolRef {
                name: "delete_workspace",
                description: "Supprimer un workspace",
            },
            ToolRef {
                name: "get_workspace_overview",
                description: "Vue d'ensemble complète",
            },
            ToolRef {
                name: "list_workspace_projects",
                description: "Projets du workspace",
            },
            ToolRef {
                name: "add_project_to_workspace",
                description: "Ajouter un projet",
            },
            ToolRef {
                name: "remove_project_from_workspace",
                description: "Retirer un projet",
            },
            ToolRef {
                name: "search_workspace_code",
                description: "Recherche cross-projets",
            },
            ToolRef {
                name: "list_workspace_milestones",
                description: "Milestones cross-projets",
            },
            ToolRef {
                name: "create_workspace_milestone",
                description: "Créer milestone workspace",
            },
            ToolRef {
                name: "get_workspace_milestone",
                description: "Détails milestone workspace",
            },
            ToolRef {
                name: "update_workspace_milestone",
                description: "Modifier milestone workspace",
            },
            ToolRef {
                name: "delete_workspace_milestone",
                description: "Supprimer milestone workspace",
            },
            ToolRef {
                name: "add_task_to_workspace_milestone",
                description: "Lier tâche → ws milestone",
            },
            ToolRef {
                name: "link_plan_to_workspace_milestone",
                description: "Lier plan → ws milestone (TARGETS_MILESTONE)",
            },
            ToolRef {
                name: "unlink_plan_from_workspace_milestone",
                description: "Délier plan d'un ws milestone",
            },
            ToolRef {
                name: "get_workspace_milestone_progress",
                description: "Progression ws milestone",
            },
            ToolRef {
                name: "list_all_workspace_milestones",
                description: "Tous les ws milestones",
            },
            ToolRef {
                name: "list_resources",
                description: "Ressources partagées",
            },
            ToolRef {
                name: "create_resource",
                description: "Créer une ressource",
            },
            ToolRef {
                name: "get_resource",
                description: "Détails d'une ressource",
            },
            ToolRef {
                name: "update_resource",
                description: "Modifier une ressource",
            },
            ToolRef {
                name: "delete_resource",
                description: "Supprimer une ressource",
            },
            ToolRef {
                name: "link_resource_to_project",
                description: "Lier ressource → projet",
            },
            ToolRef {
                name: "list_components",
                description: "Composants du workspace",
            },
            ToolRef {
                name: "create_component",
                description: "Créer un composant",
            },
            ToolRef {
                name: "get_component",
                description: "Détails d'un composant",
            },
            ToolRef {
                name: "update_component",
                description: "Modifier un composant",
            },
            ToolRef {
                name: "delete_component",
                description: "Supprimer un composant",
            },
            ToolRef {
                name: "add_component_dependency",
                description: "Dépendance entre composants",
            },
            ToolRef {
                name: "remove_component_dependency",
                description: "Retirer dépendance composant",
            },
            ToolRef {
                name: "map_component_to_project",
                description: "Lier composant → projet",
            },
            ToolRef {
                name: "get_workspace_topology",
                description: "Graphe composants + dépendances",
            },
        ],
    },
    // ── Sync & Admin (7 tools) ───────────────────────────────────────
    ToolGroup {
        name: "sync_admin",
        description: "Synchronisation code et administration",
        keywords: &["sync", "watch", "watcher", "meilisearch", "index", "admin"],
        tools: &[
            ToolRef {
                name: "sync_directory",
                description: "Sync manuelle d'un répertoire",
            },
            ToolRef {
                name: "start_watch",
                description: "Démarrer le file watcher",
            },
            ToolRef {
                name: "stop_watch",
                description: "Arrêter le file watcher",
            },
            ToolRef {
                name: "watch_status",
                description: "Statut du file watcher",
            },
            ToolRef {
                name: "get_meilisearch_stats",
                description: "Stats de l'index Meilisearch",
            },
            ToolRef {
                name: "delete_meilisearch_orphans",
                description: "Nettoyer docs orphelins",
            },
            ToolRef {
                name: "cleanup_cross_project_calls",
                description: "Supprimer les CALLS cross-projet",
            },
        ],
    },
    // ── Chat (5 tools) ───────────────────────────────────────────────
    ToolGroup {
        name: "chat",
        description: "Sessions de conversation et messages",
        keywords: &["chat", "session", "conversation", "message", "historique"],
        tools: &[
            ToolRef {
                name: "chat_send_message",
                description: "Envoyer un message (non-streaming)",
            },
            ToolRef {
                name: "list_chat_sessions",
                description: "Lister les sessions chat",
            },
            ToolRef {
                name: "get_chat_session",
                description: "Détails d'une session",
            },
            ToolRef {
                name: "delete_chat_session",
                description: "Supprimer une session",
            },
            ToolRef {
                name: "list_chat_messages",
                description: "Historique des messages",
            },
        ],
    },
    // ── Feature Graphs (6 tools) ────────────────────────────────────
    ToolGroup {
        name: "feature_graphs",
        description: "Sous-graphes de code par feature, réutilisables entre sessions",
        keywords: &[
            "feature",
            "graphe",
            "sous-graphe",
            "subgraph",
            "feature graph",
            "exploration",
            "call graph",
            "auto-build",
            "entité",
        ],
        tools: &[
            ToolRef {
                name: "create_feature_graph",
                description: "Créer un feature graph nommé",
            },
            ToolRef {
                name: "get_feature_graph",
                description: "Détails avec entités incluses",
            },
            ToolRef {
                name: "list_feature_graphs",
                description: "Lister les feature graphs",
            },
            ToolRef {
                name: "add_to_feature_graph",
                description: "Ajouter une entité (file/function/struct)",
            },
            ToolRef {
                name: "auto_build_feature_graph",
                description: "Construire auto depuis un point d'entrée",
            },
            ToolRef {
                name: "delete_feature_graph",
                description: "Supprimer un feature graph",
            },
        ],
    },
    // ── Implementation Planner (1 tool) ─────────────────────────────
    ToolGroup {
        name: "implementation_planner",
        description: "Analyser le graphe de connaissances pour planifier une implémentation",
        keywords: &[
            "plan",
            "implementation",
            "dag",
            "phases",
            "dépendances",
            "ordre",
            "planifier",
            "impact",
        ],
        tools: &[ToolRef {
            name: "plan_implementation",
            description: "Produire un DAG de phases d'implémentation à partir du graphe",
        }],
    },
];

/// Total number of unique tools across all groups.
/// Must match the MCP tools.rs count (currently 145).
pub fn tool_catalog_tool_count() -> usize {
    let mut names: Vec<&str> = TOOL_GROUPS
        .iter()
        .flat_map(|g| g.tools.iter().map(|t| t.name))
        .collect();
    names.sort();
    names.dedup();
    names.len()
}

/// Serialize the full tool catalog to compact JSON for the oneshot Opus prompt.
pub fn tools_catalog_to_json(groups: &[ToolGroup]) -> String {
    let json_groups: Vec<serde_json::Value> = groups
        .iter()
        .map(|g| {
            let tools: Vec<serde_json::Value> = g
                .tools
                .iter()
                .map(|t| serde_json::json!({ "n": t.name, "d": t.description }))
                .collect();
            serde_json::json!({
                "group": g.name,
                "desc": g.description,
                "tools": tools,
            })
        })
        .collect();
    serde_json::to_string(&json_groups).unwrap_or_default()
}

/// Format selected tool groups as concise markdown for injection into the system prompt.
pub fn format_tool_groups_markdown(groups: &[&ToolGroup]) -> String {
    let mut md = String::from("## Tools recommandés\n\n");
    for group in groups {
        md.push_str(&format!("### {}\n", group.description));
        for tool in group.tools.iter() {
            md.push_str(&format!("- `{}` — {}\n", tool.name, tool.description));
        }
        md.push('\n');
    }
    md
}

/// Keyword-based heuristic fallback for selecting tool groups when the oneshot Opus fails.
/// Matches words in the user message against each group's keywords (case-insensitive).
/// Always returns at least one group (`planning` + `code_exploration` as default).
pub fn select_tool_groups_by_keywords(user_message: &str) -> Vec<&'static ToolGroup> {
    let msg_lower = user_message.to_lowercase();
    let words: Vec<&str> = msg_lower.split_whitespace().collect();

    let mut matched: Vec<&'static ToolGroup> = TOOL_GROUPS
        .iter()
        .filter(|group| {
            group
                .keywords
                .iter()
                .any(|kw| words.iter().any(|w| w.contains(kw)))
        })
        .collect();

    // Fallback: if nothing matched, return planning + code_exploration
    if matched.is_empty() {
        matched = TOOL_GROUPS
            .iter()
            .filter(|g| g.name == "planning" || g.name == "code_exploration")
            .collect();
    }

    matched
}

use crate::neo4j::models::{
    ConnectedFileNode, ConstraintNode, FeatureGraphNode, LanguageStatsNode, MilestoneNode,
    PlanNode, ProjectNode, ReleaseNode, WorkspaceNode,
};
use crate::neo4j::GraphStore;
use crate::notes::models::{Note, NoteFilters, NoteImportance, NoteStatus, NoteType};

// ============================================================================
// ProjectContext — all dynamic data fetched from Neo4j
// ============================================================================

/// Contextual data fetched from Neo4j for the current project.
/// Used to build the dynamic section of the system prompt.
#[derive(Default)]
pub struct ProjectContext {
    pub project: Option<ProjectNode>,
    pub workspace: Option<WorkspaceNode>,
    /// Other projects in the same workspace (excludes the current project)
    pub sibling_projects: Vec<ProjectNode>,
    pub active_plans: Vec<PlanNode>,
    pub plan_constraints: Vec<ConstraintNode>,
    pub guidelines: Vec<Note>,
    pub gotchas: Vec<Note>,
    /// Global notes (no project_id) — cross-project knowledge
    pub global_guidelines: Vec<Note>,
    pub global_gotchas: Vec<Note>,
    pub milestones: Vec<MilestoneNode>,
    pub releases: Vec<ReleaseNode>,
    pub language_stats: Vec<LanguageStatsNode>,
    pub key_files: Vec<ConnectedFileNode>,
    pub feature_graphs: Vec<FeatureGraphNode>,
    pub last_synced: Option<DateTime<Utc>>,
    /// Pre-built GDS topology section (communities, bridges, health alerts)
    pub structural_topology: Option<String>,
}

// ============================================================================
// Fetcher — populates ProjectContext from GraphStore
// ============================================================================

/// Fetch all project context from Neo4j. Individual fetch errors are handled
/// gracefully (empty defaults) to never block the prompt building.
pub async fn fetch_project_context(
    graph: &Arc<dyn GraphStore>,
    slug: &str,
) -> Result<ProjectContext> {
    let mut ctx = ProjectContext::default();

    // 1. Project
    let project = graph.get_project_by_slug(slug).await.unwrap_or(None);
    if project.is_none() {
        return Ok(ctx);
    }
    let project = project.unwrap();
    let project_id = project.id;
    ctx.last_synced = project.last_synced;
    ctx.project = Some(project);

    // 2. Workspace + sibling projects
    ctx.workspace = graph
        .get_project_workspace(project_id)
        .await
        .unwrap_or(None);

    // 2b. Sibling projects (other projects in the same workspace)
    if let Some(ref ws) = ctx.workspace {
        let mut siblings = graph
            .list_workspace_projects(ws.id)
            .await
            .unwrap_or_default();
        siblings.retain(|p| p.id != project_id);
        ctx.sibling_projects = siblings;
    }

    // 3. Active plans for this project
    let (plans, _) = graph
        .list_plans_for_project(
            project_id,
            Some(vec![
                "draft".to_string(),
                "approved".to_string(),
                "in_progress".to_string(),
            ]),
            50,
            0,
        )
        .await
        .unwrap_or_default();
    ctx.active_plans = plans;

    // 4. Constraints for each active plan
    let mut all_constraints = Vec::new();
    for plan in &ctx.active_plans {
        if let Ok(constraints) = graph.get_plan_constraints(plan.id).await {
            all_constraints.extend(constraints);
        }
    }
    ctx.plan_constraints = all_constraints;

    // 5. Guidelines (critical/high importance, active)
    let guideline_filters = NoteFilters {
        note_type: Some(vec![NoteType::Guideline]),
        importance: Some(vec![NoteImportance::Critical, NoteImportance::High]),
        status: Some(vec![NoteStatus::Active]),
        ..Default::default()
    };
    let (guidelines, _) = graph
        .list_notes(Some(project_id), None, &guideline_filters)
        .await
        .unwrap_or_default();
    ctx.guidelines = guidelines;

    // 6. Gotchas (active)
    let gotcha_filters = NoteFilters {
        note_type: Some(vec![NoteType::Gotcha]),
        status: Some(vec![NoteStatus::Active]),
        ..Default::default()
    };
    let (gotchas, _) = graph
        .list_notes(Some(project_id), None, &gotcha_filters)
        .await
        .unwrap_or_default();
    ctx.gotchas = gotchas;

    // 6b. Global guidelines (no project_id, cross-project knowledge)
    let global_guideline_filters = NoteFilters {
        note_type: Some(vec![NoteType::Guideline]),
        importance: Some(vec![NoteImportance::Critical, NoteImportance::High]),
        status: Some(vec![NoteStatus::Active]),
        global_only: Some(true),
        ..Default::default()
    };
    let (global_guidelines, _) = graph
        .list_notes(None, None, &global_guideline_filters)
        .await
        .unwrap_or_default();
    ctx.global_guidelines = global_guidelines;

    // 6c. Global gotchas
    let global_gotcha_filters = NoteFilters {
        note_type: Some(vec![NoteType::Gotcha]),
        status: Some(vec![NoteStatus::Active]),
        global_only: Some(true),
        ..Default::default()
    };
    let (global_gotchas, _) = graph
        .list_notes(None, None, &global_gotcha_filters)
        .await
        .unwrap_or_default();
    ctx.global_gotchas = global_gotchas;

    // 7. Milestones
    ctx.milestones = graph
        .list_project_milestones(project_id)
        .await
        .unwrap_or_default();

    // 8. Releases
    ctx.releases = graph
        .list_project_releases(project_id)
        .await
        .unwrap_or_default();

    // 9. Language stats (scoped to project)
    ctx.language_stats = graph
        .get_language_stats_for_project(project_id)
        .await
        .unwrap_or_default();

    // 10. Key files (most connected, scoped to project)
    ctx.key_files = graph
        .get_most_connected_files_for_project(project_id, 5)
        .await
        .unwrap_or_default();

    // 11. Feature graphs (lightweight catalogue for the prompt)
    ctx.feature_graphs = graph
        .list_feature_graphs(Some(project_id))
        .await
        .unwrap_or_default();

    // 12. Structural topology (GDS communities, bridges, health alerts)
    ctx.structural_topology = build_gds_topology_section(graph, project_id).await;

    Ok(ctx)
}

/// Build a Markdown section with GDS topology data for the system prompt.
/// Returns None if no GDS data is available (graceful degradation).
async fn build_gds_topology_section(
    graph: &Arc<dyn GraphStore>,
    project_id: Uuid,
) -> Option<String> {
    let mut sections: Vec<String> = Vec::new();

    // 1. Code Communities
    let communities = graph
        .get_project_communities(project_id)
        .await
        .unwrap_or_default();

    if communities.is_empty() {
        return None; // No GDS data at all
    }

    let mut comm_lines = vec!["### Code Communities (structural clusters)".to_string()];
    let mut sorted_communities = communities;
    sorted_communities.sort_by(|a, b| b.file_count.cmp(&a.file_count));
    for c in sorted_communities.iter().take(5) {
        let key_files_str = if c.key_files.len() > 3 {
            format!("{}, ...", c.key_files[..3].join(", "))
        } else {
            c.key_files.join(", ")
        };
        comm_lines.push(format!(
            "- **{}** ({} files): {}",
            c.community_label, c.file_count, key_files_str
        ));
    }
    sections.push(comm_lines.join("\n"));

    // 2. Bridge Files
    let bridges = graph
        .get_top_bridges_by_betweenness(project_id, 3)
        .await
        .unwrap_or_default();

    if !bridges.is_empty() {
        let mut bridge_lines = vec!["### Bridge Files (high blast radius)".to_string()];
        for b in &bridges {
            let label = b.community_label.as_deref().unwrap_or("unknown");
            bridge_lines.push(format!(
                "- {} — betweenness {:.2}, community: {}",
                b.path, b.betweenness, label
            ));
        }
        bridge_lines.push("Changes to bridge files affect multiple communities.".to_string());
        sections.push(bridge_lines.join("\n"));
    }

    // 3. Structural Alerts (god functions + circular deps)
    let health = graph.get_code_health_report(project_id, 10).await.ok();

    let circular = graph
        .get_circular_dependencies(project_id)
        .await
        .unwrap_or_default();

    let has_god_functions = health
        .as_ref()
        .map(|h| !h.god_functions.is_empty())
        .unwrap_or(false);
    let has_cycles = !circular.is_empty();

    if has_god_functions || has_cycles {
        let mut alert_lines = vec!["### Structural Alerts".to_string()];

        if let Some(ref h) = health {
            for gf in h.god_functions.iter().take(5) {
                alert_lines.push(format!(
                    "- God function: {} ({} callers, {} callees) in {}",
                    gf.name, gf.in_degree, gf.out_degree, gf.file
                ));
            }
        }

        for cycle in circular.iter().take(3) {
            alert_lines.push(format!("- Circular dep: {}", cycle.join(" -> ")));
        }

        sections.push(alert_lines.join("\n"));
    }

    if sections.is_empty() {
        None
    } else {
        Some(sections.join("\n\n"))
    }
}

// ============================================================================
// Serializers — JSON (for oneshot) and Markdown (for fallback)
// ============================================================================

/// Serialize ProjectContext to compact JSON for the oneshot Opus prompt.
/// Only includes non-empty fields to reduce payload size.
pub fn context_to_json(ctx: &ProjectContext) -> String {
    let mut map = serde_json::Map::new();

    if let Some(ref p) = ctx.project {
        map.insert(
            "project".into(),
            serde_json::json!({
                "name": p.name,
                "slug": p.slug,
                "root_path": p.root_path,
                "description": p.description,
            }),
        );
    }

    if let Some(ref w) = ctx.workspace {
        map.insert(
            "workspace".into(),
            serde_json::json!({
                "name": w.name,
                "slug": w.slug,
                "description": w.description,
            }),
        );
    }

    if !ctx.sibling_projects.is_empty() {
        let siblings: Vec<_> = ctx
            .sibling_projects
            .iter()
            .map(|p| {
                serde_json::json!({
                    "name": p.name,
                    "slug": p.slug,
                    "description": p.description,
                })
            })
            .collect();
        map.insert(
            "sibling_projects".into(),
            serde_json::Value::Array(siblings),
        );
    }

    if !ctx.active_plans.is_empty() {
        let plans: Vec<_> = ctx
            .active_plans
            .iter()
            .map(|p| {
                serde_json::json!({
                    "title": p.title,
                    "status": format!("{:?}", p.status),
                    "priority": p.priority,
                    "description": p.description,
                })
            })
            .collect();
        map.insert("active_plans".into(), serde_json::Value::Array(plans));
    }

    if !ctx.plan_constraints.is_empty() {
        let constraints: Vec<_> = ctx
            .plan_constraints
            .iter()
            .map(|c| {
                serde_json::json!({
                    "type": format!("{:?}", c.constraint_type),
                    "description": c.description,
                })
            })
            .collect();
        map.insert("constraints".into(), serde_json::Value::Array(constraints));
    }

    if !ctx.guidelines.is_empty() {
        let notes: Vec<_> = ctx
            .guidelines
            .iter()
            .map(|n| {
                serde_json::json!({
                    "content": n.content,
                    "importance": format!("{:?}", n.importance),
                })
            })
            .collect();
        map.insert("guidelines".into(), serde_json::Value::Array(notes));
    }

    if !ctx.gotchas.is_empty() {
        let notes: Vec<_> = ctx
            .gotchas
            .iter()
            .map(|n| serde_json::json!({ "content": n.content }))
            .collect();
        map.insert("gotchas".into(), serde_json::Value::Array(notes));
    }

    if !ctx.global_guidelines.is_empty() {
        let notes: Vec<_> = ctx
            .global_guidelines
            .iter()
            .map(|n| {
                serde_json::json!({
                    "content": n.content,
                    "importance": format!("{:?}", n.importance),
                })
            })
            .collect();
        map.insert("global_guidelines".into(), serde_json::Value::Array(notes));
    }

    if !ctx.global_gotchas.is_empty() {
        let notes: Vec<_> = ctx
            .global_gotchas
            .iter()
            .map(|n| serde_json::json!({ "content": n.content }))
            .collect();
        map.insert("global_gotchas".into(), serde_json::Value::Array(notes));
    }

    if !ctx.milestones.is_empty() {
        let ms: Vec<_> = ctx
            .milestones
            .iter()
            .map(|m| {
                serde_json::json!({
                    "title": m.title,
                    "status": format!("{:?}", m.status),
                    "target_date": m.target_date,
                })
            })
            .collect();
        map.insert("milestones".into(), serde_json::Value::Array(ms));
    }

    if !ctx.releases.is_empty() {
        let rs: Vec<_> = ctx
            .releases
            .iter()
            .map(|r| {
                serde_json::json!({
                    "version": r.version,
                    "status": format!("{:?}", r.status),
                    "target_date": r.target_date,
                })
            })
            .collect();
        map.insert("releases".into(), serde_json::Value::Array(rs));
    }

    if !ctx.language_stats.is_empty() {
        let ls: Vec<_> = ctx
            .language_stats
            .iter()
            .map(|l| serde_json::json!({ "language": l.language, "file_count": l.file_count }))
            .collect();
        map.insert("language_stats".into(), serde_json::Value::Array(ls));
    }

    if !ctx.key_files.is_empty() {
        let kf: Vec<_> = ctx
            .key_files
            .iter()
            .map(|f| {
                serde_json::json!({
                    "path": f.path,
                    "imports": f.imports,
                    "dependents": f.dependents,
                })
            })
            .collect();
        map.insert("key_files".into(), serde_json::Value::Array(kf));
    }

    if !ctx.feature_graphs.is_empty() {
        let fgs: Vec<_> = ctx
            .feature_graphs
            .iter()
            .map(|fg| {
                let desc = fg.description.as_deref().unwrap_or("");
                let desc_truncated = if desc.len() > 60 {
                    format!("{}…", &desc[..desc.floor_char_boundary(60)])
                } else {
                    desc.to_string()
                };
                serde_json::json!({
                    "id": fg.id.to_string(),
                    "name": fg.name,
                    "description": desc_truncated,
                    "entity_count": fg.entity_count.unwrap_or(0),
                })
            })
            .collect();
        map.insert("feature_graphs".into(), serde_json::Value::Array(fgs));
    }

    if let Some(ref topo) = ctx.structural_topology {
        map.insert(
            "structural_topology".into(),
            serde_json::Value::String(topo.clone()),
        );
    }

    if let Some(ref ts) = ctx.last_synced {
        map.insert(
            "last_synced".into(),
            serde_json::Value::String(ts.to_rfc3339()),
        );
    }

    serde_json::to_string(&serde_json::Value::Object(map)).unwrap_or_default()
}

/// Format ProjectContext as markdown for fallback (when oneshot fails).
/// Only includes sections that have data.
/// When `user_message` is provided, appends a "Tools recommandés" section
/// selected by keyword heuristic matching.
pub fn context_to_markdown(ctx: &ProjectContext, user_message: Option<&str>) -> String {
    let mut md = String::new();

    if let Some(ref p) = ctx.project {
        md.push_str(&format!("## Projet actif : {} ({})\n", p.name, p.slug));
        md.push_str(&format!("Root: {}\n", p.root_path));
        if let Some(ref desc) = p.description {
            md.push_str(&format!("Description: {}\n", desc));
        }
        md.push('\n');
    }

    if let Some(ref w) = ctx.workspace {
        md.push_str(&format!("## Workspace : {} ({})\n", w.name, w.slug));
        if let Some(ref desc) = w.description {
            md.push_str(&format!("{}\n", desc));
        }
        md.push('\n');
    }

    if !ctx.sibling_projects.is_empty() {
        md.push_str("## Projets du workspace\n");
        for p in &ctx.sibling_projects {
            if let Some(ref desc) = p.description {
                md.push_str(&format!("- **{}** ({}) — {}\n", p.name, p.slug, desc));
            } else {
                md.push_str(&format!("- **{}** ({})\n", p.name, p.slug));
            }
        }
        md.push('\n');
    }

    if !ctx.active_plans.is_empty() {
        md.push_str("## Plans actifs\n");
        for plan in &ctx.active_plans {
            md.push_str(&format!(
                "- **{}** ({:?}, priorité {})\n",
                plan.title, plan.status, plan.priority
            ));
        }
        md.push('\n');
    }

    if !ctx.plan_constraints.is_empty() {
        md.push_str("## Contraintes\n");
        for c in &ctx.plan_constraints {
            md.push_str(&format!("- [{:?}] {}\n", c.constraint_type, c.description));
        }
        md.push('\n');
    }

    if !ctx.guidelines.is_empty() {
        md.push_str("## Guidelines\n");
        for g in &ctx.guidelines {
            md.push_str(&format!("- [{:?}] {}\n", g.importance, g.content));
        }
        md.push('\n');
    }

    if !ctx.gotchas.is_empty() {
        md.push_str("## Gotchas\n");
        for g in &ctx.gotchas {
            md.push_str(&format!("- {}\n", g.content));
        }
        md.push('\n');
    }

    // Global notes (cross-project knowledge)
    if !ctx.global_guidelines.is_empty() {
        md.push_str("## Guidelines globales\n");
        for g in &ctx.global_guidelines {
            md.push_str(&format!("- [{:?}] {}\n", g.importance, g.content));
        }
        md.push('\n');
    }

    if !ctx.global_gotchas.is_empty() {
        md.push_str("## Gotchas globaux\n");
        for g in &ctx.global_gotchas {
            md.push_str(&format!("- {}\n", g.content));
        }
        md.push('\n');
    }

    if !ctx.milestones.is_empty() {
        md.push_str("## Milestones\n");
        for m in &ctx.milestones {
            let date = m
                .target_date
                .map(|d| d.format("%Y-%m-%d").to_string())
                .unwrap_or_else(|| "pas de date".into());
            md.push_str(&format!(
                "- **{}** ({:?}) — cible: {}\n",
                m.title, m.status, date
            ));
        }
        md.push('\n');
    }

    if !ctx.releases.is_empty() {
        md.push_str("## Releases\n");
        for r in &ctx.releases {
            let date = r
                .target_date
                .map(|d| d.format("%Y-%m-%d").to_string())
                .unwrap_or_else(|| "pas de date".into());
            md.push_str(&format!(
                "- **v{}** ({:?}) — cible: {}\n",
                r.version, r.status, date
            ));
        }
        md.push('\n');
    }

    if !ctx.language_stats.is_empty() {
        md.push_str("## Langages\n");
        for l in &ctx.language_stats {
            md.push_str(&format!("- {} ({} fichiers)\n", l.language, l.file_count));
        }
        md.push('\n');
    }

    if !ctx.key_files.is_empty() {
        md.push_str("## Fichiers clés\n");
        for f in &ctx.key_files {
            md.push_str(&format!(
                "- `{}` ({} imports, {} dépendants)\n",
                f.path, f.imports, f.dependents
            ));
        }
        md.push('\n');
    }

    if let Some(ref topo) = ctx.structural_topology {
        md.push_str("## Topologie structurelle\n");
        md.push_str(topo);
        md.push_str("\n\n");
    }

    if !ctx.feature_graphs.is_empty() {
        md.push_str("## Feature Graphs\n");
        for fg in &ctx.feature_graphs {
            let desc = fg.description.as_deref().unwrap_or("");
            let desc_display = if desc.len() > 80 {
                format!("{}…", &desc[..desc.floor_char_boundary(80)])
            } else {
                desc.to_string()
            };
            let count = fg.entity_count.unwrap_or(0);
            if desc_display.is_empty() {
                md.push_str(&format!("- **{}** ({} entités)\n", fg.name, count));
            } else {
                md.push_str(&format!(
                    "- **{}** — {} ({} entités)\n",
                    fg.name, desc_display, count
                ));
            }
        }
        md.push_str(
            "\n→ Utiliser `get_feature_graph(id)` pour explorer les entités d'un graph\n\n",
        );
    }

    // Only show sync warnings if we have a project
    if ctx.project.is_some() {
        match ctx.last_synced {
            Some(ts) => {
                let ago = Utc::now().signed_duration_since(ts);
                if ago.num_hours() < 1 {
                    md.push_str(&format!(
                        "- **Dernière sync** : il y a {} min (à jour)\n\n",
                        ago.num_minutes().max(1)
                    ));
                } else if ago.num_hours() < 24 {
                    md.push_str(&format!(
                        "- **Dernière sync** : il y a {}h\n\n",
                        ago.num_hours()
                    ));
                } else {
                    md.push_str(&format!(
                        "- **Dernière sync** : il y a {}j — si le code a changé depuis, lance `sync_project`\n\n",
                        ago.num_days()
                    ));
                }
            }
            None => {
                md.push_str("⚠️ **Aucun sync** — le code n'a jamais été synchronisé. Lance `sync_project` avant d'explorer le code.\n\n");
            }
        }
    }

    // Append keyword-matched tool groups when user message is available
    if let Some(msg) = user_message {
        if !msg.is_empty() {
            let groups = select_tool_groups_by_keywords(msg);
            let refs: Vec<&ToolGroup> = groups.into_iter().collect();
            md.push_str(&format_tool_groups_markdown(&refs));
        }
    }

    md
}

// ============================================================================
// Oneshot refinement prompt
// ============================================================================

/// Build the prompt sent to the oneshot Opus model for context refinement.
/// The oneshot analyzes the user's request + raw project context JSON + tool catalog
/// and produces a concise "## Contexte actif" section (<500 words) including recommended tools.
pub fn build_refinement_prompt(
    user_message: &str,
    context_json: &str,
    tools_catalog_json: &str,
) -> String {
    format!(
        "Tu es un constructeur de contexte pour un agent de développement.\n\
         \n\
         ## Demande de l'utilisateur\n\
         ---\n\
         {user_message}\n\
         ---\n\
         \n\
         ## Données du projet actif (JSON)\n\
         ---\n\
         {context_json}\n\
         ---\n\
         \n\
         ## Catalogue des outils MCP disponibles (JSON)\n\
         ---\n\
         {tools_catalog_json}\n\
         ---\n\
         \n\
         Génère une section unique \"## Contexte actif\" pour le prompt système de l'agent.\n\
         Cette section doit contenir EXACTEMENT deux parties :\n\
         \n\
         **1. Contexte projet** (seulement ce qui est pertinent pour la demande) :\n\
         - Infos projet (nom, slug, état du sync)\n\
         - Plans en cours si la demande touche à la planification\n\
         - Guidelines et gotchas pertinents\n\
         - Milestones/releases si la demande touche à la roadmap\n\
         - Stats du code si la demande touche au code\n\
         \n\
         **2. Tools recommandés** : sélectionne les groupes d'outils pertinents\n\
         pour cette demande spécifique dans le catalogue. Pour chaque groupe retenu,\n\
         liste les outils sous forme `- \\`nom\\` — description`.\n\
         Inclus TOUJOURS le groupe le plus pertinent. N'inclus PAS les groupes inutiles.\n\
         Exemples :\n\
         - Demande sur le code → code_exploration + knowledge\n\
         - Demande de planification → planning + steps + constraints\n\
         - Demande de débogage → code_exploration + knowledge + git_tracking\n\
         - Demande générale/vague → planning + code_exploration\n\
         \n\
         Format : markdown, bullet points, court et actionnable.\n\
         Budget total : <500 mots (contexte + tools combinés).\n\
         Arbitre : si le contexte projet est riche, réduis les descriptions de tools.\n\
         Si le contexte est pauvre, détaille davantage les tools.",
        user_message = user_message,
        context_json = context_json,
        tools_catalog_json = tools_catalog_json,
    )
}

// ============================================================================
// Assembler — combines base + dynamic context
// ============================================================================

/// Assemble the final system prompt by combining the hardcoded base
/// with the dynamic contextual section.
pub fn assemble_prompt(base: &str, dynamic_context: &str) -> String {
    if dynamic_context.is_empty() {
        return base.to_string();
    }
    format!("{}\n\n---\n\n{}", base, dynamic_context)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_system_prompt_contains_key_sections() {
        assert!(BASE_SYSTEM_PROMPT.contains("EXCLUSIVEMENT les outils MCP"));
        assert!(BASE_SYSTEM_PROMPT.contains("Modèle de données"));
        assert!(BASE_SYSTEM_PROMPT.contains("Tree-sitter"));
        assert!(BASE_SYSTEM_PROMPT.contains("Workflow Git"));
        assert!(BASE_SYSTEM_PROMPT.contains("Protocole d'exécution"));
        assert!(BASE_SYSTEM_PROMPT.contains("Protocole de planification"));
        assert!(BASE_SYSTEM_PROMPT.contains("Gestion des statuts"));
        assert!(BASE_SYSTEM_PROMPT.contains("Bonnes pratiques"));
        assert!(BASE_SYSTEM_PROMPT.contains("Stratégie de recherche"));
        assert!(BASE_SYSTEM_PROMPT.contains("MCP-first"));
        assert!(BASE_SYSTEM_PROMPT.contains("path_prefix"));
        assert!(BASE_SYSTEM_PROMPT.contains("Sync incrémentale au commit"));
        // T6 behavioral directives
        assert!(BASE_SYSTEM_PROMPT.contains("Warm-up"));
        assert!(BASE_SYSTEM_PROMPT.contains("search_notes"));
        assert!(BASE_SYSTEM_PROMPT.contains("search_notes_semantic"));
        assert!(BASE_SYSTEM_PROMPT.contains("Capture des connaissances (OBLIGATOIRE)"));
        assert!(BASE_SYSTEM_PROMPT.contains("create_note"));
        assert!(BASE_SYSTEM_PROMPT.contains("link_note_to_entity"));
    }

    #[test]
    fn test_base_system_prompt_mcp_first_directive() {
        assert!(BASE_SYSTEM_PROMPT.contains("PAS"));
        assert!(BASE_SYSTEM_PROMPT.contains("TodoWrite"));
        assert!(BASE_SYSTEM_PROMPT.contains("EnterPlanMode"));
        assert!(BASE_SYSTEM_PROMPT.contains("create_plan"));
        assert!(BASE_SYSTEM_PROMPT.contains("create_task"));
        assert!(BASE_SYSTEM_PROMPT.contains("create_step"));
    }

    #[test]
    fn test_base_system_prompt_has_task_decomposition_example() {
        assert!(BASE_SYSTEM_PROMPT.contains("Ajouter l'endpoint GET /api/releases/:id"));
        assert!(BASE_SYSTEM_PROMPT.contains("Step 1"));
        assert!(BASE_SYSTEM_PROMPT.contains("Step 5"));
    }

    #[test]
    fn test_base_system_prompt_has_status_transitions() {
        assert!(BASE_SYSTEM_PROMPT.contains("draft"));
        assert!(BASE_SYSTEM_PROMPT.contains("approved"));
        assert!(BASE_SYSTEM_PROMPT.contains("in_progress"));
        assert!(BASE_SYSTEM_PROMPT.contains("completed"));
        assert!(BASE_SYSTEM_PROMPT.contains("blocked"));
    }

    #[test]
    fn test_context_to_json_empty() {
        let ctx = ProjectContext::default();
        let json = context_to_json(&ctx);
        assert_eq!(json, "{}");
    }

    #[test]
    fn test_context_to_json_with_project() {
        let ctx = ProjectContext {
            project: Some(ProjectNode {
                id: uuid::Uuid::new_v4(),
                name: "TestProject".into(),
                slug: "test-project".into(),
                root_path: "/tmp/test".into(),
                description: Some("A test project".into()),
                created_at: Utc::now(),
                last_synced: None,
                analytics_computed_at: None,
            }),
            ..Default::default()
        };
        let json = context_to_json(&ctx);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["project"]["name"], "TestProject");
        assert_eq!(parsed["project"]["slug"], "test-project");
    }

    #[test]
    fn test_context_to_markdown_empty() {
        let ctx = ProjectContext::default();
        let md = context_to_markdown(&ctx, None);
        assert!(md.is_empty());
    }

    #[test]
    fn test_context_to_markdown_with_project() {
        let ctx = ProjectContext {
            project: Some(ProjectNode {
                id: uuid::Uuid::new_v4(),
                name: "MyProject".into(),
                slug: "my-project".into(),
                root_path: "/home/user/code".into(),
                description: None,
                created_at: Utc::now(),
                last_synced: None,
                analytics_computed_at: None,
            }),
            ..Default::default()
        };
        let md = context_to_markdown(&ctx, None);
        assert!(md.contains("MyProject"));
        assert!(md.contains("my-project"));
        assert!(md.contains("Aucun sync"));
    }

    #[test]
    fn test_context_to_markdown_partial_data() {
        let ctx = ProjectContext {
            project: Some(ProjectNode {
                id: uuid::Uuid::new_v4(),
                name: "Partial".into(),
                slug: "partial".into(),
                root_path: "/tmp".into(),
                description: None,
                created_at: Utc::now(),
                last_synced: Some(Utc::now()),
                analytics_computed_at: None,
            }),
            language_stats: vec![LanguageStatsNode {
                language: "Rust".into(),
                file_count: 42,
            }],
            ..Default::default()
        };
        let md = context_to_markdown(&ctx, None);
        assert!(md.contains("Partial"));
        assert!(md.contains("Rust"));
        assert!(md.contains("42 fichiers"));
        // Should NOT contain sections with no data
        assert!(!md.contains("Guidelines"));
        assert!(!md.contains("Gotchas"));
        assert!(!md.contains("Milestones"));
    }

    #[test]
    fn test_build_refinement_prompt_contains_inputs() {
        let tools_json = tools_catalog_to_json(TOOL_GROUPS);
        let prompt =
            build_refinement_prompt("Implémente le login", r#"{"project":"test"}"#, &tools_json);
        // User message & context JSON are injected
        assert!(prompt.contains("Implémente le login"));
        assert!(prompt.contains(r#"{"project":"test"}"#));
        // Tools catalog JSON is injected
        assert!(prompt.contains("planning"));
        assert!(prompt.contains("code_exploration"));
        assert!(prompt.contains("create_plan"));
        // Instructions for Opus to select tool groups
        assert!(prompt.contains("Tools recommandés"));
        assert!(prompt.contains("groupes d'outils pertinents"));
        // Budget constraint
        assert!(prompt.contains("500 mots"));
        assert!(prompt.contains("## Contexte actif"));
    }

    #[test]
    fn test_assemble_prompt_no_context() {
        let result = assemble_prompt("base prompt", "");
        assert_eq!(result, "base prompt");
    }

    #[test]
    fn test_assemble_prompt_with_context() {
        let result = assemble_prompt("base prompt", "## Contexte actif\n- info");
        assert!(result.contains("base prompt"));
        assert!(result.contains("---"));
        assert!(result.contains("## Contexte actif"));
    }

    // ================================================================
    // fetch_project_context tests (with MockGraphStore)
    // ================================================================

    #[tokio::test]
    async fn test_fetch_project_context_existing_project() {
        use crate::test_helpers::{mock_app_state, test_project};

        let state = mock_app_state();
        let project = test_project();
        state.neo4j.create_project(&project).await.unwrap();

        let ctx = fetch_project_context(&state.neo4j, &project.slug)
            .await
            .unwrap();

        assert!(ctx.project.is_some());
        assert_eq!(ctx.project.as_ref().unwrap().name, project.name);
        assert!(ctx.active_plans.is_empty()); // no plans created
        assert!(ctx.guidelines.is_empty());
    }

    #[tokio::test]
    async fn test_fetch_project_context_nonexistent_project() {
        use crate::test_helpers::mock_app_state;

        let state = mock_app_state();

        let ctx = fetch_project_context(&state.neo4j, "nonexistent-slug")
            .await
            .unwrap();

        assert!(ctx.project.is_none());
        assert!(ctx.active_plans.is_empty());
    }

    #[tokio::test]
    async fn test_fetch_project_context_with_plans() {
        use crate::test_helpers::{mock_app_state, test_plan, test_project};

        let state = mock_app_state();
        let project = test_project();
        state.neo4j.create_project(&project).await.unwrap();

        let plan = test_plan();
        state.neo4j.create_plan(&plan).await.unwrap();
        state
            .neo4j
            .link_plan_to_project(plan.id, project.id)
            .await
            .unwrap();

        let ctx = fetch_project_context(&state.neo4j, &project.slug)
            .await
            .unwrap();

        assert!(ctx.project.is_some());
        assert!(!ctx.active_plans.is_empty());
        assert_eq!(ctx.active_plans[0].title, plan.title);
    }

    #[tokio::test]
    async fn test_fetch_project_context_includes_global_notes() {
        use crate::notes::{NoteImportance, NoteType};
        use crate::test_helpers::{mock_app_state, test_project};

        let state = mock_app_state();
        let project = test_project();
        state.neo4j.create_project(&project).await.unwrap();

        // Create a project-specific guideline (must be Critical/High to be fetched)
        let mut project_note = crate::notes::Note::new(
            Some(project.id),
            NoteType::Guideline,
            "Project-specific guideline".to_string(),
            "test".to_string(),
        );
        project_note.importance = NoteImportance::High;
        state.neo4j.create_note(&project_note).await.unwrap();

        // Create a global guideline (no project_id)
        let mut global_note = crate::notes::Note::new(
            None,
            NoteType::Guideline,
            "Global team convention".to_string(),
            "test".to_string(),
        );
        global_note.importance = NoteImportance::Critical;
        state.neo4j.create_note(&global_note).await.unwrap();

        let ctx = fetch_project_context(&state.neo4j, &project.slug)
            .await
            .unwrap();

        // Project guidelines include only project-specific ones
        assert_eq!(ctx.guidelines.len(), 1);
        assert_eq!(ctx.guidelines[0].content, "Project-specific guideline");

        // Global guidelines include only global ones
        assert_eq!(ctx.global_guidelines.len(), 1);
        assert_eq!(ctx.global_guidelines[0].content, "Global team convention");
    }

    // ================================================================
    // sibling_projects tests
    // ================================================================

    #[test]
    fn test_context_to_json_with_sibling_projects() {
        let ctx = ProjectContext {
            project: Some(ProjectNode {
                id: uuid::Uuid::new_v4(),
                name: "Current".into(),
                slug: "current".into(),
                root_path: "/tmp/current".into(),
                description: None,
                created_at: Utc::now(),
                last_synced: None,
                analytics_computed_at: None,
            }),
            sibling_projects: vec![
                ProjectNode {
                    id: uuid::Uuid::new_v4(),
                    name: "SiblingA".into(),
                    slug: "sibling-a".into(),
                    root_path: "/tmp/a".into(),
                    description: Some("First sibling".into()),
                    created_at: Utc::now(),
                    last_synced: None,
                    analytics_computed_at: None,
                },
                ProjectNode {
                    id: uuid::Uuid::new_v4(),
                    name: "SiblingB".into(),
                    slug: "sibling-b".into(),
                    root_path: "/tmp/b".into(),
                    description: None,
                    created_at: Utc::now(),
                    last_synced: None,
                    analytics_computed_at: None,
                },
            ],
            ..Default::default()
        };
        let json = context_to_json(&ctx);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let siblings = parsed["sibling_projects"].as_array().unwrap();
        assert_eq!(siblings.len(), 2);
        assert_eq!(siblings[0]["name"], "SiblingA");
        assert_eq!(siblings[0]["slug"], "sibling-a");
        assert_eq!(siblings[0]["description"], "First sibling");
        assert_eq!(siblings[1]["name"], "SiblingB");
        assert_eq!(siblings[1]["slug"], "sibling-b");
        assert!(siblings[1]["description"].is_null());
    }

    #[test]
    fn test_context_to_markdown_with_sibling_projects() {
        let ctx = ProjectContext {
            project: Some(ProjectNode {
                id: uuid::Uuid::new_v4(),
                name: "Current".into(),
                slug: "current".into(),
                root_path: "/tmp/current".into(),
                description: None,
                created_at: Utc::now(),
                last_synced: Some(Utc::now()),
                analytics_computed_at: None,
            }),
            workspace: Some(WorkspaceNode {
                id: uuid::Uuid::new_v4(),
                name: "MyWorkspace".into(),
                slug: "my-ws".into(),
                description: None,
                created_at: Utc::now(),
                updated_at: None,
                metadata: serde_json::json!({}),
            }),
            sibling_projects: vec![
                ProjectNode {
                    id: uuid::Uuid::new_v4(),
                    name: "Backend".into(),
                    slug: "backend".into(),
                    root_path: "/tmp/backend".into(),
                    description: Some("The API server".into()),
                    created_at: Utc::now(),
                    last_synced: None,
                    analytics_computed_at: None,
                },
                ProjectNode {
                    id: uuid::Uuid::new_v4(),
                    name: "Frontend".into(),
                    slug: "frontend".into(),
                    root_path: "/tmp/frontend".into(),
                    description: None,
                    created_at: Utc::now(),
                    last_synced: None,
                    analytics_computed_at: None,
                },
            ],
            ..Default::default()
        };
        let md = context_to_markdown(&ctx, None);
        assert!(md.contains("## Projets du workspace"));
        assert!(md.contains("**Backend** (backend) — The API server"));
        assert!(md.contains("**Frontend** (frontend)"));
        // No description → no " — " suffix
        assert!(!md.contains("**Frontend** (frontend) —"));
    }

    #[tokio::test]
    async fn test_fetch_project_context_with_sibling_projects() {
        use crate::test_helpers::{mock_app_state, test_project_named, test_workspace};

        let state = mock_app_state();
        let ws = test_workspace();
        state.neo4j.create_workspace(&ws).await.unwrap();

        let project_a = test_project_named("ProjectA");
        state.neo4j.create_project(&project_a).await.unwrap();
        state
            .neo4j
            .add_project_to_workspace(ws.id, project_a.id)
            .await
            .unwrap();

        let project_b = test_project_named("ProjectB");
        state.neo4j.create_project(&project_b).await.unwrap();
        state
            .neo4j
            .add_project_to_workspace(ws.id, project_b.id)
            .await
            .unwrap();

        let ctx = fetch_project_context(&state.neo4j, &project_a.slug)
            .await
            .unwrap();

        assert!(ctx.project.is_some());
        assert_eq!(ctx.project.as_ref().unwrap().name, "ProjectA");
        assert!(ctx.workspace.is_some());
        assert_eq!(ctx.sibling_projects.len(), 1);
        assert_eq!(ctx.sibling_projects[0].name, "ProjectB");
    }

    #[tokio::test]
    async fn test_fetch_project_context_no_siblings_without_workspace() {
        use crate::test_helpers::{mock_app_state, test_project};

        let state = mock_app_state();
        let project = test_project();
        state.neo4j.create_project(&project).await.unwrap();

        let ctx = fetch_project_context(&state.neo4j, &project.slug)
            .await
            .unwrap();

        assert!(ctx.project.is_some());
        assert!(ctx.workspace.is_none());
        assert!(ctx.sibling_projects.is_empty());
    }

    #[test]
    fn test_context_to_markdown_with_global_notes() {
        let ctx = ProjectContext {
            project: Some(ProjectNode {
                id: uuid::Uuid::new_v4(),
                name: "TestProj".into(),
                slug: "test-proj".into(),
                root_path: "/tmp".into(),
                description: None,
                created_at: Utc::now(),
                last_synced: Some(Utc::now()),
                analytics_computed_at: None,
            }),
            global_guidelines: vec![{
                let mut n = crate::notes::Note::new(
                    None,
                    crate::notes::NoteType::Guideline,
                    "Always write tests".to_string(),
                    "test".to_string(),
                );
                n.importance = crate::notes::NoteImportance::Critical;
                n
            }],
            global_gotchas: vec![crate::notes::Note::new(
                None,
                crate::notes::NoteType::Gotcha,
                "Beware of circular deps".to_string(),
                "test".to_string(),
            )],
            ..Default::default()
        };
        let md = context_to_markdown(&ctx, None);
        assert!(md.contains("Guidelines globales"));
        assert!(md.contains("Always write tests"));
        assert!(md.contains("Gotchas globaux"));
        assert!(md.contains("Beware of circular deps"));
    }

    // ================================================================
    // Tool catalog tests
    // ================================================================

    #[test]
    fn test_tool_groups_cover_all_158_tools() {
        let count = tool_catalog_tool_count();
        assert_eq!(
            count, 158,
            "TOOL_GROUPS must cover exactly 158 unique tools (got {}). \
             Update the catalog when adding/removing MCP tools.",
            count
        );
    }

    #[test]
    fn test_tool_groups_no_duplicates() {
        let mut all_names: Vec<&str> = TOOL_GROUPS
            .iter()
            .flat_map(|g| g.tools.iter().map(|t| t.name))
            .collect();
        let total = all_names.len();
        all_names.sort();
        all_names.dedup();
        assert_eq!(
            all_names.len(),
            total,
            "Duplicate tool names found in TOOL_GROUPS"
        );
    }

    #[test]
    fn test_tool_groups_match_mcp_tools() {
        let mcp_tools = crate::mcp::tools::all_tools();
        let catalog_set: std::collections::HashSet<&str> = TOOL_GROUPS
            .iter()
            .flat_map(|g| g.tools.iter().map(|t| t.name))
            .collect();
        let mcp_set: std::collections::HashSet<String> =
            mcp_tools.iter().map(|t| t.name.clone()).collect();

        for tool in &mcp_tools {
            assert!(
                catalog_set.contains(tool.name.as_str()),
                "MCP tool '{}' missing from TOOL_GROUPS catalog",
                tool.name
            );
        }
        for name in &catalog_set {
            assert!(
                mcp_set.contains(*name),
                "Catalog tool '{}' not found in MCP all_tools()",
                name
            );
        }
    }

    #[test]
    fn test_tool_groups_count() {
        assert_eq!(TOOL_GROUPS.len(), 15, "Expected 15 tool groups");
    }

    #[test]
    fn test_tool_groups_all_have_keywords_and_tools() {
        for group in TOOL_GROUPS {
            assert!(
                !group.keywords.is_empty(),
                "Group '{}' has no keywords",
                group.name
            );
            assert!(
                !group.tools.is_empty(),
                "Group '{}' has no tools",
                group.name
            );
        }
    }

    #[test]
    fn test_tools_catalog_to_json() {
        let json = tools_catalog_to_json(TOOL_GROUPS);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.len(), TOOL_GROUPS.len());
        assert_eq!(parsed[0]["group"], "planning");
        assert!(parsed[0]["tools"].as_array().unwrap().len() > 5);
    }

    #[test]
    fn test_format_tool_groups_markdown() {
        let groups: Vec<&ToolGroup> = vec![&TOOL_GROUPS[0], &TOOL_GROUPS[2]];
        let md = format_tool_groups_markdown(&groups);
        assert!(md.contains("## Tools recommandés"));
        assert!(md.contains("create_plan"));
        assert!(md.contains("search_code"));
        assert!(!md.contains("create_note")); // knowledge group not included
    }

    // ================================================================
    // select_tool_groups_by_keywords tests
    // ================================================================

    #[test]
    fn test_select_tool_groups_planning_message() {
        let groups = select_tool_groups_by_keywords("Je veux planifier une nouvelle feature");
        let names: Vec<&str> = groups.iter().map(|g| g.name).collect();
        assert!(
            names.contains(&"planning"),
            "Expected planning group, got {:?}",
            names
        );
    }

    #[test]
    fn test_select_tool_groups_code_message() {
        let groups =
            select_tool_groups_by_keywords("Explore le code de la fonction handle_request");
        let names: Vec<&str> = groups.iter().map(|g| g.name).collect();
        assert!(
            names.contains(&"code_exploration"),
            "Expected code_exploration, got {:?}",
            names
        );
    }

    #[test]
    fn test_select_tool_groups_knowledge_message() {
        let groups = select_tool_groups_by_keywords("Crée une note gotcha pour ce pattern");
        let names: Vec<&str> = groups.iter().map(|g| g.name).collect();
        assert!(
            names.contains(&"knowledge"),
            "Expected knowledge, got {:?}",
            names
        );
    }

    #[test]
    fn test_select_tool_groups_empty_message_fallback() {
        let groups = select_tool_groups_by_keywords("bonjour comment ça va");
        let names: Vec<&str> = groups.iter().map(|g| g.name).collect();
        // Fallback: planning + code_exploration
        assert!(
            names.contains(&"planning"),
            "Fallback should include planning, got {:?}",
            names
        );
        assert!(
            names.contains(&"code_exploration"),
            "Fallback should include code_exploration, got {:?}",
            names
        );
    }

    #[test]
    fn test_select_tool_groups_mixed_message() {
        let groups =
            select_tool_groups_by_keywords("Planifie le refactoring du code et crée un commit");
        let names: Vec<&str> = groups.iter().map(|g| g.name).collect();
        assert!(
            names.contains(&"planning"),
            "Expected planning, got {:?}",
            names
        );
        assert!(
            names.contains(&"code_exploration"),
            "Expected code_exploration, got {:?}",
            names
        );
        assert!(
            names.contains(&"git_tracking"),
            "Expected git_tracking, got {:?}",
            names
        );
    }

    #[test]
    fn test_select_tool_groups_always_returns_at_least_one() {
        let groups = select_tool_groups_by_keywords("");
        assert!(
            !groups.is_empty(),
            "Should always return at least one group"
        );
    }

    #[test]
    fn test_context_to_markdown_with_tools() {
        let ctx = ProjectContext {
            project: Some(ProjectNode {
                id: uuid::Uuid::new_v4(),
                name: "TestProj".into(),
                slug: "test-proj".into(),
                root_path: "/tmp".into(),
                description: None,
                created_at: Utc::now(),
                last_synced: Some(Utc::now()),
                analytics_computed_at: None,
            }),
            ..Default::default()
        };
        let md = context_to_markdown(&ctx, Some("Explore le code"));
        assert!(md.contains("TestProj"));
        assert!(md.contains("## Tools recommandés"));
        assert!(md.contains("search_code")); // code_exploration group
    }

    fn make_feature_graphs(count: usize) -> Vec<FeatureGraphNode> {
        (0..count)
            .map(|i| FeatureGraphNode {
                id: uuid::Uuid::new_v4(),
                name: format!("Feature-{}", i),
                description: Some(format!("Description for feature graph number {}", i)),
                project_id: uuid::Uuid::new_v4(),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                entity_count: Some((i as i64 + 1) * 10),
                entry_function: None,
                build_depth: None,
                include_relations: None,
            })
            .collect()
    }

    #[test]
    fn test_context_to_json_with_feature_graphs() {
        let ctx = ProjectContext {
            feature_graphs: make_feature_graphs(3),
            ..Default::default()
        };
        let json = context_to_json(&ctx);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let fgs = parsed["feature_graphs"].as_array().unwrap();
        assert_eq!(fgs.len(), 3);
        assert_eq!(fgs[0]["name"], "Feature-0");
        assert_eq!(fgs[0]["entity_count"], 10);
        assert_eq!(fgs[1]["name"], "Feature-1");
        assert_eq!(fgs[2]["entity_count"], 30);
        // Verify id is present as a string
        assert!(fgs[0]["id"].as_str().is_some());
    }

    #[test]
    fn test_context_to_markdown_with_feature_graphs() {
        let ctx = ProjectContext {
            project: Some(ProjectNode {
                id: uuid::Uuid::new_v4(),
                name: "TestProj".into(),
                slug: "test-proj".into(),
                root_path: "/tmp".into(),
                description: None,
                created_at: Utc::now(),
                last_synced: Some(Utc::now()),
                analytics_computed_at: None,
            }),
            feature_graphs: make_feature_graphs(3),
            ..Default::default()
        };
        let md = context_to_markdown(&ctx, None);
        assert!(md.contains("## Feature Graphs"));
        assert!(md.contains("**Feature-0**"));
        assert!(md.contains("**Feature-1**"));
        assert!(md.contains("**Feature-2**"));
        assert!(md.contains("10 entités"));
        assert!(md.contains("20 entités"));
        assert!(md.contains("30 entités"));
        assert!(md.contains("get_feature_graph(id)"));
    }

    #[test]
    fn test_context_to_markdown_empty_feature_graphs() {
        let ctx = ProjectContext {
            project: Some(ProjectNode {
                id: uuid::Uuid::new_v4(),
                name: "TestProj".into(),
                slug: "test-proj".into(),
                root_path: "/tmp".into(),
                description: None,
                created_at: Utc::now(),
                last_synced: Some(Utc::now()),
                analytics_computed_at: None,
            }),
            ..Default::default()
        };
        let md = context_to_markdown(&ctx, None);
        assert!(!md.contains("Feature Graphs"));
        assert!(!md.contains("get_feature_graph"));
    }

    #[test]
    fn test_feature_graphs_token_budget() {
        // Build 15 feature graphs with long descriptions
        let fgs: Vec<FeatureGraphNode> = (0..15)
            .map(|i| FeatureGraphNode {
                id: uuid::Uuid::new_v4(),
                name: format!("Feature-Graph-{}", i),
                description: Some("A".repeat(200)), // 200 char description, will be truncated
                project_id: uuid::Uuid::new_v4(),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                entity_count: Some(100),
                entry_function: None,
                build_depth: None,
                include_relations: None,
            })
            .collect();
        let ctx = ProjectContext {
            project: Some(ProjectNode {
                id: uuid::Uuid::new_v4(),
                name: "TestProj".into(),
                slug: "test-proj".into(),
                root_path: "/tmp".into(),
                description: None,
                created_at: Utc::now(),
                last_synced: Some(Utc::now()),
                analytics_computed_at: None,
            }),
            feature_graphs: fgs,
            ..Default::default()
        };
        let md = context_to_markdown(&ctx, None);
        // Extract only the Feature Graphs section
        let section_start = md.find("## Feature Graphs").unwrap();
        let section_end = md[section_start..]
            .find("\n## ")
            .or_else(|| md[section_start..].find("\n- **Dernière sync**"))
            .map(|pos| section_start + pos)
            .unwrap_or(md.len());
        let section = &md[section_start..section_end];
        // ~300 tokens ≈ ~1200 chars. With 15 graphs at 80 char truncated desc, should be well under 2000 chars.
        assert!(
            section.len() < 2500,
            "Feature Graphs section is too large: {} chars (should be < 2500)",
            section.len()
        );
        // Verify descriptions are truncated (original 200 chars → max 80 + "…")
        assert!(
            section.contains("…"),
            "Long descriptions should be truncated with …"
        );
    }

    // ================================================================
    // GDS topology section tests
    // ================================================================

    #[tokio::test]
    async fn test_topology_section_with_full_data() {
        use crate::graph::models::FileAnalyticsUpdate;
        use crate::meilisearch::mock::MockSearchStore;
        use crate::neo4j::mock::MockGraphStore;
        use crate::test_helpers::{mock_app_state_with, test_project_named};

        let graph = MockGraphStore::new();
        let project = test_project_named("topo-proj");
        graph.create_project(&project).await.unwrap();

        let file_paths = vec![
            "src/api/handlers.rs",
            "src/api/routes.rs",
            "src/neo4j/client.rs",
            "src/neo4j/models.rs",
        ];
        graph
            .project_files
            .write()
            .await
            .entry(project.id)
            .or_default()
            .extend(file_paths.iter().map(|s| s.to_string()));

        for path in &file_paths {
            let file = crate::neo4j::models::FileNode {
                path: path.to_string(),
                language: "rust".to_string(),
                hash: "test".to_string(),
                last_parsed: chrono::Utc::now(),
                project_id: Some(project.id),
            };
            graph.upsert_file(&file).await.unwrap();
        }

        // Analytics → 2 communities + bridge
        graph
            .batch_update_file_analytics(&[
                FileAnalyticsUpdate {
                    path: "src/api/handlers.rs".to_string(),
                    pagerank: 0.8,
                    betweenness: 0.5,
                    community_id: 0,
                    community_label: "api".to_string(),
                    clustering_coefficient: 0.4,
                    component_id: 0,
                },
                FileAnalyticsUpdate {
                    path: "src/api/routes.rs".to_string(),
                    pagerank: 0.4,
                    betweenness: 0.1,
                    community_id: 0,
                    community_label: "api".to_string(),
                    clustering_coefficient: 0.3,
                    component_id: 0,
                },
                FileAnalyticsUpdate {
                    path: "src/neo4j/client.rs".to_string(),
                    pagerank: 0.9,
                    betweenness: 0.8,
                    community_id: 1,
                    community_label: "neo4j".to_string(),
                    clustering_coefficient: 0.6,
                    component_id: 0,
                },
                FileAnalyticsUpdate {
                    path: "src/neo4j/models.rs".to_string(),
                    pagerank: 0.3,
                    betweenness: 0.05,
                    community_id: 1,
                    community_label: "neo4j".to_string(),
                    clustering_coefficient: 0.5,
                    component_id: 0,
                },
            ])
            .await
            .unwrap();

        let state = mock_app_state_with(graph, MockSearchStore::new());

        let ctx = fetch_project_context(&state.neo4j, "topo-proj")
            .await
            .unwrap();

        assert!(
            ctx.structural_topology.is_some(),
            "should have structural topology"
        );
        let topo = ctx.structural_topology.unwrap();

        // Check communities section
        assert!(
            topo.contains("Code Communities"),
            "should have communities header"
        );
        assert!(topo.contains("api"), "should mention api community");
        assert!(topo.contains("neo4j"), "should mention neo4j community");

        // Check bridges section
        assert!(topo.contains("Bridge Files"), "should have bridges header");
        assert!(
            topo.contains("neo4j/client.rs"),
            "neo4j/client.rs should be a bridge (highest betweenness)"
        );

        // Size check: < 2000 chars
        assert!(
            topo.len() < 2000,
            "topology section should be compact, got {} chars",
            topo.len()
        );
    }

    #[tokio::test]
    async fn test_topology_section_no_gds_data() {
        use crate::test_helpers::{mock_app_state, test_project_named};

        let state = mock_app_state();
        let project = test_project_named("no-gds-proj");
        state.neo4j.create_project(&project).await.unwrap();

        let ctx = fetch_project_context(&state.neo4j, "no-gds-proj")
            .await
            .unwrap();

        assert!(
            ctx.structural_topology.is_none(),
            "should be None when no GDS data"
        );

        // Verify markdown output has no topology
        let md = context_to_markdown(&ctx, None);
        assert!(
            !md.contains("Topologie structurelle"),
            "markdown should not contain topology section"
        );
    }

    #[tokio::test]
    async fn test_topology_section_in_json() {
        use crate::graph::models::FileAnalyticsUpdate;
        use crate::meilisearch::mock::MockSearchStore;
        use crate::neo4j::mock::MockGraphStore;
        use crate::test_helpers::{mock_app_state_with, test_project_named};

        let graph = MockGraphStore::new();
        let project = test_project_named("json-topo");
        graph.create_project(&project).await.unwrap();

        graph
            .project_files
            .write()
            .await
            .entry(project.id)
            .or_default()
            .push("src/main.rs".to_string());

        let file = crate::neo4j::models::FileNode {
            path: "src/main.rs".to_string(),
            language: "rust".to_string(),
            hash: "test".to_string(),
            last_parsed: chrono::Utc::now(),
            project_id: Some(project.id),
        };
        graph.upsert_file(&file).await.unwrap();

        graph
            .batch_update_file_analytics(&[FileAnalyticsUpdate {
                path: "src/main.rs".to_string(),
                pagerank: 0.5,
                betweenness: 0.2,
                community_id: 0,
                community_label: "main".to_string(),
                clustering_coefficient: 0.3,
                component_id: 0,
            }])
            .await
            .unwrap();

        let state = mock_app_state_with(graph, MockSearchStore::new());
        let ctx = fetch_project_context(&state.neo4j, "json-topo")
            .await
            .unwrap();

        let json_str = context_to_json(&ctx);
        assert!(
            json_str.contains("structural_topology"),
            "JSON should contain structural_topology key"
        );
        assert!(
            json_str.contains("Code Communities"),
            "JSON should contain communities data"
        );
    }
}
