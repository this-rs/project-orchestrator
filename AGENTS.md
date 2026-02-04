# AGENTS.md

Guidelines for AI agents working with Project Orchestrator.

## Workflow

### 1. Récupérer le plan et la tâche

```bash
# Récupérer le plan complet (1 call)
GET /api/plans/{plan_id}

# Ou juste la prochaine tâche disponible
GET /api/plans/{plan_id}/next-task
```

### 2. Marquer la tâche comme "in_progress"

```bash
PATCH /api/tasks/{task_id}
{"status": "in_progress", "assigned_to": "agent-id"}
```

### 3. Suivre les steps

Pour chaque step de la tâche :
1. Lire la description et le critère de vérification
2. Effectuer le travail
3. Marquer le step comme complété

```bash
PATCH /api/steps/{step_id}
{"status": "completed"}
```

### 4. Respecter les constraints

Avant de valider le travail, vérifier que toutes les constraints du plan sont respectées :
- `style` → Lancer les linters (ex: `cargo clippy`, `eslint`)
- `testing` → Lancer les tests (ex: `cargo test`)
- `security` → Vérifier les vulnérabilités
- `compatibility` → Vérifier la rétro-compatibilité

### 5. Commit des changements

**IMPORTANT : Toujours commiter à la fin de chaque tâche complétée.**

Format du commit :
```
<type>(<scope>): <description courte>

<Task title>

Changes:
- <changement 1>
- <changement 2>

Tests:
- <test ajouté/modifié>

Files modified:
- <fichier 1>
- <fichier 2>

Co-Authored-By: <Agent Name> <email>
```

Types de commit :
- `feat` : Nouvelle fonctionnalité
- `fix` : Correction de bug
- `refactor` : Refactoring sans changement de comportement
- `docs` : Documentation
- `test` : Ajout/modification de tests
- `chore` : Maintenance, dépendances

### 6. Marquer la tâche comme complétée

```bash
PATCH /api/tasks/{task_id}
{"status": "completed"}
```

### 7. Reporter via webhook (optionnel)

```bash
POST /api/wake
{
  "task_id": "uuid",
  "success": true,
  "summary": "Description du travail effectué",
  "files_modified": ["src/file1.rs", "src/file2.rs"]
}
```

## Récupérer du contexte additionnel

### Contexte code

```bash
# Recherche sémantique
GET /api/code/search?q=error+handling&limit=10

# Symboles d'un fichier
GET /api/code/symbols/{path}

# Dépendances
GET /api/code/dependencies/{path}

# Impact d'une modification
GET /api/code/impact?target={path}&target_type=file
```

### Décisions passées

```bash
# Rechercher des décisions similaires
GET /api/decisions/search?q=authentication&limit=5
```

### Enregistrer une décision

Quand vous prenez une décision architecturale importante :

```bash
POST /api/tasks/{task_id}/decisions
{
  "description": "Utiliser JWT au lieu des sessions",
  "rationale": "Meilleur pour une API stateless",
  "alternatives": ["Sessions", "OAuth"],
  "chosen_option": "JWT avec refresh tokens"
}
```

## Bonnes pratiques

### Code

1. **Lire avant d'écrire** - Toujours lire les fichiers existants avant de les modifier
2. **Tests** - Ajouter des tests pour chaque nouvelle fonctionnalité
3. **Petit commits** - Un commit par tâche, pas de commits géants
4. **Documentation** - Mettre à jour README/CLAUDE.md si nécessaire

### Communication

1. **Steps** - Mettre à jour le statut des steps au fur et à mesure
2. **Décisions** - Enregistrer les décisions importantes dans l'API
3. **Blocages** - Si bloqué, marquer la tâche comme `blocked` et expliquer

### Gestion des erreurs

Si une tâche échoue :
1. Ne PAS marquer comme `completed`
2. Marquer comme `failed` avec un message explicatif
3. Reporter via webhook avec `success: false`

```bash
POST /api/wake
{
  "task_id": "uuid",
  "success": false,
  "summary": "Échec: tests ne passent pas après modification de X"
}
```

## Checklist avant de terminer une tâche

- [ ] Tous les steps sont complétés
- [ ] Les tests passent
- [ ] Les constraints sont respectées (clippy, tests, etc.)
- [ ] Les fichiers modifiés sont commités
- [ ] Le commit suit le format standard
- [ ] La tâche est marquée comme `completed`
