# Branche de consolidation — Obrain par défaut + this-rs comme framework

Cette branche (`consolidation/obrain-default`) est le lieu unique du chantier.
Plan PO : `bf3ab074-467e-4b34-b92e-4b19040d8ece` (projet PO Backend).

## Règles de la branche

1. **Rien ne remonte dans `main` avant que la porte de bascule soit verte.** Aucune PR vers `main` n'est ouverte tant que ce fichier dit autre chose.
2. **`main` redescend ICI chaque semaine** (`git merge origin/main` dans cette branche), jamais l'inverse. Une branche qui ne voit jamais `main` pendant des mois devient infusible : c'est exactement ce qui est arrivé à `consolidation/obrain` (2 commits, abandonnée).
3. **Chaque agent travaille dans son propre worktree**, sur une branche fille de celle-ci. Jamais dans un checkout principal, jamais directement sur cette branche.
4. **Un lot = un domaine.** Un commit qui touche stockage ET routes n'est pas relisible : on le coupe.

## Transferts assumés (décisions consignées)

- **Obrain porte un index vectoriel à dimension native** (HNSW persistant sous substrat). Mesure du 10/11 : `create_vector_index(dim=768)` refusé (L2_DIM = 384) ; cascade 384-d recall@10 0,63 à N=8 131 contre 0,94 pour Neo4j ; latence obrain 0,30 ms contre 1,27-2,05 ms.
- **PO est l'écrivain unique de la base obrain.** La v0.0.1 ne garantit pas les propriétés, les transactions ni la visibilité inter-processus.
- **NATS disparaît.** EventBus this-rs + journal d'événements durable dans obrain + WS/SSE. Conséquence assumée : topologie mono-instance pour le chat, plus d'interrupt cross-instance.
- **this-rs devient le framework**, intégration complète, par lots. MCP JSON-RPC reste hors framework.

## Porte de bascule (« ça fonctionne complètement »)

| # | Condition | État |
|---|---|---|
| a | Conformité générique identique sur Neo4j et Obrain (mêmes valeurs, même ordre) | à faire |
| b | recall@10 ≥ 0,95 à 25 000 vecteurs 768-d, après réouverture | à faire |
| c | 4,5 M nœuds migrés, aller-retour vérifié | à faire |
| d | Double-run sans divergence de justesse | à faire |
| e | Retour arrière = une variable de configuration (`storage.backend=default`) | à faire |

## État des lots

| Lot | Tâche PO | État |
|---|---|---|
| Base obrain figée | O-BASE | à faire |
| Spike exécuté (7 verdicts) | O-SPIKE | à faire |
| HNSW natif côté obrain | O-HNSW | à faire |
| Socle this-rs à côté d'axum | O-THISRS-1 | à faire |
| Backend obrain derrière la fabrique | O-DOUBLE | à faire |
| Retrait de NATS | O-NATS-OUT | à faire |

## Les portillons de cette branche (en place)

| Portillon | Ce qu'il empêche | Commande |
|---|---|---|
| `containment` | la branche qui dérive de `main` pendant des mois sans que personne ne le voie | `cargo xtask containment` |
| `path-pins` | un voisin (this, obrain, atelier) qui bouge sous les pieds sans qu'aucun diff ne le montre | `cargo xtask path-pins` |

Les deux tournent dans le job « Lint & Format », donc ils sont bloquants par héritage — sans toucher à la protection de branche de `main`.

`pins.toml` dit contre QUELS commits on construit. Mettre un `rev` à jour est un acte délibéré, dans le même commit que le besoin qui le motive.

`path-pins` ne fait pas de `git fetch` par défaut : un voisin dont le remote est en ssh bloquerait le portillon. `--fetch` l'active.

## À savoir avant d'écrire une ligne ici : `backend/` est un LIEN

`/Users/triviere/projects/project-orchestrator/backend` est un lien symbolique vers `/Users/triviere/.openclaw/workspace/skills/project-orchestrator`, qui est le **répertoire du service vivant**.

Conséquences, mesurées le 10/11/2026 :
- La « copie principale » du dépôt n'est pas `backend/` : c'est ce répertoire-là. `git rev-parse --show-toplevel` depuis `backend/` le dit.
- Éditer `backend/` **c'est** éditer le service vivant. La consigne « ne jamais toucher `~/.openclaw/workspace/skills/project-orchestrator` » est donc impossible à tenir en même temps que « éditer `backend/` » : ce sont les mêmes fichiers (même inode).
- Le `pre-push` partage `CARGO_TARGET_DIR` avec cette copie principale : un build lancé depuis un worktree écrit dans le `target/` du service vivant.
- D'où la règle : **tout travail passe par un worktree**, jamais par `backend/`. C'est déjà le cas pour cette branche.
