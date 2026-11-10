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
