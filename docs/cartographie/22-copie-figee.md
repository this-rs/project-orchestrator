# Copie figée de la base de PO : le tableau auquel toute migration se compare

Copie prise le **2026-11-11T11:36:02Z → 11:36:15Z** (13 s), Neo4j 5.26.20 community. C'est la **source figée** que `20-reference-migration.md` annonçait : contrairement à la base vivante, elle ne bouge plus, donc une comparaison source/cible y a un sens.

## Totaux

| Mesure | Copie figée (11:36Z) | Base vivante (01:17Z, doc 20) |
|---|---:|---:|
| Nœuds | 3 901 702 | 3 847 985 |
| Relations | 6 013 360 | 5 913 357 |
| Labels déclarés / peuplés | 70 / 64 | 70 / 64 |
| Types de relation déclarés / peuplés | 82 / 80 | 82 / 80 |

L'écart entre les deux colonnes est l'**évolution** de la base vivante en dix heures, pas seulement sa croissance : trois comptes ont **baissé** entre 01:17Z et 11:36Z — `Skill` 448 → 423 (−25), `INCLUDES_ENTITY` 334 304 → 334 202 (−102), `BELONGS_TO` 1 285 → 1 260 (−25). L'hypothèse est une suppression légitime dans la base vivante (resynchronisation des skills, réindexation), pas une perte de la copie, puisque les deux méthodes concordent sur la copie : **non vérifié** (aucun journal des suppressions n'a été relevé). Dans tous les cas, la copie est sa propre référence : on ne compare jamais une cible à la colonne de droite.

## Deux méthodes indépendantes, un seul résultat

| Méthode | Ce qu'elle lit | Durée |
|---|---|---:|
| 1 — *count store* | `MATCH (n:L) RETURN count(n)` et `MATCH ()-[r:T]->() RETURN count(r)`, une requête par label / type : Neo4j répond depuis ses compteurs, sans parcourir | 142 s (152 requêtes) |
| 2 — parcours complet | `MATCH (n) UNWIND labels(n) AS l RETURN l, count(*)` et `MATCH ()-[r]->() RETURN type(r), count(*)` : chaque nœud et chaque relation est lu | 6 s |

`diff` des deux sorties, labels puis types : **vide**. Somme par label = 3 901 702 = nombre de nœuds ; chaque nœud porte **exactement un** label (`size(labels(n)) = 1` pour tous).

## Ce que la copie confirme ou précise

- **ChatEvent sans `HAS_EVENT` : 418** (418 relevés sur la base vivante à 01:17Z). Égal à la différence `ChatEvent` − `HAS_EVENT` des tableaux ci-dessous (2 702 931 − 2 702 513) ; la requête qui l'aurait compté directement n'est pas consignée : **non vérifié** comme compte direct. L'écart est dans la source ; la migration le reproduit et le signale.
- **Labels déclarés sans données (6)** : `DayPlan` `ExternalEvent` `PredictedLink` `SynapseShadowTrace` `TopologyRule` `TriggerSignal` 
- **Types de relation déclarés sans données (2)** : `CO_CHANGED_TRANSITIVE` `PREDICTED_LINK` 
- Ces huit noms font partie du schéma : la cible doit les connaître, même vides.

## Comment la copie a été prise, et ce que ça garantit

La base vivante n'a jamais été ouverte en écriture : son volume est monté **`:ro`** dans un conteneur de copie (un `touch` y rend `Read-only file system`, vérifié avant la copie).

```sh
docker volume create po-mig-copie-data
docker run --rm -v project-orchestrator_neo4j_data:/src:ro -v po-mig-copie-data:/dst alpine sh -c '
  mkdir -p /dst/databases /dst/transactions
  cp -a /src/dbms /src/server_id /dst/
  cp -a /src/databases/neo4j /src/databases/system /dst/databases/      # la base d abord
  cp -a /src/transactions/neo4j /src/transactions/system /dst/transactions/   # les journaux en dernier
  rm -f /dst/databases/store_lock'
docker run -d --name po-mig-copie-neo4j --memory 6g \
  -p 127.0.0.1:27474:7474 -p 127.0.0.1:27687:7687 -v po-mig-copie-data:/data \
  -e NEO4J_AUTH=none -e NEO4J_server_memory_heap_max__size=2G -e NEO4J_server_memory_pagecache_size=2G \
  neo4j:5.26.20-community
```

Ports décalés (27474 / 27687), liés à la boucle locale, sans authentification : aucun mot de passe n'est manipulé.

**Limite à connaître.** C'est une copie de fichiers d'une base en marche, pas un `neo4j-admin database dump` (qui exige d'arrêter la base vivante). Au démarrage, Neo4j a déclaré « Transaction logs recovery is required » et a rejoué les journaux depuis le dernier point de contrôle ; la base est `online`, et les deux comptages concordent. `neo4j-admin database check` n'a **pas** été lancé sur la copie : la cohérence est établie par la reprise et par le double comptage, pas par le vérificateur.

## Nettoyage

```sh
docker rm -f po-mig-copie-neo4j && docker volume rm po-mig-copie-data
```

## Nœuds par label

| Label | Nœuds |
|---|---:|
| ChatEvent | 2702931 |
| Function | 572516 |
| Import | 324693 |
| Impl | 73493 |
| Struct | 52721 |
| File | 34512 |
| Note | 31212 |
| TriggerFiring | 27489 |
| Step | 18365 |
| RefreshToken | 12963 |
| Commit | 9882 |
| Task | 6623 |
| RoutingDecision | 6243 |
| Trait | 4711 |
| Enum | 4135 |
| LlmSetting | 2681 |
| ChatSession | 2439 |
| ConversationSegment | 2342 |
| Constraint | 1667 |
| ConversationBoundary | 1642 |
| Plan | 1154 |
| Decision | 1041 |
| TrajectoryNode | 745 |
| FeatureGraph | 719 |
| ProtocolRun | 629 |
| SessionEpisode | 535 |
| ProtocolTransition | 474 |
| Process | 437 |
| ProtocolState | 427 |
| Skill | 423 |
| AgentExecution | 335 |
| Trajectory | 293 |
| TouchedNode | 167 |
| Alert | 145 |
| Milestone | 114 |
| DocumentChunk | 105 |
| PersistedReasoningTree | 94 |
| PlanRun | 92 |
| Document | 83 |
| Protocol | 72 |
| Component | 60 |
| RoutingArm | 56 |
| Persona | 55 |
| Project | 51 |
| Agent | 19 |
| Release | 19 |
| ExternalDelivery | 18 |
| DataMigration | 15 |
| Resource | 11 |
| AnalysisProfile | 9 |
| Workspace | 7 |
| WorkspaceMilestone | 7 |
| McpToken | 5 |
| Deployment | 4 |
| Trigger | 4 |
| Environment | 3 |
| McpServer | 3 |
| ToolNode | 3 |
| User | 3 |
| ExecutionPlaceEvent | 2 |
| AgentIdentityRoot | 1 |
| Anchor | 1 |
| AnchorEvent | 1 |
| LifecycleHook | 1 |
| DayPlan | 0 |
| ExternalEvent | 0 |
| PredictedLink | 0 |
| SynapseShadowTrace | 0 |
| TopologyRule | 0 |
| TriggerSignal | 0 |

## Relations par type

| Type | Relations |
|---|---:|
| HAS_EVENT | 2702513 |
| CALLS | 1048016 |
| CONTAINS | 741618 |
| LINKED_TO | 409594 |
| INCLUDES_ENTITY | 334202 |
| HAS_IMPORT | 213851 |
| SYNAPSE | 100326 |
| IMPORTS_SYMBOL | 98618 |
| IMPLEMENTS_FOR | 72773 |
| IMPORTS | 71624 |
| IMPLEMENTS_TRAIT | 65463 |
| HAS_NOTE | 28004 |
| FIRED_BY | 27489 |
| HAS_STEP | 18361 |
| EXTENDS | 10760 |
| TOUCHES | 10693 |
| DISCUSSED | 8210 |
| HAS_TASK | 6623 |
| ROUTED_FOR | 6168 |
| DEPENDS_ON | 4800 |
| RESOLVED_BY | 3882 |
| ASSOCIATED_WITH | 2992 |
| RESULTED_IN | 2086 |
| HAS_CHAT_SESSION | 2077 |
| CONSTRAINED_BY | 1665 |
| PRECEDES | 1642 |
| TOUCHED_ENTITY | 1474 |
| STEP_IN_PROCESS | 1350 |
| BELONGS_TO | 1260 |
| HAS_PLAN | 1154 |
| SPAWNED_BY | 1142 |
| CO_CHANGED | 989 |
| AFFECTS | 942 |
| INFORMED_BY | 941 |
| KNOWS | 941 |
| MEMBER_OF | 810 |
| AUTHORED | 696 |
| USED_TOOL | 671 |
| INSTANCE_OF | 629 |
| EPISODE_OF | 535 |
| LIVED | 524 |
| INCLUDES_TASK | 522 |
| HAS_TRANSITION | 474 |
| NEXT_DECISION | 452 |
| HAS_STATE | 427 |
| TARGETS_MILESTONE | 397 |
| USES | 373 |
| PRODUCED_DURING | 338 |
| EXECUTES | 335 |
| PART_OF | 335 |
| HAS_DOCUMENT | 146 |
| FORGOTTEN | 136 |
| HAS_ALERT | 127 |
| REASONING_FOR | 118 |
| HAS_MILESTONE | 114 |
| HAS_CHUNK | 105 |
| IMPLEMENTS | 101 |
| RUNS | 92 |
| HAS_DECISION | 87 |
| MEMBER_OF_SKILL | 82 |
| SUPERSEDES | 71 |
| HAS_COMPONENT | 60 |
| BELONGS_TO_SKILL | 50 |
| DEPENDS_ON_COMPONENT | 48 |
| MAPS_TO_PROJECT | 47 |
| FOLLOWS | 42 |
| BELONGS_TO_WORKSPACE | 40 |
| MASTERS | 27 |
| INCLUDES_COMMIT | 20 |
| FACET_OF | 19 |
| HAS_RELEASE | 19 |
| HAS_RESOURCE | 11 |
| SPAWNED_BY_RUN | 9 |
| USED_SKILL | 9 |
| HAS_WORKSPACE_MILESTONE | 7 |
| HAS_DEPLOYMENT | 4 |
| HAS_ENVIRONMENT | 3 |
| HAS_PLACE_EVENT | 2 |
| LINKED_TO_PLAN | 2 |
| DURING_RUN | 1 |
| CO_CHANGED_TRANSITIVE | 0 |
| PREDICTED_LINK | 0 |
