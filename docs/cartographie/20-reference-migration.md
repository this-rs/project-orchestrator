# Référence de migration : comptage par type de la base vivante

Relevé le **2026-11-11T01:17:15Z**, en lecture seule (`cypher-shell --access-mode read`), par `apoc.meta.stats()` qui lit le *count store* de Neo4j sans parcourir les données. Neo4j 5.26.20 community, conteneur `orchestrator-neo4j`, 15 Go de base + 2,3 Go de journaux de transactions.

C'est la **méthode 1**. Le critère exige une seconde méthode indépendante (parcours complet par label), qui sera faite sur la **copie**, pas sur la base vivante.

## Totaux

| Mesure | Valeur |
|---|---:|
| Nœuds | 3 847 985 |
| Relations | 5 913 357 |
| Labels déclarés / peuplés | 70 / 64 |
| Types de relation déclarés / peuplés | 82 / 80 |

## Constats à préserver, pas à corriger

1. **Un label par nœud** : la somme des comptes par label égale le nombre de nœuds. Cette égalité est nécessaire mais ne suffit pas (un nœud sans label compenserait un nœud à deux labels) ; le fait est établi sur la copie par `size(labels(n)) = 1` pour tous les nœuds (`22-copie-figee.md`).
2. **6 labels et 2 types de relation déclarés sans données.** Ils font partie du schéma et doivent exister dans la cible ; leurs noms ont été relevés sur la copie (`22-copie-figee.md` : `DayPlan`, `ExternalEvent`, `PredictedLink`, `SynapseShadowTrace`, `TopologyRule`, `TriggerSignal` ; `CO_CHANGED_TRANSITIVE`, `PREDICTED_LINK`).
3. **418 `ChatEvent` sans `HAS_EVENT`** (2 692 558 événements, 2 692 140 liens). Sur la base vivante, le chiffre est **déduit** de la différence des deux compteurs (le *count store* ne permet pas de le compter). Il est **compté directement** sur la copie le 13/11 : 418, et chaque `HAS_EVENT` vise un `ChatEvent` distinct (`22-copie-figee.raw.txt`).
4. **La base vivante écrit en continu** (`ChatEvent` domine à 70 %). Ces chiffres sont un instantané : la comparaison source/cible n'a de sens que sur une source figée — d'où la migration au démarrage, avant toute écriture (décision f8e1f200).

## Défauts du convertisseur existant (`src/neo4j_to_obrain.rs` du dépôt `obrain-db/neo4j2obrain`, hors de ce dépôt ; relu au commit afaaaf5)

- `bolt_to_value` (l. 51) retire les `null` des listes (`filter_map`, l. 62) et des maps (l. 66-72) : une liste change de longueur et ses positions bougent.
- Dates, heures et durées deviennent `format!("{:?}")` (l. 78) : le type est perdu.
- Le mot de passe a une valeur par défaut écrite dans le source (`default_value`, l. 34).

Il sert de point de départ ; il ne satisfait pas la garantie « aucune perte ».

## Nœuds par label

| Label | Nœuds |
|---|---:|
| ChatEvent | 2692558 |
| Function | 559733 |
| Import | 308447 |
| Impl | 63287 |
| Struct | 50085 |
| File | 33789 |
| Note | 31179 |
| TriggerFiring | 27489 |
| Step | 18365 |
| RefreshToken | 12944 |
| Commit | 9880 |
| Task | 6619 |
| RoutingDecision | 6175 |
| Trait | 4639 |
| Enum | 3669 |
| LlmSetting | 2640 |
| ChatSession | 2415 |
| ConversationSegment | 2333 |
| Constraint | 1667 |
| ConversationBoundary | 1633 |
| Plan | 1153 |
| Decision | 1034 |
| TrajectoryNode | 745 |
| FeatureGraph | 719 |
| ProtocolRun | 628 |
| SessionEpisode | 518 |
| ProtocolTransition | 474 |
| Skill | 448 |
| Process | 437 |
| ProtocolState | 427 |
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
| Persona | 55 |
| RoutingArm | 54 |
| Project | 51 |
| Release | 19 |
| Agent | 19 |
| ExternalDelivery | 18 |
| DataMigration | 15 |
| Resource | 11 |
| AnalysisProfile | 9 |
| WorkspaceMilestone | 7 |
| Workspace | 7 |
| McpToken | 5 |
| Trigger | 4 |
| Deployment | 4 |
| User | 3 |
| ToolNode | 3 |
| McpServer | 3 |
| Environment | 3 |
| ExecutionPlaceEvent | 2 |
| LifecycleHook | 1 |
| AnchorEvent | 1 |
| Anchor | 1 |
| AgentIdentityRoot | 1 |

## Relations par type

| Type | Relations |
|---|---:|
| HAS_EVENT | 2692140 |
| CALLS | 1026589 |
| CONTAINS | 714732 |
| LINKED_TO | 403927 |
| INCLUDES_ENTITY | 334304 |
| HAS_IMPORT | 208399 |
| SYNAPSE | 99724 |
| IMPORTS_SYMBOL | 89627 |
| IMPORTS | 70932 |
| IMPLEMENTS_FOR | 62570 |
| IMPLEMENTS_TRAIT | 55933 |
| HAS_NOTE | 27973 |
| FIRED_BY | 27489 |
| HAS_STEP | 18361 |
| EXTENDS | 10760 |
| TOUCHES | 10689 |
| DISCUSSED | 8185 |
| HAS_TASK | 6619 |
| ROUTED_FOR | 6100 |
| DEPENDS_ON | 4799 |
| RESOLVED_BY | 3882 |
| ASSOCIATED_WITH | 2983 |
| RESULTED_IN | 2086 |
| HAS_CHAT_SESSION | 2053 |
| CONSTRAINED_BY | 1665 |
| PRECEDES | 1633 |
| TOUCHED_ENTITY | 1474 |
| STEP_IN_PROCESS | 1350 |
| BELONGS_TO | 1285 |
| HAS_PLAN | 1153 |
| SPAWNED_BY | 1118 |
| CO_CHANGED | 989 |
| KNOWS | 941 |
| AFFECTS | 941 |
| INFORMED_BY | 940 |
| MEMBER_OF | 775 |
| AUTHORED | 677 |
| USED_TOOL | 671 |
| INSTANCE_OF | 628 |
| INCLUDES_TASK | 522 |
| EPISODE_OF | 518 |
| LIVED | 507 |
| HAS_TRANSITION | 474 |
| NEXT_DECISION | 452 |
| HAS_STATE | 427 |
| TARGETS_MILESTONE | 397 |
| USES | 371 |
| PRODUCED_DURING | 336 |
| PART_OF | 335 |
| EXECUTES | 335 |
| HAS_DOCUMENT | 146 |
| FORGOTTEN | 130 |
| HAS_ALERT | 127 |
| REASONING_FOR | 118 |
| HAS_MILESTONE | 114 |
| HAS_CHUNK | 105 |
| IMPLEMENTS | 101 |
| RUNS | 92 |
| MEMBER_OF_SKILL | 82 |
| HAS_DECISION | 81 |
| SUPERSEDES | 71 |
| HAS_COMPONENT | 60 |
| BELONGS_TO_SKILL | 50 |
| DEPENDS_ON_COMPONENT | 48 |
| MAPS_TO_PROJECT | 47 |
| FOLLOWS | 42 |
| BELONGS_TO_WORKSPACE | 40 |
| MASTERS | 27 |
| INCLUDES_COMMIT | 20 |
| HAS_RELEASE | 19 |
| FACET_OF | 19 |
| HAS_RESOURCE | 11 |
| USED_SKILL | 9 |
| SPAWNED_BY_RUN | 9 |
| HAS_WORKSPACE_MILESTONE | 7 |
| HAS_DEPLOYMENT | 4 |
| HAS_ENVIRONMENT | 3 |
| LINKED_TO_PLAN | 2 |
| HAS_PLACE_EVENT | 2 |
| DURING_RUN | 1 |
