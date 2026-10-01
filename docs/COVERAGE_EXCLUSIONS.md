# Registre des exclusions de couverture (backend)

Toute exclusion de couverture est inscrite ici, avec sa raison, son propriétaire
et sa date. La règle : **le nombre d'exclusions ne peut que baisser.** On ne
gonfle jamais le chiffre affiché en retirant du code du dénominateur.

Deux chiffres sont toujours publiés (voir [COVERAGE.md](COVERAGE.md)) :

- **brut** : tout `src/`, sans aucune exclusion.
- **gated** : ce que Codecov affiche, c'est-à-dire le brut privé de la section
  `ignore:` de [`codecov.yml`](../codecov.yml).

Mécanismes d'exclusion utilisés dans ce dépôt : la section `ignore:` de
`codecov.yml`. Aucun `#[coverage(off)]` n'est en place à ce jour ; s'il en
apparaît un, il s'inscrit dans le tableau ci-dessous comme les autres.

## Compteur

| Date | Exclusions `codecov.yml` | dont `src/neo4j/*` | Référence |
|---|---|---|---|
| 2026-10-01 (baseline, tâche 0.3) | 31 | **22** | `origin/main` @ 962e2119 |
| 2026-10-01 (cette PR, tâche 3.1) | 22 | **13** | branche `test/neo4j-coverage-ci` |

Les 9 exclusions retirées sont `src/neo4j/{client,traits,workspace,milestone,constraint,release,commit,decision,user}.rs`.

**Provenance des chiffres.** Les pourcentages des tableaux ci-dessous viennent
de la mesure de la tâche 0.3 (`origin/main` @ 962e2119, voir COVERAGE.md) : ce
sont les chiffres d'AVANT cette PR, c'est-à-dire ce que l'exclusion masquait.
Les chiffres d'APRÈS sont produits par le premier run du job `coverage` sur
cette branche — la machine de développement était à une charge moyenne de ~350
(une vingtaine de sessions parallèles) et un run instrumenté complet y a été
tué par le système, donc ils ne sont pas recopiés ici de mémoire. Ce qui est
vérifié en local, lui, l'est pour de bon : les 7 suites de
`tests/neo4j_store_tests.rs` passent contre un Neo4j réel (conteneur jetable sur
le port 37687, jamais l'instance de développement sur 7687), en 1,4 s, et
chacune appelle les méthodes des 9 fichiers dés-exclus.
Elles le sont dans la **même PR** que les tests qui les couvrent
(`tests/neo4j_store_tests.rs`, et `tests/workspace_tests.rs` désormais inclus
dans la commande de couverture de la CI).

## Pourquoi `src/neo4j/*` n'était pas intestable

La justification inscrite dans `codecov.yml` était « all methods require a
running Neo4j instance ». C'est vrai et ce n'est pas un motif d'exclusion : la
CI provisionne **déjà** un service Neo4j dans les jobs `integration-tests`,
`api-tests` et `coverage` (`.github/workflows/ci.yml`). Les fichiers n'étaient
pas intestables, ils étaient **non testés** — mesuré le 2026-10-01, Neo4j allumé :
`commit`, `constraint`, `decision`, `milestone`, `release`, `user`, `workspace`,
`feature_graph`, `plan_run`, `event_trigger` et `chat` étaient à **0 %**.

Ce qui a changé dans cette PR :

1. Le job `coverage` collecte en **plusieurs passes fusionnées**
   (`cargo llvm-cov --no-report` ×3 puis `cargo llvm-cov report --lcov`) :
   tests unitaires du workspace entier, tests de parser, puis les suites qui
   parlent au vrai Neo4j. Une seule invocation ne pouvait pas faire les deux.
2. `tests/workspace_tests.rs` (16 tests, déjà écrits, jamais mesurés) et le
   nouveau `tests/neo4j_store_tests.rs` entrent dans cette passe.
3. Un garde-fou, `scripts/assert_lcov_covers.py`, échoue si le lcov fusionné ne
   contient aucune ligne couverte pour les fichiers `src/neo4j` listés. Sans
   lui, une suite qui se saute elle-même (Neo4j injoignable → `store()` renvoie
   `None`) téléverserait un rapport à 0 % qui ressemblerait à « ce code n'est
   pas testable ». Les suites elles-mêmes paniquent désormais quand `CI` est
   défini et que Neo4j ne répond pas.

## Exclusions restantes

Légende de la classe : **J** = justifiée durablement (amorçage, non compilé) ·
**C** = contournable, dette de test assumée avec un propriétaire · **R** = à
retirer, le travail est identifié.

### Hors `src/neo4j`

| Exclusion | Lignes | % mesuré | Classe | Raison | Propriétaire / suite |
|---|---|---|---|---|---|
| `desktop/**` | non mesuré | n/a | J | Crate Tauri séparé, non compilé par le `cargo llvm-cov` du crate racine. À mesurer dans son propre job si on le veut. | tâche 2.G |
| `src/main.rs` | 184 | 0 % | J | Point d'entrée, amorçage pur. | — |
| `src/bin/**` (`mcp_server.rs`) | 34 | 0 % | J | Point d'entrée, amorçage pur. | — |
| `src/lib.rs` | 1 561 | 57,8 % | C | Mélange de la config (testée) et de `start_server` (bind Axum). Contournable : bind sur le port 0, ou extraire l'amorçage hors de `lib.rs`. | tâche 4.1 |
| `src/events/nats.rs` | 396 | 41,2 % | C | Wrappers pub/sub. Le motif inscrit dans `codecov.yml` (« fait baisser le patch gate ») est un motif de confort, pas une impossibilité : testable avec un faux ou le service NATS en CI. | tâche 4.1 |
| `src/chat/**` (35 fichiers) | 28 377 | 79,7 % | R | Déjà couvert à ~80 % par des mocks ; l'exclusion masque 28 k lignes pour rien. | tâche 3.2 |
| `src/api/chat_handlers.rs` | 1 384 | 59,5 % | R | Mesuré sans aucun service. | tâche 3.2 |
| `src/api/ws_chat_handler.rs` | 740 | 16,1 % | C | Pipeline WebSocket : testable avec un client WS en mémoire et un faux `ChatManager`. Vraie dette de test. | tâche 3.2 |
| `src/api/routes.rs` | 1 577 | 99,9 % | R | La justification inscrite (« pas testable ») est fausse : 99,9 % couvert. L'exclusion ne sert à rien. | tâche 4.1 |

### `src/neo4j` — 13 restantes

Toutes de classe **R** : chacune a une suite d'intégration à écrire, pas un
obstacle technique. L'ordre suit la baseline 0.3 (les moins couvertes d'abord).

| Exclusion | Lignes src | % mesuré (Neo4j allumé) | Pourquoi encore exclue | Propriétaire / suite |
|---|---|---|---|---|
| `src/neo4j/chat.rs` | 1 780 | 0 % | Sessions de chat : les écrire demande le faux CLI de la tâche 3.2 pour produire des sessions réalistes. | tâche 3.2 |
| `src/neo4j/feature_graph.rs` | 1 538 | 0 % | Gros domaine (entités, communautés) ; une suite à part entière. | tâche 2.B |
| `src/neo4j/event_trigger.rs` | 318 | 0 % | Déclencheurs d'événements ; à écrire avec les triggers builtin de la tâche 2.C. | tâche 2.C |
| `src/neo4j/plan_run.rs` | 313 | 0 % | État du runner ; `RunnerState` est volumineux à construire — à écrire avec le lot runner. | tâche 2.C |
| `src/neo4j/analytics.rs` | 3 175 | partiel | GDS : demande les projections de graphe (`gds.graph.project`), donc le plugin GDS en CI, absent aujourd'hui (seul APOC est chargé). | tâche 2.B |
| `src/neo4j/code.rs` | 4 341 | partiel | Couche code-intelligence ; couverte en partie par `parser_tests`, le reste avec le lot 2.B. | tâche 2.B |
| `src/neo4j/impl_graph_store.rs` | 4 337 | partiel | Délégations du trait. Monte mécaniquement à mesure que les suites par domaine arrivent. | tâches 2.B / 3.x |
| `src/neo4j/note.rs` | 3 641 | partiel | Notes et énergie ; recoupe la branche active `status-energy-consistency`. | tâche 2.D |
| `src/neo4j/persona.rs` | 3 272 | partiel | Personas ; lot connaissance. | tâche 2.D |
| `src/neo4j/plan.rs` | 1 427 | partiel | Couvert en partie par `integration_tests` ; à compléter. | tâche 2.C |
| `src/neo4j/project.rs` | 620 | partiel | Couvert en partie ; à compléter. | tâche 2.B |
| `src/neo4j/step.rs` | 271 | partiel | Couvert en partie ; à compléter. | tâche 2.C |
| `src/neo4j/task.rs` | 920 | partiel | Couvert en partie par `integration_tests` ; à compléter. | tâche 2.C |

Note : `src/neo4j/mock.rs` (15 889 lignes) n'est **pas** exclu et n'a pas à
l'être : il est compilé sous `#[cfg(test)]`, donc absent du binaire de
production et du rapport.

## Comment retirer une exclusion

1. Écrire les tests dans la **même PR** que le retrait. Un test qui ne casse pas
   sans le correctif, ou qui n'asserte rien, ne compte pas.
2. Mesurer le fichier avant/après (commandes dans [COVERAGE.md](COVERAGE.md)) et
   mettre les deux chiffres dans la description de la PR.
3. Retirer la ligne de `codecov.yml` et la ligne correspondante de ce fichier,
   puis mettre à jour le compteur en tête.
4. Si le fichier doit rester exclu, dire **pourquoi** et **qui** s'en occupe. Une
   exclusion sans propriétaire est un trou dans le chiffre, pas une décision.
