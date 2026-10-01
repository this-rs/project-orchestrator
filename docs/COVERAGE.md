# Baseline de couverture honnête (backend)

Mesuré le 2026-10-01 sur `origin/main` @ 94c04bce. Deux chiffres sont toujours publiés :

- **gated** : ce que Codecov affiche, c'est-à-dire le lcov de la CI **privé** des fichiers de la section `ignore:` de `codecov.yml` (l'exclusion se fait à l'upload ; `cargo llvm-cov` lui-même mesure tout).
- **brut** : tout `src/`, sans aucune exclusion.

| | Lignes | Couvertes | % |
|---|---|---|---|
| **brut** (CI + Neo4j) | 235 845 | 172 880 | **73,3 %** |
| **gated** (CI + Neo4j) | 179 990 | 143 941 | **80,0 %** |
| brut sans Neo4j (unit + parser) | 235 845 | 170 198 | 72,2 % |
| gated sans Neo4j | 179 990 | 143 245 | 79,6 % |
| partie ignorée (64 fichiers) | 55 855 | 28 939 | 51,8 % |

Les exclusions masquent 23,7 % des lignes. Ne pas fixer un objectif de 100 % sur le gated. N'utilisez pas `tarpaulin-report.json` (mars 2026, périmé).

Périmètre à la date de la mesure : la CI exécutait `cargo llvm-cov --lib --test parser_tests --test integration_tests --test data_migrations`, soit le crate racine uniquement. Les crates `crates/*` (neural-routing-*, tree-sitter-*) et `desktop/` n'étaient pas mesurés, ni les tests `api_tests`, `workspace_tests`, `p2p_*`, `mcp_federation_integration`, etc.

**Depuis la tâche 3.1**, le job `coverage` collecte en plusieurs passes puis les **fusionne** (`cargo llvm-cov --no-report` ×3, puis `cargo llvm-cov report --lcov`) : tests unitaires du workspace entier (`--workspace --exclude project-orchestrator-desktop --lib`, ce qui inclut enfin les crates membres), `parser_tests`, puis les suites qui parlent au vrai Neo4j (`integration_tests`, `data_migrations`, `workspace_tests`, `neo4j_store_tests`). Les chiffres de ce document sont donc un plancher : ils datent d'avant cet élargissement.

Le registre des exclusions, avec le propriétaire de chacune et le compteur qui ne peut que baisser, est dans [COVERAGE_EXCLUSIONS.md](COVERAGE_EXCLUSIONS.md).

## Par module (run CI-équivalent, avec Neo4j)

| Module | Fichiers | Lignes | Couvertes | % brut | Lignes gated | % gated |
|---|---|---|---|---|---|---|
| `src/(root)` | 6 | 3312 | 2053 | 62.0 | 1567 | 73.5 |
| `src/analytics` | 7 | 1080 | 1007 | 93.2 | 1080 | 93.2 |
| `src/api` | 32 | 40308 | 27606 | 68.5 | 36607 | 68.5 |
| `src/architecture` | 6 | 1590 | 1545 | 97.2 | 1590 | 97.2 |
| `src/auth` | 7 | 1865 | 1320 | 70.8 | 1865 | 70.8 |
| `src/bin` | 1 | 34 | 0 | 0.0 | 0 | n/a |
| `src/chat` | 35 | 28377 | 22612 | 79.7 | 0 | n/a |
| `src/documents` | 6 | 1130 | 1066 | 94.3 | 1130 | 94.3 |
| `src/embeddings` | 3 | 385 | 283 | 73.5 | 385 | 73.5 |
| `src/episodes` | 7 | 2369 | 2289 | 96.6 | 2369 | 96.6 |
| `src/events` | 11 | 4325 | 3551 | 82.1 | 3929 | 86.2 |
| `src/feedback` | 5 | 1042 | 870 | 83.5 | 1042 | 83.5 |
| `src/graph` | 11 | 9953 | 9139 | 91.8 | 9953 | 91.8 |
| `src/heartbeat` | 11 | 699 | 638 | 91.3 | 699 | 91.3 |
| `src/identity` | 4 | 448 | 399 | 89.1 | 448 | 89.1 |
| `src/lifecycle` | 2 | 849 | 843 | 99.3 | 849 | 99.3 |
| `src/mcp` | 8 | 14445 | 11954 | 82.8 | 14445 | 82.8 |
| `src/mcp_federation` | 7 | 4088 | 3836 | 93.8 | 4088 | 93.8 |
| `src/meilisearch` | 4 | 1315 | 975 | 74.1 | 1315 | 74.1 |
| `src/neo4j` | 41 | 35389 | 10297 | 29.1 | 13787 | 54.8 |
| `src/neurons` | 4 | 1749 | 1600 | 91.5 | 1749 | 91.5 |
| `src/notes` | 6 | 4354 | 3520 | 80.8 | 4354 | 80.8 |
| `src/orchestrator` | 6 | 12961 | 10380 | 80.1 | 12961 | 80.1 |
| `src/parser` | 21 | 8524 | 6720 | 78.8 | 8524 | 78.8 |
| `src/pipeline` | 13 | 6082 | 5340 | 87.8 | 6082 | 87.8 |
| `src/plan` | 2 | 1925 | 1731 | 89.9 | 1925 | 89.9 |
| `src/profile` | 5 | 589 | 543 | 92.2 | 589 | 92.2 |
| `src/protocol` | 10 | 7447 | 6977 | 93.7 | 7447 | 93.7 |
| `src/reasoning` | 3 | 1257 | 906 | 72.1 | 1257 | 72.1 |
| `src/reception` | 7 | 943 | 933 | 98.9 | 943 | 98.9 |
| `src/reflex` | 5 | 622 | 550 | 88.4 | 622 | 88.4 |
| `src/resolver` | 4 | 854 | 787 | 92.2 | 854 | 92.2 |
| `src/runner` | 17 | 14952 | 11745 | 78.6 | 14952 | 78.6 |
| `src/sharing` | 4 | 600 | 583 | 97.2 | 600 | 97.2 |
| `src/skills` | 19 | 16079 | 15024 | 93.4 | 16079 | 93.4 |
| `src/transport` | 4 | 862 | 647 | 75.1 | 862 | 75.1 |
| `src/update` | 4 | 1117 | 872 | 78.1 | 1117 | 78.1 |
| `src/utils` | 3 | 350 | 339 | 96.9 | 350 | 96.9 |
| `src/vault` | 6 | 1575 | 1400 | 88.9 | 1575 | 88.9 |

(« % gated » = n/a : module entièrement ignoré.)

## Classement des exclusions de `codecov.yml`

Légende : **J** = justifiée (amorçage / non compilé), **C** = contournable avec un faux (le code est testable sans service vivant), **R** = à retirer (déjà mesuré ou testable en CI, qui provisionne Neo4j/Meilisearch).
Colonnes : lignes, % avec Neo4j (CI) / % sans Neo4j.

| Exclusion | Lignes | % CI / sans Neo4j | Classe | Raison |
|---|---|---|---|---|
| `desktop/**` | n/a | non mesuré | J | Crate Tauri séparé, non compilé par llvm-cov du crate racine. À mesurer dans son propre job si voulu. |
| `src/chat/**` (35 fichiers) | 28 377 | 79,7 % / 79,7 % | R | Déjà testé par mocks à ~80 % : l'ignorer masque 28 k lignes (12 % du total). Seuls `drain.rs` (49,5 %), `cli_auth.rs` (40,7 %), `stages/biomimicry.rs` (11 %) sont faibles. `manager.rs` fait à lui seul 8 906 lignes à 67,8 %. |
| `src/api/chat_handlers.rs` | 1 384 | 59,5 % / 59,5 % | R | Mesuré à 59,5 % sans service. |
| `src/api/ws_chat_handler.rs` | 740 | 16,1 % / 15,7 % | C | Pipeline WebSocket/stream : testable avec un client WS en mémoire et un faux ChatManager. Vraie dette de test. |
| `src/neo4j/*.rs` (20 fichiers listés, hors client/traits : analytics, chat, code, commit, constraint, decision, event_trigger, feature_graph, impl_graph_store, milestone, note, persona, plan, plan_run, project, release, step, task, user, workspace) | 21 146 | 11,5 % (2 426) / 3,8 % (798) | R | La CI a déjà un service Neo4j et ignore ces fichiers alors qu'il est branché. Avec Neo4j ils passent de 798 à 2 426 lignes couvertes ; la plupart restent à 0 % faute de tests d'intégration, pas par impossibilité (`commit`, `constraint`, `decision`, `milestone`, `release`, `user`, `workspace`, `feature_graph`, `plan_run`, `event_trigger`, `chat` : 0 %). Piste : ajouter `--test api_tests` et `workspace_tests` à la commande de couverture CI. |
| `src/neo4j/client.rs` | 437 | 69,8 % avec Neo4j / 0 % sans | R | Testé avec le Neo4j de la CI. |
| `src/neo4j/traits.rs` | 19 | 68,4 % | R | Mesuré, négligeable : l'exclusion est inutile. |
| `src/api/routes.rs` | 1 577 | 99,9 % / 99,9 % | R | Justification fausse (« pas testable ») : 99,9 % couvert. L'ignorer ne sert à rien. |
| `src/lib.rs` | 1 561 | 57,8 % / 54,6 % | C | Mélange config (testée) et `start_server` (bind Axum). Contournable : bind sur le port 0 dans un test, ou extraire le bootstrap hors de `lib.rs`. |
| `src/main.rs` | 184 | 0 % | J | Point d'entrée ; amorçage pur. |
| `src/bin/**` (`mcp_server.rs`) | 34 | 0 % | J | Point d'entrée ; amorçage pur. |
| `src/events/nats.rs` | 396 | 41,2 % / 41,2 % | C | Les wrappers pub/sub sont testables avec un faux ou un service NATS en CI (l'image nats est déjà utilisée en prod). La raison invoquée (« fait baisser le patch gate ») est une raison de confort, pas technique. |

Résumé : J = 3 (desktop, main.rs, bin), C = 3 (ws_chat_handler, lib.rs, nats.rs), R = tout le reste (chat/**, chat_handlers, neo4j/*, routes.rs), soit environ 90 % des lignes ignorées.

Détail par fichier : voir les commandes ci-dessous (`lcov-an.py files`).

## Constats lors de la mesure

- `resolver::suffix_index::tests::test_large_index_performance` échoue sous instrumentation llvm-cov (test de performance, 6 411 autres tests lib passent). Il a été exclu avec `--skip` pour obtenir un lcov. Un run sans `--skip` ne produit pas de rapport (cargo-llvm-cov s'arrête au premier échec). À traiter à part, non corrigé ici.
- Avec le Neo4j de test : 17 tests `integration_tests` + 1 `data_migrations` passent, 0 échec.
- Attention : `integration_tests`, `data_migrations` et `workspace_tests` se connectent par défaut à `bolt://localhost:7687`. Sur une machine de dev, ce port peut être un Neo4j réel : toujours définir `NEO4J_URI` explicitement vers une instance jetable.

- Une suite d'intégration qui ne trouve pas Neo4j **se saute elle-même** et sort en 0 : le rapport téléversé serait alors plein de zéros sur `src/neo4j`, ce qui se lit « ce code n'est pas testable » au lieu de « le service n'était pas là ». Deux garde-fous depuis la tâche 3.1 : `tests/neo4j_store_tests.rs` panique quand `CI` est défini et que Neo4j ne répond pas, et `scripts/assert_lcov_covers.py` échoue si le lcov fusionné ne contient aucune ligne couverte pour les fichiers `src/neo4j` dont l'exclusion a été retirée.

## Commandes exactes (reproductibles)

```bash
export PATH=$HOME/.cargo/bin:$PATH
git worktree add --detach /chemin/wt-cov-be origin/main && cd /chemin/wt-cov-be
export CARGO_TARGET_DIR=$PWD/../target-cov-be   # un target dir par worktree

# Services jetables sur des ports non standard (ne pas toucher à un Neo4j existant)
docker run -d --rm --name po-cov-neo4j -p 27687:7687 -e NEO4J_AUTH=neo4j/testpassword -e 'NEO4J_PLUGINS=["apoc"]' neo4j:5
docker run -d --rm --name po-cov-meili -p 27700:7700 -e MEILI_MASTER_KEY=test-master-key -e MEILI_NO_ANALYTICS=true getmeili/meilisearch:v1.34.2

# (a) comme la CI (avec Neo4j) -> lcov complet, non filtré
NEO4J_URI=bolt://localhost:27687 NEO4J_USER=neo4j NEO4J_PASSWORD=testpassword \
MEILISEARCH_URL=http://localhost:27700 MEILISEARCH_KEY=test-master-key \
cargo llvm-cov --no-fail-fast --lib --test parser_tests --test integration_tests --test data_migrations \
  --lcov --output-path lcov-ci.info -- --skip test_large_index_performance

# (b) sans Neo4j (port volontairement mort)
NEO4J_URI=bolt://127.0.0.1:1 cargo llvm-cov --no-fail-fast --lib --test parser_tests \
  --lcov --output-path lcov-noneo4j.info -- --skip test_large_index_performance

docker stop po-cov-neo4j po-cov-meili
```

Le lcov brut contient tous les fichiers ; `scripts/coverage_by_module.py` calcule brut et gated par module et par fichier :

```bash
# extraire la liste d'ignores de codecov.yml, puis agréger
python3 - <<'PY'
import re
y=open('codecov.yml').read().split('ignore:')[1]
open('ignores.txt','w').write('\n'.join(re.findall(r'^\s*-\s*"([^"]+)"',y,re.M))+'\n')
PY
python3 scripts/coverage_by_module.py lcov-ci.info "$PWD" ignores.txt          # par module
python3 scripts/coverage_by_module.py lcov-ci.info "$PWD" ignores.txt files    # par fichier (IGN = ignoré)
```

Durée : environ 10 min de tests lib sous instrumentation (+ compilation).

## Reproduire le lcov FUSIONNÉ de la CI (tâche 3.1)

Ce que fait le job `coverage` depuis la tâche 3.1 : plusieurs passes de collecte,
une seule fusion. `--no-report` accumule les `.profraw` ; `llvm-cov report` les
fusionne. Une invocation unique ne pouvait pas couvrir à la fois le workspace
entier et les suites par cible de test.

```bash
export PATH=$HOME/.cargo/bin:$PATH
export CARGO_TARGET_DIR=$PWD/../target-cov-be        # un target dir par worktree

# Services JETABLES sur des ports non standard : 7687 est souvent un vrai Neo4j.
docker run -d --rm --name po-cov-neo4j -p 37687:7687 \
  -e NEO4J_AUTH=neo4j/testpassword -e 'NEO4J_PLUGINS=["apoc"]' neo4j:5
docker run -d --rm --name po-cov-meili -p 37700:7700 \
  -e MEILI_MASTER_KEY=test-master-key -e MEILI_NO_ANALYTICS=true getmeili/meilisearch:v1.34

export NEO4J_URI=bolt://localhost:37687 NEO4J_USER=neo4j NEO4J_PASSWORD=testpassword
export MEILISEARCH_URL=http://localhost:37700 MEILISEARCH_KEY=test-master-key

cargo llvm-cov clean --workspace

# (1) unité, workspace entier (les crates membres ont leurs propres suites)
cargo llvm-cov --no-report --no-fail-fast \
  --workspace --exclude project-orchestrator-desktop --lib \
  -- --skip test_large_index_performance
# (2) parser
cargo llvm-cov --no-report --no-fail-fast --test parser_tests
# (3) les suites qui parlent au VRAI Neo4j
cargo llvm-cov --no-report --no-fail-fast \
  --test integration_tests --test data_migrations \
  --test workspace_tests --test neo4j_store_tests
# (4) fusion
cargo llvm-cov report --lcov --output-path lcov.info

# garde-fou : une suite sautée en silence ne doit pas passer pour du code intestable
python3 scripts/assert_lcov_covers.py lcov.info \
  src/neo4j/workspace.rs src/neo4j/milestone.rs src/neo4j/constraint.rs \
  src/neo4j/release.rs src/neo4j/commit.rs src/neo4j/decision.rs \
  src/neo4j/user.rs src/neo4j/traits.rs src/neo4j/client.rs

docker stop po-cov-neo4j po-cov-meili
```

`test_large_index_performance` (`src/resolver/suffix_index.rs`) est une assertion
de temps de parois qui ne tient pas sous l'instrumentation llvm-cov. Elle est
sautée **dans la passe de couverture uniquement** ; le job `Unit Tests` continue
de l'exécuter sans instrumentation.
