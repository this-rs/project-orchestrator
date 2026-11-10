# Base et modules : PO comme application de référence composée sur this-rs

> Mission d'architecture, lecture seule. Question : « une base programme et des modules construits par-dessus ; PO deviendrait une démonstration de this-rs et obrain. »
> Convention : `chemin:ligne` = ce qui **existe déjà** ; « À ÉCRIRE » = ce qui n'existe pas. La distinction est le seul garde-fou entre démonstration et chantier.

## 0. La thèse en trois phrases

1. this-rs fournit **déjà** le mécanisme complet d'un module — descripteur déclaratif (entités, champs, vues, actions, opérations) + hôte qui stocke et dessine — et `atelier` est **déjà** une démonstration composée, y compris son propre module « chat » déclaré en JSON (`atelier/src/lib.rs:101`).
2. PO peut devenir une application composée **pour sa surface CRUD + affichage** (entités, tableaux, formulaires, dashboards : ~60 % des domaines), mais **pas** pour le chat/agent, le runner, ni l'analyse de code — ceux-là sont du *socle*, pas des modules.
3. Le coût dominant n'est pas « écrire des modules », c'est **inverser l'accès au stockage** de PO (aujourd'hui neo4j/meilisearch en direct) vers le contrat hôte `Request`/`Effect` de this-rs. C'est un chantier de fond, pas une démonstration.

## 1. Ce que this-rs sait déjà (le mécanisme)

Un module this-rs est défini par le trait `Module` (`this-module-sdk/src/module.rs:7`) :
- `manifest()` — déclare les **entités** (`this-descriptor/src/module.rs:23`), champs, liens, erreurs, vues, actions, opérations ;
- `handle(Request) -> Response` — CRUD + `Call`, avec effets `put`/`delete` (`this-module-sdk/examples/taches/src/lib.rs` est le patron complet).

Le contrat de **stockage est inversé** : le module ne touche jamais la base. Il reçoit des lignes (`Request.rows`) et rend des effets (`Effect::put/delete`) ; l'hôte fait l'écriture. C'est **exactement** ce qui réalise le critère « un module ne parle pas à la base d'un autre » : le module ne parle à *aucune* base.

Le module est exécuté en **WASM borné** (`this-module-harness/src/lib.rs` : compile → stocke → relit → exécute, fuel et mémoire bornés), ou via un langage total par construction (`this-module-lang/src/lib.rs` : pas de boucle non bornée, pas d'état survivant, pas d'I/O implicite). La sûreté vient de la grammaire **et** des bornes d'exécution, jamais du module lui-même.

`atelier` compose déjà tout cela (`atelier/Cargo.toml` : `this-agent`, `this-harness`, `this-llm-providers`, `this-descriptor`, `this-ui`, `this-module-harness`, `this-module-lang`, `this-vfs`, `this-documents`) et dessine l'écran d'un module à partir de son descripteur (`atelier/src/ecran.rs:11` : « `this-ui` dessine et ne fait rien ; chaque clic remonte une Intent »).

## 2. Les modules proposés

Découpage par domaine PO, aligné sur les domaines déjà identifiés. Tailles mesurées (`find src/<dom> -name '*.rs' | xargs wc -l`) sur `backend` (total **655 154** lignes).

| Module | Entités déclarées | Actions MCP qui lui appartiennent | Dépendances | Taille PO concernée |
|---|---|---|---|---|
| **plan** | plan, task, step, decision, constraint, milestone, release, environment, day_plan | `plan.*`, `task.*`, `step.*`, `decision.*`, `constraint.*`, `release.*`, `milestone.*`, `environment.*`, `day_plan.*` | note (les décisions deviennent des notes ? non : elles restent une entité du module plan) | `src/plan` 3 065 + part d'`orchestrator` |
| **knowledge (note)** | note, rfc | `note.*` | — | `src/notes` 14 732 |
| **graph** | workspace, component, resource, feature_graph | `workspace.*`, `component.*`, `resource.*`, `feature_graph.*`, `project.get_graph` | plan (le graphe lit plans/tâches) | `src/graph` 21 098 (hors neo4j) |
| **cognitif** | skill, protocol, persona, episode, analysis_profile, protocol_run | `skill.*`, `protocol.*`, `persona.*`, `episode.*`, `analysis_profile.*` | graph, note | `src/skills` 26 841 + `src/protocol` 14 327 + `src/neurons` 6 592 + `src/profile` 1 229 + `src/episodes` 6 003 + `src/reasoning` 2 282 |
| **chat** | conversation, message | `chat.*` (send_message, list_messages, get_session…) | — | `src/chat` 205 577 (voir §3 : quasi tout est socle) |
| **activity** | event, episode de session | `chat.list_sessions` (lecture), `activity` | chat, plan | `src/events` 9 792 + `src/analytics` 2 600 |
| **sharing** | consentement de partage | `sharing.*` | note, graph | `src/sharing` 2 063 |
| **identity/vault** | credential_ref, provider | `provider.*`, `vault.*` | — | `src/auth` 6 765 + `src/vault` 2 767 + `src/identity` 945 |

**Ce qui ne rentre dans AUCUN module** (c'est du socle, voir §3) : `src/chat` 205 577, `src/runner` 33 548, `src/orchestrator` 20 440, `src/mcp` 19 115 + `src/mcp_federation` 6 638, `src/neo4j` 87 088, `src/parser` 12 196, `src/pipeline` 8 434, `src/meilisearch` 2 299, `src/embeddings` 1 010, `src/events` (le bus), `src/transport`, `src/reception`, `src/ingress`, `src/lifecycle`, `src/boot.rs`, `src/storage.rs`.

Le « 427 actions MCP » du brief est un instantané : `grep` sur `src/mcp/` relève **≈490 noms d'action distincts** (le compteur exact dépend du filtre). Le point n'est pas le chiffre, c'est que ces ~490 actions sont **un seul point de répartition** (`src/mcp` 19 115 lignes), pas 32 crates. Pour les modulariser, il faut réattribuer chaque `(outil, action)` au module qui possède l'entité — la table ci-dessus en est le squelette — et laisser au socle les ~10 outils transverses (chat, code, admin, lifecycle_hook).

## 3. Ce que le socle doit fournir (et ce que PO refuse d'être un socle)

**Le socle this-rs fournit déjà** : stockage hôte (`Request.rows` / `Effect`), catalogue de descripteurs (`DescriptorSet`, `this-descriptor/src/module.rs:171`), génération d'UI (`this-ui`), exécution WASM bornée, i18n, permissions par opération (`EntityDesc.permissions`). `atelier` ajoute le harnais d'agent (`this-harness`), les providers (`this-llm-providers`), le VFS, le consentement d'appareil (`this-platform`).

**Ce que PO devrait construire comme socle (À ÉCRIRE)** :
1. **Le contrat `Request`/`Effect` sur les vrais stores de PO** (neo4j, meilisearch, embeddings). C'est le chantier de fond : aujourd'hui chaque domaine fait ses requêtes en direct ; un module ne recevrait jamais ses `rows`.
2. **Le bus d'événements et la réception** (`src/events`, `src/reception`, `src/transport`) : le socle, pas un module.
3. **Le harnais d'agent et le runner** (`src/chat`/`src/runner`) : la boucle Claude Code/Codex/ACP, le streaming, l'interrupt, le contrôle de permissions — c'est **impératif, temps réel, avec sous-processus** ; rien dans le vocabulaire déclaratif de this-rs ne l'exprime, et `this-harness` le fait déjà hors de tout module.
4. **L'analyse de code et les embeddings** (`src/parser`, `src/pipeline`, `src/embeddings`, `code.*`, `meilisearch`) : du socle de connaissance, appelé par les modules (le module `graph` lit des résultats, il ne recalcule pas les communautés).

**Le constat qui fâche** : PO « refuse d'être un socle » exactement là où il est le plus gros — `chat` (31 % du code), `runner`, `orchestrator`, `mcp`, `neo4j`. Un socle propre = sortir ces cinq blocs du périmètre « module » et les fournir comme services que les modules appellent. Le module `chat` de `atelier` (`atelier/src/lib.rs:101`) montre ce qui EST modulaire dans le chat : la **déclaration** des entités conversation/message et l'**affichage** du fil (`Brick::Chat`, `Brick::Message`). Ce qui ne l'est pas : le moteur d'agent qui produit les messages.

## 4. L'UI, tranchée avec des faits

**OUI — le descripteur porte assez de métadonnées pour engendrer le rendu d'un champ et d'un tableau.** La preuve est dans `this-ui` :
- `Kind::of(FieldType)` mappe chaque type du descripteur vers une nature d'interface (`this-ui/src/model.rs:33`) : `String/Text→Text`, `Bool→Bool`, `I32..U64→Int`, `F32/F64/Decimal→Decimal`, `Date→Date`, `DateTime→DateTime`, `Uuid→Uuid`, `Enum→Enum(variants)`, `Ref→Ref(entity)`, `List→List`, `Custom→Other`.
- `Kind::input_type()` donne l'`<input type=…>` (`this-ui/src/model.rs:63`) : `number`, `date`, `datetime-local`, sinon `text`.
- `Field::of(FieldDesc)` porte `optional`, `read_only`, `default` (`this-ui/src/model.rs:102`) : un formulaire peut donc dériver « requis ou non », « éditable ou non ».
- Le tableau générique existe : `EntityTable` (`this-ui/src/browser.rs:95`) avec `Column`/`Row`/`Cell` (`:72`, `:79`, `:23`).
- Le vocabulaire de vues est fermé et complet : `Brick` (`this-descriptor/src/views.rs:214`) sait `List` (tri, filtre, pagination, clic), `Card`, `Form` (`FormField`), `Metrics`, `Chat`, `Message`, `Composer`, `Settings`, `Secret`… et `Input` (`:615`) choisit Text/Multiline/Number/File/Reference/Toggle.

**NON — le descripteur n'engendre pas l'écran lui-même ; il ne sait pas composer, et ne sait pas exécuter la logique.**
- **Pas de « derive views ».** `this-descriptor` dérive TypeScript, GraphQL, Proto, JSON-schema (`crates/this-descriptor/src/derive/`) — jamais une vue. Un module **doit** déclarer ses `views` (`ViewDesc`/`Node`/`Brick`). L'entité porte le *rendu d'un champ* ; la *composition* (quels champs dans quel formulaire, quelle liste, quels onglets) est un choix déclaré, pas inféré.
- `atelier` confirme : `entites(descriptor)` (`atelier/src/ecran.rs:85`) ne fait que résoudre la **liste des entités à lire** ; les écrans viennent des `views` déclarées — le chat d'atelier déclare explicitement `views: [{surface: sidebar, brick: list…}, {surface: main, brick: stack[metrics, form, list]}]` (`atelier/src/lib.rs:101`).
- **Aucune brique n'exprime la logique métier ni les états** : pas de transition de FSM (protocol), pas de streaming de run, pas d'étape d'outil temps réel (hors `ToolStep` qui ne fait que lire un champ). Un module `protocol` qui exécute un run est une **opération** (`OperationDesc`, `this-descriptor/src/module.rs:73`) au corps libre, rendue par `handle`, pas par une vue.

**Réponse synthétique** : oui pour le **rendu** (un champ → un contrôle, une entité → un tableau, validations et i18n comprises) ; non pour la **composition automatique des écrans** (elle est déclarée, volontairement, vocabulaire fermé) et pour la **logique** (elle est dans `handle`, en WASM borné, jamais dans le descripteur).

## 5. Le coût honnête

**Ce que la direction fait gagner**
- **Démonstration** : chaque module = un descripteur testable isolément (`this-module-sdk/examples/taches` a ses tests unitaires sur `manifest` + `handle`, sans base). PO devient la preuve que le socle porte des modules variés.
- **Réutilisabilité** : un module se transporte (`skill`/`persona` export déjà des paquets portables ; le format `ModuleDescriptor` est sérialisable JSON/YAML).
- **Tests par module** : le contrat `handle` pur (lignes en entrée, effets en sortie) se teste sans neo4j — aujourd'hui les tests de PO touchent le graphe réel.
- **Bascule à chaud** : `atelier` recharge un module sans recompiler l'app.

**Ce qu'elle coûte (l'honnêteté)**
- **L'inversion du stockage** : réécrire l'accès neo4j/meilisearch de ~10 domaines derrière `Request`/`Effect`. C'est le gros du travail, invisible à l'écran.
- **Le front ne se réécrit pas d'un coup, mais il se dédouble** : PO a 397 942 lignes de front React (`frontend/src`). Le front `this-ui`/`atelier` est Leptos/wasm et ne rend que le vocabulaire fermé. Migrer les écrans = réimplémenter les vues riches (graphe three.js, kanban, chat temps réel) hors vocabulaire. Le graphe et le chat ne passeront **pas** par `Brick`.
- **Le chat ne se réécrit pas comme module** : ses 205 577 lignes restent du socle. Le module `chat` ne peut être que la *persistance et l'affichage* de la conversation, pas le moteur.
- **Les 490 actions MCP ne deviennent pas des opérations propres** : celles qui ne sont pas du CRUD (analyse de code, runs de protocole, requêtes graphe) restent des services du socle exposés à l'agent, avec un corps d'opération « libre » qui ne bénéficie plus du vocabulaire fermé.

**Verdict** : la direction **tient** pour la couche CRUD + affichage (entités, tableaux, formulaires, dashboards) — c'est exactement ce que this-rs + atelier démontrent déjà. Elle **ne tient pas** pour le chat/agent, le runner, l'analyse de code et le graphe de visualisation : ces blocs sont du socle, et le dire franchement évite de se retrouver avec un « module chat » de 205 k lignes qui n'en est pas un. Le risque de la fausse promesse (« tout est module ») est pire que le chantier : un socle honnête + 8 modules CRUD est une démonstration ; 14 pseudo-modules dont le chat est une preuve négative.

## 6. Le plus petit premier module

**`plan`** (entités : `plan`, `task`, `step`, `decision`, `constraint`). Pourquoi lui :
- c'est le **plus petit** domaine de données de PO (`src/plan` 3 065 lignes, l'un des plus purs) et il est **déjà CRUD** : ses actions MCP sont des `create/get/update/list/delete/link` sans logique temps réel ;
- il a **une vraie vue** : liste des plans, fiche de tâche, formulaire de tâche — tout est exprimable avec `Brick::List`/`Card`/`Form`/`Metrics` ;
- sa dépendance est minimale (aucune entité d'un autre module n'est requise pour un premier écran).

**Ce qu'il faut construire pour le prouver de bout en bout (de la déclaration à l'écran)** :
1. **Le manifeste** : `ModuleDescriptor` JSON/YAML pour `plan` (entités + champs typés + une vue `main` = `stack[Metrics(plan), List(plan), Form(task)]`). C'est déclaratif, ~1 jour.
2. **Le `handle`** : CRUD de `plan`/`task`/`step`/`decision`/`constraint` en Rust pur (lignes en entrée, effets en sortie), compilé en WASM via `this-module-harness`. Les règles métier (transition d'état d'une tâche, dépendances) deviennent des `OperationDesc` + `Call`.
3. **L'adaptateur de stockage (À ÉCRIRE, le vrai coût)** : un pont qui sert `Request.rows` depuis neo4j et applique `Effect::put/delete` vers neo4j, pour ces cinq entités. C'est la **preuve** que l'inversion de stockage est faisable — ou qu'elle ne l'est pas.
4. **Le front** : soit réutiliser `this-ui`/`atelier` (Leptos/wasm) pour un écran `plan` autonome, soit — plus proche de PO — un petit moteur qui traduit `ViewDesc` en composants React pour une seule vue. Ce second chemin est un **faux ami** (réécrire `this-ui` en React) ; le premier est la vraie démonstration.

**Critère de réussite** : modifier la déclaration d'un champ `task` (ajouter `priority: i32`, `sortable`) fait apparaître la colonne triable dans l'écran, **sans toucher une ligne de front ni de stockage**. Si ce critère passe, la direction est prouvée ; s'il exige de retoucher le pont neo4j ou le composant, c'est que l'inversion de stockage ou la génération d'UI ne sont pas encore au niveau — et c'est exactement ce qu'un premier module doit révéler avant d'en écrire dix.

## 7. En une phrase

Faire de PO une application composée sur this-rs est **la bonne démonstration pour la couche CRUD + affichage** (le socle et le mécanisme de module existent, atelier le prouve) ; le piège serait d'y fourrer le chat, le runner et l'analyse de code, qui sont et resteront du **socle** — la valeur de la démonstration est dans la franchise de cette frontière, pas dans le nombre de modules.
