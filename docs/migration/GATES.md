# Portes de qualité de la migration Neo4j → obrain (+ this-rs)

Ce document est le **contrat** des contrôles qui empêchent la migration de régresser
et qui permettent de la contrôler de bout en bout. Pour chaque porte : sa commande,
ce qu'elle prouve, ce qu'elle ne prouve PAS, et son **état réel**. « Non fait » est un
statut honnête : une porte absente n'est pas une porte verte.

Directive qui gouverne le reste (20/11/2026) : **on ne branche rien en production tant
que la migration vers obrain et this-rs n'est pas faite.** Le backend par défaut reste
`default` (Neo4j) ; la garde de bascule (#1248) est du code inactif.

Référence de la cible : `CONSOLIDATION.md` (règles, barre de performance, porte de bascule).

## ⚠ Ce qui rend une porte bloquante

Une étape CI rouge n'empêche une fusion que si le check est **requis** sur la branche
cible. Mesuré le 20/11/2026 par l'API GitHub : `consolidation/obrain-default` n'était
protégée par **rien** (`Branch not protected`, HTTP 404) — les portes ci-dessous
signalaient sans bloquer.

**C'est réglé depuis le 21/11/2026** (API, relu et non recopié) :

- checks requis : `Lint & Format`, `Unit Tests`, `Integration Tests`, `Obrain backend` ;
- `strict: false` — la branche n'a pas à être à jour, mais les checks doivent être verts
  **sur le SHA de tête de la PR** ;
- `enforce_admins: true` — le blocage s'applique aussi à l'orchestrateur, qui est la seule
  identité qui fusionne : sans ce réglage, la protection ne gênerait personne ;
- pull request obligatoire, mais **zéro approbation exigée** (`required_approving_review_count: 0`, pas de code owners) : la PR est le passage obligé, pas une revue ; force-push et suppression de branche interdits.

Effet vérifié le jour même : une PR dont un de ces quatre checks n'a pas conclu passe en
`mergeStateStatus: BLOCKED`, et une PR verte sur son SHA de tête passe `CLEAN`.

Ce que ça **ne** couvre pas, et qu'il faut savoir :
- un job **instable** (le test `kill9` de `Obrain backend`) bloque une PR saine : on
  relance le job, **jamais** on ne lève la protection ;
- `strict: false` veut dire que la combinaison « tête de PR + nouvelle base » n'est pas
  testée par GitHub : c'est à l'orchestrateur de rebaser et de relancer avant de fusionner ;
- un changement qui réécrit `xtask/` ou le cliquet reste couvert par la relecture, pas
  par la protection : une PR peut modifier la porte elle-même et rester verte.

## Vue d'ensemble

| # | Porte | Ce qu'elle prouve | Où elle tourne | État réel (20/11/2026) |
|---|---|---|---|---|
| G1 | containment | la branche contient `origin/main` | CI, Lint & Format | vert sur la consolidation après la descente #1253 |
| G2 | cliquet des traits portés | le cliquet est exactement l'ensemble des traits portés **selon le compilateur**, et il ne rétrécit pas | CI : Obrain backend (test du cliquet) + Lint & Format (`migration-gate --strict --base`) | **22 / 22** portés ; `AnalyticsStore` en partie — 79 des 145 méthodes servent (25 A1 #1250, 23 A2 #1258, 15 mémoire #1260, 15 agents et vues #1261, plus `run_data_migrations`), **66 refusées par nom** ; `SearchStore` porté (#1251, barre de performance non tenue, G8) |
| G3 | défaut neo4j | détecteur sur la source : le défaut reste Neo4j (définition, fabrique, câblage, déploiement) | CI, Lint & Format (même commande) | **conforme** ; détecteur, pas preuve de comportement (angles morts listés) |
| G4 | conformité des stores | les contrôles par domaine répondent pareil sur Neo4j, le mock et obrain | CI : Integration Tests (Neo4j réel) et Obrain backend | jambe **Neo4j : tourne en CI**. Jambe **obrain : élargie dans cette PR** de `obrain_project::` (10 tests) aux 18 modules `obrain_` (86 tests). **Trou** : les 28 contrôles transverses (`all_checks!`) ne tournent que sur Neo4j et le mock |
| G5 | refus nommés | une méthode non portée refuse par son nom (`NonPorte`), jamais un résultat vide | build + CI, Obrain backend | le générateur refuse la compilation si un impl manuel oublie une méthode ; les refus sont vérifiés par `not_ported`. **Trou** : les ~114 tests unitaires de `src/obrain/` ne tournent dans aucun job (pas de `--lib --features obrain-backend`) |
| G6 | parité de lecture sur copie | les mêmes requêtes REST rendent les mêmes réponses sur obrain et sur une copie Neo4j | **manuel** | **à rejouer** sur la consolidation (dernière mesure : 16/11, branche antérieure) |
| G7 | oracle de copie | comptes par type, empreinte par label, zéro quarantaine après migration, datetimes comptées | CI : Obrain backend (tests `migrator`) + **non automatisé** pour la copie réelle | **en cours** (PR #1256) : migrateur réel dans le dépôt, empreinte et comptes vérifiés, quarantaine PO vérifiée à l'ouverture de la copie ; datetimes tronquées COMPTÉES et enregistrées dans le marqueur mais **pas sans perte** (décision en attente) |
| G8 | performance | barre de `CONSOLIDATION.md` : rien ne croît avec N sur un chemin de requête | **manuel** | **non tenue** sur SearchStore (#1251) : ouverture, filtre par projet, suppression |
| G9 | double-run | un même trafic sur Neo4j et obrain, divergences classées | — | **non conçu** |
| G10 | retour arrière | rollback = une variable de configuration, la source jamais modifiée | tests de `storage::migrate` (fusionné par #1248) | **sélection de la source** implémentée et testée (migrateur factice). Les écritures faites sur obrain après la bascule ne sont ni recopiées ni réconciliées : **elles sont perdues** pour la source. Cette perte est annoncée (message + marqueur) mais non testée sur données réelles. Pas de test de bout en bout avec le vrai migrateur. **Porte rouge** tant qu'une stratégie explicite pour les écritures post-bascule n'est pas choisie |
| G11 | stabilité des tests | aucun test de la migration ne passe ou n'échoue au hasard | — | **ouvert** : `kill9_during_creations…` instable (tâche `TEST-KILL9-INSTABLE`) |

## Détail des portes automatisées

### G1 — containment
`cargo xtask containment`. Une PR vers la consolidation est rouge tant que la branche
ne contient pas `origin/main` ; cela se règle par une descente de `main` dans la
consolidation, jamais l'inverse.

### G2 — cliquet des traits portés
« Porté » est la notion **du compilateur** : le générateur `build/refus_obrain.rs` lit
`src/obrain/` avec `syn` et ne génère pas de refus pour un trait implémenté à la main ; il
publie la liste dans `project_orchestrator::obrain::methodes_portees()`. Une recherche de
texte serait en désaccord avec lui (module désactivé par `#[cfg]`, macro, chaîne) : la porte
n'en fait donc aucune.

Deux contrôles complémentaires :
1. Le test `the_migration_ratchet_is_exactly_the_set_of_ported_traits`
   (`tests/obrain_backend.rs`, job Obrain backend) exige que `xtask/migration-portees.txt`
   soit **exactement** l'ensemble des traits portés :
   - un trait listé qui n'est plus porté échoue (`RÉGRESSION`) ;
   - un trait porté absent du cliquet échoue (`NON CONSIGNÉ`) : un nouveau port est protégé
     dès la PR qui l'introduit.
2. `cargo xtask migration-gate --strict --base <rev>` (job Lint & Format) refuse une ligne
   retirée du cliquet depuis la base (`CLIQUET RÉDUIT`). Sans ce second contrôle, retirer un
   port **et** sa ligne dans la même PR passerait le test (les deux ensembles rétrécissent
   ensemble). En CI, la base est la cible de la PR, ou le commit précédent sur un push.

Preuves rejouées (20/11/2026) :

| Scénario | Résultat |
|---|---|
| état réel | test vert ; porte : 20 traits, exit 0 |
| ligne `ProjectStore` retirée du cliquet | test : `NON CONSIGNÉ ["ProjectStore"]` ; porte avec base : `CLIQUET RÉDUIT`, exit 1 |
| `AnalyticsStore` listé sans être porté | test : `RÉGRESSION ["AnalyticsStore"]` |

**Limites connues** : un `impl` dont toutes les méthodes refusent à la main compte comme
porté (le générateur compte les méthodes écrites, voir `REFUS_ECRITS_A_LA_MAIN`) ; `GraphStore`
figure parmi les portés parce que son `impl` manuel est imposé (`capabilities` n'a pas de
défaut), ce qui rend le compte « 20 / 22 » un peu flatteur. La porte exécute le code de la PR. Une PR qui réécrit la porte elle-même
(changer le chemin du cliquet, par exemple) peut la neutraliser. Ce risque se règle par la
relecture des changements de `xtask/` et du cliquet, et par la protection de la branche.

### G3 — défaut neo4j
Même commande que G2 (second contrôle). C'est un **détecteur** sur la source, pas une preuve
de comportement. Refusé quand :
- `DEFAULT_BACKEND` n'est pas défini **exactement une fois** avec la valeur `"default"`
  (une seconde définition sous `#[cfg]`, ou cachée dans une chaîne, est refusée) ;
- le bras `BackendKind::Default` de la fabrique n'appelle plus `Self::default_backend(`
  (contrôle de texte : ce que fait `default_backend` n'est pas vérifié) ;
- `src/lib.rs` ne retombe plus, **dans le code**, sur `DEFAULT_BACKEND` quand rien n'est
  configuré (une copie dans une chaîne ne compte pas) ;
- un fichier de déploiement suivi par git fixe `STORAGE_BACKEND` ou `backend:` à obrain :
  `*.yml`, `*.yaml`, `*.env`, `*.service`, `*.sh`, `*.toml`, `*.json`, `*.tpl`, `*.j2`,
  `Dockerfile*`, `docker-compose*`, leurs variantes `.example`, y compris `.github/workflows/` et une liste
  d'environnement Kubernetes (`- name: STORAGE_BACKEND` / `value: obrain`). Sont exclus :
  `docs/`, `tests/`, `examples/`, `tools/`, `benches/`, `xtask/`, `src/`, `build/`.

Preuves rejouées sur une copie indépendante : bras `Default` recâblé sur obrain, câblage
remplacé avec une copie gardée dans une chaîne, service systemd, `config.yaml.example`,
liste d'environnement k8s : tous refusés, exit 1. L'arbre réel ne produit aucun faux positif.

**Angles morts** (non détectés, mesurés le 20/11/2026) :
- dans `src/lib.rs` : un repli sur obrain placé **avant** le défaut, ou une transformation
  **après** lui (`….replace("default", "obrain")`) ; une copie du câblage dans du code mort ;
- dans la fabrique : un `return Self::obrain_backend(…)` placé au début de `default_backend`
  (le bras appelle toujours `default_backend`, la porte reste verte) ;
- dans le déploiement : une valeur atteinte par indirection (`ARG B=obrain` puis
  `ENV STORAGE_BACKEND=$B`) ; une liste k8s où `value: obrain` précède
  `name: STORAGE_BACKEND` ; une clé `backend:` dont la valeur est sur la ligne suivante ;
  une extension hors de la liste ci-dessus.

Ce que la constante
sélectionne est couvert par le test `default_backend_is_selected_for_the_default_value`,
qui vérifie la correspondance valeur → variant, et non que le variant construit Neo4j.

### G4 — conformité des stores
`tests/store_conformance.rs` exécute les contrôles par domaine (`*_checks!`) sur Neo4j, le
mock et obrain ; les 28 contrôles transverses de `all_checks!` ne tournent que sur Neo4j et le
mock (aucune jambe obrain), et `note_checks` n'a pas de jambe mock. Les
divergences connues sont listées dans `EXPECTED_DIVERGENCES` ; le test échoue si une
divergence listée est corrigée (« now conforms »), ce qui force à retirer l'entrée.
- Jambe Neo4j : job **Integration Tests**, avec un service Neo4j et `HARNESS_NEO4J_REQUIRED=1`
  (un Neo4j absent est un échec, pas un saut).
- Jambe obrain : job **Obrain backend**. Le filtre CI ne retenait que `obrain_project::`
  (10 tests, 241 filtrés) ; il retient désormais les 18 modules `obrain_` (86 tests), et
  aucun test `mock_` ni `neo4j_`.
- En local sans Neo4j, la jambe Neo4j passe sans s'exécuter : seule la CI fait foi.

## Portes manuelles ou non faites

### G6 — parité de lecture (LECTURE-NATIVE)
Rejouer les routes REST sur obrain et sur une copie Neo4j, champ à champ, et classer chaque
écart (non porté / perte de migration / lecture du portage / bénin). La mesure du 16/11 ne
vaut plus : elle portait sur une branche où les dates étaient mal lues.

### G7 — oracle de copie
Copie jetable, comptes par type égaux, empreinte par label égale, zéro nœud en quarantaine,
dates relues à la microseconde. C'est la porte que demande #1246.

**État au 21/11/2026 (PR #1256, branche `obrain/migrateur-reel`)** : le migrateur réel
(`storage::migrator::Neo4jToObrain`, feature `obrain-backend`) est dans le dépôt.

Ce qui est vérifié :
- comptes par type source vs cible (refus si différents) ;
- empreinte par label (hash Mérkle sur les propriétés, refus si divergence) ;
- quarantaine PO : la copie est ouverte par `ObrainGraphStore::open` après l'empreinte
  — refusée si l'ouverture met un nœud en quarantaine (test : un nœud Plan sans statut
  passe les comptes et l'empreinte mais PO le met en quarantaine → refus nommé) ;
- dérive de la source : les comptes source sont relus APRÈS la copie et refusés s'ils
  ont bougé (détecte un écrivain externe pendant la copie).

Ce qui n'est **pas** sans perte :
- les datetimes avec des nanosecondes non multiples de 1000 sont tronquées à la
  microseconde (obrain ne stocke pas plus fin). Le compte des troncatures est enregistré
  dans le marqueur `DONE_MARKER` (champ `narrowings.sub_microsecond_datetimes`) et
  tracé dans les logs. Un prototype de compagnon `<clé>__nanos` démontre l'aller-retour
  exact en test, mais il n'est **pas branché** en production.
- **Décision en attente** : accepter la troncature ou la résoudre sans perte. Tant que
  cette décision n'est pas prise, la porte est **rouge**.

### G8 — performance
Barre de `CONSOLIDATION.md`. Mesurée le 20/11 à 100 k et 1 M documents sur SearchStore, avec
l'exemple `examples/obrain_mesures_recherche.rs` **de la branche de #1251** (absent de la
consolidation) : ouverture 0,6 s → 7 s, filtre par projet 0,4 ms → 8,9 ms, suppression par id
5 ms → 21 ms. Barre non tenue. Une PR qui touche un chemin chaud doit livrer ses mesures à
1 M et 5 M avant et après.

### G9 — double-run
La porte de justesse décisive avant la bascule. Non conçue.

### G10 — retour arrière
Fusionné par #1248 (`src/storage/migrate.rs`, `src/storage/mod.rs`).

**Sélection de la source** par variable de configuration : implémentée et testée avec un
migrateur factice. La source n'est jamais modifiée par PO (par construction du trait
`Migrator`).

**Écritures post-bascule** : les écritures faites sur obrain après la bascule ne sont ni
recopiées ni réconciliées vers Neo4j — elles sont **perdues** pour la source. Cette perte
est annoncée dans le message de rollback et par un marqueur (`.po-migration-rolled-back`),
refusée en rollback sans `from` si la cible contient des données, mais :
- **non détectée** par comparaison de contenu ;
- **non testée** sur des données réelles (aucun test ne crée d'écriture après la bascule
  pour vérifier qu'elle manque côté source) ;
- pas de test de bout en bout avec le vrai migrateur.

**Porte rouge** tant qu'une des conditions suivantes n'est pas remplie :
- une stratégie explicite pour les écritures post-bascule (décision écrite d'accepter leur
  perte, ou recopie vers Neo4j), avec un test qui écrit après la bascule puis fait le
  rollback ; OU
- un test de bout en bout du rollback avec le vrai migrateur.

### G11 — stabilité des tests
`kill9_during_creations…` échoue au hasard, de deux façons mesurées : le garde-fou de timing
(« no kill landed inside a creation over 12 rounds ») et un `unwrap()` sur une ligne de
journal tronquée par le SIGKILL. Le correctif du premier mode existe sur #1222 (manche
« suspendue ») ; le second n'est corrigé nulle part. Une porte instable ne prouve rien.

## Règles de contrôle

- Toute PR de la migration consigne ses portes dans un commentaire : critères, rouge-sans /
  vert-avec rejoué, relecture indépendante, contrat et doc, à jour de la base, CI verte.
- Le cliquet G2 ne rétrécit pas ; un retrait voulu est une décision écrite ici, et la porte
  le refusera quand même tant que la base contient la ligne.
- La bascule en production n'est pas une porte de plus : elle exige G4, G6, G7, G8 et G9
  vertes, et la levée de la directive du 20/11.
