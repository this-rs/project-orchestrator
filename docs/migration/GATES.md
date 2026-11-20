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
cible. Mesuré le 20/11/2026 par l'API GitHub : `consolidation/obrain-default` **n'est
pas protégée** (aucune règle de branche, aucun ruleset visible). Sur cette branche, les
portes ci-dessous signalent mais **ne bloquent pas** : une PR rouge reste fusionnable.
Pour qu'elles bloquent, il faut rendre « Lint & Format », « Unit Tests », « Integration
Tests » et « Obrain backend » requis sur cette branche (réglage d'administration du
dépôt). Tant que ce n'est pas fait, le blocage repose sur la règle du portillon :
aucune fusion avec un check rouge.

## Vue d'ensemble

| # | Porte | Ce qu'elle prouve | Où elle tourne | État réel (20/11/2026) |
|---|---|---|---|---|
| G1 | containment | la branche contient `origin/main` | CI, Lint & Format | vert sur la consolidation après la descente #1253 |
| G2 | cliquet des traits portés | un trait de contrat porté ne perd pas son port ; le cliquet ne rétrécit pas | CI, Lint & Format (`migration-gate --strict --base`) | **20 / 22** portés ; restent `AnalyticsStore` (#1250), `SearchStore` (#1251) |
| G3 | défaut neo4j | le backend par défaut reste Neo4j, par définition, câblage et déploiement | CI, Lint & Format (même commande) + test unitaire | **conforme** |
| G4 | conformité des stores | chaque store répond pareil sur Neo4j, le mock et obrain | CI : Integration Tests (Neo4j réel) et Obrain backend | jambe **Neo4j : tourne en CI** (service Neo4j, `HARNESS_NEO4J_REQUIRED=1`). Jambe **obrain : élargie dans cette PR** de `obrain_project::` (10 tests) à tous les modules `obrain_` (86 tests, verts en local, ~10 s) |
| G5 | refus nommés | une méthode non portée refuse par son nom (`NonPorte`), jamais un résultat vide | CI, Obrain backend | couvert par les assertions `not_ported` ; G2 suit les traits, pas les méthodes |
| G6 | parité de lecture sur copie | les mêmes requêtes REST rendent les mêmes réponses sur obrain et sur une copie Neo4j | **manuel** | **à rejouer** sur la consolidation (dernière mesure : 16/11, branche antérieure) |
| G7 | oracle de copie | comptes par type, empreinte par label, zéro quarantaine après migration | **non automatisé** | **non fait** ; le migrateur réel est hors dépôt (attendu avant la bascule) |
| G8 | performance | barre de `CONSOLIDATION.md` : rien ne croît avec N sur un chemin de requête | **manuel** | **non tenue** sur SearchStore (#1251) : ouverture, filtre par projet, suppression |
| G9 | double-run | un même trafic sur Neo4j et obrain, divergences classées | — | **non conçu** |
| G10 | retour arrière | rollback = une variable de configuration, la source jamais modifiée | tests de `storage::migrate` (branche de #1248) | implémenté et testé sur #1248, **non fusionné** |
| G11 | stabilité des tests | aucun test de la migration ne passe ou n'échoue au hasard | — | **ouvert** : `kill9_during_creations…` instable (tâche `TEST-KILL9-INSTABLE`) |

## Détail des portes automatisées

### G1 — containment
`cargo xtask containment`. Une PR vers la consolidation est rouge tant que la branche
ne contient pas `origin/main` ; cela se règle par une descente de `main` dans la
consolidation, jamais l'inverse.

### G2 — cliquet des traits portés
Commande : `cargo xtask migration-gate --strict [--base <rev>]`.

- **Contrat** : `pub trait` de `src/neo4j/traits.rs` et `src/meilisearch/traits.rs` (22 traits).
- **Porté** : un `impl <Trait> for Obrain…` existe sous `src/obrain/`, **lu récursivement**.
  Les commentaires et les chaînes sont retirés avant la recherche : un `impl` laissé en
  commentaire ou dans une chaîne n'est pas un port. La mise en page est libre (`impl`
  sur plusieurs lignes, génériques, préfixes `crate::obrain::`, `super::`, `self::`).
  Seuls les traits du contrat comptent (`impl Debug for ObrainGraphStore` n'est pas un port).
- **Cliquet** `xtask/migration-portees.txt`. En `--strict`, refusé :
  - un trait listé qui n'est plus porté (`RÉGRESSION`) ;
  - un trait porté qui n'est pas listé (`NON CONSIGNÉ`) : un nouveau port est protégé dès sa PR ;
  - avec `--base`, une ligne retirée du cliquet depuis la base (`CLIQUET RÉDUIT`). En CI, la
    base est la cible de la PR (ou le commit précédent sur un push) : supprimer un port
    **et** sa ligne dans la même PR est refusé.

Preuves rejouées sur une copie indépendante du dépôt (20/11/2026) :

| Scénario | Résultat |
|---|---|
| état réel | 20 / 22, exit 0 |
| port retiré et ligne du cliquet retirée | `CLIQUET RÉDUIT ProjectStore`, exit 1 |
| port retiré, laissé en commentaire | `RÉGRESSION ProjectStore`, exit 1 |
| ligne retirée du cliquet sans `--base` | `NON CONSIGNÉ`, exit 1 |
| fichier déplacé en sous-dossier, `impl` sur deux lignes | aucun refus, exit 0 |

**Limites connues** (non détectées) : un `impl` dont toutes les méthodes refusent compte
comme porté (voir G5) ; un trait importé sous un alias (`use … as PS; impl PS for …`) n'est
pas reconnu, ce qui produit une régression signalée à tort, jamais une régression masquée.

### G3 — défaut neo4j
Même commande. Refusé quand :
- `DEFAULT_BACKEND` n'est pas défini **exactement une fois** avec la valeur `"default"`
  (une seconde définition sous `#[cfg]`, ou cachée dans une chaîne, est refusée) ;
- `src/lib.rs` ne retombe plus sur `DEFAULT_BACKEND` quand rien n'est configuré ;
- un fichier de déploiement suivi par git (`*.yml`, `*.yaml`, `*.env`, `Dockerfile*`,
  `docker-compose*`, hors `docs/`, `tests/`, `examples/`, `.github/`) fixe
  `STORAGE_BACKEND` ou `backend:` à obrain.

Que `DEFAULT_BACKEND` sélectionne bien le backend Neo4j est prouvé par le test unitaire
`default_backend_is_selected_for_the_default_value` (module `storage`, job Unit Tests).

Preuves rejouées sur copie : config recâblée sur `OBRAIN_BACKEND` → refusé ; deuxième
définition sous `cfg` → refusé ; `deploy/prod.env` avec `STORAGE_BACKEND=obrain` → refusé ;
tous en exit 1.

### G4 — conformité des stores
`tests/store_conformance.rs` exécute chaque contrôle sur Neo4j, le mock et obrain. Les
divergences connues sont listées dans `EXPECTED_DIVERGENCES` ; le test échoue si une
divergence listée est corrigée (« now conforms »), ce qui force à retirer l'entrée.
- Jambe Neo4j : job **Integration Tests**, avec un service Neo4j et `HARNESS_NEO4J_REQUIRED=1`
  (un Neo4j absent est un échec, pas un saut).
- Jambe obrain : job **Obrain backend**. Le filtre CI ne retenait que `obrain_project::`
  (10 tests, 241 filtrés) ; il retient désormais tous les modules `obrain_` (86 tests).
- En local sans Neo4j, la jambe Neo4j passe sans s'exécuter : seule la CI fait foi.

## Portes manuelles ou non faites

### G6 — parité de lecture (LECTURE-NATIVE)
Rejouer les routes REST sur obrain et sur une copie Neo4j, champ à champ, et classer chaque
écart (non porté / perte de migration / lecture du portage / bénin). La mesure du 16/11 ne
vaut plus : elle portait sur une branche où les dates étaient mal lues.

### G7 — oracle de copie
Copie jetable, comptes par type égaux, empreinte par label égale, zéro nœud en quarantaine,
dates relues à la microseconde. C'est la porte que demande #1246. La spécification
`obrain-migration-spec.md` est **hors dépôt** (à la racine du dossier de travail) ; le
migrateur réel aussi (tâche `MIGRATEUR-REEL`, reportée jusqu'à la fin de la migration).

### G8 — performance
Barre de `CONSOLIDATION.md`. Mesurée le 20/11 à 100 k et 1 M documents sur SearchStore, avec
l'exemple `examples/obrain_mesures_recherche.rs` **de la branche de #1251** (absent de la
consolidation) : ouverture 0,6 s → 7 s, filtre par projet 0,4 ms → 8,9 ms, suppression par id
5 ms → 21 ms. Barre non tenue. Une PR qui touche un chemin chaud doit livrer ses mesures à
1 M et 5 M avant et après.

### G9 — double-run
La porte de justesse décisive avant la bascule. Non conçue.

### G10 — retour arrière
Sur la branche de #1248 (`src/storage/migrate.rs`, absent de la consolidation) : rollback par
configuration, source jamais modifiée, testé. Non fusionné.

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
