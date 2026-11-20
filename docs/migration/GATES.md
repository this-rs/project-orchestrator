# Portes de qualité de la migration Neo4j → obrain (+ this-rs)

Ce document est le **contrat** des contrôles qui empêchent la migration de régresser,
et qui permettent de la contrôler de bout en bout. Il liste chaque porte avec sa
commande, ce qu'elle prouve, ce qu'elle ne prouve PAS, et son **état réel**.
Un statut « non fait » est un statut honnête : une porte absente n'est pas une porte verte.

Directive qui gouverne tout le reste (20/11/2026) : **on ne branche rien en production
tant que la migration vers obrain et this-rs n'est pas faite.** Le backend par défaut
reste donc `default` (Neo4j) ; la garde de bascule (#1248) est du code inactif.

Référence de la cible : `CONSOLIDATION.md` (règles, barre de performance, porte de bascule).

## Vue d'ensemble

| # | Porte | Ce qu'elle prouve | Automatisée | Commande | État réel (20/11/2026) |
|---|---|---|---|---|---|
| G1 | containment | la branche contient `origin/main` (pas de dérive) | oui (CI, Lint) | `cargo xtask containment` | **vert** sur la consolidation (tête `7e1b92dd`, après #1253) |
| G2 | cliquet des traits portés | un trait de contrat porté à la main ne perd pas son port | oui (CI, Lint) | `cargo xtask migration-gate --strict` | **20 / 22** traits portés ; restent `AnalyticsStore` (#1250) et `SearchStore` (#1251) |
| G3 | défaut neo4j | le backend par défaut reste `default` tant que la migration n'est pas finie (directive prod) | oui (CI, Lint) | même commande | **conforme** |
| G4 | conformité des stores | chaque store répond pareil sur Neo4j, le mock et obrain (`tests/store_conformance.rs`, cliquet `EXPECTED_DIVERGENCES`) | partiel | `cargo test --features obrain-backend,mock-store --test store_conformance` | mock et obrain : **vert** en CI (job « Obrain backend », sous-ensemble `obrain_`). **La jambe Neo4j ne tourne pas en CI** : aucune instance n'y est jointe, les tests `neo4j_*` passent sans s'exécuter. |
| G5 | refus nommés | une méthode non portée refuse par son nom (`NonPorte`), jamais un résultat vide | partiel | assertions `not_ported` de `tests/obrain_backend.rs` | couvert par trait, pas par méthode ; le cliquet G2 suit les traits, pas les méthodes |
| G6 | parité de lecture sur copie | les mêmes requêtes REST rendent les mêmes réponses sur obrain et sur une copie Neo4j | **non** | rejeu manuel (LECTURE-NATIVE, 209 routes) | **à rejouer** sur la consolidation : la dernière mesure date du 16/11, sur une branche antérieure, avant le décodeur de dates |
| G7 | oracle de copie | comptes par type, empreinte par label, zéro quarantaine après migration sur copie | **non** | `obrain-migration-spec.md` (spec) ; migrateur **hors dépôt** | **non fait** sur la consolidation ; le migrateur réel n'est pas branché (c'est attendu avant la bascule) |
| G8 | performance | barre de `CONSOLIDATION.md` : rien ne croît avec N sur un chemin de requête, à 100 k et 1 M, médiane et p99 | **non** | `examples/obrain_mesures_recherche.rs` (SearchStore) | **non tenue** : ouverture (0,6 s → 7 s), filtre par projet (0,4 ms → 8,9 ms), suppression par id (5 ms → 21 ms) |
| G9 | double-run sans divergence | un run Neo4j et un run obrain sur le même trafic ne divergent pas (justesse) | **non** | à concevoir (O-SOAK) | **non fait** |
| G10 | retour arrière par configuration | rollback = une variable, la source n'est jamais modifiée | partiel | tests de `storage::migrate` (#1248) | implémenté et testé sur #1248 ; **non fusionné** |
| G11 | stabilité des tests | le test `kill9_during_creations…` ne dépend ni du timing ni d'une ligne de journal tronquée | non | tâche `TEST-KILL9-INSTABLE` (#843c1406) | **ouvert** : échoue au hasard (deux modes d'échec mesurés le 20/11) |

## Détail des portes

### G1 — containment
Garde-fou de dérive : la branche doit contenir `origin/main`, sinon une PR vers la
consolidation est rouge pour une raison qui n'est pas son code. Se débloque par une
descente de `main` dans la consolidation (#1253, fusionnée).

### G2 — cliquet des traits portés (`cargo xtask migration-gate`)
- Source de vérité du contrat : `pub trait` de `src/neo4j/traits.rs` et `src/meilisearch/traits.rs` (22 traits).
- Un trait est **porté** s'il a un `impl <Trait> for Obrain…` sous `src/obrain/`. Seuls les traits du contrat comptent.
- Le cliquet `xtask/migration-portees.txt` liste les traits portés à la date de revue. Il **ne fait que grossir**.
  - Un trait listé qui n'est plus porté → **RÉGRESSION**, exit 1 en `--strict`.
  - Un trait porté mais non listé → à consigner (exit 0).
- Ce que la porte ne prouve pas : que le port est juste. C'est le rôle de G4.

Preuve de fonctionnement (20/11) : retirer `impl ProjectStore for ObrainGraphStore`
fait sortir `RÉGRESSION ProjectStore`, exit 1 ; rétablir le défaut fait repasser en vert.

### G3 — défaut neo4j
Même commande que G2. Refuse si `DEFAULT_BACKEND` n'est plus `"default"` dans
`src/storage.rs` ou `src/storage/mod.rs`. Preuve : changer le défaut en `"obrain"`
fait sortir `BACKEND PAR DÉFAUT … refusé`, exit 1.

### G4 — conformité des stores
`tests/store_conformance.rs` exécute chaque contrôle sur le mock et sur obrain. Les
divergences connues sont listées dans `EXPECTED_DIVERGENCES` ; le test échoue si une
divergence listée est corrigée (« now conforms »), ce qui force à retirer l'entrée.
**Trou connu** : la jambe Neo4j exige une instance joignable, qui n'existe ni en CI ni
en local. Tant qu'elle n'est pas branchée, la conformité Neo4j n'est pas prouvée.

### G5 — refus nommés
Chaque méthode d'un trait non porté répond `NonPorte("Trait::méthode")`. Le cliquet G2
suit les traits ; le détail par méthode est couvert par les assertions `not_ported`.
Une méthode qui rendrait `Ok(vide)` au lieu de refuser serait une régression silencieuse
non détectée par G2 : à surveiller dans les relectures.

### G6 — parité de lecture (LECTURE-NATIVE)
Rejouer les routes REST sur obrain et sur une copie Neo4j, champ à champ, classer chaque
écart (non porté / perte de migration / lecture du portage / bénin). **À refaire** sur la
consolidation : la mesure du 16/11 ne vaut plus, elle portait sur une branche où les
dates étaient lues à tort (130 713 nœuds en quarantaine avant le décodeur unique).

### G7 — oracle de copie
Avant toute bascule : copie jetable, comptes par type égaux, empreinte par label égale,
zéro nœud en quarantaine, aller-retour des dates à la microseconde. C'est la porte de
#1246 (« qualification »). Le migrateur réel est hors dépôt : cette porte attend qu'il
soit intégré (tâche `MIGRATEUR-REEL`, à reporter jusqu'à la fin de la migration).

### G8 — performance
Barre de `CONSOLIDATION.md` : une recherche indexée, un décompte, un contrôle
d'unicité ne croissent pas avec N ; aucune reconstruction proportionnelle à N sur le
chemin d'une requête, d'une écriture ou d'une ouverture. Mesurée à 100 k et 1 M sur
SearchStore : **non tenue** sur trois chemins (détail dans le tableau ci-dessus).
Une PR qui touche un chemin chaud doit livrer ses mesures à 1 M et 5 M avant et après.

### G9 — double-run
Porte de justesse la plus importante avant la bascule : un même trafic joué sur Neo4j et
sur obrain, divergences classées. **Non conçue.**

### G10 — retour arrière
Implémenté dans #1248 (`rollback` par configuration, source jamais modifiée), testé.
Non fusionné : il dépend de la correction des défauts relevés par la relecture.

### G11 — stabilité des tests
`kill9_during_creations…` échoue au hasard, avec deux modes distincts :
(1) le garde-fou de timing « no kill landed inside a creation over 12 rounds » ;
(2) un `unwrap()` sur une ligne tronquée du journal `acquittes` quand le SIGKILL tombe
pendant l'écriture. Une porte instable est une porte qui ne prouve rien : elle doit être
corrigée par construction, pas relancée jusqu'au vert.

## Règle de contrôle

- Une PR qui touche la migration doit consigner ses portes dans un commentaire (critères
  → rouge-sans / vert-avec → relecture indépendante → contrat et doc → à jour de la base → CI).
- Le cliquet G2 ne se modifie que dans le sens d'un ajout ; un retrait est une décision
  écrite dans ce document.
- La bascule en production n'est pas une porte de plus : elle exige G4 (Neo4j joint),
  G6, G7, G8 et G9 **toutes vertes**, plus la levée de la directive du 20/11.
