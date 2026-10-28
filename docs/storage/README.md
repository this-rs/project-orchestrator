# Abstraction du stockage

Objectif : le code applicatif ne dépend plus d'un moteur de stockage concret. Il
parle à des **traits de domaine** ; chaque moteur devient un **adaptateur**
interchangeable qui déclare ce qu'il sait faire.

Ce chantier ne change aucun comportement observable. La référence n'est pas la
lecture du code : c'est la **suite de conformité**, exécutée par chaque
adaptateur, et les jeux dorés (requêtes, ordre de classement) qui figent les
résultats attendus.

## Les quatre pièces

| Pièce | Ce qu'elle garantit | Où |
|---|---|---|
| **Manifeste de capacités** | Chaque adaptateur déclare ce qu'il sait faire (vectoriel filtré ou non, unicité simple et composite, chemins de longueur variable, `shortestPath`, sous-requêtes, arithmétique temporelle, index de propriété de relation, transactions explicites). Une capacité absente est **refusée au démarrage**, pas découverte en production. | `src/neo4j/traits.rs`, `src/api/handlers.rs` |
| **Contrats écrits** | Ce qu'un adaptateur doit reproduire, dans le rustdoc des traits : tri et départage, pagination (total indépendant de `limit`/`offset`), politique « entité absente », `NULL` vs chaîne vide, horodatage, entier vs flottant. | rustdoc de `src/neo4j/traits.rs` |
| **Suite de conformité** | Un même scénario exécuté par tous les adaptateurs, avec un test par adaptateur généré par macro. Divergence = rouge, pas « connu ». | `tests/store_conformance.rs` |
| **Garde-fou** | `cargo xtask storage-boundary` : aucun accès direct au pilote hors des adaptateurs, y compris via les `Cargo.toml` des crates du workspace. Mode avertissement d'abord, liste d'exceptions qui ne fait que décroître. | `xtask/src/` |

## Règles de la branche

1. Les sous-PR ciblent **`storage-trait`**, jamais `main`.
2. Un domaine = une PR, un seul écrivain à la fois sur les traits.
3. La branche reste compilable à chaque merge (`cargo check --all-targets`).
4. Tout écart de comportement découvert est **testé avant** d'être corrigé, et consigné.

## Ordre d'avancement

1. Capacités + contrats (ce que l'abstraction promet).
2. Garde-fou (ce qu'elle interdit).
3. Conformité dorée (ce qu'elle prouve).
4. Découpe des traits de domaine (ce qu'elle rend possible).
5. Fabrique et sélection de l'adaptateur (ce qu'elle branche).
