# 04 — Le cognitif d'obrain comme source de vérité des synapses

Analyse en lecture seule de `/Users/triviere/projects/obrain/obrain-db` (v0.0.1, `Cargo.toml:17`). Aucun build, aucun test lancé : tout ce qui suit est lu, pas mesuré. Chemins relatifs à `crates/`.

## 0. Cinq constats qui décident

1. **Aucune décroissance ne tourne en production.** `EnergyStore::decay_all` (`obrain-cognitive/src/energy.rs:448`) et `SynapseStore::decay_all` (`synapse.rs:1012`) n'ont aucun appelant hors tests. Le thinker `Consolidator`, censé les piloter, est un **stub** : `thinker/consolidator.rs:105-123` incrémente un compteur et renvoie (`TODO(T13-S2)`). `spawn_standard_fleet` (`thinker/mod.rs:80`) n'est appelée nulle part hors tests. Le moteur serait donc, tel quel, un stockage de poids qui ne s'oublient jamais.
2. **La décroissance substrat n'est pas rejouable, par construction.** En mode substrat elle est un facteur multiplicatif par tick (`decay_all_edge_synapse(factor)`, `store.rs:13442` ; `decay_factor: 0.99` par tick de 60 s, `consolidator.rs:53-56`), sans horodatage dans la colonne. Un processus arrêté ne décroît pas ; deux appels = deux décroissances ; le résultat dépend du nombre de ticks, pas du temps écoulé. PO fait l'inverse : poids valide « as of » `decayed_at` (`backend/src/neo4j/note.rs:3375-3382`), décroissance recalculée à partir de l'horloge, donc idempotente.
3. **La perte de `reinforcement_count` / `last_reinforced` est un câblage manquant, pas un format manquant.** Les deux sont déjà écrits en propriétés d'arête (`PROP_SYNAPSE_LAST_REINFORCED_EPOCH`, `PROP_SYNAPSE_REINFORCEMENT_COUNT`, `store_trait.rs:62-65`) via la file `MetadataDelta` vidée par `flush_pending_metadata` (`synapse.rs:892-925`) et lus par `load_from_graph` (`synapse.rs:1398-1405`). Mais le chemin de boot réel, `load_from_substrate_column` (`synapse.rs:628`, `1440-1465`), ne les lit pas et met `1` / `Instant::now()`. Et la file n'est vidée que si quelqu'un lance `SynapseConsolidator::spawn` (`synapse.rs:379`) : personne ne le fait dans `database/mod.rs`.
4. **Les cicatrices ne sont pas câblées du tout.** `ScarStore` n'apparaît ni dans le trait `CognitiveEngine` (`engine.rs:56`), ni dans le builder, ni dans `database/mod.rs` : seul `tenant.rs:84` l'instancie. En mode substrat, seul un champ de 5 bits (`record.rs:330-345`, max 4.0, pas ≈ 0,129) est écrit (`scar.rs:315-323`) : toute cicatrice cumulée < ≈ 0,065 s'arrondit à 0, tout > 4 sature, raison et id sont perdus.
5. **L'API externe est étroite et non stable.** Les procédures `obrain.cognitive.*` prennent des **ids entiers de nœud** (`cognitive_procedures.rs:151-169`) ; `search` exige un `query_embedding` (`:574`) et passe `vector_candidates` vide (`:603`) ; il n'existe aucune procédure d'écriture d'énergie, de cicatrice, de décroissance ou de soin. Version `0.0.1`, aucun `#[non_exhaustive]` dans `obrain-cognitive/src`.

## 1. Pertes de persistance : où et à quel coût

### Ce qui existe déjà

| Donnée | Où | Format | Coût |
|---|---|---|---|
| énergie | `NodeRecord.energy` (`record.rs:91`) | u16 Q1.15, `0x8000` ≈ 1.0 ; boost WAL-first (`energy.rs:384`) | 0 octet de plus (dans les 40 o du nœud, `record.rs:114`) |
| poids de synapse | `EdgeRecord.weight_u16` (`record.rs:412`) | Q0.16 normalisé par `max_synapse_weight`, écrêté à 1,0 (`synapse.rs:849`) | 36 o par arête (`record.rs:434`) |
| cicatrice (somme) | `scar_util_affinity` (`record.rs:93`) | 5 bits | 0 o |

Place disponible mesurée dans les structs : `NodeRecord._reserve: u32` (« toujours 0 en v3 », `record.rs:109-111`) ; `EdgeRecord._pad: [u8; 2]`, `ricci_u8`, `flags`, `engram_tag` (`record.rs:425-431`).

### Mécanismes existants à réutiliser

- **Propriétés d'arête / de nœud** (`set_edge_property`, `store.rs:23274`) : encodage `PropertyValue::F64`/`I64` en pages de 4 096 o (`page.rs:63`). C'est le chemin déjà utilisé pour `epoch` + `count` ; amorti, mais une page 4 Ko par entité qui porte une propriété et un WAL par écriture. À 30 000 synapses (ordre de grandeur cité `synapse.rs:1500`), c'est le coût que `MetadataDelta` cherche à sortir du chemin chaud (`synapse.rs:811-826`).
- **Zones nommées** (`open_named_zone`, `store.rs:13323`), utilisées pour `synapse_incidence` et `engram_members`. C'est le bon support d'un journal de cicatrices.

### Recommandation

| Champ | Support proposé | Octets | Remarque |
|---|---|---|---|
| `last_reinforced` | u32 secondes epoch **dans la zone** ou dans `_pad` (2 o) sous forme de **jour relatif** | 2–4 / arête | 2 o de `_pad` = u16 en minutes depuis une base du store (45 jours) : trop court. Préférer une zone parallèle `synapse_meta` indexée par slot (u32 epoch + u16 count = 6 o / arête, soit 180 Ko pour 30 000) |
| `reinforcement_count` | idem, u16 saturant | 2 | PO protège à partir d'un seuil (`protected_after`, `note.rs:3335`) : un u16 suffit |
| `energy.last_activated` | `NodeRecord._reserve` (u32 epoch secondes) | 0 de plus | aujourd'hui absent : l'énergie substrat n'a pas d'horodatage, donc rien à rejouer |
| cicatrices individuelles | zone journal `scars` : `(node u32, id u64, intensité f32, reason u8+arg, created u32, healed u32)` ≈ 28 o | 28 / cicatrice | le champ 5 bits reste en cache de lecture rapide, dérivé du journal |

Point de vigilance : la zone doit être écrite dans le WAL comme les colonnes, sinon la perte de 30 s sur `kill -9` (`synapse.rs:812-815`) devient une dérive silencieuse du compte.

## 2. Le temps

- État : `Synapse.last_reinforced`, `NodeEnergy.last_activated`, `Scar.created_at` sont des `web_time::Instant` (`synapse.rs:83`, `energy.rs:92`, `scar.rs:127`), non sérialisables. Le pont est `epoch_to_instant` (`store_trait.rs:178-194`) : il **plafonne à 365 jours**, retombe sur `Instant::now()` si l'epoch manque, et lit l'horloge système (pas de saut d'horloge géré).
- Ce que ça donne : sur le chemin `load_from_graph`, la décroissance est correcte (poids « as of » epoch). Sur `load_from_substrate_column`, tout repart « renforcé à l'instant du boot » : un poids vieux d'un an redevient plein.
- Pour une décroissance rejouable il faut : (a) un horodatage persistant par entité (§1) ; (b) remplacer `decay_all(factor)` par `decay_to(now_epoch)` qui calcule `w · 2^(−Δt/half_life)` depuis l'horodatage stocké puis le met à jour (idempotent, indépendant de la cadence) ; (c) fournir l'horloge en paramètre (`Clock` injectable) pour rejouer un WAL.
- Comparaison PO : `decayed_at` stocké par arête, `s1.decayed_at = datetime()` (`note.rs:3091`), poids lu « as of » `coalesce(decayed_at, now)` (`note.rs:3375`). C'est exactement (b). Le moteur doit adopter ce modèle ; c'est la seule condition pour que PO et obrain donnent le même poids au même instant.

## 3. Démarrage des planificateurs

État :

| Planificateur | Défaut | Instancié ? |
|---|---|---|
| `GdsRefreshScheduler` (`gds_refresh.rs:69`) | intervalle 300 s, seuil 1 000 mutations (`:48-49`) | **non**, et **n'a pas de boucle** : `refresh_interval` n'est lu que par les tests (grep). C'est un compteur + une méthode `refresh(&dyn GraphStore)` à appeler soi-même |
| `Consolidator` | 60 s, `decay_factor 0.99`, prune u16 64 | non, et corps **no-op** |
| `CommunityWardenThinker` | 300 s (`warden.rs:38`) | non. **Attention au sens** : c'est la compaction de pages Hilbert par fragmentation de communauté (`obrain-substrate/src/warden.rs:213`), pas un « warden d'oubli » façon PO |
| `Predictor` | 30 s | non |
| `Dreamer` | 600 s | non |
| `SynapseConsolidator` (vidage `MetadataDelta`) | ~30 s | non |

Qui doit démarrer : le **processus hôte** appelle `spawn_standard_fleet(store, &ThinkersConfig, sensor)` (`thinker/mod.rs:80`), documentée comme « point d'entrée unique », volontairement sans auto-démarrage. Un câblage dans `database/mod.rs` : juste après `engine_dyn` (`database/mod.rs:892-905`), sous `#[cfg(feature = "cognitive")]`, stocker le `ThinkerRuntime` et le `SynapseConsolidator` à côté de `_cognitive_scheduler` (`mod.rs:306`) pour qu'ils soient arrêtés au `Drop` de la base. Il faut aussi : un `GdsRefreshScheduler` branché sur le `MutationBus` (le `Scheduler` réactif existe déjà, `mod.rs:846`) qui appelle `record_mutations` puis `refresh` quand le seuil est franchi ET que l'intervalle est écoulé (les deux conditions, aujourd'hui aucune n'est évaluée).

Coût indicatif (à mesurer, non mesuré ici) : `refresh` recalcule PageRank (≤100 itérations), betweenness et Louvain sur tout le graphe (`gds_refresh.rs:117-123`) : betweenness exacte est O(V·E). À prévoir : seuil de mutations plus haut et exécution sur thread dédié, avec le `PressureSensor` du runtime thinker (`runtime.rs`) pour céder la place.

Feature : `cognitive` est déclarée dans `obrain-engine/Cargo.toml:136` et relayée par `obrain/Cargo.toml:47`, mais `default = ["embedded"]` (`obrain/Cargo.toml:27`) : un consommateur doit l'activer explicitement, et `with_substrate` n'est alors câblé que parce que `database/mod.rs:842` est sous `cfg(feature = "cognitive")`.

## 4. API pour un consommateur externe

| Surface | Signature | Stabilité |
|---|---|---|
| `spread` | `fn spread(sources: &[(NodeId, f64)], graph: &dyn ActivationSource, config: &SpreadConfig) -> ActivationMap` (`activation.rs:356`) ; `SpreadConfig { max_hops, min_propagated_energy, decay_factor, activation_threshold, max_activated_nodes (défaut 1000) }` (`:35-47`) | lib Rust uniquement ; `ActivationSource` a des méthodes par défaut et un préfetch (`:136-156`) : trait évolutif sans garde `#[non_exhaustive]` |
| `SearchPipeline::search` | `fn search(&self, vector_candidates: &[(NodeId, f64)], config: &SearchConfig) -> Vec<SearchResult>` (`search.rs:260`) ; résultat à 4 signaux (similarité, énergie, topologie, synapse) | **l'appelant fournit les candidats vectoriels** ; le moteur ne fait pas la recherche texte |
| `CALL obrain.cognitive.spread(seeds, {hops, causal_edge_types, causal_boost}) YIELD activated, energy` | entiers d'id ; (`cognitive_procedures.rs:121-205`) | seeds = ids numériques, pas d'identifiant applicatif. Renvoie des lignes seulement si `synapse_store()` est présent, sinon **résultat vide sans erreur** (`:184`) |
| `obrain.cognitive.reinforce(a, b, {amount})` | `:223` | seule écriture exposée |
| `obrain.cognitive.search({query_embedding, weights, limit})` | `:556`, candidats vides (`:603`) | **vide en pratique** tant que l'appelant ne peut pas passer de candidats |
| `bootstrap`, `distill`, `detect_gaps`, `synthesize`, `metrics`, `engrams.*` | — | présents |
| UDF `obrain.energy`, `obrain.risk`, `obrain.synapses` | `database/mod.rs:900` | lecture |

Absences pour PO : pas de procédure pour écrire une énergie, poser/soigner une cicatrice, décroître à un instant donné, lire `last_reinforced`/`count`, lister les synapses d'un nœud avec métadonnées, ni correspondance entre l'`NodeId` obrain (u64 dense, `activation.rs:367`) et l'identifiant de note PO. Stabilité : version 0.0.1, pas de politique de versionnage de procédures.

## 5. Ce que PO a et que le moteur n'a pas

| Fonction PO | Dans obrain ? | Où la mettre |
|---|---|---|
| Archivage réversible `:FORGOTTEN` | **non** (seul `engrams.forget`, `cognitive_procedures.rs:952`, qui est destructif : `relations_removed`) | **Moteur** : un drapeau `FORGOTTEN` dans `EdgeRecord.flags` / `node_flags` exclu des lectures par défaut est O(1) et évite deux sources de vérité. La *politique* (quand oublier) reste côté appli |
| Bornes anti-emballement (`max_archive_fraction`, `protected_after`/`protection_floor`, `note.rs:3335-3407`) | **non** : seul `synapse_prune_threshold_u16` fixe (`consolidator.rs:54`), qui supprime | **Moteur** pour `protected_after` + plancher (c'est une propriété du poids, elle doit vivre avec lui, et dépend du `count` persisté §1) ; `max_archive_fraction` aussi, car il garde l'invariant d'un balayage non borné. Les valeurs par défaut restent configuration de l'appli |
| Score de risque composite | **oui, en partie** : `compute_risk_score` (`fabric.rs:541`), 6 termes (pagerank, fréquence de mutation, écart d'annotation, écart de connaissance, betweenness, cicatrice), `RiskWeights`, persisté en `PROP_FABRIC_RISK` | Moteur déjà. À aligner terme à terme avec la formule PO avant de basculer ; les poids restent côté appli |
| `find_bridges` | algorithmes `bridges` / `articulation_points` dans `obrain-adapters/src/plugins/algorithms/structure.rs`, exposés par `obrain-engine/src/procedures.rs:94` | Déjà là (algorithme générique). Le « pont de risque » PO (betweenness × churn) reste une requête appli |
| `decayed_at` rejouable | non | Moteur (§2) |
| Cicatrice individuelle avec raison | type `Scar` en mémoire, pas câblé | Moteur (§1) |

## 6. Ce qui manque au moteur pour être la vérité des synapses

Par ordre de dépendance :

1. Horodatage persistant par synapse, par énergie de nœud et par cicatrice ; `decay_to(now)` idempotent à la place de `decay_all(factor)`.
2. `reinforcement_count` et `last_reinforced` rechargés par `load_from_substrate_column` ; file `MetadataDelta` vidée au boot de la base et à l'arrêt.
3. Journal de cicatrices individuelles et `ScarStore` câblé dans le builder et `CognitiveEngine`.
4. Consolidator réel (décroissance + élagage) ; `spawn_standard_fleet`, `SynapseConsolidator`, `GdsRefreshScheduler` démarrés et arrêtés par `database/mod.rs` ; boucle de GDS qui évalue intervalle ET seuil.
5. Archivage réversible (`FORGOTTEN`) et bornes `protected_after`, `max_archive_fraction` dans le moteur.
6. Procédures d'écriture et de lecture complètes avec métadonnées, et un mapping stable id applicatif ↔ `NodeId` ; `search` qui accepte du texte ou des candidats.
7. Politique de version des procédures et `#[non_exhaustive]` sur les structs de configuration publiques, avant 0.1.
8. Test de redémarrage de bout en bout : renforcer, fermer, rouvrir, comparer poids, compte, horodatage et cicatrices (le bench du 2026-08-31 cité `synapse.rs:624` ne couvre que le poids).

## Limites de cette analyse

Lecture seule : aucun comportement n'a été exécuté. Non vérifié : si `set_edge_property` réussit sur un slot SYNAPSE sans adjacence (`is_live_edge_on_disk`, `store.rs:23275`), ce dont dépend la persistance de `epoch`/`count` ; le coût réel en octets des pages de propriétés ; l'existence d'un autre binaire (hub) qui démarrerait la flotte hors de ce dépôt.
