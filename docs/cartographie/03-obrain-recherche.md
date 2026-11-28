# 03 — Recherche du Project Orchestrator sur obrain (remplacement de Meilisearch)

Analyse en lecture seule de `/Users/triviere/projects/obrain/obrain-db` (aucun build, aucun test rejoué). Contrat couvert : `backend/src/meilisearch/traits.rs` — les 27 méthodes de `SearchStore`, et leur implémentation `backend/src/meilisearch/client.rs`. Étiquettes : **[lu]** = lu dans le code ; **[déduit]** = conclusion de lecture, non mesurée ; **[doc]** = affirmé par un commentaire.

## Les 5 constats qui décident

1. **`text_search` n'a aucun paramètre de filtre.** Signature `text_search(label, property, query, k) -> Vec<(NodeId, f64)>` **[lu]** `obrain-engine/src/database/search.rs:581` ; le BM25 sous-jacent est `InvertedIndex::search(query, k)` **[lu]** `obrain-core/src/index/text/inverted_index.rs:314`. Le seul « filtre » est le couple `(label, propriété)` choisi à `create_text_index(label, property)` **[lu]** `obrain-engine/src/database/index.rs:462`. Aucun moyen de filtrer par `project_slug`/`note_type`/`status`/`importance`/`language`/`path` à l'intérieur de la recherche.

2. **La machinerie de filtre existe, mais elle est privée, gate `vector-index`, et jamais branchée sur le texte.** `compute_filter_allowlist` **[lu]** `search.rs:99` (liste blanche intersectée à partir de filtres d'égalité ET d'opérateurs `$eq/$ne/$gt/$gte/$lt/$lte/$in/$nin/$contains` évalués par `value_matches_operators` **[lu]** `search.rs:37`) n'est utilisée que par `vector_search`, qui sur-échantillonne ×4 pour compenser la perte de rappel du top-k filtré **[lu]** `search.rs:259`. `text_search` ne l'appelle pas. → c'est le plus petit point d'appui : ce code couvre déjà `$in` et le statut par défaut.

3. **`find_nodes_by_property` est une égalité stricte, globale, sans opérateur ni label.** `create_property_index(property)` (pas de label) **[lu]** `index.rs:34` ; `find_nodes_by_property(property, value)` **[lu]** `index.rs:113`. Composer « filtre structurel + recherche texte » se fait donc côté PO : 1..n lookups d'égalité intersectés (chaque `$in` exige autant de lookups que de valeurs), puis intersection avec les hits BM25. Coût : O(1) par filtre si la propriété est indexée, sinon balayage O(n) — mais l'intersection APRÈS `text_search(k)` perd le rappel (le top-k est déjà coupé), sauf à sur-échantillonner k soi-même. `$in` multi-valeur et préfixe ne sont pas exprimables par ce chemin.

4. **La persistance de l'index texte existe ; la 1re recherche après réouverture est ~gratuite OU un rebuild complet.** Fichier `substrate.textidx.<label>.<propriété>` écrit atomiquement (tmp + rename) **[lu]** `text_index_persist.rs:54-61,45`, validé par une empreinte `(offset WAL, nb nœuds vivants)` **[lu]** `text_index_persist.rs:70-76`. À l'ouverture, `hydrater_index_textuels` **[lu]** `:317` CHARGE depuis le disque si l'empreinte colle ; sinon il DIFFÈRE (pas de rebuild à l'ouverture — défaut mesuré corrigé **[doc]** `:335-359`). Conséquence : 1re recherche = 0 si la base a été rouverte proprement en lecture (empreinte valide) ; = balayage complet du label + reconstruction BM25 (~1,6 s mesuré sur les bases runtime **[doc]** `:1-10`) si une écriture a invalidé l'empreinte — empreinte conservatrice : TOUTE écriture invalide, même hors label indexé.

5. **`hybrid_search` fusionne texte+vecteur par RRF (k=60) et TOLÈRE l'index manquant.** `hybrid_search` **[lu]** `search.rs:647` collecte les sources texte (`text_search`) et vecteur (`vector_search`) puis `fuse_results` avec `FusionMethod::Rrf { k: 60 }` par défaut **[lu]** `obrain-core/src/index/text/fusion.rs:29,57`. Un index manquant = source SAUTÉE (pas d'erreur) ; les deux manquants = `Ok(Vec::new())` **[lu]** `search.rs:688-727` ; une vraie panne (dimension incohérente) PROPAGE (`?`). Mais : (a) toujours aucun filtre, (b) il exige un index vectoriel (embeddings), hors du contrat `SearchStore` qui est purement lexical.

## Tableau des 27 méthodes

Verdicts — **direct** = API obrain publique suffit, sans perte sémantique ni changement d'obrain ; **adaptation** = faisable côté PO avec un compromis (post-filtre qui perd le rappel, score BM25 au lieu du ranking 0–1 de Meili, champ `search_text` concaténé qui perd la pondération de champs, `is_indexing` toujours `false`) ; **impossible** = exige un ajout à obrain (filtre sur index texte / préfixe).

| # | Méthode | Verdict | Preuve / remarque |
|---|---|---|---|
| 1 | `index_code` | direct | `create_node_with_props(&["code"], …)` `crud.rs:167`. Les 5 champs recherchables (`symbols,docstrings,signatures,path,imports` `client.rs:90-95`) concaténés en une propriété `search_text` (perte de pondération). |
| 2 | `index_code_batch` | direct | `batch_create_nodes` `crud.rs:1081` ou boucle. |
| 3 | `search_code` | adaptation | `text_search("code","search_text",q,k)` + filtre `language` post-hoc (perte de rappel sans sur-échantillonnage). |
| 4 | `search_code_in_project` | adaptation | idem + filtre `project_slug`. |
| 5 | `search_code_with_scores` | **impossible** | `language` + `project_slug` + `path_prefix` (préfixe `path STARTS WITH` `client.rs:234`) + scores. Le préfixe n'a aucune primitive obrain. |
| 6 | `delete_code` | adaptation | `find_nodes_by_property("path", path)` `index.rs:113` + `delete_node` `crud.rs:325` (index de propriété sur `path` requis). |
| 7 | `delete_code_for_project` | adaptation | `find_nodes_by_property("project_slug", slug)` + delete. |
| 8 | `delete_orphan_code_documents` | adaptation | balayage `nodes_by_label("code")` + filtre `project_id`/`project_slug` vide + delete. |
| 9 | `delete_all_code` | direct | balayage `nodes_by_label("code")` + `delete_node`. |
| 10 | `get_code_stats` | adaptation | `total_documents` = `node_count_by_label("code")` **[lu]** `admin.rs:27` (O(1), public) ; `is_indexing` = `false` (indexation synchrone). |
| 11 | `index_decision` | direct | `create_node_with_props(&["decisions"], …)`. |
| 12 | `search_decisions` | direct | **Seule méthode sans filtre.** `text_search("decisions","search_text",q,k)` `search.rs:581`. |
| 13 | `search_decisions_in_project` | adaptation | filtre `project_slug` post-hoc. |
| 14 | `search_decisions_in_projects` | **impossible** | `project_slug IN […]` `client.rs:383` : `$in` multi-valeur sur le filtre texte + fusion de scores, non exprimable sans filtre sur index texte. |
| 15 | `delete_decision` | adaptation | `find_nodes_by_property("id", id)` + delete. |
| 16 | `delete_decisions_for_project` | adaptation | `find_nodes_by_property("project_slug", slug)` + delete. |
| 17 | `delete_decisions_for_task` | adaptation | `find_nodes_by_property("task_id", task_id)` + delete. |
| 18 | `index_note` | direct | `create_node_with_props(&["notes"], …)`. |
| 19 | `index_notes_batch` | direct | `batch_create_nodes`. |
| 20 | `search_notes` | adaptation | filtre statut IMPLICITE `status IN ["active","needs_review"]` `client.rs:504-505` — déjà un filtre, pas une recherche nue. |
| 21 | `search_notes_with_filters` | adaptation | 4 filtres égalité + statut par défaut, post-hoc (perte de rappel). |
| 22 | `search_notes_with_scores` | **impossible** | 4 filtres (`project_slug,note_type,status,importance`) + statut par défaut + scores `client.rs:482-536`. |
| 23 | `delete_note` | adaptation | `find_nodes_by_property("id", id)` + delete. |
| 24 | `delete_notes_for_project` | adaptation | `find_nodes_by_property("project_slug", slug)` + delete. |
| 25 | `update_note_status` | direct | `set_node_property(id, "status", …)` `crud.rs:451` (entretient l'index texte automatiquement `crud.rs:540-560`). |
| 26 | `get_notes_stats` | adaptation | `node_count_by_label("notes")` ; `is_indexing=false`. |
| 27 | `health_check` | adaptation | obrain est embarqué (pas de remote) : `Ok(true)` après `open` réussi, pas d'endpoint à sonder. |

Bilan : **8 direct, 16 adaptation, 3 impossible**. Les 16 « adaptation » se regroupent en 3 familles : (a) les recherches filtrées à 1–2 égalités (3, 4, 13, 20, 21), (b) les suppressions par propriété (6–8, 15–17, 23, 24), (c) les stats (10, 26, 27). Chaque famille se résout par le même petit morceau de glue PO, mais la famille (a) porte un compromis réel de rappel tant que le filtre n'est pas natif.

## Ce qu'il faudrait ajouter à obrain (plus petit changement)

**1. Filtre sur index texte — un paramètre, pas une nouvelle API.** Rendre `compute_filter_allowlist` (`search.rs:99`) `pub` et le dé-gate de `#[cfg(feature = "vector-index")]` (il ne dépend que de `data_store()` et `value_matches_operators`, tous deux hors feature) ; ajouter à `text_search` un `filters: Option<&HashMap<String, Value>>` avec le même sur-échantillonnage ×4 borné à 1024 que `vector_search` (`search.rs:259`), puis intersecter la liste blanche sur les hits BM25. Ceci couvre d'un coup : `language`, `project_slug`, `note_type`, `status`, `importance` (égalité) ET `project_slug IN […]` et `status IN [active,needs_review]` (`$in` déjà implémenté `search.rs:68-71`) — soit 8 méthodes de recherche.

**2. Préfixe sur propriété — un opérateur.** Pour `path_prefix` (`search_code_with_scores`), ajouter `$starts_with` dans `value_matches_operators` (`search.rs:37`), évalué par `Value::String(s) => s.starts_with(n)`. La liste blanche filtre déjà les nœuds du label en mémoire, donc aucun index de préfixe n'est nécessaire : coût O(|label|) par filtre, acceptable pour le volume code d'un projet. (Alternative plus lourde côté PO : indexer un préfixe précalculé en égalité — fragile.)

**3. Stats — rien à ajouter.** `total_documents` = `node_count_by_label(label)` (`admin.rs:27`, O(1), déjà public). `is_indexing` : obrain indexe de façon SYNCHRONE (pas de file de tâches comme Meili), donc toujours `false` — à documenter, pas à implémenter.

Rien d'autre n'est requis : les suppressions et `update_note_status` se font avec `find_nodes_by_property` + `delete_node`/`set_node_property`, tous publics.

## Les 3 méthodes qui ne se mappent pas

1. **`search_code_with_scores`** — cumule le filtre d'égalité (`language`, `project_slug`) et le filtre de PRÉFIXE (`path_prefix`), tous deux absents de `text_search`, plus les scores. C'est la seule méthode à exiger le préfixe, primitive qu'obrain n'a nulle part.
2. **`search_notes_with_scores`** — 4 filtres d'égalité simultanés + le filtre statut PAR DÉFAUT (`status IN [active, needs_review]`, une disjonction) + les scores. Un post-filtre côté PO ne peut pas restituer un top-k exact sans réécrire la boucle de recherche (sur-échantillonnage + liste blanche), c'est-à-dire réimplémenter le filtre natif.
3. **`search_decisions_in_projects`** — `project_slug IN [liste]` : un filtre `$in` multi-valeur sur le texte + une fusion de scores entre bases/projets. `find_nodes_by_property` ne rend qu'une égalité stricte ; l'union de N recherches + re-tri n'est pas le contrat (scores non comparables sans fusion).

Les 3 partagent la même cause : `text_search` n'accepte aucun filtre. Les deux premières imposent en plus (respectivement) un préfixe et une disjonction par défaut — ce qui fait de `search_code_with_scores` le plus coûteux à honorer fidèlement.
