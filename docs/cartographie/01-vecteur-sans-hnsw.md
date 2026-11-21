# 01 — Vectoriel sans HNSW : ce dont obrain a besoin

Mesuré le 10/11/2026 sur Apple M4 Max (16 cœurs), `RUSTFLAGS="-C target-cpu=native"`,
crate `vecscan` (obrain-sandbox/vecbench), données réelles PO 768-d unit-normées
(`note_embeddings.json` 8 131 + `file_embeddings.json` 24 390 = 32 521 vecteurs).
Métrique de similarité : produit scalaire (cosinus sur vecteurs normalisés).

## 1. Balayage exact (recall@10 = 1,0 par construction)

| N | synthétique | p50 1T | p95 1T | p99 1T | p95 16T | p99 16T | qps 1T | qps 16T | mémoire |
|---|---|---|---|---|---|---|---|---|---|
| 8 131 (notes) | 0 | 1,70 ms | 1,81 ms | 1,82 ms | 0,85 ms | 1,17 ms | 585 | 1 529 | 24 Mo |
| 24 390 (fichiers) | 0 | 5,03 ms | 5,63 ms | 6,70 ms | 1,42 ms | 1,59 ms | 196 | 820 | 74 Mo |
| 32 521 (fusion) | 0 | 6,71 ms | 7,26 ms | 7,35 ms | 1,57 ms | 1,69 ms | 147 | 714 | 99 Mo |
| 100 000 | 67 479 | 20,45 ms | 21,91 ms | 22,78 ms | 4,36 ms | 4,98 ms | 48 | 270 | 307 Mo |
| 500 000 | 467 479 | 103,6 ms | 111,1 ms | 113,0 ms | 17,81 ms | 18,49 ms | 10 | 59 | 1 536 Mo |

## 2. Le seuil (extrapolation linéaire des points mesurés)

p95 mono ≈ **0,222 ms / 1 000 vecteurs** (linéaire, écart < 2 % sur les 5 points).

- **5 ms** → ≈ **22 k** vecteurs (interpolation mesurée 21,7 k entre 8 131 et 24 390)
- **20 ms** → ≈ **90 k** vecteurs (interpolation mesurée 91,2 k entre 32 521 et 100 000)

Les tailles 100 k / 500 k sont **synthétiques** : duplication du pool réel + bruit
gaussien σ = 0,05 puis re-normalisation (67 479 / 467 479 vecteurs synthétiques).

## 3. Partition grossière — recall@10 × p95 (N = 100 000, sauf mention)

| Stratégie | nlist | nprobe | recall@10 (32,5 k) | p95 (32,5 k) | recall@10 (100 k) | p95 (100 k) | mémoire |
|---|---|---|---|---|---|---|---|
| k-means IVF | 64 | 4 | 0,959 | 1,15 ms | 0,986 | 15,0 ms | 3 080 o/vec |
| k-means IVF | 64 | 16 | 1,000 | 3,82 ms | 1,000 | 37,5 ms | 3 080 o/vec |
| k-means IVF | 256 | 4 | 0,913 | 0,44 ms | 0,914 | 3,99 ms | 3 080 o/vec |
| k-means IVF | 256 | 16 | 0,986 | 1,14 ms | 0,992 | 11,5 ms | 3 080 o/vec |
| Hilbert (PCA-2D) | 256 | 16 | 0,603 | 0,74 ms | 0,263 | 2,49 ms | 3 084 o/vec |

À 500 k, k-means nlist=256/nprobe=4 dégrade (p95 121 ms, variance forte, listes
déséquilibrées après 12 itérations) : le gain sur l'exact (111 ms) disparaît.

## 4. Mémoire (768-d, f32 = 3 072 o/vecteur)

| Stratégie | o/vecteur | 25 k | 100 k | 500 k |
|---|---|---|---|---|
| f32 brut | 3 072 | 76,8 Mo | 307,2 Mo | 1 536 Mo |
| k-means IVF (+8 o index +centroïdes) | 3 080 | 77 Mo | 308 Mo | 1 540 Mo |
| Hilbert PCA-2D (+12 o ordre+valeur) | 3 084 | 77 Mo | 308 Mo | 1 542 Mo |

Le surcoût d'index des partitions grossières est **négligeable** (< 0,3 %) : la
mémoire est dictée par le stockage f32 des vecteurs eux-mêmes.

## Stratégie recommandée

**Balayage exact f32 multithread (rayon), sans HNSW ni partition.** À la taille
réelle d'obrain (32,5 k), p95 = 7,3 ms mono / 1,6 ms sur 16 threads pour recall
1,0 ; il faut ~22 k vecteurs pour dépasser 5 ms mono et ~90 k pour 20 ms mono.

**Invariants.** Garantit recall@10 = 1,0 (exact, zéro paramètre, zéro approximation) ;
ne garantit ni latence < 5 ms mono au-delà de ~22 k (alors utiliser les threads), ni
compression mémoire. Reconstruction après réouverture : chargement du tableau f32
(mmap), rien d'autre à construire. Insertion : O(D) (normalisation + écriture),
O(1) amorti, aucun re-clustering ; suppression = retrait/tombstone.

Si un accélérateur devient nécessaire (N > 100 k **et** latence < 5 ms exigée),
k-means IVF nlist=256/nprobe=4 (recall 0,91–0,98, ~5× plus rapide à 100 k). L'ordre
de Hilbert sur projection PCA-2D est **à écarter** : recall 0,26 contre 0,99 à budget
égal — la projection 2D perd la structure discriminante des plongements 768-d.

## Non mesuré (et pourquoi)

- HNSW / IVF-PQ : hors périmètre « sans HNSW » (PQ désactivé, non demandé).
- 100 k / 500 k : **synthétiques** (duplication + bruit), pas de vraies données futures.
- Contention : banc mono-requête, pas de requêtes concurrentes sur le même index.
- Indexation incrémentale, persistance disque, recall@1/@100 (seul recall@10 mesuré).
- Hilbert n-D (compact/3D+) : le `hilbert.rs` du substrat est 2D (centralité×degré,
mise en page), pas conçu pour 768-d ; seul un Hilbert sur PCA-2D a été testé.

## Preuve sur les vecteurs réels de PO (ExactVectorIndex, 10/11/2026)

Le banc `vecscan` ci-dessus mesure le *concept* du balayage exact. Cette section
prouve l'**implémentation obrain** elle-même — `ExactVectorIndex`
(`crates/obrain-substrate/src/exact.rs`, branche `feat/scan-exact-dim-native`)
branchée sur le moteur (`create_vector_index`, `vector_search`), dimension native
768, **sans rembourrage** — sur les **vrais** embeddings PO
(`note_embeddings.json` 8 131 × 768, `file_embeddings.json` 24 390 × 768).

Protocole : 200 requêtes tirées de l'échantillon (indice `j·N/200`), vecteur
requêté **exclu** (top-11 demandé puis le soi écarté), recall@10 contre une force
brute qui réplique bit-à-bit la fonction `cosine` de l'index (même accumulation f32,
même ordre de boucle, même départage par id croissant). Build `--release`
(opt-level 3, thin LTO). Machine M4 Max **partagée** : les queues (p99) bougent
avec la charge des autres agents — le recall, lui, est stable.

| N | recall@10 | ordre exact | p50 | p95 | p99 |
|---|---|---|---|---|---|
| 8 131 (notes) | **1,0000** | 200/200 | 0,97 ms | 1,51 ms | 2,08 ms |
| 24 390 (fichiers) | **1,0000** | 200/200 | 2,25 ms | 3,10 ms | 3,48 ms |

- **Recall = 1,0000 aux deux tailles, ordre exact 200/200.** Plus fort que le
  rappel : le top-10 rendu est **bit-identique** à la force brute sur les 200
  requêtes. Aucun plafond 384, aucune quantification f16, aucune projection —
  l'index stocke les vecteurs à 768 en f32 et balaye exhaustivement (rayon).
- Latence cohérente avec le balayage exact de `vecscan` (p95 16T ≈ 0,85 ms à
  8 131 / 1,42 ms à 24 390) : l'écart tient au chemin end-to-end du moteur
  (clone d'`Arc`, mapping `NodeOffset`→`NodeId`, conversion `1,0−cosinus`) et à
  la charge de la machine partagée. La queue p99 reste sous ~3,5 ms à 24 390 en
  régime calme (un pic à 8,5 ms mesuré sous charge).

**Persistance (réouverture par mmap).** L'index est persisté en zone nommée
`substrate.exact.<xxh3(label,property)>` (ici `substrate.exact.1c247ef0371ed324`,
64 o d'en-tête + `[u32; n]` + `n×768×4 o` de f32, CRC32). Fermeture → réouverture :
**recall@10 = 1,0000 identique**. Temps d'ouverture ≈ 6–28 ms ; première recherche
après réouverture (rechargement mmap de la zone + scan) ≈ 3–17 ms — la zone est
relue **directement du mmap**, zéro re-balayage des propriétés (c'est le pendant de
`tier_persist`, étendu à la dimension native).

**Verdict zone orpheline — BUG confirmé.** `drop_vector_index` retire l'entrée du
registre en mémoire et la déclaration durable (`retirer_declaration_substrat`), mais
**ne supprime pas** la zone `substrate.exact.*` : elle reste sur disque après le drop
(mesuré : 1 zone restante). Pire : un `create_vector_index` suivant (même
label/propriété, même dimension) **recharge la zone orpheline périmée** au lieu de
reconstruire — un nœud ajouté après le drop n'est pas retrouvé (top-1 = un ancien id,
pas le nouveau). `drop_vector_index` doit aussi **effacer la zone** (ou invalider son
CRC) pour que la reconstruction reparte des données vivantes.
