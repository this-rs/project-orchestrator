# Rétention du CDC d'obrain — ce que vaut vraiment la fenêtre

Mesure du 10/11/2026, branche `mesure/cdc-retention` de `obrain-db`, base `38faeac4`.
Méthode : lecture du code + **exécution** (5 tests, `crates/obrain-engine/tests/mesure_cdc_retention.rs`,
tous verts). Aucun chiffre n'est une impression : chacun est imprimé par un test rejouable.

Question posée par le producteur : le Project Orchestrator veut remplacer NATS par le CDC
d'obrain (`history` / `history_since` / `search_delta`) pour qu'un client qui se reconnecte
rattrape ce qu'il a manqué. **Cette fenêtre suffit-elle ?**

---

## 0. Verdict, en une phrase

**Non.** Le journal ne tronque rien sur disque, mais la **vue interrogeable** est bornée à
**20 000 événements** (plancher, après tout redémarrage) et **50 000** (plafond, et seulement
si le curseur du client tombe encore dans l'index de queue) — soit **33 minutes** dans le cas
ordinaire et **1 h 23** dans le meilleur cas, à 10 mutations/s. Au-delà, la lecture ne rend
**aucune erreur** : elle rend silencieusement moins d'événements. PO doit tenir son propre
journal.

---

## 1. La rétention réelle, et où elle est écrite

Il y a **trois** bornes distinctes, et aucune n'est une borne d'âge.

| Borne | Constante | Lieu exact | Nature |
|---|---|---|---|
| Rechargement au démarrage (« vue chaude ») | `20_000` par défaut | `crates/obrain-engine/src/cdc_persist.rs:124` (`map_or(20_000, …)`) | **plancher de la fenêtre utile** |
| Rechargement, plafond absolu | `CDC_RELOAD_CAP = 100_000` | `crates/obrain-engine/src/cdc_persist.rs:112` | surchargeable par `OBRAIN_CDC_RELOAD_CAP`, borné à 100 k |
| Index de queue (« vue froide ») | `TAIL_CAP = 50_000` | `crates/obrain-engine/src/cdc_persist.rs:94` | **anneau** : le plus récent écrase le plus ancien |
| Journal lui-même (`substrate.cdc_journal`) | **aucune** | `compact()` à `cdc_persist.rs:342` | append-only non borné |

**Le point qui décide** : `CdcPersist::compact(skip)` — la seule fonction qui avance le pointeur
de fenêtre `first_off` (`cdc_persist.rs:342-381`) — **n'a aucun appelant en production**. Vérifié
par `grep -rn "\.compact(" --include=*.rs . | grep -i "cdc\|persist"` : uniquement
`cdc_persist.rs:648` et `:774`, qui sont deux tests unitaires. Rien dans `crates/obrain-*`,
rien dans `packages/`.

Conséquence : le fichier `substrate.cdc_journal` **grossit indéfiniment**. Aucune troncature par
nombre, par taille, ni par âge. Le commentaire du module le dit d'ailleurs sans détour
(`cdc_persist.rs:148-152`) : « la zone garde TOUT (append-only) » — le plafond ne borne que la
vue mémoire.

`admin.rs` n'expose que trois lectures, sans paramètre de rétention :
`history` (`database/admin.rs:960`), `history_since` (`:969`), `changes_between` (`:981`).
Aucune purge, aucun `compact`, aucune configuration de fenêtre.

---

## 2. Ce qui est mesuré à l'exécution

### 2.1 Coût et taille

| Mesure | Valeur |
|---|---|
| Événement `Task` réaliste (6 propriétés) | **29,9 o/événement** (746 496 o pour 25 000) |
| Événement minimal (`A` + 1 entier) | 22,1 o/événement (221 184 o pour 10 000) |
| Journal à 55 000 événements | 1 679 616 o (≈ 30,5 o/év.) |
| `substrate.cdc_tail` | **1 000 024 o, fixe**, dès la première écriture (`50 000 × 20 + 24`) |
| Débit avec CDC actif | 62–91 µs/mutation (≈ 11 000–16 000 mut/s) — le CDC n'est pas le facteur limitant |

### 2.2 Temps de lecture

| Appel | Fenêtre | Durée |
|---|---|---|
| `changes_since_timestamp(0)` | 10 000 (process chaud) | **17,9 ms** |
| `changes_since_timestamp(0)` | 20 000 (après redémarrage, rechargement) | **35–49 ms** |
| `changes_since_timestamp(ts_min)` | 30 000 (voie froide, anneau) | **47,4 ms** |
| `CdcPersist::load(20_000)` brut | 20 000 frames décodées | **19,2 ms** |

### 2.3 Ce qui se passe au-delà de la limite — **rien, et c'est le problème**

Trois expériences, trois instances neuves (donc la voie froide est tentée avant tout rechargement :
c'est exactement le premier appel d'un PO qui vient de redémarrer).

| Base | Curseur du client | Événements rendus | Attendu | Verdict |
|---|---|---|---|---|
| 10 000 | `0` | 10 000 | 10 000 | complet |
| 25 000 | `0` | **20 000** | 25 000 | **5 000 perdus, sans erreur** |
| 25 000 | ts du 10 000ᵉ | 15 005 | 15 001 | complet (dans la fenêtre chaude) |
| 30 000 | ts du 1ᵉʳ | 30 000 | 30 000 | complet (anneau de 50 k couvre) |
| 55 000 | `0` | **20 000** | 55 000 | **35 000 perdus, sans erreur** |
| 55 000 | ts du 1ᵉʳ | **20 000** | 55 000 | **35 000 perdus** (curseur hors anneau) |

Le journal, lui, garde bien tout : `CdcPersist::count()` rend **55 000** et `load(0)` rend
**55 000 événements, ts 1..55 000**. C'est la **vue** qui est bornée, pas le stockage.

**Point de justesse à retenir** : un client qui revient avec un curseur trop vieux reçoit un
diff **tronqué par le bas et silencieux**. Aucun code d'erreur, aucun drapeau « fenêtre
dépassée ». Un consommateur qui applique ce diff se croit à jour et ne l'est pas.

**Second point de justesse** : même dans la fenêtre, le rejeu n'est pas fidèle.
`retenir_leger` (`crates/obrain-engine/src/cdc.rs:687-692`) met `before` et `after` à `None`
pour tout événement qui n'est pas `Update` — un `Create` rejoué depuis la zone perd son
instantané. Seuls les `Update` conservent `before`/`after`.

---

## 3. L'extrapolation : combien de temps la fenêtre couvre

Durée = nombre d'événements ÷ taux de mutation. Les deux bornes mesurées encadrent la réalité.

| Taux de mutation | 20 000 événements (plancher) | 50 000 événements (plafond) |
|---|---|---|
| 1 /s | 5 h 33 min | 13 h 53 min |
| **10 /s** | **33 min 20 s** | **1 h 23 min 20 s** |
| 100 /s | 3 min 20 s | 8 min 20 s |
| 1 000 /s | 20 s | 50 s |

Le chiffre « ~50 000 événements » du brief est donc exact — c'est `TAIL_CAP`
(`cdc_persist.rs:94`) — et son arithmétique aussi : 1 h 23 à 10 mutations/s. Mais c'est le
**meilleur** cas. Le cas ordinaire est `cdc_reload_cap()`, **20 000**, soit 33 minutes : il
suffit d'un seul redémarrage de PO pour tomber du plafond au plancher.

**Ce que cette absence de rétention coûte en disque** (l'autre face de la médaille) : à
29,9 o/événement, 10 mutations/s produisent ≈ 26 Mo/jour, **≈ 0,77 Go/mois** de journal que
rien ne réclame. À 100 mutations/s, ≈ 7,8 Go/mois. Le fichier n'est jamais purgé.

---

## 4. Peut-on suivre le journal en continu, ou seulement interroger ?

**Seulement interroger.** Il n'existe aucun flux, aucun abonnement, aucun callback côté CDC.

L'API publique de `CdcLog` (`crates/obrain-engine/src/cdc.rs`) est intégralement en lecture
tirée : `history` (`:372`), `history_since` (`:383`), `changes_between` (`:400`),
`changes_since_timestamp` (`:435`), `event_count` (`:558`). Plus `search_delta`
(`database/search_delta.rs:39`). Rien d'autre.

`CdcObserver` (`crates/obrain-engine/src/database/cdc_observer.rs`) n'est **pas** un point
d'écoute pour un consommateur : c'est le chemin d'écriture interne (mutation substrate →
`CdcLog` → `CdcPersist`). Il n'expose pas de souscription.

**`set_post_commit_hook` existe** : `crates/obrain-engine/src/session/mod.rs:584`
(`pub fn set_post_commit_hook(&mut self, hook: Arc<dyn Fn(&dyn GraphStoreMut) + Send + Sync>)`),
invoqué en `session/mod.rs:3806`. **Il n'a aucun appelant** — vérifié par
`grep -rn "set_post_commit_hook" --include=*.rs .` : la définition, et rien d'autre. C'est une
couture posée, jamais branchée, et de toute façon elle ne concerne que les sessions de
l'engine, pas le CDC persistant.

Conséquence directe pour PO : **pas de `watch`, pas de `subscribe`, pas de notification au
commit**. Le seul mode possible est le **sondage périodique** de `changes_since_timestamp`
avec le dernier timestamp vu. Et ce sondage a le défaut du §2.3 : si le sondeur prend du
retard au-delà de la fenêtre (32 s d'arrêt à 1 000 mut/s, 33 min à 10 mut/s), il perd des
événements sans le savoir.

---

## 5. Verdict pour le producteur

**Le CDC ne suffit pas pour une reconnexion de plus de ~33 minutes** (10 mutations/s, cas
ordinaire après un redémarrage), et il ne suffit **pas de façon fiable** au-delà de ~1 h 23
même sans redémarrage.

Trois raisons, dans l'ordre de gravité :

1. **La perte est silencieuse.** Un curseur trop vieux rend un diff tronqué par le bas, sans
   erreur ni drapeau. Un repli sur le CDC seul fabrique des clients qui se croient à jour.
   C'est disqualifiant à soi seul : on ne peut pas bâtir une reprise sur une source dont le
   mode de défaillance est « moins de données, l'air de rien ».
2. **Le plancher est 20 000, pas 50 000** (`cdc_persist.rs:124`). Tout redémarrage de PO
   ramène la fenêtre à 33 minutes à 10 mut/s — et à 3 min 20 à 100 mut/s.
3. **Aucune borne d'âge, donc aucune promesse tenable.** Ce qui est borné est un *nombre*
   d'événements, pas une *durée* : la fenêtre rétrécit quand la charge monte, c'est-à-dire
   exactement quand un client a le plus de choses à rattraper.

**Ce qu'il faut faire côté PO** : persister son propre journal d'événements (le diff qu'il
sert à ses clients), avec une rétention en **durée** et un **marqueur explicite de trou**
quand un client revient avec un curseur antérieur à sa fenêtre — jamais un diff partiel
présenté comme complet. Le CDC d'obrain reste utilisable comme **source de capture** (il est
correct, peu coûteux — 30 o/événement, 60–90 µs/mutation — et déjà branché sur toutes les
mutations), pas comme **mémoire de reprise**.

Deux réserves de contrat à ne pas oublier si le CDC est utilisé comme source :
`before`/`after` sont effacés pour les `Create`/`Delete` au rejeu (`cdc.rs:687`), et il
n'existe aucun mécanisme de notification — donc aucune reprise événementielle, seulement du
sondage.

---

## 6. Rejouer

```sh
cd <worktree mesure/cdc-retention>
cargo test -p obrain-engine --features cdc --test mesure_cdc_retention \
  -- --nocapture --test-threads=1
```

5 tests, ~14 s après compilation, tous verts. Ils impriment chaque chiffre cité ici.
Dernier `git status --porcelain` : `?? crates/obrain-engine/tests/mesure_cdc_retention.rs`
(le fichier de mesure, non commité ; aucun autre fichier touché, aucun push).
