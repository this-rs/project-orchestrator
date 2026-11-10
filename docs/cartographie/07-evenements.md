# 07 — Le bus d'événements de remplacement (sortie de NATS)

Analyse en lecture seule. Rien n'a été construit, lancé ni mesuré sur un store vivant : tout ce qui suit est **lu** (`[lu]`) ou **déduit** (`[déduit]`). Chemins :
- `PO` = `/Users/triviere/projects/project-orchestrator/backend/src/`
- `TH` = `/Users/triviere/projects/this-rs/this/src/` (HEAD `3978baf`)
- `OB` = `/Users/triviere/projects/obrain/obrain-db/crates/obrain-engine/src/`

## 0. Cinq constats qui décident

1. **NATS ne donne aucune durabilité à PO aujourd'hui.** Le code n'utilise que du NATS *core* (publish / subscribe / request) : `PO/events/nats.rs:196-299`, `:426`, `:512`. Le mot « JetStream » n'apparaît qu'une fois dans `backend/src`, dans un commentaire de doc (`PO/events/mod.rs:19`). Retirer NATS ne retire donc **aucune** garantie de livraison : il n'y en avait pas. La durabilité à gagner est un gain net.
2. **Le rattrapage existe déjà, deux fois, hors NATS.** (a) `ActivityHub` : `seq` strictement croissant, anneau de 1 000 événements, `replay_since(last_seq)` et `lost_before` (`PO/events/activity.rs:257-278`, capacité `:35`), alimenté par l'unique entonnoir `EventBus::emit` (`PO/events/bus.rs:68-72`). (b) le chat : rejeu de `ChatEventRecord` depuis le graphe au `last_event` du client (`PO/api/ws_chat_handler.rs:8-9`, `:270-291`), les `stream_delta` étant volontairement non persistés (`seq` 0, `:8`).
3. **Le journal durable ordonné existe déjà dans this-rs, adossé à obrain.** `ObrainJournalLog` (`TH/events/log_obrain.rs`) implémente le trait `EventLog` (`append`, `subscribe(consumer, SeekPosition)`, `ack`, `seek`, `last_seq_no`) sur la zone `substrate.journal.<canal>`, sans modifier obrain. Mais il a **trois défauts** (§1.4) qui interdisent de le prendre tel quel.
4. **Le CDC d'obrain n'est pas un bus d'événements applicatifs.** C'est un journal de mutations d'**entités du graphe** (avant/après de propriétés), sans abonnement ni callback, interrogeable par sondage. Il ne porte ni les deltas de chat, ni les commandes (interrupt, cancel). Il convient à l'audit et à `search_delta`, pas à ce flux (§2).
5. **En mono-instance, presque tout survit sans transport.** Les sujets NATS de PO ne servent qu'à joindre *une autre* instance ; le chemin local passe déjà en premier (§3).

## 1. this-rs

### 1.1 `EventBus` — sémantique exacte

| Propriété | Réalité | Source |
|---|---|---|
| Portée | **un processus** ; clonable (Arc) ; aucun transport | `TH/core/events.rs:322-336` |
| Mécanisme | `tokio::sync::broadcast`, capacité passée à `new` (défaut 1024) | `:412-413`, `:513-516` |
| Ordre | total **par canal broadcast** ; l'ordre d'ajout au journal est celui de `publish` (file unique `AppendQueue`, un seul écrivain) | `:338-392`, `:471-488` |
| Perte | un abonné lent reçoit `Lagged(n)` et **perd** n événements ; sans abonné, l'événement est jeté | `:445-449`, `:487` |
| Abonnement | `subscribe()` → `Receiver` ; **les événements antérieurs ne sont pas reçus** | `:494-496` |
| Filtre | aucun dans le bus ; filtrage chez le consommateur (SSE, WS) | `TH/server/exposure/rest/sse.rs:140-` |
| Locataire | `publish_for_tenant` pose `tenant_id` sur l'enveloppe ; le SSE refuse fail-closed | `:462`, `sse.rs:121-138` |
| `revision` | nombre de publications **de ce processus**, indice de resynchronisation, pas une preuve | `:500-511` |
| Journal | `with_event_log` branche un `EventLog` : chaque publication est aussi ajoutée au journal | `:430-433`, `:471-490` |

### 1.2 `FrameworkEvent`

Enum balisée `kind` : `Entity(EntityEvent)`, `Link(LinkEvent)`, `Cognitive(CognitiveSignal)`, `Module(ModuleEvent)`, `GdprErasure{…}` (`TH/core/events.rs:151-172`). Enveloppe `EventEnvelope { id, timestamp, event, seq_no: Option<SeqNo>, tenant_id }` (`:260-272`). C'est un vocabulaire **d'entités génériques** (`entity_type: String`, `data: Value`) : il ne connaît ni `ChatEvent`, ni `RunnerEvent`. PO devrait y plaquer ses événements en `data` — inutile, puisque `CrudEvent`/`ActivityEvent` jouent déjà ce rôle.

### 1.3 Sinks, SSE, WebSocket

- **Sinks** (`TH/events/sinks/`) : trait `Sink::deliver(payload, recipient_id, context_vars)` (`mod.rs:83-101`), registre `SinkRegistry` (`:116`), types `InApp, Push, WebSocket, Webhook, Counter, Feed, Custom` (`:311-352`). Ce sont des **destinations de flux YAML**, pas un bus : ils n'apportent ni diffusion ni rattrapage.
- **Flux YAML** : `FlowRuntime` consomme un `EventLog` (`subscribe` à une position, `ack` après traitement : `TH/events/runtime.rs:200`, `:338`). C'est le seul consommateur qui exploite `SeekPosition::{Beginning, LastAcknowledged, Latest, Sequence}` (`TH/events/types.rs:12-22`).
- **SSE `/events/stream`** (`TH/server/exposure/rest/mod.rs:91`, handler `sse.rs:78-112`) : `BroadcastStream` sur le bus, filtres `kind` / `entity_type` / `event_type` (`sse.rs:54-66`), battement toutes les 30 s. **Pas de `id:` SSE, pas de `Last-Event-ID`, pas de `seq` dans la charge** (`sse.rs:189-210`) : un client qui se reconnecte **ne peut pas** réclamer ce qu'il a manqué. `Lagged` devient un événement `warning` (`:98-107`).
- **WebSocket** (`TH/server/exposure/websocket/`) : protocole `subscribe{filter}` / `unsubscribe` (`protocol.rs:9-12`), un `mpsc::unbounded` par connexion (`manager.rs:214`) — **aucune perte** mais **mémoire non bornée** pour un client lent. Aucun curseur de reprise.
- **Piège** `[lu]` : l'enveloppe est clonée vers la file du journal **avant** que le journal ne lui attribue un `seq_no` (`events.rs:471-490` vs `log_obrain.rs:372-381`) ; la copie diffusée en temps réel porte donc `seq_no = None`. Le temps réel ne transporte pas le curseur du journal.
- `builder.rs:1474-1476` câble par défaut un `InMemoryEventLog` : non durable.

### 1.4 `ObrainJournalLog` : à adapter, pas à prendre tel quel

1. **`append` est O(N)** `[lu]` : `append` appelle `last_seq()` (`log_obrain.rs:372-374`), qui appelle `read_from(0, 1<<20)` (`:309-311`), qui décode **toutes** les lignes de la zone ; `append_entries` le refait (`:218`). Deux balayages complets par événement. Non mesuré ; à corriger par une tête tenue en mémoire (`AtomicU64`) initialisée une fois au démarrage.
2. **Le rattrapage peut sauter un tronçon** `[déduit de la lecture]` : `read_session_journal` remonte les cadres **à rebours** et s'arrête dès `limit` lignes retenues (`OB/journal_zone.rs:307-323`) : avec `after_seq` et `limit`, elle rend les `limit` **plus récentes** lignes, pas les `limit` premières après le curseur. `subscribe` lit par pages de 1 024 et place son curseur sur la dernière ligne reçue (`log_obrain.rs:405-409`) ; au-delà de 1 024 en retard, le milieu est perdu **sans erreur** (`JournalPage.truncated` existe mais n'est pas lue). À rejouer par un test avant tout usage.
3. **Pas de poussée** : `subscribe` sonde toutes les 25 ms (`:415`).
4. **Pas de rétention** : `append_session_journal` n'ajoute que (`OB/journal_zone.rs:92`) ; aucune troncature. Un `write_session_journal` remplace la zone (`:41`).

## 2. Le CDC d'obrain

### 2.1 Ce que c'est

`CdcLog` : table mémoire `HashMap<EntityId, Vec<ChangeEvent>>` + index temporel `by_ts` + persistance optionnelle (`OB/cdc.rs:113-142`). `ChangeEvent { entity_id, kind (Create|Update|Delete), epoch, timestamp (ms), before, after }` (`:94-108`). Alimenté par `CdcObserver`, branché sur `set_mutation_observer` du substrat (`OB/database/mod.rs:821-834`), désactivable par `OBRAIN_DISABLE_CDC` (`:813`). Persisté dans la zone `substrate.cdc_journal` (bincode, en-tête 32 o, trames `len u32 + bytes`) (`OB/cdc_persist.rs:1-30`, `:67`).

### 2.2 API de lecture — **interrogation, pas flux**

| API | Clé | Source |
|---|---|---|
| `history(entity)`, `history_since(entity, epoch)` | **par entité** | `OB/database/admin.rs:959-975`, `cdc.rs:372-396` |
| `changes_between(start_epoch, end_epoch)` | plage d'époques MVCC, tout le monde | `admin.rs:980-985`, `cdc.rs:400-413` |
| `changes_since_timestamp(since_ms, filtre)` | **horloge murale ms**, tout le monde, triée | `cdc.rs:435-` |
| `cold_zone_delta(since_ms)` | idem, par l'index de queue, sans recharger | `cdc_persist.rs:535` |

**Pas de callback, pas d'abonnement, pas de canal** : `cdc.rs` et `cdc_persist.rs` ne contiennent ni `subscribe`, ni `broadcast`, ni `mpsc`, ni `Notify` (recherche faite). On « tail » en **sondant** `changes_since_timestamp(dernier_ts)`. Le seul crochet poussé est `StoreMutationObserver` du substrat, un emplacement **unique** déjà occupé par le CDC (`database/mod.rs:834`).

### 2.3 Fenêtre de rétention

Trois bornes distinctes, à ne pas confondre :

| Borne | Valeur | Rôle | Source |
|---|---|---|---|
| **Zone `cdc_journal` sur disque** | **aucune borne appliquée** | append-only, croissance ×1,5 par sauts depuis 64 Ko ; `compact(skip)` avance `first_off` mais **n'a aucun appelant hors tests** | `cdc_persist.rs:77`, `:344`, appels seulement `:653-778` |
| **Index de queue `cdc_tail`** | **50 000 événements**, anneau, 24 + 50 000 × 20 o ≈ **1 Mo** | répond à `changes_since_timestamp` sans décoder la zone ; si `since` est plus ancien que sa plus vieille entrée → repli sur la voie chaude | `cdc_persist.rs:94-99`, `:535-560` |
| **Vue mémoire rechargée** | **20 000 événements par défaut** (`OBRAIN_CDC_RELOAD_CAP`, plafonné à 100 000) ; payloads `before/after` conservés **uniquement pour les `Update`** | ce que `history*` et `changes_between` voient après redémarrage | `cdc_persist.rs:112`, `:120-125`; `cdc.rs:218-233` |

**Fenêtre de rattrapage effective par horloge** : les 50 000 derniers événements **par la voie rapide**, les 20 000 derniers par la voie lente `[lu]`. **En durée, elle dépend du débit d'écriture du graphe et n'est pas mesurée ici** (pas de store vivant, lecture seule) : à 10 mutations/s, 50 000 événements ≈ 1 h 23 ; à 100/s ≈ 8 min `[calcul, pas mesure]`. Le `ts` est une horloge murale supposée croissante (`cdc_persist.rs:44-47`).

Autres limites : l'identité de déduplication est l'événement entier (`cdc.rs:~590`) ; le journal capte **le graphe**, donc `Update` de propriétés — jamais une commande ni un delta de flux.

## 3. PO : les sujets NATS réellement utilisés

`PO/events/nats.rs` (941 l.) ; transport déjà abstrait (`Subject`, `Message`, `Subscriber` : `PO/events/transport.rs:14-74`).

| Sujet | Usage | Appelants | Mono-instance |
|---|---|---|---|
| `events.crud` | CRUD vers les autres instances (`hybrid.rs:182` pont) | `lib.rs:1539` | **inutile** : bus local (`bus.rs`) |
| `events.graph` | événements de visualisation | `hybrid.rs:246`, `lib.rs:1541` | **inutile** : `subscribe_graph` local (`hybrid.rs:132`) |
| `events.chat.{sid}` | flux de chat vers une instance distante | publié `chat/post_stream.rs:235,350`, `drain.rs:177`, `mirror.rs:1037` ; lu `ws_chat_handler.rs:391,829` | **inutile** : le handler saute NATS quand le broadcast local est actif (`ws_chat_handler.rs:380-391`) |
| `events.chat.{sid}.interrupt` | arrêter un tour | `chat/manager.rs:15065` (pub), `:5187`, `:18683` (sub) | **local d'abord** (`outcome.delivered`, `:15055-15062`) ; NATS = relais vers le propriétaire distant |
| `events.chat.{sid}.cancel_tools` | annuler les outils | sub `manager.rs:15545`, `:18707` | `[déduit]` même schéma ; non vérifié ligne à ligne |
| `events.chat.{sid}.cancel_task` | `request_cancel` (req/rep, 10 s) | `chat/cancel_relay.rs:280` | `[déduit]` local d'abord |
| `events.chat.{sid}.snapshot` | req/rep : état d'un tour en cours pour une instance qui rejoint | `nats.rs:314`, `:363` | **inutile** (une seule instance détient la session) |
| `rpc.chat.{sid}.send` | req/rep : envoyer un message à l'instance propriétaire | `manager.rs:5394`, `:18800` | **inutile** (propriétaire = soi) |

Conclusion `[lu + déduit]` : **zéro sujet n'est nécessaire à une topologie mono-instance** ; la machinerie NATS est un *relais vers un propriétaire distant*. Le contrat à conserver est `trait EventEmitter` (`PO/events/types.rs:192-`) : `emit(CrudEvent)`, `emit_graph`, `transport_state()`. Le mode « local seul, zéro surcoût » existe déjà : `HybridEmitter::new(local_bus)` (`PO/events/hybrid.rs:47`).

## 4. Conception minimale

**Principe** : ne pas adopter l'`EventBus` de this-rs (il dupliquerait `EventBus` + `ActivityHub` de PO sans leur rattrapage) ; ne pas bâtir sur le CDC (graphe, pas flux). **Ajouter une seule chose : un journal durable derrière l'`ActivityHub` existant**, sur la primitive que this-rs utilise déjà (`obrain_engine::journal_zone`).

### 4.1 Types

```rust
// backend/src/events/durable.rs  (nouveau, ~150 lignes)
pub struct DurableActivity {
    hub: Arc<ActivityHub>,                       // inchangé : seq, anneau 1000, broadcast
    journal: Arc<ActivityJournal>,
    tx: mpsc::UnboundedSender<ActivityEvent>,    // file unique, un seul écrivain
}
pub struct ActivityJournal {
    substrate: Arc<SubstrateStore>,
    channel: String,                             // "po.activity"
    head: AtomicU64,                             // tête EN MÉMOIRE (corrige O(N)/append)
}
impl ActivityJournal {
    fn open(..) -> Self;                         // head = scan unique au démarrage
    fn append(&self, ev: &ActivityEvent);        // JournalEntry{seq: ev.seq, kind:"activity", payload}
    fn read_after(&self, after_seq: u64, max: usize) -> JournalPage; // FORWARD, les max PREMIÈRES > curseur
}
```

`JournalEntry` et le format de ligne (`seq` au premier niveau) sont repris de `TH/events/log_obrain.rs:60-130`.

### 4.2 Flux

```
emit(CrudEvent)                              [trait EventEmitter, inchangé]
  └─ EventBus::emit (bus.rs:68)
       ├─ ActivityHub::publish → seq, anneau, broadcast   [temps réel : (a)]
       │       └─ tx.send(ActivityEvent)  ──▶ writer task ──▶ journal.append (seq imposé)
       └─ broadcast<CrudEvent>                             [consommateurs existants]

/ws/activity?last_seq=N                       [rattrapage : (b)]
  ├─ hub.lost_before(N) == 0 → hub.replay_since(N)         (anneau, comme aujourd'hui)
  └─ sinon → journal.read_after(N, 1000) par pages jusqu'à la tête, PUIS l'anneau, PUIS le live
     (frontière = dernier seq rejoué ; dédoublonnage par seq)
```

Deux points sans lesquels ce serait faux :
1. **Le `seq` doit survivre au redémarrage.** Aujourd'hui il est « strictement croissant dans un processus » (`activity.rs`, doc de `seq`) : il repart de zéro au boot, donc `last_seq` d'avant redémarrage est **supérieur** au nouveau `seq` et le rejeu est faux. Il faut amorcer le compteur à `journal.head` à l'ouverture. La règle « seq imposé ; déjà présent ou périmé ⇒ ignoré » de `append_entries` (`TH/events/log_obrain.rs:211-228`) rend l'écriture idempotente.
2. **La lecture de reprise doit être *avant*, pas *les plus récentes*** (§1.4-2). Écrire `read_after` pour qu'elle rende les `max` premières lignes `> N`, et la couvrir par un test à 3 000 événements de retard.

### 4.3 (c) Aucun consommateur à réécrire

- `trait EventEmitter` : inchangé (`types.rs:192`). `events/types.rs` : inchangé ; `ActivityEvent` reçoit **zéro** champ nouveau (il porte déjà `seq`, `activity.rs:104-130`).
- Supprimer : `with_nats`/ponts (`hybrid.rs:60`, `:182`, `:246`), `nats.rs` (941 l.), `nats_broker_test.rs` (256 l.), les 8 sujets (§3), `state.nats_emitter` côté handlers. Les appels `if let Some(nats) = …` deviennent des branches mortes retirées une à une.
- Ce qui n'est **pas** diffusé par le journal : les `stream_delta` (déjà non persistés, `ws_chat_handler.rs:8`) et le chat (déjà rejoué depuis le graphe, `:270-291`).

### 4.4 Rétention du journal (à décider, pas à subir)

Aucune troncature n'existe (`journal_zone.rs:92`). Proposition : au démarrage, si `head − premier_seq > 100 000`, réécrire la zone avec `write_session_journal` en ne gardant que les 100 000 dernières lignes (le `seq` est dans la ligne, donc conservé). Mesurer la taille moyenne d'un `ActivityEvent` avant de fixer ce nombre : **non mesuré**.

## 5. Ce qui est perdu par rapport à NATS

| Perdu | Gravité en mono-instance | Remarque |
|---|---|---|
| **Diffusion inter-instances** (CRUD, graph, chat) | nulle | Le jour où une 2ᵉ instance existe, il faut un transport ; `transport.rs` est déjà l'abstraction à ré-implémenter. Le journal partagé ne le remplace pas (pas de poussée, sondage 25 ms). |
| **Requête/réponse avec `NoResponders`** (`RelayFailure::NoResponders`, `nats.rs:55`) : savoir *tout de suite* qu'aucune instance ne détient la session | nulle | Remplacé par un appel local synchrone. |
| **Routage par propriétaire** (`rpc.chat.{sid}.send`, `cancel_task`) | nulle | Idem. |
| **Clients externes sur le courtier** (tout processus ou hôte abonné à `events.>`) | **réelle si elle existe** | À inventorier : la recherche n'a pas été faite hors de `backend/src`. |
| **Jonctions de sujets / jokers, files de consommateurs** | nulle | Aucun usage constaté. |
| **Isolation de panne du courtier** : le bus local ne dépend plus d'un service externe | *gagné* | — |
| **Dédoublonnage ponts** (fenêtre bornée, `hybrid.rs:~201`) | disparaît avec les ponts | — |

**Gagné** : rattrapage au-delà de 1 000 événements et au-delà d'un redémarrage (aujourd'hui impossible avec l'anneau seul) ; aucune dépendance de service.

**Non résolu par cette conception** : la latence de poussée du journal pour un lecteur *externe* (sondage) ; la diffusion SSE/WS de this-rs sans `Last-Event-ID` (§1.3) si PO exposait ces routes — PO a ses propres `/ws/activity` et `/ws/chat`, donc non concerné.

## 6. Ce qui reste à prouver avant de coder

1. Test rouge/vert : `read_session_journal(after_seq, limit)` rend-elle bien les `limit` **dernières** lignes ? (`journal_zone.rs:307-323`) — si oui, `read_after` doit paginer par fenêtres `before_seq`.
2. Coût de `append` du journal sur 100 000 lignes (le O(N) de §1.4-1).
3. Taille moyenne d'un `ActivityEvent` sérialisé ; débit d'événements réel de PO (fixe la fenêtre en durée).
4. Inventaire des abonnés NATS externes à `backend/src` (scripts, mobile, nexus).
5. Les trois sujets marqués `[déduit]` (§3) : vérifier le chemin local avant de supprimer la branche NATS.
