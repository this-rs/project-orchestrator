//! `SearchStore` on obrain — the 27 methods of the Meilisearch contract, served by
//! the embedded BM25 text index of obrain. Nothing here is refused wholesale; the
//! only refusal is a query the index cannot answer faithfully (see "Refused").
//!
//! # Storage
//!
//! A document is one node, labelled `RechercheCode`, `RechercheDecision` or
//! `RechercheNote`, with the properties `id`, `doc` (the whole document as JSON:
//! the SINGLE source of truth), `texte` (the searchable fields of the document,
//! concatenated: what the text index of obrain tokenizes) and `_lot` (the batch
//! that created it, see "Atomicity"). `texte` is a pure function of `doc`, so open
//! recomputes it and rewrites a node whose `texte` disagrees (an update cut between
//! the two writes): the repair is counted in `RapportOuverture::recherches_reparees`.
//!
//! The searchable fields are Meilisearch's own (`configure_*_index`): code =
//! `symbols`, `docstrings`, `signatures`, `path`, `imports`; decisions =
//! `description`, `rationale`, `tags`; notes = `content`, `tags`, `scope_path`,
//! `anchor_entities`.
//!
//! # The text index: built at open, never trusted from disk
//!
//! obrain persists a text index and reloads it at the next open when a fingerprint
//! of the store (WAL offset, live node count) matches; it re-persists it at close
//! when an epoch says nothing was written since. Neither signal is proven, for this
//! adapter's writes through the direct API, to change on every write that matters
//! (a full flush resets the WAL offset; the direct API does not advance the epoch).
//! So the adapter never serves a persisted index: at open, BEFORE any writer can
//! run, it drops whatever obrain loaded and builds each index from the documents
//! (`create_text_index`). From then on obrain maintains it on every write (its
//! write hooks), and every write of this adapter holds the writer lock. If an index
//! is missing at search time anyway, it is built under the WRITER lock, so no write
//! can slip between the scan and the hooks.
//!
//! # Filters: AFTER the search, by over-fetching, never wrong
//!
//! obrain's `text_search(label, property, query, k)` takes no filter. A filter
//! (project, language, path prefix, note type, status, importance, task) is applied
//! to the hits afterwards. The search asks for `max(32, 4 × limit)` hits and
//! multiplies `k` by 4 until `limit` documents pass the filter or the index returns
//! fewer hits than asked (every document matching the query has been seen). The
//! number of DOCUMENTS of the kind is not a bound: the index also holds nodes that
//! are not served (a batch in progress, a quarantined node), which would otherwise
//! use up the budget and leave a silent empty answer.
//!
//! The filter is checked twice: on the in-memory copy of the filterable fields
//! (cheap, before reading the node) and on the document READ, so a document being
//! rewritten is never returned with fields that contradict the filter.
//!
//! # Score and order
//!
//! `SearchHit::score` is in `[0, 1]` and the hits are returned in decreasing score:
//! `score = (1 − δ) × coverage + δ × b/(1 + b)` with `coverage` the fraction of the
//! query's terms the document carries, `b` its raw BM25 score and `δ = 10⁻³`. The
//! coverage decides (as Meilisearch's "words" rule ranks documents matching more of
//! the query first); BM25 breaks ties. The score is ABSOLUTE for one query in one
//! index (BM25 is computed on the same index statistics whatever the filter), so
//! lists of several projects merged by score (`api/code_handlers.rs`) keep one
//! order, and a duplicate of the query scores ≥ 0.999 (`chat/feedback.rs`
//! thresholds at 0.85 on the first hit). The order is computed over the documents
//! of the window fetched above (at least `max(32, 4 × limit)` BM25 hits that pass
//! the filter, or all of them): a document with full coverage but a BM25 rank past
//! that window is not considered. Named, not hidden.
//!
//! # Differences with Meilisearch, all named
//!
//! - **Matching.** BM25 over whole tokens (lowercased, split on non-alphanumerics,
//!   English stop words and tokens shorter than 2 bytes dropped), any query token
//!   suffices (OR). Meilisearch also matches by prefix and tolerates typos.
//! - **Ranking.** See "Score and order"; Meilisearch also ranks by attribute order
//!   and proximity. The score is not Meilisearch's ranking score.
//! - **Empty query.** A query with no alphanumeric character (`""`, `"*"`) is
//!   Meilisearch's placeholder search: documents that pass the filters, here by
//!   increasing `id`, with score 1.0.
//! - **Filter comparison** is case-insensitive (Meilisearch compares filter values
//!   without case); `STARTS WITH` too.
//! - **Notes**: when no `status` filter is given, only `active` and `needs_review`
//!   notes are searched (Meilisearch adapter, `search_notes_with_scores`).
//! - **`delete_code(path)`** deletes the document whose `id` is `path_to_id(path)`,
//!   as the Meilisearch adapter does (not every document whose `path` field equals
//!   `path`, as the mock does).
//! - **`delete_orphan_code_documents`**: `project_id` empty OR `project_slug` empty
//!   (the Meilisearch filter; the mock only tests `project_id`).
//! - **`search_decisions_in_projects(&[])`** matches no document (`IN []`), and
//!   answers without reading the index.
//! - **`is_indexing`** is always false: a write is searchable when it returns.
//! - Dates: no date is decoded here. A document is stored and returned verbatim as
//!   JSON (`DecisionDocument::timestamp` is the string the caller wrote,
//!   `NoteDocument::created_at` an integer), so there is no date for the single
//!   decoder (`date_valeur`) to read.
//!
//! # Refused
//!
//! A query whose characters include alphanumerics but whose tokens are ALL dropped
//! by the tokenizer (only English stop words — `"the"`, `"how are you"` — or
//! one-character words) is refused by name (`SearchStore::search_*`), not answered
//! empty: Meilisearch matches those words, this index cannot.
//!
//! # Quarantine
//!
//! A node of a search label whose `doc` cannot be read, or whose id is already
//! taken, is quarantined at open: never served. It is NOT left in the way: the
//! deletions remove it (by id when its `id` is readable; by project, task, orphan
//! or "all" according to what of its JSON can still be read — a node whose JSON is
//! unreadable has no readable project, so it is an orphan and goes with
//! `delete_all_code`), and indexing its id again replaces it. The search index is
//! derived data: the caller re-supplies the document.
//!
//! # Durability and atomicity
//!
//! One writer section and ONE fsync per call, through the same `Ecriture` as every
//! domain. A call that creates documents first creates ONE lot node
//! (`RechercheLot`, with an `id`), then each new document carrying `_lot = <that
//! id>`, and commits by DELETING the lot node: one engine write. At open, every lot
//! node still there is an uncommitted batch: its documents are deleted with it (in
//! `brouillons_supprimes`). A crash therefore leaves all of a batch's new documents
//! or none, never a prefix. The documents a batch REPLACES are rewritten in place
//! before the commit point: a crash may leave some replaced and some not (indexing
//! is idempotent; the sync that calls it runs again). A bulk deletion
//! (`delete_*_for_project`, `delete_all_code`, `delete_orphan_code_documents`)
//! deletes node by node and may stop half-way on a crash; running it again
//! finishes it.
//!
//! # Cost
//!
//! Lookups by id (`delete_code`, `delete_decision`, `delete_note`,
//! `update_note_status`, the replacement check of an indexing): O(log N) on the
//! in-memory map, plus O(Q) over the quarantined nodes of the kind (empty on a
//! healthy base). `get_*_stats`: O(1). `search_decisions_in_projects(&[])`: O(1).
//! A search: O(k) hits for the window, one node read per candidate.
//!
//! Proportional to the documents of the kind, and MEASURED (see the PR):
//! - open: one scan per label to rebuild the in-memory map, and one build of each
//!   text index (see above);
//! - the placeholder search WITH a selective filter (it walks the ids in order
//!   until `limit` pass; without a filter it stops after `limit`);
//! - `delete_*_for_project`, `delete_decisions_for_task`,
//!   `delete_orphan_code_documents`, `delete_all_code`: a pass over the in-memory
//!   copies (no index by project or task).
//!
//! `ProjectStore::delete_project` refuses while the base holds any node outside the
//! projects and plans (`project.rs`): these documents are such nodes, listed under
//! « recherche » in its refusal. `Orchestrator::delete_project` therefore deletes
//! the code documents AFTER the graph deletion succeeded, never before.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use obrain_common::types::{NodeId, Value};
use obrain_core::index::text::{SimpleTokenizer, Tokenizer};
use obrain_engine::ObrainDB;
use serde::{de::DeserializeOwned, Serialize};

use super::{Ecriture, Interne, NonPorte, ObrainGraphStore, RapportOuverture};
use crate::meilisearch::indexes::{
    CodeDocument, DecisionDocument, IndexStats, NoteDocument, SearchHit,
};
use crate::meilisearch::MeiliClient;
use crate::meilisearch::SearchStore;

/// Legacy draft marker (a node carrying it is a draft at open).
const BROUILLON: &str = "_brouillon";
const LOT: &str = "_lot";
const DOC: &str = "doc";
const TEXTE: &str = "texte";
const ID: &str = "id";
const LABEL_LOT: &str = "RechercheLot";

/// The labels this domain writes (`LABELS_PAR_DOMAINE` of `project.rs`).
pub(super) const LABELS: &[&str] = &[
    "RechercheCode",
    "RechercheDecision",
    "RechercheNote",
    LABEL_LOT,
];

/// The statuses a note search covers when the caller gives none.
const STATUTS_PAR_DEFAUT: &[&str] = &["active", "needs_review"];

/// The weight of BM25 in the score (see the module doc).
const DELTA: f64 = 1e-3;

/// The three corpora.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Genre {
    Code,
    Decision,
    Note,
}

const GENRES: [Genre; 3] = [Genre::Code, Genre::Decision, Genre::Note];

impl Genre {
    fn label(self) -> &'static str {
        match self {
            Genre::Code => LABELS[0],
            Genre::Decision => LABELS[1],
            Genre::Note => LABELS[2],
        }
    }

    /// The filterable fields of a document of this kind, from its JSON as a
    /// `Value`: what can still be read of a node whose document does not decode.
    fn cles_tolerantes(self, v: &serde_json::Value) -> Cles {
        let champ = |k: &str| v.get(k).and_then(|x| x.as_str()).map(str::to_string);
        match self {
            Genre::Code => Cles {
                slug: champ("project_slug"),
                projet_id: champ("project_id"),
                a: champ("language"),
                b: champ("path"),
                c: None,
            },
            Genre::Decision => Cles {
                slug: champ("project_slug"),
                projet_id: champ("project_id"),
                a: champ("task_id"),
                b: None,
                c: None,
            },
            Genre::Note => Cles {
                slug: champ("project_slug"),
                projet_id: champ("project_id"),
                a: champ("note_type"),
                b: champ("status"),
                c: champ("importance"),
            },
        }
    }
}

/// The filterable fields of a document, copied small. `None` is a field the
/// document does not carry (a decision without task or project).
#[derive(Clone, Default, Debug)]
pub(super) struct Cles {
    slug: Option<String>,
    projet_id: Option<String>,
    /// code: language · decision: task id · note: note type.
    a: Option<String>,
    /// code: path · note: status.
    b: Option<String>,
    /// note: importance.
    c: Option<String>,
}

/// A searchable document.
trait Document: Serialize + DeserializeOwned + Send + 'static {
    const GENRE: Genre;
    fn id(&self) -> &str;
    /// The searchable fields, concatenated.
    fn texte(&self) -> String;
    fn cles(&self) -> Cles;
}

fn joindre(parts: &[&str], listes: &[&[String]]) -> String {
    let mut s = parts.join("\n");
    for l in listes {
        for e in *l {
            s.push('\n');
            s.push_str(e);
        }
    }
    s
}

impl Document for CodeDocument {
    const GENRE: Genre = Genre::Code;
    fn id(&self) -> &str {
        &self.id
    }
    fn texte(&self) -> String {
        joindre(
            &[&self.docstrings, &self.path],
            &[&self.symbols, &self.signatures, &self.imports],
        )
    }
    fn cles(&self) -> Cles {
        Cles {
            slug: Some(self.project_slug.clone()),
            projet_id: Some(self.project_id.clone()),
            a: Some(self.language.clone()),
            b: Some(self.path.clone()),
            c: None,
        }
    }
}

impl Document for DecisionDocument {
    const GENRE: Genre = Genre::Decision;
    fn id(&self) -> &str {
        &self.id
    }
    fn texte(&self) -> String {
        joindre(&[&self.description, &self.rationale], &[&self.tags])
    }
    fn cles(&self) -> Cles {
        Cles {
            slug: self.project_slug.clone(),
            projet_id: self.project_id.clone(),
            a: self.task_id.clone(),
            b: None,
            c: None,
        }
    }
}

impl Document for NoteDocument {
    const GENRE: Genre = Genre::Note;
    fn id(&self) -> &str {
        &self.id
    }
    fn texte(&self) -> String {
        joindre(
            &[&self.content, &self.scope_path],
            &[&self.tags, &self.anchor_entities],
        )
    }
    fn cles(&self) -> Cles {
        Cles {
            slug: Some(self.project_slug.clone()),
            projet_id: Some(self.project_id.clone()),
            a: Some(self.note_type.clone()),
            b: Some(self.status.clone()),
            c: Some(self.importance.clone()),
        }
    }
}

/// Meilisearch compares filter values without case.
fn egal(champ: &Option<String>, voulu: &str) -> bool {
    champ
        .as_deref()
        .is_some_and(|c| c.to_lowercase() == voulu.to_lowercase())
}

fn vide(champ: &Option<String>) -> bool {
    champ.as_deref().is_none_or(str::is_empty)
}

struct Entree {
    noeud: NodeId,
    cles: Cles,
}

/// A node of a search label that open could not serve.
struct Quarantaine {
    noeud: NodeId,
    /// Its `id` property, when readable.
    id: Option<String>,
    /// What of its JSON could be read, when it is JSON at all.
    cles: Option<Cles>,
}

/// The documents of one kind: by id (ordered, for the placeholder search), and by
/// node (to name a hit); and the quarantined nodes.
#[derive(Default)]
struct Corpus {
    docs: BTreeMap<String, Entree>,
    par_noeud: HashMap<NodeId, String>,
    quarantaine: Vec<Quarantaine>,
}

impl Corpus {
    fn inscrire(&mut self, id: String, e: Entree) {
        if let Some(ancien) = self.docs.get(&id) {
            self.par_noeud.remove(&ancien.noeud);
        }
        self.par_noeud.insert(e.noeud, id.clone());
        self.docs.insert(id, e);
    }

    fn retirer(&mut self, id: &str) {
        if let Some(e) = self.docs.remove(id) {
            self.par_noeud.remove(&e.noeud);
        }
    }

    fn retirer_quarantaine(&mut self, noeuds: &HashSet<NodeId>) {
        self.quarantaine.retain(|q| !noeuds.contains(&q.noeud));
    }

    /// The quarantined nodes carrying this id.
    fn quarantaine_de<'a>(&'a self, id: &'a str) -> impl Iterator<Item = NodeId> + 'a {
        self.quarantaine
            .iter()
            .filter(move |q| q.id.as_deref() == Some(id))
            .map(|q| q.noeud)
    }
}

/// The indexes of the three corpora.
#[derive(Default)]
pub(super) struct IndexRecherche {
    code: Corpus,
    decisions: Corpus,
    notes: Corpus,
}

impl IndexRecherche {
    fn corpus(&self, g: Genre) -> &Corpus {
        match g {
            Genre::Code => &self.code,
            Genre::Decision => &self.decisions,
            Genre::Note => &self.notes,
        }
    }

    fn corpus_mut(&mut self, g: Genre) -> &mut Corpus {
        match g {
            Genre::Code => &mut self.code,
            Genre::Decision => &mut self.decisions,
            Genre::Note => &mut self.notes,
        }
    }

    /// Rebuild the maps from the base, finish or undo what a crash left, and build
    /// the text indexes from the documents (module doc). Runs at open, before any
    /// writer.
    pub(super) fn reconstruire(db: &ObrainDB, rapport: &mut RapportOuverture) -> Result<Self> {
        let store = db.store();
        // Every lot node still there is a batch that never reached its commit point.
        let mut lots_ouverts: HashSet<String> = HashSet::new();
        let noeuds_lots = store.nodes_by_label(LABEL_LOT);
        for noeud in &noeuds_lots {
            if let Some(id) = db.get_node(*noeud).and_then(|n| {
                n.get_property(ID)
                    .and_then(|v| v.as_str().map(str::to_string))
            }) {
                lots_ouverts.insert(id);
            }
        }
        let mut index = Self::default();
        reconstruire_genre::<CodeDocument>(db, rapport, &lots_ouverts, &mut index.code);
        reconstruire_genre::<DecisionDocument>(db, rapport, &lots_ouverts, &mut index.decisions);
        reconstruire_genre::<NoteDocument>(db, rapport, &lots_ouverts, &mut index.notes);
        for noeud in noeuds_lots {
            db.delete_node(noeud);
        }
        for g in GENRES {
            // Whatever obrain reloaded from disk is not trusted (module doc).
            db.drop_text_index(g.label(), TEXTE);
            db.create_text_index(g.label(), TEXTE)
                .map_err(|e| anyhow!("index texte de {} : {e}", g.label()))?;
        }
        Ok(index)
    }
}

fn reconstruire_genre<D: Document>(
    db: &ObrainDB,
    rapport: &mut RapportOuverture,
    lots_ouverts: &HashSet<String>,
    corpus: &mut Corpus,
) {
    let label = D::GENRE.label();
    let store = db.store();
    for noeud in store.nodes_by_label(label) {
        let Some(n) = db.get_node(noeud) else {
            rapport
                .en_quarantaine
                .push(format!("nœud {label} {noeud:?} : illisible"));
            continue;
        };
        let get = |cle: &str| {
            n.get_property(cle)
                .filter(|v| !matches!(v, Value::Null))
                .cloned()
        };
        let chaine = |cle: &str| get(cle).and_then(|v| v.as_str().map(str::to_string));
        let de_lot_ouvert = chaine(LOT).is_some_and(|l| lots_ouverts.contains(&l));
        // A document of an uncommitted batch, a legacy draft, or an empty node.
        if de_lot_ouvert
            || get(BROUILLON).is_some()
            || (n.properties.is_empty() && store.out_degree(noeud) == 0)
        {
            db.delete_node(noeud);
            rapport.brouillons_supprimes += 1;
            continue;
        }
        let texte_ecrit = chaine(TEXTE);
        let doc_json = chaine(DOC);
        let id_ecrit = chaine(ID);
        let lu = doc_json
            .as_deref()
            .ok_or_else(|| anyhow!("propriété `doc` absente"))
            .and_then(|j| {
                serde_json::from_str::<D>(j).map_err(|e| anyhow!("`doc` illisible : {e}"))
            });
        let doc = match lu {
            Ok(d) if id_ecrit.as_deref() == Some(d.id()) => d,
            autre => {
                let raison = match autre {
                    Ok(_) => "`id` différent de celui du document".to_string(),
                    Err(e) => e.to_string(),
                };
                let cles = doc_json
                    .as_deref()
                    .and_then(|j| serde_json::from_str::<serde_json::Value>(j).ok())
                    .map(|v| D::GENRE.cles_tolerantes(&v));
                corpus.quarantaine.push(Quarantaine {
                    noeud,
                    id: id_ecrit,
                    cles,
                });
                rapport
                    .en_quarantaine
                    .push(format!("nœud {label} {noeud:?} : {raison}"));
                continue;
            }
        };
        let id = doc.id().to_string();
        if corpus.docs.contains_key(&id) {
            corpus.quarantaine.push(Quarantaine {
                noeud,
                id: Some(id.clone()),
                cles: Some(doc.cles()),
            });
            rapport
                .en_quarantaine
                .push(format!("nœud {label} {noeud:?} : id {id} déjà pris"));
            continue;
        }
        let texte = doc.texte();
        if texte_ecrit.as_deref() != Some(texte.as_str()) {
            // hors-poser: open-time repair of a derived property, made durable by the
            // flush that follows any non-empty opening report (`ouvrir`).
            db.set_node_property(noeud, TEXTE, Value::from(texte));
            rapport.recherches_reparees += 1;
        }
        corpus.inscrire(
            id,
            Entree {
                noeud,
                cles: doc.cles(),
            },
        );
    }
}

/// A refusal, by name, with the reason.
fn refus<T>(methode: &'static str, raison: &'static str) -> Result<T> {
    Err(anyhow::Error::new(NonPorte(methode)).context(raison))
}

type Filtre = Box<dyn Fn(&Cles) -> bool + Send + 'static>;

/// The search store of the obrain backend: the same base, writer and durability as
/// the graph store it comes from ([`ObrainGraphStore::recherche`]).
pub struct ObrainSearchStore {
    interne: Arc<Interne>,
}

impl ObrainGraphStore {
    /// The search store of this base: same engine, same writer, same durability.
    pub fn recherche(&self) -> ObrainSearchStore {
        ObrainSearchStore {
            interne: Arc::clone(&self.interne),
        }
    }
}

/// What an indexing call writes for one document.
struct Prepare {
    id: String,
    json: String,
    texte: String,
    cles: Cles,
}

impl ObrainSearchStore {
    /// The graph store over the same base, to reuse its single write path.
    fn graphe(&self) -> ObrainGraphStore {
        ObrainGraphStore {
            interne: Arc::clone(&self.interne),
        }
    }

    /// Index (insert or replace by id) `docs`: one writer section, one fsync, the
    /// new documents committed together (module doc, "Atomicity").
    async fn indexer<D: Document>(&self, docs: &[D]) -> Result<()> {
        if docs.is_empty() {
            return Ok(());
        }
        // The last document of an id wins, as in a Meilisearch batch.
        let mut ordre: Vec<String> = Vec::new();
        let mut derniers: HashMap<String, Prepare> = HashMap::new();
        for d in docs {
            let id = d.id().to_string();
            if id.is_empty() {
                bail!("document sans `id` : rien n'est indexé");
            }
            let p = Prepare {
                id: id.clone(),
                json: serde_json::to_string(d)?,
                texte: d.texte(),
                cles: d.cles(),
            };
            if derniers.insert(id.clone(), p).is_none() {
                ordre.push(id);
            }
        }
        let prepares: Vec<Prepare> = ordre
            .into_iter()
            .map(|id| derniers.remove(&id).expect("id vu"))
            .collect();
        let g = D::GENRE;
        self.graphe()
            .ecrire(
                move || prepares,
                move |prepares, w: &mut Ecriture<'_>| {
                    // Under the writer lock: what exists, what is quarantined.
                    let (existants, a_remplacer): (Vec<Option<NodeId>>, Vec<NodeId>) = {
                        let etat = w.index()?;
                        let corpus = etat.recherche.corpus(g);
                        let existants = prepares
                            .iter()
                            .map(|p| corpus.docs.get(&p.id).map(|e| e.noeud))
                            .collect();
                        let a_remplacer = prepares
                            .iter()
                            .flat_map(|p| corpus.quarantaine_de(&p.id).collect::<Vec<_>>())
                            .collect();
                        (existants, a_remplacer)
                    };
                    let vivants: Vec<NodeId> = existants.iter().flatten().copied().collect();
                    w.verifier_vivants(&vivants, "indexation")?;
                    // A quarantined node of the same id is replaced: deleted first.
                    if !a_remplacer.is_empty() {
                        w.supprimer_noeuds(&a_remplacer, "indexation (remplacement)")?;
                    }
                    let nouveaux = existants.iter().any(Option::is_none);
                    let lot = if nouveaux {
                        let lot = w.db.create_node(&[LABEL_LOT]);
                        let id_lot = uuid::Uuid::new_v4().to_string();
                        w.poser(lot, ID, Value::from(id_lot.clone()));
                        Some((lot, id_lot))
                    } else {
                        None
                    };
                    let mut cibles: Vec<NodeId> = Vec::with_capacity(prepares.len());
                    for (p, existant) in prepares.iter().zip(&existants) {
                        w.etape()?;
                        match existant {
                            Some(noeud) => {
                                // `doc` first, `texte` second: open recomputes `texte`
                                // from `doc`, so a cut between the two is repaired.
                                w.poser(*noeud, DOC, Value::from(p.json.clone()));
                                w.etape()?;
                                w.poser(*noeud, TEXTE, Value::from(p.texte.clone()));
                                cibles.push(*noeud);
                            }
                            None => {
                                let (_, id_lot) = lot.as_ref().expect("lot créé");
                                let noeud = w.db.create_node(&[g.label()]);
                                w.poser(noeud, LOT, Value::from(id_lot.clone()));
                                w.poser(noeud, ID, Value::from(p.id.clone()));
                                w.poser(noeud, DOC, Value::from(p.json.clone()));
                                w.poser(noeud, TEXTE, Value::from(p.texte.clone()));
                                cibles.push(noeud);
                            }
                        }
                    }
                    #[cfg(any(test, feature = "mock-store"))]
                    w.point(super::Interruption::AvantLeRetraitDuMarqueur)?;
                    if let Some((lot, _)) = lot {
                        // THE commit point of every new document of this call.
                        w.etape()?;
                        if !w.db.delete_node(lot) {
                            bail!(
                                "obrain a refusé la suppression du lot {lot:?} : rien n'est validé"
                            );
                        }
                    }
                    w.rendre_durable()?;
                    let mut etat = w.index_mut()?;
                    let corpus = etat.recherche.corpus_mut(g);
                    let remplaces: HashSet<NodeId> = a_remplacer.into_iter().collect();
                    corpus.retirer_quarantaine(&remplaces);
                    for (p, noeud) in prepares.into_iter().zip(cibles) {
                        corpus.inscrire(
                            p.id,
                            Entree {
                                noeud,
                                cles: p.cles,
                            },
                        );
                    }
                    Ok(())
                },
            )
            .await
    }

    /// Delete the document of this id, and any quarantined node carrying it.
    /// Absent: the no-op of Meilisearch. O(log N + Q).
    async fn supprimer_id(&self, g: Genre, quoi: &'static str, id: String) -> Result<()> {
        self.graphe()
            .ecrire(
                || (),
                move |(), w: &mut Ecriture<'_>| {
                    let (doc, quarantaine): (Option<NodeId>, Vec<NodeId>) = {
                        let etat = w.index()?;
                        let corpus = etat.recherche.corpus(g);
                        (
                            corpus.docs.get(&id).map(|e| e.noeud),
                            corpus.quarantaine_de(&id).collect(),
                        )
                    };
                    let noeuds: Vec<NodeId> = doc.into_iter().chain(quarantaine).collect();
                    if noeuds.is_empty() {
                        return Ok(());
                    }
                    w.supprimer_noeuds(&noeuds, quoi)?;
                    w.rendre_durable()?;
                    let mut etat = w.index_mut()?;
                    let corpus = etat.recherche.corpus_mut(g);
                    corpus.retirer(&id);
                    corpus.retirer_quarantaine(&noeuds.into_iter().collect());
                    Ok(())
                },
            )
            .await
    }

    /// Delete the documents of kind `g` whose filterable fields satisfy `choisit`,
    /// and the quarantined nodes whose readable fields (`None`: no readable JSON)
    /// satisfy it too.
    async fn supprimer_si(
        &self,
        g: Genre,
        quoi: &'static str,
        choisit: impl Fn(Option<&Cles>) -> bool + Send + 'static,
    ) -> Result<()> {
        self.graphe()
            .ecrire(
                || (),
                move |(), w: &mut Ecriture<'_>| {
                    let (ids, mut noeuds, quarantaine): (Vec<String>, Vec<NodeId>, Vec<NodeId>) = {
                        let etat = w.index()?;
                        let corpus = etat.recherche.corpus(g);
                        let (ids, noeuds) = corpus
                            .docs
                            .iter()
                            .filter(|(_, e)| choisit(Some(&e.cles)))
                            .map(|(id, e)| (id.clone(), e.noeud))
                            .unzip();
                        let quarantaine = corpus
                            .quarantaine
                            .iter()
                            .filter(|q| choisit(q.cles.as_ref()))
                            .map(|q| q.noeud)
                            .collect();
                        (ids, noeuds, quarantaine)
                    };
                    noeuds.extend(quarantaine.iter().copied());
                    if noeuds.is_empty() {
                        return Ok(());
                    }
                    w.supprimer_noeuds(&noeuds, quoi)?;
                    w.rendre_durable()?;
                    let mut etat = w.index_mut()?;
                    let corpus = etat.recherche.corpus_mut(g);
                    for id in ids {
                        corpus.retirer(&id);
                    }
                    corpus.retirer_quarantaine(&quarantaine.into_iter().collect());
                    Ok(())
                },
            )
            .await
    }

    /// The text index of kind `g`, built under the WRITER lock if it is missing:
    /// no write can run between the scan and obrain's write hooks (module doc).
    /// Called BEFORE taking the read lock of the indexes (a writer holds the writer
    /// lock while it waits for the write lock of the indexes).
    fn assurer_index_texte(interne: &Interne, g: Genre) -> Result<()> {
        if interne.db.has_text_index(g.label(), TEXTE) {
            return Ok(());
        }
        let _ecrivain = interne
            .ecrivain
            .lock()
            .map_err(|_| anyhow!("verrou d'écriture obrain empoisonné"))?;
        if interne.ferme.load(Ordering::Acquire) {
            bail!(
                "la base obrain {} est fermée : l'opération est refusée",
                interne.chemin.display()
            );
        }
        if !interne.db.has_text_index(g.label(), TEXTE) {
            interne
                .db
                .create_text_index(g.label(), TEXTE)
                .map_err(|e| anyhow!("index texte de {} : {e}", g.label()))?;
        }
        Ok(())
    }

    /// The best `limite` documents of kind `D` for `requete` that pass `filtre`.
    async fn chercher<D: Document>(
        &self,
        methode: &'static str,
        requete: &str,
        limite: usize,
        filtre: Filtre,
    ) -> Result<Vec<SearchHit<D>>> {
        let requete = requete.to_string();
        let store = self.graphe();
        // On the blocking pool: the search reads nodes and runs the BM25 scoring.
        tokio::task::spawn_blocking(move || {
            store.garde()?;
            Self::assurer_index_texte(&store.interne, D::GENRE)?;
            store.lire(|db, etat| {
                chercher_dans::<D>(db, &etat.recherche, methode, &requete, limite, &*filtre)
            })
        })
        .await
        .context("tâche bloquante obrain interrompue")?
    }

    fn stats(&self, g: Genre) -> Result<IndexStats> {
        let total = self
            .graphe()
            .lire(|_, etat| Ok(etat.recherche.corpus(g).docs.len()))?;
        Ok(IndexStats {
            total_documents: total,
            is_indexing: false,
        })
    }

    /// Test hook: how many nodes of `label` the engine holds (served, quarantined
    /// or of a batch in progress).
    #[cfg(any(test, feature = "mock-store"))]
    #[doc(hidden)]
    pub fn noeuds_du_label(&self, label: &str) -> usize {
        self.interne.db.store().nodes_by_label(label).len()
    }

    /// Test hook: how many nodes the RAW text index of `label` returns for `requete`
    /// (served or not). Tells "deleted from the index" from "filtered out".
    #[cfg(any(test, feature = "mock-store"))]
    #[doc(hidden)]
    pub fn bm25_brut(&self, label: &str, requete: &str, k: usize) -> Result<usize> {
        self.interne
            .db
            .text_search(label, TEXTE, requete, k)
            .map(|h| h.len())
            .map_err(|e| anyhow!("recherche texte de {label} : {e}"))
    }
}

/// The score of a hit (module doc, "Score and order").
fn score(couverture: f64, bm25: f64) -> f64 {
    let b = bm25.max(0.0);
    (1.0 - DELTA) * couverture + DELTA * (b / (1.0 + b))
}

fn chercher_dans<D: Document>(
    db: &ObrainDB,
    index: &IndexRecherche,
    methode: &'static str,
    requete: &str,
    limite: usize,
    filtre: &(dyn Fn(&Cles) -> bool + Send),
) -> Result<Vec<SearchHit<D>>> {
    let g = D::GENRE;
    let corpus = index.corpus(g);
    let tokenizer = SimpleTokenizer::new();
    let termes: Vec<String> = {
        let mut t = tokenizer.tokenize(requete);
        t.sort();
        t.dedup();
        t
    };
    // A query the index could not answer is refused whatever the corpus holds.
    if termes.is_empty() && requete.chars().any(char::is_alphanumeric) {
        return refus(methode, REFUS_REQUETE);
    }
    if limite == 0 || corpus.docs.is_empty() {
        return Ok(Vec::new());
    }

    // The document of a hit, if it is served AND passes the filter on the version
    // read (module doc, "Filters").
    let lire_doc = |noeud: NodeId| -> Result<Option<D>> {
        let Some(n) = db.get_node(noeud) else {
            // Deleted between the index and the read: not there any more.
            return Ok(None);
        };
        let json = n
            .get_property(DOC)
            .and_then(|v| v.as_str().map(str::to_string))
            .ok_or_else(|| anyhow!("nœud {} {noeud:?} sans `doc`", g.label()))?;
        let d = serde_json::from_str::<D>(&json)
            .map_err(|e| anyhow!("nœud {} {noeud:?} : `doc` illisible : {e}", g.label()))?;
        Ok(filtre(&d.cles()).then_some(d))
    };

    // The placeholder search: no term to match, no alphanumeric character either.
    if termes.is_empty() {
        let mut out = Vec::new();
        for e in corpus.docs.values() {
            if out.len() == limite {
                break;
            }
            if !filtre(&e.cles) {
                continue;
            }
            if let Some(document) = lire_doc(e.noeud)? {
                out.push(SearchHit {
                    document,
                    score: 1.0,
                });
            }
        }
        return Ok(out);
    }

    let mut k = limite.saturating_mul(4).max(32);
    loop {
        let hits = db
            .text_search(g.label(), TEXTE, requete, k)
            .map_err(|e| anyhow!("recherche texte de {} : {e}", g.label()))?;
        let epuise = hits.len() < k;
        let mut candidats: Vec<SearchHit<D>> = Vec::new();
        for (noeud, bm25) in &hits {
            let Some(id) = corpus.par_noeud.get(noeud) else {
                continue; // not served: a batch in progress, a quarantined node
            };
            if !corpus.docs.get(id).is_some_and(|e| filtre(&e.cles)) {
                continue;
            }
            let Some(document) = lire_doc(*noeud)? else {
                continue;
            };
            let presents: HashSet<String> =
                tokenizer.tokenize(&document.texte()).into_iter().collect();
            let trouves = termes.iter().filter(|t| presents.contains(*t)).count();
            candidats.push(SearchHit {
                score: score(trouves as f64 / termes.len() as f64, *bm25),
                document,
            });
        }
        if candidats.len() >= limite || epuise {
            // Stable: equal scores keep the BM25 order.
            candidats.sort_by(|a, b| b.score.total_cmp(&a.score));
            candidats.truncate(limite);
            return Ok(candidats);
        }
        k = k.saturating_mul(4);
    }
}

const REFUS_REQUETE: &str = "requête sans aucun terme que l'index texte d'obrain sache chercher \
    (uniquement des mots vides anglais ou des mots d'un caractère) : Meilisearch les cherche, \
    cet index les écarte ; répondre vide serait faux";

/// The filter of a code search.
fn filtre_code(langue: Option<&str>, slug: Option<&str>, prefixe: Option<&str>) -> Filtre {
    let (langue, slug, prefixe) = (
        langue.map(str::to_string),
        slug.map(str::to_string),
        prefixe.map(str::to_lowercase),
    );
    Box::new(move |c| {
        langue.as_deref().is_none_or(|l| egal(&c.a, l))
            && slug.as_deref().is_none_or(|s| egal(&c.slug, s))
            && prefixe.as_deref().is_none_or(|p| {
                c.b.as_deref()
                    .is_some_and(|path| path.to_lowercase().starts_with(p))
            })
    })
}

fn filtre_note(
    slug: Option<&str>,
    note_type: Option<&str>,
    status: Option<&str>,
    importance: Option<&str>,
) -> Filtre {
    let (slug, note_type, status, importance) = (
        slug.map(str::to_string),
        note_type.map(str::to_string),
        status.map(str::to_string),
        importance.map(str::to_string),
    );
    Box::new(move |c| {
        slug.as_deref().is_none_or(|s| egal(&c.slug, s))
            && note_type.as_deref().is_none_or(|t| egal(&c.a, t))
            && match status.as_deref() {
                Some(s) => egal(&c.b, s),
                None => STATUTS_PAR_DEFAUT.iter().any(|s| egal(&c.b, s)),
            }
            && importance.as_deref().is_none_or(|i| egal(&c.c, i))
    })
}

fn sans_score<D>(hits: Vec<SearchHit<D>>) -> Vec<D> {
    hits.into_iter().map(|h| h.document).collect()
}

#[async_trait]
impl SearchStore for ObrainSearchStore {
    // ---- Code -----------------------------------------------------------------

    async fn index_code(&self, doc: &CodeDocument) -> Result<()> {
        self.indexer(std::slice::from_ref(doc)).await
    }

    async fn index_code_batch(&self, docs: &[CodeDocument]) -> Result<()> {
        self.indexer(docs).await
    }

    async fn search_code(
        &self,
        query: &str,
        limit: usize,
        language_filter: Option<&str>,
    ) -> Result<Vec<CodeDocument>> {
        self.search_code_in_project(query, limit, language_filter, None)
            .await
    }

    async fn search_code_in_project(
        &self,
        query: &str,
        limit: usize,
        language_filter: Option<&str>,
        project_slug: Option<&str>,
    ) -> Result<Vec<CodeDocument>> {
        Ok(sans_score(
            self.search_code_with_scores(query, limit, language_filter, project_slug, None)
                .await?,
        ))
    }

    async fn search_code_with_scores(
        &self,
        query: &str,
        limit: usize,
        language_filter: Option<&str>,
        project_slug: Option<&str>,
        path_prefix: Option<&str>,
    ) -> Result<Vec<SearchHit<CodeDocument>>> {
        self.chercher::<CodeDocument>(
            "SearchStore::search_code_with_scores",
            query,
            limit,
            filtre_code(language_filter, project_slug, path_prefix),
        )
        .await
    }

    async fn delete_code(&self, path: &str) -> Result<()> {
        self.supprimer_id(
            Genre::Code,
            "SearchStore::delete_code",
            MeiliClient::path_to_id(path),
        )
        .await
    }

    async fn delete_code_for_project(&self, project_slug: &str) -> Result<()> {
        let slug = project_slug.to_string();
        self.supprimer_si(
            Genre::Code,
            "SearchStore::delete_code_for_project",
            move |c| c.is_some_and(|c| egal(&c.slug, &slug)),
        )
        .await
    }

    async fn delete_orphan_code_documents(&self) -> Result<()> {
        // A node whose JSON is unreadable has no readable project: an orphan.
        self.supprimer_si(
            Genre::Code,
            "SearchStore::delete_orphan_code_documents",
            |c| c.is_none_or(|c| vide(&c.projet_id) || vide(&c.slug)),
        )
        .await
    }

    async fn delete_all_code(&self) -> Result<()> {
        self.supprimer_si(Genre::Code, "SearchStore::delete_all_code", |_| true)
            .await
    }

    async fn get_code_stats(&self) -> Result<IndexStats> {
        self.stats(Genre::Code)
    }

    // ---- Decisions --------------------------------------------------------------

    async fn index_decision(&self, doc: &DecisionDocument) -> Result<()> {
        self.indexer(std::slice::from_ref(doc)).await
    }

    async fn search_decisions(&self, query: &str, limit: usize) -> Result<Vec<DecisionDocument>> {
        self.search_decisions_in_project(query, limit, None).await
    }

    async fn search_decisions_in_project(
        &self,
        query: &str,
        limit: usize,
        project_slug: Option<&str>,
    ) -> Result<Vec<DecisionDocument>> {
        let slug = project_slug.map(str::to_string);
        Ok(sans_score(
            self.chercher::<DecisionDocument>(
                "SearchStore::search_decisions_in_project",
                query,
                limit,
                Box::new(move |c| slug.as_deref().is_none_or(|s| egal(&c.slug, s))),
            )
            .await?,
        ))
    }

    async fn search_decisions_in_projects(
        &self,
        query: &str,
        limit: usize,
        project_slugs: &[String],
    ) -> Result<Vec<DecisionDocument>> {
        // `project_slug IN []` matches nothing: no need to read the index.
        if project_slugs.is_empty() {
            return Ok(Vec::new());
        }
        let slugs = project_slugs.to_vec();
        Ok(sans_score(
            self.chercher::<DecisionDocument>(
                "SearchStore::search_decisions_in_projects",
                query,
                limit,
                Box::new(move |c| slugs.iter().any(|s| egal(&c.slug, s))),
            )
            .await?,
        ))
    }

    async fn delete_decision(&self, id: &str) -> Result<()> {
        self.supprimer_id(
            Genre::Decision,
            "SearchStore::delete_decision",
            id.to_string(),
        )
        .await
    }

    async fn delete_decisions_for_project(&self, project_slug: &str) -> Result<()> {
        let slug = project_slug.to_string();
        self.supprimer_si(
            Genre::Decision,
            "SearchStore::delete_decisions_for_project",
            move |c| c.is_some_and(|c| egal(&c.slug, &slug)),
        )
        .await
    }

    async fn delete_decisions_for_task(&self, task_id: &str) -> Result<()> {
        let tache = task_id.to_string();
        self.supprimer_si(
            Genre::Decision,
            "SearchStore::delete_decisions_for_task",
            move |c| c.is_some_and(|c| egal(&c.a, &tache)),
        )
        .await
    }

    // ---- Notes -------------------------------------------------------------------

    async fn index_note(&self, doc: &NoteDocument) -> Result<()> {
        self.indexer(std::slice::from_ref(doc)).await
    }

    async fn index_notes_batch(&self, docs: &[NoteDocument]) -> Result<()> {
        self.indexer(docs).await
    }

    async fn search_notes(&self, query: &str, limit: usize) -> Result<Vec<NoteDocument>> {
        self.search_notes_with_filters(query, limit, None, None, None, None)
            .await
    }

    async fn search_notes_with_filters(
        &self,
        query: &str,
        limit: usize,
        project_slug: Option<&str>,
        note_type: Option<&str>,
        status: Option<&str>,
        importance: Option<&str>,
    ) -> Result<Vec<NoteDocument>> {
        Ok(sans_score(
            self.search_notes_with_scores(
                query,
                limit,
                project_slug,
                note_type,
                status,
                importance,
            )
            .await?,
        ))
    }

    async fn search_notes_with_scores(
        &self,
        query: &str,
        limit: usize,
        project_slug: Option<&str>,
        note_type: Option<&str>,
        status: Option<&str>,
        importance: Option<&str>,
    ) -> Result<Vec<SearchHit<NoteDocument>>> {
        self.chercher::<NoteDocument>(
            "SearchStore::search_notes_with_scores",
            query,
            limit,
            filtre_note(project_slug, note_type, status, importance),
        )
        .await
    }

    async fn delete_note(&self, id: &str) -> Result<()> {
        self.supprimer_id(Genre::Note, "SearchStore::delete_note", id.to_string())
            .await
    }

    async fn delete_notes_for_project(&self, project_slug: &str) -> Result<()> {
        let slug = project_slug.to_string();
        self.supprimer_si(
            Genre::Note,
            "SearchStore::delete_notes_for_project",
            move |c| c.is_some_and(|c| egal(&c.slug, &slug)),
        )
        .await
    }

    async fn update_note_status(&self, id: &str, status: &str) -> Result<()> {
        let (id, status) = (id.to_string(), status.to_string());
        self.graphe()
            .ecrire(
                || (),
                move |(), w: &mut Ecriture<'_>| {
                    let noeud = {
                        let etat = w.index()?;
                        match etat.recherche.notes.docs.get(&id) {
                            Some(e) => e.noeud,
                            // Absent: the no-op of Meilisearch (`get_document` fails,
                            // nothing indexed).
                            None => return Ok(()),
                        }
                    };
                    w.verifier_vivants(&[noeud], "update_note_status")?;
                    let json =
                        w.db.get_node(noeud)
                            .and_then(|n| {
                                n.get_property(DOC)
                                    .and_then(|v| v.as_str().map(str::to_string))
                            })
                            .ok_or_else(|| anyhow!("note indexée {id} sans `doc`"))?;
                    let mut doc: NoteDocument = serde_json::from_str(&json)
                        .map_err(|e| anyhow!("note indexée {id} : `doc` illisible : {e}"))?;
                    doc.status = status;
                    let cles = doc.cles();
                    w.etape()?;
                    // `texte` does not hold the status: one write, nothing to tear.
                    w.poser(noeud, DOC, Value::from(serde_json::to_string(&doc)?));
                    w.rendre_durable()?;
                    let mut etat = w.index_mut()?;
                    etat.recherche.notes.inscrire(id, Entree { noeud, cles });
                    Ok(())
                },
            )
            .await
    }

    async fn get_notes_stats(&self) -> Result<IndexStats> {
        self.stats(Genre::Note)
    }

    // ---- Health ------------------------------------------------------------------

    /// Reachable = the base is open and determinate (the engine is in-process).
    async fn health_check(&self) -> Result<bool> {
        Ok(self.graphe().garde().is_ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::models::{ProjectNode, ProjectProfile};
    use crate::neo4j::traits::ProjectStore;
    use crate::orchestrator::Orchestrator;

    fn note(id: &str, content: &str) -> NoteDocument {
        NoteDocument {
            id: id.into(),
            project_id: "pid".into(),
            project_slug: "p".into(),
            note_type: "gotcha".into(),
            status: "active".into(),
            importance: "high".into(),
            scope_type: "project".into(),
            scope_path: String::new(),
            content: content.into(),
            tags: vec![],
            anchor_entities: vec![],
            created_at: 0,
            created_by: "t".into(),
            staleness_score: 0.0,
            author_kind: "agent".into(),
            human_confirmed: false,
            memory_horizon: "operational".into(),
        }
    }

    /// The whole chain of `Orchestrator::delete_project` on obrain: the graph
    /// deletion refuses (a search document is a node outside the projects), so the
    /// code index of the project must be left whole. REGRESSION: the code documents
    /// were deleted BEFORE the graph deletion, which then refused — the project
    /// stayed, its code index was gone.
    #[tokio::test]
    async fn a_refused_project_deletion_keeps_the_project_s_code_index() {
        let graphe = ObrainGraphStore::open_temporary().unwrap();
        let recherche = Arc::new(graphe.recherche());
        let id = uuid::Uuid::new_v4();
        let slug = format!("p-{id}");
        graphe
            .create_project(&ProjectNode {
                id,
                name: "p".into(),
                slug: slug.clone(),
                root_path: String::new(),
                description: Some("d".into()),
                created_at: chrono::Utc::now(),
                last_synced: None,
                analytics_computed_at: None,
                last_co_change_computed_at: None,
                default_note_energy: None,
                scaffolding_override: None,
                sharing_policy: None,
                watch_enabled: false,
                profile: ProjectProfile::Software,
            })
            .await
            .unwrap();
        recherche
            .index_code(&CodeDocument {
                id: MeiliClient::path_to_id("src/a.rs"),
                path: "src/a.rs".into(),
                language: "rust".into(),
                symbols: vec!["zork".into()],
                docstrings: String::new(),
                signatures: vec![],
                imports: vec![],
                project_id: id.to_string(),
                project_slug: slug.clone(),
            })
            .await
            .unwrap();

        let mut state = crate::test_helpers::mock_app_state();
        state.neo4j = Arc::new(graphe);
        state.meili = recherche.clone();
        let orchestrateur = Orchestrator::new(state).await.unwrap();

        let err = orchestrateur.delete_project(id).await.unwrap_err();
        assert!(
            format!("{err:#}").contains("ProjectStore::delete_project (cascade)"),
            "the graph deletion refuses: {err:#}"
        );
        assert_eq!(
            recherche
                .search_code_in_project("zork", 10, None, Some(&slug))
                .await
                .unwrap()
                .len(),
            1,
            "the project is still there, so is its code index"
        );
    }

    /// The text index built at open is not served from obrain's persisted copy: a
    /// document written after the copy was persisted is found after the reopen.
    #[tokio::test]
    async fn the_text_index_is_built_at_open_from_the_documents() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("b");
        let g = ObrainGraphStore::open(&base, super::super::Creation::Permise).unwrap();
        let s = g.recherche();
        s.index_note(&note("n1", "wombat")).await.unwrap();
        assert_eq!(s.search_notes("wombat", 5).await.unwrap().len(), 1);
        s.index_note(&note("n2", "wombat quokka")).await.unwrap();
        g.close().await.unwrap();
        drop((s, g));
        let g = ObrainGraphStore::open(&base, super::super::Creation::Interdite).unwrap();
        let s = g.recherche();
        assert_eq!(s.bm25_brut("RechercheNote", "wombat", 10).unwrap(), 2);
        assert_eq!(s.search_notes("quokka", 5).await.unwrap().len(), 1);
    }
}
