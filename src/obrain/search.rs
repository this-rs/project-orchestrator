//! `SearchStore` on obrain — the 27 methods of the Meilisearch contract, served by
//! the embedded BM25 text index of obrain. Nothing here is refused wholesale; the
//! only refusal is a query the index cannot answer faithfully (see "Refused").
//!
//! # Storage
//!
//! A document is one node, labelled `RechercheCode`, `RechercheDecision` or
//! `RechercheNote`, with three properties: `id`, `doc` (the whole document as
//! JSON: the SINGLE source of truth) and `texte` (the searchable fields of the
//! document, concatenated: what the text index of obrain tokenizes). `texte` is a
//! pure function of `doc`, so open recomputes it and rewrites a node whose `texte`
//! disagrees (an update cut between the two writes): the repair is counted in
//! `RapportOuverture::recherches_reparees`.
//!
//! The searchable fields are Meilisearch's own (`configure_*_index`): code =
//! `symbols`, `docstrings`, `signatures`, `path`, `imports`; decisions =
//! `description`, `rationale`, `tags`; notes = `content`, `tags`, `scope_path`,
//! `anchor_entities`.
//!
//! The text index (`create_text_index(label, "texte")`) is built on the first
//! search of a kind (and again on the first search after a reopen) and then
//! maintained by obrain on every write. A search never scans the corpus unless a
//! filter rejects enough of the top-k to need more (below).
//!
//! # Filters: AFTER the search, by over-fetching, never wrong
//!
//! obrain's `text_search(label, property, query, k)` takes no filter. A filter
//! (project, language, path prefix, note type, status, importance, task) is applied
//! to the hits afterwards. To keep the answer exact, the search asks for
//! `max(32, 4 × limit)` hits, filters them, and multiplies `k` by 4 until either
//! `limit` documents survive, or the index returned fewer than it was asked for
//! (every document that matches the query has been seen), or `k` reached the number
//! of documents of the kind. What is returned is therefore exactly the first
//! `limit` documents of the BM25 ranking that pass the filter — a selective filter
//! costs more rounds, never a missing hit. The filters read a small in-memory copy
//! of the filterable fields of each document (`Cles`), not the document.
//!
//! # Differences with Meilisearch, all named
//!
//! - **Matching.** BM25 over whole tokens (lowercased, split on non-alphanumerics,
//!   English stop words and tokens shorter than 2 bytes dropped), any query token
//!   suffices (OR). Meilisearch also matches by prefix, tolerates typos and prefers
//!   documents that match every word. A query that Meilisearch answers through a
//!   typo or a prefix is answered with fewer hits here.
//! - **Ranking.** One BM25 score over the concatenated fields; Meilisearch ranks by
//!   attribute order (`symbols` before `docstrings`…) and proximity.
//! - **`SearchHit::score`.** In `[0, 1]` like Meilisearch's ranking score, but
//!   it is the FRACTION OF THE QUERY'S TERMS found in the document, not Meilisearch's
//!   score; the order of the hits is BM25's, so the score is not monotone in it.
//!   Callers that threshold on the score (`chat/feedback.rs`, 0.85) therefore ask
//!   "almost every term of the probe is in that note", which is what a duplicate is.
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
//! - **`search_decisions_in_projects(&[])`** matches no document (`IN []`).
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
//! # Durability and atomicity
//!
//! One writer section and ONE fsync per call, through the same `Ecriture` as every
//! domain. A new document follows the draft-marker protocol of `decision.rs` (the
//! marker is the first property, removed last; open deletes a draft). A batch
//! marks ALL its new nodes and unmarks them together at the end: a crash leaves
//! none of its new documents, never a prefix. The documents it REPLACES are
//! written in place, one by one: a crash may leave some replaced and some not
//! (indexing is idempotent; the sync that calls it runs again). A bulk deletion
//! (`delete_*_for_project`, `delete_all_code`, `delete_orphan_code_documents`)
//! deletes node by node and may stop half-way on a crash; running it again
//! finishes it.
//!
//! # Cost (measured: nothing yet; see the PR)
//!
//! `get_*_stats`, `update_note_status`: O(1) (hash lookups). Every deletion
//! (`delete_code`, `delete_decision`, `delete_note`, the `delete_*_for_*` family,
//! `delete_all_code`) SCANS the documents of its kind, O(documents of the kind),
//! because the filterable copies are kept by id and not by project or task.
//! A search answers the first build of the text index of a kind with one scan of
//! that kind (`create_text_index`), after each open and after the first search of
//! the process: the cost of the search path grows with the corpus until the index
//! is kept open across requests. Open is O(documents): the index is rebuilt by one
//! scan per label, like every other domain. None of this is measured at 1 M or
//! 5 M nodes; consolidation's performance bar is NOT claimed here.
//!
//! `ProjectStore::delete_project` refuses while the base holds any node outside the
//! projects and plans (`project.rs`): these documents are such nodes, listed under
//! « recherche » in its refusal.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Result};
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

const BROUILLON: &str = "_brouillon";
const DOC: &str = "doc";
const TEXTE: &str = "texte";
const ID: &str = "id";

/// The labels this domain writes (`LABELS_PAR_DOMAINE` of `project.rs`).
pub(super) const LABELS: &[&str] = &["RechercheCode", "RechercheDecision", "RechercheNote"];

/// The statuses a note search covers when the caller gives none.
const STATUTS_PAR_DEFAUT: &[&str] = &["active", "needs_review"];

/// The three corpora.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Genre {
    Code,
    Decision,
    Note,
}

impl Genre {
    fn label(self) -> &'static str {
        match self {
            Genre::Code => LABELS[0],
            Genre::Decision => LABELS[1],
            Genre::Note => LABELS[2],
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

/// The documents of one kind: by id, and by node (to name a hit).
#[derive(Default)]
struct Corpus {
    docs: HashMap<String, Entree>,
    par_noeud: HashMap<NodeId, String>,
    /// Ids on disk but unreadable or doubled at open (quarantined): a write of
    /// such an id refuses instead of landing next to the node left there.
    reserves: HashSet<String>,
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

    pub(super) fn reconstruire(db: &ObrainDB, rapport: &mut RapportOuverture) -> Self {
        let mut index = Self::default();
        reconstruire_genre::<CodeDocument>(db, rapport, &mut index.code);
        reconstruire_genre::<DecisionDocument>(db, rapport, &mut index.decisions);
        reconstruire_genre::<NoteDocument>(db, rapport, &mut index.notes);
        index
    }
}

fn reconstruire_genre<D: Document>(
    db: &ObrainDB,
    rapport: &mut RapportOuverture,
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
        let texte_ecrit = get(TEXTE).and_then(|v| v.as_str().map(str::to_string));
        let doc_json = get(DOC).and_then(|v| v.as_str().map(str::to_string));
        // A creation that never reached its commit point, or an empty node.
        if get(BROUILLON).is_some() || (n.properties.is_empty() && store.out_degree(noeud) == 0) {
            db.delete_node(noeud);
            rapport.brouillons_supprimes += 1;
            continue;
        }
        let id_ecrit = get(ID).and_then(|v| v.as_str().map(str::to_string));
        let lu = doc_json
            .ok_or_else(|| anyhow!("propriété `doc` absente"))
            .and_then(|j| {
                serde_json::from_str::<D>(&j).map_err(|e| anyhow!("`doc` illisible : {e}"))
            });
        let doc = match lu {
            Ok(d) if id_ecrit.as_deref() == Some(d.id()) => d,
            autre => {
                if let Some(id) = id_ecrit {
                    corpus.reserves.insert(id);
                }
                let raison = match autre {
                    Ok(_) => "`id` différent de celui du document".to_string(),
                    Err(e) => e.to_string(),
                };
                rapport
                    .en_quarantaine
                    .push(format!("nœud {label} {noeud:?} : {raison}"));
                continue;
            }
        };
        let id = doc.id().to_string();
        if corpus.docs.contains_key(&id) {
            corpus.reserves.insert(id.clone());
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

/// The search store of the obrain backend.
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

impl ObrainSearchStore {
    /// The graph store over the same base, to reuse its single write path.
    fn graphe(&self) -> ObrainGraphStore {
        ObrainGraphStore {
            interne: Arc::clone(&self.interne),
        }
    }

    /// Index (insert or replace by id) `docs`: one writer section, one fsync.
    async fn indexer<D: Document>(&self, docs: &[D]) -> Result<()> {
        if docs.is_empty() {
            return Ok(());
        }
        // The last document of an id wins, as in a Meilisearch batch.
        let mut ordre: Vec<String> = Vec::new();
        let mut derniers: HashMap<String, (String, String, Cles)> = HashMap::new();
        for d in docs {
            let id = d.id().to_string();
            if id.is_empty() {
                bail!("document sans `id` : rien n'est indexé");
            }
            let json = serde_json::to_string(d)?;
            if derniers
                .insert(id.clone(), (json, d.texte(), d.cles()))
                .is_none()
            {
                ordre.push(id);
            }
        }
        let prepares: Vec<(String, String, String, Cles)> = ordre
            .into_iter()
            .map(|id| {
                let (json, texte, cles) = derniers.remove(&id).expect("id vu");
                (id, json, texte, cles)
            })
            .collect();
        let g = D::GENRE;
        self.graphe()
            .ecrire(
                move || prepares,
                move |prepares, w: &mut Ecriture<'_>| {
                    let mut existants: Vec<(NodeId, usize)> = Vec::new();
                    {
                        let etat = w.index()?;
                        let corpus = etat.recherche.corpus(g);
                        for (k, (id, ..)) in prepares.iter().enumerate() {
                            if corpus.reserves.contains(id) {
                                bail!(
                                    "indexation du document {id} refusée : un nœud {} de cet id \
                                     est sur disque, illisible ou en double à l'ouverture \
                                     (quarantaine). Il n'est ni repris ni écrasé",
                                    g.label()
                                );
                            }
                            if let Some(e) = corpus.docs.get(id) {
                                existants.push((e.noeud, k));
                            }
                        }
                    }
                    let noeuds: Vec<NodeId> = existants.iter().map(|(n, _)| *n).collect();
                    w.verifier_vivants(&noeuds, "indexation")?;
                    let mut nouveaux: Vec<NodeId> = Vec::new();
                    let mut cibles: Vec<NodeId> = Vec::with_capacity(prepares.len());
                    let par_rang: HashMap<usize, NodeId> =
                        existants.iter().map(|(n, k)| (*k, *n)).collect();
                    for (k, (id, json, texte, _)) in prepares.iter().enumerate() {
                        match par_rang.get(&k) {
                            Some(noeud) => {
                                // `doc` first, `texte` second: open recomputes `texte`
                                // from `doc`, so a cut between the two is repaired.
                                w.etape()?;
                                w.poser(*noeud, DOC, Value::from(json.clone()));
                                w.etape()?;
                                w.poser(*noeud, TEXTE, Value::from(texte.clone()));
                                cibles.push(*noeud);
                            }
                            None => {
                                let noeud = w.db.create_node(&[g.label()]);
                                w.etape()?;
                                w.poser(noeud, BROUILLON, Value::from(true));
                                w.poser(noeud, ID, Value::from(id.clone()));
                                w.poser(noeud, DOC, Value::from(json.clone()));
                                w.poser(noeud, TEXTE, Value::from(texte.clone()));
                                nouveaux.push(noeud);
                                cibles.push(noeud);
                            }
                        }
                    }
                    #[cfg(any(test, feature = "mock-store"))]
                    w.point(super::Interruption::AvantLeRetraitDuMarqueur)?;
                    for noeud in &nouveaux {
                        w.etape()?;
                        w.db.remove_node_property(*noeud, BROUILLON);
                    }
                    w.rendre_durable()?;
                    let mut etat = w.index_mut()?;
                    let corpus = etat.recherche.corpus_mut(g);
                    for ((id, _, _, cles), noeud) in prepares.into_iter().zip(cibles) {
                        corpus.inscrire(id, Entree { noeud, cles });
                    }
                    Ok(())
                },
            )
            .await
    }

    /// Delete the documents of kind `g` whose id and filterable fields satisfy
    /// `choisit`. Absent ids and empty selections are the no-op of Meilisearch.
    async fn supprimer_si(
        &self,
        g: Genre,
        quoi: &'static str,
        choisit: impl Fn(&str, &Cles) -> bool + Send + 'static,
    ) -> Result<()> {
        self.graphe()
            .ecrire(
                || (),
                move |(), w: &mut Ecriture<'_>| {
                    let (ids, noeuds): (Vec<String>, Vec<NodeId>) = {
                        let etat = w.index()?;
                        etat.recherche
                            .corpus(g)
                            .docs
                            .iter()
                            .filter(|(id, e)| choisit(id, &e.cles))
                            .map(|(id, e)| (id.clone(), e.noeud))
                            .unzip()
                    };
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
                    Ok(())
                },
            )
            .await
    }

    /// The top `limite` documents of kind `D` for `requete` that pass `filtre`.
    async fn chercher<D: Document>(
        &self,
        methode: &'static str,
        requete: &str,
        limite: usize,
        filtre: Filtre,
    ) -> Result<Vec<SearchHit<D>>> {
        let requete = requete.to_string();
        // On the blocking pool: the first search of a kind builds the text index.
        let store = self.graphe();
        let verrou = self.construction_partagee();
        tokio::task::spawn_blocking(move || {
            store.lire(|db, etat| {
                chercher_dans::<D>(
                    db,
                    &etat.recherche,
                    methode,
                    &requete,
                    limite,
                    &*filtre,
                    &verrou,
                )
            })
        })
        .await
        .map_err(|e| anyhow!("tâche bloquante obrain interrompue : {e}"))?
    }

    fn construction_partagee(&self) -> Arc<Mutex<()>> {
        Arc::clone(&self.interne.construction_texte)
    }
}

fn chercher_dans<D: Document>(
    db: &ObrainDB,
    index: &IndexRecherche,
    methode: &'static str,
    requete: &str,
    limite: usize,
    filtre: &(dyn Fn(&Cles) -> bool + Send),
    verrou: &Mutex<()>,
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
    if limite == 0 || corpus.docs.is_empty() {
        // Nothing can match (no document, or nothing asked): still refuse a query
        // the index could not have answered, so the answer does not depend on the corpus.
        if termes.is_empty() && requete.chars().any(char::is_alphanumeric) {
            return refus(methode, REFUS_REQUETE);
        }
        return Ok(Vec::new());
    }

    let lire_doc = |noeud: NodeId| -> Result<Option<D>> {
        let Some(n) = db.get_node(noeud) else {
            // Deleted between the index and the read: not there any more.
            return Ok(None);
        };
        let json = n
            .get_property(DOC)
            .and_then(|v| v.as_str().map(str::to_string))
            .ok_or_else(|| anyhow!("nœud {} {noeud:?} sans `doc`", g.label()))?;
        Ok(Some(serde_json::from_str::<D>(&json).map_err(|e| {
            anyhow!("nœud {} {noeud:?} : `doc` illisible : {e}", g.label())
        })?))
    };

    // The placeholder search: no term to match, no alphanumeric character either.
    if termes.is_empty() {
        if requete.chars().any(char::is_alphanumeric) {
            return refus(methode, REFUS_REQUETE);
        }
        let mut ids: Vec<(&String, NodeId)> = corpus
            .docs
            .iter()
            .filter(|(_, e)| filtre(&e.cles))
            .map(|(id, e)| (id, e.noeud))
            .collect();
        ids.sort();
        let mut out = Vec::new();
        for (_, noeud) in ids {
            if out.len() == limite {
                break;
            }
            if let Some(document) = lire_doc(noeud)? {
                out.push(SearchHit {
                    document,
                    score: 1.0,
                });
            }
        }
        return Ok(out);
    }

    {
        let _g = verrou
            .lock()
            .map_err(|_| anyhow!("verrou de construction de l'index texte empoisonné"))?;
        if !db.has_text_index(g.label(), TEXTE) {
            db.create_text_index(g.label(), TEXTE)
                .map_err(|e| anyhow!("index texte de {} : {e}", g.label()))?;
        }
    }

    let total = corpus.docs.len();
    let mut k = limite.saturating_mul(4).max(32).min(total);
    loop {
        let hits = db
            .text_search(g.label(), TEXTE, requete, k)
            .map_err(|e| anyhow!("recherche texte de {} : {e}", g.label()))?;
        let epuise = hits.len() < k;
        let mut out: Vec<SearchHit<D>> = Vec::new();
        for (noeud, _bm25) in &hits {
            let Some(id) = corpus.par_noeud.get(noeud) else {
                continue;
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
            out.push(SearchHit {
                score: trouves as f64 / termes.len() as f64,
                document,
            });
            if out.len() == limite {
                break;
            }
        }
        if out.len() == limite || epuise || k >= total {
            return Ok(out);
        }
        k = k.saturating_mul(4).min(total);
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
        let id = MeiliClient::path_to_id(path);
        self.supprimer_si(Genre::Code, "SearchStore::delete_code", move |i, _| i == id)
            .await
    }

    async fn delete_code_for_project(&self, project_slug: &str) -> Result<()> {
        let slug = project_slug.to_string();
        self.supprimer_si(
            Genre::Code,
            "SearchStore::delete_code_for_project",
            move |_, c| egal(&c.slug, &slug),
        )
        .await
    }

    async fn delete_orphan_code_documents(&self) -> Result<()> {
        self.supprimer_si(
            Genre::Code,
            "SearchStore::delete_orphan_code_documents",
            |_, c| vide(&c.projet_id) || vide(&c.slug),
        )
        .await
    }

    async fn delete_all_code(&self) -> Result<()> {
        self.supprimer_si(Genre::Code, "SearchStore::delete_all_code", |_, _| true)
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
        let id = id.to_string();
        self.supprimer_si(
            Genre::Decision,
            "SearchStore::delete_decision",
            move |i, _| i == id,
        )
        .await
    }

    async fn delete_decisions_for_project(&self, project_slug: &str) -> Result<()> {
        let slug = project_slug.to_string();
        self.supprimer_si(
            Genre::Decision,
            "SearchStore::delete_decisions_for_project",
            move |_, c| egal(&c.slug, &slug),
        )
        .await
    }

    async fn delete_decisions_for_task(&self, task_id: &str) -> Result<()> {
        let tache = task_id.to_string();
        self.supprimer_si(
            Genre::Decision,
            "SearchStore::delete_decisions_for_task",
            move |_, c| egal(&c.a, &tache),
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
        let id = id.to_string();
        self.supprimer_si(Genre::Note, "SearchStore::delete_note", move |i, _| i == id)
            .await
    }

    async fn delete_notes_for_project(&self, project_slug: &str) -> Result<()> {
        let slug = project_slug.to_string();
        self.supprimer_si(
            Genre::Note,
            "SearchStore::delete_notes_for_project",
            move |_, c| egal(&c.slug, &slug),
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
                            // Absent: the no-op of Meilisearch (`get_document` fails, nothing indexed).
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

impl ObrainSearchStore {
    fn stats(&self, g: Genre) -> Result<IndexStats> {
        let total = self
            .graphe()
            .lire(|_, etat| Ok(etat.recherche.corpus(g).docs.len()))?;
        Ok(IndexStats {
            total_documents: total,
            is_indexing: false,
        })
    }
}
