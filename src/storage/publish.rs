//! Fail-closed publication of a migrated copy (Neo4j -> obrain): COMPONENTS ONLY.
//!
//! Implements the testable pieces of the publication protocol v6 with its amendments
//! v6.1 and v6.2 (the amendments win over the body). Nothing here is wired into the real
//! startup path: no caller in the crate builds a [`StagingHandle`] or calls [`publish`], and the
//! only [`SourceExclusion`] shipped outside the tests is [`NoExclusion`], which ALWAYS refuses.
//! As long as that holds, no real publication is possible.
//!
//! What lives here:
//! - [`SourceExclusion`] (§7) and its default [`NoExclusion`] (refusal).
//! - [`NoReplaceRename`] (§2.10) and [`RustixNoReplace`]: `renameat` WITHOUT replacement, raw
//!   errno kept, "not supported" is a named refusal with NO check-then-rename fallback.
//! - §1 preconditions on `dest` as a pure function over observed facts ([`evaluer_preconditions`]).
//! - The witness `.<name>.po-publish-pending` (§2.8): written tmp + fsync + rename + fsync of the
//!   parent, bound to source identity, nonce, manifest fingerprint and canonical `dest`.
//!   v6.2: a truncated / invalid witness is an AMBIGUOUS history => refusal + conservation.
//! - The manifest (§2.7): name, size, inode, mtime, ctime, hash of small files; walked by
//!   descriptors without following any link, compared by `stat` + hash only, never by opening
//!   the engine.
//! - [`classify`] (§3, §5, v6.1, v6.2): closed by default, every ambiguity is `Refuse`, adoption
//!   only when ALL its preconditions hold.
//! - [`decider_suppression`] (§6 + v6.1): NEVER deletes. Ownership of each entry and exclusion
//!   of third parties are not demonstrated, so every staging is CONSERVED with a named reason.
//! - [`publish`]: witness -> barrier -> `renameat` -> fsync of the parent; `dest` must be
//!   absent (no `rmdir(dest)`, replacing v6 §2.10 for this lot); no destructive cleanup.
//!
//! Not covered (see the PR): the canonical-key lock, free-space threshold, steps 2-6 (copy,
//! engine close, read-only checks, DONE/MARKER, mode/uid/ACL + recursive fsync), operator
//! commands (§5), rollback (§8) and any real source exclusion (§7: none exists today).
//!
//! This file depends only on std, `rustix`, `libc`, `sha2`, `hex`, `uuid`, `thiserror` and
//! `tracing` (no `crate::` item), so it can be compiled and mutated on its own.

use std::ffi::{CStr, CString, OsStr};
use std::fmt;
use std::fs;
use std::io::{self, Read, Write};
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path, PathBuf};

use rustix::fs::{AtFlags, FileType, Mode, OFlags};
use sha2::{Digest, Sha256};

/// Name of the ownership file written in a staging at its creation (§2.1).
pub const FICHIER_PROPRIETAIRE: &str = ".po-staging-owner";
/// Files strictly smaller than this are hashed in the manifest (§2.7); larger ones are
/// compared by size + inode + ctime only (mmap: the engine may write without touching mtime).
pub const SEUIL_PETIT_FICHIER: u64 = 1 << 20;
const TEMOIN_ENTETE: &str = "po-publish-pending v1";
const PROPRIETAIRE_ENTETE: &str = "po-staging-owner v1";
const TEMOIN_TAILLE_MAX: u64 = 64 * 1024;

// ---------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------

/// Name of an errno, for the messages (the raw value is always kept next to it).
pub fn nom_errno(errno: i32) -> &'static str {
    if errno == libc::EEXIST {
        "EEXIST"
    } else if errno == libc::ENOTEMPTY {
        "ENOTEMPTY"
    } else if errno == libc::EINVAL {
        "EINVAL"
    } else if errno == libc::ENOTSUP {
        "ENOTSUP"
    } else if errno == libc::EOPNOTSUPP {
        "EOPNOTSUPP"
    } else if errno == libc::ENOSYS {
        "ENOSYS"
    } else if errno == libc::EXDEV {
        "EXDEV"
    } else if errno == libc::EBUSY {
        "EBUSY"
    } else if errno == libc::ENOENT {
        "ENOENT"
    } else if errno == libc::EACCES {
        "EACCES"
    } else if errno == libc::EIO {
        "EIO"
    } else {
        "errno"
    }
}

/// Outcome of a failed `renameat` without replacement. The errno is the RAW value returned by
/// the system call, never re-derived.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum RenameError {
    /// EEXIST / ENOTEMPTY as reported by the primitive. Taken as a clean failure only once
    /// confirmed by `statat` (see the publication); otherwise the outcome is ambiguous.
    #[error("renommage sans remplacement refusé franchement : {} ({errno})", nom_errno(*errno))]
    Franc { errno: i32 },
    /// EINVAL / ENOTSUP / EOPNOTSUPP / ENOSYS: the primitive is not available here. Refusal,
    /// NO check-then-rename fallback.
    #[error(
        "renommage sans remplacement non supporté : {} ({errno}) ; aucun repli check-then-rename",
        nom_errno(*errno)
    )]
    NonSupporte { errno: i32 },
    /// Anything else (EXDEV, EBUSY, EIO, ...): the outcome is not provably "nothing happened".
    #[error("renommage sans remplacement : issue ambiguë {} ({errno}), tout est conservé", nom_errno(*errno))]
    Ambigu { errno: i32 },
}

impl RenameError {
    /// Classify a raw errno.
    pub fn depuis_errno(errno: i32) -> Self {
        if errno == libc::EEXIST || errno == libc::ENOTEMPTY {
            RenameError::Franc { errno }
        } else if errno == libc::EINVAL
            || errno == libc::ENOTSUP
            || errno == libc::EOPNOTSUPP
            || errno == libc::ENOSYS
        {
            RenameError::NonSupporte { errno }
        } else {
            RenameError::Ambigu { errno }
        }
    }

    /// The raw errno.
    pub fn errno(&self) -> i32 {
        match *self {
            RenameError::Franc { errno }
            | RenameError::NonSupporte { errno }
            | RenameError::Ambigu { errno } => errno,
        }
    }
}

/// Every refusal of this module is named. None of them deletes anything by itself.
#[derive(Debug, thiserror::Error)]
pub enum PublishError {
    #[error("source non gelée : aucune exclusion des écrivains démontrée")]
    SourceNonGelee,
    #[error("exclusion de la source rompue : {0}")]
    ExclusionRompue(String),
    #[error("identité de la source changée : {attendue} -> {observee}")]
    IdentiteSourceChangee { attendue: String, observee: String },
    #[error("identité de source inutilisable : {0:?}")]
    IdentiteSourceInvalide(String),
    #[error("dest doit être un chemin absolu, sans `.` ni `..`, avec un parent : {}", .0.display())]
    DestNonAbsolu(PathBuf),
    #[error("dest est un lien symbolique (refusé, même cassé) : {}", .0.display())]
    DestLienSymbolique(PathBuf),
    #[error("dest est un point de montage (st_dev {dev_dest} != parent {dev_parent}) : {}", .dest.display())]
    DestPointDeMontage {
        dest: PathBuf,
        dev_dest: u64,
        dev_parent: u64,
    },
    #[error("dest existe et n'est pas un répertoire : {}", .0.display())]
    DestPasUnRepertoire(PathBuf),
    #[error("dest existe et n'est pas vide : {}", .0.display())]
    DestNonVide(PathBuf),
    /// Replaces the `rmdir(dest)` of v6 §2.10 for this lot: without a demonstrated exclusion of
    /// third parties and ownership, an existing `dest`, EVEN EMPTY, is never removed (a third
    /// party could recreate it between the check and the rename and lose it to our `rmdir`).
    #[error("destination préexistante : publication refusée, rien supprimé : {}", .0.display())]
    DestPreexistante(PathBuf),
    #[error("exclusion des écrivains du répertoire non démontrée : {0}")]
    RepertoireNonExclu(String),
    #[error("le chemin du parent ne désigne plus le répertoire tenu : {}", .0.display())]
    ParentChange(PathBuf),
    #[error("objet publié différent de l'objet vérifié (rien supprimé, ne pas servir, état conservé pour l'opérateur) : {0}")]
    ObjetPublieDivergent(String),
    #[error("parent de dest absent ou pas un répertoire : {}", .0.display())]
    ParentAbsent(PathBuf),
    #[error("parent de dest non inscriptible : {} ({errno}) : {}", nom_errno(*errno), .parent.display())]
    ParentNonInscriptible { parent: PathBuf, errno: i32 },
    #[error("{} n'est pas sur le système de fichiers du parent (st_dev {dev} != {dev_parent})", .chemin.display())]
    AutreSystemeDeFichiers {
        chemin: PathBuf,
        dev: u64,
        dev_parent: u64,
    },
    #[error("traces d'une publication antérieure pour ce dest, conservées : {0}")]
    TracesExistantes(String),
    #[error("identité du staging changée, rien n'est fait : {0}")]
    StagingIdentiteChangee(String),
    #[error("dest donné ({}) différent du dest du staging ({})", .donne.display(), .attendu.display())]
    DestDifferent { donne: PathBuf, attendu: PathBuf },
    #[error(transparent)]
    Renommage(RenameError),
    /// The primitive reported a clean failure (or "not supported") that `statat` does not
    /// confirm: the outcome is AMBIGUOUS, everything is kept, the guard stays held.
    #[error(
        "renommage annoncé en échec {} ({errno}) mais non confirmé par statat : issue ambiguë, tout est conservé",
        nom_errno(*errno)
    )]
    FrancNonConfirme { errno: i32 },
    #[error("témoin invalide (historique ambigu, conservé) : {0}")]
    TemoinInvalide(String),
    #[error("témoin non lié à cette publication : {0}")]
    TemoinNonLie(String),
    #[error("témoin d'un autre nonce : attendu {attendu}, trouvé {trouve}")]
    TemoinAutreNonce { attendu: String, trouve: String },
    #[error("chemin canonique de dest changé : {} -> {}", .avant.display(), .apres.display())]
    CheminCanoniqueChange { avant: PathBuf, apres: PathBuf },
    #[error("manifeste divergent : {0}")]
    ManifesteDivergent(String),
    #[error("contenu refusé dans le manifeste : {0}")]
    ManifesteRefuse(String),
    #[error("manifeste non fiable sous concurrence : {0}")]
    ManifesteNonFiable(String),
    /// Returned after the rename: `dest` is MAYBE published (durability or exclusion not
    /// confirmed). Keep everything, do not serve; the next start applies §3.
    #[error("peut-être publié, durabilité non confirmée (conservé, ne pas servir) : {0}")]
    PublieDurabiliteNonConfirmee(String),
    /// Returned after a confirmed durable publication when the witness could not be removed
    /// (or its removal made durable): published, but not to be served until §3 runs.
    #[error("publié durable, non servable (témoin non retiré durablement) : {0}")]
    PublieDurableNonServable(String),
    #[error("{etape} : {} : {source}", .chemin.display())]
    Io {
        etape: &'static str,
        chemin: PathBuf,
        #[source]
        source: io::Error,
    },
}

fn io_err<'a>(
    etape: &'static str,
    chemin: &'a Path,
) -> impl FnOnce(io::Error) -> PublishError + 'a {
    move |source| PublishError::Io {
        etape,
        chemin: chemin.to_path_buf(),
        source,
    }
}

fn rx_err<'a>(
    etape: &'static str,
    chemin: &'a Path,
) -> impl FnOnce(rustix::io::Errno) -> PublishError + 'a {
    move |e| PublishError::Io {
        etape,
        chemin: chemin.to_path_buf(),
        source: io::Error::from_raw_os_error(e.raw_os_error()),
    }
}

// ---------------------------------------------------------------------------------------------
// §7 Source exclusion
// ---------------------------------------------------------------------------------------------

/// Identity of the source being copied. Must not contain a line break (it is written in the
/// witness and in the ownership file).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SourceId(pub String);

impl SourceId {
    fn valider(&self) -> Result<(), PublishError> {
        if self.0.is_empty() || self.0.contains(['\n', '\r']) {
            return Err(PublishError::IdentiteSourceInvalide(self.0.clone()));
        }
        Ok(())
    }
}

impl fmt::Display for SourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Proof that the writers of the source are excluded, from `acquire` to the barrier and the
/// switch (and beyond: v6.1). `generation` lets an implementation detect "released then taken
/// again", which v6.2 says is NOT a continuous exclusion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExclusionGuard {
    pub source: SourceId,
    pub generation: u64,
}

/// §7. An implementation must prove a CONTINUOUS exclusion of every writer of the source.
/// Reading a status or a counter is only a hint and is not such a proof.
pub trait SourceExclusion {
    /// Take the exclusion before anything is written.
    fn acquire(&self) -> Result<ExclusionGuard, PublishError>;
    /// Re-check at the barrier that the exclusion taken by `guard` has been held WITHOUT
    /// interruption up to now.
    fn verify_at_barrier(&self, guard: &ExclusionGuard) -> Result<(), PublishError>;
    /// Current identity of the source.
    fn identity(&self) -> Result<SourceId, PublishError>;
}

/// Default implementation: no mechanism excludes the writers of the source today (§7), so it
/// ALWAYS refuses, and nothing can be published.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoExclusion;

impl SourceExclusion for NoExclusion {
    fn acquire(&self) -> Result<ExclusionGuard, PublishError> {
        Err(PublishError::SourceNonGelee)
    }
    fn verify_at_barrier(&self, _guard: &ExclusionGuard) -> Result<(), PublishError> {
        Err(PublishError::SourceNonGelee)
    }
    fn identity(&self) -> Result<SourceId, PublishError> {
        Err(PublishError::SourceNonGelee)
    }
}

/// Barrier check: the exclusion is still the one of `guard`, and the source is still the same.
fn verifier_exclusion(
    excl: &dyn SourceExclusion,
    guard: &ExclusionGuard,
) -> Result<(), PublishError> {
    excl.verify_at_barrier(guard)?;
    let observee = excl.identity()?;
    if observee != guard.source {
        return Err(PublishError::IdentiteSourceChangee {
            attendue: guard.source.0.clone(),
            observee: observee.0,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// §2.10 Rename without replacement, directory exclusion
// ---------------------------------------------------------------------------------------------

/// `renameat` WITHOUT replacement (`RENAME_EXCL` macOS / `RENAME_NOREPLACE` Linux), both names
/// relative to the SAME directory descriptor (the parent of `dest`, opened once and kept).
///
/// NOREPLACE protects the DESTINATION against being overwritten. It does NOT protect the
/// identity of the SOURCE: a same-uid third party can substitute the source name before the
/// call (there is no "rename this descriptor"). Hence the publication also requires an
/// [`ExclusionRepertoire`] and a dev/ino check right after the rename.
pub trait NoReplaceRename {
    fn rename_noreplace(
        &self,
        dir: BorrowedFd<'_>,
        from: &OsStr,
        to: &OsStr,
    ) -> Result<(), RenameError>;
}

/// Real primitive: `rustix::fs::renameat_with(dir, from, dir, to, RenameFlags::NOREPLACE)`. The
/// raw errno is kept; an unsupported primitive is an error, there is no fallback of any kind.
#[derive(Debug, Default, Clone, Copy)]
pub struct RustixNoReplace;

impl NoReplaceRename for RustixNoReplace {
    fn rename_noreplace(
        &self,
        dir: BorrowedFd<'_>,
        from: &OsStr,
        to: &OsStr,
    ) -> Result<(), RenameError> {
        rustix::fs::renameat_with(dir, from, dir, to, rustix::fs::RenameFlags::NOREPLACE)
            .map_err(|e| RenameError::depuis_errno(e.raw_os_error()))
    }
}

/// Guard of an exclusion of every OTHER writer of a DIRECTORY (the parent of `dest`, hence the
/// staging in it), bound to the IDENTITY (dev/ino) of the parent descriptor opened at the
/// acquisition. `jeton` lets an implementation detect "lost then re-acquired", which is NOT a
/// continuous exclusion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GardeRepertoire {
    dev: u64,
    ino: u64,
    jeton: u64,
}

impl GardeRepertoire {
    /// Private constructor: only implementations of [`ExclusionRepertoire`] written in this
    /// module can issue a guard. No real implementation exists yet (only the test fakes use it).
    #[cfg_attr(not(test), allow(dead_code))]
    fn nouvelle(parent: &StatLite, jeton: u64) -> Self {
        GardeRepertoire {
            dev: parent.dev,
            ino: parent.ino,
            jeton,
        }
    }
    pub fn jeton(&self) -> u64 {
        self.jeton
    }
    pub fn identite(&self) -> (u64, u64) {
        (self.dev, self.ino)
    }
}

/// Exclusion of the writers of a DIRECTORY. Distinct from [`SourceExclusion`] (writers of the
/// Neo4j source): excluding the source's writers proves nothing about who writes in the file
/// system. It is acquired BEFORE the staging is created and must be held continuously through
/// the creation, the copy, the manifest, the barrier, the rename, up to AFTER the fsync of the
/// parent. Every rename and every deletion of this module requires a continuous guard. An
/// implementation must prove it; none exists today.
///
/// Contract: dropping the implementation must NOT release a guard it issued; only `relacher`
/// releases (a guard may have to stay held after a "maybe published" outcome). `relacher` may
/// be called while unwinding a panic and must not panic (a panic there is swallowed, logged,
/// and the guard is then considered released).
pub trait ExclusionRepertoire {
    /// Acquire the exclusion for the directory `parent` (descriptor and its `fstat`).
    fn acquerir(
        &self,
        parent: BorrowedFd<'_>,
        parent_stat: &StatLite,
    ) -> Result<GardeRepertoire, PublishError>;
    /// `Ok` only if the exclusion of `garde` has been held WITHOUT interruption since
    /// `acquerir`. Lost then re-acquired => `Err`, even if held again now.
    fn verifier_continuite(&self, garde: &GardeRepertoire) -> Result<(), PublishError>;
    /// Release the exclusion.
    fn relacher(&self, garde: &GardeRepertoire);
}

/// Default implementation: ALWAYS refuses (no mechanism excludes same-uid writers today).
#[derive(Debug, Default, Clone, Copy)]
pub struct RepertoireNonExclu;

const PAS_D_EXCLUSION_REPERTOIRE: &str =
    "aucun mécanisme n'exclut les autres écrivains du répertoire";

impl ExclusionRepertoire for RepertoireNonExclu {
    fn acquerir(
        &self,
        _parent: BorrowedFd<'_>,
        _parent_stat: &StatLite,
    ) -> Result<GardeRepertoire, PublishError> {
        Err(PublishError::RepertoireNonExclu(
            PAS_D_EXCLUSION_REPERTOIRE.into(),
        ))
    }
    fn verifier_continuite(&self, _garde: &GardeRepertoire) -> Result<(), PublishError> {
        Err(PublishError::RepertoireNonExclu(
            PAS_D_EXCLUSION_REPERTOIRE.into(),
        ))
    }
    fn relacher(&self, _garde: &GardeRepertoire) {}
}

/// The parent of `dest`: its path, its descriptor held, its identity, and the directory
/// exclusion with its guard. Every operation that writes, renames or deletes in it checks
/// `ContexteParent::verifier` first.
struct ContexteParent<'a> {
    chemin: &'a Path,
    fd: BorrowedFd<'a>,
    stat: StatLite,
    excl: &'a dyn ExclusionRepertoire,
    garde: &'a GardeRepertoire,
}

impl ContexteParent<'_> {
    /// Continuous guard, bound to this parent, and the path and descriptor still designate it.
    fn verifier(&self) -> Result<(), PublishError> {
        self.excl.verifier_continuite(self.garde)?;
        let fst = StatLite::depuis_rustix(
            &rustix::fs::fstat(self.fd).map_err(rx_err("fstat du parent", self.chemin))?,
        );
        if self.garde.identite() != (self.stat.dev, self.stat.ino)
            || (fst.dev, fst.ino) != (self.stat.dev, self.stat.ino)
        {
            return Err(PublishError::ParentChange(self.chemin.to_path_buf()));
        }
        verifier_parent(self.chemin, &self.stat)
    }
}

/// fsync of a directory through its descriptor, injectable so that a failure can be tested.
pub trait SyncRepertoire {
    fn fsync_repertoire(&self, dir: BorrowedFd<'_>) -> io::Result<()>;
}

/// Real fsync of a directory (`File::sync_all`: `F_FULLFSYNC` on macOS).
#[derive(Debug, Default, Clone, Copy)]
pub struct SyncReel;

impl SyncRepertoire for SyncReel {
    fn fsync_repertoire(&self, dir: BorrowedFd<'_>) -> io::Result<()> {
        fs::File::from(dir.try_clone_to_owned()?).sync_all()
    }
}

/// Open the parent of `dest` once (`O_DIRECTORY`), checking that the path still resolves to the
/// opened directory.
pub fn ouvrir_parent(dest: &Path) -> Result<(OwnedFd, StatLite), PublishError> {
    verifier_chemin_dest(dest)?;
    let parent = dest.parent().expect("checked above");
    let fd = rustix::fs::open(
        parent,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(rx_err("ouverture du parent", parent))?;
    let st = StatLite::depuis_rustix(
        &rustix::fs::fstat(&fd).map_err(rx_err("fstat du parent", parent))?,
    );
    verifier_parent(parent, &st)?;
    Ok((fd, st))
}

/// The parent path still designates the directory whose descriptor is held.
fn verifier_parent(parent: &Path, st: &StatLite) -> Result<(), PublishError> {
    let m = StatLite::depuis(&fs::metadata(parent).map_err(io_err("stat du parent", parent))?);
    if (m.dev, m.ino) != (st.dev, st.ino) {
        return Err(PublishError::ParentChange(parent.to_path_buf()));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// §1 Preconditions
// ---------------------------------------------------------------------------------------------

/// The `stat` fields this module uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StatLite {
    pub genre: Genre,
    pub dev: u64,
    pub ino: u64,
    pub nlink: u64,
    pub mode: u32,
    pub uid: u32,
    pub gid: u32,
    pub taille: u64,
    pub mtime: (i64, i64),
    pub ctime: (i64, i64),
}

/// File type, as seen WITHOUT following links.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Genre {
    Repertoire,
    Fichier,
    Lien,
    Autre,
}

impl StatLite {
    pub fn depuis(m: &fs::Metadata) -> Self {
        let ft = m.file_type();
        let genre = if ft.is_symlink() {
            Genre::Lien
        } else if ft.is_dir() {
            Genre::Repertoire
        } else if ft.is_file() {
            Genre::Fichier
        } else {
            Genre::Autre
        };
        StatLite {
            genre,
            dev: m.dev(),
            ino: m.ino(),
            nlink: m.nlink(),
            mode: m.mode(),
            uid: m.uid(),
            gid: m.gid(),
            taille: m.size(),
            mtime: (m.mtime(), m.mtime_nsec()),
            ctime: (m.ctime(), m.ctime_nsec()),
        }
    }

    fn depuis_rustix(st: &rustix::fs::Stat) -> Self {
        let genre = match FileType::from_raw_mode(st.st_mode as rustix::fs::RawMode) {
            FileType::Directory => Genre::Repertoire,
            FileType::RegularFile => Genre::Fichier,
            FileType::Symlink => Genre::Lien,
            _ => Genre::Autre,
        };
        #[allow(clippy::unnecessary_cast)]
        StatLite {
            genre,
            dev: st.st_dev as u64,
            ino: st.st_ino as u64,
            nlink: st.st_nlink as u64,
            mode: st.st_mode as u32,
            uid: st.st_uid as u32,
            gid: st.st_gid as u32,
            taille: st.st_size as u64,
            mtime: (st.st_mtime as i64, st.st_mtime_nsec as i64),
            ctime: (st.st_ctime as i64, st.st_ctime_nsec as i64),
        }
    }
}

/// Facts observed about `dest` and its parent. Gathered by [`collecter_faits_dest`], judged by
/// the pure [`evaluer_preconditions`] (so that, e.g., a mount point can be simulated).
#[derive(Debug, Clone)]
pub struct FaitsDest {
    pub dest: PathBuf,
    /// `lstat(dest)`; `None` = absent.
    pub dest_stat: Option<StatLite>,
    /// Only for a real directory (not a link): is it empty?
    pub dest_vide: Option<bool>,
    /// `stat(parent)`; `None` = absent.
    pub parent_stat: Option<StatLite>,
    /// `access(parent, W_OK|X_OK)`: `Ok` or the raw errno.
    pub parent_inscriptible: Result<(), i32>,
}

/// `dest` absolute, with a name, without `.` or `..` components.
pub fn verifier_chemin_dest(dest: &Path) -> Result<(), PublishError> {
    // Raw segments: `Path::components` silently drops interior `.` components.
    let segments_propres = dest
        .as_os_str()
        .as_bytes()
        .split(|b| *b == b'/')
        .all(|seg| seg != b"." && seg != b"..");
    let propre = dest.is_absolute()
        && segments_propres
        && dest.file_name().is_some()
        && dest.parent().is_some()
        && dest
            .components()
            .all(|c| matches!(c, Component::RootDir | Component::Normal(_)));
    if propre {
        Ok(())
    } else {
        Err(PublishError::DestNonAbsolu(dest.to_path_buf()))
    }
}

/// Gather the facts about `dest` (no writes). Refuses a relative `dest` before touching anything.
pub fn collecter_faits_dest(dest: &Path) -> Result<FaitsDest, PublishError> {
    verifier_chemin_dest(dest)?;
    let parent = dest.parent().expect("checked above");
    let dest_stat = match fs::symlink_metadata(dest) {
        Ok(m) => Some(StatLite::depuis(&m)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => None,
        Err(e) => return Err(io_err("lstat de dest", dest)(e)),
    };
    let dest_vide = match dest_stat {
        Some(s) if s.genre == Genre::Repertoire => Some(
            fs::read_dir(dest)
                .map_err(io_err("lecture de dest", dest))?
                .next()
                .is_none(),
        ),
        _ => None,
    };
    let parent_stat = match fs::metadata(parent) {
        Ok(m) => Some(StatLite::depuis(&m)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => None,
        Err(e) => return Err(io_err("stat du parent", parent)(e)),
    };
    let parent_inscriptible = rustix::fs::access(
        parent,
        rustix::fs::Access::WRITE_OK | rustix::fs::Access::EXEC_OK,
    )
    .map_err(|e| e.raw_os_error());
    Ok(FaitsDest {
        dest: dest.to_path_buf(),
        dest_stat,
        dest_vide,
        parent_stat,
        parent_inscriptible,
    })
}

/// §1, pure: absolute, not a link (even broken), not a mount point (`st_dev` != parent), parent
/// writable, and ABSENT: an existing `dest`, even an empty directory, is refused for this lot
/// (see [`PublishError::DestPreexistante`]).
pub fn evaluer_preconditions(f: &FaitsDest) -> Result<(), PublishError> {
    verifier_chemin_dest(&f.dest)?;
    let parent = f.dest.parent().expect("checked above").to_path_buf();
    let parent_stat = match f.parent_stat {
        Some(p) if p.genre == Genre::Repertoire => p,
        _ => return Err(PublishError::ParentAbsent(parent)),
    };
    if let Err(errno) = f.parent_inscriptible {
        return Err(PublishError::ParentNonInscriptible { parent, errno });
    }
    let Some(d) = f.dest_stat else {
        return Ok(());
    };
    match d.genre {
        Genre::Lien => return Err(PublishError::DestLienSymbolique(f.dest.clone())),
        Genre::Repertoire => {}
        Genre::Fichier | Genre::Autre => {
            return Err(PublishError::DestPasUnRepertoire(f.dest.clone()))
        }
    }
    if d.dev != parent_stat.dev {
        return Err(PublishError::DestPointDeMontage {
            dest: f.dest.clone(),
            dev_dest: d.dev,
            dev_parent: parent_stat.dev,
        });
    }
    match f.dest_vide {
        Some(true) => Err(PublishError::DestPreexistante(f.dest.clone())),
        _ => Err(PublishError::DestNonVide(f.dest.clone())),
    }
}

/// §1 "same file system": the staging must be on the parent's device.
pub fn verifier_meme_systeme(
    chemin: &Path,
    stat: &StatLite,
    parent: &StatLite,
) -> Result<(), PublishError> {
    if stat.dev != parent.dev {
        return Err(PublishError::AutreSystemeDeFichiers {
            chemin: chemin.to_path_buf(),
            dev: stat.dev,
            dev_parent: parent.dev,
        });
    }
    Ok(())
}

/// [`collecter_faits_dest`] then [`evaluer_preconditions`].
pub fn verifier_preconditions(dest: &Path) -> Result<(), PublishError> {
    evaluer_preconditions(&collecter_faits_dest(dest)?)
}

/// Canonical `dest`: canonical parent + name (`dest` itself may not exist).
pub fn dest_canonique(dest: &Path) -> Result<PathBuf, PublishError> {
    verifier_chemin_dest(dest)?;
    let parent = dest.parent().expect("checked above");
    let nom = dest.file_name().expect("checked above");
    Ok(fs::canonicalize(parent)
        .map_err(io_err("chemin canonique du parent", parent))?
        .join(nom))
}

fn nom_dest(dest: &Path) -> &OsStr {
    dest.file_name()
        .expect("dest checked by verifier_chemin_dest")
}

/// Sibling names derived from `dest` (§2.1, §2.8).
pub fn prefixe_staging(dest: &Path) -> Vec<u8> {
    let mut v = b".".to_vec();
    v.extend_from_slice(nom_dest(dest).as_bytes());
    v.extend_from_slice(b".po-staging-");
    v
}

pub fn chemin_temoin(dest: &Path) -> PathBuf {
    let mut nom = b".".to_vec();
    nom.extend_from_slice(nom_dest(dest).as_bytes());
    nom.extend_from_slice(b".po-publish-pending");
    dest.with_file_name(OsStr::from_bytes(&nom))
}

fn chemin_temoin_tmp(dest: &Path) -> PathBuf {
    let t = chemin_temoin(dest);
    let mut nom = t.file_name().expect("has a name").as_bytes().to_vec();
    nom.extend_from_slice(b".tmp");
    t.with_file_name(OsStr::from_bytes(&nom))
}

/// Traces of a (previous or current) publication for `dest` found in the parent.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Traces {
    pub stagings: Vec<PathBuf>,
    pub temoin: bool,
    pub temoin_tmp: bool,
}

/// Read-only scan of the parent of `dest` for stagings and witnesses (§1, §5).
pub fn scanner_traces(dest: &Path) -> Result<Traces, PublishError> {
    verifier_chemin_dest(dest)?;
    let parent = dest.parent().expect("checked above");
    let prefixe = prefixe_staging(dest);
    let temoin = chemin_temoin(dest);
    let temoin_tmp = chemin_temoin_tmp(dest);
    let mut t = Traces::default();
    for e in fs::read_dir(parent).map_err(io_err("lecture du parent", parent))? {
        let e = e.map_err(io_err("lecture du parent", parent))?;
        let nom = e.file_name();
        if nom.as_bytes().starts_with(&prefixe) {
            t.stagings.push(e.path());
        } else if Some(nom.as_os_str()) == temoin.file_name() {
            t.temoin = true;
        } else if Some(nom.as_os_str()) == temoin_tmp.file_name() {
            t.temoin_tmp = true;
        }
    }
    t.stagings.sort();
    Ok(t)
}

// ---------------------------------------------------------------------------------------------
// §2.7 Manifest
// ---------------------------------------------------------------------------------------------

/// One entry of the manifest (everything below the root).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntreeManifeste {
    pub chemin: String,
    pub genre: Genre,
    pub taille: u64,
    pub ino: u64,
    pub mtime: (i64, i64),
    pub ctime: (i64, i64),
    /// sha256 for a regular file smaller than the threshold.
    pub hash: Option<String>,
}

/// The root of the manifest: mode, uid, gid and inode. Not its ctime/mtime (the rename may
/// change them).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RacineManifeste {
    pub mode: u32,
    pub uid: u32,
    pub gid: u32,
    pub ino: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifeste {
    pub racine: RacineManifeste,
    pub entrees: Vec<EntreeManifeste>,
}

/// Maximum number of nested directory levels below the root: 64 levels are walked, a 65th is
/// refused.
pub const PROFONDEUR_MAX: usize = 64;

impl Manifeste {
    /// Fingerprint bound in the witness: sha256 of a canonical serialisation.
    pub fn empreinte(&self) -> String {
        let mut h = Sha256::new();
        let r = &self.racine;
        h.update(format!("r\t{}\t{}\t{}\t{}\n", r.mode, r.uid, r.gid, r.ino).as_bytes());
        for e in &self.entrees {
            let g = match e.genre {
                Genre::Repertoire => "d",
                Genre::Fichier => "f",
                Genre::Lien => "l",
                Genre::Autre => "o",
            };
            h.update(
                format!(
                    "{g}\t{}\t{}\t{}\t{}.{}\t{}.{}\t{}\n",
                    e.chemin,
                    e.taille,
                    e.ino,
                    e.mtime.0,
                    e.mtime.1,
                    e.ctime.0,
                    e.ctime.1,
                    e.hash.as_deref().unwrap_or("-")
                )
                .as_bytes(),
            );
        }
        hex::encode(h.finalize())
    }
}

/// Build the manifest of ALL the content under `racine` (stat + hash; never opens the engine).
///
/// Walked BY DESCRIPTORS, never by path: the root is opened once with `O_DIRECTORY |
/// O_NOFOLLOW`, each entry is `fstatat(.., AT_SYMLINK_NOFOLLOW)`-ed relative to its directory
/// descriptor, subdirectories are entered with `openat(.., O_DIRECTORY | O_NOFOLLOW)` and small
/// files hashed through `openat(.., O_NOFOLLOW)`; the identity (dev/ino) seen by the stat must
/// be the one of the opened descriptor. Refused, never a result: any divergence ("manifeste non
/// fiable sous concurrence"), a link, a special file (fifo, socket, device), a regular file
/// with `nlink > 1` (an outside hard link keeps write access to the published inode), an entry
/// on another device than the root (mount point), more than [`PROFONDEUR_MAX`] nested directory levels below the root. The
/// components ABOVE `racine` are resolved by path once (the publication uses the descriptor of
/// the [`StagingHandle`]).
///
/// Requires a continuous directory guard on the parent of `racine`, checked before and after
/// the walk: without it, a writer could alter the content while it is being recorded.
/// No caller outside the tests yet (the §3 comparison of `dest` at startup is not wired):
/// hence `allow(dead_code)` outside the tests.
#[cfg_attr(not(test), allow(dead_code))]
fn construire_manifeste(
    racine: &Path,
    seuil: u64,
    ctx: &ContexteParent<'_>,
) -> Result<Manifeste, PublishError> {
    if racine.parent() != Some(ctx.chemin) {
        return Err(PublishError::ManifesteNonFiable(format!(
            "{} n'est pas dans le répertoire gardé {}",
            racine.display(),
            ctx.chemin.display()
        )));
    }
    ctx.verifier()?;
    let fd = rustix::fs::open(
        racine,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(|e| PublishError::ManifesteNonFiable(format!("racine {} : {e}", racine.display())))?;
    let m = manifeste_depuis_fd(&fd, seuil, &|_| {})?;
    ctx.verifier()?;
    Ok(m)
}

/// Same, from an already opened directory descriptor. `crochet` is called with the relative
/// path of each entry between its stat and its opening (test seam for a concurrent
/// substitution; a no-op otherwise).
fn manifeste_depuis_fd(
    fd: &OwnedFd,
    seuil: u64,
    crochet: &dyn Fn(&str),
) -> Result<Manifeste, PublishError> {
    manifeste_interne(fd, seuil, crochet, None)
}

/// `dev_racine`: test seam to simulate a root on another device (mount point below it).
fn manifeste_interne(
    fd: &OwnedFd,
    seuil: u64,
    crochet: &dyn Fn(&str),
    dev_racine: Option<u64>,
) -> Result<Manifeste, PublishError> {
    let st = rustix::fs::fstat(fd)
        .map_err(|e| PublishError::ManifesteNonFiable(format!("fstat de la racine : {e}")))?;
    let racine = StatLite::depuis_rustix(&st);
    let mut entrees = Vec::new();
    let dev = dev_racine.unwrap_or(racine.dev);
    parcourir_fd(fd, "", seuil, crochet, &mut entrees, dev, 0)?;
    entrees.sort_by(|a, b| a.chemin.cmp(&b.chemin));
    Ok(Manifeste {
        racine: RacineManifeste {
            mode: racine.mode,
            uid: racine.uid,
            gid: racine.gid,
            ino: racine.ino,
        },
        entrees,
    })
}

fn parcourir_fd(
    fd: &OwnedFd,
    prefixe: &str,
    seuil: u64,
    crochet: &dyn Fn(&str),
    out: &mut Vec<EntreeManifeste>,
    dev_racine: u64,
    profondeur: usize,
) -> Result<(), PublishError> {
    let non_fiable = |m: String| PublishError::ManifesteNonFiable(m);
    if profondeur > PROFONDEUR_MAX {
        return Err(PublishError::ManifesteRefuse(format!(
            "profondeur supérieure à {PROFONDEUR_MAX} : {prefixe}"
        )));
    }
    let mut noms = Vec::new();
    {
        let mut dir = rustix::fs::Dir::read_from(fd)
            .map_err(|e| non_fiable(format!("readdir {prefixe:?} : {e}")))?;
        while let Some(e) = dir.read() {
            let e = e.map_err(|e| non_fiable(format!("readdir {prefixe:?} : {e}")))?;
            let nom = e.file_name();
            if nom.to_bytes() != b"." && nom.to_bytes() != b".." {
                noms.push(nom.to_owned());
            }
        }
    }
    for nom in noms {
        let nom_str = nom
            .to_str()
            .ok()
            .filter(|s| !s.contains(['\t', '\n', '\r', '/']))
            .ok_or_else(|| PublishError::ManifesteRefuse(format!("nom illisible : {nom:?}")))?;
        let rel = if prefixe.is_empty() {
            nom_str.to_string()
        } else {
            format!("{prefixe}/{nom_str}")
        };
        let st = StatLite::depuis_rustix(
            &rustix::fs::statat(fd, nom.as_c_str(), AtFlags::SYMLINK_NOFOLLOW)
                .map_err(|e| non_fiable(format!("statat {rel} : {e}")))?,
        );
        crochet(&rel);
        if st.dev != dev_racine {
            return Err(PublishError::ManifesteRefuse(format!(
                "autre système de fichiers (point de montage ?) st_dev {} != {dev_racine} : {rel}",
                st.dev
            )));
        }
        let ouvrir = |flags: OFlags| -> Result<OwnedFd, PublishError> {
            let o = rustix::fs::openat(
                fd,
                nom.as_c_str(),
                flags | OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK,
                Mode::empty(),
            )
            .map_err(|e| non_fiable(format!("openat {rel} : {e}")))?;
            let fst = StatLite::depuis_rustix(
                &rustix::fs::fstat(&o).map_err(|e| non_fiable(format!("fstat {rel} : {e}")))?,
            );
            if (fst.dev, fst.ino, fst.genre) != (st.dev, st.ino, st.genre) {
                return Err(non_fiable(format!(
                    "{rel} substitué entre stat et ouverture"
                )));
            }
            Ok(o)
        };
        let hash = match st.genre {
            Genre::Lien | Genre::Autre => {
                return Err(PublishError::ManifesteRefuse(format!(
                    "lien symbolique ou fichier spécial (fifo, socket, périphérique) : {rel}"
                )))
            }
            Genre::Fichier if st.nlink > 1 => {
                return Err(PublishError::ManifesteRefuse(format!(
                    "lien dur (nlink {}) : {rel}",
                    st.nlink
                )))
            }
            Genre::Fichier if st.taille < seuil => Some(hash_fd(ouvrir(OFlags::empty())?, &rel)?),
            _ => None,
        };
        let sous = if st.genre == Genre::Repertoire {
            Some(ouvrir(OFlags::DIRECTORY)?)
        } else {
            None
        };
        out.push(EntreeManifeste {
            chemin: rel.clone(),
            genre: st.genre,
            taille: st.taille,
            ino: st.ino,
            mtime: st.mtime,
            ctime: st.ctime,
            hash,
        });
        if let Some(sfd) = sous {
            parcourir_fd(&sfd, &rel, seuil, crochet, out, dev_racine, profondeur + 1)?;
        }
    }
    Ok(())
}

fn hash_fd(fd: OwnedFd, rel: &str) -> Result<String, PublishError> {
    let mut f = fs::File::from(fd);
    let mut h = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = f
            .read(&mut buf)
            .map_err(|e| PublishError::ManifesteNonFiable(format!("lecture {rel} : {e}")))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex::encode(h.finalize()))
}

/// Compare two manifests; the first difference is named.
pub fn comparer_manifestes(attendu: &Manifeste, observe: &Manifeste) -> Result<(), PublishError> {
    if observe == attendu {
        return Ok(());
    }
    let premier = attendu
        .entrees
        .iter()
        .zip(observe.entrees.iter())
        .find(|(a, b)| a != b)
        .map(|(a, b)| format!("{} : {a:?} != {b:?}", a.chemin))
        .unwrap_or_else(|| {
            format!(
                "{} entrées attendues, {} observées",
                attendu.entrees.len(),
                observe.entrees.len()
            )
        });
    Err(PublishError::ManifesteDivergent(premier))
}

/// Compare a manifest recorded earlier with the content now under `racine`, by stat + hash only.
/// No caller outside the tests yet (same reason as `construire_manifeste`).
#[cfg_attr(not(test), allow(dead_code))]
fn comparer_manifeste(
    attendu: &Manifeste,
    racine: &Path,
    seuil: u64,
    ctx: &ContexteParent<'_>,
) -> Result<(), PublishError> {
    comparer_manifestes(attendu, &construire_manifeste(racine, seuil, ctx)?)
}

// ---------------------------------------------------------------------------------------------
// §2.8 Witness
// ---------------------------------------------------------------------------------------------

/// Content of the witness `.<name>.po-publish-pending`.
///
/// Its `controle` line is an UNKEYED sha256 of the rest: it detects truncation and corruption,
/// NOT forgery. Anyone able to write in the parent can write a well-formed witness, so the
/// witness does NOT prove its origin; [`classify`] therefore also binds the nonce of the
/// witness to the `.po-staging-owner` found in `dest` (which does NOT protect against a writer
/// of the same uid, able to write both), and adoption still requires the other proofs
/// (DONE + MARKER, manifest, continuous exclusion).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Temoin {
    pub source: SourceId,
    pub nonce: String,
    pub manifeste: String,
    pub dest: PathBuf,
}

impl Temoin {
    fn corps(&self) -> Result<String, PublishError> {
        self.source.valider()?;
        let dest = self
            .dest
            .to_str()
            .filter(|s| !s.contains(['\n', '\r']))
            .ok_or_else(|| PublishError::DestNonAbsolu(self.dest.clone()))?;
        if self.nonce.contains(['\n', '\r']) || self.manifeste.contains(['\n', '\r']) {
            return Err(PublishError::TemoinInvalide("champ multiligne".into()));
        }
        Ok(format!(
            "{TEMOIN_ENTETE}\nsource={}\nnonce={}\nmanifeste={}\ndest={dest}\n",
            self.source.0, self.nonce, self.manifeste
        ))
    }

    /// Serialised form, closed by a checksum of everything before it (detects truncation).
    pub fn serialiser(&self) -> Result<String, PublishError> {
        let corps = self.corps()?;
        let controle = hex::encode(Sha256::digest(corps.as_bytes()));
        Ok(format!("{corps}controle={controle}\n"))
    }

    /// Strict parsing: any deviation is `TemoinInvalide`.
    pub fn analyser(texte: &str) -> Result<Temoin, PublishError> {
        let invalide = |m: &str| PublishError::TemoinInvalide(m.to_string());
        let Some(sans_fin) = texte.strip_suffix('\n') else {
            return Err(invalide("pas de fin de ligne finale (tronqué ?)"));
        };
        let lignes: Vec<&str> = sans_fin.split('\n').collect();
        if lignes.len() != 6 || lignes[0] != TEMOIN_ENTETE {
            return Err(invalide("structure inattendue"));
        }
        let champ = |i: usize, cle: &str| {
            lignes[i]
                .strip_prefix(cle)
                .and_then(|r| r.strip_prefix('='))
                .ok_or_else(|| invalide(cle))
        };
        let t = Temoin {
            source: SourceId(champ(1, "source")?.to_string()),
            nonce: champ(2, "nonce")?.to_string(),
            manifeste: champ(3, "manifeste")?.to_string(),
            dest: PathBuf::from(champ(4, "dest")?),
        };
        let controle = champ(5, "controle")?;
        let attendu = hex::encode(Sha256::digest(t.corps()?.as_bytes()));
        if controle != attendu {
            return Err(invalide("somme de contrôle fausse"));
        }
        Ok(t)
    }

    /// Is this witness the one of THIS publication?
    pub fn valider(&self, attendu: &Temoin) -> Result<(), PublishError> {
        if self.nonce != attendu.nonce {
            return Err(PublishError::TemoinAutreNonce {
                attendu: attendu.nonce.clone(),
                trouve: self.nonce.clone(),
            });
        }
        if self.dest != attendu.dest {
            return Err(PublishError::CheminCanoniqueChange {
                avant: self.dest.clone(),
                apres: attendu.dest.clone(),
            });
        }
        if self.source != attendu.source {
            return Err(PublishError::TemoinNonLie(format!(
                "source {} != {}",
                self.source, attendu.source
            )));
        }
        if self.manifeste != attendu.manifeste {
            return Err(PublishError::TemoinNonLie(format!(
                "manifeste {} != {}",
                self.manifeste, attendu.manifeste
            )));
        }
        Ok(())
    }
}

/// Write the witness relative to the parent descriptor: `.tmp` created exclusively + fsync, rename
/// WITHOUT replacement, fsync of the parent. Requires a continuous directory guard (the `.tmp`
/// name could otherwise be substituted before the rename). On any failure nothing is removed.
fn ecrire_temoin(
    dest: &Path,
    ctx: &ContexteParent<'_>,
    temoin: &Temoin,
    ren: &dyn NoReplaceRename,
    sync: &dyn SyncRepertoire,
) -> Result<PathBuf, PublishError> {
    let contenu = temoin.serialiser()?;
    let parent = ctx.chemin;
    let parent_fd = ctx.fd;
    if dest.parent() != Some(parent) {
        return Err(PublishError::ParentChange(parent.to_path_buf()));
    }
    ctx.verifier()?;
    let tmp = chemin_temoin_tmp(dest);
    let fin = chemin_temoin(dest);
    let tmp_nom = tmp.file_name().expect("named");
    let fin_nom = fin.file_name().expect("named");
    let fd = rustix::fs::openat(
        parent_fd,
        tmp_nom,
        OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::RUSR | Mode::WUSR,
    )
    .map_err(rx_err("création du témoin temporaire", &tmp))?;
    let mut f = fs::File::from(fd);
    f.write_all(contenu.as_bytes())
        .map_err(io_err("écriture du témoin", &tmp))?;
    f.sync_all().map_err(io_err("fsync du témoin", &tmp))?;
    drop(f);
    ctx.verifier()?;
    ren.rename_noreplace(parent_fd, tmp_nom, fin_nom)
        .map_err(PublishError::Renommage)?;
    sync.fsync_repertoire(parent_fd)
        .map_err(io_err("fsync du parent après le témoin", parent))?;
    Ok(fin)
}

fn analyser_fd(fd: OwnedFd) -> Result<Temoin, PublishError> {
    let mut texte = String::new();
    fs::File::from(fd)
        .take(TEMOIN_TAILLE_MAX)
        .read_to_string(&mut texte)
        .map_err(|e| PublishError::TemoinInvalide(format!("lecture : {e}")))?;
    Temoin::analyser(&texte)
}

/// Read a witness by path without following the last component; any problem is `TemoinInvalide`.
pub fn lire_temoin(chemin: &Path) -> Result<Temoin, PublishError> {
    let fd = rustix::fs::open(
        chemin,
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK,
        Mode::empty(),
    )
    .map_err(|e| PublishError::TemoinInvalide(format!("ouverture : {e}")))?;
    analyser_fd(fd)
}

/// Read the witness of `dest` relative to the parent descriptor.
pub fn lire_temoin_at(dest: &Path, parent_fd: BorrowedFd<'_>) -> Result<Temoin, PublishError> {
    let fin = chemin_temoin(dest);
    let fd = rustix::fs::openat(
        parent_fd,
        fin.file_name().expect("named"),
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK,
        Mode::empty(),
    )
    .map_err(|e| PublishError::TemoinInvalide(format!("ouverture : {e}")))?;
    analyser_fd(fd)
}

/// Prefix of the decision returned when the witness is kept for lack of directory exclusion.
pub const TEMOIN_CONSERVE_SANS_EXCLUSION: &str =
    "témoin conservé : exclusion du répertoire non démontrée";

/// Remove OUR witness: re-read and validated through the parent descriptor, then `unlinkat`
/// relative to it, then fsync of the parent. POSIX has no "unlink this name only if it is this
/// inode": between the validation and the unlink a same-uid third party can substitute the
/// entry. So the continuous directory guard is checked right before the unlink; without it the witness
/// is CONSERVED with a named decision, nothing is deleted. `crochet` runs between the validation
/// and that check (test seam; a no-op otherwise).
/// Why the witness could not be removed durably; `supprime` says whether it was unlinked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErreurRetrait {
    pub message: String,
    pub supprime: bool,
}

fn retirer_temoin(
    dest: &Path,
    ctx: &ContexteParent<'_>,
    attendu: &Temoin,
    sync: &dyn SyncRepertoire,
    crochet: &dyn Fn(),
) -> Result<(), ErreurRetrait> {
    let conserve = |message: String| ErreurRetrait {
        message,
        supprime: false,
    };
    if dest.parent() != Some(ctx.chemin) {
        return Err(conserve(format!(
            "{} n'est pas le parent gardé de {}",
            ctx.chemin.display(),
            dest.display()
        )));
    }
    let parent_fd = ctx.fd;
    lire_temoin_at(dest, parent_fd)
        .and_then(|t| t.valider(attendu))
        .map_err(|e| conserve(e.to_string()))?;
    crochet();
    ctx.verifier()
        .map_err(|e| conserve(format!("{TEMOIN_CONSERVE_SANS_EXCLUSION} ({e})")))?;
    let fin = chemin_temoin(dest);
    rustix::fs::unlinkat(parent_fd, fin.file_name().expect("named"), AtFlags::empty())
        .map_err(|e| conserve(format!("suppression du témoin : {e}")))?;
    sync.fsync_repertoire(parent_fd).map_err(|e| ErreurRetrait {
        message: format!("fsync du parent après retrait du témoin : {e}"),
        supprime: true,
    })
}

/// Nonce recorded in the `.po-staging-owner` of a directory (`dest` after publication), read
/// without following links. `None` if absent or malformed.
pub fn lire_nonce_proprietaire(repertoire: &Path) -> Option<String> {
    let fd = rustix::fs::open(
        repertoire.join(FICHIER_PROPRIETAIRE),
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK,
        Mode::empty(),
    )
    .ok()?;
    let mut texte = String::new();
    fs::File::from(fd)
        .take(4096)
        .read_to_string(&mut texte)
        .ok()?;
    let mut lignes = texte.lines();
    if lignes.next() != Some(PROPRIETAIRE_ENTETE) {
        return None;
    }
    lignes
        .find_map(|l| l.strip_prefix("nonce="))
        .map(str::to_string)
}

/// Observed state of the witness, for [`classify`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemoinObserve {
    Absent,
    Invalide(String),
    Valide(Temoin),
}

pub fn observer_temoin(dest: &Path) -> TemoinObserve {
    let chemin = chemin_temoin(dest);
    match fs::symlink_metadata(&chemin) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => TemoinObserve::Absent,
        Err(e) => TemoinObserve::Invalide(format!("lstat : {e}")),
        Ok(_) => match lire_temoin(&chemin) {
            Ok(t) => TemoinObserve::Valide(t),
            Err(e) => TemoinObserve::Invalide(e.to_string()),
        },
    }
}

// ---------------------------------------------------------------------------------------------
// §3, §5, v6.1, v6.2 Decision at startup (pure)
// ---------------------------------------------------------------------------------------------

/// What is observed of `dest` itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EtatDestObserve {
    Absent,
    RepertoireVide,
    RepertoireNonVide,
    /// Link (even broken), non-directory, mount point: a dangerous state.
    Dangereux,
}

/// Everything [`classify`] needs; gathered by the caller UNDER the lock (v6.1: the fast path
/// never reads "no witness" then DONE without the lock).
#[derive(Debug, Clone)]
pub struct EtatDisque {
    pub dest: EtatDestObserve,
    /// DONE + MARQUEUR_PO valid in `dest` (judged by the existing migrator code).
    pub dest_publie_valide: bool,
    pub temoin: TemoinObserve,
    pub temoin_tmp_present: bool,
    /// Every staging found for this `dest` (none can belong to this run at startup).
    pub stagings: Vec<PathBuf>,
    /// Source identity and canonical `dest` as they are NOW.
    pub source_actuelle: Option<SourceId>,
    pub dest_canonique_actuel: PathBuf,
    /// Fingerprint of the manifest of `dest` recomputed by stat + hash (if `dest` is present).
    pub manifeste_dest: Option<String>,
    /// Nonce of the `.po-staging-owner` found in `dest` ([`lire_nonce_proprietaire`]).
    pub nonce_proprietaire_dest: Option<String>,
    /// A [`SourceExclusion`] can be acquired.
    pub exclusion_disponible: bool,
    /// v6.2: continuous exclusion from the capture to the switch is demonstrated.
    pub exclusion_continue_demontree: bool,
    /// Operator resumption requested for this nonce (`storage republish --nonce N`).
    pub reprise_demandee: Option<String>,
}

/// Named reason of a refusal. A refusal never deletes anything.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Motif {
    DestDangereux,
    CheminCanoniqueChange { avant: PathBuf, apres: PathBuf },
    TemoinTemporairePresent,
    TemoinInvalide(String),
    TemoinNonLie(String),
    TemoinAutreNonce { demande: String, trouve: String },
    TemoinSansDest,
    DestPublieInvalide,
    ManifesteDivergent,
    ProprietaireNonLie,
    AdoptionSansExclusionContinue,
    DestPreexistante,
    RepriseSansExclusionContinue,
    RepriseNonImplementee,
    OrphelinConserve(Vec<PathBuf>),
    DestNonVideNonPublie,
    SourceNonGelee,
}

impl Motif {
    /// P5: only a dangerous state of `dest` is a named `bail`; everything else is StayOnSource.
    pub fn est_bail(&self) -> bool {
        matches!(self, Motif::DestDangereux)
    }
}

impl fmt::Display for Motif {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Motif::DestDangereux => write!(f, "dest dans un état dangereux (lien, non-répertoire ou montage)"),
            Motif::CheminCanoniqueChange { avant, apres } => write!(
                f,
                "chemin canonique de dest changé ({} -> {}) : refus définitif, procédure opérateur",
                avant.display(),
                apres.display()
            ),
            Motif::TemoinTemporairePresent => write!(f, "témoin temporaire présent : historique ambigu, conservé"),
            Motif::TemoinInvalide(m) => write!(f, "témoin tronqué ou invalide ({m}) : historique ambigu, conservé"),
            Motif::TemoinNonLie(m) => write!(f, "témoin non lié ({m}), conservé"),
            Motif::TemoinAutreNonce { demande, trouve } => {
                write!(f, "témoin d'un autre nonce (demandé {demande}, trouvé {trouve}), conservé")
            }
            Motif::TemoinSansDest => write!(
                f,
                "témoin présent, dest absent : conservé, sans renommage (reprise par commande opérateur)"
            ),
            Motif::DestPublieInvalide => write!(
                f,
                "témoin présent, dest sans DONE + MARQUEUR_PO valides : adoption refusée, rien supprimé"
            ),
            Motif::ManifesteDivergent => write!(f, "témoin présent, manifeste de dest divergent : rien supprimé"),
            Motif::ProprietaireNonLie => write!(
                f,
                "témoin présent, .po-staging-owner de dest absent ou d'un autre nonce : adoption refusée"
            ),
            Motif::AdoptionSansExclusionContinue => write!(
                f,
                "adoption refusée : exclusion continue des écrivains jusqu'à la durabilité non démontrée (v6.1)"
            ),
            Motif::DestPreexistante => write!(f, "destination préexistante (même vide) : rien n'est touché"),
            Motif::RepriseSansExclusionContinue => write!(
                f,
                "reprise refusée : exclusion continue de la source depuis la capture non démontrée"
            ),
            Motif::RepriseNonImplementee => write!(f, "reprise opérateur non implémentée dans ce lot"),
            Motif::OrphelinConserve(v) => write!(
                f,
                "staging d'une autre exécution conservé, aucune nouvelle copie : {v:?}"
            ),
            Motif::DestNonVideNonPublie => write!(f, "dest non vide et non publié : rien n'est touché"),
            Motif::SourceNonGelee => write!(f, "source non gelée : aucune exclusion des écrivains démontrée"),
        }
    }
}

/// Decision at startup. There is deliberately no "delete" decision: deletion of anything found
/// on disk at startup (another run's) is never automatic (§5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    /// No trace, `dest` absent or empty, exclusion available: a new copy may start.
    NouvelleCopie,
    /// `dest` published (DONE + MARKER valid), no witness, no staging: serve it.
    ServirPublie,
    /// §3: witness bound + `dest` whose manifest matches: fsync `dest` and the parent, remove the
    /// witness, then adopt. Serve only after these fsyncs succeed.
    AdopterApresFsync {
        nonce: String,
    },
    Refuse {
        motif: Motif,
    },
}

/// §3, §5, v6.1, v6.2. CLOSED BY DEFAULT: ambiguity and refusals are evaluated BEFORE any
/// adoption, and adoption requires ALL its preconditions; any missing or contradictory one is a
/// `Refuse` (and thus conservation, nothing deleted).
pub fn classify(e: &EtatDisque) -> Decision {
    let refuse = |motif| Decision::Refuse { motif };
    // 1. Dangerous or ambiguous states, whatever else is observed.
    if e.dest == EtatDestObserve::Dangereux {
        return refuse(Motif::DestDangereux);
    }
    if e.temoin_tmp_present {
        return refuse(Motif::TemoinTemporairePresent);
    }
    let temoin = match &e.temoin {
        // v6.2: a truncated / invalid witness does NOT prove the rename never happened.
        TemoinObserve::Invalide(m) => return refuse(Motif::TemoinInvalide(m.clone())),
        TemoinObserve::Valide(t) => Some(t),
        TemoinObserve::Absent => None,
    };
    if let Some(t) = temoin {
        if t.dest != e.dest_canonique_actuel {
            return refuse(Motif::CheminCanoniqueChange {
                avant: t.dest.clone(),
                apres: e.dest_canonique_actuel.clone(),
            });
        }
        if e.source_actuelle.as_ref() != Some(&t.source) {
            return refuse(Motif::TemoinNonLie(format!(
                "source du témoin {} != source actuelle {:?}",
                t.source, e.source_actuelle
            )));
        }
    }
    // 2. Operator resumption (§5, v6.2): refused unless continuous exclusion is demonstrated,
    //    and even then not implemented in this lot.
    if let Some(n) = &e.reprise_demandee {
        if !e.exclusion_continue_demontree {
            return refuse(Motif::RepriseSansExclusionContinue);
        }
        if let Some(t) = temoin {
            if &t.nonce != n {
                return refuse(Motif::TemoinAutreNonce {
                    demande: n.clone(),
                    trouve: t.nonce.clone(),
                });
            }
        }
        return refuse(Motif::RepriseNonImplementee);
    }
    // 3. Witness present: adoption (§3). PRECONDITIONS, ALL required, in this order:
    //    (a) no staging or orphan present (otherwise the history is ambiguous);
    //    (b) `dest` present and not empty;
    //    (c) `dest` published valid: DONE + MARQUEUR_PO valid;
    //    (d) manifest of `dest` (stat + hash) equal to the one bound in the witness;
    //    (d') `.po-staging-owner` of `dest` bound to the nonce of the witness (the witness alone
    //         does not prove its origin). This binding does NOT protect against a writer of the
    //         same uid, who can write both files: only the exclusions can;
    //    (e) continuous exclusion of the writers demonstrated (v6.1: old binaries may have
    //        opened and written `dest` from the rename on, before the fsync of the parent).
    if let Some(t) = temoin {
        if !e.stagings.is_empty() {
            return refuse(Motif::OrphelinConserve(e.stagings.clone()));
        }
        if e.dest != EtatDestObserve::RepertoireNonVide {
            return refuse(Motif::TemoinSansDest);
        }
        if !e.dest_publie_valide {
            return refuse(Motif::DestPublieInvalide);
        }
        if e.manifeste_dest.as_deref() != Some(t.manifeste.as_str()) {
            return refuse(Motif::ManifesteDivergent);
        }
        if e.nonce_proprietaire_dest.as_deref() != Some(t.nonce.as_str()) {
            return refuse(Motif::ProprietaireNonLie);
        }
        if !e.exclusion_continue_demontree {
            return refuse(Motif::AdoptionSansExclusionContinue);
        }
        return Decision::AdopterApresFsync {
            nonce: t.nonce.clone(),
        };
    }
    // 4. No witness.
    if !e.stagings.is_empty() {
        return refuse(Motif::OrphelinConserve(e.stagings.clone()));
    }
    match e.dest {
        EtatDestObserve::RepertoireNonVide if e.dest_publie_valide => Decision::ServirPublie,
        EtatDestObserve::RepertoireNonVide => refuse(Motif::DestNonVideNonPublie),
        EtatDestObserve::RepertoireVide => refuse(Motif::DestPreexistante),
        _ if !e.exclusion_disponible => refuse(Motif::SourceNonGelee),
        _ => Decision::NouvelleCopie,
    }
}

// ---------------------------------------------------------------------------------------------
// §2.1 Staging of THIS run, §6 deletion
// ---------------------------------------------------------------------------------------------

/// The staging created by THIS process in THIS run. Only [`creer_staging`] builds one; a staging
/// found on disk (another run's) never gets a handle. Holds the descriptors of the staging and
/// of the parent of `dest`, both kept from the creation on, and the TWO exclusions that issued
/// its guards: they are fixed here at the creation, so a publication cannot check a guard
/// against another implementation than the one that issued it (no exclusion parameter on
/// [`publish`]). The directory guard is released once: by a durable publication, or by the
/// Drop when the staging cannot have been published (see the Drop). It stays held until the
/// end of the process after any "maybe published" outcome, and also after
/// [`PublishError::PublieDurableNonServable`] (durable, witness not removed).
pub struct StagingHandle<'x> {
    chemin: PathBuf,
    dest: PathBuf,
    dest_canonique: PathBuf,
    nonce: String,
    guard: ExclusionGuard,
    excl_source: &'x dyn SourceExclusion,
    excl_rep: &'x dyn ExclusionRepertoire,
    fd: OwnedFd,
    parent_fd: OwnedFd,
    parent_stat: StatLite,
    garde: GardeRepertoire,
    garde_relachee: std::cell::Cell<bool>,
    /// Set right BEFORE the rename is attempted, reset only on a failure (EEXIST, ENOTEMPTY,
    /// not supported) VERIFIED by `statat`: while set, the guard is released only by a durable
    /// publication, never by the Drop (v6.1: excluded up to the fsync).
    peut_etre_publie: std::cell::Cell<bool>,
    dev: u64,
    ino: u64,
    uid: u32,
    proprietaire: String,
}

impl fmt::Debug for StagingHandle<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StagingHandle")
            .field("chemin", &self.chemin)
            .field("dest", &self.dest)
            .field("nonce", &self.nonce)
            .finish_non_exhaustive()
    }
}

/// The Drop releases the directory guard unless the staging is MAYBE published: from the
/// moment the rename is attempted (and until it is known to have failed cleanly) the guard is
/// only released by a durable publication; otherwise it stays held until the end of the process
/// (v6.1: writers excluded up to the fsync). A panic does not change this rule: before any
/// rename the guard is released even while unwinding (the release goes through
/// `relacher_sans_paniquer`, so that a panicking `relacher` cannot abort the process).
impl Drop for StagingHandle<'_> {
    fn drop(&mut self) {
        if self.peut_etre_publie.get() {
            return;
        }
        // relacher_garde goes through relacher_sans_paniquer: no panic can escape the Drop.
        self.relacher_garde();
    }
}

impl<'x> StagingHandle<'x> {
    pub fn nonce(&self) -> &str {
        &self.nonce
    }
    /// Path of the staging (read only: the handle's paths cannot be redirected).
    pub fn chemin(&self) -> &Path {
        &self.chemin
    }
    pub fn dest(&self) -> &Path {
        &self.dest
    }
    pub fn dest_canonique(&self) -> &Path {
        &self.dest_canonique
    }
    fn nom(&self) -> &OsStr {
        self.chemin.file_name().expect("named")
    }
    fn nom_dest(&self) -> &OsStr {
        nom_dest(&self.dest)
    }
    /// The guarded parent, checked against the directory exclusion that issued the guard.
    fn contexte(&self) -> ContexteParent<'_> {
        ContexteParent {
            chemin: self.dest.parent().expect("dest checked"),
            fd: self.parent_fd.as_fd(),
            stat: self.parent_stat,
            excl: self.excl_rep,
            garde: &self.garde,
        }
    }
    fn verifier_source(&self) -> Result<(), PublishError> {
        verifier_exclusion(self.excl_source, &self.guard)
    }
    fn relacher_garde(&self) {
        if !self.garde_relachee.replace(true) {
            relacher_sans_paniquer(self.excl_rep, &self.garde);
        }
    }
}

/// Release a guard without ever propagating a panic of the implementation (the release may
/// run while unwinding, where a second panic would abort the process). A swallowed panic is
/// logged.
fn relacher_sans_paniquer(excl: &dyn ExclusionRepertoire, garde: &GardeRepertoire) {
    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| excl.relacher(garde))).is_err() {
        tracing::warn!(
            garde = ?garde.identite(),
            "storage::publish : panique de ExclusionRepertoire::relacher avalée (garde considérée relâchée)"
        );
    }
}

/// A directory guard acquired but not yet owned by a [`StagingHandle`]: released by its Drop
/// (even while unwinding) unless handed over with `GardeEnAttente::remettre`. Nothing can
/// be published before the handle exists, so releasing here is always safe.
struct GardeEnAttente<'x> {
    excl: &'x dyn ExclusionRepertoire,
    garde: Option<GardeRepertoire>,
}

impl GardeEnAttente<'_> {
    fn garde(&self) -> &GardeRepertoire {
        self.garde
            .as_ref()
            .expect("garde présente jusqu'à la remise")
    }
    fn remettre(mut self) -> GardeRepertoire {
        self.garde.take().expect("garde présente jusqu'à la remise")
    }
}

impl Drop for GardeEnAttente<'_> {
    fn drop(&mut self) {
        if let Some(g) = self.garde.take() {
            relacher_sans_paniquer(self.excl, &g);
        }
    }
}

/// §2.1 with the real fsync (the variant with an injected fsync is a private test seam):
/// acquire the source exclusion, then the directory guard, check §1 and the traces under the
/// guard, create the staging and its owner file. See `creer_staging_avec`.
pub fn creer_staging<'x>(
    dest: &Path,
    excl: &'x dyn SourceExclusion,
    excl_rep: &'x dyn ExclusionRepertoire,
) -> Result<StagingHandle<'x>, PublishError> {
    creer_staging_avec(dest, excl, excl_rep, &SyncReel)
}

/// §2.1: acquire the source exclusion FIRST, open the parent once and acquire the DIRECTORY
/// guard on it (nothing is written without both); THEN, under the guard, check §1 and the
/// absence of traces; then `mkdirat` 0700 exclusive with a random nonce, keep the descriptor,
/// check its identity and device, write `.po-staging-owner`. Both exclusions are fixed in the
/// handle: the guard must stay continuous through the copy up to the publication. Until the
/// handle exists the guard is held by a `GardeEnAttente`, released on any error or panic.
/// Private: the injected `sync` is a test seam only.
fn creer_staging_avec<'x>(
    dest: &Path,
    excl: &'x dyn SourceExclusion,
    excl_rep: &'x dyn ExclusionRepertoire,
    sync: &dyn SyncRepertoire,
) -> Result<StagingHandle<'x>, PublishError> {
    let guard = excl.acquire()?;
    verifier_exclusion(excl, &guard)?;
    guard.source.valider()?;
    let parent = dest
        .parent()
        .ok_or_else(|| PublishError::DestNonAbsolu(dest.to_path_buf()))?;
    let (parent_fd, parent_stat) = ouvrir_parent(dest)?;
    let attente = GardeEnAttente {
        excl: excl_rep,
        garde: Some(excl_rep.acquerir(parent_fd.as_fd(), &parent_stat)?),
    };
    let ctx = ContexteParent {
        chemin: parent,
        fd: parent_fd.as_fd(),
        stat: parent_stat,
        excl: excl_rep,
        garde: attente.garde(),
    };
    ctx.verifier()?;
    // Under the guard: preconditions and traces (not before it).
    verifier_preconditions(dest)?;
    let traces = scanner_traces(dest)?;
    if traces != Traces::default() {
        return Err(PublishError::TracesExistantes(format!("{traces:?}")));
    }
    let dest_canonique = dest_canonique(dest)?;
    let nonce = uuid::Uuid::new_v4().simple().to_string();
    let mut nom = prefixe_staging(dest);
    nom.extend_from_slice(nonce.as_bytes());
    let chemin = dest.with_file_name(OsStr::from_bytes(&nom));
    let nom = OsStr::from_bytes(&nom);
    rustix::fs::mkdirat(&parent_fd, nom, Mode::RWXU)
        .map_err(rx_err("mkdir du staging", &chemin))?;
    let fd = rustix::fs::openat(
        &parent_fd,
        nom,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(rx_err("ouverture du staging", &chemin))?;
    let st = StatLite::depuis_rustix(
        &rustix::fs::fstat(&fd).map_err(rx_err("fstat du staging", &chemin))?,
    );
    // The guard is handed over to the handle by the LAST field computed below, so that no
    // unwinding can leave it without an owner; from then on the handle's Drop releases it.
    let h = StagingHandle {
        chemin: chemin.clone(),
        dest: dest.to_path_buf(),
        dest_canonique,
        proprietaire: format!(
            "{PROPRIETAIRE_ENTETE}\nsource={}\nnonce={nonce}\nuid={}\n",
            guard.source.0, st.uid
        ),
        nonce,
        guard,
        excl_source: excl,
        excl_rep,
        fd,
        parent_fd,
        parent_stat,
        garde_relachee: std::cell::Cell::new(false),
        peut_etre_publie: std::cell::Cell::new(false),
        dev: st.dev,
        ino: st.ino,
        uid: st.uid,
        garde: attente.remettre(),
    };
    let lst = StatLite::depuis_rustix(
        &rustix::fs::statat(&h.parent_fd, nom, AtFlags::SYMLINK_NOFOLLOW)
            .map_err(rx_err("lstat du staging", &chemin))?,
    );
    if (lst.dev, lst.ino) != (st.dev, st.ino) || lst.genre != Genre::Repertoire {
        return Err(PublishError::StagingIdentiteChangee(format!(
            "{} remplacé entre mkdir et open",
            chemin.display()
        )));
    }
    verifier_meme_systeme(&chemin, &st, &parent_stat)?;
    h.contexte().verifier()?;
    let pfd = rustix::fs::openat(
        &h.fd,
        FICHIER_PROPRIETAIRE,
        OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::RUSR | Mode::WUSR,
    )
    .map_err(rx_err("création du fichier propriétaire", &chemin))?;
    let mut pf = fs::File::from(pfd);
    pf.write_all(h.proprietaire.as_bytes())
        .map_err(io_err("écriture du fichier propriétaire", &chemin))?;
    pf.sync_all()
        .map_err(io_err("fsync du fichier propriétaire", &chemin))?;
    rustix::fs::fsync(&h.fd).map_err(rx_err("fsync du staging", &chemin))?;
    sync.fsync_repertoire(h.parent_fd.as_fd())
        .map_err(io_err("fsync du parent après le staging", parent))?;
    Ok(h)
}

/// What happened to the `renameat` of this staging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatutRenommage {
    NonTente,
    EchecFranc { errno: i32 },
    Reussi,
    Ambigu,
    NonSupporte,
}

/// A staging that is kept, with its path and every named reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conservation {
    pub chemin: PathBuf,
    /// True when §6 would have allowed the deletion in principle (renameat not attempted, or an
    /// EEXIST / ENOTEMPTY failure verified by `statat`). It is still CONSERVED: see
    /// [`decider_suppression`].
    pub eligible_section6: bool,
    pub raisons: Vec<String>,
}

/// What became of the staging.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StagingApres {
    Conserve(Conservation),
    /// It was renamed to `dest` (or may have been).
    PeutEtrePublie,
}

/// Raison always present: v6.1, ownership of each entry is not demonstrated.
pub const RAISON_PROPRIETE_NON_DEMONTREE: &str = "propriété de chaque entrée et exclusion des tiers pendant toute la vie du staging non démontrées (v6.1) : conservé, rien supprimé";

fn stat_at(fd: &OwnedFd, nom: &CStr) -> Result<StatLite, String> {
    rustix::fs::statat(fd, nom, AtFlags::SYMLINK_NOFOLLOW)
        .map(|s| StatLite::depuis_rustix(&s))
        .map_err(|e| format!("statat {nom:?} : {e}"))
}

/// Read-only examination of the staging through its descriptor; returns the first doubt (link,
/// special file, `nlink > 1`, other `st_dev`, other uid). NOTE v6.1: the ABSENCE of doubt does
/// NOT prove ownership of each entry (another process of the same uid can write in it).
fn examiner(fd: &OwnedFd, h: &StagingHandle, profondeur: usize) -> Result<(), String> {
    if profondeur > PROFONDEUR_MAX {
        return Err(format!("profondeur supérieure à {PROFONDEUR_MAX}"));
    }
    let mut dir = rustix::fs::Dir::read_from(fd).map_err(|e| format!("readdir : {e}"))?;
    while let Some(e) = dir.read() {
        let e = e.map_err(|e| format!("readdir : {e}"))?;
        let nom = e.file_name();
        if nom.to_bytes() == b"." || nom.to_bytes() == b".." {
            continue;
        }
        let st = stat_at(fd, nom)?;
        if st.genre == Genre::Lien {
            return Err(format!("lien symbolique dans le staging : {nom:?}"));
        }
        if st.genre == Genre::Autre {
            return Err(format!("fichier spécial dans le staging : {nom:?}"));
        }
        if st.genre == Genre::Fichier && st.nlink > 1 {
            return Err(format!("nlink {} > 1 : {nom:?}", st.nlink));
        }
        if st.dev != h.dev {
            return Err(format!("st_dev changé : {nom:?}"));
        }
        if st.uid != h.uid {
            return Err(format!("uid changé : {nom:?}"));
        }
        if st.genre == Genre::Repertoire {
            let sfd = rustix::fs::openat(
                fd,
                nom,
                OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::empty(),
            )
            .map_err(|e| format!("openat {nom:?} : {e}"))?;
            examiner(&sfd, h, profondeur + 1)?;
        }
    }
    Ok(())
}

fn verifier_proprietaire(h: &StagingHandle) -> Result<(), String> {
    let st = stat_at(&h.fd, c".po-staging-owner")?;
    if st.genre != Genre::Fichier || st.nlink != 1 || st.uid != h.uid || st.dev != h.dev {
        return Err("fichier propriétaire altéré".into());
    }
    let pfd = rustix::fs::openat(
        &h.fd,
        FICHIER_PROPRIETAIRE,
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(|e| format!("ouverture du fichier propriétaire : {e}"))?;
    let mut texte = String::new();
    fs::File::from(pfd)
        .take(4096)
        .read_to_string(&mut texte)
        .map_err(|e| format!("lecture du fichier propriétaire : {e}"))?;
    if texte != h.proprietaire {
        return Err("fichier propriétaire d'une autre exécution".into());
    }
    Ok(())
}

/// Identity of the staging: its name in the held parent descriptor and its own held descriptor
/// both designate the directory created by this run, and the parent path still designates the
/// held parent.
fn verifier_identite_staging(h: &StagingHandle) -> Result<(), String> {
    let lst = stat_at(
        &h.parent_fd,
        &CString::new(h.nom().as_bytes()).map_err(|e| e.to_string())?,
    )?;
    let fst = StatLite::depuis_rustix(
        &rustix::fs::fstat(&h.fd).map_err(|e| format!("fstat du staging : {e}"))?,
    );
    if lst.genre != Genre::Repertoire
        || (lst.dev, lst.ino) != (h.dev, h.ino)
        || (fst.dev, fst.ino) != (h.dev, h.ino)
        || fst.uid != h.uid
    {
        return Err(format!(
            "{} n'est plus le staging de cette exécution (dev/ino/uid)",
            h.chemin.display()
        ));
    }
    verifier_parent(h.dest.parent().expect("dest checked"), &h.parent_stat)
        .map_err(|e| e.to_string())
}

/// §6 + v6.1: decide what to do with the staging of THIS run after a failure.
///
/// §6 would allow deleting it before `renameat` or after an EEXIST / ENOTEMPTY failure verified
/// by `statat`,
/// on the basis of container identity and per-entry type / nlink / dev / uid. v6.1 says those
/// attributes do NOT prove that each entry was created by this process (same uid, other
/// process), and that what is not demonstrated is CONSERVED. Nothing in this lot demonstrates
/// the ownership of each entry nor the exclusion of third parties for the whole life of the
/// staging, so this function NEVER deletes: it returns a named `Conserve` with the path and the
/// reasons it found (the examination is read-only). There is deliberately no deletion
/// function in this module; one can only come with a proof of ownership that does not exist yet.
pub fn decider_suppression(h: &StagingHandle, statut: StatutRenommage) -> StagingApres {
    let eligible_section6 = match statut {
        StatutRenommage::NonTente | StatutRenommage::EchecFranc { .. } => true,
        StatutRenommage::Reussi | StatutRenommage::Ambigu => return StagingApres::PeutEtrePublie,
        StatutRenommage::NonSupporte => false,
    };
    let mut raisons = Vec::new();
    if !eligible_section6 {
        raisons.push(format!("statut {statut:?} hors du cas de §6"));
    }
    let identite = verifier_identite_staging(h);
    if let Err(m) = &identite {
        raisons.push(m.clone());
    }
    if let Err(m) = verifier_proprietaire(h) {
        raisons.push(m);
    }
    if identite.is_ok() {
        if let Err(m) = examiner(&h.fd, h, 0) {
            raisons.push(m);
        }
    }
    raisons.push(RAISON_PROPRIETE_NON_DEMONTREE.to_string());
    StagingApres::Conserve(Conservation {
        chemin: h.chemin.clone(),
        eligible_section6,
        raisons,
    })
}

// ---------------------------------------------------------------------------------------------
// §2.7-2.12 Orchestration
// ---------------------------------------------------------------------------------------------

/// What became of the witness after a failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemoinApres {
    JamaisEcrit,
    Conserve,
    /// Unlinked, but the fsync of the parent that follows failed: its removal is not durable.
    RetireSansFsync,
}

/// A failed publication: the named error and what was kept. Nothing must be served, and
/// nothing was deleted (no destructive cleanup in this lot). A failure reported after the
/// rename (`staging == PeutEtrePublie`) means "MAYBE published", except
/// [`PublishError::PublieDurableNonServable`] (durable, witness not removed). The directory
/// guard is NOT released by a failure: the Drop of the [`StagingHandle`] releases it only if the
/// staging cannot have been published; otherwise it stays held until the end of the process.
#[derive(Debug)]
pub struct Echec {
    pub erreur: PublishError,
    pub staging: StagingApres,
    pub temoin: TemoinApres,
}

/// A durable publication: `dest` renamed, identity of the renamed object checked, parent
/// fsynced, both exclusions still continuous after it, witness removed and fsynced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Publication {
    pub dest: PathBuf,
    pub nonce: String,
    pub manifeste: String,
}

/// Steps 7-12 with the REAL primitives only ([`RustixNoReplace`], the real fsync) and the
/// default small-file threshold. The exclusions are the ones fixed in the handle at its
/// creation (the default implementations of both refuse). The variants that accept injected
/// primitives are private test seams: no code can substitute an always-Ok fsync or rename.
pub fn publish(staging: &StagingHandle<'_>, dest: &Path) -> Result<Publication, Echec> {
    publier(staging, dest, &RustixNoReplace)
}

/// Private: injected rename (test seam), real fsync.
fn publier(
    staging: &StagingHandle<'_>,
    dest: &Path,
    ren: &dyn NoReplaceRename,
) -> Result<Publication, Echec> {
    publish_with(staging, dest, ren, &SyncReel, SEUIL_PETIT_FICHIER)
}

/// Barrier (§2.9): continuous source exclusion, continuous directory guard, same source, same
/// canonical `dest`, `dest` still absent, same staging and parent, our witness, manifest by
/// stat + hash only (through the descriptor).
///
/// The two exclusion checks here are REDUNDANT for safety with the ones made right before the
/// rename (a lost exclusion is irreversible by contract): they are kept because §2.9 requires
/// them, and so that a loss is reported as such (not as a manifest divergence) without hashing
/// the staging for nothing. Tests pin that priority of errors.
fn barriere(
    h: &StagingHandle<'_>,
    temoin: &Temoin,
    manifeste: &Manifeste,
    seuil: u64,
) -> Result<(), PublishError> {
    h.verifier_source()?;
    h.contexte().verifier()?;
    let canon = dest_canonique(&h.dest)?;
    if canon != h.dest_canonique {
        return Err(PublishError::CheminCanoniqueChange {
            avant: h.dest_canonique.clone(),
            apres: canon,
        });
    }
    verifier_preconditions(&h.dest)?;
    verifier_identite_staging(h).map_err(PublishError::StagingIdentiteChangee)?;
    lire_temoin_at(&h.dest, h.parent_fd.as_fd())?.valider(temoin)?;
    comparer_manifestes(manifeste, &manifeste_depuis_fd(&h.fd, seuil, &|_| {})?)
}

/// After a rename reported as a clean failure (decision recorded on #1270): the staging is still
/// under its name in the held parent (same dev, same inode, a directory) AND `dest` is absent
/// or a DIFFERENT inode (a real EEXIST: `dest` exists and is not ours). `dest` carrying the
/// staging's identity, a failing `statat` on the staging, or any failing `statat` on `dest`
/// OTHER than ENOENT, => `false` (ambiguous).
fn renommage_certainement_non_effectue(h: &StagingHandle<'_>) -> bool {
    let (Ok(nom), Ok(nom_dest)) = (
        CString::new(h.nom().as_bytes()),
        CString::new(h.nom_dest().as_bytes()),
    ) else {
        return false;
    };
    let staging = rustix::fs::statat(&h.parent_fd, nom.as_c_str(), AtFlags::SYMLINK_NOFOLLOW)
        .map(|s| StatLite::depuis_rustix(&s));
    let dest = rustix::fs::statat(&h.parent_fd, nom_dest.as_c_str(), AtFlags::SYMLINK_NOFOLLOW)
        .map(|s| StatLite::depuis_rustix(&s));
    echec_franc_confirme(&staging, &dest, (h.dev, h.ino))
}

/// Pure decision behind `renommage_certainement_non_effectue`, from the two `statat` results
/// (staging name, `dest` name) and the identity (dev, ino) of the staging held by the handle.
///
/// Residual TOCTOU window, NOT closed by this check: with a lying primitive AND a third party
/// that does not honour the directory guard, our object may have been visible under `dest`
/// for an instant (renamed there, then renamed back) before a failure "verified" here. Only a
/// real directory exclusion closes it; none exists today.
///
/// No exploitable window BETWEEN the two `statat`: the staging is examined first, then `dest`;
/// if the staging is renamed to `dest` in between, `dest` carries our (dev, ino) and the
/// result is `false`.
fn echec_franc_confirme(
    staging: &Result<StatLite, rustix::io::Errno>,
    dest: &Result<StatLite, rustix::io::Errno>,
    identite: (u64, u64),
) -> bool {
    let staging_present = matches!(
        staging,
        Ok(s) if (s.dev, s.ino) == identite && s.genre == Genre::Repertoire
    );
    let dest_pas_le_staging = match dest {
        Err(e) => *e == rustix::io::Errno::NOENT,
        Ok(d) => (d.dev, d.ino) != identite,
    };
    staging_present && dest_pas_le_staging
}

/// Right after the rename: the object now named `dest` in the held parent is the one whose
/// identity was checked at the barrier, and the staging descriptor held since the creation
/// still designates that inode.
fn verifier_objet_publie(h: &StagingHandle<'_>) -> Result<(), PublishError> {
    let nom = CString::new(h.nom_dest().as_bytes())
        .map_err(|e| PublishError::ObjetPublieDivergent(e.to_string()))?;
    let publie = stat_at(&h.parent_fd, &nom).map_err(PublishError::ObjetPublieDivergent)?;
    let tenu = StatLite::depuis_rustix(
        &rustix::fs::fstat(&h.fd)
            .map_err(|e| PublishError::ObjetPublieDivergent(format!("fstat du staging : {e}")))?,
    );
    if publie.genre != Genre::Repertoire
        || (publie.dev, publie.ino) != (h.dev, h.ino)
        || (tenu.dev, tenu.ino) != (h.dev, h.ino)
    {
        return Err(PublishError::ObjetPublieDivergent(format!(
            "dest {}:{} / descripteur {}:{} / vérifié {}:{}",
            publie.dev, publie.ino, tenu.dev, tenu.ino, h.dev, h.ino
        )));
    }
    Ok(())
}

/// Steps 7-12 of §2 on a staging already filled, closed and checked (steps 2-6 are not part of
/// this lot): manifest -> witness -> barrier -> both exclusions again -> `renameat` WITHOUT
/// replacement relative to the held parent descriptor -> identity of the renamed object ->
/// fsync of the parent -> both exclusions again -> removal of the witness -> release.
///
/// - Both exclusions and the preconditions are checked BEFORE anything is written: with
///   [`NoExclusion`] or [`RepertoireNonExclu`] nothing is written (and no handle can even be
///   created).
/// - `dest` must be ABSENT (no `rmdir(dest)`, which replaces v6 §2.10 for this lot).
/// - On ANY failure nothing is deleted: the witness (once written) and the staging are kept
///   and reported ([`decider_suppression`] never deletes, v6.1).
/// - If the object renamed to `dest` is not the checked one, or an exclusion is lost after the
///   rename, the error is named, nothing is deleted, nothing must be served, the state is kept
///   for the operator.
fn publish_with(
    h: &StagingHandle<'_>,
    dest: &Path,
    ren: &dyn NoReplaceRename,
    sync: &dyn SyncRepertoire,
    seuil: u64,
) -> Result<Publication, Echec> {
    let sans_ecriture = |erreur| Echec {
        erreur,
        staging: decider_suppression(h, StatutRenommage::NonTente),
        temoin: TemoinApres::JamaisEcrit,
    };
    let conserve = |erreur, statut| Echec {
        erreur,
        staging: decider_suppression(h, statut),
        temoin: TemoinApres::Conserve,
    };
    // Before any write.
    if dest != h.dest {
        return Err(sans_ecriture(PublishError::DestDifferent {
            donne: dest.to_path_buf(),
            attendu: h.dest.clone(),
        }));
    }
    h.verifier_source().map_err(sans_ecriture)?;
    let ctx = h.contexte();
    ctx.verifier().map_err(sans_ecriture)?;
    verifier_preconditions(dest).map_err(sans_ecriture)?;
    verifier_identite_staging(h)
        .map_err(|m| sans_ecriture(PublishError::StagingIdentiteChangee(m)))?;
    let traces = scanner_traces(dest).map_err(sans_ecriture)?;
    if traces.temoin || traces.temoin_tmp || traces.stagings != [h.chemin.clone()] {
        return Err(sans_ecriture(PublishError::TracesExistantes(format!(
            "{traces:?}"
        ))));
    }
    // 7. Manifest (read only, through the descriptor).
    let manifeste = manifeste_depuis_fd(&h.fd, seuil, &|_| {}).map_err(sans_ecriture)?;
    let temoin = Temoin {
        source: h.guard.source.clone(),
        nonce: h.nonce.clone(),
        manifeste: manifeste.empreinte(),
        dest: h.dest_canonique.clone(),
    };
    // 8. Witness.
    ecrire_temoin(dest, &ctx, &temoin, ren, sync)
        .map_err(|e| conserve(e, StatutRenommage::NonTente))?;
    // 9. Barrier. Failure => nothing published, everything kept.
    barriere(h, &temoin, &manifeste, seuil).map_err(|e| conserve(e, StatutRenommage::NonTente))?;
    // 10. Both exclusions right before the rename (§7, v6.1), then renameat WITHOUT
    //     replacement relative to the held parent; raw errno; no fallback.
    h.verifier_source()
        .and_then(|()| ctx.verifier())
        .map_err(|e| conserve(e, StatutRenommage::NonTente))?;
    // Maybe published from the moment the rename is attempted (it may rename then panic).
    h.peut_etre_publie.set(true);
    let issue = ren.rename_noreplace(h.parent_fd.as_fd(), h.nom(), h.nom_dest());
    // A clean failure is not taken on the word of the primitive: verified by statat
    // (`renommage_certainement_non_effectue`); otherwise it stays "maybe published".
    let echec_franc_verifie = matches!(
        issue,
        Err(RenameError::Franc { .. } | RenameError::NonSupporte { .. })
    ) && renommage_certainement_non_effectue(h);
    if echec_franc_verifie {
        h.peut_etre_publie.set(false);
    }
    if let Err(re) = issue {
        let statut = match re {
            RenameError::Franc { errno } if echec_franc_verifie => {
                StatutRenommage::EchecFranc { errno }
            }
            RenameError::NonSupporte { .. } if echec_franc_verifie => StatutRenommage::NonSupporte,
            _ => StatutRenommage::Ambigu,
        };
        let erreur = match re {
            RenameError::Franc { errno } | RenameError::NonSupporte { errno }
                if !echec_franc_verifie =>
            {
                PublishError::FrancNonConfirme { errno }
            }
            _ => PublishError::Renommage(re),
        };
        return Err(conserve(erreur, statut));
    }
    // The renamed object must be the checked one.
    verifier_objet_publie(h).map_err(|e| conserve(e, StatutRenommage::Reussi))?;
    // 11. fsync of the parent; failure => MAYBE published, durability not confirmed (§4).
    if let Err(e) = sync.fsync_repertoire(h.parent_fd.as_fd()) {
        return Err(conserve(
            PublishError::PublieDurabiliteNonConfirmee(format!("fsync du parent : {e}")),
            StatutRenommage::Reussi,
        ));
    }
    // Both exclusions must still be continuous AFTER the fsync (v6.1: writers excluded up to
    // durability). A loss here: MAYBE published (durable but writers not excluded), not to be
    // served, kept for the operator.
    h.verifier_source()
        .and_then(|()| ctx.verifier())
        .map_err(|e| conserve(e, StatutRenommage::Reussi))?;
    // Durability and both exclusions are confirmed here: a failure to remove the witness
    // leaves a DURABLE publication that must not be served (the witness is still there).
    if let Err(r) = retirer_temoin(dest, &ctx, &temoin, sync, &|| {}) {
        return Err(Echec {
            erreur: PublishError::PublieDurableNonServable(r.message),
            staging: decider_suppression(h, StatutRenommage::Reussi),
            temoin: if r.supprime {
                TemoinApres::RetireSansFsync
            } else {
                TemoinApres::Conserve
            },
        });
    }
    // The guard was held up to after the fsync of the parent: release it.
    h.relacher_garde();
    // 12. Serving is the caller's business, and only on this `Ok`.
    Ok(Publication {
        dest: dest.to_path_buf(),
        nonce: h.nonce.clone(),
        manifeste: temoin.manifeste,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::sync::Mutex;
    use std::time::Duration;

    // ---- fakes ---------------------------------------------------------------------------

    #[derive(Debug, Clone, Copy)]
    enum Rupture {
        Levee,
        LeveeReposee,
        IdentiteChangee,
    }

    struct EtatTest {
        tenue: bool,
        generation: u64,
        identite: String,
        appels_verify: usize,
        rupture: Option<(usize, Rupture)>,
    }

    /// Fake exclusion: held, released, released-then-taken-again, identity changed; the
    /// break happens on the N-th `verify_at_barrier` (1 = creer_staging, 2 = publish pre-check,
    /// 3 = barrier).
    struct TestExclusion(Mutex<EtatTest>);

    impl TestExclusion {
        fn tenue() -> Self {
            Self::avec(None)
        }
        fn rompue_au(appel: usize, r: Rupture) -> Self {
            Self::avec(Some((appel, r)))
        }
        fn avec(rupture: Option<(usize, Rupture)>) -> Self {
            TestExclusion(Mutex::new(EtatTest {
                tenue: true,
                generation: 1,
                identite: "source-A".into(),
                appels_verify: 0,
                rupture,
            }))
        }
    }

    impl SourceExclusion for TestExclusion {
        fn acquire(&self) -> Result<ExclusionGuard, PublishError> {
            let e = self.0.lock().unwrap();
            if !e.tenue {
                return Err(PublishError::SourceNonGelee);
            }
            Ok(ExclusionGuard {
                source: SourceId(e.identite.clone()),
                generation: e.generation,
            })
        }
        fn verify_at_barrier(&self, g: &ExclusionGuard) -> Result<(), PublishError> {
            let mut e = self.0.lock().unwrap();
            e.appels_verify += 1;
            if let Some((n, r)) = e.rupture {
                if n == e.appels_verify {
                    match r {
                        Rupture::Levee => e.tenue = false,
                        Rupture::LeveeReposee => e.generation += 1,
                        Rupture::IdentiteChangee => e.identite = "source-B".into(),
                    }
                }
            }
            if !e.tenue {
                return Err(PublishError::ExclusionRompue("levée".into()));
            }
            if e.generation != g.generation {
                return Err(PublishError::ExclusionRompue(
                    "levée puis reposée : continuité non démontrée".into(),
                ));
            }
            Ok(())
        }
        fn identity(&self) -> Result<SourceId, PublishError> {
            Ok(SourceId(self.0.lock().unwrap().identite.clone()))
        }
    }

    type Action = Box<dyn Fn() + Send + Sync>;

    /// Fake that CLAIMS a continuous directory exclusion (tests only: nothing demonstrates it).
    struct RepertoireExcluPretendu;

    impl ExclusionRepertoire for RepertoireExcluPretendu {
        fn acquerir(
            &self,
            _parent: BorrowedFd<'_>,
            parent_stat: &StatLite,
        ) -> Result<GardeRepertoire, PublishError> {
            Ok(GardeRepertoire::nouvelle(parent_stat, 1))
        }
        fn verifier_continuite(&self, _garde: &GardeRepertoire) -> Result<(), PublishError> {
            Ok(())
        }
        fn relacher(&self, _garde: &GardeRepertoire) {}
    }

    /// Stateful fake directory exclusion: held, released, released then re-acquired, or
    /// released at the N-th continuity check.
    struct TestRepertoire(Mutex<EtatRep>);

    #[derive(Default)]
    struct EtatRep {
        tenue: bool,
        jeton: u64,
        relachements: usize,
        verifications: usize,
        levee_au: Option<usize>,
        panique_a_la_verification: Option<usize>,
        panique_au_relacher: bool,
    }

    impl TestRepertoire {
        fn tenue() -> Self {
            TestRepertoire(Mutex::new(EtatRep {
                tenue: true,
                jeton: 1,
                ..Default::default()
            }))
        }
        /// Released at the `n`-th call of `verifier_continuite`.
        fn levee_au(n: usize) -> Self {
            let r = Self::tenue();
            r.0.lock().unwrap().levee_au = Some(n);
            r
        }
        fn lever(&self) {
            self.0.lock().unwrap().tenue = false;
        }
        fn reprendre(&self) {
            let mut e = self.0.lock().unwrap();
            e.tenue = true;
            e.jeton += 1;
        }
        fn relachements(&self) -> usize {
            self.0.lock().unwrap().relachements
        }
        fn verifications(&self) -> usize {
            self.0.lock().unwrap().verifications
        }
    }

    impl ExclusionRepertoire for TestRepertoire {
        fn acquerir(
            &self,
            _parent: BorrowedFd<'_>,
            parent_stat: &StatLite,
        ) -> Result<GardeRepertoire, PublishError> {
            let e = self.0.lock().unwrap();
            if !e.tenue {
                return Err(PublishError::RepertoireNonExclu("garde non tenue".into()));
            }
            Ok(GardeRepertoire::nouvelle(parent_stat, e.jeton))
        }
        fn verifier_continuite(&self, garde: &GardeRepertoire) -> Result<(), PublishError> {
            let mut e = self.0.lock().unwrap();
            e.verifications += 1;
            if e.panique_a_la_verification == Some(e.verifications) {
                drop(e); // never panic while holding the lock
                panic!("panique injectée dans verifier_continuite");
            }
            if e.levee_au == Some(e.verifications) {
                e.tenue = false;
            }
            if !e.tenue {
                return Err(PublishError::RepertoireNonExclu("garde levée".into()));
            }
            if garde.jeton() != e.jeton {
                return Err(PublishError::RepertoireNonExclu(
                    "garde perdue puis réacquise : non continue".into(),
                ));
            }
            Ok(())
        }
        fn relacher(&self, _garde: &GardeRepertoire) {
            let panique = {
                let mut e = self.0.lock().unwrap();
                e.relachements += 1;
                e.panique_au_relacher
            };
            if panique {
                panic!("panique injectée dans relacher");
            }
        }
    }

    /// Stateless fake that CLAIMS a frozen source (tests only).
    struct SourceTenue;

    impl SourceExclusion for SourceTenue {
        fn acquire(&self) -> Result<ExclusionGuard, PublishError> {
            Ok(ExclusionGuard {
                source: SourceId("source-A".into()),
                generation: 1,
            })
        }
        fn verify_at_barrier(&self, _guard: &ExclusionGuard) -> Result<(), PublishError> {
            Ok(())
        }
        fn identity(&self) -> Result<SourceId, PublishError> {
            Ok(SourceId("source-A".into()))
        }
    }

    /// The guarded parent of `enfant`, owned, for the tests that need a context.
    struct Parent {
        chemin: PathBuf,
        fd: OwnedFd,
        stat: StatLite,
        garde: GardeRepertoire,
    }

    fn parent_garde(enfant: &Path, excl: &dyn ExclusionRepertoire) -> Parent {
        let (fd, stat) = ouvrir_parent(enfant).unwrap();
        let garde = excl.acquerir(fd.as_fd(), &stat).unwrap();
        Parent {
            chemin: enfant.parent().unwrap().to_path_buf(),
            fd,
            stat,
            garde,
        }
    }

    impl Parent {
        fn ctx<'a>(&'a self, excl: &'a dyn ExclusionRepertoire) -> ContexteParent<'a> {
            ContexteParent {
                chemin: &self.chemin,
                fd: self.fd.as_fd(),
                stat: self.stat,
                excl,
                garde: &self.garde,
            }
        }
    }

    fn manifeste_de(racine: &Path, seuil: u64) -> Result<Manifeste, PublishError> {
        let p = parent_garde(racine, &RepertoireExcluPretendu);
        construire_manifeste(racine, seuil, &p.ctx(&RepertoireExcluPretendu))
    }

    fn comparer_a(m: &Manifeste, racine: &Path, seuil: u64) -> Result<(), PublishError> {
        let p = parent_garde(racine, &RepertoireExcluPretendu);
        comparer_manifeste(m, racine, seuil, &p.ctx(&RepertoireExcluPretendu))
    }

    fn ecrire_temoin_test(dest: &Path, t: &Temoin) -> PathBuf {
        let p = parent_garde(dest, &RepertoireExcluPretendu);
        ecrire_temoin(
            dest,
            &p.ctx(&RepertoireExcluPretendu),
            t,
            &RustixNoReplace,
            &SyncReel,
        )
        .unwrap()
    }

    /// Fake rename: for `to == cible`, runs `avant` then returns `issue` (or does the real
    /// rename when `issue` is None); other renames (the witness) are real.
    struct FauxRenommage {
        cible: std::ffi::OsString,
        issue: Option<RenameError>,
        avant: Option<Action>,
        /// Rename for real, then panic (an implementation that panics after renaming).
        panique_apres: bool,
        /// Rename for real, then still return `issue` (a lying implementation).
        renommer_puis_mentir: bool,
        appels: Mutex<usize>,
    }

    impl FauxRenommage {
        fn new(cible: &Path, issue: Option<RenameError>) -> Self {
            FauxRenommage {
                cible: cible.file_name().unwrap().to_os_string(),
                issue,
                avant: None,
                panique_apres: false,
                renommer_puis_mentir: false,
                appels: Mutex::new(0),
            }
        }
        fn appels(&self) -> usize {
            *self.appels.lock().unwrap()
        }
    }

    impl NoReplaceRename for FauxRenommage {
        fn rename_noreplace(
            &self,
            dir: BorrowedFd<'_>,
            from: &OsStr,
            to: &OsStr,
        ) -> Result<(), RenameError> {
            if to == self.cible {
                *self.appels.lock().unwrap() += 1;
                if let Some(f) = &self.avant {
                    f();
                }
                if let Some(e) = self.issue {
                    if self.renommer_puis_mentir {
                        RustixNoReplace.rename_noreplace(dir, from, to).unwrap();
                    } else if matches!(e, RenameError::Franc { .. }) {
                        // A real EEXIST / ENOTEMPTY: `dest` exists, another inode than ours.
                        let _ = rustix::fs::mkdirat(dir, to, Mode::RWXU);
                    }
                    return Err(e);
                }
                if self.panique_apres {
                    RustixNoReplace.rename_noreplace(dir, from, to).unwrap();
                    panic!("rename effectué puis panique");
                }
            }
            RustixNoReplace.rename_noreplace(dir, from, to)
        }
    }

    /// Fake fsync: call N fails with EIO, or runs an action first.
    #[derive(Default)]
    struct FauxSync {
        echec_appel: Option<usize>,
        action: Option<(usize, Action)>,
        panique_appel: Option<usize>,
        appels: Mutex<usize>,
    }

    impl SyncRepertoire for FauxSync {
        fn fsync_repertoire(&self, dir: BorrowedFd<'_>) -> io::Result<()> {
            let n = {
                let mut a = self.appels.lock().unwrap();
                *a += 1;
                *a
            };
            if let Some((k, f)) = &self.action {
                if *k == n {
                    f();
                }
            }
            if self.panique_appel == Some(n) {
                panic!("panique injectée au fsync n°{n}");
            }
            if self.echec_appel == Some(n) {
                return Err(io::Error::from_raw_os_error(libc::EIO));
            }
            SyncReel.fsync_repertoire(dir)
        }
    }

    // ---- helpers ---------------------------------------------------------------------------

    fn base() -> (tempfile::TempDir, PathBuf) {
        let t = tempfile::tempdir().unwrap();
        let dest = t.path().join("base");
        (t, dest)
    }

    fn lister(dir: &Path) -> Vec<String> {
        let mut v: Vec<String> = fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        v.sort();
        v
    }

    fn staging_rempli<'x>(dest: &Path, excl: &'x dyn SourceExclusion) -> StagingHandle<'x> {
        let h = creer_staging(dest, excl, &RepertoireExcluPretendu).unwrap();
        fs::write(h.chemin.join("data.bin"), b"abc").unwrap();
        fs::create_dir(h.chemin.join("sub")).unwrap();
        fs::write(h.chemin.join("sub/x"), b"xyz").unwrap();
        h
    }

    fn conservation(s: &StagingApres) -> &Conservation {
        match s {
            StagingApres::Conserve(c) => c,
            autre => panic!("attendu Conserve, obtenu {autre:?}"),
        }
    }

    fn assert_staging_intact(h: &StagingHandle) {
        assert!(h.chemin.join(FICHIER_PROPRIETAIRE).is_file());
        assert_eq!(fs::read(h.chemin.join("data.bin")).unwrap(), b"abc");
        assert_eq!(fs::read(h.chemin.join("sub/x")).unwrap(), b"xyz");
    }

    fn faits(dest: &Path, dest_stat: Option<StatLite>, vide: Option<bool>) -> FaitsDest {
        FaitsDest {
            dest: dest.to_path_buf(),
            dest_stat,
            dest_vide: vide,
            parent_stat: Some(stat_rep(1)),
            parent_inscriptible: Ok(()),
        }
    }

    fn stat_rep(dev: u64) -> StatLite {
        StatLite {
            genre: Genre::Repertoire,
            dev,
            ino: 7,
            nlink: 2,
            mode: 0o40700,
            uid: 501,
            gid: 20,
            taille: 0,
            mtime: (0, 0),
            ctime: (0, 0),
        }
    }

    // ---- §7 exclusion ------------------------------------------------------------------------

    #[test]
    fn no_exclusion_creer_staging_refuse_sans_rien_ecrire() {
        let (t, dest) = base();
        let avant = lister(t.path());
        let e = creer_staging(&dest, &NoExclusion, &RepertoireExcluPretendu).unwrap_err();
        assert!(matches!(e, PublishError::SourceNonGelee), "{e}");
        assert_eq!(
            e.to_string(),
            "source non gelée : aucune exclusion des écrivains démontrée"
        );
        assert_eq!(lister(t.path()), avant);
    }

    #[test]
    fn exclusion_source_perdue_avant_publish_refus_avant_toute_ecriture() {
        let (t, dest) = base();
        // 2 = the pre-check of publish.
        let excl = TestExclusion::rompue_au(2, Rupture::Levee);
        let h = staging_rempli(&dest, &excl);
        let avant = lister(t.path());
        let ren = FauxRenommage::new(&dest, None);
        let echec = publier(&h, &dest, &ren).unwrap_err();
        assert!(matches!(echec.erreur, PublishError::ExclusionRompue(_)));
        assert_eq!(echec.temoin, TemoinApres::JamaisEcrit);
        assert_eq!(lister(t.path()), avant, "rien écrit dans le parent");
        assert_eq!(ren.appels(), 0);
        assert_staging_intact(&h);
    }

    #[test]
    fn garde_verifiee_par_son_emetteur_et_non_par_une_autre_implementation() {
        // The handle is created with A; B (which would answer Ok) cannot be given to publish:
        // there is no exclusion parameter. Losing A is enough to refuse.
        let (_t, dest) = base();
        let a = TestRepertoire::tenue();
        let h = creer_staging(&dest, &SourceTenue, &a).unwrap();
        let _b = RepertoireExcluPretendu;
        a.lever();
        let ren = FauxRenommage::new(&dest, None);
        let echec = publier(&h, &dest, &ren).unwrap_err();
        assert!(
            matches!(echec.erreur, PublishError::RepertoireNonExclu(_)),
            "{}",
            echec.erreur
        );
        assert_eq!(ren.appels(), 0);
        assert!(a.verifications() >= 2, "la garde est vérifiée auprès de A");
    }

    /// Publish with a source exclusion broken at its N-th check (3 = barrier); returns the
    /// failure, the number of renames of `dest`, whether `dest` exists, whether the witness
    /// exists, and checks the staging is intact.
    fn refus_source(appel: usize, r: Rupture) -> (Echec, usize, bool, bool) {
        let (_t, dest) = base();
        let excl = TestExclusion::rompue_au(appel, r);
        let h = staging_rempli(&dest, &excl);
        let ren = FauxRenommage::new(&dest, None);
        let echec = publier(&h, &dest, &ren).unwrap_err();
        assert_staging_intact_ou_publie(&h, &dest);
        (
            echec,
            ren.appels(),
            dest.exists(),
            chemin_temoin(&dest).exists(),
        )
    }

    fn assert_staging_intact_ou_publie(h: &StagingHandle, dest: &Path) {
        let racine = if h.chemin.exists() {
            h.chemin.clone()
        } else {
            dest.to_path_buf()
        };
        assert!(racine.join(FICHIER_PROPRIETAIRE).is_file());
        assert_eq!(fs::read(racine.join("data.bin")).unwrap(), b"abc");
        assert_eq!(fs::read(racine.join("sub/x")).unwrap(), b"xyz");
    }

    #[test]
    fn exclusion_levee_puis_reposee_refus_a_la_barriere() {
        let (echec, appels, dest_existe, temoin_existe) = refus_source(3, Rupture::LeveeReposee);
        assert!(
            matches!(&echec.erreur, PublishError::ExclusionRompue(m) if m.contains("reposée")),
            "{}",
            echec.erreur
        );
        assert_eq!(appels, 0, "aucun renommage de dest");
        assert!(!dest_existe);
        assert_eq!(echec.temoin, TemoinApres::Conserve);
        assert!(temoin_existe);
        conservation(&echec.staging);
    }

    #[test]
    fn exclusion_levee_refus_a_la_barriere() {
        let (echec, appels, dest_existe, _) = refus_source(3, Rupture::Levee);
        assert!(matches!(echec.erreur, PublishError::ExclusionRompue(_)));
        assert_eq!(appels, 0);
        assert!(!dest_existe);
    }

    #[test]
    fn exclusion_source_levee_juste_avant_le_rename_refus() {
        // 1 = creer_staging, 2 = pre-check, 3 = barrier, 4 = right before the rename.
        let (echec, appels, dest_existe, temoin_existe) = refus_source(4, Rupture::Levee);
        assert!(
            matches!(echec.erreur, PublishError::ExclusionRompue(_)),
            "{}",
            echec.erreur
        );
        assert_eq!(appels, 0, "aucun renommage");
        assert!(!dest_existe && temoin_existe);
        conservation(&echec.staging);
    }

    #[test]
    fn exclusion_source_levee_apres_le_fsync_publie_non_servi() {
        // 5 = after the fsync of the parent.
        let (echec, appels, dest_existe, temoin_existe) = refus_source(5, Rupture::Levee);
        assert!(
            matches!(echec.erreur, PublishError::ExclusionRompue(_)),
            "{}",
            echec.erreur
        );
        assert_eq!(appels, 1);
        assert!(
            dest_existe && temoin_existe,
            "publié, témoin conservé, ne pas servir"
        );
        assert_eq!(echec.staging, StagingApres::PeutEtrePublie);
    }

    #[test]
    fn identite_source_changee_refus_a_la_barriere() {
        let (echec, appels, dest_existe, _) = refus_source(3, Rupture::IdentiteChangee);
        assert!(
            matches!(echec.erreur, PublishError::IdentiteSourceChangee { .. }),
            "{}",
            echec.erreur
        );
        assert_eq!(appels, 0);
        assert!(!dest_existe);
    }

    // ---- §1 preconditions ----------------------------------------------------------------------

    #[test]
    fn dest_absent_accepte() {
        let (_t, dest) = base();
        verifier_preconditions(&dest).unwrap();
    }

    #[test]
    fn dest_vide_refuse_destination_preexistante() {
        let (_t, dest) = base();
        fs::create_dir(&dest).unwrap();
        let e = verifier_preconditions(&dest).unwrap_err();
        assert!(matches!(e, PublishError::DestPreexistante(_)), "{e}");
        assert!(e
            .to_string()
            .starts_with("destination préexistante : publication refusée, rien supprimé"));
        assert!(dest.is_dir());
    }

    #[test]
    fn dest_non_vide_refuse() {
        let (_t, dest) = base();
        fs::create_dir(&dest).unwrap();
        fs::write(dest.join("f"), b"1").unwrap();
        assert!(matches!(
            verifier_preconditions(&dest).unwrap_err(),
            PublishError::DestNonVide(_)
        ));
    }

    #[test]
    fn dest_fichier_refuse() {
        let (_t, dest) = base();
        fs::write(&dest, b"1").unwrap();
        assert!(matches!(
            verifier_preconditions(&dest).unwrap_err(),
            PublishError::DestPasUnRepertoire(_)
        ));
    }

    #[test]
    fn dest_lien_vers_vide_refuse() {
        let (t, dest) = base();
        fs::create_dir(t.path().join("vide")).unwrap();
        std::os::unix::fs::symlink(t.path().join("vide"), &dest).unwrap();
        assert!(matches!(
            verifier_preconditions(&dest).unwrap_err(),
            PublishError::DestLienSymbolique(_)
        ));
    }

    #[test]
    fn dest_lien_vers_non_vide_refuse() {
        let (t, dest) = base();
        fs::create_dir(t.path().join("plein")).unwrap();
        fs::write(t.path().join("plein/f"), b"1").unwrap();
        std::os::unix::fs::symlink(t.path().join("plein"), &dest).unwrap();
        assert!(matches!(
            verifier_preconditions(&dest).unwrap_err(),
            PublishError::DestLienSymbolique(_)
        ));
    }

    #[test]
    fn dest_lien_casse_refuse() {
        let (t, dest) = base();
        std::os::unix::fs::symlink(t.path().join("nulle-part"), &dest).unwrap();
        assert!(matches!(
            verifier_preconditions(&dest).unwrap_err(),
            PublishError::DestLienSymbolique(_)
        ));
    }

    #[test]
    fn dest_relatif_ou_avec_point_point_refuse() {
        for d in ["relatif/base", "/tmp/a/../base", "/tmp/./base", "/"] {
            assert!(
                matches!(
                    verifier_preconditions(Path::new(d)).unwrap_err(),
                    PublishError::DestNonAbsolu(_)
                ),
                "{d}"
            );
        }
    }

    #[test]
    fn dest_point_de_montage_refuse() {
        let d = Path::new("/x/base");
        let e = evaluer_preconditions(&faits(d, Some(stat_rep(2)), Some(true))).unwrap_err();
        assert!(
            matches!(
                e,
                PublishError::DestPointDeMontage {
                    dev_dest: 2,
                    dev_parent: 1,
                    ..
                }
            ),
            "{e}"
        );
    }

    #[test]
    fn staging_autre_systeme_de_fichiers_refuse() {
        let e = verifier_meme_systeme(Path::new("/x/s"), &stat_rep(2), &stat_rep(1)).unwrap_err();
        assert!(matches!(e, PublishError::AutreSystemeDeFichiers { .. }));
        verifier_meme_systeme(Path::new("/x/s"), &stat_rep(1), &stat_rep(1)).unwrap();
    }

    #[test]
    fn parent_non_inscriptible_refuse() {
        let mut f = faits(Path::new("/x/base"), None, None);
        f.parent_inscriptible = Err(libc::EACCES);
        let e = evaluer_preconditions(&f).unwrap_err();
        assert!(
            matches!(e, PublishError::ParentNonInscriptible { errno, .. } if errno == libc::EACCES),
            "{e}"
        );
    }

    // ---- §2.10 renameat without replacement ----------------------------------------------------

    #[test]
    fn errno_classes_brutes() {
        for e in [libc::EEXIST, libc::ENOTEMPTY] {
            assert_eq!(
                RenameError::depuis_errno(e),
                RenameError::Franc { errno: e }
            );
        }
        for e in [libc::EINVAL, libc::ENOTSUP, libc::EOPNOTSUPP, libc::ENOSYS] {
            assert_eq!(
                RenameError::depuis_errno(e),
                RenameError::NonSupporte { errno: e }
            );
        }
        for e in [libc::EXDEV, libc::EBUSY, libc::EIO] {
            assert_eq!(
                RenameError::depuis_errno(e),
                RenameError::Ambigu { errno: e }
            );
        }
        assert!(RenameError::depuis_errno(libc::EEXIST)
            .to_string()
            .contains(&format!("EEXIST ({})", libc::EEXIST)));
    }

    #[test]
    fn renommage_reel_vers_cible_existante_eexist_brut() {
        let t = tempfile::tempdir().unwrap();
        let a = t.path().join("a");
        let b = t.path().join("b");
        fs::create_dir(&a).unwrap();
        fs::create_dir(&b).unwrap();
        fs::write(b.join("f"), b"1").unwrap();
        let dir =
            rustix::fs::open(t.path(), OFlags::RDONLY | OFlags::DIRECTORY, Mode::empty()).unwrap();
        let e = RustixNoReplace
            .rename_noreplace(dir.as_fd(), OsStr::new("a"), OsStr::new("b"))
            .unwrap_err();
        assert_eq!(
            e,
            RenameError::Franc {
                errno: libc::EEXIST
            }
        );
        assert!(a.is_dir() && b.join("f").is_file());
    }

    #[test]
    fn publication_reelle_reussie() {
        let (t, dest) = base();
        let excl = TestExclusion::tenue();
        let h = staging_rempli(&dest, &excl);
        let p = publier(&h, &dest, &RustixNoReplace).unwrap();
        assert_eq!(p.nonce, h.nonce);
        assert_eq!(fs::read(dest.join("sub/x")).unwrap(), b"xyz");
        assert!(!h.chemin.exists());
        assert!(!chemin_temoin(&dest).exists());
        assert_eq!(lister(t.path()), vec!["base".to_string()]);
    }

    fn echec_franc(errno: i32) {
        let (_t, dest) = base();
        let excl = TestExclusion::tenue();
        let h = staging_rempli(&dest, &excl);
        let ren = FauxRenommage::new(&dest, Some(RenameError::depuis_errno(errno)));
        let echec = publier(&h, &dest, &ren).unwrap_err();
        assert!(matches!(
            echec.erreur,
            PublishError::Renommage(RenameError::Franc { errno: e }) if e == errno
        ));
        let c = conservation(&echec.staging);
        assert!(c.eligible_section6, "§6 : supprimable en principe");
        assert_eq!(c.chemin, h.chemin);
        assert!(c
            .raisons
            .iter()
            .any(|r| r == RAISON_PROPRIETE_NON_DEMONTREE));
        assert_staging_intact(&h);
        assert!(
            dest.is_dir() && lister(&dest).is_empty(),
            "dest d'un tiers (autre inode) intact"
        );
    }

    #[test]
    fn eexist_staging_supprimable_mais_conserve() {
        echec_franc(libc::EEXIST);
    }

    #[test]
    fn enotempty_staging_supprimable_mais_conserve() {
        echec_franc(libc::ENOTEMPTY);
    }

    #[test]
    fn non_supporte_refus_sans_repli() {
        for errno in [libc::EINVAL, libc::ENOTSUP, libc::ENOSYS] {
            let (_t, dest) = base();
            let excl = TestExclusion::tenue();
            let h = staging_rempli(&dest, &excl);
            let ren = FauxRenommage::new(&dest, Some(RenameError::depuis_errno(errno)));
            let echec = publier(&h, &dest, &ren).unwrap_err();
            assert!(matches!(
                echec.erreur,
                PublishError::Renommage(RenameError::NonSupporte { errno: e }) if e == errno
            ));
            assert_eq!(ren.appels(), 1, "une seule tentative");
            assert!(!dest.exists(), "aucun repli check-then-rename");
            assert!(!conservation(&echec.staging).eligible_section6);
            assert_staging_intact(&h);
        }
    }

    #[test]
    fn renommage_ambigu_tout_conserve() {
        let (_t, dest) = base();
        let excl = TestExclusion::tenue();
        let h = staging_rempli(&dest, &excl);
        let ren = FauxRenommage::new(&dest, Some(RenameError::depuis_errno(libc::EIO)));
        let echec = publier(&h, &dest, &ren).unwrap_err();
        assert_eq!(echec.staging, StagingApres::PeutEtrePublie);
        assert_eq!(echec.temoin, TemoinApres::Conserve);
        assert!(chemin_temoin(&dest).exists());
        assert_staging_intact(&h);
    }

    // ---- §4 fsync of the parent ----------------------------------------------------------------

    #[test]
    fn succes_puis_fsync_parent_en_echec_conserve_non_servi() {
        let (_t, dest) = base();
        let excl = TestExclusion::tenue();
        let h = staging_rempli(&dest, &excl);
        // Call 1 = after the witness, call 2 = after the rename.
        let sync = FauxSync {
            echec_appel: Some(2),
            ..Default::default()
        };
        let echec =
            publish_with(&h, &dest, &RustixNoReplace, &sync, SEUIL_PETIT_FICHIER).unwrap_err();
        assert!(matches!(
            echec.erreur,
            PublishError::PublieDurabiliteNonConfirmee(_)
        ));
        assert_eq!(echec.staging, StagingApres::PeutEtrePublie);
        assert_eq!(echec.temoin, TemoinApres::Conserve);
        assert!(dest.join("data.bin").is_file(), "dest conservé");
        assert!(chemin_temoin(&dest).exists(), "témoin conservé pour §3");
    }

    // ---- concurrent creation of `dest` ---------------------------------------------------------

    #[test]
    fn dest_vide_cree_par_un_tiers_avant_la_barriere_subsiste() {
        let (_t, dest) = base();
        let excl = TestExclusion::tenue();
        let h = staging_rempli(&dest, &excl);
        let (tx, rx) = mpsc::channel::<u64>();
        let d = dest.clone();
        // A third party creates an empty `dest` right after the witness (fsync call 1).
        let sync = FauxSync {
            action: Some((
                1,
                Box::new(move || {
                    let d = d.clone();
                    let tx = tx.clone();
                    std::thread::spawn(move || {
                        fs::create_dir(&d).unwrap();
                        tx.send(fs::metadata(&d).unwrap().ino()).unwrap();
                    })
                    .join()
                    .unwrap();
                }),
            )),
            ..Default::default()
        };
        let echec =
            publish_with(&h, &dest, &RustixNoReplace, &sync, SEUIL_PETIT_FICHIER).unwrap_err();
        let ino_tiers = rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(
            matches!(echec.erreur, PublishError::DestPreexistante(_)),
            "{}",
            echec.erreur
        );
        assert_eq!(
            fs::metadata(&dest).unwrap().ino(),
            ino_tiers,
            "dest du tiers intact"
        );
        assert!(lister(&dest).is_empty());
        conservation(&echec.staging);
        assert_staging_intact(&h);
    }

    #[test]
    fn dest_vide_cree_par_un_tiers_entre_barriere_et_renommage_subsiste() {
        // The REAL renameat answers EEXIST (dest exists, another inode): a verified clean
        // failure, hence EchecFranc (eligible to §6 in principle, still conserved) and the guard
        // released by the Drop.
        let (_t, dest) = base();
        let rep = TestRepertoire::tenue();
        {
            let h = staging_garde(&dest, &rep);
            let mut ren = FauxRenommage::new(&dest, None);
            let d = dest.clone();
            ren.avant = Some(Box::new(move || fs::create_dir(&d).unwrap()));
            let echec = publier(&h, &dest, &ren).unwrap_err();
            assert!(
                matches!(
                    echec.erreur,
                    PublishError::Renommage(RenameError::Franc { errno }) if errno == libc::EEXIST
                ),
                "{}",
                echec.erreur
            );
            assert!(
                conservation(&echec.staging).eligible_section6,
                "statut EchecFranc"
            );
            assert!(dest.is_dir() && lister(&dest).is_empty());
            assert_staging_intact(&h);
        }
        assert_eq!(rep.relachements(), 1);
    }

    // ---- §2.8 witness ---------------------------------------------------------------------------

    fn temoin_de(dest: &Path, nonce: &str) -> Temoin {
        Temoin {
            source: SourceId("source-A".into()),
            nonce: nonce.into(),
            manifeste: "m1".into(),
            dest: dest_canonique(dest).unwrap(),
        }
    }

    #[test]
    fn temoin_aller_retour() {
        let (_t, dest) = base();
        let t = temoin_de(&dest, "n1");
        let p = ecrire_temoin_test(&dest, &t);
        assert_eq!(lire_temoin(&p).unwrap(), t);
        lire_temoin(&p).unwrap().valider(&t).unwrap();
    }

    #[test]
    fn temoin_tronque_refus_et_conservation() {
        let (t, dest) = base();
        let excl = TestExclusion::tenue();
        let h = staging_rempli(&dest, &excl);
        let tm = temoin_de(&dest, &h.nonce);
        let p = ecrire_temoin_test(&dest, &tm);
        let complet = fs::read(&p).unwrap();
        for coupe in [complet.len() / 2, complet.len() - 5, complet.len() - 1] {
            fs::write(&p, &complet[..coupe]).unwrap();
            let obs = observer_temoin(&dest);
            assert!(matches!(obs, TemoinObserve::Invalide(_)), "coupe {coupe}");
            for stagings in [vec![h.chemin.clone()], vec![]] {
                let e = EtatDisque {
                    temoin: obs.clone(),
                    stagings,
                    ..etat_vierge(&dest)
                };
                assert!(
                    matches!(
                        classify(&e),
                        Decision::Refuse {
                            motif: Motif::TemoinInvalide(_)
                        }
                    ),
                    "coupe {coupe}"
                );
            }
        }
        assert!(p.exists(), "témoin conservé");
        assert_staging_intact(&h);
        let _ = t;
    }

    #[test]
    fn temoin_d_un_autre_nonce_refuse() {
        let (_t, dest) = base();
        let ecrit = temoin_de(&dest, "n1");
        let attendu = temoin_de(&dest, "n2");
        assert!(matches!(
            ecrit.valider(&attendu).unwrap_err(),
            PublishError::TemoinAutreNonce { .. }
        ));
        let e = EtatDisque {
            temoin: TemoinObserve::Valide(ecrit),
            exclusion_continue_demontree: true,
            reprise_demandee: Some("n2".into()),
            ..etat_vierge(&dest)
        };
        assert!(matches!(
            classify(&e),
            Decision::Refuse {
                motif: Motif::TemoinAutreNonce { .. }
            }
        ));
    }

    // ---- §2.7 manifest ---------------------------------------------------------------------------

    #[test]
    fn manifeste_hash_des_petits_fichiers_seulement() {
        let t = tempfile::tempdir().unwrap();
        fs::write(t.path().join("petit"), b"abc").unwrap();
        fs::write(t.path().join("gros"), vec![0u8; 64]).unwrap();
        let m = manifeste_de(t.path(), 10).unwrap();
        let petit = m.entrees.iter().find(|e| e.chemin == "petit").unwrap();
        let gros = m.entrees.iter().find(|e| e.chemin == "gros").unwrap();
        assert_eq!(
            petit.hash.as_deref(),
            Some(hex::encode(Sha256::digest(b"abc")).as_str())
        );
        assert_eq!(gros.hash, None);
        assert_eq!(gros.taille, 64);
    }

    #[test]
    fn manifeste_ecriture_apres_manifeste_detectee() {
        let t = tempfile::tempdir().unwrap();
        fs::write(t.path().join("f"), b"abc").unwrap();
        let m = manifeste_de(t.path(), SEUIL_PETIT_FICHIER).unwrap();
        comparer_a(&m, t.path(), SEUIL_PETIT_FICHIER).unwrap();
        fs::write(t.path().join("f"), b"abd").unwrap();
        assert!(matches!(
            comparer_a(&m, t.path(), SEUIL_PETIT_FICHIER).unwrap_err(),
            PublishError::ManifesteDivergent(_)
        ));
    }

    #[test]
    fn barriere_refuse_une_ecriture_dans_le_staging_apres_le_manifeste() {
        let (_t, dest) = base();
        let excl = TestExclusion::tenue();
        let h = staging_rempli(&dest, &excl);
        let f = h.chemin.join("data.bin");
        let sync = FauxSync {
            action: Some((1, Box::new(move || fs::write(&f, b"abX").unwrap()))),
            ..Default::default()
        };
        let ren = FauxRenommage::new(&dest, None);
        let echec = publish_with(&h, &dest, &ren, &sync, SEUIL_PETIT_FICHIER).unwrap_err();
        assert!(
            matches!(echec.erreur, PublishError::ManifesteDivergent(_)),
            "{}",
            echec.erreur
        );
        assert_eq!(ren.appels(), 0);
        assert!(!dest.exists());
    }

    #[test]
    fn manifeste_substitution_concurrente_par_un_lien_refusee() {
        let t = tempfile::tempdir().unwrap();
        let racine = t.path().join("staging");
        fs::create_dir_all(racine.join("sub")).unwrap();
        fs::write(racine.join("sub/x"), b"1").unwrap();
        let dehors = t.path().join("dehors");
        fs::create_dir(&dehors).unwrap();
        fs::write(dehors.join("marqueur-hors-staging"), b"!").unwrap();

        let (go_tx, go_rx) = mpsc::channel::<()>();
        let (fait_tx, fait_rx) = mpsc::channel::<()>();
        let (r2, d2, t2) = (racine.clone(), dehors.clone(), t.path().to_path_buf());
        let fil = std::thread::spawn(move || {
            go_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            fs::rename(r2.join("sub"), t2.join("sub-deplace")).unwrap();
            std::os::unix::fs::symlink(&d2, r2.join("sub")).unwrap();
            fait_tx.send(()).unwrap();
        });
        let fd = rustix::fs::open(
            &racine,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .unwrap();
        let crochet = |rel: &str| {
            if rel == "sub" {
                go_tx.send(()).unwrap();
                fait_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            }
        };
        let r = manifeste_depuis_fd(&fd, SEUIL_PETIT_FICHIER, &crochet);
        fil.join().unwrap();
        match r {
            Err(PublishError::ManifesteNonFiable(m)) => assert!(m.contains("sub"), "{m}"),
            Err(e) => panic!("refus inattendu : {e}"),
            Ok(m) => panic!(
                "aurait dû refuser ; entrées : {:?}",
                m.entrees.iter().map(|e| &e.chemin).collect::<Vec<_>>()
            ),
        }
        assert!(PublishError::ManifesteNonFiable(String::new())
            .to_string()
            .starts_with("manifeste non fiable sous concurrence"));
    }

    #[test]
    fn manifeste_refuse_un_lien_dans_le_staging() {
        let t = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink("/", t.path().join("l")).unwrap();
        assert!(matches!(
            manifeste_de(t.path(), SEUIL_PETIT_FICHIER).unwrap_err(),
            PublishError::ManifesteRefuse(_)
        ));
    }

    // ---- §6 + v6.1: nothing is ever deleted --------------------------------------------------

    #[test]
    fn fichier_tiers_de_meme_uid_dans_le_staging_rien_supprime() {
        let (_t, dest) = base();
        let excl = TestExclusion::tenue();
        let h = staging_rempli(&dest, &excl);
        let c = h.chemin.clone();
        // Another thread of the same uid writes an arbitrary name and an engine-like name.
        std::thread::spawn(move || {
            fs::write(c.join("intrus-tiers"), b"tiers").unwrap();
            fs::write(c.join("fence"), b"tiers").unwrap();
        })
        .join()
        .unwrap();
        let ren = FauxRenommage::new(
            &dest,
            Some(RenameError::Franc {
                errno: libc::EEXIST,
            }),
        );
        let echec = publier(&h, &dest, &ren).unwrap_err();
        let cons = conservation(&echec.staging);
        assert_eq!(cons.chemin, h.chemin);
        assert!(cons
            .raisons
            .iter()
            .any(|r| r == RAISON_PROPRIETE_NON_DEMONTREE));
        assert_eq!(fs::read(h.chemin.join("intrus-tiers")).unwrap(), b"tiers");
        assert_eq!(fs::read(h.chemin.join("fence")).unwrap(), b"tiers");
        assert_staging_intact(&h);
        // And directly, for every status.
        for s in [
            StatutRenommage::NonTente,
            StatutRenommage::EchecFranc {
                errno: libc::ENOTEMPTY,
            },
        ] {
            conservation(&decider_suppression(&h, s));
            assert_staging_intact(&h);
            assert!(h.chemin.join("intrus-tiers").exists());
        }
    }

    #[test]
    fn staging_avec_lien_symbolique_aucune_suppression() {
        let (t, dest) = base();
        let h = staging_rempli(&dest, &SourceTenue);
        std::os::unix::fs::symlink(t.path(), h.chemin.join("lien")).unwrap();
        let c = decider_suppression(&h, StatutRenommage::NonTente);
        assert!(conservation(&c)
            .raisons
            .iter()
            .any(|r| r.contains("lien symbolique")));
        assert!(fs::symlink_metadata(h.chemin.join("lien")).is_ok());
        assert_staging_intact(&h);
    }

    #[test]
    fn staging_avec_nlink_superieur_a_1_aucune_suppression() {
        let (t, dest) = base();
        let h = staging_rempli(&dest, &SourceTenue);
        fs::hard_link(h.chemin.join("data.bin"), t.path().join("ailleurs")).unwrap();
        let c = decider_suppression(&h, StatutRenommage::NonTente);
        assert!(conservation(&c).raisons.iter().any(|r| r.contains("nlink")));
        assert_staging_intact(&h);
    }

    #[test]
    fn staging_remplace_identite_changee_conserve() {
        let (t, dest) = base();
        let h = staging_rempli(&dest, &SourceTenue);
        let deplace = t.path().join("deplace");
        fs::rename(&h.chemin, &deplace).unwrap();
        fs::create_dir(&h.chemin).unwrap();
        fs::write(h.chemin.join("autre"), b"1").unwrap();
        let c = decider_suppression(&h, StatutRenommage::NonTente);
        assert!(conservation(&c)
            .raisons
            .iter()
            .any(|r| r.contains("dev/ino")));
        assert!(h.chemin.join("autre").is_file());
        assert_eq!(fs::read(deplace.join("data.bin")).unwrap(), b"abc");
    }

    #[test]
    fn suppression_jamais_apres_renommage_reussi_ou_ambigu() {
        let (_t, dest) = base();
        let h = staging_rempli(&dest, &SourceTenue);
        for s in [StatutRenommage::Reussi, StatutRenommage::Ambigu] {
            assert_eq!(decider_suppression(&h, s), StagingApres::PeutEtrePublie);
        }
        assert_staging_intact(&h);
    }

    #[test]
    fn orphelin_d_une_autre_execution_conserve() {
        let (_t, dest) = base();
        let excl = TestExclusion::tenue();
        let h = staging_rempli(&dest, &excl);
        let chemin = h.chemin.clone();
        drop(h); // another run: no handle survives
        let traces = scanner_traces(&dest).unwrap();
        assert_eq!(traces.stagings, vec![chemin.clone()]);
        let e = EtatDisque {
            stagings: traces.stagings.clone(),
            ..etat_vierge(&dest)
        };
        assert!(matches!(
            classify(&e),
            Decision::Refuse {
                motif: Motif::OrphelinConserve(_)
            }
        ));
        assert!(matches!(
            creer_staging(&dest, &excl, &RepertoireExcluPretendu).unwrap_err(),
            PublishError::TracesExistantes(_)
        ));
        assert!(chemin.join("sub/x").is_file());
        assert_eq!(scanner_traces(&dest).unwrap().stagings, vec![chemin]);
    }

    // ---- classify ---------------------------------------------------------------------------

    fn etat_vierge(dest: &Path) -> EtatDisque {
        EtatDisque {
            dest: EtatDestObserve::Absent,
            dest_publie_valide: false,
            temoin: TemoinObserve::Absent,
            temoin_tmp_present: false,
            stagings: vec![],
            source_actuelle: Some(SourceId("source-A".into())),
            dest_canonique_actuel: dest_canonique(dest).unwrap(),
            manifeste_dest: None,
            nonce_proprietaire_dest: None,
            exclusion_disponible: true,
            exclusion_continue_demontree: false,
            reprise_demandee: None,
        }
    }

    fn adoptable(dest: &Path) -> EtatDisque {
        EtatDisque {
            dest: EtatDestObserve::RepertoireNonVide,
            dest_publie_valide: true,
            temoin: TemoinObserve::Valide(temoin_de(dest, "n1")),
            manifeste_dest: Some("m1".into()),
            nonce_proprietaire_dest: Some("n1".into()),
            exclusion_continue_demontree: true,
            ..etat_vierge(dest)
        }
    }

    fn motif(d: Decision) -> Motif {
        match d {
            Decision::Refuse { motif } => motif,
            autre => panic!("attendu Refuse, obtenu {autre:?}"),
        }
    }

    #[test]
    fn classify_adoption_quand_tout_est_demontre() {
        let (_t, dest) = base();
        assert_eq!(
            classify(&adoptable(&dest)),
            Decision::AdopterApresFsync { nonce: "n1".into() }
        );
    }

    #[test]
    fn classify_adoption_contradictoire_refusee() {
        let (_t, dest) = base();
        let a = adoptable(&dest);
        let cas: Vec<(EtatDisque, Motif)> = vec![
            (
                EtatDisque {
                    dest_publie_valide: false,
                    ..a.clone()
                },
                Motif::DestPublieInvalide,
            ),
            (
                EtatDisque {
                    stagings: vec![dest.with_file_name(".base.po-staging-x")],
                    ..a.clone()
                },
                Motif::OrphelinConserve(vec![dest.with_file_name(".base.po-staging-x")]),
            ),
            (
                EtatDisque {
                    exclusion_continue_demontree: false,
                    ..a.clone()
                },
                Motif::AdoptionSansExclusionContinue,
            ),
            (
                EtatDisque {
                    dest: EtatDestObserve::Absent,
                    ..a.clone()
                },
                Motif::TemoinSansDest,
            ),
            (
                EtatDisque {
                    dest: EtatDestObserve::RepertoireVide,
                    ..a.clone()
                },
                Motif::TemoinSansDest,
            ),
            (
                EtatDisque {
                    manifeste_dest: Some("m2".into()),
                    ..a.clone()
                },
                Motif::ManifesteDivergent,
            ),
            (
                EtatDisque {
                    manifeste_dest: None,
                    ..a.clone()
                },
                Motif::ManifesteDivergent,
            ),
            (
                EtatDisque {
                    temoin_tmp_present: true,
                    ..a.clone()
                },
                Motif::TemoinTemporairePresent,
            ),
        ];
        for (e, attendu) in cas {
            assert_eq!(motif(classify(&e)), attendu);
        }
    }

    /// Expected decision, written from the documented priority order of [`classify`].
    fn decision_attendue(e: &EtatDisque) -> Decision {
        let r = |motif| Decision::Refuse { motif };
        if e.dest == EtatDestObserve::Dangereux {
            return r(Motif::DestDangereux);
        }
        if e.temoin_tmp_present {
            return r(Motif::TemoinTemporairePresent);
        }
        let t = match &e.temoin {
            TemoinObserve::Invalide(m) => return r(Motif::TemoinInvalide(m.clone())),
            TemoinObserve::Valide(t) => Some(t),
            TemoinObserve::Absent => None,
        };
        if let Some(t) = t {
            if t.dest != e.dest_canonique_actuel {
                return r(Motif::CheminCanoniqueChange {
                    avant: t.dest.clone(),
                    apres: e.dest_canonique_actuel.clone(),
                });
            }
            if e.source_actuelle.as_ref() != Some(&t.source) {
                return r(Motif::TemoinNonLie(String::new()));
            }
        }
        if let Some(n) = &e.reprise_demandee {
            if !e.exclusion_continue_demontree {
                return r(Motif::RepriseSansExclusionContinue);
            }
            if let Some(t) = t {
                if &t.nonce != n {
                    return r(Motif::TemoinAutreNonce {
                        demande: n.clone(),
                        trouve: t.nonce.clone(),
                    });
                }
            }
            return r(Motif::RepriseNonImplementee);
        }
        if let Some(t) = t {
            if !e.stagings.is_empty() {
                return r(Motif::OrphelinConserve(e.stagings.clone()));
            }
            if e.dest != EtatDestObserve::RepertoireNonVide {
                return r(Motif::TemoinSansDest);
            }
            if !e.dest_publie_valide {
                return r(Motif::DestPublieInvalide);
            }
            if e.manifeste_dest.as_deref() != Some(t.manifeste.as_str()) {
                return r(Motif::ManifesteDivergent);
            }
            if e.nonce_proprietaire_dest.as_deref() != Some(t.nonce.as_str()) {
                return r(Motif::ProprietaireNonLie);
            }
            if !e.exclusion_continue_demontree {
                return r(Motif::AdoptionSansExclusionContinue);
            }
            return Decision::AdopterApresFsync {
                nonce: t.nonce.clone(),
            };
        }
        if !e.stagings.is_empty() {
            return r(Motif::OrphelinConserve(e.stagings.clone()));
        }
        match e.dest {
            EtatDestObserve::RepertoireNonVide if e.dest_publie_valide => Decision::ServirPublie,
            EtatDestObserve::RepertoireNonVide => r(Motif::DestNonVideNonPublie),
            EtatDestObserve::RepertoireVide => r(Motif::DestPreexistante),
            _ if !e.exclusion_disponible => r(Motif::SourceNonGelee),
            _ => Decision::NouvelleCopie,
        }
    }

    /// Same decision; for the two motives carrying a free-form message, the variant only.
    fn meme_decision(a: &Decision, b: &Decision) -> bool {
        match (a, b) {
            (
                Decision::Refuse {
                    motif: Motif::TemoinNonLie(_),
                },
                Decision::Refuse {
                    motif: Motif::TemoinNonLie(_),
                },
            ) => true,
            _ => a == b,
        }
    }

    #[test]
    fn classify_table_exhaustive_aucune_combinaison_contradictoire_n_adopte() {
        let (_t, dest) = base();
        let canon = dest_canonique(&dest).unwrap();
        let lie = temoin_de(&dest, "n1");
        let temoins = [
            ("absent", TemoinObserve::Absent),
            ("invalide", TemoinObserve::Invalide("tronqué".into())),
            ("lie", TemoinObserve::Valide(lie.clone())),
            (
                "autre-source",
                TemoinObserve::Valide(Temoin {
                    source: SourceId("source-Z".into()),
                    ..lie.clone()
                }),
            ),
            (
                "autre-dest",
                TemoinObserve::Valide(Temoin {
                    dest: PathBuf::from("/ailleurs/base"),
                    ..lie.clone()
                }),
            ),
        ];
        let dests = [
            EtatDestObserve::Absent,
            EtatDestObserve::RepertoireVide,
            EtatDestObserve::RepertoireNonVide,
            EtatDestObserve::Dangereux,
        ];
        let manifestes = [None, Some("m1".to_string()), Some("m2".to_string())];
        let proprietaires = [None, Some("n1".to_string()), Some("n9".to_string())];
        let reprises = [None, Some("n1".to_string()), Some("n9".to_string())];
        let sources = [
            Some(SourceId("source-A".into())),
            Some(SourceId("source-Z".into())),
            None,
        ];
        let canons = [canon.clone(), PathBuf::from("/autre/volume/base")];
        let mut n = 0;
        let mut adoptions = 0;
        for d in dests {
            for publie in [false, true] {
                for (_nom_t, t) in &temoins {
                    for tmp in [false, true] {
                        for stg in [false, true] {
                            for m in &manifestes {
                                for prop in &proprietaires {
                                    for dispo in [false, true] {
                                        for cont in [false, true] {
                                            for rep in &reprises {
                                                for src in &sources {
                                                    for cn in &canons {
                                                        n += 1;
                                                        let e = EtatDisque {
                                                            dest: d,
                                                            dest_publie_valide: publie,
                                                            temoin: t.clone(),
                                                            temoin_tmp_present: tmp,
                                                            stagings: if stg {
                                                                vec![dest.with_file_name(
                                                                    ".base.po-staging-x",
                                                                )]
                                                            } else {
                                                                vec![]
                                                            },
                                                            source_actuelle: src.clone(),
                                                            dest_canonique_actuel: cn.clone(),
                                                            manifeste_dest: m.clone(),
                                                            nonce_proprietaire_dest: prop.clone(),
                                                            exclusion_disponible: dispo,
                                                            exclusion_continue_demontree: cont,
                                                            reprise_demandee: rep.clone(),
                                                        };
                                                        let obtenu = classify(&e);
                                                        let attendu = decision_attendue(&e);
                                                        assert!(
                                                            meme_decision(&obtenu, &attendu),
                                                            "{e:?}\nobtenu {obtenu:?}\nattendu {attendu:?}"
                                                        );
                                                        if let Decision::Refuse { motif } = &obtenu
                                                        {
                                                            assert_eq!(
                                                                motif.est_bail(),
                                                                d == EtatDestObserve::Dangereux
                                                            );
                                                        }
                                                        let lie_ici = matches!(t, TemoinObserve::Valide(t)
                                                            if src.as_ref() == Some(&t.source) && t.dest == *cn);
                                                        let tout_demontre = !tmp
                                                            && !stg
                                                            && rep.is_none()
                                                            && d == EtatDestObserve::RepertoireNonVide
                                                            && publie
                                                            && lie_ici
                                                            && m.as_deref() == Some("m1")
                                                            && prop.as_deref() == Some("n1")
                                                            && cont;
                                                        let adopte = matches!(
                                                            obtenu,
                                                            Decision::AdopterApresFsync { .. }
                                                        );
                                                        assert_eq!(adopte, tout_demontre, "{e:?}");
                                                        adoptions += adopte as usize;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(n, 4 * 2 * 5 * 2 * 2 * 3 * 3 * 2 * 2 * 3 * 3 * 2);
        // (témoin lié, source A, chemin canonique) et (témoin de source Z, source actuelle Z) :
        // les seules combinaisons entièrement démontrées, chacune x 2 (exclusion disponible).
        assert_eq!(adoptions, 4);
    }

    #[test]
    fn classify_temoin_sans_dest_conserve_sans_renommage() {
        let (_t, dest) = base();
        let e = EtatDisque {
            temoin: TemoinObserve::Valide(temoin_de(&dest, "n1")),
            stagings: vec![],
            ..etat_vierge(&dest)
        };
        assert_eq!(motif(classify(&e)), Motif::TemoinSansDest);
    }

    #[test]
    fn classify_reprise_refusee_sans_exclusion_continue() {
        let (_t, dest) = base();
        let mut e = EtatDisque {
            temoin: TemoinObserve::Valide(temoin_de(&dest, "n1")),
            stagings: vec![dest.with_file_name(".base.po-staging-n1")],
            reprise_demandee: Some("n1".into()),
            ..etat_vierge(&dest)
        };
        assert_eq!(motif(classify(&e)), Motif::RepriseSansExclusionContinue);
        e.exclusion_continue_demontree = true;
        assert_eq!(motif(classify(&e)), Motif::RepriseNonImplementee);
    }

    #[test]
    fn classify_chemin_canonique_change_refuse() {
        let (_t, dest) = base();
        let e = EtatDisque {
            dest_canonique_actuel: PathBuf::from("/autre/volume/base"),
            ..adoptable(&dest)
        };
        assert!(matches!(
            motif(classify(&e)),
            Motif::CheminCanoniqueChange { .. }
        ));
    }

    // ---- directory exclusion, substitution races ------------------------------------------

    #[test]
    fn defaut_repertoire_refuse_a_l_acquisition_rien_cree() {
        let (t, dest) = base();
        let avant = lister(t.path());
        let e = creer_staging(&dest, &TestExclusion::tenue(), &RepertoireNonExclu).unwrap_err();
        assert!(matches!(e, PublishError::RepertoireNonExclu(_)), "{e}");
        assert_eq!(lister(t.path()), avant, "rien créé");
    }

    #[test]
    fn garde_tenue_de_bout_en_bout_publication_acceptee_puis_relachee() {
        let (_t, dest) = base();
        let excl = TestExclusion::tenue();
        let rep = TestRepertoire::tenue();
        let h = creer_staging(&dest, &excl, &rep).unwrap();
        fs::write(h.chemin.join("data.bin"), b"abc").unwrap();
        publier(&h, &dest, &RustixNoReplace).unwrap();
        assert_eq!(fs::read(dest.join("data.bin")).unwrap(), b"abc");
        assert_eq!(rep.relachements(), 1);
        drop(h);
        assert_eq!(rep.relachements(), 1, "pas de double relâchement au Drop");
    }

    #[test]
    fn garde_rompue_pendant_la_copie_puis_reacquise_refus_et_conservation() {
        let (t, dest) = base();
        let excl = TestExclusion::tenue();
        let rep = TestRepertoire::tenue();
        let h = creer_staging(&dest, &excl, &rep).unwrap();
        rep.lever(); // lost during the copy ...
        fs::write(h.chemin.join("data.bin"), b"abc").unwrap();
        rep.reprendre(); // ... and held again before the barrier
        let avant = lister(t.path());
        let ren = FauxRenommage::new(&dest, None);
        let echec = publier(&h, &dest, &ren).unwrap_err();
        assert!(
            matches!(&echec.erreur, PublishError::RepertoireNonExclu(m) if m.contains("non continue")),
            "{}",
            echec.erreur
        );
        assert_eq!(ren.appels(), 0);
        assert_eq!(lister(t.path()), avant);
        conservation(&echec.staging);
        assert_eq!(fs::read(h.chemin.join("data.bin")).unwrap(), b"abc");
    }

    #[test]
    fn parent_dont_l_identite_a_change_refus() {
        let t = tempfile::tempdir().unwrap();
        let p = t.path().join("p");
        fs::create_dir(&p).unwrap();
        let dest = p.join("base");
        let excl = TestExclusion::tenue();
        let h = staging_rempli(&dest, &excl);
        fs::rename(&p, t.path().join("p-deplace")).unwrap();
        fs::create_dir(&p).unwrap();
        let ren = FauxRenommage::new(&dest, None);
        let echec = publier(&h, &dest, &ren).unwrap_err();
        assert!(
            matches!(echec.erreur, PublishError::ParentChange(_)),
            "{}",
            echec.erreur
        );
        assert_eq!(ren.appels(), 0);
        assert!(lister(&p).is_empty(), "rien créé dans le nouveau parent");
        assert!(t
            .path()
            .join("p-deplace")
            .join(h.chemin.file_name().unwrap())
            .join("sub/x")
            .is_file());
    }

    /// Fake rename that, right before the real rename of the staging, moves the checked staging
    /// away and puts another directory (other content, same name) in its place.
    fn renommage_substituant(dest: &Path, h: &StagingHandle, ailleurs: &Path) -> FauxRenommage {
        let mut ren = FauxRenommage::new(dest, None);
        let (s, a) = (h.chemin.clone(), ailleurs.to_path_buf());
        ren.avant = Some(Box::new(move || {
            fs::rename(&s, &a).unwrap();
            fs::create_dir(&s).unwrap();
            fs::write(s.join("substitut"), b"tiers").unwrap();
        }));
        ren
    }

    #[test]
    fn staging_substitue_avant_rename_sans_exclusion_repertoire_refus_avant_rename() {
        let (t, dest) = base();
        let rep = TestRepertoire::tenue();
        let h = creer_staging(&dest, &SourceTenue, &rep).unwrap();
        fs::write(h.chemin.join("data.bin"), b"abc").unwrap();
        let ailleurs = t.path().join("staging-deplace");
        let ren = renommage_substituant(&dest, &h, &ailleurs);
        // The guard that issued the staging is no longer held.
        rep.lever();
        let echec = publier(&h, &dest, &ren).unwrap_err();
        assert!(matches!(echec.erreur, PublishError::RepertoireNonExclu(_)));
        assert_eq!(ren.appels(), 0, "refus avant tout rename");
        assert!(!dest.exists() && !ailleurs.exists());
        assert_eq!(fs::read(h.chemin.join("data.bin")).unwrap(), b"abc");
    }

    #[test]
    fn staging_substitue_avant_rename_detecte_apres_rename_rien_supprime() {
        let (t, dest) = base();
        let excl = TestExclusion::tenue();
        let h = staging_rempli(&dest, &excl);
        let ailleurs = t.path().join("staging-deplace");
        let ren = renommage_substituant(&dest, &h, &ailleurs);
        let echec = publier(&h, &dest, &ren).unwrap_err();
        assert!(
            matches!(echec.erreur, PublishError::ObjetPublieDivergent(_)),
            "{}",
            echec.erreur
        );
        assert_eq!(echec.staging, StagingApres::PeutEtrePublie);
        assert_eq!(echec.temoin, TemoinApres::Conserve);
        assert!(
            chemin_temoin(&dest).exists(),
            "témoin conservé pour l'opérateur"
        );
        assert_eq!(
            fs::read(dest.join("substitut")).unwrap(),
            b"tiers",
            "rien supprimé"
        );
        assert_eq!(
            fs::read(ailleurs.join("sub/x")).unwrap(),
            b"xyz",
            "staging vérifié intact"
        );
    }

    #[test]
    fn temoin_substitue_entre_validation_et_suppression_conserve() {
        let (_t, dest) = base();
        let rep = TestRepertoire::tenue();
        let parent = parent_garde(&dest, &rep);
        let ctx = parent.ctx(&rep);
        let tm = temoin_de(&dest, "n1");
        let p = ecrire_temoin(&dest, &ctx, &tm, &RustixNoReplace, &SyncReel).unwrap();
        let (p2, tiers) = (p.clone(), dest.with_file_name("tiers.tmp"));
        // Between the validation and the unlink: the guard is lost and a third party
        // substitutes the witness (same name, other content).
        let crochet = || {
            rep.lever();
            fs::write(&tiers, b"fichier tiers").unwrap();
            fs::rename(&tiers, &p2).unwrap();
        };
        let r = retirer_temoin(&dest, &ctx, &tm, &SyncReel, &crochet).unwrap_err();
        assert!(
            r.message.starts_with(TEMOIN_CONSERVE_SANS_EXCLUSION),
            "{}",
            r.message
        );
        assert!(!r.supprime);
        assert_eq!(
            fs::read(&p).unwrap(),
            b"fichier tiers",
            "fichier tiers intact"
        );
    }

    // ---- review of #1270: guards at every step, traces, witness, manifest ------------------

    use std::sync::Arc;

    /// Staging created under `rep`, filled like [`staging_rempli`].
    fn staging_garde<'x>(dest: &Path, rep: &'x dyn ExclusionRepertoire) -> StagingHandle<'x> {
        let h = creer_staging(dest, &SourceTenue, rep).unwrap();
        fs::write(h.chemin.join("data.bin"), b"abc").unwrap();
        fs::create_dir(h.chemin.join("sub")).unwrap();
        fs::write(h.chemin.join("sub/x"), b"xyz").unwrap();
        h
    }

    // Continuity checks of the directory guard, in order: creer_staging (1, 2), pre-check (3),
    // witness (4, 5), barrier (6), right before the rename (7), after the fsync (8), witness
    // removal (9).

    #[test]
    fn garde_perdue_apres_le_temoin_refus_a_la_barriere() {
        let (_t, dest) = base();
        let rep = TestRepertoire::levee_au(6);
        let h = staging_garde(&dest, &rep);
        let ren = FauxRenommage::new(&dest, None);
        let echec = publier(&h, &dest, &ren).unwrap_err();
        assert!(
            matches!(echec.erreur, PublishError::RepertoireNonExclu(_)),
            "{}",
            echec.erreur
        );
        assert_eq!(ren.appels(), 0);
        assert!(!dest.exists() && chemin_temoin(&dest).exists());
        assert_staging_intact(&h);
    }

    #[test]
    fn garde_perdue_entre_la_barriere_et_le_rename_refus() {
        let (_t, dest) = base();
        let rep = TestRepertoire::levee_au(7);
        let h = staging_garde(&dest, &rep);
        let ren = FauxRenommage::new(&dest, None);
        let echec = publier(&h, &dest, &ren).unwrap_err();
        assert!(
            matches!(echec.erreur, PublishError::RepertoireNonExclu(_)),
            "{}",
            echec.erreur
        );
        assert_eq!(ren.appels(), 0, "aucun rename");
        assert!(!dest.exists());
        assert_staging_intact(&h);
    }

    #[test]
    fn garde_perdue_entre_le_rename_et_le_fsync_publie_non_servi() {
        let (_t, dest) = base();
        let rep = Arc::new(TestRepertoire::tenue());
        let h = staging_garde(&dest, &*rep);
        let r2 = Arc::clone(&rep);
        // fsync call 2 is the one right after the rename.
        let sync = FauxSync {
            action: Some((2, Box::new(move || r2.lever()))),
            ..Default::default()
        };
        let echec =
            publish_with(&h, &dest, &RustixNoReplace, &sync, SEUIL_PETIT_FICHIER).unwrap_err();
        assert!(
            matches!(echec.erreur, PublishError::RepertoireNonExclu(_)),
            "{}",
            echec.erreur
        );
        assert_eq!(echec.staging, StagingApres::PeutEtrePublie);
        assert_eq!(echec.temoin, TemoinApres::Conserve);
        assert!(dest.join("data.bin").is_file() && chemin_temoin(&dest).exists());
    }

    #[test]
    fn garde_relachee_une_fois_par_le_drop_apres_un_echec() {
        let (_t, dest) = base();
        let rep = TestRepertoire::levee_au(6);
        {
            let h = staging_garde(&dest, &rep);
            let _ = publier(&h, &dest, &FauxRenommage::new(&dest, None)).unwrap_err();
            assert_eq!(rep.relachements(), 0, "un échec ne relâche pas");
        }
        assert_eq!(rep.relachements(), 1);
    }

    #[test]
    fn second_staging_present_au_moment_de_publish_refus_sans_ecriture() {
        let (t, dest) = base();
        let h = staging_rempli(&dest, &SourceTenue);
        fs::create_dir(dest.with_file_name(".base.po-staging-autre")).unwrap();
        let avant = lister(t.path());
        let ren = FauxRenommage::new(&dest, None);
        let echec = publier(&h, &dest, &ren).unwrap_err();
        assert!(
            matches!(echec.erreur, PublishError::TracesExistantes(_)),
            "{}",
            echec.erreur
        );
        assert_eq!(echec.temoin, TemoinApres::JamaisEcrit);
        assert_eq!(lister(t.path()), avant);
    }

    #[test]
    fn temoin_preexistant_au_moment_de_publish_refus_sans_ecriture() {
        let (t, dest) = base();
        let h = staging_rempli(&dest, &SourceTenue);
        fs::write(chemin_temoin(&dest), b"tiers").unwrap();
        let avant = lister(t.path());
        let echec = publier(&h, &dest, &FauxRenommage::new(&dest, None)).unwrap_err();
        assert!(
            matches!(echec.erreur, PublishError::TracesExistantes(_)),
            "{}",
            echec.erreur
        );
        assert_eq!(lister(t.path()), avant);
        assert_eq!(fs::read(chemin_temoin(&dest)).unwrap(), b"tiers");
    }

    #[test]
    fn temoin_altere_avant_la_barriere_refus() {
        let (_t, dest) = base();
        let h = staging_rempli(&dest, &SourceTenue);
        let p = chemin_temoin(&dest);
        // fsync call 1 is the one right after the witness.
        let sync = FauxSync {
            action: Some((
                1,
                Box::new(move || {
                    let t = fs::read_to_string(&p).unwrap().replace("nonce=", "nonce=x");
                    fs::write(&p, t).unwrap();
                }),
            )),
            ..Default::default()
        };
        let ren = FauxRenommage::new(&dest, None);
        let echec = publish_with(&h, &dest, &ren, &sync, SEUIL_PETIT_FICHIER).unwrap_err();
        assert!(
            matches!(echec.erreur, PublishError::TemoinInvalide(_)),
            "{}",
            echec.erreur
        );
        assert_eq!(ren.appels(), 0);
        assert!(!dest.exists());
        assert_staging_intact(&h);
    }

    #[test]
    fn chemin_canonique_change_refus_a_la_barriere() {
        let t = tempfile::tempdir().unwrap();
        let reel = t.path().join("reel");
        fs::create_dir(&reel).unwrap();
        let lien = t.path().join("lien");
        std::os::unix::fs::symlink(&reel, &lien).unwrap();
        let dest = lien.join("base");
        let h = staging_rempli(&dest, &SourceTenue);
        // Same parent directory (same inode), reached through another canonical path.
        let reel2 = t.path().join("reel2");
        fs::rename(&reel, &reel2).unwrap();
        fs::remove_file(&lien).unwrap();
        std::os::unix::fs::symlink(&reel2, &lien).unwrap();
        let ren = FauxRenommage::new(&dest, None);
        let echec = publier(&h, &dest, &ren).unwrap_err();
        assert!(
            matches!(echec.erreur, PublishError::CheminCanoniqueChange { .. }),
            "{}",
            echec.erreur
        );
        assert_eq!(ren.appels(), 0);
        assert!(!dest.exists());
    }

    #[test]
    fn fsync_numero_1_en_echec_rien_publie() {
        let (_t, dest) = base();
        let h = staging_rempli(&dest, &SourceTenue);
        let sync = FauxSync {
            echec_appel: Some(1),
            ..Default::default()
        };
        let ren = FauxRenommage::new(&dest, None);
        let echec = publish_with(&h, &dest, &ren, &sync, SEUIL_PETIT_FICHIER).unwrap_err();
        assert!(
            matches!(echec.erreur, PublishError::Io { .. }),
            "{}",
            echec.erreur
        );
        assert_eq!(ren.appels(), 0);
        assert!(!dest.exists());
        assert_eq!(echec.temoin, TemoinApres::Conserve);
        assert_staging_intact(&h);
    }

    #[test]
    fn fsync_numero_3_en_echec_temoin_retire_non_durable() {
        let (_t, dest) = base();
        let h = staging_rempli(&dest, &SourceTenue);
        let sync = FauxSync {
            echec_appel: Some(3),
            ..Default::default()
        };
        let echec =
            publish_with(&h, &dest, &RustixNoReplace, &sync, SEUIL_PETIT_FICHIER).unwrap_err();
        assert!(
            matches!(echec.erreur, PublishError::PublieDurableNonServable(_)),
            "{}",
            echec.erreur
        );
        assert_eq!(echec.temoin, TemoinApres::RetireSansFsync);
        assert_eq!(echec.staging, StagingApres::PeutEtrePublie);
        assert!(dest.join("data.bin").is_file());
    }

    #[test]
    fn creer_staging_examine_dest_seulement_sous_la_garde() {
        // dest is a link (refused by §1) AND the guard cannot be acquired: the guard is
        // acquired first, so the refusal is the guard's, not the precondition's.
        let (t, dest) = base();
        std::os::unix::fs::symlink(t.path(), &dest).unwrap();
        let e = creer_staging(&dest, &SourceTenue, &RepertoireNonExclu).unwrap_err();
        assert!(matches!(e, PublishError::RepertoireNonExclu(_)), "{e}");
    }

    fn refus_manifeste(prep: impl FnOnce(&Path)) -> PublishError {
        let t = tempfile::tempdir().unwrap();
        let r = t.path().join("racine");
        fs::create_dir(&r).unwrap();
        fs::write(r.join("ok"), b"1").unwrap();
        prep(&r);
        manifeste_de(&r, SEUIL_PETIT_FICHIER).unwrap_err()
    }

    #[test]
    fn manifeste_refuse_une_fifo() {
        let e = refus_manifeste(|r| {
            let c = CString::new(r.join("fifo").as_os_str().as_bytes()).unwrap();
            // SAFETY: valid NUL-terminated path, plain syscall.
            assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o600) }, 0);
        });
        assert!(
            matches!(&e, PublishError::ManifesteRefuse(m) if m.contains("fifo")),
            "{e}"
        );
    }

    #[test]
    fn manifeste_refuse_une_socket() {
        let t = tempfile::tempdir().unwrap();
        let r = t.path().join("r");
        fs::create_dir(&r).unwrap();
        let _l = std::os::unix::net::UnixListener::bind(r.join("s")).unwrap();
        let e = manifeste_de(&r, SEUIL_PETIT_FICHIER).unwrap_err();
        assert!(matches!(e, PublishError::ManifesteRefuse(_)), "{e}");
    }

    #[test]
    fn manifeste_refuse_un_lien_dur() {
        let e = refus_manifeste(|r| fs::hard_link(r.join("ok"), r.join("dur")).unwrap());
        assert!(
            matches!(&e, PublishError::ManifesteRefuse(m) if m.contains("lien dur")),
            "{e}"
        );
    }

    #[test]
    fn manifeste_refuse_un_point_de_montage_simule() {
        let t = tempfile::tempdir().unwrap();
        fs::write(t.path().join("f"), b"1").unwrap();
        let fd =
            rustix::fs::open(t.path(), OFlags::RDONLY | OFlags::DIRECTORY, Mode::empty()).unwrap();
        let dev = StatLite::depuis_rustix(&rustix::fs::fstat(&fd).unwrap()).dev;
        let e = manifeste_interne(&fd, SEUIL_PETIT_FICHIER, &|_| {}, Some(dev + 1)).unwrap_err();
        assert!(
            matches!(&e, PublishError::ManifesteRefuse(m) if m.contains("autre système de fichiers")),
            "{e}"
        );
        manifeste_interne(&fd, SEUIL_PETIT_FICHIER, &|_| {}, None).unwrap();
    }

    #[test]
    fn manifeste_enregistre_mode_uid_gid_ino_de_la_racine() {
        use std::os::unix::fs::PermissionsExt;
        let t = tempfile::tempdir().unwrap();
        let r = t.path().join("r");
        fs::create_dir(&r).unwrap();
        let m = manifeste_de(&r, SEUIL_PETIT_FICHIER).unwrap();
        let st = fs::metadata(&r).unwrap();
        assert_eq!(
            (m.racine.mode, m.racine.uid, m.racine.gid, m.racine.ino),
            (st.mode(), st.uid(), st.gid(), st.ino())
        );
        fs::set_permissions(&r, fs::Permissions::from_mode(0o700)).unwrap();
        assert!(matches!(
            comparer_a(&m, &r, SEUIL_PETIT_FICHIER).unwrap_err(),
            PublishError::ManifesteDivergent(_)
        ));
    }

    #[test]
    fn classify_adoption_refusee_si_le_proprietaire_de_dest_n_est_pas_lie() {
        let (_t, dest) = base();
        for prop in [None, Some("n9".to_string())] {
            let e = EtatDisque {
                nonce_proprietaire_dest: prop,
                ..adoptable(&dest)
            };
            assert_eq!(motif(classify(&e)), Motif::ProprietaireNonLie);
        }
    }

    #[test]
    fn nonce_proprietaire_lu_depuis_dest() {
        let (_t, dest) = base();
        let h = staging_rempli(&dest, &SourceTenue);
        assert_eq!(
            lire_nonce_proprietaire(&h.chemin).as_deref(),
            Some(h.nonce())
        );
        assert_eq!(lire_nonce_proprietaire(dest.parent().unwrap()), None);
    }

    // ---- second review of #1270 -------------------------------------------------------------

    /// fsync call 1 (right after the witness) writes a file into the staging.
    fn ecriture_dans_le_staging_apres_le_temoin(h: &StagingHandle) -> FauxSync {
        let f = h.chemin().join("ecrit-apres-le-manifeste");
        FauxSync {
            action: Some((1, Box::new(move || fs::write(&f, b"x").unwrap()))),
            ..Default::default()
        }
    }

    #[test]
    fn barriere_priorite_garde_perdue_avant_divergence_du_manifeste() {
        let (_t, dest) = base();
        let rep = TestRepertoire::levee_au(6);
        let h = staging_garde(&dest, &rep);
        let sync = ecriture_dans_le_staging_apres_le_temoin(&h);
        let ren = FauxRenommage::new(&dest, None);
        let echec = publish_with(&h, &dest, &ren, &sync, SEUIL_PETIT_FICHIER).unwrap_err();
        assert!(
            matches!(echec.erreur, PublishError::RepertoireNonExclu(_)),
            "{}",
            echec.erreur
        );
    }

    #[test]
    fn barriere_priorite_source_perdue_avant_divergence_du_manifeste() {
        let (_t, dest) = base();
        let excl = TestExclusion::rompue_au(3, Rupture::Levee);
        let h = staging_rempli(&dest, &excl);
        let sync = ecriture_dans_le_staging_apres_le_temoin(&h);
        let ren = FauxRenommage::new(&dest, None);
        let echec = publish_with(&h, &dest, &ren, &sync, SEUIL_PETIT_FICHIER).unwrap_err();
        assert!(
            matches!(echec.erreur, PublishError::ExclusionRompue(_)),
            "{}",
            echec.erreur
        );
    }

    fn arbre(profondeur: usize) -> Result<Manifeste, PublishError> {
        let t = tempfile::tempdir().unwrap();
        let r = t.path().join("r");
        let mut d = r.clone();
        for _ in 0..profondeur {
            d = d.join("d");
        }
        fs::create_dir_all(&d).unwrap();
        manifeste_de(&r, SEUIL_PETIT_FICHIER)
    }

    #[test]
    fn manifeste_profondeur_64_acceptee_65_refusee() {
        let m = arbre(PROFONDEUR_MAX).unwrap();
        assert_eq!(m.entrees.len(), PROFONDEUR_MAX);
        let e = arbre(PROFONDEUR_MAX + 1).unwrap_err();
        assert!(
            matches!(&e, PublishError::ManifesteRefuse(m) if m.contains("profondeur")),
            "{e}"
        );
    }

    #[test]
    fn temoin_forge_bien_forme_d_un_autre_nonce_refuse() {
        let (_t, dest) = base();
        let h = staging_rempli(&dest, &SourceTenue);
        let p = chemin_temoin(&dest);
        // A well-formed witness with another nonce and a recomputed checksum.
        let sync = FauxSync {
            action: Some((
                1,
                Box::new(move || {
                    let mut t = Temoin::analyser(&fs::read_to_string(&p).unwrap()).unwrap();
                    t.nonce = "forge".into();
                    fs::write(&p, t.serialiser().unwrap()).unwrap();
                }),
            )),
            ..Default::default()
        };
        let ren = FauxRenommage::new(&dest, None);
        let echec = publish_with(&h, &dest, &ren, &sync, SEUIL_PETIT_FICHIER).unwrap_err();
        assert!(
            matches!(&echec.erreur, PublishError::TemoinAutreNonce { trouve, .. } if trouve == "forge"),
            "{}",
            echec.erreur
        );
        assert_eq!(ren.appels(), 0);
        assert!(!dest.exists());
    }

    #[test]
    fn garde_non_relachee_au_drop_apres_un_rename_sans_durabilite() {
        let (_t, dest) = base();
        let rep = TestRepertoire::tenue();
        {
            let h = staging_garde(&dest, &rep);
            // fsync call 2 is the one right after the rename.
            let sync = FauxSync {
                echec_appel: Some(2),
                ..Default::default()
            };
            let echec =
                publish_with(&h, &dest, &RustixNoReplace, &sync, SEUIL_PETIT_FICHIER).unwrap_err();
            assert_eq!(echec.staging, StagingApres::PeutEtrePublie);
        }
        assert_eq!(
            rep.relachements(),
            0,
            "peut-être publié : garde tenue jusqu'à la fin"
        );
    }

    fn panique_pendant_publish(appel: usize) -> (TestRepertoire, PathBuf, tempfile::TempDir) {
        let (t, dest) = base();
        let rep = TestRepertoire::tenue();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let h = staging_garde(&dest, &rep);
            let sync = FauxSync {
                panique_appel: Some(appel),
                ..Default::default()
            };
            let _ = publish_with(&h, &dest, &RustixNoReplace, &sync, SEUIL_PETIT_FICHIER);
        }));
        assert!(r.is_err(), "la panique injectée doit remonter");
        (rep, dest, t)
    }

    #[test]
    fn garde_relachee_pendant_une_panique_avant_le_rename() {
        // fsync call 1: after the witness, before any rename: nothing can be published, the
        // guard is released even while unwinding (no leak in a long-running process).
        let (rep, dest, _t) = panique_pendant_publish(1);
        assert!(!dest.exists());
        assert_eq!(rep.relachements(), 1);
    }

    #[test]
    fn garde_non_relachee_pendant_une_panique_entre_le_rename_et_le_fsync() {
        // fsync call 2: right after the rename.
        let (rep, dest, _t) = panique_pendant_publish(2);
        assert!(dest.is_dir());
        assert_eq!(rep.relachements(), 0);
    }

    // ---- third review of #1270 --------------------------------------------------------------

    #[test]
    fn rename_puis_panique_attrapee_handle_hors_closure_garde_tenue() {
        let (_t, dest) = base();
        let rep = TestRepertoire::tenue();
        let h = staging_garde(&dest, &rep);
        let mut ren = FauxRenommage::new(&dest, None);
        ren.panique_apres = true;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = publier(&h, &dest, &ren);
        }));
        assert!(r.is_err(), "la panique du rename remonte");
        assert!(dest.is_dir(), "le rename a eu lieu");
        drop(h); // outside the unwound scope, not panicking any more
        assert_eq!(rep.relachements(), 0, "peut-être publié : garde tenue");
    }

    #[test]
    fn garde_relachee_au_drop_apres_un_echec_franc() {
        let (_t, dest) = base();
        let rep = TestRepertoire::tenue();
        {
            let h = staging_garde(&dest, &rep);
            let ren = FauxRenommage::new(
                &dest,
                Some(RenameError::Franc {
                    errno: libc::EEXIST,
                }),
            );
            let echec = publier(&h, &dest, &ren).unwrap_err();
            assert!(matches!(
                echec.erreur,
                PublishError::Renommage(RenameError::Franc { .. })
            ));
        }
        assert_eq!(
            rep.relachements(),
            1,
            "rien n'a pu être publié : garde relâchée"
        );
    }

    #[test]
    fn controle_ecriture_apres_le_temoin_seule_donne_manifeste_divergent() {
        let (_t, dest) = base();
        let h = staging_rempli(&dest, &SourceTenue);
        let sync = ecriture_dans_le_staging_apres_le_temoin(&h);
        let ren = FauxRenommage::new(&dest, None);
        let echec = publish_with(&h, &dest, &ren, &sync, SEUIL_PETIT_FICHIER).unwrap_err();
        assert!(
            matches!(echec.erreur, PublishError::ManifesteDivergent(_)),
            "{}",
            echec.erreur
        );
        assert_eq!(ren.appels(), 0);
        assert!(!dest.exists());
    }

    // ---- fourth review of #1270 -------------------------------------------------------------

    fn rep_avec(f: impl FnOnce(&mut EtatRep)) -> TestRepertoire {
        let r = TestRepertoire::tenue();
        f(&mut r.0.lock().unwrap());
        r
    }

    #[test]
    fn rename_menteur_renomme_puis_renvoie_franc_garde_tenue() {
        let (_t, dest) = base();
        let rep = TestRepertoire::tenue();
        {
            let h = staging_garde(&dest, &rep);
            let mut ren = FauxRenommage::new(
                &dest,
                Some(RenameError::Franc {
                    errno: libc::EEXIST,
                }),
            );
            ren.renommer_puis_mentir = true;
            let echec = publier(&h, &dest, &ren).unwrap_err();
            assert!(dest.is_dir(), "le rename a eu lieu malgré EEXIST");
            assert_eq!(
                echec.staging,
                StagingApres::PeutEtrePublie,
                "aucun eligible_section6"
            );
            assert!(
                matches!(echec.erreur, PublishError::FrancNonConfirme { errno } if errno == libc::EEXIST),
                "{}",
                echec.erreur
            );
            assert!(echec.erreur.to_string().contains("non confirmé"));
        }
        assert_eq!(rep.relachements(), 0, "peut-être publié : garde tenue");
    }

    #[test]
    fn garde_tenue_apres_ambigu_relachee_apres_non_supporte() {
        for (errno, attendu) in [(libc::EIO, 0), (libc::EINVAL, 1)] {
            let (_t, dest) = base();
            let rep = TestRepertoire::tenue();
            {
                let h = staging_garde(&dest, &rep);
                let ren = FauxRenommage::new(&dest, Some(RenameError::depuis_errno(errno)));
                let _ = publier(&h, &dest, &ren).unwrap_err();
            }
            assert_eq!(rep.relachements(), attendu, "errno {}", nom_errno(errno));
        }
    }

    #[test]
    fn relacher_qui_panique_au_drop_est_avale() {
        let (_t, dest) = base();
        let rep = rep_avec(|e| {
            e.levee_au = Some(6);
            e.panique_au_relacher = true;
        });
        {
            let h = staging_garde(&dest, &rep);
            let _ = publier(&h, &dest, &FauxRenommage::new(&dest, None)).unwrap_err();
        } // Drop: relacher panics, the panic is swallowed (no abort, no propagation)
        assert_eq!(rep.relachements(), 1);
    }

    #[test]
    fn panique_entre_acquerir_et_le_handle_garde_relachee() {
        let (_t, dest) = base();
        // Verification 1 is the one inside creer_staging, before the handle exists.
        let rep = rep_avec(|e| e.panique_a_la_verification = Some(1));
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = creer_staging(&dest, &SourceTenue, &rep);
        }));
        assert!(r.is_err());
        assert_eq!(
            rep.relachements(),
            1,
            "la garde en attente est relâchée au déroulement"
        );
        assert!(
            scanner_traces(&dest).unwrap().stagings.is_empty(),
            "rien créé"
        );
    }

    #[test]
    fn erreur_entre_acquerir_et_le_handle_garde_relachee() {
        // dest is a link: refused under the guard; the pending guard is released.
        let (t, dest) = base();
        std::os::unix::fs::symlink(t.path(), &dest).unwrap();
        let rep = TestRepertoire::tenue();
        let e = creer_staging(&dest, &SourceTenue, &rep).unwrap_err();
        assert!(matches!(e, PublishError::DestLienSymbolique(_)), "{e}");
        assert_eq!(rep.relachements(), 1);
    }

    #[test]
    fn relacher_qui_panique_pour_une_garde_en_attente_est_avale() {
        // dest is a link: refused under the pending guard, whose relacher panics.
        let (t, dest) = base();
        std::os::unix::fs::symlink(t.path(), &dest).unwrap();
        let rep = rep_avec(|e| e.panique_au_relacher = true);
        let e = creer_staging(&dest, &SourceTenue, &rep).unwrap_err();
        assert!(matches!(e, PublishError::DestLienSymbolique(_)), "{e}");
        assert_eq!(rep.relachements(), 1);
    }

    #[test]
    fn relacher_qui_panique_pendant_un_deroulement_pas_d_abort() {
        // Covers "a relacher that panics during unwinding does not abort". It does NOT cover
        // the hand-over order of the guard (`garde: attente.remettre()` as the last field):
        // the panic here happens before remettre(); that ordering is defensive and untested.
        // Verification 1 panics inside creer_staging; while unwinding, the pending guard is
        // released and relacher panics too: swallowed, no double panic, no abort.
        let (_t, dest) = base();
        let rep = rep_avec(|e| {
            e.panique_a_la_verification = Some(1);
            e.panique_au_relacher = true;
        });
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = creer_staging(&dest, &SourceTenue, &rep);
        }));
        assert!(r.is_err());
        assert_eq!(rep.relachements(), 1);
    }

    // ---- sixth review: the clean-failure decision, directly -----------------------------------

    #[test]
    fn decision_echec_franc_sur_les_resultats_de_statat() {
        let id = (1, 7); // stat_rep(1) has dev 1, ino 7, a directory
        let staging_ok: Result<StatLite, rustix::io::Errno> = Ok(stat_rep(1));
        let autre = |ino| {
            let mut s = stat_rep(1);
            s.ino = ino;
            s
        };
        let enoent = Err(rustix::io::Errno::NOENT);
        // dest absent: confirmed.
        assert!(echec_franc_confirme(&staging_ok, &enoent, id));
        // dest of another inode (a real EEXIST): confirmed.
        assert!(echec_franc_confirme(&staging_ok, &Ok(autre(8)), id));
        // dest carrying the staging's inode: ambiguous.
        assert!(!echec_franc_confirme(&staging_ok, &Ok(stat_rep(1)), id));
        // dest statat failing otherwise than ENOENT: ambiguous.
        assert!(!echec_franc_confirme(
            &staging_ok,
            &Err(rustix::io::Errno::IO),
            id
        ));
        // staging of another inode: ambiguous.
        assert!(!echec_franc_confirme(&Ok(autre(9)), &enoent, id));
        // staging not a directory: ambiguous.
        let mut fichier = stat_rep(1);
        fichier.genre = Genre::Fichier;
        assert!(!echec_franc_confirme(&Ok(fichier), &enoent, id));
        // staging statat failing: ambiguous.
        assert!(!echec_franc_confirme(&enoent, &enoent, id));
        // The dev component counts on both sides.
        let autre_dev = {
            let mut s = stat_rep(2);
            s.ino = 7;
            s
        };
        // staging of the same ino on ANOTHER device: not ours, ambiguous.
        assert!(!echec_franc_confirme(&Ok(autre_dev), &enoent, id));
        // dest of the same ino on another device: another object, a real EEXIST.
        assert!(echec_franc_confirme(&staging_ok, &Ok(autre_dev), id));
    }
}
