//! Garde-fou du contrat « tri total » (voir `docs/cartographie/16-tris.md`).
//!
//! Un `ORDER BY` partiel (timestamp/score/poids/priorité sans identifiant en
//! fin de tri) laisse des égalités que le moteur tranche arbitrairement : Neo4j
//! et obrain peuvent sortir les mêmes lignes dans un ordre différent, sans
//! erreur. La règle : à égalité, l'ordre est celui de l'identifiant unique du
//! nœud, dans le même sens que la dernière colonne.
//!
//! Ce test tourne **sans** Neo4j : il relit les sources et vérifie que les tris
//! partiels connus portent désormais leur clé de départage totale. Une preuve de
//! stabilité sur données réelles (insertion d'égalités volontaires) exigerait le
//! binaire Neo4j ; elle est décrite dans le document de cartographie.

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    /// (fichier, fragment attendu) — le fragment doit contenir la clé de
    /// départage totale ajoutée à la fin du tri partiel.
    const CASES: &[(&str, &str)] = &[
        // L'exemple nommé dans la tâche : priorité puis horodatage.
        (
            "alert.rs",
            "ORDER BY coalesce(a.priority, 0.0) DESC, coalesce(a.last_seen, a.created_at) DESC, a.id DESC",
        ),
        ("task.rs", "ORDER BY COALESCE(t.priority, 0) DESC, t.created_at, t.id ASC"),
        ("plan.rs", "ORDER BY p.priority DESC, p.created_at DESC, p.id DESC"),
        ("decision.rs", "ORDER BY d.decided_at DESC, d.id DESC"),
        ("note.rs", "ORDER BY n.importance DESC, n.created_at DESC, n.id DESC"),
        ("skill.rs", "ORDER BY s.energy DESC, s.name ASC, s.id ASC"),
        // Nœuds sans `id` : File -> path, Commit -> hash.
        ("commit.rs", "ORDER BY c.timestamp DESC, c.hash DESC"),
        ("code.rs", "ORDER BY func.line_start, func.id ASC"),
        ("code.rs", "ORDER BY COALESCE(f.pagerank, 0) DESC, connections DESC, f.path ASC"),
        // Tri dynamique des endpoints de liste : id ajouté en départage.
        ("milestone.rs", "ORDER BY {} {}, m.id"),
        ("project.rs", "ORDER BY {} {}, p.id"),
        // Relations/événements journalisés : id ajouté en départage.
        ("ingress.rs", "ORDER BY e.received_at DESC, e.id DESC"),
        ("trigger.rs", "ORDER BY f.fired_at DESC, f.id DESC"),
        // Score de recherche sémantique (flottant) : id du nœud retourné.
        ("decision.rs", "ORDER BY score DESC, d.id DESC"),
        ("document.rs", "ORDER BY score DESC, c.id DESC"),
        ("note.rs", "ORDER BY score DESC, n.id DESC"),
        ("note.rs", "ORDER BY score DESC, f.path DESC"),
        ("note.rs", "ORDER BY score DESC, fn.id DESC"),
        ("trajectory_store.rs", "ORDER BY score DESC, t.id DESC"),
        // Poids SYNAPSE : extrémités de l'arête.
        ("note.rs", "ORDER BY weight DESC, neighbor_id DESC"),
        ("note.rs", "ORDER BY s.weight DESC, source_id DESC, target_id DESC"),
        // Agrégats analytics : identifiant du groupe.
        ("analytics.rs", "ORDER BY SIZE(members) DESC, wl_hash DESC"),
        ("analytics.rs", "ORDER BY file_count DESC, cid DESC"),
        ("analytics.rs", "ORDER BY f.churn_score DESC, f.path DESC"),
        ("data_migrations.rs", "ORDER BY notes DESC, status, note_type, importance"),
        // Entités hétérogènes : coalesce(id, path) en départage.
        ("chat.rs", "ORDER BY r.mention_count DESC, COALESCE(e.id, e.path) DESC"),
        // Chemin critique : liste d'identifiants du chemin.
        ("plan.rs", "ORDER BY pathLength DESC, path_ids DESC"),
        ("graph_schema.rs", "ORDER BY depth, n.id ASC LIMIT 200"),
    ];

    #[test]
    fn partial_sorts_end_with_a_total_tiebreaker() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/neo4j");
        for (file, needle) in CASES {
            let src =
                fs::read_to_string(root.join(file)).unwrap_or_else(|e| panic!("read {file}: {e}"));
            assert!(
                src.contains(needle),
                "{file} n'a plus la clé de départage attendue :\n  {needle}"
            );
        }
    }

    /// Aucun des fragments ne doit avoir été réintroduit sans départage :
    /// la forme partielle (sans l'id en fin) est rejetée.
    #[test]
    fn no_partial_form_remains_for_fixed_sorts() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/neo4j");
        let alert = fs::read_to_string(root.join("alert.rs")).unwrap();
        // La forme partielle exacte de l'exemple ne doit plus exister.
        assert!(!alert.contains(
            "ORDER BY coalesce(a.priority, 0.0) DESC, coalesce(a.last_seen, a.created_at) DESC\n"
        ));
    }
}
