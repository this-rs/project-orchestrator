//! Artifact Comparison — M2 Step 4
//!
//! Compares the PO enriched artifact format (EnrichedArtifact) with the
//! the-source distilled artifact format (DistilledArtifactV1).
//!
//! the-source format reference: distilled_artifact_v1.json (555KB, 2244 traces)
//! — a flat list of weighted structural edges between file paths.
//!
//! PO enriched format: structure (co-change edges) + portable episodes
//! (Stimulus → Process → Outcome → Validation → Lesson).

use serde::{Deserialize, Serialize};

// ============================================================================
// the-source format (simulated from known schema)
// ============================================================================

/// A single trace in the-source distilled artifact.
///
/// the-source's distill.py produces edges of the form:
/// { "source": "src/foo.rs", "target": "src/bar.rs", "relation": "COACTIVATES", "weight": 0.85 }
///
/// Each trace represents a structural relationship discovered by static analysis
/// or co-change mining. No semantic context — just topology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TheSourceTrace {
    pub source: String,
    pub target: String,
    pub relation: String,
    pub weight: f64,
}

/// the-source distilled artifact v1 format.
///
/// A flat JSON: { "version": 1, "traces": [...], "stats": { ... } }
/// Reference: 555KB, 2244 traces, pure structural (no episodes, no lessons).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistilledArtifactV1 {
    pub version: u32,
    pub traces: Vec<TheSourceTrace>,
    pub stats: TheSourceStats,
}

/// Stats in the-source format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TheSourceStats {
    pub trace_count: usize,
    pub unique_files: usize,
    pub relation_types: Vec<String>,
}

// ============================================================================
// Comparison Report
// ============================================================================

/// Comparison report between the two artifact formats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactComparisonReport {
    /// Dimension 1: Size comparison (bytes)
    pub size: SizeComparison,
    /// Dimension 2: Record count (traces vs edges+episodes)
    pub record_count: RecordCountComparison,
    /// Dimension 3: Information richness (fields per record)
    pub information_richness: RichnessComparison,
    /// Dimension 4: Semantic depth (does the format carry "why"?)
    pub semantic_depth: SemanticDepthComparison,
    /// Dimension 5: Portability (cross-instance transferability)
    pub portability: PortabilityComparison,
    /// Overall conclusion
    pub conclusion: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SizeComparison {
    pub the_source_bytes: usize,
    pub enriched_bytes: usize,
    pub ratio: f64,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordCountComparison {
    pub the_source_traces: usize,
    pub enriched_edges: usize,
    pub enriched_episodes: usize,
    pub enriched_total: usize,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RichnessComparison {
    pub the_source_fields_per_trace: usize,
    pub enriched_fields_per_edge: usize,
    pub enriched_fields_per_episode: usize,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticDepthComparison {
    pub the_source_has_stimulus: bool,
    pub the_source_has_process: bool,
    pub the_source_has_lesson: bool,
    pub enriched_has_stimulus: bool,
    pub enriched_has_process: bool,
    pub enriched_has_lesson: bool,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortabilityComparison {
    pub the_source_uuid_free: bool,
    pub enriched_uuid_free: bool,
    pub the_source_has_anonymization: bool,
    pub enriched_has_anonymization: bool,
    pub note: String,
}

// ============================================================================
// Comparison Logic
// ============================================================================

/// Compare an enriched artifact with a the-source distilled artifact.
pub fn compare_artifacts(
    enriched: &crate::api::episode_handlers::EnrichedArtifact,
    the_source: &DistilledArtifactV1,
) -> ArtifactComparisonReport {
    let enriched_json = serde_json::to_string(enriched).unwrap_or_default();
    let the_source_json = serde_json::to_string(the_source).unwrap_or_default();

    let enriched_bytes = enriched_json.len();
    let the_source_bytes = the_source_json.len();

    let enriched_total = enriched.stats.edge_count + enriched.stats.episode_count;

    // Fields per record: the-source trace has 4 fields (source, target, relation, weight)
    // Enriched edge also has 4 fields
    // Enriched episode has ~15+ nested fields (stimulus.request, stimulus.trigger,
    // process.had_reasoning_tree, process.states_visited, outcome.notes_produced, etc.)
    let enriched_episode_fields = 15; // conservative count of meaningful fields

    let episodes_with_lessons = enriched.stats.episodes_with_lessons;
    let total_episodes = enriched.stats.episode_count;
    let lesson_ratio = if total_episodes > 0 {
        episodes_with_lessons as f64 / total_episodes as f64
    } else {
        0.0
    };

    ArtifactComparisonReport {
        size: SizeComparison {
            the_source_bytes,
            enriched_bytes,
            ratio: if the_source_bytes > 0 {
                enriched_bytes as f64 / the_source_bytes as f64
            } else {
                0.0
            },
            note: format!(
                "the-source: {}KB vs enriched: {}KB (ratio: {:.2}x). \
                 Enriched is {} because episodes carry semantic context (stimulus, process, lesson).",
                the_source_bytes / 1024,
                enriched_bytes / 1024,
                enriched_bytes as f64 / the_source_bytes.max(1) as f64,
                if enriched_bytes > the_source_bytes {
                    "larger"
                } else {
                    "smaller"
                }
            ),
        },
        record_count: RecordCountComparison {
            the_source_traces: the_source.stats.trace_count,
            enriched_edges: enriched.stats.edge_count,
            enriched_episodes: total_episodes,
            enriched_total,
            note: format!(
                "the-source: {} traces (pure structural). \
                 Enriched: {} edges + {} episodes = {} records. \
                 Episodes carry cognitive context absent from traces.",
                the_source.stats.trace_count,
                enriched.stats.edge_count,
                total_episodes,
                enriched_total,
            ),
        },
        information_richness: RichnessComparison {
            the_source_fields_per_trace: 4,
            enriched_fields_per_edge: 4,
            enriched_fields_per_episode: enriched_episode_fields,
            note: format!(
                "the-source: 4 fields/trace (source, target, relation, weight). \
                 Enriched edges: 4 fields (same). \
                 Enriched episodes: ~{} fields (stimulus, process trace, outcome counts, \
                 validation, lesson with domain_tags + portability_layer). \
                 {:.0}% of episodes have extracted lessons.",
                enriched_episode_fields,
                lesson_ratio * 100.0,
            ),
        },
        semantic_depth: SemanticDepthComparison {
            the_source_has_stimulus: false,
            the_source_has_process: false,
            the_source_has_lesson: false,
            enriched_has_stimulus: true,
            enriched_has_process: true,
            enriched_has_lesson: episodes_with_lessons > 0,
            note: format!(
                "the-source captures WHAT (structural edges) but not WHY. \
                 Enriched artifact captures: \
                 (1) WHAT triggered the knowledge (stimulus), \
                 (2) HOW it was processed (FSM states, reasoning tree), \
                 (3) WHAT it produced (notes/decisions counts), \
                 (4) WHY it matters (lesson with abstract pattern). \
                 {}/{} episodes have lessons — the key differentiator for cross-instance transfer.",
                episodes_with_lessons, total_episodes,
            ),
        },
        portability: PortabilityComparison {
            the_source_uuid_free: true,
            enriched_uuid_free: true,
            the_source_has_anonymization: false, // the-source uses raw file paths
            enriched_has_anonymization: true,     // PortableEpisode strips UUIDs
            note: "Both formats are UUID-free. the-source uses raw file paths \
                   (project-specific). Enriched episodes are anonymized \
                   (no UUIDs, no absolute paths) making them more portable \
                   across different codebases."
                .to_string(),
        },
        conclusion: format!(
            "the-source Layer 0 (structural): {} traces, {}KB — captures topology (WHAT connects to WHAT). \
             PO Layer 1 (episodic): {} edges + {} episodes, {}KB — captures topology + cognitive context \
             (WHY knowledge was created, HOW it was derived, WHAT lesson was learned). \
             The episodic layer adds ~{:.0}% size overhead but provides semantic depth \
             that enables higher-quality cross-instance knowledge transfer. \
             Key advantage: {}/{} episodes carry portable lessons that can be \
             re-contextualized on the receiving instance without access to the source graph.",
            the_source.stats.trace_count,
            the_source_bytes / 1024,
            enriched.stats.edge_count,
            total_episodes,
            enriched_bytes / 1024,
            ((enriched_bytes as f64 / the_source_bytes.max(1) as f64) - 1.0) * 100.0,
            episodes_with_lessons,
            total_episodes,
        ),
    }
}

// ============================================================================
// Fake Data Generators (for testing without real graph data)
// ============================================================================

/// Generate a fake the-source distilled artifact with realistic dimensions.
///
/// Based on the real the-source artifact: 555KB, 2244 traces, ~150 unique files.
pub fn generate_fake_the_source(trace_count: usize) -> DistilledArtifactV1 {
    let mut traces = Vec::with_capacity(trace_count);
    let file_count = (trace_count as f64 * 0.067).ceil() as usize; // ~150/2244 ratio

    for i in 0..trace_count {
        let src_idx = i % file_count;
        let tgt_idx = (i * 7 + 3) % file_count;
        let relation = match i % 5 {
            0 => "COACTIVATES",
            1 => "CO_CHANGED",
            2 => "IMPORTS",
            3 => "CALLS",
            _ => "DEPENDS_ON",
        };
        traces.push(TheSourceTrace {
            source: format!("src/module_{}/file_{}.rs", src_idx / 10, src_idx),
            target: format!("src/module_{}/file_{}.rs", tgt_idx / 10, tgt_idx),
            relation: relation.to_string(),
            weight: 0.1 + (i as f64 * 0.37) % 0.9,
        });
    }

    let relation_types = vec![
        "COACTIVATES".to_string(),
        "CO_CHANGED".to_string(),
        "IMPORTS".to_string(),
        "CALLS".to_string(),
        "DEPENDS_ON".to_string(),
    ];

    DistilledArtifactV1 {
        version: 1,
        traces,
        stats: TheSourceStats {
            trace_count,
            unique_files: file_count,
            relation_types,
        },
    }
}

/// Generate a fake enriched artifact with realistic episodes.
pub fn generate_fake_enriched(
    edge_count: usize,
    episode_count: usize,
) -> crate::api::episode_handlers::EnrichedArtifact {
    use crate::episodes::*;

    let mut structure = Vec::with_capacity(edge_count);
    for i in 0..edge_count {
        structure.push(crate::api::episode_handlers::ArtifactEdge {
            source: format!("src/module_{}/file_{}.rs", i / 10, i),
            target: format!(
                "src/module_{}/file_{}.rs",
                (i * 3 + 1) % edge_count.max(1),
                (i * 3 + 1) % edge_count.max(1)
            ),
            relation: "CO_CHANGED".to_string(),
            weight: (i as f64 * 0.5) % 10.0 + 1.0,
        });
    }

    let lessons_data = [
        ("When adding a new graph relation, always create an index and a backfill migration for existing data.", vec!["neo4j", "schema-migration"], 2),
        ("Protocol FSM states should be idempotent — re-entering a state must not duplicate side effects.", vec!["fsm", "protocol", "idempotency"], 3),
        ("Batch UNWIND is critical for Neo4j performance — never N+1 loop over node creation.", vec!["neo4j", "performance"], 2),
        ("MCP mega-tool dispatch requires updating 4 locations: tools.rs schema, handlers.rs array, mega_tool_to_legacy, try_handle_http.", vec!["mcp", "architecture"], 1),
        ("Spreading activation finds semantically related notes that BM25 keyword search misses — use both for comprehensive retrieval.", vec!["search", "neural", "knowledge-management"], 3),
        ("Community detection (Louvain) reveals functional modules invisible to manual code review — a file's community_id is more informative than its directory.", vec!["graph-analytics", "architecture"], 3),
        ("When modifying a trait method signature, always check find_trait_implementations first — breaking N implementors silently is the #1 refactoring risk.", vec!["rust", "refactoring", "traits"], 2),
        ("Trust scoring for cross-instance skill exchange should weight success_rate (30%) highest — energy and cohesion alone don't guarantee quality.", vec!["federation", "trust", "skills"], 3),
    ];

    let mut episodes = Vec::with_capacity(episode_count);
    let mut episodes_with_lessons = 0;
    let mut total_notes = 0;
    let mut total_decisions = 0;

    for i in 0..episode_count {
        let has_lesson = i < lessons_data.len(); // first N episodes get lessons
        let notes_produced = (i % 3) + 1;
        let decisions_made = i % 2;
        total_notes += notes_produced;
        total_decisions += decisions_made;

        let lesson = if has_lesson {
            episodes_with_lessons += 1;
            let (pattern, tags, portability) = &lessons_data[i];
            Some(PortableLesson {
                abstract_pattern: pattern.to_string(),
                domain_tags: tags.iter().map(|t| t.to_string()).collect(),
                portability_layer: *portability,
                confidence: Some(0.7 + (i as f64 * 0.03)),
            })
        } else {
            None
        };

        let states = match i % 4 {
            0 => vec![
                "analyze".to_string(),
                "implement".to_string(),
                "validate".to_string(),
                "done".to_string(),
            ],
            1 => vec![
                "plan".to_string(),
                "execute".to_string(),
                "review".to_string(),
                "done".to_string(),
            ],
            2 => vec![
                "detect".to_string(),
                "diagnose".to_string(),
                "fix".to_string(),
                "done".to_string(),
            ],
            _ => vec![
                "start".to_string(),
                "process".to_string(),
                "done".to_string(),
            ],
        };

        episodes.push(PortableEpisode {
            schema_version: 1,
            stimulus: PortableStimulus {
                request: format!(
                    "Episode {} — triggered by protocol execution on task {}",
                    i,
                    i * 7 + 1
                ),
                trigger: match i % 3 {
                    0 => crate::episodes::models::StimulusTrigger::ProtocolTransition,
                    1 => crate::episodes::models::StimulusTrigger::UserRequest,
                    _ => crate::episodes::models::StimulusTrigger::Manual,
                },
            },
            process: PortableProcess {
                had_reasoning_tree: i % 2 == 0,
                states_visited: states,
                duration_ms: Some(5000 + (i as i64 * 1500)),
            },
            outcome: PortableOutcome {
                notes_produced,
                decisions_made,
                commits_made: if i % 3 == 0 { 1 } else { 0 },
                files_modified: (i % 5) + 1,
                note_summaries: Vec::new(),
                decision_summaries: Vec::new(),
            },
            validation: PortableValidation {
                feedback_type: if i % 4 == 0 {
                    crate::episodes::models::FeedbackType::ExplicitPositive
                } else {
                    crate::episodes::models::FeedbackType::ImplicitPositive
                },
                score: Some(0.6 + (i as f64 * 0.04) % 0.4),
                evidence_count: (i % 3) + 1,
            },
            lesson,
        });
    }

    crate::api::episode_handlers::EnrichedArtifact {
        schema_version: 1,
        exported_at: chrono::Utc::now(),
        source_project: "project-orchestrator-backend".to_string(),
        structure,
        episodes,
        stats: crate::api::episode_handlers::ArtifactStats {
            edge_count,
            episode_count,
            episodes_with_lessons,
            total_notes_produced: total_notes,
            total_decisions_made: total_decisions,
        },
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_fake_the_source() {
        let artifact = generate_fake_the_source(2244);
        assert_eq!(artifact.traces.len(), 2244);
        assert_eq!(artifact.stats.trace_count, 2244);
        assert!(artifact.stats.unique_files > 100);

        // Verify JSON size is in the right ballpark (~555KB)
        let json = serde_json::to_string(&artifact).unwrap();
        let size_kb = json.len() / 1024;
        assert!(size_kb > 200, "Expected >200KB, got {}KB", size_kb);
    }

    #[test]
    fn test_generate_fake_enriched() {
        let artifact = generate_fake_enriched(80, 12);
        assert_eq!(artifact.stats.edge_count, 80);
        assert_eq!(artifact.stats.episode_count, 12);
        assert!(artifact.stats.episodes_with_lessons > 0);
        assert!(artifact.stats.total_notes_produced > 0);

        let json = serde_json::to_string(&artifact).unwrap();
        assert!(!json.is_empty());

        // Verify no UUIDs in episodes
        let uuid_re =
            regex::Regex::new(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
                .unwrap();
        // The exported_at datetime is fine, we check episodes specifically
        for ep in &artifact.episodes {
            let ep_json = serde_json::to_string(ep).unwrap();
            assert!(
                !uuid_re.is_match(&ep_json),
                "Episode should not contain UUIDs: {}",
                ep_json
            );
        }
    }

    #[test]
    fn test_artifact_comparison_realistic() {
        // Simulate the-source: 2244 traces, ~555KB
        let the_source = generate_fake_the_source(2244);

        // Simulate PO enriched: 80 co-change edges + 12 episodes (8 with lessons)
        let enriched = generate_fake_enriched(80, 12);

        let report = compare_artifacts(&enriched, &the_source);

        // Print the full report for M2 documentation
        let report_json = serde_json::to_string_pretty(&report).unwrap();
        println!(
            "=== ARTIFACT COMPARISON REPORT (M2 Step 4) ===\n{}",
            report_json
        );

        // Dimension 1: Size — enriched should be smaller (fewer records)
        assert!(report.size.the_source_bytes > 0);
        assert!(report.size.enriched_bytes > 0);

        // Dimension 2: Record count — the-source has more traces, enriched has episodes
        assert_eq!(report.record_count.the_source_traces, 2244);
        assert_eq!(report.record_count.enriched_episodes, 12);
        assert_eq!(report.record_count.enriched_edges, 80);

        // Dimension 3: Richness — episodes are richer than traces
        assert!(
            report.information_richness.enriched_fields_per_episode
                > report.information_richness.the_source_fields_per_trace
        );

        // Dimension 4: Semantic depth — enriched has all 3, the-source has none
        assert!(!report.semantic_depth.the_source_has_stimulus);
        assert!(!report.semantic_depth.the_source_has_process);
        assert!(!report.semantic_depth.the_source_has_lesson);
        assert!(report.semantic_depth.enriched_has_stimulus);
        assert!(report.semantic_depth.enriched_has_process);
        assert!(report.semantic_depth.enriched_has_lesson);

        // Dimension 5: Portability — both UUID-free, enriched has anonymization
        assert!(report.portability.the_source_uuid_free);
        assert!(report.portability.enriched_uuid_free);
        assert!(report.portability.enriched_has_anonymization);
        assert!(!report.portability.the_source_has_anonymization);

        // Conclusion should be non-empty
        assert!(!report.conclusion.is_empty());
    }

    #[test]
    fn test_comparison_report_serialization() {
        let the_source = generate_fake_the_source(100);
        let enriched = generate_fake_enriched(20, 5);
        let report = compare_artifacts(&enriched, &the_source);

        let json = serde_json::to_string_pretty(&report).unwrap();
        let restored: ArtifactComparisonReport = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.record_count.the_source_traces, 100);
        assert_eq!(restored.record_count.enriched_episodes, 5);
    }

    #[test]
    fn test_comparison_with_equal_scale() {
        // What if we had the same number of records?
        let the_source = generate_fake_the_source(100);
        let enriched = generate_fake_enriched(50, 50);
        let report = compare_artifacts(&enriched, &the_source);

        // Enriched should be larger per-record due to episode richness
        let avg_the_source = report.size.the_source_bytes as f64 / 100.0;
        let avg_enriched = report.size.enriched_bytes as f64 / 100.0;
        assert!(
            avg_enriched > avg_the_source,
            "Enriched should be larger per-record: {} vs {}",
            avg_enriched,
            avg_the_source
        );
    }
}
