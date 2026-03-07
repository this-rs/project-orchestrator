//! VizBlock builder — tag parsing and data-fetching helpers.
//!
//! Parses structured `<viz>` tags in LLM text output and replaces them with
//! real [`VizBlock`]s populated from the knowledge graph.
//!
//! ## Tag format
//!
//! ```text
//! <viz type="impact_graph" target="src/chat/manager.rs" />
//! <viz type="reasoning_tree" />
//! <viz type="progress_bar" plan_id="57746b78-..." />
//! <viz type="knowledge_card" note_id="d7376565-..." />
//! ```
//!
//! ## Architecture
//!
//! [`VizBlockProcessor`] holds a reference to the [`GraphStore`] and processes
//! LLM text in a single pass:
//! 1. Regex finds all `<viz ... />` tags
//! 2. For each tag, extract `type` and key-value attributes
//! 3. Call the appropriate helper to fetch real data from the graph
//! 4. Build a [`VizBlock`] with real data and auto-generated fallback text
//! 5. Return `Vec<ContentBlock>` with alternating Text and Viz blocks

use crate::neo4j::traits::GraphStore;
use anyhow::Result;
use regex::Regex;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};
use uuid::Uuid;

use super::viz::{
    ContentBlock, KnowledgeCardVizBuilder, ProgressBarVizBuilder, ReasoningTreeVizBuilder,
    TaskProgress, VizBlock, VizDataBuilder, VizType,
};
use crate::reasoning::models::ReasoningTree;

// ============================================================================
// Tag parsing
// ============================================================================

/// Regex for matching `<viz type="..." [key="value"]* />` tags.
///
/// Captures the entire tag including all attributes.
/// Supports both `<viz ... />` (self-closing) and `<viz ...>...</viz>` forms.
static VIZ_TAG_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"<viz\s+([^>]*?)\s*/>"#).expect("invalid viz tag regex")
});

/// Regex for extracting key="value" attribute pairs from a viz tag.
static ATTR_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(\w+)\s*=\s*"([^"]*)""#).expect("invalid attr regex")
});

/// A parsed viz tag with its type and attributes.
#[derive(Debug, Clone)]
pub struct ParsedVizTag {
    /// The viz type string (e.g., "impact_graph", "reasoning_tree").
    pub viz_type: String,
    /// Key-value attributes from the tag.
    pub attrs: HashMap<String, String>,
    /// The full matched tag string (for replacement).
    pub full_match: String,
    /// Byte offset of the tag start in the original text.
    pub start: usize,
    /// Byte offset of the tag end in the original text.
    pub end: usize,
}

/// Parse all `<viz ... />` tags from a text string.
pub fn parse_viz_tags(text: &str) -> Vec<ParsedVizTag> {
    VIZ_TAG_RE
        .captures_iter(text)
        .filter_map(|cap| {
            let full = cap.get(0)?;
            let attrs_str = cap.get(1)?.as_str();

            // Extract attributes
            let mut attrs = HashMap::new();
            for attr_cap in ATTR_RE.captures_iter(attrs_str) {
                let key = attr_cap.get(1)?.as_str().to_string();
                let value = attr_cap.get(2)?.as_str().to_string();
                attrs.insert(key, value);
            }

            let viz_type = attrs.get("type").cloned()?;

            Some(ParsedVizTag {
                viz_type,
                attrs,
                full_match: full.as_str().to_string(),
                start: full.start(),
                end: full.end(),
            })
        })
        .collect()
}

// ============================================================================
// Helper functions — build VizBlocks from graph data
// ============================================================================

/// Build an impact graph VizBlock from a file target.
///
/// Queries the graph for files impacted by changes to `target`,
/// then builds a VizBlock with nodes and edges for frontend rendering.
pub async fn build_impact_viz(
    graph: &dyn GraphStore,
    target: &str,
    project_id: Option<Uuid>,
) -> Result<VizBlock> {
    let impacted_files = graph.find_impacted_files(target, 3, project_id).await?;
    let dependent_files = graph.find_dependent_files(target, 2, project_id).await?;

    // Build node list (target + impacted + dependents)
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    // Target node
    nodes.push(serde_json::json!({
        "id": target,
        "label": target.rsplit('/').next().unwrap_or(target),
        "path": target,
        "role": "target",
    }));

    // Impacted files (downstream)
    for file in &impacted_files {
        if file != target {
            nodes.push(serde_json::json!({
                "id": file,
                "label": file.rsplit('/').next().unwrap_or(file),
                "path": file,
                "role": "impacted",
            }));
            edges.push(serde_json::json!({
                "source": target,
                "target": file,
                "relation": "impacts",
            }));
        }
    }

    // Dependent files (upstream)
    for file in &dependent_files {
        if file != target && !impacted_files.contains(file) {
            nodes.push(serde_json::json!({
                "id": file,
                "label": file.rsplit('/').next().unwrap_or(file),
                "path": file,
                "role": "dependent",
            }));
            edges.push(serde_json::json!({
                "source": file,
                "target": target,
                "relation": "depends_on",
            }));
        }
    }

    // Build fallback text
    let mut fallback_lines = vec![format!("Impact analysis for: {target}")];
    if !impacted_files.is_empty() {
        fallback_lines.push(format!(
            "Impacted files ({}):",
            impacted_files.len()
        ));
        for file in &impacted_files {
            if file != target {
                fallback_lines.push(format!("  → {file}"));
            }
        }
    }
    if !dependent_files.is_empty() {
        fallback_lines.push(format!(
            "Dependent files ({}):",
            dependent_files.len()
        ));
        for file in &dependent_files {
            if file != target && !impacted_files.contains(file) {
                fallback_lines.push(format!("  ← {file}"));
            }
        }
    }
    if impacted_files.is_empty() && dependent_files.is_empty() {
        fallback_lines.push("No impact detected.".to_string());
    }

    Ok(VizBlock::new(
        VizType::ImpactGraph,
        serde_json::json!({
            "target": target,
            "nodes": nodes,
            "edges": edges,
            "impacted_count": impacted_files.len(),
            "dependent_count": dependent_files.len(),
        }),
        fallback_lines.join("\n"),
    )
    .with_interactive(true)
    .with_title(format!(
        "Impact: {}",
        target.rsplit('/').next().unwrap_or(target)
    )))
}

/// Build a reasoning tree VizBlock from a [`ReasoningTree`].
///
/// Pure conversion — no graph queries needed. Serializes the tree
/// to JSON and generates an ASCII fallback.
pub fn build_reasoning_tree_viz(tree: &ReasoningTree) -> Result<VizBlock> {
    let tree_data = serde_json::to_value(tree)?;
    let builder = ReasoningTreeVizBuilder {
        tree_data,
        confidence: tree.confidence,
        node_count: tree.node_count,
        depth: tree.depth,
        request: tree.request.clone(),
    };
    builder.build()
}

/// Build a progress bar VizBlock from a plan ID.
///
/// Queries the graph for the plan and its tasks, then builds a
/// segmented progress bar visualization.
pub async fn build_progress_viz(
    graph: &dyn GraphStore,
    plan_id: Uuid,
) -> Result<VizBlock> {
    let plan = graph
        .get_plan(plan_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Plan {plan_id} not found"))?;

    let tasks = graph.get_plan_tasks(plan_id).await?;

    let task_progress: Vec<TaskProgress> = tasks
        .iter()
        .map(|t| {
            // TaskStatus uses serde rename_all = snake_case, serialize to get the string
            let status_str = serde_json::to_value(&t.status)
                .ok()
                .and_then(|v| v.as_str().map(|s| s.to_string()))
                .unwrap_or_else(|| format!("{:?}", t.status).to_lowercase());
            TaskProgress {
                title: t.title.clone().unwrap_or_else(|| t.description.chars().take(60).collect()),
                status: status_str,
                priority: t.priority.unwrap_or(50),
            }
        })
        .collect();

    let builder = ProgressBarVizBuilder {
        title: plan.title.clone(),
        tasks: task_progress,
        plan_id: Some(plan_id.to_string()),
    };
    builder.build()
}

/// Build a knowledge card VizBlock from a note ID.
///
/// Queries the graph for the note and its linked entities,
/// then builds an inline knowledge card.
pub async fn build_knowledge_card_viz(
    graph: &dyn GraphStore,
    note_id: Uuid,
) -> Result<VizBlock> {
    let note = graph
        .get_note(note_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Note {note_id} not found"))?;

    // Get anchors to find linked entities
    let anchors = graph.get_note_anchors(note_id).await?;
    let linked_entities: Vec<String> = anchors
        .iter()
        .map(|a| format!("{}:{}", a.entity_type, a.entity_id))
        .collect();

    let builder = KnowledgeCardVizBuilder {
        entity_type: "note".to_string(),
        entity_id: note_id.to_string(),
        kind: note.note_type.to_string(),
        content: note.content.clone(),
        importance: note.importance.to_string(),
        tags: note.tags.clone(),
        linked_entities,
    };
    builder.build()
}

/// Build a context radar VizBlock from 5 dimension scores.
///
/// The ContextVector is a T4 feature; this helper accepts pre-computed
/// dimension scores for forward-compatibility.
pub fn build_radar_viz(dimensions: &[(String, f64)]) -> Result<VizBlock> {
    let axes: Vec<serde_json::Value> = dimensions
        .iter()
        .map(|(name, value)| {
            serde_json::json!({
                "name": name,
                "value": value,
            })
        })
        .collect();

    // Build fallback text as a table
    let mut fallback_lines = vec!["Context Radar:".to_string()];
    for (name, value) in dimensions {
        let bar_len = (*value * 20.0).round() as usize;
        let bar = "█".repeat(bar_len) + &"░".repeat(20 - bar_len.min(20));
        fallback_lines.push(format!("  {name:.<20} [{bar}] {:.0}%", value * 100.0));
    }

    Ok(VizBlock::new(
        VizType::ContextRadar,
        serde_json::json!({
            "axes": axes,
            "dimension_count": dimensions.len(),
        }),
        fallback_lines.join("\n"),
    )
    .with_title("Context Radar"))
}

/// Build a dependency tree VizBlock from a file path.
///
/// Queries the graph for the file's imports and dependents.
pub async fn build_dependency_tree_viz(
    graph: &dyn GraphStore,
    file_path: &str,
    project_id: Option<Uuid>,
) -> Result<VizBlock> {
    let dependents = graph.find_dependent_files(file_path, 2, project_id).await?;

    // Build fallback
    let mut fallback_lines = vec![format!("Dependencies for: {file_path}")];
    if !dependents.is_empty() {
        fallback_lines.push(format!("Dependents ({}):", dependents.len()));
        for dep in &dependents {
            fallback_lines.push(format!("  ← {dep}"));
        }
    } else {
        fallback_lines.push("No dependents found.".to_string());
    }

    Ok(VizBlock::new(
        VizType::DependencyTree,
        serde_json::json!({
            "root": file_path,
            "dependents": dependents,
            "dependent_count": dependents.len(),
        }),
        fallback_lines.join("\n"),
    )
    .with_interactive(true)
    .with_title(format!(
        "Dependencies: {}",
        file_path.rsplit('/').next().unwrap_or(file_path)
    )))
}

// ============================================================================
// VizBlockProcessor — orchestrates tag parsing and block generation
// ============================================================================

/// Processes LLM text output, replacing `<viz>` tags with real VizBlocks.
///
/// Holds a reference to the graph for fetching real data.
/// Also accepts an optional [`ReasoningTree`] for the current query
/// (injected by the enrichment pipeline).
pub struct VizBlockProcessor {
    graph: Arc<dyn GraphStore>,
    /// Optional reasoning tree for the current query.
    reasoning_tree: Option<ReasoningTree>,
    /// Project scope for graph queries.
    project_id: Option<Uuid>,
}

impl VizBlockProcessor {
    /// Create a new processor.
    pub fn new(
        graph: Arc<dyn GraphStore>,
        project_id: Option<Uuid>,
    ) -> Self {
        Self {
            graph,
            reasoning_tree: None,
            project_id,
        }
    }

    /// Set the reasoning tree for the current query.
    pub fn with_reasoning_tree(mut self, tree: ReasoningTree) -> Self {
        self.reasoning_tree = Some(tree);
        self
    }

    /// Process LLM text and return a sequence of ContentBlocks.
    ///
    /// Text between `<viz>` tags becomes `TextBlock`s.
    /// Each `<viz>` tag becomes a `VizBlock` with real data.
    /// If a viz tag fails to resolve, it's replaced with a fallback text block.
    pub async fn process(&self, text: &str) -> Vec<ContentBlock> {
        let tags = parse_viz_tags(text);

        if tags.is_empty() {
            return vec![ContentBlock::text(text)];
        }

        let mut blocks = Vec::new();
        let mut last_end = 0;

        for tag in &tags {
            // Add text before this tag (if any)
            if tag.start > last_end {
                let before = text[last_end..tag.start].trim();
                if !before.is_empty() {
                    blocks.push(ContentBlock::text(before));
                }
            }

            // Build the VizBlock
            match self.build_viz_for_tag(tag).await {
                Ok(viz_block) => {
                    blocks.push(ContentBlock::viz(viz_block));
                }
                Err(e) => {
                    // Graceful degradation: show error as text
                    blocks.push(ContentBlock::text(format!(
                        "[viz:{} error: {}]",
                        tag.viz_type, e
                    )));
                }
            }

            last_end = tag.end;
        }

        // Add remaining text after the last tag
        if last_end < text.len() {
            let after = text[last_end..].trim();
            if !after.is_empty() {
                blocks.push(ContentBlock::text(after));
            }
        }

        blocks
    }

    /// Build a VizBlock from a parsed tag by dispatching to the appropriate helper.
    async fn build_viz_for_tag(&self, tag: &ParsedVizTag) -> Result<VizBlock> {
        let viz_type = VizType::from_str_loose(&tag.viz_type);

        match viz_type {
            VizType::ImpactGraph => {
                let target = tag
                    .attrs
                    .get("target")
                    .ok_or_else(|| anyhow::anyhow!("impact_graph requires 'target' attribute"))?;
                build_impact_viz(self.graph.as_ref(), target, self.project_id).await
            }

            VizType::ReasoningTree => {
                if let Some(ref tree) = self.reasoning_tree {
                    build_reasoning_tree_viz(tree)
                } else {
                    Ok(VizBlock::new(
                        VizType::ReasoningTree,
                        serde_json::Value::Null,
                        "No reasoning tree available for this query.",
                    ))
                }
            }

            VizType::ProgressBar => {
                let plan_id_str = tag
                    .attrs
                    .get("plan_id")
                    .ok_or_else(|| anyhow::anyhow!("progress_bar requires 'plan_id' attribute"))?;
                let plan_id = Uuid::parse_str(plan_id_str)?;
                build_progress_viz(self.graph.as_ref(), plan_id).await
            }

            VizType::KnowledgeCard => {
                let note_id_str = tag
                    .attrs
                    .get("note_id")
                    .ok_or_else(|| anyhow::anyhow!("knowledge_card requires 'note_id' attribute"))?;
                let note_id = Uuid::parse_str(note_id_str)?;
                build_knowledge_card_viz(self.graph.as_ref(), note_id).await
            }

            VizType::ContextRadar => {
                // ContextVector is a T4 feature. Accept pre-built dimensions or return stub.
                Ok(VizBlock::new(
                    VizType::ContextRadar,
                    serde_json::json!({"status": "pending", "message": "ContextVector available in TP4"}),
                    "Context Radar: available after TP4 — Session Continuity & Neural Feedback.",
                ))
            }

            VizType::DependencyTree => {
                let target = tag
                    .attrs
                    .get("target")
                    .ok_or_else(|| {
                        anyhow::anyhow!("dependency_tree requires 'target' attribute")
                    })?;
                build_dependency_tree_viz(self.graph.as_ref(), target, self.project_id).await
            }

            // Pattern Federation reserved types → stub
            viz_type if viz_type.is_pattern_federation() => {
                Ok(VizBlock::new(
                    viz_type.clone(),
                    serde_json::json!({"status": "not_installed"}),
                    format!(
                        "[{viz_type}] Available after Pattern Federation installation."
                    ),
                ))
            }

            // Unknown/custom types → generic stub
            _ => Ok(VizBlock::new(
                viz_type.clone(),
                serde_json::Value::Null,
                format!("[{viz_type}] Unknown visualization type."),
            )),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::mock::MockGraphStore;
    use crate::neo4j::models::{PlanNode, PlanStatus, TaskNode, TaskStatus};
    use crate::reasoning::models::{EntitySource, ReasoningNode};

    fn make_mock_graph() -> Arc<MockGraphStore> {
        Arc::new(MockGraphStore::new())
    }

    // --- Tag parsing tests ---

    #[test]
    fn test_parse_single_viz_tag() {
        let text = r#"Here is the analysis: <viz type="impact_graph" target="src/main.rs" /> And more text."#;
        let tags = parse_viz_tags(text);
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].viz_type, "impact_graph");
        assert_eq!(tags[0].attrs.get("target").unwrap(), "src/main.rs");
    }

    #[test]
    fn test_parse_multiple_viz_tags() {
        let text = r#"Text <viz type="impact_graph" target="a.rs" /> middle <viz type="progress_bar" plan_id="abc-123" /> end"#;
        let tags = parse_viz_tags(text);
        assert_eq!(tags.len(), 2);
        assert_eq!(tags[0].viz_type, "impact_graph");
        assert_eq!(tags[1].viz_type, "progress_bar");
        assert_eq!(tags[1].attrs.get("plan_id").unwrap(), "abc-123");
    }

    #[test]
    fn test_parse_no_viz_tags() {
        let text = "Just regular text without any tags.";
        let tags = parse_viz_tags(text);
        assert!(tags.is_empty());
    }

    #[test]
    fn test_parse_tag_with_many_attrs() {
        let text = r#"<viz type="knowledge_card" note_id="abc" importance="high" kind="gotcha" />"#;
        let tags = parse_viz_tags(text);
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].attrs.get("note_id").unwrap(), "abc");
        assert_eq!(tags[0].attrs.get("importance").unwrap(), "high");
        assert_eq!(tags[0].attrs.get("kind").unwrap(), "gotcha");
    }

    #[test]
    fn test_parse_preserves_offsets() {
        let text = r#"before <viz type="test" /> after"#;
        let tags = parse_viz_tags(text);
        assert_eq!(tags.len(), 1);
        assert_eq!(&text[tags[0].start..tags[0].end], r#"<viz type="test" />"#);
    }

    // --- Helper function tests ---

    #[tokio::test]
    async fn test_build_impact_viz_no_impacts() {
        let graph = make_mock_graph();
        let block = build_impact_viz(graph.as_ref(), "src/unknown.rs", None)
            .await
            .unwrap();

        assert_eq!(block.viz_type, VizType::ImpactGraph);
        assert!(block.interactive);
        assert!(block.fallback_text.contains("src/unknown.rs"));
        assert!(block.fallback_text.contains("No impact detected"));
    }

    #[tokio::test]
    async fn test_build_impact_viz_with_data() {
        let graph = make_mock_graph();

        // Seed import relationships: b.rs imports a.rs
        {
            let mut ir = graph.import_relationships.write().await;
            ir.insert(
                "src/b.rs".to_string(),
                vec!["src/a.rs".to_string()],
            );
        }

        let block = build_impact_viz(graph.as_ref(), "src/a.rs", None)
            .await
            .unwrap();

        assert_eq!(block.viz_type, VizType::ImpactGraph);
        assert!(block.data["impacted_count"].as_u64().unwrap() > 0);
        assert!(block.fallback_text.contains("src/b.rs"));
    }

    #[test]
    fn test_build_reasoning_tree_viz() {
        let mut tree = ReasoningTree::new("test query", None);
        let node = ReasoningNode::new(EntitySource::Note, "note-1", 0.9, "Test reasoning");
        tree.add_root(node);

        let block = build_reasoning_tree_viz(&tree).unwrap();
        assert_eq!(block.viz_type, VizType::ReasoningTree);
        assert!(block.interactive);
        assert!(block.fallback_text.contains("90%"));
        assert!(block.fallback_text.contains("1 nodes"));
    }

    #[tokio::test]
    async fn test_build_progress_viz() {
        let graph = make_mock_graph();
        let plan_id = Uuid::new_v4();
        let project_id = Uuid::new_v4();

        // Seed plan
        let plan = PlanNode {
            id: plan_id,
            title: "Test Plan".to_string(),
            description: "Test description".to_string(),
            status: PlanStatus::InProgress,
            priority: 80,
            created_at: chrono::Utc::now(),
            created_by: "test".to_string(),
            project_id: Some(project_id),
        };
        graph.plans.write().await.insert(plan_id, plan);

        // Seed tasks
        let task1_id = Uuid::new_v4();
        let task2_id = Uuid::new_v4();
        let task1 = TaskNode {
            id: task1_id,
            title: Some("Task 1".to_string()),
            description: "Done".to_string(),
            status: TaskStatus::Completed,
            priority: Some(90),
            tags: vec![],
            acceptance_criteria: vec![],
            affected_files: vec![],
            created_at: chrono::Utc::now(),
            updated_at: None,
            assigned_to: None,
            estimated_complexity: None,
            actual_complexity: None,
            started_at: None,
            completed_at: None,
        };
        let task2 = TaskNode {
            id: task2_id,
            title: Some("Task 2".to_string()),
            description: "Pending".to_string(),
            status: TaskStatus::Pending,
            priority: Some(80),
            tags: vec![],
            acceptance_criteria: vec![],
            affected_files: vec![],
            created_at: chrono::Utc::now(),
            updated_at: None,
            assigned_to: None,
            estimated_complexity: None,
            actual_complexity: None,
            started_at: None,
            completed_at: None,
        };
        graph.tasks.write().await.insert(task1_id, task1);
        graph.tasks.write().await.insert(task2_id, task2);
        graph
            .plan_tasks
            .write()
            .await
            .insert(plan_id, vec![task1_id, task2_id]);

        let block = build_progress_viz(graph.as_ref(), plan_id)
            .await
            .unwrap();

        assert_eq!(block.viz_type, VizType::ProgressBar);
        assert_eq!(block.data["total"], 2);
        assert_eq!(block.data["completed"], 1);
        assert_eq!(block.data["percentage"], 50.0);
        assert!(block.fallback_text.contains("50%"));
        assert!(block.fallback_text.contains("1/2 tasks"));
    }

    #[test]
    fn test_build_radar_viz() {
        let dimensions = vec![
            ("Knowledge".to_string(), 0.8),
            ("Complexity".to_string(), 0.6),
            ("Risk".to_string(), 0.3),
            ("Coupling".to_string(), 0.5),
            ("Activity".to_string(), 0.9),
        ];

        let block = build_radar_viz(&dimensions).unwrap();
        assert_eq!(block.viz_type, VizType::ContextRadar);
        assert_eq!(block.data["dimension_count"], 5);
        assert!(block.fallback_text.contains("Knowledge"));
        assert!(block.fallback_text.contains("80%"));
    }

    #[tokio::test]
    async fn test_build_dependency_tree_viz() {
        let graph = make_mock_graph();

        // Seed: c.rs imports b.rs, b.rs imports a.rs
        {
            let mut ir = graph.import_relationships.write().await;
            ir.insert("src/b.rs".to_string(), vec!["src/a.rs".to_string()]);
            ir.insert("src/c.rs".to_string(), vec!["src/b.rs".to_string()]);
        }

        let block = build_dependency_tree_viz(graph.as_ref(), "src/a.rs", None)
            .await
            .unwrap();

        assert_eq!(block.viz_type, VizType::DependencyTree);
        assert!(block.interactive);
        assert!(block.data["dependent_count"].as_u64().unwrap() > 0);
    }

    // --- VizBlockProcessor tests ---

    #[tokio::test]
    async fn test_processor_no_tags() {
        let graph = make_mock_graph();
        let processor = VizBlockProcessor::new(graph, None);
        let blocks = processor.process("Just plain text.").await;

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].fallback_text(), "Just plain text.");
    }

    #[tokio::test]
    async fn test_processor_with_impact_tag() {
        let graph = make_mock_graph();

        // Seed some data
        {
            let mut ir = graph.import_relationships.write().await;
            ir.insert("src/b.rs".to_string(), vec!["src/a.rs".to_string()]);
        }

        let processor = VizBlockProcessor::new(graph, None);
        let text = r#"Here's the impact: <viz type="impact_graph" target="src/a.rs" /> See above."#;
        let blocks = processor.process(text).await;

        assert_eq!(blocks.len(), 3); // text + viz + text
        assert_eq!(blocks[0].fallback_text(), "Here's the impact:");
        if let ContentBlock::Viz(viz) = &blocks[1] {
            assert_eq!(viz.viz_type, VizType::ImpactGraph);
            assert!(viz.data["impacted_count"].as_u64().unwrap() >= 1);
        } else {
            panic!("Expected Viz block");
        }
        assert_eq!(blocks[2].fallback_text(), "See above.");
    }

    #[tokio::test]
    async fn test_processor_with_reasoning_tree() {
        let graph = make_mock_graph();
        let mut tree = ReasoningTree::new("test query", None);
        tree.add_root(ReasoningNode::new(
            EntitySource::Note,
            "note-1",
            0.9,
            "Test",
        ));

        let processor = VizBlockProcessor::new(graph, None).with_reasoning_tree(tree);

        let text = r#"Analysis: <viz type="reasoning_tree" /> Done."#;
        let blocks = processor.process(text).await;

        assert_eq!(blocks.len(), 3);
        if let ContentBlock::Viz(viz) = &blocks[1] {
            assert_eq!(viz.viz_type, VizType::ReasoningTree);
            assert!(viz.interactive);
        } else {
            panic!("Expected Viz block");
        }
    }

    #[tokio::test]
    async fn test_processor_graceful_degradation() {
        let graph = make_mock_graph();
        let processor = VizBlockProcessor::new(graph, None);

        // Missing required attribute
        let text = r#"Test: <viz type="impact_graph" /> end"#;
        let blocks = processor.process(text).await;

        assert_eq!(blocks.len(), 3); // text + error text + text
        let error_text = blocks[1].fallback_text();
        assert!(error_text.contains("error"), "Got: {error_text}");
    }

    #[tokio::test]
    async fn test_processor_pattern_federation_stub() {
        let graph = make_mock_graph();
        let processor = VizBlockProcessor::new(graph, None);

        let text = r#"<viz type="protocol_run" />"#;
        let blocks = processor.process(text).await;

        assert_eq!(blocks.len(), 1);
        if let ContentBlock::Viz(viz) = &blocks[0] {
            assert_eq!(viz.viz_type, VizType::ProtocolRun);
            assert!(viz.fallback_text.contains("Pattern Federation"));
        } else {
            panic!("Expected Viz block");
        }
    }

    #[tokio::test]
    async fn test_processor_multiple_tags_interleaved() {
        let graph = make_mock_graph();
        let plan_id = Uuid::new_v4();

        // Seed plan and tasks
        let plan = PlanNode {
            id: plan_id,
            title: "My Plan".to_string(),
            description: "desc".to_string(),
            status: PlanStatus::InProgress,
            priority: 80,
            created_at: chrono::Utc::now(),
            created_by: "test".to_string(),
            project_id: None,
        };
        graph.plans.write().await.insert(plan_id, plan);
        graph
            .plan_tasks
            .write()
            .await
            .insert(plan_id, vec![]);

        let processor = VizBlockProcessor::new(graph, None);
        let text = format!(
            r#"Before <viz type="progress_bar" plan_id="{plan_id}" /> middle <viz type="protocol_run" /> after"#
        );
        let blocks = processor.process(&text).await;

        // Should be: text + viz + text + viz + text
        assert_eq!(blocks.len(), 5);
        assert!(matches!(&blocks[0], ContentBlock::Text(_)));
        assert!(matches!(&blocks[1], ContentBlock::Viz(_)));
        assert!(matches!(&blocks[2], ContentBlock::Text(_)));
        assert!(matches!(&blocks[3], ContentBlock::Viz(_)));
        assert!(matches!(&blocks[4], ContentBlock::Text(_)));
    }
}
