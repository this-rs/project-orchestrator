//! Completeness guard for the assembled tool reference.
//!
//! ## Why this exists
//!
//! `extract_tool_reference` keeps only the `## <tool>` sections of `TOOL_REFERENCE`
//! whose header matches a tool name declared by the selected `ToolRefGroupId`s. When a
//! declared tool has no section, the group silently renders nothing — no error, no log,
//! just missing documentation in the system prompt. That is exactly how
//! `ToolRefGroupId::External` (`mcp_federation`) shipped with zero lines.
//!
//! `test_tool_groups_no_overlap` (in `src/chat/prompt_sections.rs`) checks that no tool
//! belongs to two groups. That is *overlap*, not *coverage*: it cannot see this class of
//! defect. These tests check coverage, in both directions:
//!
//!  - declared in a group but absent from `TOOL_REFERENCE` → silently undocumented;
//!  - present in `TOOL_REFERENCE` but in no group → can never be selected, dead weight.
//!
//! Both assertions name the offending tools explicitly, so a failure is actionable
//! without reading the test.

use std::collections::BTreeSet;

use project_orchestrator::chat::prompt::TOOL_REFERENCE;
use project_orchestrator::chat::prompt_sections::{extract_tool_reference, ToolRefGroupId};

/// Tool names carrying a `## <name>` section in `TOOL_REFERENCE`.
///
/// Mirrors the header detection of `extract_tool_reference`: a line starting with
/// `"## "`, the tool name being the first whitespace-delimited word after the marker.
fn documented_tools(tool_reference: &str) -> BTreeSet<&str> {
    tool_reference
        .lines()
        .filter(|line| line.starts_with("## "))
        .filter_map(|line| line[3..].split_whitespace().next())
        .collect()
}

/// Union of the tool names declared by every group.
fn declared_tools() -> BTreeSet<&'static str> {
    ToolRefGroupId::ALL
        .iter()
        .flat_map(|g| g.tool_names().iter().copied())
        .collect()
}

/// Direction 1: every declared tool must have a section, or its group renders nothing.
#[test]
fn every_declared_tool_has_a_tool_reference_section() {
    let documented = documented_tools(TOOL_REFERENCE);

    let mut missing: Vec<String> = Vec::new();
    for group in ToolRefGroupId::ALL {
        for tool in group.tool_names() {
            if !documented.contains(tool) {
                missing.push(format!("{} (group {:?})", tool, group));
            }
        }
    }

    assert!(
        missing.is_empty(),
        "tools declared in ToolRefGroupId but missing a `## <name>` section in \
         TOOL_REFERENCE — their group renders no documentation at all: {}",
        missing.join(", ")
    );
}

/// Direction 2: every documented tool must belong to a group, or it is unreachable.
#[test]
fn every_tool_reference_section_belongs_to_a_group() {
    let declared = declared_tools();

    let orphans: Vec<&str> = documented_tools(TOOL_REFERENCE)
        .into_iter()
        .filter(|tool| !declared.contains(tool))
        .collect();

    assert!(
        orphans.is_empty(),
        "tools with a `## <name>` section in TOOL_REFERENCE but in no ToolRefGroupId — \
         they can never be selected and will never reach the prompt: {}",
        orphans.join(", ")
    );
}

/// Empirical check on the group that was broken: `External` must render real content.
///
/// `extract_tool_reference` always emits a fixed preamble, so "non-empty" is not
/// enough — the output is compared against the preamble-only baseline.
#[test]
fn external_group_renders_tool_documentation() {
    let baseline = extract_tool_reference(TOOL_REFERENCE, &[]);
    let external = extract_tool_reference(TOOL_REFERENCE, &[ToolRefGroupId::External]);

    assert!(
        external.len() > baseline.len(),
        "ToolRefGroupId::External rendered nothing beyond the preamble \
         ({} bytes, same as the empty selection)",
        external.len()
    );
    assert!(
        external.contains("## mcp_federation"),
        "ToolRefGroupId::External output is missing the `## mcp_federation` section:\n{}",
        external
    );
    for action in [
        "| connect ",
        "| disconnect ",
        "| list ",
        "| status ",
        "| tools ",
        "| probe ",
        "| reconnect ",
        "| backfill_relations ",
        "| backfill_sequences ",
    ] {
        assert!(
            external.contains(action),
            "`## mcp_federation` section is missing the row for action `{}`",
            action.trim().trim_start_matches("| ")
        );
    }
}

/// Every group must render something — the failure mode generalised beyond `External`.
#[test]
fn every_group_renders_tool_documentation() {
    let baseline = extract_tool_reference(TOOL_REFERENCE, &[]);

    let empty: Vec<String> = ToolRefGroupId::ALL
        .iter()
        .filter(|group| extract_tool_reference(TOOL_REFERENCE, &[**group]).len() <= baseline.len())
        .map(|group| format!("{:?}", group))
        .collect();

    assert!(
        empty.is_empty(),
        "tool groups rendering no documentation beyond the preamble: {}",
        empty.join(", ")
    );
}
