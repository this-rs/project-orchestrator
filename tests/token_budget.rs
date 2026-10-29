//! The per-turn token budget of the STATIC prompt, measured offline.
//!
//! ## Why this exists
//!
//! Three static blocks ride in **every** request a model receives, before a single
//! token of conversation, knowledge or plan status:
//!
//! | block | where it is built | measured (chars) | ≈ tokens (chars / 4) |
//! |---|---|---|---|
//! | tool definitions (33 mega-tools, JSON schemas) | `crate::mcp::tools::all_tools` | 71 558 | ≈ 17 900 |
//! | tool reference (markdown, group-filtered) | `chat::prompt::TOOL_REFERENCE` | 48 747 | ≈ 12 200 |
//! | base system prompt (17 sections) | `chat::prompt_sections::all_base_sections` | 45 632 | ≈ 11 400 |
//!
//! The dynamic context (notes, plan status, reasoning) is a fourth block, budgeted
//! separately by `chat::composer::compute_dynamic_budget` (5 K to 40 K chars according
//! to the model's context window), and is not measured here.
//!
//! The first two blocks document **the same 33 tools twice**, once as a machine-readable
//! schema and once as prose: ≈ 30 100 tokens, 73 % of the static prompt. Reducing that
//! duplication is the real lever, and it is the job of the group filtering /
//! deferred-loading path (`chat::prompt_sections::select_tool_groups`,
//! `chat::deferred_tools`), not of this file. This file only guards the blocks from
//! growing back.
//!
//! ## What was removed, and why it was safe
//!
//! Every tool description used to re-list its own actions verbatim —
//! `"Manage plans. Actions: list, create, get, …"` — while the same list is the
//! `action` property's `enum`, three lines below in the same definition. That echo was
//! 5 874 chars (≈ 1 470 tokens) re-sent at every turn, and it rotted: the copy in
//! `description` is what an author forgets to update when adding an action. With it
//! gone, the `enum` alone names the actions, and the markdown reference documents them.
//! `src/mcp/handlers.rs` (`test_description_does_not_re_list_the_action_enum`) keeps it
//! gone.
//!
//! The same echo appeared once more, inside `TOOL_REFERENCE`: `## plan` was followed by
//! `Manage plans. Actions: list, create, …` and then by the table whose first column is
//! that very list. Removing it (3 165 chars net, ≈ 790 tokens, after adding the 23 rows
//! that were missing) also exposed a real documentation gap, now closed and guarded by
//! `tests/prompt_completeness.rs` (`every_action_has_a_row_in_its_tool_reference_section`):
//! 23 actions accepted by a schema (`chat.add_anchor`, `sharing.retract`,
//! `workspace.derive_topology`, …) had no row anywhere in the prompt's reference.
//!
//! ## gzip is NOT a lever
//!
//! Gzip (and every transport encoding) shrinks **bytes on the wire**. Billing, the
//! context window and the cache are counted in **tokens**, and a model sees the
//! decompressed text. Compressing the request changes neither. The levers that do move
//! tokens are: fewer/leaner tool definitions, lazy loading by group, prompt caching of
//! the byte-stable prefix, and compaction of the history.
//!
//! ## The ceilings
//!
//! A failure here is not a bug in the test: it means the static prompt grew. Measure the
//! split again (`cargo test --test token_budget -- --nocapture` and read the numbers),
//! say in the PR what the growth buys, then raise the ceiling deliberately.
use project_orchestrator::neo4j::traits::*;

use project_orchestrator::chat::prompt::{BASE_SYSTEM_PROMPT, TOOL_REFERENCE};
use project_orchestrator::chat::prompt_sections::all_base_sections;
use project_orchestrator::mcp::tools::all_tools;

/// Chars of JSON for the whole tool catalog, `name` + `description` + `input_schema`.
/// Measured 2026-10-26: 71 558 (was 77 432 before the action-echo removal).
const TOOL_DEFINITIONS_CHAR_BUDGET: usize = 74_000;

/// Chars of the markdown tool reference (every group selected; the composer renders
/// only the selected groups, so a real turn is at or below this).
const TOOL_REFERENCE_CHAR_BUDGET: usize = 54_000;

/// Chars of the base system prompt sections, full and compact variants together.
const BASE_SECTIONS_CHAR_BUDGET: usize = 47_000;

/// The JSON of every tool definition, as a provider receives it.
fn tool_definitions_chars() -> usize {
    all_tools()
        .iter()
        .map(|t| {
            serde_json::to_string(t)
                .expect("a tool definition serialises")
                .len()
        })
        .sum()
}

#[test]
fn the_tool_definitions_fit_their_token_budget() {
    let chars = tool_definitions_chars();
    assert!(
        chars <= TOOL_DEFINITIONS_CHAR_BUDGET,
        "tool definitions are {chars} chars (≈ {} tokens), over the {TOOL_DEFINITIONS_CHAR_BUDGET}-char \
         budget. This block is re-sent at every turn: measure before raising the ceiling \
         (see the module docs).",
        chars / 4
    );
}

#[test]
fn the_tool_reference_fits_its_token_budget() {
    let chars = TOOL_REFERENCE.len();
    assert!(
        chars <= TOOL_REFERENCE_CHAR_BUDGET,
        "TOOL_REFERENCE is {chars} chars (≈ {} tokens), over the {TOOL_REFERENCE_CHAR_BUDGET}-char \
         budget: it is extracted into the system prompt of every turn.",
        chars / 4
    );
}

#[test]
fn the_base_sections_fit_their_token_budget() {
    let chars: usize = all_base_sections()
        .iter()
        .map(|s| s.content.len() + s.compact_content.map(str::len).unwrap_or(0))
        .sum();
    assert!(
        chars <= BASE_SECTIONS_CHAR_BUDGET,
        "the base prompt sections are {chars} chars (≈ {} tokens), over the \
         {BASE_SECTIONS_CHAR_BUDGET}-char budget.",
        chars / 4
    );
}

/// Stripping the echo (guarded by `mcp::handlers`
/// `test_description_does_not_re_list_the_action_enum`) must not leave a tool without a
/// description: the description is the only thing a model reads about a tool it has not
/// loaded yet (a deferred tool is named to it by the system prompt, its schema withheld).
#[test]
fn every_tool_keeps_a_meaningful_description() {
    let mut too_short: Vec<(String, usize)> = Vec::new();
    for tool in all_tools() {
        let len = tool.description.trim().len();
        if len < 8 {
            too_short.push((tool.name.clone(), len));
        }
    }
    assert!(
        too_short.is_empty(),
        "these tools have no usable description: {too_short:?}"
    );
}

/// The split, printed for whoever raises a ceiling.
#[test]
fn report_the_static_prompt_split() {
    let tools = tool_definitions_chars();
    let reference = TOOL_REFERENCE.len();
    let base = BASE_SYSTEM_PROMPT.len();
    let sections: usize = all_base_sections()
        .iter()
        .map(|s| s.content.len() + s.compact_content.map(str::len).unwrap_or(0))
        .sum();
    let total = tools + reference + sections;
    println!("\n=== STATIC PROMPT (chars / ≈tokens at 4 chars per token) ===");
    println!("tool definitions     {tools:>7} / {:>6}", tools / 4);
    println!("tool reference       {reference:>7} / {:>6}", reference / 4);
    println!("base sections        {sections:>7} / {:>6}", sections / 4);
    println!("(BASE_SYSTEM_PROMPT  {base:>7} — the monolithic source of the sections)");
    println!("TOTAL                {total:>7} / {:>6}", total / 4);
    println!(
        "tool schemas + reference = {} tokens, {:.0}% of the static prompt\n",
        (tools + reference) / 4,
        100.0 * (tools + reference) as f64 / total as f64
    );
}
