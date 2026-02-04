//! Note Lifecycle Management
//!
//! Handles automatic detection of note obsolescence during code sync,
//! staleness calculation, and note status transitions.

use super::hashing::{hash_function_body, hash_function_signature, similarity_score};
use super::models::*;
use crate::neo4j::models::FunctionNode;
use crate::parser::ParsedFile;
use chrono::{DateTime, Utc};
use uuid::Uuid;

// Re-export types needed for assertion verification
pub use super::models::{AssertionCheckType, AssertionResult};

/// Result of verifying a note's anchors
#[derive(Debug, Clone)]
pub struct AnchorVerificationResult {
    /// The note ID
    pub note_id: Uuid,
    /// Whether all anchors are still valid
    pub all_valid: bool,
    /// Individual anchor results
    pub anchor_results: Vec<AnchorCheckResult>,
    /// Suggested status update if any
    pub suggested_update: Option<NoteStatusUpdate>,
}

/// Result of checking a single anchor
#[derive(Debug, Clone)]
pub struct AnchorCheckResult {
    /// Entity type of the anchor
    pub entity_type: EntityType,
    /// Entity ID
    pub entity_id: String,
    /// Whether the anchor is still valid
    pub is_valid: bool,
    /// Reason for the result
    pub reason: String,
    /// New hashes if entity changed
    pub new_hashes: Option<(String, String)>,
    /// Possible migration target if entity was renamed
    pub migration_target: Option<MigrationTarget>,
}

/// Information about a parsed function for verification
#[derive(Debug, Clone)]
pub struct FunctionInfo {
    pub name: String,
    pub params: Vec<(String, Option<String>)>,
    pub return_type: Option<String>,
    pub is_async: bool,
    pub is_unsafe: bool,
    pub body: String,
}

/// Information about a parsed file for verification
#[derive(Debug, Clone)]
pub struct FileInfo {
    pub path: String,
    pub functions: Vec<FunctionInfo>,
    pub structs: Vec<StructInfo>,
    pub symbols: Vec<String>,
    pub imports: Vec<String>,
}

/// Information about a parsed struct for verification
#[derive(Debug, Clone)]
pub struct StructInfo {
    pub name: String,
    pub fields: Vec<(String, String, bool)>, // (name, type, is_public)
    pub generics: Vec<String>,
}

/// Lifecycle manager for notes
pub struct NoteLifecycleManager {
    /// Similarity threshold for detecting renamed functions
    rename_similarity_threshold: f64,
}

impl Default for NoteLifecycleManager {
    fn default() -> Self {
        Self::new()
    }
}

impl NoteLifecycleManager {
    /// Create a new lifecycle manager
    pub fn new() -> Self {
        Self {
            rename_similarity_threshold: 0.7,
        }
    }

    /// Create with custom similarity threshold
    pub fn with_threshold(threshold: f64) -> Self {
        Self {
            rename_similarity_threshold: threshold,
        }
    }

    /// Create FileInfo from a ParsedFile and source content
    ///
    /// This extracts the information needed for note lifecycle verification
    /// from a parsed file structure.
    pub fn create_file_info(parsed: &ParsedFile, source: &str) -> FileInfo {
        let functions = parsed
            .functions
            .iter()
            .map(|f| Self::function_node_to_info(f, source))
            .collect();

        let structs = parsed
            .structs
            .iter()
            .map(|s| StructInfo {
                name: s.name.clone(),
                fields: Vec::new(), // Struct fields not stored in parser currently
                generics: s.generics.clone(),
            })
            .collect();

        let symbols = parsed.symbols.clone();
        let imports = parsed.imports.iter().map(|i| i.path.clone()).collect();

        FileInfo {
            path: parsed.path.clone(),
            functions,
            structs,
            symbols,
            imports,
        }
    }

    /// Convert a FunctionNode to FunctionInfo, extracting body from source
    fn function_node_to_info(func: &FunctionNode, source: &str) -> FunctionInfo {
        // Extract function body from source using line numbers
        let body =
            Self::extract_function_body(source, func.line_start as usize, func.line_end as usize);

        let params = func
            .params
            .iter()
            .map(|p| (p.name.clone(), p.type_name.clone()))
            .collect();

        FunctionInfo {
            name: func.name.clone(),
            params,
            return_type: func.return_type.clone(),
            is_async: func.is_async,
            is_unsafe: func.is_unsafe,
            body,
        }
    }

    /// Extract function body from source code using line numbers
    fn extract_function_body(source: &str, start_line: usize, end_line: usize) -> String {
        let lines: Vec<&str> = source.lines().collect();

        if start_line == 0 || end_line == 0 || start_line > lines.len() {
            return String::new();
        }

        // Convert 1-indexed to 0-indexed
        let start = (start_line - 1).min(lines.len());
        let end = end_line.min(lines.len());

        if start >= end {
            return String::new();
        }

        lines[start..end].join("\n")
    }

    /// Verify all anchors for a note against the current file state
    pub fn verify_note_anchors(
        &self,
        note: &Note,
        file_info: &FileInfo,
    ) -> AnchorVerificationResult {
        let mut anchor_results = Vec::new();
        let mut all_valid = true;
        let mut suggested_update: Option<NoteStatusUpdate> = None;

        for anchor in &note.anchors {
            let result = self.verify_single_anchor(anchor, file_info);

            if !result.is_valid {
                all_valid = false;

                // Determine the most severe update needed
                if suggested_update.is_none() {
                    suggested_update = Some(NoteStatusUpdate {
                        note_id: note.id,
                        new_status: if result.migration_target.is_some() {
                            NoteStatus::NeedsReview
                        } else {
                            NoteStatus::Obsolete
                        },
                        reason: result.reason.clone(),
                        new_hashes: result.new_hashes.clone(),
                        migration_target: result.migration_target.clone(),
                    });
                }
            }

            anchor_results.push(result);
        }

        AnchorVerificationResult {
            note_id: note.id,
            all_valid,
            anchor_results,
            suggested_update,
        }
    }

    /// Verify a single anchor
    fn verify_single_anchor(&self, anchor: &NoteAnchor, file_info: &FileInfo) -> AnchorCheckResult {
        match anchor.entity_type {
            EntityType::Function => self.verify_function_anchor(anchor, file_info),
            EntityType::File => self.verify_file_anchor(anchor, file_info),
            EntityType::Struct => self.verify_struct_anchor(anchor, file_info),
            _ => AnchorCheckResult {
                entity_type: anchor.entity_type.clone(),
                entity_id: anchor.entity_id.clone(),
                is_valid: true, // Assume valid for unsupported types
                reason: "Verification not implemented for this entity type".to_string(),
                new_hashes: None,
                migration_target: None,
            },
        }
    }

    /// Verify a function anchor
    fn verify_function_anchor(
        &self,
        anchor: &NoteAnchor,
        file_info: &FileInfo,
    ) -> AnchorCheckResult {
        // Find the function by name
        if let Some(func) = file_info
            .functions
            .iter()
            .find(|f| f.name == anchor.entity_id)
        {
            // Function found - check if it changed
            let new_sig_hash = hash_function_signature(
                &func.name,
                &func.params,
                func.return_type.as_deref(),
                func.is_async,
                func.is_unsafe,
            );
            let new_body_hash = hash_function_body(&func.body);

            let sig_matches = anchor.signature_hash.as_ref() == Some(&new_sig_hash);
            let body_matches = anchor.body_hash.as_ref() == Some(&new_body_hash);

            if sig_matches && body_matches {
                // No changes
                AnchorCheckResult {
                    entity_type: EntityType::Function,
                    entity_id: anchor.entity_id.clone(),
                    is_valid: true,
                    reason: "Function unchanged".to_string(),
                    new_hashes: None,
                    migration_target: None,
                }
            } else if sig_matches {
                // Body changed, signature same
                AnchorCheckResult {
                    entity_type: EntityType::Function,
                    entity_id: anchor.entity_id.clone(),
                    is_valid: false,
                    reason: "Function body changed".to_string(),
                    new_hashes: Some((new_sig_hash, new_body_hash)),
                    migration_target: None,
                }
            } else {
                // Signature changed
                AnchorCheckResult {
                    entity_type: EntityType::Function,
                    entity_id: anchor.entity_id.clone(),
                    is_valid: false,
                    reason: "Function signature changed".to_string(),
                    new_hashes: Some((new_sig_hash, new_body_hash)),
                    migration_target: None,
                }
            }
        } else {
            // Function not found - try to find a similar one (possible rename)
            if let Some(body_hash) = &anchor.body_hash {
                if let Some(similar) = self.find_similar_function(file_info, body_hash) {
                    return AnchorCheckResult {
                        entity_type: EntityType::Function,
                        entity_id: anchor.entity_id.clone(),
                        is_valid: false,
                        reason: format!("Function possibly renamed to '{}'", similar.0),
                        new_hashes: Some(similar.1.clone()),
                        migration_target: Some(MigrationTarget {
                            entity_type: EntityType::Function,
                            new_entity_id: similar.0.clone(),
                            similarity_score: similar.2,
                        }),
                    };
                }
            }

            // Function deleted
            AnchorCheckResult {
                entity_type: EntityType::Function,
                entity_id: anchor.entity_id.clone(),
                is_valid: false,
                reason: "Function deleted".to_string(),
                new_hashes: None,
                migration_target: None,
            }
        }
    }

    /// Verify a file anchor
    fn verify_file_anchor(&self, anchor: &NoteAnchor, file_info: &FileInfo) -> AnchorCheckResult {
        // File exists (we're verifying against its info)
        // Check structure hash if available
        let new_structure_hash =
            super::hashing::hash_file_structure(&file_info.symbols, &file_info.imports);

        if anchor.signature_hash.as_ref() == Some(&new_structure_hash) {
            AnchorCheckResult {
                entity_type: EntityType::File,
                entity_id: anchor.entity_id.clone(),
                is_valid: true,
                reason: "File structure unchanged".to_string(),
                new_hashes: None,
                migration_target: None,
            }
        } else {
            AnchorCheckResult {
                entity_type: EntityType::File,
                entity_id: anchor.entity_id.clone(),
                is_valid: false,
                reason: "File structure changed".to_string(),
                new_hashes: Some((new_structure_hash, String::new())),
                migration_target: None,
            }
        }
    }

    /// Verify a struct anchor
    fn verify_struct_anchor(&self, anchor: &NoteAnchor, file_info: &FileInfo) -> AnchorCheckResult {
        if let Some(s) = file_info
            .structs
            .iter()
            .find(|s| s.name == anchor.entity_id)
        {
            let new_hash = super::hashing::hash_struct_signature(&s.name, &s.fields, &s.generics);

            if anchor.signature_hash.as_ref() == Some(&new_hash) {
                AnchorCheckResult {
                    entity_type: EntityType::Struct,
                    entity_id: anchor.entity_id.clone(),
                    is_valid: true,
                    reason: "Struct unchanged".to_string(),
                    new_hashes: None,
                    migration_target: None,
                }
            } else {
                AnchorCheckResult {
                    entity_type: EntityType::Struct,
                    entity_id: anchor.entity_id.clone(),
                    is_valid: false,
                    reason: "Struct definition changed".to_string(),
                    new_hashes: Some((new_hash, String::new())),
                    migration_target: None,
                }
            }
        } else {
            AnchorCheckResult {
                entity_type: EntityType::Struct,
                entity_id: anchor.entity_id.clone(),
                is_valid: false,
                reason: "Struct deleted".to_string(),
                new_hashes: None,
                migration_target: None,
            }
        }
    }

    /// Find a function with similar body hash (for rename detection)
    fn find_similar_function(
        &self,
        file_info: &FileInfo,
        original_body_hash: &str,
    ) -> Option<(String, (String, String), f64)> {
        let mut best_match: Option<(String, (String, String), f64)> = None;

        for func in &file_info.functions {
            let body_hash = hash_function_body(&func.body);
            let sim = similarity_score(original_body_hash, &body_hash);

            if sim >= self.rename_similarity_threshold {
                let sig_hash = hash_function_signature(
                    &func.name,
                    &func.params,
                    func.return_type.as_deref(),
                    func.is_async,
                    func.is_unsafe,
                );

                if let Some((_, _, best_sim)) = &best_match {
                    if sim > *best_sim {
                        best_match = Some((func.name.clone(), (sig_hash, body_hash), sim));
                    }
                } else {
                    best_match = Some((func.name.clone(), (sig_hash, body_hash), sim));
                }
            }
        }

        best_match
    }

    /// Calculate staleness score for a note
    pub fn calculate_staleness_score(&self, note: &Note, now: DateTime<Utc>) -> f64 {
        let base_decay_days = note.base_decay_days();

        // Assertion notes never go stale (verified by code)
        if base_decay_days == f64::MAX {
            return 0.0;
        }

        let last_activity = note.last_confirmed_at.unwrap_or(note.created_at);
        let duration = now.signed_duration_since(last_activity);
        let days_since_activity = duration.num_days() as f64;

        // Exponential decay
        let staleness = 1.0 - (-days_since_activity / base_decay_days).exp();

        // Adjust by importance
        let importance_factor = note.importance.decay_factor();

        (staleness * importance_factor).clamp(0.0, 1.0)
    }

    /// Determine if a note should transition to stale
    pub fn should_become_stale(&self, note: &Note, staleness_score: f64) -> bool {
        note.status == NoteStatus::Active && staleness_score > 0.8
    }

    /// Batch verify notes for a file
    pub fn verify_notes_for_file(
        &self,
        notes: &[Note],
        file_info: &FileInfo,
    ) -> Vec<AnchorVerificationResult> {
        notes
            .iter()
            .map(|note| self.verify_note_anchors(note, file_info))
            .collect()
    }

    /// Verify an assertion note against a file
    ///
    /// Returns the verification result based on the assertion rule.
    pub fn verify_assertion(
        &self,
        note: &Note,
        file_info: &FileInfo,
    ) -> AssertionVerificationResult {
        if note.note_type != NoteType::Assertion {
            return AssertionVerificationResult {
                note_id: note.id,
                passed: true,
                message: "Not an assertion note".to_string(),
                check_type: None,
            };
        }

        let rule = match &note.assertion_rule {
            Some(r) => r,
            None => {
                return AssertionVerificationResult {
                    note_id: note.id,
                    passed: false,
                    message: "Assertion note has no rule defined".to_string(),
                    check_type: None,
                };
            }
        };

        // Check if this assertion applies to the current file
        if let Some(ref pattern) = rule.file_pattern {
            if !file_info.path.contains(pattern) {
                return AssertionVerificationResult {
                    note_id: note.id,
                    passed: true,
                    message: "File does not match pattern".to_string(),
                    check_type: Some(rule.check_type.clone()),
                };
            }
        }

        let result = match rule.check_type {
            AssertionCheckType::Exists => self.check_exists(&rule.target, file_info),
            AssertionCheckType::NotExists => self.check_not_exists(&rule.target, file_info),
            AssertionCheckType::SignatureContains => {
                self.check_signature_contains(&rule.target, file_info, &rule.parameters)
            }
            AssertionCheckType::DependsOn => {
                // This requires graph traversal, handled separately
                AssertionResult {
                    passed: true,
                    message: "DependsOn check requires graph access".to_string(),
                    details: None,
                    checked_at: chrono::Utc::now(),
                }
            }
            AssertionCheckType::Calls => {
                // This requires graph traversal, handled separately
                AssertionResult {
                    passed: true,
                    message: "Calls check requires graph access".to_string(),
                    details: None,
                    checked_at: chrono::Utc::now(),
                }
            }
            AssertionCheckType::NoCalls => {
                // This requires graph traversal, handled separately
                AssertionResult {
                    passed: true,
                    message: "NoCalls check requires graph access".to_string(),
                    details: None,
                    checked_at: chrono::Utc::now(),
                }
            }
            AssertionCheckType::Implements => {
                // This requires graph traversal, handled separately
                AssertionResult {
                    passed: true,
                    message: "Implements check requires graph access".to_string(),
                    details: None,
                    checked_at: chrono::Utc::now(),
                }
            }
        };

        AssertionVerificationResult {
            note_id: note.id,
            passed: result.passed,
            message: result.message,
            check_type: Some(rule.check_type.clone()),
        }
    }

    /// Check if an entity exists in the file
    fn check_exists(&self, target: &str, file_info: &FileInfo) -> AssertionResult {
        // Parse target format: "type:name" e.g., "function:validate_user"
        let parts: Vec<&str> = target.splitn(2, ':').collect();
        if parts.len() != 2 {
            return AssertionResult {
                passed: false,
                message: format!("Invalid target format: {}", target),
                details: None,
                checked_at: chrono::Utc::now(),
            };
        }

        let (entity_type, name) = (parts[0], parts[1]);
        let exists = match entity_type {
            "function" => file_info.functions.iter().any(|f| f.name == name),
            "struct" => file_info.structs.iter().any(|s| s.name == name),
            "symbol" => file_info.symbols.contains(&name.to_string()),
            "import" => file_info.imports.iter().any(|i| i.contains(name)),
            _ => false,
        };

        AssertionResult {
            passed: exists,
            message: if exists {
                format!("{} '{}' exists", entity_type, name)
            } else {
                format!("{} '{}' not found", entity_type, name)
            },
            details: None,
            checked_at: chrono::Utc::now(),
        }
    }

    /// Check if an entity does NOT exist in the file
    fn check_not_exists(&self, target: &str, file_info: &FileInfo) -> AssertionResult {
        let result = self.check_exists(target, file_info);
        AssertionResult {
            passed: !result.passed,
            message: if !result.passed {
                result.message.replace("not found", "correctly absent")
            } else {
                format!("{} should not exist but was found", target)
            },
            details: None,
            checked_at: chrono::Utc::now(),
        }
    }

    /// Check if a function signature contains specific elements
    fn check_signature_contains(
        &self,
        target: &str,
        file_info: &FileInfo,
        params: &Option<serde_json::Value>,
    ) -> AssertionResult {
        let parts: Vec<&str> = target.splitn(2, ':').collect();
        if parts.len() != 2 || parts[0] != "function" {
            return AssertionResult {
                passed: false,
                message: "SignatureContains only works with functions".to_string(),
                details: None,
                checked_at: chrono::Utc::now(),
            };
        }

        let func_name = parts[1];
        let func = match file_info.functions.iter().find(|f| f.name == func_name) {
            Some(f) => f,
            None => {
                return AssertionResult {
                    passed: false,
                    message: format!("Function '{}' not found", func_name),
                    details: None,
                    checked_at: chrono::Utc::now(),
                };
            }
        };

        // Check parameters if specified
        if let Some(params) = params {
            if let Some(expected_params) = params.get("params").and_then(|v| v.as_array()) {
                for expected in expected_params {
                    if let Some(param_name) = expected.as_str() {
                        if !func.params.iter().any(|(name, _)| name == param_name) {
                            return AssertionResult {
                                passed: false,
                                message: format!(
                                    "Parameter '{}' not found in function '{}'",
                                    param_name, func_name
                                ),
                                details: None,
                                checked_at: chrono::Utc::now(),
                            };
                        }
                    }
                }
            }

            // Check return type if specified
            if let Some(expected_return) = params.get("return_type").and_then(|v| v.as_str()) {
                if func.return_type.as_deref() != Some(expected_return) {
                    return AssertionResult {
                        passed: false,
                        message: format!(
                            "Return type mismatch: expected '{}', got '{:?}'",
                            expected_return, func.return_type
                        ),
                        details: None,
                        checked_at: chrono::Utc::now(),
                    };
                }
            }

            // Check async/unsafe flags
            if let Some(is_async) = params.get("is_async").and_then(|v| v.as_bool()) {
                if func.is_async != is_async {
                    return AssertionResult {
                        passed: false,
                        message: format!(
                            "Async mismatch: expected {}, got {}",
                            is_async, func.is_async
                        ),
                        details: None,
                        checked_at: chrono::Utc::now(),
                    };
                }
            }
        }

        AssertionResult {
            passed: true,
            message: format!("Function '{}' signature matches", func_name),
            details: None,
            checked_at: chrono::Utc::now(),
        }
    }

    /// Batch verify assertion notes for a file
    pub fn verify_assertions_for_file(
        &self,
        notes: &[Note],
        file_info: &FileInfo,
    ) -> Vec<AssertionVerificationResult> {
        notes
            .iter()
            .filter(|n| n.note_type == NoteType::Assertion)
            .map(|note| self.verify_assertion(note, file_info))
            .collect()
    }
}

/// Result of verifying an assertion
#[derive(Debug, Clone)]
pub struct AssertionVerificationResult {
    /// The note ID
    pub note_id: Uuid,
    /// Whether the assertion passed
    pub passed: bool,
    /// Human-readable message
    pub message: String,
    /// The type of check that was performed
    pub check_type: Option<AssertionCheckType>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_note(anchors: Vec<NoteAnchor>) -> Note {
        let mut note = Note::new(
            Uuid::new_v4(),
            NoteType::Guideline,
            "Test note".to_string(),
            "test".to_string(),
        );
        note.anchors = anchors;
        note
    }

    fn create_test_file_info() -> FileInfo {
        FileInfo {
            path: "test.rs".to_string(),
            functions: vec![FunctionInfo {
                name: "test_func".to_string(),
                params: vec![("x".to_string(), Some("i32".to_string()))],
                return_type: Some("String".to_string()),
                is_async: false,
                is_unsafe: false,
                body: "let result = x.to_string(); result".to_string(),
            }],
            structs: vec![StructInfo {
                name: "TestStruct".to_string(),
                fields: vec![("field1".to_string(), "String".to_string(), true)],
                generics: vec![],
            }],
            symbols: vec!["test_func".to_string(), "TestStruct".to_string()],
            imports: vec!["std::string::String".to_string()],
        }
    }

    #[test]
    fn test_verify_function_unchanged() {
        let manager = NoteLifecycleManager::new();
        let file_info = create_test_file_info();

        // Create anchor with matching hashes
        let sig_hash = hash_function_signature(
            "test_func",
            &[("x".to_string(), Some("i32".to_string()))],
            Some("String"),
            false,
            false,
        );
        let body_hash = hash_function_body("let result = x.to_string(); result");

        let anchor = NoteAnchor::with_hashes(
            EntityType::Function,
            "test_func".to_string(),
            Some(sig_hash),
            Some(body_hash),
        );

        let note = create_test_note(vec![anchor]);
        let result = manager.verify_note_anchors(&note, &file_info);

        assert!(result.all_valid);
        assert!(result.suggested_update.is_none());
    }

    #[test]
    fn test_verify_function_body_changed() {
        let manager = NoteLifecycleManager::new();
        let file_info = create_test_file_info();

        // Create anchor with old body hash
        let sig_hash = hash_function_signature(
            "test_func",
            &[("x".to_string(), Some("i32".to_string()))],
            Some("String"),
            false,
            false,
        );
        let old_body_hash = hash_function_body("different code");

        let anchor = NoteAnchor::with_hashes(
            EntityType::Function,
            "test_func".to_string(),
            Some(sig_hash),
            Some(old_body_hash),
        );

        let note = create_test_note(vec![anchor]);
        let result = manager.verify_note_anchors(&note, &file_info);

        assert!(!result.all_valid);
        assert!(result.suggested_update.is_some());
        assert!(result.anchor_results[0].reason.contains("body changed"));
    }

    #[test]
    fn test_verify_function_deleted() {
        let manager = NoteLifecycleManager::new();
        let file_info = create_test_file_info();

        let anchor = NoteAnchor::with_hashes(
            EntityType::Function,
            "deleted_func".to_string(),
            Some("hash1".to_string()),
            Some("hash2".to_string()),
        );

        let note = create_test_note(vec![anchor]);
        let result = manager.verify_note_anchors(&note, &file_info);

        assert!(!result.all_valid);
        assert!(result.anchor_results[0].reason.contains("deleted"));
    }

    #[test]
    fn test_staleness_calculation() {
        let manager = NoteLifecycleManager::new();

        // Create a note from 100 days ago
        let mut note = Note::new(
            Uuid::new_v4(),
            NoteType::Tip, // Base decay: 90 days
            "Test".to_string(),
            "test".to_string(),
        );
        note.created_at = Utc::now() - chrono::Duration::days(100);
        note.last_confirmed_at = Some(note.created_at);

        let staleness = manager.calculate_staleness_score(&note, Utc::now());

        // After 100 days with 90-day decay, staleness should be > 0.6
        assert!(staleness > 0.6);
        assert!(staleness < 1.0);
    }

    #[test]
    fn test_staleness_assertion_never_stale() {
        let manager = NoteLifecycleManager::new();

        let mut note = Note::new(
            Uuid::new_v4(),
            NoteType::Assertion,
            "Test assertion".to_string(),
            "test".to_string(),
        );
        note.created_at = Utc::now() - chrono::Duration::days(1000);
        note.last_confirmed_at = Some(note.created_at);

        let staleness = manager.calculate_staleness_score(&note, Utc::now());

        // Assertions should never go stale
        assert_eq!(staleness, 0.0);
    }

    #[test]
    fn test_should_become_stale() {
        let manager = NoteLifecycleManager::new();

        let mut note = Note::new(
            Uuid::new_v4(),
            NoteType::Context, // Fast decay
            "Test".to_string(),
            "test".to_string(),
        );
        note.status = NoteStatus::Active;

        // Should become stale at > 0.8
        assert!(manager.should_become_stale(&note, 0.85));
        assert!(!manager.should_become_stale(&note, 0.75));

        // Already stale notes shouldn't trigger again
        note.status = NoteStatus::Stale;
        assert!(!manager.should_become_stale(&note, 0.9));
    }
}
