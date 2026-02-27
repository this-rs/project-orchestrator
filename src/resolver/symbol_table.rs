//! SymbolTable: dual-index in-memory symbol resolution
//!
//! Provides two lookup strategies:
//! - **Exact**: `(file_path, symbol_name) → node_id` for same-file resolution
//! - **Fuzzy**: `symbol_name → Vec<SymbolDefinition>` for cross-file resolution
//!
//! Populated from `ParsedFile` results during sync, then used by all
//! language-specific import resolvers.

use std::collections::HashMap;

/// Type of symbol (function, struct, trait, enum)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymbolType {
    Function,
    Struct,
    Trait,
    Enum,
}

impl std::fmt::Display for SymbolType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SymbolType::Function => write!(f, "function"),
            SymbolType::Struct => write!(f, "struct"),
            SymbolType::Trait => write!(f, "trait"),
            SymbolType::Enum => write!(f, "enum"),
        }
    }
}

/// A symbol definition with its location and type
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolDefinition {
    /// Unique node ID (format: "file_path:name:line_start")
    pub node_id: String,
    /// File where the symbol is defined
    pub file_path: String,
    /// Type of symbol
    pub symbol_type: SymbolType,
    /// Line where the symbol starts
    pub line_start: u32,
}

/// Statistics about the symbol table contents
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolTableStats {
    /// Number of files with symbols
    pub file_count: usize,
    /// Total unique symbol names in the global index
    pub global_symbol_count: usize,
    /// Total symbol definitions (may be > global_symbol_count due to same-name symbols in different files)
    pub total_definitions: usize,
    /// Breakdown by symbol type
    pub by_type: HashMap<SymbolType, usize>,
}

/// Dual-index symbol table for import resolution.
///
/// Built once per sync from all `ParsedFile` results, then queried
/// by language-specific resolvers.
#[derive(Debug, Clone)]
pub struct SymbolTable {
    /// file_path → (symbol_name → node_id) for exact lookups
    file_index: HashMap<String, HashMap<String, String>>,
    /// symbol_name → Vec<SymbolDefinition> for cross-file lookups
    global_index: HashMap<String, Vec<SymbolDefinition>>,
}

impl SymbolTable {
    /// Create an empty symbol table.
    pub fn new() -> Self {
        Self {
            file_index: HashMap::new(),
            global_index: HashMap::new(),
        }
    }

    /// Create a symbol table with estimated capacity.
    pub fn with_capacity(files: usize, symbols_per_file: usize) -> Self {
        Self {
            file_index: HashMap::with_capacity(files),
            global_index: HashMap::with_capacity(files * symbols_per_file),
        }
    }

    /// Add a symbol definition to both indexes.
    pub fn add(
        &mut self,
        name: &str,
        node_id: &str,
        file_path: &str,
        symbol_type: SymbolType,
        line_start: u32,
    ) {
        // File index
        self.file_index
            .entry(file_path.to_string())
            .or_default()
            .insert(name.to_string(), node_id.to_string());

        // Global index (deduplicate by node_id)
        let defs = self.global_index.entry(name.to_string()).or_default();

        // Avoid duplicates (same node_id)
        if !defs.iter().any(|d| d.node_id == node_id) {
            defs.push(SymbolDefinition {
                node_id: node_id.to_string(),
                file_path: file_path.to_string(),
                symbol_type,
                line_start,
            });
        }
    }

    /// Look up a symbol by exact file path and name.
    ///
    /// Returns the node_id if found in that specific file.
    pub fn lookup_exact(&self, file_path: &str, name: &str) -> Option<&str> {
        self.file_index
            .get(file_path)
            .and_then(|symbols| symbols.get(name))
            .map(|s| s.as_str())
    }

    /// Look up a symbol by name across all files.
    ///
    /// Returns all definitions (may be in different files).
    pub fn lookup_fuzzy(&self, name: &str) -> &[SymbolDefinition] {
        self.global_index
            .get(name)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get all symbol names defined in a specific file.
    pub fn symbols_in_file(&self, file_path: &str) -> Vec<&str> {
        self.file_index
            .get(file_path)
            .map(|symbols| symbols.keys().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Populate the symbol table from parsed files.
    ///
    /// Iterates all functions, structs, traits, and enums from each
    /// parsed file and adds them to both indexes.
    pub fn populate_from_parsed(&mut self, parsed_files: &[crate::parser::ParsedFile]) {
        for parsed in parsed_files {
            let file_path = &parsed.path;

            for func in &parsed.functions {
                let node_id = format!("{}:{}:{}", file_path, func.name, func.line_start);
                self.add(
                    &func.name,
                    &node_id,
                    file_path,
                    SymbolType::Function,
                    func.line_start,
                );
            }

            for s in &parsed.structs {
                let node_id = format!("{}:{}:{}", file_path, s.name, s.line_start);
                self.add(
                    &s.name,
                    &node_id,
                    file_path,
                    SymbolType::Struct,
                    s.line_start,
                );
            }

            for t in &parsed.traits {
                let node_id = format!("{}:{}:{}", file_path, t.name, t.line_start);
                self.add(
                    &t.name,
                    &node_id,
                    file_path,
                    SymbolType::Trait,
                    t.line_start,
                );
            }

            for e in &parsed.enums {
                let node_id = format!("{}:{}:{}", file_path, e.name, e.line_start);
                self.add(&e.name, &node_id, file_path, SymbolType::Enum, e.line_start);
            }
        }
    }

    /// Clear all entries from the symbol table.
    pub fn clear(&mut self) {
        self.file_index.clear();
        self.global_index.clear();
    }

    /// Get statistics about the symbol table.
    pub fn stats(&self) -> SymbolTableStats {
        let mut by_type: HashMap<SymbolType, usize> = HashMap::new();
        let mut total_definitions = 0;

        for defs in self.global_index.values() {
            for def in defs {
                *by_type.entry(def.symbol_type).or_insert(0) += 1;
                total_definitions += 1;
            }
        }

        SymbolTableStats {
            file_count: self.file_index.len(),
            global_symbol_count: self.global_index.len(),
            total_definitions,
            by_type,
        }
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neo4j::models::*;
    use crate::parser::ParsedFile;

    fn make_parsed_file(
        path: &str,
        functions: &[(&str, u32)],
        structs: &[(&str, u32)],
    ) -> ParsedFile {
        ParsedFile {
            path: path.to_string(),
            language: "rust".to_string(),
            hash: "abc123".to_string(),
            functions: functions
                .iter()
                .map(|(name, line)| FunctionNode {
                    name: name.to_string(),
                    visibility: Visibility::Public,
                    params: vec![],
                    return_type: None,
                    generics: vec![],
                    is_async: false,
                    is_unsafe: false,
                    complexity: 1,
                    file_path: path.to_string(),
                    line_start: *line,
                    line_end: line + 5,
                    docstring: None,
                })
                .collect(),
            structs: structs
                .iter()
                .map(|(name, line)| StructNode {
                    name: name.to_string(),
                    visibility: Visibility::Public,
                    generics: vec![],
                    file_path: path.to_string(),
                    line_start: *line,
                    line_end: line + 5,
                    docstring: None,
                    parent_class: None,
                    interfaces: vec![],
                })
                .collect(),
            traits: vec![],
            enums: vec![],
            imports: vec![],
            impl_blocks: vec![],
            function_calls: vec![],
            symbols: vec![],
        }
    }

    #[test]
    fn test_add_and_stats() {
        let mut table = SymbolTable::new();
        table.add("foo", "a.rs:foo:1", "a.rs", SymbolType::Function, 1);
        table.add("bar", "a.rs:bar:10", "a.rs", SymbolType::Function, 10);
        table.add("Baz", "b.rs:Baz:1", "b.rs", SymbolType::Struct, 1);

        let stats = table.stats();
        assert_eq!(stats.file_count, 2);
        assert_eq!(stats.global_symbol_count, 3);
        assert_eq!(stats.total_definitions, 3);
        assert_eq!(stats.by_type[&SymbolType::Function], 2);
        assert_eq!(stats.by_type[&SymbolType::Struct], 1);
    }

    #[test]
    fn test_lookup_exact() {
        let mut table = SymbolTable::new();
        table.add(
            "foo",
            "src/main.rs:foo:1",
            "src/main.rs",
            SymbolType::Function,
            1,
        );
        table.add(
            "bar",
            "src/lib.rs:bar:5",
            "src/lib.rs",
            SymbolType::Function,
            5,
        );

        assert_eq!(
            table.lookup_exact("src/main.rs", "foo"),
            Some("src/main.rs:foo:1")
        );
        assert_eq!(
            table.lookup_exact("src/lib.rs", "bar"),
            Some("src/lib.rs:bar:5")
        );
        // Wrong file
        assert_eq!(table.lookup_exact("src/lib.rs", "foo"), None);
        // Non-existent
        assert_eq!(table.lookup_exact("src/main.rs", "nonexistent"), None);
    }

    #[test]
    fn test_lookup_fuzzy_cross_file() {
        let mut table = SymbolTable::new();
        // Same name in two different files
        table.add(
            "handle",
            "src/api.rs:handle:1",
            "src/api.rs",
            SymbolType::Function,
            1,
        );
        table.add(
            "handle",
            "src/ws.rs:handle:10",
            "src/ws.rs",
            SymbolType::Function,
            10,
        );
        table.add(
            "unique",
            "src/lib.rs:unique:1",
            "src/lib.rs",
            SymbolType::Function,
            1,
        );

        let results = table.lookup_fuzzy("handle");
        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|d| d.file_path == "src/api.rs"));
        assert!(results.iter().any(|d| d.file_path == "src/ws.rs"));

        let results = table.lookup_fuzzy("unique");
        assert_eq!(results.len(), 1);

        let results = table.lookup_fuzzy("nonexistent");
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_deduplication() {
        let mut table = SymbolTable::new();
        // Adding the same node_id twice should not duplicate
        table.add("foo", "a.rs:foo:1", "a.rs", SymbolType::Function, 1);
        table.add("foo", "a.rs:foo:1", "a.rs", SymbolType::Function, 1);

        let results = table.lookup_fuzzy("foo");
        assert_eq!(results.len(), 1, "Should deduplicate by node_id");
    }

    #[test]
    fn test_symbols_in_file() {
        let mut table = SymbolTable::new();
        table.add("alpha", "a.rs:alpha:1", "a.rs", SymbolType::Function, 1);
        table.add("beta", "a.rs:beta:10", "a.rs", SymbolType::Struct, 10);
        table.add("gamma", "b.rs:gamma:1", "b.rs", SymbolType::Function, 1);

        let mut symbols = table.symbols_in_file("a.rs");
        symbols.sort();
        assert_eq!(symbols, vec!["alpha", "beta"]);

        let symbols = table.symbols_in_file("b.rs");
        assert_eq!(symbols, vec!["gamma"]);

        let symbols = table.symbols_in_file("nonexistent.rs");
        assert!(symbols.is_empty());
    }

    #[test]
    fn test_populate_from_parsed() {
        let files = vec![
            make_parsed_file(
                "src/main.rs",
                &[("main", 1), ("helper", 10)],
                &[("Config", 20)],
            ),
            make_parsed_file("src/lib.rs", &[("init", 1)], &[("State", 10)]),
            make_parsed_file("src/api.rs", &[("handle_request", 1)], &[]),
        ];

        let mut table = SymbolTable::new();
        table.populate_from_parsed(&files);

        let stats = table.stats();
        assert_eq!(stats.file_count, 3);
        assert_eq!(stats.total_definitions, 6); // 4 functions + 2 structs

        // Exact lookups
        assert!(table.lookup_exact("src/main.rs", "main").is_some());
        assert!(table.lookup_exact("src/main.rs", "helper").is_some());
        assert!(table.lookup_exact("src/main.rs", "Config").is_some());
        assert!(table.lookup_exact("src/lib.rs", "init").is_some());
        assert!(table.lookup_exact("src/api.rs", "handle_request").is_some());

        // Cross-file lookup should not find main.rs symbols in lib.rs
        assert!(table.lookup_exact("src/lib.rs", "main").is_none());

        // Fuzzy lookup
        let results = table.lookup_fuzzy("handle_request");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].file_path, "src/api.rs");
        assert_eq!(results[0].symbol_type, SymbolType::Function);
    }

    #[test]
    fn test_clear() {
        let mut table = SymbolTable::new();
        table.add("foo", "a.rs:foo:1", "a.rs", SymbolType::Function, 1);
        assert_eq!(table.stats().total_definitions, 1);

        table.clear();
        assert_eq!(table.stats().total_definitions, 0);
        assert_eq!(table.stats().file_count, 0);
        assert!(table.lookup_exact("a.rs", "foo").is_none());
    }

    #[test]
    fn test_with_capacity() {
        let table = SymbolTable::with_capacity(100, 10);
        assert_eq!(table.stats().file_count, 0);
        assert_eq!(table.stats().total_definitions, 0);
    }
}
