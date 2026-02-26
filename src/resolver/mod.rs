//! Import resolution infrastructure
//!
//! Provides data structures and algorithms for resolving import paths
//! to actual files in the project:
//! - `SuffixIndex`: O(1) file lookup by path suffix
//! - `SymbolTable`: dual-index symbol resolution (by name and by file)
//! - `ResolveCache`: LRU cache for resolved imports

pub mod suffix_index;
pub mod symbol_table;

pub use suffix_index::SuffixIndex;
pub use symbol_table::SymbolTable;
