//! Common query parameter structs for pagination and filtering

use serde::{Deserialize, Deserializer, Serialize};
use std::str::FromStr;

/// Helper to deserialize numbers from query string (which are always strings)
fn deserialize_from_str<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: FromStr + Default,
    T::Err: std::fmt::Display,
{
    use serde::de::Error;
    let s: Option<String> = Option::deserialize(deserializer)?;
    match s {
        Some(s) if !s.is_empty() => s.parse().map_err(D::Error::custom),
        _ => Ok(T::default()),
    }
}

/// Helper to deserialize optional numbers from query string
fn deserialize_option_from_str<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: FromStr,
    T::Err: std::fmt::Display,
{
    use serde::de::Error;
    let s: Option<String> = Option::deserialize(deserializer)?;
    match s {
        Some(s) if !s.is_empty() => s.parse().map(Some).map_err(D::Error::custom),
        _ => Ok(None),
    }
}

/// Pagination parameters for list endpoints
#[derive(Debug, Deserialize, Clone)]
pub struct PaginationParams {
    /// Max items to return (default: 50, max: 100)
    #[serde(default = "default_limit", deserialize_with = "deserialize_from_str")]
    pub limit: usize,
    /// Items to skip (default: 0)
    #[serde(default, deserialize_with = "deserialize_from_str")]
    pub offset: usize,
    /// Sort field (e.g., "created_at", "priority", "title")
    pub sort_by: Option<String>,
    /// Sort direction: "asc" or "desc" (default: "desc")
    #[serde(default = "default_sort_order")]
    pub sort_order: String,
}

fn default_limit() -> usize {
    50
}

fn default_sort_order() -> String {
    "desc".to_string()
}

impl Default for PaginationParams {
    fn default() -> Self {
        Self {
            limit: default_limit(),
            offset: 0,
            sort_by: None,
            sort_order: default_sort_order(),
        }
    }
}

impl PaginationParams {
    /// Validate pagination parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.limit > 100 {
            return Err("limit cannot exceed 100".to_string());
        }
        if !["asc", "desc"].contains(&self.sort_order.as_str()) {
            return Err("sort_order must be 'asc' or 'desc'".to_string());
        }
        Ok(())
    }

    /// Get validated limit (capped at 100)
    pub fn validated_limit(&self) -> usize {
        self.limit.min(100)
    }
}

/// Status filter - accepts comma-separated values
#[derive(Debug, Deserialize, Default, Clone)]
pub struct StatusFilter {
    /// Comma-separated status values, e.g., "pending,in_progress"
    pub status: Option<String>,
}

impl StatusFilter {
    /// Convert comma-separated string to Vec<String>
    pub fn to_vec(&self) -> Option<Vec<String>> {
        self.status.as_ref().map(|s| {
            s.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
    }
}

/// Priority filter with min/max range
#[derive(Debug, Deserialize, Default, Clone)]
pub struct PriorityFilter {
    /// Minimum priority (inclusive)
    #[serde(default, deserialize_with = "deserialize_option_from_str")]
    pub priority_min: Option<i32>,
    /// Maximum priority (inclusive)
    #[serde(default, deserialize_with = "deserialize_option_from_str")]
    pub priority_max: Option<i32>,
}

impl PriorityFilter {
    /// Check if any priority filter is set
    pub fn is_set(&self) -> bool {
        self.priority_min.is_some() || self.priority_max.is_some()
    }
}

/// Tags filter - accepts comma-separated values
#[derive(Debug, Deserialize, Default, Clone)]
pub struct TagsFilter {
    /// Comma-separated tag values, e.g., "backend,api"
    pub tags: Option<String>,
}

impl TagsFilter {
    /// Convert comma-separated string to Vec<String>
    pub fn to_vec(&self) -> Option<Vec<String>> {
        self.tags.as_ref().map(|s| {
            s.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
    }
}

/// Search filter for text-based queries
#[derive(Debug, Deserialize, Default, Clone)]
pub struct SearchFilter {
    /// Search query string
    pub search: Option<String>,
}

impl SearchFilter {
    /// Check if search filter is set
    pub fn is_set(&self) -> bool {
        self.search.as_ref().is_some_and(|s| !s.trim().is_empty())
    }
}

/// Paginated response wrapper
#[derive(Debug, Serialize)]
pub struct PaginatedResponse<T> {
    /// Items in the current page
    pub items: Vec<T>,
    /// Total count of items matching the filter
    pub total: usize,
    /// Maximum items per page (as requested)
    pub limit: usize,
    /// Number of items skipped
    pub offset: usize,
    /// Whether there are more items after this page
    pub has_more: bool,
}

impl<T> PaginatedResponse<T> {
    /// Create a new paginated response
    pub fn new(items: Vec<T>, total: usize, limit: usize, offset: usize) -> Self {
        Self {
            has_more: offset + items.len() < total,
            items,
            total,
            limit,
            offset,
        }
    }

    /// Create an empty paginated response
    pub fn empty(limit: usize, offset: usize) -> Self {
        Self {
            items: Vec::new(),
            total: 0,
            limit,
            offset,
            has_more: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // PaginationParams Tests
    // =========================================================================

    #[test]
    fn test_pagination_defaults() {
        let params = PaginationParams::default();
        assert_eq!(params.limit, 50);
        assert_eq!(params.offset, 0);
        assert_eq!(params.sort_order, "desc");
        assert!(params.sort_by.is_none());
    }

    #[test]
    fn test_pagination_validation_success() {
        let params = PaginationParams {
            limit: 50,
            offset: 0,
            sort_by: Some("created_at".to_string()),
            sort_order: "asc".to_string(),
        };
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_pagination_validation_limit_too_high() {
        let params = PaginationParams {
            limit: 150,
            ..Default::default()
        };
        assert!(params.validate().is_err());
        assert!(params.validate().unwrap_err().contains("limit"));
    }

    #[test]
    fn test_pagination_validation_invalid_sort_order() {
        let params = PaginationParams {
            sort_order: "invalid".to_string(),
            ..Default::default()
        };
        assert!(params.validate().is_err());
        assert!(params.validate().unwrap_err().contains("sort_order"));
    }

    #[test]
    fn test_pagination_validated_limit() {
        let params = PaginationParams {
            limit: 150,
            ..Default::default()
        };
        assert_eq!(params.validated_limit(), 100);

        let params2 = PaginationParams {
            limit: 50,
            ..Default::default()
        };
        assert_eq!(params2.validated_limit(), 50);
    }

    // =========================================================================
    // StatusFilter Tests
    // =========================================================================

    #[test]
    fn test_status_filter_to_vec() {
        let filter = StatusFilter {
            status: Some("pending, in_progress, completed".to_string()),
        };
        let vec = filter.to_vec().unwrap();
        assert_eq!(vec, vec!["pending", "in_progress", "completed"]);
    }

    #[test]
    fn test_status_filter_to_vec_none() {
        let filter = StatusFilter { status: None };
        assert!(filter.to_vec().is_none());
    }

    #[test]
    fn test_status_filter_to_vec_empty_string() {
        let filter = StatusFilter {
            status: Some("".to_string()),
        };
        let vec = filter.to_vec().unwrap();
        assert!(vec.is_empty());
    }

    #[test]
    fn test_status_filter_to_vec_single_value() {
        let filter = StatusFilter {
            status: Some("pending".to_string()),
        };
        let vec = filter.to_vec().unwrap();
        assert_eq!(vec, vec!["pending"]);
    }

    // =========================================================================
    // PriorityFilter Tests
    // =========================================================================

    #[test]
    fn test_priority_filter_is_set() {
        let filter = PriorityFilter {
            priority_min: Some(1),
            priority_max: None,
        };
        assert!(filter.is_set());

        let filter2 = PriorityFilter {
            priority_min: None,
            priority_max: Some(10),
        };
        assert!(filter2.is_set());

        let filter3 = PriorityFilter {
            priority_min: None,
            priority_max: None,
        };
        assert!(!filter3.is_set());
    }

    // =========================================================================
    // TagsFilter Tests
    // =========================================================================

    #[test]
    fn test_tags_filter_to_vec() {
        let filter = TagsFilter {
            tags: Some("backend, api, rust".to_string()),
        };
        let vec = filter.to_vec().unwrap();
        assert_eq!(vec, vec!["backend", "api", "rust"]);
    }

    #[test]
    fn test_tags_filter_to_vec_none() {
        let filter = TagsFilter { tags: None };
        assert!(filter.to_vec().is_none());
    }

    #[test]
    fn test_tags_filter_to_vec_with_extra_whitespace() {
        let filter = TagsFilter {
            tags: Some("  backend  ,  api  ,  rust  ".to_string()),
        };
        let vec = filter.to_vec().unwrap();
        assert_eq!(vec, vec!["backend", "api", "rust"]);
    }

    // =========================================================================
    // SearchFilter Tests
    // =========================================================================

    #[test]
    fn test_search_filter_is_set() {
        let filter = SearchFilter {
            search: Some("query".to_string()),
        };
        assert!(filter.is_set());
    }

    #[test]
    fn test_search_filter_is_not_set_when_none() {
        let filter = SearchFilter { search: None };
        assert!(!filter.is_set());
    }

    #[test]
    fn test_search_filter_is_not_set_when_empty() {
        let filter = SearchFilter {
            search: Some("".to_string()),
        };
        assert!(!filter.is_set());
    }

    #[test]
    fn test_search_filter_is_not_set_when_whitespace() {
        let filter = SearchFilter {
            search: Some("   ".to_string()),
        };
        assert!(!filter.is_set());
    }

    // =========================================================================
    // PaginatedResponse Tests
    // =========================================================================

    #[test]
    fn test_paginated_response_has_more() {
        let items = vec![1, 2, 3, 4, 5];
        let response = PaginatedResponse::new(items, 10, 5, 0);
        assert_eq!(response.items.len(), 5);
        assert_eq!(response.total, 10);
        assert!(response.has_more);
    }

    #[test]
    fn test_paginated_response_no_more() {
        let response = PaginatedResponse::new(vec![6, 7, 8, 9, 10], 10, 5, 5);
        assert!(!response.has_more);
    }

    #[test]
    fn test_paginated_response_empty() {
        let response: PaginatedResponse<i32> = PaginatedResponse::empty(10, 0);
        assert!(response.items.is_empty());
        assert_eq!(response.total, 0);
        assert!(!response.has_more);
    }

    #[test]
    fn test_paginated_response_exact_boundary() {
        // Exactly fills the page
        let response = PaginatedResponse::new(vec![1, 2, 3, 4, 5], 5, 5, 0);
        assert!(!response.has_more);
    }

    #[test]
    fn test_paginated_response_serialization() {
        let response = PaginatedResponse::new(vec!["a", "b", "c"], 10, 3, 0);
        let json = serde_json::to_string(&response).unwrap();

        assert!(json.contains("\"items\""));
        assert!(json.contains("\"total\":10"));
        assert!(json.contains("\"limit\":3"));
        assert!(json.contains("\"offset\":0"));
        assert!(json.contains("\"has_more\":true"));
    }
}
