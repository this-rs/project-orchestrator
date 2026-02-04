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

    #[test]
    fn test_pagination_defaults() {
        let params = PaginationParams::default();
        assert_eq!(params.limit, 50);
        assert_eq!(params.offset, 0);
        assert_eq!(params.sort_order, "desc");
    }

    #[test]
    fn test_pagination_validation() {
        let mut params = PaginationParams::default();
        assert!(params.validate().is_ok());

        params.limit = 150;
        assert!(params.validate().is_err());

        params.limit = 50;
        params.sort_order = "invalid".to_string();
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_status_filter_to_vec() {
        let filter = StatusFilter {
            status: Some("pending, in_progress, completed".to_string()),
        };
        let vec = filter.to_vec().unwrap();
        assert_eq!(vec, vec!["pending", "in_progress", "completed"]);

        let empty_filter = StatusFilter { status: None };
        assert!(empty_filter.to_vec().is_none());
    }

    #[test]
    fn test_tags_filter_to_vec() {
        let filter = TagsFilter {
            tags: Some("backend, api, rust".to_string()),
        };
        let vec = filter.to_vec().unwrap();
        assert_eq!(vec, vec!["backend", "api", "rust"]);
    }

    #[test]
    fn test_paginated_response() {
        let items = vec![1, 2, 3, 4, 5];
        let response = PaginatedResponse::new(items, 10, 5, 0);
        assert_eq!(response.items.len(), 5);
        assert_eq!(response.total, 10);
        assert!(response.has_more);

        let response2 = PaginatedResponse::new(vec![6, 7, 8, 9, 10], 10, 5, 5);
        assert!(!response2.has_more);
    }
}
