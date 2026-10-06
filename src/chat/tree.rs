//! Annotation of a session tree with what the frontend reads: which provider and
//! model ran each node, what each cost, and what a whole subtree cost.
//!
//! Pure over data already loaded. A cost that is not known is not zero: a
//! subtree cost is given only when every node under it has one.

use std::collections::HashMap;

use crate::neo4j::models::SessionTreeNode;

/// What is known of one session.
#[derive(Debug, Clone, Default)]
pub struct NodeInfo {
    /// Provider instance; `None` = claude-code.
    pub provider_id: Option<String>,
    /// Model.
    pub model: Option<String>,
    /// Own cost; `None` = unknown.
    pub cost_usd: Option<f64>,
}

/// Fills `provider_id`, `model`, `cost_usd`, `subtree_cost_usd` and, on the
/// root, the delegation limits.
pub fn annotate(tree: &mut [SessionTreeNode], info: &HashMap<String, NodeInfo>) {
    for node in tree.iter_mut() {
        if let Some(i) = info.get(&node.session_id) {
            node.provider_id = i.provider_id.clone();
            node.model = i.model.clone();
            node.cost_usd = i.cost_usd;
        }
    }
    // Subtree cost: the node itself plus every descendant, all known.
    let children: HashMap<&str, Vec<&str>> = {
        let mut m: HashMap<&str, Vec<&str>> = HashMap::new();
        for n in tree.iter() {
            if let Some(p) = n.parent_session_id.as_deref() {
                m.entry(p).or_default().push(n.session_id.as_str());
            }
        }
        m
    };
    let own: HashMap<&str, Option<f64>> = tree
        .iter()
        .map(|n| (n.session_id.as_str(), n.cost_usd))
        .collect();
    fn total(
        id: &str,
        children: &HashMap<&str, Vec<&str>>,
        own: &HashMap<&str, Option<f64>>,
        depth: usize,
    ) -> Option<f64> {
        if depth > 16 {
            return None;
        }
        let mut sum = (*own.get(id)?)?;
        for child in children.get(id).into_iter().flatten() {
            sum += total(child, children, own, depth + 1)?;
        }
        Some(sum)
    }
    let subtree: Vec<Option<f64>> = tree
        .iter()
        .map(|n| total(&n.session_id, &children, &own, 0))
        .collect();
    for (node, cost) in tree.iter_mut().zip(subtree) {
        node.subtree_cost_usd = cost;
    }
    if let Some(root) = tree.first_mut() {
        root.max_depth = Some(super::envelope::MAX_DELEGATION_DEPTH);
        root.max_children = u32::try_from(super::envelope::MAX_LIVE_CHILDREN).ok();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, parent: Option<&str>) -> SessionTreeNode {
        SessionTreeNode {
            session_id: id.into(),
            parent_session_id: parent.map(str::to_string),
            spawn_type: None,
            run_id: None,
            task_id: None,
            depth: 0,
            created_at: None,
            provider_id: None,
            model: None,
            cost_usd: None,
            subtree_cost_usd: None,
            max_depth: None,
            max_children: None,
        }
    }

    fn info(provider: Option<&str>, cost: Option<f64>) -> NodeInfo {
        NodeInfo {
            provider_id: provider.map(str::to_string),
            model: Some("m".into()),
            cost_usd: cost,
        }
    }

    #[test]
    fn a_subtree_cost_is_the_sum_when_every_node_is_known() {
        let mut tree = vec![
            node("root", None),
            node("a", Some("root")),
            node("b", Some("root")),
        ];
        let infos = HashMap::from([
            ("root".to_string(), info(None, Some(1.0))),
            ("a".to_string(), info(Some("ds"), Some(0.5))),
            ("b".to_string(), info(Some("ds"), Some(0.25))),
        ]);
        annotate(&mut tree, &infos);
        assert_eq!(tree[0].subtree_cost_usd, Some(1.75));
        assert_eq!(tree[1].subtree_cost_usd, Some(0.5));
        assert_eq!(tree[1].provider_id.as_deref(), Some("ds"));
        assert_eq!(tree[0].provider_id, None, "absent = claude-code");
        assert_eq!(tree[0].max_depth, Some(1));
        assert_eq!(tree[0].max_children, Some(4));
        assert_eq!(tree[1].max_depth, None, "limits ride on the root only");
    }

    #[test]
    fn an_unknown_cost_is_not_a_zero_and_hides_the_subtree_total() {
        let mut tree = vec![node("root", None), node("a", Some("root"))];
        let infos = HashMap::from([
            ("root".to_string(), info(None, Some(1.0))),
            ("a".to_string(), info(Some("local"), None)),
        ]);
        annotate(&mut tree, &infos);
        assert_eq!(tree[1].cost_usd, None);
        assert_eq!(tree[1].subtree_cost_usd, None);
        assert_eq!(
            tree[0].subtree_cost_usd, None,
            "one unknown below: no exact total"
        );
    }

    #[test]
    fn a_node_that_could_not_be_read_stays_unannotated() {
        let mut tree = vec![node("root", None)];
        annotate(&mut tree, &HashMap::new());
        assert_eq!(tree[0].cost_usd, None);
        assert_eq!(tree[0].model, None);
    }
}
